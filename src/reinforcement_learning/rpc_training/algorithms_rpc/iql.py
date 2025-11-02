import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
import torch.distributed.rpc as rpc

from src.reinforcement_learning.rpc_training.algorithms_rpc.model_rpc import (
    GaussianPolicy,
    QNetwork,
    epsilon,
    weights_init_,
)
from src.reinforcement_learning.rpc_training.algorithms_rpc.replay_memory_rpc import ReplayMemory
from src.reinforcement_learning.rpc_training.algorithms_rpc.utils import soft_update, hard_update
from src.reinforcement_learning.rpc_training.helper_rpc.helper_pure_rpc import _remote_method
from src.reinforcement_learning.rpc_training.train_rpc import TrainerRPC


class ValueNetwork(nn.Module):
    """Simple feed-forward value network used by IQL."""

    def __init__(self, num_inputs: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.input_layer = nn.Linear(num_inputs, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for _ in range(max(num_layers - 1, 0)):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_layer(state))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.output_layer(x)


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(diff > 0, expectile, 1 - expectile)
    return (weight * diff.pow(2)).mean()


class IQLAgent(object):
    """Implicit Q-Learning agent with an AWAC-style online fine-tuning stage."""

    def __init__(
        self,
        num_inputs: int,
        action_space: np.ndarray,
        config,
        rank: int,
        num_gpus: int,
    ) -> None:
        self.action_space = action_space.shape[0]
        self.state_space = num_inputs
        self.config = config

        self.rpc_id = rpc.get_worker_info().id
        if num_gpus <= 0:
            device = "cpu"
            self.device = torch.device(device)
        else:
            device_index = (self.rpc_id - 1) % num_gpus
            device = f"cuda:{device_index}" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device)

        self.worker_id = rank
        print(f"0. Worker id {self.worker_id} Rpc id {self.rpc_id} Device {device}")

        self.gamma = config.sac['gamma']
        self.tau = config.sac['tau']
        self.target_update_interval = config.sac['target_update_interval']
        self.expectile = config.sac['expectile']
        self.offline_temperature = max(config.sac['awac_temperature'], 1e-6)
        self.awac_lambda = max(config.sac['awac_lambda'], 1e-6)
        self.adv_max_weight = config.sac['adv_max_weight']
        self.offline_updates_per_call = max(config.sac['offline_updates_per_call'], 1)
        self.online_updates_per_call = max(config.sac['updates_per_episode_rpc'], 1)
        self.value_target_tau = config.sac['value_target_tau']

        self.policy_type = config.sac['policy']
        self.initialize_last_layer_zero = config.sac['initialize_last_layer_0']
        self.initialize_last_layer_near_zero = config.sac['initialize_last_layer_near_0']

        self.lr = config.sac['lr']
        hidden_size_critic = config.sac['hidden_size_critic']
        num_layers_critic = config.sac['num_layers_critic']
        hidden_size_actor = config.sac['hidden_size_actor']
        num_layers_actor = config.sac['num_layers_actor']

        if isinstance(hidden_size_critic, list):
            value_hidden_dim = hidden_size_critic[0]
        else:
            value_hidden_dim = hidden_size_critic

        self.critic, self.critic_optim = self.initialise_critic(
            num_inputs=num_inputs,
            action_space=action_space,
            hidden_size_critic=hidden_size_critic,
            num_layers_critic=num_layers_critic,
        )
        self.value_net, self.value_target, self.value_optim = self.initialise_value(
            num_inputs=num_inputs,
            hidden_dim=value_hidden_dim,
            num_layers=num_layers_critic,
        )
        self.policy, self.policy_optim = self.initialise_policy(
            num_inputs=num_inputs,
            action_space=action_space,
            hidden_size_actor=hidden_size_actor,
            num_layers_actor=num_layers_actor,
        )

        self.memory = ReplayMemory(config.sac['memory_size'])

        self.total_update = 0
        self.q_loss_list: Optional[Tuple[int, float]] = None
        self.value_loss_list: Optional[Tuple[int, float]] = None
        self.policy_loss_list: Optional[Tuple[int, float]] = None

    def initialise_critic(
        self,
        num_inputs: int,
        action_space: np.ndarray,
        hidden_size_critic,
        num_layers_critic: int,
    ):
        print("1. Initialasing IQL Critic")
        critic = QNetwork(num_inputs, action_space.shape[0], hidden_size_critic, num_layers_critic).to(self.device)
        critic_optim = Adam(critic.parameters(), lr=self.lr)
        return critic, critic_optim

    def initialise_value(self, num_inputs: int, hidden_dim: int, num_layers: int):
        print("2. Initialasing IQL Value network")
        value_net = ValueNetwork(num_inputs, hidden_dim, num_layers).to(self.device)
        value_target = ValueNetwork(num_inputs, hidden_dim, num_layers).to(self.device)
        hard_update(value_target, value_net)
        value_optim = Adam(value_net.parameters(), lr=self.lr)
        return value_net, value_target, value_optim

    def initialise_policy(
        self,
        num_inputs: int,
        action_space: np.ndarray,
        hidden_size_actor: int,
        num_layers_actor: int,
    ):
        print("3. Initialising IQL Policy; Type:", self.policy_type)
        if self.policy_type != "Gaussian":
            raise NotImplementedError("Only Gaussian policies are currently supported for IQL")

        policy = GaussianPolicy(
            num_inputs=num_inputs,
            num_actions=action_space.shape[0],
            hidden_dim=hidden_size_actor,
            action_scale=self.config.sac['gaussian_std'],
            action_bias=self.config.sac['gaussian_mu'],
            num_layers=num_layers_actor,
            initialize_last_layer_zero=self.initialize_last_layer_zero,
            initialize_last_layer_near_zero=self.initialize_last_layer_near_zero,
            activation=self.config.sac['activation'],
            LOG_SIG_MAX=self.config.sac['LOG_SIG_MAX'],
        ).to(self.device)
        policy_optim = Adam(policy.parameters(), lr=self.lr)
        return policy, policy_optim

    def load_policy(self, master_rref, worker_id):
        assert self.worker_id == worker_id
        policy_model_path = self.config.sac.get('pretrained_model_path', None)
        if policy_model_path in [None, 'None']:
            return
        model_dict = torch.load(policy_model_path, map_location=self.device)
        model_state_dict = model_dict.get("model_state_dict", model_dict)
        self.policy.load_state_dict(model_state_dict)

    def select_action(self, state, eval_mode: bool = False, only_choosing_action: bool = False):
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _, mean = self.policy.sample(state_tensor, only_choosing_action=only_choosing_action)
        mean_np = mean.detach().cpu().numpy()[0]
        if eval_mode:
            action_np = mean_np
        else:
            action_np = action.detach().cpu().numpy()[0]
        return action_np, mean_np

    def master_ask_action(self, master_rref, state, worker_id, eval_mode):
        assert self.worker_id == worker_id
        with torch.no_grad():
            action, mean = self.select_action(state=state, eval_mode=eval_mode, only_choosing_action=True)
        _remote_method(TrainerRPC.report_action, master_rref, action, mean, self.worker_id)

    def master_ask_metrics(self, master_rref):
        _remote_method(
            TrainerRPC.report_metrics,
            master_rref,
            self.worker_id,
            self.value_loss_list,
            self.q_loss_list,
            None,
            None,
            self.policy_loss_list,
        )

    def get_tensors_from_memory(self, memory: ReplayMemory, batch_size: int):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        return state_batch, action_batch, reward_batch, next_state_batch, mask_batch

    def _log_prob_from_actions(self, state_batch: torch.Tensor, action_batch: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.policy.forward(state_batch)
        std = log_std.exp()
        normal = Normal(mean, std)
        action_scale = self.policy.action_scale.to(self.device)
        action_bias = self.policy.action_bias.to(self.device)
        y = (action_batch - action_bias) / action_scale
        y = torch.clamp(y, -1 + 1e-6, 1 - 1e-6)
        x = 0.5 * (torch.log1p(y) - torch.log1p(-y))
        log_prob = normal.log_prob(x)
        correction = torch.log(action_scale * (1 - y.pow(2)).clamp(min=1e-6) + epsilon)
        log_prob = log_prob - correction
        return log_prob.sum(dim=-1, keepdim=True)

    def update_critic(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
        reward_batch: torch.Tensor,
        next_state_batch: torch.Tensor,
        mask_batch: torch.Tensor,
    ) -> float:
        with torch.no_grad():
            target_v = self.value_target(next_state_batch)
            q_target = reward_batch + mask_batch * self.gamma * target_v
        q1, q2 = self.critic(state_batch, action_batch)
        q_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()
        return q_loss.detach().item()

    def update_value(self, state_batch: torch.Tensor, action_batch: torch.Tensor) -> float:
        with torch.no_grad():
            q1, q2 = self.critic(state_batch, action_batch)
            q_estimate = torch.min(q1, q2)
        value = self.value_net(state_batch)
        diff = q_estimate - value
        value_loss = expectile_loss(diff, self.expectile)
        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()
        return value_loss.detach().item()

    def update_actor(self, state_batch: torch.Tensor, action_batch: torch.Tensor, temperature: float) -> float:
        with torch.no_grad():
            q1, q2 = self.critic(state_batch, action_batch)
            value = self.value_net(state_batch)
            advantage = torch.min(q1, q2) - value
        weights = torch.exp(advantage / temperature)
        if self.adv_max_weight > 0:
            weights = torch.clamp(weights, max=self.adv_max_weight)
        log_prob = self._log_prob_from_actions(state_batch, action_batch)
        actor_loss = -(weights.detach() * log_prob).mean()
        self.policy_optim.zero_grad()
        actor_loss.backward()
        self.policy_optim.step()
        return actor_loss.detach().item()

    def update_parameters(
        self,
        master_memory: ReplayMemory,
        batch_size: int,
        updates: int,
        total_step: int,
        phase: str,
    ) -> None:
        self.total_update = updates
        master_memory_idx = 0
        updates_per_call = self.offline_updates_per_call if phase == "offline" else self.online_updates_per_call
        q_loss_value: Optional[float] = None
        value_loss_value: Optional[float] = None
        policy_loss_value: Optional[float] = None

        for _ in range(updates_per_call):
            if master_memory_idx < len(master_memory):
                state_master, action_master, reward_master, next_state_master, mask_master = (
                    master_memory.buffer[master_memory_idx]
                )
                self.memory.push(state_master, action_master, reward_master, next_state_master, mask_master)
                master_memory_idx += 1

            if len(self.memory) > batch_size:
                state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.get_tensors_from_memory(
                    self.memory, batch_size
                )
                q_loss_value = self.update_critic(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
                value_loss_value = self.update_value(state_batch, action_batch)
                temperature = self.offline_temperature if phase == "offline" else self.awac_lambda
                policy_loss_value = self.update_actor(state_batch, action_batch, temperature)

                if updates % self.target_update_interval == 0:
                    soft_update(self.value_target, self.value_net, self.value_target_tau)

        if (
            len(self.memory) > batch_size
            and total_step % max(10 * updates_per_call, 1) == 0
            and value_loss_value is not None
            and q_loss_value is not None
            and policy_loss_value is not None
        ):
            self.value_loss_list = [total_step, value_loss_value]
            self.q_loss_list = [total_step, q_loss_value]
            self.policy_loss_list = [total_step, policy_loss_value]

    def save_model(self, experiment_name, episode, modes_controlled, worker_id):
        assert worker_id == self.worker_id
        folder = (
            "outputgain_0.4_noice3_layer3_GM4_para0.16_train0.16_no_auencoder_worker4_hidden32_criticpolicy_kan_test/"
            "output_models/models_rpc/" + experiment_name + "/"
        )
        if not os.path.exists(folder):
            os.makedirs(folder)

        actor_path = (
            folder
            + experiment_name
            + f"_worker_{worker_id}_iql_actor_{experiment_name}_episode_{episode}"
        )
        critic_path = (
            folder
            + experiment_name
            + f"_worker_{worker_id}_iql_critic_{experiment_name}_episode_{episode}"
        )

        print(f'Saving IQL actor to {actor_path}')
        print(f'Saving IQL critic to {critic_path}')

        torch.save(
            {
                'worker_id': worker_id,
                'models_controlled': modes_controlled,
                'model_state_dict': self.policy.state_dict(),
            },
            actor_path,
        )
        torch.save(self.critic.state_dict(), critic_path)
