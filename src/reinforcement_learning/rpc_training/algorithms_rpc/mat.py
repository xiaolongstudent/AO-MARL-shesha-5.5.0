import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
import torch.distributed.rpc as rpc

from src.reinforcement_learning.rpc_training.helper_rpc.helper_pure_rpc import _remote_method

LOG_SIG_MIN = -20
LOG_SIG_MAX = 2
epsilon = 1e-5


class TransformerPolicy(nn.Module):
    """Simple transformer based policy used by MAT."""

    def __init__(self, num_inputs: int, num_actions: int, d_model: int, nhead: int, num_layers: int):
        super().__init__()
        self.input_layer = nn.Linear(num_inputs, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mean_layer = nn.Linear(d_model, num_actions)
        self.log_std_layer = nn.Linear(d_model, num_actions)

    def forward(self, x: torch.Tensor):
        # x: [batch, num_inputs]
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        x = self.input_layer(x).unsqueeze(1)  # [batch, 1, d_model]
        x = self.encoder(x)
        x = x.squeeze(1)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=10.0, neginf=-10.0)
        log_std = torch.nan_to_num(log_std, nan=0.0, posinf=LOG_SIG_MAX, neginf=LOG_SIG_MIN)
        return mean.clamp(-10.0, 10.0), log_std

    def sample(self, state: torch.Tensor, only_mean: bool = False):
        mean, log_std = self.forward(state)
        if only_mean:
            action = torch.tanh(mean)
            log_prob = None
        else:
            std = torch.nan_to_num(log_std, nan=0.0, posinf=LOG_SIG_MAX, neginf=LOG_SIG_MIN)
            std = std.exp().clamp(min=1e-6, max=1e6)
            std = torch.nan_to_num(std, nan=1.0, posinf=1e6, neginf=1e-6)
            normal = Normal(mean, std)
            z = normal.rsample()
            z = torch.nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0)
            action = torch.tanh(z)
            log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
            log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)


class ValueNetwork(nn.Module):
    """Critic network for MAT."""

    def __init__(self, num_inputs: int, hidden_dims):
        super().__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        elif isinstance(hidden_dims, (list, tuple)):
            hidden_dims = list(hidden_dims)
        else:
            raise TypeError("hidden_dims must be an int or sequence of ints")

        layers = []
        last_dim = num_inputs
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim

        layers.append(nn.Linear(last_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.model(x)


class MAT(object):
    """Multi-Agent Transformer training class with a PPO style update."""

    def __init__(self, num_inputs, action_space, config, rank, num_gpus, model_dir=None):
        self.state_dim = num_inputs
        self.action_dim = action_space.shape[0]
        self.config = config

        self.rpc_id = rpc.get_worker_info().id
        if num_gpus <= 0:
            device_id = "cpu"
        else:
            device_id = (self.rpc_id - 1) % num_gpus
        self.device = torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")

        hidden_actor = config.sac['hidden_size_actor']
        hidden_critic = config.sac['hidden_size_critic']
        layers_actor = config.sac['num_layers_actor']

        self.policy = TransformerPolicy(num_inputs, self.action_dim,
                                        d_model=hidden_actor,
                                        nhead=4,
                                        num_layers=layers_actor).to(self.device)
        self.value = ValueNetwork(num_inputs, hidden_dims=hidden_critic).to(self.device)

        self.policy_optim = Adam(self.policy.parameters(), lr=config.sac['lr'])
        self.value_optim = Adam(self.value.parameters(), lr=config.sac['lr'])

        self.gamma = config.sac['gamma']
        self.updates_per_episode = max(1, int(config.sac.get('updates_per_episode_rpc', 1)))
        self.worker_id = rank

        self.policy_loss_list = None
        self.value_loss_list = None
        self.entropy_list = None
        # ``entropy_loss_list`` was previously used by the trainer RPC.  Keep
        # it as an alias so older code that still references the legacy name
        # keeps functioning until all callers are updated.
        self.entropy_loss_list = None

        self.entropy_coef = float(config.sac.get('entropy_coef', 0.0))
        normalize_advantage = config.sac.get('normalize_advantage', True)
        if isinstance(normalize_advantage, str):
            normalize_advantage = normalize_advantage.lower() == 'true'
        self.normalize_advantage = normalize_advantage
        self.advantage_norm_epsilon = float(config.sac.get('advantage_norm_epsilon', 1e-5))
        self.gradient_clip_norm = float(config.sac.get('gradient_clip_norm', 0.0))
        self.gae_lambda = float(config.sac.get('gae_lambda', 0.95))
        self.value_coef = float(config.sac.get('value_coef', 0.5))
        self.ppo_clip_param = float(config.sac.get('ppo_clip_param', 0.2))
        self.value_clip_param = float(config.sac.get('value_clip_param', 0.0))
        normalize_rewards = config.sac.get('normalize_rewards', True)
        if isinstance(normalize_rewards, str):
            normalize_rewards = normalize_rewards.lower() == 'true'
        self.normalize_rewards = normalize_rewards
        self.reward_clip = float(config.sac.get('reward_clip', 0.0))
        self.model_dir = model_dir

    @staticmethod
    def _sanitize_tensor(tensor: torch.Tensor, clamp_min=None, clamp_max=None) -> torch.Tensor:
        """Replace NaN/Inf values and optionally clamp the tensor."""

        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e6, neginf=-1e6)
        if clamp_min is not None or clamp_max is not None:
            tensor = tensor.clamp(min=clamp_min, max=clamp_max)
        return tensor

    # ------------------------------------------------------------------
    # Interaction with trainer
    # ------------------------------------------------------------------
    def select_action(self, state, eval_mode=False):
        state = np.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        state = torch.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)
        action, _, mean = self.policy.sample(state, only_mean=eval_mode)
        action_np = action.detach().cpu().numpy()[0]
        mean_np = mean.detach().cpu().numpy()[0]
        return action_np, mean_np

    def master_ask_action(self, master_rref, state, worker_id, eval_mode):
        assert worker_id == self.worker_id
        with torch.no_grad():
            a, mu = self.select_action(state, eval_mode=eval_mode)
        from src.reinforcement_learning.rpc_training.train_rpc import TrainerRPC
        _remote_method(TrainerRPC.report_action, master_rref, a, mu, self.worker_id)

    def master_ask_metrics(self, master_rref):
        """Send latest training metrics to the trainer if available.

        MAT collects policy, value and entropy losses lazily inside
        :meth:`update_parameters`.  During some iterations these values
        might still be ``None`` which previously caused RPC calls with
        missing arguments.  To avoid spurious errors we only perform the
        remote call once all metrics are populated and reset them
        afterwards to prevent re-reporting the same values."""

        if not all(
            v is not None
            for v in (self.policy_loss_list, self.value_loss_list, self.entropy_list)
        ):
            return

        from src.reinforcement_learning.rpc_training.train_rpc import TrainerRPC

        def _coerce(entry, default_value=0.0):
            if isinstance(entry, (tuple, list)) and len(entry) == 2:
                step, value = entry
                if step is None:
                    step = 0
                if value is None:
                    value = default_value
                return step, value
            return 0, default_value

        policy_entry = _coerce(self.policy_loss_list, 0.0)
        value_entry = _coerce(self.value_loss_list, 0.0)
        entropy_entry = _coerce(self.entropy_list, 0.0)

        _remote_method(
            TrainerRPC.report_metrics,
            master_rref,
            self.worker_id,
            value_entry,   # qf1 proxy
            value_entry,   # qf2 proxy
            entropy_entry, # alpha loss proxy
            entropy_entry, # alpha_tlogs proxy
            policy_entry,  # policy loss
        )

        self.policy_loss_list = None
        self.value_loss_list = None
        self.entropy_list = None
        self.entropy_loss_list = None

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def _compute_log_probs(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return log probabilities of ``actions`` under the current policy."""

        mean, log_std = self.policy.forward(states)
        mean = self._sanitize_tensor(mean)
        log_std = self._sanitize_tensor(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = log_std.exp().clamp(min=1e-6)
        normal = Normal(mean, std)

        squashed = self._sanitize_tensor(actions, -0.999, 0.999)
        pre_tanh = 0.5 * (torch.log1p(squashed) - torch.log1p(-squashed))
        log_prob = normal.log_prob(pre_tanh) - torch.log(1 - squashed.pow(2) + epsilon)
        log_prob = self._sanitize_tensor(log_prob)
        return log_prob.sum(-1, keepdim=True)

    def update_parameters(self, memory, batch_size, _total_update, total_step):
        transitions = [transition for transition in getattr(memory, "buffer", []) if transition is not None]
        if not transitions:
            return

        states, actions, rewards, next_states, masks = zip(*transitions)

        states_np = np.nan_to_num(np.stack(states), nan=0.0, posinf=1e6, neginf=-1e6)
        actions_np = np.clip(np.nan_to_num(np.stack(actions), nan=0.0, posinf=1.0, neginf=-1.0), -0.999, 0.999)
        rewards_np = np.nan_to_num(np.array(rewards, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
        next_states_np = np.nan_to_num(np.stack(next_states), nan=0.0, posinf=1e6, neginf=-1e6)
        masks_np = np.nan_to_num(np.array(masks, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)

        states = torch.as_tensor(states_np, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions_np, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards_np, device=self.device)
        next_states = torch.as_tensor(next_states_np, dtype=torch.float32, device=self.device)
        masks = torch.as_tensor(masks_np, dtype=torch.float32, device=self.device)
        if masks.numel() > 0:
            masks[-1] = 0.0

        states = self._sanitize_tensor(states)
        actions = self._sanitize_tensor(actions, -0.999, 0.999)
        rewards = self._sanitize_tensor(rewards)
        next_states = self._sanitize_tensor(next_states)
        masks = self._sanitize_tensor(masks, 0.0, 1.0)

        if self.reward_clip > 0.0:
            rewards = rewards.clamp(-self.reward_clip, self.reward_clip)

        with torch.no_grad():
            values = self.value(states).squeeze(-1)
            next_values = self.value(next_states).squeeze(-1)
            deltas = rewards + self.gamma * next_values * masks - values
            advantages = torch.zeros_like(rewards, device=self.device)
            gae = torch.zeros(1, device=self.device)
            for step in reversed(range(len(transitions))):
                gae = deltas[step] + self.gamma * self.gae_lambda * masks[step] * gae
                advantages[step] = gae
            returns = advantages + values
            old_log_probs = self._compute_log_probs(states, actions)

            values = self._sanitize_tensor(values)
            next_values = self._sanitize_tensor(next_values)
            advantages = self._sanitize_tensor(advantages)
            returns = self._sanitize_tensor(returns)
            old_log_probs = self._sanitize_tensor(old_log_probs)

            if self.normalize_rewards:
                rew_std = rewards.std()
                if torch.isfinite(rew_std) and rew_std.item() > 0:
                    rewards = (rewards - rewards.mean()) / (rew_std + self.advantage_norm_epsilon)
                else:
                    rewards = rewards - rewards.mean()
                deltas = rewards + self.gamma * next_values * masks - values
                advantages.zero_()
                gae.zero_()
                for step in reversed(range(len(transitions))):
                    gae = deltas[step] + self.gamma * self.gae_lambda * masks[step] * gae
                    advantages[step] = gae
                returns = advantages + values

            advantages = self._sanitize_tensor(advantages)
            returns = self._sanitize_tensor(returns)

        if self.normalize_advantage:
            adv_std = advantages.std()
            if torch.isfinite(adv_std) and adv_std.item() > 0:
                advantages = (advantages - advantages.mean()) / (adv_std + self.advantage_norm_epsilon)
            else:
                advantages = advantages - advantages.mean()
        else:
            advantages = advantages - advantages.mean()

        advantages = advantages.unsqueeze(1)
        returns = returns.unsqueeze(1)
        old_values = values.unsqueeze(1).detach()
        old_log_probs = old_log_probs.detach()
        advantages = advantages.detach()
        returns = returns.detach()

        dataset_size = states.size(0)
        if dataset_size == 0:
            return

        batch_size = max(1, min(batch_size, dataset_size))

        policy_loss_acc = 0.0
        value_loss_acc = 0.0
        entropy_acc = 0.0
        updates = 0

        for _ in range(self.updates_per_episode):
            permutation = torch.randperm(dataset_size, device=self.device)
            for start in range(0, dataset_size, batch_size):
                idx = permutation[start:start + batch_size]

                state_batch = states[idx]
                action_batch = actions[idx]
                advantage_batch = advantages[idx]
                return_batch = returns[idx]
                old_log_prob_batch = old_log_probs[idx]
                old_value_batch = old_values[idx]

                mean, log_std = self.policy.forward(state_batch)
                mean = self._sanitize_tensor(mean)
                log_std = self._sanitize_tensor(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
                std = log_std.exp().clamp(min=1e-6)
                normal = Normal(mean, std)
                squashed = self._sanitize_tensor(action_batch, -0.999, 0.999)
                pre_tanh = 0.5 * (torch.log1p(squashed) - torch.log1p(-squashed))
                log_prob = normal.log_prob(pre_tanh) - torch.log(1 - squashed.pow(2) + epsilon)
                log_prob = log_prob.sum(1, keepdim=True)
                log_prob = self._sanitize_tensor(log_prob)
                entropy = -log_prob.mean()

                value_pred = self.value(state_batch)
                if not torch.isfinite(log_prob).all() or not torch.isfinite(value_pred).all():
                    continue
                value_pred = self._sanitize_tensor(value_pred)

                ratio = torch.exp(log_prob - old_log_prob_batch)
                surr1 = ratio * advantage_batch
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_param, 1.0 + self.ppo_clip_param) * advantage_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                if self.entropy_coef != 0.0:
                    policy_loss = policy_loss - self.entropy_coef * entropy

                if self.value_clip_param > 0.0:
                    value_pred_clipped = old_value_batch + (value_pred - old_value_batch).clamp(
                        -self.value_clip_param, self.value_clip_param)
                    value_losses = (value_pred - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    raw_value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    raw_value_loss = F.mse_loss(value_pred, return_batch)
                value_loss = raw_value_loss * self.value_coef

                if not torch.isfinite(policy_loss) or not torch.isfinite(value_loss):
                    continue

                self.policy_optim.zero_grad()
                policy_loss.backward()
                if self.gradient_clip_norm > 0.0:
                    clip_grad_norm_(self.policy.parameters(), self.gradient_clip_norm)
                self.policy_optim.step()

                self.value_optim.zero_grad()
                value_loss.backward()
                if self.gradient_clip_norm > 0.0:
                    clip_grad_norm_(self.value.parameters(), self.gradient_clip_norm)
                self.value_optim.step()

                policy_loss_acc += float(policy_loss.item())
                value_loss_acc += float(raw_value_loss.item())
                entropy_acc += float(entropy.item())
                updates += 1

        if updates:
            avg_policy = policy_loss_acc / updates
            avg_value = value_loss_acc / updates
            avg_entropy = entropy_acc / updates
            self.policy_loss_list = (total_step, avg_policy)
            self.value_loss_list = (total_step, avg_value)
            self.entropy_list = (total_step, avg_entropy)
            self.entropy_loss_list = self.entropy_list
        else:
            self.policy_loss_list = None
            self.value_loss_list = None
            self.entropy_list = None
            self.entropy_loss_list = None

    # ------------------------------------------------------------------
    # Saving and loading
    # ------------------------------------------------------------------
    def save_model(self, experiment_name, episode, modes_controlled, worker_id):
        assert worker_id == self.worker_id
        folder = self.model_dir or os.path.join(self.config.savedir, "output_models", "models_rpc")
        os.makedirs(folder, exist_ok=True)
        actor_path = os.path.join(folder, f"{experiment_name}_worker_{worker_id}_mat_actor_episode_{episode}.pth")
        critic_path = os.path.join(folder, f"{experiment_name}_worker_{worker_id}_mat_value_episode_{episode}.pth")
        torch.save({'model_state_dict': self.policy.state_dict()}, actor_path)
        torch.save({'model_state_dict': self.value.state_dict()}, critic_path)

    def load_policy(self, master_rref, worker_id):
        assert worker_id == self.worker_id
        folder = self.model_dir or os.path.join(self.config.savedir, "output_models", "models_rpc")
        actor_path = os.path.join(folder, f"trainingexperiment_name_worker_{worker_id}_mat_actor_training_episode_500.pth")
        if os.path.exists(actor_path):
            model_dict = torch.load(actor_path)
            self.policy.load_state_dict(model_dict["model_state_dict"])

