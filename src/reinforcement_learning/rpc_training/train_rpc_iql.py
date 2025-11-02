import numpy as np
import torch.distributed.rpc as rpc
from torch.distributed.rpc import rpc_async, rpc_sync, remote

from src.reinforcement_learning.rpc_training.train_rpc import TrainerRPC
from src.reinforcement_learning.rpc_training.helper_rpc.helper_pure_rpc import _call_method
from src.reinforcement_learning.rpc_training.algorithms_rpc.iql import IQLAgent


class TrainerRPCIQL(TrainerRPC):
    """Trainer that orchestrates offline IQL pretraining followed by AWAC fine-tuning."""

    def __init__(self, config_rl, writer_performance, writer_metrics_1, experiment_name, seed, world_size, num_gpus):
        self.offline_phase_episodes = config_rl.sac.get('offline_phase_episodes', 0)
        self.training_phase = 'offline' if self.offline_phase_episodes > 0 else 'online'
        super().__init__(
            config_rl=config_rl,
            writer_performance=writer_performance,
            writer_metrics_1=writer_metrics_1,
            experiment_name=experiment_name,
            seed=seed,
            world_size=world_size,
            num_gpus=num_gpus,
        )
        if self.offline_phase_episodes > 0:
            print(f"[TrainerRPCIQL] Offline phase configured for {self.offline_phase_episodes} episodes")
        else:
            print("[TrainerRPCIQL] No offline phase configured, starting in online AWAC mode")

    def _determine_phase(self) -> str:
        return 'offline' if self.num_episode < self.offline_phase_episodes else 'online'

    def load_soft_actor_critic(self, config_rl):
        self.ag_rrefs = []
        for worker_id in range(1, self.world_size):
            agent_value = self.dictionary_agents[worker_id]
            state_shape = self.get_state_shape_worker(agent_value, config_rl, worker_id)
            print("[TrainerRPCIQL] Worker id {} state shape {}".format(worker_id, state_shape))
            action_shape = np.zeros([agent_value[1] - agent_value[0]])
            ag_info = rpc.get_worker_info("Agent{}".format(worker_id))
            self.ag_rrefs.append(
                remote(
                    ag_info,
                    IQLAgent,
                    kwargs={
                        "num_inputs": state_shape,
                        "action_space": action_shape,
                        "config": config_rl,
                        "rank": worker_id,
                        "num_gpus": self.num_gpus,
                    },
                )
            )

    def load_model_dict(self, config_rl):
        futs = []
        worker_id = 1
        for ag_rreff in self.ag_rrefs:
            futs.append(
                rpc_async(
                    ag_rreff.owner(),
                    _call_method,
                    args=(IQLAgent.load_policy, ag_rreff, self.master_rref, worker_id),
                    timeout=12000,
                )
            )
            worker_id += 1
        for fut in futs:
            fut.wait()

    def manage_saving_networks(self):
        futs = []
        worker_id = 1
        for ag_rreff in self.ag_rrefs:
            futs.append(
                rpc_async(
                    ag_rreff.owner(),
                    _call_method,
                    args=(IQLAgent.save_model, ag_rreff, self.experiment_name, self.num_episode, self.dictionary_agents[worker_id], worker_id),
                    timeout=12000,
                )
            )
            worker_id += 1
        for fut in futs:
            fut.wait()

    def write_update_losses_for_each_agent(self):
        futs = []
        for ag_rreff in self.ag_rrefs:
            futs.append(
                rpc_sync(
                    ag_rreff.owner(),
                    _call_method,
                    args=(IQLAgent.master_ask_metrics, ag_rreff, self.master_rref),
                    timeout=12000,
                )
            )
        for fut in futs:
            fut.wait()

    def choose_action(self, s, eval_mode=False):
        self.current_action = np.zeros(self.env.action_space.shape[0], dtype=np.float32)
        self.current_action_divided = {}
        self.current_mu = np.zeros(self.env.action_space.shape[0], dtype=np.float32)
        futs = []
        worker_id = 1
        for ag_rreff in self.ag_rrefs:
            futs.append(
                rpc_async(
                    ag_rreff.owner(),
                    _call_method,
                    args=(IQLAgent.master_ask_action, ag_rreff, self.master_rref, s[worker_id], worker_id, eval_mode),
                    timeout=12000,
                )
            )
            worker_id += 1
        for fut in futs:
            fut.wait()
        return self.current_action, self.current_action_divided, self.current_mu

    def update_all_agents(self):
        phase = self._determine_phase()
        if phase != self.training_phase:
            print(f"[TrainerRPCIQL] Switching training phase from {self.training_phase} to {phase} at episode {self.num_episode}")
            self.training_phase = phase

        worker_id = 1
        futs = []
        for ag_rreff in self.ag_rrefs:
            futs.append(
                rpc_async(
                    ag_rreff.owner(),
                    _call_method,
                    args=(
                        IQLAgent.update_parameters,
                        ag_rreff,
                        self.memorys_master[worker_id],
                        self.config_rl.sac['batch_size'],
                        self.total_update,
                        self.total_step,
                        phase,
                    ),
                    timeout=12000,
                )
            )
            worker_id += 1
        for fut in futs:
            fut.wait()

        updates_increment = (
            self.config_rl.sac['offline_updates_per_call'] if phase == 'offline' else self.config_rl.sac['updates_per_episode_rpc']
        )
        self.total_update += updates_increment
        for worker_id in range(1, self.world_size):
            self.memorys_master[worker_id].reset()

    def report_metrics(self, worker_id, value_loss_list, q_loss_list, _alpha_loss_list, _alpha_tlogs_list, policy_loss_list):
        if not (value_loss_list and q_loss_list and policy_loss_list):
            return
        total_step_value, value_loss = value_loss_list
        total_step_q, q_loss = q_loss_list
        total_step_policy, policy_loss = policy_loss_list
        self.writer_metrics_1.add_scalar("value_loss/" + str(worker_id), value_loss, total_step_value)
        self.writer_metrics_1.add_scalar("critic_loss/" + str(worker_id), q_loss, total_step_q)
        self.writer_metrics_1.add_scalar("policy_loss/" + str(worker_id), policy_loss, total_step_policy)


