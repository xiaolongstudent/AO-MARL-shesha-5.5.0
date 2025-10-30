import logging
import os
import time

import numpy as np

from src.reinforcement_learning.environment import ao_env
from src.reinforcement_learning.environment.delayed_mdp import DelayedMDP
from src.reinforcement_learning.rpc_training.algorithms_rpc.mat import MAT, DecisionTransformer
from src.reinforcement_learning.rpc_training.algorithms_rpc.replay_memory_rpc import \
            ReplayMemory
from src.reinforcement_learning.rpc_training.helper_rpc.helper_pure_rpc import _call_method
from src.reinforcement_learning.rpc_training.helper_rpc.helper_states import get_modes_chosen
from torch.distributed.rpc import RRef, rpc_async, rpc_sync, remote
import torch.distributed.rpc as rpc

"""
1. worker_id: int
+ id of worker starting from 1 to args.world_size

2. dictionary_agents: dict
+ dictionary of modes controlled per agent
+ key: worker_id
+ value: modes controlled
e.g. 8m 40x40 1281 modes, let say zernike_start_end 0 1200 and world_size 11, 10 workers, 120 modes per worker
+ {1: [0:120]}
+ {2: [120:240]}
+ NOTE: changes in the state via things like window_n_zernike will be taken into account in separate_state part

3. memorys_master: ReplayMemory
+ The memory of 1 episode saved by master
+ It is reset after each episode and only serves the purpose of sending it to the worker at the end of each episode

4. indices_of_state: dict
+ The environment returns an array for (c_t, C_t-1, C_t-2, ...)
+ Indices of state has the indices for each element in this array
e.g. 8m 40x40 n_zernike_start_end 0 1200 without window_n_zernike
+ {"state_dm_before_linear":0:1200}
+ {"state_dm_history_1":1200:2400}
+ {"state_dm_history_2":2400:3600}
+ {"state_d_err": 3600:4800}

5. self.modes_chosen: dict
+ dictionary that has keys: worker_id and values: elements of the state that are to be assigned to the RL agent

6. state_separated, reward_separated: dict, dict
+ each contains in key: worker_id and value states seen by that agent and reward seen by that agent

"""


class TrainerRPC:
    """
    Class that manages everything related to training of the AO RL agent
    """

    _DEFAULT_COORDINATION_FEATURE_DIM = 8

    def __init__(self,
                 config_rl,
                 writer_performance,
                 writer_metrics_1,
                 experiment_name,
                 seed,
                 world_size,
                 num_gpus):

        # 0) a. RPC

        self.savedir = os.path.abspath(config_rl.savedir)
        os.makedirs(self.savedir, exist_ok=True)
        config_rl.savedir = self.savedir

        self.n_filtered = config_rl.env_rl['n_reverse_filtered_from_cmat']
        self.ag_rrefs = []
        self.master_rref = RRef(self)
        self.world_size = world_size
        self.num_gpus = num_gpus
        self.sr_list = []
        # self.save_dict = config_rl.env_rl['save_dict']
        self.model_root = os.path.join(self.savedir, "output_models", "models_rpc")
        self.model_dir = os.path.join(self.model_root, experiment_name)
        os.makedirs(self.model_dir, exist_ok=True)
        self._sr_log_path = os.path.join(self.savedir, "sr_list.npy")

        # Reward shaping defaults.  These are overwritten once the
        # configuration is parsed but having them in place up-front prevents
        # attribute errors when other helper methods run before the config has
        # been fully applied (for instance when legacy code paths access reward
        # helpers during initialisation).
        self._reward_scale = 1.0
        self._reward_clip = 0.0
        self._reward_epsilon = 1e-6
        self._reward_center = 0.0
        self._reward_momentum = 0.0
        self._agent_reward_state = {}
        # ``divide_states_for_agents`` appends a vector of shared coordination
        # statistics (global residual summaries, mean DM commands, Strehl,
        # etc.) to every worker observation.  Some legacy code paths
        # instantiate :class:`TrainerRPC` before the module defining the
        # constant is reloaded which meant ``_coordination_feature_dim`` could
        # be missing entirely.  Prime the attribute up-front so even if other
        # helpers run before we finalise configuration the state shape
        # adjustments have a sensible default.
        self._coordination_feature_dim = getattr(
            self,
            "_coordination_feature_dim",
            self._DEFAULT_COORDINATION_FEATURE_DIM,
        )

        # Default to the non-Decision-Transformer path.  This gets overwritten
        # once the configuration is parsed but having a guard value prevents
        # AttributeError crashes if helper methods are invoked before the
        # algorithm-specific branch runs (for example when legacy entry points
        # still import :mod:`train_rpc_mat` and call reward utilities during
        # initialisation).
        self._is_decision_transformer = False


        # 1) Initializing AO env

        self.env = ao_env.AoEnv(config_rl=config_rl,
                                normalization_bool=True,
                                initial_seed=seed)

        if config_rl.env_rl['gain_change'] > -1:
            print("-CONFIG: Changing gain to:", config_rl.env_rl['gain_change'])
            self.env.supervisor.rtc.set_gain(0, config_rl.env_rl['gain_change'])

        config_rl.original_gain = self.env.supervisor.rtc._rtc.d_control[0].gain

        # Set environment seed given for the experiment

        self.env.set_sim_seed(seed)

        # 2) b. Choose RPC experiment

        self.dictionary_agents, self.total_controlled_modes, self.local_controlled_modes, self.starting_mode,\
            self.total_existing_modes = self.create_agents_dictionary(config_rl)

        # Reward trackers depend on the final agent map.  Recreate the
        # per-worker entries so late configuration changes (for example
        # toggling tip-tilt control) cannot leave us without a bookkeeping
        # slot for a worker that later contributes to the reward signal.
        self._agent_reward_state = {
            worker_id: {
                "prev_residual": None,
                "ema_residual": None,
                "smoothed": 0.0,
                "prev_strehl": None,
                "momentum_reward": 0.0,
            }
            for worker_id in self.dictionary_agents
        }

        self.indices_of_state = self.prepare_indices_of_state()

        self.modes_chosen = get_modes_chosen(self.dictionary_agents, self.indices_of_state, config_rl, self.n_filtered,
                                             experiment_name,
                                             self.total_existing_modes, self.total_controlled_modes, self.starting_mode)

        # 3) Loading transformer-based agent (MAT or Decision Transformer)

        algorithm_name = str(config_rl.algorithm).strip().lower()
        if algorithm_name in ("dt", "decision_transformer", "decision-transformer"):
            self.agent_cls = DecisionTransformer
            self._uses_return_tracking = True
        else:
            self.agent_cls = MAT
            self._uses_return_tracking = False

        self._is_decision_transformer = self._uses_return_tracking

        self.load_transformer_agents(config_rl)

        load_prev = config_rl.env_rl.get('load_previous_weights', False)
        if isinstance(load_prev, str):
            load_prev = load_prev.lower() == 'true'
        if load_prev:
            self.load_agent_models(config_rl)

        # 4) Loading Replay Memory

        self.memorys_master = {}
        for worker_id in range(1, world_size):
            self.memorys_master[worker_id] = ReplayMemory(config_rl.env_rl['max_steps_per_episode'])

        # 5) Load pretrained weights for the chosen transformer agent

        if config_rl.sac['pretrained_model_path'] is not None:


            raise NotImplementedError

        # Other initializations that are needed
        self.writer_performance, self.writer_metrics_1 = writer_performance, writer_metrics_1
        self.config_rl, self.experiment_name = config_rl, experiment_name
        self.initial_seed, self.seed = seed, seed
        self.len_actions = len(self.env.action_space.sample())
        self.total_update, self.total_step = 0, 0
        self.num_test_episode, self.num_episode = 0, 0
        self.delayed_mdp_object = None
        self.max_num_steps = 1e6  #最大步长
        self.save_networks_every_episodes = 50

        print("-----------------------------RPC TRAINING-----------------------------")

        print("\n Shape of state:", self.env.observation_space,
              "\n Original gain:", self.env.supervisor.rtc._rtc.d_control[0].gain,
              "\n Action shape:", self.env.action_space.sample().shape[0])

        print("Dictionary agents {} Total modes {} Local modes {} Total existing modes {}"
              .format(self.dictionary_agents, self.total_controlled_modes,
                      self.local_controlled_modes, self.total_existing_modes))

        self.current_action = np.zeros(self.env.action_space.shape[0], dtype=np.float32)
        self.current_action_divided = {}
        self.current_mu = np.zeros(self.env.action_space.shape[0], dtype=np.float32)

        self.current_r0 = self.env.supervisor.config.p_atmos.r0
        self.current_windspeed_layer_0 = self.env.supervisor.config.p_atmos.windspeed[0]
        # Track Strehl measurements to keep logging robust when the backend
        # momentarily fails to estimate the ratio.
        self._last_safe_strehl = (0.0, 0.0, 0.0, 0.0)
        self._strehl_warning_emitted = False
        self._latest_strehl = 0.0
        # Number of coordination features appended to each agent observation to
        # provide awareness of the global DM activity and residual statistics.
        self._coordination_feature_dim = self._DEFAULT_COORDINATION_FEATURE_DIM

        sac_cfg = self.config_rl.sac

        def _as_float(value, default=0.0):
            if value is None:
                return default
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        self._reward_scale = _as_float(sac_cfg.get('reward_scale'), 1.0)
        self._reward_clip = abs(_as_float(sac_cfg.get('reward_clip'), 0.0))
        self._reward_epsilon = max(_as_float(sac_cfg.get('reward_epsilon'), 1e-6), 1e-12)
        self._reward_residual_weight = _as_float(
            sac_cfg.get('reward_residual_weight'),
            0.5,
        )
        self._reward_strehl_weight = _as_float(
            sac_cfg.get('reward_strehl_weight'),
            0.2,
        )
        self._reward_delta_weight = _as_float(
            sac_cfg.get('reward_delta_weight'),
            0.0,
        )
        self._reward_residual_smoothing = float(
            self.config_rl.env_rl.get('reward_residual_smoothing', 0.0)
        )
        self._reward_residual_smoothing = min(max(self._reward_residual_smoothing, 0.0), 0.99)

        if self._is_decision_transformer:
            self._reward_scale = _as_float(
                sac_cfg.get('dt_reward_scale'),
                self._reward_scale,
            )
            self._reward_clip = abs(_as_float(
                sac_cfg.get('dt_reward_clip'),
                self._reward_clip,
            ))
            self._reward_center = _as_float(
                sac_cfg.get('dt_reward_center'),
                0.0,
            )
            self._reward_momentum = min(
                max(_as_float(sac_cfg.get('dt_reward_momentum'), 0.0), 0.0),
                0.99,
            )
            self._reward_residual_weight = _as_float(
                sac_cfg.get('dt_residual_weight'),
                self._reward_residual_weight,
            )
            self._reward_strehl_weight = _as_float(
                sac_cfg.get('dt_strehl_weight'),
                self._reward_strehl_weight,
            )
            self._reward_delta_weight = _as_float(
                sac_cfg.get('dt_reward_delta_weight'),
                self._reward_delta_weight,
            )
            self._reward_residual_smoothing = min(
                max(
                    _as_float(
                        sac_cfg.get('dt_reward_smoothing'),
                        0.0,
                    ),
                    0.0,
                ),
                0.99,
            )

        # ``_agent_reward_state`` has already been aligned with
        # ``dictionary_agents`` above.  Here we only make sure the helper
        # attributes respect the configuration values.

    def write_test_performances(self, rl_performance_dict, linear_performance_dict, geo_performance_dict):
        """
        agent_performance = {"r_per_agent_test": r_per_agent_test,
                             "r_total_test": r_total_test,
                             "sr_le_test": sr_le_test,
                             "sr_se_test": sr_se_test}

        geometric_performance = {"r_geo_per_agent_test": r_geo_per_agent_test,
                                 "r_geo_total_test": r_geo_total_test,
                                 "sr_le_geo_test": sr_le_geo_test}
        """

        if self.num_episode % 100 == 0:
            for i in range(len(rl_performance_dict['r_per_agent_test'])):
                worker_id = i + 1
                self.writer_performance.add_scalar("Evaluation_Agent_Rewards/RL_worker_" + str(worker_id),
                                                   rl_performance_dict['r_per_agent_test'][i],
                                                   self.total_step)
                self.writer_performance.add_scalar("Evaluation_Agent_Rewards/Linear_worker_" + str(worker_id),
                                                   linear_performance_dict['r_per_agent_test'][i],
                                                   self.total_step)
                self.writer_performance.add_scalar("Evaluation_Agent_Rewards/Error_worker_" + str(worker_id),
                                                   rl_performance_dict['r_per_agent_test'][i] -
                                                   linear_performance_dict['r_per_agent_test'][i],
                                                   self.total_step)

                # Geometric
                if len(self.env.supervisor.config.p_controllers) > 1:
                    self.writer_performance.add_scalar("Evaluation_Agent_Rewards_Geometric/RL_worker_"
                                                       + str(worker_id),
                                                       rl_performance_dict['r_per_agent_test'][i],
                                                       self.total_step)
                    self.writer_performance.add_scalar("Evaluation_Agent_Rewards_Geometric/Geo_worker_"
                                                       + str(worker_id),
                                                       geo_performance_dict['r_geo_per_agent_test'][i],
                                                       self.total_step)
                    self.writer_performance.add_scalar("Evaluation_Agent_Rewards_Geometric/Error_geo_worker_"
                                                       + str(worker_id),
                                                       rl_performance_dict['r_per_agent_test'][i] -
                                                       geo_performance_dict['r_geo_per_agent_test'][i],
                                                       self.total_step)

            self.write_update_losses_for_each_agent()

        self.writer_performance.add_scalar("Evaluation_CR/RL_CR", rl_performance_dict['r_total_test'],
                                           self.total_step)
        self.writer_performance.add_scalar('Evaluation_CR/Linear_CR', linear_performance_dict['r_total_test'],
                                           self.total_step)
        self.writer_performance.add_scalar('Evaluation_CR/Error_CR', rl_performance_dict['r_total_test'] -
                                           linear_performance_dict['r_total_test'], self.total_step)

        self.writer_performance.add_scalar("Evaluation_Strehl_LE/RL_SR_LE", rl_performance_dict['sr_le_test'],
                                           self.total_step)
        self.writer_performance.add_scalar('Evaluation_Strehl_LE/Linear_SR_LE',
                                           linear_performance_dict['sr_le_test'], self.total_step)
        self.writer_performance.add_scalar('Evaluation_Strehl_LE/Error_SR_LE', rl_performance_dict['sr_le_test']
                                           - linear_performance_dict['sr_le_test'], self.total_step)

        # Geometric LE
        if len(self.env.supervisor.config.p_controllers) > 1:
            self.writer_performance.add_scalar("Evaluation_Strehl_LE/GEO_SR_LE", geo_performance_dict['sr_le_geo_test'],
                                               self.total_step)

        self.writer_performance.add_scalar("Evaluation_Strehl_SE/RL_SR_SE", rl_performance_dict['sr_se_test'],
                                           self.total_step)
        self.writer_performance.add_scalar('Evaluation_Strehl_SE/Linear_SR_SE',
                                           linear_performance_dict['sr_se_test'], self.total_step)
        self.writer_performance.add_scalar('Evaluation_Strehl_SE/Error_SR_SE', rl_performance_dict['sr_se_test'] -
                                           linear_performance_dict['sr_se_test'],
                                           self.total_step)
        self.sr_list.append(rl_performance_dict['sr_se_test'])
        np.save(self._sr_log_path, self.sr_list)



    def manage_saving_networks(self):

        futs = []
        worker_id = 1
        for ag_rreff in self.ag_rrefs:
            futs.append(
                rpc_async(
                    ag_rreff.owner(),
                    _call_method,
                    args=(
                        self.agent_cls.save_model,
                        ag_rreff,
                        self.experiment_name,
                        self.num_episode,
                        self.dictionary_agents[worker_id],
                        worker_id,
                    ),
                    timeout=12000,
                )
            )
            worker_id += 1

        for fut in futs:
            fut.wait()

    def create_agents_dictionary_tt_treated_as_mode(self, config_rl):

        total_existing_modes = self.env.supervisor.modes2volts.shape[1]
        total_controlled_modes =\
            config_rl.env_rl['n_zernike_start_end'][1] - config_rl.env_rl['n_zernike_start_end'][0] + 2
        local_controlled_modes = int(total_controlled_modes / (self.world_size - 1))
        assert total_controlled_modes % (self.world_size - 1) == 0
        assert config_rl.env_rl['n_zernike_start_end'][0] > -1 and config_rl.env_rl['n_zernike_start_end'][1] > -1
        assert total_controlled_modes > 0
        assert config_rl.env_rl['window_n_zernike'] == -1

        dictionary_agents = {}
        worker_id = 1

        # 0-28 if 30 modes per agent
        modes = np.arange(config_rl.env_rl['n_zernike_start_end'][0],
                          config_rl.env_rl['n_zernike_start_end'][0] + local_controlled_modes - 2)
        # 2 last
        tt = np.arange(total_existing_modes - 2, total_existing_modes)

        dictionary_agents[worker_id] = np.concatenate([tt, modes])
        # agent 1 that has TT and initial modes
        # other agents will controll other
        worker_id = 2
        for modes in range(config_rl.env_rl['n_zernike_start_end'][0] + local_controlled_modes - 2,
                           config_rl.env_rl['n_zernike_start_end'][1],
                           local_controlled_modes):
            dictionary_agents[worker_id] = np.arange(modes, modes + local_controlled_modes)
            worker_id += 1

        return dictionary_agents, total_controlled_modes, local_controlled_modes,  total_existing_modes

    def create_agents_dictionary_original(self, config_rl):
        total_existing_modes = self.env.supervisor.modes2volts.shape[1]
        total_controlled_modes = config_rl.env_rl['n_zernike_start_end'][1] - config_rl.env_rl['n_zernike_start_end'][0]
        if config_rl.env_rl['include_tip_tilt']:
            # 1 worker will be TT, the other local controlled modes
            local_controlled_modes = int(total_controlled_modes / (self.world_size - 2))
            assert total_controlled_modes % (self.world_size - 2) == 0
        else:
            # all workers will be local controlled modes
            local_controlled_modes = int(total_controlled_modes / (self.world_size - 1))
            assert total_controlled_modes % (self.world_size - 1) == 0
        # assert we have zernike start end
        assert config_rl.env_rl['n_zernike_start_end'][0] > -1 and config_rl.env_rl['n_zernike_start_end'][1] > -1
        assert total_controlled_modes > 0

        dictionary_agents = {}
        # id 0: Compass
        # worker_id 1, 2, 3, 4...: Agents
        worker_id = 1
        for modes in range(config_rl.env_rl['n_zernike_start_end'][0],
                           config_rl.env_rl['n_zernike_start_end'][1],
                           local_controlled_modes):
            dictionary_agents[worker_id] = [modes, modes + local_controlled_modes]

            worker_id += 1

        if config_rl.env_rl['include_tip_tilt']:
            dictionary_agents[worker_id] = [total_existing_modes - 2, total_existing_modes]
            total_controlled_modes += 2

        return dictionary_agents, total_controlled_modes, local_controlled_modes, total_existing_modes

    def create_agents_dictionary(self, config_rl):

        if config_rl.env_rl['tt_treated_as_mode']:
            dictionary_agents, total_controlled_modes, local_controlled_modes, total_existing_modes =\
                self.create_agents_dictionary_tt_treated_as_mode(config_rl)
        else:
            dictionary_agents, total_controlled_modes, local_controlled_modes, total_existing_modes =\
                self.create_agents_dictionary_original(config_rl)

        return dictionary_agents, total_controlled_modes, local_controlled_modes,\
               config_rl.env_rl['n_zernike_start_end'][0], total_existing_modes

    def get_state_shape_worker(self, agent_value, config_rl, worker_id):
        """
        Gets state shape for the current worker_id
        """
        state_multiplier = (int(config_rl.env_rl['state_dm_residual']) +
                            int(config_rl.env_rl['state_dm_after_linear']) +
                            int(config_rl.env_rl['state_dm_before_linear']) +
                            config_rl.env_rl['number_of_previous_dm'] +
                            config_rl.env_rl['number_of_previous_dm_residuals'])

        original_state_shape = (agent_value[1] - agent_value[0]) * state_multiplier

        if config_rl.env_rl['window_n_zernike'] > -1:
            # worker_id == len(self.dictionary_agents) implies TT
            if config_rl.env_rl['include_tip_tilt'] and worker_id == len(self.dictionary_agents):
                state_shape = original_state_shape
                if config_rl.env_rl['include_tip_tilt_windowed']:
                    state_shape = original_state_shape + int(2*config_rl.env_rl['window_n_zernike']) * state_multiplier
            else:
                state_shape = original_state_shape + int(
                    config_rl.env_rl['window_n_zernike'] * 2) * state_multiplier  # TODO generalize the 4
        else:
            state_shape = original_state_shape

        state_shape += self._coordination_feature_dim

        return state_shape

    def load_agent_models(self, config_rl):
        """Load previously saved transformer-based policies."""
        futs = []
        worker_id = 1
        for ag_rreff in self.ag_rrefs:
            # make async RPC to kick off an episode on all observers
            futs.append(
                rpc_async(
                    ag_rreff.owner(),
                    _call_method,
                    args=(self.agent_cls.load_policy, ag_rreff, self.master_rref, worker_id),
                    timeout=12000
                )

            )
            worker_id += 1

        for fut in futs:
            fut.wait()
            
        

    def load_transformer_agents(self, config_rl):
        """Create remote transformer agents (MAT or Decision Transformer)."""
        for worker_id in range(1, self.world_size):
            agent_value = self.dictionary_agents[worker_id]

            state_shape = self.get_state_shape_worker(agent_value, config_rl, worker_id)

            print("Worker id {} state shape {}".format(worker_id, state_shape))

            action_shape = np.zeros([agent_value[1] - agent_value[0]])

            ag_info = rpc.get_worker_info("Agent{}".format(worker_id))
            worker_model_dir = os.path.join(self.model_dir, f"worker_{worker_id}")
            os.makedirs(worker_model_dir, exist_ok=True)

            agent_rref = remote(
                ag_info,
                self.agent_cls,
                kwargs={
                    "num_inputs": state_shape,
                    "action_space": action_shape,
                    "config": config_rl,
                    "rank": worker_id,
                    "num_gpus": self.num_gpus,
                    "model_dir": worker_model_dir,
                },
            )
            self.ag_rrefs.append(agent_rref)

            if self._uses_return_tracking:
                try:
                    summary = rpc_sync(
                        ag_info,
                        _call_method,
                        args=(self.agent_cls.get_offline_summary, agent_rref),
                        timeout=120,
                    )
                except Exception:
                    summary = None
                self._log_offline_dataset_summary(worker_id, summary)

    def _log_offline_dataset_summary(self, worker_id, summary):
        if not summary:
            return

        num_eps = summary.get("num_eps", 0)
        if num_eps == 0:
            print(
                f"[DecisionTransformer][worker {worker_id}] No offline episodes were loaded."
            )
            return

        ret_mean = summary.get("ret_mean", 0.0)
        ret_std = summary.get("ret_std", 0.0)
        top20_ratio = summary.get("top20_ratio", 0.0)
        strehl_mean = summary.get("strehl_mean")
        strehl_std = summary.get("strehl_std")

        message = (
            f"[DecisionTransformer][worker {worker_id}] offline episodes={num_eps} "
            f"return_mean={ret_mean:.3f} return_std={ret_std:.3f} top20_ratio={top20_ratio:.3f}"
        )
        if strehl_mean is not None:
            message += f" strehl_mean={strehl_mean:.4f}"
        if strehl_std is not None:
            message += f" strehl_std={strehl_std:.4f}"
        print(message)

        if self.writer_metrics_1 is not None:
            step = getattr(self, "total_step", 0)
            prefix = f"Offline/worker{worker_id}"
            try:
                self.writer_metrics_1.add_scalar(f"{prefix}/return_mean", ret_mean, step)
                self.writer_metrics_1.add_scalar(f"{prefix}/return_std", ret_std, step)
                self.writer_metrics_1.add_scalar(f"{prefix}/top20_ratio", top20_ratio, step)
                if strehl_mean is not None:
                    self.writer_metrics_1.add_scalar(f"{prefix}/strehl_mean", strehl_mean, step)
                if strehl_std is not None:
                    self.writer_metrics_1.add_scalar(f"{prefix}/strehl_std", strehl_std, step)
                quantiles = summary.get("q", {})
                for key, value in quantiles.items():
                    self.writer_metrics_1.add_scalar(
                        f"{prefix}/return_q{key}", value, step
                    )
            except AttributeError:
                # Some writer implementations might not expose ``add_scalar``
                # (for instance when running in a minimal evaluation setup).
                pass

    def prepare_indices_of_state(self):
        """
        Prepares the dictionary self.indices_of_state.
        For each key in the state usually e.g. (s_wfs, s_dm_before_linear, s_dm_after_linear, ...)
        We have a list of two elements:
        · The first element is the starting point of that part of the state
        · The second element is the ending por of that part of the state
        """

        s = self.env.reset(return_dict=True)

        indices_of_state = dict()
        initial_index = 0
        for key in s.keys():
            indices_of_state[key] = [initial_index, initial_index + s[key].shape[0]]
            initial_index += s[key].shape[0]

        return indices_of_state

    def divide_rewards_for_agents_geometric(self, geometric_modes):
        """
        :geometric_modes: array of geometric_modes (N steps x modes)
        Creates a list that contains in an orderly manner the rewards for each agent for the geometric controller
        :return: reward_list
        """

        residuals = geometric_modes[1:, :]-geometric_modes[:-1, :]

        reward = np.sum(np.square(residuals), axis=0)

        if self.config_rl.env_rl['reward_type'] != "avg_squared_modes":
            factor = float(self.config_rl.env_rl['reward_type'].split("_")[-1])
            separated_reward = {worker_id: -factor*np.average(reward[agent_value[0]:agent_value[1]])
                                for (worker_id, agent_value) in self.dictionary_agents.items()}
        else:
            separated_reward = {worker_id: -np.sum(reward[agent_value[0]:agent_value[1]])
                                for (worker_id, agent_value) in self.dictionary_agents.items()}

        return list(separated_reward.values())

    def divide_rewards_for_agents(self):
        """
        Creates a list that contains in an orderly manner the rewards for each agent
        :return: reward_list
        """

        s_dm_residual_modes = self.env.supervisor.volts2modes.dot(
            self.env.supervisor.rtc.get_err(0)
        )
        residual_energy = np.square(np.nan_to_num(s_dm_residual_modes, nan=0.0))
        rewards = {}
        residual_smoothing = self._reward_residual_smoothing
        scale = self._reward_scale
        epsilon = self._reward_epsilon
        clip = self._reward_clip
        strehl_weight = self._reward_strehl_weight
        delta_weight = self._reward_delta_weight
        residual_weight = self._reward_residual_weight
        sr_values = self._safe_get_strehl()
        if len(sr_values) == 0:
            strehl_current = 0.0
        elif len(sr_values) > 1:
            strehl_current = float(sr_values[1])
        else:
            strehl_current = float(sr_values[0])

        center = self._reward_center if self._is_decision_transformer else 0.0
        momentum = self._reward_momentum if self._is_decision_transformer else 0.0

        # For the Decision Transformer we focus on Strehl improvements rather
        # than the absolute Strehl level to avoid starting each episode with a
        # large positive reward simply because the optics already operate at a
        # reasonable baseline.  MASAC/MAT users can still keep the absolute
        # Strehl contribution via configuration.
        strehl_gain_weight = 0.0
        strehl_absolute_weight = strehl_weight
        if self._is_decision_transformer:
            strehl_absolute_weight = 0.0
            strehl_gain_weight = strehl_weight

        self._latest_strehl = strehl_current

        for worker_id, agent_value in self.dictionary_agents.items():
            agent_state = self._agent_reward_state.get(worker_id)
            if agent_state is None:
                agent_state = self._agent_reward_state[worker_id] = {
                    "prev_residual": None,
                    "ema_residual": None,
                    "smoothed": 0.0,
                    "prev_strehl": None,
                    "momentum_reward": 0.0,
                }

            mean_residual = self._mean_residual_for_agent(agent_value, residual_energy)
            if mean_residual is None:
                rewards[worker_id] = 0.0
                agent_state["prev_residual"] = None
                agent_state["smoothed"] = 0.0
                agent_state["ema_residual"] = None
                agent_state["prev_strehl"] = strehl_current
                agent_state["momentum_reward"] = center
                continue

            previous_ema = agent_state.get("ema_residual")
            baseline = previous_ema if previous_ema is not None else mean_residual

            if residual_smoothing > 0.0:
                ema_residual = (
                    residual_smoothing * baseline
                    + (1.0 - residual_smoothing) * mean_residual
                )
            else:
                ema_residual = mean_residual

            improvement = baseline - ema_residual
            denom = abs(baseline) + epsilon
            if denom > 0.0:
                improvement /= denom
            else:
                improvement = 0.0

            prev_strehl = agent_state.get("prev_strehl")
            strehl_delta = 0.0
            if prev_strehl is not None:
                strehl_delta = strehl_current - prev_strehl

            if prev_strehl is None:
                strehl_delta_norm = 0.0
            else:
                strehl_scale = max(abs(prev_strehl), abs(strehl_current), epsilon)
                strehl_delta_norm = strehl_delta / strehl_scale

            strehl_gain_norm = 0.0
            if self._is_decision_transformer:
                baseline_strehl = max(abs(getattr(self, "_episode_strehl_baseline", strehl_current)), epsilon)
                strehl_gain = strehl_current - getattr(self, "_episode_strehl_baseline", strehl_current)
                strehl_gain_norm = strehl_gain / baseline_strehl

            reward_raw = center
            if residual_weight != 0.0:
                reward_raw += residual_weight * improvement
            if strehl_absolute_weight != 0.0:
                reward_raw += strehl_absolute_weight * strehl_current
            if strehl_gain_weight != 0.0 and strehl_gain_norm != 0.0:
                reward_raw += strehl_gain_weight * strehl_gain_norm
            if delta_weight != 0.0 and strehl_delta_norm != 0.0:
                reward_raw += delta_weight * strehl_delta_norm

            if momentum > 0.0:
                prev_momentum = agent_state.get("momentum_reward", center)
                reward_raw = momentum * prev_momentum + (1.0 - momentum) * reward_raw

            reward_value = reward_raw * scale
            if clip > 0.0:
                reward_value = float(np.clip(reward_value, -clip, clip))

            rewards[worker_id] = reward_value

            if agent_state is not None:
                agent_state["prev_residual"] = mean_residual
                agent_state["ema_residual"] = ema_residual
                agent_state["smoothed"] = reward_value
                agent_state["prev_strehl"] = strehl_current
                agent_state["momentum_reward"] = reward_raw

        return rewards

    def _safe_get_strehl(self, tar_index=0):
        """Return Strehl metrics while shielding against estimator failures.

        Compass occasionally fails to fit the PSF and raises a low-level
        exception that bubbles up as soon as we query ``get_strehl``.  That
        failure manifested as noisy "can not estimate the SR" messages after
        every episode once we started querying the Strehl ratio for reward
        shaping.  To keep the training loop robust we try to read the Strehl
        without fitting the PSF; if the backend still raises, we fall back to
        the last valid reading and warn just once.
        """

        try:
            values = self.env.supervisor.target.get_strehl(tar_index, do_fit=False)
        except Exception as exc:  # pragma: no cover - backend specific
            if not self._strehl_warning_emitted:
                logging.warning(
                    "Failed to estimate Strehl for target %s: %s", tar_index, exc
                )
                self._strehl_warning_emitted = True
            values = self._last_safe_strehl
        else:
            values = tuple(float(np.nan_to_num(val, nan=0.0)) for val in values)
            self._last_safe_strehl = values
            self._strehl_warning_emitted = False

        return values

    def _reset_reward_tracking(self):
        """Reset reward bookkeeping before starting a new episode."""

        self._strehl_warning_emitted = False

        residual_modes = self.env.supervisor.volts2modes.dot(
            self.env.supervisor.rtc.get_err(0)
        )
        residual_energy = np.square(np.nan_to_num(residual_modes, nan=0.0))

        sr_values = self._safe_get_strehl()
        if len(sr_values) == 0:
            strehl_current = 0.0
        elif len(sr_values) > 1:
            strehl_current = float(sr_values[1])
        else:
            strehl_current = float(sr_values[0])

        center = self._reward_center if self._is_decision_transformer else 0.0

        self._episode_strehl_baseline = strehl_current

        for worker_id, agent_value in self.dictionary_agents.items():
            worker_state = self._agent_reward_state.get(worker_id)
            if worker_state is None:
                worker_state = self._agent_reward_state[worker_id] = {
                    "prev_residual": None,
                    "ema_residual": None,
                    "smoothed": 0.0,
                    "prev_strehl": None,
                    "momentum_reward": center,
                }
            baseline = self._mean_residual_for_agent(agent_value, residual_energy)
            worker_state["prev_residual"] = baseline
            worker_state["ema_residual"] = baseline
            worker_state["smoothed"] = 0.0
            worker_state["prev_strehl"] = strehl_current
            worker_state["momentum_reward"] = center

    @staticmethod
    def _mean_residual_for_agent(agent_value, residual_energy):
        if isinstance(agent_value, (list, tuple)):
            if len(agent_value) == 2 and all(isinstance(v, (int, np.integer)) for v in agent_value):
                start, end = agent_value
                if end <= start:
                    return None
                return float(np.mean(residual_energy[start:end]))
            agent_indices = np.asarray(agent_value, dtype=int)
        else:
            agent_indices = np.asarray(agent_value, dtype=int)

        if agent_indices.size == 0:
            return None
        return float(np.mean(residual_energy[agent_indices]))

    def divide_states_for_agents(self, state):
        """
        Creates a list that contains in an orderly manner the states for each agent
        :return: divided_state
        """
        divided_states = {}

        residual_modes = self.env.supervisor.volts2modes.dot(
            self.env.supervisor.rtc.get_err(0)
        )
        residual_energy = np.square(np.nan_to_num(residual_modes, nan=0.0))
        total_residual_count = residual_energy.size
        if total_residual_count > 0:
            global_residual_mean = float(np.mean(residual_energy))
            global_residual_std = float(np.std(residual_energy))
            total_residual_sum = float(np.sum(residual_energy))
        else:
            global_residual_mean = 0.0
            global_residual_std = 0.0
            total_residual_sum = 0.0

        if self.current_action.size:
            action_vector = self.current_action
        else:
            action_vector = np.zeros(1, dtype=np.float32)
        action_count = action_vector.size
        global_action_mean = float(np.mean(action_vector))
        global_action_std = float(np.std(action_vector))
        total_action_sum = float(np.sum(action_vector))

        strehl_current = float(getattr(self, "_latest_strehl", 0.0))

        for worker_id in range(1, self.world_size):
            base_slice = self.modes_chosen[worker_id]
            base_state = state[base_slice]
            base_state = np.atleast_1d(np.asarray(base_state, dtype=np.float32))

            agent_value = self.dictionary_agents[worker_id]
            start_idx = max(0, int(agent_value[0]))
            end_idx = min(total_residual_count, int(agent_value[1]))
            if end_idx > start_idx:
                agent_residual = residual_energy[start_idx:end_idx]
                agent_residual_mean = float(np.mean(agent_residual))
                agent_residual_sum = float(np.sum(agent_residual))
                other_count_residual = total_residual_count - agent_residual.size
                if other_count_residual > 0:
                    other_residual_mean = float(
                        (total_residual_sum - agent_residual_sum) / other_count_residual
                    )
                else:
                    other_residual_mean = agent_residual_mean
            else:
                agent_residual_mean = 0.0
                other_residual_mean = global_residual_mean

            bottom_mode_for_array, top_mode_for_array = self.select_correct_modes_for_array(
                agent_value[0], agent_value[1]
            )
            bottom_mode_for_array = max(0, int(bottom_mode_for_array))
            top_mode_for_array = min(action_count, int(top_mode_for_array))
            if top_mode_for_array > bottom_mode_for_array:
                agent_actions = action_vector[bottom_mode_for_array:top_mode_for_array]
                agent_action_sum = float(np.sum(agent_actions))
                agent_action_mean = float(np.mean(agent_actions))
                other_action_count = action_count - agent_actions.size
                if other_action_count > 0:
                    other_action_mean = float(
                        (total_action_sum - agent_action_sum) / other_action_count
                    )
                else:
                    other_action_mean = agent_action_mean
            else:
                agent_action_mean = 0.0
                other_action_mean = global_action_mean

            coordination_features = np.array(
                [
                    global_residual_mean,
                    global_residual_std,
                    agent_residual_mean,
                    other_residual_mean,
                    global_action_mean,
                    global_action_std,
                    other_action_mean,
                    strehl_current,
                ],
                dtype=np.float32,
            )

            divided_states[worker_id] = np.concatenate(
                [base_state, coordination_features],
                axis=0,
            )

        return divided_states

    def manage_changing_conditions(self):
        if self.config_rl.env_rl['change_atmospheric_3_layers_1'] and self.num_episode == 1000:
            # From wind direction 0 0 0 to 0 15 30
            self.env.supervisor.atmos.set_wind(screen_index=1, winddir=15)
            self.env.supervisor.atmos.set_wind(screen_index=2, winddir=30)
        elif self.config_rl.env_rl['change_atmospheric_3_layers_2'] and self.num_episode == 1000:
            # From wind speed 2 to wind speed 1
            self.env.supervisor.atmos.set_wind(screen_index=0, windspeed=15)
            self.env.supervisor.atmos.set_wind(screen_index=1, windspeed=10)
            self.env.supervisor.atmos.set_wind(screen_index=2, windspeed=20)
        elif self.config_rl.env_rl['change_atmospheric_3_layers_3'] and self.num_episode == 1000:
            # From r0 0.16 to 0.08
            self.env.supervisor.atmos.set_r0(0.08)
        elif self.config_rl.env_rl['change_atmospheric_3_layers_4'] and self.num_episode == 1000:
            # From wind speed 2 to wind speed 1
            self.env.supervisor.atmos.set_wind(screen_index=0, windspeed=10)
            self.env.supervisor.atmos.set_wind(screen_index=1, windspeed=5)
            self.env.supervisor.atmos.set_wind(screen_index=2, windspeed=15)
        elif self.config_rl.env_rl['change_atmospheric_3_layers_5'] and self.num_episode == 1000:
            # From wind direction 0 0 0 to 0 45 90
            self.env.supervisor.atmos.set_wind(screen_index=1, winddir=45)
            self.env.supervisor.atmos.set_wind(screen_index=2, winddir=90)
    def set_seed(self, seed):
        
        self.env.set_sim_seed(seed)

    def train_agent(self):
        """
        Method that manages the full training loop of the agent
        """

        while True:
            # An episode of the environment
            r_total = self.episode()

            sr_values = self._safe_get_strehl()
            if len(sr_values) > 1:
                self.writer_performance.add_scalar(
                    "Training_Reward/Evolution of SR LE",
                    sr_values[1],
                    self.num_episode,
                )
            self.writer_performance.add_scalar(
                "Training_Reward/Average Reward of last 10 episodes",
                r_total,
                self.num_episode,
            )

            if (self.config_rl.env_rl['change_atmospheric_3_layers_1'] or
                self.config_rl.env_rl['change_atmospheric_3_layers_2'] or
                self.config_rl.env_rl['change_atmospheric_3_layers_3'] or
                self.config_rl.env_rl['change_atmospheric_3_layers_4'] or
                self.config_rl.env_rl['change_atmospheric_3_layers_5'])\
                    and self.num_episode >= 1000:
                if self.num_episode == 1000 \
                        or self.num_episode == 1001\
                        or self.num_episode == 1002:
                    num_test = 1
                elif 1002 < self.num_episode <= 1100:
                    num_test = 10
                else:
                    num_test = 50
            elif self.config_rl.env_rl['do_more_evaluations']:
                num_test = 5
            else:
                num_test = 50
            if self.num_episode % num_test == 0:
                self.seed += 1
                self.env.set_sim_seed(self.seed)
                rl_performance_dict, geo_performance_dict = self.test_episode(controller="RL")
                linear_performance_dict, _ = self.test_episode(controller="Integrator")
                self.write_test_performances(rl_performance_dict, linear_performance_dict, geo_performance_dict)

            if self.num_episode % self.save_networks_every_episodes == 0 and self.num_episode > 50:
                self.manage_saving_networks()

            self.seed += 1
            self.env.set_sim_seed(self.seed)

            self.manage_changing_conditions()

            if self.total_step > self.max_num_steps:
                break

    def episode(self):
        """
        Does an episode of the environment
        The episode definition depends on the configuration
        """

        step, r_total, done, s, start_time = 0, 0, False, self.env.reset(), time.time()

        # Reset the reward baseline so the first step in the episode is
        # measured relative to the initial residual energy.
        self._reset_reward_tracking()

        self.delayed_mdp_object = DelayedMDP(self.config_rl.env_rl['delayed_assignment'],
                                             self.config_rl.env_rl['modification_online'])

        begin_futs = []
        for ag_rreff in self.ag_rrefs:
            begin_futs.append(
                rpc_async(
                    ag_rreff.owner(),
                    _call_method,
                    args=(self.agent_cls.begin_episode, ag_rreff),
                    timeout=12000,
                )
            )
        for fut in begin_futs:
            fut.wait()

        for _ in range(self.config_rl.env_rl['max_steps_per_episode']):

            # 0. Divided states for agents
            s_divided = self.divide_states_for_agents(s)

            # 1. Choose action based on state
            a, a_divided, mu = self.choose_action(s_divided)

            # 2. Step on the environment
            s_next, reward_divided, done = self.env_step(a)

            # 3. Divided states for agents
            s_next_divided = self.divide_states_for_agents(s_next)

            # 4. if the delayed_mdp is ready
            # Save on replay (s, a, r, s_next) which comes from delayed_mdp and the s_next and reward this timestep
            if self.delayed_mdp_object.check_update_possibility():
                self.manage_memory(reward_divided, done)

            # 5. Save s, a, s_next, r, a_next to do the correct credit assignment in replay memory later
            # We use this object because we have delay
            self.manage_delayed_mdp(s_divided, a_divided, s_next_divided)

            step += 1
            r_total += np.sum(list(reward_divided.values()))
            self.total_step += 1
            # 6. s = s_next
            s = s_next.copy()


        self.update_all_agents()

        sr_values = self._safe_get_strehl()
        print('Episode: {} \tTotal steps: {} \tEpisode steps: {} \tNum updates: {}'
              ' \tSeed: {} \tCurrent Reward: {:.4f} \tSR SE: {:.4f} \tTime {:.4f}'
              .format(
                  self.num_episode,
                  self.total_step,
                  step,
                  self.total_update,
                  self.seed,
                  r_total,
                  sr_values[0],
                  time.time() - start_time,
              ))

        self.num_episode += 1

        return r_total


    def test_episode(self, controller):
        """
        Does an episode of the environment
        The episode definition depends on the configuration
        """
        geometric_modes = np.zeros((self.config_rl.env_rl['max_steps_per_episode'],
                                    self.env.supervisor.volts2modes.shape[0]))
        r_per_agent_test, r_total_test, done, s = np.zeros(len(self.dictionary_agents)), 0, False, self.env.reset()

        # Align the test run reward baseline with the freshly reset
        # environment to ensure comparable metrics across controllers.
        self._reset_reward_tracking()

        sr_se_test_list = []
        sr_sl_test_list = []
        dm_images = []  #
        for test_step in range(self.config_rl.env_rl['max_steps_per_episode']):

            # 0. Divided states for agents
            s_divided = self.divide_states_for_agents(s)


            # 1. Choose action based on state
            if controller == "RL":
                a, _, _ = self.choose_action(s_divided, eval_mode=True)
            else:
                a = None

            # 2. Step on the environment

            s_next, reward_divided, done =\
                self.env_step(a, linear_control=True if controller == "Integrator" else False)
            dm_image = self.env.supervisor.dms.get_dm_shape(0)
            dm_images.append(dm_image)
            # Agent metrics
            r_total_test += np.sum(list(reward_divided.values()))
            r_per_agent_test += np.array(list(reward_divided.values()))
            sr_values = self._safe_get_strehl()
            sr_se_test_list.append(sr_values[0])
            sr_sl_test_list.append(sr_values[1])

            # Geometric save commands
            if len(self.env.supervisor.config.p_controllers) > 1:
                geometric_modes[test_step, :] =\
                    self.env.supervisor.volts2modes.dot(self.env.supervisor.rtc.get_command(1))

            # 3. s = s_next
            s = s_next.copy()

        sr_le_test = self._safe_get_strehl()[1]
        sr_se_test = np.average(sr_se_test_list)
        print('Test episode: {} \tSeed: {} \tCurrent Reward: {:.4f} \tSR LE: {:.4f} \tAvg SR SE: {:.4f}'
              .format(self.num_test_episode, self.seed, r_total_test, sr_le_test, sr_se_test))

        self.num_test_episode += 1

        agent_performance = {"r_per_agent_test": r_per_agent_test,
                             "r_total_test": r_total_test,
                             "sr_le_test": sr_le_test,
                             "sr_se_test": sr_se_test,
                             "sr_se_test_list":sr_se_test_list,
                             "sr_sl_test_list":sr_sl_test_list}

        if len(self.env.supervisor.config.p_controllers) > 1:
            # Geometric metrics, geometric index is 1
            r_geo_per_agent_test = self.divide_rewards_for_agents_geometric(geometric_modes=geometric_modes)
            r_geo_total_test = np.sum(r_geo_per_agent_test)
            sr_le_geo_test = self._safe_get_strehl(tar_index=1)[1]

            geometric_performance = {"r_geo_per_agent_test": r_geo_per_agent_test,
                                     "r_geo_total_test": r_geo_total_test,
                                     "sr_le_geo_test": sr_le_geo_test}
        else:
            geometric_performance = None

        return agent_performance,geometric_performance

    def manage_delayed_mdp(self, s, a, s_next):
        """
        delayed_mdp object serves the purpose to remember s, a and s_next for when the correct r appears in the train
        loop and we can assign it correctly. This effect happens because we have delay in the system.
        e.g. delay 1
        s, a -> s', r'
        s', a' -> s'', r''
        s'', a'' -> s''', r'''
        The correct assignment would be (s, a, s'', r''') due to how the simulator works
        e.g. delay 0 -> (s, a, s', r'')
        """
        self.delayed_mdp_object.save(s, a, s_next)

    def env_step(self, a, return_dict=False, linear_control=False):
        """
        Does an step inside the environment
        a: action that will change the environment
        return_dict: if we want the next_state as an array or as a ordered_dict
        """
        if self.config_rl.env_rl['level'] == "only_rl" or self.config_rl.env_rl['level'] == "correction":

           #做一步强化学习，把强化学习的命令设传入系统
            _, done, info = self.env.rl_step(a, linear_control)
            #奖励用的是变形镜的总误差，不明白为什么不用传感器测量误差
            r = self.divide_rewards_for_agents()

            s_next = self.env.linear_step(return_dict)
        else:
            raise NotImplementedError

        return s_next, r, done

    def select_correct_modes_for_array(self, agent_value_0, agent_value_1):
        # 1. As self.current_action = np.zeros(len_actions)
        # 2. And self.dictionary we have the current modes controlled e.g. 300 to 600
        # 3. From 300 and 600 we will remove 300 (self.starting_mode) to fit into the array
        # 4. Ending with bottom_mode 0 and top_mode 300
        # The same goes for the state when doing divide_states

        # First if is due to tip tilt
        if self.config_rl.env_rl['include_tip_tilt']\
                and agent_value_0 == (self.total_existing_modes-2) and agent_value_1 == self.total_existing_modes:
            bottom_mode = self.total_controlled_modes-2-self.starting_mode
            top_mode = self.total_controlled_modes-self.starting_mode
        else:
            bottom_mode = agent_value_0 - self.starting_mode
            top_mode = agent_value_1 - self.starting_mode
        return bottom_mode, top_mode

    def report_action(self, a, mu, worker_id):

        # TODO disable record mu?
        bottom_mode_for_array, top_mode_for_array =\
            self.select_correct_modes_for_array(
                self.dictionary_agents[worker_id][0], self.dictionary_agents[worker_id][1])
        self.current_action[bottom_mode_for_array:top_mode_for_array] = a
        self.current_mu[bottom_mode_for_array:top_mode_for_array] = mu
        self.current_action_divided[worker_id] = a

    def report_metrics(self, worker_id, *metric_args, **metric_kwargs):
        """Record training metrics reported by a worker.

        Historical SAC workers reported five positional metrics in a fixed
        order, whereas newer agents such as MAT prefer descriptive keyword
        arguments (or even pass a single dictionary).  The previous
        implementation attempted to cover both cases by declaring optional
        parameters with ``None`` defaults.  However, RPC calls coming from
        legacy workers still triggered ``TypeError`` exceptions because the
        remote site resolved the signature before the module reload, keeping
        the stricter positional contract.  To shield the trainer from such
        version skew we now accept an arbitrary combination of positional
        arguments and keyword payloads and normalise them at runtime.
        """

        alias_map = {
            "qf1_loss_list": "qf1_loss_list",
            "qf1_loss": "qf1_loss_list",
            "qf2_loss_list": "qf2_loss_list",
            "qf2_loss": "qf2_loss_list",
            "alpha_loss_list": "alpha_loss_list",
            "alpha_loss": "alpha_loss_list",
            "alpha_tlogs_list": "alpha_tlogs_list",
            "alpha_tlogs": "alpha_tlogs_list",
            "policy_loss_list": "policy_loss_list",
            "policy_loss": "policy_loss_list",
            "value_loss_list": "value_loss_list",
            "value_loss": "value_loss_list",
            "critic_loss_list": "value_loss_list",
            "entropy_list": "entropy_list",
            "entropy": "entropy_list",
            "entropy_loss_list": "entropy_list",
        }

        canonical_keys = set(alias_map.values())
        metrics = {key: None for key in canonical_keys}

        def _assign_if_missing(name, value):
            if value is None:
                return
            canonical = alias_map.get(name, name)
            if isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    _assign_if_missing(sub_key, sub_val)
                return
            if canonical is None:
                return
            if canonical not in metrics:
                metrics[canonical] = value
            elif metrics[canonical] is None:
                metrics[canonical] = value

        legacy_keys = [
            "qf1_loss_list",
            "qf2_loss_list",
            "alpha_loss_list",
            "alpha_tlogs_list",
            "policy_loss_list",
        ]
        optional_legacy_keys = [
            "value_loss_list",
            "entropy_list",
        ]
        mat_keys = [
            "policy_loss_list",
            "value_loss_list",
            "entropy_list",
        ]

        positional_args = []
        for value in metric_args:
            if isinstance(value, dict):
                _assign_if_missing(None, value)
            else:
                positional_args.append(value)

        if positional_args:
            if len(positional_args) >= len(legacy_keys):
                for key, value in zip(legacy_keys, positional_args):
                    _assign_if_missing(key, value)

                remaining = positional_args[len(legacy_keys):]
                for key, value in zip(optional_legacy_keys, remaining):
                    _assign_if_missing(key, value)
            elif len(positional_args) == len(mat_keys):
                for key, value in zip(mat_keys, positional_args):
                    _assign_if_missing(key, value)
            else:
                for key, value in zip(legacy_keys, positional_args):
                    _assign_if_missing(key, value)

        for key, value in metric_kwargs.items():
            _assign_if_missing(key, value)

        def _log(tag, entry):
            if entry is None:
                return
            try:
                total_step, val = entry
            except (TypeError, ValueError):
                return
            if total_step is None or val is None:
                return
            self.writer_metrics_1.add_scalar(f"{tag}/{worker_id}", val, total_step)

        for tag, key in (
            ("qf1_loss", "qf1_loss_list"),
            ("qf2_loss", "qf2_loss_list"),
            ("alpha_loss", "alpha_loss_list"),
            ("alpha_tlogs", "alpha_tlogs_list"),
            ("policy_loss", "policy_loss_list"),
            ("value_loss", "value_loss_list"),
            ("entropy", "entropy_list"),
        ):
            _log(tag, metrics.get(key))

    def write_update_losses_for_each_agent(self):
        """
        Writing losses had to be syncronized otherwise it seemed not to work
        """
        futs = []
        for ag_rreff in self.ag_rrefs:
            futs.append(
                rpc_async(
                    ag_rreff.owner(),
                    _call_method,
                    args=(self.agent_cls.master_ask_metrics, ag_rreff, self.master_rref),
                    timeout=12000,
                )
            )

        for fut in futs:
            fut.wait()

    def choose_action(self, s, eval_mode=False):
        """
        Query the remote transformer agent for an action.
        Based on some config parameters the behaviour changes
        """

        self.current_action = np.zeros(self.env.action_space.shape[0], dtype=np.float32)
        self.current_action_divided = {}
        self.current_mu = np.zeros(self.env.action_space.shape[0], dtype=np.float32)

        futs = []
        worker_id = 1
        for ag_rreff in self.ag_rrefs:
            # make async RPC to kick off an episode on all observers
            futs.append(
                rpc_async(
                    ag_rreff.owner(),
                    _call_method,
                    args=(
                        self.agent_cls.master_ask_action,
                        ag_rreff,
                        self.master_rref,
                        s[worker_id],
                        worker_id,
                        eval_mode,
                    ),
                    timeout=12000,
                )
            )
            worker_id += 1

        for fut in futs:
            fut.wait()

        return self.current_action, self.current_action_divided, self.current_mu

    def manage_memory(self, reward_divided, done=False):
        """
        Pushes to memory and does updates. Some explanations below.
        ------------------------------------------------------------------------------------------------------------
        Following what is explained in manage_delayed_mdp
        manage_delayed_mdp has deques of len(delay+1)
        So once we have the reward, r, the input of this function we can extract the following:
        s = delayed_mdp.state_list[0], a = delayed_mdp.action_list[0], s' = delayed_mdp.next_state_list[-1]
        r = input
        This should work for all delays and the credit assignment should be ok
        -------------------------------------------------------------------------------------------------------------
        """

        # a) Credit assignment for different configs
        state_divided, action_divided, state_next_divided = self.delayed_mdp_object.credit_assignment()

        for worker_id in range(1, self.world_size):
            state = state_divided[worker_id]
            action = action_divided[worker_id]
            state_next = state_next_divided[worker_id]
            reward = reward_divided[worker_id]
            mask = float(not done)

            state = np.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)
            action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
            state_next = np.nan_to_num(state_next, nan=0.0, posinf=1e6, neginf=-1e6)
            reward = float(np.nan_to_num(reward, nan=0.0, posinf=1e6, neginf=-1e6))
            action = np.clip(action, -0.999, 0.999)

            self.memorys_master[worker_id].push(state, action, reward, state_next, mask)

            if self._uses_return_tracking:
                agent_rref = self.ag_rrefs[worker_id - 1]
                rpc_async(
                    agent_rref.owner(),
                    _call_method,
                    args=(
                        self.agent_cls.record_reward,
                        agent_rref,
                        reward,
                        bool(not mask),
                        float(self._latest_strehl),
                    ),
                    timeout=12000,
                ).wait()

    def update_all_agents(self):

        worker_id = 1
        futs = []
        for ag_rreff in self.ag_rrefs:
            futs.append(
                rpc_async(
                    ag_rreff.owner(),
                    _call_method,
                    args=(
                        self.agent_cls.update_parameters,
                        ag_rreff,
                        self.memorys_master[worker_id],
                        self.config_rl.sac['batch_size'],
                        self.total_update,
                        self.total_step,
                    ),
                    timeout=12000,
                )

            )
            worker_id += 1
        for fut in futs:
            fut.wait()
        self.total_update += self.config_rl.sac['updates_per_episode_rpc']
        for worker_id in range(1, self.world_size):
            self.memorys_master[worker_id].reset()


