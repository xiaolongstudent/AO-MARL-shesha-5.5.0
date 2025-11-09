import json
import os
import sys
from json import JSONDecodeError

# sys.path.append('src/reinforcement_learning/config')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("src/reinforcement_learning"))))


def _strip_inline_comments(text: str) -> str:
    """Remove trailing // or # comments that may appear in config files."""

    cleaned_lines = []
    for original_line in text.splitlines():
        line = []
        i = 0
        while i < len(original_line):
            char = original_line[i]
            if char == '"':
                line.append(char)
                i += 1
                # Copy the rest of the string verbatim, taking escaped quotes into account.
                while i < len(original_line):
                    line.append(original_line[i])
                    if original_line[i] == '"' and original_line[i - 1] != '\\':
                        i += 1
                        break
                    i += 1
                continue
            if char == '/' and i + 1 < len(original_line) and original_line[i + 1] == '/':
                break
            if char == '#':
                break
            line.append(char)
            i += 1
        cleaned_lines.append(''.join(line).rstrip())
    return '\n'.join(cleaned_lines)


def _strip_trailing_commas(text: str) -> str:
    """Remove trailing commas before closing braces/brackets."""

    result = []
    i = 0
    length = len(text)
    while i < length:
        char = text[i]
        if char == '"':
            result.append(char)
            i += 1
            while i < length:
                result.append(text[i])
                if text[i] == '"' and text[i - 1] != '\\':
                    i += 1
                    break
                i += 1
            continue
        if char == ',':
            j = i + 1
            while j < length and text[j] in ' \t\r\n':
                j += 1
            if j < length and text[j] in '}]':
                i += 1
                continue
        result.append(char)
        i += 1
    return ''.join(result)


def _load_json_config(path: str) -> dict:
    """Load a JSON config file, tolerating minor formatting issues."""

    with open(path, "r", encoding="utf-8") as datafile:
        raw_text = datafile.read()
    try:
        return json.loads(raw_text)
    except JSONDecodeError:
        sanitized = _strip_trailing_commas(_strip_inline_comments(raw_text))
        try:
            return json.loads(sanitized)
        except JSONDecodeError as exc:
            raise ValueError(f"Configuration file {path} is not valid JSON: {exc}") from exc

class Config:
    """
    Config object that reads all the files from reinforcement_learning/config/*.cfg
    """
    def __init__(self, in_p="src/reinforcement_learning"):
        config_file_path = in_p + '/config/parameters.cfg'
        # config_file_path = 'parameters.cfg'

        config_file_path_sac = in_p + '/config/parameters_sac.cfg'
        # config_file_path_sac = 'parameters_sac.cfg'

        config_file_path_autoencoder = in_p + '/config/parameters_autoencoder.cfg'
        # config_file_path_autoencoder = 'parameters_autoencoder.cfg'

        config = _load_json_config(config_file_path)
        config_sac = _load_json_config(config_file_path_sac)
        config_autoencoder = _load_json_config(config_file_path_autoencoder)

        self.sac = dict()
        self.autoencoder = dict()
        self.env_rl = dict()

        self.cuda = True if config['cuda'] == "True" else False
        self.algorithm = str(config['algorithm'])
        self.savedir = str(config['savedir'])
        # 1) Soft Actor Critic Config

        # 1.1 Traditional parameters

        self.sac['alpha'] = float(config_sac['alpha'])
        self.sac['automatic_entropy_tuning'] = str(config_sac['automatic_entropy_tuning'])
        self.sac['batch_size'] = int(config_sac['batch_size'])
        self.sac['gamma'] = float(config_sac['gamma'])
        hidden_size_critic = int(config_sac['hidden_size_critic'])
        num_layers_critic = max(1, int(config_sac['num_layers_critic']))
        self.sac['num_layers_critic'] = num_layers_critic
        self.sac['hidden_size_critic'] = [hidden_size_critic] * num_layers_critic
        self.sac['hidden_size_actor'] = int(config_sac['hidden_size_actor'])
        self.sac['num_layers_actor'] = int(config_sac['num_layers_actor'])
        self.sac['lr'] = float(config_sac['lr'])
        self.sac['policy'] = str(config_sac['policy'])
        self.sac['target_update_interval'] = int(config_sac['target_update_interval'])
        self.sac['tau'] = float(config_sac['tau'])
        self.sac['updates_per_step'] = int(config_sac['updates_per_step'])
        self.sac['memory_size'] = int(config_sac['memory_size'])

        # 1.2 Other parameters

        self.sac['gaussian_mu'] = float(config_sac['gaussian_mu'])
        self.sac['gaussian_std'] = float(config_sac['gaussian_std'])

        self.sac['sac_reward_scaling'] = float(1.0)
        self.sac['activation'] = str(config_sac['activation'])
        self.sac['initialize_last_layer_0'] = str(config_sac['initialize_last_layer_0'])
        self.sac['initialize_last_layer_near_0'] = str(config_sac['initialize_last_layer_near_0'])
        self.sac['initialize_last_layer_init_kan'] = str(config_sac['initialize_last_layer_init_kan'])

        self.sac['save_replay_buffer'] = str(config_sac['save_replay_buffer'])
        self.sac['save_rewards_buffer'] = str(config_sac['save_rewards_buffer'])


        offline_enabled_raw = config_sac.get('offline_dataset_enabled', "False")
        self.sac['offline_dataset_enabled'] = str(offline_enabled_raw).lower() == 'true'
        offline_dir_raw = config_sac.get('offline_dataset_dir')
        if offline_dir_raw is None or str(offline_dir_raw).lower() in {"none", ""}:
            offline_dir = None
        else:
            offline_dir = str(offline_dir_raw)
        self.sac['offline_dataset_dir'] = offline_dir
        self.sac['offline_dataset_format'] = str(config_sac.get('offline_dataset_format', "npz"))
        self.sac['offline_dataset_flush_episodes'] = int(
            config_sac.get('offline_dataset_flush_episodes', "10")
        )
        include_mu_raw = config_sac.get('offline_dataset_include_mu', "False")
        self.sac['offline_dataset_include_mu'] = str(include_mu_raw).lower() == 'true'


        self.sac['l2_norm_policy'] = -1
        self.sac['LOG_SIG_MAX'] = 2.0
        self.sac['updates_per_episode_rpc'] = int(config_sac.get('updates_per_episode_rpc', "4"))

        # Stabilisation helpers used by transformer based policies.  Default
        # values are provided so older configuration files remain valid.
        self.sac['entropy_coef'] = float(config_sac.get('entropy_coef', "0.0"))
        self.sac['normalize_advantage'] = config_sac.get('normalize_advantage', "True")
        self.sac['advantage_norm_epsilon'] = float(config_sac.get('advantage_norm_epsilon', "1e-5"))
        self.sac['gradient_clip_norm'] = float(config_sac.get('gradient_clip_norm', "0.0"))
        self.sac['gae_lambda'] = float(config_sac.get('gae_lambda', "0.95"))
        self.sac['value_coef'] = float(config_sac.get('value_coef', "0.5"))
        self.sac['ppo_clip_param'] = float(config_sac.get('ppo_clip_param', "0.2"))
        self.sac['value_clip_param'] = float(config_sac.get('value_clip_param', "0.0"))
        self.sac['normalize_rewards'] = config_sac.get('normalize_rewards', "True")
        self.sac['reward_scale'] = float(config_sac.get('reward_scale', "1.0"))
        self.sac['reward_clip'] = float(config_sac.get('reward_clip', "0.0"))
        self.sac['reward_epsilon'] = float(config_sac.get('reward_epsilon', "1e-6"))
        self.sac['reward_residual_weight'] = float(config_sac.get('reward_residual_weight', "1.0"))
        self.sac['reward_strehl_weight'] = float(config_sac.get('reward_strehl_weight', "0.1"))
        self.sac['reward_delta_weight'] = float(config_sac.get('reward_delta_weight', "0.0"))
        self.sac['transformer_dropout'] = float(config_sac.get('transformer_dropout', "0.1"))
        self.sac['transformer_ff_multiplier'] = float(config_sac.get('transformer_ff_multiplier', "2.0"))
        self.sac['mat_replay_window'] = int(config_sac.get('mat_replay_window', "0"))
        self.sac['dt_context_len'] = int(config_sac.get('dt_context_len', "8"))
        self.sac['dt_nhead'] = max(1, int(config_sac.get('dt_nhead', "4")))
        dt_num_layers_raw = config_sac.get('dt_num_layers')
        if dt_num_layers_raw is None or str(dt_num_layers_raw).strip() == "":
            self.sac['dt_num_layers'] = None
        else:
            self.sac['dt_num_layers'] = max(1, int(dt_num_layers_raw))
        self.sac['dt_target_return'] = float(config_sac.get('dt_target_return', "800.0"))
        self.sac['dt_discount'] = float(config_sac.get('dt_discount', "0.99"))
        self.sac['dt_action_scale'] = float(config_sac.get('dt_action_scale', "1.0"))
        dt_return_scale_raw = config_sac.get('dt_return_scale')
        if dt_return_scale_raw is None:
            target = max(self.sac['dt_target_return'], 1.0)
            self.sac['dt_return_scale'] = max(1.0 / target, 0.1)
        else:
            self.sac['dt_return_scale'] = float(dt_return_scale_raw)
        self.sac['dt_return_clip'] = float(config_sac.get('dt_return_clip', "2000.0"))
        self.sac['dt_return_floor_ratio'] = float(
            config_sac.get('dt_return_floor_ratio', "0.0")
        )
        self.sac['dt_step_return_scale'] = float(config_sac.get('dt_step_return_scale', "0.5"))
        self.sac['dt_step_return_beta'] = float(config_sac.get('dt_step_return_beta', "0.15"))
        self.sac['dt_reward_scale'] = float(config_sac.get('dt_reward_scale', "1.0"))
        self.sac['dt_reward_center'] = float(config_sac.get('dt_reward_center', "0.0"))
        self.sac['dt_reward_clip'] = float(config_sac.get('dt_reward_clip', "0.0"))
        self.sac['dt_strehl_weight'] = float(config_sac.get('dt_strehl_weight', "0.05"))
        self.sac['dt_residual_weight'] = float(config_sac.get('dt_residual_weight', "1.0"))
        self.sac['dt_reward_delta_weight'] = float(config_sac.get('dt_reward_delta_weight', "0.0"))
        self.sac['dt_reward_momentum'] = float(config_sac.get('dt_reward_momentum', "0.2"))
        self.sac['dt_reward_smoothing'] = float(config_sac.get('dt_reward_smoothing', "0.02"))
        self.sac['dt_target_momentum'] = float(config_sac.get('dt_target_momentum', "0.9"))
        self.sac['dt_strehl_momentum'] = float(
            config_sac.get('dt_strehl_momentum', self.sac['dt_target_momentum'])
        )
        self.sac['dt_online_gain'] = float(config_sac.get('dt_online_gain', "1.0"))
        self.sac['dt_online_gain_min'] = float(config_sac.get('dt_online_gain_min', "1.0"))
        self.sac['dt_online_gain_max'] = float(config_sac.get('dt_online_gain_max', "1.5"))
        self.sac['dt_online_gain_quantile'] = float(
            config_sac.get('dt_online_gain_quantile', "0.8")
        )
        self.sac['dt_target_offset'] = float(config_sac.get('dt_target_offset', "0.0"))
        self.sac['dt_target_gain'] = float(config_sac.get('dt_target_gain', "0.0"))
        self.sac['dt_target_min'] = float(config_sac.get('dt_target_min', "0.0"))
        self.sac['dt_replay_episodes'] = max(1, int(config_sac.get('dt_replay_episodes', "32")))
        self.sac['dt_recent_episodes'] = max(1, int(config_sac.get('dt_recent_episodes', "10")))
        self.sac['dt_best_episodes'] = max(
            1,
            int(
                config_sac.get(
                    'dt_best_episodes',
                    str(self.sac['dt_replay_episodes'])
                )
            ),
        )
        self.sac['dt_sequence_stride'] = max(1, int(config_sac.get('dt_sequence_stride', "1")))
        self.sac['dt_loss_temperature'] = float(config_sac.get('dt_loss_temperature', "1.0"))
        self.sac['dt_recent_weight'] = float(config_sac.get('dt_recent_weight', "0.45"))
        dt_normalize_returns = config_sac.get('dt_normalize_returns', "True")
        if isinstance(dt_normalize_returns, str):
            dt_normalize_returns = dt_normalize_returns.lower() == "true"
        else:
            dt_normalize_returns = bool(dt_normalize_returns)
        self.sac['dt_normalize_returns'] = dt_normalize_returns
        self.sac['dt_return_norm_epsilon'] = float(config_sac.get('dt_return_norm_epsilon', "1e-6"))
        self.sac['dt_quality_strehl_weight'] = float(
            config_sac.get('dt_quality_strehl_weight', "0.0")
        )
        self.sac['dt_sequences_topk'] = max(
            0, int(config_sac.get('dt_sequences_topk', "384"))
        )
        self.sac['dt_sequences_min_keep_recent'] = max(
            0, int(config_sac.get('dt_sequences_min_keep_recent', "32"))
        )
        replay_offline_ratio = float(
            config_sac.get('dt_replay_offline_ratio', "0.2")
        )
        replay_recent_ratio = float(
            config_sac.get('dt_replay_recent_ratio', "0.1")
        )
        self.sac['dt_replay_offline_ratio'] = replay_offline_ratio
        self.sac['dt_replay_recent_ratio'] = replay_recent_ratio
        self.sac['dt_replay_recent_ratio_cap'] = float(
            config_sac.get('dt_replay_recent_ratio_cap', "0.2")
        )
        self.sac['dt_replay_online_top_percentile'] = float(
            config_sac.get('dt_replay_online_top_percentile', "0.2")
        )
        self.sac['dt_online_keep_percentile'] = float(
            config_sac.get('dt_online_keep_percentile', "0.7")
        )
        self.sac['dt_online_keep_min_samples'] = max(
            1, int(config_sac.get('dt_online_keep_min_samples', "32"))
        )
        self.sac['dt_online_history_limit'] = int(
            config_sac.get('dt_online_history_limit', "2048")
        )
        self.sac['dt_low_weight_maxlen'] = int(
            config_sac.get('dt_low_weight_maxlen', "128")
        )
        updates_override = config_sac.get('dt_updates_per_episode', None)
        if updates_override is None:
            self.sac['dt_updates_per_episode'] = None
        else:
            updates_str = str(updates_override).strip().lower()
            if updates_str in ("", "none"):
                self.sac['dt_updates_per_episode'] = None
            else:
                self.sac['dt_updates_per_episode'] = max(1, int(float(updates_str)))
        target_entropy_scale_raw = config_sac.get('target_entropy_scale', "1.0")
        try:
            self.sac['target_entropy_scale'] = float(target_entropy_scale_raw)
        except (TypeError, ValueError):
            self.sac['target_entropy_scale'] = 1.0
        target_entropy_offset_raw = config_sac.get('target_entropy_offset', "0.0")
        try:
            self.sac['target_entropy_offset'] = float(target_entropy_offset_raw)
        except (TypeError, ValueError):
            self.sac['target_entropy_offset'] = 0.0
        alpha_clip_min_raw = config_sac.get('alpha_clip_min')
        if isinstance(alpha_clip_min_raw, str) and alpha_clip_min_raw.strip().lower() in {"", "none"}:
            alpha_clip_min = None
        else:
            try:
                alpha_clip_min = float(alpha_clip_min_raw)
            except (TypeError, ValueError):
                alpha_clip_min = None
        alpha_clip_max_raw = config_sac.get('alpha_clip_max')
        if isinstance(alpha_clip_max_raw, str) and alpha_clip_max_raw.strip().lower() in {"", "none"}:
            alpha_clip_max = None
        else:
            try:
                alpha_clip_max = float(alpha_clip_max_raw)
            except (TypeError, ValueError):
                alpha_clip_max = None
        if alpha_clip_min is not None and alpha_clip_max is not None and alpha_clip_min > alpha_clip_max:
            alpha_clip_min, alpha_clip_max = alpha_clip_max, alpha_clip_min
        self.sac['alpha_clip_min'] = alpha_clip_min
        self.sac['alpha_clip_max'] = alpha_clip_max
        self.sac['dt_offline_dataset_glob'] = config_sac.get('dt_offline_dataset_glob', None)
        self.sac['dt_offline_mix_ratio'] = float(config_sac.get('dt_offline_mix_ratio', "0.0"))
        self.sac['dt_offline_mix_ratio_start'] = float(
            config_sac.get('dt_offline_mix_ratio_start', self.sac['dt_offline_mix_ratio'])
        )
        self.sac['dt_offline_mix_ratio_final'] = float(
            config_sac.get('dt_offline_mix_ratio_final', self.sac['dt_offline_mix_ratio'])
        )
        self.sac['dt_offline_mix_ratio_decay'] = max(
            1, int(config_sac.get('dt_offline_mix_ratio_decay', "1"))
        )
        self.sac['dt_offline_mix_ratio_warmup'] = max(
            0, int(config_sac.get('dt_offline_mix_ratio_warmup', "0"))
        )
        self.sac['dt_offline_lock_ratio'] = float(
            config_sac.get('dt_offline_lock_ratio', "0.6")
        )
        self.sac['dt_offline_lock_updates'] = max(
            0, int(config_sac.get('dt_offline_lock_updates', "2000"))
        )
        self.sac['dt_offline_final_ratio'] = float(
            config_sac.get('dt_offline_final_ratio', "0.2")
        )
        self.sac['dt_offline_decay_updates'] = max(
            1, int(config_sac.get('dt_offline_decay_updates', "4000"))
        )
        self.sac['dt_offline_improvement_window'] = max(
            1, int(config_sac.get('dt_offline_improvement_window', "256"))
        )
        self.sac['dt_offline_improvement_threshold'] = max(
            0.0, float(config_sac.get('dt_offline_improvement_threshold', "0.08"))
        )
        self.sac['dt_offline_improvement_min_delta'] = float(
            config_sac.get('dt_offline_improvement_min_delta', "50.0")
        )
        self.sac['dt_offline_improvement_patience'] = max(
            1, int(config_sac.get('dt_offline_improvement_patience', "3"))
        )
        self.sac['dt_offline_max_episodes'] = int(config_sac.get('dt_offline_max_episodes', "0"))
        keep_ratio = float(config_sac.get('dt_offline_keep_top_ratio', "0.0"))
        self.sac['dt_offline_keep_top_ratio'] = min(max(keep_ratio, 0.0), 1.0)
        keep_strehl_ratio = float(config_sac.get('dt_offline_keep_strehl_ratio', "0.0"))
        self.sac['dt_offline_keep_strehl_ratio'] = min(
            max(keep_strehl_ratio, 0.0), 1.0
        )
        self.sac['dt_offline_elite_count'] = max(
            0, int(config_sac.get('dt_offline_elite_count', "0"))
        )
        self.sac['dt_offline_elite_fraction'] = float(
            config_sac.get('dt_offline_elite_fraction', "0.0")
        )
        self.sac['dt_offline_reserve_limit'] = max(
            0, int(config_sac.get('dt_offline_reserve_limit', "0"))
        )
        min_return_cfg = config_sac.get('dt_offline_min_return', None)
        if min_return_cfg is None:
            self.sac['dt_offline_min_return'] = None
        else:
            min_return_str = str(min_return_cfg).strip().lower()
            if min_return_str in ("", "none"):
                self.sac['dt_offline_min_return'] = None
            else:
                self.sac['dt_offline_min_return'] = float(min_return_cfg)
        min_strehl_cfg = config_sac.get('dt_offline_min_strehl', None)
        if min_strehl_cfg is None:
            self.sac['dt_offline_min_strehl'] = None
        else:
            min_strehl_str = str(min_strehl_cfg).strip().lower()
            if min_strehl_str in ("", "none"):
                self.sac['dt_offline_min_strehl'] = None
            else:
                self.sac['dt_offline_min_strehl'] = float(min_strehl_cfg)

        # 2) Environment Reinforcement Learning Config

        self.env_rl['write_every'] = int(config['env_rl_parameters']['write_every'])
        self.env_rl['verbose'] = False
        self.env_rl['check_every'] = int(config['env_rl_parameters']['check_every'])
        self.env_rl['move_atmos'] = str(config['env_rl_parameters']['move_atmos'])
        self.env_rl['max_steps_per_episode'] = int(config['env_rl_parameters']['max_steps_per_episode'])
        training_eps_raw = config['env_rl_parameters'].get('training_episodes')
        training_eps = None
        if training_eps_raw is not None:
            try:
                training_eps = max(1, int(float(training_eps_raw)))
            except (TypeError, ValueError):
                training_eps = None
        if training_eps is None:
            training_eps = 1000
        self.env_rl['training_episodes'] = training_eps

        # Parameters of telescope

        self.env_rl['parameters_telescope'] = str(config['env_rl_parameters']['parameters_telescope'])
        self.env_rl['integration_mode'] = str(config['env_rl_parameters']['integration_mode'])

        # Related to how do we do the process and we optimize

        self.env_rl['level'] = str(config['env_rl_parameters']['level'])
        self.env_rl['basis'] = str(config['env_rl_parameters']['basis'])
        self.env_rl['influence'] = str(config['env_rl_parameters']['influence'])

        # Related to reward

        self.env_rl['reward_type'] = str(config['env_rl_parameters'].get('reward_type', "avg_squared_modes_200"))
        self.env_rl['delayed_assignment'] = int(config['env_rl_parameters']['delayed_assignment'])

        # Related to state

        self.env_rl['state_dm_before_linear'] = str(config['env_rl_parameters']['state_dm_before_linear'])
        self.env_rl['state_dm_after_linear'] = str(config['env_rl_parameters']['state_dm_after_linear'])
        self.env_rl['state_wfs'] = str(config['env_rl_parameters']['state_wfs'])
        self.env_rl['state_gain'] = str(config['env_rl_parameters']['state_gain'])
        self.env_rl['state_dm_residual'] = str(config['env_rl_parameters']['state_dm_residual'])
        self.env_rl['number_of_previous_dm'] = int(config['env_rl_parameters']['number_of_previous_dm'])
        self.env_rl['number_of_previous_wfs'] = int(config['env_rl_parameters']['number_of_previous_wfs'])
        self.env_rl['number_of_previous_dm_residuals'] = 0

        self.env_rl['reward_mode'] = str(config['env_rl_parameters']['reward_mode'])
        self.env_rl['reward_residual_smoothing'] = float(
            config['env_rl_parameters'].get('reward_residual_smoothing', "0.0")
        )

        self.env_rl['n_zernike_start_end'] = [0, 80]
        self.env_rl['n_reverse_filtered_from_cmat'] = 5
        self.env_rl['include_tip_tilt'] = "False"

        # Related to reward

        self.env_rl['reward_mode'] = str(config['env_rl_parameters']['reward_mode'])
        self.env_rl['TT_reward'] = 'absolute'

        # Other

        self.env_rl['normalization_std_inside_environment'] =\
            float(config['env_rl_parameters']['normalization_std_inside_environment'])
        self.env_rl['normalization_mean_inside_environment'] =\
            float(config['env_rl_parameters']['normalization_mean_inside_environment'])

        self.env_rl['norm_scale_zernike_actions'] = float(config['env_rl_parameters']['norm_scale_zernike_actions'])
        self.env_rl['modification_online'] = str(config['env_rl_parameters']['modification_online'])
        self.env_rl["custom_freedom_path"] = None

        self.env_rl['create_norm_param'] = str(config['env_rl_parameters']['create_norm_param'])
        self.env_rl['window_n_zernike'] = -1
        self.env_rl['save_dict'] = str(config['env_rl_parameters']['save_dict'])

        self.env_rl['do_more_evaluations'] = "False"
        self.env_rl['change_atmospheric_3_layers_1'] = "False"
        self.env_rl['change_atmospheric_3_layers_2'] = "False"
        self.env_rl['change_atmospheric_3_layers_3'] = "False"
        self.env_rl['change_atmospheric_3_layers_4'] = "False"
        self.env_rl['change_atmospheric_3_layers_5'] = "False"

        self.env_rl['change_atmospheric_conditions_wind_direction_1'] = "False"
        self.env_rl['change_atmospheric_conditions_wind_direction_2'] = "False"

        self.env_rl['include_tip_tilt_windowed'] = "False"
        self.env_rl['record_autoencoder_time'] = "False"

        self.env_rl['gain_change'] = -1.0
        self.env_rl['tt_treated_as_mode'] = "False"

        # 3) Variable to save original gain from integrator controller
        self.original_gain = None

        # 4) Autoencoder
        self.autoencoder['path'] = None #'output/autoencoder/save_model2.pth'  #None
        self.autoencoder['type'] = str(config_autoencoder['type'])

        # Loading previous weights/replay

        self.env_rl['load_previous_weights'] = str(config['env_rl_parameters']['load_previous_weights'])
        self.sac['pretrained_replay_path'] = None
        self.sac['replay_path'] = None
        self.sac['pretrained_model_path'] = str(config_sac['pretrained_model_path'])

        # TODO ??

        self.dictionary_agent_values = None

    def update_conf_with_args(self, args):
        """
        Updates config object with the arguments
        """

        algorithm_name = str(args.algorithm).strip().lower()
        if algorithm_name in {"sac", "mat", "multi-agent-transformer", "dt",
                              "decision_transformer", "decision-transformer"}:
            # MAT/DT currently reuse the same hyper-parameter structure that the
            # historical SAC implementation relied on.  Accepting the extra
            # names keeps backwards compatibility with configuration files that
            # select one of the transformer agents while still funnelling all of
            # the tuning knobs through the existing SAC parsing logic.
            self.update_sac(args)
            # Preserve the requested algorithm name so the trainer can decide
            # whether to build SAC, MAT or Decision Transformer instances.
            self.algorithm = args.algorithm
        else:
            raise NotImplementedError

        self.update_env(args)

        self.update_autoencoder(args)

        self.strings_to_bools()

    def update_sac(self, args):
        self.algorithm = args.algorithm
        self.sac['policy'] = args.policy

        # Traditional parameters

        self.sac['gamma'] = args.gamma
        self.sac['batch_size'] = args.batch_size
        self.sac['alpha'] = args.alpha
        self.sac['automatic_entropy_tuning'] = True if args.automatic_entropy_tuning == "True" else False
        self.sac['memory_size'] = args.memory_size
        self.sac['lr'] = args.lr
        if len(args.hidden_size_critic) == 1:
            self.sac['hidden_size_critic'] = args.hidden_size_critic[0]
        else:
            self.sac['hidden_size_critic'] = args.hidden_size_critic
        self.sac['num_layers_critic'] = args.num_layers_critic
        self.sac['hidden_size_actor'] = args.hidden_size_actor
        self.sac['num_layers_actor'] = args.num_layers_actor
        self.sac['tau'] = float(args.tau)

        # Other parameters

        self.sac['gaussian_mu'] = args.gaussian_mu
        self.sac['gaussian_std'] = args.gaussian_std
        self.sac['updates_per_step'] = args.updates_per_step
        self.sac['sac_reward_scaling'] = float(args.sac_reward_scaling)
        self.sac['activation'] = str(args.activation)
        self.sac['initialize_last_layer_0'] = True if str(args.initialize_last_layer_0) == "True" else False
        self.sac['initialize_last_layer_near_0'] = True if str(args.initialize_last_layer_near_0) == "True" else False
        self.sac['initialize_last_layer_init_kan'] = True if str(args.initialize_last_layer_init_kan) == "True" else False

        self.sac['l2_norm_policy'] = float(args.l2_norm_policy)
        self.sac['updates_per_episode_rpc'] = int(args.updates_per_episode_rpc)
        self.sac['LOG_SIG_MAX'] = float(args.LOG_SIG_MAX)
        if hasattr(args, 'mat_replay_window'):
            self.sac['mat_replay_window'] = int(args.mat_replay_window)

        # Loading
        self.sac['pretrained_replay_path'] = args.pretrained_replay_path
        self.sac['pretrained_model_path'] = None if args.pretrained_model_path == "None" else args.pretrained_model_path
        self.sac['replay_path'] = args.replay_path if args.replay_path is not None else None

    def update_env(self, args):

        self.env_rl['move_atmos'] = True if args.move_atmos == "True" else False
        self.env_rl['max_steps_per_episode'] = args.max_steps_per_episode
        self.env_rl['level'] = args.level
        self.env_rl['parameters_telescope'] = args.parameters_telescope
        self.env_rl['integration_mode'] = args.integration_mode

        if len(args.reward_type) > 1:
            self.env_rl['reward_type'] = args.reward_type
        else:
            self.env_rl['reward_type'] = args.reward_type[0]
        self.env_rl['delayed_assignment'] = args.delayed_assignment

        # Related to the state
        self.env_rl['state_dm_before_linear'] = True if args.state_dm_before_linear == "True" else False
        self.env_rl['state_dm_after_linear'] = True if args.state_dm_after_linear == "True" else False
        self.env_rl['state_wfs'] = True if args.state_wfs == "True" else False
        self.env_rl['state_dm_residual'] = True if args.state_dm_residual == "True" else False

        self.env_rl['n_zernike_start_end'] = args.n_zernike_start_end
        self.env_rl['include_tip_tilt'] = True if args.include_tip_tilt == "True" else False
        self.env_rl['number_of_previous_dm'] = int(args.number_of_previous_dm)
        self.env_rl['number_of_previous_wfs'] = int(args.number_of_previous_wfs)
        self.env_rl['number_of_previous_dm_residuals'] = int(args.number_of_previous_dm_residuals)

        # Related to the reward
        self.env_rl['reward_mode'] = args.reward_mode
        if hasattr(args, 'reward_residual_smoothing'):
            self.env_rl['reward_residual_smoothing'] = float(args.reward_residual_smoothing)


        self.env_rl['max_steps_episode'] = args.max_steps_per_episode
        self.env_rl['basis'] = args.basis
        self.env_rl['normalization_std_inside_environment'] = float(args.normalization_std_inside_environment)
        self.env_rl['normalization_mean_inside_environment'] = float(args.normalization_mean_inside_environment)
        self.env_rl['norm_scale_zernike_actions'] = float(args.norm_scale_zernike_actions)

        self.env_rl['modification_online'] = True if args.modification_online == "True" else False

        self.env_rl["custom_freedom_path"] = args.custom_freedom_path
        self.env_rl["load_previous_weights"] = True if args.load_previous_weights == "True" else False

        self.env_rl['create_norm_param'] = True if args.create_norm_param in ['True',"True",1] else False

        self.env_rl['n_reverse_filtered_from_cmat'] = int(args.n_reverse_filtered_from_cmat)

        self.env_rl['window_n_zernike'] = int(args.window_n_zernike)

        self.env_rl['TT_reward'] = str(args.TT_reward)
        self.env_rl['tt_treated_as_mode'] = True if args.tt_treated_as_mode == "True" else False

        self.env_rl['do_more_evaluations'] = True if args.do_more_evaluations == "True" else False

        self.env_rl['include_tip_tilt_windowed'] = True if args.include_tip_tilt_windowed == "True" else False
        if int(args.window_n_zernike) >= 0:
            self.env_rl['include_tip_tilt_windowed'] = True

        self.env_rl['gain_change'] = float(args.gain_change)

        self.env_rl['change_atmospheric_3_layers_1'] = True if args.change_atmospheric_3_layers_1 == "True" else False
        self.env_rl['change_atmospheric_3_layers_2'] = True if args.change_atmospheric_3_layers_2 == "True" else False
        self.env_rl['change_atmospheric_3_layers_3'] = True if args.change_atmospheric_3_layers_3 == "True" else False
        self.env_rl['change_atmospheric_3_layers_4'] = True if args.change_atmospheric_3_layers_4 == "True" else False
        self.env_rl['change_atmospheric_3_layers_5'] = True if args.change_atmospheric_3_layers_5 == "True" else False

    def update_autoencoder(self, args):
        self.autoencoder['path'] = args.autoencoder_path
        self.autoencoder['type'] = str(args.autoencoder_type)

    def strings_to_bools(self):
        for key, item in self.env_rl.items():
            print(type(item), key, item)
            if item == "True":
                self.env_rl[key] = True
                if self.env_rl['verbose']:
                    print("correcting,", key, "to True")
            elif item == "False":
                self.env_rl[key] = False
                if self.env_rl['verbose']:
                    print("correcting,", key, "to False")

        for key, item in self.sac.items():
            print(type(item), key, item)
            if item == "True":
                self.sac[key] = True
                if self.env_rl['verbose']:
                    print("correcting,", key, "to True")
            elif item == "False":
                self.sac[key] = False
                if self.env_rl['verbose']:
                    print("correcting,", key, "to False")


    def set_original_gain(self, gain):
        self.original_gain = gain

if __name__ == "__main__":
    Config()