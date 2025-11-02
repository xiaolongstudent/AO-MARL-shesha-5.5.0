import json
import math
import os
import random
from dataclasses import dataclass
from collections import deque
from shutil import copy2
from glob import glob
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
import torch.distributed.rpc as rpc

from src.reinforcement_learning.rpc_training.helper_rpc.helper_pure_rpc import _remote_method

try:  # pragma: no cover - defensive import for older environments
    from .data_stats import EpisodeStats, compute_episode_stats, summarize_dataset
except (ImportError, AttributeError):

    @dataclass
    class EpisodeStats:  # type: ignore[override]
        ep_id: str
        sum_return: float
        mean_strehl: float
        T: int

    def _fallback_mean_strehl(ep, strehl_key):
        if strehl_key is None or strehl_key not in ep:
            return float("nan")
        try:
            arr = np.asarray(ep[strehl_key], dtype=np.float64).reshape(-1)
        except Exception:
            return float("nan")
        if arr.size == 0:
            return float("nan")
        return float(np.nanmean(arr))

    def compute_episode_stats(ep, *, reward_key="reward", strehl_key=None):  # type: ignore[override]
        rewards = np.asarray(ep[reward_key], dtype=np.float64).reshape(-1)
        return EpisodeStats(
            ep_id=str(ep.get("ep_id", "")),
            sum_return=float(np.sum(rewards)),
            mean_strehl=_fallback_mean_strehl(ep, strehl_key),
            T=int(rewards.size),
        )

    def summarize_dataset(stats_list, *, quantiles=(0.5, 0.8, 0.9), histogram_bins=20):  # type: ignore[override]
        stats = list(stats_list)
        if not stats:
            return {
                "num_eps": 0,
                "ret_mean": 0.0,
                "ret_std": 0.0,
                "strehl_mean": None,
                "strehl_std": None,
                "q": {},
                "top20_thr": 0.0,
                "top20_ratio": 0.0,
                "hist": [],
            }

        returns = np.array([s.sum_return for s in stats], dtype=np.float64)
        quantiles = tuple(float(q) for q in quantiles)
        qs = np.quantile(returns, q=np.array(quantiles))
        top20_thr = float(np.quantile(returns, 0.8))
        top20_ratio = float(np.mean(returns >= top20_thr))
        strehl_vals = np.array(
            [s.mean_strehl for s in stats if not np.isnan(s.mean_strehl)], dtype=np.float64
        )
        strehl_mean = float(np.mean(strehl_vals)) if strehl_vals.size else None
        strehl_std = float(np.std(strehl_vals)) if strehl_vals.size else None

        hist_counts, _ = np.histogram(returns, bins=histogram_bins)

        return {
            "num_eps": len(stats),
            "ret_mean": float(np.mean(returns)),
            "ret_std": float(np.std(returns)),
            "strehl_mean": strehl_mean,
            "strehl_std": strehl_std,
            "q": {f"{q:.2f}": float(v) for q, v in zip(quantiles, qs)},
            "top20_thr": top20_thr,
            "top20_ratio": top20_ratio,
            "hist": hist_counts.astype(int).tolist(),
        }

LOG_SIG_MIN = -20
LOG_SIG_MAX = 2
epsilon = 1e-5


def _coerce_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return default


class DecisionTransformerFeatureEngineer:
    """Build compact, collaboration-aware state embeddings for ODT."""

    def __init__(self, raw_state_dim: int, action_dim: int, config):
        self.raw_dim = int(raw_state_dim)
        self.action_dim = int(action_dim)
        sac_cfg = getattr(config, "sac", {})

        feature_dim = sac_cfg.get("dt_feature_dim", self.raw_dim)
        try:
            feature_dim = int(feature_dim)
        except (TypeError, ValueError):
            feature_dim = self.raw_dim
        self.output_dim = max(1, feature_dim)

        use_residual = sac_cfg.get("dt_feature_use_residual", True)
        self.use_residual = _coerce_bool(use_residual, True)
        use_action_mean = sac_cfg.get("dt_feature_use_action_mean", True)
        self.use_action_mean = _coerce_bool(use_action_mean, True)

        normalize = sac_cfg.get("dt_feature_normalize", True)
        self.normalize = _coerce_bool(normalize, True)

        seed = sac_cfg.get("dt_feature_projection_seed", None)
        try:
            seed = int(seed) if seed is not None else None
        except (TypeError, ValueError):
            seed = None
        if seed is None:
            fallback_seed = getattr(config, "seed", None)
            try:
                seed = int(fallback_seed) if fallback_seed is not None else None
            except (TypeError, ValueError):
                seed = None
        rng = np.random.default_rng(seed)

        enriched_dim = self.raw_dim
        if self.use_residual:
            enriched_dim += self.raw_dim
        if self.use_action_mean:
            enriched_dim += self.action_dim

        if self.output_dim == enriched_dim:
            self._projection = np.eye(enriched_dim, dtype=np.float32)
        else:
            scale = 1.0 / math.sqrt(max(self.output_dim, 1))
            self._projection = rng.standard_normal((self.output_dim, enriched_dim)).astype(np.float32)
            self._projection *= float(scale)

        self._running_mean = np.zeros(self.raw_dim, dtype=np.float32)
        self._running_count = 0.0
        self._last_action_mean = np.zeros(self.action_dim, dtype=np.float32)
        self._eps = 1e-6

    # ------------------------------------------------------------------
    # Running statistics
    # ------------------------------------------------------------------
    def _align_state(self, state: np.ndarray) -> np.ndarray:
        arr = np.asarray(state, dtype=np.float32).reshape(-1)
        if arr.size < self.raw_dim:
            pad = np.zeros(self.raw_dim, dtype=np.float32)
            pad[: arr.size] = arr
            return pad
        if arr.size > self.raw_dim:
            return arr[: self.raw_dim]
        return arr

    def observe_batch(self, states: np.ndarray) -> None:
        if not self.use_residual:
            return
        if states is None:
            return
        arr = np.asarray(states, dtype=np.float32)
        if arr.size == 0:
            return
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        aligned = np.stack([self._align_state(row) for row in arr], axis=0)
        batch_count = float(aligned.shape[0])
        batch_mean = np.mean(aligned, axis=0)
        total = self._running_count + batch_count
        if total <= self._eps:
            self._running_mean = batch_mean
            self._running_count = batch_count
            return
        weight_prev = self._running_count / total
        weight_new = batch_count / total
        self._running_mean = (
            weight_prev * self._running_mean + weight_new * batch_mean
        ).astype(np.float32, copy=False)
        self._running_count = total

    def _prepare_features(
        self,
        state: np.ndarray,
        action_mean: Optional[np.ndarray],
    ) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        components = [state]
        if self.use_residual and state.size == self.raw_dim:
            residual = state - self._running_mean
            if self.normalize:
                denom = np.linalg.norm(residual, ord=2) + self._eps
                residual = residual / denom
            components.append(residual.astype(np.float32, copy=False))
        if self.use_action_mean and self.action_dim > 0:
            if action_mean is None:
                action_mean = self._last_action_mean
            else:
                action_mean = np.asarray(action_mean, dtype=np.float32).reshape(-1)
                if action_mean.size != self.action_dim:
                    action_mean = self._last_action_mean
                else:
                    self._last_action_mean = action_mean
            components.append(action_mean.astype(np.float32, copy=False))
        enriched = np.concatenate(components, axis=0)
        return enriched

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def transform_state(
        self,
        state: np.ndarray,
        *,
        action_mean: Optional[np.ndarray] = None,
        update_stats: bool = False,
    ) -> np.ndarray:
        aligned_state = self._align_state(state)
        if update_stats:
            self.observe_batch(aligned_state.reshape(1, -1))
        enriched = self._prepare_features(aligned_state, action_mean)
        projected = self._projection @ enriched
        return projected.astype(np.float32, copy=False)

    def transform_sequence(
        self,
        states: np.ndarray,
        *,
        action_sequence: Optional[np.ndarray] = None,
        update_stats: bool = False,
    ) -> np.ndarray:
        arr = np.asarray(states, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.size == 0 or arr.shape[0] == 0:
            return np.zeros((0, self.output_dim), dtype=np.float32)
        if update_stats:
            self.observe_batch(arr)
        if action_sequence is not None:
            actions = np.asarray(action_sequence, dtype=np.float32)
            if actions.ndim == 1:
                actions = actions.reshape(1, -1)
            if actions.shape[-1] == self.action_dim:
                action_mean = np.mean(actions, axis=0)
            else:
                action_mean = None
        else:
            action_mean = None
        encoded = [
            self.transform_state(state, action_mean=action_mean, update_stats=False)
            for state in arr
        ]
        if not encoded:
            return np.zeros((0, self.output_dim), dtype=np.float32)
        return np.stack(encoded, axis=0)

    @property
    def projection_matrix(self) -> np.ndarray:
        return self._projection



class TransformerPolicy(nn.Module):
    """Simple transformer based policy used by MAT."""

    def __init__(
        self,
        num_inputs: int,
        num_actions: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_layer = nn.Linear(num_inputs, d_model)
        ff_dim = max(d_model, int(ff_dim) if ff_dim is not None else d_model * 4)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.mean_layer = nn.Linear(d_model, num_actions)
        self.log_std_layer = nn.Linear(d_model, num_actions)

    def forward(self, x: torch.Tensor):
        # x: [batch, num_inputs]
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        x = self.input_layer(x).unsqueeze(1)  # [batch, 1, d_model]
        x = self.encoder(x)
        x = self.output_norm(x.squeeze(1))
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

        dropout = float(config.sac.get('transformer_dropout', 0.1))
        ff_multiplier = float(config.sac.get('transformer_ff_multiplier', 2.0))
        ff_dim = max(hidden_actor, int(hidden_actor * ff_multiplier))

        self.policy = TransformerPolicy(
            num_inputs,
            self.action_dim,
            d_model=hidden_actor,
            nhead=4,
            num_layers=layers_actor,
            ff_dim=ff_dim,
            dropout=dropout,
        ).to(self.device)
        self.value = ValueNetwork(num_inputs, hidden_dims=hidden_critic).to(self.device)

        self.policy_optim = Adam(self.policy.parameters(), lr=config.sac['lr'])
        self.value_optim = Adam(self.value.parameters(), lr=config.sac['lr'])

        self.gamma = config.sac['gamma']
        self.updates_per_episode = max(1, int(config.sac.get('updates_per_episode_rpc', 1)))
        replay_window = int(config.sac.get('mat_replay_window', 0))
        self.replay_window = max(0, replay_window)
        self.worker_id = rank

        self.policy_loss_list = None
        self.value_loss_list = None
        self.entropy_list = None
        self._offline_stats: list[EpisodeStats] = []
        self._offline_summary = None
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
        if self.model_dir is not None:
            os.makedirs(self.model_dir, exist_ok=True)

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

    # ``begin_episode`` and ``record_reward`` are used by Decision Transformer
    # agents.  Provide benign defaults so MAT instances can be invoked through
    # the same RPC helpers without additional guards.

    def begin_episode(self):
        return

    def record_reward(self, *_args, **_kwargs):
        return

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
        if self.replay_window > 0 and len(transitions) > self.replay_window:
            transitions = transitions[-self.replay_window:]
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

        episode_id = int(episode)
        actor_filename = f"worker_{worker_id}_mat_actor_episode_{episode_id:06d}.pth"
        critic_filename = f"worker_{worker_id}_mat_value_episode_{episode_id:06d}.pth"
        actor_path = os.path.join(folder, actor_filename)
        critic_path = os.path.join(folder, critic_filename)

        torch.save({'model_state_dict': self.policy.state_dict()}, actor_path)
        torch.save({'model_state_dict': self.value.state_dict()}, critic_path)

        # Maintain deterministic "latest" checkpoints so training can resume
        # without having to know the exact episode number.
        latest_actor = os.path.join(folder, f"worker_{worker_id}_mat_actor_latest.pth")
        latest_critic = os.path.join(folder, f"worker_{worker_id}_mat_value_latest.pth")
        try:
            copy2(actor_path, latest_actor)
            copy2(critic_path, latest_critic)
        except OSError:
            # Fall back to saving the current state directly when running on a
            # filesystem that does not support copy-on-write semantics.
            torch.save({'model_state_dict': self.policy.state_dict()}, latest_actor)
            torch.save({'model_state_dict': self.value.state_dict()}, latest_critic)

    def load_policy(self, master_rref, worker_id):
        assert worker_id == self.worker_id
        folder = self.model_dir or os.path.join(self.config.savedir, "output_models", "models_rpc")
        if not os.path.isdir(folder):
            return

        def _resolve_latest(prefix: str) -> Optional[str]:
            latest_path = os.path.join(folder, f"{prefix}_latest.pth")
            if os.path.exists(latest_path):
                return latest_path
            pattern = os.path.join(folder, f"{prefix}_episode_*.pth")
            candidates = sorted(glob(pattern))
            if candidates:
                return candidates[-1]
            return None

        actor_path = _resolve_latest(f"worker_{worker_id}_mat_actor")
        critic_path = _resolve_latest(f"worker_{worker_id}_mat_value")

        if actor_path and os.path.exists(actor_path):
            model_dict = torch.load(actor_path, map_location=self.device)
            self.policy.load_state_dict(model_dict["model_state_dict"])
        if critic_path and os.path.exists(critic_path):
            model_dict = torch.load(critic_path, map_location=self.device)
            self.value.load_state_dict(model_dict["model_state_dict"])


class DecisionTransformerPolicy(nn.Module):
    """Decision Transformer style policy operating on return/state/action sequences."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        context_len: int,
        dropout: float = 0.1,
        ff_dim: Optional[int] = None,
    ):
        super().__init__()
        self.context_len = context_len
        self.state_embed = nn.Linear(state_dim, d_model)
        self.action_embed = nn.Linear(action_dim, d_model)
        self.return_embed = nn.Linear(1, d_model)
        ff_dim = max(d_model, int(ff_dim) if ff_dim is not None else d_model * 4)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.position_embedding = nn.Parameter(torch.zeros(context_len, d_model))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.norm = nn.LayerNorm(d_model)
        self.mean_head = nn.Linear(d_model, action_dim)
        self.log_std_head = nn.Linear(d_model, action_dim)

    def forward(self, states, actions, returns, padding_mask=None):
        """Forward pass for a batch of sequences.

        Args:
            states: Tensor of shape [B, T, state_dim].
            actions: Tensor of shape [B, T, action_dim].  The last entry is a
                placeholder for the action to be predicted.
            returns: Tensor of shape [B, T, 1] with return-to-go targets.
            padding_mask: Optional boolean mask of shape [B, T] marking padded
                positions as ``True``.
        """

        x = (
            self.state_embed(states)
            + self.action_embed(actions)
            + self.return_embed(returns)
        )
        pos = self.position_embedding.unsqueeze(0)[:, : x.size(1), :]
        x = x + pos
        if padding_mask is not None:
            x = self.transformer(x, src_key_padding_mask=padding_mask)
        else:
            x = self.transformer(x)
        x = self.norm(x[:, -1])
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std


class DecisionTransformer(MAT):
    """Decision Transformer agent compatible with the RPC trainer interface."""

    def __init__(self, num_inputs, action_space, config, rank, num_gpus, model_dir=None):
        super().__init__(num_inputs, action_space, config, rank, num_gpus, model_dir=model_dir)

        self.raw_state_dim = int(getattr(self, "state_dim", num_inputs))
        self.feature_engineer = DecisionTransformerFeatureEngineer(
            self.raw_state_dim,
            self.action_dim,
            config,
        )
        self.state_dim = self.feature_engineer.output_dim

        hidden_actor = config.sac['hidden_size_actor']
        layers_actor = config.sac['num_layers_actor']
        dropout = float(config.sac.get('transformer_dropout', 0.1))
        context_len = int(config.sac.get('dt_context_len', 8))
        context_len = max(2, context_len)
        nhead = max(1, int(config.sac.get('dt_nhead', 4)))
        dt_layers_override = config.sac.get('dt_num_layers')
        if dt_layers_override:
            layers_actor = int(dt_layers_override)
        if hidden_actor % nhead != 0:
            raise ValueError(
                f"Transformer hidden size {hidden_actor} must be divisible by nhead={nhead}."
            )
        ff_multiplier = float(config.sac.get('transformer_ff_multiplier', 2.0))
        ff_dim = max(hidden_actor, int(hidden_actor * ff_multiplier))

        self.policy = DecisionTransformerPolicy(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            d_model=hidden_actor,
            nhead=nhead,
            num_layers=layers_actor,
            context_len=context_len,
            dropout=dropout,
            ff_dim=ff_dim,
        ).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=config.sac['lr'])

        # Value network is not used for optimisation but kept for checkpoint
        # compatibility with MAT.
        self.value = ValueNetwork(self.state_dim, hidden_dims=config.sac['hidden_size_critic']).to(self.device)
        self.value_optim = Adam(self.value.parameters(), lr=config.sac['lr'])

        self.context_len = context_len
        self.target_return = float(config.sac.get('dt_target_return', 1.0))
        self.dt_discount = float(config.sac.get('dt_discount', self.gamma))
        self.action_scale = float(config.sac.get('dt_action_scale', 1.0))
        self.return_scale = float(config.sac.get('dt_return_scale', 1.0))
        if abs(self.return_scale) < 1e-12:
            self.return_scale = 1.0
        self.return_clip = float(config.sac.get('dt_return_clip', 0.0))
        self.return_floor_ratio = float(config.sac.get('dt_return_floor_ratio', 0.0))
        if self.return_floor_ratio < 0.0:
            self.return_floor_ratio = 0.0
        self.step_return_scale = max(0.0, float(config.sac.get('dt_step_return_scale', 0.5)))
        self.step_return_beta = float(config.sac.get('dt_step_return_beta', 0.15))
        self.step_return_beta = min(max(self.step_return_beta, 0.0), 1.0)
        self.sequence_stride = max(1, int(config.sac.get('dt_sequence_stride', 1)))
        normalize_returns = config.sac.get('dt_normalize_returns', True)
        if isinstance(normalize_returns, str):
            normalize_returns = normalize_returns.lower() == 'true'
        self.normalize_returns = bool(normalize_returns)
        self.return_norm_epsilon = float(config.sac.get('dt_return_norm_epsilon', 1e-6))
        self.target_momentum = float(config.sac.get('dt_target_momentum', 0.9))
        self.target_momentum = min(max(self.target_momentum, 0.0), 1.0)
        self.strehl_momentum = float(config.sac.get('dt_strehl_momentum', self.target_momentum))
        self.strehl_momentum = min(max(self.strehl_momentum, 0.0), 1.0)
        self.target_offset = float(config.sac.get('dt_target_offset', 0.0))
        self.target_gain = float(config.sac.get('dt_target_gain', 0.0))
        self.target_min = float(config.sac.get('dt_target_min', 0.0))
        self.strehl_goal = float(config.sac.get('dt_target_strehl', 0.8))
        self.strehl_goal_weight = float(
            config.sac.get('dt_target_strehl_weight', self.target_return)
        )
        if self.strehl_goal_weight < 0.0:
            self.strehl_goal_weight = 0.0
        self.replay_episodes = max(1, int(config.sac.get('dt_replay_episodes', 4)))
        self.best_history_capacity = max(
            self.replay_episodes,
            int(config.sac.get('dt_best_episodes', self.replay_episodes)),
        )
        self.recent_window = max(1, int(config.sac.get('dt_recent_episodes', 1)))
        self.loss_temperature = max(0.0, float(config.sac.get('dt_loss_temperature', 1.0)))
        self.recent_weight = max(0.0, float(config.sac.get('dt_recent_weight', 0.0)))
        self.strehl_quality_weight = float(config.sac.get('dt_quality_strehl_weight', 0.0))
        self.sequences_topk = int(config.sac.get('dt_sequences_topk', 0))
        self.sequences_min_keep_recent = int(
            config.sac.get('dt_sequences_min_keep_recent', 0)
        )
        def _clamp_unit(value: float) -> float:
            return min(max(float(value), 0.0), 1.0)

        offline_ratio_cfg = _clamp_unit(
            float(config.sac.get('dt_replay_offline_ratio', 0.5))
        )
        recent_ratio_cfg = _clamp_unit(
            float(config.sac.get('dt_replay_recent_ratio', 0.2))
        )
        recent_ratio_cap = _clamp_unit(
            float(config.sac.get('dt_replay_recent_ratio_cap', recent_ratio_cfg))
        )
        if recent_ratio_cap > 0.0:
            recent_ratio_cfg = min(recent_ratio_cfg, recent_ratio_cap)
        if offline_ratio_cfg + recent_ratio_cfg > 1.0:
            total = offline_ratio_cfg + recent_ratio_cfg
            offline_ratio_cfg = offline_ratio_cfg / total
            recent_ratio_cfg = recent_ratio_cfg / total
        self.replay_offline_ratio = offline_ratio_cfg
        self.replay_recent_ratio = recent_ratio_cfg
        self.replay_recent_ratio_cap = recent_ratio_cap
        if (
            self.sequences_topk > 0
            and self.sequences_min_keep_recent > 0
            and self.replay_recent_ratio_cap > 0.0
        ):
            max_recent_allowed = int(
                math.floor(self.sequences_topk * self.replay_recent_ratio_cap)
            )
            if max_recent_allowed > 0:
                self.sequences_min_keep_recent = min(
                    self.sequences_min_keep_recent, max_recent_allowed
                )
        self.replay_online_top_percentile = _clamp_unit(
            float(config.sac.get('dt_replay_online_top_percentile', 0.2))
        )
        self.online_keep_percentile = _clamp_unit(
            float(config.sac.get('dt_online_keep_percentile', 0.7))
        )
        self.online_keep_min_samples = max(
            1, int(config.sac.get('dt_online_keep_min_samples', 32))
        )
        history_limit_cfg = int(config.sac.get('dt_online_history_limit', 2048))
        self._online_history_limit = history_limit_cfg if history_limit_cfg > 0 else None
        self._online_return_history: list[float] = []
        self._offline_improvement_baseline: Optional[float] = None
        self._offline_improvement_votes = 0
        self._offline_decay_triggered = False
        self._offline_decay_trigger_update: Optional[int] = None
        self._total_update_counter = 0
        low_weight_maxlen = int(config.sac.get('dt_low_weight_maxlen', max(self.recent_window, 32)))
        if low_weight_maxlen <= 0:
            low_weight_maxlen = self.recent_window
        self._low_weight_episodes: deque = deque(maxlen=low_weight_maxlen)
        self.bc_logprob_coef = float(config.sac.get('dt_bc_logprob_coef', 1.0))
        if self.bc_logprob_coef < 0.0:
            self.bc_logprob_coef = 0.0
        self.bc_mse_coef = float(config.sac.get('dt_bc_mse_coef', 0.1))
        if self.bc_mse_coef < 0.0:
            self.bc_mse_coef = 0.0
        self.bc_reg_coef = float(config.sac.get('dt_bc_reg_coef', 0.01))
        if self.bc_reg_coef < 0.0:
            self.bc_reg_coef = 0.0
        self.offline_weight_gain = float(config.sac.get('dt_offline_weight_gain', 0.0))
        if self.offline_weight_gain < 0.0:
            self.offline_weight_gain = 0.0
        offline_patterns = []
        raw_pattern = config.sac.get('dt_offline_dataset_glob')

        class _SafeFormatDict(dict):
            def __missing__(self, key):
                return ""

        def _append_if_matches(pattern: str) -> None:
            if not pattern:
                return
            matches = glob(pattern, recursive=True)
            if matches and pattern not in offline_patterns:
                offline_patterns.append(pattern)

        if isinstance(raw_pattern, str):
            pattern_candidates = [chunk.strip() for chunk in raw_pattern.split(';') if chunk.strip()]
            formatter = _SafeFormatDict(
                savedir=getattr(config, 'savedir', ''),
                experiment=getattr(config, 'experiment_name', ''),
                algorithm=getattr(config, 'algorithm', ''),
            )
            for candidate in pattern_candidates:
                if candidate.lower() == 'none':
                    continue
                try:
                    formatted = candidate.format_map(formatter)
                except (KeyError, ValueError):
                    formatted = candidate

                search_paths = []
                if os.path.isabs(formatted):
                    search_paths.append(formatted)
                else:
                    search_paths.append(formatted)
                    savedir = getattr(config, 'savedir', '') or ''
                    if savedir:
                        search_paths.append(os.path.join(savedir, formatted))
                    search_paths.append(os.path.join(os.getcwd(), formatted))

                chosen = None
                for option in search_paths:
                    if not option:
                        continue
                    if glob(option, recursive=True):
                        chosen = option
                        break
                if chosen is None and search_paths:
                    chosen = search_paths[0]
                if chosen:
                    _append_if_matches(chosen)

        if not offline_patterns:
            fallback_patterns = self._discover_offline_patterns(config)
            for pattern in fallback_patterns:
                _append_if_matches(pattern)

        self.offline_dataset_globs = offline_patterns
        self.offline_dataset_glob = offline_patterns[0] if offline_patterns else None
        self.offline_mix_ratio = max(0.0, float(config.sac.get('dt_offline_mix_ratio', 0.0)))
        self.offline_mix_ratio_start = max(
            0.0, float(config.sac.get('dt_offline_mix_ratio_start', self.offline_mix_ratio))
        )
        self.offline_mix_ratio_final = max(
            0.0, float(config.sac.get('dt_offline_mix_ratio_final', self.offline_mix_ratio))
        )
        self.offline_mix_ratio_decay = max(
            1, int(config.sac.get('dt_offline_mix_ratio_decay', 1))
        )
        self.offline_mix_ratio_warmup = max(
            0, int(config.sac.get('dt_offline_mix_ratio_warmup', 0))
        )
        self.offline_lock_ratio = _clamp_unit(
            float(config.sac.get('dt_offline_lock_ratio', self.offline_mix_ratio_start))
        )
        self.offline_final_ratio = _clamp_unit(
            float(config.sac.get('dt_offline_final_ratio', self.offline_mix_ratio_final))
        )
        if self.offline_final_ratio > self.offline_lock_ratio:
            self.offline_final_ratio = self.offline_lock_ratio
        self.offline_lock_updates = max(
            0, int(config.sac.get('dt_offline_lock_updates', 0))
        )
        self.offline_decay_updates = max(
            1, int(config.sac.get('dt_offline_decay_updates', 1))
        )
        self.offline_improvement_window = max(
            1, int(config.sac.get('dt_offline_improvement_window', 128))
        )
        self.offline_improvement_threshold = max(
            0.0, float(config.sac.get('dt_offline_improvement_threshold', 0.05))
        )
        self.offline_improvement_min_delta = max(
            0.0, float(config.sac.get('dt_offline_improvement_min_delta', 0.0))
        )
        self.offline_improvement_patience = max(
            1, int(config.sac.get('dt_offline_improvement_patience', 1))
        )
        self.offline_max_episodes = max(0, int(config.sac.get('dt_offline_max_episodes', 0)))
        self.offline_keep_top_ratio = float(config.sac.get('dt_offline_keep_top_ratio', 0.0))
        self.offline_keep_strehl_ratio = float(config.sac.get('dt_offline_keep_strehl_ratio', 0.0))
        self.offline_min_return = config.sac.get('dt_offline_min_return', None)
        self.offline_min_strehl = config.sac.get('dt_offline_min_strehl', None)
        self.offline_elite_count_cfg = max(
            0, int(config.sac.get('dt_offline_elite_count', 0))
        )
        self.offline_elite_fraction = max(
            0.0, float(config.sac.get('dt_offline_elite_fraction', 0.0))
        )
        self.offline_reserve_limit = max(
            0, int(config.sac.get('dt_offline_reserve_limit', 0))
        )
        self._offline_episode_counter = 0
        self._offline_episodes = []
        self._offline_stats = []
        self._offline_elite: list[tuple[float, dict]] = []
        self._offline_reserve: list[tuple[float, dict]] = []

        self.context_states = []
        self.context_actions = []
        self.context_returns = []
        self.current_return = self.target_return
        self._best_reward_episodes = []
        self._best_strehl_episodes = []
        if self.offline_dataset_globs:
            loaded_total = []
            for pattern in self.offline_dataset_globs:
                loaded = self._load_offline_dataset(pattern)
                if loaded:
                    loaded_total.extend(loaded)
            if loaded_total:
                stats_all = [
                    ep[1].get('stats') for ep in loaded_total if ep[1].get('stats') is not None
                ]
                filtered_eps = loaded_total
                filter_drop = 0
                return_cut = None
                strehl_cut = None
                if stats_all:
                    filtered_eps, filter_drop, return_cut, strehl_cut = self._apply_offline_filters(
                        loaded_total, stats_all
                    )
                if filter_drop > 0:
                    conditions = []
                    if return_cut is not None:
                        conditions.append(f"return>={return_cut:.3f}")
                    if strehl_cut is not None:
                        conditions.append(f"strehl>={strehl_cut:.4f}")
                    suffix = f" ({', '.join(conditions)})" if conditions else ""
                    print(
                        f"[DecisionTransformer] Filtered {filter_drop} offline episodes below thresholds"
                        f"{suffix}"
                    )
                self._offline_episodes = filtered_eps
                self._offline_stats = [
                    ep[1].get('stats')
                    for ep in self._offline_episodes
                    if ep[1].get('stats') is not None
                ]
                self._refresh_offline_pools(self._offline_episodes)
                for score, episode in self._offline_episodes:
                    self._insert_episode_sorted(
                        self._best_reward_episodes,
                        score,
                        self.replay_episodes,
                        episode,
                    )
                    self._insert_episode_sorted(
                        self._best_strehl_episodes,
                        float(episode.get('avg_strehl', 0.0)),
                        self.best_history_capacity,
                        episode,
                    )
                summary = summarize_dataset(self._offline_stats)
                self._offline_summary = summary
                summary_msg = (
                    f"[DecisionTransformer] Loaded {summary['num_eps']} offline episodes "
                    f"from {len(self.offline_dataset_globs)} pattern(s). "
                    f"return_mean={summary['ret_mean']:.3f} ret_std={summary['ret_std']:.3f} "
                    f"top20_ratio={summary['top20_ratio']:.3f}"
                )
                if summary.get("strehl_mean") is not None:
                    strehl_mean = summary.get("strehl_mean") or 0.0
                    strehl_std = summary.get("strehl_std") or 0.0
                    summary_msg += (
                        f" strehl_mean={strehl_mean:.4f} strehl_std={strehl_std:.4f}"
                    )
                print(summary_msg)
                if self.model_dir:
                    try:
                        with open(
                            os.path.join(self.model_dir, "offline_stats.json"),
                            "w",
                            encoding="utf-8",
                        ) as fh:
                            json.dump(summary, fh, indent=2)
                    except OSError:
                        pass
        else:
            print(
                "[DecisionTransformer] No offline dataset found. Set dt_offline_dataset_glob or "
                "place .npz files under an offline_datasets directory to enable offline mixing."
            )
        self._episode_return = 0.0
        self._return_ema = self.target_return
        self._strehl_ema = None
        self._recent_episodes = deque(maxlen=self.recent_window)
        self._dt_episode_counter = 0
        self._episode_strehl_values = []
        self._last_episode_avg_strehl = 0.0

        updates_override = config.sac.get('dt_updates_per_episode')
        if updates_override is not None:
            self.updates_per_episode = max(1, int(updates_override))

    def _discover_offline_patterns(self, config) -> list:
        """Return a list of glob patterns pointing at offline dataset files."""

        patterns = []
        seen_dirs = set()

        def _add_directory(directory: str) -> None:
            if not directory:
                return
            abs_dir = os.path.abspath(directory)
            if abs_dir in seen_dirs or not os.path.isdir(abs_dir):
                return
            seen_dirs.add(abs_dir)
            pattern = os.path.join(abs_dir, "**", "*.npz")
            if glob(pattern, recursive=True):
                patterns.append(pattern)

        savedir = getattr(config, 'savedir', '') or ''
        experiment = getattr(config, 'experiment_name', '') or ''
        cwd = os.getcwd()

        candidate_roots = []
        if savedir:
            candidate_roots.append(savedir)
            candidate_roots.append(os.path.join(cwd, savedir))
        candidate_roots.append(cwd)

        for root in list(candidate_roots):
            if not root:
                continue
            if experiment:
                candidate_roots.append(os.path.join(root, experiment))

        for root in candidate_roots:
            offline_dir = os.path.join(root, 'offline_datasets')
            _add_directory(offline_dir)

        wildcard_dirs = set()
        wildcard_dirs.update(
            glob(os.path.join(cwd, 'output*', 'offline_datasets'))
        )
        wildcard_dirs.update(
            glob(os.path.join(cwd, '*', 'offline_datasets'))
        )
        for directory in sorted(wildcard_dirs):
            _add_directory(directory)

        return patterns

    def _insert_episode_sorted(self, store, score, capacity, episode):
        if capacity <= 0:
            return
        episode_id = episode.get("episode_id")
        filtered = []
        for existing_score, existing_episode in store:
            if episode_id is not None and existing_episode.get("episode_id") == episode_id:
                continue
            filtered.append((existing_score, existing_episode))
        filtered.append((float(score), episode))
        filtered.sort(key=lambda item: item[0], reverse=True)
        if len(filtered) > capacity:
            del filtered[capacity:]
        store[:] = filtered

    def _record_online_return(self, value: float) -> None:
        if not np.isfinite(value):
            return
        self._online_return_history.append(float(value))
        limit = self._online_history_limit
        if limit is not None and len(self._online_return_history) > limit:
            excess = len(self._online_return_history) - limit
            if excess > 0:
                del self._online_return_history[:excess]
        self._maybe_update_offline_decay()

    def _compute_online_quantile(self, quantile: float, default: float = float("-inf")) -> float:
        data = self._online_return_history
        if not data:
            return default
        q = min(max(float(quantile), 0.0), 1.0)
        try:
            return float(np.quantile(np.asarray(data, dtype=np.float64), q))
        except (ValueError, IndexError, TypeError):
            return default

    def _should_keep_online_episode(self, reward: float) -> bool:
        if not np.isfinite(reward):
            reward = float("-inf")
        if len(self._online_return_history) < self.online_keep_min_samples:
            return True
        threshold = self._compute_online_quantile(self.online_keep_percentile, default=float("-inf"))
        if threshold == float("-inf"):
            return True
        return reward >= threshold

    def _maybe_update_offline_decay(self) -> None:
        if self.offline_lock_ratio <= 0.0:
            return
        if self._offline_decay_triggered:
            return

        window = max(1, self.offline_improvement_window)
        history = self._online_return_history
        if len(history) < window:
            return

        recent_avg = float(np.mean(history[-window:]))
        if not np.isfinite(recent_avg):
            return

        if self._offline_improvement_baseline is None:
            baseline_slice = history[:window]
            baseline = float(np.mean(baseline_slice))
            if not np.isfinite(baseline):
                return
            self._offline_improvement_baseline = baseline
            self._offline_improvement_votes = 0
            if len(history) < window * 2:
                return

        baseline = self._offline_improvement_baseline
        if baseline is None or not np.isfinite(baseline):
            return

        improvement = recent_avg - baseline
        threshold_abs = max(
            self.offline_improvement_min_delta,
            abs(baseline) * self.offline_improvement_threshold,
        )
        if improvement >= threshold_abs:
            self._offline_improvement_votes += 1
            if self._offline_improvement_votes >= self.offline_improvement_patience:
                self._offline_decay_triggered = True
                self._offline_decay_trigger_update = self._total_update_counter
                self._offline_improvement_baseline = recent_avg
        else:
            self._offline_improvement_votes = 0
            if improvement < 0.0:
                blend = 0.1
                self._offline_improvement_baseline = (1.0 - blend) * baseline + blend * recent_avg

    def _select_balanced_sequences(self, sequence_items, target_total):
        if not sequence_items:
            return []

        total_available = len(sequence_items)
        target_total = int(max(1, min(int(target_total), total_available)))

        categories = {
            "offline": [],
            "recent": [],
            "online_top": [],
            "online_other": [],
        }
        for item in sequence_items:
            categories.setdefault(item.get("category", "online_other"), []).append(item)

        offline_candidates = sorted(
            categories.get("offline", []), key=lambda x: x["quality"], reverse=True
        )
        recent_candidates = sorted(
            categories.get("recent", []), key=lambda x: x.get("order", 0), reverse=True
        )
        online_top_candidates = sorted(
            categories.get("online_top", []), key=lambda x: x["quality"], reverse=True
        )
        online_other_candidates = sorted(
            categories.get("online_other", []), key=lambda x: x["quality"], reverse=True
        )

        offline_target = int(round(target_total * self.replay_offline_ratio))
        if self.replay_offline_ratio > 0.0 and offline_candidates and offline_target <= 0:
            offline_target = 1
        offline_target = min(offline_target, len(offline_candidates), target_total)

        recent_target = int(round(target_total * self.replay_recent_ratio))
        if self.replay_recent_ratio > 0.0 and recent_candidates and recent_target <= 0:
            recent_target = 1
        recent_target = min(recent_target, len(recent_candidates))
        if self.sequences_min_keep_recent > 0 and recent_candidates:
            required_recent = min(self.sequences_min_keep_recent, len(recent_candidates))
            if required_recent > recent_target:
                recent_target = min(required_recent, target_total)

        selected = []
        used_indices = set()

        for item in offline_candidates[:offline_target]:
            selected.append(item)
            used_indices.add(item["index"])

        for item in recent_candidates[:recent_target]:
            if item["index"] in used_indices:
                continue
            selected.append(item)
            used_indices.add(item["index"])

        remaining_slots = target_total - len(selected)
        online_target = max(0, target_total - offline_target - recent_target)
        online_selected = []
        for item in online_top_candidates:
            if len(online_selected) >= online_target or remaining_slots <= 0:
                break
            if item["index"] in used_indices:
                continue
            selected.append(item)
            online_selected.append(item)
            used_indices.add(item["index"])
            remaining_slots -= 1

        if remaining_slots > 0:
            def _fill_from(pool):
                nonlocal remaining_slots
                for candidate in pool:
                    if remaining_slots <= 0:
                        break
                    if candidate["index"] in used_indices:
                        continue
                    selected.append(candidate)
                    used_indices.add(candidate["index"])
                    remaining_slots -= 1

            # Prioritize remaining offline, then top online, then recent, then fallback
            _fill_from(offline_candidates[offline_target:])
            _fill_from(online_top_candidates[len(online_selected):])
            _fill_from(recent_candidates[recent_target:])
            _fill_from(online_other_candidates)

        if len(selected) > target_total:
            selected = selected[:target_total]

        return sorted(selected, key=lambda x: x["index"])

    def _match_feature_dim(self, array: np.ndarray, target_dim: int) -> np.ndarray:
        if array.shape[1] == target_dim:
            return array
        if array.shape[1] < target_dim:
            pad_width = target_dim - array.shape[1]
            pad = np.zeros((array.shape[0], pad_width), dtype=array.dtype)
            return np.concatenate([array, pad], axis=1)
        return array[:, :target_dim]

    def _coerce_offline_arrays(self, states, actions, rewards, masks, strehl, base_id):
        episodes = []
        if states is None or actions is None or rewards is None:
            return episodes

        states_np = np.asarray(states)
        if states_np.size == 0:
            return episodes

        if states_np.dtype == object and states_np.ndim == 1:
            for idx, sub_states in enumerate(states_np.tolist()):
                sub_actions = None
                if isinstance(actions, (list, tuple)):
                    sub_actions = actions[idx]
                else:
                    actions_np = np.asarray(actions)
                    if actions_np.ndim >= 1 and idx < actions_np.shape[0]:
                        sub_actions = actions_np[idx]
                    else:
                        sub_actions = actions
                sub_rewards = None
                rewards_np = np.asarray(rewards)
                if rewards_np.ndim >= 2 and idx < rewards_np.shape[0]:
                    sub_rewards = rewards_np[idx]
                elif rewards_np.ndim == 1 and rewards_np.size > 0:
                    sub_rewards = rewards_np
                else:
                    sub_rewards = rewards
                sub_masks = None
                if masks is not None:
                    masks_np = np.asarray(masks)
                    if masks_np.ndim >= 2 and idx < masks_np.shape[0]:
                        sub_masks = masks_np[idx]
                    elif masks_np.ndim == 1:
                        sub_masks = masks_np
                sub_strehl = None
                if strehl is not None:
                    strehl_np = np.asarray(strehl)
                    if strehl_np.ndim >= 1 and idx < strehl_np.shape[0]:
                        sub_strehl = strehl_np[idx]
                    else:
                        sub_strehl = strehl
                episodes.extend(
                    self._coerce_offline_arrays(
                        sub_states,
                        sub_actions,
                        sub_rewards,
                        sub_masks,
                        sub_strehl,
                        f"{base_id}-{idx}",
                    )
                )
            return episodes

        if states_np.ndim == 3:
            for idx in range(states_np.shape[0]):
                sub_states = states_np[idx]
                actions_np = np.asarray(actions)
                if actions_np.ndim >= 3:
                    sub_actions = actions_np[idx]
                else:
                    sub_actions = actions
                rewards_np = np.asarray(rewards)
                if rewards_np.ndim >= 2:
                    sub_rewards = rewards_np[idx]
                else:
                    sub_rewards = rewards
                sub_masks = None
                if masks is not None:
                    masks_np = np.asarray(masks)
                    if masks_np.ndim >= 2:
                        sub_masks = masks_np[idx]
                    else:
                        sub_masks = masks
                sub_strehl = None
                if strehl is not None:
                    strehl_np = np.asarray(strehl)
                    if strehl_np.ndim >= 1 and idx < strehl_np.shape[0]:
                        sub_strehl = strehl_np[idx]
                    else:
                        sub_strehl = strehl
                episodes.extend(
                    self._coerce_offline_arrays(
                        sub_states,
                        sub_actions,
                        sub_rewards,
                        sub_masks,
                        sub_strehl,
                        f"{base_id}-{idx}",
                    )
                )
            return episodes

        states_np = np.nan_to_num(states_np, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        if states_np.ndim != 2:
            return episodes
        timestep = states_np.shape[0]
        states_np = self._match_feature_dim(states_np, self.raw_state_dim)

        actions_np = np.asarray(actions)
        actions_np = np.nan_to_num(actions_np, nan=0.0, posinf=0.0, neginf=0.0)
        if actions_np.ndim == 1:
            actions_np = actions_np.reshape(-1, self.action_dim)
        if actions_np.ndim == 2 and actions_np.shape[1] != self.action_dim:
            actions_np = self._match_feature_dim(actions_np.astype(np.float32, copy=False), self.action_dim)
        elif actions_np.ndim != 2:
            return episodes
        actions_np = actions_np.astype(np.float32, copy=False)
        if actions_np.shape[0] < timestep:
            pad = np.zeros((timestep, self.action_dim), dtype=np.float32)
            pad[: actions_np.shape[0]] = actions_np[: actions_np.shape[0]]
            actions_np = pad
        elif actions_np.shape[0] > timestep:
            actions_np = actions_np[:timestep]

        self.feature_engineer.observe_batch(states_np)
        encoded_states = self.feature_engineer.transform_sequence(
            states_np,
            action_sequence=actions_np,
            update_stats=False,
        )

        rewards_np = np.asarray(rewards, dtype=np.float32).reshape(-1)
        if rewards_np.size < timestep:
            rewards_np = np.pad(rewards_np, (0, timestep - rewards_np.size), mode='constant')
        elif rewards_np.size > timestep:
            rewards_np = rewards_np[:timestep]

        if masks is None:
            masks_np = np.ones(timestep, dtype=np.float32)
            masks_np[-1] = 0.0
        else:
            masks_np = np.asarray(masks, dtype=np.float32).reshape(-1)
            if masks_np.size < timestep:
                pad = np.ones(timestep, dtype=np.float32)
                pad[: masks_np.size] = masks_np
                pad[masks_np.size:] = 0.0
                masks_np = pad
            elif masks_np.size > timestep:
                masks_np = masks_np[:timestep]
            if masks_np.size == 0:
                masks_np = np.ones(timestep, dtype=np.float32)
            masks_np[-1] = 0.0

        avg_strehl = 0.0
        strehl_np = None
        if strehl is not None:
            strehl_np = np.asarray(strehl)
            if strehl_np.size == 1:
                avg_strehl = float(strehl_np.reshape(-1)[0])
            elif strehl_np.ndim >= 1:
                avg_strehl = float(np.nanmean(strehl_np))

        episode_id = f"{base_id}-{self._offline_episode_counter}"
        self._offline_episode_counter += 1
        episode = {
            "states": encoded_states,
            "actions": actions_np,
            "rewards": rewards_np,
            "masks": masks_np,
            "episode_id": episode_id,
            "avg_strehl": avg_strehl,
            "strehl_series": strehl_np,
        }
        stats_payload = {
            "ep_id": episode_id,
            "reward": rewards_np,
            "strehl": strehl_np,
        }
        episode_stats = compute_episode_stats(stats_payload, reward_key="reward", strehl_key="strehl")
        episode["stats"] = episode_stats
        self._offline_stats.append(episode_stats)
        episodes.append(episode)
        return episodes

    def _load_offline_dataset(self, pattern: str):
        episodes = []
        paths = sorted(glob(pattern, recursive=True))
        if not paths:
            return episodes

        total_limit = self.offline_max_episodes if self.offline_max_episodes > 0 else None
        count = 0
        for path_idx, path in enumerate(paths):
            if total_limit is not None and count >= total_limit:
                break
            try:
                with np.load(path, allow_pickle=True) as data:
                    if 'episodes' in data.files:
                        raw_episodes = data['episodes']
                        try:
                            iterable = list(raw_episodes)
                        except Exception:
                            iterable = raw_episodes
                        for idx, entry in enumerate(iterable):
                            if isinstance(entry, np.ndarray) and entry.dtype == object and entry.shape == ():
                                entry = entry.item()
                            if isinstance(entry, dict):
                                sub_states = entry.get('states')
                                sub_actions = entry.get('actions')
                                sub_rewards = entry.get('rewards')
                                sub_masks = entry.get('masks')
                                sub_strehl = entry.get('avg_strehl', entry.get('strehl'))
                                parsed = self._coerce_offline_arrays(
                                    sub_states,
                                    sub_actions,
                                    sub_rewards,
                                    sub_masks,
                                    sub_strehl,
                                    f"offline-file{path_idx}-entry{idx}",
                                )
                                for episode in parsed:
                                    episodes.append(episode)
                                    count += 1
                                    if total_limit is not None and count >= total_limit:
                                        break
                            if total_limit is not None and count >= total_limit:
                                break
                    else:
                        states = data.get('states')
                        actions = data.get('actions')
                        rewards = data.get('rewards')
                        masks = data.get('masks')
                        strehl = data.get('avg_strehl', data.get('strehl'))
                        parsed = self._coerce_offline_arrays(
                            states,
                            actions,
                            rewards,
                            masks,
                            strehl,
                            f"offline-file{path_idx}",
                        )
                        for episode in parsed:
                            episodes.append(episode)
                            count += 1
                            if total_limit is not None and count >= total_limit:
                                break
            except OSError:
                continue
        prepared = []
        for episode in episodes:
            rewards_np = episode['rewards']
            quality_reward = float(np.sum(rewards_np))
            avg_strehl = float(episode.get('avg_strehl', 0.0))
            quality_metric = quality_reward + self.strehl_quality_weight * avg_strehl
            episode['quality_reward'] = quality_reward
            episode['quality_metric'] = quality_metric
            episode['is_offline'] = True
            prepared.append((quality_metric, episode))
        prepared.sort(key=lambda item: item[0], reverse=True)
        if total_limit is not None and len(prepared) > total_limit:
            prepared = prepared[:total_limit]
        return prepared

    def _apply_offline_filters(self, episodes, stats_list):
        """Filter offline episodes using configured thresholds."""

        if not episodes or not stats_list:
            return episodes, 0, None, None

        min_return = self.offline_min_return
        min_strehl = self.offline_min_strehl

        returns = np.array([s.sum_return for s in stats_list], dtype=np.float64)
        if self.offline_keep_top_ratio > 0.0:
            quantile = np.clip(1.0 - self.offline_keep_top_ratio, 0.0, 1.0)
            top_threshold = float(np.quantile(returns, quantile))
            if min_return is None or top_threshold > min_return:
                min_return = top_threshold

        strehl_values = [
            s.mean_strehl for s in stats_list if not np.isnan(s.mean_strehl)
        ]
        if strehl_values and self.offline_keep_strehl_ratio > 0.0:
            quantile = np.clip(1.0 - self.offline_keep_strehl_ratio, 0.0, 1.0)
            strehl_threshold = float(np.quantile(strehl_values, quantile))
            if min_strehl is None or strehl_threshold > min_strehl:
                min_strehl = strehl_threshold

        if min_return is None and min_strehl is None:
            return episodes, 0, None, None

        kept = []
        kept_stats = []
        dropped = 0
        for item in episodes:
            score, episode = item
            stats = episode.get('stats')
            if stats is None:
                kept.append(item)
                continue
            if min_return is not None and stats.sum_return < min_return:
                dropped += 1
                continue
            if min_strehl is not None:
                mean_strehl = stats.mean_strehl
                if np.isnan(mean_strehl) or mean_strehl < min_strehl:
                    dropped += 1
                    continue
            kept.append(item)
            kept_stats.append(stats)

        if not kept:
            # Avoid wiping the offline buffer entirely; fall back to original episodes.
            kept = episodes
            kept_stats = [episode.get('stats') for _score, episode in episodes if episode.get('stats')]

        # Update the stats cache so downstream summaries reflect the filtered set.
        self._offline_stats = kept_stats
        self._refresh_offline_pools(kept)
        return kept, dropped, min_return, min_strehl

    def _refresh_offline_pools(self, episodes):
        """Partition offline data into elite and reserve buffers."""

        if not episodes:
            self._offline_elite = []
            self._offline_reserve = []
            return

        episodes_sorted = sorted(episodes, key=lambda item: item[0], reverse=True)
        if self.offline_max_episodes > 0 and len(episodes_sorted) > self.offline_max_episodes:
            episodes_sorted = episodes_sorted[: self.offline_max_episodes]
        total = len(episodes_sorted)

        elite_count = self.offline_elite_count_cfg
        if elite_count <= 0 and self.offline_elite_fraction > 0.0:
            elite_count = int(math.ceil(total * min(self.offline_elite_fraction, 1.0)))
        elite_count = max(0, min(elite_count, total))

        elite = episodes_sorted[:elite_count]
        reserve = episodes_sorted[elite_count:]
        if self.offline_reserve_limit > 0:
            reserve = reserve[: self.offline_reserve_limit]

        self._offline_elite = elite
        self._offline_reserve = reserve

    def _scheduled_offline_ratio(self) -> float:
        if self.offline_lock_ratio <= 0.0:
            return 0.0

        updates = max(0, int(self._total_update_counter))
        if updates < self.offline_lock_updates:
            return self.offline_lock_ratio

        if not self._offline_decay_triggered:
            return self.offline_lock_ratio

        start_update = self._offline_decay_trigger_update
        if start_update is None or start_update < self.offline_lock_updates:
            start_update = self.offline_lock_updates

        if updates <= start_update:
            return self.offline_lock_ratio

        progress = (updates - start_update) / float(self.offline_decay_updates)
        progress = max(0.0, min(progress, 1.0))
        ratio = self.offline_lock_ratio + (self.offline_final_ratio - self.offline_lock_ratio) * progress
        return min(max(ratio, 0.0), 1.0)

    def _current_offline_ratio(self):
        start = self.offline_mix_ratio_start
        final = self.offline_mix_ratio_final
        effective_base = max(self.offline_mix_ratio, start, final)
        scheduled = self._scheduled_offline_ratio()
        if effective_base <= 0.0 and scheduled <= 0.0:
            return 0.0

        episode_idx = max(0, self._dt_episode_counter - 1)

        if episode_idx < self.offline_mix_ratio_warmup:
            base_ratio = start
        else:
            progress = (episode_idx - self.offline_mix_ratio_warmup) / float(self.offline_mix_ratio_decay)
            progress = max(0.0, min(progress, 1.0))
            base_ratio = start + (final - start) * progress

        base_ratio = max(base_ratio, self.offline_mix_ratio, final)
        base_ratio = min(max(base_ratio, 0.0), 1.0)
        return max(base_ratio, scheduled)

    def get_offline_summary(self):
        """Return aggregate statistics about loaded offline trajectories."""

        return self._offline_summary

    # ------------------------------------------------------------------
    # Episode bookkeeping
    # ------------------------------------------------------------------
    def begin_episode(self):
        self.context_states = []
        self.context_actions = []
        self.context_returns = []
        self.current_return = self.target_return
        self._episode_return = 0.0
        self._episode_strehl_values = []
        self._last_episode_avg_strehl = 0.0

    def record_reward(self, reward, done, strehl=None):
        reward = float(np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0))
        self._episode_return += reward
        strehl_gap_term = 0.0
        if strehl is not None and self.strehl_goal_weight != 0.0:
            strehl_gap = self.strehl_goal - strehl
            strehl_gap_term = self.step_return_scale * self.strehl_goal_weight * strehl_gap

        if self.step_return_beta > 0.0:
            target = self.target_return
            adjustment = self.step_return_scale * reward
            blended = target - adjustment + strehl_gap_term
            self.current_return = (
                (1.0 - self.step_return_beta) * self.current_return
                + self.step_return_beta * blended
            )
        else:
            self.current_return = (
                self.current_return
                - self.step_return_scale * reward
                + strehl_gap_term
            )

        if self.dt_discount != 1.0:
            self.current_return *= self.dt_discount

        if self.return_floor_ratio > 0.0 and self.target_return > 0.0:
            min_return = self.return_floor_ratio * self.target_return
            if self.current_return < min_return:
                self.current_return = min_return

        if self.return_clip > 0.0:
            self.current_return = float(
                np.clip(self.current_return, -self.return_clip, self.return_clip)
            )

        if strehl is not None:
            strehl = float(np.nan_to_num(strehl, nan=0.0, posinf=0.0, neginf=0.0))
            self._episode_strehl_values.append(strehl)
            if self._strehl_ema is None:
                self._strehl_ema = strehl
            else:
                beta_s = self.strehl_momentum
                self._strehl_ema = beta_s * self._strehl_ema + (1.0 - beta_s) * strehl

        if done:
            if self._return_ema is None:
                self._return_ema = self._episode_return
            else:
                beta = self.target_momentum
                self._return_ema = beta * self._return_ema + (1.0 - beta) * self._episode_return

            if self._episode_strehl_values:
                self._last_episode_avg_strehl = float(
                    np.mean(self._episode_strehl_values)
                )
            elif self._strehl_ema is not None:
                self._last_episode_avg_strehl = float(self._strehl_ema)
            else:
                self._last_episode_avg_strehl = 0.0
            self._episode_strehl_values = []

            candidate = self._return_ema + self.target_offset
            if self.target_gain != 0.0 and self._strehl_ema is not None:
                candidate = max(candidate, self._strehl_ema * self.target_gain + self.target_offset)
            if self.strehl_goal_weight != 0.0:
                target_from_goal = self.strehl_goal_weight * max(
                    self.strehl_goal,
                    self._strehl_ema if self._strehl_ema is not None else 0.0,
                )
                candidate = max(candidate, target_from_goal)
            if self.target_min is not None:
                candidate = max(candidate, self.target_min)

            self.target_return = candidate
            self.current_return = self.target_return
            if self.return_floor_ratio > 0.0 and self.target_return > 0.0:
                floor_value = self.return_floor_ratio * self.target_return
                if self.current_return < floor_value:
                    self.current_return = floor_value
            if self.return_clip > 0.0:
                self.current_return = float(
                    np.clip(self.current_return, -self.return_clip, self.return_clip)
                )
            self._episode_return = 0.0

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------
    def _build_context_batch(self):
        states = list(self.context_states)
        returns = list(self.context_returns)
        seq_len = len(states)
        actions = list(self.context_actions)
        actions.append(np.zeros(self.action_dim, dtype=np.float32))

        pad = max(self.context_len - seq_len, 0)
        context_states = np.zeros((self.context_len, self.state_dim), dtype=np.float32)
        context_actions = np.zeros((self.context_len, self.action_dim), dtype=np.float32)
        context_returns = np.zeros((self.context_len, 1), dtype=np.float32)
        padding_mask = np.zeros((self.context_len,), dtype=bool)
        if pad > 0:
            padding_mask[:pad] = True
        if seq_len > 0:
            context_states[pad:] = np.array(states[-self.context_len:], dtype=np.float32)
            context_returns[pad:, 0] = np.array(returns[-self.context_len:], dtype=np.float32)
            context_actions[pad:] = np.array(actions[-self.context_len:], dtype=np.float32)

        batch_states = torch.as_tensor(context_states, device=self.device).unsqueeze(0)
        batch_actions = torch.as_tensor(context_actions, device=self.device).unsqueeze(0)
        batch_returns = torch.as_tensor(context_returns, device=self.device).unsqueeze(0)
        batch_mask = torch.as_tensor(padding_mask, device=self.device).unsqueeze(0)
        return batch_states, batch_actions, batch_returns, batch_mask

    def _context_action_mean(self) -> np.ndarray:
        if not self.context_actions:
            return np.zeros(self.action_dim, dtype=np.float32)
        try:
            stacked = np.asarray(self.context_actions, dtype=np.float32)
            if stacked.ndim == 1:
                stacked = stacked.reshape(1, -1)
        except ValueError:
            stacked = np.stack(
                [np.asarray(action, dtype=np.float32).reshape(-1) for action in self.context_actions],
                axis=0,
            )
        mean = np.mean(stacked, axis=0)
        if mean.size != self.action_dim:
            return np.zeros(self.action_dim, dtype=np.float32)
        return mean.astype(np.float32, copy=False)

    def select_action(self, state, eval_mode=False):
        state = np.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        action_mean = self._context_action_mean()
        encoded_state = self.feature_engineer.transform_state(
            state,
            action_mean=action_mean,
            update_stats=True,
        )
        self.context_states.append(encoded_state)
        self.context_returns.append(self.current_return * self.return_scale)
        if len(self.context_states) > self.context_len:
            self.context_states.pop(0)
            self.context_returns.pop(0)
            if self.context_actions:
                self.context_actions.pop(0)

        batch_states, batch_actions, batch_returns, batch_mask = self._build_context_batch()
        with torch.no_grad():
            mean, log_std = self.policy.forward(batch_states, batch_actions, batch_returns, padding_mask=batch_mask)
            std = log_std.exp().clamp(min=1e-6)
            if eval_mode:
                z = mean
            else:
                normal = Normal(mean, std)
                z = normal.rsample()
            action = torch.tanh(z) * self.action_scale
        action_np = action.squeeze(0).cpu().numpy()
        mean_np = torch.tanh(mean).squeeze(0).cpu().numpy() * self.action_scale

        self.context_actions.append(action_np)
        if len(self.context_actions) > len(self.context_states):
            self.context_actions = self.context_actions[-len(self.context_states):]

        return action_np, mean_np

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def update_parameters(self, memory, batch_size, _total_update, total_step):
        self._total_update_counter += 1
        try:
            if _total_update is not None:
                update_int = int(float(_total_update))
                if update_int > self._total_update_counter:
                    self._total_update_counter = update_int
        except (TypeError, ValueError):
            pass
        transitions = [transition for transition in getattr(memory, "buffer", []) if transition is not None]
        if self.replay_window > 0 and len(transitions) > self.replay_window:
            transitions = transitions[-self.replay_window:]
        if not transitions:
            return

        states, actions, rewards, _next_states, masks = zip(*transitions)

        states_np = np.nan_to_num(np.stack(states), nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        actions_np = np.clip(
            np.nan_to_num(np.stack(actions), nan=0.0, posinf=1.0, neginf=-1.0),
            -0.999,
            0.999,
        ).astype(np.float32)
        rewards_np = np.nan_to_num(np.array(rewards, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
        masks_np = np.nan_to_num(np.array(masks, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)

        self.feature_engineer.observe_batch(states_np)
        encoded_states_np = self.feature_engineer.transform_sequence(
            states_np,
            action_sequence=actions_np,
            update_stats=False,
        )

        episode_id = self._dt_episode_counter
        self._dt_episode_counter += 1
        avg_strehl = float(getattr(self, "_last_episode_avg_strehl", 0.0))
        episode = {
            "states": encoded_states_np,
            "actions": actions_np,
            "rewards": rewards_np,
            "masks": masks_np,
            "episode_id": episode_id,
            "avg_strehl": avg_strehl,
            "is_offline": False,
        }
        quality_reward = float(np.sum(rewards_np))
        quality_metric = quality_reward + self.strehl_quality_weight * avg_strehl
        episode["quality_reward"] = quality_reward
        episode["quality_metric"] = quality_metric

        is_offline_episode = bool(episode.get("is_offline"))
        keep_episode = True
        if not is_offline_episode:
            keep_episode = self._should_keep_online_episode(quality_reward)
            episode["is_low_weight"] = not keep_episode
            if keep_episode:
                self._record_online_return(quality_reward)
        else:
            episode["is_low_weight"] = False

        if keep_episode:
            self._insert_episode_sorted(
                self._best_reward_episodes, quality_metric, self.replay_episodes, episode
            )
            self._insert_episode_sorted(
                self._best_strehl_episodes, avg_strehl, self.best_history_capacity, episode
            )
            if not is_offline_episode:
                self._recent_episodes.append(episode)
        else:
            self._low_weight_episodes.append(episode)

        episodes_iterable = []
        seen_ids = set()

        def _append_episode(entry, is_recent=False, is_low_weight=False):
            ep_id = entry.get("episode_id")
            if ep_id in seen_ids:
                return
            episodes_iterable.append((entry, bool(is_recent), bool(is_low_weight)))
            seen_ids.add(ep_id)

        for _score, stored in self._best_reward_episodes:
            _append_episode(stored, False, stored.get("is_low_weight", False))

        for _score, stored in self._best_strehl_episodes:
            _append_episode(stored, False, stored.get("is_low_weight", False))

        for stored in reversed(self._recent_episodes):
            _append_episode(stored, True, stored.get("is_low_weight", False))

        offline_available = len(self._offline_elite) + len(self._offline_reserve)
        offline_ratio = max(self._current_offline_ratio(), self.replay_offline_ratio)
        if offline_ratio > 0.0 and offline_available:
            base_count = max(len(episodes_iterable), 1)
            desired = int(math.ceil(offline_ratio * base_count))
            desired = min(desired, offline_available)
            if desired <= 0 and offline_available > 0:
                desired = 1
            if desired > 0:
                selection: list[tuple[float, dict]] = []
                elite_pool = list(self._offline_elite)
                if elite_pool:
                    elite_take = min(len(elite_pool), desired)
                    selection.extend(elite_pool[:elite_take])
                remaining = desired - len(selection)
                reserve_pool = list(self._offline_reserve)
                if remaining > 0 and reserve_pool:
                    if remaining < len(reserve_pool):
                        selection.extend(random.sample(reserve_pool, remaining))
                    else:
                        selection.extend(reserve_pool[:remaining])
                for score, stored in selection:
                    _append_episode(stored, False, stored.get("is_low_weight", False))

        for stored in reversed(self._low_weight_episodes):
            _append_episode(stored, False, True)

        top_reward_threshold = float("-inf")
        if self.replay_online_top_percentile > 0.0:
            quantile = max(0.0, 1.0 - self.replay_online_top_percentile)
            top_reward_threshold = self._compute_online_quantile(quantile, default=float("-inf"))

        sequence_items = []
        sequence_index = 0
        order_counter = 0

        for stored, is_recent, is_low_weight in episodes_iterable:
            states_ep = stored["states"]
            actions_ep = stored["actions"]
            rewards_ep = stored["rewards"]
            masks_ep = stored["masks"]

            if states_ep.size == 0:
                continue

            episode_quality = float(stored.get("quality_reward", float(np.sum(rewards_ep))))
            is_offline = bool(stored.get("is_offline"))
            if is_offline:
                category = "offline"
            elif is_recent:
                category = "recent"
            else:
                if top_reward_threshold == float("-inf") or episode_quality >= top_reward_threshold:
                    category = "online_top"
                else:
                    category = "online_other"
                if is_low_weight:
                    category = "online_other"

            returns_ep = np.zeros_like(rewards_ep)
            running_return = 0.0
            for idx in reversed(range(len(rewards_ep))):
                running_return = rewards_ep[idx] + self.dt_discount * running_return * masks_ep[idx]
                returns_ep[idx] = running_return
                if masks_ep[idx] == 0.0:
                    running_return = 0.0
            returns_raw = returns_ep.copy()
            returns_scaled = returns_ep * self.return_scale
            if self.return_clip > 0.0:
                returns_scaled = np.clip(returns_scaled, -self.return_clip, self.return_clip)

            horizon = len(states_ep)
            for idx in range(0, horizon, self.sequence_stride):
                start = max(0, idx - self.context_len + 1)
                seq_states = states_ep[start : idx + 1]
                seq_returns = returns_scaled[start : idx + 1]
                seq_actions = actions_ep[start:idx]

                seq_len = idx - start + 1
                pad = self.context_len - seq_len

                states_pad = np.zeros((self.context_len, self.state_dim), dtype=np.float32)
                returns_pad = np.zeros((self.context_len, 1), dtype=np.float32)
                actions_pad = np.zeros((self.context_len, self.action_dim), dtype=np.float32)
                mask_pad = np.zeros((self.context_len,), dtype=bool)
                if pad > 0:
                    mask_pad[:pad] = True

                states_pad[pad:] = seq_states
                returns_pad[pad:, 0] = seq_returns
                if seq_actions.size:
                    actions_pad[pad : pad + seq_actions.shape[0]] = seq_actions
                actions_pad[-1] = 0.0

                item = {
                    "index": sequence_index,
                    "states": states_pad,
                    "actions": actions_pad,
                    "returns": returns_pad,
                    "mask": mask_pad,
                    "target": actions_ep[idx],
                    "quality": float(returns_raw[idx]),
                    "is_recent": bool(is_recent),
                    "is_offline": is_offline,
                    "category": category,
                    "order": order_counter,
                }
                sequence_items.append(item)
                sequence_index += 1
                order_counter += 1

        if not sequence_items:
            return

        total_sequences = len(sequence_items)
        target_total = self.sequences_topk if self.sequences_topk > 0 else total_sequences
        selected_items = self._select_balanced_sequences(sequence_items, target_total)
        if not selected_items:
            return

        sequences_states = [item["states"] for item in selected_items]
        sequences_actions = [item["actions"] for item in selected_items]
        sequences_returns = [item["returns"] for item in selected_items]
        sequences_masks = [item["mask"] for item in selected_items]
        targets = [item["target"] for item in selected_items]
        sequence_quality = [item["quality"] for item in selected_items]
        sequence_recent = [1.0 if item["is_recent"] else 0.0 for item in selected_items]
        sequence_offline = [1.0 if item["is_offline"] else 0.0 for item in selected_items]

        state_array = np.stack(sequences_states)
        action_array = np.stack(sequences_actions)
        return_array = np.stack(sequences_returns)
        mask_array = np.stack(sequences_masks)
        target_array = np.stack(targets)
        quality_array = np.asarray(sequence_quality, dtype=np.float32)
        recent_array = np.asarray(sequence_recent, dtype=np.float32)
        offline_array = np.asarray(sequence_offline, dtype=np.float32)

        if self.normalize_returns:
            valid = ~mask_array
            valid_returns = return_array[valid]
            if valid_returns.size > 0:
                mean = float(valid_returns.mean())
                std = float(valid_returns.std())
                if np.isfinite(std) and std > self.return_norm_epsilon:
                    return_array = (return_array - mean) / (std + self.return_norm_epsilon)
                else:
                    return_array = return_array - mean

        state_tensor = torch.as_tensor(state_array, device=self.device)
        action_tensor = torch.as_tensor(action_array, device=self.device)
        return_tensor = torch.as_tensor(return_array, device=self.device)
        mask_tensor = torch.as_tensor(mask_array, device=self.device)
        target_tensor = torch.as_tensor(target_array, device=self.device)

        state_tensor = self._sanitize_tensor(state_tensor)
        action_tensor = self._sanitize_tensor(action_tensor, -0.999, 0.999)
        return_tensor = self._sanitize_tensor(return_tensor)
        target_tensor = self._sanitize_tensor(target_tensor, -0.999, 0.999)
        quality_tensor = torch.as_tensor(quality_array, device=self.device)
        recent_tensor = torch.as_tensor(recent_array, device=self.device)
        offline_tensor = torch.as_tensor(offline_array, device=self.device)

        dataset_size = state_tensor.size(0)
        if dataset_size == 0:
            return

        batch_size = max(1, min(batch_size, dataset_size))

        policy_loss_acc = 0.0
        entropy_acc = 0.0
        updates = 0

        if dataset_size:
            if self.normalize_returns:
                q_mean = quality_tensor.mean()
                q_std = quality_tensor.std()
                if torch.isfinite(q_std) and q_std > self.return_norm_epsilon:
                    quality_norm = (quality_tensor - q_mean) / (q_std + self.return_norm_epsilon)
                else:
                    quality_norm = quality_tensor - q_mean
            else:
                denom = torch.mean(torch.abs(quality_tensor))
                denom = torch.nan_to_num(denom, nan=1.0, posinf=1.0, neginf=1.0)
                if torch.isfinite(denom) and denom > self.return_norm_epsilon:
                    quality_norm = quality_tensor / (denom + self.return_norm_epsilon)
                else:
                    quality_norm = quality_tensor

            if self.loss_temperature > 0.0:
                temp = max(self.loss_temperature, 1e-6)
                scaled = torch.clamp(quality_norm / temp, -10.0, 10.0)
                quality_weights = torch.softmax(scaled, dim=0) * float(dataset_size)
            else:
                quality_weights = torch.ones_like(quality_tensor)

            if self.recent_weight > 0.0 and torch.any(recent_tensor > 0):
                recent_total = torch.sum(recent_tensor)
                recent_total = torch.nan_to_num(recent_total, nan=0.0, posinf=0.0, neginf=0.0)
                if recent_total > 0:
                    recent_weights = recent_tensor / recent_total * float(dataset_size)
                    weight_tensor = (
                        (1.0 - self.recent_weight) * quality_weights
                        + self.recent_weight * recent_weights
                    )
                else:
                    weight_tensor = quality_weights
            else:
                weight_tensor = quality_weights

            weight_tensor = torch.nan_to_num(weight_tensor, nan=1.0, posinf=10.0, neginf=0.0)
            weight_tensor = weight_tensor.clamp(min=1e-3)
        else:
            weight_tensor = torch.ones_like(quality_tensor)

        if self.offline_weight_gain > 0.0 and torch.any(offline_tensor > 0):
            offline_boost = 1.0 + self.offline_weight_gain * offline_tensor
            weight_tensor = weight_tensor * offline_boost
            weight_tensor = torch.nan_to_num(weight_tensor, nan=1.0, posinf=10.0, neginf=0.0)
            weight_tensor = weight_tensor.clamp(min=1e-3)

        for _ in range(self.updates_per_episode):
            permutation = torch.randperm(dataset_size, device=self.device)
            for start in range(0, dataset_size, batch_size):
                idx = permutation[start:start + batch_size]
                state_batch = state_tensor.index_select(0, idx)
                action_batch = action_tensor.index_select(0, idx)
                return_batch = return_tensor.index_select(0, idx)
                mask_batch = mask_tensor.index_select(0, idx)
                target_batch = target_tensor.index_select(0, idx)
                weight_batch = weight_tensor.index_select(0, idx).unsqueeze(1)
                weight_batch = weight_batch / (weight_batch.mean() + 1e-6)

                mean, log_std = self.policy.forward(
                    state_batch,
                    action_batch,
                    return_batch,
                    padding_mask=mask_batch,
                )
                pred_action = torch.tanh(mean) * self.action_scale
                mse_elements = (pred_action - target_batch).pow(2)
                mse_loss = (mse_elements * weight_batch).mean()
                if self.action_scale != 0.0:
                    scale_den = float(abs(self.action_scale))
                else:
                    scale_den = 1.0
                target_scaled = torch.clamp(
                    target_batch / scale_den,
                    min=-0.999,
                    max=0.999,
                )
                target_pre_tanh = 0.5 * (
                    torch.log1p(target_scaled) - torch.log1p(-target_scaled)
                )
                std = log_std.exp().clamp(min=1e-6, max=1e6)
                normal = Normal(mean, std)
                log_prob = normal.log_prob(target_pre_tanh) - torch.log(
                    1 - target_scaled.pow(2) + epsilon
                )
                log_prob = log_prob.sum(-1, keepdim=True)
                nll = -(log_prob)
                nll_loss = (nll * weight_batch).mean()

                reg = self.bc_reg_coef * log_std.pow(2).mean()
                total_loss = (
                    self.bc_logprob_coef * nll_loss
                    + self.bc_mse_coef * mse_loss
                    + reg
                )

                if not torch.isfinite(total_loss):
                    continue

                self.policy_optim.zero_grad()
                total_loss.backward()
                if self.gradient_clip_norm > 0.0:
                    clip_grad_norm_(self.policy.parameters(), self.gradient_clip_norm)
                self.policy_optim.step()

                policy_loss_acc += float(total_loss.item())
                entropy_components = 0.5 * (math.log(2 * math.pi * math.e)) + log_std
                entropy_batch = entropy_components.sum(dim=-1, keepdim=True)
                entropy_weighted = (entropy_batch * weight_batch).mean()
                entropy_acc += float(entropy_weighted.item())
                updates += 1

        if updates:
            avg_policy = policy_loss_acc / updates
            avg_entropy = entropy_acc / updates
            self.policy_loss_list = (total_step, avg_policy)
            self.value_loss_list = (total_step, 0.0)
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

        episode_id = int(episode)
        actor_filename = f"worker_{worker_id}_dt_policy_episode_{episode_id:06d}.pth"
        critic_filename = f"worker_{worker_id}_dt_value_episode_{episode_id:06d}.pth"
        actor_path = os.path.join(folder, actor_filename)
        critic_path = os.path.join(folder, critic_filename)

        torch.save({'model_state_dict': self.policy.state_dict()}, actor_path)
        torch.save({'model_state_dict': self.value.state_dict()}, critic_path)

        latest_actor = os.path.join(folder, f"worker_{worker_id}_dt_policy_latest.pth")
        latest_critic = os.path.join(folder, f"worker_{worker_id}_dt_value_latest.pth")
        try:
            copy2(actor_path, latest_actor)
            copy2(critic_path, latest_critic)
        except OSError:
            torch.save({'model_state_dict': self.policy.state_dict()}, latest_actor)
            torch.save({'model_state_dict': self.value.state_dict()}, latest_critic)

    def load_policy(self, master_rref, worker_id):
        assert worker_id == self.worker_id
        folder = self.model_dir or os.path.join(self.config.savedir, "output_models", "models_rpc")
        if not os.path.isdir(folder):
            return

        def _resolve_latest(prefix: str) -> Optional[str]:
            latest_path = os.path.join(folder, f"{prefix}_latest.pth")
            if os.path.exists(latest_path):
                return latest_path
            pattern = os.path.join(folder, f"{prefix}_episode_*.pth")
            candidates = sorted(glob(pattern))
            if candidates:
                return candidates[-1]
            return None

        actor_path = _resolve_latest(f"worker_{worker_id}_dt_policy")
        critic_path = _resolve_latest(f"worker_{worker_id}_dt_value")

        if actor_path and os.path.exists(actor_path):
            model_dict = torch.load(actor_path, map_location=self.device)
            self.policy.load_state_dict(model_dict["model_state_dict"])
        if critic_path and os.path.exists(critic_path):
            model_dict = torch.load(critic_path, map_location=self.device)
            self.value.load_state_dict(model_dict["model_state_dict"])

