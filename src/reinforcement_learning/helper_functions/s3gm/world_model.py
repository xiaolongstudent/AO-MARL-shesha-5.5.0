"""Utilities for integrating S3GM-style world models inside AO-MARL."""

from __future__ import annotations

import importlib
import logging
import os
import sys
from collections import OrderedDict
from typing import Any, Dict, Optional

import numpy as np
import torch


LOGGER = logging.getLogger(__name__)


def _maybe_numpy(value: Any) -> np.ndarray:
    """Convert tensors or sequences to a numpy array."""
    if value is None:
        return None  # type: ignore[return-value]
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    if np.isscalar(value):
        return np.asarray([value])
    return np.asarray(value)


class S3GMWorldModel:
    """Runtime wrapper for Structured-3D Generative Models (S3GM).

    The wrapper purposefully keeps the interface generic so that different
    S3GM implementations (TorchScript checkpoints, python modules, etc.) can
    be plugged without touching the RL training code. When an external model
    cannot be loaded, a lightweight exponential smoother is used as a
    fallback so that the training loop still benefits from temporally
    consistent predictions.
    """

    def __init__(self, config: Dict[str, Any], observation_size: int):
        self.config = config
        self.observation_size = int(observation_size)
        self.enabled = bool(config.get("enabled", False))
        self.prediction_horizon = max(0, int(config.get("prediction_horizon", 1)))
        self.latent_size = max(0, int(config.get("latent_size", 0)))
        self.inference_interval = max(1, int(config.get("inference_interval", 1)))
        self.use_prediction = bool(config.get("use_prediction", True))
        self.use_reconstruction = bool(config.get("use_reconstruction", True))
        self.reconstruction_blend = float(config.get("reconstruction_blend", 0.5))
        self.fallback_alpha = float(config.get("fallback_alpha", 0.25))
        self.device = torch.device(config.get("device", "cpu"))
        self.repo_path = config.get("repo_path")
        self.entrypoint = config.get("entrypoint")
        self.checkpoint = config.get("checkpoint")

        self.model: Optional[Any] = None
        self.available = False
        self.current_step = 0

        self.prediction_cache = np.zeros(
            (max(1, self.prediction_horizon), self.observation_size), dtype=np.float64
        )
        self.latent_cache = np.zeros(max(1, self.latent_size), dtype=np.float64)
        self.reconstruction_cache = np.zeros(self.observation_size, dtype=np.float64)
        self.fallback_state = np.zeros(self.observation_size, dtype=np.float64)

        if not self.enabled:
            return

        if self.repo_path and os.path.isdir(self.repo_path):
            if self.repo_path not in sys.path:
                sys.path.append(self.repo_path)
        elif self.repo_path:
            LOGGER.warning("Provided S3GM repo path '%s' does not exist", self.repo_path)

        self._load_model()
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset cached predictions."""
        self.current_step = 0
        self.prediction_cache.fill(0.0)
        self.latent_cache.fill(0.0)
        self.reconstruction_cache.fill(0.0)
        self.fallback_state.fill(0.0)

    def update(self, observation: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Update the world model with a new observation.

        Parameters
        ----------
        observation:
            Latest slopes/modes vector already projected in the space used by
            the RL agent.
        metadata:
            Optional dictionary with auxiliary signals (DM commands, residuals,
            etc.) that can be consumed by an external S3GM implementation.
        Returns
        -------
        np.ndarray
            Reconstructed/denoised observation. If reconstruction is disabled
            the original measurement is returned.
        """

        if observation is None:
            return observation

        obs = np.asarray(observation, dtype=np.float64).reshape(-1)
        self.current_step += 1
        metadata = metadata or {}

        if not self.enabled or not self.available:
            self._fallback_prediction(obs)
            return obs

        if self.current_step % self.inference_interval != 0:
            # Keep previous reconstruction but refresh the fallback filter so the
            # predictions remain close to the most recent measurement.
            self._fallback_prediction(obs)
            if self.use_reconstruction:
                self.reconstruction_cache = obs.copy()
            return self.reconstruction_cache.copy() if self.use_reconstruction else obs

        try:
            raw_outputs = self._execute_model(obs, metadata)
            self._parse_outputs(raw_outputs, obs)
        except Exception as exc:  # pragma: no cover - defensive path
            LOGGER.warning("S3GM inference failed, reverting to fallback. Error: %s", exc)
            self.available = False
            self._fallback_prediction(obs)
            return obs

        if self.use_reconstruction:
            return self.reconstruction_cache.copy()
        return obs

    def get_state_features(self) -> OrderedDict:
        """Return the extra state features produced by S3GM."""
        if not self.enabled:
            return OrderedDict()

        features = OrderedDict()
        if self.use_prediction and self.prediction_horizon > 0:
            features['s3gm_future_wfs'] = self.prediction_cache.reshape(-1).copy()
        if self.latent_size > 0:
            features['s3gm_latent'] = self.latent_cache[: self.latent_size].copy()
        return features

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        if self.checkpoint and os.path.isfile(self.checkpoint):
            if self._load_torch_checkpoint(self.checkpoint):
                return
        if self.entrypoint:
            self._load_from_entrypoint(self.entrypoint)

    def _load_torch_checkpoint(self, checkpoint_path: str) -> bool:
        try:
            self.model = torch.jit.load(checkpoint_path, map_location=self.device)
            self.model.eval()
            self.available = True
            LOGGER.info("Loaded TorchScript S3GM checkpoint from %s", checkpoint_path)
            return True
        except Exception:
            pass
        try:
            state = torch.load(checkpoint_path, map_location=self.device)
            if hasattr(state, 'state_dict') and hasattr(state, 'load_state_dict'):
                state = state  # pragma: no cover - extremely unlikely branch
            self.model = state
            if hasattr(self.model, 'to'):
                self.model.to(self.device)
            if hasattr(self.model, 'eval'):
                self.model.eval()
            self.available = True
            LOGGER.info("Loaded pickled S3GM checkpoint from %s", checkpoint_path)
            return True
        except Exception as exc:
            LOGGER.warning("Failed loading S3GM checkpoint %s: %s", checkpoint_path, exc)
        return False

    def _load_from_entrypoint(self, entrypoint: str) -> None:
        module_name, _, attr = entrypoint.partition(':')
        if not attr:
            raise ValueError("Entrypoint must follow 'module:callable' format")
        module = importlib.import_module(module_name)
        builder = getattr(module, attr)
        try:
            self.model = builder(config=self.config, observation_size=self.observation_size)
        except TypeError:
            self.model = builder()
        if hasattr(self.model, 'to'):
            self.model.to(self.device)
        if hasattr(self.model, 'eval'):
            self.model.eval()
        self.available = True
        LOGGER.info("Instantiated S3GM model from %s", entrypoint)

    def _execute_model(self, observation: np.ndarray, metadata: Dict[str, Any]):
        tensor = torch.from_numpy(observation.astype(np.float32)).to(self.device)
        tensor = tensor.unsqueeze(0)

        if hasattr(self.model, 'reconstruct_and_predict'):
            return self.model.reconstruct_and_predict(tensor, metadata)
        if hasattr(self.model, 'predict'):
            try:
                return self.model.predict(tensor, metadata)
            except TypeError:
                return self.model.predict(tensor)
        if hasattr(self.model, '__call__'):
            try:
                return self.model(tensor, metadata)
            except TypeError:
                return self.model(tensor)
        raise RuntimeError("S3GM model does not expose a callable interface")

    def _parse_outputs(self, raw_outputs: Any, observation: np.ndarray) -> None:
        reconstruction = None
        prediction = None
        latent = None

        if isinstance(raw_outputs, dict):
            reconstruction = raw_outputs.get('reconstruction') or raw_outputs.get('rec')
            prediction = raw_outputs.get('prediction') or raw_outputs.get('pred')
            latent = raw_outputs.get('latent')
        elif isinstance(raw_outputs, (list, tuple)):
            if len(raw_outputs) >= 1:
                reconstruction = raw_outputs[0]
            if len(raw_outputs) >= 2:
                prediction = raw_outputs[1]
            if len(raw_outputs) >= 3:
                latent = raw_outputs[2]
        else:
            reconstruction = raw_outputs

        if reconstruction is None:
            reconstruction = observation
        reconstruction = self._sanitize_vector(_maybe_numpy(reconstruction), observation)

        if prediction is not None:
            prediction = _maybe_numpy(prediction)
        if latent is not None:
            latent = _maybe_numpy(latent)

        self.reconstruction_cache = (
            self.reconstruction_blend * reconstruction + (1 - self.reconstruction_blend) * observation
        )

        if self.prediction_horizon > 0:
            if prediction is None:
                self._fallback_prediction(observation)
            else:
                prediction = prediction.reshape(-1)
                expected = self.prediction_horizon * self.observation_size
                if prediction.size < expected:
                    prediction = np.pad(prediction, (0, expected - prediction.size))
                elif prediction.size > expected:
                    prediction = prediction[:expected]
                self.prediction_cache = prediction.reshape(self.prediction_horizon, self.observation_size)
        if self.latent_size > 0 and latent is not None:
            latent = latent.reshape(-1)
            if latent.size < self.latent_size:
                latent = np.pad(latent, (0, self.latent_size - latent.size))
            elif latent.size > self.latent_size:
                latent = latent[: self.latent_size]
            self.latent_cache = latent

    def _fallback_prediction(self, observation: np.ndarray) -> None:
        self.fallback_state = (
            self.fallback_alpha * observation + (1 - self.fallback_alpha) * self.fallback_state
        )
        if self.prediction_horizon > 0:
            for idx in range(self.prediction_horizon):
                self.prediction_cache[idx] = self.fallback_state

    def _sanitize_vector(self, candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
        if candidate is None:
            return reference
        candidate = candidate.reshape(-1)
        if candidate.size < reference.size:
            candidate = np.pad(candidate, (0, reference.size - candidate.size))
        elif candidate.size > reference.size:
            candidate = candidate[: reference.size]
        return candidate
