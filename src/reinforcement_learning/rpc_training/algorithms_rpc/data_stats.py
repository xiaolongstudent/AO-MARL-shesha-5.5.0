from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np


__all__ = [
    "EpisodeStats",
    "compute_episode_stats",
    "summarize_dataset",
]


@dataclass
class EpisodeStats:
    ep_id: str
    sum_return: float
    mean_strehl: float
    T: int


def _safe_mean_strehl(ep: Dict[str, Any], strehl_key: Optional[str]) -> float:
    if strehl_key is None:
        return float("nan")
    if strehl_key not in ep:
        return float("nan")
    series = ep[strehl_key]
    if series is None:
        return float("nan")
    try:
        arr = np.asarray(series, dtype=np.float64).reshape(-1)
    except Exception:
        return float("nan")
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def compute_episode_stats(
    ep: Dict[str, Any],
    *,
    reward_key: str = "reward",
    strehl_key: Optional[str] = None,
) -> EpisodeStats:
    rewards = np.asarray(ep[reward_key], dtype=np.float64).reshape(-1)
    sum_return = float(np.sum(rewards))
    mean_strehl = _safe_mean_strehl(ep, strehl_key)
    return EpisodeStats(
        ep_id=str(ep.get("ep_id", "")),
        sum_return=sum_return,
        mean_strehl=mean_strehl,
        T=int(rewards.size),
    )


def summarize_dataset(
    stats_list: Iterable[EpisodeStats],
    *,
    quantiles: Sequence[float] = (0.5, 0.8, 0.9),
    histogram_bins: int = 20,
) -> Dict[str, Any]:
    """Return aggregate statistics for a collection of episodes.

    Parameters
    ----------
    stats_list:
        Iterable of :class:`EpisodeStats` objects.
    quantiles:
        Iterable of quantiles (between 0 and 1) to compute for the summed
        returns.  Defaults to median/80th/90th percentiles as those were the
        values highlighted in the analysis request.
    histogram_bins:
        Number of bins to use when computing a coarse histogram of the return
        distribution.  The resulting counts are useful for quickly assessing
        how heavy the distribution tails are without storing full data dumps.
    """

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
    ret_mean = float(np.mean(returns))
    ret_std = float(np.std(returns))

    quantiles = tuple(float(q) for q in quantiles)
    qs = np.quantile(returns, q=np.array(quantiles))
    quantile_dict = {f"{q:.2f}": float(v) for q, v in zip(quantiles, qs)}

    top20_threshold = float(np.quantile(returns, 0.8))
    top20_ratio = float(np.mean(returns >= top20_threshold))

    hist_counts, _ = np.histogram(returns, bins=histogram_bins)

    strehl_vals = np.array(
        [s.mean_strehl for s in stats if not np.isnan(s.mean_strehl)], dtype=np.float64
    )
    strehl_mean = float(np.mean(strehl_vals)) if strehl_vals.size else None
    strehl_std = float(np.std(strehl_vals)) if strehl_vals.size else None

    return {
        "num_eps": len(stats),
        "ret_mean": ret_mean,
        "ret_std": ret_std,
        "strehl_mean": strehl_mean,
        "strehl_std": strehl_std,
        "q": quantile_dict,
        "top20_thr": top20_threshold,
        "top20_ratio": top20_ratio,
        "hist": hist_counts.astype(int).tolist(),
    }
