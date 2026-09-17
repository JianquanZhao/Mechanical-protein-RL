"""State-macro action ranking and regression metrics."""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import spearmanr


def _ndcg(scores: np.ndarray, targets: np.ndarray) -> float:
    gains = np.maximum(targets, 0.0)
    order = np.argsort(-scores)
    ideal = np.argsort(-gains)
    discounts = 1.0 / np.log2(np.arange(len(gains)) + 2.0)
    dcg = float(np.sum(gains[order] * discounts))
    idcg = float(np.sum(gains[ideal] * discounts))
    return 0.0 if idcg <= 0.0 else dcg / idcg


def compute_sft_metrics(
    q_values: np.ndarray,
    targets: np.ndarray,
    observed_mask: np.ndarray,
    positive_mask: np.ndarray,
    strength_deltas: np.ndarray | None = None,
    toughness_deltas: np.ndarray | None = None,
) -> Dict[str, float]:
    spearman_values: list[float] = []
    ndcg_values: list[float] = []
    regret_values: list[float] = []
    selected_rewards: list[float] = []
    hit_values = {1: [], 5: [], 10: []}
    pareto: list[float] = []
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for index in range(q_values.shape[0]):
        mask = observed_mask[index].astype(bool)
        if not np.any(mask):
            continue
        q = q_values[index, mask]
        y = targets[index, mask]
        positive = positive_mask[index, mask].astype(bool)
        all_predictions.append(q)
        all_targets.append(y)
        if len(y) >= 2 and np.std(y) > 0 and np.std(q) > 0:
            spearman_values.append(float(spearmanr(q, y).statistic))
        ndcg_values.append(_ndcg(q, y))
        greedy = int(np.argmax(q))
        selected_rewards.append(float(y[greedy]))
        regret_values.append(float(np.max(y) - y[greedy]))
        order = np.argsort(-q)
        for k in hit_values:
            hit_values[k].append(float(np.any(positive[order[: min(k, len(order))]])))
        if strength_deltas is not None and toughness_deltas is not None:
            strength = strength_deltas[index, mask][greedy]
            toughness = toughness_deltas[index, mask][greedy]
            if np.isfinite(strength) and np.isfinite(toughness):
                pareto.append(float(strength > 0.0 and toughness > 0.0))
    if not all_targets:
        return {"state_count": 0.0}
    predicted = np.concatenate(all_predictions)
    actual = np.concatenate(all_targets)
    metrics = {
        "state_count": float(len(all_targets)),
        "action_count": float(len(actual)),
        "mae": float(np.mean(np.abs(predicted - actual))),
        "rmse": float(np.sqrt(np.mean(np.square(predicted - actual)))),
        "spearman_macro": float(np.mean(spearman_values)) if spearman_values else float("nan"),
        "ndcg_macro": float(np.mean(ndcg_values)),
        "greedy_regret_mean": float(np.mean(regret_values)),
        "greedy_reward_mean": float(np.mean(selected_rewards)),
        "greedy_reward_median": float(np.median(selected_rewards)),
        "pareto_positive_fraction": float(np.mean(pareto)) if pareto else float("nan"),
    }
    for k, values in hit_values.items():
        metrics[f"positive_hit_at_{k}"] = float(np.mean(values))
    return metrics

