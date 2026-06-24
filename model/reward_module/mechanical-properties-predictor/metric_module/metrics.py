from __future__ import annotations

import math
from typing import Mapping

import numpy as np


TARGET_NAMES = ("strength", "toughness")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    top_fractions: tuple[float, ...] = (0.05, 0.10),
) -> dict[str, float]:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.shape != pred.shape:
        raise ValueError(f"y_true and y_pred shapes differ: {true.shape} vs {pred.shape}.")
    if true.ndim != 2 or true.shape[1] != 2:
        raise ValueError("Expected arrays with shape (n_samples, 2): strength, toughness.")

    metrics: dict[str, float] = {}
    for column, name in enumerate(TARGET_NAMES):
        target_true = true[:, column]
        target_pred = pred[:, column]
        metrics[f"{name}/r2"] = _r2_score(target_true, target_pred)
        metrics[f"{name}/mae"] = float(np.mean(np.abs(target_true - target_pred)))
        metrics[f"{name}/rmse"] = float(np.sqrt(np.mean((target_true - target_pred) ** 2)))
        metrics[f"{name}/spearman"] = _spearman(target_true, target_pred)
        for fraction in top_fractions:
            suffix = f"top_{int(round(fraction * 100))}pct"
            metrics[f"{name}/{suffix}_hit_rate"] = _top_hit_rate(target_true, target_pred, fraction)
            metrics[f"{name}/{suffix}_enrichment_factor"] = _enrichment_factor(
                target_true,
                target_pred,
                fraction,
            )

    # A simple scalar summary for model selection.
    metrics["mean/r2"] = float(np.mean([metrics[f"{name}/r2"] for name in TARGET_NAMES]))
    metrics["mean/mae"] = float(np.mean([metrics[f"{name}/mae"] for name in TARGET_NAMES]))
    metrics["mean/rmse"] = float(np.mean([metrics[f"{name}/rmse"] for name in TARGET_NAMES]))
    metrics["mean/spearman"] = float(np.mean([metrics[f"{name}/spearman"] for name in TARGET_NAMES]))
    return metrics


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = float(np.sum((y_true - y_pred) ** 2))
    total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if total <= 0:
        return 0.0
    return 1.0 - residual / total


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    true_rank = _rankdata(y_true)
    pred_rank = _rankdata(y_pred)
    true_centered = true_rank - true_rank.mean()
    pred_centered = pred_rank - pred_rank.mean()
    denominator = float(np.linalg.norm(true_centered) * np.linalg.norm(pred_centered))
    if denominator <= 0:
        return 0.0
    return float(np.dot(true_centered, pred_centered) / denominator)


def _top_indices(values: np.ndarray, fraction: float) -> set[int]:
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("top fraction must satisfy 0 < fraction <= 1.")
    count = max(1, int(math.ceil(len(values) * float(fraction))))
    return set(map(int, np.argsort(values)[-count:]))


def _top_hit_rate(y_true: np.ndarray, y_pred: np.ndarray, fraction: float) -> float:
    true_top = _top_indices(y_true, fraction)
    pred_top = _top_indices(y_pred, fraction)
    return float(len(true_top & pred_top) / max(1, len(pred_top)))


def _enrichment_factor(y_true: np.ndarray, y_pred: np.ndarray, fraction: float) -> float:
    hit_rate = _top_hit_rate(y_true, y_pred, fraction)
    baseline = float(fraction)
    if baseline <= 0:
        return 0.0
    return float(hit_rate / baseline)


def format_metrics(metrics: Mapping[str, float]) -> str:
    return ", ".join(f"{key}={value:.6g}" for key, value in sorted(metrics.items()))
