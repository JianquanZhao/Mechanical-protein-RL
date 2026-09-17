"""Deterministic summaries for paired greedy validation episodes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    samples: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if values.size == 1 or samples == 0:
        value = float(values.mean())
        return value, value
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(means, tail)),
        float(np.quantile(means, 1.0 - tail)),
    )


def summarize_greedy_validation(
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 1_000,
    confidence: float = 0.95,
    seed: int = 17_291,
    top_fractions: Sequence[float] = (0.05, 0.10),
) -> dict[str, float | int]:
    """Aggregate paired final-minus-initial validation improvements."""

    if not records:
        raise ValueError("Greedy validation requires at least one record.")
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be >= 0.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be within (0, 1).")
    fractions = tuple(float(value) for value in top_fractions)
    if not fractions or any(not 0.0 < value <= 1.0 for value in fractions):
        raise ValueError("top_fractions must contain values within (0, 1].")

    rng = np.random.default_rng(seed)
    summary: dict[str, float | int] = {"validation_count": len(records)}
    fields = {
        "terminal_reward": "terminal_reward",
        "strength": "strength_delta",
        "toughness": "toughness_delta",
    }
    for output_name, source_name in fields.items():
        values = np.asarray([record[source_name] for record in records], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Validation field {source_name!r} contains non-finite values.")
        ci_low, ci_high = _bootstrap_mean_interval(
            values,
            samples=int(bootstrap_samples),
            confidence=float(confidence),
            rng=rng,
        )
        summary[f"{output_name}_mean"] = float(values.mean())
        summary[f"{output_name}_median"] = float(np.median(values))
        summary[f"{output_name}_positive_fraction"] = float(np.mean(values > 0.0))
        summary[f"{output_name}_mean_ci_low"] = ci_low
        summary[f"{output_name}_mean_ci_high"] = ci_high
        descending = np.sort(values)[::-1]
        for fraction in fractions:
            count = max(1, int(np.ceil(values.size * fraction)))
            label = f"top_{int(round(fraction * 100)):02d}pct_mean"
            summary[f"{output_name}_{label}"] = float(descending[:count].mean())
    return summary
