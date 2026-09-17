"""Replicate aggregation, bootstrap confidence intervals and action labels."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .hashing import stable_hash
from .table_io import read_table, write_table


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    confidence_level: float = 0.95,
    samples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1 or int(samples) <= 0:
        value = float(array[0])
        return value, value
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.size, size=(int(samples), array.size))
    means = array[indices].mean(axis=1)
    alpha = 1.0 - float(confidence_level)
    return float(np.quantile(means, alpha / 2.0)), float(np.quantile(means, 1.0 - alpha / 2.0))


def classify_label(
    *,
    mean: float,
    lower: float,
    upper: float,
    positive_boundary: float,
    negative_boundary: float,
) -> str:
    if lower > positive_boundary:
        return "confirmed_positive"
    if upper < negative_boundary:
        return "confirmed_negative"
    if mean > positive_boundary:
        return "hard_negative"
    return "near_zero_or_uncertain"


def aggregate_replicates(
    replicates: pd.DataFrame,
    *,
    target_column: str = "marginal_reward_scaled",
    confidence_level: float = 0.95,
    bootstrap_samples: int = 2000,
    positive_noise_boundary: float = 0.0,
    negative_noise_boundary: float = 0.0,
) -> pd.DataFrame:
    required = {"state_id", "action_index", "task_id", "success", target_column}
    missing = sorted(required - set(replicates.columns))
    if missing:
        raise ValueError(f"Replicate table is missing required columns: {missing}")
    rows: list[dict[str, Any]] = []
    group_columns = ["state_id", "action_index"]
    for (state_id, action_index), group in replicates.groupby(group_columns, sort=True):
        success_mask = group["success"].astype(str).str.lower().isin({"true", "1"})
        successful = group[success_mask].copy()
        first = group.iloc[0]
        base = {
            column: first[column]
            for column in (
                "schema_version",
                "state_id",
                "root_state_id",
                "parent_state_id",
                "protein_id",
                "cluster_id",
                "split",
                "depth",
                "pose_path",
                "root_pdb_path",
                "sequence",
                "candidate_sequence",
                "visited_positions",
                "action_index",
                "position",
                "wild_type_amino_acid",
                "mutant_amino_acid",
                "proposal_sources",
                "reward_contract_hash",
            )
            if column in group.columns
        }
        base["repeat_count_requested"] = int(len(group))
        base["repeat_count_successful"] = int(len(successful))
        base["failure_count"] = int(len(group) - len(successful))
        if successful.empty:
            base.update(
                {
                    "target_column": target_column,
                    "reward_mean": float("nan"),
                    "reward_std": float("nan"),
                    "reward_lcb": float("nan"),
                    "reward_ucb": float("nan"),
                    "sign_agreement": 0.0,
                    "confidence_weight": 0.0,
                    "label_class": "structural_failure",
                    "failure_reason": " | ".join(sorted(set(map(str, group["failure_reason"])))),
                }
            )
            rows.append(base)
            continue

        values = successful[target_column].astype(float).to_numpy()
        bootstrap_seed = int(stable_hash([state_id, int(action_index)], length=8), 16)
        lower, upper = bootstrap_mean_interval(
            values,
            confidence_level=confidence_level,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        mean = float(np.mean(values))
        nonzero = values[np.abs(values) > 1e-12]
        sign_agreement = (
            1.0
            if nonzero.size == 0
            else float(max(np.mean(nonzero > 0), np.mean(nonzero < 0)))
        )
        repeat_factor = len(successful) / (len(successful) + 2.0)
        base.update(
            {
                "target_column": target_column,
                "reward_mean": mean,
                "reward_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "reward_lcb": lower,
                "reward_ucb": upper,
                "sign_agreement": sign_agreement,
                "confidence_weight": float(repeat_factor * sign_agreement),
                "label_class": classify_label(
                    mean=mean,
                    lower=lower,
                    upper=upper,
                    positive_boundary=positive_noise_boundary,
                    negative_boundary=negative_noise_boundary,
                ),
                "failure_reason": "",
            }
        )
        for source, destination in (
            ("delta_strength_marginal", "delta_strength_mean"),
            ("delta_toughness_marginal", "delta_toughness_mean"),
            ("delta_strength_absolute", "delta_strength_absolute_mean"),
            ("delta_toughness_absolute", "delta_toughness_absolute_mean"),
            ("absolute_reward_scaled", "absolute_reward_scaled_mean"),
            ("strength_after", "strength_after_mean"),
            ("toughness_after", "toughness_after_mean"),
        ):
            if source in successful.columns:
                base[destination] = float(successful[source].astype(float).mean())
        if "candidate_structure_path" in successful.columns:
            paths = [str(value) for value in successful["candidate_structure_path"] if str(value)]
            base["candidate_structure_path"] = paths[0] if paths else ""
        rows.append(base)
    return pd.DataFrame(rows).sort_values(["protein_id", "state_id", "action_index"]).reset_index(drop=True)


def aggregate_file(
    replicates_path: str,
    output_path: str,
    **kwargs: Any,
) -> pd.DataFrame:
    labels = aggregate_replicates(read_table(replicates_path), **kwargs)
    write_table(labels, output_path)
    return labels
