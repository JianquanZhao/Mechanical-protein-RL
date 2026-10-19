"""Validation and stable release export consumed by train_sft.py."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from .hashing import stable_hash
from .manifest import assert_no_cluster_leakage
from .table_io import read_table, write_json, write_table


COMPACT_FORMAT_VERSION = "compact_state_v1"
LABEL_UNOBSERVED = np.int8(0)
LABEL_POSITIVE = np.int8(1)
LABEL_NEGATIVE = np.int8(2)
LABEL_UNCERTAIN = np.int8(3)


def validate_release_inputs(states: pd.DataFrame, actions: pd.DataFrame) -> dict[str, Any]:
    required_states = {"state_id", "protein_id", "cluster_id", "split", "sequence"}
    required_actions = {
        "state_id",
        "action_index",
        "reward_mean",
        "reward_lcb",
        "label_class",
    }
    missing_states = sorted(required_states - set(states.columns))
    missing_actions = sorted(required_actions - set(actions.columns))
    if missing_states or missing_actions:
        raise ValueError(
            f"Release schema missing states={missing_states}, actions={missing_actions}."
        )
    assert_no_cluster_leakage(states)
    unknown = sorted(set(actions["state_id"]) - set(states["state_id"]))
    if unknown:
        raise ValueError(f"Action labels reference {len(unknown)} unknown states.")
    valid_actions = actions[actions["label_class"] != "structural_failure"]
    return {
        "state_count": int(len(states)),
        "protein_count": int(states["protein_id"].nunique()),
        "cluster_count": int(states["cluster_id"].nunique()),
        "action_count": int(len(actions)),
        "usable_action_count": int(len(valid_actions)),
        "split_state_counts": {
            key: int(value) for key, value in states["split"].value_counts().items()
        },
        "label_counts": {
            key: int(value) for key, value in actions["label_class"].value_counts().items()
        },
    }


def create_release(
    *,
    states_path: str | Path,
    actions_path: str | Path,
    output_dir: str | Path,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    states = read_table(states_path)
    actions = read_table(actions_path)
    summary = validate_release_inputs(states, actions)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    write_table(states, destination / "states.csv")
    write_table(actions, destination / "actions.csv")
    payload = {**dict(metadata), **summary}
    write_json(payload, destination / "dataset_version.json")
    return {**summary, "release_dir": str(destination)}


def _split_counts(count: int, fractions: Mapping[str, float]) -> dict[str, int]:
    names = ("train", "validation", "test")
    exact = np.asarray([count * float(fractions[name]) for name in names])
    assigned = np.floor(exact).astype(int)
    remainder = count - int(assigned.sum())
    for index in np.argsort(-(exact - assigned))[:remainder]:
        assigned[int(index)] += 1
    nonzero = [index for index, name in enumerate(names) if float(fractions[name]) > 0.0]
    if count >= len(nonzero):
        for index in nonzero:
            if assigned[index] == 0:
                donor = max(nonzero, key=lambda candidate: assigned[candidate])
                if assigned[donor] <= 1:
                    break
                assigned[donor] -= 1
                assigned[index] += 1
    return {name: int(value) for name, value in zip(names, assigned)}


def nested_split_states(
    states: pd.DataFrame,
    *,
    source_split: str = "train",
    fractions: Mapping[str, float] | None = None,
    seed: int = 20260917,
) -> pd.DataFrame:
    """Repartition one labelled manifest split into pilot train/validation/test."""

    fractions = fractions or {"train": 0.8, "validation": 0.1, "test": 0.1}
    expected_splits = {"train", "validation", "test"}
    if set(fractions) != expected_splits:
        raise ValueError(f"Nested split fractions must contain {sorted(expected_splits)}.")
    if any(float(value) < 0.0 for value in fractions.values()):
        raise ValueError("Nested split fractions must be non-negative.")
    if abs(sum(float(value) for value in fractions.values()) - 1.0) > 1e-8:
        raise ValueError("Nested split fractions must sum to 1.0.")
    required = {"state_id", "protein_id", "cluster_id", "split", "sequence"}
    missing = sorted(required - set(states.columns))
    if missing:
        raise ValueError(f"State manifest is missing required columns: {missing}")
    selected = states[states["split"].astype(str) == str(source_split)].copy()
    if selected.empty:
        raise ValueError(f"No states found in source split {source_split!r}.")
    if selected["state_id"].duplicated().any():
        raise ValueError("State manifest contains duplicate state_id values.")

    clusters = np.asarray(sorted(selected["cluster_id"].astype(str).unique()), dtype=object)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(clusters)
    counts = _split_counts(len(clusters), fractions)
    train_end = counts["train"]
    validation_end = train_end + counts["validation"]
    mapping = {
        **{str(value): "train" for value in clusters[:train_end]},
        **{str(value): "validation" for value in clusters[train_end:validation_end]},
        **{str(value): "test" for value in clusters[validation_end:]},
    }
    selected["manifest_split"] = selected["split"].astype(str)
    selected["split"] = selected["cluster_id"].astype(str).map(mapping)
    selected["sft_split_seed"] = int(seed)
    assert_no_cluster_leakage(selected)
    return selected.reset_index(drop=True)


def _state_relative_path(state_id: str) -> Path:
    digest = stable_hash(str(state_id), length=32)
    return Path("state_data") / digest[:2] / f"{digest}.npz"


def _embedding_relative_path(state_id: str, embedding_dim: int) -> Path:
    digest = stable_hash(str(state_id), length=32)
    return Path("embeddings") / f"esm2_{int(embedding_dim)}" / digest[:2] / f"{digest}.npy"


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _action_columns(path: Path, target_column: str) -> list[str]:
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as parquet

            available = list(parquet.read_schema(path).names)
        except ImportError:
            available = list(pd.read_parquet(path).columns)
    else:
        available = list(pd.read_csv(path, nrows=0).columns)
    required = {"state_id", "action_index", "reward_mean", target_column}
    missing = sorted(required - set(available))
    if missing:
        raise ValueError(f"Action labels are missing required columns: {missing}")
    optional = {
        "reward_lcb",
        "reward_ucb",
        "confidence_weight",
        "label_class",
        "delta_strength_mean",
        "delta_toughness_mean",
    }
    return [column for column in available if column in required | optional]


def _iter_state_groups(
    path: Path,
    *,
    columns: Sequence[str],
    chunk_size: int,
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield contiguous state groups without loading a large action table."""

    if path.suffix.lower() == ".parquet":
        chunks: Iterator[pd.DataFrame] = iter([pd.read_parquet(path, columns=list(columns))])
    else:
        chunks = pd.read_csv(
            path,
            usecols=list(columns),
            keep_default_na=False,
            chunksize=int(chunk_size),
        )
    pending = pd.DataFrame(columns=list(columns))
    finished: set[str] = set()
    for chunk in chunks:
        if chunk.empty:
            continue
        combined = pd.concat((pending, chunk), ignore_index=True) if not pending.empty else chunk
        last_state = str(combined.iloc[-1]["state_id"])
        pending = combined[combined["state_id"].astype(str) == last_state].copy()
        ready = combined[combined["state_id"].astype(str) != last_state]
        for state_id, group in ready.groupby("state_id", sort=False):
            key = str(state_id)
            if key in finished:
                raise ValueError(
                    "Action labels must be contiguous by state_id for streaming compact export; "
                    f"state {key!r} appeared more than once."
                )
            finished.add(key)
            yield key, group.copy()
    if not pending.empty:
        state_id = str(pending.iloc[0]["state_id"])
        if state_id in finished:
            raise ValueError(f"Action labels for state {state_id!r} are not contiguous.")
        yield state_id, pending.copy()


def _compact_arrays(
    group: pd.DataFrame,
    *,
    sequence_length: int,
    target_column: str,
    dead_zone_tau: float,
    uncertain_weight: float,
) -> dict[str, np.ndarray]:
    action_count = int(sequence_length) * 20
    indices = pd.to_numeric(group["action_index"], errors="raise").astype(np.int64).to_numpy()
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"Duplicate action_index values found for state {group.iloc[0]['state_id']!r}.")
    if np.any(indices < 0) or np.any(indices >= action_count):
        raise ValueError(f"Action index outside action space for state {group.iloc[0]['state_id']!r}.")

    targets = np.full(action_count, np.nan, dtype=np.float32)
    reward_means = np.full(action_count, np.nan, dtype=np.float32)
    confidence = np.zeros(action_count, dtype=np.float32)
    label_codes = np.full(action_count, LABEL_UNOBSERVED, dtype=np.int8)
    strength = np.full(action_count, np.nan, dtype=np.float32)
    toughness = np.full(action_count, np.nan, dtype=np.float32)
    observed = np.zeros(action_count, dtype=np.bool_)

    target_values = pd.to_numeric(group[target_column], errors="coerce").to_numpy(dtype=np.float64)
    mean_values = pd.to_numeric(group["reward_mean"], errors="coerce").to_numpy(dtype=np.float64)
    lcb_values = (
        pd.to_numeric(group["reward_lcb"], errors="coerce").to_numpy(dtype=np.float64)
        if "reward_lcb" in group
        else target_values
    )
    ucb_values = (
        pd.to_numeric(group["reward_ucb"], errors="coerce").to_numpy(dtype=np.float64)
        if "reward_ucb" in group
        else target_values
    )
    weights = (
        pd.to_numeric(group["confidence_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        if "confidence_weight" in group
        else np.ones(len(group), dtype=np.float64)
    )
    structural_failure = (
        group["label_class"].astype(str).to_numpy() == "structural_failure"
        if "label_class" in group
        else np.zeros(len(group), dtype=np.bool_)
    )
    finite = np.isfinite(target_values) & np.isfinite(mean_values) & ~structural_failure
    positive = finite & (lcb_values > float(dead_zone_tau))
    negative = finite & (ucb_values < -float(dead_zone_tau))
    uncertain = finite & ~(positive | negative)
    effective_targets = target_values.copy()
    effective_targets[uncertain] = 0.0
    effective_weights = np.clip(weights, 0.0, None)
    effective_weights[uncertain] *= float(uncertain_weight)

    valid_indices = indices[finite]
    targets[valid_indices] = effective_targets[finite].astype(np.float32)
    reward_means[valid_indices] = mean_values[finite].astype(np.float32)
    confidence[valid_indices] = effective_weights[finite].astype(np.float32)
    observed[valid_indices] = True
    label_codes[indices[positive]] = LABEL_POSITIVE
    label_codes[indices[negative]] = LABEL_NEGATIVE
    label_codes[indices[uncertain]] = LABEL_UNCERTAIN
    if "delta_strength_mean" in group:
        values = pd.to_numeric(group["delta_strength_mean"], errors="coerce").to_numpy(dtype=np.float64)
        strength[valid_indices] = values[finite].astype(np.float32)
    if "delta_toughness_mean" in group:
        values = pd.to_numeric(group["delta_toughness_mean"], errors="coerce").to_numpy(dtype=np.float64)
        toughness[valid_indices] = values[finite].astype(np.float32)
    shape = (int(sequence_length), 20)
    return {
        "targets": targets.reshape(shape),
        "reward_means": reward_means.reshape(shape),
        "confidence_weights": confidence.reshape(shape),
        "label_codes": label_codes.reshape(shape),
        "strength_deltas": strength.reshape(shape),
        "toughness_deltas": toughness.reshape(shape),
        "observed_mask": observed.reshape(shape),
    }


def _state_file_summary(path: Path) -> dict[str, int]:
    with np.load(path, allow_pickle=False) as payload:
        observed = payload["observed_mask"].astype(bool)
        labels = payload["label_codes"]
        return {
            "action_count": int(observed.sum()),
            "confirmed_positive": int(((labels == LABEL_POSITIVE) & observed).sum()),
            "confirmed_negative": int(((labels == LABEL_NEGATIVE) & observed).sum()),
            "near_zero_or_uncertain": int(((labels == LABEL_UNCERTAIN) & observed).sum()),
        }


def create_compact_release(
    *,
    states_path: str | Path,
    actions_path: str | Path,
    output_dir: str | Path,
    metadata: Mapping[str, Any],
    source_split: str = "train",
    split_fractions: Mapping[str, float] | None = None,
    split_seed: int = 20260917,
    target_column: str = "reward_lcb",
    dead_zone_tau: float = 0.01,
    uncertain_weight: float = 0.25,
    chunk_size: int = 100000,
) -> dict[str, Any]:
    """Create an ID-indexed release with one compact target file per state."""

    if float(dead_zone_tau) < 0.0:
        raise ValueError("dead_zone_tau must be non-negative.")
    if not 0.0 <= float(uncertain_weight) <= 1.0:
        raise ValueError("uncertain_weight must be between 0 and 1.")
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive.")
    source_states_path = Path(states_path).expanduser().resolve()
    source_states = read_table(source_states_path)
    states = nested_split_states(
        source_states,
        source_split=source_split,
        fractions=split_fractions,
        seed=split_seed,
    )
    state_lookup = states.set_index("state_id", drop=False)
    eligible_ids = set(map(str, state_lookup.index))
    action_path = Path(actions_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    effective_fractions = dict(
        split_fractions or {"train": 0.8, "validation": 0.1, "test": 0.1}
    )
    state_stat = source_states_path.stat()
    action_stat = action_path.stat()
    build_contract = {
        "format_version": COMPACT_FORMAT_VERSION,
        "source_states": str(source_states_path),
        "source_states_size": int(state_stat.st_size),
        "source_states_mtime_ns": int(state_stat.st_mtime_ns),
        "source_actions": str(action_path),
        "source_actions_size": int(action_stat.st_size),
        "source_actions_mtime_ns": int(action_stat.st_mtime_ns),
        "source_split": str(source_split),
        "split_fractions": effective_fractions,
        "split_seed": int(split_seed),
        "target_column": str(target_column),
        "dead_zone_tau": float(dead_zone_tau),
        "uncertain_weight": float(uncertain_weight),
        "metadata": dict(metadata),
    }
    build_contract_path = destination / "release_build_config.json"
    if build_contract_path.is_file():
        existing_contract = json.loads(build_contract_path.read_text(encoding="utf-8"))
        if existing_contract != build_contract:
            raise ValueError(
                "Compact release directory contains state files generated with a different "
                f"configuration: {build_contract_path}. Use a new release directory."
            )
    else:
        write_json(build_contract, build_contract_path)
    columns = _action_columns(action_path, target_column)

    released_rows: list[dict[str, Any]] = []
    counts = {
        "action_count": 0,
        "confirmed_positive": 0,
        "confirmed_negative": 0,
        "near_zero_or_uncertain": 0,
    }
    for state_id, group in _iter_state_groups(
        action_path,
        columns=columns,
        chunk_size=chunk_size,
    ):
        if state_id not in eligible_ids:
            continue
        state = state_lookup.loc[state_id]
        relative = _state_relative_path(state_id)
        output = destination / relative
        if not output.is_file():
            arrays = _compact_arrays(
                group,
                sequence_length=len(str(state["sequence"])),
                target_column=target_column,
                dead_zone_tau=dead_zone_tau,
                uncertain_weight=uncertain_weight,
            )
            _write_npz_atomic(output, **arrays)
        summary = _state_file_summary(output)
        if summary["action_count"] <= 0:
            continue
        for key in counts:
            counts[key] += summary[key]
        row = state.to_dict()
        row["state_data_path"] = str(relative)
        row["embedding_path"] = ""
        row["embedding_dim"] = 0
        row["embedding_dtype"] = ""
        released_rows.append(row)

    if not released_rows:
        raise ValueError("No labelled states were exported to the compact release.")
    released = pd.DataFrame(released_rows).sort_values(["split", "state_id"]).reset_index(drop=True)
    assert_no_cluster_leakage(released)
    missing_splits = sorted({"train", "validation", "test"} - set(released["split"]))
    if missing_splits:
        raise ValueError(f"Compact release has no labelled states for splits: {missing_splits}")
    states_output = write_table(released, destination / "states.csv")
    payload = {
        **dict(metadata),
        "format_version": COMPACT_FORMAT_VERSION,
        "states_file": states_output.name,
        "source_states": str(source_states_path),
        "source_actions": str(action_path),
        "source_split": str(source_split),
        "split_fractions": effective_fractions,
        "split_seed": int(split_seed),
        "target_column": str(target_column),
        "dead_zone_tau": float(dead_zone_tau),
        "uncertain_weight": float(uncertain_weight),
        "state_count": int(len(released)),
        "protein_count": int(released["protein_id"].nunique()),
        "cluster_count": int(released["cluster_id"].nunique()),
        "split_state_counts": {
            key: int(value) for key, value in released["split"].value_counts().items()
        },
        "split_cluster_counts": {
            key: int(value)
            for key, value in released.groupby("split")["cluster_id"].nunique().items()
        },
        "label_counts": {
            "confirmed_positive": counts["confirmed_positive"],
            "confirmed_negative": counts["confirmed_negative"],
            "near_zero_or_uncertain": counts["near_zero_or_uncertain"],
        },
        "action_count": counts["action_count"],
        "embedding_cache": {"complete": False},
    }
    write_json(payload, destination / "dataset_version.json")
    return {**payload, "release_dir": str(destination)}
