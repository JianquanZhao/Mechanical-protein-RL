"""Validation and stable release export consumed by train_sft.py."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .manifest import assert_no_cluster_leakage
from .table_io import read_table, write_json, write_table


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

