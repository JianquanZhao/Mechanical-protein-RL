"""Deterministic action-task construction for root and retained states."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .constants import AA_TO_INDEX, AMINO_ACIDS, SCHEMA_VERSION
from .hashing import task_id
from .table_io import read_table, write_table


def decode_action(action_index: int) -> tuple[int, str]:
    index = int(action_index)
    if index < 0:
        raise ValueError("action_index must be non-negative.")
    return index // len(AMINO_ACIDS) + 1, AMINO_ACIDS[index % len(AMINO_ACIDS)]


def encode_action(position: int, amino_acid: str) -> int:
    position = int(position)
    amino_acid = str(amino_acid).upper()
    if position <= 0:
        raise ValueError("position must be 1-indexed and positive.")
    if amino_acid not in AA_TO_INDEX:
        raise ValueError(f"Unsupported amino acid: {amino_acid!r}")
    return (position - 1) * len(AMINO_ACIDS) + AA_TO_INDEX[amino_acid]


def full_scan_actions(sequence: str, visited_positions: Sequence[int] = ()) -> list[int]:
    visited = {int(value) for value in visited_positions}
    actions: list[int] = []
    for position, wild_type in enumerate(str(sequence), start=1):
        if position in visited:
            continue
        for amino_acid in AMINO_ACIDS:
            if amino_acid != wild_type:
                actions.append(encode_action(position, amino_acid))
    return actions


def _parse_positions(value: Any) -> tuple[int, ...]:
    text = str(value).strip()
    if not text:
        return tuple()
    return tuple(sorted({int(item) for item in text.split(",") if item.strip()}))


def build_action_tasks(
    states: pd.DataFrame,
    *,
    repeat_seeds: Sequence[int],
    reward_contract_hash: str,
    split_names: Sequence[str] = ("train", "validation", "test"),
    candidate_actions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one immutable task row per state/action/repeat seed."""

    rows: list[dict[str, Any]] = []
    allowed_splits = set(split_names)
    candidate_lookup: dict[str, list[tuple[int, str]]] = {}
    if candidate_actions is not None:
        for row in candidate_actions.itertuples(index=False):
            candidate_lookup.setdefault(str(row.state_id), []).append(
                (int(row.action_index), str(getattr(row, "proposal_sources", "external")))
            )

    for state in states.itertuples(index=False):
        split = str(getattr(state, "split", "train"))
        if split not in allowed_splits:
            continue
        sequence = str(state.sequence)
        visited = _parse_positions(getattr(state, "visited_positions", ""))
        if candidate_actions is None:
            actions = [(action, "full_scan") for action in full_scan_actions(sequence, visited)]
        else:
            actions = candidate_lookup.get(str(state.state_id), [])
        for action_index, proposal_sources in actions:
            position, mutant = decode_action(action_index)
            if position > len(sequence):
                raise ValueError(
                    f"Action {action_index} exceeds sequence length {len(sequence)} "
                    f"for state {state.state_id}."
                )
            wild_type = sequence[position - 1]
            if wild_type == mutant or position in visited:
                continue
            candidate_sequence = sequence[: position - 1] + mutant + sequence[position:]
            for seed in repeat_seeds:
                identifier = task_id(
                    state_id_value=str(state.state_id),
                    action_index=action_index,
                    repeat_seed=int(seed),
                    contract_hash=reward_contract_hash,
                )
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "task_id": identifier,
                        "state_id": str(state.state_id),
                        "root_state_id": str(getattr(state, "root_state_id", state.state_id)),
                        "parent_state_id": str(getattr(state, "parent_state_id", "")),
                        "protein_id": str(state.protein_id),
                        "cluster_id": str(getattr(state, "cluster_id", "")),
                        "split": split,
                        "depth": int(getattr(state, "depth", 0)),
                        "pose_path": str(state.pdb_path),
                        "root_pdb_path": str(getattr(state, "root_pdb_path", state.pdb_path)),
                        "sequence": sequence,
                        "candidate_sequence": candidate_sequence,
                        "visited_positions": ",".join(map(str, visited)),
                        "action_index": int(action_index),
                        "position": int(position),
                        "wild_type_amino_acid": wild_type,
                        "mutant_amino_acid": mutant,
                        "repeat_seed": int(seed),
                        "proposal_sources": proposal_sources,
                        "reward_contract_hash": reward_contract_hash,
                    }
                )
    if not rows:
        raise ValueError("No valid action tasks were generated.")
    result = pd.DataFrame(rows).sort_values(
        ["protein_id", "state_id", "action_index", "repeat_seed"]
    )
    if result["task_id"].duplicated().any():
        raise ValueError("Task construction produced duplicate task IDs.")
    return result.reset_index(drop=True)


def build_tasks_from_files(
    states_path: str | Path,
    output_path: str | Path,
    *,
    repeat_seeds: Sequence[int],
    reward_contract_hash: str,
    split_names: Sequence[str],
    candidate_actions_path: str | Path | None = None,
) -> pd.DataFrame:
    states = read_table(states_path)
    candidates = None if candidate_actions_path is None else read_table(candidate_actions_path)
    tasks = build_action_tasks(
        states,
        repeat_seeds=repeat_seeds,
        reward_contract_hash=reward_contract_hash,
        split_names=split_names,
        candidate_actions=candidates,
    )
    write_table(tasks, output_path)
    return tasks
