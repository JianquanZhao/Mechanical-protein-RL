"""Global beam selection and next-state manifest construction."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .hashing import file_sha256, state_id


LABEL_PRIORITY = {
    "confirmed_positive": 0,
    "hard_negative": 1,
    "near_zero_or_uncertain": 2,
    "confirmed_negative": 3,
    "structural_failure": 4,
}


def select_global_beam(
    labels: pd.DataFrame,
    *,
    beam_width: int = 3,
    require_materialized_structure: bool = False,
) -> pd.DataFrame:
    """Select B states per root across all parents, not B children per parent."""

    if int(beam_width) <= 0:
        raise ValueError("beam_width must be positive.")
    candidates = labels.copy()
    candidates = candidates[candidates["label_class"] != "structural_failure"]
    candidates = candidates[candidates["reward_lcb"].notna()]
    if require_materialized_structure:
        candidates = candidates[candidates["candidate_structure_path"].astype(str).str.len() > 0]
    candidates["_label_priority"] = candidates["label_class"].map(LABEL_PRIORITY).fillna(99)
    candidates = candidates.sort_values(
        ["root_state_id", "_label_priority", "reward_lcb", "reward_mean", "action_index"],
        ascending=[True, True, False, False, True],
    )
    selected = candidates.groupby("root_state_id", sort=True).head(int(beam_width)).copy()
    selected["beam_rank"] = selected.groupby("root_state_id").cumcount() + 1
    return selected.drop(columns=["_label_priority"]).reset_index(drop=True)


def frontier_to_states(frontier: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in frontier.itertuples(index=False):
        structure_path = str(getattr(row, "candidate_structure_path", ""))
        if not structure_path:
            raise ValueError(
                "Frontier cannot become the next-depth state until selected candidate "
                "structures have been materialized by a scan pass."
            )
        visited = {
            int(value)
            for value in str(getattr(row, "visited_positions", "")).split(",")
            if value.strip()
        }
        visited.add(int(row.position))
        visited_text = ",".join(map(str, sorted(visited)))
        checksum = file_sha256(structure_path)
        identifier = state_id(
            protein_id=str(row.protein_id),
            sequence=str(row.candidate_sequence),
            pose_checksum=checksum,
            visited_positions=visited_text,
        )
        rows.append(
            {
                "schema_version": str(row.schema_version),
                "state_id": identifier,
                "root_state_id": str(row.root_state_id),
                "parent_state_id": str(row.state_id),
                "protein_id": str(row.protein_id),
                "cluster_id": str(getattr(row, "cluster_id", "")),
                "split": str(row.split),
                "depth": int(row.depth) + 1,
                "pdb_path": structure_path,
                "root_pdb_path": str(row.root_pdb_path),
                "pdb_sha256": checksum,
                "sequence": str(row.candidate_sequence),
                "sequence_length": len(str(row.candidate_sequence)),
                "visited_positions": visited_text,
                "source_action_index": int(row.action_index),
                "beam_rank": int(row.beam_rank),
                "selection_reward_lcb": float(row.reward_lcb),
            }
        )
    return pd.DataFrame(rows)
