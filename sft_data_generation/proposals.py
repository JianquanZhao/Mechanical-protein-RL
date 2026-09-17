"""Merge SFT, ESM2 and model-free candidates under a fixed per-state budget."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .hashing import stable_hash
from .tasks import full_scan_actions


def build_candidate_manifest(
    states: pd.DataFrame,
    *,
    budget: int = 64,
    random_count: int = 16,
    seed: int = 20260917,
    proposal_tables: Sequence[pd.DataFrame] = (),
) -> pd.DataFrame:
    """Deduplicate model proposals, then fill the remaining budget randomly."""

    if int(budget) <= 0 or not 0 <= int(random_count) <= int(budget):
        raise ValueError("Candidate budget must be positive and random_count within [0, budget].")
    external: dict[str, list[dict[str, Any]]] = {}
    for table in proposal_tables:
        required = {"state_id", "action_index"}
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"Proposal table is missing columns: {sorted(missing)}")
        for row in table.to_dict(orient="records"):
            row.setdefault("proposal_source", "external")
            row.setdefault("proposal_score", 0.0)
            external.setdefault(str(row["state_id"]), []).append(row)

    rows: list[dict[str, Any]] = []
    for state in states.itertuples(index=False):
        state_key = str(state.state_id)
        visited = [
            int(value)
            for value in str(getattr(state, "visited_positions", "")).split(",")
            if value.strip()
        ]
        valid = set(full_scan_actions(str(state.sequence), visited))
        merged: dict[int, dict[str, Any]] = {}
        ranked = sorted(
            external.get(state_key, []),
            key=lambda row: (-float(row.get("proposal_score", 0.0)), int(row["action_index"])),
        )
        model_budget = max(0, int(budget) - int(random_count))
        for proposal in ranked:
            action = int(proposal["action_index"])
            if action not in valid:
                continue
            if action in merged:
                sources = set(str(merged[action]["proposal_sources"]).split("+"))
                sources.add(str(proposal["proposal_source"]))
                merged[action]["proposal_sources"] = "+".join(sorted(sources))
                merged[action]["proposal_score"] = max(
                    float(merged[action]["proposal_score"]),
                    float(proposal.get("proposal_score", 0.0)),
                )
            elif len(merged) < model_budget:
                merged[action] = {
                    "proposal_sources": str(proposal["proposal_source"]),
                    "proposal_score": float(proposal.get("proposal_score", 0.0)),
                }

        remaining = np.asarray(sorted(valid - set(merged)), dtype=np.int64)
        state_seed = int(stable_hash([seed, state_key], length=16), 16) % (2**32)
        rng = np.random.default_rng(state_seed)
        if len(remaining):
            rng.shuffle(remaining)
        fill_count = min(int(budget) - len(merged), len(remaining))
        for action in remaining[:fill_count]:
            merged[int(action)] = {"proposal_sources": "random", "proposal_score": float("nan")}

        for rank, (action, metadata) in enumerate(
            sorted(
                merged.items(),
                key=lambda item: (
                    item[1]["proposal_sources"] == "random",
                    -float(item[1]["proposal_score"])
                    if np.isfinite(item[1]["proposal_score"])
                    else 0.0,
                    item[0],
                ),
            ),
            start=1,
        ):
            rows.append(
                {
                    "state_id": state_key,
                    "action_index": int(action),
                    "proposal_sources": metadata["proposal_sources"],
                    "proposal_score": metadata["proposal_score"],
                    "proposal_rank": rank,
                }
            )
    return pd.DataFrame(rows)

