"""State-grouped SFT dataset with variable-length action labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_TO_INDEX = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path, keep_default_na=False)


def _positions(value: Any) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                int(item)
                for item in str(value).split(",")
                if str(item).strip()
            }
        )
    )


@dataclass(frozen=True)
class SFTStateExample:
    state_id: str
    protein_id: str
    cluster_id: str
    split: str
    sequence: str
    visited_positions: tuple[int, ...]
    action_indices: np.ndarray
    reward_targets: np.ndarray
    reward_means: np.ndarray
    confidence_weights: np.ndarray
    label_classes: tuple[str, ...]
    strength_deltas: np.ndarray
    toughness_deltas: np.ndarray


class SFTActionDataset(Dataset[SFTStateExample]):
    """Load one item per state so long proteins cannot dominate by row count."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        split: str,
        target_column: str = "reward_lcb",
    ) -> None:
        root = Path(data_dir).expanduser().resolve()
        states = _read_table(root / "states.csv")
        actions = _read_table(root / "actions.csv")
        states = states[states["split"] == split].copy()
        actions = actions[actions["state_id"].isin(states["state_id"])].copy()
        actions = actions[actions["label_class"] != "structural_failure"]
        actions = actions[pd.to_numeric(actions[target_column], errors="coerce").notna()]
        if states.empty:
            raise ValueError(f"No states found for split={split!r} in {root}.")
        state_lookup = states.set_index("state_id", drop=False)
        examples: list[SFTStateExample] = []
        for state_id, group in actions.groupby("state_id", sort=True):
            state = state_lookup.loc[state_id]
            sequence = str(state["sequence"])
            action_indices = group["action_index"].astype(int).to_numpy()
            if np.any(action_indices < 0) or np.any(action_indices >= len(sequence) * 20):
                raise ValueError(f"Action index outside sequence action space for state {state_id}.")
            examples.append(
                SFTStateExample(
                    state_id=str(state_id),
                    protein_id=str(state["protein_id"]),
                    cluster_id=str(state["cluster_id"]),
                    split=split,
                    sequence=sequence,
                    visited_positions=_positions(state.get("visited_positions", "")),
                    action_indices=action_indices,
                    reward_targets=group[target_column].astype(float).to_numpy(dtype=np.float32),
                    reward_means=group["reward_mean"].astype(float).to_numpy(dtype=np.float32),
                    confidence_weights=group.get(
                        "confidence_weight", pd.Series(np.ones(len(group)), index=group.index)
                    ).astype(float).to_numpy(dtype=np.float32),
                    label_classes=tuple(group["label_class"].astype(str)),
                    strength_deltas=pd.to_numeric(
                        group.get("delta_strength_mean", pd.Series(np.nan, index=group.index)),
                        errors="coerce",
                    ).to_numpy(dtype=np.float32),
                    toughness_deltas=pd.to_numeric(
                        group.get("delta_toughness_mean", pd.Series(np.nan, index=group.index)),
                        errors="coerce",
                    ).to_numpy(dtype=np.float32),
                )
            )
        if not examples:
            raise ValueError(f"No usable action labels found for split={split!r} in {root}.")
        self.examples = examples
        self.split = split
        self.data_dir = root
        self.target_column = target_column

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> SFTStateExample:
        return self.examples[int(index)]


def collate_sft_states(examples: Sequence[SFTStateExample]) -> Dict[str, Any]:
    if not examples:
        raise ValueError("Cannot collate an empty SFT batch.")
    batch_size = len(examples)
    max_length = max(len(example.sequence) for example in examples)
    action_count = max_length * 20
    targets = torch.full((batch_size, action_count), float("nan"), dtype=torch.float32)
    reward_means = torch.full_like(targets, float("nan"))
    weights = torch.zeros_like(targets)
    observed = torch.zeros((batch_size, action_count), dtype=torch.bool)
    positive = torch.zeros_like(observed)
    negative = torch.zeros_like(observed)
    valid = torch.zeros_like(observed)
    strength = torch.full_like(targets, float("nan"))
    toughness = torch.full_like(targets, float("nan"))
    visited = torch.zeros((batch_size, max_length, 1), dtype=torch.float32)
    lengths = torch.tensor([len(example.sequence) for example in examples], dtype=torch.long)

    for batch_index, example in enumerate(examples):
        sequence_length = len(example.sequence)
        valid[batch_index, : sequence_length * 20] = True
        for position, amino_acid in enumerate(example.sequence):
            valid[batch_index, position * 20 + AA_TO_INDEX[amino_acid]] = False
        for position in example.visited_positions:
            if 1 <= position <= sequence_length:
                valid[batch_index, (position - 1) * 20 : position * 20] = False
                visited[batch_index, position - 1, 0] = 1.0
        indices = torch.as_tensor(example.action_indices, dtype=torch.long)
        observed[batch_index, indices] = True
        targets[batch_index, indices] = torch.from_numpy(example.reward_targets)
        reward_means[batch_index, indices] = torch.from_numpy(example.reward_means)
        weights[batch_index, indices] = torch.from_numpy(example.confidence_weights)
        strength[batch_index, indices] = torch.from_numpy(example.strength_deltas)
        toughness[batch_index, indices] = torch.from_numpy(example.toughness_deltas)
        for action_index, label in zip(example.action_indices, example.label_classes):
            if label == "confirmed_positive":
                positive[batch_index, int(action_index)] = True
            elif label in {"hard_negative", "confirmed_negative"}:
                negative[batch_index, int(action_index)] = True

    observed &= valid
    return {
        "state_ids": [example.state_id for example in examples],
        "protein_ids": [example.protein_id for example in examples],
        "cluster_ids": [example.cluster_id for example in examples],
        "sequences": [example.sequence for example in examples],
        "lengths": lengths,
        "visited": visited,
        "targets": targets,
        "reward_means": reward_means,
        "confidence_weights": weights,
        "observed_mask": observed,
        "valid_mask": valid,
        "positive_mask": positive,
        "negative_mask": negative,
        "strength_deltas": strength,
        "toughness_deltas": toughness,
    }

