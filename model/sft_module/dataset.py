"""ID-indexed SFT dataset with lazy state-level target and embedding I/O."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_TO_INDEX = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}
LABEL_POSITIVE = 1
LABEL_NEGATIVE = 2
LABEL_UNCERTAIN = 3


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
    """Materialized compatibility representation used by legacy releases/tests."""

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


@dataclass(frozen=True)
class SFTStateReference:
    """Small state index item; numerical payloads are loaded by the collator."""

    state_id: str
    protein_id: str
    cluster_id: str
    split: str
    sequence: str
    visited_positions: tuple[int, ...]
    state_data_path: str
    embedding_path: str | None
    embedding_dim: int | None


class SFTActionDataset(Dataset[SFTStateExample | SFTStateReference]):
    """Keep only state IDs and paths in memory for compact releases."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        split: str,
        target_column: str = "reward_lcb",
        expected_embedding_dim: int | None = None,
        require_cached_embeddings: bool = False,
    ) -> None:
        root = Path(data_dir).expanduser().resolve()
        states_path = root / "states.csv"
        if not states_path.is_file():
            raise FileNotFoundError(f"SFT state index not found: {states_path}")
        states = _read_table(states_path)
        states = states[states["split"].astype(str) == str(split)].copy()
        if states.empty:
            raise ValueError(f"No states found for split={split!r} in {root}.")

        version_path = root / "dataset_version.json"
        self.dataset_version = (
            json.loads(version_path.read_text(encoding="utf-8"))
            if version_path.is_file()
            else {}
        )
        self.compact = "state_data_path" in states.columns
        if self.compact:
            examples: list[SFTStateReference] = []
            embedding_dimensions: set[int] = set()
            for row in states.itertuples(index=False):
                state_path = root / str(row.state_data_path)
                if not state_path.is_file():
                    raise FileNotFoundError(f"Compact state payload not found: {state_path}")
                raw_embedding_path = str(getattr(row, "embedding_path", "")).strip()
                embedding_path = str(root / raw_embedding_path) if raw_embedding_path else None
                raw_dimension = int(getattr(row, "embedding_dim", 0) or 0)
                embedding_dim = raw_dimension if raw_dimension > 0 else None
                if require_cached_embeddings and embedding_path is None:
                    raise ValueError(
                        f"State {row.state_id!r} has no cached ESM2 embedding. "
                        "Run sft_data_generation.cli cache-embeddings first."
                    )
                if embedding_path is not None and not Path(embedding_path).is_file():
                    raise FileNotFoundError(f"Cached embedding not found: {embedding_path}")
                if embedding_dim is not None:
                    embedding_dimensions.add(embedding_dim)
                examples.append(
                    SFTStateReference(
                        state_id=str(row.state_id),
                        protein_id=str(row.protein_id),
                        cluster_id=str(row.cluster_id),
                        split=str(split),
                        sequence=str(row.sequence),
                        visited_positions=_positions(getattr(row, "visited_positions", "")),
                        state_data_path=str(state_path),
                        embedding_path=embedding_path,
                        embedding_dim=embedding_dim,
                    )
                )
            if len(embedding_dimensions) > 1:
                raise ValueError(f"Split {split!r} mixes embedding dimensions: {embedding_dimensions}")
            self.embedding_dim = next(iter(embedding_dimensions), None)
            if (
                expected_embedding_dim is not None
                and self.embedding_dim is not None
                and self.embedding_dim != int(expected_embedding_dim)
            ):
                raise ValueError(
                    f"Cached embedding dimension {self.embedding_dim} does not match "
                    f"requested {expected_embedding_dim}."
                )
            self.examples: list[SFTStateExample | SFTStateReference] = examples
            self.uses_cached_embeddings = all(item.embedding_path is not None for item in examples)
        else:
            actions_path = root / "actions.csv"
            if not actions_path.is_file():
                raise FileNotFoundError(
                    f"Neither compact state_data_path entries nor legacy actions.csv exist in {root}."
                )
            actions = _read_table(actions_path)
            actions = actions[actions["state_id"].isin(states["state_id"])].copy()
            actions = actions[actions["label_class"] != "structural_failure"]
            actions = actions[pd.to_numeric(actions[target_column], errors="coerce").notna()]
            state_lookup = states.set_index("state_id", drop=False)
            legacy_examples: list[SFTStateExample] = []
            for state_id, group in actions.groupby("state_id", sort=True):
                state = state_lookup.loc[state_id]
                sequence = str(state["sequence"])
                action_indices = group["action_index"].astype(int).to_numpy()
                if np.any(action_indices < 0) or np.any(action_indices >= len(sequence) * 20):
                    raise ValueError(f"Action index outside sequence action space for state {state_id}.")
                legacy_examples.append(
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
            if not legacy_examples:
                raise ValueError(f"No usable action labels found for split={split!r} in {root}.")
            self.examples = legacy_examples
            self.embedding_dim = None
            self.uses_cached_embeddings = False

        self.split = split
        self.data_dir = root
        self.target_column = target_column

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> SFTStateExample | SFTStateReference:
        return self.examples[int(index)]


def _materialize_reference(
    reference: SFTStateReference,
) -> tuple[SFTStateExample, np.ndarray | None]:
    with np.load(reference.state_data_path, allow_pickle=False) as payload:
        required = {
            "targets",
            "reward_means",
            "confidence_weights",
            "label_codes",
            "strength_deltas",
            "toughness_deltas",
            "observed_mask",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"Compact state {reference.state_id!r} is missing arrays: {missing}")
        observed = np.asarray(payload["observed_mask"], dtype=np.bool_).reshape(-1)
        expected_shape = (len(reference.sequence), 20)
        target_shape = np.asarray(payload["targets"]).shape
        if target_shape != expected_shape:
            raise ValueError(
                f"Compact state {reference.state_id!r} target shape "
                f"{target_shape} does not match {expected_shape}."
            )
        indices = np.flatnonzero(observed).astype(np.int64)
        label_codes = np.asarray(payload["label_codes"], dtype=np.int8).reshape(-1)[indices]
        label_classes = tuple(
            "confirmed_positive"
            if code == LABEL_POSITIVE
            else "confirmed_negative"
            if code == LABEL_NEGATIVE
            else "near_zero_or_uncertain"
            for code in label_codes
        )
        example = SFTStateExample(
            state_id=reference.state_id,
            protein_id=reference.protein_id,
            cluster_id=reference.cluster_id,
            split=reference.split,
            sequence=reference.sequence,
            visited_positions=reference.visited_positions,
            action_indices=indices,
            reward_targets=np.asarray(payload["targets"], dtype=np.float32).reshape(-1)[indices],
            reward_means=np.asarray(payload["reward_means"], dtype=np.float32).reshape(-1)[indices],
            confidence_weights=np.asarray(payload["confidence_weights"], dtype=np.float32).reshape(-1)[indices],
            label_classes=label_classes,
            strength_deltas=np.asarray(payload["strength_deltas"], dtype=np.float32).reshape(-1)[indices],
            toughness_deltas=np.asarray(payload["toughness_deltas"], dtype=np.float32).reshape(-1)[indices],
        )
    embedding = None
    if reference.embedding_path is not None:
        embedding = np.array(
            np.load(reference.embedding_path, mmap_mode="r", allow_pickle=False),
            dtype=np.float32,
            copy=True,
        )
        expected = (len(reference.sequence), int(reference.embedding_dim or 0))
        if embedding.shape != expected:
            raise ValueError(
                f"Embedding for state {reference.state_id!r} has shape {embedding.shape}, "
                f"expected {expected}."
            )
    return example, embedding


def collate_sft_states(
    items: Sequence[SFTStateExample | SFTStateReference],
) -> Dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty SFT batch.")
    examples: list[SFTStateExample] = []
    embeddings: list[np.ndarray | None] = []
    for item in items:
        if isinstance(item, SFTStateReference):
            example, embedding = _materialize_reference(item)
            examples.append(example)
            embeddings.append(embedding)
        else:
            examples.append(item)
            embeddings.append(None)
    has_embeddings = [embedding is not None for embedding in embeddings]
    if any(has_embeddings) and not all(has_embeddings):
        raise ValueError("A batch cannot mix cached and uncached ESM2 embeddings.")

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
    batch: Dict[str, Any] = {
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
        "positive_mask": positive & observed,
        "negative_mask": negative & observed,
        "strength_deltas": strength,
        "toughness_deltas": toughness,
    }
    if all(has_embeddings):
        first_embedding = embeddings[0]
        assert first_embedding is not None
        embedding_dim = int(first_embedding.shape[1])
        padded = torch.zeros(
            (batch_size, max_length, embedding_dim),
            dtype=torch.float32,
        )
        for index, embedding in enumerate(embeddings):
            assert embedding is not None
            padded[index, : len(embedding)] = torch.from_numpy(embedding)
        batch["embeddings"] = padded
    return batch
