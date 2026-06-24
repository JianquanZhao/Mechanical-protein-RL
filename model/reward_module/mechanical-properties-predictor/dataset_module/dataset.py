from __future__ import annotations

import csv
import hashlib
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


LOGGER = logging.getLogger(__name__)

ESM2_MODEL_SPECS = {
    1280: ("esm2_t33_650M_UR50D", 33),
    2560: ("esm2_t36_3B_UR50D", 36),
    5120: ("esm2_t48_15B_UR50D", 48),
}


@dataclass(frozen=True)
class SequenceRecord:
    record_id: str
    sequence: str
    toughness: float
    strength: float

    @property
    def targets(self) -> np.ndarray:
        return np.asarray([self.strength, self.toughness], dtype=np.float32)


@dataclass(frozen=True)
class TargetNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, targets: np.ndarray) -> "TargetNormalizer":
        array = np.asarray(targets, dtype=np.float32)
        mean = array.mean(axis=0)
        std = array.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, targets: np.ndarray) -> np.ndarray:
        return (np.asarray(targets, dtype=np.float32) - self.mean) / self.std

    def inverse_transform(self, targets: np.ndarray) -> np.ndarray:
        return np.asarray(targets, dtype=np.float32) * self.std + self.mean

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


class MechanicalPropertyDataset(Dataset):
    def __init__(
        self,
        embeddings: np.ndarray,
        targets: np.ndarray,
        records: Sequence[SequenceRecord],
    ) -> None:
        if len(embeddings) != len(targets) or len(records) != len(targets):
            raise ValueError("embeddings, targets, and records must have the same length.")
        self.embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.records = tuple(records)

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[index], self.targets[index]


def load_mechanical_property_records(
    csv_path: str | Path,
    *,
    sequence_column: str = "Sequence",
    toughness_column: str = "v127",
    strength_column: str = "v128",
    id_column: str = "PDB_ID",
    max_samples: int | None = None,
) -> list[SequenceRecord]:
    path = Path(csv_path).expanduser().resolve()
    records: list[SequenceRecord] = []
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {sequence_column, toughness_column, strength_column}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        for row_index, row in enumerate(reader):
            sequence = str(row.get(sequence_column, "")).strip().upper()
            if not sequence:
                continue
            try:
                toughness = float(row[toughness_column])
                strength = float(row[strength_column])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(toughness) or not math.isfinite(strength):
                continue
            record_id = str(row.get(id_column) or f"row_{row_index}")
            records.append(
                SequenceRecord(
                    record_id=record_id,
                    sequence=sequence,
                    toughness=toughness,
                    strength=strength,
                )
            )
            if max_samples is not None and len(records) >= int(max_samples):
                break
    if not records:
        raise ValueError(f"No valid records were loaded from {path}.")
    LOGGER.info("Loaded mechanical-property records path=%s count=%s", path, len(records))
    return records


def build_split(
    records: Sequence[SequenceRecord],
    *,
    split_method: str = "random",
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 7,
    similarity_threshold: float = 0.5,
    kmer_size: int = 5,
) -> dict[str, list[int]]:
    if split_method == "random":
        return _random_split(
            len(records),
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
    if split_method == "similarity":
        groups = _cluster_by_kmer_jaccard(
            [record.sequence for record in records],
            threshold=similarity_threshold,
            kmer_size=kmer_size,
        )
        return _group_split(
            groups,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
    raise ValueError("split_method must be one of {'random', 'similarity'}.")


def _random_split(
    size: int,
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[int]]:
    _validate_split_fractions(val_fraction, test_fraction)
    rng = np.random.default_rng(seed)
    order = list(map(int, rng.permutation(size)))
    test_count = int(round(size * test_fraction))
    val_count = int(round(size * val_fraction))
    test_indices = sorted(order[:test_count])
    val_indices = sorted(order[test_count : test_count + val_count])
    train_indices = sorted(order[test_count + val_count :])
    return {"train": train_indices, "val": val_indices, "test": test_indices}


def _group_split(
    groups: Sequence[Sequence[int]],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[int]]:
    _validate_split_fractions(val_fraction, test_fraction)
    rng = np.random.default_rng(seed)
    order = list(map(int, rng.permutation(len(groups))))
    total = sum(len(group) for group in groups)
    target_test = int(round(total * test_fraction))
    target_val = int(round(total * val_fraction))

    split = {"train": [], "val": [], "test": []}
    for group_index in order:
        group = list(groups[group_index])
        if len(split["test"]) < target_test:
            split["test"].extend(group)
        elif len(split["val"]) < target_val:
            split["val"].extend(group)
        else:
            split["train"].extend(group)
    return {name: sorted(indices) for name, indices in split.items()}


def _validate_split_fractions(val_fraction: float, test_fraction: float) -> None:
    if not 0.0 <= float(val_fraction) < 1.0:
        raise ValueError("val_fraction must satisfy 0 <= val_fraction < 1.")
    if not 0.0 <= float(test_fraction) < 1.0:
        raise ValueError("test_fraction must satisfy 0 <= test_fraction < 1.")
    if float(val_fraction) + float(test_fraction) >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.")


def _kmers(sequence: str, kmer_size: int) -> set[str]:
    if len(sequence) <= kmer_size:
        return {sequence}
    return {sequence[index : index + kmer_size] for index in range(len(sequence) - kmer_size + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = len(left | right)
    if union == 0:
        return 0.0
    return len(left & right) / union


def _cluster_by_kmer_jaccard(
    sequences: Sequence[str],
    *,
    threshold: float,
    kmer_size: int,
) -> list[list[int]]:
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("similarity_threshold must satisfy 0 <= value <= 1.")
    if int(kmer_size) <= 0:
        raise ValueError("kmer_size must be positive.")
    kmer_sets = [_kmers(sequence, int(kmer_size)) for sequence in sequences]
    groups: list[list[int]] = []
    representatives: list[set[str]] = []
    for index, kmer_set in enumerate(kmer_sets):
        assigned = False
        for group_index, representative in enumerate(representatives):
            if _jaccard(kmer_set, representative) >= threshold:
                groups[group_index].append(index)
                assigned = True
                break
        if not assigned:
            groups.append([index])
            representatives.append(kmer_set)
    LOGGER.info(
        "Built sequence-similarity groups records=%s groups=%s threshold=%s kmer_size=%s",
        len(sequences),
        len(groups),
        threshold,
        kmer_size,
    )
    return groups


def encode_records(
    records: Sequence[SequenceRecord],
    *,
    embedding_dim: int = 1280,
    device: str = "auto",
    batch_size: int = 4,
    cache_dir: str | Path | None = None,
) -> np.ndarray:
    if int(embedding_dim) not in ESM2_MODEL_SPECS:
        raise ValueError(f"embedding_dim must be one of {tuple(ESM2_MODEL_SPECS)}.")
    cache = None if cache_dir is None else Path(cache_dir).expanduser().resolve()
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)

    embeddings: list[np.ndarray | None] = [None] * len(records)
    missing_indices: list[int] = []
    for index, record in enumerate(records):
        path = _embedding_cache_path(cache, record.sequence, embedding_dim)
        if path is not None and path.is_file():
            embeddings[index] = np.load(path).astype(np.float32, copy=False)
        else:
            missing_indices.append(index)

    if missing_indices:
        computed = _compute_esm2_embeddings(
            [records[index].sequence for index in missing_indices],
            embedding_dim=embedding_dim,
            device=device,
            batch_size=batch_size,
        )
        for index, embedding in zip(missing_indices, computed):
            array = embedding.astype(np.float32, copy=False)
            embeddings[index] = array
            path = _embedding_cache_path(cache, records[index].sequence, embedding_dim)
            if path is not None:
                np.save(path, array)

    stacked = np.stack([embedding for embedding in embeddings if embedding is not None]).astype(np.float32)
    if stacked.shape != (len(records), int(embedding_dim)):
        raise RuntimeError(f"Unexpected embedding array shape: {stacked.shape}.")
    return stacked


def _embedding_cache_path(cache_dir: Path | None, sequence: str, embedding_dim: int) -> Path | None:
    if cache_dir is None:
        return None
    digest = hashlib.sha1(sequence.encode("utf-8")).hexdigest()
    return cache_dir / f"esm2_{embedding_dim}_{digest}.npy"


def _compute_esm2_embeddings(
    sequences: Sequence[str],
    *,
    embedding_dim: int,
    device: str,
    batch_size: int,
) -> list[np.ndarray]:
    try:
        import esm
    except ImportError as exc:
        raise ImportError("fair-esm is required. Install with: pip install fair-esm") from exc
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name, representation_layer = ESM2_MODEL_SPECS[int(embedding_dim)]
    model_loader = getattr(esm.pretrained, model_name)
    model, alphabet = model_loader()
    model.eval()
    model.to(torch.device(device))
    batch_converter = alphabet.get_batch_converter()

    output: list[np.ndarray] = []
    for start in range(0, len(sequences), int(batch_size)):
        batch_sequences = sequences[start : start + int(batch_size)]
        labels = [(f"seq_{start + offset}", sequence) for offset, sequence in enumerate(batch_sequences)]
        _, _, tokens = batch_converter(labels)
        tokens = tokens.to(torch.device(device))
        with torch.no_grad():
            result = model(tokens, repr_layers=[representation_layer])
            reps = result["representations"][representation_layer]
        for offset, sequence in enumerate(batch_sequences):
            per_residue = reps[offset, 1 : len(sequence) + 1]
            pooled = per_residue.mean(dim=0).detach().cpu().numpy().astype(np.float32)
            output.append(pooled)
        LOGGER.info("Encoded ESM2 batch %s/%s", min(start + int(batch_size), len(sequences)), len(sequences))
    return output


def subset_records(records: Sequence[SequenceRecord], indices: Iterable[int]) -> list[SequenceRecord]:
    return [records[int(index)] for index in indices]
