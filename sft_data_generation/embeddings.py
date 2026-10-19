"""Precompute one per-residue ESM2 embedding file for every released state."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from model.encoding_module.esm2_encoder import ESM2SequenceEncoder

from .hashing import stable_hash
from .table_io import read_table, write_json, write_table


def _embedding_relative_path(state_id: str, embedding_dim: int) -> Path:
    digest = stable_hash(str(state_id), length=32)
    return Path("embeddings") / f"esm2_{int(embedding_dim)}" / digest[:2] / f"{digest}.npy"


def _save_array_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)


def _batches(values: Sequence[int], batch_size: int) -> Iterable[Sequence[int]]:
    for start in range(0, len(values), int(batch_size)):
        yield values[start : start + int(batch_size)]


def cache_state_embeddings(
    release_dir: str | Path,
    *,
    embedding_dim: int = 1280,
    dtype: str = "float16",
    batch_size: int = 4,
    device: str = "auto",
    model_dir: str | Path | None = None,
    splits: Sequence[str] | None = None,
    overwrite: bool = False,
    encoder: Any | None = None,
) -> dict[str, Any]:
    """Encode released sequences and atomically store one embedding per state."""

    root = Path(release_dir).expanduser().resolve()
    states_path = root / "states.csv"
    version_path = root / "dataset_version.json"
    if not states_path.is_file() or not version_path.is_file():
        raise FileNotFoundError(f"Compact release is incomplete: {root}")
    states = read_table(states_path)
    required = {"state_id", "split", "sequence", "state_data_path"}
    missing = sorted(required - set(states.columns))
    if missing:
        raise ValueError(f"Compact state index is missing columns: {missing}")
    if dtype not in {"float16", "float32"}:
        raise ValueError("dtype must be float16 or float32.")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive.")
    selected_splits = set(map(str, splits or ("train", "validation", "test")))
    indices = [
        int(index)
        for index, value in states["split"].items()
        if str(value) in selected_splits
    ]
    if not indices:
        raise ValueError(f"No states found for requested splits: {sorted(selected_splits)}")
    if encoder is None:
        encoder = ESM2SequenceEncoder(
            embedding_dim=int(embedding_dim),
            device=device,
            mutable_only=True,
            model_dir=model_dir,
        )
    encoder_dim = int(getattr(encoder, "embedding_dim", embedding_dim))
    if encoder_dim != int(embedding_dim):
        raise ValueError(
            f"Encoder dimension {encoder_dim} does not match requested {embedding_dim}."
        )

    output_dtype = np.float16 if dtype == "float16" else np.float32
    encoded_count = 0
    reused_count = 0
    for batch_indices in _batches(indices, int(batch_size)):
        pending: list[int] = []
        for index in batch_indices:
            state_id = str(states.at[index, "state_id"])
            relative = _embedding_relative_path(state_id, embedding_dim)
            output = root / relative
            states.at[index, "embedding_path"] = str(relative)
            states.at[index, "embedding_dim"] = int(embedding_dim)
            states.at[index, "embedding_dtype"] = str(dtype)
            if output.is_file() and not overwrite:
                existing = np.load(output, mmap_mode="r", allow_pickle=False)
                expected = (len(str(states.at[index, "sequence"])), int(embedding_dim))
                if existing.shape != expected:
                    raise ValueError(
                        f"Cached embedding {output} has shape {existing.shape}, expected {expected}."
                    )
                if existing.dtype != np.dtype(output_dtype):
                    raise ValueError(
                        f"Cached embedding {output} has dtype {existing.dtype}, "
                        f"expected {np.dtype(output_dtype)}. Use --overwrite to replace it."
                    )
                reused_count += 1
            else:
                pending.append(index)
        if not pending:
            continue
        sequences = [str(states.at[index, "sequence"]) for index in pending]
        arrays = encoder.encode_sequences(sequences)
        if len(arrays) != len(pending):
            raise RuntimeError("ESM2 encoder returned a different number of embeddings than inputs.")
        for index, array in zip(pending, arrays):
            expected = (len(str(states.at[index, "sequence"])), int(embedding_dim))
            matrix = np.asarray(array)
            if matrix.shape != expected:
                raise ValueError(
                    f"Embedding for state {states.at[index, 'state_id']} has shape "
                    f"{matrix.shape}, expected {expected}."
                )
            output = root / str(states.at[index, "embedding_path"])
            _save_array_atomic(output, matrix.astype(output_dtype, copy=False))
            encoded_count += 1

    write_table(states, states_path)
    version = json.loads(version_path.read_text(encoding="utf-8"))
    complete = bool(
        states["embedding_path"].astype(str).ne("").all()
        and pd.to_numeric(states["embedding_dim"], errors="coerce")
        .eq(int(embedding_dim))
        .all()
        and states["embedding_dtype"].astype(str).eq(str(dtype)).all()
        and all((root / str(path)).is_file() for path in states["embedding_path"])
    )
    version["embedding_cache"] = {
        "complete": complete,
        "embedding_dim": int(embedding_dim),
        "dtype": str(dtype),
        "model_dir": None if model_dir is None else str(Path(model_dir).expanduser().resolve()),
        "selected_splits": sorted(selected_splits),
        "encoded_count": int(encoded_count),
        "reused_count": int(reused_count),
    }
    write_json(version, version_path)
    return {
        "release_dir": str(root),
        "state_count": int(len(indices)),
        "encoded_count": int(encoded_count),
        "reused_count": int(reused_count),
        "complete": complete,
        "embedding_dim": int(embedding_dim),
        "dtype": str(dtype),
    }
