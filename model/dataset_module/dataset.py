"""
Dataset helpers for folders of protein structure files.

The training entry point uses this module to create deterministic train/val
index files and to sample PDB paths for episode resets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np


PathLike = Union[str, Path]
LOGGER = logging.getLogger(__name__)
DEFAULT_EXTENSIONS = (".pdb", ".ent")


def discover_structure_files(
    pdb_dir: PathLike,
    *,
    extensions: Sequence[str] = DEFAULT_EXTENSIONS,
) -> List[Path]:
    """Return sorted protein structure files under ``pdb_dir``."""

    root = Path(pdb_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Protein structure directory does not exist: {root}")

    normalized_extensions = tuple(ext.lower() for ext in extensions)
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in normalized_extensions
    )
    if not files:
        raise FileNotFoundError(
            f"No protein structure files with extensions {normalized_extensions} under {root}."
        )
    LOGGER.info("Discovered structure files root=%s count=%s", root, len(files))
    return files


@dataclass(frozen=True)
class ProteinStructureDataset:
    """Train/validation split over a directory of PDB-like structure files."""

    pdb_dir: Path
    train_paths: Tuple[Path, ...]
    val_paths: Tuple[Path, ...]
    train_index_path: Path
    val_index_path: Path

    @classmethod
    def from_folder(
        cls,
        pdb_dir: PathLike,
        *,
        train_index_path: PathLike | None = None,
        val_index_path: PathLike | None = None,
        val_fraction: float = 0.1,
        seed: int = 7,
        extensions: Sequence[str] = DEFAULT_EXTENSIONS,
        recreate_indices: bool = False,
    ) -> "ProteinStructureDataset":
        root = Path(pdb_dir).expanduser().resolve()
        train_index = (
            root / "train_index.txt"
            if train_index_path is None
            else Path(train_index_path).expanduser().resolve()
        )
        val_index = (
            root / "val_index.txt"
            if val_index_path is None
            else Path(val_index_path).expanduser().resolve()
        )

        if recreate_indices or not train_index.exists() or not val_index.exists():
            files = discover_structure_files(root, extensions=extensions)
            train_paths, val_paths = cls._split(files, val_fraction=val_fraction, seed=seed)
            cls._write_index(root, train_index, train_paths)
            cls._write_index(root, val_index, val_paths)
        else:
            train_paths = cls._read_index(root, train_index)
            val_paths = cls._read_index(root, val_index)

        if not train_paths:
            raise ValueError("Training split is empty.")
        LOGGER.info(
            "ProteinStructureDataset ready pdb_dir=%s train=%s val=%s train_index=%s val_index=%s",
            root,
            len(train_paths),
            len(val_paths),
            train_index,
            val_index,
        )
        return cls(
            pdb_dir=root,
            train_paths=tuple(train_paths),
            val_paths=tuple(val_paths),
            train_index_path=train_index,
            val_index_path=val_index,
        )

    def sample_train_path(self, rng: np.random.Generator) -> Path:
        index = int(rng.integers(len(self.train_paths)))
        return self.train_paths[index]

    def validation_paths(self, *, limit: int | None = None) -> Tuple[Path, ...]:
        if limit is None or limit <= 0:
            return self.val_paths
        return self.val_paths[:limit]

    @staticmethod
    def _split(
        files: Sequence[Path],
        *,
        val_fraction: float,
        seed: int,
    ) -> Tuple[List[Path], List[Path]]:
        if not 0.0 <= float(val_fraction) < 1.0:
            raise ValueError("val_fraction must satisfy 0 <= val_fraction < 1.")
        files = list(files)
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(files))
        shuffled = [files[int(index)] for index in order]
        val_count = int(round(len(shuffled) * float(val_fraction)))
        if len(shuffled) > 1 and val_fraction > 0.0:
            val_count = max(1, val_count)
        val_count = min(val_count, max(0, len(shuffled) - 1))
        val_paths = sorted(shuffled[:val_count])
        train_paths = sorted(shuffled[val_count:])
        return train_paths, val_paths

    @staticmethod
    def _write_index(root: Path, index_path: Path, paths: Iterable[Path]) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for path in paths:
            try:
                line = path.resolve().relative_to(root).as_posix()
            except ValueError:
                line = str(path.resolve())
            lines.append(line)
        index_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        LOGGER.info("Wrote dataset index path=%s rows=%s", index_path, len(lines))

    @staticmethod
    def _read_index(root: Path, index_path: Path) -> List[Path]:
        paths: List[Path] = []
        for line_number, line in enumerate(index_path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            path = Path(stripped)
            if not path.is_absolute():
                path = root / path
            path = path.expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(
                    f"Index file {index_path} line {line_number} points to missing file: {path}"
                )
            paths.append(path)
        LOGGER.info("Read dataset index path=%s rows=%s", index_path, len(paths))
        return paths
