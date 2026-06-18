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
DEFAULT_BACKBONE_ATOMS = ("N", "CA", "C", "O")
DEFAULT_MAX_MISSING_BACKBONE_FRACTION = 0.05
CANONICAL_PROTEIN_RESIDUES = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }
)


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


def _canonical_residue_atoms(path: PathLike) -> dict[tuple[str, str, str, str], set[str]]:
    """Collect atom names for canonical protein residues from PDB ATOM records."""

    residue_atoms: dict[tuple[str, str, str, str], set[str]] = {}
    path = Path(path).expanduser().resolve()
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as file:
            for line in file:
                if not line.startswith("ATOM"):
                    continue
                atom_name = line[12:16].strip().upper()
                residue_name = line[17:20].strip().upper()
                if residue_name not in CANONICAL_PROTEIN_RESIDUES:
                    continue
                chain_id = line[21:22].strip()
                residue_number = line[22:26].strip()
                insertion_code = line[26:27].strip()
                residue_key = (chain_id, residue_number, insertion_code, residue_name)
                residue_atoms.setdefault(residue_key, set()).add(atom_name)
    except OSError as exc:
        LOGGER.warning("Skipping unreadable structure file path=%s error=%s", path, exc)
        return {}
    return residue_atoms


def count_canonical_protein_residues(path: PathLike) -> int:
    """Count unique canonical protein residues from PDB ATOM records."""

    return len(_canonical_residue_atoms(path))


def backbone_missing_fraction(
    path: PathLike,
    *,
    backbone_atoms: Sequence[str] = DEFAULT_BACKBONE_ATOMS,
) -> float:
    """Return the fraction of canonical residues missing required backbone atoms."""

    required_atoms = {str(atom).strip().upper() for atom in backbone_atoms if str(atom).strip()}
    if not required_atoms:
        raise ValueError("backbone_atoms must not be empty.")

    residue_atoms = _canonical_residue_atoms(path)
    if not residue_atoms:
        return 1.0
    missing_count = sum(1 for atoms in residue_atoms.values() if not required_atoms.issubset(atoms))
    return float(missing_count / len(residue_atoms))


def is_protein_structure_file(
    path: PathLike,
    *,
    min_protein_residues: int = 1,
    max_missing_backbone_fraction: float = DEFAULT_MAX_MISSING_BACKBONE_FRACTION,
    backbone_atoms: Sequence[str] = DEFAULT_BACKBONE_ATOMS,
) -> bool:
    """Return True when ``path`` contains enough canonical protein residues."""

    if int(min_protein_residues) <= 0:
        raise ValueError("min_protein_residues must be a positive integer.")
    if not 0.0 <= float(max_missing_backbone_fraction) <= 1.0:
        raise ValueError("max_missing_backbone_fraction must satisfy 0 <= value <= 1.")
    residue_count = count_canonical_protein_residues(path)
    missing_fraction = backbone_missing_fraction(path, backbone_atoms=backbone_atoms)
    is_valid = (
        residue_count >= int(min_protein_residues)
        and missing_fraction <= float(max_missing_backbone_fraction)
    )
    if not is_valid:
        LOGGER.info(
            "Filtered invalid structure path=%s canonical_residues=%s required=%s "
            "missing_backbone_fraction=%.4f max_missing_backbone_fraction=%.4f",
            Path(path).expanduser(),
            residue_count,
            min_protein_residues,
            missing_fraction,
            max_missing_backbone_fraction,
        )
    return is_valid


def filter_protein_structure_files(
    files: Sequence[Path],
    *,
    min_protein_residues: int = 1,
    require_non_empty: bool = True,
    max_missing_backbone_fraction: float = DEFAULT_MAX_MISSING_BACKBONE_FRACTION,
    backbone_atoms: Sequence[str] = DEFAULT_BACKBONE_ATOMS,
) -> List[Path]:
    """Keep only PDB-like files that contain canonical protein residues."""

    valid: List[Path] = []
    invalid_count = 0
    for path in files:
        if is_protein_structure_file(
            path,
            min_protein_residues=min_protein_residues,
            max_missing_backbone_fraction=max_missing_backbone_fraction,
            backbone_atoms=backbone_atoms,
        ):
            valid.append(Path(path).expanduser().resolve())
        else:
            invalid_count += 1

    LOGGER.info(
        "Protein structure preprocessing complete total=%s valid=%s filtered=%s "
        "min_residues=%s max_missing_backbone_fraction=%.4f",
        len(files),
        len(valid),
        invalid_count,
        min_protein_residues,
        max_missing_backbone_fraction,
    )
    if require_non_empty and not valid:
        raise FileNotFoundError(
            "No valid protein structure files were found after preprocessing. "
            f"min_protein_residues={min_protein_residues} "
            f"max_missing_backbone_fraction={max_missing_backbone_fraction}"
        )
    return sorted(valid)


@dataclass(frozen=True)
class ProteinStructureDataset:
    """Train/validation split over a directory of PDB-like structure files."""

    pdb_dir: Path
    train_paths: Tuple[Path, ...]
    val_paths: Tuple[Path, ...]
    train_index_path: Path
    val_index_path: Path
    min_protein_residues: int = 1
    max_missing_backbone_fraction: float = DEFAULT_MAX_MISSING_BACKBONE_FRACTION

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
        min_protein_residues: int = 1,
        max_missing_backbone_fraction: float = DEFAULT_MAX_MISSING_BACKBONE_FRACTION,
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

        if int(min_protein_residues) <= 0:
            raise ValueError("min_protein_residues must be a positive integer.")
        if not 0.0 <= float(max_missing_backbone_fraction) <= 1.0:
            raise ValueError("max_missing_backbone_fraction must satisfy 0 <= value <= 1.")

        if recreate_indices or not train_index.exists() or not val_index.exists():
            discovered_files = discover_structure_files(root, extensions=extensions)
            files = filter_protein_structure_files(
                discovered_files,
                min_protein_residues=min_protein_residues,
                max_missing_backbone_fraction=max_missing_backbone_fraction,
            )
            train_paths, val_paths = cls._split(files, val_fraction=val_fraction, seed=seed)
            cls._write_index(root, train_index, train_paths)
            cls._write_index(root, val_index, val_paths)
        else:
            train_paths = cls._preprocess_index_paths(
                root,
                train_index,
                min_protein_residues=min_protein_residues,
                max_missing_backbone_fraction=max_missing_backbone_fraction,
            )
            val_paths = cls._preprocess_index_paths(
                root,
                val_index,
                min_protein_residues=min_protein_residues,
                max_missing_backbone_fraction=max_missing_backbone_fraction,
            )

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
            min_protein_residues=int(min_protein_residues),
            max_missing_backbone_fraction=float(max_missing_backbone_fraction),
        )

    def sample_train_path(self, rng: np.random.Generator) -> Path:
        index = int(rng.integers(len(self.train_paths)))
        return self.train_paths[index]

    def train_epoch_paths(
        self,
        rng: np.random.Generator,
        *,
        episodes_per_epoch: int | None = None,
        shuffle: bool = True,
    ) -> Tuple[Path, ...]:
        """Return the ordered training PDB paths for one dataset epoch."""

        if episodes_per_epoch is None:
            count = len(self.train_paths)
        else:
            count = int(episodes_per_epoch)
        if count <= 0:
            raise ValueError("episodes_per_epoch must be a positive integer.")

        paths = list(self.train_paths)
        selected: List[Path] = []
        while len(selected) < count:
            if shuffle:
                order = rng.permutation(len(paths))
                cycle = [paths[int(index)] for index in order]
            else:
                cycle = paths
            selected.extend(cycle)
        return tuple(selected[:count])

    def iter_train_batches(
        self,
        rng: np.random.Generator,
        *,
        batch_size: int,
        episodes_per_epoch: int | None = None,
        shuffle: bool = True,
    ) -> Iterable[Tuple[Path, ...]]:
        """Yield batches of PDB paths for one training epoch."""

        if int(batch_size) <= 0:
            raise ValueError("batch_size must be a positive integer.")
        epoch_paths = self.train_epoch_paths(
            rng,
            episodes_per_epoch=episodes_per_epoch,
            shuffle=shuffle,
        )
        batch_size = int(batch_size)
        for start in range(0, len(epoch_paths), batch_size):
            yield epoch_paths[start : start + batch_size]

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

    @classmethod
    def _preprocess_index_paths(
        cls,
        root: Path,
        index_path: Path,
        *,
        min_protein_residues: int,
        max_missing_backbone_fraction: float,
    ) -> List[Path]:
        paths = cls._read_index(root, index_path)
        valid_paths = filter_protein_structure_files(
            paths,
            min_protein_residues=min_protein_residues,
            require_non_empty=False,
            max_missing_backbone_fraction=max_missing_backbone_fraction,
        )
        if len(valid_paths) != len(paths):
            LOGGER.warning(
                "Rewriting dataset index after protein preprocessing path=%s before=%s after=%s",
                index_path,
                len(paths),
                len(valid_paths),
            )
            cls._write_index(root, index_path, valid_paths)
        return valid_paths
