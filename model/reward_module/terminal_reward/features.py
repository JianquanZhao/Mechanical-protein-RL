"""Hydrogen-bond topology features used by the terminal reward model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


CANONICAL_RESIDUES = {
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
BACKBONE_HBOND_ATOMS = {"N", "O", "OXT"}
DONOR_ELEMENTS = {"N", "O", "S"}
ACCEPTOR_ELEMENTS = {"N", "O", "S"}


@dataclass(frozen=True)
class ParsedAtom:
    atom_index: int
    atom_name: str
    residue_name: str
    chain_id: str
    residue_number: int
    insertion_code: str
    element: str
    coord: np.ndarray
    b_factor: float

    @property
    def residue_key(self) -> Tuple[str, int, str]:
        return self.chain_id, self.residue_number, self.insertion_code

    @property
    def is_backbone_hbond_atom(self) -> bool:
        return self.atom_name.strip().upper() in BACKBONE_HBOND_ATOMS


@dataclass(frozen=True)
class HbondRecord:
    donor_atom: ParsedAtom
    hydrogen_atom: ParsedAtom
    acceptor_atom: ParsedAtom
    donor_acceptor_distance: float
    hydrogen_acceptor_distance: float
    donor_hydrogen_acceptor_angle: float

    @property
    def is_strong(self) -> bool:
        return (
            self.donor_acceptor_distance <= 3.0
            and self.donor_hydrogen_acceptor_angle >= 150.0
        )


@dataclass(frozen=True)
class HbondFeatureResult:
    """Feature vector and diagnostics extracted from one protein structure."""

    features: Mapping[str, float]
    sequence_length: int
    atom_count: int
    hydrogen_atom_count: int
    hbond_count: int
    mean_plddt: Optional[float]
    source_path: Optional[str]
    warnings: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def feature_vector(self, feature_names: Sequence[str]) -> np.ndarray:
        return np.asarray([self.features[name] for name in feature_names], dtype=float)


class HbondFeatureExtractor:
    """
    Extract the seven selected topology / hydrogen-bond features.

    The criteria match the hbond analysis notebooks used to build the random
    forest terminal reward model:
    donor-heavy to acceptor distance <= 3.5 A, H to acceptor distance <= 2.7 A,
    and D-H-A angle >= 120 degrees. Strong nonlocal bonds additionally satisfy
    donor-heavy to acceptor distance <= 3.0 A and D-H-A angle >= 150 degrees.
    """

    selected_features = (
        "sequence_length",
        "hbond_per_residue",
        "seq_class_nonlocal_per_residue",
        "strong_nonlocal_fraction",
        "strong_nonlocal_per_residue",
        "nonlocal_backbone_backbone_per_residue",
        "hbond_contact_order",
    )

    def __init__(
        self,
        *,
        donor_hydrogen_distance_cutoff: float = 1.35,
        donor_acceptor_distance_cutoff: float = 3.5,
        hydrogen_acceptor_distance_cutoff: float = 2.7,
        angle_cutoff: float = 120.0,
    ) -> None:
        self.donor_hydrogen_distance_cutoff = float(donor_hydrogen_distance_cutoff)
        self.donor_acceptor_distance_cutoff = float(donor_acceptor_distance_cutoff)
        self.hydrogen_acceptor_distance_cutoff = float(hydrogen_acceptor_distance_cutoff)
        self.angle_cutoff = float(angle_cutoff)

    def extract_from_pose(self, pose: Any) -> HbondFeatureResult:
        if pose is None or not hasattr(pose, "dump_pdb"):
            raise ValueError("extract_from_pose expects a PyRosetta-like Pose with dump_pdb().")

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=True) as handle:
            pose.dump_pdb(handle.name)
            return self.extract_from_pdb(handle.name)

    def extract_from_pdb(self, pdb_path: str | Path) -> HbondFeatureResult:
        source_path = str(pdb_path)
        atoms = self._parse_pdb(Path(pdb_path))
        return self.extract_from_atoms(atoms, source_path=source_path)

    def extract_from_atoms(
        self,
        atoms: Sequence[ParsedAtom],
        *,
        source_path: Optional[str] = None,
    ) -> HbondFeatureResult:
        warnings: List[str] = []
        protein_atoms = [atom for atom in atoms if atom.residue_name in CANONICAL_RESIDUES]
        residue_keys = sorted({atom.residue_key for atom in protein_atoms})
        sequence_length = len(residue_keys)
        hydrogen_count = sum(1 for atom in protein_atoms if atom.element == "H")

        if sequence_length <= 0:
            raise ValueError(f"No canonical protein residues found in {source_path or 'structure'}.")
        if hydrogen_count == 0:
            warnings.append(
                "No hydrogen atoms were found; hbond features will be near zero. "
                "Use PyRosetta full-atom output or add hydrogens before reward evaluation."
            )

        hbonds = self._find_hbonds(protein_atoms)
        features = self._summarize_features(hbonds, sequence_length)
        features["sequence_length"] = float(sequence_length)

        mean_plddt = self._mean_plddt(protein_atoms)
        return HbondFeatureResult(
            features={name: float(features[name]) for name in self.selected_features},
            sequence_length=int(sequence_length),
            atom_count=int(len(protein_atoms)),
            hydrogen_atom_count=int(hydrogen_count),
            hbond_count=int(len(hbonds)),
            mean_plddt=mean_plddt,
            source_path=source_path,
            warnings=tuple(warnings),
        )

    @staticmethod
    def _parse_pdb(path: Path) -> List[ParsedAtom]:
        atoms: List[ParsedAtom] = []
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.startswith(("ATOM  ", "HETATM")):
                    continue
                altloc = line[16:17].strip()
                if altloc not in {"", "A"}:
                    continue
                try:
                    atom_name = line[12:16].strip()
                    residue_name = line[17:20].strip().upper()
                    chain_id = line[21:22].strip() or "_"
                    residue_number = int(line[22:26])
                    insertion_code = line[26:27].strip()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    b_factor = float(line[60:66]) if line[60:66].strip() else 0.0
                except ValueError:
                    continue
                element = line[76:78].strip().upper()
                if not element:
                    element = "".join(ch for ch in atom_name if ch.isalpha())[:1].upper()
                atoms.append(
                    ParsedAtom(
                        atom_index=len(atoms),
                        atom_name=atom_name,
                        residue_name=residue_name,
                        chain_id=chain_id,
                        residue_number=residue_number,
                        insertion_code=insertion_code,
                        element=element,
                        coord=np.asarray([x, y, z], dtype=float),
                        b_factor=b_factor,
                    )
                )
        return atoms

    def _find_hbonds(self, atoms: Sequence[ParsedAtom]) -> List[HbondRecord]:
        hydrogens = [atom for atom in atoms if atom.element == "H"]
        donors = [atom for atom in atoms if atom.element in DONOR_ELEMENTS]
        acceptors = [atom for atom in atoms if atom.element in ACCEPTOR_ELEMENTS]

        donor_hydrogen_pairs: List[Tuple[ParsedAtom, ParsedAtom]] = []
        for donor in donors:
            for hydrogen in hydrogens:
                if donor.residue_key != hydrogen.residue_key:
                    continue
                if self._distance(donor.coord, hydrogen.coord) <= self.donor_hydrogen_distance_cutoff:
                    donor_hydrogen_pairs.append((donor, hydrogen))

        hbonds: List[HbondRecord] = []
        seen = set()
        for donor, hydrogen in donor_hydrogen_pairs:
            for acceptor in acceptors:
                if acceptor.atom_index == donor.atom_index:
                    continue
                donor_acceptor_distance = self._distance(donor.coord, acceptor.coord)
                if donor_acceptor_distance > self.donor_acceptor_distance_cutoff:
                    continue
                hydrogen_acceptor_distance = self._distance(hydrogen.coord, acceptor.coord)
                if hydrogen_acceptor_distance > self.hydrogen_acceptor_distance_cutoff:
                    continue
                angle = self._angle(donor.coord, hydrogen.coord, acceptor.coord)
                if angle < self.angle_cutoff:
                    continue
                key = (donor.atom_index, hydrogen.atom_index, acceptor.atom_index)
                if key in seen:
                    continue
                seen.add(key)
                hbonds.append(
                    HbondRecord(
                        donor_atom=donor,
                        hydrogen_atom=hydrogen,
                        acceptor_atom=acceptor,
                        donor_acceptor_distance=float(donor_acceptor_distance),
                        hydrogen_acceptor_distance=float(hydrogen_acceptor_distance),
                        donor_hydrogen_acceptor_angle=float(angle),
                    )
                )
        return hbonds

    @staticmethod
    def _summarize_features(hbonds: Sequence[HbondRecord], sequence_length: int) -> Dict[str, float]:
        hbond_count = len(hbonds)
        nonlocal_count = 0
        strong_nonlocal_count = 0
        nonlocal_backbone_backbone_count = 0
        sequence_separations: List[int] = []

        for hbond in hbonds:
            donor = hbond.donor_atom
            acceptor = hbond.acceptor_atom
            if donor.chain_id == acceptor.chain_id:
                seq_sep = abs(donor.residue_number - acceptor.residue_number)
                sequence_separations.append(seq_sep)
                is_nonlocal = seq_sep > 4
            else:
                is_nonlocal = False

            if not is_nonlocal:
                continue
            nonlocal_count += 1
            if donor.is_backbone_hbond_atom and acceptor.is_backbone_hbond_atom:
                nonlocal_backbone_backbone_count += 1
            if hbond.is_strong:
                strong_nonlocal_count += 1

        safe_length = max(float(sequence_length), 1.0)
        safe_hbond_count = max(float(hbond_count), 1.0)
        mean_seq_sep = float(np.mean(sequence_separations)) if sequence_separations else 0.0

        return {
            "hbond_per_residue": hbond_count / safe_length,
            "seq_class_nonlocal_per_residue": nonlocal_count / safe_length,
            "strong_nonlocal_fraction": strong_nonlocal_count / safe_hbond_count,
            "strong_nonlocal_per_residue": strong_nonlocal_count / safe_length,
            "nonlocal_backbone_backbone_per_residue": nonlocal_backbone_backbone_count / safe_length,
            "hbond_contact_order": mean_seq_sep / safe_length,
        }

    @staticmethod
    def _distance(first: np.ndarray, second: np.ndarray) -> float:
        return float(np.linalg.norm(first - second))

    @staticmethod
    def _angle(donor: np.ndarray, hydrogen: np.ndarray, acceptor: np.ndarray) -> float:
        first = donor - hydrogen
        second = acceptor - hydrogen
        denom = np.linalg.norm(first) * np.linalg.norm(second)
        if denom <= 0:
            return 0.0
        cosine = float(np.dot(first, second) / denom)
        cosine = max(-1.0, min(1.0, cosine))
        return math.degrees(math.acos(cosine))

    @staticmethod
    def _mean_plddt(atoms: Sequence[ParsedAtom]) -> Optional[float]:
        if not atoms:
            return None
        values = [atom.b_factor for atom in atoms if np.isfinite(atom.b_factor)]
        if not values:
            return None
        mean_value = float(np.mean(values))
        if mean_value < 1.0 or mean_value > 100.0:
            return None
        return mean_value
