"""Stable constants shared by all offline SFT data stages."""

from __future__ import annotations

from pathlib import Path


SCHEMA_VERSION = "1.0"
AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_TO_INDEX = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}
THREE_TO_ONE = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}

DEFAULT_BASE_ROOT = Path("/mnt/nas/jianquanzhao/data/mprl/SFT")
DEFAULT_PDB_DIR = Path("/mnt/nas/jianquanzhao/data/mprl/pdbs/cath")
DEFAULT_TRAIN_INDEX = Path(
    "/mnt/nas/jianquanzhao/data/mprl/outputs/train/"
    "add_terminal_reward_train/indices/train_index.txt"
)
DEFAULT_VAL_INDEX = Path(
    "/mnt/nas/jianquanzhao/data/mprl/outputs/train/"
    "add_terminal_reward_train/indices/val_index.txt"
)
DEFAULT_REWARD_ARTIFACT = Path("params/hbond_random_forest.joblib")

