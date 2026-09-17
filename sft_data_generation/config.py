"""Configuration loading and path resolution for one-machine SFT workflows."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from .constants import (
    DEFAULT_BASE_ROOT,
    DEFAULT_PDB_DIR,
    DEFAULT_REWARD_ARTIFACT,
    DEFAULT_TRAIN_INDEX,
    DEFAULT_VAL_INDEX,
)
from .hashing import config_hash


DEFAULT_CONFIG: Dict[str, Any] = {
    "version": "sft_v001",
    "paths": {
        "base_root": str(DEFAULT_BASE_ROOT),
        "source_root": str(DEFAULT_BASE_ROOT / "data" / "source"),
        "generated_root": str(DEFAULT_BASE_ROOT / "data" / "generated"),
        "model_root": str(DEFAULT_BASE_ROOT / "model"),
        "log_root": str(DEFAULT_BASE_ROOT / "log"),
        "tensorboard_root": str(DEFAULT_BASE_ROOT / "tensorboard"),
    },
    "source": {
        "pdb_dir": str(DEFAULT_PDB_DIR),
        "train_index": str(DEFAULT_TRAIN_INDEX),
        "reserved_validation_index": str(DEFAULT_VAL_INDEX),
        "materialize": "symlink",
        "sequence_parser": "environment",
        "require_single_chain": True,
    },
    "split": {
        "cluster_method": "exact",
        "mmseqs_binary": "mmseqs",
        "min_sequence_identity": 0.3,
        "coverage": 0.8,
        "fractions": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "seed": 20260917,
    },
    "reward": {
        "artifact_path": str(DEFAULT_REWARD_ARTIFACT),
        "local_repack_radius": 8.0,
        "perform_repack": True,
        "perform_minimize": False,
        "minimize_backbone": False,
        "step_reward_scale": 0.025,
        "terminal_reward_scale": 8.0,
        "pyrosetta_init_options": "-mute all",
    },
    "scan": {
        "repeat_seeds": [11],
        "num_shards": 1,
        "save_candidate_structures": False,
    },
    "statistics": {
        "confidence_level": 0.95,
        "bootstrap_samples": 2000,
        "positive_noise_boundary": 0.0,
        "negative_noise_boundary": 0.0,
        "target_column": "marginal_reward_scaled",
    },
    "search": {"beam_width": 3, "candidate_budget": 64, "max_depth": 8},
}


def _deep_update(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    config_path = None
    if path is not None:
        config_path = Path(path).expanduser().resolve()
        user_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(user_config, Mapping):
            raise TypeError("SFT config root must be a mapping.")
        _deep_update(config, user_config)
    validate_config(config)
    config["_config_path"] = None if config_path is None else str(config_path)
    config["_config_hash"] = config_hash(
        {key: value for key, value in config.items() if not key.startswith("_")}
    )
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    fractions = config["split"]["fractions"]
    expected = {"train", "validation", "test"}
    if set(fractions) != expected:
        raise ValueError(f"split.fractions must contain exactly {sorted(expected)}.")
    if abs(sum(float(value) for value in fractions.values()) - 1.0) > 1e-8:
        raise ValueError("split.fractions must sum to 1.0.")
    if any(float(value) < 0.0 for value in fractions.values()):
        raise ValueError("split fractions must be non-negative.")
    if config["source"]["materialize"] not in {"copy", "symlink", "none"}:
        raise ValueError("source.materialize must be copy, symlink, or none.")
    if config["source"].get("sequence_parser", "environment") not in {
        "environment",
        "pdb_text",
    }:
        raise ValueError("source.sequence_parser must be environment or pdb_text.")
    if config["split"]["cluster_method"] not in {"exact", "mmseqs"}:
        raise ValueError("split.cluster_method must be exact or mmseqs.")
    repeat_seeds = [int(value) for value in config["scan"]["repeat_seeds"]]
    if not repeat_seeds or len(set(repeat_seeds)) != len(repeat_seeds):
        raise ValueError("scan.repeat_seeds must contain unique integer seeds.")


def dataset_root(config: Mapping[str, Any]) -> Path:
    generated = Path(config["paths"]["generated_root"]).expanduser().resolve()
    return generated / str(config["version"])


def make_run_id(version: str) -> str:
    return f"{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
