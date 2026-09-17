"""Freeze the mutation/reward semantics used to generate labels."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Mapping

from .constants import AMINO_ACIDS, SCHEMA_VERSION
from .hashing import config_hash, file_sha256


def reward_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    reward = dict(config["reward"])
    artifact = Path(reward["artifact_path"]).expanduser().resolve()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        commit = "unknown"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "amino_acid_order": "".join(AMINO_ACIDS),
        "action_index": "(position_1_based - 1) * 20 + amino_acid_index",
        "artifact_path": str(artifact),
        "artifact_sha256": file_sha256(artifact),
        "git_commit": commit,
        **reward,
    }
    payload["reward_contract_hash"] = config_hash(payload)
    return payload

