"""Content hashes used for reproducible states, tasks and contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def stable_hash(value: Any, *, length: int = 20) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def state_id(
    *,
    protein_id: str,
    sequence: str,
    pose_checksum: str,
    visited_positions: str = "",
) -> str:
    return "state_" + stable_hash(
        {
            "protein_id": protein_id,
            "sequence": sequence,
            "pose_checksum": pose_checksum,
            "visited_positions": visited_positions,
        }
    )


def task_id(
    *, state_id_value: str, action_index: int, repeat_seed: int, contract_hash: str
) -> str:
    return "task_" + stable_hash(
        {
            "state_id": state_id_value,
            "action_index": int(action_index),
            "repeat_seed": int(repeat_seed),
            "reward_contract_hash": contract_hash,
        }
    )


def config_hash(config: Mapping[str, Any]) -> str:
    return stable_hash(config, length=64)

