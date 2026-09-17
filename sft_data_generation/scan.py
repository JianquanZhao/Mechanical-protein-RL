"""Resumable, deterministic PyRosetta task-shard execution."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .adapters import RLContractAdapter
from .config import dataset_root
from .hashing import stable_hash
from .table_io import append_jsonl, read_table, write_json, write_table


LOGGER = logging.getLogger(__name__)


def shard_for_task(task_id: str, num_shards: int) -> int:
    if int(num_shards) <= 0:
        raise ValueError("num_shards must be positive.")
    return int(stable_hash(str(task_id), length=16), 16) % int(num_shards)


def _existing_task_ids(progress_path: Path) -> set[str]:
    if not progress_path.is_file():
        return set()
    identifiers: set[str] = set()
    with progress_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                identifiers.add(str(json.loads(line)["task_id"]))
    return identifiers


def scan_shard(
    *,
    tasks_path: str | Path,
    output_dir: str | Path,
    reward_config: Mapping[str, Any],
    shard_index: int,
    num_shards: int,
    save_candidate_structures: bool = False,
) -> dict[str, Any]:
    shard_index = int(shard_index)
    num_shards = int(num_shards)
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards.")
    tasks = read_table(tasks_path)
    mask = tasks["task_id"].map(lambda value: shard_for_task(str(value), num_shards) == shard_index)
    selected = tasks[mask].copy()
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / f"shard_{shard_index:05d}.jsonl"
    completed = _existing_task_ids(progress_path)
    pending = selected[~selected["task_id"].isin(completed)]
    adapter = RLContractAdapter(reward_config)
    success = 0
    failed = 0
    started = time.perf_counter()
    for row in pending.to_dict(orient="records"):
        task_started = time.perf_counter()
        output: dict[str, Any] = dict(row)
        try:
            structure_path = None
            if save_candidate_structures:
                structure_path = output_root / "candidate_structures" / f"{row['task_id']}.pdb"
            result = adapter.scan_task(row, candidate_structure_path=structure_path)
            output.update(result)
            output["success"] = True
            output["failure_reason"] = ""
            success += 1
        except Exception as exc:
            LOGGER.warning("SFT scan task failed task_id=%s error=%s", row["task_id"], exc)
            output["success"] = False
            output["failure_reason"] = f"{type(exc).__name__}: {exc}"
            failed += 1
        output["elapsed_seconds"] = time.perf_counter() - task_started
        append_jsonl([output], progress_path)

    records = read_table(progress_path) if progress_path.is_file() else pd.DataFrame()
    csv_path = output_root / f"shard_{shard_index:05d}.csv"
    if not records.empty:
        write_table(records, csv_path)
    summary = {
        "shard_index": shard_index,
        "num_shards": num_shards,
        "assigned_tasks": int(len(selected)),
        "already_complete": int(len(completed)),
        "executed": int(len(pending)),
        "succeeded": int(success),
        "failed": int(failed),
        "elapsed_seconds": time.perf_counter() - started,
        "records": str(csv_path),
    }
    write_json(summary, output_root / f"shard_{shard_index:05d}.complete.json")
    return summary


def merge_shards(shard_dir: str | Path, output_path: str | Path) -> pd.DataFrame:
    root = Path(shard_dir).expanduser().resolve()
    paths = sorted(root.glob("shard_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No completed shard CSVs found under {root}.")
    merged = pd.concat([read_table(path) for path in paths], ignore_index=True)
    merged = merged.sort_values("task_id").drop_duplicates("task_id", keep="last")
    write_table(merged, output_path)
    return merged.reset_index(drop=True)

