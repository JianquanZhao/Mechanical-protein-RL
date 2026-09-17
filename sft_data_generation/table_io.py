"""Atomic table I/O with CSV as the dependency-light default."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path).expanduser().resolve()
    if table_path.suffix.lower() == ".parquet":
        return pd.read_parquet(table_path)
    if table_path.suffix.lower() in {".jsonl", ".ndjson"}:
        return pd.read_json(table_path, lines=True)
    return pd.read_csv(table_path, keep_default_na=False)


def write_table(frame: pd.DataFrame, path: str | Path) -> Path:
    table_path = Path(path).expanduser().resolve()
    table_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = table_path.with_name(f".{table_path.name}.{os.getpid()}.tmp")
    suffix = table_path.suffix.lower()
    if suffix == ".parquet":
        frame.to_parquet(temporary, index=False)
    elif suffix in {".jsonl", ".ndjson"}:
        frame.to_json(temporary, orient="records", lines=True)
    else:
        frame.to_csv(temporary, index=False)
    os.replace(temporary, table_path)
    return table_path


def write_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    return output_path


def append_jsonl(rows: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")
    return output_path

