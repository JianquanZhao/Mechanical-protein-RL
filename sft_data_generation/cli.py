"""Explicit, resumable command-line stages for one-machine SFT workflows."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml

from .config import dataset_root, load_config
from .contract import reward_contract
from .manifest import prepare_manifest
from .proposals import build_candidate_manifest
from .release import create_release
from .scan import merge_shards, scan_shard
from .search import frontier_to_states, select_global_beam
from .statistics import aggregate_file
from .table_io import read_table, write_json, write_table
from .tasks import build_tasks_from_files


LOGGER = logging.getLogger(__name__)


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _round_dir(root: Path, round_index: int, depth: int) -> Path:
    return root / "rounds" / f"round_{int(round_index):03d}" / f"depth_{int(depth):03d}"


def _load(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    config = load_config(args.config)
    return config, dataset_root(config)


def command_init(args: argparse.Namespace) -> None:
    config, root = _load(args)
    roots = [
        Path(config["paths"][name]).expanduser().resolve()
        for name in ("source_root", "generated_root", "model_root", "log_root", "tensorboard_root")
    ]
    for path in roots:
        path.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    resolved = {key: value for key, value in config.items() if not key.startswith("_")}
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "metadata" / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    contract = reward_contract(config)
    write_json(contract, root / "metadata" / "reward_contract.json")
    _print({"dataset_root": str(root), "created": [str(path) for path in roots]})


def command_prepare(args: argparse.Namespace) -> None:
    config, root = _load(args)
    contract = reward_contract(config)
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    write_json(contract, root / "metadata" / "reward_contract.json")
    summary = prepare_manifest(config, limit=args.limit)
    summary["reserved_rl_validation_index"] = config["source"]["reserved_validation_index"]
    _print(summary)


def command_build_tasks(args: argparse.Namespace) -> None:
    config, root = _load(args)
    stage = _round_dir(root, args.round, args.depth)
    stage.mkdir(parents=True, exist_ok=True)
    states = Path(args.states).expanduser().resolve() if args.states else root / "manifests" / "split_manifest.csv"
    output = Path(args.output).expanduser().resolve() if args.output else stage / "tasks.csv"
    contract = reward_contract(config)
    repeat_seeds = args.repeat_seed or [int(value) for value in config["scan"]["repeat_seeds"]]
    tasks = build_tasks_from_files(
        states,
        output,
        repeat_seeds=repeat_seeds,
        reward_contract_hash=str(contract["reward_contract_hash"]),
        split_names=args.split,
        candidate_actions_path=args.candidates,
    )
    _print({"task_count": len(tasks), "states": str(states), "tasks": str(output)})


def command_build_candidates(args: argparse.Namespace) -> None:
    config, root = _load(args)
    stage = _round_dir(root, args.round, args.depth)
    states_path = Path(args.states).expanduser().resolve()
    tables = [read_table(path) for path in args.proposals]
    candidates = build_candidate_manifest(
        read_table(states_path),
        budget=int(args.budget or config["search"]["candidate_budget"]),
        random_count=int(args.random_count),
        seed=int(args.seed),
        proposal_tables=tables,
    )
    output = Path(args.output).expanduser().resolve() if args.output else stage / "candidate_actions.csv"
    write_table(candidates, output)
    _print(
        {
            "state_count": int(candidates["state_id"].nunique()),
            "candidate_count": int(len(candidates)),
            "output": str(output),
        }
    )


def command_scan(args: argparse.Namespace) -> None:
    config, root = _load(args)
    stage = _round_dir(root, args.round, args.depth)
    tasks = Path(args.tasks).expanduser().resolve() if args.tasks else stage / "tasks.csv"
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else stage / "replicates"
    summary = scan_shard(
        tasks_path=tasks,
        output_dir=output,
        reward_config=config["reward"],
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        save_candidate_structures=args.save_candidate_structures,
    )
    _print(summary)


def command_merge(args: argparse.Namespace) -> None:
    _, root = _load(args)
    stage = _round_dir(root, args.round, args.depth)
    shard_dir = Path(args.shard_dir).expanduser().resolve() if args.shard_dir else stage / "replicates"
    output = Path(args.output).expanduser().resolve() if args.output else stage / "replicates.csv"
    merged = merge_shards(shard_dir, output)
    _print({"replicate_count": len(merged), "output": str(output)})


def command_aggregate(args: argparse.Namespace) -> None:
    config, root = _load(args)
    stage = _round_dir(root, args.round, args.depth)
    source = Path(args.replicates).expanduser().resolve() if args.replicates else stage / "replicates.csv"
    output = Path(args.output).expanduser().resolve() if args.output else stage / "labels.csv"
    settings = config["statistics"]
    labels = aggregate_file(
        str(source),
        str(output),
        target_column=str(settings["target_column"]),
        confidence_level=float(settings["confidence_level"]),
        bootstrap_samples=int(settings["bootstrap_samples"]),
        positive_noise_boundary=float(settings["positive_noise_boundary"]),
        negative_noise_boundary=float(settings["negative_noise_boundary"]),
    )
    _print({"label_count": len(labels), "label_counts": labels["label_class"].value_counts().to_dict(), "output": str(output)})


def command_frontier(args: argparse.Namespace) -> None:
    config, root = _load(args)
    stage = _round_dir(root, args.round, args.depth)
    source = Path(args.labels).expanduser().resolve() if args.labels else stage / "labels.csv"
    labels = read_table(source)
    width = int(args.beam_width or config["search"]["beam_width"])
    frontier = select_global_beam(
        labels,
        beam_width=width,
        require_materialized_structure=args.require_materialized_structure,
    )
    output = Path(args.output).expanduser().resolve() if args.output else stage / "frontier.csv"
    write_table(frontier, output)
    result: dict[str, Any] = {"frontier_count": len(frontier), "output": str(output)}
    if args.next_states:
        states = frontier_to_states(frontier)
        write_table(states, args.next_states)
        result["next_states"] = str(Path(args.next_states).expanduser().resolve())
    _print(result)


def command_release(args: argparse.Namespace) -> None:
    config, root = _load(args)
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / "release" / args.release_name
    summary = create_release(
        states_path=args.states or root / "manifests" / "split_manifest.csv",
        actions_path=args.actions,
        output_dir=output,
        metadata={
            "dataset_version": str(config["version"]),
            "config_hash": str(config["_config_hash"]),
            "reward_contract_hash": reward_contract(config)["reward_contract_hash"],
        },
    )
    _print(summary)


def command_status(args: argparse.Namespace) -> None:
    _, root = _load(args)
    files = sorted(str(path.relative_to(root)) for path in root.rglob("*.complete.json")) if root.exists() else []
    _print({"dataset_root": str(root), "completed_shards": len(files), "completion_files": files})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="YAML config; defaults match this project.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init-layout")
    init.set_defaults(func=command_init)

    prepare = subparsers.add_parser("prepare-manifest")
    prepare.add_argument("--limit", type=int, default=None)
    prepare.set_defaults(func=command_prepare)

    tasks = subparsers.add_parser("build-tasks")
    tasks.add_argument("--round", type=int, default=0)
    tasks.add_argument("--depth", type=int, default=0)
    tasks.add_argument("--states", default=None)
    tasks.add_argument("--candidates", default=None)
    tasks.add_argument("--output", default=None)
    tasks.add_argument("--split", action="append", default=["train"])
    tasks.add_argument("--repeat-seed", action="append", type=int, default=[])
    tasks.set_defaults(func=command_build_tasks)

    candidates = subparsers.add_parser("build-candidates")
    candidates.add_argument("--round", type=int, default=1)
    candidates.add_argument("--depth", type=int, default=1)
    candidates.add_argument("--states", required=True)
    candidates.add_argument("--proposals", action="append", default=[])
    candidates.add_argument("--budget", type=int, default=None)
    candidates.add_argument("--random-count", type=int, default=16)
    candidates.add_argument("--seed", type=int, default=20260917)
    candidates.add_argument("--output", default=None)
    candidates.set_defaults(func=command_build_candidates)

    scan = subparsers.add_parser("scan-shard")
    scan.add_argument("--round", type=int, default=0)
    scan.add_argument("--depth", type=int, default=0)
    scan.add_argument("--tasks", default=None)
    scan.add_argument("--output-dir", default=None)
    scan.add_argument("--shard-index", type=int, required=True)
    scan.add_argument("--num-shards", type=int, required=True)
    scan.add_argument("--save-candidate-structures", action="store_true")
    scan.set_defaults(func=command_scan)

    merge = subparsers.add_parser("merge-shards")
    merge.add_argument("--round", type=int, default=0)
    merge.add_argument("--depth", type=int, default=0)
    merge.add_argument("--shard-dir", default=None)
    merge.add_argument("--output", default=None)
    merge.set_defaults(func=command_merge)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--round", type=int, default=0)
    aggregate.add_argument("--depth", type=int, default=0)
    aggregate.add_argument("--replicates", default=None)
    aggregate.add_argument("--output", default=None)
    aggregate.set_defaults(func=command_aggregate)

    frontier = subparsers.add_parser("select-frontier")
    frontier.add_argument("--round", type=int, default=0)
    frontier.add_argument("--depth", type=int, default=0)
    frontier.add_argument("--labels", default=None)
    frontier.add_argument("--output", default=None)
    frontier.add_argument("--next-states", default=None)
    frontier.add_argument("--beam-width", type=int, default=None)
    frontier.add_argument("--require-materialized-structure", action="store_true")
    frontier.set_defaults(func=command_frontier)

    release = subparsers.add_parser("release")
    release.add_argument("--states", default=None)
    release.add_argument("--actions", required=True)
    release.add_argument("--release-name", default="sft_actions_v001")
    release.add_argument("--output-dir", default=None)
    release.set_defaults(func=command_release)

    status = subparsers.add_parser("status")
    status.set_defaults(func=command_status)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
