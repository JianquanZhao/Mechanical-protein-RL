"""Source PDB parsing, exact deduplication and cluster-level splitting."""

from __future__ import annotations

import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .constants import SCHEMA_VERSION, THREE_TO_ONE
from .hashing import file_sha256, stable_hash, state_id
from .table_io import write_table


def parse_single_chain_sequence(
    pdb_path: str | Path, *, require_single_chain: bool = True
) -> tuple[str, str]:
    """Parse the residues retained by the environment's PDB cleaning contract."""

    path = Path(pdb_path).expanduser().resolve()
    residue_order: list[tuple[str, str, str]] = []
    residue_names: dict[tuple[str, str, str], str] = {}
    residue_atoms: dict[tuple[str, str, str], set[str]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            record = line[:6].strip().upper()
            if record not in {"ATOM", "HETATM"}:
                continue
            chain = line[21:22].strip() or "_"
            key = (chain, line[22:26].strip(), line[26:27].strip())
            if key not in residue_names:
                residue_order.append(key)
                residue_names[key] = line[17:20].strip().upper()
                residue_atoms[key] = set()
            residue_atoms[key].add(line[12:16].strip().upper())
    required_backbone = {"N", "CA", "C", "O"}
    chains: dict[str, list[str]] = defaultdict(list)
    for key in residue_order:
        amino_acid = THREE_TO_ONE.get(residue_names[key])
        if amino_acid is None or not required_backbone.issubset(residue_atoms[key]):
            continue
        chains[key[0]].append(amino_acid)
    non_empty = {chain: "".join(sequence) for chain, sequence in chains.items() if sequence}
    if not non_empty:
        raise ValueError(f"No canonical protein residues found in {path}.")
    if require_single_chain and len(non_empty) != 1:
        raise ValueError(
            f"Expected one canonical protein chain in {path}, found {sorted(non_empty)}."
        )
    chain = sorted(non_empty, key=lambda key: (-len(non_empty[key]), key))[0]
    return chain, non_empty[chain]


def read_index(index_path: str | Path, pdb_dir: str | Path) -> list[Path]:
    root = Path(pdb_dir).expanduser().resolve()
    paths: list[Path] = []
    for raw in Path(index_path).expanduser().resolve().read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value:
            continue
        candidate = Path(value).expanduser()
        path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Indexed PDB does not exist: {path}")
        paths.append(path)
    if not paths:
        raise ValueError(f"No PDB entries found in index {index_path}.")
    return paths


def _materialize(source: Path, destination: Path, mode: str) -> Path:
    if mode == "none":
        return source
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise FileExistsError(f"Materialized source already exists with another target: {destination}")
        return destination
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source)
    else:
        raise ValueError(f"Unknown materialization mode: {mode}")
    return destination


def build_source_manifest(config: Mapping[str, Any], *, limit: int | None = None) -> pd.DataFrame:
    source = config["source"]
    indexed = read_index(source["train_index"], source["pdb_dir"])
    if limit is not None:
        indexed = indexed[: int(limit)]
    source_root = Path(config["paths"]["source_root"]).expanduser().resolve()
    source_root.mkdir(parents=True, exist_ok=True)
    rows: list[Dict[str, Any]] = []
    failures: list[Dict[str, str]] = []
    parser_mode = str(source.get("sequence_parser", "environment"))
    inspector = None
    if parser_mode == "environment":
        from .adapters import SourceStructureInspector

        inspector = SourceStructureInspector(config["reward"])
    for path in indexed:
        try:
            if inspector is None:
                chain, sequence = parse_single_chain_sequence(
                    path,
                    require_single_chain=bool(source.get("require_single_chain", True)),
                )
            else:
                sequence, chain_count = inspector.inspect(path)
                if bool(source.get("require_single_chain", True)) and chain_count != 1:
                    raise ValueError(
                        f"Expected one PyRosetta chain in {path}, found {chain_count}."
                    )
                chain = "pyrosetta_single_chain" if chain_count == 1 else f"pyrosetta_{chain_count}_chains"
            checksum = file_sha256(path)
            destination = source_root / path.name
            materialized = _materialize(path, destination, str(source["materialize"]))
            protein_id = path.stem
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "protein_id": protein_id,
                    "source_pdb_path": str(path),
                    "pdb_path": str(materialized),
                    "pdb_sha256": checksum,
                    "chain_id": chain,
                    "sequence_parser": parser_mode,
                    "sequence": sequence,
                    "sequence_length": len(sequence),
                    "sequence_hash": stable_hash(sequence, length=64),
                }
            )
        except Exception as exc:
            failures.append({"pdb_path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    if not rows:
        raise ValueError("No valid single-chain source PDBs remained after parsing.")
    frame = pd.DataFrame(rows).sort_values(["protein_id", "pdb_path"]).reset_index(drop=True)
    frame["exact_duplicate_count"] = frame.groupby("sequence_hash")["protein_id"].transform("size")
    frame["is_canonical_sequence_structure"] = ~frame.duplicated("sequence_hash", keep="first")
    if failures:
        write_table(pd.DataFrame(failures), source_root / "source_failures.csv")
    return frame


def _run_mmseqs(source: pd.DataFrame, config: Mapping[str, Any], work_dir: Path) -> dict[str, str]:
    split = config["split"]
    fasta = work_dir / "sequences.fasta"
    lines: list[str] = []
    for row in source.itertuples(index=False):
        lines.extend([f">{row.protein_id}", str(row.sequence)])
    fasta.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output_prefix = work_dir / "mmseqs_clusters"
    temp_dir = work_dir / "mmseqs_tmp"
    command = [
        str(split["mmseqs_binary"]),
        "easy-cluster",
        str(fasta),
        str(output_prefix),
        str(temp_dir),
        "--min-seq-id",
        str(float(split["min_sequence_identity"])),
        "-c",
        str(float(split["coverage"])),
        "--cov-mode",
        "0",
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"MMseqs2 executable not found: {split['mmseqs_binary']}. "
            "Install mmseqs2 or set split.cluster_method=exact for a smoke test."
        ) from exc
    cluster_tsv = Path(f"{output_prefix}_cluster.tsv")
    mapping: dict[str, str] = {}
    with cluster_tsv.open("r", encoding="utf-8") as handle:
        for line in handle:
            representative, member = line.rstrip("\n").split("\t")[:2]
            mapping[member] = "cluster_" + stable_hash(representative)
    missing = sorted(set(source["protein_id"]) - set(mapping))
    if missing:
        raise RuntimeError(f"MMseqs2 cluster output omitted {len(missing)} proteins.")
    return mapping


def assign_clusters(source: pd.DataFrame, config: Mapping[str, Any], work_dir: Path) -> pd.DataFrame:
    canonical = source[source["is_canonical_sequence_structure"]].copy()
    method = str(config["split"]["cluster_method"])
    if method == "exact":
        canonical["cluster_id"] = canonical["sequence_hash"].map(lambda value: f"exact_{value[:20]}")
    else:
        work_dir.mkdir(parents=True, exist_ok=True)
        mapping = _run_mmseqs(canonical, config, work_dir)
        canonical["cluster_id"] = canonical["protein_id"].map(mapping)
    canonical["cluster_method"] = method
    return canonical


def split_clusters(frame: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    fractions = config["split"]["fractions"]
    rng = np.random.default_rng(int(config["split"]["seed"]))
    clusters = np.asarray(sorted(frame["cluster_id"].unique()), dtype=object)
    rng.shuffle(clusters)
    count = len(clusters)
    split_names = ("train", "validation", "test")
    exact_counts = np.asarray([count * float(fractions[name]) for name in split_names])
    split_counts = np.floor(exact_counts).astype(int)
    for index in np.argsort(-(exact_counts - split_counts))[: count - int(split_counts.sum())]:
        split_counts[int(index)] += 1
    nonzero = [index for index, name in enumerate(split_names) if float(fractions[name]) > 0.0]
    if count >= len(nonzero):
        for index in nonzero:
            if split_counts[index] == 0:
                donor = max(nonzero, key=lambda candidate: split_counts[candidate])
                if split_counts[donor] <= 1:
                    break
                split_counts[donor] -= 1
                split_counts[index] += 1
    train_end = int(split_counts[0])
    validation_end = train_end + int(split_counts[1])
    mapping = {
        **{value: "train" for value in clusters[:train_end]},
        **{value: "validation" for value in clusters[train_end:validation_end]},
        **{value: "test" for value in clusters[validation_end:]},
    }
    result = frame.copy()
    result["split"] = result["cluster_id"].map(mapping)
    result["state_id"] = [
        state_id(
            protein_id=str(row.protein_id),
            sequence=str(row.sequence),
            pose_checksum=str(row.pdb_sha256),
        )
        for row in result.itertuples(index=False)
    ]
    result["root_state_id"] = result["state_id"]
    result["parent_state_id"] = ""
    result["depth"] = 0
    result["visited_positions"] = ""
    return result


def prepare_manifest(config: Mapping[str, Any], *, limit: int | None = None) -> dict[str, Any]:
    from .config import dataset_root

    if str(config["split"]["cluster_method"]) == "mmseqs":
        executable = str(config["split"]["mmseqs_binary"])
        if shutil.which(executable) is None:
            raise FileNotFoundError(
                f"MMseqs2 executable not found: {executable}. Install environment-cpu.yaml "
                "before parsing the full source set, or use cluster_method=exact only for smoke tests."
            )
    root = dataset_root(config)
    manifest_dir = root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    source = build_source_manifest(config, limit=limit)
    canonical = assign_clusters(source, config, manifest_dir / "mmseqs")
    split = split_clusters(canonical, config)
    source_path = write_table(source, manifest_dir / "source_manifest.csv")
    split_path = write_table(split, manifest_dir / "split_manifest.csv")
    summary = {
        "indexed_count": int(len(source)),
        "canonical_sequence_count": int(source["is_canonical_sequence_structure"].sum()),
        "exact_duplicate_count": int(len(source) - source["is_canonical_sequence_structure"].sum()),
        "cluster_count": int(split["cluster_id"].nunique()),
        "split_counts": {key: int(value) for key, value in split["split"].value_counts().items()},
        "source_manifest": str(source_path),
        "split_manifest": str(split_path),
    }
    return summary


def assert_no_cluster_leakage(frame: pd.DataFrame) -> None:
    counts = frame.groupby("cluster_id")["split"].nunique()
    leaked = counts[counts > 1]
    if not leaked.empty:
        raise ValueError(f"Sequence clusters cross SFT splits: {list(leaked.index[:10])}")
