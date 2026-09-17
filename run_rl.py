#!/usr/bin/env python3
"""Batch protein-sequence design with a trained per-residue DDQN policy."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

from model.agent_module.ddqn_agent import AMINO_ACID_ACTION_DIM, QNetwork
from model.encoding_module import ESM2SequenceEncoder
from model.environment_module.environment import (
    AA_ONE_TO_THREE,
    CANONICAL_AMINO_ACIDS,
)


LOGGER = logging.getLogger("run_rl")
PDB_EXTENSIONS = frozenset({".pdb", ".ent"})
FASTA_EXTENSIONS = frozenset({".fa", ".faa", ".fas", ".fasta", ".fsa"})
AA_THREE_TO_ONE = {three: one for one, three in AA_ONE_TO_THREE.items()}


@dataclass(frozen=True)
class ProteinInput:
    name: str
    sequence: str
    source_path: Path
    source_detail: str


@dataclass(frozen=True)
class LoadedPolicy:
    network: QNetwork
    checkpoint_path: Path
    embedding_dim: int
    state_feature_dim: int
    device: torch.device

    @property
    def includes_visited_mask(self) -> bool:
        return self.state_feature_dim == self.embedding_dim + 1


@dataclass(frozen=True)
class Mutation:
    step: int
    position: int
    previous_amino_acid: str
    target_amino_acid: str
    q_value: float


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Design every single-chain protein in a folder with a trained "
            "per-residue DDQN checkpoint."
        )
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help=(
            "Agent checkpoint file. A checkpoint directory is also accepted; "
            "the most recently modified agent*.pt file is selected."
        ),
    )
    parser.add_argument(
        "--input-type",
        required=True,
        choices=("structure", "sequence"),
        help="Use PDB/ENT structures or single-record FASTA sequence files.",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder recursively searched for supported protein input files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Folder that receives designed FASTA files.",
    )
    parser.add_argument(
        "--updates",
        default=None,
        help=(
            "Requested unique mutation counts. Accepts one integer (24), a "
            "comma list (24,48), or inclusive ranges (24:48). By default, "
            "outputs every count from 24 through sequence length. Counts above "
            "the sequence length are capped to the sequence length."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for the Q head: auto, cpu, cuda, or cuda:N.",
    )
    parser.add_argument(
        "--esm-device",
        default=None,
        help="Optional separate ESM2 device; defaults to --device.",
    )
    parser.add_argument(
        "--esm-model-dir",
        default=None,
        help="Optional folder containing the local fair-esm checkpoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Torch/NumPy seed used for reproducible inference setup.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace FASTA outputs that already exist.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first invalid input instead of continuing the folder.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, str(level).upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )


def resolve_device(value: str) -> torch.device:
    normalized = str(value).strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {value!r} was requested but CUDA is unavailable.")
    if device.type == "cuda" and device.index is not None:
        if device.index < 0 or device.index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {device.index} is out of range for "
                f"{torch.cuda.device_count()} visible device(s)."
            )
    return device


def resolve_checkpoint_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Model path does not exist: {path}")
    candidates = [candidate for candidate in path.glob("agent*.pt") if candidate.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No agent*.pt checkpoint found in directory: {path}")
    selected = max(candidates, key=lambda candidate: (candidate.stat().st_mtime_ns, candidate.name))
    LOGGER.info("Selected latest checkpoint from directory path=%s", selected)
    return selected


def _torch_load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with old PyTorch.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"Checkpoint payload must be a mapping: {path}")
    return payload


def _without_data_parallel_prefix(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    if state_dict and all(str(name).startswith("module.") for name in state_dict):
        return {str(name)[7:]: value for name, value in state_dict.items()}
    return {str(name): value for name, value in state_dict.items()}


def load_policy(model_path: str | Path, *, device: torch.device) -> LoadedPolicy:
    checkpoint_path = resolve_checkpoint_path(model_path)
    payload = _torch_load_checkpoint(checkpoint_path)
    required = ("state_shape", "action_dim", "config", "online_network")
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {missing}.")

    state_shape = tuple(int(value) for value in payload["state_shape"])
    if len(state_shape) != 2:
        raise ValueError(
            "run_rl.py requires a per-residue checkpoint with state_shape=(L, D); "
            f"got {state_shape}."
        )
    action_dim = int(payload["action_dim"])
    if action_dim != state_shape[0] * AMINO_ACID_ACTION_DIM:
        raise ValueError(
            "Checkpoint does not use the per-residue L*20 action layout: "
            f"state_shape={state_shape}, action_dim={action_dim}."
        )
    config = payload["config"]
    if not isinstance(config, Mapping):
        raise ValueError("Checkpoint config must be a mapping.")
    embedding_dim = int(config["embedding_dim"])
    state_feature_dim = int(state_shape[-1])
    if state_feature_dim not in (embedding_dim, embedding_dim + 1):
        raise ValueError(
            "Checkpoint feature dimension must be embedding_dim or embedding_dim+1 "
            f"for a visited mask; got {state_feature_dim} and {embedding_dim}."
        )
    hidden_dims = tuple(int(value) for value in config["hidden_dims"])
    network = QNetwork(
        state_shape=(1, state_feature_dim),
        action_dim=AMINO_ACID_ACTION_DIM,
        hidden_dims=hidden_dims,
        embedding_dim=embedding_dim,
    )
    network.load_state_dict(_without_data_parallel_prefix(payload["online_network"]))
    network.to(device)
    network.eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)

    LOGGER.info(
        "Loaded DDQN online policy checkpoint=%s embedding_dim=%s "
        "state_feature_dim=%s hidden_dims=%s device=%s environment_steps=%s "
        "optimization_steps=%s",
        checkpoint_path,
        embedding_dim,
        state_feature_dim,
        hidden_dims,
        device,
        payload.get("environment_steps", "unknown"),
        payload.get("optimization_steps", "unknown"),
    )
    return LoadedPolicy(
        network=network,
        checkpoint_path=checkpoint_path,
        embedding_dim=embedding_dim,
        state_feature_dim=state_feature_dim,
        device=device,
    )


def validate_sequence(sequence: str, *, source: Path) -> str:
    normalized = "".join(str(sequence).split()).upper()
    if not normalized:
        raise ValueError(f"Protein sequence is empty: {source}")
    invalid = sorted(set(normalized) - set(CANONICAL_AMINO_ACIDS))
    if invalid:
        details = {
            amino_acid: [index + 1 for index, value in enumerate(normalized) if value == amino_acid][
                :10
            ]
            for amino_acid in invalid
        }
        raise ValueError(
            f"Only the 20 canonical amino acids are supported in {source}; "
            f"invalid residues and first positions: {details}."
        )
    return normalized


def parse_fasta_input(path: Path) -> ProteinInput:
    records: list[tuple[str, list[str]]] = []
    current_header: Optional[str] = None
    current_lines: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith(">"):
            if current_header is not None:
                records.append((current_header, current_lines))
            current_header = line[1:].strip() or f"record_at_line_{line_number}"
            current_lines = []
            continue
        if current_header is None:
            raise ValueError(f"FASTA sequence appears before the first header in {path}:{line_number}.")
        current_lines.append(line)
    if current_header is not None:
        records.append((current_header, current_lines))
    if len(records) != 1:
        raise ValueError(
            f"Each FASTA file must contain exactly one single-chain record; "
            f"found {len(records)} in {path}."
        )
    header, sequence_lines = records[0]
    sequence = validate_sequence("".join(sequence_lines), source=path)
    return ProteinInput(path.stem, sequence, path, f"FASTA header={header!r}")


def parse_pdb_input(path: Path) -> ProteinInput:
    residues_by_chain: OrderedDict[str, OrderedDict[tuple[str, str], str]] = OrderedDict()
    first_model_seen = False
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        record_name = raw_line[0:6].strip().upper()
        if record_name == "MODEL":
            if first_model_seen:
                break
            first_model_seen = True
            continue
        if record_name == "ENDMDL" and first_model_seen:
            break
        if record_name != "ATOM" or len(raw_line) < 27:
            continue
        alternate_location = raw_line[16].strip()
        if alternate_location not in ("", "A", "1"):
            continue
        residue_name = raw_line[17:20].strip().upper()
        amino_acid = AA_THREE_TO_ONE.get(residue_name)
        if amino_acid is None:
            continue
        chain_id = raw_line[21].strip() or "_"
        residue_key = (raw_line[22:26].strip(), raw_line[26].strip())
        chain_residues = residues_by_chain.setdefault(chain_id, OrderedDict())
        previous = chain_residues.get(residue_key)
        if previous is not None and previous != amino_acid:
            raise ValueError(
                f"Conflicting residue identities in {path}: chain={chain_id!r} "
                f"residue={residue_key} values={previous}/{amino_acid}."
            )
        chain_residues.setdefault(residue_key, amino_acid)

    protein_chains = OrderedDict(
        (chain_id, residues)
        for chain_id, residues in residues_by_chain.items()
        if residues
    )
    if len(protein_chains) != 1:
        chain_sizes = {chain_id: len(residues) for chain_id, residues in protein_chains.items()}
        raise ValueError(
            "Structure input must contain exactly one canonical protein chain in "
            f"the first model; found {len(protein_chains)} in {path}: {chain_sizes}."
        )
    chain_id, residues = next(iter(protein_chains.items()))
    sequence = validate_sequence("".join(residues.values()), source=path)
    return ProteinInput(path.stem, sequence, path, f"PDB chain={chain_id!r}")


def discover_input_files(
    input_dir: Path,
    *,
    input_type: str,
    excluded_dir: Optional[Path] = None,
) -> list[Path]:
    extensions = PDB_EXTENSIONS if input_type == "structure" else FASTA_EXTENSIONS
    files = []
    for path in input_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        resolved = path.resolve()
        if excluded_dir is not None and resolved.is_relative_to(excluded_dir):
            continue
        files.append(resolved)
    return sorted(files, key=lambda path: str(path.relative_to(input_dir)).lower())


def parse_update_counts(specification: Optional[str], sequence_length: int) -> tuple[int, ...]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    if specification is None:
        if sequence_length < 24:
            LOGGER.warning(
                "Default minimum of 24 unique updates exceeds sequence length %s; "
                "using %s updates instead.",
                sequence_length,
                sequence_length,
            )
            return (sequence_length,)
        return tuple(range(24, sequence_length + 1))

    values: set[int] = set()
    for raw_token in str(specification).split(","):
        token = raw_token.strip()
        if not token:
            raise ValueError(f"Invalid empty token in --updates {specification!r}.")
        if ":" in token:
            bounds = token.split(":")
            if len(bounds) != 2 or not all(bound.strip() for bound in bounds):
                raise ValueError(f"Invalid update range {token!r}; use START:STOP.")
            start, stop = (int(bound.strip()) for bound in bounds)
            if start > stop:
                raise ValueError(f"Update range start exceeds stop: {token!r}.")
            values.update(range(start, stop + 1))
        else:
            values.add(int(token))
    if not values or min(values) <= 0:
        raise ValueError("Every requested update count must be a positive integer.")
    if max(values) > sequence_length:
        LOGGER.warning(
            "Requested update count(s) up to %s exceed sequence length %s; "
            "capping them to %s unique updates.",
            max(values),
            sequence_length,
            sequence_length,
        )
        values = {min(value, sequence_length) for value in values}
    return tuple(sorted(values))


def build_action_mask(sequence: str, visited_positions: set[int]) -> np.ndarray:
    mask = np.ones(len(sequence) * AMINO_ACID_ACTION_DIM, dtype=np.bool_)
    amino_acid_to_index = {
        amino_acid: index for index, amino_acid in enumerate(CANONICAL_AMINO_ACIDS)
    }
    for position, current_amino_acid in enumerate(sequence):
        start = position * AMINO_ACID_ACTION_DIM
        if position in visited_positions:
            mask[start : start + AMINO_ACID_ACTION_DIM] = False
        else:
            mask[start + amino_acid_to_index[current_amino_acid]] = False
    return mask


def build_observation(
    sequence: str,
    *,
    encoder: ESM2SequenceEncoder,
    policy: LoadedPolicy,
    visited_positions: set[int],
) -> np.ndarray:
    embedding = np.asarray(encoder.encode_sequence(sequence), dtype=np.float32)
    expected_shape = (len(sequence), policy.embedding_dim)
    if embedding.shape != expected_shape:
        raise ValueError(
            f"ESM2 embedding has shape {embedding.shape}, expected {expected_shape}."
        )
    if policy.includes_visited_mask:
        visited = np.fromiter(
            (1.0 if position in visited_positions else 0.0 for position in range(len(sequence))),
            dtype=np.float32,
            count=len(sequence),
        ).reshape(-1, 1)
        embedding = np.concatenate((embedding, visited), axis=1)
    return np.asarray(embedding, dtype=np.float32)


def select_greedy_action(
    observation: np.ndarray,
    action_mask: np.ndarray,
    *,
    policy: LoadedPolicy,
) -> tuple[int, float]:
    state_tensor = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=policy.device,
    ).unsqueeze(0)
    with torch.inference_mode():
        q_values = policy.network(state_tensor)[0]
    expected_actions = observation.shape[0] * AMINO_ACID_ACTION_DIM
    if q_values.shape != (expected_actions,):
        raise RuntimeError(
            f"Q head returned shape {tuple(q_values.shape)}, expected ({expected_actions},)."
        )
    mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool, device=policy.device)
    valid_q_values = q_values[mask_tensor]
    if valid_q_values.numel() == 0:
        raise RuntimeError("No valid mutation action remains.")
    if not torch.isfinite(valid_q_values).all():
        raise FloatingPointError("Valid Q values contain NaN or infinity.")
    masked_q_values = q_values.masked_fill(~mask_tensor, -torch.inf)
    action = int(masked_q_values.argmax().item())
    return action, float(q_values[action].item())


def design_sequence(
    sequence: str,
    *,
    update_counts: Sequence[int],
    encoder: ESM2SequenceEncoder,
    policy: LoadedPolicy,
) -> tuple[dict[int, str], list[Mutation]]:
    normalized = validate_sequence(sequence, source=Path("<in-memory-sequence>"))
    requested = tuple(sorted(set(int(value) for value in update_counts)))
    if not requested or requested[0] <= 0 or requested[-1] > len(normalized):
        raise ValueError(
            f"update_counts must be within 1..{len(normalized)}, got {requested}."
        )

    current = list(normalized)
    visited_positions: set[int] = set()
    snapshots: dict[int, str] = {}
    mutations: list[Mutation] = []
    requested_set = set(requested)
    for step in range(1, requested[-1] + 1):
        current_sequence = "".join(current)
        observation = build_observation(
            current_sequence,
            encoder=encoder,
            policy=policy,
            visited_positions=visited_positions,
        )
        action_mask = build_action_mask(current_sequence, visited_positions)
        action, q_value = select_greedy_action(
            observation,
            action_mask,
            policy=policy,
        )
        position = action // AMINO_ACID_ACTION_DIM
        amino_acid_index = action % AMINO_ACID_ACTION_DIM
        target_amino_acid = CANONICAL_AMINO_ACIDS[amino_acid_index]
        previous_amino_acid = current[position]
        if position in visited_positions or target_amino_acid == previous_amino_acid:
            raise RuntimeError("Masked greedy policy selected an invalid mutation action.")
        current[position] = target_amino_acid
        visited_positions.add(position)
        mutations.append(
            Mutation(
                step=step,
                position=position + 1,
                previous_amino_acid=previous_amino_acid,
                target_amino_acid=target_amino_acid,
                q_value=q_value,
            )
        )
        if step in requested_set:
            snapshots[step] = "".join(current)
        LOGGER.debug(
            "Mutation step=%s position=%s previous=%s target=%s q_value=%.6f",
            step,
            position + 1,
            previous_amino_acid,
            target_amino_acid,
            q_value,
        )
    return snapshots, mutations


def write_fasta(path: Path, *, header: str, sequence: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists; use --overwrite to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f">{header}"]
    lines.extend(sequence[index : index + 80] for index in range(0, len(sequence), 80))
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text("\n".join(lines) + "\n", encoding="ascii")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _check_unique_input_names(files: Iterable[Path]) -> None:
    owners: dict[str, Path] = {}
    for path in files:
        key = path.stem.casefold()
        previous = owners.get(key)
        if previous is not None:
            raise ValueError(
                "Input basenames must be unique because output names preserve each stem: "
                f"{previous} and {path}."
            )
        owners[key] = path


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    files = discover_input_files(
        input_dir,
        input_type=args.input_type,
        excluded_dir=output_dir,
    )
    if not files:
        expected = sorted(PDB_EXTENSIONS if args.input_type == "structure" else FASTA_EXTENSIONS)
        raise FileNotFoundError(
            f"No {args.input_type} inputs with extensions {expected} found under {input_dir}."
        )
    # _check_unique_input_names(files)

    device = resolve_device(args.device)
    esm_device = resolve_device(args.esm_device or str(device))
    policy = load_policy(args.model_path, device=device)
    encoder = ESM2SequenceEncoder(
        embedding_dim=policy.embedding_dim,
        device=str(esm_device),
        mutable_only=True,
        model_dir=args.esm_model_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    parser = parse_pdb_input if args.input_type == "structure" else parse_fasta_input

    LOGGER.info(
        "Batch design started input_type=%s input_dir=%s files=%s output_dir=%s "
        "updates=%s q_device=%s esm_device=%s",
        args.input_type,
        input_dir,
        len(files),
        output_dir,
        args.updates or "24..sequence_length",
        device,
        esm_device,
    )
    completed = 0
    failed = 0
    output_count = 0
    for index, path in enumerate(files, start=1):
        protein_started = time.perf_counter()
        try:
            protein = parser(path)
            update_counts = parse_update_counts(args.updates, len(protein.sequence))
            LOGGER.info(
                "Designing protein index=%s/%s name=%s length=%s source=%s detail=%s "
                "requested_outputs=%s max_updates=%s",
                index,
                len(files),
                protein.name,
                len(protein.sequence),
                protein.source_path,
                protein.source_detail,
                len(update_counts),
                update_counts[-1],
            )
            snapshots, mutations = design_sequence(
                protein.sequence,
                update_counts=update_counts,
                encoder=encoder,
                policy=policy,
            )
            for update_count in update_counts:
                output_path = output_dir / f"{protein.name}_{update_count}.fasta"
                write_fasta(
                    output_path,
                    header=f"{protein.name}_{update_count}",
                    sequence=snapshots[update_count],
                    overwrite=bool(args.overwrite),
                )
                output_count += 1
            completed += 1
            LOGGER.info(
                "Protein design complete name=%s outputs=%s first_output=%s last_output=%s "
                "mutations=%s elapsed_sec=%.3f",
                protein.name,
                len(update_counts),
                output_dir / f"{protein.name}_{update_counts[0]}.fasta",
                output_dir / f"{protein.name}_{update_counts[-1]}.fasta",
                len(mutations),
                time.perf_counter() - protein_started,
            )
        except Exception:
            failed += 1
            LOGGER.exception("Protein design failed index=%s/%s source=%s", index, len(files), path)
            if args.fail_fast:
                raise

    LOGGER.info(
        "Batch design finished completed=%s failed=%s inputs=%s outputs=%s "
        "elapsed_sec=%.3f",
        completed,
        failed,
        len(files),
        output_count,
        time.perf_counter() - started,
    )
    if completed == 0:
        return 2
    return 1 if failed else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    try:
        return run(args)
    except Exception:
        LOGGER.exception("RL protein design terminated")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
