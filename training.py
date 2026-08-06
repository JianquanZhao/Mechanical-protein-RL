"""
Root training entry point for the mechanical-protein DDQN project.

Examples
--------
Single machine, single GPU:
    python training.py --mode single --device cuda:0

Single machine, multiple GPUs:
    python training.py --mode multi --gpu-ids 0,1,2,3 --batch-size 128

CPU/debug run:
    python training.py --mode single --device cpu --epochs 1 --max-steps 1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch

from model.agent_module.ddqn_agent import DDQNAgent, DDQNConfig, OptimizationResult
from model.dataset_module import ProteinStructureDataset
from model.environment_module.environment import MechanicalProteinEnv
from model.logging_module.training_logger import TrainingLogger, TrainingLoggerConfig
from model.replay_buffer_module.replay_buffer import ReplayBuffer


DEFAULT_PDB_DIR = "model/reward_module"
DEFAULT_OUTPUT_DIR = "outputs/ddqn_base"
RESUME_AGENT_FILENAME = "agent.pt"
RESUME_REPLAY_FILENAME = "replay_buffer.npz"
RESUME_STATE_FILENAME = "training_state.json"
RESUME_STATE_VERSION = 1
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the mechanical-protein DDQN agent in single-GPU or "
            "single-machine multi-GPU mode."
        )
    )

    parser.add_argument(
        "--mode",
        choices=("single", "multi"),
        default="single",
        help="Training mode. 'multi' uses torch.nn.DataParallel on one machine.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "Device for single mode, for example cpu, cuda, cuda:0, or auto. "
            "In multi mode this is derived from --gpu-ids."
        ),
    )
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help="Comma-separated GPU ids for multi mode, for example 0,1,2,3.",
    )

    parser.add_argument("--pdb-dir", default=DEFAULT_PDB_DIR)
    parser.add_argument("--train-index", default=None)
    parser.add_argument("--val-index", default=None)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--dataset-seed", type=int, default=7)
    parser.add_argument("--recreate-splits", action="store_true")
    parser.add_argument(
        "--min-protein-residues",
        type=int,
        default=1,
        help="Minimum canonical protein residues required for a structure file to enter train/val splits.",
    )
    parser.add_argument(
        "--max-missing-backbone-fraction",
        type=float,
        default=0.05,
        help=(
            "Maximum fraction of canonical protein residues allowed to miss "
            "N/CA/C/O atoms during dataset preprocessing."
        ),
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help=(
            "Legacy total episode count. Used only when --epochs is omitted. "
            "Prefer --epochs for dataset-style multi-batch training."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Number of dataset training epochs. Each epoch iterates over train PDB "
            "batches and runs one RL episode per selected PDB."
        ),
    )
    parser.add_argument(
        "--episodes-per-epoch",
        type=int,
        default=None,
        help=(
            "Number of PDB episodes sampled in each epoch. Defaults to the full "
            "training split size. Values larger than the split repeat shuffled cycles."
        ),
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=1,
        help="Number of PDB episodes grouped into one dataset batch inside each epoch.",
    )
    parser.add_argument(
        "--no-shuffle-train",
        action="store_true",
        help="Disable shuffling of train PDB paths within each dataset epoch.",
    )
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--mutable-positions",
        default=None,
        help="Optional comma-separated 1-indexed pose positions, for example 38,39,40.",
    )
    parser.add_argument("--local-repack-radius", type=float, default=8.0)
    parser.add_argument("--pyrosetta-options", default="-mute all")
    parser.add_argument(
        "--no-clean-pdb-before-load",
        action="store_true",
        help="Disable environment-side PDB text cleaning before PyRosetta pose_from_pdb.",
    )
    parser.add_argument(
        "--keep-cleaned-pdbs",
        action="store_true",
        help="Keep temporary cleaned PDB files generated before PyRosetta loading.",
    )
    parser.add_argument(
        "--cleaned-pdb-dir",
        default=None,
        help="Optional directory for cleaned PDB files when --keep-cleaned-pdbs is enabled.",
    )
    parser.add_argument(
        "--rmsd-missing-atom-policy",
        choices=("raise", "skip_residue", "penalize"),
        default="penalize",
        help=(
            "How StepRewardCalculator handles residues missing local-RMSD atoms. "
            "'penalize' is recommended for long mixed-quality training runs."
        ),
    )
    parser.add_argument("--rmsd-missing-penalty", type=float, default=5.0)
    parser.add_argument("--min-rmsd-atoms", type=int, default=3)
    parser.add_argument(
        "--step-reward-scale",
        type=float,
        default=1.0,
        help="Scale applied after bounded 0-1 step reward normalization.",
    )
    parser.add_argument(
        "--terminal-reward-scale",
        type=float,
        default=1.0,
        help="Scale applied to the structure-based terminal reward.",
    )
    parser.add_argument(
        "--no-terminal-reward",
        action="store_true",
        help="Disable the hbond-topology mechanical-property terminal reward.",
    )
    parser.add_argument(
        "--terminal-reward-artifact",
        default="params/hbond_random_forest.joblib",
        help="Path to the hbond random-forest artifact used by terminal reward.",
    )
    parser.add_argument(
        "--terminal-predicted-pdb-dir",
        default=None,
        help=(
            "Optional directory containing sequence-predicted PDBs. When a matching "
            "PDB is found, terminal reward averages PyRosetta terminal pose and "
            "predicted structure 1:1."
        ),
    )
    parser.add_argument(
        "--observation-encoder",
        choices=("default", "esm2"),
        default="default",
        help="Use default one-hot environment observations or per-residue ESM2 embeddings.",
    )
    parser.add_argument(
        "--esm2-device",
        default="auto",
        help="Device for ESM2 observation encoding when --observation-encoder esm2.",
    )
    parser.add_argument(
        "--esm-model-dir",
        default=None,
        help=(
            "Optional directory containing local fair-esm checkpoints. "
            "For embedding_dim=1280 it should contain esm2_t33_650M_UR50D.pt."
        ),
    )
    parser.add_argument(
        "--esm2-mutable-only",
        action="store_true",
        default=True,
        help="Encode only mutable positions with ESM2 instead of the full sequence.",
    )
    parser.add_argument(
        "--esm2-full-sequence",
        action="store_false",
        dest="esm2_mutable_only",
        help=(
            "Encode the full sequence with ESM2. This requires the full sequence "
            "length to match the environment action positions."
        ),
    )
    parser.add_argument("--no-repack", action="store_true")
    parser.add_argument("--no-minimize", action="store_true")
    parser.add_argument("--minimize-backbone", action="store_true")
    parser.add_argument("--prevent-revisit-positions", action="store_true")
    parser.add_argument("--raise-on-update-error", action="store_true", default=True)
    parser.add_argument(
        "--continue-on-update-error",
        action="store_false",
        dest="raise_on_update_error",
        help="Rollback failed mutation updates and keep training.",
    )

    parser.add_argument("--hidden-dims", default="256,256")
    parser.add_argument(
        "--embedding-dim",
        type=int,
        choices=(1280, 2560, 5120),
        default=1280,
        help="Protein-language-model encoding dimension used by the DDQN Q head.",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Full replay/optimizer batch processed by one backward pass. "
            "When supplied, this sets --micro-batch-size to the same value and "
            "--gradient-accumulation-steps to 1. This is the recommended option "
            "for multi-GPU training; it is distinct from --train-batch-size."
        ),
    )
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--replay-warmup-size", type=int, default=1_000)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument(
        "--train-frequency",
        type=int,
        default=1,
        help=(
            "Collect this many environment transitions between replay training "
            "events. A value of 1 preserves the original update-after-every-step behavior."
        ),
    )
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=1,
        help="Number of replay optimizer steps executed at each training event.",
    )
    parser.add_argument("--target-sync-interval", type=int, default=250)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=50_000)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument(
        "--amp-dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
        help=(
            "Autocast dtype used with --use-amp. bfloat16 is recommended on "
            "RTX 4090/Ampere-or-newer GPUs because it avoids FP16 loss-scale overflow."
        ),
    )
    parser.add_argument(
        "--amp-max-retries",
        type=int,
        default=4,
        help="Maximum automatic reduced-loss-scale retries after an FP16 gradient overflow.",
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Save the small agent checkpoint every N completed episodes; 0 disables it.",
    )
    parser.add_argument(
        "--replay-checkpoint-every",
        type=int,
        default=0,
        help=(
            "Save a resumable agent+replay bundle every N completed episodes. "
            "This is expensive for ESM2 replay, so 0 disables periodic replay snapshots."
        ),
    )
    parser.add_argument(
        "--resume-checkpoint-dir",
        default=None,
        help=(
            "Resume agent and replay from a directory containing agent.pt, "
            "replay_buffer.npz, and preferably training_state.json."
        ),
    )
    parser.add_argument(
        "--resume-next-episode",
        type=int,
        default=None,
        help=(
            "Manual next episode for legacy checkpoint directories without "
            "training_state.json. Prefer the recorded value when available."
        ),
    )
    parser.add_argument(
        "--no-save-final-replay",
        action="store_false",
        dest="save_final_replay",
        help="Skip the expensive final replay snapshot; final agent weights are still saved.",
    )
    parser.set_defaults(save_final_replay=True)
    parser.add_argument("--save-candidates", action="store_true", default=True)
    parser.add_argument(
        "--no-save-candidates",
        action="store_false",
        dest="save_candidates",
    )
    parser.add_argument("--plot-every-episodes", type=int, default=10)
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument("--enable-tensorboard", action="store_true")
    parser.add_argument("--no-resume-logs", action="store_true")
    parser.add_argument("--validate-every", type=int, default=25)
    parser.add_argument("--validation-episodes", type=int, default=5)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="stdout logging verbosity.",
    )
    parser.add_argument(
        "--log-every-steps",
        type=int,
        default=1,
        help="Emit a training-loop progress log every N environment steps.",
    )

    return parser.parse_args()


def configure_stdout_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def parse_int_tuple(value: Optional[str], *, name: str) -> Optional[Tuple[int, ...]]:
    if value is None or value.strip() == "":
        return None
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated integer list.") from exc
    if not parsed:
        return None
    return parsed


def parse_hidden_dims(value: str) -> Tuple[int, ...]:
    parsed = parse_int_tuple(value, name="hidden_dims")
    if parsed is None:
        raise ValueError("hidden_dims must contain at least one layer size.")
    return parsed


def parse_gpu_ids(value: Optional[str]) -> Tuple[int, ...]:
    if value is None or value.strip() == "":
        return tuple(range(torch.cuda.device_count()))
    parsed = parse_int_tuple(value, name="gpu_ids")
    return tuple() if parsed is None else parsed


def build_env(args: argparse.Namespace) -> MechanicalProteinEnv:
    observation_encoder = None
    if args.observation_encoder == "esm2":
        from model.encoding_module import ESM2SequenceEncoder

        LOGGER.info(
            "Building ESM2 observation encoder embedding_dim=%s device=%s output=per_residue mutable_only=%s",
            args.embedding_dim,
            args.esm2_device,
            args.esm2_mutable_only,
        )
        observation_encoder = ESM2SequenceEncoder(
            embedding_dim=args.embedding_dim,
            device=args.esm2_device,
            mutable_only=args.esm2_mutable_only,
            model_dir=args.esm_model_dir,
        )

    terminal_reward_calculator = None
    if not args.no_terminal_reward:
        from model.reward_module.terminal_reward import (
            EqualWeightDualStructureTerminalRewardCalculator,
        )

        terminal_reward_calculator = EqualWeightDualStructureTerminalRewardCalculator(
            artifact_path=args.terminal_reward_artifact,
            predicted_pdb_dir=args.terminal_predicted_pdb_dir,
        )
        LOGGER.info(
            "Enabled hbond topology terminal reward artifact=%s predicted_pdb_dir=%s "
            "structure_weights=pyrosetta_terminal:1,predicted_structure:1 "
            "objective_weights=strength:1,toughness:1",
            args.terminal_reward_artifact,
            args.terminal_predicted_pdb_dir,
        )
    else:
        LOGGER.info("Terminal reward disabled by --no-terminal-reward")

    LOGGER.info(
        "Building MechanicalProteinEnv max_steps=%s mutable_positions=%s "
        "repack=%s minimize=%s minimize_backbone=%s local_repack_radius=%s observation_encoder=%s "
        "step_reward_scale=%s terminal_reward_scale=%s",
        args.max_steps,
        args.mutable_positions or "all canonical residues",
        not args.no_repack,
        not args.no_minimize,
        args.minimize_backbone,
        args.local_repack_radius,
        args.observation_encoder,
        args.step_reward_scale,
        args.terminal_reward_scale,
    )
    return MechanicalProteinEnv(
        max_steps=args.max_steps,
        mutable_positions=parse_int_tuple(
            args.mutable_positions,
            name="mutable_positions",
        ),
        local_repack_radius=args.local_repack_radius,
        perform_repack=not args.no_repack,
        perform_minimize=not args.no_minimize,
        minimize_backbone=args.minimize_backbone,
        prevent_revisit_positions=args.prevent_revisit_positions,
        raise_on_update_error=args.raise_on_update_error,
        step_reward_scale=args.step_reward_scale,
        terminal_reward_scale=args.terminal_reward_scale,
        terminal_reward_calculator=terminal_reward_calculator,
        observation_encoder=observation_encoder,
        step_reward_kwargs={
            "rmsd_missing_atom_policy": args.rmsd_missing_atom_policy,
            "rmsd_missing_penalty": args.rmsd_missing_penalty,
            "min_rmsd_atoms": args.min_rmsd_atoms,
        },
        pyrosetta_init_options=args.pyrosetta_options,
        clean_pdb_before_load=not args.no_clean_pdb_before_load,
        load_max_missing_backbone_fraction=args.max_missing_backbone_fraction,
        keep_cleaned_pdbs=args.keep_cleaned_pdbs,
        cleaned_pdb_dir=args.cleaned_pdb_dir,
        seed=args.seed,
    )


def build_dataset(args: argparse.Namespace) -> ProteinStructureDataset:
    return ProteinStructureDataset.from_folder(
        args.pdb_dir,
        train_index_path=args.train_index,
        val_index_path=args.val_index,
        val_fraction=args.val_fraction,
        seed=args.dataset_seed,
        recreate_indices=args.recreate_splits,
        min_protein_residues=args.min_protein_residues,
        max_missing_backbone_fraction=args.max_missing_backbone_fraction,
    )


def build_agent_config(args: argparse.Namespace, *, device: str) -> DDQNConfig:
    return DDQNConfig(
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        embedding_dim=args.embedding_dim,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        replay_warmup_size=args.replay_warmup_size,
        target_sync_interval=args.target_sync_interval,
        max_grad_norm=args.max_grad_norm,
        huber_beta=args.huber_beta,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        device=device,
        use_amp=args.use_amp,
        amp_dtype=args.amp_dtype,
        amp_max_retries=args.amp_max_retries,
        seed=args.seed,
    )


def apply_full_batch_shortcut(args: argparse.Namespace) -> None:
    """Resolve --batch-size to one full batch with no gradient accumulation."""

    batch_size = getattr(args, "batch_size", None)
    if batch_size is None:
        return
    if int(batch_size) <= 0:
        raise ValueError("--batch-size must be a positive integer.")

    args.micro_batch_size = int(batch_size)
    args.gradient_accumulation_steps = 1
    LOGGER.info(
        "Configured full-batch optimizer mode batch_size=%s "
        "micro_batch_size=%s gradient_accumulation_steps=1",
        batch_size,
        args.micro_batch_size,
    )


def configure_training_mode(args: argparse.Namespace) -> Tuple[str, Tuple[int, ...]]:
    if args.mode == "single":
        LOGGER.info("Configured single-device training mode device=%s", args.device)
        return args.device, tuple()

    if not torch.cuda.is_available():
        raise RuntimeError("Multi-GPU mode requires CUDA, but CUDA is unavailable.")

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if len(gpu_ids) < 2:
        raise ValueError("Multi-GPU mode requires at least two GPU ids.")

    visible_count = torch.cuda.device_count()
    invalid = [gpu_id for gpu_id in gpu_ids if gpu_id < 0 or gpu_id >= visible_count]
    if invalid:
        raise ValueError(
            f"Invalid GPU ids {invalid}; this process sees {visible_count} CUDA devices."
        )

    torch.cuda.set_device(gpu_ids[0])
    LOGGER.info(
        "Configured single-machine multi-GPU training mode primary_device=cuda:%s gpu_ids=%s",
        gpu_ids[0],
        list(gpu_ids),
    )
    return f"cuda:{gpu_ids[0]}", gpu_ids


def data_parallel_batch_plan(
    config: DDQNConfig,
    gpu_ids: Sequence[int],
) -> dict[str, int]:
    """Describe how each backward batch is split across DataParallel devices."""

    gpu_count = len(gpu_ids)
    batch_per_backward = int(config.micro_batch_size)
    if gpu_count and batch_per_backward < gpu_count:
        raise ValueError(
            "Multi-GPU batch per backward pass must be at least the number of GPUs: "
            f"batch_per_backward={batch_per_backward}, gpu_count={gpu_count}."
        )

    divisor = gpu_count if gpu_count else 1
    return {
        "gpu_count": gpu_count,
        "effective_batch_size": int(config.effective_batch_size),
        "batch_per_backward": batch_per_backward,
        "gradient_accumulation_steps": int(config.gradient_accumulation_steps),
        "per_gpu_batch_size": batch_per_backward // divisor,
        "uneven_batch_remainder": batch_per_backward % divisor,
    }


def enable_data_parallel(agent: DDQNAgent, gpu_ids: Sequence[int]) -> None:
    if not gpu_ids:
        LOGGER.info("DataParallel disabled; using agent device=%s", agent.device)
        return

    plan = data_parallel_batch_plan(agent.config, gpu_ids)
    if plan["uneven_batch_remainder"]:
        LOGGER.warning(
            "DataParallel batch is not evenly divisible by GPU count; "
            "batch_per_backward=%s gpu_count=%s remainder=%s",
            plan["batch_per_backward"],
            plan["gpu_count"],
            plan["uneven_batch_remainder"],
        )
    LOGGER.info(
        "Wrapping online and target networks with DataParallel gpu_ids=%s "
        "effective_batch_size=%s batch_per_backward=%s per_gpu_batch_size=%s "
        "gradient_accumulation_steps=%s",
        list(gpu_ids),
        plan["effective_batch_size"],
        plan["batch_per_backward"],
        plan["per_gpu_batch_size"],
        plan["gradient_accumulation_steps"],
    )
    agent.online_network = torch.nn.DataParallel(
        agent.online_network,
        device_ids=list(gpu_ids),
        output_device=gpu_ids[0],
    )
    agent.target_network = torch.nn.DataParallel(
        agent.target_network,
        device_ids=list(gpu_ids),
        output_device=gpu_ids[0],
    )
    agent.target_network.eval()


def unwrapped_state_dict(module: torch.nn.Module) -> dict:
    if isinstance(module, torch.nn.DataParallel):
        return module.module.state_dict()
    return module.state_dict()


def save_agent_checkpoint(agent: DDQNAgent, path: Path) -> None:
    started = time.perf_counter()
    LOGGER.info("Saving agent checkpoint path=%s", path)
    payload = agent.state_dict()
    payload["online_network"] = unwrapped_state_dict(agent.online_network)
    payload["target_network"] = unwrapped_state_dict(agent.target_network)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    LOGGER.info(
        "Saved agent checkpoint path=%s bytes=%s elapsed_sec=%.3f",
        path,
        path.stat().st_size,
        time.perf_counter() - started,
    )


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def save_resume_checkpoint(
    *,
    agent: DDQNAgent,
    replay_buffer: ReplayBuffer,
    checkpoint_dir: Path,
    next_episode: int,
    args: argparse.Namespace,
) -> None:
    """Save an episode-boundary agent/replay pair and consistency metadata."""

    started = time.perf_counter()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    replay_path = checkpoint_dir / RESUME_REPLAY_FILENAME
    agent_path = checkpoint_dir / RESUME_AGENT_FILENAME
    state_path = checkpoint_dir / RESUME_STATE_FILENAME

    LOGGER.info(
        "Resume checkpoint started dir=%s next_episode=%s replay_size=%s",
        checkpoint_dir,
        next_episode,
        len(replay_buffer),
    )
    replay_buffer.save(replay_path)
    save_agent_checkpoint(agent, agent_path)
    state = {
        "version": RESUME_STATE_VERSION,
        "next_episode": int(next_episode),
        "agent_environment_steps": int(agent.environment_steps),
        "agent_optimization_steps": int(agent.optimization_steps),
        "replay_size": int(len(replay_buffer)),
        "replay_capacity": int(replay_buffer.capacity),
        "replay_position": int(replay_buffer.position),
        "state_shape": list(agent.state_shape),
        "action_dim": int(agent.action_dim),
        "schedule": {
            "dataset_seed": int(args.dataset_seed),
            "train_batch_size": int(args.train_batch_size),
            "no_shuffle_train": bool(args.no_shuffle_train),
            "pdb_dir": str(args.pdb_dir),
            "train_index": None if args.train_index is None else str(args.train_index),
        },
    }
    _write_json_atomic(state_path, state)
    LOGGER.info(
        "Resume checkpoint complete dir=%s next_episode=%s elapsed_sec=%.3f",
        checkpoint_dir,
        next_episode,
        time.perf_counter() - started,
    )


def load_resume_checkpoint(
    *,
    checkpoint_dir: Path,
    agent_config: DDQNConfig,
    expected_state_shape: Sequence[int],
    expected_action_dim: int,
    args: argparse.Namespace,
) -> tuple[DDQNAgent, ReplayBuffer, int]:
    """Restore an agent/replay pair and return the next dataset episode index."""

    started = time.perf_counter()
    agent_path = checkpoint_dir / RESUME_AGENT_FILENAME
    replay_path = checkpoint_dir / RESUME_REPLAY_FILENAME
    state_path = checkpoint_dir / RESUME_STATE_FILENAME
    if not agent_path.is_file():
        raise FileNotFoundError(f"Resume agent checkpoint not found: {agent_path}")
    if not replay_path.is_file():
        raise FileNotFoundError(f"Resume replay checkpoint not found: {replay_path}")

    agent = DDQNAgent.from_checkpoint(
        agent_path,
        map_location=agent_config.device,
        config_override=agent_config,
    )
    replay_buffer = ReplayBuffer.load(replay_path)

    expected_shape = tuple(int(value) for value in expected_state_shape)
    if tuple(agent.state_shape) != expected_shape:
        raise ValueError(
            "Resume agent state shape is incompatible with the current environment: "
            f"{agent.state_shape} vs {expected_shape}."
        )
    if int(agent.action_dim) != int(expected_action_dim):
        raise ValueError(
            "Resume agent action dimension is incompatible with the current environment: "
            f"{agent.action_dim} vs {expected_action_dim}."
        )
    if tuple(replay_buffer.state_shape) != expected_shape:
        raise ValueError(
            "Resume replay state shape is incompatible with the current environment: "
            f"{replay_buffer.state_shape} vs {expected_shape}."
        )
    if int(replay_buffer.action_dim) != int(expected_action_dim):
        raise ValueError(
            "Resume replay action dimension is incompatible with the current environment: "
            f"{replay_buffer.action_dim} vs {expected_action_dim}."
        )

    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if int(state.get("version", -1)) != RESUME_STATE_VERSION:
            raise ValueError(f"Unsupported training-state version in {state_path}.")
        next_episode = int(state["next_episode"])
        if args.resume_next_episode is not None and int(args.resume_next_episode) != next_episode:
            raise ValueError(
                "--resume-next-episode disagrees with training_state.json: "
                f"{args.resume_next_episode} vs {next_episode}."
            )

        consistency_checks = {
            "agent_environment_steps": int(agent.environment_steps),
            "agent_optimization_steps": int(agent.optimization_steps),
            "replay_size": int(len(replay_buffer)),
            "replay_capacity": int(replay_buffer.capacity),
            "replay_position": int(replay_buffer.position),
        }
        for key, actual in consistency_checks.items():
            if int(state[key]) != actual:
                raise ValueError(
                    f"Resume checkpoint is inconsistent for {key}: "
                    f"metadata={state[key]} actual={actual}."
                )

        expected_schedule = {
            "dataset_seed": int(args.dataset_seed),
            "train_batch_size": int(args.train_batch_size),
            "no_shuffle_train": bool(args.no_shuffle_train),
            "pdb_dir": str(args.pdb_dir),
            "train_index": None if args.train_index is None else str(args.train_index),
        }
        if state.get("schedule") != expected_schedule:
            raise ValueError(
                "Resume dataset schedule differs from the saved training state. "
                f"saved={state.get('schedule')} current={expected_schedule}."
            )
    else:
        if args.resume_next_episode is None:
            raise ValueError(
                f"Legacy resume directory {checkpoint_dir} has no {RESUME_STATE_FILENAME}; "
                "supply --resume-next-episode explicitly."
            )
        next_episode = int(args.resume_next_episode)
        LOGGER.warning(
            "Resuming legacy checkpoint without consistency metadata dir=%s next_episode=%s",
            checkpoint_dir,
            next_episode,
        )

    if next_episode < 0:
        raise ValueError("Resume next episode must be >= 0.")
    LOGGER.info(
        "Resume checkpoint loaded dir=%s next_episode=%s environment_steps=%s "
        "optimization_steps=%s replay_size=%s replay_capacity=%s elapsed_sec=%.3f",
        checkpoint_dir,
        next_episode,
        agent.environment_steps,
        agent.optimization_steps,
        len(replay_buffer),
        replay_buffer.capacity,
        time.perf_counter() - started,
    )
    return agent, replay_buffer, next_episode


def write_run_config(
    args: argparse.Namespace,
    output_dir: Path,
    agent_config: DDQNConfig,
    *,
    state_shape: Iterable[int],
    action_dim: int,
    gpu_ids: Sequence[int],
) -> None:
    payload = {
        "args": vars(args),
        "agent_config": asdict(agent_config),
        "state_shape": list(state_shape),
        "action_dim": int(action_dim),
        "gpu_ids": list(gpu_ids),
        "parallel_batch_plan": data_parallel_batch_plan(agent_config, gpu_ids),
        "update_schedule": update_schedule_summary(
            batch_size=agent_config.effective_batch_size,
            train_frequency=args.train_frequency,
            gradient_steps=args.gradient_steps,
        ),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Wrote run configuration path=%s", output_dir / "run_config.json")


def run_validation(
    *,
    agent: DDQNAgent,
    env: MechanicalProteinEnv,
    dataset: ProteinStructureDataset,
    episode: int,
    limit: int,
) -> list[dict]:
    records: list[dict] = []
    for validation_index, pdb_path in enumerate(dataset.validation_paths(limit=limit)):
        state, info = env.reset(pdb_path=str(pdb_path))
        total_reward = 0.0
        steps = 0
        while True:
            action = agent.select_action(
                state,
                action_mask=info["action_mask"],
                evaluate=True,
                advance_step=False,
            )
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            if terminated or truncated:
                break
        record = {
            "episode": episode,
            "validation_index": validation_index,
            "pdb_path": str(pdb_path),
            "total_reward": total_reward,
            "steps": steps,
            "sequence": info.get("sequence"),
        }
        records.append(record)
        LOGGER.debug("Validation episode complete record=%s", record)
    return records


def validate_training_schedule_args(args: argparse.Namespace) -> None:
    if args.epochs is not None and int(args.epochs) <= 0:
        raise ValueError("--epochs must be a positive integer when supplied.")
    if args.episodes_per_epoch is not None and int(args.episodes_per_epoch) <= 0:
        raise ValueError("--episodes-per-epoch must be a positive integer when supplied.")
    if int(args.train_batch_size) <= 0:
        raise ValueError("--train-batch-size must be a positive integer.")
    if args.epochs is None and int(args.episodes) <= 0:
        raise ValueError("--episodes must be a positive integer in legacy episode mode.")
    if int(args.train_frequency) <= 0:
        raise ValueError("--train-frequency must be a positive integer.")
    if int(args.gradient_steps) <= 0:
        raise ValueError("--gradient-steps must be a positive integer.")
    if int(args.checkpoint_every) < 0:
        raise ValueError("--checkpoint-every must be >= 0.")
    if int(args.replay_checkpoint_every) < 0:
        raise ValueError("--replay-checkpoint-every must be >= 0.")
    if args.resume_next_episode is not None and int(args.resume_next_episode) < 0:
        raise ValueError("--resume-next-episode must be >= 0.")
    if args.resume_next_episode is not None and args.resume_checkpoint_dir is None:
        raise ValueError("--resume-next-episode requires --resume-checkpoint-dir.")


def update_schedule_summary(
    *,
    batch_size: int,
    train_frequency: int,
    gradient_steps: int,
) -> dict[str, float | int]:
    """Return the effective replay-update ratios for one collection schedule."""

    if int(batch_size) <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if int(train_frequency) <= 0:
        raise ValueError("train_frequency must be a positive integer.")
    if int(gradient_steps) <= 0:
        raise ValueError("gradient_steps must be a positive integer.")

    return {
        "train_frequency": int(train_frequency),
        "gradient_steps": int(gradient_steps),
        "gradient_updates_per_transition": float(gradient_steps / train_frequency),
        "replay_samples_per_transition": float(
            batch_size * gradient_steps / train_frequency
        ),
    }


def should_run_optimizer_event(
    completed_environment_steps: int,
    train_frequency: int,
) -> bool:
    """Return whether a replay-training event is due after the latest step."""

    if int(completed_environment_steps) <= 0:
        raise ValueError("completed_environment_steps must be a positive integer.")
    if int(train_frequency) <= 0:
        raise ValueError("train_frequency must be a positive integer.")
    return int(completed_environment_steps) % int(train_frequency) == 0


def run_replay_optimization_event(
    *,
    agent: DDQNAgent,
    replay_buffer: ReplayBuffer,
    gradient_steps: int,
) -> list[OptimizationResult]:
    """Execute up to ``gradient_steps`` updates from the unchanged replay buffer."""

    if int(gradient_steps) <= 0:
        raise ValueError("gradient_steps must be a positive integer.")

    results: list[OptimizationResult] = []
    for _ in range(int(gradient_steps)):
        result = agent.optimize_from_replay_buffer(replay_buffer)
        if result is None:
            break
        results.append(result)
    return results


def planned_episode_count(args: argparse.Namespace, dataset: ProteinStructureDataset) -> int:
    if args.epochs is None:
        return int(args.episodes)
    episodes_per_epoch = (
        len(dataset.train_paths)
        if args.episodes_per_epoch is None
        else int(args.episodes_per_epoch)
    )
    return int(args.epochs) * episodes_per_epoch


def iter_training_episode_paths(
    *,
    args: argparse.Namespace,
    dataset: ProteinStructureDataset,
    rng: np.random.Generator,
) -> Iterable[tuple[int, int, int, Path]]:
    """Yield (epoch, batch_index, batch_item_index, pdb_path)."""

    if args.epochs is None:
        for episode in range(int(args.episodes)):
            yield 0, episode, 0, dataset.sample_train_path(rng)
        return

    episodes_per_epoch = (
        len(dataset.train_paths)
        if args.episodes_per_epoch is None
        else int(args.episodes_per_epoch)
    )
    for epoch in range(int(args.epochs)):
        for batch_index, batch_paths in enumerate(
            dataset.iter_train_batches(
                rng,
                batch_size=args.train_batch_size,
                episodes_per_epoch=episodes_per_epoch,
                shuffle=not args.no_shuffle_train,
            )
        ):
            LOGGER.debug(
                "Dataset batch ready epoch=%s/%s batch=%s batch_size=%s paths=%s",
                epoch + 1,
                int(args.epochs),
                batch_index,
                len(batch_paths),
                [str(path) for path in batch_paths],
            )
            for batch_item_index, pdb_path in enumerate(batch_paths):
                yield epoch, batch_index, batch_item_index, pdb_path


def train(args: argparse.Namespace) -> None:
    if args.log_every_steps <= 0:
        raise ValueError("--log-every-steps must be a positive integer.")
    apply_full_batch_shortcut(args)
    validate_training_schedule_args(args)

    run_started = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Training entry started output_dir=%s args=%s", output_dir, vars(args))
    LOGGER.info(
        "Torch status version=%s cuda_built=%s cuda_available=%s cuda_device_count=%s",
        torch.__version__,
        torch.version.cuda,
        torch.cuda.is_available(),
        torch.cuda.device_count(),
    )
    if torch.cuda.is_available():
        for device_index in range(torch.cuda.device_count()):
            LOGGER.info("CUDA device %s: %s", device_index, torch.cuda.get_device_name(device_index))

    dataset = build_dataset(args)
    dataset_rng = np.random.default_rng(args.dataset_seed)
    total_planned_episodes = planned_episode_count(args, dataset)
    LOGGER.info(
        "Dataset ready train_size=%s val_size=%s train_index=%s val_index=%s "
        "min_protein_residues=%s max_missing_backbone_fraction=%.4f "
        "training_schedule=%s planned_episodes=%s epochs=%s episodes_per_epoch=%s train_batch_size=%s",
        len(dataset.train_paths),
        len(dataset.val_paths),
        dataset.train_index_path,
        dataset.val_index_path,
        dataset.min_protein_residues,
        dataset.max_missing_backbone_fraction,
        "legacy_episodes" if args.epochs is None else "dataset_epochs",
        total_planned_episodes,
        args.epochs,
        args.episodes_per_epoch if args.episodes_per_epoch is not None else len(dataset.train_paths),
        args.train_batch_size,
    )

    device, gpu_ids = configure_training_mode(args)
    env = build_env(args)
    initial_pdb_path = dataset.train_paths[0]
    LOGGER.info("Resetting environment with seed=%s initial_pdb=%s", args.seed, initial_pdb_path)
    state, info = env.reset(seed=args.seed, pdb_path=str(initial_pdb_path))
    LOGGER.info(
        "Initial environment state ready state_shape=%s action_dim=%s valid_actions=%s sequence_len=%s",
        state.shape,
        env.action_space.n,
        info.get("valid_action_count"),
        len(str(info.get("sequence", ""))),
    )

    agent_config = build_agent_config(args, device=device)
    LOGGER.info("Building DDQNAgent config=%s", asdict(agent_config))
    if args.resume_checkpoint_dir is None:
        agent = DDQNAgent(
            state_shape=state.shape,
            action_dim=env.action_space.n,
            config=agent_config,
        )
        replay_buffer = ReplayBuffer(
            capacity=args.replay_capacity,
            state_shape=state.shape,
            action_dim=env.action_space.n,
            seed=args.seed,
            store_action_masks=True,
            variable_length=args.observation_encoder == "esm2",
        )
        start_episode = 0
    else:
        agent, replay_buffer, start_episode = load_resume_checkpoint(
            checkpoint_dir=Path(args.resume_checkpoint_dir).expanduser().resolve(),
            agent_config=agent_config,
            expected_state_shape=state.shape,
            expected_action_dim=env.action_space.n,
            args=args,
        )
    enable_data_parallel(agent, gpu_ids)

    if start_episode > total_planned_episodes:
        raise ValueError(
            "Resume episode exceeds the configured training schedule: "
            f"next_episode={start_episode}, planned_episodes={total_planned_episodes}."
        )
    LOGGER.info(
        "ReplayBuffer ready capacity=%s size=%s effective_batch_size=%s warmup=%s "
        "variable_length=%s resumed=%s start_episode=%s",
        replay_buffer.capacity,
        len(replay_buffer),
        agent_config.effective_batch_size,
        agent_config.replay_warmup_size,
        replay_buffer.variable_length,
        args.resume_checkpoint_dir is not None,
        start_episode,
    )
    update_schedule = update_schedule_summary(
        batch_size=agent_config.effective_batch_size,
        train_frequency=args.train_frequency,
        gradient_steps=args.gradient_steps,
    )
    LOGGER.info("Replay optimization schedule=%s", update_schedule)

    logger = TrainingLogger(
        TrainingLoggerConfig(
            output_dir=output_dir,
            rolling_window=args.rolling_window,
            plot_every_episodes=args.plot_every_episodes,
            save_step_records=True,
            save_optimization_records=True,
            enable_tensorboard=args.enable_tensorboard,
            resume=not args.no_resume_logs,
            gradient_clip_threshold=agent_config.max_grad_norm,
        )
    )
    LOGGER.info(
        "TrainingLogger ready logs_dir=%s tensorboard=%s resume_logs=%s",
        output_dir / "logs",
        args.enable_tensorboard,
        not args.no_resume_logs,
    )

    write_run_config(
        args,
        output_dir,
        agent_config,
        state_shape=state.shape,
        action_dim=env.action_space.n,
        gpu_ids=gpu_ids,
    )

    total_environment_steps = int(agent.environment_steps)

    try:
        episode_paths = iter_training_episode_paths(
            args=args,
            dataset=dataset,
            rng=dataset_rng,
        )
        for episode, (epoch, batch_index, batch_item_index, episode_pdb_path) in enumerate(
            islice(episode_paths, start_episode, None),
            start=start_episode,
        ):
            episode_started = time.perf_counter()
            LOGGER.debug(
                "Episode %s/%s starting epoch=%s batch=%s batch_item=%s",
                episode + 1,
                total_planned_episodes,
                epoch,
                batch_index,
                batch_item_index,
            )
            state, info = env.reset(pdb_path=str(episode_pdb_path))
            LOGGER.debug(
                "Episode %s reset epoch=%s batch=%s batch_item=%s pdb_path=%s "
                "valid_actions=%s accepted_mutations=%s",
                episode,
                epoch,
                batch_index,
                batch_item_index,
                episode_pdb_path,
                info.get("valid_action_count"),
                info.get("accepted_mutation_count"),
            )
            episode_reward = 0.0
            episode_steps = 0

            while True:
                step_started = time.perf_counter()
                action_mask = info["action_mask"]
                valid_action_count = int(action_mask.sum())
                LOGGER.debug(
                    "Episode %s step %s global_step=%s selecting_action epsilon=%.6f valid_actions=%s",
                    episode,
                    episode_steps,
                    total_environment_steps,
                    agent.epsilon,
                    valid_action_count,
                )
                action = agent.select_action(state, action_mask=action_mask)
                LOGGER.debug("Selected action=%s for episode=%s step=%s", action, episode, episode_steps)

                next_state, reward, terminated, truncated, next_info = env.step(action)
                decoded_action = next_info.get("decoded_action", {})
                LOGGER.debug(
                    "Environment step finished episode=%s step=%s action=%s decoded=%s "
                    "reward=%.6f step_reward=%.6f terminal_reward=%.6f accepted=%s reason=%s "
                    "terminated=%s truncated=%s truncation_reason=%s next_valid_actions=%s elapsed_sec=%.3f",
                    episode,
                    episode_steps,
                    action,
                    decoded_action,
                    float(reward),
                    float(next_info.get("step_reward", 0.0)),
                    float(next_info.get("terminal_reward", 0.0)),
                    next_info.get("accepted"),
                    next_info.get("reason"),
                    terminated,
                    truncated,
                    next_info.get("truncation_reason"),
                    next_info.get("valid_action_count"),
                    time.perf_counter() - step_started,
                )

                replay_buffer.add(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    terminated=terminated,
                    truncated=truncated,
                    action_mask=action_mask,
                    next_action_mask=next_info["action_mask"],
                )
                LOGGER.debug(
                    "ReplayBuffer add complete size=%s capacity=%s position=%s",
                    len(replay_buffer),
                    replay_buffer.capacity,
                    replay_buffer.position,
                )

                completed_environment_steps = total_environment_steps + 1
                if should_run_optimizer_event(
                    completed_environment_steps,
                    args.train_frequency,
                ):
                    optimizer_event = completed_environment_steps // args.train_frequency
                    LOGGER.debug(
                        "Replay optimization event started event=%s "
                        "completed_environment_steps=%s gradient_steps=%s",
                        optimizer_event,
                        completed_environment_steps,
                        args.gradient_steps,
                    )
                    optimization_results = run_replay_optimization_event(
                        agent=agent,
                        replay_buffer=replay_buffer,
                        gradient_steps=args.gradient_steps,
                    )
                    if optimization_results:
                        for gradient_step_in_event, optimization_result in enumerate(
                            optimization_results,
                            start=1,
                        ):
                            LOGGER.debug(
                                "Optimization step complete event=%s "
                                "gradient_step=%s/%s step=%s loss=%.8f mean_q=%.8f "
                                "mean_target=%.8f mean_abs_td=%.8f grad_norm=%.8f "
                                "target_synced=%s amp_retries=%s amp_scale=%s",
                                optimizer_event,
                                gradient_step_in_event,
                                args.gradient_steps,
                                optimization_result.optimization_step,
                                optimization_result.loss,
                                optimization_result.mean_q_value,
                                optimization_result.mean_target_q_value,
                                optimization_result.mean_absolute_td_error,
                                optimization_result.grad_norm,
                                optimization_result.target_synced,
                                optimization_result.amp_retries,
                                optimization_result.amp_scale,
                            )
                            logger.log_optimization(
                                optimization_result,
                                global_step=total_environment_steps,
                                extra={
                                    "optimizer_event": optimizer_event,
                                    "completed_environment_steps": completed_environment_steps,
                                    "gradient_step_in_event": gradient_step_in_event,
                                    "gradient_steps_requested": args.gradient_steps,
                                    "train_frequency": args.train_frequency,
                                },
                            )
                    else:
                        required_size = max(
                            agent_config.replay_warmup_size,
                            agent_config.effective_batch_size,
                        )
                        LOGGER.debug(
                            "Replay optimization event skipped for warmup "
                            "event=%s replay_buffer_size=%s required_size=%s",
                            optimizer_event,
                            len(replay_buffer),
                            required_size,
                        )
                else:
                    LOGGER.debug(
                        "Replay optimization deferred completed_environment_steps=%s "
                        "next_event_in=%s",
                        completed_environment_steps,
                        args.train_frequency
                        - (completed_environment_steps % args.train_frequency),
                    )

                logger.log_step(
                    episode=episode,
                    episode_step=episode_steps,
                    global_step=total_environment_steps,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=next_info,
                )

                state = next_state
                info = next_info
                episode_reward += float(reward)
                episode_steps += 1
                total_environment_steps += 1

                if total_environment_steps % args.log_every_steps == 0:
                    LOGGER.info(
                        "Progress global_step=%s episode=%s episode_step=%s "
                        "episode_reward=%.6f replay_size=%s optim_steps=%s",
                        total_environment_steps,
                        episode,
                        episode_steps,
                        episode_reward,
                        len(replay_buffer),
                        agent.optimization_steps,
                    )

                if terminated or truncated:
                    break

            candidate_path = None
            if args.save_candidates:
                candidate_path = output_dir / "candidates" / f"episode_{episode:04d}.pdb"
                candidate_path.parent.mkdir(parents=True, exist_ok=True)
                LOGGER.debug("Saving candidate PDB episode=%s path=%s", episode, candidate_path)
                env.save_current_pose(candidate_path)
                LOGGER.debug("Saved candidate PDB episode=%s path=%s", episode, candidate_path)

            logger.end_episode(
                episode=episode,
                total_reward=episode_reward,
                episode_steps=episode_steps,
                epsilon=agent.epsilon,
                optimization_steps=agent.optimization_steps,
                info=info,
                extra={
                    "candidate_pdb": None if candidate_path is None else str(candidate_path),
                    "source_pdb": str(episode_pdb_path),
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "batch_item_index": batch_item_index,
                    "planned_episodes": total_planned_episodes,
                    "training_schedule": "legacy_episodes" if args.epochs is None else "dataset_epochs",
                    "mode": args.mode,
                    "device": str(agent.device),
                    "gpu_ids": list(gpu_ids),
                },
            )

            if (
                args.validate_every > 0
                and args.validation_episodes > 0
                and dataset.val_paths
                and (episode + 1) % args.validate_every == 0
            ):
                validation_records = run_validation(
                    agent=agent,
                    env=env,
                    dataset=dataset,
                    episode=episode,
                    limit=args.validation_episodes,
                )
                validation_path = output_dir / "logs" / "validation.jsonl"
                validation_path.parent.mkdir(parents=True, exist_ok=True)
                with validation_path.open("a", encoding="utf-8") as file:
                    for record in validation_records:
                        file.write(json.dumps(record, sort_keys=True))
                        file.write("\n")
                LOGGER.info(
                    "Validation records appended path=%s count=%s",
                    validation_path,
                    len(validation_records),
                )

            if args.checkpoint_every > 0 and (episode + 1) % args.checkpoint_every == 0:
                checkpoint_dir = output_dir / "checkpoints"
                LOGGER.info("Periodic agent checkpoint triggered episode=%s dir=%s", episode, checkpoint_dir)
                save_agent_checkpoint(agent, checkpoint_dir / "agent.pt")
                LOGGER.info("Periodic agent checkpoint complete episode=%s dir=%s", episode, checkpoint_dir)

            if (
                args.replay_checkpoint_every > 0
                and (episode + 1) % args.replay_checkpoint_every == 0
            ):
                save_resume_checkpoint(
                    agent=agent,
                    replay_buffer=replay_buffer,
                    checkpoint_dir=output_dir / "checkpoints" / "resume",
                    next_episode=episode + 1,
                    args=args,
                )

            LOGGER.info(
                "Episode summary episode=%s/%s epoch=%s batch=%s batch_item=%s reward=%.4f steps=%s "
                "epsilon=%.4f optim_steps=%s elapsed_sec=%.3f",
                episode,
                total_planned_episodes,
                epoch,
                batch_index,
                batch_item_index,
                episode_reward,
                episode_steps,
                agent.epsilon,
                agent.optimization_steps,
                time.perf_counter() - episode_started,
            )

    finally:
        LOGGER.info("Finalization started")
        save_agent_checkpoint(agent, output_dir / "checkpoints" / "agent_final.pt")
        if args.save_final_replay:
            replay_buffer.save(output_dir / "checkpoints" / "replay_buffer_final.npz")
        else:
            LOGGER.info("Final replay checkpoint skipped by --no-save-final-replay")
        plot_paths = logger.generate_plots()
        LOGGER.info("Generated final plots count=%s paths=%s", len(plot_paths), plot_paths)
        logger.close()
        env.close()
        LOGGER.info("Training finished total_elapsed_sec=%.3f", time.perf_counter() - run_started)


def main() -> None:
    args = parse_args()
    configure_stdout_logging(args.log_level)
    train(args)


if __name__ == "__main__":
    main()
