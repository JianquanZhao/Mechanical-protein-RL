"""
Root training entry point for the mechanical-protein DDQN project.

Examples
--------
Single machine, single GPU:
    python training.py --mode single --device cuda:0

Single machine, multiple GPUs:
    python training.py --mode multi --gpu-ids 0,1

CPU/debug run:
    python training.py --mode single --device cpu --episodes 2 --max-steps 1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch

from model.agent_module.ddqn_agent import DDQNAgent, DDQNConfig
from model.dataset_module import ProteinStructureDataset
from model.environment_module.environment import MechanicalProteinEnv
from model.logging_module.training_logger import TrainingLogger, TrainingLoggerConfig
from model.replay_buffer_module.replay_buffer import ReplayBuffer


DEFAULT_PDB_DIR = "model/reward_module"
DEFAULT_OUTPUT_DIR = "outputs/ddqn_base"
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
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episodes", type=int, default=500)
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
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--replay-warmup-size", type=int, default=1_000)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--target-sync-interval", type=int, default=250)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=50_000)
    parser.add_argument("--use-amp", action="store_true")

    parser.add_argument("--checkpoint-every", type=int, default=25)
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
        )

    LOGGER.info(
        "Building MechanicalProteinEnv max_steps=%s mutable_positions=%s "
        "repack=%s minimize=%s minimize_backbone=%s local_repack_radius=%s observation_encoder=%s",
        args.max_steps,
        args.mutable_positions or "all canonical residues",
        not args.no_repack,
        not args.no_minimize,
        args.minimize_backbone,
        args.local_repack_radius,
        args.observation_encoder,
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
        observation_encoder=observation_encoder,
        pyrosetta_init_options=args.pyrosetta_options,
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
        seed=args.seed,
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


def enable_data_parallel(agent: DDQNAgent, gpu_ids: Sequence[int]) -> None:
    if not gpu_ids:
        LOGGER.info("DataParallel disabled; using agent device=%s", agent.device)
        return

    LOGGER.info("Wrapping online and target networks with DataParallel gpu_ids=%s", list(gpu_ids))
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
    torch.save(payload, path)
    LOGGER.info("Saved agent checkpoint path=%s elapsed_sec=%.3f", path, time.perf_counter() - started)


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
        LOGGER.info("Validation episode complete record=%s", record)
    return records


def train(args: argparse.Namespace) -> None:
    if args.log_every_steps <= 0:
        raise ValueError("--log-every-steps must be a positive integer.")

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
    LOGGER.info(
        "Dataset ready train_size=%s val_size=%s train_index=%s val_index=%s",
        len(dataset.train_paths),
        len(dataset.val_paths),
        dataset.train_index_path,
        dataset.val_index_path,
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
    agent = DDQNAgent(
        state_shape=state.shape,
        action_dim=env.action_space.n,
        config=agent_config,
    )
    enable_data_parallel(agent, gpu_ids)

    replay_buffer = ReplayBuffer(
        capacity=args.replay_capacity,
        state_shape=state.shape,
        action_dim=env.action_space.n,
        seed=args.seed,
        store_action_masks=True,
        variable_length=args.observation_encoder == "esm2",
    )
    LOGGER.info(
        "ReplayBuffer ready capacity=%s effective_batch_size=%s warmup=%s variable_length=%s",
        args.replay_capacity,
        agent_config.effective_batch_size,
        agent_config.replay_warmup_size,
        args.observation_encoder == "esm2",
    )

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

    total_environment_steps = 0

    try:
        for episode in range(args.episodes):
            episode_started = time.perf_counter()
            LOGGER.info("Episode %s/%s starting", episode + 1, args.episodes)
            episode_pdb_path = dataset.sample_train_path(dataset_rng)
            state, info = env.reset(pdb_path=str(episode_pdb_path))
            LOGGER.info(
                "Episode %s reset pdb_path=%s valid_actions=%s accepted_mutations=%s",
                episode,
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
                LOGGER.info(
                    "Episode %s step %s global_step=%s selecting_action epsilon=%.6f valid_actions=%s",
                    episode,
                    episode_steps,
                    total_environment_steps,
                    agent.epsilon,
                    valid_action_count,
                )
                action = agent.select_action(state, action_mask=action_mask)
                LOGGER.info("Selected action=%s for episode=%s step=%s", action, episode, episode_steps)

                next_state, reward, terminated, truncated, next_info = env.step(action)
                decoded_action = next_info.get("decoded_action", {})
                LOGGER.info(
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
                LOGGER.info(
                    "ReplayBuffer add complete size=%s capacity=%s position=%s",
                    len(replay_buffer),
                    replay_buffer.capacity,
                    replay_buffer.position,
                )

                optimization_result = agent.optimize_from_replay_buffer(replay_buffer)
                if optimization_result is not None:
                    LOGGER.info(
                        "Optimization step complete step=%s loss=%.8f mean_q=%.8f "
                        "mean_target=%.8f mean_abs_td=%.8f grad_norm=%.8f target_synced=%s",
                        optimization_result.optimization_step,
                        optimization_result.loss,
                        optimization_result.mean_q_value,
                        optimization_result.mean_target_q_value,
                        optimization_result.mean_absolute_td_error,
                        optimization_result.grad_norm,
                        optimization_result.target_synced,
                    )
                    logger.log_optimization(
                        optimization_result,
                        global_step=total_environment_steps,
                    )
                else:
                    required_size = max(
                        agent_config.replay_warmup_size,
                        agent_config.effective_batch_size,
                    )
                    LOGGER.info(
                        "Optimization skipped replay_buffer_size=%s required_size=%s",
                        len(replay_buffer),
                        required_size,
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
                LOGGER.info("Saving candidate PDB episode=%s path=%s", episode, candidate_path)
                env.save_current_pose(candidate_path)
                LOGGER.info("Saved candidate PDB episode=%s path=%s", episode, candidate_path)

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
                LOGGER.info("Periodic checkpoint triggered episode=%s dir=%s", episode, checkpoint_dir)
                save_agent_checkpoint(agent, checkpoint_dir / "agent.pt")
                replay_buffer.save(checkpoint_dir / "replay_buffer.npz")
                LOGGER.info("Periodic checkpoint complete episode=%s dir=%s", episode, checkpoint_dir)

            LOGGER.info(
                "Episode summary episode=%s reward=%.4f steps=%s "
                "epsilon=%.4f optim_steps=%s elapsed_sec=%.3f",
                episode,
                episode_reward,
                episode_steps,
                agent.epsilon,
                agent.optimization_steps,
                time.perf_counter() - episode_started,
            )

    finally:
        LOGGER.info("Finalization started")
        save_agent_checkpoint(agent, output_dir / "checkpoints" / "agent_final.pt")
        replay_buffer.save(output_dir / "checkpoints" / "replay_buffer_final.npz")
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
