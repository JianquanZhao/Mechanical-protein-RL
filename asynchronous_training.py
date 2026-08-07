"""Central learner for process-based asynchronous mechanical-protein DDQN."""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from model.agent_module.ddqn_agent import AMINO_ACID_ACTION_DIM, DDQNAgent
from model.asynchronous_module import actor_worker_main, run_esm_inference_worker
from model.logging_module.training_logger import TrainingLogger, TrainingLoggerConfig
from model.replay_buffer_module.replay_buffer import ReplayBuffer

from training import (
    apply_full_batch_shortcut,
    build_agent_config,
    build_dataset,
    configure_stdout_logging,
    iter_training_episode_paths,
    parse_args,
    parse_gpu_ids,
    planned_episode_count,
    run_replay_optimization_event,
    save_agent_checkpoint,
    should_run_optimizer_event,
    update_schedule_summary,
    validate_training_schedule_args,
    write_run_config,
)


LOGGER = logging.getLogger(__name__)


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def _offer_latest(target_queue: Any, payload: Mapping[str, Any]) -> None:
    try:
        target_queue.put_nowait(payload)
        return
    except queue.Full:
        pass
    try:
        target_queue.get_nowait()
    except queue.Empty:
        pass
    target_queue.put_nowait(payload)


def _broadcast_policy(
    agent: DDQNAgent,
    policy_queues: Sequence[Any],
) -> None:
    snapshot = {
        "version": int(agent.optimization_steps),
        "environment_steps": int(agent.environment_steps),
        "state_dict": _cpu_state_dict(agent.online_network),
    }
    for policy_queue in policy_queues:
        _offer_latest(policy_queue, snapshot)


def _actor_config(args: Any, agent_config: Any, output_dir: Path) -> dict[str, Any]:
    return {
        "max_steps": args.max_steps,
        "mutable_positions": args.mutable_positions,
        "local_repack_radius": args.local_repack_radius,
        "no_repack": args.no_repack,
        "no_minimize": args.no_minimize,
        "minimize_backbone": args.minimize_backbone,
        "prevent_revisit_positions": args.prevent_revisit_positions,
        "raise_on_update_error": args.raise_on_update_error,
        "step_reward_scale": args.step_reward_scale,
        "terminal_reward_scale": args.terminal_reward_scale,
        "no_terminal_reward": args.no_terminal_reward,
        "terminal_reward_artifact": args.terminal_reward_artifact,
        "terminal_predicted_pdb_dir": args.terminal_predicted_pdb_dir,
        "rmsd_missing_atom_policy": args.rmsd_missing_atom_policy,
        "rmsd_missing_penalty": args.rmsd_missing_penalty,
        "min_rmsd_atoms": args.min_rmsd_atoms,
        "pyrosetta_options": args.pyrosetta_options,
        "no_clean_pdb_before_load": args.no_clean_pdb_before_load,
        "max_missing_backbone_fraction": args.max_missing_backbone_fraction,
        "keep_cleaned_pdbs": args.keep_cleaned_pdbs,
        "cleaned_pdb_dir": args.cleaned_pdb_dir,
        "seed": args.seed,
        "embedding_dim": args.embedding_dim,
        "hidden_dims": tuple(agent_config.hidden_dims),
        "epsilon_start": agent_config.epsilon_start,
        "epsilon_end": agent_config.epsilon_end,
        "epsilon_decay_steps": agent_config.epsilon_decay_steps,
        "initial_environment_steps": 0,
        "esm2_mutable_only": args.esm2_mutable_only,
        "actor_torch_threads": args.async_actor_torch_threads,
        "async_timeout_seconds": args.async_timeout_seconds,
        "save_candidates": args.save_candidates,
        "output_dir": str(output_dir),
    }


def _wait_for_workers(
    ready_queue: Any,
    *,
    expected_esm_workers: int,
    expected_actors: int,
    timeout_seconds: float,
) -> None:
    ready_esm: set[int] = set()
    ready_actors: set[int] = set()
    deadline = time.monotonic() + float(timeout_seconds)
    while len(ready_esm) < expected_esm_workers or len(ready_actors) < expected_actors:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                "Timed out starting asynchronous workers: "
                f"esm={len(ready_esm)}/{expected_esm_workers}, "
                f"actors={len(ready_actors)}/{expected_actors}."
            )
        try:
            message = ready_queue.get(timeout=min(1.0, remaining))
        except queue.Empty:
            continue
        if message["kind"] == "fatal":
            raise RuntimeError(
                f"{message['worker']} worker failed during startup:\n"
                f"{message['traceback']}"
            )
        if message["kind"] == "esm_ready":
            ready_esm.add(int(message["worker_id"]))
        elif message["kind"] == "actor_ready":
            ready_actors.add(int(message["actor_id"]))
    LOGGER.info(
        "Asynchronous workers ready actors=%s esm_workers=%s",
        len(ready_actors),
        len(ready_esm),
    )


def _next_task(
    episode_iterator: Iterable[tuple[int, int, int, Path]],
    episode_number: int,
) -> dict[str, Any]:
    epoch, batch_index, batch_item_index, pdb_path = next(episode_iterator)
    return {
        "episode": int(episode_number),
        "epoch": int(epoch),
        "batch_index": int(batch_index),
        "batch_item_index": int(batch_item_index),
        "pdb_path": str(pdb_path),
    }


def _check_worker_health(ready_queue: Any, processes: Sequence[mp.Process]) -> None:
    while True:
        try:
            message = ready_queue.get_nowait()
        except queue.Empty:
            break
        if message.get("kind") == "fatal":
            raise RuntimeError(
                f"{message['worker']} worker failed:\n{message['traceback']}"
            )
    failed = [process for process in processes if process.exitcode not in (None, 0)]
    if failed:
        details = ", ".join(
            f"{process.name}:exitcode={process.exitcode}" for process in failed
        )
        raise RuntimeError(f"Asynchronous worker exited unexpectedly: {details}.")


def train_asynchronously(args: Any) -> None:
    """Run many CPU environments against batched ESM2 workers and one learner."""

    if args.observation_encoder != "esm2":
        raise ValueError("Asynchronous mode currently requires --observation-encoder esm2.")
    if args.resume_checkpoint_dir is not None:
        raise ValueError(
            "Asynchronous resume is not enabled yet because in-flight episode IDs "
            "must be checkpointed atomically. Start a new asynchronous run."
        )
    if args.replay_checkpoint_every:
        raise ValueError(
            "Use --replay-checkpoint-every 0 in asynchronous mode; an in-flight "
            "replay snapshot is not an exact scheduling checkpoint."
        )
    if int(args.async_actors) <= 0:
        raise ValueError("--async-actors must be positive.")
    if int(args.async_inference_batch_size) <= 0:
        raise ValueError("--async-inference-batch-size must be positive.")
    if int(args.async_policy_sync_interval) <= 0:
        raise ValueError("--async-policy-sync-interval must be positive.")
    if int(args.async_queue_size) <= 0:
        raise ValueError("--async-queue-size must be positive.")
    if float(args.async_inference_batch_wait_ms) < 0:
        raise ValueError("--async-inference-batch-wait-ms must be non-negative.")
    if int(args.async_actor_torch_threads) <= 0:
        raise ValueError("--async-actor-torch-threads must be positive.")
    if float(args.async_timeout_seconds) <= 0:
        raise ValueError("--async-timeout-seconds must be positive.")

    apply_full_batch_shortcut(args)
    validate_training_schedule_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    if args.device == "auto":
        learner_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        learner_device = str(args.device)
    inference_gpu_ids = parse_gpu_ids(args.async_inference_gpu_ids)
    if not inference_gpu_ids:
        raise ValueError("--async-inference-gpu-ids must specify at least one GPU.")
    inference_devices = tuple(f"cuda:{gpu_id}" for gpu_id in inference_gpu_ids)
    if not torch.cuda.is_available():
        raise RuntimeError("Asynchronous ESM2 GPU inference requires CUDA.")
    invalid_gpu_ids = [
        gpu_id
        for gpu_id in inference_gpu_ids
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count()
    ]
    if invalid_gpu_ids:
        raise ValueError(
            f"Invalid ESM2 inference GPU ids {invalid_gpu_ids}; "
            f"visible CUDA device count is {torch.cuda.device_count()}."
        )
    learner_index = (
        (torch.device(learner_device).index or 0)
        if learner_device.startswith("cuda")
        else None
    )
    if learner_index is not None and learner_index in inference_gpu_ids:
        raise ValueError(
            "Learner and ESM inference GPUs must be disjoint; "
            f"learner={learner_device}, inference={inference_devices}."
        )

    dataset = build_dataset(args)
    dataset_rng = np.random.default_rng(args.dataset_seed)
    total_episodes = planned_episode_count(args, dataset)
    episode_iterator = iter(
        iter_training_episode_paths(args=args, dataset=dataset, rng=dataset_rng)
    )
    agent_config = build_agent_config(args, device=learner_device)
    state_shape = (1, int(args.embedding_dim))
    agent = DDQNAgent(
        state_shape=state_shape,
        action_dim=AMINO_ACID_ACTION_DIM,
        config=agent_config,
    )
    replay_buffer = ReplayBuffer(
        capacity=args.replay_capacity,
        state_shape=state_shape,
        action_dim=AMINO_ACID_ACTION_DIM,
        seed=args.seed,
        store_action_masks=True,
        variable_length=True,
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
            episode_csv_every=args.episode_csv_every,
            max_step_records_in_memory=args.max_step_records_in_memory,
            max_optimization_records_in_memory=args.max_optimization_records_in_memory,
        )
    )
    write_run_config(
        args,
        output_dir,
        agent_config,
        state_shape=state_shape,
        action_dim=AMINO_ACID_ACTION_DIM,
        gpu_ids=(),
    )
    LOGGER.info(
        "Asynchronous training started episodes=%s actors=%s learner_device=%s "
        "esm_devices=%s esm_batch_size=%s queue_size=%s schedule=%s",
        total_episodes,
        args.async_actors,
        learner_device,
        inference_devices,
        args.async_inference_batch_size,
        args.async_queue_size,
        update_schedule_summary(
            batch_size=agent_config.effective_batch_size,
            train_frequency=args.train_frequency,
            gradient_steps=args.gradient_steps,
        ),
    )
    if args.validate_every > 0 and args.validation_episodes > 0:
        LOGGER.warning(
            "Periodic validation is deferred in asynchronous mode so the learner "
            "does not block transition draining. Evaluate saved checkpoints separately."
        )

    context = mp.get_context(args.async_start_method)
    stop_event = context.Event()
    request_queue = context.Queue(maxsize=int(args.async_queue_size))
    event_queue = context.Queue(maxsize=int(args.async_queue_size))
    ready_queue = context.Queue()
    task_queue = context.Queue(maxsize=max(int(args.async_actors) * 2, 1))
    response_queues = [context.Queue(maxsize=2) for _ in range(int(args.async_actors))]
    policy_queues = [context.Queue(maxsize=1) for _ in range(int(args.async_actors))]
    processes: list[mp.Process] = []
    actor_config = _actor_config(args, agent_config, output_dir)
    initial_policy_state = _cpu_state_dict(agent.online_network)

    for worker_id, device in enumerate(inference_devices):
        process = context.Process(
            target=run_esm_inference_worker,
            name=f"esm-worker-{worker_id}",
            kwargs={
                "worker_id": worker_id,
                "device": device,
                "embedding_dim": args.embedding_dim,
                "model_dir": args.esm_model_dir,
                "request_queue": request_queue,
                "response_queues": response_queues,
                "ready_queue": ready_queue,
                "stop_event": stop_event,
                "batch_size": args.async_inference_batch_size,
                "batch_wait_ms": args.async_inference_batch_wait_ms,
            },
        )
        process.start()
        processes.append(process)

    for actor_id in range(int(args.async_actors)):
        process = context.Process(
            target=actor_worker_main,
            name=f"pyrosetta-actor-{actor_id}",
            kwargs={
                "actor_id": actor_id,
                "actor_count": int(args.async_actors),
                "config": actor_config,
                "initial_policy_state": initial_policy_state,
                "task_queue": task_queue,
                "event_queue": event_queue,
                "policy_queue": policy_queues[actor_id],
                "inference_request_queue": request_queue,
                "inference_response_queue": response_queues[actor_id],
                "ready_queue": ready_queue,
                "stop_event": stop_event,
            },
        )
        process.start()
        processes.append(process)

    submitted = 0
    completed = 0
    last_policy_sync_step = 0
    try:
        _wait_for_workers(
            ready_queue,
            expected_esm_workers=len(inference_devices),
            expected_actors=int(args.async_actors),
            timeout_seconds=args.async_timeout_seconds,
        )
        initial_tasks = min(int(args.async_actors), total_episodes)
        for _ in range(initial_tasks):
            task_queue.put(_next_task(episode_iterator, submitted))
            submitted += 1

        while completed < total_episodes:
            _check_worker_health(ready_queue, processes)
            if stop_event.is_set():
                raise RuntimeError("An asynchronous worker requested global shutdown.")
            try:
                event = event_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            kind = event.get("kind")
            if kind == "fatal":
                raise RuntimeError(
                    f"{event['worker']} worker failed:\n{event['traceback']}"
                )
            if kind == "transition":
                replay_buffer.add(
                    state=event["state"],
                    action=event["action"],
                    reward=event["reward"],
                    next_state=event["next_state"],
                    terminated=event["terminated"],
                    truncated=event["truncated"],
                    action_mask=event["action_mask"],
                    next_action_mask=event["next_action_mask"],
                )
                agent.environment_steps += 1
                global_step = int(agent.environment_steps)
                if should_run_optimizer_event(global_step, args.train_frequency):
                    results = run_replay_optimization_event(
                        agent=agent,
                        replay_buffer=replay_buffer,
                        gradient_steps=args.gradient_steps,
                    )
                    for result in results:
                        logger.log_optimization(
                            result,
                            global_step=global_step,
                            extra={
                                "collector_actor_id": event["actor_id"],
                                "train_frequency": args.train_frequency,
                                "gradient_steps_requested": args.gradient_steps,
                            },
                        )
                    if (
                        results
                        and agent.optimization_steps - last_policy_sync_step
                        >= int(args.async_policy_sync_interval)
                    ):
                        _broadcast_policy(agent, policy_queues)
                        last_policy_sync_step = int(agent.optimization_steps)

                logger.log_step(
                    episode=event["episode"],
                    episode_step=event["episode_step"],
                    global_step=global_step,
                    reward=event["reward"],
                    terminated=event["terminated"],
                    truncated=event["truncated"],
                    info=event["info"],
                    extra={
                        "actor_id": event["actor_id"],
                        "actor_policy_version": event["policy_version"],
                    },
                )
                if global_step % args.log_every_steps == 0:
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    LOGGER.info(
                        "Async progress global_step=%s completed_episodes=%s/%s "
                        "replay_size=%s optim_steps=%s transitions_per_sec=%.3f",
                        global_step,
                        completed,
                        total_episodes,
                        len(replay_buffer),
                        agent.optimization_steps,
                        global_step / elapsed,
                    )
                continue

            if kind != "episode":
                raise RuntimeError(f"Unknown asynchronous event kind: {kind!r}.")

            completed += 1
            logger.end_episode(
                episode=event["episode"],
                total_reward=event["total_reward"],
                episode_steps=event["episode_steps"],
                epsilon=event["epsilon"],
                optimization_steps=agent.optimization_steps,
                info=event["info"],
                extra={
                    "actor_id": event["actor_id"],
                    "actor_policy_version": event["policy_version"],
                    "candidate_pdb": event["candidate_path"],
                    "source_pdb": event["pdb_path"],
                    "epoch": event["epoch"],
                    "batch_index": event["batch_index"],
                    "batch_item_index": event["batch_item_index"],
                    "completion_order": completed,
                    "elapsed_seconds": event["elapsed_seconds"],
                    "mode": "asynchronous",
                },
            )
            if args.checkpoint_every > 0 and completed % args.checkpoint_every == 0:
                save_agent_checkpoint(
                    agent,
                    output_dir / "checkpoints" / f"agent_{completed:08d}.pt",
                )
            if submitted < total_episodes:
                task_queue.put(_next_task(episode_iterator, submitted))
                submitted += 1
            else:
                task_queue.put(None)

            LOGGER.info(
                "Async episode complete episode=%s completion=%s/%s actor=%s "
                "reward=%.6f steps=%s elapsed_sec=%.3f",
                event["episode"],
                completed,
                total_episodes,
                event["actor_id"],
                event["total_reward"],
                event["episode_steps"],
                event["elapsed_seconds"],
            )
    finally:
        stop_event.set()
        for _ in inference_devices:
            try:
                request_queue.put_nowait(None)
            except queue.Full:
                break
        for _ in range(int(args.async_actors)):
            try:
                task_queue.put_nowait(None)
            except queue.Full:
                break
        for process in processes:
            process.join(timeout=10.0)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)

        save_agent_checkpoint(agent, output_dir / "checkpoints" / "agent_final.pt")
        if args.save_final_replay:
            replay_buffer.save(output_dir / "checkpoints" / "replay_buffer_final.npz")
        logger.generate_plots()
        logger.close()
        LOGGER.info(
            "Asynchronous training finalized completed_episodes=%s/%s "
            "environment_steps=%s optimization_steps=%s elapsed_sec=%.3f",
            completed,
            total_episodes,
            agent.environment_steps,
            agent.optimization_steps,
            time.perf_counter() - started,
        )


def main() -> None:
    args = parse_args()
    configure_stdout_logging(args.log_level)
    if args.mode != "asynchronous":
        raise ValueError("asynchronous_training.py requires --mode asynchronous.")
    train_asynchronously(args)


if __name__ == "__main__":
    main()
