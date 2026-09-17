"""Central learner for process-based asynchronous mechanical-protein DDQN."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import time
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from model.agent_module.ddqn_agent import AMINO_ACID_ACTION_DIM, DDQNAgent
from model.asynchronous_module import actor_worker_main, run_esm_inference_worker
from model.evaluation_module import summarize_greedy_validation
from model.logging_module.training_logger import TrainingLogger, TrainingLoggerConfig
from model.replay_buffer_module import NStepTransitionAccumulator, ReplayBuffer

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
        "include_visited_mask_in_observation": (
            args.include_visited_mask_in_observation
        ),
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
        "state_feature_dim": int(args.embedding_dim)
        + int(args.include_visited_mask_in_observation),
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
        "kind": "train",
        "episode": int(episode_number),
        "epoch": int(epoch),
        "batch_index": int(batch_index),
        "batch_item_index": int(batch_item_index),
        "pdb_path": str(pdb_path),
    }


def _validation_tasks(
    validation_paths: Sequence[Path],
    *,
    validation_run: int,
    trigger_episode: int,
    seed: int,
) -> list[dict[str, Any]]:
    return [
        {
            "kind": "validation",
            "episode": -1,
            "validation_run": int(validation_run),
            "validation_index": int(index),
            "trigger_episode": int(trigger_episode),
            "pdb_path": str(pdb_path),
            "seed": int(seed) + int(index),
        }
        for index, pdb_path in enumerate(validation_paths)
    ]


def _prefixed_diagnostics(
    prefix: str, values: Mapping[str, Any]
) -> dict[str, Any]:
    return {f"per/{prefix}/{key}": value for key, value in values.items()}


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
    if int(args.validation_bootstrap_samples) < 0:
        raise ValueError("--validation-bootstrap-samples must be >= 0.")

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
    state_shape = (
        1,
        int(args.embedding_dim) + int(args.include_visited_mask_in_observation),
    )
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
        sampling_strategy=args.replay_sampling,
        priority_alpha=args.priority_alpha,
        priority_beta_start=args.priority_beta_start,
        priority_beta_end=args.priority_beta_end,
        priority_beta_steps=args.priority_beta_steps,
        priority_epsilon=args.priority_epsilon,
        positive_sample_fraction=args.positive_sample_fraction,
        positive_replay_reserve_fraction=args.positive_replay_reserve_fraction,
        positive_reward_threshold=args.positive_reward_threshold,
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
        "esm_devices=%s esm_batch_size=%s queue_size=%s replay_sampling=%s "
        "positive_sample_fraction=%s positive_replay_reserve_fraction=%s "
        "positive_reward_lower_bound=%s n_step=%s step_reward_scale=%s schedule=%s",
        total_episodes,
        args.async_actors,
        learner_device,
        inference_devices,
        args.async_inference_batch_size,
        args.async_queue_size,
        args.replay_sampling,
        args.positive_sample_fraction,
        args.positive_replay_reserve_fraction,
        args.positive_reward_threshold,
        args.n_step,
        args.step_reward_scale,
        update_schedule_summary(
            batch_size=agent_config.effective_batch_size,
            train_frequency=args.train_frequency,
            gradient_steps=args.gradient_steps,
        ),
    )
    validation_paths = (
        dataset.validation_paths(limit=int(args.validation_episodes))
        if args.validate_every > 0
        and args.validation_episodes > 0
        and dataset.val_paths
        else tuple()
    )
    if validation_paths and args.no_terminal_reward:
        raise ValueError(
            "Fixed mechanical greedy validation requires terminal reward metrics; "
            "remove --no-terminal-reward or disable validation."
        )

    context = mp.get_context(args.async_start_method)
    stop_event = context.Event()
    request_queue = context.Queue(maxsize=int(args.async_queue_size))
    event_queue = context.Queue(maxsize=int(args.async_queue_size))
    ready_queue = context.Queue()
    task_queue = context.Queue(maxsize=max(int(args.async_actors) * 2, 1))
    validation_task_queue = context.Queue(
        maxsize=max(int(args.validation_episodes), 1)
    )
    worker_count = int(args.async_actors) + int(bool(validation_paths))
    response_queues = [context.Queue(maxsize=2) for _ in range(worker_count)]
    policy_queues = [context.Queue(maxsize=1) for _ in range(worker_count)]
    processes: list[mp.Process] = []
    actor_config = _actor_config(args, agent_config, output_dir)
    initial_policy_state = _cpu_state_dict(agent.online_network)
    learner_pid = os.getpid()

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
                "learner_pid": learner_pid,
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
                "learner_pid": learner_pid,
            },
        )
        process.start()
        processes.append(process)

    validation_actor_id = int(args.async_actors)
    if validation_paths:
        process = context.Process(
            target=actor_worker_main,
            name="pyrosetta-validation-actor",
            kwargs={
                "actor_id": validation_actor_id,
                "actor_count": int(args.async_actors),
                "config": actor_config,
                "initial_policy_state": initial_policy_state,
                "task_queue": validation_task_queue,
                "event_queue": event_queue,
                "policy_queue": policy_queues[validation_actor_id],
                "inference_request_queue": request_queue,
                "inference_response_queue": response_queues[validation_actor_id],
                "ready_queue": ready_queue,
                "stop_event": stop_event,
                "learner_pid": learner_pid,
            },
        )
        process.start()
        processes.append(process)

    submitted = 0
    completed = 0
    last_policy_sync_step = 0
    validation_run = 0
    validation_pending: dict[str, Any] | None = None
    next_validation_at = int(args.validate_every)
    n_step_accumulators = [
        NStepTransitionAccumulator(n_step=args.n_step, gamma=agent_config.gamma)
        for _ in range(int(args.async_actors))
    ]
    try:
        _wait_for_workers(
            ready_queue,
            expected_esm_workers=len(inference_devices),
            expected_actors=worker_count,
            timeout_seconds=args.async_timeout_seconds,
        )
        if validation_paths:
            snapshot = {
                "version": int(agent.optimization_steps),
                "environment_steps": int(agent.environment_steps),
                "state_dict": _cpu_state_dict(agent.online_network),
            }
            _offer_latest(policy_queues[validation_actor_id], snapshot)
            for task in _validation_tasks(
                validation_paths,
                validation_run=validation_run,
                trigger_episode=0,
                seed=args.validation_seed,
            ):
                validation_task_queue.put(task)
            validation_pending = {
                "run": validation_run,
                "trigger_episode": 0,
                "global_step": 0,
                "policy_version": int(agent.optimization_steps),
                "records": [],
            }
            validation_run += 1
        initial_tasks = min(int(args.async_actors), total_episodes)
        for _ in range(initial_tasks):
            task_queue.put(_next_task(episode_iterator, submitted))
            submitted += 1

        while completed < total_episodes or validation_pending is not None:
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
                actor_id = int(event["actor_id"])
                replay_rows = n_step_accumulators[actor_id].append(event)
                for replay_row in replay_rows:
                    replay_buffer.add(**replay_row)
                agent.environment_steps += 1
                global_step = int(agent.environment_steps)
                if should_run_optimizer_event(global_step, args.train_frequency):
                    results = run_replay_optimization_event(
                        agent=agent,
                        replay_buffer=replay_buffer,
                        gradient_steps=args.gradient_steps,
                    )
                    for result in results:
                        optimization_extra = {
                            "collector_actor_id": event["actor_id"],
                            "train_frequency": args.train_frequency,
                            "gradient_steps_requested": args.gradient_steps,
                        }
                        if global_step % args.log_every_steps == 0:
                            optimization_extra.update(
                                _prefixed_diagnostics(
                                    "replay",
                                    replay_buffer.terminal_outcome_diagnostics(),
                                )
                            )
                            optimization_extra.update(
                                _prefixed_diagnostics(
                                    "priority_top_1pct",
                                    replay_buffer.terminal_outcome_diagnostics(
                                        top_priority_fraction=0.01
                                    ),
                                )
                            )
                            if result.sample_indices is not None:
                                optimization_extra.update(
                                    _prefixed_diagnostics(
                                        "sampled_batch",
                                        replay_buffer.terminal_outcome_diagnostics(
                                            indices=result.sample_indices
                                        ),
                                    )
                                )
                        logger.log_optimization(
                            result,
                            global_step=global_step,
                            extra=optimization_extra,
                        )
                    if (
                        results
                        and agent.optimization_steps - last_policy_sync_step
                        >= int(args.async_policy_sync_interval)
                    ):
                        _broadcast_policy(
                            agent, policy_queues[: int(args.async_actors)]
                        )
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
                        "replay_rows_emitted": len(replay_rows),
                        "configured_n_step": args.n_step,
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

            if kind == "validation":
                if validation_pending is None:
                    raise RuntimeError("Received validation result with no pending run.")
                if int(event["validation_run"]) != int(validation_pending["run"]):
                    raise RuntimeError("Received validation result for the wrong run.")
                record = {
                    "validation_run": int(event["validation_run"]),
                    "validation_index": int(event["validation_index"]),
                    "trigger_episode": int(event["trigger_episode"]),
                    "global_step": int(validation_pending["global_step"]),
                    "pdb_path": event["pdb_path"],
                    "seed": int(event["seed"]),
                    "policy_version": int(event["policy_version"]),
                    "total_reward": float(event["total_reward"]),
                    "terminal_reward": float(event["terminal_reward"]),
                    "strength_delta": float(event["terminal_strength_delta"]),
                    "toughness_delta": float(event["terminal_toughness_delta"]),
                    "episode_steps": int(event["episode_steps"]),
                    "elapsed_seconds": float(event["elapsed_seconds"]),
                }
                validation_pending["records"].append(record)
                validation_episode_path = output_dir / "logs" / "validation_episodes.jsonl"
                with validation_episode_path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(record, sort_keys=True))
                    file.write("\n")
                if len(validation_pending["records"]) == len(validation_paths):
                    summary = summarize_greedy_validation(
                        validation_pending["records"],
                        bootstrap_samples=args.validation_bootstrap_samples,
                        seed=args.validation_seed,
                    )
                    logger.log_validation(
                        summary,
                        global_step=int(validation_pending["global_step"]),
                        extra={
                            "validation_run": int(validation_pending["run"]),
                            "trigger_episode": int(
                                validation_pending["trigger_episode"]
                            ),
                            "policy_version": int(
                                validation_pending["policy_version"]
                            ),
                        },
                    )
                    LOGGER.info(
                        "Fixed greedy validation complete run=%s trigger_episode=%s "
                        "global_step=%s metrics=%s",
                        validation_pending["run"],
                        validation_pending["trigger_episode"],
                        validation_pending["global_step"],
                        summary,
                    )
                    validation_pending = None
                continue

            if kind != "episode":
                raise RuntimeError(f"Unknown asynchronous event kind: {kind!r}.")

            actor_id = int(event["actor_id"])
            if len(n_step_accumulators[actor_id]) != 0:
                raise RuntimeError(
                    f"Actor {actor_id} ended an episode with pending n-step transitions."
                )

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
            if (
                validation_paths
                and validation_pending is None
                and completed >= next_validation_at
            ):
                snapshot = {
                    "version": int(agent.optimization_steps),
                    "environment_steps": int(agent.environment_steps),
                    "state_dict": _cpu_state_dict(agent.online_network),
                }
                _offer_latest(policy_queues[validation_actor_id], snapshot)
                for task in _validation_tasks(
                    validation_paths,
                    validation_run=validation_run,
                    trigger_episode=completed,
                    seed=args.validation_seed,
                ):
                    validation_task_queue.put(task)
                validation_pending = {
                    "run": validation_run,
                    "trigger_episode": completed,
                    "global_step": int(agent.environment_steps),
                    "policy_version": int(agent.optimization_steps),
                    "records": [],
                }
                validation_run += 1
                while next_validation_at <= completed:
                    next_validation_at += int(args.validate_every)
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
        if validation_paths:
            try:
                validation_task_queue.put_nowait(None)
            except queue.Full:
                pass
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
