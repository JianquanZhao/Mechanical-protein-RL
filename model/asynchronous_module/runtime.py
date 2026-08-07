"""Worker-side primitives for asynchronous DDQN training.

The learner remains in the parent process.  PyRosetta environments run in
isolated CPU processes and request ESM2 embeddings from dedicated GPU workers.
Only compact, serializable transition data crosses process boundaries.
"""

from __future__ import annotations

import logging
import queue
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch

from model.agent_module.ddqn_agent import AMINO_ACID_ACTION_DIM, QNetwork
from model.environment_module.environment import MechanicalProteinEnv


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    client_id: int
    sequence: str


@dataclass(frozen=True)
class InferenceResponse:
    request_id: str
    embedding: Optional[np.ndarray] = None
    error: Optional[str] = None


class RemoteESM2Encoder:
    """Synchronous environment encoder backed by a remote batched worker."""

    def __init__(
        self,
        *,
        client_id: int,
        request_queue: Any,
        response_queue: Any,
        mutable_only: bool = True,
        timeout_seconds: float = 300.0,
    ) -> None:
        self.client_id = int(client_id)
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.mutable_only = bool(mutable_only)
        self.timeout_seconds = float(timeout_seconds)
        self._request_index = 0

    def __call__(self, pose: Any, env: Any) -> np.ndarray:
        del pose
        sequence = str(env.current_sequence(mutable_only=self.mutable_only))
        return self.encode_sequence(sequence)

    def encode_sequence(self, sequence: str) -> np.ndarray:
        sequence = str(sequence).strip().upper()
        if not sequence:
            raise ValueError("Cannot remotely encode an empty protein sequence.")
        request_id = f"{self.client_id}:{self._request_index}"
        self._request_index += 1
        self.request_queue.put(
            InferenceRequest(request_id, self.client_id, sequence),
            timeout=self.timeout_seconds,
        )
        try:
            response = self.response_queue.get(timeout=self.timeout_seconds)
        except queue.Empty as exc:
            raise TimeoutError(
                f"Timed out waiting for ESM2 response request_id={request_id}."
            ) from exc
        if not isinstance(response, InferenceResponse):
            raise TypeError(f"Unexpected ESM2 response type: {type(response).__name__}.")
        if response.request_id != request_id:
            raise RuntimeError(
                "ESM2 response/request mismatch: "
                f"expected={request_id} received={response.request_id}."
            )
        if response.error is not None:
            raise RuntimeError(
                f"Remote ESM2 inference failed for request_id={request_id}: "
                f"{response.error}"
            )
        if response.embedding is None:
            raise RuntimeError(f"Remote ESM2 response {request_id} has no embedding.")
        return np.asarray(response.embedding, dtype=np.float32)


def _load_esm_encoder(
    *,
    embedding_dim: int,
    device: str,
    model_dir: Optional[str],
) -> Any:
    from model.encoding_module import ESM2SequenceEncoder

    return ESM2SequenceEncoder(
        embedding_dim=embedding_dim,
        device=device,
        mutable_only=True,
        model_dir=model_dir,
    )


def run_esm_inference_worker(
    *,
    worker_id: int,
    device: str,
    embedding_dim: int,
    model_dir: Optional[str],
    request_queue: Any,
    response_queues: Sequence[Any],
    ready_queue: Any,
    stop_event: Any,
    batch_size: int,
    batch_wait_ms: float,
    encoder_factory: Optional[Callable[..., Any]] = None,
) -> None:
    """Load one ESM2 replica and dynamically batch requests from all actors."""

    try:
        if str(device).startswith("cuda"):
            torch.cuda.set_device(torch.device(device))
        factory = _load_esm_encoder if encoder_factory is None else encoder_factory
        encoder = factory(
            embedding_dim=int(embedding_dim),
            device=str(device),
            model_dir=model_dir,
        )
        ready_queue.put(
            {"kind": "esm_ready", "worker_id": int(worker_id), "device": str(device)}
        )

        while not stop_event.is_set():
            try:
                first = request_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if first is None:
                break

            requests = [first]
            deadline = time.monotonic() + max(0.0, float(batch_wait_ms)) / 1000.0
            while len(requests) < int(batch_size):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    request = request_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if request is None:
                    request_queue.put(None)
                    break
                requests.append(request)

            try:
                embeddings = encoder.encode_sequences(
                    [request.sequence for request in requests]
                )
                if len(embeddings) != len(requests):
                    raise RuntimeError(
                        "ESM2 encoder returned a different number of embeddings "
                        f"({len(embeddings)}) than requests ({len(requests)})."
                    )
            except Exception:
                error = traceback.format_exc()
                for request in requests:
                    response_queues[request.client_id].put(
                        InferenceResponse(request.request_id, error=error)
                    )
                raise

            for request, embedding in zip(requests, embeddings):
                response_queues[request.client_id].put(
                    InferenceResponse(
                        request.request_id,
                        embedding=np.asarray(embedding, dtype=np.float32),
                    )
                )
    except Exception:
        ready_queue.put(
            {
                "kind": "fatal",
                "worker": "esm",
                "worker_id": int(worker_id),
                "traceback": traceback.format_exc(),
            }
        )
        stop_event.set()


class ActorPolicy:
    """CPU epsilon-greedy policy used by a PyRosetta actor process."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        hidden_dims: Sequence[int],
        state_dict: Mapping[str, Any],
        seed: int,
    ) -> None:
        self.network = QNetwork(
            (1, int(embedding_dim)),
            AMINO_ACID_ACTION_DIM,
            hidden_dims=hidden_dims,
            embedding_dim=int(embedding_dim),
        ).cpu()
        self.network.load_state_dict(state_dict)
        self.network.eval()
        self.rng = np.random.default_rng(seed)
        self.version = 0

    def update(self, state_dict: Mapping[str, Any], *, version: int) -> None:
        self.network.load_state_dict(state_dict)
        self.network.eval()
        self.version = int(version)

    def select_action(
        self,
        state: Any,
        action_mask: Any,
        *,
        epsilon: float,
    ) -> int:
        state_array = np.asarray(state, dtype=np.float32)
        mask = np.asarray(action_mask, dtype=np.bool_)
        expected_action_dim = int(state_array.shape[0]) * AMINO_ACID_ACTION_DIM
        if state_array.ndim != 2 or mask.shape != (expected_action_dim,):
            raise ValueError(
                "Actor state/mask mismatch: "
                f"state_shape={state_array.shape} mask_shape={mask.shape}."
            )
        valid_actions = np.flatnonzero(mask)
        if valid_actions.size == 0:
            raise ValueError("Actor received a state with no valid action.")
        if self.rng.random() < float(epsilon):
            return int(self.rng.choice(valid_actions))
        with torch.inference_mode():
            q_values = self.network(torch.from_numpy(state_array).unsqueeze(0))[0]
            mask_tensor = torch.from_numpy(mask)
            q_values = q_values.masked_fill(~mask_tensor, -torch.inf)
            return int(q_values.argmax().item())


def epsilon_at_step(config: Mapping[str, Any], environment_steps: int) -> float:
    progress = min(
        1.0,
        int(environment_steps) / max(1, int(config["epsilon_decay_steps"])),
    )
    return float(
        float(config["epsilon_start"])
        + progress * (float(config["epsilon_end"]) - float(config["epsilon_start"]))
    )


def _parse_mutable_positions(value: Optional[str]) -> Optional[tuple[int, ...]]:
    if value is None or not str(value).strip():
        return None
    return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())


def _build_actor_environment(
    config: Mapping[str, Any],
    observation_encoder: RemoteESM2Encoder,
) -> MechanicalProteinEnv:
    terminal_reward_calculator = None
    if not bool(config["no_terminal_reward"]):
        from model.reward_module.terminal_reward import (
            EqualWeightDualStructureTerminalRewardCalculator,
        )

        terminal_reward_calculator = EqualWeightDualStructureTerminalRewardCalculator(
            artifact_path=config["terminal_reward_artifact"],
            predicted_pdb_dir=config["terminal_predicted_pdb_dir"],
        )

    return MechanicalProteinEnv(
        max_steps=int(config["max_steps"]),
        mutable_positions=_parse_mutable_positions(config["mutable_positions"]),
        local_repack_radius=float(config["local_repack_radius"]),
        perform_repack=not bool(config["no_repack"]),
        perform_minimize=not bool(config["no_minimize"]),
        minimize_backbone=bool(config["minimize_backbone"]),
        prevent_revisit_positions=bool(config["prevent_revisit_positions"]),
        raise_on_update_error=bool(config["raise_on_update_error"]),
        step_reward_scale=float(config["step_reward_scale"]),
        terminal_reward_scale=float(config["terminal_reward_scale"]),
        terminal_reward_calculator=terminal_reward_calculator,
        observation_encoder=observation_encoder,
        step_reward_kwargs={
            "rmsd_missing_atom_policy": config["rmsd_missing_atom_policy"],
            "rmsd_missing_penalty": float(config["rmsd_missing_penalty"]),
            "min_rmsd_atoms": int(config["min_rmsd_atoms"]),
        },
        pyrosetta_init_options=config["pyrosetta_options"],
        clean_pdb_before_load=not bool(config["no_clean_pdb_before_load"]),
        load_max_missing_backbone_fraction=float(
            config["max_missing_backbone_fraction"]
        ),
        keep_cleaned_pdbs=bool(config["keep_cleaned_pdbs"]),
        cleaned_pdb_dir=config["cleaned_pdb_dir"],
        seed=int(config["seed"]),
    )


def _latest_policy_snapshot(policy_queue: Any) -> Optional[Mapping[str, Any]]:
    latest = None
    while True:
        try:
            latest = policy_queue.get_nowait()
        except queue.Empty:
            return latest


def actor_worker_main(
    *,
    actor_id: int,
    actor_count: int,
    config: Mapping[str, Any],
    initial_policy_state: Mapping[str, Any],
    task_queue: Any,
    event_queue: Any,
    policy_queue: Any,
    inference_request_queue: Any,
    inference_response_queue: Any,
    ready_queue: Any,
    stop_event: Any,
) -> None:
    """Collect complete episodes in one isolated PyRosetta process."""

    env: Optional[MechanicalProteinEnv] = None
    try:
        torch.set_num_threads(max(1, int(config["actor_torch_threads"])))
        remote_encoder = RemoteESM2Encoder(
            client_id=int(actor_id),
            request_queue=inference_request_queue,
            response_queue=inference_response_queue,
            mutable_only=bool(config["esm2_mutable_only"]),
            timeout_seconds=float(config["async_timeout_seconds"]),
        )
        env = _build_actor_environment(config, remote_encoder)
        policy = ActorPolicy(
            embedding_dim=int(config["embedding_dim"]),
            hidden_dims=tuple(config["hidden_dims"]),
            state_dict=initial_policy_state,
            seed=int(config["seed"]) + int(actor_id) * 1009,
        )
        estimated_global_step = int(config.get("initial_environment_steps", 0)) + int(
            actor_id
        )
        ready_queue.put({"kind": "actor_ready", "actor_id": int(actor_id)})

        while not stop_event.is_set():
            try:
                task = task_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if task is None:
                break

            snapshot = _latest_policy_snapshot(policy_queue)
            if snapshot is not None:
                policy.update(snapshot["state_dict"], version=snapshot["version"])
                estimated_global_step = max(
                    estimated_global_step,
                    int(snapshot["environment_steps"]),
                )

            episode = int(task["episode"])
            pdb_path = str(task["pdb_path"])
            started = time.perf_counter()
            state, info = env.reset(pdb_path=pdb_path)
            episode_reward = 0.0
            episode_step = 0

            while not stop_event.is_set():
                snapshot = _latest_policy_snapshot(policy_queue)
                if snapshot is not None:
                    policy.update(snapshot["state_dict"], version=snapshot["version"])
                    estimated_global_step = max(
                        estimated_global_step,
                        int(snapshot["environment_steps"]),
                    )

                epsilon = epsilon_at_step(config, estimated_global_step)
                action_mask = np.asarray(info["action_mask"], dtype=np.bool_)
                action = policy.select_action(
                    state,
                    action_mask,
                    epsilon=epsilon,
                )
                next_state, reward, terminated, truncated, next_info = env.step(action)
                event_queue.put(
                    {
                        "kind": "transition",
                        "actor_id": int(actor_id),
                        "episode": episode,
                        "episode_step": episode_step,
                        "state": state,
                        "action": int(action),
                        "reward": float(reward),
                        "next_state": next_state,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "action_mask": action_mask,
                        "next_action_mask": np.asarray(
                            next_info["action_mask"], dtype=np.bool_
                        ),
                        "info": next_info,
                        "epsilon": epsilon,
                        "policy_version": policy.version,
                    },
                    timeout=float(config["async_timeout_seconds"]),
                )
                state = next_state
                info = next_info
                episode_reward += float(reward)
                episode_step += 1
                estimated_global_step += int(actor_count)
                if terminated or truncated:
                    break

            if stop_event.is_set():
                break
            candidate_path = None
            if bool(config["save_candidates"]):
                candidate_path = (
                    Path(config["output_dir"])
                    / "candidates"
                    / f"episode_{episode:08d}.pdb"
                )
                candidate_path.parent.mkdir(parents=True, exist_ok=True)
                env.save_current_pose(str(candidate_path))

            event_queue.put(
                {
                    "kind": "episode",
                    "actor_id": int(actor_id),
                    "episode": episode,
                    "epoch": int(task["epoch"]),
                    "batch_index": int(task["batch_index"]),
                    "batch_item_index": int(task["batch_item_index"]),
                    "pdb_path": pdb_path,
                    "total_reward": episode_reward,
                    "episode_steps": episode_step,
                    "epsilon": epsilon_at_step(config, estimated_global_step),
                    "policy_version": policy.version,
                    "candidate_path": None
                    if candidate_path is None
                    else str(candidate_path),
                    "info": info,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                timeout=float(config["async_timeout_seconds"]),
            )
    except Exception:
        event_queue.put(
            {
                "kind": "fatal",
                "worker": "actor",
                "actor_id": int(actor_id),
                "traceback": traceback.format_exc(),
            }
        )
        stop_event.set()
    finally:
        if env is not None:
            env.close()
