"""
Uniform replay buffer for a standard DDQN workflow.

Stores current and next action masks because the protein-mutation environment
can invalidate actions such as mutating a residue to its current amino acid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReplayBatch:
    """Independent NumPy-array copies sampled from ReplayBuffer."""

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    terminateds: np.ndarray
    truncateds: np.ndarray
    dones: np.ndarray
    action_masks: Optional[np.ndarray]
    next_action_masks: Optional[np.ndarray]
    indices: np.ndarray

    @property
    def batch_size(self) -> int:
        return int(self.actions.shape[0])

    def to_dict(self) -> Dict[str, Optional[np.ndarray]]:
        return {
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_states": self.next_states,
            "terminateds": self.terminateds,
            "truncateds": self.truncateds,
            "dones": self.dones,
            "action_masks": self.action_masks,
            "next_action_masks": self.next_action_masks,
            "indices": self.indices,
        }


class ReplayBuffer:
    """
    Fixed-capacity uniform replay buffer backed by preallocated NumPy arrays.

    Parameters
    ----------
    capacity:
        Maximum number of transitions. Once full, new transitions overwrite
        the oldest ones.
    state_shape:
        Shape of a single observation, excluding the batch dimension.
    action_dim:
        Number of discrete actions. For this project it is usually L * 20.
    seed:
        Optional sampling seed.
    store_action_masks:
        Keep enabled for the current protein-mutation environment.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        capacity: int,
        state_shape: Sequence[int],
        action_dim: int,
        *,
        seed: Optional[int] = None,
        state_dtype: Union[str, np.dtype, type] = np.float32,
        reward_dtype: Union[str, np.dtype, type] = np.float32,
        store_action_masks: bool = True,
    ) -> None:
        self.capacity = self._validate_positive_int(capacity, "capacity")
        self.state_shape = self._validate_state_shape(state_shape)
        self.action_dim = self._validate_positive_int(action_dim, "action_dim")
        self.state_dtype = np.dtype(state_dtype)
        self.reward_dtype = np.dtype(reward_dtype)
        self.store_action_masks = bool(store_action_masks)

        self._rng = np.random.default_rng(seed)
        self._position = 0
        self._size = 0
        LOGGER.info(
            "ReplayBuffer initialized capacity=%s state_shape=%s action_dim=%s "
            "store_action_masks=%s seed=%s",
            self.capacity,
            self.state_shape,
            self.action_dim,
            self.store_action_masks,
            seed,
        )

        self._states = np.empty((self.capacity, *self.state_shape), dtype=self.state_dtype)
        self._next_states = np.empty((self.capacity, *self.state_shape), dtype=self.state_dtype)
        self._actions = np.empty(self.capacity, dtype=np.int64)
        self._rewards = np.empty(self.capacity, dtype=self.reward_dtype)
        self._terminateds = np.empty(self.capacity, dtype=np.bool_)
        self._truncateds = np.empty(self.capacity, dtype=np.bool_)
        self._dones = np.empty(self.capacity, dtype=np.bool_)

        if self.store_action_masks:
            self._action_masks: Optional[np.ndarray] = np.empty(
                (self.capacity, self.action_dim), dtype=np.bool_
            )
            self._next_action_masks: Optional[np.ndarray] = np.empty(
                (self.capacity, self.action_dim), dtype=np.bool_
            )
        else:
            self._action_masks = None
            self._next_action_masks = None

    def __len__(self) -> int:
        return self._size

    @property
    def position(self) -> int:
        return self._position

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity

    def can_sample(self, batch_size: int, *, replace: bool = False) -> bool:
        batch_size = self._validate_positive_int(batch_size, "batch_size")
        return self._size > 0 if replace else self._size >= batch_size

    def add(
        self,
        *,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        terminated: bool = False,
        truncated: bool = False,
        done: Optional[bool] = None,
        action_mask: Optional[Any] = None,
        next_action_mask: Optional[Any] = None,
    ) -> None:
        """Add one transition to the ring buffer."""

        state_array = self._coerce_state(state, "state")
        next_state_array = self._coerce_state(next_state, "next_state")
        action_value = self._validate_action(action)
        reward_value = self._validate_finite_scalar(reward, "reward")

        terminated_value = bool(terminated)
        truncated_value = bool(truncated)
        inferred_done = terminated_value or truncated_value
        if done is None:
            done_value = inferred_done
        else:
            done_value = bool(done)
            if done_value != inferred_done:
                raise ValueError(
                    "Explicit done is inconsistent with terminated/truncated: "
                    f"done={done_value}, terminated={terminated_value}, "
                    f"truncated={truncated_value}."
                )

        if self.store_action_masks:
            current_mask = self._coerce_mask(action_mask, "action_mask")
            next_mask = self._coerce_mask(next_action_mask, "next_action_mask")
        else:
            if action_mask is not None or next_action_mask is not None:
                raise ValueError("Mask values were supplied, but store_action_masks=False.")
            current_mask = None
            next_mask = None

        index = self._position
        self._states[index] = state_array
        self._actions[index] = action_value
        self._rewards[index] = reward_value
        self._next_states[index] = next_state_array
        self._terminateds[index] = terminated_value
        self._truncateds[index] = truncated_value
        self._dones[index] = done_value

        if self.store_action_masks:
            assert self._action_masks is not None
            assert self._next_action_masks is not None
            assert current_mask is not None and next_mask is not None
            self._action_masks[index] = current_mask
            self._next_action_masks[index] = next_mask

        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        LOGGER.info(
            "ReplayBuffer added transition index=%s action=%s reward=%.6f "
            "terminated=%s truncated=%s done=%s size=%s next_position=%s",
            index,
            action_value,
            reward_value,
            terminated_value,
            truncated_value,
            done_value,
            self._size,
            self._position,
        )

    def sample(self, batch_size: int, *, replace: bool = False) -> ReplayBatch:
        """Uniformly sample a mini-batch."""

        batch_size = self._validate_positive_int(batch_size, "batch_size")
        if not self.can_sample(batch_size, replace=replace):
            raise ValueError(
                f"Cannot sample batch_size={batch_size} with replace={replace}: "
                f"buffer size is {self._size}."
            )

        indices = self._rng.choice(self._size, size=batch_size, replace=replace)
        indices = np.asarray(indices, dtype=np.int64)
        LOGGER.info(
            "ReplayBuffer sampled batch_size=%s replace=%s size=%s indices_preview=%s",
            batch_size,
            replace,
            self._size,
            indices[: min(10, len(indices))].tolist(),
        )

        return ReplayBatch(
            states=self._states[indices].copy(),
            actions=self._actions[indices].copy(),
            rewards=self._rewards[indices].copy(),
            next_states=self._next_states[indices].copy(),
            terminateds=self._terminateds[indices].copy(),
            truncateds=self._truncateds[indices].copy(),
            dones=self._dones[indices].copy(),
            action_masks=None if self._action_masks is None else self._action_masks[indices].copy(),
            next_action_masks=None
            if self._next_action_masks is None
            else self._next_action_masks[indices].copy(),
            indices=indices.copy(),
        )

    def clear(self) -> None:
        """Remove transitions without reallocating the arrays."""

        self._position = 0
        self._size = 0

    def state_dict(self) -> Dict[str, Any]:
        """Return a serialization-friendly buffer snapshot."""

        size = self._size
        return {
            "format_version": self.FORMAT_VERSION,
            "capacity": self.capacity,
            "state_shape": self.state_shape,
            "action_dim": self.action_dim,
            "state_dtype": self.state_dtype.str,
            "reward_dtype": self.reward_dtype.str,
            "store_action_masks": self.store_action_masks,
            "position": self._position,
            "size": size,
            "states": self._states[:size].copy(),
            "actions": self._actions[:size].copy(),
            "rewards": self._rewards[:size].copy(),
            "next_states": self._next_states[:size].copy(),
            "terminateds": self._terminateds[:size].copy(),
            "truncateds": self._truncateds[:size].copy(),
            "dones": self._dones[:size].copy(),
            "action_masks": None if self._action_masks is None else self._action_masks[:size].copy(),
            "next_action_masks": None
            if self._next_action_masks is None
            else self._next_action_masks[:size].copy(),
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        """Restore a snapshot into a compatible buffer."""

        if int(payload.get("format_version", -1)) != self.FORMAT_VERSION:
            raise ValueError("Unsupported replay-buffer format version.")

        self._assert_compatible_snapshot(payload)
        size = int(payload["size"])
        position = int(payload["position"])
        if not 0 <= size <= self.capacity:
            raise ValueError(f"Snapshot size must be within 0..{self.capacity}.")
        if not 0 <= position < self.capacity:
            raise ValueError(f"Snapshot position must be within 0..{self.capacity - 1}.")

        self._states[:size] = self._snapshot_array(payload["states"], (size, *self.state_shape), self.state_dtype, "states")
        self._actions[:size] = self._snapshot_array(payload["actions"], (size,), np.dtype(np.int64), "actions")
        self._rewards[:size] = self._snapshot_array(payload["rewards"], (size,), self.reward_dtype, "rewards")
        self._next_states[:size] = self._snapshot_array(payload["next_states"], (size, *self.state_shape), self.state_dtype, "next_states")
        self._terminateds[:size] = self._snapshot_array(payload["terminateds"], (size,), np.dtype(np.bool_), "terminateds")
        self._truncateds[:size] = self._snapshot_array(payload["truncateds"], (size,), np.dtype(np.bool_), "truncateds")
        self._dones[:size] = self._snapshot_array(payload["dones"], (size,), np.dtype(np.bool_), "dones")

        if self.store_action_masks:
            assert self._action_masks is not None and self._next_action_masks is not None
            self._action_masks[:size] = self._snapshot_array(
                payload["action_masks"], (size, self.action_dim), np.dtype(np.bool_), "action_masks"
            )
            self._next_action_masks[:size] = self._snapshot_array(
                payload["next_action_masks"], (size, self.action_dim), np.dtype(np.bool_), "next_action_masks"
            )

        self._position = position
        self._size = size
        self._rng.bit_generator.state = payload["rng_state"]

    def save(self, path: PathLike) -> None:
        """Save a compressed .npz snapshot."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("ReplayBuffer save started path=%s size=%s capacity=%s", path, self._size, self.capacity)
        payload = self.state_dict()
        metadata = {
            key: payload[key]
            for key in (
                "format_version", "capacity", "state_shape", "action_dim",
                "state_dtype", "reward_dtype", "store_action_masks",
                "position", "size", "rng_state"
            )
        }
        np.savez_compressed(
            path,
            metadata=np.asarray(metadata, dtype=object),
            states=payload["states"],
            actions=payload["actions"],
            rewards=payload["rewards"],
            next_states=payload["next_states"],
            terminateds=payload["terminateds"],
            truncateds=payload["truncateds"],
            dones=payload["dones"],
            action_masks=np.asarray([], dtype=np.bool_) if payload["action_masks"] is None else payload["action_masks"],
            next_action_masks=np.asarray([], dtype=np.bool_)
            if payload["next_action_masks"] is None
            else payload["next_action_masks"],
        )
        LOGGER.info("ReplayBuffer save complete path=%s", path)

    @classmethod
    def load(cls, path: PathLike) -> "ReplayBuffer":
        """Load a ReplayBuffer saved by save()."""

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Replay-buffer snapshot not found: {path}")

        LOGGER.info("ReplayBuffer load started path=%s", path)
        with np.load(path, allow_pickle=True) as archive:
            metadata = archive["metadata"].item()
            buffer = cls(
                capacity=int(metadata["capacity"]),
                state_shape=tuple(metadata["state_shape"]),
                action_dim=int(metadata["action_dim"]),
                state_dtype=np.dtype(metadata["state_dtype"]),
                reward_dtype=np.dtype(metadata["reward_dtype"]),
                store_action_masks=bool(metadata["store_action_masks"]),
            )
            payload = dict(metadata)
            payload.update(
                states=archive["states"],
                actions=archive["actions"],
                rewards=archive["rewards"],
                next_states=archive["next_states"],
                terminateds=archive["terminateds"],
                truncateds=archive["truncateds"],
                dones=archive["dones"],
                action_masks=archive["action_masks"] if buffer.store_action_masks else None,
                next_action_masks=archive["next_action_masks"] if buffer.store_action_masks else None,
            )
            buffer.load_state_dict(payload)
            LOGGER.info("ReplayBuffer load complete path=%s size=%s capacity=%s", path, len(buffer), buffer.capacity)
            return buffer

    @staticmethod
    def _validate_positive_int(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer.")
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be > 0.")
        return value

    @staticmethod
    def _validate_state_shape(state_shape: Sequence[int]) -> Tuple[int, ...]:
        shape = tuple(int(dimension) for dimension in state_shape)
        if not shape or any(dimension <= 0 for dimension in shape):
            raise ValueError("state_shape must contain positive dimensions.")
        return shape

    def _validate_action(self, action: Any) -> int:
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer.")
        action = int(action)
        if not 0 <= action < self.action_dim:
            raise ValueError(f"action must be within 0..{self.action_dim - 1}.")
        return action

    @staticmethod
    def _validate_finite_scalar(value: Any, name: str) -> float:
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")
        return value

    def _coerce_state(self, state: Any, name: str) -> np.ndarray:
        array = np.asarray(state, dtype=self.state_dtype)
        if array.shape != self.state_shape:
            raise ValueError(f"{name} must have shape {self.state_shape}, got {array.shape}.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains NaN or infinity.")
        return array

    def _coerce_mask(self, mask: Optional[Any], name: str) -> np.ndarray:
        if mask is None:
            return np.ones(self.action_dim, dtype=np.bool_)
        array = np.asarray(mask, dtype=np.bool_)
        if array.shape != (self.action_dim,):
            raise ValueError(f"{name} must have shape {(self.action_dim,)}, got {array.shape}.")
        return array

    def _assert_compatible_snapshot(self, payload: Mapping[str, Any]) -> None:
        expected = (
            self.capacity,
            self.state_shape,
            self.action_dim,
            self.state_dtype.str,
            self.reward_dtype.str,
            self.store_action_masks,
        )
        actual = (
            int(payload["capacity"]),
            tuple(payload["state_shape"]),
            int(payload["action_dim"]),
            np.dtype(payload["state_dtype"]).str,
            np.dtype(payload["reward_dtype"]).str,
            bool(payload["store_action_masks"]),
        )
        if actual != expected:
            raise ValueError("Snapshot configuration is incompatible with this buffer.")

    @staticmethod
    def _snapshot_array(value: Any, shape: Tuple[int, ...], dtype: np.dtype, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=dtype)
        if array.shape != shape:
            raise ValueError(f"Snapshot array '{name}' must have shape {shape}, got {array.shape}.")
        return array


UniformReplayBuffer = ReplayBuffer
