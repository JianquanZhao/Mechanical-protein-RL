"""
Uniform or TD-error-prioritized replay buffer for a DDQN workflow.

Stores current and next action masks because the protein-mutation environment
can invalidate actions such as mutating a residue to its current amino acid.
Positive terminal-outcome retention and stratified sampling are optional so
rare improving trajectories can remain visible without duplicating ESM states.
"""

from __future__ import annotations

import logging
import time
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
    n_steps: Optional[np.ndarray] = None
    importance_weights: Optional[np.ndarray] = None
    sampling_probabilities: Optional[np.ndarray] = None
    priority_beta: Optional[float] = None
    terminal_rewards: Optional[np.ndarray] = None
    terminal_reward_lcbs: Optional[np.ndarray] = None
    terminal_strength_deltas: Optional[np.ndarray] = None
    terminal_toughness_deltas: Optional[np.ndarray] = None
    episode_ids: Optional[np.ndarray] = None

    @property
    def batch_size(self) -> int:
        return int(self.actions.shape[0])

    def to_dict(self) -> Dict[str, Any]:
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
            "n_steps": self.n_steps,
            "importance_weights": self.importance_weights,
            "sampling_probabilities": self.sampling_probabilities,
            "priority_beta": self.priority_beta,
            "terminal_rewards": self.terminal_rewards,
            "terminal_reward_lcbs": self.terminal_reward_lcbs,
            "terminal_strength_deltas": self.terminal_strength_deltas,
            "terminal_toughness_deltas": self.terminal_toughness_deltas,
            "episode_ids": self.episode_ids,
        }


class ReplayBuffer:
    """
    Fixed-capacity replay buffer with uniform and prioritized sampling modes.

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

    FORMAT_VERSION = 5
    SUPPORTED_SAMPLING_STRATEGIES = ("uniform", "prioritized")

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
        variable_length: bool = False,
        sampling_strategy: str = "uniform",
        priority_alpha: float = 0.6,
        priority_beta_start: float = 0.4,
        priority_beta_end: float = 1.0,
        priority_beta_steps: int = 1_000_000,
        priority_epsilon: float = 1e-6,
        positive_sample_fraction: float = 0.0,
        positive_replay_reserve_fraction: float = 0.0,
        positive_reward_threshold: float = 0.0,
    ) -> None:
        self.capacity = self._validate_positive_int(capacity, "capacity")
        self.state_shape = self._validate_state_shape(state_shape)
        self.action_dim = self._validate_positive_int(action_dim, "action_dim")
        self.state_dtype = np.dtype(state_dtype)
        self.reward_dtype = np.dtype(reward_dtype)
        self.store_action_masks = bool(store_action_masks)
        self.variable_length = bool(variable_length)
        self.sampling_strategy = str(sampling_strategy).strip().lower()
        if self.sampling_strategy not in self.SUPPORTED_SAMPLING_STRATEGIES:
            raise ValueError(
                "sampling_strategy must be one of "
                f"{self.SUPPORTED_SAMPLING_STRATEGIES}."
            )
        self.priority_alpha = self._validate_unit_interval(
            priority_alpha,
            "priority_alpha",
            lower_inclusive=True,
        )
        self.priority_beta_start = self._validate_unit_interval(
            priority_beta_start,
            "priority_beta_start",
            lower_inclusive=True,
        )
        self.priority_beta_end = self._validate_unit_interval(
            priority_beta_end,
            "priority_beta_end",
            lower_inclusive=True,
        )
        if self.priority_beta_end < self.priority_beta_start:
            raise ValueError("priority_beta_end must be >= priority_beta_start.")
        self.priority_beta_steps = self._validate_positive_int(
            priority_beta_steps,
            "priority_beta_steps",
        )
        self.priority_epsilon = self._validate_finite_scalar(
            priority_epsilon,
            "priority_epsilon",
        )
        if self.priority_epsilon <= 0.0:
            raise ValueError("priority_epsilon must be > 0.")
        self.positive_sample_fraction = self._validate_unit_interval(
            positive_sample_fraction,
            "positive_sample_fraction",
            lower_inclusive=True,
        )
        self.positive_replay_reserve_fraction = self._validate_unit_interval(
            positive_replay_reserve_fraction,
            "positive_replay_reserve_fraction",
            lower_inclusive=True,
        )
        if self.positive_replay_reserve_fraction >= 1.0:
            raise ValueError("positive_replay_reserve_fraction must be within [0, 1).")
        self.positive_reward_threshold = self._validate_finite_scalar(
            positive_reward_threshold,
            "positive_reward_threshold",
        )

        self._rng = np.random.default_rng(seed)
        self._position = 0
        self._size = 0
        self._positive_count = 0
        LOGGER.info(
            "ReplayBuffer initialized capacity=%s state_shape=%s action_dim=%s "
            "store_action_masks=%s variable_length=%s sampling_strategy=%s "
            "positive_sample_fraction=%s positive_replay_reserve_fraction=%s "
            "positive_reward_threshold=%s seed=%s",
            self.capacity,
            self.state_shape,
            self.action_dim,
            self.store_action_masks,
            self.variable_length,
            self.sampling_strategy,
            self.positive_sample_fraction,
            self.positive_replay_reserve_fraction,
            self.positive_reward_threshold,
            seed,
        )

        if self.variable_length:
            self._states = np.empty(self.capacity, dtype=object)
            self._next_states = np.empty(self.capacity, dtype=object)
        else:
            self._states = np.empty((self.capacity, *self.state_shape), dtype=self.state_dtype)
            self._next_states = np.empty((self.capacity, *self.state_shape), dtype=self.state_dtype)
        self._actions = np.empty(self.capacity, dtype=np.int64)
        self._rewards = np.empty(self.capacity, dtype=self.reward_dtype)
        self._terminateds = np.empty(self.capacity, dtype=np.bool_)
        self._truncateds = np.empty(self.capacity, dtype=np.bool_)
        self._dones = np.empty(self.capacity, dtype=np.bool_)
        self._n_steps = np.ones(self.capacity, dtype=np.int32)
        self._priorities = np.ones(self.capacity, dtype=np.float32)
        self._terminal_rewards = np.full(self.capacity, np.nan, dtype=np.float32)
        self._terminal_reward_lcbs = np.full(
            self.capacity, np.nan, dtype=np.float32
        )
        self._terminal_strength_deltas = np.full(
            self.capacity, np.nan, dtype=np.float32
        )
        self._terminal_toughness_deltas = np.full(
            self.capacity, np.nan, dtype=np.float32
        )
        self._episode_ids = np.full(self.capacity, -1, dtype=np.int64)
        self._max_priority = 1.0

        if self.store_action_masks:
            if self.variable_length:
                self._action_masks = np.empty(self.capacity, dtype=object)
                self._next_action_masks = np.empty(self.capacity, dtype=object)
            else:
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

    @property
    def positive_count(self) -> int:
        """Number of retained transitions with a positive terminal outcome."""

        return self._positive_count

    @property
    def positive_fraction(self) -> float:
        """Fraction of retained transitions with a positive terminal outcome."""

        return float(self._positive_count / self._size) if self._size else 0.0

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
        n_steps: int = 1,
        terminal_reward: Optional[float] = None,
        terminal_reward_lcb: Optional[float] = None,
        terminal_strength_delta: Optional[float] = None,
        terminal_toughness_delta: Optional[float] = None,
        episode_id: Optional[int] = None,
    ) -> None:
        """Add one transition to the ring buffer."""

        state_array = self._coerce_state(state, "state")
        next_state_array = self._coerce_state(next_state, "next_state")
        action_dim = (
            self._action_dim_for_state(state_array)
            if self.variable_length
            else self.action_dim
        )
        action_value = self._validate_action(action, action_dim=action_dim)
        reward_value = self._validate_finite_scalar(reward, "reward")
        n_steps_value = self._validate_positive_int(n_steps, "n_steps")
        terminal_reward_value = self._optional_finite_scalar(
            terminal_reward, "terminal_reward"
        )
        terminal_reward_lcb_value = self._optional_finite_scalar(
            terminal_reward_lcb, "terminal_reward_lcb"
        )
        if not np.isfinite(terminal_reward_lcb_value):
            terminal_reward_lcb_value = terminal_reward_value
        terminal_strength_value = self._optional_finite_scalar(
            terminal_strength_delta, "terminal_strength_delta"
        )
        terminal_toughness_value = self._optional_finite_scalar(
            terminal_toughness_delta, "terminal_toughness_delta"
        )
        episode_id_value = self._optional_nonnegative_int(episode_id, "episode_id")

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
            current_mask = self._coerce_mask(
                action_mask,
                "action_mask",
                expected_dim=action_dim,
            )
            next_mask = self._coerce_mask(
                next_action_mask,
                "next_action_mask",
                expected_dim=self._action_dim_for_state(next_state_array)
                if self.variable_length
                else self.action_dim,
            )
        else:
            if action_mask is not None or next_action_mask is not None:
                raise ValueError("Mask values were supplied, but store_action_masks=False.")
            current_mask = None
            next_mask = None

        incoming_positive = self._is_positive_terminal_value(
            terminal_reward_lcb_value
        )
        index = self._storage_index_for(incoming_positive)
        replaced_positive = (
            self._is_positive_terminal_value(self._terminal_reward_lcbs[index])
            if self._size == self.capacity
            else False
        )
        self._states[index] = state_array
        self._actions[index] = action_value
        self._rewards[index] = reward_value
        self._next_states[index] = next_state_array
        self._terminateds[index] = terminated_value
        self._truncateds[index] = truncated_value
        self._dones[index] = done_value
        self._n_steps[index] = n_steps_value
        self._priorities[index] = self._max_priority
        self._terminal_rewards[index] = terminal_reward_value
        self._terminal_reward_lcbs[index] = terminal_reward_lcb_value
        self._terminal_strength_deltas[index] = terminal_strength_value
        self._terminal_toughness_deltas[index] = terminal_toughness_value
        self._episode_ids[index] = episode_id_value
        self._positive_count += int(incoming_positive) - int(replaced_positive)

        if self.store_action_masks:
            assert self._action_masks is not None
            assert self._next_action_masks is not None
            assert current_mask is not None and next_mask is not None
            self._action_masks[index] = current_mask
            self._next_action_masks[index] = next_mask

        self._position = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        LOGGER.debug(
            "ReplayBuffer added transition index=%s action=%s reward=%.6f "
            "terminated=%s truncated=%s done=%s size=%s next_position=%s "
            "positive_count=%s positive_fraction=%.6f",
            index,
            action_value,
            reward_value,
            terminated_value,
            truncated_value,
            done_value,
            self._size,
            self._position,
            self._positive_count,
            self.positive_fraction,
        )

    def _is_positive_terminal_value(self, value: float) -> bool:
        return bool(
            np.isfinite(value) and value > self.positive_reward_threshold
        )

    def _storage_index_for(self, incoming_positive: bool) -> int:
        """Choose a ring slot while preserving the configured positive reserve."""

        candidate = self._position
        if self._size < self.capacity or self.positive_replay_reserve_fraction <= 0.0:
            return candidate

        reserve_count = int(
            np.ceil(self.capacity * self.positive_replay_reserve_fraction)
        )
        candidate_positive = self._is_positive_terminal_value(
            self._terminal_reward_lcbs[candidate]
        )
        must_replace_non_positive = candidate_positive and (
            (incoming_positive and self._positive_count < reserve_count)
            or (not incoming_positive and self._positive_count <= reserve_count)
        )
        if not must_replace_non_positive:
            return candidate

        for offset in range(1, self.capacity):
            index = (candidate + offset) % self.capacity
            if not self._is_positive_terminal_value(
                self._terminal_reward_lcbs[index]
            ):
                return index
        return candidate

    def _pool_probabilities(self, pool: np.ndarray) -> np.ndarray:
        if pool.size == 0:
            return np.empty(0, dtype=np.float64)
        if self.sampling_strategy == "prioritized":
            weights = np.power(
                np.maximum(
                    self._priorities[pool].astype(np.float64),
                    self.priority_epsilon,
                ),
                self.priority_alpha,
            )
            weight_sum = float(weights.sum())
            if not np.isfinite(weight_sum) or weight_sum <= 0.0:
                raise FloatingPointError(
                    "Replay priorities produced invalid probabilities."
                )
            return weights / weight_sum
        return np.full(pool.size, 1.0 / pool.size, dtype=np.float64)

    def _draw_from_pool(
        self,
        pool: np.ndarray,
        count: int,
        *,
        replace: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        if count == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
            )
        probabilities = self._pool_probabilities(pool)
        offsets = self._rng.choice(
            pool.size,
            size=count,
            replace=replace,
            p=probabilities,
        )
        offsets = np.asarray(offsets, dtype=np.int64)
        return pool[offsets], probabilities[offsets]

    def _positive_episode_representatives(
        self,
        positive_pool: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """Keep at most one highest-priority row for each known episode."""

        episode_ids = self._episode_ids[positive_pool]
        known_mask = episode_ids >= 0
        if not np.any(known_mask):
            return positive_pool, False

        known_pool = positive_pool[known_mask]
        known_episode_ids = episode_ids[known_mask]
        row_weights = (
            np.power(
                np.maximum(
                    self._priorities[known_pool].astype(np.float64),
                    self.priority_epsilon,
                ),
                self.priority_alpha,
            )
            if self.sampling_strategy == "prioritized"
            else np.ones(known_pool.size, dtype=np.float64)
        )
        order = np.lexsort((-row_weights, known_episode_ids))
        sorted_episode_ids = known_episode_ids[order]
        first_per_episode = np.concatenate(
            (
                np.asarray([True], dtype=np.bool_),
                sorted_episode_ids[1:] != sorted_episode_ids[:-1],
            )
        )
        representatives = known_pool[order[first_per_episode]]
        unknown_rows = positive_pool[~known_mask]
        return np.concatenate((representatives, unknown_rows)), True

    def _sample_indices(
        self,
        batch_size: int,
        *,
        replace: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return indices, marginal probabilities, and within-stratum correction."""

        all_indices = np.arange(self._size, dtype=np.int64)
        positive_mask = (
            np.isfinite(self._terminal_reward_lcbs[: self._size])
            & (
                self._terminal_reward_lcbs[: self._size]
                > self.positive_reward_threshold
            )
        )
        positive_transition_pool = all_indices[positive_mask]
        positive_pool, deduplicated_by_episode = (
            self._positive_episode_representatives(positive_transition_pool)
        )
        other_pool = all_indices[~positive_mask]
        use_strata = (
            self.positive_sample_fraction > 0.0
            and positive_pool.size > 0
            and other_pool.size > 0
        )
        if not use_strata:
            indices, conditional = self._draw_from_pool(
                all_indices,
                batch_size,
                replace=replace,
            )
            return indices, conditional, self._size * conditional

        # A buffer can briefly contain fewer unique positive episodes plus
        # negative rows than a no-replacement batch requires. Preserve the
        # ability to sample in that corner case and expose it in diagnostics.
        if (
            deduplicated_by_episode
            and not replace
            and positive_pool.size + other_pool.size < batch_size
        ):
            positive_pool = positive_transition_pool
            deduplicated_by_episode = False

        positive_draws = int(
            np.ceil(batch_size * self.positive_sample_fraction)
        )
        positive_draws = min(positive_draws, batch_size)
        if deduplicated_by_episode:
            positive_draws = min(positive_draws, int(positive_pool.size))
        if not replace:
            positive_draws = min(positive_draws, int(positive_pool.size))
            other_draws = batch_size - positive_draws
            if other_draws > other_pool.size:
                positive_draws += other_draws - int(other_pool.size)
                other_draws = int(other_pool.size)
        else:
            other_draws = batch_size - positive_draws

        positive_indices, positive_conditional = self._draw_from_pool(
            positive_pool,
            positive_draws,
            replace=replace and not deduplicated_by_episode,
        )
        other_indices, other_conditional = self._draw_from_pool(
            other_pool,
            other_draws,
            replace=replace,
        )
        indices = np.concatenate((positive_indices, other_indices))
        positive_mass = positive_draws / batch_size
        other_mass = other_draws / batch_size
        marginal_probabilities = np.concatenate(
            (
                positive_mass * positive_conditional,
                other_mass * other_conditional,
            )
        )
        within_stratum_correction = np.concatenate(
            (
                positive_pool.size * positive_conditional,
                other_pool.size * other_conditional,
            )
        )
        permutation = self._rng.permutation(batch_size)
        return (
            indices[permutation],
            marginal_probabilities[permutation],
            within_stratum_correction[permutation],
        )

    def sample(
        self,
        batch_size: int,
        *,
        replace: bool = False,
        beta: Optional[float] = None,
        step: Optional[int] = None,
    ) -> ReplayBatch:
        """Sample a uniform or TD-error-prioritized mini-batch."""

        batch_size = self._validate_positive_int(batch_size, "batch_size")
        if not self.can_sample(batch_size, replace=replace):
            raise ValueError(
                f"Cannot sample batch_size={batch_size} with replace={replace}: "
                f"buffer size is {self._size}."
            )

        indices, sampled_probabilities, correction_factors = self._sample_indices(
            batch_size,
            replace=replace,
        )
        if self.sampling_strategy == "prioritized":
            beta_value = self.priority_beta(step or 0) if beta is None else float(beta)
            beta_value = self._validate_unit_interval(
                beta_value,
                "beta",
                lower_inclusive=True,
            )
            importance_weights = np.power(
                correction_factors,
                -beta_value,
            )
            importance_weights /= max(float(importance_weights.max()), np.finfo(float).eps)
        else:
            importance_weights = np.ones(batch_size, dtype=np.float64)
            beta_value = None
        indices = np.asarray(indices, dtype=np.int64)
        LOGGER.debug(
            "ReplayBuffer sampled batch_size=%s replace=%s size=%s indices_preview=%s",
            batch_size,
            replace,
            self._size,
            indices[: min(10, len(indices))].tolist(),
        )

        if self.variable_length:
            states = self._pad_state_batch([self._states[index] for index in indices])
            next_states = self._pad_state_batch([self._next_states[index] for index in indices])
            action_masks = (
                None
                if self._action_masks is None
                else self._pad_mask_batch([self._action_masks[index] for index in indices])
            )
            next_action_masks = (
                None
                if self._next_action_masks is None
                else self._pad_mask_batch([self._next_action_masks[index] for index in indices])
            )
        else:
            states = self._states[indices].copy()
            next_states = self._next_states[indices].copy()
            action_masks = None if self._action_masks is None else self._action_masks[indices].copy()
            next_action_masks = (
                None
                if self._next_action_masks is None
                else self._next_action_masks[indices].copy()
            )

        return ReplayBatch(
            states=states,
            actions=self._actions[indices].copy(),
            rewards=self._rewards[indices].copy(),
            next_states=next_states,
            terminateds=self._terminateds[indices].copy(),
            truncateds=self._truncateds[indices].copy(),
            dones=self._dones[indices].copy(),
            action_masks=action_masks,
            next_action_masks=next_action_masks,
            indices=indices.copy(),
            n_steps=self._n_steps[indices].astype(np.int64, copy=True),
            importance_weights=np.asarray(importance_weights, dtype=np.float32),
            sampling_probabilities=np.asarray(sampled_probabilities, dtype=np.float32),
            priority_beta=beta_value,
            terminal_rewards=self._terminal_rewards[indices].copy(),
            terminal_reward_lcbs=self._terminal_reward_lcbs[indices].copy(),
            terminal_strength_deltas=self._terminal_strength_deltas[indices].copy(),
            terminal_toughness_deltas=self._terminal_toughness_deltas[indices].copy(),
            episode_ids=self._episode_ids[indices].copy(),
        )

    def priority_beta(self, step: int) -> float:
        """Linearly anneal importance-sampling beta toward one."""

        progress = min(1.0, max(0, int(step)) / self.priority_beta_steps)
        return float(
            self.priority_beta_start
            + progress * (self.priority_beta_end - self.priority_beta_start)
        )

    def update_priorities(self, indices: Any, td_errors: Any) -> None:
        """Update sampled priorities from absolute learner TD errors."""

        if self.sampling_strategy != "prioritized":
            return
        index_array = np.asarray(indices, dtype=np.int64)
        error_array = np.asarray(td_errors, dtype=np.float64)
        if index_array.ndim != 1 or error_array.shape != index_array.shape:
            raise ValueError("indices and td_errors must be matching one-dimensional arrays.")
        if np.any(index_array < 0) or np.any(index_array >= self._size):
            raise ValueError("priority update contains an out-of-range replay index.")
        if not np.all(np.isfinite(error_array)):
            raise ValueError("td_errors contains NaN or infinity.")
        priorities = np.abs(error_array) + self.priority_epsilon
        self._priorities[index_array] = priorities.astype(np.float32)
        if priorities.size:
            self._max_priority = max(self._max_priority, float(priorities.max()))

    def terminal_outcome_diagnostics(
        self,
        *,
        indices: Optional[Any] = None,
        top_priority_fraction: Optional[float] = None,
    ) -> Dict[str, float | int]:
        """Summarize terminal-outcome signs for replay, top-priority, or sampled rows."""

        if indices is not None and top_priority_fraction is not None:
            raise ValueError("Specify indices or top_priority_fraction, not both.")
        if indices is None:
            selected = np.arange(self._size, dtype=np.int64)
        else:
            selected = np.asarray(indices, dtype=np.int64).reshape(-1)
            if np.any(selected < 0) or np.any(selected >= self._size):
                raise ValueError("diagnostic indices contain an out-of-range replay index.")
        if top_priority_fraction is not None:
            fraction = float(top_priority_fraction)
            if not 0.0 < fraction <= 1.0:
                raise ValueError("top_priority_fraction must be within (0, 1].")
            count = min(self._size, max(1, int(np.ceil(self._size * fraction))))
            if count:
                selected = np.argpartition(
                    self._priorities[: self._size], -count
                )[-count:]

        terminal_rewards = self._terminal_rewards[selected]
        terminal_reward_lcbs = self._terminal_reward_lcbs[selected]
        terminal_mask = np.isfinite(terminal_rewards)
        positive_terminal_mask = np.isfinite(terminal_reward_lcbs) & (
            terminal_reward_lcbs > self.positive_reward_threshold
        )
        terminal_count = int(terminal_mask.sum())
        positive_terminal_count = int(positive_terminal_mask.sum())
        selected_count = int(selected.size)
        result: Dict[str, float | int] = {
            "selected_count": selected_count,
            "terminal_count": terminal_count,
            "terminal_fraction": (
                float(terminal_count / selected_count) if selected_count else 0.0
            ),
            "positive_terminal_count": positive_terminal_count,
            "positive_terminal_fraction": (
                float(positive_terminal_count / selected_count)
                if selected_count
                else 0.0
            ),
            "positive_reward_lower_bound": self.positive_reward_threshold,
            "positive_reward_threshold": self.positive_reward_threshold,
        }
        known_episode_ids = self._episode_ids[selected]
        known_episode_ids = known_episode_ids[known_episode_ids >= 0]
        unique_episode_count = int(np.unique(known_episode_ids).size)
        result["known_episode_transition_count"] = int(known_episode_ids.size)
        result["unique_episode_count"] = unique_episode_count
        result["episode_duplicate_fraction"] = (
            float(1.0 - unique_episode_count / known_episode_ids.size)
            if known_episode_ids.size
            else 0.0
        )
        positive_episode_ids = self._episode_ids[selected][positive_terminal_mask]
        positive_episode_ids = positive_episode_ids[positive_episode_ids >= 0]
        positive_unique_episode_count = int(np.unique(positive_episode_ids).size)
        result["positive_known_episode_transition_count"] = int(
            positive_episode_ids.size
        )
        result["positive_unique_episode_count"] = positive_unique_episode_count
        result["positive_episode_duplicate_fraction"] = (
            float(1.0 - positive_unique_episode_count / positive_episode_ids.size)
            if positive_episode_ids.size
            else 0.0
        )
        for name, values in (
            ("terminal_reward", terminal_rewards),
            ("terminal_reward_lcb", terminal_reward_lcbs),
            ("strength", self._terminal_strength_deltas[selected]),
            ("toughness", self._terminal_toughness_deltas[selected]),
        ):
            finite = values[np.isfinite(values)]
            denominator = int(finite.size)
            result[f"{name}_count"] = denominator
            result[f"{name}_positive_fraction"] = (
                float(np.mean(finite > 0.0)) if denominator else 0.0
            )
            result[f"{name}_negative_fraction"] = (
                float(np.mean(finite < 0.0)) if denominator else 0.0
            )
            result[f"{name}_zero_fraction"] = (
                float(np.mean(finite == 0.0)) if denominator else 0.0
            )
            if denominator:
                result[f"{name}_mean"] = float(np.mean(finite))
        return result

    def clear(self) -> None:
        """Remove transitions without reallocating the arrays."""

        self._position = 0
        self._size = 0
        self._max_priority = 1.0
        self._positive_count = 0

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
            "variable_length": self.variable_length,
            "sampling_strategy": self.sampling_strategy,
            "priority_alpha": self.priority_alpha,
            "priority_beta_start": self.priority_beta_start,
            "priority_beta_end": self.priority_beta_end,
            "priority_beta_steps": self.priority_beta_steps,
            "priority_epsilon": self.priority_epsilon,
            "positive_sample_fraction": self.positive_sample_fraction,
            "positive_replay_reserve_fraction": (
                self.positive_replay_reserve_fraction
            ),
            "positive_reward_threshold": self.positive_reward_threshold,
            "max_priority": self._max_priority,
            "position": self._position,
            "size": size,
            "states": self._states[:size].copy(),
            "actions": self._actions[:size].copy(),
            "rewards": self._rewards[:size].copy(),
            "next_states": self._next_states[:size].copy(),
            "terminateds": self._terminateds[:size].copy(),
            "truncateds": self._truncateds[:size].copy(),
            "dones": self._dones[:size].copy(),
            "n_steps": self._n_steps[:size].copy(),
            "priorities": self._priorities[:size].copy(),
            "terminal_rewards": self._terminal_rewards[:size].copy(),
            "terminal_reward_lcbs": self._terminal_reward_lcbs[:size].copy(),
            "terminal_strength_deltas": self._terminal_strength_deltas[:size].copy(),
            "terminal_toughness_deltas": self._terminal_toughness_deltas[:size].copy(),
            "episode_ids": self._episode_ids[:size].copy(),
            "action_masks": None if self._action_masks is None else self._action_masks[:size].copy(),
            "next_action_masks": None
            if self._next_action_masks is None
            else self._next_action_masks[:size].copy(),
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        """Restore a snapshot into a compatible buffer."""

        format_version = int(payload.get("format_version", -1))
        if format_version not in (1, 2, 3, 4, self.FORMAT_VERSION):
            raise ValueError("Unsupported replay-buffer format version.")

        self._assert_compatible_snapshot(payload)
        size = int(payload["size"])
        position = int(payload["position"])
        if not 0 <= size <= self.capacity:
            raise ValueError(f"Snapshot size must be within 0..{self.capacity}.")
        if not 0 <= position < self.capacity:
            raise ValueError(f"Snapshot position must be within 0..{self.capacity - 1}.")

        if self.variable_length:
            self._states[:size] = self._snapshot_object_array(payload["states"], size, "states")
        else:
            self._states[:size] = self._snapshot_array(payload["states"], (size, *self.state_shape), self.state_dtype, "states")
        self._actions[:size] = self._snapshot_array(payload["actions"], (size,), np.dtype(np.int64), "actions")
        self._rewards[:size] = self._snapshot_array(payload["rewards"], (size,), self.reward_dtype, "rewards")
        if self.variable_length:
            self._next_states[:size] = self._snapshot_object_array(payload["next_states"], size, "next_states")
        else:
            self._next_states[:size] = self._snapshot_array(payload["next_states"], (size, *self.state_shape), self.state_dtype, "next_states")
        self._terminateds[:size] = self._snapshot_array(payload["terminateds"], (size,), np.dtype(np.bool_), "terminateds")
        self._truncateds[:size] = self._snapshot_array(payload["truncateds"], (size,), np.dtype(np.bool_), "truncateds")
        self._dones[:size] = self._snapshot_array(payload["dones"], (size,), np.dtype(np.bool_), "dones")
        if format_version >= 2:
            self._n_steps[:size] = self._snapshot_array(
                payload["n_steps"], (size,), np.dtype(np.int32), "n_steps"
            )
            self._priorities[:size] = self._snapshot_array(
                payload["priorities"], (size,), np.dtype(np.float32), "priorities"
            )
            self._max_priority = float(payload.get("max_priority", 1.0))
        else:
            self._n_steps[:size] = 1
            self._priorities[:size] = 1.0
            self._max_priority = 1.0
        if format_version >= 3:
            self._terminal_rewards[:size] = self._snapshot_array(
                payload["terminal_rewards"], (size,), np.dtype(np.float32), "terminal_rewards"
            )
            self._terminal_strength_deltas[:size] = self._snapshot_array(
                payload["terminal_strength_deltas"],
                (size,),
                np.dtype(np.float32),
                "terminal_strength_deltas",
            )
            self._terminal_toughness_deltas[:size] = self._snapshot_array(
                payload["terminal_toughness_deltas"],
                (size,),
                np.dtype(np.float32),
                "terminal_toughness_deltas",
            )
        else:
            self._terminal_rewards[:size] = np.nan
            self._terminal_strength_deltas[:size] = np.nan
            self._terminal_toughness_deltas[:size] = np.nan
        if format_version >= 5:
            self._terminal_reward_lcbs[:size] = self._snapshot_array(
                payload["terminal_reward_lcbs"],
                (size,),
                np.dtype(np.float32),
                "terminal_reward_lcbs",
            )
            self._episode_ids[:size] = self._snapshot_array(
                payload["episode_ids"],
                (size,),
                np.dtype(np.int64),
                "episode_ids",
            )
        else:
            self._terminal_reward_lcbs[:size] = self._terminal_rewards[:size]
            self._episode_ids[:size] = -1
        self._positive_count = int(
            np.sum(
                np.isfinite(self._terminal_reward_lcbs[:size])
                & (
                    self._terminal_reward_lcbs[:size]
                    > self.positive_reward_threshold
                )
            )
        )

        if self.store_action_masks:
            assert self._action_masks is not None and self._next_action_masks is not None
            if self.variable_length:
                self._action_masks[:size] = self._snapshot_object_array(payload["action_masks"], size, "action_masks")
                self._next_action_masks[:size] = self._snapshot_object_array(payload["next_action_masks"], size, "next_action_masks")
            else:
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
        started = time.perf_counter()
        LOGGER.info("ReplayBuffer save started path=%s size=%s capacity=%s", path, self._size, self.capacity)
        payload = self.state_dict()
        metadata = {
            key: payload[key]
            for key in (
                "format_version", "capacity", "state_shape", "action_dim",
                "state_dtype", "reward_dtype", "store_action_masks",
                "variable_length", "sampling_strategy", "priority_alpha",
                "priority_beta_start", "priority_beta_end", "priority_beta_steps",
                "priority_epsilon", "positive_sample_fraction",
                "positive_replay_reserve_fraction", "positive_reward_threshold",
                "max_priority", "position", "size", "rng_state"
            )
        }
        temporary_path = path.with_name(f".{path.name}.tmp.npz")
        try:
            np.savez_compressed(
                temporary_path,
                metadata=np.asarray(metadata, dtype=object),
                states=payload["states"],
                actions=payload["actions"],
                rewards=payload["rewards"],
                next_states=payload["next_states"],
                terminateds=payload["terminateds"],
                truncateds=payload["truncateds"],
                dones=payload["dones"],
                n_steps=payload["n_steps"],
                priorities=payload["priorities"],
                terminal_rewards=payload["terminal_rewards"],
                terminal_reward_lcbs=payload["terminal_reward_lcbs"],
                terminal_strength_deltas=payload["terminal_strength_deltas"],
                terminal_toughness_deltas=payload["terminal_toughness_deltas"],
                episode_ids=payload["episode_ids"],
                action_masks=np.asarray([], dtype=np.bool_)
                if payload["action_masks"] is None
                else payload["action_masks"],
                next_action_masks=np.asarray([], dtype=np.bool_)
                if payload["next_action_masks"] is None
                else payload["next_action_masks"],
            )
            temporary_path.replace(path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        LOGGER.info(
            "ReplayBuffer save complete path=%s bytes=%s elapsed_sec=%.3f",
            path,
            path.stat().st_size,
            time.perf_counter() - started,
        )

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
                variable_length=bool(metadata.get("variable_length", False)),
                sampling_strategy=str(metadata.get("sampling_strategy", "uniform")),
                priority_alpha=float(metadata.get("priority_alpha", 0.6)),
                priority_beta_start=float(metadata.get("priority_beta_start", 0.4)),
                priority_beta_end=float(metadata.get("priority_beta_end", 1.0)),
                priority_beta_steps=int(metadata.get("priority_beta_steps", 1_000_000)),
                priority_epsilon=float(metadata.get("priority_epsilon", 1e-6)),
                positive_sample_fraction=float(
                    metadata.get("positive_sample_fraction", 0.0)
                ),
                positive_replay_reserve_fraction=float(
                    metadata.get("positive_replay_reserve_fraction", 0.0)
                ),
                positive_reward_threshold=float(
                    metadata.get("positive_reward_threshold", 0.0)
                ),
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
                n_steps=archive["n_steps"] if "n_steps" in archive else np.ones(int(metadata["size"]), dtype=np.int32),
                priorities=archive["priorities"] if "priorities" in archive else np.ones(int(metadata["size"]), dtype=np.float32),
                terminal_rewards=archive["terminal_rewards"] if "terminal_rewards" in archive else np.full(int(metadata["size"]), np.nan, dtype=np.float32),
                terminal_reward_lcbs=archive["terminal_reward_lcbs"] if "terminal_reward_lcbs" in archive else archive["terminal_rewards"] if "terminal_rewards" in archive else np.full(int(metadata["size"]), np.nan, dtype=np.float32),
                terminal_strength_deltas=archive["terminal_strength_deltas"] if "terminal_strength_deltas" in archive else np.full(int(metadata["size"]), np.nan, dtype=np.float32),
                terminal_toughness_deltas=archive["terminal_toughness_deltas"] if "terminal_toughness_deltas" in archive else np.full(int(metadata["size"]), np.nan, dtype=np.float32),
                episode_ids=archive["episode_ids"] if "episode_ids" in archive else np.full(int(metadata["size"]), -1, dtype=np.int64),
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
    def _validate_unit_interval(
        value: Any,
        name: str,
        *,
        lower_inclusive: bool,
    ) -> float:
        value = float(value)
        lower_ok = value >= 0.0 if lower_inclusive else value > 0.0
        if not np.isfinite(value) or not lower_ok or value > 1.0:
            bracket = "[0, 1]" if lower_inclusive else "(0, 1]"
            raise ValueError(f"{name} must be within {bracket}.")
        return value

    @staticmethod
    def _validate_state_shape(state_shape: Sequence[int]) -> Tuple[int, ...]:
        shape = tuple(int(dimension) for dimension in state_shape)
        if not shape or any(dimension <= 0 for dimension in shape):
            raise ValueError("state_shape must contain positive dimensions.")
        return shape

    def _validate_action(self, action: Any, *, action_dim: Optional[int] = None) -> int:
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer.")
        action = int(action)
        upper = self.action_dim if action_dim is None else int(action_dim)
        if not 0 <= action < upper:
            raise ValueError(f"action must be within 0..{upper - 1}.")
        return action

    @staticmethod
    def _validate_finite_scalar(value: Any, name: str) -> float:
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")
        return value

    @classmethod
    def _optional_finite_scalar(cls, value: Any, name: str) -> float:
        if value is None:
            return float("nan")
        return cls._validate_finite_scalar(value, name)

    @staticmethod
    def _optional_nonnegative_int(value: Any, name: str) -> int:
        if value is None:
            return -1
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer or None.")
        value = int(value)
        if value < 0:
            raise ValueError(f"{name} must be >= 0.")
        return value

    def _coerce_state(self, state: Any, name: str) -> np.ndarray:
        array = np.asarray(state, dtype=self.state_dtype)
        if self.variable_length:
            if array.ndim != len(self.state_shape):
                raise ValueError(
                    f"{name} must have {len(self.state_shape)} dimensions, got {array.shape}."
                )
            if tuple(array.shape[1:]) != self.state_shape[1:]:
                raise ValueError(
                    f"{name} must have trailing shape {self.state_shape[1:]}, got {array.shape[1:]}."
                )
        elif array.shape != self.state_shape:
            raise ValueError(f"{name} must have shape {self.state_shape}, got {array.shape}.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains NaN or infinity.")
        return array

    def _coerce_mask(
        self,
        mask: Optional[Any],
        name: str,
        *,
        expected_dim: Optional[int] = None,
    ) -> np.ndarray:
        expected = self.action_dim if expected_dim is None else int(expected_dim)
        if mask is None:
            return np.ones(expected, dtype=np.bool_)
        array = np.asarray(mask, dtype=np.bool_)
        if self.variable_length:
            if array.shape != (expected,):
                raise ValueError(f"{name} must have shape {(expected,)}, got {array.shape}.")
        elif array.shape != (self.action_dim,):
            raise ValueError(f"{name} must have shape {(self.action_dim,)}, got {array.shape}.")
        return array

    @staticmethod
    def _action_dim_for_state(state: np.ndarray) -> int:
        if state.ndim == 2:
            return int(state.shape[0]) * 20
        return int(np.prod(state.shape))

    def _pad_state_batch(self, states: Sequence[np.ndarray]) -> np.ndarray:
        if not states:
            raise ValueError("Cannot pad an empty state batch.")
        max_length = max(int(state.shape[0]) for state in states)
        trailing_shape = states[0].shape[1:]
        padded = np.zeros((len(states), max_length, *trailing_shape), dtype=self.state_dtype)
        for row, state in enumerate(states):
            padded[row, : state.shape[0], ...] = state
        return padded

    @staticmethod
    def _pad_mask_batch(masks: Sequence[np.ndarray]) -> np.ndarray:
        if not masks:
            raise ValueError("Cannot pad an empty mask batch.")
        max_length = max(int(mask.shape[0]) for mask in masks)
        padded = np.zeros((len(masks), max_length), dtype=np.bool_)
        for row, mask in enumerate(masks):
            padded[row, : mask.shape[0]] = mask
        return padded

    def _assert_compatible_snapshot(self, payload: Mapping[str, Any]) -> None:
        expected = (
            self.capacity,
            self.state_shape,
            self.action_dim,
            self.state_dtype.str,
            self.reward_dtype.str,
            self.store_action_masks,
            self.variable_length,
        )
        actual = (
            int(payload["capacity"]),
            tuple(payload["state_shape"]),
            int(payload["action_dim"]),
            np.dtype(payload["state_dtype"]).str,
            np.dtype(payload["reward_dtype"]).str,
            bool(payload["store_action_masks"]),
            bool(payload.get("variable_length", False)),
        )
        if actual != expected:
            raise ValueError("Snapshot configuration is incompatible with this buffer.")
        if int(payload.get("format_version", 1)) >= 2:
            saved_replay_config = (
                str(payload.get("sampling_strategy", "uniform")),
                float(payload.get("priority_alpha", 0.6)),
                float(payload.get("priority_beta_start", 0.4)),
                float(payload.get("priority_beta_end", 1.0)),
                int(payload.get("priority_beta_steps", 1_000_000)),
                float(payload.get("priority_epsilon", 1e-6)),
            )
            current_replay_config = (
                self.sampling_strategy,
                self.priority_alpha,
                self.priority_beta_start,
                self.priority_beta_end,
                self.priority_beta_steps,
                self.priority_epsilon,
            )
            if saved_replay_config != current_replay_config:
                raise ValueError("Snapshot replay-sampling configuration is incompatible.")
        if int(payload.get("format_version", 1)) >= 4:
            saved_positive_config = (
                float(payload.get("positive_sample_fraction", 0.0)),
                float(payload.get("positive_replay_reserve_fraction", 0.0)),
                float(payload.get("positive_reward_threshold", 0.0)),
            )
            current_positive_config = (
                self.positive_sample_fraction,
                self.positive_replay_reserve_fraction,
                self.positive_reward_threshold,
            )
            if saved_positive_config != current_positive_config:
                raise ValueError(
                    "Snapshot positive-replay configuration is incompatible."
                )

    @staticmethod
    def _snapshot_array(value: Any, shape: Tuple[int, ...], dtype: np.dtype, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=dtype)
        if array.shape != shape:
            raise ValueError(f"Snapshot array '{name}' must have shape {shape}, got {array.shape}.")
        return array

    @staticmethod
    def _snapshot_object_array(value: Any, size: int, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=object)
        if array.shape != (size,):
            raise ValueError(f"Snapshot object array '{name}' must have shape {(size,)}, got {array.shape}.")
        return array


UniformReplayBuffer = ReplayBuffer
