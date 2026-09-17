"""Episode-local n-step transition aggregation."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Mapping, Optional

import numpy as np


def terminal_outcome_fields(
    info: Mapping[str, Any], *, done: bool
) -> Dict[str, Optional[float]]:
    """Extract terminal replay metadata, with point-reward fallback for LCB."""

    empty = {
        "terminal_reward": None,
        "terminal_reward_lcb": None,
        "terminal_strength_delta": None,
        "terminal_toughness_delta": None,
    }
    if not done:
        return empty
    metrics = info.get("terminal_reward_metrics")
    if not isinstance(metrics, Mapping):
        metrics = {}

    def finite_or_none(value: Any) -> Optional[float]:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if np.isfinite(result) else None

    terminal_reward = finite_or_none(info.get("terminal_reward"))
    terminal_reward_lcb = finite_or_none(info.get("terminal_reward_lcb"))
    if terminal_reward_lcb is None:
        terminal_reward_lcb = terminal_reward
    return {
        "terminal_reward": terminal_reward,
        "terminal_reward_lcb": terminal_reward_lcb,
        "terminal_strength_delta": finite_or_none(
            metrics.get("delta_normalized_strength")
        ),
        "terminal_toughness_delta": finite_or_none(
            metrics.get("delta_normalized_toughness")
        ),
    }


class NStepTransitionAccumulator:
    """Convert ordered one-step transitions into discounted n-step returns."""

    def __init__(self, n_step: int = 1, gamma: float = 0.99) -> None:
        if isinstance(n_step, bool) or int(n_step) <= 0:
            raise ValueError("n_step must be a positive integer.")
        if not 0.0 <= float(gamma) <= 1.0:
            raise ValueError("gamma must be within [0, 1].")
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self._queue: Deque[Dict[str, Any]] = deque()

    def __len__(self) -> int:
        return len(self._queue)

    def clear(self) -> None:
        self._queue.clear()

    def append(self, transition: Mapping[str, Any]) -> list[Dict[str, Any]]:
        """Append one ordered transition and return every replay row now ready."""

        row = dict(transition)
        required = (
            "state",
            "action",
            "reward",
            "next_state",
            "terminated",
            "truncated",
            "action_mask",
            "next_action_mask",
        )
        missing = [name for name in required if name not in row]
        if missing:
            raise ValueError(f"n-step transition is missing fields: {missing}.")
        if not np.isfinite(float(row["reward"])):
            raise ValueError("n-step transition reward must be finite.")

        row["terminated"] = bool(row["terminated"])
        row["truncated"] = bool(row["truncated"])
        self._queue.append(row)

        ready: list[Dict[str, Any]] = []
        if row["terminated"] or row["truncated"]:
            while self._queue:
                ready.append(self._aggregate_prefix())
                self._queue.popleft()
        elif len(self._queue) >= self.n_step:
            ready.append(self._aggregate_prefix())
            self._queue.popleft()
        return ready

    def _aggregate_prefix(self) -> Dict[str, Any]:
        first = self._queue[0]
        reward = 0.0
        final = first
        steps = 0
        for steps, transition in enumerate(self._queue, start=1):
            reward += (self.gamma ** (steps - 1)) * float(transition["reward"])
            final = transition
            if (
                steps >= self.n_step
                or transition["terminated"]
                or transition["truncated"]
            ):
                break

        return {
            "state": first["state"],
            "action": int(first["action"]),
            "reward": float(reward),
            "next_state": final["next_state"],
            "terminated": bool(final["terminated"]),
            "truncated": bool(final["truncated"]),
            "action_mask": first["action_mask"],
            "next_action_mask": final["next_action_mask"],
            "n_steps": int(steps),
            "terminal_reward": final.get("terminal_reward"),
            "terminal_reward_lcb": final.get("terminal_reward_lcb"),
            "terminal_strength_delta": final.get("terminal_strength_delta"),
            "terminal_toughness_delta": final.get("terminal_toughness_delta"),
            "episode_id": first.get(
                "episode_id",
                first.get("episode", final.get("episode_id", final.get("episode"))),
            ),
        }
