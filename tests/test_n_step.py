import numpy as np
import pytest

from model.replay_buffer_module import (
    NStepTransitionAccumulator,
    terminal_outcome_fields,
)


def transition(index: int, reward: float, *, done: bool = False) -> dict:
    return {
        "state": np.asarray([index], dtype=np.float32),
        "action": index,
        "reward": reward,
        "next_state": np.asarray([index + 1], dtype=np.float32),
        "terminated": False,
        "truncated": done,
        "action_mask": np.ones(4, dtype=np.bool_),
        "next_action_mask": np.ones(4, dtype=np.bool_),
    }


def test_n_step_accumulates_discounted_reward() -> None:
    accumulator = NStepTransitionAccumulator(n_step=3, gamma=0.5)

    assert accumulator.append(transition(0, 1.0)) == []
    assert accumulator.append(transition(1, 2.0)) == []
    ready = accumulator.append(transition(2, 4.0))

    assert len(ready) == 1
    assert ready[0]["reward"] == pytest.approx(3.0)
    assert ready[0]["n_steps"] == 3
    np.testing.assert_array_equal(ready[0]["state"], [0.0])
    np.testing.assert_array_equal(ready[0]["next_state"], [3.0])
    assert not ready[0]["truncated"]


def test_terminal_transition_flushes_short_prefixes() -> None:
    accumulator = NStepTransitionAccumulator(n_step=3, gamma=0.5)
    accumulator.append(transition(0, 1.0))
    accumulator.append(transition(1, 2.0))

    ready = accumulator.append(transition(2, 4.0, done=True))

    assert [row["reward"] for row in ready] == pytest.approx([3.0, 4.0, 4.0])
    assert [row["n_steps"] for row in ready] == [3, 2, 1]
    assert all(row["truncated"] for row in ready)
    assert len(accumulator) == 0


def test_terminal_outcome_metadata_is_propagated_to_flushed_prefixes() -> None:
    accumulator = NStepTransitionAccumulator(n_step=3, gamma=0.5)
    accumulator.append(transition(0, 1.0))
    final = transition(1, 2.0, done=True)
    final.update(
        terminal_reward=-0.4,
        terminal_reward_lcb=-0.7,
        terminal_strength_delta=0.2,
        terminal_toughness_delta=-1.0,
        episode_id=42,
    )

    ready = accumulator.append(final)

    assert [row["terminal_reward"] for row in ready] == pytest.approx([-0.4, -0.4])
    assert [row["terminal_reward_lcb"] for row in ready] == pytest.approx([-0.7, -0.7])
    assert [row["terminal_toughness_delta"] for row in ready] == pytest.approx([-1.0, -1.0])
    assert [row["episode_id"] for row in ready] == [42, 42]


def test_terminal_outcome_fields_use_explicit_lcb_then_point_fallback() -> None:
    metrics = {
        "delta_normalized_strength": 0.4,
        "delta_normalized_toughness": -0.2,
    }

    explicit = terminal_outcome_fields(
        {
            "terminal_reward": 0.3,
            "terminal_reward_lcb": 0.1,
            "terminal_reward_metrics": metrics,
        },
        done=True,
    )
    fallback = terminal_outcome_fields(
        {"terminal_reward": 0.3, "terminal_reward_metrics": metrics},
        done=True,
    )

    assert explicit["terminal_reward_lcb"] == pytest.approx(0.1)
    assert fallback["terminal_reward_lcb"] == pytest.approx(0.3)
    assert explicit["terminal_strength_delta"] == pytest.approx(0.4)


def test_one_step_mode_preserves_transition() -> None:
    accumulator = NStepTransitionAccumulator(n_step=1, gamma=0.99)
    source = transition(0, -0.25)

    ready = accumulator.append(source)

    assert len(ready) == 1
    assert ready[0]["reward"] == pytest.approx(-0.25)
    assert ready[0]["n_steps"] == 1
