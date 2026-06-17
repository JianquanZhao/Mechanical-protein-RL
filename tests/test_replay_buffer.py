from pathlib import Path

import numpy as np
import pytest

from model.replay_buffer_module.replay_buffer import ReplayBatch, ReplayBuffer


def make_transition(
    value: float,
    *,
    action: int = 0,
    reward: float = 1.0,
    terminated: bool = False,
    truncated: bool = False,
    action_dim: int = 4,
):
    state = np.asarray([value, value + 0.1, value + 0.2], dtype=np.float32)
    next_state = state + 1.0
    action_mask = np.ones(action_dim, dtype=bool)
    next_action_mask = np.ones(action_dim, dtype=bool)
    next_action_mask[-1] = False
    return {
        "state": state,
        "action": action,
        "reward": reward,
        "next_state": next_state,
        "terminated": terminated,
        "truncated": truncated,
        "action_mask": action_mask,
        "next_action_mask": next_action_mask,
    }


def make_buffer(*, capacity: int = 5, seed: int = 123, store_action_masks: bool = True) -> ReplayBuffer:
    return ReplayBuffer(
        capacity=capacity,
        state_shape=(3,),
        action_dim=4,
        seed=seed,
        store_action_masks=store_action_masks,
    )


def test_add_and_sample_shapes() -> None:
    buffer = make_buffer()
    for i in range(4):
        buffer.add(**make_transition(float(i), action=i))
    batch = buffer.sample(3)
    assert isinstance(batch, ReplayBatch)
    assert batch.batch_size == 3
    assert batch.states.shape == (3, 3)
    assert batch.actions.shape == (3,)
    assert batch.rewards.shape == (3,)
    assert batch.next_states.shape == (3, 3)
    assert batch.terminateds.shape == (3,)
    assert batch.truncateds.shape == (3,)
    assert batch.dones.shape == (3,)
    assert batch.action_masks is not None and batch.action_masks.shape == (3, 4)
    assert batch.next_action_masks is not None and batch.next_action_masks.shape == (3, 4)
    assert batch.indices.shape == (3,)


def test_default_masks_are_all_true() -> None:
    buffer = make_buffer()
    transition = make_transition(0.0)
    transition.pop("action_mask")
    transition.pop("next_action_mask")
    buffer.add(**transition)
    batch = buffer.sample(1)
    assert batch.action_masks is not None and np.all(batch.action_masks)
    assert batch.next_action_masks is not None and np.all(batch.next_action_masks)


def test_done_is_inferred_from_terminated_or_truncated() -> None:
    buffer = make_buffer()
    buffer.add(**make_transition(0.0, terminated=True))
    buffer.add(**make_transition(1.0, truncated=True))
    batch = buffer.sample(2)
    assert np.all(batch.dones)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"terminated": True, "truncated": False, "done": False},
        {"terminated": False, "truncated": False, "done": True},
    ],
)
def test_explicit_done_must_match_terminated_or_truncated(kwargs) -> None:
    buffer = make_buffer()
    transition = make_transition(0.0)
    transition.update(kwargs)
    with pytest.raises(ValueError, match="inconsistent"):
        buffer.add(**transition)


def test_ring_buffer_overwrites_oldest_transitions() -> None:
    buffer = make_buffer(capacity=3)
    for i in range(5):
        buffer.add(**make_transition(float(i), action=i % 4))
    assert len(buffer) == 3
    assert buffer.is_full
    assert buffer.position == 2
    assert set(float(value) for value in buffer._states[:, 0]) == {2.0, 3.0, 4.0}


def test_sample_is_reproducible_for_same_seed() -> None:
    left = make_buffer(seed=77)
    right = make_buffer(seed=77)
    for i in range(5):
        transition = make_transition(float(i), action=i % 4)
        left.add(**transition)
        right.add(**transition)
    left_batch = left.sample(4)
    right_batch = right.sample(4)
    np.testing.assert_array_equal(left_batch.indices, right_batch.indices)
    np.testing.assert_array_equal(left_batch.states, right_batch.states)


def test_sample_without_replacement_has_unique_indices() -> None:
    buffer = make_buffer()
    for i in range(5):
        buffer.add(**make_transition(float(i), action=i % 4))
    batch = buffer.sample(5, replace=False)
    assert len(set(batch.indices.tolist())) == 5


def test_sample_with_replacement_works_with_single_transition() -> None:
    buffer = make_buffer()
    buffer.add(**make_transition(0.0))
    batch = buffer.sample(4, replace=True)
    assert batch.batch_size == 4
    assert np.all(batch.indices == 0)


def test_cannot_sample_too_many_without_replacement() -> None:
    buffer = make_buffer()
    buffer.add(**make_transition(0.0))
    assert not buffer.can_sample(2)
    with pytest.raises(ValueError, match="Cannot sample"):
        buffer.sample(2)


def test_clear_resets_size_and_position() -> None:
    buffer = make_buffer()
    buffer.add(**make_transition(0.0))
    buffer.add(**make_transition(1.0))
    buffer.clear()
    assert len(buffer) == 0
    assert buffer.position == 0
    assert not buffer.is_full


def test_samples_are_copies_not_views() -> None:
    buffer = make_buffer()
    buffer.add(**make_transition(0.0))
    batch = buffer.sample(1)
    batch.states[0, 0] = 999.0
    assert buffer._states[0, 0] != pytest.approx(999.0)


def test_masks_can_be_disabled() -> None:
    buffer = make_buffer(store_action_masks=False)
    transition = make_transition(0.0)
    transition.pop("action_mask")
    transition.pop("next_action_mask")
    buffer.add(**transition)
    batch = buffer.sample(1)
    assert batch.action_masks is None
    assert batch.next_action_masks is None


def test_disabled_mask_storage_rejects_supplied_masks() -> None:
    buffer = make_buffer(store_action_masks=False)
    with pytest.raises(ValueError, match="store_action_masks=False"):
        buffer.add(**make_transition(0.0))


@pytest.mark.parametrize(
    "bad_state",
    [np.asarray([1.0, 2.0]), np.asarray([1.0, np.nan, 3.0])],
)
def test_rejects_invalid_state(bad_state) -> None:
    buffer = make_buffer()
    transition = make_transition(0.0)
    transition["state"] = bad_state
    with pytest.raises(ValueError):
        buffer.add(**transition)


@pytest.mark.parametrize("bad_action", [-1, 4, 1.5, True])
def test_rejects_invalid_action(bad_action) -> None:
    buffer = make_buffer()
    transition = make_transition(0.0)
    transition["action"] = bad_action
    with pytest.raises((TypeError, ValueError)):
        buffer.add(**transition)


def test_rejects_invalid_mask_shape() -> None:
    buffer = make_buffer()
    transition = make_transition(0.0)
    transition["next_action_mask"] = np.ones(3, dtype=bool)
    with pytest.raises(ValueError, match="next_action_mask"):
        buffer.add(**transition)


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    buffer = make_buffer(capacity=5, seed=19)
    for i in range(4):
        buffer.add(
            **make_transition(
                float(i),
                action=i % 4,
                reward=float(i) / 2.0,
                terminated=(i == 3),
            )
        )
    _ = buffer.sample(2)
    snapshot_path = tmp_path / "replay_buffer.npz"
    buffer.save(snapshot_path)
    restored = ReplayBuffer.load(snapshot_path)

    assert restored.capacity == buffer.capacity
    assert restored.state_shape == buffer.state_shape
    assert restored.action_dim == buffer.action_dim
    assert len(restored) == len(buffer)
    assert restored.position == buffer.position
    np.testing.assert_array_equal(restored._states[: len(buffer)], buffer._states[: len(buffer)])
    np.testing.assert_array_equal(
        restored._next_action_masks[: len(buffer)],
        buffer._next_action_masks[: len(buffer)],
    )

    expected_batch = buffer.sample(3)
    actual_batch = restored.sample(3)
    np.testing.assert_array_equal(actual_batch.indices, expected_batch.indices)


def test_load_state_dict_rejects_incompatible_configuration() -> None:
    source = make_buffer(capacity=5)
    source.add(**make_transition(0.0))
    target = make_buffer(capacity=6)
    with pytest.raises(ValueError, match="incompatible"):
        target.load_state_dict(source.state_dict())


def test_variable_length_sample_pads_states_and_masks() -> None:
    buffer = ReplayBuffer(
        capacity=4,
        state_shape=(2, 4),
        action_dim=40,
        seed=5,
        variable_length=True,
    )

    short_state = np.ones((2, 4), dtype=np.float32)
    long_state = np.full((3, 4), 2.0, dtype=np.float32)
    buffer.add(
        state=short_state,
        action=1,
        reward=1.0,
        next_state=short_state + 0.5,
        action_mask=np.ones(40, dtype=bool),
        next_action_mask=np.ones(40, dtype=bool),
    )
    buffer.add(
        state=long_state,
        action=45,
        reward=2.0,
        next_state=long_state + 0.5,
        action_mask=np.ones(60, dtype=bool),
        next_action_mask=np.ones(60, dtype=bool),
    )

    batch = buffer.sample(2)

    assert batch.states.shape == (2, 3, 4)
    assert batch.next_states.shape == (2, 3, 4)
    assert batch.action_masks is not None and batch.action_masks.shape == (2, 60)
    assert batch.next_action_masks is not None and batch.next_action_masks.shape == (2, 60)

    short_row = int(np.where(batch.actions == 1)[0][0])
    long_row = int(np.where(batch.actions == 45)[0][0])
    np.testing.assert_array_equal(batch.states[short_row, 2], np.zeros(4, dtype=np.float32))
    assert not np.any(batch.action_masks[short_row, 40:])
    assert np.all(batch.action_masks[long_row])


def test_variable_length_save_and_load_round_trip(tmp_path: Path) -> None:
    buffer = ReplayBuffer(
        capacity=3,
        state_shape=(2, 4),
        action_dim=40,
        seed=11,
        variable_length=True,
    )
    state = np.ones((2, 4), dtype=np.float32)
    buffer.add(
        state=state,
        action=3,
        reward=1.0,
        next_state=state,
        action_mask=np.ones(40, dtype=bool),
        next_action_mask=np.ones(40, dtype=bool),
    )

    path = tmp_path / "variable_replay.npz"
    buffer.save(path)
    restored = ReplayBuffer.load(path)

    assert restored.variable_length
    assert restored.state_shape == buffer.state_shape
    batch = restored.sample(1)
    assert batch.states.shape == (1, 2, 4)
    assert batch.action_masks is not None and batch.action_masks.shape == (1, 40)
