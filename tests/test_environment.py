"""Pure-Python tests for MechanicalProteinEnv using fake Pose components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence, Tuple

import numpy as np
import pytest

from model.environment_module.environment import (
    CANONICAL_AMINO_ACIDS,
    MechanicalProteinEnv,
)


@dataclass
class FakePose:
    sequence: list[str]

    def clone(self):
        return FakePose(self.sequence.copy())


class FakeBackend:
    scorefxn = None

    def __init__(self, sequence: str = "ACD") -> None:
        self.initial_sequence = sequence
        self.calls: list[tuple] = []

    def load_pose(self, pdb_path: str) -> FakePose:
        self.calls.append(("load", pdb_path))
        return FakePose(list(self.initial_sequence))

    @staticmethod
    def clone_pose(pose: FakePose) -> FakePose:
        return pose.clone()

    @staticmethod
    def total_residue(pose: FakePose) -> int:
        return len(pose.sequence)

    @staticmethod
    def residue_name1(pose: FakePose, position: int) -> str:
        return pose.sequence[position - 1]

    def local_residues(self, pose: FakePose, center_position: int, radius: float) -> Tuple[int, ...]:
        self.calls.append(("local", center_position, radius))
        return (center_position,)

    def mutate(self, pose: FakePose, position: int, amino_acid: str) -> None:
        self.calls.append(("mutate", position, amino_acid))
        pose.sequence[position - 1] = amino_acid

    def repack(self, pose: FakePose, local_residues: Sequence[int]) -> None:
        self.calls.append(("repack", tuple(local_residues)))

    def minimize(self, pose: FakePose, local_residues: Sequence[int], *, minimize_backbone: bool) -> None:
        self.calls.append(("minimize", tuple(local_residues), minimize_backbone))

    def dump_pose(self, pose: FakePose, output_path: str) -> None:
        Path(output_path).write_text("".join(pose.sequence), encoding="utf-8")


class FakeStepRewardCalculator:
    def evaluate(self, current_pose, *, previous_pose, mutated_positions, local_residues):
        before = "".join(previous_pose.sequence)
        after = "".join(current_pose.sequence)
        return SimpleNamespace(
            reward=2.5,
            to_dict=lambda: {
                "before": before,
                "after": after,
                "mutated_positions": tuple(mutated_positions),
                "local_residues": tuple(local_residues),
            },
        )


class FakeTerminalRewardCalculator:
    def evaluate_pose(self, pose):
        return SimpleNamespace(
            reward=10.0,
            to_dict=lambda: {"sequence": "".join(pose.sequence)},
        )


def make_env(**kwargs) -> MechanicalProteinEnv:
    backend = kwargs.pop("backend", FakeBackend())
    return MechanicalProteinEnv(
        "fake.pdb",
        backend=backend,
        step_reward_calculator=FakeStepRewardCalculator(),
        **kwargs,
    )


def action_for(env: MechanicalProteinEnv, mutable_index: int, aa: str) -> int:
    return mutable_index * env.n_amino_acids + env.amino_acids.index(aa)


def test_load_reset_and_default_observation_shape() -> None:
    env = make_env()
    observation, info = env.reset(seed=123)

    assert env.n_actions == 3 * 20
    assert env.action_space.n == 60
    assert observation.shape == (60,)
    assert observation.sum() == pytest.approx(3.0)
    assert info["sequence"] == "ACD"
    assert info["valid_action_count"] == 3 * 19


def test_decode_action_maps_mutable_index_and_amino_acid() -> None:
    env = make_env()
    env.reset()

    action = action_for(env, mutable_index=1, aa="G")
    decoded = env.decode_action(action)

    assert decoded.mutable_position_index == 1
    assert decoded.pose_position == 2
    assert decoded.previous_amino_acid == "C"
    assert decoded.target_amino_acid == "G"


def test_step_mutates_repack_minimizes_and_returns_reward() -> None:
    backend = FakeBackend()
    env = make_env(backend=backend, max_steps=3)
    env.reset()

    action = action_for(env, mutable_index=1, aa="G")
    observation, reward, terminated, truncated, info = env.step(action)

    assert env.current_sequence() == "AGD"
    assert reward == pytest.approx(2.5)
    assert terminated is False
    assert truncated is False
    assert info["accepted"] is True
    assert info["reason"] == "accepted"
    assert info["step_reward_metrics"]["before"] == "ACD"
    assert info["step_reward_metrics"]["after"] == "AGD"

    assert ("mutate", 2, "G") in backend.calls
    assert ("repack", (2,)) in backend.calls
    assert ("minimize", (2,), False) in backend.calls
    assert observation.shape == (60,)


def test_last_step_adds_terminal_reward() -> None:
    env = make_env(
        max_steps=1,
        terminal_reward_calculator=FakeTerminalRewardCalculator(),
    )
    env.reset()

    _, reward, terminated, truncated, info = env.step(action_for(env, 0, "G"))

    assert terminated is False
    assert truncated is True
    assert info["step_reward"] == pytest.approx(2.5)
    assert info["terminal_reward"] == pytest.approx(10.0)
    assert reward == pytest.approx(12.5)

    with pytest.raises(RuntimeError, match="Call reset"):
        env.step(action_for(env, 1, "A"))


def test_action_mask_blocks_current_amino_acid_and_noop_is_penalized() -> None:
    env = make_env(invalid_action_penalty=-7.0)
    env.reset()

    noop_action = action_for(env, mutable_index=0, aa="A")
    assert env.action_mask()[noop_action] == np.bool_(False)

    _, reward, _, _, info = env.step(noop_action)

    assert reward == pytest.approx(-7.0)
    assert info["accepted"] is False
    assert info["reason"] == "noop_same_amino_acid"
    assert env.current_sequence() == "ACD"


def test_restricted_mutable_positions_use_pose_indices_not_local_indices() -> None:
    env = make_env(mutable_positions=[2, 3])
    env.reset()

    assert env.n_actions == 2 * 20
    decoded = env.decode_action(action_for(env, mutable_index=0, aa="W"))
    assert decoded.pose_position == 2


def test_prevent_revisit_masks_entire_position_after_mutation() -> None:
    env = make_env(prevent_revisit_positions=True, invalid_action_penalty=-4.0)
    env.reset()

    env.step(action_for(env, mutable_index=1, aa="G"))
    mask = env.action_mask()
    start = 1 * env.n_amino_acids
    assert not mask[start : start + env.n_amino_acids].any()

    _, reward, _, _, info = env.step(action_for(env, mutable_index=1, aa="A"))
    assert reward == pytest.approx(-4.0)
    assert info["reason"] == "position_already_mutated"


def test_update_failure_rolls_back_candidate_pose_when_configured() -> None:
    class FailingBackend(FakeBackend):
        def repack(self, pose: FakePose, local_residues: Sequence[int]) -> None:
            raise RuntimeError("synthetic packing failure")

    env = make_env(
        backend=FailingBackend(),
        raise_on_update_error=False,
        update_error_penalty=-11.0,
    )
    env.reset()

    _, reward, _, _, info = env.step(action_for(env, 0, "G"))

    assert reward == pytest.approx(-11.0)
    assert info["accepted"] is False
    assert info["reason"] == "structure_update_error"
    assert "synthetic packing failure" in info["error"]
    assert env.current_sequence() == "ACD"


def test_save_pose_and_history(tmp_path: Path) -> None:
    env = make_env(max_steps=1)
    env.reset()
    env.step(action_for(env, 0, "G"))

    pose_path = tmp_path / "candidate.pdb"
    history_path = tmp_path / "history.json"
    env.save_current_pose(str(pose_path))
    env.save_history(str(history_path))

    assert pose_path.read_text(encoding="utf-8") == "GCD"
    assert '"sequence": "GCD"' in history_path.read_text(encoding="utf-8")


def test_manual_terminal_finalize_is_applied_only_once() -> None:
    env = make_env(terminal_reward_calculator=FakeTerminalRewardCalculator())
    env.reset()

    first_reward, first_metrics = env.finalize_episode()
    second_reward, second_metrics = env.finalize_episode()

    assert first_reward == pytest.approx(10.0)
    assert first_metrics["sequence"] == "ACD"
    assert second_reward == pytest.approx(0.0)
    assert second_metrics == {}
