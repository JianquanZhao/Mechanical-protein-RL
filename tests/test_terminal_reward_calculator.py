"""Unit tests for final-minus-initial mechanical terminal reward."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from model.reward_module.terminal_reward import (
    MechanicalImprovementTerminalRewardCalculator,
    StructureRewardResult,
)


@dataclass(frozen=True)
class FakePose:
    name: str


def structure_result(*, strength_z: float, toughness_z: float) -> StructureRewardResult:
    return StructureRewardResult(
        reward=0.5 * (strength_z + toughness_z),
        strength=100.0 + strength_z,
        toughness=10.0 + toughness_z,
        normalized_strength=strength_z,
        normalized_toughness=toughness_z,
        reward_components={},
        features={},
        feature_diagnostics={},
        structure_quality=None,
    )


class FakeBaseCalculator:
    def __init__(self) -> None:
        self.pose_results = {
            "initial": structure_result(strength_z=-0.5, toughness_z=0.25),
            "final": structure_result(strength_z=0.5, toughness_z=0.75),
        }
        self.pdb_result = structure_result(strength_z=-1.0, toughness_z=-0.5)

    def evaluate_pose(self, pose: FakePose) -> StructureRewardResult:
        return self.pose_results[pose.name]

    def evaluate_pdb(self, _path: str) -> StructureRewardResult:
        return self.pdb_result


def make_calculator() -> MechanicalImprovementTerminalRewardCalculator:
    calculator = MechanicalImprovementTerminalRewardCalculator.__new__(
        MechanicalImprovementTerminalRewardCalculator
    )
    calculator.base_calculator = FakeBaseCalculator()
    return calculator


def test_terminal_reward_is_equal_weight_final_minus_initial_zscore() -> None:
    result = make_calculator().evaluate_episode(
        initial_pose=FakePose("initial"),
        relaxed_pose=FakePose("final"),
        source_pdb_path="unused.pdb",
    )

    assert result.delta_normalized_strength == pytest.approx(1.0)
    assert result.delta_normalized_toughness == pytest.approx(0.5)
    assert result.reward_components == pytest.approx(
        {"delta_strength": 0.5, "delta_toughness": 0.25}
    )
    assert result.reward == pytest.approx(0.75)
    assert result.raw_predictions == pytest.approx(
        {"strength": 100.5, "toughness": 10.75}
    )


def test_terminal_reward_can_fall_back_to_source_pdb_for_initial_state() -> None:
    result = make_calculator().evaluate_episode(
        relaxed_pose=FakePose("final"),
        source_pdb_path="source.pdb",
    )

    assert result.delta_normalized_strength == pytest.approx(1.5)
    assert result.delta_normalized_toughness == pytest.approx(1.25)
    assert result.reward == pytest.approx(1.375)


def test_terminal_reward_requires_an_initial_state() -> None:
    with pytest.raises(ValueError, match="requires initial_pose or source_pdb_path"):
        make_calculator().evaluate_episode(relaxed_pose=FakePose("final"))
