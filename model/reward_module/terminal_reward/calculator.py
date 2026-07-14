"""Terminal reward calculator built on hbond topology features."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .features import HbondFeatureExtractor, HbondFeatureResult
from .model_artifact import (
    DEFAULT_ARTIFACT_PATH,
    HbondRandomForestArtifact,
    load_or_train_random_forest_artifact,
    predict_physical_units,
)


@dataclass(frozen=True)
class TerminalRewardScalarization:
    """Weights and penalties used to turn predicted objectives into one reward."""

    strength_weight: float = 1.0
    toughness_weight: float = 1.0
    disagreement_penalty: float = 0.25
    use_zscore: bool = True


@dataclass(frozen=True)
class StructureRewardResult:
    reward: float
    strength: float
    toughness: float
    normalized_strength: float
    normalized_toughness: float
    reward_components: Mapping[str, float]
    features: Mapping[str, float]
    feature_diagnostics: Mapping[str, Any]
    structure_quality: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DualStructureTerminalRewardResult:
    reward: float
    relaxed_result: Optional[StructureRewardResult]
    predicted_result: Optional[StructureRewardResult]
    relaxed_weight: float
    predicted_weight: float
    weighted_reward: float
    disagreement_penalty: float
    reward_components: Mapping[str, float]

    @property
    def raw_predictions(self) -> Mapping[str, float]:
        primary = self.relaxed_result or self.predicted_result
        if primary is None:
            return {}
        return {"strength": primary.strength, "toughness": primary.toughness}

    @property
    def objective_vector(self) -> tuple[float, float]:
        primary = self.relaxed_result or self.predicted_result
        if primary is None:
            return (0.0, 0.0)
        return (primary.strength, primary.toughness)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HbondTopologyTerminalRewardCalculator:
    """
    Structure-based terminal reward using the random-split hbond random forest.

    This calculator can score a single PyRosetta-relaxed terminal Pose, a single
    PDB, or a dual-structure pair consisting of a local-relaxed structure and a
    sequence-predicted structure. Dual scoring weights each structure by a
    confidence proxy, then subtracts a penalty when the two rewards disagree.
    """

    def __init__(
        self,
        *,
        artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
        artifact: Optional[HbondRandomForestArtifact] = None,
        feature_extractor: Optional[HbondFeatureExtractor] = None,
        scalarization: TerminalRewardScalarization = TerminalRewardScalarization(),
        force_retrain_artifact: bool = False,
    ) -> None:
        self.artifact = artifact or load_or_train_random_forest_artifact(
            artifact_path=artifact_path,
            force_retrain=force_retrain_artifact,
        )
        self.feature_extractor = feature_extractor or HbondFeatureExtractor()
        self.scalarization = scalarization

    def evaluate_pose(self, pose: Any) -> StructureRewardResult:
        feature_result = self.feature_extractor.extract_from_pose(pose)
        return self.evaluate_feature_result(feature_result)

    def evaluate_pdb(self, pdb_path: str | Path) -> StructureRewardResult:
        feature_result = self.feature_extractor.extract_from_pdb(pdb_path)
        return self.evaluate_feature_result(feature_result)

    def evaluate_feature_result(self, feature_result: HbondFeatureResult) -> StructureRewardResult:
        x = feature_result.feature_vector(self.artifact.selected_features).reshape(1, -1)
        prediction = predict_physical_units(self.artifact, x)[0]
        toughness = float(prediction[0])
        strength = float(prediction[1])

        if self.scalarization.use_zscore:
            normalized_toughness = (
                toughness - self.artifact.target_mean["toughness"]
            ) / self.artifact.target_std["toughness"]
            normalized_strength = (
                strength - self.artifact.target_mean["strength"]
            ) / self.artifact.target_std["strength"]
        else:
            normalized_toughness = toughness
            normalized_strength = strength

        components = {
            "strength": self.scalarization.strength_weight * float(normalized_strength),
            "toughness": self.scalarization.toughness_weight * float(normalized_toughness),
        }
        diagnostics = feature_result.to_dict()
        diagnostics.pop("features", None)

        return StructureRewardResult(
            reward=float(sum(components.values())),
            strength=strength,
            toughness=toughness,
            normalized_strength=float(normalized_strength),
            normalized_toughness=float(normalized_toughness),
            reward_components=components,
            features=dict(feature_result.features),
            feature_diagnostics=diagnostics,
            structure_quality=self._quality_from_feature_result(feature_result),
        )

    def evaluate_dual_structures(
        self,
        *,
        relaxed_pose: Optional[Any] = None,
        relaxed_pdb_path: Optional[str | Path] = None,
        predicted_pdb_path: Optional[str | Path] = None,
        relaxed_quality: Optional[float] = None,
        predicted_plddt: Optional[float] = None,
    ) -> DualStructureTerminalRewardResult:
        relaxed_result = None
        predicted_result = None

        if relaxed_pose is not None:
            relaxed_result = self.evaluate_pose(relaxed_pose)
        elif relaxed_pdb_path is not None:
            relaxed_result = self.evaluate_pdb(relaxed_pdb_path)

        if predicted_pdb_path is not None:
            predicted_result = self.evaluate_pdb(predicted_pdb_path)

        if relaxed_result is None and predicted_result is None:
            raise ValueError("At least one relaxed or predicted structure is required.")

        relaxed_weight = 0.0
        if relaxed_result is not None:
            relaxed_weight = self._coerce_quality_weight(relaxed_quality)
            if relaxed_weight is None:
                relaxed_weight = 1.0

        predicted_weight = 0.0
        if predicted_result is not None:
            if predicted_plddt is not None:
                predicted_quality = self._plddt_to_weight(predicted_plddt)
            else:
                predicted_quality = predicted_result.structure_quality
            predicted_weight = 1.0 if predicted_quality is None else predicted_quality

        total_weight = relaxed_weight + predicted_weight
        if total_weight <= 0.0:
            total_weight = 1.0
        weighted_reward = (
            (relaxed_weight * relaxed_result.reward if relaxed_result is not None else 0.0)
            + (predicted_weight * predicted_result.reward if predicted_result is not None else 0.0)
        ) / total_weight

        disagreement = 0.0
        if relaxed_result is not None and predicted_result is not None:
            disagreement = abs(relaxed_result.reward - predicted_result.reward)
        penalty = self.scalarization.disagreement_penalty * disagreement
        final_reward = float(weighted_reward - penalty)

        return DualStructureTerminalRewardResult(
            reward=final_reward,
            relaxed_result=relaxed_result,
            predicted_result=predicted_result,
            relaxed_weight=float(relaxed_weight),
            predicted_weight=float(predicted_weight),
            weighted_reward=float(weighted_reward),
            disagreement_penalty=float(penalty),
            reward_components={
                "weighted_reward": float(weighted_reward),
                "structure_disagreement_penalty": float(-penalty),
            },
        )

    @staticmethod
    def _quality_from_feature_result(feature_result: HbondFeatureResult) -> Optional[float]:
        if feature_result.mean_plddt is None:
            return None
        return HbondTopologyTerminalRewardCalculator._plddt_to_weight(feature_result.mean_plddt)

    @staticmethod
    def _plddt_to_weight(plddt: Optional[float]) -> Optional[float]:
        if plddt is None or not np.isfinite(plddt):
            return None
        return float(max(0.0, min(1.0, plddt / 100.0)))

    @staticmethod
    def _coerce_quality_weight(value: Optional[float]) -> Optional[float]:
        if value is None or not np.isfinite(value):
            return None
        return float(max(0.0, min(1.0, value)))
