"""Terminal reward calculator built on hbond topology features."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

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
        total_weight = self.scalarization.strength_weight + self.scalarization.toughness_weight
        if total_weight <= 0.0:
            raise ValueError("At least one terminal objective weight must be positive.")
        diagnostics = feature_result.to_dict()
        diagnostics.pop("features", None)

        return StructureRewardResult(
            reward=float(sum(components.values()) / total_weight),
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


class EqualWeightDualStructureTerminalRewardCalculator:
    """
    Terminal reward wrapper for the RL base version.

    The wrapper scores the PyRosetta terminal Pose and, when available, a
    matching sequence-predicted PDB. If both structures are available, their
    terminal rewards are averaged 1:1. If a matching predicted structure is not
    available, the relaxed PyRosetta Pose is used alone and the missing predicted
    path is recorded in the returned diagnostics.
    """

    def __init__(
        self,
        *,
        artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
        predicted_pdb_dir: Optional[str | Path] = None,
        predicted_pdb_patterns: Sequence[str] = (
            "{stem}.pdb",
            "{stem}_relaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb",
            "{stem}*relaxed_rank_001*.pdb",
            "{stem}*unrelaxed_rank_001*.pdb",
        ),
        force_retrain_artifact: bool = False,
    ) -> None:
        self.predicted_pdb_dir = None if predicted_pdb_dir is None else Path(predicted_pdb_dir).expanduser()
        self.predicted_pdb_patterns = tuple(predicted_pdb_patterns)
        self.base_calculator = HbondTopologyTerminalRewardCalculator(
            artifact_path=artifact_path,
            scalarization=TerminalRewardScalarization(
                strength_weight=1.0,
                toughness_weight=1.0,
                disagreement_penalty=0.0,
                use_zscore=True,
            ),
            force_retrain_artifact=force_retrain_artifact,
        )

    def evaluate_pose(self, pose: Any) -> StructureRewardResult:
        """Fallback API used when no source PDB context is available."""

        return self.base_calculator.evaluate_pose(pose)

    def evaluate_episode(
        self,
        *,
        relaxed_pose: Any,
        source_pdb_path: Optional[str | Path] = None,
    ) -> DualStructureTerminalRewardResult:
        predicted_pdb_path = self.resolve_predicted_pdb_path(source_pdb_path)
        relaxed_result = self.base_calculator.evaluate_pose(relaxed_pose)
        predicted_result = (
            self.base_calculator.evaluate_pdb(predicted_pdb_path)
            if predicted_pdb_path is not None
            else None
        )

        if predicted_result is None:
            return DualStructureTerminalRewardResult(
                reward=float(relaxed_result.reward),
                relaxed_result=relaxed_result,
                predicted_result=None,
                relaxed_weight=1.0,
                predicted_weight=0.0,
                weighted_reward=float(relaxed_result.reward),
                disagreement_penalty=0.0,
                reward_components={
                    "relaxed_structure_reward": float(relaxed_result.reward),
                    "predicted_structure_reward": 0.0,
                    "relaxed_structure_weight": 1.0,
                    "predicted_structure_weight": 0.0,
                    "predicted_structure_missing": 1.0,
                },
            )

        weighted_reward = 0.5 * float(relaxed_result.reward) + 0.5 * float(predicted_result.reward)
        return DualStructureTerminalRewardResult(
            reward=float(weighted_reward),
            relaxed_result=relaxed_result,
            predicted_result=predicted_result,
            relaxed_weight=1.0,
            predicted_weight=1.0,
            weighted_reward=float(weighted_reward),
            disagreement_penalty=0.0,
            reward_components={
                "relaxed_structure_reward": float(relaxed_result.reward),
                "predicted_structure_reward": float(predicted_result.reward),
                "relaxed_structure_weight": 1.0,
                "predicted_structure_weight": 1.0,
                "predicted_structure_missing": 0.0,
            },
        )

    def resolve_predicted_pdb_path(self, source_pdb_path: Optional[str | Path]) -> Optional[Path]:
        if self.predicted_pdb_dir is None or source_pdb_path is None:
            return None
        source = Path(source_pdb_path)
        stem = source.stem
        name = source.name
        search_dirs = [self.predicted_pdb_dir]
        results_dir = self.predicted_pdb_dir / "results"
        if results_dir.is_dir():
            search_dirs.append(results_dir)
        for search_dir in search_dirs:
            for pattern in self.predicted_pdb_patterns:
                expanded = pattern.format(stem=stem, name=name)
                matches = sorted(search_dir.glob(expanded))
                for match in matches:
                    if match.is_file():
                        return match
        return None
