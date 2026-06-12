"""
reward_calculators.py

A base-version reward module for RL-based optimization of mechanical proteins.

Design principles
-----------------
1. StepRewardCalculator evaluates a mutated / repacked / relaxed PyRosetta Pose.
   It does NOT mutate or relax a Pose itself. Keep environment transitions and
   reward calculation separate.
2. TerminalRewardCalculator wraps a trained predictor that outputs:
      - max_stress
      - toughness
   It supports sklearn-like predictors and generic callables.
3. Both calculators return raw, interpretable metrics and a scalar reward.
   Logging the raw metrics is strongly recommended to detect reward hacking.

PyRosetta note
--------------
Initialize PyRosetta once, before instantiating StepRewardCalculator:

    import pyrosetta
    pyrosetta.init("-mute all")

The code uses pose.get_hbonds(), the weighted fa_rep term, and local backbone
heavy-atom RMSD calculated with Kabsch superposition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Shared helper types
# ---------------------------------------------------------------------------

ArrayLike = Union[np.ndarray, Sequence[float]]
FeatureExtractor = Callable[[Any], ArrayLike]


class PredictorProtocol(Protocol):
    """Minimal sklearn-like predictor interface."""

    def predict(self, x: np.ndarray) -> Any:
        ...


@dataclass(frozen=True)
class StepRewardWeights:
    """
    Weights for the scalar step reward.

    Start with small, interpretable values and tune them after inspecting the
    distribution of each raw metric on a pilot mutation set.
    """

    collision: float = 1.0
    backbone_hbond: float = 1.0
    sidechain_hbond: float = 0.5
    local_rmsd: float = 1.0


@dataclass(frozen=True)
class StepRewardScales:
    """
    Normalization scales for step reward terms.

    A term is divided by its scale before multiplying by its weight. Replace
    the defaults using robust statistics (for example, median absolute
    deviations or interquartile ranges) from a pilot mutation set.
    """

    collision: float = 1.0
    backbone_hbond: float = 1.0
    sidechain_hbond: float = 1.0
    local_rmsd: float = 1.0

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if value <= 0:
                raise ValueError(f"Scale '{name}' must be > 0, got {value}.")


@dataclass(frozen=True)
class HBondCounts:
    """
    Hydrogen-bond count summary.

    backbone:
        Backbone-backbone hydrogen bonds.
    sidechain_involving:
        Hydrogen bonds for which at least one participating atom belongs to a
        side chain. This combines bb-sc, sc-bb and sc-sc bonds in the base
        version.
    total:
        Number of accepted hydrogen bonds after applying the energy cutoff.
    """

    backbone: int
    sidechain_involving: int
    total: int


@dataclass(frozen=True)
class StepRewardResult:
    """Interpretable result returned after one environment transition."""

    reward: float

    # Collision metrics
    collision_score: float
    previous_collision_score: float
    reference_collision_score: float
    collision_delta_from_previous: float
    collision_excess_over_reference: float
    collision_loss_used: float

    # Hydrogen-bond metrics
    backbone_hbonds: int
    previous_backbone_hbonds: int
    backbone_hbond_delta: int
    sidechain_hbonds: int
    previous_sidechain_hbonds: int
    sidechain_hbond_delta: int
    total_hbonds: int

    # Local structural drift
    local_rmsd_to_previous: float
    local_rmsd_to_reference: float
    local_rmsd_used: float
    local_residues: Tuple[int, ...]

    # Weighted scalar-reward components
    reward_components: Mapping[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TerminalRewardWeights:
    """Weights for scalarizing predicted mechanical objectives."""

    max_stress: float = 1.0
    toughness: float = 1.0


@dataclass(frozen=True)
class TerminalRewardResult:
    """
    Result returned at the end of an RL episode.

    raw_predictions:
        Predictor outputs in their original physical units.
    normalized_predictions:
        Values used for scalarization. These are z-scores when target means
        and standard deviations are supplied; otherwise they equal the raw
        predictions.
    objective_vector:
        Ordered objective vector: (max_stress, toughness). Keep this vector if
        you later switch from scalar reward to Pareto-front selection.
    """

    reward: float
    raw_predictions: Mapping[str, float]
    normalized_predictions: Mapping[str, float]
    baseline_normalized_predictions: Optional[Mapping[str, float]]
    reward_components: Mapping[str, float]
    objective_vector: Tuple[float, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Step reward calculator
# ---------------------------------------------------------------------------

class StepRewardCalculator:
    """
    Calculate a base-version RL step reward from PyRosetta Pose objects.

    Reward formula
    --------------
    reward =
        - w_collision       * normalized(collision_loss)
        + w_backbone_hbond  * normalized(delta backbone-backbone H-bonds)
        + w_sidechain_hbond * normalized(delta sidechain-involving H-bonds)
        - w_local_rmsd      * normalized(local RMSD)

    Recommended environment order
    -----------------------------
    previous_pose
        -> mutate residue(s)
        -> local repack
        -> local relax / minimize
        -> current_pose
        -> StepRewardCalculator.evaluate(...)

    Parameters
    ----------
    reference_pose:
        Initial / wild-type Pose. Stored internally as a clone.
    scorefxn:
        Rosetta ScoreFunction. If omitted, pyrosetta.get_fa_scorefxn() is used.
    weights:
        Weights used to scalarize metrics.
    scales:
        Normalization scales for reward terms.
    hbond_energy_cutoff:
        Only count hydrogen bonds with HBond.energy() <= cutoff. Rosetta HBond
        energies are favorable when negative. A cutoff of 0.0 is permissive;
        consider a stricter negative cutoff after pilot analysis.
    neighborhood_radius:
        Radius in Angstrom used to construct a mutation-centered local residue
        neighborhood when local_residues is not explicitly provided.
    rmsd_atom_names:
        Atoms used for local RMSD. Backbone heavy atoms avoid undefined
        comparisons when a mutation changes side-chain atom names.
    rmsd_penalty_mode:
        "previous" penalizes abrupt drift caused by the latest action.
        "reference" penalizes cumulative drift from the initial structure.
    collision_penalty_mode:
        "delta" penalizes collision-score increases from the previous Pose.
        "excess_over_reference" penalizes collision score above wild type.
        "absolute" penalizes the full current collision score.
    include_fa_intra_rep:
        If True, add weighted fa_intra_rep to weighted fa_rep.
    """

    _VALID_RMSD_MODES = {"previous", "reference"}
    _VALID_COLLISION_MODES = {"delta", "excess_over_reference", "absolute"}

    def __init__(
        self,
        reference_pose: Any,
        *,
        scorefxn: Optional[Any] = None,
        weights: StepRewardWeights = StepRewardWeights(),
        scales: StepRewardScales = StepRewardScales(),
        hbond_energy_cutoff: float = 0.0,
        neighborhood_radius: float = 8.0,
        rmsd_atom_names: Sequence[str] = ("N", "CA", "C", "O"),
        rmsd_penalty_mode: str = "previous",
        collision_penalty_mode: str = "delta",
        include_fa_intra_rep: bool = False,
    ) -> None:
        pyrosetta, score_type_from_name = self._import_pyrosetta()

        if reference_pose is None:
            raise ValueError("reference_pose must be a valid PyRosetta Pose.")
        if neighborhood_radius <= 0:
            raise ValueError("neighborhood_radius must be > 0.")
        if not rmsd_atom_names:
            raise ValueError("rmsd_atom_names must not be empty.")
        if rmsd_penalty_mode not in self._VALID_RMSD_MODES:
            raise ValueError(
                f"rmsd_penalty_mode must be one of {sorted(self._VALID_RMSD_MODES)}."
            )
        if collision_penalty_mode not in self._VALID_COLLISION_MODES:
            raise ValueError(
                "collision_penalty_mode must be one of "
                f"{sorted(self._VALID_COLLISION_MODES)}."
            )

        scales.validate()

        self._score_type_from_name = score_type_from_name
        self.reference_pose = reference_pose.clone()
        self.scorefxn = scorefxn if scorefxn is not None else pyrosetta.get_fa_scorefxn()
        self.weights = weights
        self.scales = scales
        self.hbond_energy_cutoff = float(hbond_energy_cutoff)
        self.neighborhood_radius = float(neighborhood_radius)
        self.rmsd_atom_names = tuple(str(atom).strip() for atom in rmsd_atom_names)
        self.rmsd_penalty_mode = rmsd_penalty_mode
        self.collision_penalty_mode = collision_penalty_mode
        self.include_fa_intra_rep = bool(include_fa_intra_rep)

        self.reference_collision_score = self._collision_score(self.reference_pose)
        self.reference_hbond_counts = self._count_hbonds(self.reference_pose)

    @staticmethod
    def _import_pyrosetta() -> Tuple[Any, Any]:
        try:
            import pyrosetta
            from pyrosetta.rosetta.core.scoring import score_type_from_name
        except ImportError as exc:
            raise ImportError(
                "PyRosetta is required for StepRewardCalculator. "
                "Install PyRosetta in your research environment and initialize "
                "it once with pyrosetta.init(...)."
            ) from exc
        return pyrosetta, score_type_from_name

    def _weighted_score_term(self, pose: Any, score_term_name: str) -> float:
        """
        Return a weighted Rosetta score term.

        Calling the ScoreFunction updates pose.energies() before accessing the
        requested energy component.
        """

        self.scorefxn(pose)
        score_type = self._score_type_from_name(score_term_name)
        raw_value = float(pose.energies().total_energies()[score_type])
        weight = float(self.scorefxn.get_weight(score_type))
        return raw_value * weight

    def _collision_score(self, pose: Any) -> float:
        """
        Weighted collision proxy based on fa_rep, optionally plus fa_intra_rep.

        This is a Rosetta steric-overlap proxy, not a literal atom-pair clash
        count.
        """

        score = self._weighted_score_term(pose, "fa_rep")
        if self.include_fa_intra_rep:
            score += self._weighted_score_term(pose, "fa_intra_rep")
        return float(score)

    def _count_hbonds(self, pose: Any) -> HBondCounts:
        """
        Count backbone-backbone and sidechain-involving hydrogen bonds.

        pose.get_hbonds() returns an HBondSet. The base version counts an H-bond
        if its unweighted HBond.energy() is <= hbond_energy_cutoff.
        """

        hbond_set = pose.get_hbonds()
        backbone = 0
        sidechain_involving = 0
        total = 0

        for index in range(1, int(hbond_set.nhbonds()) + 1):
            hbond = hbond_set.hbond(index)

            if float(hbond.energy()) > self.hbond_energy_cutoff:
                continue

            donor_bb = bool(hbond.don_hatm_is_protein_backbone())
            acceptor_bb = bool(hbond.acc_atm_is_protein_backbone())

            total += 1
            if donor_bb and acceptor_bb:
                backbone += 1
            else:
                sidechain_involving += 1

        return HBondCounts(
            backbone=backbone,
            sidechain_involving=sidechain_involving,
            total=total,
        )

    @staticmethod
    def _xyz_to_numpy(xyz: Any) -> np.ndarray:
        """Convert Rosetta xyzVector-like object to a NumPy coordinate vector."""

        return np.asarray([float(xyz.x), float(xyz.y), float(xyz.z)], dtype=float)

    @staticmethod
    def _validate_pose_lengths(*poses: Any) -> None:
        totals = [int(pose.total_residue()) for pose in poses]
        if len(set(totals)) != 1:
            raise ValueError(
                "All Poses must contain the same number of residues. "
                f"Observed lengths: {totals}."
            )

    def _infer_local_residues(
        self,
        pose: Any,
        mutated_positions: Sequence[int],
    ) -> Tuple[int, ...]:
        """
        Infer a mutation-centered neighborhood using neighbor-atom distances.
        """

        total_residue = int(pose.total_residue())
        if not mutated_positions:
            raise ValueError(
                "Provide mutated_positions or explicitly pass local_residues."
            )

        centers = []
        for position in mutated_positions:
            if not 1 <= int(position) <= total_residue:
                raise IndexError(
                    f"Mutated residue index {position} is outside 1..{total_residue}."
                )
            residue = pose.residue(int(position))
            centers.append(self._xyz_to_numpy(residue.nbr_atom_xyz()))

        local_positions = []
        radius_squared = self.neighborhood_radius ** 2

        for residue_index in range(1, total_residue + 1):
            residue = pose.residue(residue_index)
            candidate = self._xyz_to_numpy(residue.nbr_atom_xyz())

            if any(float(np.sum((candidate - center) ** 2)) <= radius_squared for center in centers):
                local_positions.append(residue_index)

        return tuple(local_positions)

    def _coordinates_for_local_rmsd(
        self,
        pose: Any,
        residue_indices: Sequence[int],
    ) -> np.ndarray:
        """
        Collect matching backbone-heavy-atom coordinates for local RMSD.
        """

        coordinates = []

        for residue_index in residue_indices:
            residue = pose.residue(int(residue_index))
            for atom_name in self.rmsd_atom_names:
                if not residue.has(atom_name):
                    raise ValueError(
                        f"Residue {residue_index} ({residue.name3()}) lacks atom "
                        f"'{atom_name}', required for local RMSD."
                    )
                coordinates.append(self._xyz_to_numpy(residue.xyz(atom_name)))

        if len(coordinates) < 3:
            raise ValueError(
                "At least three atoms are required to calculate a superposed RMSD."
            )

        return np.asarray(coordinates, dtype=float)

    @staticmethod
    def _kabsch_rmsd(reference_coordinates: np.ndarray, mobile_coordinates: np.ndarray) -> float:
        """
        Calculate RMSD after optimal rigid-body Kabsch superposition.

        Both arrays must be shaped (n_atoms, 3).
        """

        reference = np.asarray(reference_coordinates, dtype=float)
        mobile = np.asarray(mobile_coordinates, dtype=float)

        if reference.shape != mobile.shape:
            raise ValueError(
                f"Coordinate shapes differ: {reference.shape} vs {mobile.shape}."
            )
        if reference.ndim != 2 or reference.shape[1] != 3:
            raise ValueError("Coordinate arrays must have shape (n_atoms, 3).")

        reference_centered = reference - reference.mean(axis=0, keepdims=True)
        mobile_centered = mobile - mobile.mean(axis=0, keepdims=True)

        covariance = mobile_centered.T @ reference_centered
        u, _, vt = np.linalg.svd(covariance)

        correction = np.eye(3)
        correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
        rotation = u @ correction @ vt

        aligned_mobile = mobile_centered @ rotation
        difference = aligned_mobile - reference_centered

        return float(np.sqrt(np.mean(np.sum(difference ** 2, axis=1))))

    def _local_rmsd(
        self,
        reference_pose: Any,
        mobile_pose: Any,
        residue_indices: Sequence[int],
    ) -> float:
        reference_coordinates = self._coordinates_for_local_rmsd(
            reference_pose, residue_indices
        )
        mobile_coordinates = self._coordinates_for_local_rmsd(
            mobile_pose, residue_indices
        )
        return self._kabsch_rmsd(reference_coordinates, mobile_coordinates)

    def evaluate(
        self,
        current_pose: Any,
        *,
        previous_pose: Any,
        mutated_positions: Optional[Sequence[int]] = None,
        local_residues: Optional[Sequence[int]] = None,
    ) -> StepRewardResult:
        """
        Evaluate one mutation / repack / relaxation transition.

        Parameters
        ----------
        current_pose:
            Pose after mutation, local repacking and local relax/minimization.
        previous_pose:
            Pose before the action.
        mutated_positions:
            1-indexed Rosetta Pose residue indices modified by the action.
            Required unless local_residues is provided.
        local_residues:
            Optional explicit 1-indexed residue indices for RMSD calculation.
            This is useful when you want a fixed mechanical-lock region rather
            than an automatically inferred spatial neighborhood.
        """

        if current_pose is None or previous_pose is None:
            raise ValueError("current_pose and previous_pose must be valid Poses.")

        self._validate_pose_lengths(
            self.reference_pose,
            previous_pose,
            current_pose,
        )

        if local_residues is None:
            local_residues_tuple = self._infer_local_residues(
                previous_pose,
                mutated_positions or (),
            )
        else:
            local_residues_tuple = tuple(sorted({int(i) for i in local_residues}))
            if not local_residues_tuple:
                raise ValueError("local_residues must not be empty.")

        current_collision = self._collision_score(current_pose)
        previous_collision = self._collision_score(previous_pose)

        collision_delta = current_collision - previous_collision
        collision_excess = max(
            0.0,
            current_collision - self.reference_collision_score,
        )

        if self.collision_penalty_mode == "delta":
            collision_loss = max(0.0, collision_delta)
        elif self.collision_penalty_mode == "excess_over_reference":
            collision_loss = collision_excess
        else:
            collision_loss = max(0.0, current_collision)

        current_hbonds = self._count_hbonds(current_pose)
        previous_hbonds = self._count_hbonds(previous_pose)

        backbone_delta = current_hbonds.backbone - previous_hbonds.backbone
        sidechain_delta = (
            current_hbonds.sidechain_involving
            - previous_hbonds.sidechain_involving
        )

        local_rmsd_to_previous = self._local_rmsd(
            previous_pose,
            current_pose,
            local_residues_tuple,
        )
        local_rmsd_to_reference = self._local_rmsd(
            self.reference_pose,
            current_pose,
            local_residues_tuple,
        )

        if self.rmsd_penalty_mode == "previous":
            local_rmsd_used = local_rmsd_to_previous
        else:
            local_rmsd_used = local_rmsd_to_reference

        components = {
            "collision": (
                -self.weights.collision
                * collision_loss
                / self.scales.collision
            ),
            "backbone_hbond": (
                self.weights.backbone_hbond
                * backbone_delta
                / self.scales.backbone_hbond
            ),
            "sidechain_hbond": (
                self.weights.sidechain_hbond
                * sidechain_delta
                / self.scales.sidechain_hbond
            ),
            "local_rmsd": (
                -self.weights.local_rmsd
                * local_rmsd_used
                / self.scales.local_rmsd
            ),
        }

        return StepRewardResult(
            reward=float(sum(components.values())),
            collision_score=float(current_collision),
            previous_collision_score=float(previous_collision),
            reference_collision_score=float(self.reference_collision_score),
            collision_delta_from_previous=float(collision_delta),
            collision_excess_over_reference=float(collision_excess),
            collision_loss_used=float(collision_loss),
            backbone_hbonds=int(current_hbonds.backbone),
            previous_backbone_hbonds=int(previous_hbonds.backbone),
            backbone_hbond_delta=int(backbone_delta),
            sidechain_hbonds=int(current_hbonds.sidechain_involving),
            previous_sidechain_hbonds=int(previous_hbonds.sidechain_involving),
            sidechain_hbond_delta=int(sidechain_delta),
            total_hbonds=int(current_hbonds.total),
            local_rmsd_to_previous=float(local_rmsd_to_previous),
            local_rmsd_to_reference=float(local_rmsd_to_reference),
            local_rmsd_used=float(local_rmsd_used),
            local_residues=local_residues_tuple,
            reward_components=components,
        )


# ---------------------------------------------------------------------------
# Terminal reward calculator
# ---------------------------------------------------------------------------

class TerminalRewardCalculator:
    """
    Wrap a trained predictor for terminal RL rewards.

    The predictor must output two objectives in this order:
        [max_stress, toughness]

    Supported predictor forms
    -------------------------
    1. sklearn-like object with .predict(batch_features)
    2. callable receiving batch_features and returning predictions
    3. callable returning a mapping with keys "max_stress" and "toughness"

    Use evaluate_features(features) when your model consumes precomputed
    embeddings or structural descriptors. Use evaluate_pose(pose) when you
    supply a feature_extractor callable.

    Parameters
    ----------
    predictor:
        Trained mechanics predictor or callable wrapper.
    feature_extractor:
        Optional callable: pose -> one-dimensional feature vector.
    weights:
        Weights for scalarizing max_stress and toughness.
    target_mean / target_std:
        Optional dictionaries for z-score normalization. Strongly recommended
        because maximum stress and toughness usually have different units and
        scales. Either supply both dictionaries or neither.
    baseline_features:
        Optional feature vector for wild type. When provided and reward_mode is
        "delta", scalarization uses improvement over wild type.
    reward_mode:
        "absolute" or "delta".
    """

    OUTPUT_NAMES = ("max_stress", "toughness")
    _VALID_REWARD_MODES = {"absolute", "delta"}

    def __init__(
        self,
        predictor: Any,
        *,
        feature_extractor: Optional[FeatureExtractor] = None,
        weights: TerminalRewardWeights = TerminalRewardWeights(),
        target_mean: Optional[Mapping[str, float]] = None,
        target_std: Optional[Mapping[str, float]] = None,
        baseline_features: Optional[ArrayLike] = None,
        reward_mode: str = "absolute",
    ) -> None:
        if predictor is None:
            raise ValueError("predictor must not be None.")
        if reward_mode not in self._VALID_REWARD_MODES:
            raise ValueError(
                f"reward_mode must be one of {sorted(self._VALID_REWARD_MODES)}."
            )
        if (target_mean is None) != (target_std is None):
            raise ValueError("Supply both target_mean and target_std, or neither.")

        self.predictor = predictor
        self.feature_extractor = feature_extractor
        self.weights = weights
        self.reward_mode = reward_mode

        self.target_mean = self._validate_optional_objective_mapping(
            target_mean,
            mapping_name="target_mean",
            allow_zero=True,
        )
        self.target_std = self._validate_optional_objective_mapping(
            target_std,
            mapping_name="target_std",
            allow_zero=False,
        )

        self._baseline_normalized_predictions: Optional[Dict[str, float]] = None
        if baseline_features is not None:
            baseline_raw = self._predict_one(baseline_features)
            self._baseline_normalized_predictions = self._normalize(baseline_raw)

        if self.reward_mode == "delta" and self._baseline_normalized_predictions is None:
            raise ValueError(
                "reward_mode='delta' requires baseline_features from the "
                "wild-type / initial protein."
            )

    def _validate_optional_objective_mapping(
        self,
        values: Optional[Mapping[str, float]],
        *,
        mapping_name: str,
        allow_zero: bool,
    ) -> Optional[Dict[str, float]]:
        if values is None:
            return None

        missing = [name for name in self.OUTPUT_NAMES if name not in values]
        if missing:
            raise ValueError(f"{mapping_name} is missing keys: {missing}.")

        result = {name: float(values[name]) for name in self.OUTPUT_NAMES}

        if not allow_zero:
            invalid = {name: value for name, value in result.items() if value <= 0}
            if invalid:
                raise ValueError(
                    f"{mapping_name} values must be > 0. Invalid values: {invalid}."
                )

        return result

    @staticmethod
    def _as_numpy(value: Any) -> np.ndarray:
        """Convert NumPy / Torch-like output to a NumPy array."""

        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value, dtype=float)

    def _run_predictor(self, batch_features: np.ndarray) -> Any:
        if hasattr(self.predictor, "predict") and callable(self.predictor.predict):
            return self.predictor.predict(batch_features)
        if callable(self.predictor):
            return self.predictor(batch_features)
        raise TypeError(
            "predictor must provide .predict(batch_features) or be callable."
        )

    def _coerce_prediction(self, prediction: Any) -> Dict[str, float]:
        if isinstance(prediction, Mapping):
            missing = [name for name in self.OUTPUT_NAMES if name not in prediction]
            if missing:
                raise ValueError(f"Predictor mapping is missing keys: {missing}.")
            return {name: float(prediction[name]) for name in self.OUTPUT_NAMES}

        array = self._as_numpy(prediction)

        # Accept [stress, toughness] or [[stress, toughness]].
        if array.ndim == 2 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 1 or array.shape[0] != 2:
            raise ValueError(
                "Predictor output must be a mapping or an array with shape "
                "(2,) / (1, 2), ordered as [max_stress, toughness]. "
                f"Received shape {array.shape}."
            )

        return {
            "max_stress": float(array[0]),
            "toughness": float(array[1]),
        }

    def _predict_one(self, features: ArrayLike) -> Dict[str, float]:
        feature_array = np.asarray(features, dtype=float)

        if feature_array.ndim != 1:
            raise ValueError(
                "A single terminal-state feature vector must be one-dimensional. "
                f"Received shape {feature_array.shape}."
            )
        if not np.all(np.isfinite(feature_array)):
            raise ValueError("Feature vector contains NaN or infinity.")

        batch = feature_array.reshape(1, -1)
        # wait: change to a implemented predictor
        prediction = np.array([0., 0.]) # self._run_predictor(batch)
        coerced = self._coerce_prediction(prediction)

        if not all(np.isfinite(value) for value in coerced.values()):
            raise ValueError("Predictor output contains NaN or infinity.")

        return coerced

    def _normalize(self, predictions: Mapping[str, float]) -> Dict[str, float]:
        if self.target_mean is None or self.target_std is None:
            return {
                name: float(predictions[name])
                for name in self.OUTPUT_NAMES
            }

        return {
            name: (
                float(predictions[name]) - self.target_mean[name]
            ) / self.target_std[name]
            for name in self.OUTPUT_NAMES
        }

    def evaluate_features(self, features: ArrayLike) -> TerminalRewardResult:
        """Evaluate terminal reward from a one-dimensional feature vector."""

        raw_predictions = self._predict_one(features)
        normalized = self._normalize(raw_predictions)

        if self.reward_mode == "absolute":
            objective_for_reward = normalized
        else:
            assert self._baseline_normalized_predictions is not None
            objective_for_reward = {
                name: (
                    normalized[name]
                    - self._baseline_normalized_predictions[name]
                )
                for name in self.OUTPUT_NAMES
            }

        components = {
            "max_stress": (
                self.weights.max_stress
                * objective_for_reward["max_stress"]
            ),
            "toughness": (
                self.weights.toughness
                * objective_for_reward["toughness"]
            ),
        }

        return TerminalRewardResult(
            reward=float(sum(components.values())),
            raw_predictions=raw_predictions,
            normalized_predictions=normalized,
            baseline_normalized_predictions=(
                None
                if self._baseline_normalized_predictions is None
                else dict(self._baseline_normalized_predictions)
            ),
            reward_components=components,
            objective_vector=(
                raw_predictions["max_stress"],
                raw_predictions["toughness"],
            ),
        )

    def evaluate_pose(self, pose: Any) -> TerminalRewardResult:
        """Evaluate terminal reward by extracting features from a Pose."""

        if self.feature_extractor is None:
            raise ValueError(
                "evaluate_pose requires feature_extractor=pose_to_features."
            )
        return self.evaluate_features(self.feature_extractor(pose))
