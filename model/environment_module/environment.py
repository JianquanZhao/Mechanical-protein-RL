"""
environment.py

Gymnasium-compatible base environment for RL-based optimization of an existing
mechanical protein structure with PyRosetta.

The default action space has size L * 20, where L is the number of mutable
residue positions and 20 is the number of canonical amino acids. An action is
parsed as:

    mutable_position_index = action // 20
    amino_acid_index       = action % 20

The environment performs:

    load initial pose
    -> decode discrete mutation action
    -> clone current pose
    -> mutate candidate pose
    -> local side-chain repacking
    -> local minimization
    -> call StepRewardCalculator
    -> commit candidate pose only after success
    -> optionally call TerminalRewardCalculator at episode end

The base observation is a flattened one-hot encoding of the current sequence at
mutable positions. A custom observation_encoder can be supplied when the DDQN
state should include structural descriptors or learned embeddings.

PyRosetta is loaded lazily. Importing this module does not require PyRosetta,
which makes unit testing possible with a fake backend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple
import copy
import json

import numpy as np


# ---------------------------------------------------------------------------
# Optional Gymnasium dependency
# ---------------------------------------------------------------------------

try:  # pragma: no cover - exercised only when Gymnasium is installed.
    import gymnasium as gym
    from gymnasium import spaces

    _GymEnvBase = gym.Env
except ImportError:  # Lightweight fallback for research scripts and tests.
    class _GymEnvBase:
        """Small fallback base class used when Gymnasium is unavailable."""

        metadata: Mapping[str, Any] = {}

    class _DiscreteSpace:
        def __init__(self, n: int) -> None:
            if int(n) <= 0:
                raise ValueError("Discrete space size must be positive.")
            self.n = int(n)
            self._rng = np.random.default_rng()

        def seed(self, seed: Optional[int] = None) -> None:
            self._rng = np.random.default_rng(seed)

        def sample(self, mask: Optional[np.ndarray] = None) -> int:
            if mask is None:
                return int(self._rng.integers(self.n))
            mask_array = np.asarray(mask, dtype=bool).reshape(-1)
            valid = np.flatnonzero(mask_array)
            if len(valid) == 0:
                raise RuntimeError("No valid actions are available.")
            return int(self._rng.choice(valid))

    class _BoxSpace:
        def __init__(self, low: Any, high: Any, shape: Sequence[int], dtype: Any) -> None:
            self.low = low
            self.high = high
            self.shape = tuple(int(value) for value in shape)
            self.dtype = np.dtype(dtype)

    class _Spaces:
        Discrete = _DiscreteSpace
        Box = _BoxSpace

    spaces = _Spaces()


# ---------------------------------------------------------------------------
# Constants and small data objects
# ---------------------------------------------------------------------------

CANONICAL_AMINO_ACIDS: Tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_ONE_TO_THREE: Mapping[str, str] = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}

ObservationEncoder = Callable[[Any, "MechanicalProteinEnv"], np.ndarray]


@dataclass(frozen=True)
class MutationAction:
    """Decoded representation of one integer DDQN action."""

    action_index: int
    mutable_position_index: int
    pose_position: int
    previous_amino_acid: str
    target_amino_acid: str

    @property
    def is_noop(self) -> bool:
        return self.previous_amino_acid == self.target_amino_acid

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransitionRecord:
    """Compact episode log entry suitable for JSON serialization."""

    step_index: int
    action: Mapping[str, Any]
    accepted: bool
    reason: str
    step_reward: float
    terminal_reward: float
    total_reward: float
    sequence: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Backend protocol and default PyRosetta implementation
# ---------------------------------------------------------------------------

class PoseBackend(Protocol):
    """Backend contract used by MechanicalProteinEnv."""

    scorefxn: Any

    def load_pose(self, pdb_path: str) -> Any:
        ...

    def clone_pose(self, pose: Any) -> Any:
        ...

    def total_residue(self, pose: Any) -> int:
        ...

    def residue_name1(self, pose: Any, position: int) -> str:
        ...

    def local_residues(self, pose: Any, center_position: int, radius: float) -> Tuple[int, ...]:
        ...

    def mutate(self, pose: Any, position: int, amino_acid: str) -> None:
        ...

    def repack(self, pose: Any, local_residues: Sequence[int]) -> None:
        ...

    def minimize(
        self,
        pose: Any,
        local_residues: Sequence[int],
        *,
        minimize_backbone: bool,
    ) -> None:
        ...

    def dump_pose(self, pose: Any, output_path: str) -> None:
        ...


class PyRosettaPoseBackend:
    """
    Default PyRosetta implementation of structure updates.

    Local repacking changes side-chain rotamers only in the mutation-centered
    neighborhood. Local minimization enables chi torsions in that neighborhood;
    optional backbone minimization can also enable local bb torsions.
    """

    def __init__(
        self,
        *,
        scorefxn: Optional[Any] = None,
        pyrosetta_init_options: str = "-mute all",
        minimization_type: str = "lbfgs_armijo_nonmonotone",
        minimization_tolerance: float = 1e-3,
    ) -> None:
        if minimization_tolerance <= 0:
            raise ValueError("minimization_tolerance must be > 0.")

        try:
            import pyrosetta
        except ImportError as exc:  # pragma: no cover - needs PyRosetta install.
            raise ImportError(
                "PyRosetta is required by the default environment backend. "
                "Install and license PyRosetta, or inject a custom backend for tests."
            ) from exc

        self.pyrosetta = pyrosetta
        self._initialize_pyrosetta_once(pyrosetta, pyrosetta_init_options)
        self.scorefxn = scorefxn if scorefxn is not None else pyrosetta.get_fa_scorefxn()
        self.minimization_type = str(minimization_type)
        self.minimization_tolerance = float(minimization_tolerance)

    @staticmethod
    def _initialize_pyrosetta_once(pyrosetta: Any, options: str) -> None:
        try:
            already_initialized = bool(pyrosetta.rosetta.basic.was_init_called())
        except AttributeError:  # Older PyRosetta builds may not expose this helper.
            already_initialized = False

        if not already_initialized:
            pyrosetta.init(str(options))

    def load_pose(self, pdb_path: str) -> Any:
        path = Path(pdb_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Initial PDB file does not exist: {path}")
        return self.pyrosetta.pose_from_pdb(str(path))

    @staticmethod
    def clone_pose(pose: Any) -> Any:
        return pose.clone()

    @staticmethod
    def total_residue(pose: Any) -> int:
        return int(pose.total_residue())

    @staticmethod
    def residue_name1(pose: Any, position: int) -> str:
        return str(pose.residue(int(position)).name1()).upper()

    @staticmethod
    def _xyz_to_numpy(xyz: Any) -> np.ndarray:
        return np.asarray([float(xyz.x), float(xyz.y), float(xyz.z)], dtype=float)

    def local_residues(self, pose: Any, center_position: int, radius: float) -> Tuple[int, ...]:
        if radius <= 0:
            raise ValueError("radius must be > 0.")

        total = self.total_residue(pose)
        center_position = int(center_position)
        if not 1 <= center_position <= total:
            raise IndexError(f"Residue position {center_position} is outside 1..{total}.")

        center_xyz = self._xyz_to_numpy(pose.residue(center_position).nbr_atom_xyz())
        radius_squared = float(radius) ** 2
        neighborhood: List[int] = []

        for residue_index in range(1, total + 1):
            xyz = self._xyz_to_numpy(pose.residue(residue_index).nbr_atom_xyz())
            if float(np.sum((xyz - center_xyz) ** 2)) <= radius_squared:
                neighborhood.append(residue_index)

        if center_position not in neighborhood:
            neighborhood.append(center_position)

        return tuple(sorted(set(neighborhood)))

    @staticmethod
    def mutate(pose: Any, position: int, amino_acid: str) -> None:
        from pyrosetta.rosetta.protocols.simple_moves import MutateResidue

        amino_acid = str(amino_acid).upper()
        if amino_acid not in AA_ONE_TO_THREE:
            raise ValueError(f"Unsupported amino acid: {amino_acid!r}.")

        mover = MutateResidue()
        mover.set_target(int(position))
        mover.set_res_name(AA_ONE_TO_THREE[amino_acid])
        mover.apply(pose)

    def repack(self, pose: Any, local_residues: Sequence[int]) -> None:
        from pyrosetta import standard_packer_task
        from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover

        allowed = {int(index) for index in local_residues}
        task = standard_packer_task(pose)
        task.restrict_to_repacking()

        for residue_index in range(1, self.total_residue(pose) + 1):
            if residue_index not in allowed:
                task.nonconst_residue_task(residue_index).prevent_repacking()

        mover = PackRotamersMover(self.scorefxn, task)
        mover.apply(pose)

    def minimize(
        self,
        pose: Any,
        local_residues: Sequence[int],
        *,
        minimize_backbone: bool,
    ) -> None:
        from pyrosetta.rosetta.core.kinematics import MoveMap
        from pyrosetta.rosetta.protocols.minimization_packing import MinMover

        move_map = MoveMap()
        move_map.set_bb(False)
        move_map.set_chi(False)
        move_map.set_jump(False)

        for residue_index in local_residues:
            move_map.set_chi(int(residue_index), True)
            if minimize_backbone:
                move_map.set_bb(int(residue_index), True)

        mover = MinMover()
        mover.movemap(move_map)
        mover.score_function(self.scorefxn)
        mover.min_type(self.minimization_type)
        mover.tolerance(self.minimization_tolerance)
        mover.apply(pose)

    @staticmethod
    def dump_pose(pose: Any, output_path: str) -> None:
        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        pose.dump_pdb(str(path))


# ---------------------------------------------------------------------------
# Reward-module import helper
# ---------------------------------------------------------------------------

def _load_step_reward_calculator_class() -> Any:
    """Import StepRewardCalculator under package and script-style layouts."""

    try:
        from ..reward_module.reward_calculators import StepRewardCalculator
        return StepRewardCalculator
    except ImportError:
        pass

    try:
        from model.reward_module.reward_calculators import StepRewardCalculator
        return StepRewardCalculator
    except ImportError:
        pass

    try:
        from reward_module.reward_calculators import StepRewardCalculator
        return StepRewardCalculator
    except ImportError as exc:
        raise ImportError(
            "Cannot import StepRewardCalculator. Place reward_calculators.py in "
            "model/reward_module and add __init__.py files to the package directories."
        ) from exc


# ---------------------------------------------------------------------------
# Main RL environment
# ---------------------------------------------------------------------------

class MechanicalProteinEnv(_GymEnvBase):
    """
    RL environment for iterative amino-acid optimization of a protein Pose.

    Parameters
    ----------
    initial_pdb_path:
        Initial / wild-type PDB structure.
    max_steps:
        Maximum number of mutation actions in one episode. Reaching max_steps
        returns truncated=True and triggers the optional terminal reward.
    mutable_positions:
        Optional 1-indexed Pose residue positions open to mutation. If omitted,
        all Pose residues are mutable. Use a mechanical-lock region here to
        reduce the search space.
    step_reward_calculator:
        Optional preconstructed calculator. If omitted, the environment builds
        StepRewardCalculator(reference_pose, **step_reward_kwargs).
    terminal_reward_calculator:
        Optional TerminalRewardCalculator. When supplied, its evaluate_pose()
        method is called at episode end and added to the last step reward.
    observation_encoder:
        Optional callable: (pose, env) -> np.ndarray. The default observation
        is a sequence one-hot vector at mutable positions.
    flatten_observation:
        Flatten the default L x 20 one-hot observation to shape (L*20,).
    local_repack_radius:
        Radius in Angstrom for local repacking and local minimization.
    perform_repack / perform_minimize:
        Enable local side-chain rotamer packing and minimization after mutation.
    minimize_backbone:
        If False, minimize local chi torsions only. Start with False for a
        conservative base version; later compare against local backbone=True.
    prevent_revisit_positions:
        If True, a residue position can be mutated at most once per episode.
    invalid_action_penalty:
        Reward used for a masked action, such as changing a residue to its
        current amino acid or revisiting a protected position.
    update_error_penalty:
        Reward used when mutation/repacking/minimization fails and
        raise_on_update_error=False.
    raise_on_update_error:
        Raise structure-update errors during development. Set False during
        long training runs to roll back failed candidates and continue.
    include_action_mask_in_info:
        Include the boolean L*20 action mask in reset()/step() info.
    truncate_when_no_valid_actions:
        End an episode early when all actions are masked, for example after
        every mutable position has been used with prevent_revisit_positions=True.
    backend:
        Optional custom PoseBackend. Omit to use PyRosettaPoseBackend.

    Gymnasium API
    -------------
    reset(...) -> observation, info
    step(action) -> observation, reward, terminated, truncated, info

    This environment has no biological terminal condition in the base version,
    so terminated is always False. Episodes end through truncation at max_steps.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        initial_pdb_path: str,
        *,
        max_steps: int = 5,
        mutable_positions: Optional[Sequence[int]] = None,
        amino_acids: Sequence[str] = CANONICAL_AMINO_ACIDS,
        step_reward_calculator: Optional[Any] = None,
        step_reward_kwargs: Optional[Mapping[str, Any]] = None,
        terminal_reward_calculator: Optional[Any] = None,
        observation_encoder: Optional[ObservationEncoder] = None,
        flatten_observation: bool = True,
        local_repack_radius: float = 8.0,
        perform_repack: bool = True,
        perform_minimize: bool = True,
        minimize_backbone: bool = False,
        prevent_revisit_positions: bool = False,
        invalid_action_penalty: float = -5.0,
        update_error_penalty: float = -10.0,
        raise_on_update_error: bool = True,
        step_reward_scale: float = 1.0,
        terminal_reward_scale: float = 1.0,
        include_action_mask_in_info: bool = True,
        truncate_when_no_valid_actions: bool = True,
        backend: Optional[PoseBackend] = None,
        pyrosetta_init_options: str = "-mute all",
        seed: Optional[int] = None,
    ) -> None:
        if int(max_steps) <= 0:
            raise ValueError("max_steps must be a positive integer.")
        if float(local_repack_radius) <= 0:
            raise ValueError("local_repack_radius must be > 0.")
        if not amino_acids:
            raise ValueError("amino_acids must not be empty.")

        amino_acid_tuple = tuple(str(aa).upper() for aa in amino_acids)
        if len(set(amino_acid_tuple)) != len(amino_acid_tuple):
            raise ValueError("amino_acids must not contain duplicates.")
        unknown = sorted(set(amino_acid_tuple) - set(CANONICAL_AMINO_ACIDS))
        if unknown:
            raise ValueError(f"Only canonical amino acids are supported: {unknown}.")

        self.initial_pdb_path = str(Path(initial_pdb_path).expanduser())
        self.max_steps = int(max_steps)
        self.amino_acids = amino_acid_tuple
        self.n_amino_acids = len(self.amino_acids)
        self.local_repack_radius = float(local_repack_radius)
        self.perform_repack = bool(perform_repack)
        self.perform_minimize = bool(perform_minimize)
        self.minimize_backbone = bool(minimize_backbone)
        self.prevent_revisit_positions = bool(prevent_revisit_positions)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.update_error_penalty = float(update_error_penalty)
        self.raise_on_update_error = bool(raise_on_update_error)
        self.step_reward_scale = float(step_reward_scale)
        self.terminal_reward_scale = float(terminal_reward_scale)
        self.include_action_mask_in_info = bool(include_action_mask_in_info)
        self.truncate_when_no_valid_actions = bool(truncate_when_no_valid_actions)
        self.observation_encoder = observation_encoder
        self.flatten_observation = bool(flatten_observation)

        self.backend: PoseBackend = (
            backend
            if backend is not None
            else PyRosettaPoseBackend(pyrosetta_init_options=pyrosetta_init_options)
        )

        loaded_pose = self.backend.load_pose(self.initial_pdb_path)
        self.reference_pose = self.backend.clone_pose(loaded_pose)
        self.current_pose = self.backend.clone_pose(loaded_pose)
        self.total_residues = int(self.backend.total_residue(self.reference_pose))
        if self.total_residues <= 0:
            raise ValueError("Initial pose contains no residues.")

        if mutable_positions is None:
            # Exclude ligands and non-canonical residues from the default
            # mutation space. Users may still include selected positions
            # explicitly when a specialized workflow requires it.
            positions = tuple(
                index
                for index in range(1, self.total_residues + 1)
                if self.backend.residue_name1(self.reference_pose, index)
                in CANONICAL_AMINO_ACIDS
            )
            if not positions:
                raise ValueError(
                    "No canonical protein residues were found in the initial Pose. "
                    "Provide mutable_positions explicitly if this is intentional."
                )
        else:
            positions = tuple(int(index) for index in mutable_positions)
            if not positions:
                raise ValueError("mutable_positions must not be empty.")
            if len(set(positions)) != len(positions):
                raise ValueError("mutable_positions must not contain duplicates.")
            invalid_positions = [
                index for index in positions if not 1 <= index <= self.total_residues
            ]
            if invalid_positions:
                raise IndexError(
                    f"mutable_positions outside 1..{self.total_residues}: {invalid_positions}."
                )

        self.mutable_positions = positions
        self.n_mutable_positions = len(self.mutable_positions)
        self.n_actions = self.n_mutable_positions * self.n_amino_acids

        if step_reward_calculator is None:
            calculator_class = _load_step_reward_calculator_class()
            kwargs = dict(step_reward_kwargs or {})
            if "scorefxn" not in kwargs and hasattr(self.backend, "scorefxn"):
                kwargs["scorefxn"] = self.backend.scorefxn
            self.step_reward_calculator = calculator_class(self.reference_pose, **kwargs)
        else:
            self.step_reward_calculator = step_reward_calculator

        self.terminal_reward_calculator = terminal_reward_calculator

        self.action_space = spaces.Discrete(self.n_actions)
        self._rng = np.random.default_rng(seed)
        if hasattr(self.action_space, "seed"):
            self.action_space.seed(seed)

        self.current_step = 0
        self.accepted_mutation_count = 0
        self.visited_positions: set[int] = set()
        self.history: List[TransitionRecord] = []
        self._episode_finalized = False

        initial_observation = self._encode_observation()
        self.observation_space = spaces.Box(
            low=0.0 if self.observation_encoder is None else -np.inf,
            high=1.0 if self.observation_encoder is None else np.inf,
            shape=initial_observation.shape,
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Observation and sequence helpers
    # ------------------------------------------------------------------

    def current_sequence(self, *, mutable_only: bool = False) -> str:
        positions = (
            self.mutable_positions
            if mutable_only
            else tuple(range(1, self.total_residues + 1))
        )
        return "".join(self.backend.residue_name1(self.current_pose, index) for index in positions)

    def reference_sequence(self, *, mutable_only: bool = False) -> str:
        positions = (
            self.mutable_positions
            if mutable_only
            else tuple(range(1, self.total_residues + 1))
        )
        return "".join(self.backend.residue_name1(self.reference_pose, index) for index in positions)

    def _default_sequence_observation(self) -> np.ndarray:
        observation = np.zeros(
            (self.n_mutable_positions, self.n_amino_acids),
            dtype=np.float32,
        )
        aa_to_index = {aa: index for index, aa in enumerate(self.amino_acids)}

        for mutable_index, pose_position in enumerate(self.mutable_positions):
            current_aa = self.backend.residue_name1(self.current_pose, pose_position)
            if current_aa in aa_to_index:
                observation[mutable_index, aa_to_index[current_aa]] = 1.0

        if self.flatten_observation:
            return observation.reshape(-1)
        return observation

    def _encode_observation(self) -> np.ndarray:
        if self.observation_encoder is None:
            array = self._default_sequence_observation()
        else:
            array = np.asarray(
                self.observation_encoder(self.current_pose, self),
                dtype=np.float32,
            )

        if array.size == 0:
            raise ValueError("Observation encoder returned an empty array.")
        if not np.all(np.isfinite(array)):
            raise ValueError("Observation contains NaN or infinity.")
        return np.asarray(array, dtype=np.float32)

    def get_pose(self, *, clone: bool = True) -> Any:
        """Return the current Pose. Clone by default to protect environment state."""

        if clone:
            return self.backend.clone_pose(self.current_pose)
        return self.current_pose

    # ------------------------------------------------------------------
    # Action decoding and masking
    # ------------------------------------------------------------------

    def decode_action(self, action: int) -> MutationAction:
        """Decode one integer action from the L * 20 action space."""

        if isinstance(action, np.ndarray):
            if action.size != 1:
                raise ValueError(f"Action array must contain one integer, got shape {action.shape}.")
            action = int(action.reshape(-1)[0])
        elif isinstance(action, (np.integer, int)):
            action = int(action)
        else:
            raise TypeError(f"Action must be an integer, got {type(action).__name__}.")

        if not 0 <= action < self.n_actions:
            raise IndexError(f"Action {action} is outside 0..{self.n_actions - 1}.")

        mutable_index = action // self.n_amino_acids
        aa_index = action % self.n_amino_acids
        pose_position = self.mutable_positions[mutable_index]
        previous_aa = self.backend.residue_name1(self.current_pose, pose_position)
        target_aa = self.amino_acids[aa_index]

        return MutationAction(
            action_index=action,
            mutable_position_index=mutable_index,
            pose_position=pose_position,
            previous_amino_acid=previous_aa,
            target_amino_acid=target_aa,
        )

    def encode_action(self, pose_position: int, amino_acid: str) -> int:
        """Return the integer action for a Pose position and target amino acid."""

        pose_position = int(pose_position)
        amino_acid = str(amino_acid).upper()
        if pose_position not in self.mutable_positions:
            raise ValueError(
                f"Pose position {pose_position} is not included in mutable_positions."
            )
        if amino_acid not in self.amino_acids:
            raise ValueError(
                f"Amino acid {amino_acid!r} is not included in the action alphabet."
            )

        mutable_index = self.mutable_positions.index(pose_position)
        aa_index = self.amino_acids.index(amino_acid)
        return mutable_index * self.n_amino_acids + aa_index

    def action_mask(self) -> np.ndarray:
        """
        Return a boolean L*20 mask. True entries are valid mutation actions.

        The mask removes no-op mutations. When prevent_revisit_positions=True,
        it also removes all actions for positions already mutated in the episode.
        """

        mask = np.ones(self.n_actions, dtype=bool)

        for mutable_index, pose_position in enumerate(self.mutable_positions):
            block_start = mutable_index * self.n_amino_acids
            if self.prevent_revisit_positions and pose_position in self.visited_positions:
                mask[block_start : block_start + self.n_amino_acids] = False
                continue

            current_aa = self.backend.residue_name1(self.current_pose, pose_position)
            if current_aa in self.amino_acids:
                current_aa_index = self.amino_acids.index(current_aa)
                mask[block_start + current_aa_index] = False

        return mask

    def sample_valid_action(self) -> int:
        valid_actions = np.flatnonzero(self.action_mask())
        if len(valid_actions) == 0:
            raise RuntimeError("No valid mutation actions remain.")
        return int(self._rng.choice(valid_actions))

    def _invalid_reason(self, decoded_action: MutationAction) -> Optional[str]:
        if decoded_action.is_noop:
            return "noop_same_amino_acid"
        if (
            self.prevent_revisit_positions
            and decoded_action.pose_position in self.visited_positions
        ):
            return "position_already_mutated"
        return None

    # ------------------------------------------------------------------
    # Gymnasium-style episode API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset to the wild-type structure and return (observation, info)."""

        del options  # Reserved for future episode initialization variants.

        if seed is not None:
            self._rng = np.random.default_rng(seed)
            if hasattr(self.action_space, "seed"):
                self.action_space.seed(seed)

        self.current_pose = self.backend.clone_pose(self.reference_pose)
        self.current_step = 0
        self.accepted_mutation_count = 0
        self.visited_positions = set()
        self.history = []
        self._episode_finalized = False

        observation = self._encode_observation()
        info = self._base_info()
        info["event"] = "reset"
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Apply one mutation action and return Gymnasium's five-value tuple."""

        if self._episode_finalized:
            raise RuntimeError("Episode is finalized. Call reset() before step().")

        decoded = self.decode_action(action)
        invalid_reason = self._invalid_reason(decoded)
        step_reward_result: Optional[Any] = None
        accepted = False
        reason = "accepted"
        error_message: Optional[str] = None

        if invalid_reason is not None:
            step_reward = self.invalid_action_penalty
            reason = invalid_reason
        else:
            previous_pose = self.backend.clone_pose(self.current_pose)
            candidate_pose = self.backend.clone_pose(self.current_pose)
            local_residues = self.backend.local_residues(
                candidate_pose,
                decoded.pose_position,
                self.local_repack_radius,
            )

            try:
                self.backend.mutate(
                    candidate_pose,
                    decoded.pose_position,
                    decoded.target_amino_acid,
                )

                if self.perform_repack:
                    self.backend.repack(candidate_pose, local_residues)

                if self.perform_minimize:
                    self.backend.minimize(
                        candidate_pose,
                        local_residues,
                        minimize_backbone=self.minimize_backbone,
                    )

                step_reward_result = self.step_reward_calculator.evaluate(
                    candidate_pose,
                    previous_pose=previous_pose,
                    mutated_positions=[decoded.pose_position],
                    local_residues=local_residues,
                )
                step_reward = self.step_reward_scale * float(step_reward_result.reward)

            except Exception as exc:
                if self.raise_on_update_error:
                    raise
                step_reward = self.update_error_penalty
                reason = "structure_update_error"
                error_message = f"{type(exc).__name__}: {exc}"
            else:
                self.current_pose = candidate_pose
                self.visited_positions.add(decoded.pose_position)
                self.accepted_mutation_count += 1
                accepted = True

        self.current_step += 1
        terminated = False
        no_valid_actions_remain = not bool(np.any(self.action_mask()))
        truncated_by_step_limit = self.current_step >= self.max_steps
        truncated_by_action_exhaustion = (
            self.truncate_when_no_valid_actions and no_valid_actions_remain
        )
        truncated = truncated_by_step_limit or truncated_by_action_exhaustion

        if truncated_by_step_limit:
            truncation_reason: Optional[str] = "max_steps_reached"
        elif truncated_by_action_exhaustion:
            truncation_reason = "no_valid_actions_remain"
        else:
            truncation_reason = None

        terminal_reward = 0.0
        terminal_reward_result: Optional[Any] = None
        if truncated:
            terminal_reward, terminal_reward_result = self._finalize_episode()

        total_reward = float(step_reward + terminal_reward)
        observation = self._encode_observation()
        info = self._base_info()
        info.update(
            {
                "event": "step",
                "decoded_action": decoded.to_dict(),
                "accepted": accepted,
                "reason": reason,
                "step_reward": float(step_reward),
                "terminal_reward": float(terminal_reward),
                "total_reward": total_reward,
                "truncation_reason": truncation_reason,
                "step_reward_metrics": self._result_to_dict(step_reward_result),
                "terminal_reward_metrics": self._result_to_dict(terminal_reward_result),
            }
        )
        if error_message is not None:
            info["error"] = error_message

        self.history.append(
            TransitionRecord(
                step_index=self.current_step,
                action=decoded.to_dict(),
                accepted=accepted,
                reason=reason,
                step_reward=float(step_reward),
                terminal_reward=float(terminal_reward),
                total_reward=total_reward,
                sequence=self.current_sequence(),
            )
        )

        return observation, total_reward, terminated, truncated, info

    def finalize_episode(self) -> Tuple[float, Dict[str, Any]]:
        """
        Manually finalize an episode before max_steps.

        This is useful when an external training loop applies its own early-stop
        rule. The terminal reward is calculated at most once.
        """

        terminal_reward, result = self._finalize_episode()
        return float(terminal_reward), self._result_to_dict(result) or {}

    def _finalize_episode(self) -> Tuple[float, Optional[Any]]:
        if self._episode_finalized:
            return 0.0, None

        self._episode_finalized = True
        if self.terminal_reward_calculator is None:
            return 0.0, None

        result = self.terminal_reward_calculator.evaluate_pose(self.current_pose)
        return self.terminal_reward_scale * float(result.reward), result

    # ------------------------------------------------------------------
    # Logging and output helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _result_to_dict(result: Optional[Any]) -> Optional[Dict[str, Any]]:
        if result is None:
            return None
        if hasattr(result, "to_dict") and callable(result.to_dict):
            return dict(result.to_dict())
        if hasattr(result, "__dict__"):
            return dict(result.__dict__)
        return {"value": result}

    def _base_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "step_index": self.current_step,
            "max_steps": self.max_steps,
            "accepted_mutation_count": self.accepted_mutation_count,
            "sequence": self.current_sequence(),
            "mutable_sequence": self.current_sequence(mutable_only=True),
            "visited_positions": tuple(sorted(self.visited_positions)),
            "valid_action_count": int(np.count_nonzero(self.action_mask())),
        }
        if self.include_action_mask_in_info:
            info["action_mask"] = self.action_mask()
        return info

    def save_current_pose(self, output_path: str) -> None:
        """Write the current candidate structure to a PDB file."""

        self.backend.dump_pose(self.current_pose, output_path)

    def save_history(self, output_path: str) -> None:
        """Write the current episode transition log to JSON."""

        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [record.to_dict() for record in self.history]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def render(self) -> None:
        print(
            f"step={self.current_step}/{self.max_steps} "
            f"accepted={self.accepted_mutation_count} "
            f"sequence={self.current_sequence()}"
        )

    def close(self) -> None:
        """No external resources need to be closed in the base version."""


# Backward-friendly aliases.
ProteinMutationEnv = MechanicalProteinEnv
Env = MechanicalProteinEnv
