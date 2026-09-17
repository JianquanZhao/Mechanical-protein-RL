"""The only data-generation module allowed to import the existing RL model package."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np


LOGGER = logging.getLogger(__name__)


class _UnusedStepRewardCalculator:
    """Marker override that prevents source inspection from building reward state."""

    def evaluate(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - never called.
        raise RuntimeError("SourceStructureInspector does not execute environment steps.")


class SourceStructureInspector:
    """Read the exact sequence/action length produced by the RL PDB loader."""

    def __init__(self, reward_config: Mapping[str, Any]) -> None:
        from model.environment_module.environment import MechanicalProteinEnv

        self.env = MechanicalProteinEnv(
            max_steps=1,
            perform_repack=False,
            perform_minimize=False,
            step_reward_calculator=_UnusedStepRewardCalculator(),
            terminal_reward_calculator=None,
            observation_encoder=None,
            flatten_observation=False,
            pyrosetta_init_options=str(reward_config.get("pyrosetta_init_options", "-mute all")),
            clean_pdb_before_load=bool(reward_config.get("clean_pdb_before_load", True)),
            load_max_missing_backbone_fraction=float(
                reward_config.get("max_missing_backbone_fraction", 0.05)
            ),
        )

    def inspect(self, pdb_path: str | Path) -> tuple[str, int]:
        _, info = self.env.reset(pdb_path=str(pdb_path))
        pose = self.env.get_pose(clone=False)
        chain_count = int(pose.num_chains()) if hasattr(pose, "num_chains") else 1
        return str(info["sequence"]), chain_count


def _nested(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


class RLContractAdapter:
    """Apply exactly one RL mutation and expose stable, flattened reward fields."""

    def __init__(self, reward_config: Mapping[str, Any]) -> None:
        from model.environment_module.environment import MechanicalProteinEnv
        from model.reward_module.terminal_reward import (
            MechanicalImprovementTerminalRewardCalculator,
        )

        artifact_path = Path(reward_config["artifact_path"]).expanduser().resolve()
        calculator = MechanicalImprovementTerminalRewardCalculator(artifact_path=artifact_path)
        self.reward_config = dict(reward_config)
        self.calculator = calculator
        self.env = MechanicalProteinEnv(
            max_steps=1,
            local_repack_radius=float(reward_config.get("local_repack_radius", 8.0)),
            perform_repack=bool(reward_config.get("perform_repack", True)),
            perform_minimize=bool(reward_config.get("perform_minimize", False)),
            minimize_backbone=bool(reward_config.get("minimize_backbone", False)),
            prevent_revisit_positions=True,
            include_visited_mask_in_observation=False,
            raise_on_update_error=False,
            step_reward_scale=float(reward_config.get("step_reward_scale", 0.025)),
            terminal_reward_scale=float(reward_config.get("terminal_reward_scale", 8.0)),
            terminal_reward_calculator=calculator,
            observation_encoder=None,
            flatten_observation=False,
            step_reward_kwargs={
                "rmsd_missing_atom_policy": str(
                    reward_config.get("rmsd_missing_atom_policy", "penalize")
                ),
                "rmsd_missing_penalty": float(reward_config.get("rmsd_missing_penalty", 5.0)),
                "min_rmsd_atoms": int(reward_config.get("min_rmsd_atoms", 3)),
            },
            pyrosetta_init_options=str(reward_config.get("pyrosetta_init_options", "-mute all")),
            clean_pdb_before_load=bool(reward_config.get("clean_pdb_before_load", True)),
            load_max_missing_backbone_fraction=float(
                reward_config.get("max_missing_backbone_fraction", 0.05)
            ),
        )

    @staticmethod
    def _set_rosetta_seed(seed: int) -> bool:
        try:
            import pyrosetta

            random_generator = pyrosetta.rosetta.numeric.random.rg()
            if hasattr(random_generator, "set_seed"):
                random_generator.set_seed(int(seed))
                return True
        except Exception:
            LOGGER.debug("Unable to set the PyRosetta RNG seed", exc_info=True)
        return False

    def scan_task(
        self,
        task: Mapping[str, Any],
        *,
        candidate_structure_path: str | Path | None = None,
    ) -> dict[str, Any]:
        seed = int(task["repeat_seed"])
        seed_applied = self._set_rosetta_seed(seed)
        _, reset_info = self.env.reset(seed=seed, pdb_path=str(task["pose_path"]))
        action_index = int(task["action_index"])
        decoded = self.env.decode_action(action_index)
        if decoded.pose_position != int(task["position"]):
            raise RuntimeError("Offline task action mapping differs from MechanicalProteinEnv.")
        _, total_reward, _, truncated, info = self.env.step(action_index)
        if not truncated:
            raise RuntimeError("One-action scan did not finalize its episode.")
        metrics = info.get("terminal_reward_metrics") or {}
        if not info.get("accepted", False):
            raise RuntimeError(str(info.get("error") or info.get("reason") or "mutation rejected"))

        initial = _nested(metrics, "initial_result", default={}) or {}
        final = _nested(metrics, "final_result", default={}) or {}
        marginal_raw = float(metrics.get("reward", 0.0))
        terminal_scale = float(self.reward_config.get("terminal_reward_scale", 8.0))

        root_path = str(task.get("root_pdb_path") or task["pose_path"])
        if Path(root_path).expanduser().resolve() == Path(task["pose_path"]).expanduser().resolve():
            root = initial
        else:
            root = self.calculator.base_calculator.evaluate_pdb(root_path).to_dict()
        delta_strength_root = float(final["normalized_strength"] - root["normalized_strength"])
        delta_toughness_root = float(final["normalized_toughness"] - root["normalized_toughness"])
        absolute_raw = 0.5 * (delta_strength_root + delta_toughness_root)

        saved_path = ""
        if candidate_structure_path is not None:
            output_path = Path(candidate_structure_path).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.env.save_current_pose(str(output_path))
            saved_path = str(output_path)

        return {
            "sequence_before": str(reset_info["sequence"]),
            "sequence_after": str(info["sequence"]),
            "accepted": True,
            "rosetta_seed_applied": seed_applied,
            "step_reward_scaled": float(info["step_reward"]),
            "terminal_reward_scaled": float(info["terminal_reward"]),
            "total_reward_scaled": float(total_reward),
            "marginal_reward_raw": marginal_raw,
            "marginal_reward_scaled": marginal_raw * terminal_scale,
            "absolute_reward_raw": absolute_raw,
            "absolute_reward_scaled": absolute_raw * terminal_scale,
            "delta_strength_marginal": float(metrics["delta_normalized_strength"]),
            "delta_toughness_marginal": float(metrics["delta_normalized_toughness"]),
            "delta_strength_absolute": delta_strength_root,
            "delta_toughness_absolute": delta_toughness_root,
            "strength_before": float(initial["strength"]),
            "toughness_before": float(initial["toughness"]),
            "strength_after": float(final["strength"]),
            "toughness_after": float(final["toughness"]),
            "normalized_strength_before": float(initial["normalized_strength"]),
            "normalized_toughness_before": float(initial["normalized_toughness"]),
            "normalized_strength_after": float(final["normalized_strength"]),
            "normalized_toughness_after": float(final["normalized_toughness"]),
            "features_before_json": json.dumps(initial.get("features", {}), sort_keys=True),
            "features_after_json": json.dumps(final.get("features", {}), sort_keys=True),
            "step_metrics_json": json.dumps(info.get("step_reward_metrics") or {}, sort_keys=True),
            "candidate_structure_path": saved_path,
        }
