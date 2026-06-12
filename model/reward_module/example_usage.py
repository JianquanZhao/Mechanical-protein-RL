"""
example_usage.py

Minimal integration example. The mutation / repack / relax operation remains
inside your RL environment and should happen before StepRewardCalculator is
called.
"""

import joblib
import numpy as np
import pyrosetta

from reward_calculators import (
    StepRewardCalculator,
    StepRewardScales,
    StepRewardWeights,
    TerminalRewardCalculator,
    TerminalRewardWeights,
)

pyrosetta.init("-mute all")
wild_type_pose = pyrosetta.pose_from_pdb("wild_type.pdb")

step_reward = StepRewardCalculator(
    wild_type_pose,
    weights=StepRewardWeights(
        collision=1.0,
        backbone_hbond=1.0,
        sidechain_hbond=0.5,
        local_rmsd=1.0,
    ),
    scales=StepRewardScales(
        collision=10.0,      # Replace with pilot-set statistics.
        backbone_hbond=1.0,
        sidechain_hbond=1.0,
        local_rmsd=0.5,      # Angstrom.
    ),
    hbond_energy_cutoff=0.0,
    neighborhood_radius=8.0,
    rmsd_penalty_mode="previous",
    collision_penalty_mode="delta",
)

previous_pose = wild_type_pose.clone()
current_pose = previous_pose.clone()

# -------------------------------------------------------------
# Your RL environment applies an action here:
#     mutate(current_pose, position=42, amino_acid="V")
#     local_repack(current_pose, center_position=42)
#     local_relax(current_pose, center_position=42)
# -------------------------------------------------------------

step_result = step_reward.evaluate(
    current_pose,
    previous_pose=previous_pose,
    mutated_positions=[42],
)
print("Step reward:", step_result.reward)
print(step_result.to_dict())


# A placeholder structural feature extractor.
# Replace this with exactly the same feature pipeline used to train your model.
def pose_to_mechanics_features(pose) -> np.ndarray:
    return np.asarray(
        [
            pose.total_residue(),
            step_reward._collision_score(pose),
            step_reward._count_hbonds(pose).backbone,
            step_reward._count_hbonds(pose).sidechain_involving,
        ],
        dtype=float,
    )

# wait: change a usefule predictor
# predictor = joblib.load("mechanics_predictor.joblib")
predictor = 0
wild_type_features = pose_to_mechanics_features(wild_type_pose)

terminal_reward = TerminalRewardCalculator(
    predictor,
    feature_extractor=pose_to_mechanics_features,
    weights=TerminalRewardWeights(
        max_stress=1.0,
        toughness=1.0,
    ),
    target_mean={
        "max_stress": 120.0,  # Replace with training-set mean.
        "toughness": 20.0,
    },
    target_std={
        "max_stress": 30.0,   # Replace with training-set standard deviation.
        "toughness": 5.0,
    },
    baseline_features=wild_type_features,
    reward_mode="delta",
)

terminal_result = terminal_reward.evaluate_pose(current_pose)
print("Terminal reward:", terminal_result.reward)
print(terminal_result.to_dict())
