"""
Minimal usage example for MechanicalProteinEnv.

Run from the project root:
    python -m model.environment_module.example_environment_usage

The base example uses step rewards only. Attach a trained
TerminalRewardCalculator after your mechanical-property predictor is ready.
"""

from pathlib import Path

from model.environment_module.environment import MechanicalProteinEnv
from model.reward_module.reward_calculators import (
    StepRewardScales,
    StepRewardWeights,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDB_PATH = PROJECT_ROOT / "model" / "reward_module" / "wild_type.pdb"

env = MechanicalProteinEnv(
    initial_pdb_path=str(PDB_PATH),
    max_steps=5,
    # Replace with your mechanical-lock region to reduce the search space.
    mutable_positions=None,
    step_reward_kwargs={
        "weights": StepRewardWeights(
            collision=1.0,
            backbone_hbond=1.0,
            sidechain_hbond=0.5,
            local_rmsd=1.0,
        ),
        "scales": StepRewardScales(
            collision=10.0,
            backbone_hbond=1.0,
            sidechain_hbond=1.0,
            local_rmsd=0.5,
        ),
        "hbond_energy_cutoff": 0.0,
        "neighborhood_radius": 8.0,
        "rmsd_penalty_mode": "previous",
        "collision_penalty_mode": "delta",
    },
    local_repack_radius=8.0,
    perform_repack=True,
    perform_minimize=True,
    minimize_backbone=False,
    prevent_revisit_positions=False,
    raise_on_update_error=True,
)

observation, info = env.reset(seed=7)
print("Observation shape:", observation.shape)
print("Action-space size:", env.action_space.n)
print("Initial sequence:", info["sequence"])

for _ in range(env.max_steps):
    action = env.sample_valid_action()
    observation, reward, terminated, truncated, info = env.step(action)

    print("\nDecoded action:", info["decoded_action"])
    print("Reward:", reward)
    print("Sequence:", info["sequence"])

    if terminated or truncated:
        break

env.save_current_pose("outputs/final_candidate.pdb")
env.save_history("outputs/episode_history.json")
