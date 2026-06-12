"""
Optional real-PyRosetta environment smoke test.

This is skipped unless both PyRosetta and model/reward_module/wild_type.pdb are
available. Copy your existing wild_type.pdb into that path and run:

    python -m pytest -q tests/test_environment_pyrosetta_smoke.py
"""

from pathlib import Path
import importlib.util

import numpy as np
import pytest

from model.environment_module.environment import MechanicalProteinEnv


@pytest.mark.integration
def test_real_pyrosetta_environment_step() -> None:
    if importlib.util.find_spec("pyrosetta") is None:
        pytest.skip("PyRosetta is not installed in this environment.")

    pdb_path = Path("model/reward_module/wild_type.pdb")
    if not pdb_path.is_file():
        pytest.skip("Copy wild_type.pdb to model/reward_module before this smoke test.")

    env = MechanicalProteinEnv(
        initial_pdb_path=str(pdb_path),
        max_steps=1,
        perform_repack=True,
        perform_minimize=True,
        minimize_backbone=False,
    )

    observation, info = env.reset(seed=7)
    assert observation.shape == env.observation_space.shape
    assert env.action_space.n == env.n_mutable_positions * 20

    action = env.sample_valid_action()
    _, reward, terminated, truncated, info = env.step(action)

    assert np.isfinite(reward)
    assert terminated is False
    assert truncated is True
    assert info["accepted"] is True
    assert info["step_reward_metrics"] is not None
