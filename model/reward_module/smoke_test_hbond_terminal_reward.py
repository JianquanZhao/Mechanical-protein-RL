"""
Smoke test for the hbond-topology terminal reward.

This test does not require PyRosetta. It trains or loads the random-split
random forest artifact, extracts hbond topology features from a real PDB, and
checks both single-structure and dual-structure reward paths.
"""

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reward_calculators import HbondTopologyTerminalRewardCalculator
from terminal_reward import DEFAULT_ARTIFACT_PATH


PDB_PATH = Path("/home/jianquanzhao/data/tsinghua/mpprd/pdbs/pdbs/2w1nA02.pdb")


calculator = HbondTopologyTerminalRewardCalculator(artifact_path=DEFAULT_ARTIFACT_PATH)
assert DEFAULT_ARTIFACT_PATH.exists(), DEFAULT_ARTIFACT_PATH

single = calculator.evaluate_pdb(PDB_PATH)
assert np.isfinite(single.reward), single
assert np.isfinite(single.strength), single
assert np.isfinite(single.toughness), single
assert len(single.features) == 7, single
assert single.feature_diagnostics["hbond_count"] > 0, single

dual = calculator.evaluate_dual_structures(
    relaxed_pdb_path=PDB_PATH,
    predicted_pdb_path=PDB_PATH,
    relaxed_quality=1.0,
)
assert np.isfinite(dual.reward), dual
assert np.isclose(dual.disagreement_penalty, 0.0), dual

print("HbondTopologyTerminalRewardCalculator smoke test passed.")
print(single.to_dict())
print(dual.to_dict())
