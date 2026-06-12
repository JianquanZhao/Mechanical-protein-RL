"""
smoke_test_terminal_reward.py

This test does not require PyRosetta. It verifies that the terminal predictor
wrapper supports sklearn-like predictors and delta rewards.
"""

import numpy as np

from reward_calculators import (
    TerminalRewardCalculator,
    TerminalRewardWeights,
)


class DummyPredictor:
    def predict(self, x: np.ndarray) -> np.ndarray:
        # max_stress = 2*x0 + x1, toughness = x0 - 0.5*x1
        return np.column_stack(
            [
                2.0 * x[:, 0] + x[:, 1],
                x[:, 0] - 0.5 * x[:, 1],
            ]
        )


calculator = TerminalRewardCalculator(
    DummyPredictor(),
    weights=TerminalRewardWeights(max_stress=1.0, toughness=2.0),
    target_mean={"max_stress": 0.0, "toughness": 0.0},
    target_std={"max_stress": 1.0, "toughness": 1.0},
    baseline_features=np.asarray([1.0, 2.0]),
    reward_mode="delta",
)

result = calculator.evaluate_features(np.asarray([2.0, 2.0]))

# Baseline: stress=4, toughness=0
# Current:  stress=6, toughness=1
# Reward:   1*(6-4) + 2*(1-0) = 4
assert np.isclose(result.reward, 4.0), result
assert result.objective_vector == (6.0, 1.0), result

print("TerminalRewardCalculator smoke test passed.")
print(result.to_dict())
