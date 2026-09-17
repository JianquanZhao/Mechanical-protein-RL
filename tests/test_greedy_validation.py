import pytest

from model.evaluation_module import summarize_greedy_validation


def test_greedy_validation_summary_reports_paired_improvements() -> None:
    records = [
        {"terminal_reward": 1.0, "strength_delta": 2.0, "toughness_delta": -1.0},
        {"terminal_reward": -1.0, "strength_delta": 0.0, "toughness_delta": 3.0},
        {"terminal_reward": 2.0, "strength_delta": 4.0, "toughness_delta": 1.0},
    ]

    summary = summarize_greedy_validation(
        records,
        bootstrap_samples=100,
        seed=7,
    )

    assert summary["validation_count"] == 3
    assert summary["strength_mean"] == pytest.approx(2.0)
    assert summary["strength_positive_fraction"] == pytest.approx(2 / 3)
    assert summary["toughness_median"] == pytest.approx(1.0)
    assert summary["terminal_reward_top_10pct_mean"] == pytest.approx(2.0)
