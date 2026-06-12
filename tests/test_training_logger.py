from dataclasses import dataclass
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from model.logging_module.training_logger import (
    TrainingLogger,
    TrainingLoggerConfig,
)


class DummyWriter:
    def __init__(self):
        self.scalars = []
        self.flush_calls = 0
        self.close_calls = 0

    def add_scalar(self, tag, value, step):
        self.scalars.append((str(tag), float(value), int(step)))

    def flush(self):
        self.flush_calls += 1

    def close(self):
        self.close_calls += 1


@dataclass(frozen=True)
class FakeOptimizationResult:
    loss: float = 1.25
    mean_q_value: float = 0.4
    mean_target_q_value: float = 0.7
    mean_absolute_td_error: float = 0.3
    grad_norm: float = 2.0
    effective_batch_size: int = 64
    micro_batches: int = 4
    optimization_step: int = 3
    target_synced: bool = False
    epsilon: float = 0.8

    def to_dict(self):
        return {
            "loss": self.loss,
            "mean_q_value": self.mean_q_value,
            "mean_target_q_value": self.mean_target_q_value,
            "mean_absolute_td_error": self.mean_absolute_td_error,
            "grad_norm": self.grad_norm,
            "effective_batch_size": self.effective_batch_size,
            "micro_batches": self.micro_batches,
            "optimization_step": self.optimization_step,
            "target_synced": self.target_synced,
            "epsilon": self.epsilon,
        }


def make_logger(tmp_path: Path, **kwargs) -> TrainingLogger:
    defaults = dict(
        output_dir=tmp_path,
        rolling_window=3,
        plot_every_episodes=2,
        save_step_records=True,
        save_optimization_records=True,
        enable_tensorboard=False,
        resume=True,
        gradient_clip_threshold=10.0,
    )
    defaults.update(kwargs)
    return TrainingLogger(TrainingLoggerConfig(**defaults))


def read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_rolling_mean() -> None:
    actual = TrainingLogger.rolling_mean([1.0, 2.0, 6.0, 10.0], window=3)
    assert actual == pytest.approx([1.0, 1.5, 3.0, 6.0])


def test_rolling_mean_rejects_nonpositive_window() -> None:
    with pytest.raises(ValueError, match="window"):
        TrainingLogger.rolling_mean([1.0], window=0)


def test_log_optimization_persists_jsonl_and_tensorboard(tmp_path: Path) -> None:
    writer = DummyWriter()
    logger = TrainingLogger(
        TrainingLoggerConfig(output_dir=tmp_path),
        tensorboard_writer=writer,
    )

    record = logger.log_optimization(
        FakeOptimizationResult(),
        global_step=12,
    )

    assert record["loss"] == pytest.approx(1.25)
    assert record["global_step"] == 12

    records = read_jsonl(tmp_path / "logs" / "optimization.jsonl")
    assert len(records) == 1
    assert records[0]["mean_absolute_td_error"] == pytest.approx(0.3)

    tags = {tag for tag, _, _ in writer.scalars}
    assert "optimization/loss" in tags
    assert "optimization/grad_norm" in tags


def test_log_step_extracts_reward_components_and_terminal_reward(tmp_path: Path) -> None:
    logger = make_logger(tmp_path)

    record = logger.log_step(
        episode=1,
        episode_step=2,
        global_step=17,
        reward=3.5,
        terminated=False,
        truncated=True,
        info={
            "sequence": "ACDE",
            "step_reward_metrics": {
                "collision_score": 4.0,
                "backbone_hbond_delta": 2,
                "reward_components": {
                    "collision": -1.5,
                    "backbone_hbond": 2.0,
                    "local_rmsd": -0.25,
                },
            },
            "terminal_reward_metrics": {
                "reward": 7.0,
            },
        },
    )

    assert record["done"] is True
    assert record["sequence"] == "ACDE"
    assert record["step_metric/collision_score"] == pytest.approx(4.0)
    assert record["step_metric/backbone_hbond_delta"] == 2
    assert record["reward_component/collision"] == pytest.approx(-1.5)
    assert record["reward_component/backbone_hbond"] == pytest.approx(2.0)
    assert record["terminal_reward"] == pytest.approx(7.0)

    persisted = read_jsonl(tmp_path / "logs" / "steps.jsonl")
    assert persisted == [record]


def test_end_episode_writes_jsonl_and_csv(tmp_path: Path) -> None:
    logger = make_logger(tmp_path)

    record = logger.end_episode(
        episode=0,
        total_reward=8.5,
        episode_steps=5,
        epsilon=0.9,
        optimization_steps=3,
        info={
            "sequence": "AAAA",
            "terminal_reward": 2.0,
        },
        generate_plots=False,
    )

    assert record["total_reward"] == pytest.approx(8.5)
    assert record["terminal_reward"] == pytest.approx(2.0)

    jsonl = read_jsonl(tmp_path / "logs" / "episodes.jsonl")
    assert jsonl == [record]

    with (tmp_path / "logs" / "episodes.csv").open(
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["episode"] == "0"
    assert rows[0]["sequence"] == "AAAA"


def test_periodic_plot_generation_is_triggered(tmp_path: Path, monkeypatch) -> None:
    logger = make_logger(tmp_path, plot_every_episodes=2)
    calls = []

    monkeypatch.setattr(
        logger,
        "generate_plots",
        lambda: calls.append("generated") or {},
    )

    logger.end_episode(
        episode=0,
        total_reward=1.0,
        episode_steps=2,
        epsilon=1.0,
    )
    assert calls == []

    logger.end_episode(
        episode=1,
        total_reward=2.0,
        episode_steps=2,
        epsilon=0.9,
    )
    assert calls == ["generated"]


def test_resume_loads_existing_records(tmp_path: Path) -> None:
    first = make_logger(tmp_path)
    first.log_optimization(FakeOptimizationResult())
    first.log_step(
        episode=0,
        episode_step=0,
        global_step=0,
        reward=1.0,
        terminated=False,
        truncated=False,
    )
    first.end_episode(
        episode=0,
        total_reward=1.0,
        episode_steps=1,
        epsilon=1.0,
        generate_plots=False,
    )

    resumed = make_logger(tmp_path, resume=True)

    assert len(resumed.optimization_records) == 1
    assert len(resumed.step_records) == 1
    assert len(resumed.episode_records) == 1


def test_resume_false_starts_empty_in_memory(tmp_path: Path) -> None:
    first = make_logger(tmp_path)
    first.end_episode(
        episode=0,
        total_reward=1.0,
        episode_steps=1,
        epsilon=1.0,
        generate_plots=False,
    )

    second = make_logger(tmp_path, resume=False)
    assert second.episode_records == []


def test_generate_plots_creates_diagnostic_pngs(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    logger = make_logger(tmp_path, rolling_window=2)

    for index in range(4):
        logger.log_step(
            episode=index // 2,
            episode_step=index % 2,
            global_step=index,
            reward=float(index),
            terminated=False,
            truncated=(index % 2 == 1),
            info={
                "step_reward_metrics": {
                    "reward_components": {
                        "collision": -0.1 * index,
                        "backbone_hbond": 0.2 * index,
                    }
                }
            },
        )

        logger.log_optimization(
            {
                "optimization_step": index + 1,
                "loss": 2.0 / (index + 1),
                "mean_absolute_td_error": 1.0 / (index + 1),
                "grad_norm": float(index + 1),
                "mean_q_value": 0.25 * index,
                "mean_target_q_value": 0.3 * index,
            }
        )

    for episode in range(2):
        logger.end_episode(
            episode=episode,
            total_reward=float(episode + 1),
            episode_steps=2,
            epsilon=1.0 - 0.1 * episode,
            terminal_reward=0.5 * episode,
            generate_plots=False,
        )

    paths = logger.generate_plots()

    expected = {
        "episode_reward",
        "episode_length",
        "epsilon",
        "optimization_loss",
        "td_error",
        "grad_norm",
        "q_values",
        "step_reward",
        "reward_components",
        "terminal_reward",
    }

    assert expected <= set(paths)
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0


def test_close_flushes_and_closes_tensorboard_writer(tmp_path: Path) -> None:
    writer = DummyWriter()
    logger = TrainingLogger(
        TrainingLoggerConfig(output_dir=tmp_path),
        tensorboard_writer=writer,
    )

    logger.close()
    logger.close()

    assert writer.flush_calls == 1
    assert writer.close_calls == 1


def test_context_manager_closes_writer(tmp_path: Path) -> None:
    writer = DummyWriter()

    with TrainingLogger(
        TrainingLoggerConfig(output_dir=tmp_path),
        tensorboard_writer=writer,
    ) as logger:
        logger.log_optimization({"optimization_step": 1, "loss": 1.0})

    assert writer.flush_calls == 1
    assert writer.close_calls == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rolling_window": 0},
        {"plot_every_episodes": 0},
        {"tensorboard_flush_secs": 0},
        {"dpi": 0},
        {"gradient_clip_threshold": 0.0},
    ],
)
def test_config_validation(kwargs) -> None:
    config = TrainingLoggerConfig(**kwargs)
    with pytest.raises(ValueError):
        config.validate()


def test_invalid_jsonl_is_reported_on_resume(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / "episodes.jsonl").write_text(
        "{invalid json}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid JSONL"):
        make_logger(tmp_path, resume=True)


def test_direct_terminal_reward_has_priority(tmp_path: Path) -> None:
    logger = make_logger(tmp_path)

    record = logger.log_step(
        episode=0,
        episode_step=0,
        global_step=0,
        reward=1.0,
        terminated=True,
        truncated=False,
        info={
            "terminal_reward": 9.0,
            "terminal_reward_metrics": {"reward": 3.0},
        },
    )

    assert record["terminal_reward"] == pytest.approx(9.0)
