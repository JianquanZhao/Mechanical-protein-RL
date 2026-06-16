"""
training_logger.py

Training logging and visualization utilities for the mechanical-protein DDQN
project.

The logger is intentionally independent of PyRosetta and the DDQN
implementation. It accepts dictionaries or objects exposing ``to_dict()``.

Outputs
-------
output_dir/
├── logs/
│   ├── episodes.jsonl
│   ├── episodes.csv
│   ├── optimization.jsonl
│   └── steps.jsonl
├── plots/
│   ├── episode_reward.png
│   ├── episode_length.png
│   ├── epsilon.png
│   ├── optimization_loss.png
│   ├── td_error.png
│   ├── grad_norm.png
│   ├── q_values.png
│   ├── step_reward.png
│   ├── reward_components.png
│   └── terminal_reward.png
└── tensorboard/          # optional

Design goals
------------
- Always persist raw records before plotting.
- Use JSONL for append-only recovery after interruption.
- Export a compact episode CSV for quick inspection.
- Generate separate PNG figures rather than one crowded dashboard.
- Support TensorBoard when installed, without making it mandatory.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import warnings
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Union

import numpy as np


PathLike = Union[str, Path]
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingLoggerConfig:
    """Configuration for persistent logging and visualization."""

    output_dir: PathLike = "outputs/ddqn_base"
    rolling_window: int = 20
    plot_every_episodes: int = 10
    save_step_records: bool = True
    save_optimization_records: bool = True
    enable_tensorboard: bool = False
    tensorboard_flush_secs: int = 30
    resume: bool = True
    dpi: int = 160
    gradient_clip_threshold: Optional[float] = None

    def validate(self) -> None:
        if self.rolling_window <= 0:
            raise ValueError("rolling_window must be > 0.")
        if self.plot_every_episodes <= 0:
            raise ValueError("plot_every_episodes must be > 0.")
        if self.tensorboard_flush_secs <= 0:
            raise ValueError("tensorboard_flush_secs must be > 0.")
        if self.dpi <= 0:
            raise ValueError("dpi must be > 0.")
        if (
            self.gradient_clip_threshold is not None
            and self.gradient_clip_threshold <= 0
        ):
            raise ValueError("gradient_clip_threshold must be > 0 or None.")


class TrainingLogger:
    """
    Persist DDQN training metrics and periodically render diagnostic figures.

    Typical usage
    -------------
    logger = TrainingLogger(
        TrainingLoggerConfig(
            output_dir="outputs/ddqn_base",
            plot_every_episodes=10,
            enable_tensorboard=True,
            gradient_clip_threshold=10.0,
        )
    )

    optimization_result = agent.optimize_from_replay_buffer(replay_buffer)
    if optimization_result is not None:
        logger.log_optimization(
            optimization_result,
            global_step=total_environment_steps,
        )

    logger.log_step(
        episode=episode,
        episode_step=episode_step,
        global_step=total_environment_steps,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info=next_info,
    )

    logger.end_episode(
        episode=episode,
        total_reward=episode_reward,
        episode_steps=episode_steps,
        epsilon=agent.epsilon,
        optimization_steps=agent.optimization_steps,
        info=info,
    )
    """

    def __init__(
        self,
        config: TrainingLoggerConfig = TrainingLoggerConfig(),
        *,
        tensorboard_writer: Optional[Any] = None,
    ) -> None:
        LOGGER.info("Initializing TrainingLogger config=%s", asdict(config))
        config.validate()
        self.config = config

        self.output_dir = Path(config.output_dir)
        self.logs_dir = self.output_dir / "logs"
        self.plots_dir = self.output_dir / "plots"
        self.tensorboard_dir = self.output_dir / "tensorboard"

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.episodes_jsonl_path = self.logs_dir / "episodes.jsonl"
        self.episodes_csv_path = self.logs_dir / "episodes.csv"
        self.optimization_jsonl_path = self.logs_dir / "optimization.jsonl"
        self.steps_jsonl_path = self.logs_dir / "steps.jsonl"

        self.episode_records: List[Dict[str, Any]] = []
        self.optimization_records: List[Dict[str, Any]] = []
        self.step_records: List[Dict[str, Any]] = []

        if config.resume:
            self.episode_records = self._read_jsonl(self.episodes_jsonl_path)
            self.optimization_records = self._read_jsonl(
                self.optimization_jsonl_path
            )
            if config.save_step_records:
                self.step_records = self._read_jsonl(self.steps_jsonl_path)
            LOGGER.info(
                "TrainingLogger resumed records episodes=%s optimizations=%s steps=%s",
                len(self.episode_records),
                len(self.optimization_records),
                len(self.step_records),
            )

        self._writer = tensorboard_writer
        if self._writer is None and config.enable_tensorboard:
            self._writer = self._create_tensorboard_writer()

        self._closed = False
        LOGGER.info(
            "TrainingLogger ready output_dir=%s logs_dir=%s plots_dir=%s tensorboard=%s",
            self.output_dir,
            self.logs_dir,
            self.plots_dir,
            self._writer is not None,
        )

    # ------------------------------------------------------------------
    # Public logging API
    # ------------------------------------------------------------------

    def log_optimization(
        self,
        metrics: Any,
        *,
        global_step: Optional[int] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Record one optimizer update.

        ``metrics`` may be:
        - OptimizationResult from ``ddqn_agent.py``;
        - a dataclass;
        - a mapping;
        - any object exposing ``to_dict()``.
        """

        record = self._normalize_record(metrics)
        if global_step is not None:
            record["global_step"] = self._coerce_nonnegative_int(
                global_step,
                "global_step",
            )
        if extra:
            record.update(self._json_safe_mapping(extra))

        record = self._numeric_json_safe_record(record)
        self.optimization_records.append(record)

        if self.config.save_optimization_records:
            self._append_jsonl(self.optimization_jsonl_path, record)

        tensorboard_step = int(
            record.get(
                "optimization_step",
                record.get("global_step", len(self.optimization_records) - 1),
            )
        )
        self._tensorboard_add_scalars("optimization", record, tensorboard_step)
        LOGGER.info(
            "Logged optimization record optimization_step=%s global_step=%s loss=%s path=%s",
            record.get("optimization_step"),
            record.get("global_step"),
            record.get("loss"),
            self.optimization_jsonl_path,
        )
        return record

    def log_step(
        self,
        *,
        episode: int,
        episode_step: int,
        global_step: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Optional[Mapping[str, Any]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Record one environment transition.

        The method extracts step-reward components from
        ``info["step_reward_metrics"]["reward_components"]`` when available.
        """

        info = {} if info is None else dict(info)

        record: Dict[str, Any] = {
            "episode": self._coerce_nonnegative_int(episode, "episode"),
            "episode_step": self._coerce_nonnegative_int(
                episode_step,
                "episode_step",
            ),
            "global_step": self._coerce_nonnegative_int(
                global_step,
                "global_step",
            ),
            "reward": self._coerce_finite_float(reward, "reward"),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "done": bool(terminated or truncated),
        }

        step_reward_metrics = info.get("step_reward_metrics")
        if isinstance(step_reward_metrics, Mapping):
            numeric_metrics = self._numeric_json_safe_record(
                step_reward_metrics
            )
            for key, value in numeric_metrics.items():
                if key == "reward_components":
                    continue
                record[f"step_metric/{key}"] = value

            components = step_reward_metrics.get("reward_components")
            if isinstance(components, Mapping):
                for key, value in self._numeric_json_safe_record(
                    components
                ).items():
                    record[f"reward_component/{key}"] = value

        terminal_reward = self._extract_terminal_reward(info)
        if terminal_reward is not None:
            record["terminal_reward"] = terminal_reward

        if "sequence" in info and info["sequence"] is not None:
            record["sequence"] = str(info["sequence"])

        if extra:
            record.update(self._json_safe_mapping(extra))

        self.step_records.append(record)
        if self.config.save_step_records:
            self._append_jsonl(self.steps_jsonl_path, record)

        self._tensorboard_add_scalar(
            "reward/step_total",
            record["reward"],
            record["global_step"],
        )
        for key, value in record.items():
            if key.startswith("reward_component/") and self._is_number(value):
                self._tensorboard_add_scalar(
                    key,
                    float(value),
                    record["global_step"],
                )
        if terminal_reward is not None:
            self._tensorboard_add_scalar(
                "reward/terminal",
                terminal_reward,
                record["global_step"],
            )

        LOGGER.info(
            "Logged step record episode=%s episode_step=%s global_step=%s reward=%.6f "
            "done=%s path=%s",
            record["episode"],
            record["episode_step"],
            record["global_step"],
            record["reward"],
            record["done"],
            self.steps_jsonl_path,
        )
        return record

    def end_episode(
        self,
        *,
        episode: int,
        total_reward: float,
        episode_steps: int,
        epsilon: float,
        optimization_steps: Optional[int] = None,
        info: Optional[Mapping[str, Any]] = None,
        terminal_reward: Optional[float] = None,
        extra: Optional[Mapping[str, Any]] = None,
        generate_plots: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Record an episode summary and optionally refresh diagnostic figures."""

        info = {} if info is None else dict(info)
        episode_value = self._coerce_nonnegative_int(episode, "episode")

        if terminal_reward is None:
            terminal_reward = self._extract_terminal_reward(info)

        record: Dict[str, Any] = {
            "episode": episode_value,
            "total_reward": self._coerce_finite_float(
                total_reward,
                "total_reward",
            ),
            "episode_steps": self._coerce_nonnegative_int(
                episode_steps,
                "episode_steps",
            ),
            "epsilon": self._coerce_finite_float(epsilon, "epsilon"),
        }

        if optimization_steps is not None:
            record["optimization_steps"] = self._coerce_nonnegative_int(
                optimization_steps,
                "optimization_steps",
            )

        if terminal_reward is not None:
            record["terminal_reward"] = self._coerce_finite_float(
                terminal_reward,
                "terminal_reward",
            )

        if "sequence" in info and info["sequence"] is not None:
            record["sequence"] = str(info["sequence"])

        if extra:
            record.update(self._json_safe_mapping(extra))

        self.episode_records.append(record)
        self._append_jsonl(self.episodes_jsonl_path, record)
        self._write_episode_csv()
        LOGGER.info(
            "Logged episode record episode=%s total_reward=%.6f steps=%s path=%s csv=%s",
            record["episode"],
            record["total_reward"],
            record["episode_steps"],
            self.episodes_jsonl_path,
            self.episodes_csv_path,
        )

        self._tensorboard_add_scalar(
            "episode/total_reward",
            record["total_reward"],
            episode_value,
        )
        self._tensorboard_add_scalar(
            "episode/length",
            record["episode_steps"],
            episode_value,
        )
        self._tensorboard_add_scalar(
            "episode/epsilon",
            record["epsilon"],
            episode_value,
        )
        if terminal_reward is not None:
            self._tensorboard_add_scalar(
                "episode/terminal_reward",
                terminal_reward,
                episode_value,
            )

        if generate_plots is None:
            generate_plots = (
                len(self.episode_records) % self.config.plot_every_episodes == 0
            )

        if generate_plots:
            paths = self.generate_plots()
            LOGGER.info("Generated periodic plots count=%s paths=%s", len(paths), paths)

        return record

    # ------------------------------------------------------------------
    # Plot generation
    # ------------------------------------------------------------------

    def generate_plots(self) -> Dict[str, Path]:
        """
        Generate available diagnostic plots.

        Separate figures are deliberately used to keep each signal readable.
        """

        LOGGER.info(
            "Generating plots episodes=%s optimizations=%s steps=%s output_dir=%s",
            len(self.episode_records),
            len(self.optimization_records),
            len(self.step_records),
            self.plots_dir,
        )
        plt = self._import_pyplot()
        output_paths: Dict[str, Path] = {}

        episode_rewards = self._series(self.episode_records, "total_reward")
        episodes = self._series(self.episode_records, "episode")

        if episodes and episode_rewards:
            output_paths["episode_reward"] = self._plot_lines(
                plt,
                x=episodes,
                lines={
                    "Episode reward": episode_rewards,
                    f"Rolling mean ({self.config.rolling_window})": self.rolling_mean(
                        episode_rewards,
                        self.config.rolling_window,
                    ),
                },
                title="Episode reward",
                xlabel="Episode",
                ylabel="Reward",
                filename="episode_reward.png",
            )

        episode_lengths = self._series(self.episode_records, "episode_steps")
        if episodes and episode_lengths:
            output_paths["episode_length"] = self._plot_lines(
                plt,
                x=episodes,
                lines={"Episode length": episode_lengths},
                title="Episode length",
                xlabel="Episode",
                ylabel="Steps",
                filename="episode_length.png",
            )

        epsilons = self._series(self.episode_records, "epsilon")
        if episodes and epsilons:
            output_paths["epsilon"] = self._plot_lines(
                plt,
                x=episodes,
                lines={"Epsilon": epsilons},
                title="Exploration schedule",
                xlabel="Episode",
                ylabel="Epsilon",
                filename="epsilon.png",
            )

        optimization_steps = self._optimization_x_axis()
        losses = self._series(self.optimization_records, "loss")
        if optimization_steps and losses:
            output_paths["optimization_loss"] = self._plot_lines(
                plt,
                x=optimization_steps,
                lines={
                    "Loss": losses,
                    f"Rolling mean ({self.config.rolling_window})": self.rolling_mean(
                        losses,
                        self.config.rolling_window,
                    ),
                },
                title="Optimization loss",
                xlabel="Optimization step",
                ylabel="Huber loss",
                filename="optimization_loss.png",
            )

        td_errors = self._series(
            self.optimization_records,
            "mean_absolute_td_error",
        )
        if optimization_steps and td_errors:
            output_paths["td_error"] = self._plot_lines(
                plt,
                x=optimization_steps,
                lines={
                    "Mean absolute TD error": td_errors,
                    f"Rolling mean ({self.config.rolling_window})": self.rolling_mean(
                        td_errors,
                        self.config.rolling_window,
                    ),
                },
                title="TD error",
                xlabel="Optimization step",
                ylabel="Absolute TD error",
                filename="td_error.png",
            )

        grad_norms = self._series(self.optimization_records, "grad_norm")
        if optimization_steps and grad_norms:
            lines = {"Gradient norm": grad_norms}
            if self.config.gradient_clip_threshold is not None:
                lines["Configured clip threshold"] = [
                    float(self.config.gradient_clip_threshold)
                ] * len(grad_norms)

            output_paths["grad_norm"] = self._plot_lines(
                plt,
                x=optimization_steps,
                lines=lines,
                title="Gradient norm before clipping",
                xlabel="Optimization step",
                ylabel="Gradient norm",
                filename="grad_norm.png",
            )

        mean_q_values = self._series(
            self.optimization_records,
            "mean_q_value",
        )
        target_q_values = self._series(
            self.optimization_records,
            "mean_target_q_value",
        )
        if optimization_steps and mean_q_values and target_q_values:
            output_paths["q_values"] = self._plot_lines(
                plt,
                x=optimization_steps,
                lines={
                    "Mean predicted Q": mean_q_values,
                    "Mean target Q": target_q_values,
                },
                title="Predicted and target Q values",
                xlabel="Optimization step",
                ylabel="Q value",
                filename="q_values.png",
            )

        global_steps = self._series(self.step_records, "global_step")
        step_rewards = self._series(self.step_records, "reward")
        if global_steps and step_rewards:
            output_paths["step_reward"] = self._plot_lines(
                plt,
                x=global_steps,
                lines={
                    "Step reward": step_rewards,
                    f"Rolling mean ({self.config.rolling_window})": self.rolling_mean(
                        step_rewards,
                        self.config.rolling_window,
                    ),
                },
                title="Step reward",
                xlabel="Environment step",
                ylabel="Reward",
                filename="step_reward.png",
            )

        component_lines = self._collect_prefixed_lines(
            self.step_records,
            prefix="reward_component/",
        )
        if global_steps and component_lines:
            output_paths["reward_components"] = self._plot_lines(
                plt,
                x=global_steps,
                lines=component_lines,
                title="Step reward components",
                xlabel="Environment step",
                ylabel="Reward contribution",
                filename="reward_components.png",
            )

        episode_terminal_rewards = self._series(
            self.episode_records,
            "terminal_reward",
            require_all=False,
        )
        if episode_terminal_rewards:
            terminal_episodes = [
                float(record["episode"])
                for record in self.episode_records
                if self._is_number(record.get("terminal_reward"))
            ]
            output_paths["terminal_reward"] = self._plot_lines(
                plt,
                x=terminal_episodes,
                lines={"Terminal reward": episode_terminal_rewards},
                title="Terminal reward",
                xlabel="Episode",
                ylabel="Reward",
                filename="terminal_reward.png",
            )

        self.flush()
        LOGGER.info("Plot generation complete count=%s paths=%s", len(output_paths), output_paths)
        return output_paths

    @staticmethod
    def rolling_mean(
        values: Sequence[float],
        window: int,
    ) -> List[float]:
        """
        Return a trailing rolling mean with shorter windows at the beginning.
        """

        if window <= 0:
            raise ValueError("window must be > 0.")

        result: List[float] = []
        running_sum = 0.0
        queue: List[float] = []

        for value in values:
            float_value = float(value)
            queue.append(float_value)
            running_sum += float_value
            if len(queue) > window:
                running_sum -= queue.pop(0)
            result.append(running_sum / len(queue))

        return result

    # ------------------------------------------------------------------
    # TensorBoard and lifecycle
    # ------------------------------------------------------------------

    @property
    def tensorboard_enabled(self) -> bool:
        return self._writer is not None

    def flush(self) -> None:
        if self._writer is not None and hasattr(self._writer, "flush"):
            self._writer.flush()
            LOGGER.info("TrainingLogger tensorboard writer flushed")

    def close(self) -> None:
        if self._closed:
            return
        LOGGER.info("Closing TrainingLogger")
        self.flush()
        if self._writer is not None and hasattr(self._writer, "close"):
            self._writer.close()
            LOGGER.info("TrainingLogger tensorboard writer closed")
        self._closed = True
        LOGGER.info("TrainingLogger closed")

    def __enter__(self) -> "TrainingLogger":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_tensorboard_writer(self) -> Optional[Any]:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            warnings.warn(
                "TensorBoard logging was requested but tensorboard is not "
                "installed. Continuing with JSONL, CSV and PNG logs only.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Creating TensorBoard SummaryWriter dir=%s", self.tensorboard_dir)
        return SummaryWriter(
            log_dir=str(self.tensorboard_dir),
            flush_secs=self.config.tensorboard_flush_secs,
        )

    def _tensorboard_add_scalar(
        self,
        tag: str,
        value: Any,
        step: int,
    ) -> None:
        if self._writer is None or not self._is_number(value):
            return
        self._writer.add_scalar(tag, float(value), int(step))

    def _tensorboard_add_scalars(
        self,
        prefix: str,
        record: Mapping[str, Any],
        step: int,
    ) -> None:
        for key, value in record.items():
            if self._is_number(value) and not isinstance(value, bool):
                self._tensorboard_add_scalar(
                    f"{prefix}/{key}",
                    value,
                    step,
                )

    def _plot_lines(
        self,
        plt: Any,
        *,
        x: Sequence[float],
        lines: Mapping[str, Sequence[float]],
        title: str,
        xlabel: str,
        ylabel: str,
        filename: str,
    ) -> Path:
        fig, axis = plt.subplots(figsize=(8.0, 4.8))

        for label, values in lines.items():
            if len(values) != len(x):
                raise ValueError(
                    f"Line '{label}' has {len(values)} values but x has "
                    f"{len(x)} values."
                )
            axis.plot(x, values, label=label)

        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)

        if len(lines) > 1:
            axis.legend()

        fig.tight_layout()
        output_path = self.plots_dir / filename
        fig.savefig(output_path, dpi=self.config.dpi)
        plt.close(fig)
        return output_path

    @staticmethod
    def _import_pyplot() -> Any:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "Matplotlib is required to generate PNG plots. Install it "
                "with: python -m pip install matplotlib"
            ) from exc
        return plt

    def _optimization_x_axis(self) -> List[float]:
        values = self._series(
            self.optimization_records,
            "optimization_step",
            require_all=False,
        )
        if len(values) == len(self.optimization_records):
            return values
        return [
            float(index + 1)
            for index in range(len(self.optimization_records))
        ]

    @classmethod
    def _collect_prefixed_lines(
        cls,
        records: Sequence[Mapping[str, Any]],
        *,
        prefix: str,
    ) -> Dict[str, List[float]]:
        keys = sorted(
            {
                key
                for record in records
                for key, value in record.items()
                if key.startswith(prefix) and cls._is_number(value)
            }
        )

        lines: Dict[str, List[float]] = {}
        for key in keys:
            values = []
            for record in records:
                value = record.get(key, 0.0)
                values.append(float(value) if cls._is_number(value) else 0.0)
            lines[key.removeprefix(prefix)] = values

        return lines

    @classmethod
    def _series(
        cls,
        records: Sequence[Mapping[str, Any]],
        key: str,
        *,
        require_all: bool = True,
    ) -> List[float]:
        values: List[float] = []

        for record in records:
            value = record.get(key)
            if cls._is_number(value):
                values.append(float(value))
            elif require_all:
                return []

        return values

    @classmethod
    def _extract_terminal_reward(
        cls,
        info: Mapping[str, Any],
    ) -> Optional[float]:
        direct_value = info.get("terminal_reward")
        if cls._is_number(direct_value):
            return float(direct_value)

        metrics = info.get("terminal_reward_metrics")
        if isinstance(metrics, Mapping):
            reward_value = metrics.get("reward")
            if cls._is_number(reward_value):
                return float(reward_value)

        return None

    @staticmethod
    def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            file.write("\n")
        LOGGER.info("Appended JSONL path=%s keys=%s", path, sorted(record.keys()))

    @staticmethod
    def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            LOGGER.info("JSONL path does not exist; starting empty path=%s", path)
            return []

        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    value = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL record in {path} at line "
                        f"{line_number}."
                    ) from exc
                if not isinstance(value, dict):
                    raise ValueError(
                        f"Expected JSON object in {path} at line "
                        f"{line_number}."
                    )
                records.append(value)
        LOGGER.info("Read JSONL path=%s records=%s", path, len(records))
        return records

    def _write_episode_csv(self) -> None:
        if not self.episode_records:
            return

        preferred_fields = [
            "episode",
            "total_reward",
            "episode_steps",
            "epsilon",
            "optimization_steps",
            "terminal_reward",
            "sequence",
        ]

        observed_fields = {
            key
            for record in self.episode_records
            for key in record
        }

        fields = [
            field
            for field in preferred_fields
            if field in observed_fields
        ]
        fields.extend(sorted(observed_fields - set(fields)))

        with self.episodes_csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fields,
                extrasaction="ignore",
            )
            writer.writeheader()
            for record in self.episode_records:
                writer.writerow(record)
        LOGGER.info(
            "Wrote episode CSV path=%s rows=%s",
            self.episodes_csv_path,
            len(self.episode_records),
        )

    @classmethod
    def _normalize_record(cls, value: Any) -> Dict[str, Any]:
        if isinstance(value, Mapping):
            record = dict(value)
        elif hasattr(value, "to_dict") and callable(value.to_dict):
            record = dict(value.to_dict())
        elif is_dataclass(value):
            record = asdict(value)
        else:
            raise TypeError(
                "Expected a mapping, dataclass or object exposing to_dict()."
            )
        return cls._json_safe_mapping(record)

    @classmethod
    def _json_safe_mapping(
        cls,
        mapping: Mapping[str, Any],
    ) -> Dict[str, Any]:
        return {
            str(key): cls._json_safe_value(value)
            for key, value in mapping.items()
        }

    @classmethod
    def _numeric_json_safe_record(
        cls,
        mapping: Mapping[str, Any],
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in mapping.items():
            if isinstance(value, Mapping):
                result[str(key)] = cls._numeric_json_safe_record(value)
            elif cls._is_number(value) or isinstance(value, bool):
                result[str(key)] = cls._json_safe_value(value)
        return result

    @classmethod
    def _json_safe_value(cls, value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return cls._json_safe_mapping(value)
        if isinstance(value, (list, tuple)):
            return [cls._json_safe_value(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    @staticmethod
    def _is_number(value: Any) -> bool:
        if isinstance(value, bool):
            return False
        if not isinstance(value, (int, float, np.number)):
            return False
        return bool(np.isfinite(float(value)))

    @staticmethod
    def _coerce_nonnegative_int(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer.")
        value = int(value)
        if value < 0:
            raise ValueError(f"{name} must be >= 0.")
        return value

    @staticmethod
    def _coerce_finite_float(value: Any, name: str) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a finite scalar.") from exc
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
        return value
