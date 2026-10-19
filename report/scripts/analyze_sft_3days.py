"""Compare the SFT-initialized DDQN run with its matched random-init control."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(
    "/mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train"
)
CONTROL_NAME = (
    "async_a24_esm8_f8_g1_prioritized_possample0.25_poslcb-0.25_"
    "posreserve0.10_n3_stepx0.025_terminalx8.0_20260917_100006"
)
SFT_NAME = (
    "async_a24_esm8_f8_g1_prioritized_possample0.25_poslcb-0.25_"
    "posreserve0.10_n3_stepx0.025_terminalx8.0_sft_20261016_121023"
)
OUTPUT_DIR = Path("report/figures/sft-3days-result-analysis")
SUMMARY_PATH = Path("report/sft-3days-result-analysis-summary.json")
COLORS = {"Random init": "#247BA0", "SFT init": "#E76F51"}
SNAPSHOT_EPISODES = 47_000
SNAPSHOT_VALIDATION_RUNS = 197
SNAPSHOT_OPTIMIZATION_STEPS = 140_730


def load_jsonl(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if columns is not None:
                row = {key: row.get(key, np.nan) for key in columns}
            rows.append(row)
    return pd.DataFrame(rows)


def moving_block_ci(
    values: np.ndarray,
    *,
    block_size: int = 10,
    samples: int = 5_000,
    seed: int = 20261019,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    if values.size <= block_size:
        means = np.random.default_rng(seed).choice(
            values, size=(samples, values.size), replace=True
        ).mean(axis=1)
        return tuple(np.quantile(means, [0.025, 0.975]))
    starts = np.arange(values.size - block_size + 1)
    block_count = int(np.ceil(values.size / block_size))
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for index in range(samples):
        sampled_starts = rng.choice(starts, size=block_count, replace=True)
        sample = np.concatenate(
            [values[start : start + block_size] for start in sampled_starts]
        )[: values.size]
        means[index] = sample.mean()
    return tuple(np.quantile(means, [0.025, 0.975]))


def bootstrap_ci(
    values: np.ndarray,
    *,
    samples: int = 10_000,
    seed: int = 20261019,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(samples, values.size), replace=True).mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def rolling(series: pd.Series, window: int = 10) -> pd.Series:
    return series.rolling(window, min_periods=max(2, window // 2), center=True).mean()


def binned_episode_metrics(frame: pd.DataFrame, bin_size: int = 1_000) -> pd.DataFrame:
    data = frame.copy()
    data["episode_bin"] = ((data["completion_order"] - 1) // bin_size) * bin_size
    data["terminal_positive"] = (data["terminal_reward"] > 0).astype(float)
    data["pareto_positive"] = (
        (data["mechanical_delta_strength_z"] > 0)
        & (data["mechanical_delta_toughness_z"] > 0)
    ).astype(float)
    data["step_shaping_sum"] = data["total_reward"] - data["terminal_reward"]
    result = (
        data.groupby("episode_bin", as_index=False)
        .agg(
            total_reward=("total_reward", "mean"),
            terminal_reward=("terminal_reward", "mean"),
            terminal_positive=("terminal_positive", "mean"),
            pareto_positive=("pareto_positive", "mean"),
            strength=("mechanical_delta_strength_z", "mean"),
            toughness=("mechanical_delta_toughness_z", "mean"),
            step_shaping_sum=("step_shaping_sum", "mean"),
        )
    )
    result["episode"] = result["episode_bin"] + bin_size / 2
    return result


def window_summary(frame: pd.DataFrame, size: int = 5_000) -> dict[str, dict[str, float]]:
    def summarize(window: pd.DataFrame) -> dict[str, float]:
        return {
            "total_reward_mean": float(window["total_reward"].mean()),
            "total_reward_median": float(window["total_reward"].median()),
            "terminal_reward_mean": float(window["terminal_reward"].mean()),
            "terminal_reward_median": float(window["terminal_reward"].median()),
            "total_positive_fraction": float((window["total_reward"] > 0).mean()),
            "terminal_positive_fraction": float(
                (window["terminal_reward"] > 0).mean()
            ),
            "strength_mean": float(window["mechanical_delta_strength_z"].mean()),
            "toughness_mean": float(window["mechanical_delta_toughness_z"].mean()),
            "pareto_positive_fraction": float(
                (
                    (window["mechanical_delta_strength_z"] > 0)
                    & (window["mechanical_delta_toughness_z"] > 0)
                ).mean()
            ),
            "step_shaping_sum_mean": float(
                (window["total_reward"] - window["terminal_reward"]).mean()
            ),
        }

    return {
        "first_5000": summarize(frame.head(size)),
        "last_5000": summarize(frame.tail(size)),
        "all_matched": summarize(frame),
    }


def validation_summary(
    control: pd.DataFrame, sft: pd.DataFrame
) -> dict[str, object]:
    common = min(len(control), len(sft))
    control = control.iloc[:common].reset_index(drop=True)
    sft = sft.iloc[:common].reset_index(drop=True)
    metrics = [
        "terminal_reward_mean",
        "terminal_reward_median",
        "terminal_reward_positive_fraction",
        "strength_mean",
        "strength_positive_fraction",
        "toughness_mean",
        "toughness_positive_fraction",
    ]
    result: dict[str, object] = {
        "common_validation_runs": common,
        "max_trigger_episode": int(sft["trigger_episode"].max()),
        "random_initial": {
            metric: float(control.loc[0, metric]) for metric in metrics
        },
        "sft_initial": {metric: float(sft.loc[0, metric]) for metric in metrics},
    }
    result["initial_sft_minus_random"] = {
        metric: float(sft.loc[0, metric] - control.loc[0, metric])
        for metric in metrics
    }
    for label, frame in (("random", control), ("sft", sft)):
        result[f"{label}_first20"] = {
            metric: float(frame.head(20)[metric].mean()) for metric in metrics
        }
        result[f"{label}_last20"] = {
            metric: float(frame.tail(20)[metric].mean()) for metric in metrics
        }
        result[f"{label}_positive_mean_runs"] = int(
            (frame["terminal_reward_mean"] > 0).sum()
        )
        result[f"{label}_positive_lower_ci_runs"] = int(
            (frame["terminal_reward_mean_ci_low"] > 0).sum()
        )

    paired = {}
    for metric in ("terminal_reward_mean", "strength_mean", "toughness_mean"):
        difference = (sft[metric] - control[metric]).to_numpy()
        ci_low, ci_high = moving_block_ci(difference)
        paired[metric] = {
            "mean_difference": float(difference.mean()),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "sft_better_fraction": float((difference > 0).mean()),
            "last20_difference": float(difference[-20:].mean()),
        }
    result["paired_sft_minus_random"] = paired

    difference = sft["terminal_reward_mean"] - control["terminal_reward_mean"]
    smooth = difference.rolling(10, min_periods=10).mean()
    non_positive = np.flatnonzero(smooth.to_numpy() <= 0)
    result["terminal_advantage_first_rolling10_nonpositive_episode"] = (
        int(sft.loc[non_positive[0], "trigger_episode"])
        if non_positive.size
        else None
    )
    return result


def load_optimization(path: Path, maximum_step: int | None = None) -> pd.DataFrame:
    columns = [
        "optimization_step",
        "global_step",
        "loss",
        "grad_norm",
        "mean_absolute_td_error",
        "mean_q_value",
        "mean_target_q_value",
        "per/replay/positive_terminal_fraction",
        "per/sampled_batch/positive_terminal_fraction",
        "per/priority_top_1pct/positive_terminal_fraction",
        "per/sampled_batch/terminal_reward_mean",
    ]
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            optimization_step = int(row["optimization_step"])
            if maximum_step is not None and optimization_step > maximum_step:
                break
            rows.append({key: row.get(key, np.nan) for key in columns})
    return pd.DataFrame(rows)


def optimization_summary(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    metrics = ["loss", "grad_norm", "mean_absolute_td_error", "mean_q_value"]

    def summarize(window: pd.DataFrame) -> dict[str, float]:
        result = {}
        for metric in metrics:
            values = window[metric].dropna()
            result[f"{metric}_median"] = float(values.median())
            result[f"{metric}_p99"] = float(values.quantile(0.99))
        return result

    diagnostic = frame.dropna(
        subset=["per/sampled_batch/positive_terminal_fraction"]
    ).tail(100)
    return {
        "first_1000": summarize(frame.head(1_000)),
        "last_10000": summarize(frame.tail(10_000)),
        "per_last100": {
            column: float(diagnostic[column].mean())
            for column in [
                "per/replay/positive_terminal_fraction",
                "per/sampled_batch/positive_terminal_fraction",
                "per/priority_top_1pct/positive_terminal_fraction",
                "per/sampled_batch/terminal_reward_mean",
            ]
        },
    }


def plot_validation(
    control: pd.DataFrame, sft: pd.DataFrame, output_path: Path
) -> None:
    panels = [
        ("terminal_reward_mean", "Terminal reward mean", 0.0),
        ("terminal_reward_positive_fraction", "Terminal positive fraction", 0.5),
        ("strength_mean", "Strength delta (z)", 0.0),
        ("toughness_mean", "Toughness delta (z)", 0.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, (metric, title, reference) in zip(axes.flat, panels):
        for label, frame in (("Random init", control), ("SFT init", sft)):
            axis.plot(
                frame["trigger_episode"],
                frame[metric],
                color=COLORS[label],
                alpha=0.16,
                linewidth=0.8,
            )
            axis.plot(
                frame["trigger_episode"],
                rolling(frame[metric], 10),
                color=COLORS[label],
                linewidth=2.2,
                label=label,
            )
        axis.axhline(reference, color="#555555", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[1, 0].set_xlabel("Completed training episodes")
    axes[1, 1].set_xlabel("Completed training episodes")
    axes[0, 0].set_ylabel("Reward")
    axes[1, 0].set_ylabel("Delta z-score")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Fixed greedy validation: matched first 47,040 episodes", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_advantage(
    control: pd.DataFrame, sft: pd.DataFrame, output_path: Path
) -> None:
    metrics = [
        ("terminal_reward_mean", "Terminal reward"),
        ("strength_mean", "Strength delta"),
        ("toughness_mean", "Toughness delta"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    for metric, label in metrics:
        difference = sft[metric] - control[metric]
        axes[0].plot(
            sft["trigger_episode"],
            rolling(difference, 10),
            linewidth=2,
            label=label,
        )
    axes[0].axhline(0, color="#333333", linestyle="--", linewidth=1)
    axes[0].set_title("SFT minus random initialization (10-validation moving mean)")
    axes[0].set_ylabel("Paired difference")
    axes[0].legend(frameon=False, ncol=3)
    axes[0].grid(alpha=0.2)

    categories = ["Terminal reward", "Strength delta", "Toughness delta"]
    x = np.arange(len(categories))
    width = 0.36
    early = [
        float(sft.head(20)[metric].mean() - control.head(20)[metric].mean())
        for metric, _ in metrics
    ]
    late = [
        float(sft.tail(20)[metric].mean() - control.tail(20)[metric].mean())
        for metric, _ in metrics
    ]
    axes[1].bar(x - width / 2, early, width, label="First 20 validations", color="#2A9D8F")
    axes[1].bar(x + width / 2, late, width, label="Last 20 validations", color="#F4A261")
    axes[1].axhline(0, color="#333333", linewidth=1)
    axes[1].set_xticks(x, categories)
    axes[1].set_ylabel("SFT minus random")
    axes[1].set_title("Initialization advantage reverses during online DDQN")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_behavior(
    control: pd.DataFrame, sft: pd.DataFrame, output_path: Path
) -> None:
    control_bins = binned_episode_metrics(control)
    sft_bins = binned_episode_metrics(sft)
    panels = [
        ("total_reward", "Exploratory total reward"),
        ("terminal_reward", "Exploratory terminal reward"),
        ("terminal_positive", "Terminal-positive fraction"),
        ("pareto_positive", "Strength+toughness Pareto-positive fraction"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, (metric, title) in zip(axes.flat, panels):
        for label, frame in (("Random init", control_bins), ("SFT init", sft_bins)):
            axis.plot(
                frame["episode"],
                frame[metric],
                color=COLORS[label],
                linewidth=2,
                label=label,
            )
        axis.axhline(0 if "reward" in metric else 0.5, color="#555555", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[1, 0].set_xlabel("Episode completion order")
    axes[1, 1].set_xlabel("Episode completion order")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Training actors: 1,000-episode bins at matched horizon", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_optimization(
    control: pd.DataFrame, sft: pd.DataFrame, output_path: Path
) -> None:
    panels = [
        ("loss", "Huber loss", True),
        ("mean_absolute_td_error", "Mean absolute TD error", True),
        ("grad_norm", "Gradient norm", True),
        ("mean_q_value", "Mean Q value", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, (metric, title, log_scale) in zip(axes.flat, panels):
        for label, frame in (("Random init", control), ("SFT init", sft)):
            binned = frame.assign(
                optimization_bin=(frame["optimization_step"] // 1_000) * 1_000
            ).groupby("optimization_bin", as_index=False)[metric].median()
            axis.plot(
                binned["optimization_bin"],
                binned[metric],
                color=COLORS[label],
                linewidth=2,
                label=label,
            )
        if log_scale:
            axis.set_yscale("log")
        else:
            axis.axhline(0, color="#555555", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[1, 0].set_xlabel("Optimizer steps")
    axes[1, 1].set_xlabel("Optimizer steps")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Optimization dynamics at matched update budget", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_per_protein(
    control: pd.DataFrame, sft: pd.DataFrame, output_path: Path
) -> dict[str, object]:
    initial_control = control[control["validation_run"] == 0].set_index("pdb_path")
    initial_sft = sft[sft["validation_run"] == 0].set_index("pdb_path")
    last_runs = sorted(sft["validation_run"].unique())[-20:]
    late_control = (
        control[control["validation_run"].isin(last_runs)]
        .groupby("pdb_path")["terminal_reward"]
        .mean()
    )
    late_sft = (
        sft[sft["validation_run"].isin(last_runs)]
        .groupby("pdb_path")["terminal_reward"]
        .mean()
    )
    initial_difference = (
        initial_sft["terminal_reward"] - initial_control["terminal_reward"]
    ).sort_values()
    late_difference = (late_sft - late_control).reindex(initial_difference.index)

    labels = [Path(path).stem for path in initial_difference.index]
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    for axis, values, title in (
        (axes[0], initial_difference, "Before DDQN updates (run 0)"),
        (axes[1], late_difference, "Mean of last 20 validations"),
    ):
        colors = ["#2A9D8F" if value > 0 else "#E76F51" for value in values]
        axis.barh(y, values, color=colors)
        axis.axvline(0, color="#333333", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("SFT minus random terminal reward")
        axis.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(y, labels)
    fig.suptitle("Per-protein fixed greedy effect", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    initial_ci = bootstrap_ci(initial_difference.to_numpy())
    late_ci = bootstrap_ci(late_difference.to_numpy())
    return {
        "initial_mean_difference": float(initial_difference.mean()),
        "initial_ci_low": float(initial_ci[0]),
        "initial_ci_high": float(initial_ci[1]),
        "initial_improved_proteins": int((initial_difference > 0).sum()),
        "late20_mean_difference": float(late_difference.mean()),
        "late20_ci_low": float(late_ci[0]),
        "late20_ci_high": float(late_ci[1]),
        "late20_improved_proteins": int((late_difference > 0).sum()),
        "protein_count": int(len(initial_difference)),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    control_dir = BASE_DIR / CONTROL_NAME
    sft_dir = BASE_DIR / SFT_NAME

    validation_control = load_jsonl(control_dir / "logs/validation_summary.jsonl")
    validation_sft = load_jsonl(sft_dir / "logs/validation_summary.jsonl")
    common_validations = min(
        len(validation_control), len(validation_sft), SNAPSHOT_VALIDATION_RUNS
    )
    validation_control = validation_control.iloc[:common_validations].copy()
    validation_sft = validation_sft.iloc[:common_validations].copy()

    episode_columns = [
        "completion_order",
        "epoch",
        "total_reward",
        "terminal_reward",
        "mechanical_delta_strength_z",
        "mechanical_delta_toughness_z",
        "source_pdb",
    ]
    episodes_control = pd.read_csv(
        control_dir / "logs/episodes.csv", usecols=episode_columns
    ).sort_values("completion_order")
    episodes_sft = pd.read_csv(
        sft_dir / "logs/episodes.csv", usecols=episode_columns
    ).sort_values("completion_order")
    common_episodes = min(
        SNAPSHOT_EPISODES,
        int(
            min(
                episodes_control["completion_order"].max(),
                episodes_sft["completion_order"].max(),
            )
        ),
    )
    episodes_control = episodes_control[
        episodes_control["completion_order"] <= common_episodes
    ].copy()
    episodes_sft = episodes_sft[
        episodes_sft["completion_order"] <= common_episodes
    ].copy()

    optimization_sft = load_optimization(
        sft_dir / "logs/optimization.jsonl", SNAPSHOT_OPTIMIZATION_STEPS
    )
    common_optimization_steps = min(
        int(optimization_sft["optimization_step"].max()),
        SNAPSHOT_OPTIMIZATION_STEPS,
    )
    optimization_control = load_optimization(
        control_dir / "logs/optimization.jsonl", common_optimization_steps
    )

    validation_episodes_control = load_jsonl(
        control_dir / "logs/validation_episodes.jsonl"
    )
    validation_episodes_sft = load_jsonl(
        sft_dir / "logs/validation_episodes.jsonl"
    )
    validation_episodes_control = validation_episodes_control[
        validation_episodes_control["validation_run"] < common_validations
    ]
    validation_episodes_sft = validation_episodes_sft[
        validation_episodes_sft["validation_run"] < common_validations
    ]

    plot_validation(
        validation_control,
        validation_sft,
        OUTPUT_DIR / "01_fixed_greedy_validation.png",
    )
    plot_advantage(
        validation_control,
        validation_sft,
        OUTPUT_DIR / "02_sft_advantage_decay.png",
    )
    plot_behavior(
        episodes_control,
        episodes_sft,
        OUTPUT_DIR / "03_training_episode_rewards.png",
    )
    plot_optimization(
        optimization_control,
        optimization_sft,
        OUTPUT_DIR / "04_optimization_dynamics.png",
    )
    per_protein = plot_per_protein(
        validation_episodes_control,
        validation_episodes_sft,
        OUTPUT_DIR / "05_per_protein_validation.png",
    )

    protein_control = episodes_control.groupby("source_pdb")[[
        "total_reward",
        "terminal_reward",
        "mechanical_delta_strength_z",
        "mechanical_delta_toughness_z",
    ]].mean()
    protein_sft = episodes_sft.groupby("source_pdb")[[
        "total_reward",
        "terminal_reward",
        "mechanical_delta_strength_z",
        "mechanical_delta_toughness_z",
    ]].mean()
    protein_difference = protein_sft.join(
        protein_control, lsuffix="_sft", rsuffix="_random", how="inner"
    )
    protein_paired = {}
    for metric in (
        "total_reward",
        "terminal_reward",
        "mechanical_delta_strength_z",
        "mechanical_delta_toughness_z",
    ):
        difference = (
            protein_difference[f"{metric}_sft"]
            - protein_difference[f"{metric}_random"]
        ).to_numpy()
        ci = bootstrap_ci(difference, samples=5_000)
        protein_paired[metric] = {
            "mean_difference": float(difference.mean()),
            "ci_low": float(ci[0]),
            "ci_high": float(ci[1]),
            "sft_better_fraction": float((difference > 0).mean()),
        }

    summary = {
        "snapshot": {
            "common_episodes": common_episodes,
            "common_validation_runs": common_validations,
            "common_optimization_steps": common_optimization_steps,
            "episode_equivalent_epochs": float(common_episodes / 7_701),
            "current_epoch_index_zero_based": int(episodes_sft["epoch"].max()),
            "control_run": CONTROL_NAME,
            "sft_run": SFT_NAME,
        },
        "validation": validation_summary(validation_control, validation_sft),
        "training_episodes": {
            "random": window_summary(episodes_control),
            "sft": window_summary(episodes_sft),
            "protein_paired_sft_minus_random": protein_paired,
        },
        "optimization": {
            "random": optimization_summary(optimization_control),
            "sft": optimization_summary(optimization_sft),
        },
        "per_protein_validation": per_protein,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
