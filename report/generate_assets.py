#!/usr/bin/env python3
"""Build traceable figures and a frozen analysis snapshot for the RL report."""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections import defaultdict, deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
ASSET_DIR = REPORT_DIR / "assets"
DATA_ROOT = Path("/mnt/nas/jianquanzhao/data/mprl/outputs")
PREDICTOR_ROOT = DATA_ROOT / "mechanical_property_predictor"
EVALUATION_ROOT = DATA_ROOT / "evaluate-mechanical-predictor"
TRAIN_ROOT = DATA_ROOT / "train/add_terminal_reward_train"
LATEST_RUN_NAME = (
    "async_a24_esm8_f8_g1_prioritized_n3_stepx0.025_"
    "terminalx8.0_20260821_093310"
)
LATEST_RUN = TRAIN_ROOT / LATEST_RUN_NAME

# The long run was still active while the report was prepared. These limits freeze
# the evidence used in the report and make later rebuilds reproducible.
VALIDATION_RUN_CUTOFF = 769
GLOBAL_STEP_CUTOFF = 4_432_416
EPISODE_ROW_CUTOFF = 184_000
SNAPSHOT_TIME = "2026-08-28 20:33（UTC+8）"

COLORS = {
    "blue": "#1f4e79",
    "light_blue": "#d9eaf7",
    "orange": "#c55a11",
    "light_orange": "#fce4d6",
    "green": "#548235",
    "light_green": "#e2f0d9",
    "red": "#c00000",
    "light_red": "#f4cccc",
    "gray": "#666666",
    "light_gray": "#eeeeee",
    "gold": "#bf9000",
}


def configure_matplotlib() -> None:
    noto_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(noto_path)
    noto_name = font_manager.FontProperties(fname=noto_path).get_name()
    plt.rcParams.update(
        {
            "font.family": noto_name,
            "font.sans-serif": [noto_name, "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(ASSET_DIR / name, facecolor="white")
    plt.close(fig)


def copy_source_assets() -> dict[str, str]:
    """Copy selected source figures under short, stable report paths."""

    sources = {
        # Dataset and surrogate-model evidence.
        "target_distributions.png": PREDICTOR_ROOT
        / "analysis_data/plots/target_distributions_log1p.png",
        "target_ecdf.png": PREDICTOR_ROOT / "analysis_data/plots/target_ecdf.png",
        "length_vs_targets.png": PREDICTOR_ROOT
        / "analysis_data/plots/length_vs_targets.png",
        "hbond_final_test_r2.png": PREDICTOR_ROOT
        / "hbond_analysis/plots/hbond_final_model_test_r2.png",
        "hbond_final_predictions.png": PREDICTOR_ROOT
        / "hbond_analysis/plots/hbond_final_model_test_predictions.png",
        "hbond_feature_importance.png": PREDICTOR_ROOT
        / "hbond_analysis/plots/hbond_final_model_feature_importance.png",
        "hbond_random_similarity.png": PREDICTOR_ROOT
        / "hbond_analysis/plots/dataset_split_random_vs_similarity_test_r2.png",
        "hbond_length_ablation.png": PREDICTOR_ROOT
        / "hbond_analysis/plots/hbond_length_controlled_model_ablation_r2.png",
        "hbond_partial_correlation.png": PREDICTOR_ROOT
        / "hbond_analysis/plots/hbond_partial_correlation_after_length_control.png",
        "hbond_correlation_heatmap.png": PREDICTOR_ROOT
        / "hbond_analysis/plots/hbond_property_correlation_heatmap.png",
        "hbond_geometry_toughness.png": PREDICTOR_ROOT
        / "hbond_analysis/plots/hbond_geometry_vs_v127.png",
        "hbond_geometry_strength.png": PREDICTOR_ROOT
        / "hbond_analysis/plots/hbond_geometry_vs_v128.png",
        # CATH predicted-structure test.
        "cath_overview.png": EVALUATION_ROOT / "figures/cath_test_dataset_overview.png",
        "cath_scatter.png": EVALUATION_ROOT
        / "figures/cath_test_groundtruth_vs_predicted_scatter.png",
        "cath_quality.png": EVALUATION_ROOT
        / "figures/cath_test_predicted_structure_quality_features.png",
        "cath_delta_hist.png": EVALUATION_ROOT
        / "figures/cath_test_prediction_delta_histograms.png",
        # MechanoPro external pressure test.
        "mechanopro_force_duplicates.png": EVALUATION_ROOT
        / "compared_based_predicted-MechanoPro-DB/figures/mechanopro_force_and_duplicates.png",
        "mechanopro_scatter.png": EVALUATION_ROOT
        / "compared_based_predicted-MechanoPro-DB/figures/mechanopro_force_vs_predictions_scatter.png",
        "mechanopro_structure_pair.png": EVALUATION_ROOT
        / "compared_based_predicted-MechanoPro-DB/figures/mechanopro_predicted_vs_groundtruth_heads.png",
        "mechanopro_topk.png": EVALUATION_ROOT
        / "compared_based_predicted-MechanoPro-DB/figures/mechanopro_topk_enrichment.png",
        "mechanopro_hit_rate.png": EVALUATION_ROOT
        / "compared_based_predicted-MechanoPro-DB/figures/mechanopro_topk_hit_rate.png",
        # Spider-silk material-level pressure test.
        "spider_distribution.png": EVALUATION_ROOT
        / "evalutaed-spider_silkom/figures/spider_silkom_material_property_distribution.png",
        "spider_scatter.png": EVALUATION_ROOT
        / "evalutaed-spider_silkom/figures/spider_silkom_predicted_vs_material_scatter.png",
        "spider_heatmap.png": EVALUATION_ROOT
        / "evalutaed-spider_silkom/figures/spider_silkom_spearman_heatmap.png",
        "spider_topk.png": EVALUATION_ROOT
        / "evalutaed-spider_silkom/figures/spider_silkom_topk_enrichment.png",
        # Early integration and long-run diagnosis.
        "smoke_terminal_reward.png": TRAIN_ROOT
        / "terminal_reward_esm_mixed_length_32/plots/terminal_reward.png",
        "smoke_loss.png": TRAIN_ROOT
        / "terminal_reward_esm_mixed_length_32/plots/optimization_loss.png",
        "smoke_q_values.png": TRAIN_ROOT
        / "terminal_reward_esm_mixed_length_32/plots/q_values.png",
        "smoke_reward_components.png": TRAIN_ROOT
        / "terminal_reward_esm_mixed_length_32/plots/reward_components.png",
        "old_optimization_diagnostics.png": TRAIN_ROOT
        / "full_esm_terminal_fully-change-max-steps/analysis/optimization_diagnostics.png",
        "old_policy_diagnostics.png": TRAIN_ROOT
        / "full_esm_terminal_fully-change-max-steps/analysis/reward_policy_diagnostics.png",
        "async_a12_q_values.png": TRAIN_ROOT
        / "async_a12_esm8_f8_g1_20260807_111457/plots/q_values.png",
        "async_a12_episode_reward.png": TRAIN_ROOT
        / "async_a12_esm8_f8_g1_20260807_111457/plots/episode_reward.png",
        # Frozen copies of plots from the latest long run.
        "latest_episode_reward.png": LATEST_RUN / "plots/episode_reward.png",
        "latest_terminal_reward.png": LATEST_RUN / "plots/terminal_reward.png",
        "latest_step_reward.png": LATEST_RUN / "plots/step_reward.png",
        "latest_reward_components.png": LATEST_RUN / "plots/reward_components.png",
        "latest_loss.png": LATEST_RUN / "plots/optimization_loss.png",
        "latest_td_error.png": LATEST_RUN / "plots/td_error.png",
        "latest_grad_norm.png": LATEST_RUN / "plots/grad_norm.png",
        "latest_q_values.png": LATEST_RUN / "plots/q_values.png",
        "latest_epsilon.png": LATEST_RUN / "plots/epsilon.png",
    }
    copied: dict[str, str] = {}
    for name, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(f"Required report asset is missing: {source}")
        target = ASSET_DIR / name
        shutil.copy2(source, target)
        copied[name] = str(source)
    return copied


def draw_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str = "#444444",
    fontsize: float = 11,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.4,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#555555",
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.5,
            color=color,
            connectionstyle=connectionstyle,
        )
    )


def conceptual_figures() -> None:
    # Scientific target and interpretation levels.
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("本项目技术目标的层级边界", fontweight="bold", pad=16)
    draw_box(
        ax,
        (0.06, 0.63),
        0.25,
        0.20,
        "当前可优化代理目标\n氢键网络与结构拓扑\n（粘附/内聚潜力）",
        facecolor=COLORS["light_green"],
    )
    draw_box(
        ax,
        (0.375, 0.63),
        0.25,
        0.20,
        "训练标签层\nSMD toughness v127\nSMD strength v128",
        facecolor=COLORS["light_blue"],
    )
    draw_box(
        ax,
        (0.69, 0.63),
        0.25,
        0.20,
        "外部目标层\n单分子拉力 / 丝材料性能\n（尚未建立可靠映射）",
        facecolor=COLORS["light_orange"],
    )
    arrow(ax, (0.31, 0.73), (0.375, 0.73), color=COLORS["green"])
    arrow(ax, (0.625, 0.73), (0.69, 0.73), color=COLORS["orange"])
    ax.text(0.343, 0.78, "模型拟合", ha="center", color=COLORS["green"], fontsize=10)
    ax.text(0.657, 0.78, "跨尺度外推", ha="center", color=COLORS["orange"], fontsize=10)
    draw_box(
        ax,
        (0.15, 0.22),
        0.70,
        0.19,
        "报告中的审慎定义：力学增强在当前工程实现中被操作化为\n“由氢键网络和拓扑代理的结构粘附/内聚潜力提升”，并不等价于已验证的宏观材料增强",
        facecolor="#fff2cc",
        edgecolor=COLORS["gold"],
        fontsize=11,
    )
    arrow(ax, (0.50, 0.63), (0.50, 0.41), color=COLORS["gold"])
    save_figure(fig, "concept_target_layers.png")

    # End-to-end technical route.
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("强化学习增强结构粘附/力学代理目标的技术路线", fontweight="bold", pad=14)
    labels = [
        ("PDB 数据集\n清洗与划分", COLORS["light_gray"]),
        ("ESM2 逐残基\n状态表征", COLORS["light_blue"]),
        ("DDQN + mask\n选择残基—氨基酸", COLORS["light_green"]),
        ("PyRosetta\n突变与局部 repack", COLORS["light_orange"]),
        ("步级约束\n碰撞/H 键/RMSD", "#f4cccc"),
        ("终端代理\nΔstrength/Δtoughness", "#fff2cc"),
    ]
    xs = np.linspace(0.03, 0.82, len(labels))
    for i, ((label, color), x) in enumerate(zip(labels, xs)):
        draw_box(ax, (float(x), 0.54), 0.145, 0.22, label, facecolor=color, fontsize=10.5)
        if i < len(labels) - 1:
            arrow(ax, (float(x + 0.145), 0.65), (float(xs[i + 1]), 0.65))
    draw_box(
        ax,
        (0.34, 0.16),
        0.32,
        0.17,
        "Replay（PER + n-step）\n更新 online Q / 同步 target Q",
        facecolor="#e4dfec",
    )
    arrow(ax, (0.895, 0.54), (0.66, 0.28), connectionstyle="arc3,rad=-0.18")
    arrow(ax, (0.34, 0.25), (0.19, 0.54), connectionstyle="arc3,rad=-0.18")
    save_figure(fig, "concept_technical_route.png")

    # MDP and reward decomposition.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("有限时域突变 MDP", fontweight="bold")
    draw_box(ax, (0.07, 0.66), 0.34, 0.18, "状态 sₜ\nESM2(L×1280) + visited mask", facecolor=COLORS["light_blue"], fontsize=10)
    draw_box(ax, (0.58, 0.66), 0.34, 0.18, "动作 aₜ\n位置 × 20 种氨基酸\nno-op/已访问位置被屏蔽", facecolor=COLORS["light_green"], fontsize=10)
    draw_box(ax, (0.58, 0.21), 0.34, 0.18, "转移\n突变 + 8 Å 局部 repack\n最长 24 步", facecolor=COLORS["light_orange"], fontsize=10)
    draw_box(ax, (0.07, 0.21), 0.34, 0.18, "奖励 rₜ\n弱零中心 shaping\n末步加入力学增量", facecolor="#fff2cc", fontsize=10)
    arrow(ax, (0.41, 0.75), (0.58, 0.75))
    arrow(ax, (0.75, 0.66), (0.75, 0.39))
    arrow(ax, (0.58, 0.30), (0.41, 0.30))
    arrow(ax, (0.24, 0.39), (0.24, 0.66))

    ax = axes[1]
    ax.axis("off")
    ax.set_title("当前奖励标度", fontweight="bold")
    ax.text(
        0.5,
        0.78,
        r"$r_{terminal}=8\times\frac{1}{2}(\Delta z_{strength}+\Delta z_{toughness})$",
        ha="center",
        va="center",
        fontsize=15,
        color=COLORS["blue"],
    )
    ax.text(
        0.5,
        0.58,
        r"$r_{step}=0.025\times weighted\_mean(centered\ components)$",
        ha="center",
        va="center",
        fontsize=13,
        color=COLORS["green"],
    )
    ax.text(0.5, 0.42, "24 步合法 shaping 理论绝对上限约 0.6", ha="center", fontsize=11)
    draw_box(
        ax,
        (0.12, 0.14),
        0.76,
        0.16,
        "设计意图：局部结构约束只作弱引导，最终选择依据应由\n相对初态的氢键拓扑—力学代理增量主导",
        facecolor=COLORS["light_gray"],
        fontsize=10.5,
    )
    save_figure(fig, "concept_mdp_reward.png")

    # Asynchronous actor-learner architecture.
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("集中式异步 actor–learner 架构", fontweight="bold", pad=14)
    draw_box(ax, (0.04, 0.60), 0.25, 0.25, "24 个 CPU actor\n独立 PyRosetta episode\n策略快照 + 局部 Q head", facecolor=COLORS["light_orange"])
    draw_box(ax, (0.38, 0.67), 0.24, 0.18, "GPU 1–3\n动态批量 ESM2 推理\n变长序列 padding", facecolor=COLORS["light_blue"])
    draw_box(ax, (0.71, 0.60), 0.25, 0.25, "GPU 0 learner\nPER + n=3 + DDQN\nF8/G1，batch 128", facecolor=COLORS["light_green"])
    draw_box(ax, (0.36, 0.19), 0.28, 0.20, "有界 IPC 队列\nembedding / transition / snapshot\n提供 backpressure", facecolor="#e4dfec")
    arrow(ax, (0.29, 0.75), (0.38, 0.75), color=COLORS["blue"])
    arrow(ax, (0.62, 0.75), (0.71, 0.75), color=COLORS["green"])
    arrow(ax, (0.17, 0.60), (0.39, 0.39), color=COLORS["orange"])
    arrow(ax, (0.64, 0.30), (0.83, 0.60), color=COLORS["green"])
    arrow(ax, (0.83, 0.60), (0.29, 0.65), color=COLORS["green"], connectionstyle="arc3,rad=0.28")
    ax.text(0.55, 0.93, "专用 validation actor：固定 PDB、固定 seed、ε=0、冻结策略快照", ha="center", fontsize=11, color=COLORS["red"])
    save_figure(fig, "concept_async_architecture.png")

    # Evidence hierarchy / stage gate.
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("证据等级与结论权限", fontweight="bold", pad=14)
    rows = [
        (0.76, "A级：固定未见 PDB 的 greedy 配对增量", "决定是否声称策略有效", COLORS["light_green"]),
        (0.57, "B级：训练 behavior-policy 的终端增量", "判断采样分布，不独立证明泛化", COLORS["light_blue"]),
        (0.38, "C级：Q/TD/loss/grad 数值指标", "只证明优化稳定或 Bellman 自洽", "#fff2cc"),
        (0.19, "D级：吞吐、显存、IPC、worker 生命周期", "只证明工程可运行", COLORS["light_gray"]),
    ]
    for y, left, right, color in rows:
        draw_box(ax, (0.06, y), 0.48, 0.13, left, facecolor=color, fontsize=10.5)
        arrow(ax, (0.54, y + 0.065), (0.62, y + 0.065))
        draw_box(ax, (0.62, y), 0.32, 0.13, right, facecolor="white", fontsize=10)
    save_figure(fig, "concept_evidence_hierarchy.png")

    # Roadmap and go/no-go gates.
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("下一阶段建议：以可证伪门槛推进", fontweight="bold", pad=14)
    stages = [
        ("G0\n代理灵敏度", "单突变/多突变\nstrength 与 toughness\n非零且可重复"),
        ("G1\n评估可信度", "≥128 固定验证 PDB\n≥3 seeds\nbootstrap CI"),
        ("G2\n学习消融", "uniform/PER ×\nn=1/3/6 ×\n状态增强"),
        ("G3\n候选复核", "结构质量 + OOD\n重复 relax\n高保真模拟"),
        ("G4\n实验闭环", "粘附/单分子/材料\n按层级重新标定"),
    ]
    xs = np.linspace(0.03, 0.81, len(stages))
    palette = [COLORS["light_orange"], COLORS["light_blue"], COLORS["light_green"], "#e4dfec", "#fff2cc"]
    for i, ((title, body), x, color) in enumerate(zip(stages, xs, palette)):
        draw_box(ax, (float(x), 0.58), 0.16, 0.18, title, facecolor=color, fontsize=11)
        draw_box(ax, (float(x), 0.24), 0.16, 0.22, body, facecolor="white", fontsize=9.5)
        arrow(ax, (float(x + 0.08), 0.58), (float(x + 0.08), 0.46))
        if i < len(stages) - 1:
            arrow(ax, (float(x + 0.16), 0.67), (float(xs[i + 1]), 0.67))
    ax.text(0.50, 0.08, "任一门槛未通过：停止扩大训练预算，回到对应上游问题", ha="center", color=COLORS["red"], fontsize=11, fontweight="bold")
    save_figure(fig, "concept_stage_gates.png")


def static_quantitative_figures() -> None:
    # Horizon effect calculated from gamma=0.99 and replay batch size 128.
    horizons = np.asarray([24, 50, 1024], dtype=float)
    terminal_weight = np.power(0.99, horizons - 1)
    batch_terminal_probability = 1.0 - np.power(1.0 - 1.0 / horizons, 128)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, values, title, ylabel in [
        (axes[0], terminal_weight, "终端信号在 episode 起点的折扣权重", r"$\gamma^{H-1}$"),
        (axes[1], batch_terminal_probability, "batch 128 至少含一个终端 transition 的概率", "概率"),
    ]:
        bars = ax.bar(["H=24", "H=50", "H=1024"], values, color=[COLORS["green"], COLORS["blue"], COLORS["orange"]])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("缩短 horizon 改变的不只是训练时长，而是终端 credit 的可见性", fontweight="bold", y=1.03)
    fig.tight_layout()
    save_figure(fig, "quant_horizon_credit.png")

    # Actor scaling.
    actors = np.asarray([12, 24, 36])
    throughput = np.asarray([8.1613, 9.6780, 9.5636])
    latency = np.asarray([23.22, 48.72, 78.41])
    fig, ax1 = plt.subplots(figsize=(9.5, 5.2))
    bars = ax1.bar(actors - 1.8, throughput, width=3.6, color=COLORS["blue"], label="稳态吞吐（step/s）")
    ax1.set_xlabel("actor 数量")
    ax1.set_ylabel("吞吐（step/s）", color=COLORS["blue"])
    ax1.set_xticks(actors)
    ax1.set_ylim(0, 12)
    ax2 = ax1.twinx()
    ax2.plot(actors, latency, marker="o", linewidth=2.2, color=COLORS["orange"], label="单 actor episode 延迟（s）")
    ax2.set_ylabel("平均 episode 延迟（s）", color=COLORS["orange"])
    ax2.set_ylim(0, 95)
    for bar, value in zip(bars, throughput):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.2, f"{value:.2f}", ha="center")
    for x, value in zip(actors, latency):
        ax2.text(x + 0.5, value, f"{value:.1f}s", color=COLORS["orange"], va="center")
    ax1.grid(axis="y", alpha=0.2)
    ax1.set_title("actor 扩容收益在 A24 附近进入平台", fontweight="bold")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")
    fig.tight_layout()
    save_figure(fig, "quant_actor_scaling.png")

    # Cross-domain evidence summary.
    labels = ["SMD/CATH\n真实结构", "SMD/CATH\n预测结构", "MechanoPro\n预测结构", "蛛丝材料\ntype-balanced"]
    toughness = [0.8891, 0.7427, -0.3743, -0.1378]
    strength = [0.8326, 0.8168, -0.3915, -0.0728]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    width = 0.35
    ax.bar(x - width / 2, toughness, width, label="toughness 相关", color=COLORS["blue"])
    ax.bar(x + width / 2, strength, width, label="strength 相关", color=COLORS["orange"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Spearman 相关系数")
    ax.set_ylim(-0.55, 1.0)
    ax.set_title("随着物理层级跨越，当前代理模型的可迁移性显著下降", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, "quant_cross_domain_spearman.png")

    # Exploration timeline.
    events = [
        ("6月", "数据/环境/ESM2\n变长 DDQN 骨架"),
        ("7月上", "氢键 RF 与\n结构预测评估"),
        ("7月下", "终端 reward 接入\n长 horizon 暴露 Q 峰值"),
        ("8月上", "F/G、FP32\n异步 actor–learner"),
        ("8月中", "Δreward、零中心\nPER + n=3"),
        ("8月下", "visited mask\n固定 greedy validation"),
    ]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.set_xlim(-0.3, len(events) - 0.7)
    ax.set_ylim(-0.55, 0.75)
    ax.axis("off")
    ax.plot(range(len(events)), np.zeros(len(events)), color=COLORS["blue"], linewidth=3)
    for i, (date, label) in enumerate(events):
        ax.scatter([i], [0], s=150, color=COLORS["blue"], zorder=3, edgecolor="white", linewidth=1.5)
        y = 0.24 if i % 2 == 0 else -0.38
        ax.text(i, y, f"{date}\n{label}", ha="center", va="center", fontsize=10)
    ax.set_title("技术探索主线：从“能运行”转向“目标是否可学习、结论是否可信”", fontweight="bold", pad=18)
    save_figure(fig, "quant_exploration_timeline.png")


def read_validation_summary() -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    path = LATEST_RUN / "logs/validation_summary.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(record["validation_run"]) <= VALIDATION_RUN_CUTOFF:
                records.append(record)
    if not records or int(records[-1]["validation_run"]) != VALIDATION_RUN_CUTOFF:
        raise RuntimeError("The frozen validation cutoff is not available.")
    return records


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if values.size < window:
        return np.full_like(values, np.nan, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    result[window - 1 :] = np.convolve(values, np.ones(window) / window, mode="valid")
    return result


def validation_figures(records: list[dict[str, float]]) -> dict[str, object]:
    runs = np.asarray([int(record["validation_run"]) for record in records])
    episodes = np.asarray([int(record["trigger_episode"]) for record in records])
    fields = [
        ("terminal_reward_mean", "scaled terminal reward", COLORS["blue"]),
        ("strength_mean", "Δz strength", COLORS["orange"]),
        ("toughness_mean", "Δz toughness", COLORS["green"]),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9.2), sharex=True)
    for ax, (key, label, color) in zip(axes, fields):
        values = np.asarray([float(record[key]) for record in records])
        ax.plot(episodes, values, color=color, alpha=0.24, linewidth=0.7, label="单次 16-PDB 验证")
        ax.plot(episodes, rolling_mean(values, 25), color=color, linewidth=2.1, label="25 次滑动平均")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right")
    axes[-1].set_xlabel("完成的训练 episode")
    fig.suptitle(
        "最新长时运行的固定 greedy validation：早期改善后仍未跨过零线",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, "latest_validation_trajectory.png")

    # Block means show the non-monotonic plateau more clearly than one lucky checkpoint.
    block_size = 100
    blocks: list[tuple[int, int, dict[str, float]]] = []
    for start in range(0, len(records), block_size):
        block = records[start : start + block_size]
        block_values = {
            key: float(np.mean([float(record[key]) for record in block]))
            for key, _, _ in fields
        }
        block_values["terminal_reward_positive_fraction"] = float(
            np.mean([float(record["terminal_reward_positive_fraction"]) for record in block])
        )
        blocks.append((start, start + len(block) - 1, block_values))
    labels = [f"{start}–{end}" for start, end, _ in blocks]
    x = np.arange(len(blocks))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9))
    axes[0].bar(x - 0.27, [b[2]["terminal_reward_mean"] for b in blocks], 0.27, label="terminal", color=COLORS["blue"])
    axes[0].bar(x, [4 * b[2]["strength_mean"] for b in blocks], 0.27, label="4×Δz strength", color=COLORS["orange"])
    axes[0].bar(x + 0.27, [4 * b[2]["toughness_mean"] for b in blocks], 0.27, label="4×Δz toughness", color=COLORS["green"])
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(x, labels, rotation=35)
    axes[0].set_title("每 100 次验证的均值")
    axes[0].set_ylabel("reward 尺度")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(x, [b[2]["terminal_reward_positive_fraction"] for b in blocks], color=COLORS["blue"])
    axes[1].axhline(0.5, color=COLORS["red"], linestyle="--", linewidth=1.2, label="50%")
    axes[1].set_xticks(x, labels, rotation=35)
    axes[1].set_ylim(0, 0.55)
    axes[1].set_title("终端 reward 为正的验证蛋白比例")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("固定验证的阶段块统计：中期最好，后期没有持续单调改善", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, "latest_validation_blocks.png")

    # Confidence interval diagnostic.
    means = np.asarray([float(r["terminal_reward_mean"]) for r in records])
    lows = np.asarray([float(r["terminal_reward_mean_ci_low"]) for r in records])
    highs = np.asarray([float(r["terminal_reward_mean_ci_high"]) for r in records])
    stride = 5
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.fill_between(episodes[::stride], lows[::stride], highs[::stride], color=COLORS["light_blue"], alpha=0.55, label="单轮 bootstrap 95% CI（每 5 轮显示）")
    ax.plot(episodes, rolling_mean(means, 25), color=COLORS["blue"], linewidth=2.2, label="terminal mean 的 25 轮滑动平均")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.9)
    ax.set_xlabel("完成的训练 episode")
    ax.set_ylabel("scaled terminal reward")
    ax.set_title("16 个验证 PDB 导致区间较宽；尚无一轮平均值的 CI 明确高于零", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    save_figure(fig, "latest_validation_confidence.png")

    summary: dict[str, object] = {
        "records": len(records),
        "last_validation_run": int(records[-1]["validation_run"]),
        "last_trigger_episode": int(records[-1]["trigger_episode"]),
        "last_global_step": int(records[-1]["global_step"]),
        "terminal_ci_excludes_zero_negative": int(sum(float(r["terminal_reward_mean_ci_high"]) < 0 for r in records)),
        "terminal_ci_excludes_zero_positive": int(sum(float(r["terminal_reward_mean_ci_low"]) > 0 for r in records)),
        "terminal_mean_positive_runs": int(sum(float(r["terminal_reward_mean"]) > 0 for r in records)),
    }
    for key, _, _ in fields:
        values = np.asarray([float(record[key]) for record in records])
        roll25 = rolling_mean(values, 25)
        valid_roll25 = roll25[np.isfinite(roll25)]
        summary[key] = {
            "baseline": float(values[0]),
            "first_50_mean": float(values[:50].mean()),
            "last_50_mean": float(values[-50:].mean()),
            "last_200_mean": float(values[-200:].mean()),
            "best_25_run_rolling_mean": float(valid_roll25.max()),
            "last_25_run_rolling_mean": float(valid_roll25[-1]),
            "linear_correlation_with_run": float(np.corrcoef(runs, values)[0, 1]),
        }
    summary["block_statistics"] = [
        {"start_run": start, "end_run": end, **values}
        for start, end, values in blocks
    ]
    return summary


def validation_pdb_heatmap() -> dict[str, object]:
    path = LATEST_RUN / "logs/validation_episodes.jsonl"
    by_pdb_run: dict[str, dict[int, float]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            run = int(record["validation_run"])
            if run > VALIDATION_RUN_CUTOFF:
                continue
            pdb = Path(record["pdb_path"]).stem
            by_pdb_run[pdb][run] = float(record["terminal_reward"])
    pdbs = sorted(by_pdb_run)
    block_edges = list(range(0, VALIDATION_RUN_CUTOFF + 1, 100)) + [VALIDATION_RUN_CUTOFF + 1]
    matrix = np.full((len(pdbs), len(block_edges) - 1), np.nan)
    for i, pdb in enumerate(pdbs):
        for j, (start, end) in enumerate(zip(block_edges[:-1], block_edges[1:])):
            values = [value for run, value in by_pdb_run[pdb].items() if start <= run < end]
            if values:
                matrix[i, j] = float(np.mean(values))
    scale = max(0.5, float(np.nanpercentile(np.abs(matrix), 90)))
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    image = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-scale, vmax=scale)
    ax.set_yticks(np.arange(len(pdbs)), pdbs)
    labels = [f"{a}–{b-1}" for a, b in zip(block_edges[:-1], block_edges[1:])]
    ax.set_xticks(np.arange(len(labels)), labels, rotation=35)
    ax.set_xlabel("validation run 分块")
    ax.set_title("固定验证集的逐 PDB terminal reward：蛋白间异质性长期存在", fontweight="bold")
    fig.colorbar(image, ax=ax, label="分块平均 scaled terminal reward")
    fig.tight_layout()
    save_figure(fig, "latest_validation_pdb_heatmap.png")
    return {"validation_pdbs": len(pdbs), "validation_pdb_names": pdbs}


def training_epoch_figure() -> dict[str, object]:
    path = LATEST_RUN / "logs/episodes.csv"
    sums: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[int, int] = defaultdict(int)
    positive: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    columns = {
        "total_reward": "total_reward",
        "terminal_reward": "terminal_reward",
        "strength": "mechanical_delta_strength_z",
        "toughness": "mechanical_delta_toughness_z",
        "epsilon": "epsilon",
        "elapsed": "elapsed_seconds",
    }
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            if row_index >= EPISODE_ROW_CUTOFF:
                break
            epoch = int(float(row["epoch"]))
            counts[epoch] += 1
            for output_name, source_name in columns.items():
                value = float(row[source_name])
                sums[epoch][output_name] += value
                if output_name in {"terminal_reward", "strength", "toughness"} and value > 0:
                    positive[epoch][output_name] += 1
    epochs = sorted(counts)
    means = {
        key: np.asarray([sums[epoch][key] / counts[epoch] for epoch in epochs])
        for key in columns
    }
    fractions = {
        key: np.asarray([positive[epoch][key] / counts[epoch] for epoch in epochs])
        for key in ("terminal_reward", "strength", "toughness")
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    axes[0, 0].plot(epochs, means["total_reward"], marker="o", label="total", color=COLORS["blue"])
    axes[0, 0].plot(epochs, means["terminal_reward"], marker="o", label="terminal", color=COLORS["orange"])
    axes[0, 0].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0, 0].set_title("训练 behavior-policy reward")
    axes[0, 0].legend()
    axes[0, 1].plot(epochs, means["strength"], marker="o", label="Δz strength", color=COLORS["orange"])
    axes[0, 1].plot(epochs, means["toughness"], marker="o", label="Δz toughness", color=COLORS["green"])
    axes[0, 1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0, 1].set_title("训练集机械代理增量")
    axes[0, 1].legend()
    axes[1, 0].plot(epochs, fractions["terminal_reward"], label="terminal > 0", color=COLORS["blue"])
    axes[1, 0].plot(epochs, fractions["strength"], label="strength > 0", color=COLORS["orange"])
    axes[1, 0].plot(epochs, fractions["toughness"], label="toughness > 0", color=COLORS["green"])
    axes[1, 0].axhline(0.5, color=COLORS["red"], linestyle="--", linewidth=0.8)
    axes[1, 0].set_ylim(0.2, 0.6)
    axes[1, 0].set_title("正向 episode 比例")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot(epochs, means["elapsed"], marker="o", color=COLORS["gray"])
    axes[1, 1].set_title("单 actor episode 平均耗时")
    axes[1, 1].set_ylabel("秒")
    for ax in axes.ravel():
        ax.set_xlabel("epoch（从 0 开始）")
        ax.grid(alpha=0.2)
    fig.suptitle("最新长时运行训练集轨迹：主要表现为 toughness 少恶化，非稳定正提升", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, "latest_training_epochs.png")
    return {
        "episode_rows": int(sum(counts.values())),
        "complete_epochs": int(sum(count == 7701 for count in counts.values())),
        "last_epoch": int(epochs[-1]),
        "last_epoch_rows": int(counts[epochs[-1]]),
        "epoch_means": [
            {
                "epoch": int(epoch),
                "count": int(counts[epoch]),
                **{key: float(means[key][index]) for key in columns},
                **{f"{key}_positive_fraction": float(fractions[key][index]) for key in fractions},
            }
            for index, epoch in enumerate(epochs)
        ],
    }


def optimization_figures() -> dict[str, object]:
    path = LATEST_RUN / "logs/optimization.jsonl"
    metric_keys = ["loss", "mean_absolute_td_error", "grad_norm", "mean_q_value", "mean_target_q_value"]
    bin_width = 500_000
    bins: dict[int, dict[str, object]] = {}
    recent = deque(maxlen=10_000)
    diagnostic_recent = deque(maxlen=1_000)
    record_count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            global_step = int(record.get("global_step", 0))
            if global_step > GLOBAL_STEP_CUTOFF:
                break
            record_count += 1
            recent.append(record)
            bin_start = (global_step // bin_width) * bin_width
            if bin_start not in bins:
                bins[bin_start] = {
                    "count": 0,
                    "sum": {key: 0.0 for key in metric_keys},
                    "max": {key: -math.inf for key in metric_keys},
                }
            bucket = bins[bin_start]
            bucket["count"] = int(bucket["count"]) + 1
            for key in metric_keys:
                value = float(record[key])
                bucket["sum"][key] += value
                bucket["max"][key] = max(float(bucket["max"][key]), value)
            if "per/replay/terminal_fraction" in record:
                diagnostic_recent.append(record)

    starts = sorted(bins)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    plot_specs = [
        ("loss", "Huber loss", COLORS["blue"]),
        ("mean_absolute_td_error", "mean |TD error|", COLORS["orange"]),
        ("grad_norm", "gradient norm", COLORS["green"]),
        ("mean_q_value", "mean Q / target Q", COLORS["red"]),
    ]
    x = np.asarray(starts) / 1e6
    for ax, (key, title, color) in zip(axes.ravel(), plot_specs):
        means = np.asarray([bins[start]["sum"][key] / bins[start]["count"] for start in starts])
        ax.plot(x, means, marker="o", color=color, label=title)
        if key == "mean_q_value":
            target_means = np.asarray([bins[start]["sum"]["mean_target_q_value"] / bins[start]["count"] for start in starts])
            ax.plot(x, target_means, marker="s", color=COLORS["gray"], label="mean target Q")
            ax.legend()
        ax.set_title(title)
        ax.set_xlabel("global step 分箱起点（百万）")
        ax.grid(alpha=0.2)
    fig.suptitle("最新长时运行的优化指标已从早期峰值恢复并保持有限", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, "latest_optimization_bins.png")

    # PER diagnostics: compare the last 1,000 diagnostic snapshots.
    scopes = ["replay", "priority_top_1pct", "sampled_batch"]
    scope_labels = ["完整 replay", "priority 前 1%", "实际 sampled batch"]
    per_stats: dict[str, dict[str, float]] = {}
    for scope in scopes:
        per_stats[scope] = {}
        for metric in [
            "terminal_fraction",
            "terminal_reward_positive_fraction",
            "terminal_reward_mean",
            "strength_positive_fraction",
            "strength_mean",
            "toughness_positive_fraction",
            "toughness_mean",
        ]:
            key = f"per/{scope}/{metric}"
            values = [float(record[key]) for record in diagnostic_recent if key in record]
            per_stats[scope][metric] = float(np.mean(values))
    x = np.arange(len(scopes))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9))
    axes[0].bar(x, [per_stats[s]["terminal_fraction"] for s in scopes], color=[COLORS["blue"], COLORS["red"], COLORS["orange"]])
    axes[0].set_xticks(x, scope_labels)
    axes[0].set_ylabel("terminal transition 比例")
    axes[0].set_title("PER 显著提高终端样本可见性")
    axes[0].grid(axis="y", alpha=0.2)
    width = 0.25
    axes[1].bar(x - width, [per_stats[s]["terminal_reward_positive_fraction"] for s in scopes], width, label="terminal > 0", color=COLORS["blue"])
    axes[1].bar(x, [per_stats[s]["strength_positive_fraction"] for s in scopes], width, label="strength > 0", color=COLORS["orange"])
    axes[1].bar(x + width, [per_stats[s]["toughness_positive_fraction"] for s in scopes], width, label="toughness > 0", color=COLORS["green"])
    axes[1].axhline(0.5, color=COLORS["red"], linestyle="--", linewidth=0.9)
    axes[1].set_xticks(x, scope_labels)
    axes[1].set_ylim(0, 0.58)
    axes[1].set_title("但 priority 前 1% 主要是负向失败样本")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("最近 1,000 个 PER 诊断快照", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, "latest_per_diagnostics.png")

    recent_stats: dict[str, dict[str, float]] = {}
    for key in metric_keys + ["mean_importance_weight"]:
        values = np.asarray([float(record[key]) for record in recent])
        recent_stats[key] = {
            "mean": float(values.mean()),
            "max": float(values.max()),
            "p99": float(np.quantile(values, 0.99)),
        }
    return {
        "optimization_records": record_count,
        "last_global_step": int(recent[-1]["global_step"]),
        "last_optimization_step": int(recent[-1]["optimization_step"]),
        "recent_10000": recent_stats,
        "bins": [
            {
                "global_step_start": int(start),
                "count": int(bins[start]["count"]),
                **{
                    f"{key}_mean": float(bins[start]["sum"][key] / bins[start]["count"])
                    for key in metric_keys
                },
                **{f"{key}_max": float(bins[start]["max"][key]) for key in metric_keys},
            }
            for start in starts
        ],
        "per_recent_1000": per_stats,
    }


def write_snapshot(
    copied_assets: dict[str, str],
    validation: dict[str, object],
    validation_pdb: dict[str, object],
    training: dict[str, object],
    optimization: dict[str, object],
) -> None:
    payload = {
        "report_snapshot_time": SNAPSHOT_TIME,
        "latest_run": LATEST_RUN_NAME,
        "cutoffs": {
            "validation_run": VALIDATION_RUN_CUTOFF,
            "global_step": GLOBAL_STEP_CUTOFF,
            "episode_rows": EPISODE_ROW_CUTOFF,
        },
        "copied_source_assets": copied_assets,
        "validation": validation,
        "validation_pdb": validation_pdb,
        "training": training,
        "optimization": optimization,
        "verification": {"pytest": "139 passed, 2 skipped in 13.02s"},
    }
    (REPORT_DIR / "analysis_snapshot.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    copied_assets = copy_source_assets()
    conceptual_figures()
    static_quantitative_figures()
    records = read_validation_summary()
    validation = validation_figures(records)
    validation_pdb = validation_pdb_heatmap()
    training = training_epoch_figure()
    optimization = optimization_figures()
    write_snapshot(copied_assets, validation, validation_pdb, training, optimization)
    print(f"Generated {len(list(ASSET_DIR.glob('*.png')))} report figures in {ASSET_DIR}")


if __name__ == "__main__":
    main()
