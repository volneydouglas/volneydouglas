"""
Visualization module — generates publication-ready plots for the paper.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from analysis.metrics import ConditionMetrics

matplotlib.use("Agg")  # Non-interactive backend for server environments

# Color palette
COLORS = {
    "extended_phenotype": "#2196F3",
    "separate_entity": "#FF5722",
}


def plot_h1_cooperation_by_frame(
    metrics: dict[str, ConditionMetrics], output_dir: Path
) -> None:
    """Bar chart: overall cooperation rate by identity frame."""
    ext_rates = []
    sep_rates = []
    for m in metrics.values():
        if m.frame_type == "extended_phenotype":
            ext_rates.extend(m.cooperation_rates)
        else:
            sep_rates.extend(m.cooperation_rates)

    means = [np.mean(ext_rates), np.mean(sep_rates)]
    sems = [
        np.std(ext_rates) / np.sqrt(len(ext_rates)),
        np.std(sep_rates) / np.sqrt(len(sep_rates)),
    ]
    labels = ["Extended\nPhenotype", "Separate\nEntity"]
    colors = [COLORS["extended_phenotype"], COLORS["separate_entity"]]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, means, yerr=sems, color=colors, capsize=5, width=0.5)
    ax.set_ylabel("Mean Cooperation Rate")
    ax.set_title("H1: Identity Frame Effect on Cooperation")
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "h1_cooperation_by_frame.png", dpi=150)
    plt.close(fig)


def plot_h2_frame_x_game(
    metrics: dict[str, ConditionMetrics], output_dir: Path
) -> None:
    """Grouped bar chart: cooperation rate by frame × game type."""
    game_labels = {
        "prisoners_dilemma": "Prisoner's\nDilemma",
        "stag_hunt": "Stag\nHunt",
        "battle_of_sexes": "Battle of\nthe Sexes",
    }
    game_order = ["prisoners_dilemma", "stag_hunt", "battle_of_sexes"]

    ext_means, sep_means = [], []
    ext_sems, sep_sems = [], []

    for game in game_order:
        ext = [
            r
            for m in metrics.values()
            if m.game_type == game and m.frame_type == "extended_phenotype"
            for r in m.cooperation_rates
        ]
        sep = [
            r
            for m in metrics.values()
            if m.game_type == game and m.frame_type == "separate_entity"
            for r in m.cooperation_rates
        ]
        ext_means.append(np.mean(ext) if ext else 0)
        sep_means.append(np.mean(sep) if sep else 0)
        ext_sems.append(np.std(ext) / np.sqrt(len(ext)) if ext else 0)
        sep_sems.append(np.std(sep) / np.sqrt(len(sep)) if sep else 0)

    x = np.arange(len(game_order))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        x - width / 2, ext_means, width, yerr=ext_sems,
        label="Extended Phenotype", color=COLORS["extended_phenotype"], capsize=4,
    )
    ax.bar(
        x + width / 2, sep_means, width, yerr=sep_sems,
        label="Separate Entity", color=COLORS["separate_entity"], capsize=4,
    )
    ax.set_ylabel("Mean Cooperation Rate")
    ax.set_title("H2: Frame × Game Type Interaction")
    ax.set_xticks(x)
    ax.set_xticklabels([game_labels[g] for g in game_order])
    ax.set_ylim(0, 1)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "h2_frame_x_game.png", dpi=150)
    plt.close(fig)


def plot_h3_frame_x_partner(
    metrics: dict[str, ConditionMetrics], output_dir: Path
) -> None:
    """Grouped bar chart: cooperation by frame × partner type."""
    conditions = [
        ("extended_phenotype", "human"),
        ("extended_phenotype", "ai"),
        ("separate_entity", "human"),
        ("separate_entity", "ai"),
    ]
    labels = [
        "Extended\n+ Human",
        "Extended\n+ AI",
        "Separate\n+ Human",
        "Separate\n+ AI",
    ]
    colors_list = [
        COLORS["extended_phenotype"],
        "#64B5F6",
        COLORS["separate_entity"],
        "#FF8A65",
    ]

    means, sems = [], []
    for frame, partner in conditions:
        rates = [
            r
            for m in metrics.values()
            if m.frame_type == frame and m.partner_type == partner
            for r in m.cooperation_rates
        ]
        means.append(np.mean(rates) if rates else 0)
        sems.append(np.std(rates) / np.sqrt(len(rates)) if rates else 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, means, yerr=sems, color=colors_list, capsize=4, width=0.6)
    ax.set_ylabel("Mean Cooperation Rate")
    ax.set_title("H3: Frame × Partner Type Interaction")
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "h3_frame_x_partner.png", dpi=150)
    plt.close(fig)


def plot_h5_convergence(
    metrics: dict[str, ConditionMetrics], output_dir: Path
) -> None:
    """Line chart: round-by-round cooperation trends by frame type."""
    ext_trends: list[list[float]] = []
    sep_trends: list[list[float]] = []

    for m in metrics.values():
        if not m.round_by_round_cooperation:
            continue
        if m.frame_type == "extended_phenotype":
            ext_trends.append(m.round_by_round_cooperation)
        else:
            sep_trends.append(m.round_by_round_cooperation)

    if not ext_trends or not sep_trends:
        return

    max_rounds = max(
        max(len(t) for t in ext_trends), max(len(t) for t in sep_trends)
    )

    def avg_trend(trends: list[list[float]]) -> tuple[list[float], list[float]]:
        means, sems = [], []
        for r in range(max_rounds):
            vals = [t[r] for t in trends if r < len(t)]
            means.append(np.mean(vals) if vals else 0)
            sems.append(np.std(vals) / np.sqrt(len(vals)) if vals else 0)
        return means, sems

    ext_m, ext_s = avg_trend(ext_trends)
    sep_m, sep_s = avg_trend(sep_trends)
    rounds = list(range(1, max_rounds + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        rounds, ext_m, yerr=ext_s, label="Extended Phenotype",
        color=COLORS["extended_phenotype"], marker="o", capsize=3,
    )
    ax.errorbar(
        rounds, sep_m, yerr=sep_s, label="Separate Entity",
        color=COLORS["separate_entity"], marker="s", capsize=3,
    )
    ax.set_xlabel("Round")
    ax.set_ylabel("Cooperation Rate")
    ax.set_title("H5: Cooperation Convergence Over Rounds")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "h5_convergence.png", dpi=150)
    plt.close(fig)


def plot_all(metrics: dict[str, ConditionMetrics], output_dir: str | Path) -> None:
    """Generate all plots and save to output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plot_h1_cooperation_by_frame(metrics, output_path)
    plot_h2_frame_x_game(metrics, output_path)
    plot_h3_frame_x_partner(metrics, output_path)
    plot_h5_convergence(metrics, output_path)
    print(f"Plots saved to {output_path}")
