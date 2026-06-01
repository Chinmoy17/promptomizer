"""Publication-quality visualization for RAG comparison results.

Generates:
1. Quality comparison (grouped bar chart)
2. Efficiency comparison (tokens, cost, latency panels)
3. Quality-cost Pareto trade-off (scatter)
4. Radar chart (multi-dimensional normalized)
5. Performance heatmap
6. Score distribution (violin plots)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger


# Style configuration
SYSTEM_COLORS = {
    "langchain_rag": "#4C72B0",
    "dspy_baseline": "#55A868",
    "dspy_bootstrap": "#C44E52",
    "dspy_mipro": "#8172B2",
}

SYSTEM_LABELS = {
    "langchain_rag": "LangChain (Baseline)",
    "dspy_baseline": "DSPy (CoT)",
    "dspy_bootstrap": "DSPy + Bootstrap",
    "dspy_mipro": "DSPy + MIPROv2",
}


def setup_style():
    """Configure matplotlib for publication-quality figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_quality_comparison(
    results_df: pd.DataFrame,
    metrics: list[str] = None,
    output_path: str | Path = "results/figures/quality_comparison.png",
) -> None:
    """Grouped bar chart comparing quality metrics across systems.

    Args:
        results_df: DataFrame with columns ['system', metric1, metric2, ...].
        metrics: List of metric column names.
        output_path: Where to save the figure.
    """
    setup_style()

    if metrics is None:
        metrics = ["answer_accuracy", "context_relevance", "faithfulness"]

    fig, ax = plt.subplots(figsize=(10, 6))

    systems = results_df["system"].unique()
    x = np.arange(len(metrics))
    width = 0.8 / len(systems)

    for i, system in enumerate(systems):
        system_data = results_df[results_df["system"] == system]
        values = [system_data[m].mean() for m in metrics]
        offset = (i - len(systems) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=SYSTEM_LABELS.get(system, system),
            color=SYSTEM_COLORS.get(system, f"C{i}"),
            alpha=0.85,
        )

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("RAG System Quality Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved quality comparison: {output_path}")


def plot_efficiency_panels(
    efficiency_data: list[dict],
    output_path: str | Path = "results/figures/efficiency_comparison.png",
) -> None:
    """Three-panel efficiency comparison (tokens, cost, latency).

    Args:
        efficiency_data: List of system summary dicts from EfficiencyTracker.
        output_path: Where to save the figure.
    """
    setup_style()

    df = pd.DataFrame(efficiency_data)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel 1: Tokens
    colors = [SYSTEM_COLORS.get(s, "C0") for s in df["system"]]
    labels = [SYSTEM_LABELS.get(s, s) for s in df["system"]]

    axes[0].barh(labels, df["avg_tokens"], color=colors, alpha=0.85)
    axes[0].set_xlabel("Avg Tokens / Query")
    axes[0].set_title("Token Usage")

    # Panel 2: Cost
    axes[1].barh(labels, df["avg_cost_usd"] * 1000, color=colors, alpha=0.85)
    axes[1].set_xlabel("Avg Cost (mUSD / Query)")
    axes[1].set_title("Cost per Query")

    # Panel 3: Latency
    axes[2].barh(labels, df["avg_latency_ms"], color=colors, alpha=0.85)
    axes[2].set_xlabel("Avg Latency (ms)")
    axes[2].set_title("Latency")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved efficiency panels: {output_path}")


def plot_pareto_tradeoff(
    quality_scores: dict[str, float],
    cost_scores: dict[str, float],
    output_path: str | Path = "results/figures/pareto_tradeoff.png",
) -> None:
    """Quality-cost Pareto trade-off scatter plot.

    Args:
        quality_scores: {system_name: avg_quality_score}.
        cost_scores: {system_name: avg_cost_per_query}.
        output_path: Where to save.
    """
    setup_style()

    fig, ax = plt.subplots(figsize=(8, 6))

    for system in quality_scores:
        ax.scatter(
            cost_scores[system] * 1000,
            quality_scores[system],
            s=150,
            c=SYSTEM_COLORS.get(system, "gray"),
            label=SYSTEM_LABELS.get(system, system),
            zorder=5,
            edgecolors="black",
            linewidths=0.5,
        )

    ax.set_xlabel("Cost (mUSD / Query)")
    ax.set_ylabel("Avg Quality Score")
    ax.set_title("Quality vs. Cost Trade-off")
    ax.legend()
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved Pareto trade-off: {output_path}")


def plot_radar_chart(
    system_scores: dict[str, dict[str, float]],
    output_path: str | Path = "results/figures/radar_chart.png",
) -> None:
    """Radar chart for multi-dimensional comparison.

    Args:
        system_scores: {system: {metric: normalized_score}}.
        output_path: Where to save.
    """
    setup_style()

    categories = list(next(iter(system_scores.values())).keys())
    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for system, scores in system_scores.items():
        values = [scores[cat] for cat in categories]
        values += values[:1]
        ax.plot(
            angles,
            values,
            "o-",
            linewidth=2,
            label=SYSTEM_LABELS.get(system, system),
            color=SYSTEM_COLORS.get(system, "gray"),
        )
        ax.fill(angles, values, alpha=0.1, color=SYSTEM_COLORS.get(system, "gray"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.replace("_", " ").title() for c in categories])
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.set_title("Multi-Dimensional System Comparison", pad=20)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved radar chart: {output_path}")


def plot_heatmap(
    results_df: pd.DataFrame,
    metrics: list[str] = None,
    output_path: str | Path = "results/figures/heatmap.png",
) -> None:
    """Performance heatmap (systems × metrics).

    Args:
        results_df: DataFrame with 'system' column and metric columns.
        metrics: Metric columns to include.
        output_path: Where to save.
    """
    setup_style()

    if metrics is None:
        metrics = ["answer_accuracy", "context_relevance", "faithfulness"]

    # Pivot to system × metric
    pivot = results_df.groupby("system")[metrics].mean()
    pivot.index = [SYSTEM_LABELS.get(s, s) for s in pivot.index]
    pivot.columns = [m.replace("_", " ").title() for m in pivot.columns]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("System Performance Heatmap")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved heatmap: {output_path}")


def plot_score_distributions(
    results_df: pd.DataFrame,
    metric: str = "answer_accuracy",
    output_path: str | Path = "results/figures/score_distribution.png",
) -> None:
    """Violin plot showing score distributions per system.

    Args:
        results_df: DataFrame with 'system' column and metric column.
        metric: Which metric to plot distributions for.
        output_path: Where to save.
    """
    setup_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    plot_df = results_df[["system", metric]].copy()
    plot_df["system"] = plot_df["system"].map(lambda s: SYSTEM_LABELS.get(s, s))

    palette = [SYSTEM_COLORS.get(s, "C0") for s in results_df["system"].unique()]

    sns.violinplot(
        data=plot_df,
        x="system",
        y=metric,
        palette=palette,
        ax=ax,
        inner="box",
        cut=0,
    )

    ax.set_xlabel("")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Score Distribution: {metric.replace('_', ' ').title()}")
    ax.set_ylim(-0.05, 1.05)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved score distribution: {output_path}")


def generate_all_figures(
    results_df: pd.DataFrame,
    efficiency_data: list[dict],
    output_dir: str | Path = "results/figures",
) -> None:
    """Generate all publication figures.

    Args:
        results_df: Full evaluation results DataFrame.
        efficiency_data: Efficiency summaries from EfficiencyTracker.
        output_dir: Directory for all figure outputs.
    """
    output_dir = Path(output_dir)

    metrics = ["answer_accuracy", "context_relevance", "faithfulness"]

    plot_quality_comparison(results_df, metrics, output_dir / "quality_comparison.png")
    plot_efficiency_panels(efficiency_data, output_dir / "efficiency_comparison.png")
    plot_heatmap(results_df, metrics, output_dir / "heatmap.png")

    # Score distributions for each metric
    for metric in metrics:
        plot_score_distributions(
            results_df, metric, output_dir / f"distribution_{metric}.png"
        )

    # Radar chart
    systems = results_df["system"].unique()
    system_scores = {}
    for system in systems:
        sys_df = results_df[results_df["system"] == system]
        system_scores[system] = {m: sys_df[m].mean() for m in metrics}
    plot_radar_chart(system_scores, output_dir / "radar_chart.png")

    # Pareto trade-off
    quality_scores = {s: results_df[results_df["system"] == s][metrics].mean().mean() for s in systems}
    cost_scores = {d["system"]: d["avg_cost_usd"] for d in efficiency_data}
    if cost_scores:
        plot_pareto_tradeoff(quality_scores, cost_scores, output_dir / "pareto_tradeoff.png")

    logger.info(f"All figures generated in {output_dir}")
