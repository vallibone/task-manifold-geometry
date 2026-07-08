"""Shared plotting helpers for the experiment trilogy."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


LAYER_ORDER = ["input", "hidden_1", "hidden_2", "hidden_3", "logits"]


def layer_order_from_data(layers):
    layers = list(layers)
    ordered = ["input"]
    hidden = sorted(
        [layer for layer in layers if str(layer).startswith("hidden_")],
        key=lambda name: int(str(name).split("_")[1]),
    )
    ordered.extend(hidden)
    ordered.append("logits")
    return [layer for layer in ordered if layer in layers]


def plot_distance_correlation_by_complexity(comparison):
    fig, ax = plt.subplots(figsize=(8, 5))
    for regime in comparison["regime"].unique():
        subset = comparison[comparison["regime"] == regime]
        ax.plot(subset["n_classes"], subset["distance_correlation"], marker="o", label=regime)
    ax.set_xlabel("Number of target classes")
    ax.set_ylabel("Distance correlation at logits")
    ax.set_title("Global distance preservation across target complexity")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_seed_summary_for_circle(seed_summary):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    metrics = [
        ("knn_mean", "knn_std", "KNN Preservation"),
        ("distance_mean", "distance_std", "Distance Correlation"),
        ("first_pc_mean", "first_pc_std", "First PC Variance"),
    ]
    for ax, (mean, std, title) in zip(axes, metrics):
        for scheme in ["intrinsic_arc", "observed_x", "observed_y"]:
            subset = seed_summary[(seed_summary.regime == "circle") & (seed_summary.scheme == scheme)]
            ax.errorbar(subset.n_classes, subset[mean], yerr=subset[std], marker="o", capsize=3, label=scheme)
        ax.set_title(title)
        ax.set_xlabel("Classes")
    axes[0].legend()
    fig.tight_layout()
    return fig, axes


def plot_capacity_allocation_grid(summary):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plots = [
        ("accuracy_mean", "accuracy_std", "Accuracy"),
        ("knn_mean", "knn_std", "kNN Preservation"),
        ("distance_mean", "distance_std", "Distance Correlation"),
        ("arc_r2_mean", "arc_r2_std", "Arc Reconstruction $R^2$"),
    ]
    for ax, (metric, std_col, title) in zip(axes.flat, plots):
        for allocation in ["back_loaded", "uniform", "front_loaded"]:
            df = summary[summary["allocation"] == allocation].sort_values("parameter_count")
            if df.empty:
                continue
            ax.errorbar(df["parameter_count"], df[metric], yerr=df[std_col], marker="o", linewidth=2, capsize=3, label=allocation.replace("_", " ").title())
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("Actual parameter count")
        ax.grid(alpha=0.3)
    axes[0, 0].legend()
    fig.suptitle("Capacity Allocation vs Representation Geometry", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig, axes


def plot_layerwise_seed_distribution(df, summary, metric, mean_col, std_col, ylabel, title, highlight_seeds=None):
    highlight_seeds = set(highlight_seeds or [])
    order = layer_order_from_data(df["layer"].unique())
    fig, ax = plt.subplots(figsize=(8, 4))
    for seed in sorted(df["seed"].unique()):
        subset = df[df["seed"] == seed].assign(layer=lambda d: pd.Categorical(d["layer"], categories=order, ordered=True)).sort_values("layer")
        linewidth = 2.5 if seed in highlight_seeds else 1.0
        alpha = 0.95 if seed in highlight_seeds else 0.25
        label = f"seed={seed}" if seed in highlight_seeds else None
        ax.plot(subset["layer"].astype(str), subset[metric], marker="o" if seed in highlight_seeds else None, linewidth=linewidth, alpha=alpha, label=label)
    ax.errorbar(summary["layer"].astype(str), summary[mean_col], yerr=summary[std_col], marker="o", linewidth=3, capsize=4, label="mean +/- std")
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_scheme_layerwise_summary(summary, mean_col, std_col, ylabel, title):
    order = layer_order_from_data(summary["layer"].unique())
    fig, ax = plt.subplots(figsize=(8, 4))
    for scheme_label in summary["scheme_label"].unique():
        subset = summary[summary["scheme_label"] == scheme_label].copy()
        subset["layer"] = pd.Categorical(subset["layer"], categories=order, ordered=True)
        subset = subset.sort_values("layer")
        ax.errorbar(subset["layer"].astype(str), subset[mean_col], yerr=subset[std_col], marker="o", capsize=4, linewidth=2, label=scheme_label)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_scheme_delta_summary(summary, mean_col, std_col, ylabel, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    for scheme_label in summary["scheme_label"].unique():
        subset = summary[summary["scheme_label"] == scheme_label]
        ax.errorbar(subset["transition"], subset[mean_col], yerr=subset[std_col], marker="o", capsize=4, linewidth=2, label=scheme_label)
    ax.set_xlabel("Transition")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    return fig, ax
