"""Run the trajectory-through-layers experiment end to end."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from representation_geometry.architecture import count_mlp_parameters, scale_template_to_budget
from representation_geometry.runner import run_representation_experiment
from representation_geometry.worlds import build_world
from representation_geometry.models import MLPRepresentationModel
from representation_geometry.plotting import (
    LAYER_ORDER,
    plot_layerwise_seed_distribution,
    plot_scheme_delta_summary,
    plot_scheme_layerwise_summary,
)
from representation_geometry.probes import (
    distance_correlation_by_layer,
    geodesic_knn_preservation_by_layer,
    knn_preservation_by_layer,
    latent_reconstruction_by_layer,
    pca_summary_by_layer,
)
from representation_geometry.targets import make_target_by_scheme


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

DEFAULT_EPOCHS = 100
DEFAULT_N_SAMPLES = 3000
DEFAULT_NOISE_STD = 0.02
QUICK_EPOCHS = 5
QUICK_N_SAMPLES = 500

N_CLASSES = 32
K = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 128


def run_model_for_world(
    regime,
    regime_kwargs,
    target_scheme,
    hidden_dims,
    seed,
    config,
):
    world = build_world(
        regime=regime,
        regime_kwargs=regime_kwargs,
        seed=seed,
        n_samples=config.n_samples,
        noise_std=config.noise_std,
    )
    labels = make_target_by_scheme(world, scheme=target_scheme, n_classes=N_CLASSES)
    model = MLPRepresentationModel(
        input_dim=world["observed"].shape[1],
        output_dim=len(np.unique(labels)),
        hidden_dims=hidden_dims,
        epochs=config.epochs,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        random_state=seed,
        verbose=False,
    )
    result = run_representation_experiment(world, labels, model)
    result["seed"] = seed
    result["hidden_dims"] = hidden_dims
    result["actual_params"] = count_mlp_parameters(
        input_dim=world["observed"].shape[1],
        hidden_dims=hidden_dims,
        output_dim=len(np.unique(labels)),
    )
    result["labels"] = labels
    return result


def collect_layerwise_absolute_metrics(result, scheme_label=None, target_scheme=None):
    reps = result["representations"]
    metadata = result["world"]["metadata_df"]
    arc = metadata["arc_parameter"].to_numpy()

    euclidean_knn = knn_preservation_by_layer(
        reps,
        reference_layer="input",
        k=K,
    ).rename(columns={"knn_preservation": "euclidean_knn"})

    geodesic_knn = geodesic_knn_preservation_by_layer(
        reps,
        arc_parameter=arc,
        k=K,
        periodic=False,
    )

    distance = distance_correlation_by_layer(
        reps,
        reference_layer="input",
    )
    pca = pca_summary_by_layer(reps)
    latent = latent_reconstruction_by_layer(
        reps,
        metadata,
        target_columns=["arc_parameter"],
    ).rename(columns={"r2": "arc_r2"})

    df = (
        euclidean_knn[["layer", "euclidean_knn"]]
        .merge(geodesic_knn[["layer", "geodesic_knn_preservation"]], on="layer", how="left")
        .merge(distance[["layer", "distance_correlation"]], on="layer", how="left")
        .merge(
            pca[["layer", "first_pc_variance", "first_2pc_variance", "components_95"]],
            on="layer",
            how="left",
        )
        .merge(
            latent[latent["latent_variable"] == "arc_parameter"][["layer", "arc_r2"]],
            on="layer",
            how="left",
        )
    )

    df["seed"] = result["seed"]
    df["accuracy"] = result["accuracy"]
    df["hidden_dims"] = str(tuple(result["hidden_dims"]))
    df["parameter_count"] = result["actual_params"]
    if scheme_label is not None:
        df["scheme_label"] = scheme_label
    if target_scheme is not None:
        df["target_scheme"] = target_scheme
    return df


def summarize_layerwise(df, by_scheme=False):
    group_cols = ["scheme_label", "layer"] if by_scheme else ["layer"]
    work = df.copy()
    work["layer"] = pd.Categorical(work["layer"], categories=LAYER_ORDER, ordered=True)
    return (
        work.groupby(group_cols, observed=True)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            euclidean_knn_mean=("euclidean_knn", "mean"),
            euclidean_knn_std=("euclidean_knn", "std"),
            geodesic_knn_mean=("geodesic_knn_preservation", "mean"),
            geodesic_knn_std=("geodesic_knn_preservation", "std"),
            distance_correlation_mean=("distance_correlation", "mean"),
            distance_correlation_std=("distance_correlation", "std"),
            distance_corr_mean=("distance_correlation", "mean"),
            distance_corr_std=("distance_correlation", "std"),
            arc_r2_mean=("arc_r2", "mean"),
            arc_r2_std=("arc_r2", "std"),
            first_pc_mean=("first_pc_variance", "mean"),
            first_pc_std=("first_pc_variance", "std"),
        )
        .reset_index()
    )


def final_logit_comparison(layerwise_df):
    return (
        layerwise_df[layerwise_df["layer"] == "logits"]
        [
            [
                "seed",
                "accuracy",
                "euclidean_knn",
                "geodesic_knn_preservation",
                "distance_correlation",
                "arc_r2",
                "first_pc_variance",
                "components_95",
                "hidden_dims",
                "parameter_count",
            ]
        ]
        .sort_values("distance_correlation")
        .reset_index(drop=True)
    )


def compute_change_tables(layerwise_df, by_scheme=False):
    work = layerwise_df.copy()
    work["layer"] = pd.Categorical(work["layer"], categories=LAYER_ORDER, ordered=True)
    sort_cols = ["scheme_label", "seed", "layer"] if by_scheme else ["seed", "layer"]
    group_cols = ["scheme_label", "seed"] if by_scheme else ["seed"]
    work = work.sort_values(sort_cols).reset_index(drop=True)
    work = work.rename(
        columns={
            "geodesic_knn_preservation": "geodesic_knn",
            "distance_correlation": "distance_corr",
        }
    )

    absolute_rows = []
    delta_rows = []

    for group_key, group in work.groupby(group_cols, observed=True):
        group = group.sort_values("layer").reset_index(drop=True)
        input_row = group[group["layer"] == "input"].iloc[0]
        scheme_label = group_key[0] if by_scheme else None
        seed = group_key[1] if by_scheme else group_key

        for _, row in group.iterrows():
            record = {
                "seed": seed,
                "layer": row["layer"],
                "accuracy": row["accuracy"],
                "abs_euclidean_knn_loss": input_row["euclidean_knn"] - row["euclidean_knn"],
                "abs_geodesic_knn_loss": input_row["geodesic_knn"] - row["geodesic_knn"],
                "abs_distance_loss": input_row["distance_corr"] - row["distance_corr"],
                "abs_arc_r2_gain": row["arc_r2"] - input_row["arc_r2"],
                "abs_first_pc_gain": row["first_pc_variance"] - input_row["first_pc_variance"],
                "euclidean_knn": row["euclidean_knn"],
                "geodesic_knn": row["geodesic_knn"],
                "distance_corr": row["distance_corr"],
                "arc_r2": row["arc_r2"],
                "first_pc_variance": row["first_pc_variance"],
            }
            if by_scheme:
                record["scheme_label"] = scheme_label
            absolute_rows.append(record)

        for index in range(1, len(group)):
            prev = group.iloc[index - 1]
            curr = group.iloc[index]
            record = {
                "seed": seed,
                "previous_layer": prev["layer"],
                "layer": curr["layer"],
                "transition": f"{prev['layer']} -> {curr['layer']}",
                "accuracy": curr["accuracy"],
                "delta_euclidean_knn_loss": prev["euclidean_knn"] - curr["euclidean_knn"],
                "delta_geodesic_knn_loss": prev["geodesic_knn"] - curr["geodesic_knn"],
                "delta_distance_loss": prev["distance_corr"] - curr["distance_corr"],
                "delta_arc_r2_gain": curr["arc_r2"] - prev["arc_r2"],
                "delta_first_pc_gain": curr["first_pc_variance"] - prev["first_pc_variance"],
                "previous_euclidean_knn": prev["euclidean_knn"],
                "current_euclidean_knn": curr["euclidean_knn"],
                "previous_geodesic_knn": prev["geodesic_knn"],
                "current_geodesic_knn": curr["geodesic_knn"],
                "previous_distance_corr": prev["distance_corr"],
                "current_distance_corr": curr["distance_corr"],
                "previous_arc_r2": prev["arc_r2"],
                "current_arc_r2": curr["arc_r2"],
            }
            if by_scheme:
                record["scheme_label"] = scheme_label
            delta_rows.append(record)

    return pd.DataFrame(absolute_rows), pd.DataFrame(delta_rows)


def summarize_delta(delta_df, by_scheme=False):
    group_cols = ["scheme_label", "transition"] if by_scheme else ["transition"]
    return (
        delta_df.groupby(group_cols, observed=True)
        .agg(
            delta_euclidean_knn_loss_mean=("delta_euclidean_knn_loss", "mean"),
            delta_euclidean_knn_loss_std=("delta_euclidean_knn_loss", "std"),
            delta_geodesic_knn_loss_mean=("delta_geodesic_knn_loss", "mean"),
            delta_geodesic_knn_loss_std=("delta_geodesic_knn_loss", "std"),
            delta_distance_loss_mean=("delta_distance_loss", "mean"),
            delta_distance_loss_std=("delta_distance_loss", "std"),
            delta_arc_r2_gain_mean=("delta_arc_r2_gain", "mean"),
            delta_arc_r2_gain_std=("delta_arc_r2_gain", "std"),
            delta_first_pc_gain_mean=("delta_first_pc_gain", "mean"),
            delta_first_pc_gain_std=("delta_first_pc_gain", "std"),
        )
        .reset_index()
    )


def run_spiral_scarcity(config, output_dir):
    arch = scale_template_to_budget(
        input_dim=2,
        output_dim=N_CLASSES,
        template=[1, 2, 4],
        target_params=750,
        tolerance=0.05,
    )
    hidden_dims = arch["hidden_dims"]
    rows = []
    results = {}

    for seed in range(1, 11):
        print(f"Running spiral scarcity seed={seed}")
        result = run_model_for_world(
            regime="spiral",
            regime_kwargs={"scale": 1.0},
            target_scheme="arc_bins",
            hidden_dims=hidden_dims,
            seed=seed,
            config=config,
        )
        results[seed] = result
        rows.append(collect_layerwise_absolute_metrics(result))

    layerwise_df = pd.concat(rows, ignore_index=True)
    logit_comparison = final_logit_comparison(layerwise_df)
    logit_comparison["distance_z"] = (
        logit_comparison["distance_correlation"] - logit_comparison["distance_correlation"].mean()
    ) / logit_comparison["distance_correlation"].std()
    logit_comparison["arc_r2_z"] = (
        logit_comparison["arc_r2"] - logit_comparison["arc_r2"].mean()
    ) / logit_comparison["arc_r2"].std()

    summary = summarize_layerwise(layerwise_df)
    absolute_change, delta_change = compute_change_tables(layerwise_df)
    final_absolute = (
        absolute_change[absolute_change["layer"] == "logits"]
        .sort_values("abs_distance_loss", ascending=False)
        .reset_index(drop=True)
    )
    delta_summary = summarize_delta(delta_change)

    layerwise_df.to_csv(output_dir / "spiral_layerwise_by_seed.csv", index=False)
    logit_comparison.to_csv(output_dir / "spiral_logit_seed_comparison.csv", index=False)
    summary.to_csv(output_dir / "spiral_layerwise_summary.csv", index=False)
    absolute_change.to_csv(output_dir / "spiral_absolute_change_by_seed_layer.csv", index=False)
    delta_change.to_csv(output_dir / "spiral_delta_change_by_seed_transition.csv", index=False)
    final_absolute.to_csv(output_dir / "spiral_final_absolute_change_by_seed.csv", index=False)
    delta_summary.to_csv(output_dir / "spiral_delta_summary_by_transition.csv", index=False)

    worst_seed = int(logit_comparison.iloc[0]["seed"])
    best_seed = int(logit_comparison.iloc[-1]["seed"])

    fig, _ = plot_layerwise_seed_distribution(
        layerwise_df,
        summary,
        metric="arc_r2",
        mean_col="arc_r2_mean",
        std_col="arc_r2_std",
        ylabel="Arc reconstruction R^2",
        title="Scarce-capacity spiral: intrinsic coordinate construction",
        highlight_seeds=[worst_seed, best_seed],
    )
    fig.savefig(output_dir / "spiral_intrinsic_coordinate_construction.png", dpi=200)

    fig, _ = plot_layerwise_seed_distribution(
        layerwise_df,
        summary,
        metric="distance_correlation",
        mean_col="distance_correlation_mean",
        std_col="distance_correlation_std",
        ylabel="Distance correlation",
        title="Scarce-capacity spiral: progressive global collapse",
        highlight_seeds=[worst_seed, best_seed],
    )
    fig.savefig(output_dir / "spiral_global_collapse_through_layers.png", dpi=200)


def run_sinewave_task_comparison(config, output_dir):
    rows = []
    scheme_configs = [
        ("intrinsic_arc", "arc_bins"),
        ("observed_y", "y_bins"),
    ]
    hidden_dims = [64, 64, 32]

    for scheme_label, target_scheme in scheme_configs:
        for seed in range(1, 11):
            print(f"Running sinewave {scheme_label} seed={seed}")
            result = run_model_for_world(
                regime="sinewave",
                regime_kwargs={"amplitude": 1.0, "frequency": 2.0, "phase": 0.0},
                target_scheme=target_scheme,
                hidden_dims=hidden_dims,
                seed=seed,
                config=config,
            )
            rows.append(
                collect_layerwise_absolute_metrics(
                    result,
                    scheme_label=scheme_label,
                    target_scheme=target_scheme,
                )
            )

    layerwise_df = pd.concat(rows, ignore_index=True)
    summary = summarize_layerwise(layerwise_df, by_scheme=True)
    absolute_change, delta_change = compute_change_tables(layerwise_df, by_scheme=True)
    final_absolute = (
        absolute_change[absolute_change["layer"] == "logits"]
        .sort_values(["scheme_label", "abs_distance_loss"], ascending=[True, False])
        .reset_index(drop=True)
    )
    delta_summary = summarize_delta(delta_change, by_scheme=True)

    layerwise_df.to_csv(output_dir / "sinewave_layerwise_by_scheme_seed.csv", index=False)
    absolute_change.to_csv(output_dir / "sinewave_absolute_change.csv", index=False)
    delta_change.to_csv(output_dir / "sinewave_delta_change.csv", index=False)
    final_absolute.to_csv(output_dir / "sinewave_final_absolute_change.csv", index=False)
    summary.to_csv(output_dir / "sinewave_layerwise_summary.csv", index=False)
    delta_summary.to_csv(output_dir / "sinewave_delta_summary.csv", index=False)

    fig, _ = plot_scheme_layerwise_summary(
        summary,
        mean_col="geodesic_knn_mean",
        std_col="geodesic_knn_std",
        ylabel="Geodesic KNN preservation",
        title="Sinewave: geodesic geometry through layers",
    )
    fig.savefig(output_dir / "sinewave_geodesic_preservation_through_layers.png", dpi=200)

    fig, _ = plot_scheme_delta_summary(
        delta_summary,
        mean_col="delta_geodesic_knn_loss_mean",
        std_col="delta_geodesic_knn_loss_std",
        ylabel="Delta geodesic KNN loss",
        title="Sinewave: layerwise geodesic geometry loss",
    )
    fig.savefig(output_dir / "sinewave_geodesic_loss_by_transition.png", dpi=200)

    fig, _ = plot_scheme_layerwise_summary(
        summary,
        mean_col="distance_corr_mean",
        std_col="distance_corr_std",
        ylabel="Distance correlation",
        title="Sinewave: global geometry through layers",
    )
    fig.savefig(output_dir / "sinewave_global_geometry_through_layers.png", dpi=200)

    fig, _ = plot_scheme_layerwise_summary(
        summary,
        mean_col="arc_r2_mean",
        std_col="arc_r2_std",
        ylabel="Arc reconstruction R^2",
        title="Sinewave: intrinsic coordinate recoverability",
    )
    fig.savefig(output_dir / "sinewave_intrinsic_coordinate_recoverability.png", dpi=200)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["all", "spiral", "sinewave"],
        default="all",
        help="Run all experiment blocks or one block.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"Training epochs per MLP fit. Defaults to {DEFAULT_EPOCHS}, or {QUICK_EPOCHS} with --quick.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help=f"Samples per manifold. Defaults to {DEFAULT_N_SAMPLES}, or {QUICK_N_SAMPLES} with --quick.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=DEFAULT_NOISE_STD,
        help=f"Gaussian observation noise. Defaults to {DEFAULT_NOISE_STD}.",
    )
    parser.add_argument("--quick", action="store_true", help="Use a smaller fast run for smoke testing.")
    return parser.parse_args()


def resolve_config(args):
    epochs = args.epochs if args.epochs is not None else (QUICK_EPOCHS if args.quick else DEFAULT_EPOCHS)
    n_samples = args.n_samples if args.n_samples is not None else (QUICK_N_SAMPLES if args.quick else DEFAULT_N_SAMPLES)
    if epochs <= 0:
        raise ValueError(f"--epochs must be greater than 0, got {epochs}")
    if n_samples <= 1:
        raise ValueError(f"--n-samples must be greater than 1, got {n_samples}")
    if args.noise_std < 0:
        raise ValueError(f"--noise-std cannot be less than 0, got {args.noise_std}")
    return argparse.Namespace(
        epochs=epochs,
        n_samples=n_samples,
        noise_std=args.noise_std,
        quick=args.quick,
    )


def output_dir_for_config(config):
    if (
        config.epochs == DEFAULT_EPOCHS
        and config.n_samples == DEFAULT_N_SAMPLES
        and config.noise_std == DEFAULT_NOISE_STD
        and not config.quick
    ):
        return OUTPUT_DIR

    label = (
        f"epochs{config.epochs}_samples{config.n_samples}_"
        f"noise{str(config.noise_std).replace('.', 'p')}"
    )
    if config.quick:
        label += "_quick"
    return OUTPUT_DIR / "runs" / label


def main():
    args = parse_args()
    config = resolve_config(args)
    output_dir = output_dir_for_config(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in {"all", "spiral"}:
        run_spiral_scarcity(config, output_dir)
    if args.stage in {"all", "sinewave"}:
        run_sinewave_task_comparison(config, output_dir)

    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()

