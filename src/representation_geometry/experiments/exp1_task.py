"""Run the public local-vs-global representation experiment end to end."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from representation_geometry.runner import run_representation_experiment
from representation_geometry.worlds import build_world
from representation_geometry.models import MLPRepresentationModel
from representation_geometry.plotting import (
    plot_distance_correlation_by_complexity,
    plot_seed_summary_for_circle,
)
from representation_geometry.probes import (
    distance_correlation_by_layer,
    knn_preservation_by_layer,
    latent_reconstruction_by_layer,
    pca_summary_by_layer,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
MANIFOLDS = [
    ("circle", {"radius": 1.0}),
    ("spiral", {"scale": 1.0}),
    ("sinewave", {"amplitude": 1.0, "frequency": 2.0, "phase": 0.0}),
]
DEFAULT_EPOCHS = 100
DEFAULT_N_SAMPLES = 3000
DEFAULT_SEED = 42
QUICK_EPOCHS = 10
QUICK_N_SAMPLES = 500


def bin_continuous(values, n_classes):
    values = np.asarray(values)
    scaled = (values - values.min()) / (values.max() - values.min())
    labels = np.floor(scaled * n_classes).astype(int)
    return np.clip(labels, 0, n_classes - 1)


def bin_continuous_balanced(values, n_classes):
    values = np.asarray(values)
    ranks = pd.Series(values).rank(method="first").to_numpy()
    scaled = (ranks - 1) / (len(ranks) - 1)
    labels = np.floor(scaled * n_classes).astype(int)
    return np.clip(labels, 0, n_classes - 1)


def make_arc_target(world, n_classes):
    arc = world["metadata_df"]["arc_parameter"].to_numpy()
    return bin_continuous(arc, n_classes)


def make_target_by_scheme(world, scheme, n_classes=2, angle=0.0):
    values = world["observed"]
    metadata = world["metadata_df"]
    x = values[:, 0]
    y_coord = values[:, 1]

    if scheme == "arc_bins":
        return bin_continuous_balanced(metadata["arc_parameter"].to_numpy(), n_classes)
    if scheme == "x_bins":
        return bin_continuous_balanced(x, n_classes)
    if scheme == "y_bins":
        return bin_continuous_balanced(y_coord, n_classes)
    if scheme == "radial_bins":
        radius = np.linalg.norm(values[:, :2], axis=1)
        return bin_continuous_balanced(radius, n_classes)
    if scheme == "quadrant":
        x_sign = (x > np.median(x)).astype(int)
        y_sign = (y_coord > np.median(y_coord)).astype(int)
        return x_sign + 2 * y_sign
    if scheme == "halfspace_angle":
        direction = np.array([np.cos(angle), np.sin(angle)])
        projection = values[:, :2] @ direction
        return (projection > np.median(projection)).astype(int)

    raise ValueError(f"Unknown labelling scheme: {scheme}")


def run_arc_complexity_experiments(config):
    experiments = [
        {
            "name": f"{regime}_arc_{n_classes}",
            "regime": regime,
            "n_classes": n_classes,
            "regime_kwargs": kwargs,
        }
        for regime, kwargs in MANIFOLDS
        for n_classes in [2, 8, 32]
    ]

    results = []
    for experiment in experiments:
        print(f"Running {experiment['name']}")
        world = build_world(
            regime=experiment["regime"],
            regime_kwargs=experiment["regime_kwargs"],
            seed=config.seed,
            n_samples=config.n_samples,
        )
        labels = make_arc_target(world, experiment["n_classes"])
        model = MLPRepresentationModel(
            input_dim=world["observed"].shape[1],
            output_dim=experiment["n_classes"],
            hidden_dims=[64, 64, 32],
            epochs=config.epochs,
            learning_rate=1e-3,
            batch_size=128,
            random_state=config.seed,
            verbose=False,
        )
        result = run_representation_experiment(world, labels, model)
        result["experiment_name"] = experiment["name"]
        result["regime"] = experiment["regime"]
        result["n_classes"] = experiment["n_classes"]
        results.append(result)

    return results, summarize_arc_complexity(results)


def summarize_arc_complexity(results):
    all_knn = []
    all_dist = []
    all_pca = []
    all_latent = []

    for result in results:
        representations = result["representations"]
        metadata = result["world"]["metadata_df"]
        knn = knn_preservation_by_layer(representations, k=10)
        dist = distance_correlation_by_layer(representations)
        pca = pca_summary_by_layer(representations)
        latent = latent_reconstruction_by_layer(
            representations,
            metadata,
            target_columns=["arc_parameter"],
        )

        for frame in [knn, dist, pca, latent]:
            frame["experiment"] = result["experiment_name"]
            frame["regime"] = result["regime"]
            frame["n_classes"] = result["n_classes"]
            frame["accuracy"] = result["accuracy"]

        all_knn.append(knn)
        all_dist.append(dist)
        all_pca.append(pca)
        all_latent.append(latent)

    all_knn = pd.concat(all_knn, ignore_index=True)
    all_dist = pd.concat(all_dist, ignore_index=True)
    all_pca = pd.concat(all_pca, ignore_index=True)
    all_latent = pd.concat(all_latent, ignore_index=True)
    final_layer = "logits"

    comparison = (
        all_knn[all_knn["layer"] == final_layer]
        .merge(
            all_dist[all_dist["layer"] == final_layer],
            on=["experiment", "regime", "n_classes", "accuracy", "layer", "reference_layer"],
            how="left",
        )
        .merge(
            all_pca[all_pca["layer"] == final_layer],
            on=["experiment", "regime", "n_classes", "accuracy", "layer"],
            how="left",
        )
        .merge(
            all_latent[
                (all_latent["layer"] == final_layer)
                & (all_latent["latent_variable"] == "arc_parameter")
            ],
            on=["experiment", "regime", "n_classes", "accuracy", "layer"],
            how="left",
        )
    )
    return comparison


def run_controlled_experiments(config):
    labelling_schemes = [
        {"scheme": "arc_bins", "n_classes_list": [2, 8, 32], "label": "intrinsic_arc"},
        {"scheme": "x_bins", "n_classes_list": [2, 8, 32], "label": "observed_x"},
        {"scheme": "y_bins", "n_classes_list": [2, 8, 32], "label": "observed_y"},
        {
            "scheme": "halfspace_angle",
            "n_classes_list": [2],
            "label": "vertical_halfspace",
            "angle": np.pi / 2,
        },
        {
            "scheme": "halfspace_angle",
            "n_classes_list": [2],
            "label": "horizontal_halfspace",
            "angle": 0.0,
        },
        {"scheme": "quadrant", "n_classes_list": [4], "label": "quadrant"},
    ]

    experiments = []
    for regime, regime_kwargs in MANIFOLDS:
        for scheme_config in labelling_schemes:
            for n_classes in scheme_config["n_classes_list"]:
                experiments.append(
                    {
                        "name": f"{regime}_{scheme_config['label']}_{n_classes}",
                        "regime": regime,
                        "regime_kwargs": regime_kwargs,
                        "scheme": scheme_config["scheme"],
                        "scheme_label": scheme_config["label"],
                        "n_classes": n_classes,
                        "angle": scheme_config.get("angle", 0.0),
                    }
                )

    rows = []
    for experiment in experiments:
        print(f"Running {experiment['name']}")
        world = build_world(
            regime=experiment["regime"],
            regime_kwargs=experiment["regime_kwargs"],
            seed=config.seed,
            n_samples=config.n_samples,
        )
        labels = make_target_by_scheme(
            world,
            scheme=experiment["scheme"],
            n_classes=experiment["n_classes"],
            angle=experiment["angle"],
        )
        model = MLPRepresentationModel(
            input_dim=world["observed"].shape[1],
            output_dim=len(np.unique(labels)),
            hidden_dims=[64, 64, 32],
            epochs=config.epochs,
            learning_rate=1e-3,
            batch_size=128,
            random_state=config.seed,
            verbose=False,
        )
        result = run_representation_experiment(world, labels, model)
        rows.append(probe_logits_row(result, experiment, labels))

    return pd.DataFrame(rows)


def probe_logits_row(result, experiment, labels):
    representations = result["representations"]
    metadata = result["world"]["metadata_df"]
    knn = knn_preservation_by_layer(representations, k=10)
    dist = distance_correlation_by_layer(representations)
    pca = pca_summary_by_layer(representations)
    latent = latent_reconstruction_by_layer(
        representations,
        metadata,
        target_columns=["arc_parameter"],
    )
    final_layer = "logits"
    row = {
        "experiment": experiment["name"],
        "regime": experiment["regime"],
        "scheme": experiment["scheme"],
        "scheme_label": experiment["scheme_label"],
        "n_classes": experiment["n_classes"],
        "actual_classes": len(np.unique(labels)),
        "accuracy": result["accuracy"],
    }
    row.update(knn[knn["layer"] == final_layer].iloc[0].add_prefix("knn_").to_dict())
    row.update(dist[dist["layer"] == final_layer].iloc[0].add_prefix("dist_").to_dict())
    row.update(pca[pca["layer"] == final_layer].iloc[0].add_prefix("pca_").to_dict())
    arc_r2 = latent[
        (latent["layer"] == final_layer) & (latent["latent_variable"] == "arc_parameter")
    ]["r2"]
    row["arc_parameter_r2"] = arc_r2.iloc[0] if len(arc_r2) else np.nan
    return row


def run_seed_experiments(config, seed_rows_path=None):
    rows = _load_seed_rows(seed_rows_path, config)
    completed = {
        (row["regime"], row["scheme"], int(row["n_classes"]), int(row["seed"]))
        for row in rows
    }

    seeds = [config.seed] if config.quick else list(range(config.seed, config.seed + 5))

    for regime, kwargs in MANIFOLDS:
        for scheme in ["intrinsic_arc", "observed_x", "observed_y"]:
            for n_classes in [2, 8, 32]:
                for seed in seeds:
                    key = (regime, scheme, n_classes, seed)
                    if key in completed:
                        continue

                    name = f"{regime}_{scheme}_{n_classes}_seed_{seed}"
                    print(f"Running {name}")
                    world = build_world(
                        regime=regime,
                        regime_kwargs=kwargs,
                        seed=seed,
                        n_samples=config.n_samples,
                    )
                    if scheme == "intrinsic_arc":
                        labels = make_target_by_scheme(world, "arc_bins", n_classes)
                    elif scheme == "observed_x":
                        labels = make_target_by_scheme(world, "x_bins", n_classes)
                    else:
                        labels = make_target_by_scheme(world, "y_bins", n_classes)

                    model = MLPRepresentationModel(
                        input_dim=2,
                        output_dim=len(np.unique(labels)),
                        hidden_dims=[64, 64, 32],
                        epochs=config.epochs,
                        random_state=seed,
                        verbose=False,
                    )
                    result = run_representation_experiment(world, labels, model)
                    representations = result["representations"]
                    logits_knn = knn_preservation_by_layer(representations, k=10).query(
                        "layer == 'logits'"
                    ).iloc[0]
                    logits_dist = distance_correlation_by_layer(representations).query(
                        "layer == 'logits'"
                    ).iloc[0]
                    logits_pca = pca_summary_by_layer(representations).query(
                        "layer == 'logits'"
                    ).iloc[0]
                    row = {
                        "seed": seed,
                        "regime": regime,
                        "scheme": scheme,
                        "n_classes": n_classes,
                        "accuracy": result["accuracy"],
                        "knn": logits_knn["knn_preservation"],
                        "distance": logits_dist["distance_correlation"],
                        "first_pc": logits_pca["first_pc_variance"],
                        "epochs": config.epochs,
                        "n_samples": config.n_samples,
                        "base_seed": config.seed,
                        "quick": config.quick,
                    }
                    rows.append(row)
                    completed.add(key)
                    _save_seed_rows(rows, seed_rows_path)

    seed_results = pd.DataFrame(rows)
    return (
        seed_results.groupby(["regime", "scheme", "n_classes"])
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            knn_mean=("knn", "mean"),
            knn_std=("knn", "std"),
            distance_mean=("distance", "mean"),
            distance_std=("distance", "std"),
            first_pc_mean=("first_pc", "mean"),
            first_pc_std=("first_pc", "std"),
        )
        .reset_index()
    )


def _load_seed_rows(seed_rows_path, config):
    if seed_rows_path is None or not seed_rows_path.exists():
        return []
    rows = pd.read_csv(seed_rows_path)
    required = {"epochs", "n_samples", "base_seed", "quick"}
    if not required.issubset(rows.columns):
        return []
    rows = rows[
        (rows["epochs"] == config.epochs)
        & (rows["n_samples"] == config.n_samples)
        & (rows["base_seed"] == config.seed)
        & (rows["quick"].astype(bool) == config.quick)
    ]
    return rows.to_dict("records")


def _save_seed_rows(rows, seed_rows_path):
    if seed_rows_path is not None:
        pd.DataFrame(rows).to_csv(seed_rows_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["all", "arc", "controlled", "seed"],
        default="all",
        help="Run all experiments or a single stage.",
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
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "Base random seed. Arc and controlled stages use this seed; "
            "the seed stage uses seed..seed+4, or only seed with --quick."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a smaller fast run: fewer epochs, fewer samples, and one seed for the seed stage.",
    )
    return parser.parse_args()


def resolve_config(args):
    epochs = args.epochs if args.epochs is not None else (QUICK_EPOCHS if args.quick else DEFAULT_EPOCHS)
    n_samples = (
        args.n_samples
        if args.n_samples is not None
        else (QUICK_N_SAMPLES if args.quick else DEFAULT_N_SAMPLES)
    )
    if epochs <= 0:
        raise ValueError(f"--epochs must be greater than 0, got {epochs}")
    if n_samples <= 1:
        raise ValueError(f"--n-samples must be greater than 1, got {n_samples}")
    return argparse.Namespace(
        epochs=epochs,
        n_samples=n_samples,
        seed=args.seed,
        quick=args.quick,
    )


def output_dir_for_config(config):
    if (
        config.epochs == DEFAULT_EPOCHS
        and config.n_samples == DEFAULT_N_SAMPLES
        and config.seed == DEFAULT_SEED
        and not config.quick
    ):
        return OUTPUT_DIR

    label = f"epochs{config.epochs}_samples{config.n_samples}_seed{config.seed}"
    if config.quick:
        label += "_quick"
    return OUTPUT_DIR / "runs" / label


def main():
    args = parse_args()
    config = resolve_config(args)
    output_dir = output_dir_for_config(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in {"all", "arc"}:
        _, arc_comparison = run_arc_complexity_experiments(config)
        arc_comparison.to_csv(output_dir / "arc_complexity_comparison.csv", index=False)
        fig, _ = plot_distance_correlation_by_complexity(arc_comparison)
        fig.savefig(output_dir / "distance_correlation_by_complexity.png", dpi=200)

    if args.stage in {"all", "controlled"}:
        controlled_comparison = run_controlled_experiments(config)
        controlled_comparison.to_csv(output_dir / "controlled_comparison.csv", index=False)

    if args.stage in {"all", "seed"}:
        seed_summary = run_seed_experiments(config, output_dir / "seed_results.csv")
        seed_summary.to_csv(output_dir / "seed_summary.csv", index=False)
        fig, _ = plot_seed_summary_for_circle(seed_summary)
        fig.savefig(output_dir / "circle_seed_summary.png", dpi=200)

    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()

