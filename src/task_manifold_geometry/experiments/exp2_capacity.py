"""Run the fixed-parameter capacity allocation experiment end to end."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from task_manifold_geometry.architecture import scale_template_to_budget
from task_manifold_geometry.runner import run_representation_experiment
from task_manifold_geometry.worlds import build_world
from task_manifold_geometry.models import MLPRepresentationModel
from task_manifold_geometry.plotting import plot_capacity_allocation_grid
from task_manifold_geometry.probes import probe_logits_row
from task_manifold_geometry.targets import make_target_by_scheme


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

WORLD = "spiral"
REGIME_KWARGS = {"scale": 1.0}
N_CLASSES = 32
TARGET_SCHEME = "arc_bins"
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
TOLERANCE = 0.05

DEFAULT_EPOCHS = 100
DEFAULT_N_SAMPLES = 3000
DEFAULT_NOISE_STD = 0.02
QUICK_EPOCHS = 5
QUICK_N_SAMPLES = 500

ALLOCATION_TEMPLATES = {
    "front_loaded": [4, 2, 1],
    "uniform": [1, 1, 1],
    "back_loaded": [1, 2, 4],
}


def run_single_budget(config, target_params=1000, seed_values=range(1, 6)):
    return run_capacity_experiments(
        config=config,
        target_params_list=[target_params],
        allocation_templates=ALLOCATION_TEMPLATES,
        seed_values=seed_values,
        results_path=output_dir_for_config(config) / "single_budget_1000_results.csv",
        failed_path=output_dir_for_config(config) / "single_budget_1000_failed_architectures.csv",
    )


def run_budget_sweep(config, seed_values=range(1, 6)):
    return run_capacity_experiments(
        config=config,
        target_params_list=[750, 1000, 2000, 5000, 10000],
        allocation_templates=ALLOCATION_TEMPLATES,
        seed_values=seed_values,
        results_path=output_dir_for_config(config) / "budget_sweep_results.csv",
        failed_path=output_dir_for_config(config) / "budget_sweep_failed_architectures.csv",
    )


def run_final_seed_sweep(config, seed_values=range(1, 11)):
    return run_capacity_experiments(
        config=config,
        target_params_list=[750],
        allocation_templates={
            "back_loaded": ALLOCATION_TEMPLATES["back_loaded"],
            "uniform": ALLOCATION_TEMPLATES["uniform"],
        },
        seed_values=seed_values,
        results_path=output_dir_for_config(config) / "seed_750_back_uniform_results.csv",
        failed_path=output_dir_for_config(config) / "seed_750_back_uniform_failed_architectures.csv",
    )


def run_capacity_experiments(
    config,
    target_params_list,
    allocation_templates,
    seed_values,
    results_path,
    failed_path,
):
    rows = _load_matching_rows(results_path, config)
    failed_rows = _load_matching_rows(failed_path, config)
    completed = {
        (int(row["target_params"]), row["allocation"], int(row["seed"]))
        for row in rows
    }
    failed_completed = {
        (int(row["target_params"]), row["allocation"], int(row["seed"]))
        for row in failed_rows
    }

    for target_params in target_params_list:
        for seed in seed_values:
            world = build_world(
                regime=WORLD,
                regime_kwargs=REGIME_KWARGS,
                seed=seed,
                n_samples=config.n_samples,
                noise_std=config.noise_std,
            )
            labels = make_target_by_scheme(
                world,
                scheme=TARGET_SCHEME,
                n_classes=N_CLASSES,
            )
            output_dim = len(np.unique(labels))

            for allocation, template in allocation_templates.items():
                key = (target_params, allocation, seed)
                if key in completed or key in failed_completed:
                    continue

                try:
                    arch = scale_template_to_budget(
                        template=template,
                        target_params=target_params,
                        input_dim=world["observed"].shape[1],
                        output_dim=output_dim,
                        tolerance=TOLERANCE,
                    )
                except ValueError as err:
                    failed_rows.append(
                        _config_metadata(config)
                        | {
                            "target_params": target_params,
                            "allocation": allocation,
                            "template": str(tuple(template)),
                            "seed": seed,
                            "error": str(err),
                        }
                    )
                    failed_completed.add(key)
                    _save_rows(failed_rows, failed_path)
                    print(
                        f"Skipping target_params={target_params}, "
                        f"allocation={allocation}, seed={seed}: {err}"
                    )
                    continue

                experiment = {
                    "name": (
                        f"{WORLD}_{TARGET_SCHEME}_{N_CLASSES}_"
                        f"{allocation}_params_{target_params}_seed_{seed}"
                    ),
                    "world": WORLD,
                    "regime": WORLD,
                    "allocation": allocation,
                    "template": tuple(template),
                    "hidden_dims": tuple(arch["hidden_dims"]),
                    "parameter_count": arch["parameter_count"],
                    "target_params": target_params,
                    "relative_error": arch["relative_error"],
                    "seed": seed,
                    "scheme": TARGET_SCHEME,
                    "n_classes": N_CLASSES,
                    "epochs": config.epochs,
                    "n_samples": config.n_samples,
                }

                print(
                    f"Running {experiment['name']} "
                    f"| hidden_dims={arch['hidden_dims']} "
                    f"| params={arch['parameter_count']}"
                )

                model = MLPRepresentationModel(
                    input_dim=world["observed"].shape[1],
                    output_dim=output_dim,
                    hidden_dims=arch["hidden_dims"],
                    epochs=config.epochs,
                    learning_rate=LEARNING_RATE,
                    batch_size=BATCH_SIZE,
                    random_state=seed,
                    verbose=False,
                )
                result = run_representation_experiment(world, labels, model)
                row = probe_logits_row(
                    result=result,
                    experiment=experiment,
                    labels=labels,
                    target_columns=("arc_parameter",),
                )
                row.update(experiment)
                row.update(_config_metadata(config))
                row["template"] = str(tuple(template))
                row["hidden_dims"] = str(tuple(arch["hidden_dims"]))

                rows.append(row)
                completed.add(key)
                _save_rows(rows, results_path)

    results = pd.DataFrame(rows)
    failed = pd.DataFrame(failed_rows)
    _save_rows(rows, results_path)
    _save_rows(failed_rows, failed_path)
    return results, failed


def summarize_capacity_results(results):
    return (
        results.groupby(
            ["target_params", "allocation", "hidden_dims", "parameter_count"],
            dropna=False,
        )
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            knn_mean=("knn_knn_preservation", "mean"),
            knn_std=("knn_knn_preservation", "std"),
            distance_mean=("dist_distance_correlation", "mean"),
            distance_std=("dist_distance_correlation", "std"),
            first_pc_mean=("pca_first_pc_variance", "mean"),
            first_pc_std=("pca_first_pc_variance", "std"),
            arc_r2_mean=("arc_parameter_r2", "mean"),
            arc_r2_std=("arc_parameter_r2", "std"),
            n_runs=("seed", "count"),
        )
        .reset_index()
        .sort_values(["target_params", "allocation"])
    )


def _config_metadata(config):
    return {
        "config_epochs": config.epochs,
        "config_n_samples": config.n_samples,
        "config_noise_std": config.noise_std,
        "config_quick": config.quick,
    }


def _load_matching_rows(path, config):
    if not path.exists():
        return []

    rows = pd.read_csv(path)
    required = set(_config_metadata(config))
    if not required.issubset(rows.columns):
        return []

    mask = (
        (rows["config_epochs"] == config.epochs)
        & (rows["config_n_samples"] == config.n_samples)
        & (rows["config_noise_std"] == config.noise_std)
        & (rows["config_quick"].astype(bool) == config.quick)
    )
    return rows[mask].to_dict("records")


def _save_rows(rows, path):
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["all", "single", "budget", "final"],
        default="all",
        help="Run all experiment blocks or a single block.",
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
        help=f"Samples per spiral. Defaults to {DEFAULT_N_SAMPLES}, or {QUICK_N_SAMPLES} with --quick.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=DEFAULT_NOISE_STD,
        help=f"Gaussian observation noise. Defaults to {DEFAULT_NOISE_STD}.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a smaller fast run for smoke testing.",
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

    if args.stage in {"all", "single"}:
        single_results, single_failed = run_single_budget(config)
        single_summary = summarize_capacity_results(single_results)
        single_summary.to_csv(output_dir / "single_budget_1000_summary.csv", index=False)
        if not single_failed.empty:
            single_failed.to_csv(output_dir / "single_budget_1000_failed_architectures.csv", index=False)

    if args.stage in {"all", "budget"}:
        budget_results, budget_failed = run_budget_sweep(config)
        budget_summary = summarize_capacity_results(budget_results)
        budget_summary.to_csv(output_dir / "budget_sweep_summary.csv", index=False)
        if not budget_failed.empty:
            budget_failed.to_csv(output_dir / "budget_sweep_failed_architectures.csv", index=False)
        fig, _ = plot_capacity_allocation_grid(budget_summary)
        fig.savefig(output_dir / "capacity_allocation_vs_geometry.png", dpi=200)

    if args.stage in {"all", "final"}:
        final_results, final_failed = run_final_seed_sweep(config)
        final_summary = summarize_capacity_results(final_results)
        final_summary.to_csv(output_dir / "seed_750_back_uniform_summary.csv", index=False)
        if not final_failed.empty:
            final_failed.to_csv(output_dir / "seed_750_back_uniform_failed_architectures.csv", index=False)

    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()

