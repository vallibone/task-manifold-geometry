"""Thin entry point for experiment 3."""

from pathlib import Path

from task_manifold_geometry.experiments import exp3_trajectory


exp3_trajectory.OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


if __name__ == "__main__":
    exp3_trajectory.main()
