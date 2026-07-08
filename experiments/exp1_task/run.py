"""Thin entry point for experiment 1."""

from pathlib import Path

from task_manifold_geometry.experiments import exp1_task


exp1_task.OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


if __name__ == "__main__":
    exp1_task.main()
