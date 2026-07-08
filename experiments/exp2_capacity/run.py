"""Thin entry point for experiment 2."""

from pathlib import Path

from representation_geometry.experiments import exp2_capacity


exp2_capacity.OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


if __name__ == "__main__":
    exp2_capacity.main()
