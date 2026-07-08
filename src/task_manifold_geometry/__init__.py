"""Public tools for the task-manifold-geometry experiment trilogy."""

from task_manifold_geometry.architecture import count_mlp_parameters, scale_template_to_budget
from task_manifold_geometry.models import MLPRepresentationModel
from task_manifold_geometry.runner import run_representation_experiment
from task_manifold_geometry.worlds import build_world

__all__ = [
    "MLPRepresentationModel",
    "build_world",
    "count_mlp_parameters",
    "run_representation_experiment",
    "scale_template_to_budget",
]
