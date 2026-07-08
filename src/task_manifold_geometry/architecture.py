"""Parameter-count utilities for small MLP architecture sweeps."""

from __future__ import annotations


def count_mlp_parameters(input_dim, hidden_dims, output_dim, include_bias=True):
    dims = [input_dim, *hidden_dims, output_dim]
    total = 0
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        total += in_dim * out_dim
        if include_bias:
            total += out_dim
    return total


def scale_template_to_budget(
    template,
    target_params,
    input_dim=2,
    output_dim=2,
    tolerance=0.05,
    min_scale=1,
    max_scale=512,
    include_bias=True,
):
    """Scale a relative-width template to approximately match a parameter budget."""
    if target_params <= 0:
        raise ValueError("target_params must be positive")
    if len(template) == 0:
        raise ValueError("template must contain at least one hidden layer")
    if any(value <= 0 for value in template):
        raise ValueError("template values must all be positive")

    best = None
    for scale in range(min_scale, max_scale + 1):
        hidden_dims = [max(1, round(scale * ratio)) for ratio in template]
        parameter_count = count_mlp_parameters(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            include_bias=include_bias,
        )
        relative_error = abs(parameter_count - target_params) / target_params
        candidate = {
            "template": list(template),
            "scale": scale,
            "hidden_dims": hidden_dims,
            "parameter_count": parameter_count,
            "target_params": target_params,
            "relative_error": relative_error,
        }
        if best is None or relative_error < best["relative_error"]:
            best = candidate

    if best["relative_error"] > tolerance:
        raise ValueError(
            f"No architecture found within {tolerance:.1%}. "
            f"Best was {best['hidden_dims']} with "
            f"{best['parameter_count']} params "
            f"({best['relative_error']:.2%} error)."
        )

    return best
