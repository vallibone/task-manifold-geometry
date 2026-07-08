"""Classification target construction for public manifold experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd


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


def make_target_by_scheme(world, scheme, n_classes=2, angle=0.0):
    values = world["observed"]
    metadata = world["metadata_df"]
    x = values[:, 0]
    y = values[:, 1]

    if scheme == "arc_bins":
        return bin_continuous_balanced(metadata["arc_parameter"].to_numpy(), n_classes)
    if scheme == "arc_bins_minmax":
        return bin_continuous(metadata["arc_parameter"].to_numpy(), n_classes)
    if scheme == "x_bins":
        return bin_continuous_balanced(x, n_classes)
    if scheme == "y_bins":
        return bin_continuous_balanced(y, n_classes)
    if scheme == "radial_bins":
        radius = np.linalg.norm(values[:, :2], axis=1)
        return bin_continuous_balanced(radius, n_classes)
    if scheme == "quadrant":
        x_sign = (x > np.median(x)).astype(int)
        y_sign = (y > np.median(y)).astype(int)
        return x_sign + 2 * y_sign
    if scheme == "halfspace_angle":
        direction = np.array([np.cos(angle), np.sin(angle)])
        projection = values[:, :2] @ direction
        return (projection > np.median(projection)).astype(int)

    raise ValueError(f"Unknown labelling scheme: {scheme}")
