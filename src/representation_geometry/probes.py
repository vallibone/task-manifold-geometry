"""Representation geometry probes used by the public experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def knn_preservation(reference, candidate, k=10):
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    _validate_same_rows(reference, candidate)
    if reference.shape[0] <= 1:
        return np.nan

    k_eff = min(k, reference.shape[0] - 1)
    neighbors_reference = _nearest_neighbor_indices(reference, k_eff)
    neighbors_candidate = _nearest_neighbor_indices(candidate, k_eff)
    overlaps = [
        len(set(row_reference).intersection(row_candidate)) / k_eff
        for row_reference, row_candidate in zip(neighbors_reference, neighbors_candidate)
    ]
    return float(np.mean(overlaps))


def knn_preservation_by_layer(representations, reference_layer="input", k=10):
    reference = representations[reference_layer]
    records = []
    for layer, values in representations.items():
        records.append(
            {
                "layer": layer,
                "reference_layer": reference_layer,
                "k": k,
                "knn_preservation": knn_preservation(reference, values, k=k),
            }
        )
    return pd.DataFrame(records)


def arc_neighbour_indices(arc_parameter, k=10, periodic=False, period=None):
    arc = np.asarray(arc_parameter).reshape(-1)
    n = len(arc)
    if n <= 1:
        return np.empty((n, 0), dtype=int)

    k_eff = min(k, n - 1)
    diff = np.abs(arc[:, None] - arc[None, :])
    if periodic:
        if period is None:
            period = arc.max() - arc.min()
        diff = np.minimum(diff, period - diff)
    np.fill_diagonal(diff, np.inf)
    return np.argsort(diff, axis=1)[:, :k_eff]


def neighbour_overlap_from_indices(reference_indices, candidate_values, k=10):
    candidate_values = np.asarray(candidate_values)
    if candidate_values.ndim != 2:
        raise ValueError("candidate_values must be a 2D array.")
    if reference_indices.shape[0] != candidate_values.shape[0]:
        raise ValueError("reference_indices and candidate_values must have same number of rows.")

    k_eff = min(k, candidate_values.shape[0] - 1)
    candidate_indices = _nearest_neighbor_indices(candidate_values, k_eff)
    overlaps = [
        len(set(reference).intersection(candidate)) / k_eff
        for reference, candidate in zip(reference_indices, candidate_indices)
    ]
    return float(np.mean(overlaps))


def geodesic_knn_preservation_by_layer(
    representations,
    arc_parameter,
    k=10,
    periodic=False,
    period=None,
):
    reference_indices = arc_neighbour_indices(
        arc_parameter,
        k=k,
        periodic=periodic,
        period=period,
    )
    records = []
    for layer, values in representations.items():
        records.append(
            {
                "layer": layer,
                "reference": "arc_parameter",
                "k": k,
                "geodesic_knn_preservation": neighbour_overlap_from_indices(
                    reference_indices,
                    values,
                    k=k,
                ),
            }
        )
    return pd.DataFrame(records)


def distance_correlation(reference, candidate, sample_size=1000, random_state=42):
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    _validate_same_rows(reference, candidate)
    if reference.shape[0] <= 1:
        return np.nan

    rng = np.random.default_rng(random_state)
    n_samples = min(sample_size, reference.shape[0])
    indices = rng.choice(reference.shape[0], size=n_samples, replace=False)
    distances_reference = pairwise_distances(reference[indices], metric="euclidean")
    distances_candidate = pairwise_distances(candidate[indices], metric="euclidean")
    upper = np.triu_indices(n_samples, k=1)
    flat_reference = distances_reference[upper]
    flat_candidate = distances_candidate[upper]

    if np.std(flat_reference) == 0 or np.std(flat_candidate) == 0:
        return np.nan
    return float(np.corrcoef(flat_reference, flat_candidate)[0, 1])


def distance_correlation_by_layer(
    representations,
    reference_layer="input",
    sample_size=1000,
    random_state=42,
):
    reference = representations[reference_layer]
    records = []
    for layer, values in representations.items():
        records.append(
            {
                "layer": layer,
                "reference_layer": reference_layer,
                "distance_correlation": distance_correlation(
                    reference,
                    values,
                    sample_size=sample_size,
                    random_state=random_state,
                ),
            }
        )
    return pd.DataFrame(records)


def pca_spectrum(values, max_components=None):
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError("values must be a 2D array.")

    max_possible = min(values.shape)
    n_components = max_possible if max_components is None else min(max_components, max_possible)
    if n_components < 1:
        return pd.DataFrame(
            columns=["component", "explained_variance_ratio", "cumulative_variance"]
        )

    pca = PCA(n_components=n_components)
    pca.fit(values)
    explained = pca.explained_variance_ratio_
    return pd.DataFrame(
        {
            "component": np.arange(1, len(explained) + 1),
            "explained_variance_ratio": explained,
            "cumulative_variance": np.cumsum(explained),
        }
    )


def pca_summary_by_layer(representations):
    records = []
    for layer, values in representations.items():
        values = np.asarray(values)
        spectrum = pca_spectrum(values)
        explained = spectrum["explained_variance_ratio"].to_numpy()
        cumulative = spectrum["cumulative_variance"].to_numpy()
        records.append(
            {
                "layer": layer,
                "n_features": values.shape[1] if values.ndim == 2 else np.nan,
                "components_90": _components_for_threshold(cumulative, 0.90),
                "components_95": _components_for_threshold(cumulative, 0.95),
                "components_99": _components_for_threshold(cumulative, 0.99),
                "first_pc_variance": _sum_first(explained, 1),
                "first_2pc_variance": _sum_first(explained, 2),
                "first_3pc_variance": _sum_first(explained, 3),
            }
        )
    return pd.DataFrame(records)


def latent_reconstruction_score(representation, latent_df, target_columns=None):
    values = np.asarray(representation)
    targets = pd.DataFrame(latent_df)
    columns = list(target_columns) if target_columns is not None else list(targets.columns)
    records = []

    for column in columns:
        if column not in targets.columns or not pd.api.types.is_numeric_dtype(targets[column]):
            continue

        labels = targets[column].to_numpy()
        mask = np.isfinite(labels) & np.all(np.isfinite(values), axis=1)
        if mask.sum() < 3 or np.std(labels[mask]) == 0:
            score = np.nan
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                values[mask],
                labels[mask],
                test_size=0.25,
                random_state=42,
            )
            estimator = make_pipeline(StandardScaler(), Ridge())
            estimator.fit(x_train, y_train)
            score = r2_score(y_test, estimator.predict(x_test))

        records.append(
            {
                "latent_variable": column,
                "r2": float(score) if np.isfinite(score) else np.nan,
            }
        )

    return pd.DataFrame(records, columns=["latent_variable", "r2"])


def latent_reconstruction_by_layer(representations, latent_df, target_columns=None):
    frames = []
    for layer, values in representations.items():
        scores = latent_reconstruction_score(values, latent_df, target_columns=target_columns)
        scores.insert(0, "layer", layer)
        frames.append(scores)

    if not frames:
        return pd.DataFrame(columns=["layer", "latent_variable", "r2"])
    return pd.concat(frames, ignore_index=True)


def probe_logits_row(result, experiment, labels, target_columns=("arc_parameter",)):
    representations = result["representations"]
    metadata = result["world"]["metadata_df"]
    final_layer = "logits"

    knn = knn_preservation_by_layer(representations, k=10)
    dist = distance_correlation_by_layer(representations)
    pca = pca_summary_by_layer(representations)

    row = {
        "experiment": experiment["name"],
        "shape_family": experiment.get("shape_family"),
        "hidden_dims": tuple(experiment.get("hidden_dims", [])),
        "parameter_count": experiment.get("parameter_count"),
        "regime": experiment.get("regime"),
        "scheme": experiment.get("scheme"),
        "n_classes": experiment.get("n_classes"),
        "actual_classes": len(np.unique(labels)),
        "accuracy": result["accuracy"],
    }

    row.update(knn[knn["layer"] == final_layer].iloc[0].add_prefix("knn_").to_dict())
    row.update(dist[dist["layer"] == final_layer].iloc[0].add_prefix("dist_").to_dict())
    row.update(pca[pca["layer"] == final_layer].iloc[0].add_prefix("pca_").to_dict())

    if target_columns:
        latent = latent_reconstruction_by_layer(
            representations,
            metadata,
            target_columns=list(target_columns),
        )
        for column in target_columns:
            value = latent[
                (latent["layer"] == final_layer)
                & (latent["latent_variable"] == column)
            ]["r2"]
            row[f"{column}_r2"] = value.iloc[0] if len(value) else np.nan

    return row


def _nearest_neighbor_indices(values, k):
    neighbors = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    neighbors.fit(values)
    indices = neighbors.kneighbors(values, return_distance=False)
    return indices[:, 1:]


def _validate_same_rows(reference, candidate):
    if reference.ndim != 2 or candidate.ndim != 2:
        raise ValueError("Both representations must be 2D arrays.")
    if reference.shape[0] != candidate.shape[0]:
        raise ValueError(
            "Representations must have the same number of rows, got "
            f"{reference.shape[0]} and {candidate.shape[0]}."
        )


def _components_for_threshold(cumulative, threshold):
    if len(cumulative) == 0:
        return np.nan
    hits = np.flatnonzero(cumulative >= threshold)
    return int(hits[0] + 1) if len(hits) else int(len(cumulative))


def _sum_first(values, n):
    if len(values) == 0:
        return np.nan
    return float(np.sum(values[:n]))
