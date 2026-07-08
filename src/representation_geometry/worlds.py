"""Minimal standalone manifold generators used across the trilogy."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_world(
    regime: str,
    seed: int = 42,
    n_samples: int = 3000,
    noise_std: float = 0.02,
    regime_kwargs: dict | None = None,
) -> dict:
    regime_kwargs = regime_kwargs or {}
    if regime == "circle":
        return make_circle(seed, n_samples, noise_std, **regime_kwargs)
    if regime == "spiral":
        return make_spiral(seed, n_samples, noise_std, **regime_kwargs)
    if regime == "sinewave":
        return make_sinewave(seed, n_samples, noise_std, **regime_kwargs)
    raise ValueError(f"Unknown public manifold: {regime!r}")


def make_circle(seed: int, n_samples: int = 3000, noise_std: float = 0.02, radius: float = 1.0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
    manifold = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
    observed = _add_noise(rng, manifold, noise_std)
    metadata_df = pd.DataFrame(
        {
            "arc_parameter": theta,
            "theta": theta,
            "angle": theta % (2.0 * np.pi),
            "radius": np.full(n_samples, radius, dtype=float),
            "distance_along_manifold": radius * theta,
        }
    )
    return _world_dict("circle", theta[:, None], manifold, observed, metadata_df)


def make_spiral(seed: int, n_samples: int = 3000, noise_std: float = 0.02, scale: float = 1.0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.2, 4.0 * np.pi, size=n_samples)
    manifold = np.column_stack((scale * t * np.cos(t), scale * t * np.sin(t)))
    observed = _add_noise(rng, manifold, noise_std)
    metadata_df = pd.DataFrame(
        {
            "arc_parameter": t,
            "t": t,
            "angle": t % (2.0 * np.pi),
            "radius": scale * t,
            "distance_along_manifold": t,
        }
    )
    return _world_dict("spiral", t[:, None], manifold, observed, metadata_df)


def make_sinewave(
    seed: int,
    n_samples: int = 3000,
    noise_std: float = 0.02,
    amplitude: float = 1.0,
    frequency: float = 2.0,
    phase: float = 0.0,
):
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1.0, 1.0, size=n_samples)
    wave_phase = 2.0 * np.pi * frequency * z + phase
    manifold = np.column_stack((z, amplitude * np.sin(wave_phase)))
    observed = _add_noise(rng, manifold, noise_std)
    metadata_df = pd.DataFrame(
        {
            "arc_parameter": z,
            "x": z,
            "phase": wave_phase,
            "phase_wrapped": wave_phase % (2.0 * np.pi),
            "amplitude": np.full(n_samples, amplitude, dtype=float),
            "frequency": np.full(n_samples, frequency, dtype=float),
            "distance_along_manifold": z,
        }
    )
    return _world_dict("sinewave", z[:, None], manifold, observed, metadata_df)


def _world_dict(regime_name, latent, manifold, observed, metadata_df):
    metadata = {column: metadata_df[column].to_numpy() for column in metadata_df.columns}
    return {
        "regime_name": regime_name,
        "latent": latent,
        "manifold": manifold,
        "embedded": manifold.copy(),
        "observed": observed,
        "metadata": metadata,
        "metadata_df": metadata_df,
    }


def _add_noise(rng: np.random.Generator, values: np.ndarray, noise_std: float):
    if noise_std < 0:
        raise ValueError(f"noise_std cannot be less than 0, got {noise_std}")
    if noise_std == 0.0:
        return values.copy()
    return values + rng.normal(loc=0.0, scale=noise_std, size=values.shape)
