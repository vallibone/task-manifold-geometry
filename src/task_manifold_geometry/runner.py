"""Experiment runner."""

from __future__ import annotations

import numpy as np


def run_representation_experiment(world, labels, model):
    values = world["observed"]
    model.fit(values, labels)
    predictions = model.predict(values)
    representations = model.representations(values)
    accuracy = float(np.mean(predictions == np.asarray(labels)))
    return {
        "world": world,
        "y": labels,
        "model": model,
        "predictions": predictions,
        "accuracy": accuracy,
        "representations": representations,
        "history": model.history_,
    }
