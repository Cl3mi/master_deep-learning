"""Export the feature model as plain JSON for the browser demo.

The network has a few hundred parameters, so the page evaluates it with a
hand-written matrix multiply rather than shipping a deep-learning runtime: no
build step, no CDN, and the demo keeps working as a static file.

`forward` is the Python twin of that JavaScript. It exists so a test can prove
the two agree with Keras, which is the only thing standing between the demo and
silently drifting from the reported model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def export_mlp(regressor, path: Path, feature_names: list[str], **extra) -> dict:
    """Serialise weights, standardisation statistics and target scaling."""
    layers = []
    for layer in regressor.model_.layers:
        weights = layer.get_weights()
        if len(weights) != 2:
            continue
        kernel, bias = weights
        layers.append(
            {
                "w": np.asarray(kernel, dtype=float).tolist(),
                "b": np.asarray(bias, dtype=float).tolist(),
                "activation": getattr(layer.activation, "__name__", "linear"),
            }
        )

    payload = {
        "feature_names": list(feature_names),
        "feature_mean": np.asarray(regressor.feature_mean_, dtype=float).tolist(),
        "feature_std": np.asarray(regressor.feature_std_, dtype=float).tolist(),
        "target_scale": float(regressor.target_scale),
        "layers": layers,
        **{key: value for key, value in extra.items()},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def forward(payload: dict, features: np.ndarray) -> float:
    """Reference implementation of the browser's forward pass."""
    x = (np.asarray(features, dtype=float) - np.array(payload["feature_mean"])) / np.array(
        payload["feature_std"]
    )
    for layer in payload["layers"]:
        x = x @ np.array(layer["w"]) + np.array(layer["b"])
        if layer["activation"] == "relu":
            x = np.maximum(x, 0.0)
    return float(x[0] * payload["target_scale"])
