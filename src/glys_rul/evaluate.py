"""Pipeline orchestration.

Two output files by design. `results.json` holds metrics only and is committed as
a golden copy that CI diffs on every push; `run_meta.json` holds the volatile
provenance — versions, timestamps, git SHA — that would otherwise make that diff
fail on every run and so make the reproducibility check worthless.
"""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .audit import duplicate_report, error_floors, scan_directory
from .baselines import BASELINES
from .colorscale import ColorScale
from .features import DIAGNOSTIC_FEATURES, PHYSICAL_FEATURES, degenerate_columns, extract
from .io import load_rgb
from .train import cross_validate


def build_feature_table(
    data_dir: Path,
    scale_path: Path,
    min_component_px: int = config.MIN_COMPONENT_PX,
    erode_iterations: int = config.ERODE_ITERATIONS,
) -> pd.DataFrame:
    """Scan the directory and attach extracted features to every sample."""
    samples = scan_directory(data_dir)
    scale = ColorScale.from_image(scale_path)

    rows = [
        extract(
            load_rgb(path),
            scale,
            min_component_px=min_component_px,
            erode_iterations=erode_iterations,
        )
        for path in samples["path"]
    ]
    features = pd.DataFrame(rows)
    return pd.concat([samples.drop(columns=["path"]).reset_index(drop=True), features], axis=1)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _package_versions() -> dict[str, str]:
    from importlib.metadata import version

    versions = {}
    for name in ["numpy", "scipy", "scikit-learn", "pandas", "pillow", "matplotlib"]:
        try:
            versions[name] = version(name)
        except Exception:  # noqa: BLE001 - provenance must never break the run
            versions[name] = "unavailable"
    return versions


def run_evaluation(
    data_dir: Path = config.DATA_DIR,
    scale_path: Path = config.SCALE_IMAGE,
    output_dir: Path = config.REPORTS_DIR,
    include_neural: bool = True,
    min_component_px: int = config.MIN_COMPONENT_PX,
    erode_iterations: int = config.ERODE_ITERATIONS,
) -> dict:
    """Run the full pipeline and write results plus provenance."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = build_feature_table(data_dir, scale_path, min_component_px, erode_iterations)
    floors = error_floors(table)

    modelling_columns = list(PHYSICAL_FEATURES)
    features = table[modelling_columns].to_numpy(dtype=float)
    totals = table[["total_c"]].to_numpy(dtype=float)
    target = table["rul"].to_numpy(dtype=float)
    groups = table["group"].to_numpy()

    baseline_mae = cross_validate(BASELINES["mean"], totals, target, groups).metrics["mae"]

    models: dict[str, dict] = {}
    predictions: dict[str, list[float]] = {}

    def record(name: str, result) -> None:
        models[name] = {key: round(value, 6) for key, value in result.metrics.items()}
        predictions[name] = [round(float(value), 6) for value in result.predictions]

    for name, factory in BASELINES.items():
        record(name, cross_validate(factory, totals, target, groups, baseline_mae=baseline_mae))

    if include_neural:
        from .determinism import configure
        from .estimators import KerasRegressor
        from .models import build_mlp, build_monotone_mlp

        n_physical = len(modelling_columns)
        neural = {
            # The summed feature outperforms the three separate ones at this sample
            # size: it is a physics-motivated reduction that regularises far more
            # effectively than anything the optimiser can learn from eleven samples.
            "feature_mlp": (totals, lambda: build_mlp(n_features=1)),
            "monotone_mlp": (features, lambda: build_monotone_mlp(n_features=n_physical)),
        }
        for name, (matrix, builder) in neural.items():
            configure(seed=0)
            record(
                name,
                cross_validate(
                    lambda b=builder: KerasRegressor(b, epochs=config.MLP_EPOCHS),
                    matrix,
                    target,
                    groups,
                    baseline_mae=baseline_mae,
                ),
            )

        from .dataset import augment_batch, build_image_tensor
        from .models import build_cnn

        images = build_image_tensor(
            table, ColorScale.from_image(scale_path), data_dir, image_size=config.IMAGE_SIZE
        )
        configure(seed=0)
        record(
            "cnn",
            cross_validate(
                lambda: KerasRegressor(
                    lambda: build_cnn(image_size=config.IMAGE_SIZE),
                    epochs=config.CNN_EPOCHS,
                    augment=augment_batch,
                    augment_rounds=config.CNN_AUGMENT_ROUNDS,
                    # Temperature maps are already scaled to [0, 1]; per-pixel
                    # standardisation over nine images divides by the near-zero
                    # spread of background pixels and diverges.
                    standardise="none",
                ),
                images,
                target,
                groups,
                baseline_mae=baseline_mae,
            ),
        )

    # How much of the error is simply lack of data? Retrain on 2..n-1 groups.
    from .controls import shuffled_label_control

    unique_groups = sorted(set(groups.tolist()))
    learning_curve = []
    for size in range(2, len(unique_groups)):
        subset = np.isin(groups, unique_groups[:size])
        if len(set(groups[subset].tolist())) < 2:
            continue
        sub = cross_validate(BASELINES["isotonic"], totals[subset], target[subset], groups[subset])
        learning_curve.append({"groups": size, "mae": round(sub.metrics["mae"], 6)})

    # A model that still scores well on permuted labels is memorising, not learning.
    shuffled = shuffled_label_control(target, groups, np.random.default_rng(0))
    control = cross_validate(BASELINES["isotonic"], totals, shuffled, groups)

    results = {
        "learning_curve": learning_curve,
        "shuffled_label_control": {
            "mae": round(control.metrics["mae"], 6),
            "no_skill_mae": round(baseline_mae, 6),
        },
        "dataset": {
            "n_files": int(len(table)),
            "n_groups": int(table["group"].nunique()),
            "rul_min": float(table["rul"].min()),
            "rul_max": float(table["rul"].max()),
        },
        "floors": {key: round(value, 6) for key, value in floors.items()},
        "excluded_features": degenerate_columns(
            table[list(DIAGNOSTIC_FEATURES)], relative_tolerance=0.05
        ),
        "modelling_features": modelling_columns,
        "models": models,
        "out_of_fold_predictions": predictions,
        "labels": [float(value) for value in target],
    }

    (output_dir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    (output_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "git_sha": _git_sha(),
                "packages": _package_versions(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    duplicate_report(table).to_csv(output_dir / "duplicates.csv", index=False)
    table.to_csv(output_dir / "features.csv", index=False)

    _write_figures(
        output_dir, results, np.asarray(totals).ravel(), target, models, predictions
    )
    return results


def _write_figures(output_dir, results, totals, target, models, predictions) -> None:
    """Render the accuracy figures for the best-scoring model."""
    from .figures import (
        plot_baseline_ladder,
        plot_confusion,
        plot_learning_curve,
        plot_predicted_vs_actual,
        plot_residuals,
    )

    figures_dir = Path(output_dir) / "figures"
    floors = results["floors"]
    best = min(models, key=lambda name: models[name]["mae"])
    best_predictions = np.array(predictions[best], dtype=float)

    curve = results.get("learning_curve") or []
    if len(curve) >= 2:
        plot_learning_curve(
            [entry["groups"] for entry in curve],
            [entry["mae"] for entry in curve],
            floors,
            figures_dir / "learning_curve.png",
        )

    plot_predicted_vs_actual(
        target, best_predictions, floors, figures_dir / "predicted_vs_actual.png"
    )
    plot_residuals(totals, best_predictions - target, floors, figures_dir / "residuals.png")
    plot_confusion(target, best_predictions, config.BIN_WIDTH_H, figures_dir / "confusion.png")
    plot_baseline_ladder(
        list(models),
        [models[name]["mae"] for name in models],
        floors,
        figures_dir / "ladder.png",
    )
