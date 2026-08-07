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
from .train import cross_validate, cross_validate_seeds


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
    seed_sweep: dict[str, dict] = {}

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
            # A single run is dominated by initialisation noise at this sample
            # size, so the reportable figure is the spread across seeds.
            seed_sweep[name] = cross_validate_seeds(
                lambda b=builder, s=0: KerasRegressor(b, epochs=config.MLP_EPOCHS, seed=s),
                matrix,
                target,
                groups,
                seeds=config.SEEDS,
                baseline_mae=baseline_mae,
            )

        from .dataset import augment_batch, build_image_tensor
        from .models import build_cnn

        scale_for_images = ColorScale.from_image(scale_path)

        def cnn_factory():
            return KerasRegressor(
                lambda: build_cnn(image_size=config.IMAGE_SIZE),
                epochs=config.CNN_EPOCHS,
                augment=augment_batch,
                augment_rounds=config.CNN_AUGMENT_ROUNDS,
                # Temperature maps are already scaled to [0, 1]; per-pixel
                # standardisation over nine images divides by the near-zero
                # spread of background pixels and diverges.
                standardise="none",
            )

        # White sits nearest the hot end of the colour scale, so an unmasked
        # background reads as roughly 1000 °C. Both variants are evaluated so the
        # cost of that artefact is measured rather than asserted.
        images = build_image_tensor(
            table, scale_for_images, data_dir, config.IMAGE_SIZE, mask_background=True
        )
        configure(seed=0)
        record("cnn", cross_validate(cnn_factory, images, target, groups, baseline_mae))

        unmasked = build_image_tensor(
            table, scale_for_images, data_dir, config.IMAGE_SIZE, mask_background=False
        )
        configure(seed=0)
        record(
            "cnn_unmasked_background",
            cross_validate(cnn_factory, unmasked, target, groups, baseline_mae),
        )

    attribution: dict = {}
    if include_neural:
        _export_demo_model(totals, target, models, predictions, seed_sweep)
        attribution = _explain(features, images, target, output_dir)

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

    # Where the error lives: holding out an endpoint group forces extrapolation
    # beyond every state the fold has seen, while interior groups interpolate.
    best_name = min(models, key=lambda name: models[name]["mae"])
    best_predictions = np.asarray(predictions[best_name], dtype=float)
    absolute_error = np.abs(best_predictions - target)
    endpoint = (target == target.min()) | (target == target.max())
    decomposition = {
        "model": best_name,
        "endpoint_mae": round(float(absolute_error[endpoint].mean()), 6),
        "interior_mae": round(float(absolute_error[~endpoint].mean()), 6),
        "n_endpoint": int(endpoint.sum()),
        "n_interior": int((~endpoint).sum()),
    }

    scale = ColorScale.from_image(scale_path)
    luminance = scale.lut @ np.array([0.299, 0.587, 0.114])
    scale_report = {
        "lut_entries": int(len(scale.temps)),
        "max_roundtrip_error_c": round(float(scale.max_roundtrip_error()), 4),
        "luminance_peak": round(float(luminance.max()), 4),
        "luminance_peak_at_c": round(float(scale.temps[int(luminance.argmax())]), 4),
        "luminance_at_hot_end": round(float(luminance[-1]), 4),
    }

    results = {
        "attribution": attribution,
        "colour_scale": scale_report,
        "error_decomposition": decomposition,
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
        "model_inputs": {
            "feature_mlp": ["total_c"],
            "monotone_mlp": list(modelling_columns),
            "cnn": ["temperature_map_64x64_masked"],
            "cnn_unmasked_background": ["temperature_map_64x64"],
            **{name: ["total_c"] for name in BASELINES},
        },
        "modelling_features": modelling_columns,
        "seed_sweep": seed_sweep,
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


def _explain(features, images, target, output_dir) -> dict:
    """Attribution for both architectures.

    Permutation importance runs on the monotone network, which is the model that
    consumes the three region temperatures separately — the reported model reads
    only their sum, where permuting the single input is uninformative.

    Occlusion runs on the convolutional model and answers a specific question:
    does it attend to the engine at all, or has it merely memorised?
    """
    from .determinism import configure
    from .estimators import KerasRegressor
    from .explain import occlusion_map, permutation_importance
    from .figures import plot_occlusion
    from .models import build_cnn, build_monotone_mlp

    configure(seed=0)
    monotone = KerasRegressor(
        lambda: build_monotone_mlp(n_features=features.shape[1]), epochs=config.MLP_EPOCHS
    ).fit(features, target)
    importance = permutation_importance(
        monotone, features, target, names=PHYSICAL_FEATURES, rng=np.random.default_rng(0)
    )

    configure(seed=0)
    cnn = KerasRegressor(
        lambda: build_cnn(image_size=config.IMAGE_SIZE),
        epochs=config.CNN_EPOCHS,
        standardise="none",
    ).fit(images, target)
    hottest = int(np.argmin(target))
    heatmap = occlusion_map(cnn, images[hottest], patch=16, stride=8)
    plot_occlusion(images[hottest], heatmap, Path(output_dir) / "figures" / "occlusion.png")

    return {
        "permutation_importance": {k: round(float(v), 6) for k, v in importance.items()},
        "occlusion": {
            "model": "cnn",
            "sample": f"{target[hottest]:g}h",
            "max_sensitivity_h": round(float(heatmap.max()), 6),
            "mean_sensitivity_h": round(float(heatmap.mean()), 6),
        },
    }


def _export_demo_model(totals, target, models, predictions, sweep) -> None:
    """Refit the reported model on all data and export it for the browser demo.

    The interval half-width comes from the *out-of-fold* residuals, not from the
    refit's own errors — otherwise the demo would advertise a confidence it has
    not earned.
    """
    from .conformal import residual_quantile
    from .determinism import configure
    from .estimators import KerasRegressor
    from .export import export_mlp
    from .models import build_mlp

    configure(seed=0)
    final = KerasRegressor(lambda: build_mlp(n_features=1), epochs=config.MLP_EPOCHS)
    final.fit(totals, target)

    residuals = np.asarray(predictions["feature_mlp"], dtype=float) - target
    export_mlp(
        final,
        config.WEB_DIR / "model.json",
        ["total_c"],
        model="feature_mlp",
        # Advertise the seed-averaged figure, not the single favourable draw.
        cv_mae=round(float(sweep["feature_mlp"]["mae_mean"]), 6),
        cv_mae_std=round(float(sweep["feature_mlp"]["mae_std"]), 6),
        interval_halfwidth=round(float(residual_quantile(residuals)), 4),
        coverage=int(round((1.0 - config.CONFORMAL_ALPHA) * 100)),
        # The demo flags inputs outside this range: the model has seen six
        # thermal states and anything beyond them is extrapolation, not estimation.
        observed_total_range=[
            round(float(np.min(totals)), 3),
            round(float(np.max(totals)), 3),
        ],
        observed_rul_range=[float(np.min(target)), float(np.max(target))],
    )


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
