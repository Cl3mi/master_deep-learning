"""The optimisation campaign.

Every trial is written to the ledger, including the ones that error and the ones
that score badly. A record containing only successes cannot distinguish a
thorough search from a lucky one, so the dead ends are the evidence that the
search was real.

Input representation is a search axis, not a fixed choice. It matters more than
any hyperparameter here: at eleven samples the summed temperature is a
physics-motivated dimensionality reduction that regularises far more effectively
than anything the optimiser can learn from three separate features.

The scores here are *optimistic* — configurations are selected by the same
grouped cross-validation that reports them. The honest figure is the one in
`results.json`, produced by a configuration fixed a priori. The gap between the
two is itself a finding about selection bias at this sample size.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from . import config
from .determinism import configure
from .estimators import KerasRegressor
from .models import build_cnn, build_mlp, build_monotone_mlp
from .train import cross_validate

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _dense_space(trial: optuna.Trial) -> dict:
    depth = trial.suggest_int("depth", 1, 3)
    return {
        "representation": trial.suggest_categorical("representation", ["total", "physical"]),
        "hidden": tuple(
            trial.suggest_int(f"units_{index}", 4, 64, log=True) for index in range(depth)
        ),
        "l2": trial.suggest_float("l2", 1e-6, 1e-1, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        "epochs": trial.suggest_int("epochs", 100, 1200, step=100),
        "loss": trial.suggest_categorical("loss", ["mse", "huber"]),
    }


def _cnn_space(trial: optuna.Trial) -> dict:
    return {
        "representation": "image",
        "filters": tuple(
            trial.suggest_categorical(f"filters_{index}", [8, 16, 32])
            for index in range(trial.suggest_int("blocks", 2, 3))
        ),
        "dense_units": trial.suggest_int("dense_units", 8, 64, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "epochs": trial.suggest_int("epochs", 100, 400, step=100),
        "augment_rounds": trial.suggest_int("augment_rounds", 0, 4),
        "loss": trial.suggest_categorical("loss", ["mse", "huber"]),
    }


SEARCH_SPACES = {
    "feature_mlp": _dense_space,
    "monotone_mlp": _dense_space,
    "cnn": _cnn_space,
}


def _build_estimator(family: str, params: dict, n_features: int, seed: int):
    """Return a zero-argument factory producing a fresh estimator."""
    if family == "cnn":
        from .dataset import augment_batch

        return lambda: KerasRegressor(
            lambda: build_cnn(
                image_size=config.IMAGE_SIZE,
                filters=params["filters"],
                dense_units=params["dense_units"],
                dropout=params["dropout"],
            ),
            epochs=params["epochs"],
            learning_rate=params["learning_rate"],
            loss=params["loss"],
            augment=augment_batch,
            augment_rounds=params["augment_rounds"],
            standardise="none",
            seed=seed,
        )

    builder = build_monotone_mlp if family == "monotone_mlp" else build_mlp
    kwargs = {} if family == "monotone_mlp" else {"l2": params["l2"]}
    return lambda: KerasRegressor(
        lambda: builder(n_features=n_features, hidden=params["hidden"], **kwargs),
        epochs=params["epochs"],
        learning_rate=params["learning_rate"],
        loss=params["loss"],
        seed=seed,
    )


def run_campaign(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    n_trials: int = 25,
    output: Path = config.REPORTS_DIR / "experiments.csv",
    families: tuple[str, ...] = ("feature_mlp", "monotone_mlp", "cnn"),
    images: np.ndarray | None = None,
    seed: int = 0,
    max_epochs: int | None = None,
    on_trial=None,
) -> pd.DataFrame:
    """Search every requested family and write the full ledger.

    `features` column 0 must be the summed temperature; the remaining columns are
    the per-region temperatures. `images` supplies the tensor for the CNN family.
    `max_epochs` clips the sampled epoch budget, which keeps the test suite fast
    without narrowing the space the real campaign explores.
    """
    records: list[dict] = []
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def flush() -> pd.DataFrame:
        """Persist the ledger after every trial.

        A campaign runs for tens of minutes; writing only at the end would lose
        the entire record to an interrupt or a crash.
        """
        frame = pd.DataFrame(records)
        frame.to_csv(output, index=False)
        if on_trial is not None:
            on_trial(frame)
        return frame

    def matrix_for(representation: str) -> np.ndarray:
        if representation == "image":
            if images is None:
                raise ValueError("the cnn family needs an image tensor")
            return images
        return features[:, :1] if representation == "total" else features[:, 1:]

    for family in families:
        space = SEARCH_SPACES.get(family)

        def objective(trial: optuna.Trial, family=family, space=space) -> float:
            if space is None:
                # Unknown family: record the failure rather than skipping silently.
                records.append(
                    {
                        "family": family, "params": "{}", "mae": float("nan"),
                        "rmse": float("nan"), "status": "failed",
                        "note": f"no search space registered for {family!r}",
                    }
                )
                raise optuna.TrialPruned

            params = space(trial)
            if max_epochs is not None:
                params["epochs"] = min(params["epochs"], max_epochs)
            try:
                matrix = matrix_for(params["representation"])
                configure(seed=seed)
                result = cross_validate(
                    _build_estimator(family, params, matrix.shape[-1], seed),
                    matrix, target, groups,
                )
                if not np.isfinite(result.metrics["mae"]):
                    raise ValueError("non-finite predictions")
            except Exception as error:  # noqa: BLE001 - failures belong in the ledger
                records.append(
                    {
                        "family": family, "params": json.dumps(params, default=str),
                        "mae": float("nan"), "rmse": float("nan"),
                        "status": "failed", "note": str(error)[:200],
                    }
                )
                flush()
                raise optuna.TrialPruned from error

            records.append(
                {
                    "family": family, "params": json.dumps(params, default=str),
                    "mae": result.metrics["mae"], "rmse": result.metrics["rmse"],
                    "status": "ok", "note": "",
                }
            )
            flush()
            return result.metrics["mae"]

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(objective, n_trials=n_trials, catch=(optuna.TrialPruned,))

    frame = pd.DataFrame(records)
    frame = frame.sort_values(["status", "mae"], ascending=[True, True]).reset_index(drop=True)
    frame.to_csv(output, index=False)
    return frame
