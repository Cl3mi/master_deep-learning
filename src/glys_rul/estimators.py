"""Adapter giving Keras models the same fit/predict surface as the baselines.

Standardisation statistics and target scaling are fitted inside `fit`, so they
are derived from the training fold alone and cannot leak held-out information.
That matters more than usual here: with eleven samples, statistics computed over
the whole dataset would carry a visible imprint of the held-out group.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from . import config


class KerasRegressor:
    """Trains a Keras model on standardised features and scaled targets."""

    def __init__(
        self,
        factory: Callable[[], object],
        epochs: int = config.MLP_EPOCHS,
        learning_rate: float = config.LEARNING_RATE,
        target_scale: float = config.TARGET_SCALE,
        loss: str = "mse",
        augment: Callable[[np.ndarray, np.random.Generator], np.ndarray] | None = None,
        seed: int = 0,
    ) -> None:
        self.factory = factory
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.target_scale = target_scale
        self.loss = loss
        self.augment = augment
        self.seed = seed

    def fit(self, features: np.ndarray, target: np.ndarray) -> KerasRegressor:
        import keras

        features = np.asarray(features, dtype=float)
        target = np.asarray(target, dtype=float)

        flat = features.reshape(len(features), -1)
        self.feature_mean_ = flat.mean(axis=0)
        self.feature_std_ = flat.std(axis=0)
        # A constant column has zero spread; dividing by it would yield NaN.
        self.feature_std_[self.feature_std_ == 0] = 1.0

        x = self._standardise(features)
        if self.augment is not None:
            x = self.augment(x, np.random.default_rng(self.seed))

        self.model_ = self.factory()
        self.model_.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss=self.loss
        )
        self.model_.fit(
            x,
            target / self.target_scale,
            epochs=self.epochs,
            batch_size=len(x),
            verbose=0,
            shuffle=False,
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return predictions in hours, undoing the training-time target scaling."""
        x = self._standardise(np.asarray(features, dtype=float))
        scaled = self.model_.predict(x, verbose=0).ravel()
        return scaled * self.target_scale

    def _standardise(self, features: np.ndarray) -> np.ndarray:
        flat = features.reshape(len(features), -1)
        standardised = (flat - self.feature_mean_) / self.feature_std_
        return standardised.reshape(features.shape)
