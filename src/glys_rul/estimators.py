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
        standardise: str = "feature",
        augment_rounds: int = 3,
    ) -> None:
        self.factory = factory
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.target_scale = target_scale
        self.loss = loss
        self.augment = augment
        self.seed = seed
        self.standardise = standardise
        self.augment_rounds = augment_rounds

    def fit(self, features: np.ndarray, target: np.ndarray) -> KerasRegressor:
        import keras

        features = np.asarray(features, dtype=float)
        target = np.asarray(target, dtype=float)

        flat = features.reshape(len(features), -1)
        if self.standardise == "none":
            # Temperature maps already arrive scaled to [0, 1]. Standardising them
            # per pixel across a handful of images divides by the near-zero spread
            # of background pixels and blows the inputs up by orders of magnitude.
            self.feature_mean_ = np.zeros(flat.shape[1])
            self.feature_std_ = np.ones(flat.shape[1])
        else:
            self.feature_mean_ = flat.mean(axis=0)
            self.feature_std_ = flat.std(axis=0)
            # A constant column has zero spread; dividing by it would yield NaN.
            self.feature_std_[self.feature_std_ == 0] = 1.0

        x = self._standardise(features)
        scaled_target = target / self.target_scale

        if self.augment is not None and self.augment_rounds > 0:
            # Augmentation must *extend* the training set, not replace it. Handing
            # back the same count would mean the model never sees a clean example.
            rng = np.random.default_rng(self.seed)
            rounds = [self.augment(x, rng) for _ in range(self.augment_rounds)]
            x = np.concatenate([x, *rounds])
            scaled_target = np.tile(scaled_target, 1 + self.augment_rounds)

        self.model_ = self.factory()
        self.model_.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss=self.loss
        )
        self.model_.fit(
            x,
            scaled_target,
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
