"""The baseline ladder.

Each rung answers one objection to the neural results: what does no skill
look like, is the task mere table lookup, does one parameter suffice, and
does exploiting the monotone physics close the gap.

All baselines consume the single feature `total_c` (column 0) so that the
comparison isolates model family rather than input representation.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression


class MeanPredictor:
    """Predicts the training mean regardless of input."""

    def fit(self, features: np.ndarray, target: np.ndarray) -> MeanPredictor:
        self.value_ = float(np.mean(target))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(len(features), self.value_)


class NearestNeighbourOnTotal:
    """Copies the label of the nearest training sample by total temperature."""

    def fit(self, features: np.ndarray, target: np.ndarray) -> NearestNeighbourOnTotal:
        self.totals_ = np.asarray(features)[:, 0].astype(float)
        self.target_ = np.asarray(target, dtype=float)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        totals = np.asarray(features)[:, 0].astype(float)
        nearest = np.abs(totals[:, None] - self.totals_[None, :]).argmin(axis=1)
        return self.target_[nearest]


class LinearOnTotal:
    """Ordinary least squares on total temperature."""

    def fit(self, features: np.ndarray, target: np.ndarray) -> LinearOnTotal:
        self.model_ = LinearRegression().fit(np.asarray(features)[:, :1], target)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model_.predict(np.asarray(features)[:, :1])


class IsotonicOnTotal:
    """Monotone-decreasing fit of remaining life against total temperature.

    Encodes the domain fact that a hotter engine never has more life left.
    """

    def fit(self, features: np.ndarray, target: np.ndarray) -> IsotonicOnTotal:
        self.model_ = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(
            np.asarray(features)[:, 0].astype(float), target
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model_.predict(np.asarray(features)[:, 0].astype(float))


#: Ordered ladder, cheapest and least capable first.
BASELINES = {
    "mean": MeanPredictor,
    "nearest_neighbour": NearestNeighbourOnTotal,
    "linear": LinearOnTotal,
    "isotonic": IsotonicOnTotal,
}
