import numpy as np
import pytest

from glys_rul.conformal import (
    empirical_coverage,
    interval_table,
    jackknife_plus_interval,
    residual_quantile,
)


def test_interval_contains_the_point_prediction():
    residuals = np.array([1.0, 2.0, 1.5, 3.0, 0.5])

    low, high = jackknife_plus_interval(50.0, residuals, alpha=0.1)

    assert low <= 50.0 <= high


def test_interval_widens_with_larger_residuals():
    narrow = jackknife_plus_interval(50.0, np.array([0.1, 0.2, 0.15]), alpha=0.1)
    wide = jackknife_plus_interval(50.0, np.array([10.0, 20.0, 15.0]), alpha=0.1)

    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_lower_alpha_gives_wider_intervals():
    residuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    strict = jackknife_plus_interval(50.0, residuals, alpha=0.01)
    loose = jackknife_plus_interval(50.0, residuals, alpha=0.5)

    assert (strict[1] - strict[0]) >= (loose[1] - loose[0])


def test_interval_is_symmetric_around_the_prediction():
    residuals = np.array([1.0, 2.0, 3.0])

    low, high = jackknife_plus_interval(50.0, residuals, alpha=0.1)

    assert (50.0 - low) == pytest.approx(high - 50.0)


def test_sign_of_residuals_does_not_matter():
    """Coverage depends on absolute error, so a sign flip must not change the width."""
    positive = jackknife_plus_interval(50.0, np.array([1.0, 2.0, 3.0]), alpha=0.1)
    mixed = jackknife_plus_interval(50.0, np.array([-1.0, 2.0, -3.0]), alpha=0.1)

    assert positive == mixed


def test_no_residuals_gives_an_infinite_interval():
    """With nothing to calibrate on, the honest answer is 'unknown', not a guess."""
    low, high = jackknife_plus_interval(50.0, np.array([]), alpha=0.1)

    assert low == -np.inf and high == np.inf


def test_quantile_uses_the_finite_sample_correction():
    """The (n+1)/n inflation is what makes coverage valid at small n, not asymptotic."""
    residuals = np.arange(1.0, 11.0)

    assert residual_quantile(residuals, alpha=0.1) == pytest.approx(10.0)


def test_empirical_coverage_is_close_to_the_target():
    rng = np.random.default_rng(0)
    calibration = np.abs(rng.normal(0.0, 2.0, size=200))
    fresh = np.abs(rng.normal(0.0, 2.0, size=2000))

    low, high = jackknife_plus_interval(0.0, calibration, alpha=0.1)
    covered = ((fresh >= low) & (fresh <= high)).mean()

    assert covered > 0.85


def test_interval_table_returns_one_row_per_prediction():
    predictions = np.array([10.0, 50.0, 90.0])
    residuals = np.array([1.0, 2.0, 3.0])

    table = interval_table(predictions, residuals, alpha=0.1)

    assert table.shape == (3, 2)
    assert np.all(table[:, 0] <= predictions)
    assert np.all(table[:, 1] >= predictions)


def test_empirical_coverage_counts_labels_inside_their_interval():
    truth = np.array([10.0, 50.0, 90.0])
    intervals = np.array([[9.0, 11.0], [40.0, 60.0], [0.0, 1.0]])

    assert empirical_coverage(truth, intervals) == pytest.approx(2 / 3)
