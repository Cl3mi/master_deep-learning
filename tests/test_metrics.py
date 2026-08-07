import numpy as np
import pytest

from glys_rul.metrics import evaluate_predictions, skill_score


def test_perfect_predictions_score_zero_error():
    truth = np.array([10.0, 20.0, 30.0])

    result = evaluate_predictions(truth, truth.copy())

    assert result["mae"] == pytest.approx(0.0)
    assert result["rmse"] == pytest.approx(0.0)
    assert result["r2"] == pytest.approx(1.0)


def test_mae_and_rmse_differ_under_uneven_errors():
    truth = np.array([0.0, 0.0, 0.0])
    predicted = np.array([0.0, 0.0, 3.0])

    result = evaluate_predictions(truth, predicted)

    assert result["mae"] == pytest.approx(1.0)
    assert result["rmse"] == pytest.approx(np.sqrt(3.0))


def test_rmse_never_falls_below_mae():
    rng = np.random.default_rng(0)
    truth = rng.normal(50.0, 20.0, size=50)
    predicted = truth + rng.normal(0.0, 5.0, size=50)

    result = evaluate_predictions(truth, predicted)

    assert result["rmse"] >= result["mae"]


def test_skill_score_is_one_for_perfect_and_zero_for_baseline():
    truth = np.array([10.0, 20.0, 60.0])
    baseline_mae = float(np.abs(truth - truth.mean()).mean())

    assert skill_score(0.0, baseline_mae) == pytest.approx(1.0)
    assert skill_score(baseline_mae, baseline_mae) == pytest.approx(0.0)


def test_skill_score_is_negative_when_worse_than_the_mean():
    assert skill_score(20.0, 10.0) == pytest.approx(-1.0)


def test_skill_score_is_undefined_for_a_degenerate_baseline():
    """A constant-label dataset has no spread to improve on."""
    assert np.isnan(skill_score(1.0, 0.0))


def test_evaluate_includes_skill_score_against_supplied_baseline():
    truth = np.array([10.0, 20.0, 60.0])
    predicted = np.array([10.0, 20.0, 60.0])

    result = evaluate_predictions(truth, predicted, baseline_mae=10.0)

    assert result["skill"] == pytest.approx(1.0)


def test_skill_is_omitted_when_no_baseline_is_supplied():
    result = evaluate_predictions(np.array([1.0, 2.0]), np.array([1.0, 2.0]))

    assert "skill" not in result


def test_r2_is_undefined_when_labels_have_no_variance():
    """R-squared divides by label variance; a constant target makes it meaningless."""
    result = evaluate_predictions(np.array([5.0, 5.0]), np.array([5.0, 5.1]))

    assert np.isnan(result["r2"])
