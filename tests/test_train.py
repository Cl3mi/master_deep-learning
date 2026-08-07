import numpy as np
import pytest

from glys_rul.baselines import BASELINES
from glys_rul.train import cross_validate, cross_validate_seeds


@pytest.fixture
def ladder():
    """Six samples in four content groups; two groups hold a duplicate pair."""
    features = np.array([[3215.0], [3215.0], [2679.0], [2679.0], [1485.0], [1.0]])
    target = np.array([3.0, 5.0, 24.0, 26.0, 74.0, 100.0])
    groups = np.array(["a", "a", "b", "b", "c", "d"])
    return features, target, groups


def test_every_sample_receives_an_out_of_fold_prediction(ladder):
    features, target, groups = ladder

    result = cross_validate(BASELINES["isotonic"], features, target, groups)

    assert result.predictions.shape == target.shape
    assert np.isfinite(result.predictions).all()


def test_one_fold_metric_entry_per_group(ladder):
    features, target, groups = ladder

    result = cross_validate(BASELINES["isotonic"], features, target, groups)

    assert len(result.fold_metrics) == len(set(groups))


def test_mean_baseline_cannot_beat_the_label_spread(ladder):
    features, target, groups = ladder

    result = cross_validate(BASELINES["mean"], features, target, groups)

    assert result.metrics["mae"] > 20.0


def test_isotonic_beats_the_mean_baseline(ladder):
    features, target, groups = ladder

    isotonic = cross_validate(BASELINES["isotonic"], features, target, groups)
    mean = cross_validate(BASELINES["mean"], features, target, groups)

    assert isotonic.metrics["mae"] < mean.metrics["mae"]


def test_cross_validation_is_reproducible(ladder):
    features, target, groups = ladder

    first = cross_validate(BASELINES["isotonic"], features, target, groups)
    second = cross_validate(BASELINES["isotonic"], features, target, groups)

    assert np.array_equal(first.predictions, second.predictions)


def test_skill_is_reported_when_a_baseline_is_supplied(ladder):
    features, target, groups = ladder

    result = cross_validate(BASELINES["isotonic"], features, target, groups, baseline_mae=27.0)

    assert "skill" in result.metrics


def test_cross_validate_never_fits_on_the_group_it_scores(ladder):
    """The held-out group's features must never appear in a fold's training set."""
    features, target, groups = ladder
    observed: list[tuple[set, set]] = []

    class Recorder:
        def fit(self, x, y):
            self.train_totals_ = set(np.asarray(x)[:, 0].tolist())
            self.value_ = float(np.mean(y))
            return self

        def predict(self, x):
            observed.append((self.train_totals_, set(np.asarray(x)[:, 0].tolist())))
            return np.full(len(x), self.value_)

    cross_validate(Recorder, features, target, groups)

    assert len(observed) == len(set(groups))
    for train_totals, test_totals in observed:
        assert not (train_totals & test_totals), "held-out features leaked into training"


def test_a_fresh_model_is_built_for_every_fold(ladder):
    """Reusing one instance would carry the previous fold's fitted state."""
    features, target, groups = ladder
    built = []

    class Counter:
        def __init__(self):
            built.append(self)

        def fit(self, x, y):
            self.value_ = float(np.mean(y))
            return self

        def predict(self, x):
            return np.full(len(x), self.value_)

    cross_validate(Counter, features, target, groups)

    assert len(built) == len(set(groups))
    assert len({id(model) for model in built}) == len(built)


def test_missing_prediction_is_reported_rather_than_silently_scored(ladder):
    """A model returning nothing usable must fail loudly, not produce a metric."""
    features, target, groups = ladder

    class Broken:
        def fit(self, x, y):
            return self

        def predict(self, x):
            return np.full(len(x), np.nan)

    with pytest.raises(RuntimeError, match="without a prediction"):
        cross_validate(Broken, features, target, groups)


def test_seed_sweep_reports_spread_across_seeds(ladder):
    features, target, groups = ladder

    summary = cross_validate_seeds(
        BASELINES["isotonic"], features, target, groups, seeds=(0, 1, 2)
    )

    assert summary["n_seeds"] == 3
    assert "mae_mean" in summary and "mae_std" in summary
    assert summary["mae_std"] == pytest.approx(0.0), "a deterministic model has no seed spread"
