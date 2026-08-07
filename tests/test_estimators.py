import numpy as np
import pytest

from glys_rul.determinism import configure
from glys_rul.estimators import KerasRegressor
from glys_rul.models import build_mlp

pytestmark = pytest.mark.slow


@pytest.fixture
def monotone_data():
    features = np.array([[3215.0], [2679.0], [2312.0], [1485.0], [658.0], [1.0]])
    target = np.array([4.0, 25.0, 49.0, 74.5, 80.0, 100.0])
    return features, target


def test_predictions_have_one_value_per_sample(monotone_data):
    features, target = monotone_data
    configure(seed=0)
    model = KerasRegressor(lambda: build_mlp(n_features=1), epochs=50)

    predictions = model.fit(features, target).predict(features)

    assert predictions.shape == (len(target),)


def test_target_scaling_returns_predictions_in_hours(monotone_data):
    """A model returning scaled units would look absurdly accurate at first glance."""
    features, target = monotone_data
    configure(seed=0)
    model = KerasRegressor(lambda: build_mlp(n_features=1), epochs=400)

    predictions = model.fit(features, target).predict(features)

    assert 0.0 < predictions.mean() < 200.0


def test_training_reduces_error_below_the_mean_baseline(monotone_data):
    features, target = monotone_data
    configure(seed=0)
    model = KerasRegressor(lambda: build_mlp(n_features=1), epochs=600)

    predictions = model.fit(features, target).predict(features)

    model_mae = np.abs(predictions - target).mean()
    baseline_mae = np.abs(target - target.mean()).mean()
    assert model_mae < baseline_mae


def test_input_standardisation_is_fit_on_training_data_only(monotone_data):
    features, target = monotone_data
    configure(seed=0)
    model = KerasRegressor(lambda: build_mlp(n_features=1), epochs=10).fit(features, target)

    assert model.feature_mean_.shape == (1,)
    assert model.feature_std_.shape == (1,)
    assert np.all(model.feature_std_ > 0)


def test_constant_feature_does_not_produce_nan_predictions():
    """A zero-variance column must not divide by zero during standardisation."""
    features = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0]])
    target = np.array([10.0, 20.0, 30.0, 40.0])
    configure(seed=0)
    model = KerasRegressor(lambda: build_mlp(n_features=2), epochs=20)

    predictions = model.fit(features, target).predict(features)

    assert np.isfinite(predictions).all()


def test_same_seed_gives_identical_predictions(monotone_data):
    features, target = monotone_data

    configure(seed=0)
    first = KerasRegressor(lambda: build_mlp(n_features=1), epochs=50).fit(
        features, target
    ).predict(features)
    configure(seed=0)
    second = KerasRegressor(lambda: build_mlp(n_features=1), epochs=50).fit(
        features, target
    ).predict(features)

    assert np.array_equal(first, second)


def test_augmentation_callback_is_applied_when_supplied():
    """The CNN arm trains on augmented tensors; the hook must actually fire."""
    features = np.zeros((4, 8, 8, 1))
    target = np.array([10.0, 20.0, 30.0, 40.0])
    calls = []

    def spy(batch, rng):
        calls.append(batch.shape)
        return batch

    configure(seed=0)
    KerasRegressor(
        lambda: build_mlp(n_features=64), epochs=2, augment=spy
    ).fit(features.reshape(4, 64), target)

    assert len(calls) == 1


def test_standardisation_uses_training_statistics_for_unseen_data(monotone_data):
    """Predicting on new rows must reuse the fitted statistics, not recompute them."""
    features, target = monotone_data
    configure(seed=0)
    model = KerasRegressor(lambda: build_mlp(n_features=1), epochs=50).fit(features, target)

    single = model.predict(features[:1])
    batch = model.predict(features)

    assert single[0] == pytest.approx(batch[0], abs=1e-5)
