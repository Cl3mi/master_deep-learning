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
    features = np.zeros((4, 64))
    target = np.array([10.0, 20.0, 30.0, 40.0])
    calls = []

    def spy(batch, rng):
        calls.append(batch.shape)
        return batch

    configure(seed=0)
    KerasRegressor(
        lambda: build_mlp(n_features=64), epochs=2, augment=spy, augment_rounds=3
    ).fit(features, target)

    assert len(calls) == 3


def test_augmentation_expands_the_training_set_rather_than_replacing_it():
    """Replacing would mean the model never sees a single clean example."""
    features = np.zeros((4, 64))
    target = np.array([10.0, 20.0, 30.0, 40.0])
    seen = {}

    def spy(batch, rng):
        seen["rounds"] = seen.get("rounds", 0) + 1
        return batch + 1.0

    class Recorder:
        def compile(self, **kwargs):
            pass

        def fit(self, x, y, **kwargs):
            seen["n_train"] = len(x)
            seen["n_target"] = len(y)
            seen["has_clean"] = bool((x == 0.0).all(axis=1).any())

        def predict(self, x, **kwargs):
            return np.zeros((len(x), 1))

    configure(seed=0)
    KerasRegressor(
        Recorder, epochs=1, augment=spy, augment_rounds=2, standardise="none"
    ).fit(features, target)

    assert seen["n_train"] == 4 * 3, "expected originals plus two augmented rounds"
    assert seen["n_target"] == seen["n_train"], "targets must be tiled to match"
    assert seen["has_clean"], "the clean originals must remain in the training set"


def test_image_input_is_not_standardised_per_pixel():
    """Per-pixel standardisation on a handful of images explodes.

    A background pixel that is near-constant across the training fold has a
    standard deviation of ~1e-8; dividing by it produces astronomically large
    inputs and the network diverges. Temperature maps already arrive scaled to
    [0, 1], so they must pass through untouched.
    """
    rng = np.random.default_rng(0)
    images = np.full((6, 8, 8, 1), 0.5)
    images += rng.normal(0.0, 1e-9, images.shape)  # near-constant background
    images[:, :4] = np.linspace(0.1, 0.9, 6)[:, None, None, None]  # the actual signal
    target = np.linspace(10.0, 100.0, 6)

    configure(seed=0)
    model = KerasRegressor(
        lambda: build_mlp(n_features=64), epochs=50, standardise="none"
    ).fit(images.reshape(6, 64), target)
    predictions = model.predict(images.reshape(6, 64))

    assert np.isfinite(predictions).all()
    assert np.abs(predictions).max() < 1e4, "predictions exploded"


def test_none_standardisation_leaves_inputs_untouched():
    features = np.array([[0.25], [0.5], [0.75]])
    target = np.array([10.0, 20.0, 30.0])

    configure(seed=0)
    model = KerasRegressor(
        lambda: build_mlp(n_features=1), epochs=5, standardise="none"
    ).fit(features, target)

    assert np.allclose(model.feature_mean_, 0.0)
    assert np.allclose(model.feature_std_, 1.0)


def test_standardisation_uses_training_statistics_for_unseen_data(monotone_data):
    """Predicting on new rows must reuse the fitted statistics, not recompute them."""
    features, target = monotone_data
    configure(seed=0)
    model = KerasRegressor(lambda: build_mlp(n_features=1), epochs=50).fit(features, target)

    single = model.predict(features[:1])
    batch = model.predict(features)

    assert single[0] == pytest.approx(batch[0], abs=1e-5)
