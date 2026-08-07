import numpy as np
import pytest

from glys_rul.baselines import (
    BASELINES,
    IsotonicOnTotal,
    LinearOnTotal,
    MeanPredictor,
    NearestNeighbourOnTotal,
)


@pytest.fixture
def ladder_data():
    """Total temperature falls monotonically as remaining life rises."""
    totals = np.array([[3215.0], [2679.0], [2312.0], [1485.0], [658.0], [1.0]])
    rul = np.array([4.0, 25.0, 49.0, 74.5, 80.0, 100.0])
    return totals, rul


def test_mean_predictor_returns_the_training_mean(ladder_data):
    features, target = ladder_data

    model = MeanPredictor().fit(features, target)

    assert np.allclose(model.predict(features), target.mean())


def test_mean_predictor_ignores_its_input(ladder_data):
    features, target = ladder_data

    model = MeanPredictor().fit(features, target)

    assert np.allclose(model.predict(np.zeros((3, 1))), target.mean())


def test_nearest_neighbour_reproduces_training_labels(ladder_data):
    features, target = ladder_data

    model = NearestNeighbourOnTotal().fit(features, target)

    assert np.allclose(model.predict(features), target)


def test_nearest_neighbour_snaps_to_the_closest_training_point(ladder_data):
    features, target = ladder_data

    model = NearestNeighbourOnTotal().fit(features, target)

    assert model.predict(np.array([[3200.0]]))[0] == pytest.approx(4.0)


def test_linear_recovers_an_exactly_linear_relationship():
    features = np.array([[0.0], [100.0], [200.0], [300.0]])
    target = np.array([10.0, 20.0, 30.0, 40.0])

    model = LinearOnTotal().fit(features, target)

    assert np.allclose(model.predict(np.array([[150.0]])), 25.0)


def test_isotonic_respects_monotonicity(ladder_data):
    features, target = ladder_data

    model = IsotonicOnTotal().fit(features, target)
    predictions = model.predict(np.array([[3215.0], [2312.0], [1.0]]))

    assert predictions[0] < predictions[1] < predictions[2]


def test_isotonic_fits_a_monotone_relationship_exactly(ladder_data):
    features, target = ladder_data

    model = IsotonicOnTotal().fit(features, target)

    assert np.allclose(model.predict(features), target, atol=1e-6)


def test_isotonic_interpolates_between_training_points(ladder_data):
    features, target = ladder_data

    model = IsotonicOnTotal().fit(features, target)
    middle = model.predict(np.array([[2500.0]]))

    assert 25.0 < middle[0] < 49.0


def test_isotonic_clips_outside_the_training_range(ladder_data):
    """Extrapolation must stay inside the observed label range, not run off."""
    features, target = ladder_data

    model = IsotonicOnTotal().fit(features, target)
    beyond = model.predict(np.array([[9999.0], [-9999.0]]))

    assert target.min() <= beyond[0] <= target.max()
    assert target.min() <= beyond[1] <= target.max()


def test_baselines_ignore_columns_beyond_the_first(ladder_data):
    """Every rung consumes total_c only, so the comparison isolates model family."""
    features, target = ladder_data
    padded = np.hstack([features, np.random.default_rng(0).random((len(features), 3))])

    for name, factory in BASELINES.items():
        plain = factory().fit(features, target).predict(features)
        extra = factory().fit(padded, target).predict(padded)
        assert np.allclose(plain, extra), f"{name} used a column beyond the first"


def test_every_registered_baseline_fits_and_predicts(ladder_data):
    features, target = ladder_data

    for name, factory in BASELINES.items():
        model = factory().fit(features, target)
        predictions = model.predict(features)
        assert predictions.shape == target.shape, f"{name} returned the wrong shape"
        assert np.isfinite(predictions).all(), f"{name} produced non-finite predictions"


def test_ladder_is_registered_in_increasing_capability_order():
    assert list(BASELINES) == ["mean", "nearest_neighbour", "linear", "isotonic"]
