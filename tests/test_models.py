import numpy as np
import pytest

from glys_rul.determinism import configure
from glys_rul.models import build_cnn, build_mlp, build_monotone_mlp

pytestmark = pytest.mark.slow


def test_mlp_outputs_one_linear_value_per_sample():
    configure(seed=0)
    model = build_mlp(n_features=3)

    output = model.predict(np.zeros((5, 3)), verbose=0)

    assert output.shape == (5, 1)


def test_mlp_is_small_enough_for_eleven_samples():
    configure(seed=0)
    model = build_mlp(n_features=3)

    assert model.count_params() < 1000


def test_cnn_accepts_single_channel_temperature_maps():
    configure(seed=0)
    model = build_cnn(image_size=32)

    output = model.predict(np.zeros((2, 32, 32, 1)), verbose=0)

    assert output.shape == (2, 1)


def test_cnn_capacity_is_deliberately_bounded():
    configure(seed=0)
    model = build_cnn(image_size=32)

    assert model.count_params() < 60_000


def test_output_activation_is_linear_for_regression():
    """The brief specifies Dense(1, activation="linear") for the regression head."""
    configure(seed=0)
    model = build_mlp(n_features=3)

    assert model.layers[-1].activation.__name__ == "linear"


def test_cnn_output_activation_is_linear_for_regression():
    configure(seed=0)
    model = build_cnn(image_size=32)

    assert model.layers[-1].activation.__name__ == "linear"


def test_monotone_model_never_predicts_more_life_for_a_hotter_engine():
    configure(seed=0)
    model = build_monotone_mlp(n_features=3)

    cool = model.predict(np.array([[0.0, 0.0, 0.0]]), verbose=0)[0, 0]
    hot = model.predict(np.array([[1.0, 1.0, 1.0]]), verbose=0)[0, 0]

    assert hot <= cool


def test_monotone_property_holds_across_many_random_pairs():
    """Monotonicity is a structural guarantee, not a property of one lucky seed."""
    configure(seed=0)
    model = build_monotone_mlp(n_features=3)
    rng = np.random.default_rng(0)

    cooler = rng.random((30, 3))
    hotter = cooler + rng.random((30, 3))

    assert np.all(
        model.predict(hotter, verbose=0).ravel() <= model.predict(cooler, verbose=0).ravel() + 1e-6
    )
