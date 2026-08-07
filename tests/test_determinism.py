import numpy as np
import pytest

from glys_rul.determinism import configure

pytestmark = pytest.mark.slow


def test_same_seed_produces_identical_weights():
    import keras

    configure(seed=0)
    first = keras.layers.Dense(4)
    first.build((None, 3))
    first_weights = [w.numpy().copy() for w in first.weights]

    configure(seed=0)
    second = keras.layers.Dense(4)
    second.build((None, 3))
    second_weights = [w.numpy().copy() for w in second.weights]

    for a, b in zip(first_weights, second_weights, strict=True):
        assert np.array_equal(a, b)


def test_different_seeds_produce_different_weights():
    import keras

    configure(seed=0)
    first = keras.layers.Dense(4)
    first.build((None, 3))
    first_weights = first.weights[0].numpy().copy()

    configure(seed=1)
    second = keras.layers.Dense(4)
    second.build((None, 3))
    second_weights = second.weights[0].numpy().copy()

    assert not np.array_equal(first_weights, second_weights)


def test_configure_forces_single_threaded_cpu():
    import tensorflow as tf

    configure(seed=0)

    assert tf.config.threading.get_intra_op_parallelism_threads() == 1
    assert tf.config.threading.get_inter_op_parallelism_threads() == 1
    assert tf.config.list_physical_devices("GPU") == []


def test_determinism_environment_is_set():
    import os

    from glys_rul import determinism  # noqa: F401

    assert os.environ["TF_ENABLE_ONEDNN_OPTS"] == "0"
    assert os.environ["PYTHONHASHSEED"] == "0"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
