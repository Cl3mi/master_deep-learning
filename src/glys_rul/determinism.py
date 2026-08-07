"""Determinism controls.

Every knob here removes one documented source of run-to-run or machine-to-machine
variation. Environment variables must be set before TensorFlow is imported, so
this module sets them at import time and only then imports the framework.
"""

from __future__ import annotations

import os
import random

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")   # uniform kernels across AVX2/AVX-512
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")     # CPU only: no cuDNN nondeterminism
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

_CONFIGURED_THREADS = False


def configure(seed: int = 0) -> None:
    """Seed every generator and pin execution to a single CPU thread."""
    global _CONFIGURED_THREADS

    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    if not _CONFIGURED_THREADS:
        # Thread counts cannot change once the runtime is initialised.
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.set_visible_devices([], "GPU")
        _CONFIGURED_THREADS = True

    keras.backend.clear_session()
