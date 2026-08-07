"""Network architectures.

Capacity is deliberately small. With six distinguishable training states, a model
with thousands of free parameters can only memorise, so both architectures are
sized against the sample count rather than against the input dimensionality.
"""

from __future__ import annotations

import keras
from keras import layers

from . import config


def build_mlp(n_features: int, hidden: tuple[int, ...] = (16, 8), l2: float = 0.0) -> keras.Model:
    """Architecture B: a small dense network over the physical features."""
    regulariser = keras.regularizers.l2(l2) if l2 else None
    return keras.Sequential(
        [keras.Input(shape=(n_features,))]
        + [
            layers.Dense(units, activation="relu", kernel_regularizer=regulariser)
            for units in hidden
        ]
        + [layers.Dense(1, activation="linear")],
        name="feature_mlp",
    )


def build_cnn(
    image_size: int = config.IMAGE_SIZE,
    filters: tuple[int, ...] = (16, 32, 64),
    dense_units: int = 32,
    dropout: float = 0.0,
) -> keras.Model:
    """Architecture A: a compact convolutional network over temperature maps."""
    inputs = keras.Input(shape=(image_size, image_size, 1))
    x = inputs
    for count in filters:
        x = layers.Conv2D(count, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    outputs = layers.Dense(1, activation="linear")(x)
    return keras.Model(inputs, outputs, name="temperature_cnn")


class NonNegative(keras.constraints.Constraint):
    """Clamps weights to be non-negative, enforcing a monotone response."""

    def __call__(self, w):
        return keras.ops.maximum(w, 0.0)


def build_monotone_mlp(n_features: int, hidden: tuple[int, ...] = (16, 8)) -> keras.Model:
    """A dense network guaranteed to predict less life as temperature rises.

    Non-negative weights make the network monotone increasing in its input; the
    input is negated so the composition is monotone decreasing in temperature.
    Extrapolation therefore stays physically sensible.
    """
    # kernel_constraint only clamps weights after each optimiser update, so a
    # freshly built model would otherwise start from ordinary (signed) random
    # weights and briefly violate monotonicity. Constraining the initialiser
    # too makes the guarantee hold from the first forward pass.
    initialiser = keras.initializers.RandomUniform(0.0, 0.1)

    inputs = keras.Input(shape=(n_features,))
    x = layers.Lambda(lambda t: -t, output_shape=(n_features,))(inputs)
    for units in hidden:
        x = layers.Dense(
            units,
            activation="relu",
            kernel_initializer=initialiser,
            kernel_constraint=NonNegative(),
        )(x)
    outputs = layers.Dense(
        1,
        activation="linear",
        kernel_initializer=initialiser,
        kernel_constraint=NonNegative(),
    )(x)
    return keras.Model(inputs, outputs, name="monotone_mlp")
