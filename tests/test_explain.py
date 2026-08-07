import numpy as np
import pytest

from glys_rul.explain import occlusion_map, permutation_importance


class PylonOnlyModel:
    """Depends solely on the third feature."""

    def predict(self, features):
        features = np.atleast_2d(features)
        return features[:, 2] * 2.0


class RightHalfModel:
    """Depends solely on the mean of the right half of the image."""

    def predict(self, images):
        images = np.asarray(images)
        return images[:, :, images.shape[2] // 2 :, 0].mean(axis=(1, 2))


def test_permutation_importance_identifies_the_only_used_feature():
    rng = np.random.default_rng(0)
    features = rng.random((40, 3))
    target = PylonOnlyModel().predict(features)

    importance = permutation_importance(
        PylonOnlyModel(), features, target, names=("cone", "body", "pylon"), rng=rng
    )

    assert max(importance, key=importance.get) == "pylon"


def test_unused_features_have_near_zero_importance():
    rng = np.random.default_rng(0)
    features = rng.random((40, 3))
    target = PylonOnlyModel().predict(features)

    importance = permutation_importance(
        PylonOnlyModel(), features, target, names=("cone", "body", "pylon"), rng=rng
    )

    assert importance["cone"] == pytest.approx(0.0, abs=1e-9)
    assert importance["body"] == pytest.approx(0.0, abs=1e-9)


def test_permutation_importance_is_reported_for_every_named_feature():
    rng = np.random.default_rng(0)
    features = rng.random((20, 3))
    target = PylonOnlyModel().predict(features)

    importance = permutation_importance(
        PylonOnlyModel(), features, target, names=("cone", "body", "pylon"), rng=rng
    )

    assert list(importance) == ["cone", "body", "pylon"]


def test_permutation_importance_is_reproducible():
    features = np.random.default_rng(0).random((20, 3))
    target = PylonOnlyModel().predict(features)

    first = permutation_importance(
        PylonOnlyModel(), features, target, ("a", "b", "c"), rng=np.random.default_rng(5)
    )
    second = permutation_importance(
        PylonOnlyModel(), features, target, ("a", "b", "c"), rng=np.random.default_rng(5)
    )

    assert first == second


def test_occlusion_map_has_one_value_per_patch():
    image = np.random.default_rng(0).random((16, 16, 1))

    heatmap = occlusion_map(RightHalfModel(), image, patch=4, stride=4)

    assert heatmap.shape == (4, 4)


def test_occlusion_map_highlights_the_region_the_model_uses():
    image = np.ones((16, 16, 1))

    heatmap = occlusion_map(RightHalfModel(), image, patch=4, stride=4)

    assert heatmap[:, 2:].mean() > heatmap[:, :2].mean()


def test_occlusion_values_are_non_negative():
    """The map reports magnitude of change, so it cannot be negative."""
    image = np.random.default_rng(1).random((16, 16, 1))

    heatmap = occlusion_map(RightHalfModel(), image, patch=4, stride=4)

    assert np.all(heatmap >= 0.0)


def test_occlusion_map_is_flat_for_a_model_that_ignores_the_image():
    class Constant:
        def predict(self, images):
            return np.full(len(np.asarray(images)), 42.0)

    heatmap = occlusion_map(Constant(), np.ones((16, 16, 1)), patch=4, stride=4)

    assert np.allclose(heatmap, 0.0)
