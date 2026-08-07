import json

import numpy as np
import pytest

from glys_rul.determinism import configure
from glys_rul.estimators import KerasRegressor
from glys_rul.export import export_mlp, forward
from glys_rul.models import build_mlp

pytestmark = pytest.mark.slow


@pytest.fixture
def trained(tmp_path):
    configure(seed=0)
    features = np.array([[3215.0], [2679.0], [1485.0], [1.0]])
    target = np.array([4.0, 25.0, 74.5, 100.0])
    model = KerasRegressor(lambda: build_mlp(n_features=1), epochs=200).fit(features, target)
    path = tmp_path / "model.json"
    payload = export_mlp(model, path, feature_names=["total_c"])
    return model, features, path, payload


def test_export_writes_layers_and_scaling(trained):
    _, _, path, _ = trained

    payload = json.loads(path.read_text())

    assert payload["feature_names"] == ["total_c"]
    assert len(payload["layers"]) >= 2
    assert "feature_mean" in payload and "feature_std" in payload
    assert payload["target_scale"] > 0


def test_exported_weights_reproduce_keras_predictions(trained):
    """The browser evaluates these weights by hand; the two must agree."""
    model, features, path, _ = trained

    payload = json.loads(path.read_text())
    expected = model.predict(features)
    actual = np.array([forward(payload, row) for row in features])

    assert np.allclose(actual, expected, atol=1e-4)


def test_every_layer_records_its_activation(trained):
    _, _, path, _ = trained

    payload = json.loads(path.read_text())

    assert all("activation" in layer for layer in payload["layers"])
    assert payload["layers"][-1]["activation"] == "linear"


def test_payload_is_plain_json_serialisable(trained):
    """No numpy scalars may leak in, or json.dump fails at write time."""
    _, _, _, payload = trained

    json.dumps(payload)


def test_forward_matches_keras_on_unseen_inputs(trained):
    model, _, path, _ = trained
    payload = json.loads(path.read_text())
    fresh = np.array([[2000.0], [500.0], [3600.0]])

    expected = model.predict(fresh)
    actual = np.array([forward(payload, row) for row in fresh])

    assert np.allclose(actual, expected, atol=1e-4)
