import numpy as np
import pytest

from glys_rul import config
from glys_rul.colorscale import ColorScale
from glys_rul.errors import DataContractError
from tests.conftest import BLACK, MAGENTA, ORANGE, YELLOW


def test_endpoints_map_to_scale_bounds(scale_image):
    scale = ColorScale.from_image(scale_image, vmin=0.0, vmax=1200.0)

    assert scale.to_celsius(BLACK) == pytest.approx(0.0, abs=30.0)
    assert scale.to_celsius(MAGENTA) == pytest.approx(1200.0, abs=30.0)


def test_temperature_order_follows_the_colour_ramp(scale_image):
    scale = ColorScale.from_image(scale_image, vmin=0.0, vmax=1200.0)

    temps = [scale.to_celsius(c) for c in (BLACK, ORANGE, YELLOW, MAGENTA)]

    assert temps == sorted(temps), f"ramp must be monotone, got {temps}"


def test_scale_is_invertible(scale_image):
    scale = ColorScale.from_image(scale_image, vmin=0.0, vmax=1200.0)

    assert scale.max_roundtrip_error() < config.MAX_ROUNDTRIP_ERROR_C


def test_non_invertible_scale_is_rejected(tmp_path):
    """A bar that returns to its starting colour cannot be inverted."""
    from PIL import Image

    width = 300
    bar = np.zeros((40, width, 3), dtype=np.uint8)
    ramp = np.concatenate(
        [np.linspace(0, 255, width // 2), np.linspace(255, 0, width - width // 2)]
    )
    bar[:, :, 0] = ramp.astype(np.uint8)
    path = tmp_path / "bad.png"
    Image.fromarray(bar).save(path)

    with pytest.raises(DataContractError, match="not invertible"):
        ColorScale.from_image(path, vmin=0.0, vmax=1200.0)


def test_to_map_converts_a_whole_image(scale_image, engine):
    scale = ColorScale.from_image(scale_image, vmin=0.0, vmax=1200.0)

    temperature_map = scale.to_map(engine)

    assert temperature_map.shape == engine.shape[:2]
    assert temperature_map[30, 60] == pytest.approx(1200.0, abs=60.0)  # magenta body


@pytest.mark.slow
def test_real_scale_inverts_within_fifteen_degrees():
    scale = ColorScale.from_image(config.SCALE_IMAGE, config.SCALE_VMIN, config.SCALE_VMAX)

    assert scale.max_roundtrip_error() <= 15.0


@pytest.mark.slow
def test_real_scale_luminance_is_not_monotone():
    """Documents a real trap: brightness peaks mid-scale, so grayscale
    conversion destroys the signal above the peak."""
    scale = ColorScale.from_image(config.SCALE_IMAGE, config.SCALE_VMIN, config.SCALE_VMAX)

    luminance = scale.lut @ np.array([0.299, 0.587, 0.114])

    assert luminance.argmax() < len(luminance) - 1, "luminance must peak before the hot end"
    assert luminance[-1] < luminance.max()
