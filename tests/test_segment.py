import numpy as np
import pytest

from glys_rul import config
from glys_rul.errors import DataContractError
from glys_rul.io import load_rgb
from glys_rul.segment import regions
from tests.conftest import BLACK, MAGENTA, YELLOW, make_engine


def test_returns_the_three_named_regions(engine):
    result = regions(engine, min_component_px=100, erode_iterations=1)

    assert list(result) == ["cone", "body", "pylon"]


def test_regions_are_ordered_left_to_right(engine):
    result = regions(engine, min_component_px=100, erode_iterations=1)

    centres = [np.flatnonzero(result[name].any(axis=0)).mean() for name in result]

    assert centres == sorted(centres)


def test_masks_are_disjoint_and_non_empty(engine):
    result = regions(engine, min_component_px=100, erode_iterations=1)

    for mask in result.values():
        assert mask.sum() > 0
    assert not (result["cone"] & result["body"]).any()
    assert not (result["body"] & result["pylon"]).any()


def test_black_regions_are_still_detected():
    """0 degrees C regions are pure black and must not be confused with background."""
    dark = make_engine(BLACK, BLACK, BLACK)

    result = regions(dark, min_component_px=100, erode_iterations=1)

    assert len(result) == 3
    assert all(mask.sum() > 0 for mask in result.values())


def test_wrong_component_count_raises_actionable_error():
    canvas = np.full((60, 120, 3), 255.0)
    canvas[10:50, 5:35] = YELLOW  # only one region

    with pytest.raises(DataContractError, match="expected 3"):
        regions(canvas, min_component_px=100, erode_iterations=1)


def test_noise_speckles_are_ignored():
    canvas = make_engine(YELLOW, MAGENTA, MAGENTA)
    canvas[0, 0] = BLACK  # single stray pixel, far below the size threshold

    result = regions(canvas, min_component_px=100, erode_iterations=1)

    assert len(result) == 3


def test_erosion_shrinks_masks_but_never_empties_them(engine):
    lightly = regions(engine, min_component_px=100, erode_iterations=1)
    heavily = regions(engine, min_component_px=100, erode_iterations=3)

    for name in lightly:
        assert heavily[name].sum() > 0
        assert heavily[name].sum() <= lightly[name].sum()


@pytest.mark.slow
def test_every_real_image_yields_exactly_three_regions():
    for path in sorted(config.DATA_DIR.glob(config.IMAGE_GLOB)):
        result = regions(load_rgb(path))
        assert len(result) == 3, f"{path.name} produced {len(result)} regions"
