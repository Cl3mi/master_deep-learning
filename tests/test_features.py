import numpy as np
import pandas as pd
import pytest

from glys_rul.colorscale import ColorScale
from glys_rul.features import (
    PHYSICAL_FEATURES,
    degenerate_columns,
    extract,
    temperature_histogram,
)
from tests.conftest import BLACK, MAGENTA, ORANGE, YELLOW, make_engine


@pytest.fixture
def scale(scale_image):
    return ColorScale.from_image(scale_image, vmin=0.0, vmax=1200.0)


def test_region_temperatures_follow_the_painted_colours(scale, engine):
    result = extract(engine, scale, min_component_px=100, erode_iterations=1)

    assert result["cone_c"] < result["body_c"]
    assert result["body_c"] == pytest.approx(result["pylon_c"], abs=60.0)


def test_cold_engine_reads_near_zero(scale):
    result = extract(
        make_engine(BLACK, BLACK, BLACK), scale, min_component_px=100, erode_iterations=1
    )

    for name in ("cone_c", "body_c", "pylon_c"):
        assert result[name] == pytest.approx(0.0, abs=40.0)


def test_total_temperature_is_the_sum_of_regions(scale, engine):
    result = extract(engine, scale, min_component_px=100, erode_iterations=1)

    assert result["total_c"] == pytest.approx(
        result["cone_c"] + result["body_c"] + result["pylon_c"]
    )


def test_hotter_engine_has_greater_total(scale):
    hot = extract(
        make_engine(YELLOW, MAGENTA, MAGENTA), scale, min_component_px=100, erode_iterations=1
    )
    cool = extract(
        make_engine(BLACK, ORANGE, ORANGE), scale, min_component_px=100, erode_iterations=1
    )

    assert hot["total_c"] > cool["total_c"]


def test_physical_feature_names_are_present(scale, engine):
    result = extract(engine, scale, min_component_px=100, erode_iterations=1)

    for name in PHYSICAL_FEATURES:
        assert name in result


def test_histogram_sums_to_one(scale, engine):
    from glys_rul.segment import regions

    masks = regions(engine, min_component_px=100, erode_iterations=1)
    combined = np.logical_or.reduce(list(masks.values()))

    histogram = temperature_histogram(scale.to_map(engine)[combined], bins=16)

    assert histogram.sum() == pytest.approx(1.0)
    assert len(histogram) == 16


def test_degenerate_columns_detects_zero_variance():
    frame = pd.DataFrame({"constant": [5.0, 5.0, 5.0], "varying": [1.0, 2.0, 3.0]})

    assert degenerate_columns(frame) == ["constant"]


def test_degenerate_columns_flags_near_constant_area_features():
    """Areas vary under 2 percent across the real dataset and carry no physics."""
    frame = pd.DataFrame(
        {"cone_px": [138840.0, 138802.0, 137465.0], "cone_c": [827.0, 657.0, 0.0]}
    )

    assert "cone_px" in degenerate_columns(frame, relative_tolerance=0.05)
    assert "cone_c" not in degenerate_columns(frame, relative_tolerance=0.05)
