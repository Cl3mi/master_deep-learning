import numpy as np
import pytest

from glys_rul.counterfactual import required_temperature, threshold_table


class LinearModel:
    """Predicts 100 h minus one hour per 30 degrees of total temperature."""

    def predict(self, features):
        features = np.atleast_2d(features)
        return 100.0 - features.sum(axis=1) / 30.0


def test_finds_the_temperature_that_reaches_a_target_life():
    current = np.array([600.0, 600.0, 600.0])

    result = required_temperature(LinearModel(), current, region_index=2, target_rul=60.0)

    assert result is not None
    assert result == pytest.approx(0.0, abs=30.0)


def test_returns_none_when_the_target_is_unreachable():
    current = np.array([1200.0, 1200.0, 1200.0])

    result = required_temperature(LinearModel(), current, region_index=2, target_rul=99.0)

    assert result is None


def test_reports_headroom_when_the_target_is_already_met():
    """The answer is the highest temperature still meeting the target.

    Here 50 h is reached even at the top of the scale, so the useful advice is
    that the region has full headroom — not the temperature it happens to run at.
    """
    current = np.array([0.0, 0.0, 300.0])

    result = required_temperature(LinearModel(), current, region_index=2, target_rul=50.0)

    assert result == pytest.approx(1200.0)


def test_cooling_a_region_never_reduces_predicted_life():
    current = np.array([600.0, 600.0, 600.0])

    threshold = required_temperature(LinearModel(), current, region_index=0, target_rul=50.0)

    assert threshold is not None
    assert threshold <= current[0]


def test_threshold_is_the_highest_temperature_still_reaching_the_target():
    """A lower answer would be needlessly conservative maintenance advice."""
    current = np.array([0.0, 0.0, 0.0])

    threshold = required_temperature(
        LinearModel(), current, region_index=0, target_rul=90.0, resolution=5.0
    )

    assert threshold is not None
    assert LinearModel().predict(np.array([[threshold, 0.0, 0.0]]))[0] >= 90.0
    assert LinearModel().predict(np.array([[threshold + 5.0, 0.0, 0.0]]))[0] < 90.0


def test_threshold_table_covers_every_region():
    current = np.array([600.0, 600.0, 600.0])

    table = threshold_table(LinearModel(), current, ("cone", "body", "pylon"), target_rul=60.0)

    assert list(table) == ["cone", "body", "pylon"]


def test_threshold_table_reports_none_for_regions_that_cannot_help():
    current = np.array([1200.0, 1200.0, 1200.0])

    table = threshold_table(LinearModel(), current, ("cone", "body", "pylon"), target_rul=99.0)

    assert all(value is None for value in table.values())
