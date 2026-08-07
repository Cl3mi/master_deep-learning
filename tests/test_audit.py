import pandas as pd
import pytest

from glys_rul import config
from glys_rul.audit import duplicate_report, error_floors, parse_label, scan_directory
from glys_rul.errors import DataContractError


def test_parses_hours_from_filename():
    assert parse_label("003h.jpeg") == 3.0
    assert parse_label("100h.jpeg") == 100.0


def test_unparseable_filename_raises():
    with pytest.raises(DataContractError, match="does not match"):
        parse_label("engine_alpha.jpeg")


def test_empty_directory_raises(tmp_path):
    with pytest.raises(DataContractError, match="no files matching"):
        scan_directory(tmp_path)


def test_scan_groups_identical_files(engine_dir):
    frame = scan_directory(engine_dir, pattern="*.jpeg")

    assert len(frame) == 5
    assert frame["group"].nunique() == 3
    duplicates = frame[frame["rul"].isin([3.0, 5.0])]
    assert duplicates["group"].nunique() == 1, "byte-identical files must share a group"


def test_scan_records_labels_in_hours(engine_dir):
    frame = scan_directory(engine_dir, pattern="*.jpeg")

    assert sorted(frame["rul"]) == [3.0, 5.0, 47.0, 51.0, 100.0]


def test_error_floor_is_zero_when_every_group_is_unique():
    frame = pd.DataFrame({"group": ["a", "b", "c"], "rul": [10.0, 20.0, 30.0]})

    floors = error_floors(frame)

    assert floors["mae"] == pytest.approx(0.0)
    assert floors["rmse"] == pytest.approx(0.0)


def test_error_floor_reflects_conflicting_labels():
    """Two identical images labelled 3 h and 5 h: best constant is 4 h, error 1 h each."""
    frame = pd.DataFrame({"group": ["a", "a"], "rul": [3.0, 5.0]})

    floors = error_floors(frame)

    assert floors["mae"] == pytest.approx(1.0)
    assert floors["rmse"] == pytest.approx(1.0)


def test_rmse_floor_exceeds_mae_floor_under_uneven_disagreement():
    """RMSE punishes the wider disagreement more than MAE does."""
    frame = pd.DataFrame({"group": ["a", "a", "b", "b"], "rul": [0.0, 2.0, 0.0, 10.0]})

    floors = error_floors(frame)

    assert floors["rmse"] > floors["mae"]


def test_duplicate_report_lists_files_sharing_content(engine_dir):
    frame = scan_directory(engine_dir, pattern="*.jpeg")

    report = duplicate_report(frame)

    assert len(report) == 3
    assert report["n"].sum() == 5
    assert report["n"].max() == 2


@pytest.mark.slow
def test_real_dataset_has_eleven_files_in_six_groups():
    frame = scan_directory(config.DATA_DIR, pattern=config.IMAGE_GLOB)

    assert len(frame) == 11
    assert frame["group"].nunique() == 6


@pytest.mark.slow
def test_real_dataset_error_floors_match_the_design():
    frame = scan_directory(config.DATA_DIR, pattern=config.IMAGE_GLOB)

    floors = error_floors(frame)

    assert floors["mae"] == pytest.approx(1.3636, abs=1e-3)
    assert floors["rmse"] == pytest.approx(1.4924, abs=1e-3)
