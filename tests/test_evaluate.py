import json

import pytest

from glys_rul.evaluate import build_feature_table, run_evaluation

pytestmark = pytest.mark.slow

TINY = {"min_component_px": 100, "erode_iterations": 1}


def test_feature_table_has_one_row_per_file(engine_dir, scale_image):
    frame = build_feature_table(engine_dir, scale_image, **TINY)

    assert len(frame) == 5
    assert {"rul", "group", "cone_c", "body_c", "pylon_c", "total_c"} <= set(frame.columns)


def test_feature_table_orders_rows_by_label(engine_dir, scale_image):
    frame = build_feature_table(engine_dir, scale_image, **TINY)

    assert list(frame["rul"]) == sorted(frame["rul"])


def test_feature_table_groups_byte_identical_files(engine_dir, scale_image):
    frame = build_feature_table(engine_dir, scale_image, **TINY)

    assert frame["group"].nunique() == 3


def test_evaluation_writes_results_and_metadata(tmp_path, engine_dir, scale_image):
    run_evaluation(
        data_dir=engine_dir, scale_path=scale_image, output_dir=tmp_path,
        include_neural=False, **TINY,
    )

    results = json.loads((tmp_path / "results.json").read_text())
    meta = json.loads((tmp_path / "run_meta.json").read_text())

    assert "floors" in results and "models" in results
    assert "python" in meta and "packages" in meta


def test_results_json_excludes_volatile_metadata(tmp_path, engine_dir, scale_image):
    """results.json is diffed by CI, so anything that changes per run lives elsewhere."""
    run_evaluation(
        data_dir=engine_dir, scale_path=scale_image, output_dir=tmp_path,
        include_neural=False, **TINY,
    )

    text = (tmp_path / "results.json").read_text().lower()

    for token in ("timestamp", "date", "elapsed", "duration", "version", "platform"):
        assert token not in text


def test_evaluation_is_byte_identical_across_runs(tmp_path, engine_dir, scale_image):
    first, second = tmp_path / "a", tmp_path / "b"
    for output in (first, second):
        run_evaluation(
            data_dir=engine_dir, scale_path=scale_image, output_dir=output,
            include_neural=False, **TINY,
        )

    assert (first / "results.json").read_bytes() == (second / "results.json").read_bytes()


def test_baseline_ladder_is_reported_with_floors(tmp_path, engine_dir, scale_image):
    run_evaluation(
        data_dir=engine_dir, scale_path=scale_image, output_dir=tmp_path,
        include_neural=False, **TINY,
    )

    results = json.loads((tmp_path / "results.json").read_text())

    assert set(results["models"]) >= {"mean", "nearest_neighbour", "linear", "isotonic"}
    assert results["floors"]["mae"] >= 0.0


def test_no_model_beats_the_irreducible_floor(tmp_path, engine_dir, scale_image):
    """A score below the floor is leakage, and must never appear in the output."""
    run_evaluation(
        data_dir=engine_dir, scale_path=scale_image, output_dir=tmp_path,
        include_neural=False, **TINY,
    )

    results = json.loads((tmp_path / "results.json").read_text())
    floor = results["floors"]["mae"]

    for name, metrics in results["models"].items():
        assert metrics["mae"] >= floor - 1e-6, f"{name} scored below the floor"


def test_side_tables_are_written(tmp_path, engine_dir, scale_image):
    run_evaluation(
        data_dir=engine_dir, scale_path=scale_image, output_dir=tmp_path,
        include_neural=False, **TINY,
    )

    assert (tmp_path / "features.csv").is_file()
    assert (tmp_path / "duplicates.csv").is_file()


def test_degenerate_area_features_are_recorded_as_excluded(tmp_path, engine_dir, scale_image):
    run_evaluation(
        data_dir=engine_dir, scale_path=scale_image, output_dir=tmp_path,
        include_neural=False, **TINY,
    )

    results = json.loads((tmp_path / "results.json").read_text())

    assert results["modelling_features"] == ["cone_c", "body_c", "pylon_c"]
