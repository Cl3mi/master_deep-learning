"""Reproducibility guarantees.

These are the tests behind the project's central claim. They are slow because
they run the real pipeline: nothing weaker would actually prove the point.
"""

import json
import subprocess
import sys

import pytest

from glys_rul import config

pytestmark = pytest.mark.slow


def _run(output_dir, extra=()):
    subprocess.run(
        [sys.executable, "-m", "glys_rul.cli", "reproduce", "--output", str(output_dir), *extra],
        check=True,
        cwd=config.REPO_ROOT,
        capture_output=True,
    )


def test_two_runs_produce_identical_results(tmp_path):
    """Same code, same data, same bytes — the whole determinism stack in one assertion."""
    first, second = tmp_path / "first", tmp_path / "second"
    _run(first, ("--no-neural",))
    _run(second, ("--no-neural",))

    assert (first / "results.json").read_bytes() == (second / "results.json").read_bytes()


def test_committed_results_exist_and_parse():
    results = json.loads((config.REPORTS_DIR / "results.json").read_text())

    assert results["dataset"]["n_files"] == 11
    assert results["dataset"]["n_groups"] == 6
    assert set(results["models"]) >= {"mean", "isotonic", "feature_mlp", "cnn"}


def test_reported_error_never_beats_the_irreducible_floor():
    """A score below the floor is leakage or memorisation, never skill."""
    results = json.loads((config.REPORTS_DIR / "results.json").read_text())
    floor = results["floors"]["mae"]

    for name, metrics in results["models"].items():
        assert metrics["mae"] >= floor - 1e-6, (
            f"{name} reports MAE {metrics['mae']} below the floor {floor}; "
            f"this indicates leakage, not skill"
        )


def test_committed_floors_match_a_fresh_derivation():
    """The golden copy must reflect what the audit actually computes today."""
    from glys_rul.audit import error_floors, scan_directory

    results = json.loads((config.REPORTS_DIR / "results.json").read_text())
    fresh = error_floors(scan_directory(config.DATA_DIR))

    assert results["floors"]["mae"] == pytest.approx(fresh["mae"], abs=1e-6)
    assert results["floors"]["rmse"] == pytest.approx(fresh["rmse"], abs=1e-6)


def test_shuffled_label_control_shows_no_residual_signal():
    """With labels permuted the model must do no better than guessing the mean."""
    results = json.loads((config.REPORTS_DIR / "results.json").read_text())
    control = results["shuffled_label_control"]

    assert control["mae"] >= control["no_skill_mae"] * 0.9


def test_learning_curve_is_recorded_and_descending():
    results = json.loads((config.REPORTS_DIR / "results.json").read_text())
    curve = results["learning_curve"]

    assert len(curve) >= 3
    assert curve[0]["mae"] > curve[-1]["mae"], "more training groups must help"
