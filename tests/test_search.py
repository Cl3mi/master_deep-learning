import numpy as np
import pytest

from glys_rul.search import SEARCH_SPACES, run_campaign

pytestmark = pytest.mark.slow


@pytest.fixture
def ladder():
    features = np.array(
        [[3215.0, 827.0, 1194.0], [2679.0, 657.0, 1194.0],
         [1485.0, 0.0, 827.0], [1.0, 0.0, 0.0]]
    )
    target = np.array([4.0, 25.0, 74.5, 100.0])
    groups = np.array(["a", "b", "c", "d"])
    return features, target, groups


def test_campaign_records_one_row_per_trial(tmp_path, ladder):
    features, target, groups = ladder

    frame = run_campaign(
        features, target, groups, n_trials=2, output=tmp_path / "experiments.csv",
        families=("feature_mlp",), max_epochs=20,
    )

    assert len(frame) >= 2
    assert (tmp_path / "experiments.csv").is_file()


def test_ledger_records_the_configuration_of_each_trial(tmp_path, ladder):
    features, target, groups = ladder

    frame = run_campaign(
        features, target, groups, n_trials=2, output=tmp_path / "experiments.csv",
        families=("feature_mlp",), max_epochs=20,
    )

    for column in ("family", "params", "mae", "rmse", "status"):
        assert column in frame.columns


def test_failed_trials_are_recorded_rather_than_dropped(tmp_path, ladder):
    """A ledger of only successes cannot distinguish thorough search from luck."""
    features, target, groups = ladder

    frame = run_campaign(
        features, target, groups, n_trials=2, output=tmp_path / "experiments.csv",
        families=("always_fails",), max_epochs=20,
    )

    assert len(frame) == 2
    assert set(frame["status"]) == {"failed"}
    assert frame["note"].str.len().gt(0).all()


def test_search_spaces_cover_the_documented_families():
    assert set(SEARCH_SPACES) >= {"feature_mlp", "monotone_mlp", "cnn"}


def test_campaign_is_reproducible(tmp_path, ladder):
    features, target, groups = ladder

    first = run_campaign(
        features, target, groups, n_trials=2, output=tmp_path / "a.csv",
        families=("feature_mlp",), seed=0, max_epochs=20,
    )
    second = run_campaign(
        features, target, groups, n_trials=2, output=tmp_path / "b.csv",
        families=("feature_mlp",), seed=0, max_epochs=20,
    )

    assert list(first["params"]) == list(second["params"])
    assert np.allclose(first["mae"], second["mae"])


def test_ledger_is_sorted_best_first(tmp_path, ladder):
    features, target, groups = ladder

    frame = run_campaign(
        features, target, groups, n_trials=3, output=tmp_path / "experiments.csv",
        families=("feature_mlp",), max_epochs=20,
    )
    successful = frame[frame["status"] == "ok"]["mae"].to_numpy()

    assert np.all(np.diff(successful) >= 0), "successful trials must be ordered best first"
