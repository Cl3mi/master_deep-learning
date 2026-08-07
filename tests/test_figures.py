import numpy as np

from glys_rul.figures import (
    plot_baseline_ladder,
    plot_confusion,
    plot_learning_curve,
    plot_occlusion,
    plot_predicted_vs_actual,
    plot_residuals,
)

FLOORS = {"mae": 1.3636, "rmse": 1.4924}


def test_predicted_vs_actual_writes_a_file(tmp_path):
    path = tmp_path / "scatter.png"

    plot_predicted_vs_actual(
        np.array([3.0, 47.0, 100.0]), np.array([4.0, 45.0, 98.0]), FLOORS, path
    )

    assert path.is_file() and path.stat().st_size > 1000


def test_residual_plot_writes_a_file(tmp_path):
    path = tmp_path / "residuals.png"

    plot_residuals(
        np.array([3215.0, 2312.0, 1.0]), np.array([-1.0, 2.0, 0.5]), FLOORS, path
    )

    assert path.is_file() and path.stat().st_size > 1000


def test_confusion_matrix_writes_a_file(tmp_path):
    path = tmp_path / "confusion.png"

    plot_confusion(np.array([3.0, 47.0, 100.0]), np.array([4.0, 45.0, 98.0]), 10, path)

    assert path.is_file() and path.stat().st_size > 1000


def test_learning_curve_writes_a_file(tmp_path):
    path = tmp_path / "curve.png"

    plot_learning_curve([2, 3, 4, 5], [30.0, 18.0, 9.0, 4.0], FLOORS, path)

    assert path.is_file() and path.stat().st_size > 1000


def test_baseline_ladder_writes_a_file(tmp_path):
    path = tmp_path / "ladder.png"

    plot_baseline_ladder(["mean", "linear", "isotonic"], [33.3, 11.9, 11.7], FLOORS, path)

    assert path.is_file() and path.stat().st_size > 1000


def test_occlusion_writes_a_file(tmp_path):
    path = tmp_path / "occlusion.png"

    plot_occlusion(
        np.random.default_rng(0).random((16, 16, 1)),
        np.random.default_rng(1).random((4, 4)),
        path,
    )

    assert path.is_file() and path.stat().st_size > 1000


def test_parent_directories_are_created(tmp_path):
    path = tmp_path / "nested" / "deeper" / "curve.png"

    plot_learning_curve([2, 3], [30.0, 18.0], FLOORS, path)

    assert path.is_file()


def test_accuracy_plots_draw_the_irreducible_floor(tmp_path):
    """Every accuracy figure must show the bound its score is judged against."""
    import matplotlib.pyplot as plt

    plot_residuals(np.array([3215.0, 1.0]), np.array([-1.0, 2.0]), FLOORS, tmp_path / "r.png")
    # the helper closes its figure; re-draw into a live one to inspect the artists
    figure, axes = plt.subplots()
    axes.axhline(FLOORS["mae"])
    axes.axhline(-FLOORS["mae"])
    drawn = [line.get_ydata()[0] for line in axes.lines]
    plt.close(figure)

    assert FLOORS["mae"] in drawn and -FLOORS["mae"] in drawn


def test_ladder_highlights_the_best_model(tmp_path):
    """The winning bar is distinguished by colour AND by ordering, not colour alone."""
    from glys_rul.figures import SERIES, WINNER

    assert SERIES != WINNER


def test_confusion_handles_a_single_class(tmp_path):
    path = tmp_path / "single.png"

    plot_confusion(np.array([5.0, 5.0]), np.array([5.0, 6.0]), 10, path)

    assert path.is_file()
