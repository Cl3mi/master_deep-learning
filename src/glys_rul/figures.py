"""All matplotlib output.

Every accuracy figure draws the irreducible error floor, so a reader can never
see a score without immediately seeing the bound it should be judged against.

Colour follows role rather than taste. Two categorical inks are used — one for
data marks, one to distinguish the winning model — and they were checked for
colour-vision separation rather than chosen by eye (worst-pair CVD dE 24.7,
normal-vision 33.6). Magnitude is carried by single-hue sequential ramps,
light to dark, never a rainbow. The floor is drawn in the reserved "limit" red
and is always accompanied by a text label, so it never depends on colour alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

#: Categorical ink for data marks.
SERIES = "#2a78d6"
#: Categorical ink distinguishing the best model in the ladder.
WINNER = "#eb6834"
#: Reserved status colour for a hard limit.
LIMIT = "#d03b3b"

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#dedcd6"

#: Single-hue sequential ramps, light to dark.
_BLUE_STEPS = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
_ORANGE_STEPS = ["#fde4d8", "#f9c2a6", "#f39c74", "#eb6834", "#c8531f", "#9c3f16", "#6f2b0e"]

SEQUENTIAL = LinearSegmentedColormap.from_list("glys_blue", _BLUE_STEPS)
THERMAL = LinearSegmentedColormap.from_list("glys_orange", _ORANGE_STEPS)


def _style(axes) -> None:
    """Recede the frame so the data carries the figure."""
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
    axes.tick_params(colors=MUTED, labelsize=8, length=3)
    axes.grid(True, color=GRID, linewidth=0.6, linestyle="-", alpha=0.9)
    axes.set_axisbelow(True)
    axes.title.set_color(INK)
    axes.xaxis.label.set_color(MUTED)
    axes.yaxis.label.set_color(MUTED)


def _save(figure, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_predicted_vs_actual(
    truth: np.ndarray, predicted: np.ndarray, floors: dict[str, float], path: Path
) -> None:
    """Scatter against the identity line, with the achievable-accuracy band."""
    figure, axes = plt.subplots(figsize=(5, 5))
    limits = [0.0, float(max(truth.max(), predicted.max())) * 1.1]

    axes.fill_between(
        limits,
        [value - floors["mae"] for value in limits],
        [value + floors["mae"] for value in limits],
        color=LIMIT,
        alpha=0.14,
        linewidth=0,
        label=f"irreducible band (±{floors['mae']:.2f} h)",
    )
    axes.plot(limits, limits, color=MUTED, linewidth=1.0, label="perfect prediction")
    axes.scatter(
        truth, predicted, s=70, color=SERIES, edgecolor="white", linewidth=1.5,
        zorder=3, label="out-of-fold prediction",
    )

    _style(axes)
    axes.set_xlabel("actual RUL [h]")
    axes.set_ylabel("predicted RUL [h]")
    axes.set_xlim(limits)
    axes.set_ylim(limits)
    axes.set_title("Predicted versus actual remaining useful life", fontsize=10)
    axes.legend(loc="upper left", fontsize=7.5, frameon=False, labelcolor=MUTED)
    _save(figure, path)


def plot_residuals(
    totals: np.ndarray, residuals: np.ndarray, floors: dict[str, float], path: Path
) -> None:
    """Residuals against total temperature, revealing systematic structure."""
    figure, axes = plt.subplots(figsize=(6.5, 4))
    axes.axhline(0.0, color=MUTED, linewidth=1.0)
    axes.axhline(floors["mae"], color=LIMIT, linestyle="--", linewidth=1.2)
    axes.axhline(-floors["mae"], color=LIMIT, linestyle="--", linewidth=1.2)
    axes.scatter(
        totals, residuals, s=70, color=SERIES, edgecolor="white", linewidth=1.5, zorder=3
    )

    _style(axes)
    axes.annotate(
        f"irreducible ±{floors['mae']:.2f} h",
        xy=(axes.get_xlim()[1], floors["mae"]),
        xytext=(-4, 4), textcoords="offset points",
        ha="right", fontsize=7.5, color=LIMIT,
    )
    axes.set_xlabel("total temperature Σ°C")
    axes.set_ylabel("residual [h]")
    axes.set_title("Residuals versus thermal state", fontsize=10)
    _save(figure, path)


def plot_confusion(
    truth: np.ndarray, predicted: np.ndarray, bin_width: int, path: Path
) -> None:
    """Confusion matrix of the binned classification view."""
    upper = float(max(truth.max(), predicted.max()))
    edges = np.arange(0, upper + bin_width, bin_width)
    size = max(len(edges), 1)
    truth_bins = np.clip(np.digitize(truth, edges) - 1, 0, size - 1)
    predicted_bins = np.clip(np.digitize(predicted, edges) - 1, 0, size - 1)

    matrix = np.zeros((size, size), dtype=int)
    for actual, estimated in zip(truth_bins, predicted_bins, strict=True):
        matrix[actual, estimated] += 1

    figure, axes = plt.subplots(figsize=(5.5, 4.8))
    image = axes.imshow(matrix, cmap=SEQUENTIAL, vmin=0)
    axes.set_xlabel(f"predicted class [{bin_width} h bins]")
    axes.set_ylabel("actual class")
    axes.set_title("Classification view of the regression output", fontsize=10)
    axes.tick_params(colors=MUTED, labelsize=8, length=3)
    axes.xaxis.label.set_color(MUTED)
    axes.yaxis.label.set_color(MUTED)
    axes.title.set_color(INK)
    axes.grid(False)
    bar = figure.colorbar(image, ax=axes, label="samples")
    bar.ax.tick_params(colors=MUTED, labelsize=8)
    _save(figure, path)


def plot_learning_curve(
    sizes: list[int], scores: list[float], floors: dict[str, float], path: Path
) -> None:
    """Cross-validated error against the number of training groups."""
    figure, axes = plt.subplots(figsize=(6.5, 4))
    axes.plot(sizes, scores, marker="o", markersize=8, linewidth=2, color=SERIES,
              markeredgecolor="white", markeredgewidth=1.5, label="cross-validated MAE")
    axes.axhline(floors["mae"], color=LIMIT, linestyle="--", linewidth=1.2,
                 label=f"irreducible floor {floors['mae']:.2f} h")

    _style(axes)
    axes.set_xlabel("training groups")
    axes.set_ylabel("MAE [h]")
    axes.set_title("How data-starved is this problem?", fontsize=10)
    axes.legend(fontsize=7.5, frameon=False, labelcolor=MUTED)
    _save(figure, path)


def plot_baseline_ladder(
    names: list[str], maes: list[float], floors: dict[str, float], path: Path
) -> None:
    """Horizontal bars, best model last and distinctly inked."""
    order = np.argsort(maes)[::-1]
    ordered_names = [names[index] for index in order]
    ordered_maes = [maes[index] for index in order]
    colours = [SERIES] * len(ordered_names)
    colours[-1] = WINNER

    figure, axes = plt.subplots(figsize=(6.5, 0.5 * len(names) + 1.9))
    axes.barh(ordered_names, ordered_maes, color=colours, height=0.62)
    axes.axvline(floors["mae"], color=LIMIT, linestyle="--", linewidth=1.2)

    for index, value in enumerate(ordered_maes):
        axes.annotate(
            f"{value:.2f}", xy=(value, index), xytext=(4, 0), textcoords="offset points",
            va="center", fontsize=8, color=MUTED,
        )

    _style(axes)
    axes.grid(True, axis="x", color=GRID, linewidth=0.6)
    axes.grid(False, axis="y")
    # Reserve headroom above the top bar so the floor label cannot overlap it.
    top = len(ordered_names) - 1 + 0.85
    axes.set_ylim(-0.7, top)
    axes.annotate(
        f"irreducible floor {floors['mae']:.2f} h",
        xy=(floors["mae"], top - 0.25), xytext=(5, 0), textcoords="offset points",
        va="center", fontsize=7.5, color=LIMIT,
    )
    axes.set_xlabel("cross-validated MAE [h]  (lower is better)")
    axes.set_title("Baseline ladder", fontsize=10)
    _save(figure, path)


def plot_occlusion(image: np.ndarray, heatmap: np.ndarray, path: Path) -> None:
    """Temperature map beside its occlusion sensitivity."""
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    thermal = axes[0].imshow(image[..., 0], cmap=THERMAL)
    axes[0].set_title("temperature map", fontsize=10, color=INK)
    axes[0].axis("off")
    figure.colorbar(thermal, ax=axes[0], fraction=0.046).ax.tick_params(
        colors=MUTED, labelsize=8
    )

    sensitivity = axes[1].imshow(heatmap, cmap=SEQUENTIAL)
    axes[1].set_title("occlusion sensitivity", fontsize=10, color=INK)
    axes[1].axis("off")
    bar = figure.colorbar(
        sensitivity, ax=axes[1], fraction=0.046, label="|Δ prediction| [h]"
    )
    bar.ax.tick_params(colors=MUTED, labelsize=8)
    _save(figure, path)
