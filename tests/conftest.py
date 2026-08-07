"""Synthetic fixtures mirroring the real dataset's structure at tiny scale.

A fake engine is three horizontally separated rectangles on white, matching the
cone / body / pylon layout. Sizes are chosen so erosion still leaves interior
pixels.
"""

import numpy as np
import pytest
from PIL import Image

# Colours taken from the real dataset's four-value alphabet.
BLACK = (0, 0, 0)
ORANGE = (255, 149, 3)
YELLOW = (255, 192, 2)
MAGENTA = (255, 47, 145)

COLOUR_TEMPS = {BLACK: 0.0, ORANGE: 656.0, YELLOW: 825.0, MAGENTA: 1193.0}


def make_engine(cone=YELLOW, body=MAGENTA, pylon=MAGENTA, size=(120, 60)):
    """Return a float64 RGB array with three separated regions on white."""
    width, height = size
    canvas = np.full((height, width, 3), 255.0)
    canvas[10:50, 5:35] = cone      # leftmost
    canvas[10:50, 40:90] = body     # widest
    canvas[15:35, 95:115] = pylon   # rightmost, smallest
    return canvas


@pytest.fixture
def engine():
    return make_engine()


@pytest.fixture
def scale_image(tmp_path):
    """A synthetic colour bar: black -> orange -> yellow -> magenta, RGBA with margins."""
    width, height = 400, 60
    bar = np.zeros((height, width, 4), dtype=np.uint8)
    bar[:, :, 3] = 0  # transparent margins, as in the real temp.png
    stops = np.array([BLACK, ORANGE, YELLOW, MAGENTA], dtype=float)
    positions = np.linspace(0, width - 81, len(stops))
    for x in range(width - 80):
        t = np.interp(x, positions, np.arange(len(stops)))
        low, high = int(np.floor(t)), min(int(np.ceil(t)), len(stops) - 1)
        frac = t - low
        bar[10:50, x + 40, :3] = (stops[low] * (1 - frac) + stops[high] * frac).astype(np.uint8)
        bar[10:50, x + 40, 3] = 255
    path = tmp_path / "scale.png"
    Image.fromarray(bar, mode="RGBA").save(path)
    return path


@pytest.fixture
def engine_dir(tmp_path):
    """A miniature dataset directory: five images, two of them byte-identical pairs."""
    directory = tmp_path / "engines"
    directory.mkdir()
    specs = {
        "003h": (YELLOW, MAGENTA, MAGENTA),
        "005h": (YELLOW, MAGENTA, MAGENTA),   # duplicate of 003h
        "047h": (ORANGE, YELLOW, YELLOW),
        "051h": (ORANGE, YELLOW, YELLOW),     # duplicate of 047h
        "100h": (BLACK, BLACK, BLACK),
    }
    for name, colours in specs.items():
        array = make_engine(*colours).astype(np.uint8)
        Image.fromarray(array).save(directory / f"{name}.jpeg", quality=100, subsampling=0)
    return directory
