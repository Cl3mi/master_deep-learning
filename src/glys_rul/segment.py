"""Decomposition of an engine image into its named thermal regions.

Geometry is identical across the dataset, so components are identified by
position rather than shape: sorted left to right they are cone, body, pylon.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import config
from .errors import DataContractError


def foreground(image: np.ndarray, white_threshold: float = config.WHITE_THRESHOLD) -> np.ndarray:
    """Boolean mask of non-background pixels."""
    return image.mean(axis=2) < white_threshold


def regions(
    image: np.ndarray,
    expected: int = config.EXPECTED_REGIONS,
    names: tuple[str, ...] = config.REGION_NAMES,
    min_component_px: int = config.MIN_COMPONENT_PX,
    erode_iterations: int = config.ERODE_ITERATIONS,
) -> dict[str, np.ndarray]:
    """Return eroded boolean masks keyed by region name, ordered left to right.

    Erosion removes JPEG ringing along region edges, which would otherwise pull
    sampled colours towards the white background.
    """
    labels, count = ndimage.label(foreground(image))
    slices = ndimage.find_objects(labels)

    components: list[tuple[int, np.ndarray]] = []
    for index, bounds in enumerate(slices, start=1):
        mask = labels == index
        if int(mask.sum()) < min_component_px:
            continue
        components.append((bounds[1].start, mask))

    if len(components) != expected:
        raise DataContractError(
            f"expected {expected} engine regions but found {len(components)} "
            f"components larger than {min_component_px} px "
            f"({count} raw components). The image does not match the documented "
            f"format contract."
        )

    components.sort(key=lambda item: item[0])
    eroded = {}
    for name, (_, mask) in zip(names, components, strict=True):
        shrunk = ndimage.binary_erosion(mask, iterations=erode_iterations)
        eroded[name] = shrunk if shrunk.any() else mask
    return eroded
