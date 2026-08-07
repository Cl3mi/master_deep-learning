"""Splitting and augmentation.

Splits separate *content groups*, never filenames: the dataset contains
byte-identical images with different labels, so a filename-level split places a
pixel-identical copy of the test sample into training.

Augmentation is geometric plus sensor noise only. Brightness, contrast, hue and
gamma jitter are excluded by design: colour is the label, so photometric
transforms silently relabel the sample.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from scipy import ndimage

from . import config


def grouped_splits(
    groups: np.ndarray,
    max_logo_groups: int = config.MAX_LOGO_GROUPS,
    k: int = config.GROUP_KFOLD_SPLITS,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_index, test_index) pairs that never share a group.

    Leave-one-group-out while the group count is small enough to afford it,
    otherwise grouped k-fold.
    """
    groups = np.asarray(groups)
    unique = np.array(sorted(set(groups.tolist())))

    if len(unique) <= max_logo_groups:
        held_out: list[np.ndarray] = [np.array([value]) for value in unique]
    else:
        held_out = [unique[index::k] for index in range(k)]

    for fold in held_out:
        mask = np.isin(groups, fold)
        yield np.flatnonzero(~mask), np.flatnonzero(mask)


def build_image_tensor(
    table, scale, data_dir, image_size: int = config.IMAGE_SIZE, mask_background: bool = False
) -> np.ndarray:
    """Convert every sample to a normalised single-channel temperature map.

    Feeding calibrated temperature rather than raw RGB means the convolutional
    and feature-based models consume the same physical quantity, so a difference
    in their scores reflects architecture rather than input representation.

    Rows follow `table` order, so tensor row i pairs with label i.
    """
    from pathlib import Path

    from .io import load_rgb
    from .segment import foreground

    maps = []
    for filename in table["filename"]:
        image = load_rgb(Path(data_dir) / filename)
        temperatures = scale.to_map(image)
        if mask_background:
            # White is nearest the hot end of the colour scale, so an unmasked
            # background reads as roughly 1000 °C — hotter than much of the
            # engine. Zero it so the network sees temperature, not an artefact.
            temperatures = np.where(foreground(image), temperatures, 0.0)
        zoom = (image_size / temperatures.shape[0], image_size / temperatures.shape[1])
        resized = _fit_to(ndimage.zoom(temperatures, zoom, order=1, mode="nearest"),
                          image_size, image_size)
        maps.append(resized / config.SCALE_VMAX)
    return np.clip(np.stack(maps)[..., None], 0.0, 1.0)


def augment_batch(
    images: np.ndarray,
    rng: np.random.Generator,
    max_shift: float = 0.08,
    max_scale: float = 0.10,
    max_rotation_deg: float = 5.0,
    noise_std: float = 0.005,
    calibration_bias_std: float = 0.002,
) -> np.ndarray:
    """Apply label-preserving augmentation to a batch of temperature maps.

    Geometry: translation, scaling and small rotations. Photometry: additive
    sensor noise and a small global calibration bias, both modelled as
    measurement error rather than as a change of temperature.
    """
    output = np.empty_like(images)
    height, width = images.shape[1:3]

    for index, image in enumerate(images):
        frame = image[..., 0]

        if max_rotation_deg > 0:
            angle = rng.uniform(-max_rotation_deg, max_rotation_deg)
            frame = ndimage.rotate(frame, angle, reshape=False, order=1, mode="nearest")

        if max_scale > 0:
            zoom = 1.0 + rng.uniform(-max_scale, max_scale)
            scaled = ndimage.zoom(frame, zoom, order=1, mode="nearest")
            frame = _fit_to(scaled, height, width)

        if max_shift > 0:
            shift = (
                rng.uniform(-max_shift, max_shift) * height,
                rng.uniform(-max_shift, max_shift) * width,
            )
            frame = ndimage.shift(frame, shift, order=1, mode="nearest")

        if calibration_bias_std > 0:
            frame = frame + rng.normal(0.0, calibration_bias_std)
        if noise_std > 0:
            frame = frame + rng.normal(0.0, noise_std, size=frame.shape)

        output[index, ..., 0] = frame

    return output


def _fit_to(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    """Centre-crop or edge-pad `frame` to exactly height x width."""
    current_height, current_width = frame.shape
    if current_height >= height:
        top = (current_height - height) // 2
        frame = frame[top : top + height]
    else:
        pad = height - current_height
        frame = np.pad(frame, ((pad // 2, pad - pad // 2), (0, 0)), mode="edge")
    current_height, current_width = frame.shape
    if current_width >= width:
        left = (current_width - width) // 2
        frame = frame[:, left : left + width]
    else:
        pad = width - current_width
        frame = np.pad(frame, ((0, 0), (pad // 2, pad - pad // 2)), mode="edge")
    return frame
