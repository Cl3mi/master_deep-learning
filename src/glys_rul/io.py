"""Image and file I/O. The only module that imports PIL.

All pixel reads go through `load_rgb` so that the alpha-compositing rule is
applied exactly once, in one place.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .errors import DataContractError

_ALPHA_MODES = {"RGBA", "LA", "P", "PA"}


def load_rgb(path: Path | str) -> np.ndarray:
    """Load an image as float64 RGB, compositing any alpha over opaque white.

    Transparent pixels must become white, not black: the reference colour scale
    is RGBA, and black is a meaningful temperature (0 °C). Getting this wrong
    silently corrupts the entire calibration.
    """
    image = Image.open(path)
    if image.mode in _ALPHA_MODES:
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
    return np.asarray(image.convert("RGB"), dtype=np.float64)


def md5_of(path: Path | str) -> str:
    """Return the md5 hex digest of a file's bytes."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(directory: Path, destination: Path) -> dict[str, str]:
    """Record the md5 of every file in `directory` for provenance."""
    entries = {p.name: md5_of(p) for p in sorted(directory.iterdir()) if p.is_file()}
    entries.pop(destination.name, None)
    destination.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")
    return entries


def verify_manifest(directory: Path, manifest_path: Path) -> None:
    """Raise if any recorded file is missing or its content changed."""
    if not manifest_path.is_file():
        return
    recorded = json.loads(manifest_path.read_text())
    for name, expected in recorded.items():
        candidate = directory / name
        if not candidate.is_file():
            raise DataContractError(f"{name} is listed in {manifest_path.name} but missing")
        actual = md5_of(candidate)
        if actual != expected:
            raise DataContractError(
                f"{name} content changed: manifest records {expected}, found {actual}"
            )
