"""Dataset audit: label parsing, content grouping and achievable-accuracy bounds.

The supplied dataset contains byte-identical images carrying different labels.
Two consequences follow, and both are enforced downstream: splits must separate
content hashes rather than filenames, and estimation error is bounded below by
the disagreement among labels sharing a hash.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .errors import DataContractError
from .io import md5_of


def parse_label(filename: str) -> float:
    """Extract remaining useful life in hours from a filename stem."""
    stem = Path(filename).stem
    match = re.match(config.LABEL_PATTERN, stem)
    if match is None:
        raise DataContractError(
            f"filename {filename!r} does not match the documented label pattern "
            f"{config.LABEL_PATTERN!r}; expected e.g. '047h.jpeg'"
        )
    return float(match.group("rul"))


def scan_directory(directory: Path, pattern: str = config.IMAGE_GLOB) -> pd.DataFrame:
    """Build the sample table: one row per file, with label and content group."""
    paths = sorted(Path(directory).glob(pattern))
    if not paths:
        raise DataContractError(
            f"no files matching {pattern!r} in {directory}. The data folder must "
            f"contain images named like '047h.jpeg'."
        )
    records = [
        {"path": path, "filename": path.name, "rul": parse_label(path.name), "group": md5_of(path)}
        for path in paths
    ]
    return pd.DataFrame.from_records(records).sort_values("rul").reset_index(drop=True)


def error_floors(frame: pd.DataFrame) -> dict[str, float]:
    """Lowest MAE and RMSE any estimator can achieve on this data.

    Samples sharing a content group are indistinguishable by any function of the
    pixels, so the best possible prediction for a group is a single constant:
    the median for MAE, the mean for RMSE.
    """
    absolute_error = 0.0
    squared_error = 0.0
    for _, group in frame.groupby("group"):
        labels = group["rul"].to_numpy(dtype=float)
        absolute_error += float(np.abs(labels - np.median(labels)).sum())
        squared_error += float(((labels - labels.mean()) ** 2).sum())
    count = len(frame)
    return {"mae": absolute_error / count, "rmse": math.sqrt(squared_error / count)}


def duplicate_report(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per content group listing the files and labels that share it."""
    rows = []
    for group, members in frame.groupby("group"):
        rows.append(
            {
                "group": group[:6],
                "files": ", ".join(sorted(members["filename"])),
                "labels": ", ".join(f"{value:g}" for value in sorted(members["rul"])),
                "n": len(members),
            }
        )
    return pd.DataFrame(rows).sort_values("labels").reset_index(drop=True)
