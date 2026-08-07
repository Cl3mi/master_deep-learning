"""Negative controls.

A model that still fits well after its labels are randomised has enough capacity
to memorise the dataset outright. Comparing its cross-validated error under
permuted labels against the mean baseline turns "the model might be memorising"
from a worry into a measurement.
"""

from __future__ import annotations

import numpy as np


def shuffled_label_control(
    target: np.ndarray, groups: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Return the labels permuted across samples, leaving `groups` untouched.

    Permutation is per sample rather than per group. Group-level permutation
    would need equal group sizes, and it would also impose a structure the real
    data does not have: byte-identical images here already carry *different*
    labels, so identical labels within a group is not a property worth
    preserving. The label multiset is preserved exactly, so the comparison
    against the unshuffled run isolates the loss of signal and nothing else.
    """
    target = np.asarray(target, dtype=float)
    return rng.permutation(target)
