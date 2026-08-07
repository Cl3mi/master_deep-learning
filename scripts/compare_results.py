#!/usr/bin/env python3
"""Compare two results.json files at the precision the pipeline actually guarantees.

Everything derived from the data, every baseline, and both dense networks
reproduce **bit-identically** on any machine. The convolutional model does not:
Conv2D dispatches on CPU SIMD features, so a different processor takes a
different kernel and the result shifts in the last few digits.

Rather than weaken the whole check to a tolerance, this compares each part at the
precision it genuinely holds — exact where exactness is guaranteed, bounded where
it is not. Stdlib only, so CI needs no environment set up.
"""

from __future__ import annotations

import json
import sys

#: Anything trained in TensorFlow depends on the host's SIMD capabilities, so its
#: result shifts in the low-order digits between CPU models. Two tiers, because the
#: magnitude differs by four orders: dense layers are plain matrix multiplies and
#: agree to ~1e-6, while convolution dispatches far more aggressively.
DENSE_MODELS = ("feature_mlp", "monotone_mlp")
CONVOLUTIONAL_PREFIX = "cnn"

DENSE_TOLERANCE = 1e-4
CONVOLUTIONAL_TOLERANCE = 1e-2


def tolerance_for(name: str) -> float | None:
    """Relative tolerance for a model, or None if it must match exactly."""
    if name.startswith(CONVOLUTIONAL_PREFIX):
        return CONVOLUTIONAL_TOLERANCE
    if name in DENSE_MODELS:
        return DENSE_TOLERANCE
    return None

EXACT_SECTIONS = (
    "dataset",
    "floors",
    "excluded_features",
    "modelling_features",
    "labels",
    "learning_curve",
    "shuffled_label_control",
)


def compare(golden: dict, fresh: dict) -> list[str]:
    """Return a list of human-readable failures; empty means the run reproduced."""
    failures: list[str] = []

    for section in EXACT_SECTIONS:
        if golden.get(section) != fresh.get(section):
            failures.append(f"{section}: differs, but must be exactly reproducible")

    if set(golden["models"]) != set(fresh["models"]):
        failures.append(
            f"model set differs: {sorted(golden['models'])} vs {sorted(fresh['models'])}"
        )
        return failures

    for name in sorted(golden["models"]):
        for metric, want in golden["models"][name].items():
            got = fresh["models"][name][metric]
            tolerance = tolerance_for(name)
            if tolerance is not None:
                scale = max(abs(want), 1e-9)
                if abs(got - want) / scale > tolerance:
                    failures.append(
                        f"{name}.{metric}: {got} vs {want} "
                        f"(relative {abs(got - want) / scale:.2e} > {tolerance:.0e})"
                    )
            elif got != want:
                failures.append(
                    f"{name}.{metric}: {got} vs {want}, but must be exactly reproducible"
                )
    return failures


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: compare_results.py GOLDEN.json FRESH.json", file=sys.stderr)
        return 2

    with open(argv[1]) as handle:
        golden = json.load(handle)
    with open(argv[2]) as handle:
        fresh = json.load(handle)

    failures = compare(golden, fresh)
    if failures:
        print("Reproduction FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    exact = [n for n in sorted(golden["models"]) if tolerance_for(n) is None]
    dense = [n for n in sorted(golden["models"]) if tolerance_for(n) == DENSE_TOLERANCE]
    conv = [n for n in sorted(golden["models"]) if tolerance_for(n) == CONVOLUTIONAL_TOLERANCE]
    print("Reproduction verified.")
    print(f"  exact:  {', '.join(EXACT_SECTIONS)}")
    print(f"  exact:  {', '.join(exact)}")
    print(f"  <{DENSE_TOLERANCE:.0e}:  {', '.join(dense)}")
    print(f"  <{CONVOLUTIONAL_TOLERANCE:.0e}:  {', '.join(conv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
