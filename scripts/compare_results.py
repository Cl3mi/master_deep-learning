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

#: Models whose kernels dispatch on CPU features and so vary across machines.
#: Every convolutional variant qualifies, so match by prefix rather than by an
#: explicit list that a new ablation would silently fall outside of.
HARDWARE_DEPENDENT_PREFIX = "cnn"


def is_hardware_dependent(name: str) -> bool:
    """True for models whose result depends on the host's SIMD capabilities."""
    return name.startswith(HARDWARE_DEPENDENT_PREFIX)

#: Relative tolerance allowed for those models. Observed across two machines: 1.5e-3.
RELATIVE_TOLERANCE = 1e-2

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
            if is_hardware_dependent(name):
                scale = max(abs(want), 1e-9)
                if abs(got - want) / scale > RELATIVE_TOLERANCE:
                    failures.append(
                        f"{name}.{metric}: {got} vs {want} "
                        f"(relative {abs(got - want) / scale:.2e} > {RELATIVE_TOLERANCE:.0e})"
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

    exact = [n for n in sorted(golden["models"]) if not is_hardware_dependent(n)]
    tolerated = [n for n in sorted(golden["models"]) if is_hardware_dependent(n)]
    print("Reproduction verified.")
    print(f"  exact:     {', '.join(EXACT_SECTIONS)}")
    print(f"  exact:     {', '.join(exact)}")
    print(f"  tolerance: {', '.join(tolerated)} within {RELATIVE_TOLERANCE:.0e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
