"""Command-line entry points.

`validate` checks the data contract and prints a report without training
anything, so a swapped dataset can be diagnosed in seconds. `reproduce` runs the
full pipeline and writes `reports/`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config
from .audit import duplicate_report, error_floors, scan_directory
from .colorscale import ColorScale
from .evaluate import run_evaluation
from .io import load_rgb, verify_manifest
from .segment import regions


def validate(data_dir: Path, scale_path: Path) -> int:
    """Print the data contract report. Raises DataContractError on violation."""
    verify_manifest(data_dir, data_dir / config.MANIFEST.name)
    samples = scan_directory(data_dir)
    scale = ColorScale.from_image(scale_path)
    floors = error_floors(samples)

    print(f"files                 {len(samples)}")
    print(f"unique content groups {samples['group'].nunique()}")
    print(f"labels                {samples['rul'].min():g} .. {samples['rul'].max():g} h")
    print(f"scale LUT entries     {len(scale.temps)}")
    print(f"scale roundtrip error {scale.max_roundtrip_error():.1f} °C")
    print(f"MAE floor             {floors['mae']:.4f} h")
    print(f"RMSE floor            {floors['rmse']:.4f} h")
    print()
    print(duplicate_report(samples).to_string(index=False))
    print()
    for path in samples["path"]:
        print(f"  {path.name}: {len(regions(load_rgb(path)))} regions")
    return 0


def _search(args, scale_path: Path) -> int:
    """Run the optimisation campaign and print the leading configurations."""
    import numpy as np

    from .colorscale import ColorScale
    from .dataset import build_image_tensor
    from .evaluate import build_feature_table
    from .features import PHYSICAL_FEATURES
    from .search import run_campaign

    table = build_feature_table(args.data_dir, scale_path)
    # Column 0 is the summed load; the rest are the per-region temperatures.
    features = np.column_stack(
        [table["total_c"].to_numpy(dtype=float), table[list(PHYSICAL_FEATURES)].to_numpy(float)]
    )
    images = None
    if "cnn" in args.families:
        images = build_image_tensor(
            table, ColorScale.from_image(scale_path), args.data_dir, config.IMAGE_SIZE
        )

    frame = run_campaign(
        features,
        table["rul"].to_numpy(dtype=float),
        table["group"].to_numpy(),
        n_trials=args.trials,
        output=args.output,
        families=tuple(args.families),
        images=images,
    )
    successful = frame[frame["status"] == "ok"]
    failed = len(frame) - len(successful)
    print(f"{len(frame)} trials, {len(successful)} successful, {failed} failed")
    print(successful.head(10).to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="glys-rul", description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--scale", type=Path, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("validate", help="check the data contract and print a report")
    reproduce = subparsers.add_parser("reproduce", help="run the full pipeline")
    reproduce.add_argument("--output", type=Path, default=config.REPORTS_DIR)
    reproduce.add_argument("--no-neural", action="store_true")

    search = subparsers.add_parser("search", help="run the optimisation campaign")
    search.add_argument("--trials", type=int, default=25)
    search.add_argument("--output", type=Path, default=config.REPORTS_DIR / "experiments.csv")
    search.add_argument(
        "--families", nargs="+", default=["feature_mlp", "monotone_mlp", "cnn"]
    )

    args = parser.parse_args(argv)
    scale_path = args.scale or (args.data_dir / config.SCALE_IMAGE.name)

    if args.command == "validate":
        return validate(args.data_dir, scale_path)

    if args.command == "search":
        return _search(args, scale_path)

    results = run_evaluation(
        data_dir=args.data_dir,
        scale_path=scale_path,
        output_dir=args.output,
        include_neural=not args.no_neural,
    )
    print(json.dumps(results["models"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
