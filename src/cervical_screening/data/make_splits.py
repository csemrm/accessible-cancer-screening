"""Build a stratified train/val/calibration/test split manifest for SIPaKMeD.

Four splits, not the usual three, because the project's uncertainty work
needs a calibration set held separate from both training and the final test
set (see docs/methodology.md): fitting conformal thresholds on the same data
used to pick the risk-coverage operating point would invalidate the coverage
guarantee.

Output is a single CSV manifest (path, label, split) rather than four
directory trees, so re-splitting never requires copying image files.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

CLASSES = [
    "im_Dyskeratotic",
    "im_Koilocytotic",
    "im_Metaplastic",
    "im_Parabasal",
    "im_Superficial-Intermediate",
]

# train / val / calibration / test
DEFAULT_FRACTIONS = (0.60, 0.15, 0.10, 0.15)


def stratified_split(paths: list[Path], fractions: tuple[float, float, float, float], seed: int
                      ) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    shuffled = paths[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * fractions[0])
    n_val = round(n * fractions[1])
    n_cal = round(n * fractions[2])
    # test gets the remainder so rounding never drops an image
    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "calibration": shuffled[n_train + n_val:n_train + n_val + n_cal],
        "test": shuffled[n_train + n_val + n_cal:],
    }
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops-root", type=Path, required=True,
                         help="directory with one subfolder per class, containing crop images")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--fractions", type=float, nargs=4, default=DEFAULT_FRACTIONS,
                         metavar=("TRAIN", "VAL", "CALIBRATION", "TEST"))
    args = parser.parse_args()

    if abs(sum(args.fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0, got {args.fractions}")

    rows = []
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = args.crops_root / class_name
        images = sorted(p for p in class_dir.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        if not images:
            raise FileNotFoundError(f"no images found under {class_dir}")

        splits = stratified_split(images, tuple(args.fractions), seed=args.seed + class_idx)
        for split_name, split_paths in splits.items():
            for p in split_paths:
                rows.append((str(p), class_name, class_idx, split_name))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "class_name", "class_idx", "split"])
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for *_rest, split_name in rows:
        counts[split_name] = counts.get(split_name, 0) + 1
    print(f"wrote {len(rows)} rows to {args.out_csv}")
    print("split sizes:", counts)


if __name__ == "__main__":
    main()
