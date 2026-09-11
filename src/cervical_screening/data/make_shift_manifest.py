"""Build a binary (normal/abnormal) manifest for a cross-dataset domain-shift
evaluation set (Herlev or Mendeley LBC) — proposal S16/RQ4.

The whole dataset becomes one "test" split: unlike SIPaKMeD (the training
source, see make_splits.py), Herlev and Mendeley LBC exist here purely to be
evaluated on zero-shot, never trained or calibrated on.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from cervical_screening.data.harmonize_labels import (
    HERLEV_TO_BINARY,
    MENDELEY_LBC_TO_BINARY,
    scan_binary_manifest,
)

DATASET_MAPS = {
    "herlev": HERLEV_TO_BINARY,
    "mendeley_lbc": MENDELEY_LBC_TO_BINARY,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_MAPS), required=True)
    parser.add_argument("--root", type=Path, required=True,
                         help="directory containing this dataset's class subfolders")
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    rows = scan_binary_manifest(args.root, DATASET_MAPS[args.dataset])

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "binary_label", "split", "source_dataset"])
        for path, label in rows:
            writer.writerow([str(path), label, "test", args.dataset])

    counts = {}
    for _path, label in rows:
        counts[label] = counts.get(label, 0) + 1
    print(f"wrote {len(rows)} rows to {args.out_csv}")
    print("label counts:", counts)


if __name__ == "__main__":
    main()
