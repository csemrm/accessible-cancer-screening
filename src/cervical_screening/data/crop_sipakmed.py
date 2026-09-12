"""Build native-resolution single-cell crops from raw SIPaKMeD cluster images.

SIPaKMeD (Plissiti et al., ICIP 2018) ships as 966 cluster images (one .bmp per
slide field of view) plus, for every annotated cell in that field, a pair of
polygon boundary files: ``<id>_cytNN.dat`` (cytoplasm boundary) and
``<id>_nucNN.dat`` (nucleus boundary), each a newline-separated list of
``x,y`` floating-point coordinates.

The commonly-circulated "cropped" Kaggle mirrors resize every crop to a fixed
66x66px, which throws away the nucleus-level detail this project's
explainability evaluation (RQ3) depends on. This script re-derives the crops
directly from the cytoplasm polygon's bounding box, at native resolution,
with a small margin, so downstream preprocessing controls the resize step
explicitly instead of inheriting one baked into a third-party mirror.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image

CLASSES = [
    "im_Dyskeratotic",
    "im_Koilocytotic",
    "im_Metaplastic",
    "im_Parabasal",
    "im_Superficial-Intermediate",
]


def read_polygon(path: Path) -> list[tuple[float, float]]:
    points = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            x, y = float(row[0]), float(row[1])
            points.append((x, y))
    return points


def bbox(points: list[tuple[float, float]], margin: float) -> tuple[int, int, int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (
        int(min(xs) - margin),
        int(min(ys) - margin),
        int(max(xs) + margin),
        int(max(ys) + margin),
    )


def crop_class(raw_root: Path, out_root: Path, class_dir: str, margin: float,
                nucleus_rows: list[dict] | None = None) -> int:
    src_dir = raw_root / class_dir / class_dir
    if not src_dir.exists():
        raise FileNotFoundError(f"expected raw class directory not found: {src_dir}")

    out_dir = out_root / class_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for bmp_path in sorted(src_dir.glob("*.bmp")):
        cluster_id = bmp_path.stem  # e.g. "001"
        cyt_files = sorted(src_dir.glob(f"{cluster_id}_cyt*.dat"))
        if not cyt_files:
            continue

        image = Image.open(bmp_path).convert("RGB")
        w, h = image.size

        for cyt_path in cyt_files:
            cell_idx = cyt_path.stem.split("_cyt")[-1]  # e.g. "01"
            points = read_polygon(cyt_path)
            if len(points) < 3:
                continue
            left, top, right, bottom = bbox(points, margin)
            left, top = max(0, left), max(0, top)
            right, bottom = min(w, right), min(h, bottom)
            if right <= left or bottom <= top:
                continue

            crop = image.crop((left, top, right, bottom))
            out_path = out_dir / f"{cluster_id}_{cell_idx}.png"
            crop.save(out_path)
            n_written += 1

            if nucleus_rows is not None:
                # RQ3's nucleus-overlap explainability metric needs the nucleus
                # boundary in crop-local coordinates. SIPaKMeD occasionally
                # ships more than one nucleus polygon per cell (binucleated
                # cells), hence a list rather than a single polygon.
                nuc_files = sorted(src_dir.glob(f"{cluster_id}_nuc{cell_idx}*.dat"))
                polygons_local = []
                for nuc_path in nuc_files:
                    nuc_points = read_polygon(nuc_path)
                    if len(nuc_points) < 3:
                        continue
                    polygons_local.append([[x - left, y - top] for x, y in nuc_points])
                nucleus_rows.append({
                    "crop_path": str(out_path.relative_to(out_root)),
                    "crop_width": right - left,
                    "crop_height": bottom - top,
                    "nucleus_polygons": json.dumps(polygons_local),
                })

    return n_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True,
                         help="directory containing the im_<Class>/im_<Class>/ raw folders")
    parser.add_argument("--out-root", type=Path, required=True,
                         help="directory to write native-resolution crops into, one subfolder per class")
    parser.add_argument("--margin", type=float, default=4.0,
                         help="pixels of margin added around the cytoplasm bounding box")
    parser.add_argument("--nucleus-manifest", type=Path, default=None,
                         help="if set, also write a CSV mapping each crop to its crop-local nucleus "
                              "polygon(s), for RQ3's nucleus-overlap explainability metric")
    args = parser.parse_args()

    total = 0
    nucleus_rows: list[dict] | None = [] if args.nucleus_manifest else None
    for class_dir in CLASSES:
        n = crop_class(args.raw_root, args.out_root, class_dir, args.margin, nucleus_rows=nucleus_rows)
        print(f"{class_dir}: {n} crops")
        total += n
    print(f"TOTAL: {total} crops")

    if args.nucleus_manifest and nucleus_rows is not None:
        args.nucleus_manifest.parent.mkdir(parents=True, exist_ok=True)
        with open(args.nucleus_manifest, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["crop_path", "crop_width", "crop_height", "nucleus_polygons"])
            writer.writeheader()
            writer.writerows(nucleus_rows)
        print(f"nucleus manifest: {len(nucleus_rows)} rows -> {args.nucleus_manifest}")


if __name__ == "__main__":
    main()
