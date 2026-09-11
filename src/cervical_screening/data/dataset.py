"""PyTorch Dataset reading from a split manifest CSV produced by make_splits.py."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset

CLASSES = [
    "im_Dyskeratotic",
    "im_Koilocytotic",
    "im_Metaplastic",
    "im_Parabasal",
    "im_Superficial-Intermediate",
]


class SipakmedManifestDataset(Dataset):
    """Loads (image, label) pairs for one split ("train"/"val"/"calibration"/"test")."""

    def __init__(self, manifest_csv: Path, split: str, transform: Optional[Callable] = None):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        with open(manifest_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row["split"] != split:
                    continue
                self.samples.append((Path(row["path"]), int(row["class_idx"])))

        if not self.samples:
            raise ValueError(f"no rows found for split={split!r} in {manifest_csv}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def class_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in CLASSES}
        for _path, label in self.samples:
            counts[CLASSES[label]] += 1
        return counts
