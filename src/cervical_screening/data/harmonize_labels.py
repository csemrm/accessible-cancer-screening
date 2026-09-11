"""Harmonize SIPaKMeD, Herlev, and Mendeley LBC onto a shared binary label
space for the cross-dataset domain-shift evaluation (proposal S16/RQ4).

The three source datasets use incompatible fine-grained taxonomies with no
1:1 correspondence — SIPaKMeD's 5 cytomorphology classes, Herlev's 7
dysplasia-grade classes, and Mendeley LBC's 4 Bethesda-system classes don't
line up cell-for-cell. The only label space all three datasets can be
honestly compared on is the coarse Normal / Abnormal split every one of them
already implies, which is also the standard regrouping used elsewhere in
this literature for exactly this reason. Fine-grained cross-dataset
transfer is out of scope: it would require assuming a correspondence
between, say, SIPaKMeD's "Dyskeratotic" and Herlev's "severe_dysplastic"
that neither dataset's documentation actually supports.
"""

from __future__ import annotations

from pathlib import Path

NORMAL, ABNORMAL = "normal", "abnormal"

SIPAKMED_TO_BINARY = {
    "im_Parabasal": NORMAL,
    "im_Metaplastic": NORMAL,
    "im_Superficial-Intermediate": NORMAL,
    "im_Koilocytotic": ABNORMAL,
    "im_Dyskeratotic": ABNORMAL,
}

HERLEV_TO_BINARY = {
    "normal_columnar": NORMAL,
    "normal_intermediate": NORMAL,
    "normal_superficiel": NORMAL,
    "light_dysplastic": ABNORMAL,
    "moderate_dysplastic": ABNORMAL,
    "severe_dysplastic": ABNORMAL,
    "carcinoma_in_situ": ABNORMAL,
}

# Mendeley LBC folder names as unpacked from blank1508/mendeley-lbc-cervical-cancer
# on Kaggle — confirm these exact strings against the extracted directory
# listing before relying on this mapping; Bethesda-system full names are
# used as folder names rather than the NILM/LSIL/HSIL/SCC abbreviations.
MENDELEY_LBC_TO_BINARY = {
    "Negative for Intraepithelial malignancy": NORMAL,
    "Low squamous intra-epithelial lesion": ABNORMAL,
    "High squamous intra-epithelial lesion": ABNORMAL,
    "Squamous cell carcinoma": ABNORMAL,
}


def scan_binary_manifest(root: Path, class_to_binary: dict[str, str],
                          extensions=(".jpg", ".jpeg", ".png", ".bmp")) -> list[tuple[Path, str]]:
    """Walk root/<class_name>/... (any depth) and label each image file
    found under a known class_name by the binary mapping. Raises if a class
    directory present in class_to_binary yields zero images, since a silent
    empty class is more likely a naming mismatch than a genuinely empty
    dataset."""
    rows: list[tuple[Path, str]] = []
    for class_name, binary_label in class_to_binary.items():
        class_dir = root / class_name
        if not class_dir.exists():
            # allow a nested "root/<class_name>/<class_name>/..." layout too
            nested = list(root.glob(f"**/{class_name}"))
            if not nested:
                raise FileNotFoundError(f"class directory not found: {class_dir}")
            class_dir = nested[0]

        images = [p for p in class_dir.rglob("*") if p.suffix.lower() in extensions]
        if not images:
            raise FileNotFoundError(f"no images found under {class_dir}")
        rows.extend((p, binary_label) for p in images)

    return rows
