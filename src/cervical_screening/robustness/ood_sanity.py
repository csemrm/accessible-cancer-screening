"""OOD sanity check (proposal S16): does the model's confidence appropriately
drop on inputs that are obviously not cytology images at all? Uses
synthetic images (random noise, solid color, checkerboard) rather than a
downloaded natural-image dataset, since the only property that matters here
is "clearly not a cervical cell," not any specific alternative domain.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def random_noise_image(size: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def solid_color_image(size: int, color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (size, size), color)


def checkerboard_image(size: int, tile: int, color_a: tuple[int, int, int],
                        color_b: tuple[int, int, int]) -> Image.Image:
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for r in range(0, size, tile):
        for c in range(0, size, tile):
            color = color_a if ((r // tile) + (c // tile)) % 2 == 0 else color_b
            arr[r:r + tile, c:c + tile] = color
    return Image.fromarray(arr, mode="RGB")


def build_ood_batch(size: int = 224, n_noise: int = 10) -> list[Image.Image]:
    images = [random_noise_image(size, seed=i) for i in range(n_noise)]
    images += [
        solid_color_image(size, (0, 0, 0)),
        solid_color_image(size, (255, 255, 255)),
        solid_color_image(size, (200, 30, 30)),
        checkerboard_image(size, tile=16, color_a=(0, 0, 0), color_b=(255, 255, 255)),
        checkerboard_image(size, tile=32, color_a=(255, 0, 0), color_b=(0, 255, 0)),
    ]
    return images
