"""Three synthetic degradations (proposal S16): JPEG compression, Gaussian
blur, and brightness/contrast shift, each applied at several severities to
the in-distribution SIPaKMeD test set to approximate field-camera
conditions (compression artifacts, defocus, poor/uneven lighting) that no
available public cytology dataset captures. Applied to the raw PIL image,
before the model's own resize/normalize transform.
"""

from __future__ import annotations

import io

from PIL import Image, ImageEnhance, ImageFilter


def jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gaussian_blur(image: Image.Image, radius: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def brightness_contrast(image: Image.Image, brightness_factor: float, contrast_factor: float) -> Image.Image:
    """factor 1.0 = no change; <1.0 darker/lower-contrast, >1.0 brighter/higher-contrast."""
    image = ImageEnhance.Brightness(image).enhance(brightness_factor)
    return ImageEnhance.Contrast(image).enhance(contrast_factor)


# Severity ladders, mild -> severe, used identically across the eval script
# so every corruption type reports the same number of severity levels.
JPEG_QUALITIES = [90, 70, 50, 30, 10]
BLUR_RADII = [0.0, 1.0, 2.0, 4.0, 8.0]
BRIGHTNESS_CONTRAST_FACTORS = [1.0, 0.8, 0.6, 1.4, 1.7]  # symmetric-ish mild->severe under/over-exposure
