"""Core segmentation utilities for splitting a grayscale image into two phases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image, ImageFilter


ArrayLike = np.ndarray


@dataclass
class PhaseSegmentationResult:
    """Result of running the :class:`Segmenter` on an image."""

    mask: ArrayLike
    boundary: ArrayLike
    threshold: float
    grayscale: ArrayLike

    def save_mask(self, path: Path | str) -> None:
        """Save the binary mask to ``path`` as an 8-bit image."""

        path = Path(path)
        mask_img = Image.fromarray((self.mask.astype(np.uint8) * 255), mode="L")
        mask_img.save(path)

    def overlay_on(
        self,
        image: Image.Image,
        *,
        boundary_color: Tuple[int, int, int] = (255, 0, 0),
        boundary_alpha: int = 255,
    ) -> Image.Image:
        """Return an RGB image with the segmentation boundary drawn on top."""

        if image.mode != "RGB":
            image = image.convert("RGB")

        rgb = np.array(image)
        overlay = rgb.copy()
        overlay[self.boundary] = np.array(boundary_color, dtype=np.uint8)

        if boundary_alpha >= 255:
            blended = overlay
        else:
            alpha = np.clip(boundary_alpha / 255.0, 0.0, 1.0)
            blended = (alpha * overlay + (1.0 - alpha) * rgb).astype(np.uint8)

        return Image.fromarray(blended, mode="RGB")


def otsu_threshold(image: ArrayLike) -> float:
    """Return the global Otsu threshold for the supplied grayscale ``image``."""

    flat = image.ravel().astype(np.uint8)
    hist, _ = np.histogram(flat, bins=256, range=(0, 256))
    total = flat.size
    sum_total = np.dot(hist, np.arange(256))

    sum_background = 0.0
    weight_background = 0.0
    max_variance = -1.0
    threshold = 0

    for intensity, count in enumerate(hist):
        weight_background += count
        if weight_background == 0:
            continue

        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += intensity * count
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        variance_between = (
            weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        )
        if variance_between > max_variance:
            max_variance = variance_between
            threshold = intensity

    return float(threshold)


def _ensure_grayscale(image: Image.Image) -> Image.Image:
    if image.mode == "L":
        return image
    return image.convert("L")


def _gaussian_blur(image: Image.Image, radius: float) -> Image.Image:
    if radius <= 0:
        return image
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def _binary_sliding_window(mask: ArrayLike, radius: int, *, reducer) -> ArrayLike:
    if radius <= 0:
        return mask

    size = 2 * radius + 1
    padded = np.pad(mask, radius, mode="constant", constant_values=reducer is np.all)
    windows = sliding_window_view(padded, (size, size))
    return reducer(windows, axis=(-1, -2))


def binary_erosion(mask: ArrayLike, radius: int) -> ArrayLike:
    return _binary_sliding_window(mask, radius, reducer=np.all)


def binary_dilation(mask: ArrayLike, radius: int) -> ArrayLike:
    return _binary_sliding_window(mask, radius, reducer=np.any)


def binary_opening(mask: ArrayLike, radius: int) -> ArrayLike:
    if radius <= 0:
        return mask
    return binary_dilation(binary_erosion(mask, radius), radius)


def binary_closing(mask: ArrayLike, radius: int) -> ArrayLike:
    if radius <= 0:
        return mask
    return binary_erosion(binary_dilation(mask, radius), radius)


def extract_boundary(mask: ArrayLike) -> ArrayLike:
    eroded = binary_erosion(mask, 1)
    return mask & ~eroded


@dataclass
class Segmenter:
    """Segment a grayscale image into two phases using Otsu's method."""

    blur_radius: float = 1.5
    morphology_radius: int = 2
    select_dark_phase: bool = True
    closing: bool = True

    def segment(self, image: Image.Image) -> PhaseSegmentationResult:
        grayscale_image = _ensure_grayscale(image)
        smoothed = _gaussian_blur(grayscale_image, self.blur_radius)
        as_array = np.array(smoothed, dtype=np.uint8)

        threshold = otsu_threshold(as_array)
        if self.select_dark_phase:
            mask = as_array <= threshold
        else:
            mask = as_array >= threshold

        if self.morphology_radius > 0:
            if self.closing:
                mask = binary_closing(mask, self.morphology_radius)
            else:
                mask = binary_opening(mask, self.morphology_radius)

        boundary = extract_boundary(mask)
        return PhaseSegmentationResult(mask=mask, boundary=boundary, threshold=threshold, grayscale=as_array)


def segment_image(
    image_path: Path | str,
    *,
    blur_radius: float = 1.5,
    morphology_radius: int = 2,
    select_dark_phase: bool = True,
    closing: bool = True,
) -> PhaseSegmentationResult:
    image = Image.open(image_path)
    segmenter = Segmenter(
        blur_radius=blur_radius,
        morphology_radius=morphology_radius,
        select_dark_phase=select_dark_phase,
        closing=closing,
    )
    try:
        return segmenter.segment(image)
    finally:
        image.close()
