"""Command line interface for two-phase image segmentation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from PIL import Image

from .segmentation import Segmenter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Segment an image into two phases using an automatic threshold.",
    )
    parser.add_argument("image", type=Path, help="Path to the input grayscale image.")
    parser.add_argument(
        "--mask-path",
        type=Path,
        default=None,
        help="Where to store the binary mask image (defaults to '<stem>_mask.png').",
    )
    parser.add_argument(
        "--overlay-path",
        type=Path,
        default=None,
        help="Where to store the overlay image (defaults to '<stem>_overlay.png').",
    )
    parser.add_argument(
        "--blur-radius",
        type=float,
        default=1.5,
        help="Gaussian blur radius applied before thresholding (default: 1.5).",
    )
    parser.add_argument(
        "--morphology-radius",
        type=int,
        default=2,
        help="Radius of the morphological cleanup in pixels (default: 2).",
    )
    parser.add_argument(
        "--prefer-bright-phase",
        action="store_true",
        help="Select the bright phase instead of the darker phase.",
    )
    parser.add_argument(
        "--use-opening",
        action="store_true",
        help="Use morphological opening instead of closing for cleanup.",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Disable overlay generation (only the mask will be saved).",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Disable mask saving (only the overlay will be saved).",
    )
    return parser


def _resolve_output_path(path: Optional[Path], image: Path, suffix: str) -> Path:
    if path is not None:
        return path
    return image.with_name(f"{image.stem}{suffix}")


def run_cli(args: argparse.Namespace) -> None:
    image_path: Path = args.image
    if not image_path.exists():
        raise FileNotFoundError(f"Input image '{image_path}' does not exist.")

    with Image.open(image_path) as img:
        segmenter = Segmenter(
            blur_radius=args.blur_radius,
            morphology_radius=args.morphology_radius,
            select_dark_phase=not args.prefer_bright_phase,
            closing=not args.use_opening,
        )
        result = segmenter.segment(img)

    if not args.no_mask:
        mask_path = _resolve_output_path(args.mask_path, image_path, "_mask.png")
        result.save_mask(mask_path)

    if not args.no_overlay:
        overlay_path = _resolve_output_path(args.overlay_path, image_path, "_overlay.png")
        with Image.open(image_path) as original:
            overlay = result.overlay_on(original)
        overlay.save(overlay_path)

    print(
        f"Threshold={result.threshold:.1f} | mask saved={not args.no_mask} | overlay saved={not args.no_overlay}"
    )


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_cli(args)


if __name__ == "__main__":
    main()
