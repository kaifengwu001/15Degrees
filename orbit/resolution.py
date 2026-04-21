"""Aspect-preserving resolution calculation matching the HF Space logic."""

from __future__ import annotations

from typing import Tuple

from PIL import Image


def compute_output_size(
    source: Image.Image,
    longest_side: int,
) -> Tuple[int, int]:
    """Scale so the longest side == longest_side, preserve aspect, snap each dim down to /8.

    Matches the HF Space `update_dimensions_on_upload` exactly so the LoRA sees the
    same resolutions it was validated against.

    Returns:
        (width, height) both >= 8, both divisible by 8.
    """
    if longest_side < 8 or longest_side % 8 != 0:
        raise ValueError("longest_side must be >= 8 and divisible by 8")

    original_width, original_height = source.size
    if original_width <= 0 or original_height <= 0:
        raise ValueError(f"Invalid source dimensions: {source.size}")

    if original_width > original_height:
        new_width = longest_side
        new_height = int(longest_side * (original_height / original_width))
    else:
        new_height = longest_side
        new_width = int(longest_side * (original_width / original_height))

    new_width = max(8, (new_width // 8) * 8)
    new_height = max(8, (new_height // 8) * 8)
    return new_width, new_height
