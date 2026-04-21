"""Bilingual camera-control prompt builder matching the HF Space format exactly."""

from __future__ import annotations


def bilingual_rotate_prompt(rotate_degrees: float) -> str:
    """Build the bilingual rotate prompt used by linoyts/Qwen-Image-Edit-Angles.

    Format: Chinese directive first, English second, terminal period.
    The LoRA was trained on this bilingual conditioning — English-only changes the look.

    Args:
        rotate_degrees: Non-zero. Positive rotates the camera right, negative rotates left.
            The sign of the Space's slider is flipped (positive = left), but we follow
            the natural convention here and flip internally when building the string.
    """
    if rotate_degrees == 0:
        raise ValueError("rotate_degrees must be nonzero")

    # Match the Space verbatim: its slider always yields float, so abs(15.0) -> "15.0".
    magnitude_str = str(abs(float(rotate_degrees)))

    if rotate_degrees > 0:
        return (
            f"将镜头向右旋转{magnitude_str}度 "
            f"Rotate the camera {magnitude_str} degrees to the right."
        )
    return (
        f"将镜头向左旋转{magnitude_str}度 "
        f"Rotate the camera {magnitude_str} degrees to the left."
    )
