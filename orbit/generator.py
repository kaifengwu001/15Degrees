"""Core autoregressive frame generation loop."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from tqdm import tqdm

from orbit.runs import RunPaths
from orbit.seeds import pick_seed
from orbit.state import State, record_frame, save_state


def generate_frames(
    *,
    pipe: Any,
    initial_image: Image.Image,
    state: State,
    paths: RunPaths,
    prompt: str,
    total_frames: int,
    inference_steps: int,
    true_cfg_scale: float,
    randomize_seed: bool,
    fixed_seed: Optional[int],
    logger: logging.Logger,
) -> State:
    """Run the autoregressive rotation loop from `state.last_completed_frame + 1`.

    Returns the final State. The caller is responsible for persisting the returned
    state if the loop completes normally; every per-frame update is already flushed
    to disk atomically as it happens.

    On KeyboardInterrupt the exception is re-raised — in-progress frame is never
    written to disk, so state.json remains consistent.
    """
    start_index = state.last_completed_frame + 1
    if start_index >= total_frames:
        logger.info("Nothing to do: state is already at frame %d of %d.", start_index, total_frames)
        return state

    current_state = state
    current_image = initial_image.convert("RGB")
    width, height = state.width, state.height

    logger.info(
        "Generating frames %d..%d (%d total) at %dx%d, steps=%d, true_cfg=%.2f",
        start_index,
        total_frames - 1,
        total_frames - start_index,
        width,
        height,
        inference_steps,
        true_cfg_scale,
    )

    device = _pipeline_device(pipe)

    progress = tqdm(
        range(start_index, total_frames),
        initial=start_index,
        total=total_frames,
        unit="frame",
        desc="orbit",
    )

    try:
        for i in progress:
            frame_start = time.monotonic()
            seed_i = pick_seed(randomize=randomize_seed, fixed_seed=fixed_seed)
            generator = torch.Generator(device=device).manual_seed(seed_i)

            result = pipe(
                image=[current_image],
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=inference_steps,
                true_cfg_scale=true_cfg_scale,
                generator=generator,
                num_images_per_prompt=1,
            )
            output_image = result.images[0]

            frame_path = paths.frame_path(i)
            _save_frame(output_image, frame_path)

            current_state = record_frame(current_state, i, seed_i)
            save_state(paths.state_path, current_state)

            current_image = output_image
            if device == "cuda":
                torch.cuda.empty_cache()

            elapsed = time.monotonic() - frame_start
            logger.info("frame %04d seed=%d elapsed=%.2fs", i, seed_i, elapsed)
            progress.set_postfix(seed=seed_i, last_s=f"{elapsed:.1f}")
    finally:
        progress.close()

    return current_state


def _pipeline_device(pipe: Any) -> str:
    """Best-effort device detection for building the generator on the right backend."""
    try:
        device = pipe.device
        return str(device.type if hasattr(device, "type") else device)
    except Exception:  # pragma: no cover
        return "cuda" if torch.cuda.is_available() else "cpu"


def _save_frame(image: Image.Image, path: Path) -> None:
    """Save PNG via a temp file + rename so a Ctrl+C mid-write can't leave a half-PNG in place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    image.save(tmp_path, format="PNG", compress_level=6)
    tmp_path.replace(path)
