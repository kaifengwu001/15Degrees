"""Stage 1b CLI: non-autoregressive backfill between coarse-pass frames.

Usage:
    python backfill.py --coarse-run fight_coarse --run-name fight_backfill

For each coarse frame N (from `runs/<coarse-run>/frames/frame_NNNN.png`), this
script runs a single fresh edit step -- `pipe(coarse_N, "rotate <half>°")` --
and writes the result to `runs/<run-name>/frames/frame_NNNN.png`. Every
backfill frame's source is a clean coarse anchor, so drift cannot propagate
between backfill frames (they are independent of each other).

The backfill angle is exactly half the coarse `rotate_degrees` from config.yaml
(e.g. coarse 30° -> backfill 15°). Combined with the coarse pass, the
interleaved sequence coarse_0, fine_0, coarse_1, fine_1, ... gives a 2*N-frame
orbit that shares the coarse pass's angular range but halves the per-step
rotation shown in playback.

This script reuses `orbit.pipeline.load_pipeline` unchanged, so the same fp8
residency / NF4 / CPU offload regimes apply. Existing backfill frames are
skipped on re-run unless `--overwrite` is passed, which makes Ctrl+C safe.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from PIL import Image

from orbit.config import ConfigError, load_config
from orbit.logging_setup import configure_logger
from orbit.prompts import bilingual_rotate_prompt
from orbit.runs import ensure_run_dirs, paths_for


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    try:
        config = load_config(Path(args.config))
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    cfg = config.generation

    coarse_paths = paths_for(args.coarse_run)
    if not coarse_paths.frames_dir.is_dir():
        print(
            f"ERROR: coarse frames not found at {coarse_paths.frames_dir}",
            file=sys.stderr,
        )
        return 2

    coarse_frames = sorted(coarse_paths.frames_dir.glob("frame_*.png"))
    if not coarse_frames:
        print(f"ERROR: no coarse frames in {coarse_paths.frames_dir}", file=sys.stderr)
        return 2

    out_paths = paths_for(args.run_name)
    ensure_run_dirs(out_paths)

    logger = configure_logger(out_paths.log_path)
    logger.info("=" * 72)
    logger.info(
        "orbit backfill.py starting — run=%s coarse_run=%s (%d coarse frames)",
        args.run_name,
        args.coarse_run,
        len(coarse_frames),
    )

    if cfg.rotate_degrees == 0:
        logger.error("rotate_degrees must be nonzero")
        return 2
    backfill_degrees = cfg.rotate_degrees / 2.0
    prompt = bilingual_rotate_prompt(backfill_degrees)
    logger.info(
        "Backfill angle = %.2f° (half of coarse rotate_degrees=%.2f)",
        backfill_degrees,
        cfg.rotate_degrees,
    )
    logger.info("prompt=%s", prompt)

    # Heavy imports deferred so --help / arg errors don't require torch/diffusers.
    import torch

    from orbit.generator import _pipeline_device, _save_frame
    from orbit.pipeline import load_pipeline
    from orbit.seeds import pick_seed

    try:
        pipe = load_pipeline(
            lora_fuse_scale=cfg.lora_fuse_scale,
            enable_cpu_offload=cfg.enable_cpu_offload,
            quantize_4bit=cfg.quantize_4bit,
            logger=logger,
        )
    except Exception as exc:  # pragma: no cover - env dependent
        logger.exception("Pipeline load failed: %s", exc)
        return 3

    device = _pipeline_device(pipe)

    total = len(coarse_frames)
    if args.limit is not None:
        total = min(total, max(1, args.limit))
    if args.dry_run:
        total = 1
        logger.info("--dry-run: capping backfill at 1 frame")

    logger.info("Backfilling %d frame(s) at %.2f° each ...", total, backfill_degrees)

    written = 0
    skipped = 0

    try:
        for i in range(total):
            coarse_path = coarse_frames[i]
            out_path = out_paths.frame_path(i)

            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue

            t0 = time.monotonic()
            source = Image.open(coarse_path).convert("RGB")
            seed_i = pick_seed(randomize=cfg.randomize_seed, fixed_seed=cfg.fixed_seed)
            generator = torch.Generator(device=device).manual_seed(seed_i)

            result = pipe(
                image=[source],
                prompt=prompt,
                height=source.height,
                width=source.width,
                num_inference_steps=cfg.inference_steps,
                true_cfg_scale=cfg.true_cfg_scale,
                generator=generator,
                num_images_per_prompt=1,
            )
            out_image = result.images[0]
            _save_frame(out_image, out_path)

            if device == "cuda":
                torch.cuda.empty_cache()

            elapsed = time.monotonic() - t0
            written += 1
            logger.info(
                "backfill %04d <- %s seed=%d elapsed=%.2fs",
                i,
                coarse_path.name,
                seed_i,
                elapsed,
            )
    except KeyboardInterrupt:
        logger.warning(
            "Interrupted by user (Ctrl+C). Wrote %d, skipped %d so far; re-run "
            "the same command to resume (existing frames are skipped).",
            written,
            skipped,
        )
        return 130
    except Exception as exc:
        logger.exception("Backfill failed: %s", exc)
        return 4

    logger.info(
        "Done. Wrote %d, skipped %d (already existed). Output: %s",
        written,
        skipped,
        out_paths.frames_dir,
    )
    return 0


def _parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orbit Decay — non-autoregressive backfill between coarse frames"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: ./config.yaml)",
    )
    parser.add_argument(
        "--coarse-run",
        required=True,
        help="Name of the coarse run (under runs/) whose frames serve as anchors",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Name of the backfill run to create (under runs/)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Cap this invocation at N frames (for scouting drift)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load pipeline, backfill 1 frame, exit",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing backfill frames (default is skip on re-run)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
