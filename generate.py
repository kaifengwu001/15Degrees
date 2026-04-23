"""Stage 1 CLI: autoregressive frame generation.

Usage examples:
    python generate.py --source portrait.jpg --run-name portrait_v1 --frames 300
    python generate.py --run-name portrait_v1 --resume
    python generate.py --run-name portrait_v1 --from-frame 85
    python generate.py --run-name portrait_v1 --from-frame 85 --fixed-seed 99
    python generate.py --source portrait.jpg --run-name smoke --dry-run
    python generate.py --source portrait.jpg --run-name burn30 --limit 30
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Optional

if os.environ.get("ORBIT_SDPA_MATH_ONLY", "0") == "1":
    # Force PyTorch's scaled_dot_product_attention onto the pure math backend.
    # Windows + sm_89 + very-long-sequence bf16 attention hits native crashes in
    # cuDNN / flash / mem-efficient fused kernels on some driver/cuDNN combos.
    # Math backend is slower but always correct.
    import torch

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

from PIL import Image

from orbit.config import (
    Config,
    ConfigError,
    GenerationConfig,
    config_hash,
    load_config,
    with_overrides,
)
from orbit.logging_setup import configure_logger
from orbit.prompts import bilingual_rotate_prompt
from orbit.resolution import compute_output_size
from orbit.runs import RunPaths, ensure_run_dirs, paths_for
from orbit.state import (
    State,
    StateError,
    load_state,
    new_state,
    save_state,
    truncate_for_fork,
)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    try:
        config = load_config(Path(args.config))
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    gen_cfg = _apply_cli_overrides(config.generation, args)
    paths = paths_for(gen_cfg.run_name)
    ensure_run_dirs(paths)

    logger = configure_logger(paths.log_path)
    logger.info("=" * 72)
    logger.info("orbit generate.py starting — run=%s", gen_cfg.run_name)
    logger.info("config=%s config_hash=%s", args.config, config_hash(config))

    try:
        current_state, current_image, total_frames = _prepare_run(
            args=args,
            config=config,
            gen_cfg=gen_cfg,
            paths=paths,
            logger=logger,
        )
    except (ConfigError, StateError, FileNotFoundError) as exc:
        logger.error("Setup failed: %s", exc)
        return 2

    prompt = bilingual_rotate_prompt(gen_cfg.rotate_degrees)
    logger.info("prompt=%s", prompt)

    # Import heavy deps only now, after arg parsing / setup is confirmed good.
    from orbit.generator import generate_frames
    from orbit.pipeline import load_pipeline

    try:
        pipe = load_pipeline(
            lora_fuse_scale=gen_cfg.lora_fuse_scale,
            enable_cpu_offload=gen_cfg.enable_cpu_offload,
            quantize_4bit=gen_cfg.quantize_4bit,
            logger=logger,
        )
    except Exception as exc:  # pragma: no cover - pipeline loading is environment-dependent
        logger.exception("Pipeline load failed: %s", exc)
        return 3

    try:
        final_state = generate_frames(
            pipe=pipe,
            initial_image=current_image,
            state=current_state,
            paths=paths,
            prompt=prompt,
            total_frames=total_frames,
            inference_steps=gen_cfg.inference_steps,
            true_cfg_scale=gen_cfg.true_cfg_scale,
            randomize_seed=gen_cfg.randomize_seed,
            fixed_seed=gen_cfg.fixed_seed,
            logger=logger,
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl+C). state.json reflects the last saved frame.")
        return 130
    except Exception as exc:
        logger.exception("Generation failed: %s", exc)
        return 4

    logger.info(
        "Done. %d/%d frames written to %s",
        final_state.last_completed_frame + 1,
        final_state.total_frames,
        paths.frames_dir,
    )
    return 0


def _parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orbit Decay — frame generation")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml (default: ./config.yaml)")
    parser.add_argument("--source", help="Override source_image from config (fresh runs only)")
    parser.add_argument("--run-name", help="Override run_name from config")
    parser.add_argument("--frames", type=int, help="Override total_frames from config")
    parser.add_argument("--rotate-degrees", type=float, help="Override per-step rotation (fresh runs only)")
    parser.add_argument("--limit", type=int, help="Cap this invocation at N frames (for scouting drift)")
    parser.add_argument("--fixed-seed", type=int, help="Force deterministic seed (overrides randomize_seed)")
    parser.add_argument("--randomize-seed", action="store_true", help="Force randomize_seed=true")
    parser.add_argument("--cpu-offload", action="store_true", help="Force CPU offload on")
    parser.add_argument("--no-cpu-offload", action="store_true", help="Disable CPU offload")
    parser.add_argument("--4bit", dest="quantize_4bit", action="store_true", help="Force 4-bit NF4 quantization of the transformer")
    parser.add_argument("--no-4bit", dest="no_quantize_4bit", action="store_true", help="Disable 4-bit quantization (use bf16)")
    parser.add_argument("--resume", action="store_true", help="Continue from an interrupted run")
    parser.add_argument("--from-frame", type=int, help="Fork from an existing frame index")
    parser.add_argument("--force", action="store_true", help="Ignore config-hash mismatch on resume")
    parser.add_argument("--dry-run", action="store_true", help="Load pipeline, generate 1 frame, exit")
    return parser.parse_args(argv)


def _apply_cli_overrides(gen_cfg: GenerationConfig, args: argparse.Namespace) -> GenerationConfig:
    overrides: dict = {}
    if args.source:
        overrides["source_image"] = args.source
    if args.run_name:
        overrides["run_name"] = args.run_name
    if args.frames is not None:
        overrides["total_frames"] = args.frames
    if args.rotate_degrees is not None:
        overrides["rotate_degrees"] = args.rotate_degrees
    if args.fixed_seed is not None:
        overrides["fixed_seed"] = args.fixed_seed
    if args.randomize_seed:
        overrides["randomize_seed"] = True
    if args.cpu_offload:
        overrides["enable_cpu_offload"] = True
    if args.no_cpu_offload:
        overrides["enable_cpu_offload"] = False
    if args.quantize_4bit:
        overrides["quantize_4bit"] = True
    if args.no_quantize_4bit:
        overrides["quantize_4bit"] = False
    if not overrides:
        return gen_cfg
    return with_overrides(gen_cfg, **overrides)


def _prepare_run(
    *,
    args: argparse.Namespace,
    config: Config,
    gen_cfg: GenerationConfig,
    paths: RunPaths,
    logger,
) -> tuple[State, Image.Image, int]:
    """Dispatch to fresh/resume/fork setup; returns (state, starting_image, total_frames)."""
    current_hash = config_hash(config)

    if args.resume and args.from_frame is not None:
        raise ConfigError("--resume and --from-frame are mutually exclusive")

    if args.resume:
        return _setup_resume(paths=paths, current_hash=current_hash, force=args.force, logger=logger)

    if args.from_frame is not None:
        return _setup_fork(
            paths=paths,
            from_frame=args.from_frame,
            gen_cfg=gen_cfg,
            current_hash=current_hash,
            logger=logger,
        )

    return _setup_fresh(
        config=config,
        gen_cfg=gen_cfg,
        paths=paths,
        current_hash=current_hash,
        dry_run=args.dry_run,
        limit=args.limit,
        logger=logger,
    )


def _setup_fresh(
    *,
    config: Config,
    gen_cfg: GenerationConfig,
    paths: RunPaths,
    current_hash: str,
    dry_run: bool,
    limit: Optional[int],
    logger,
) -> tuple[State, Image.Image, int]:
    source_path = Path(gen_cfg.source_image)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source image not found: {source_path}")

    source = Image.open(source_path).convert("RGB")
    width, height = compute_output_size(source, gen_cfg.longest_side)
    logger.info("Source %s -> output size %dx%d", source_path, width, height)

    prompt = bilingual_rotate_prompt(gen_cfg.rotate_degrees)
    total = gen_cfg.total_frames
    if dry_run:
        total = 1
        logger.info("--dry-run: capping run at 1 frame")
    elif limit is not None:
        total = min(total, max(1, limit))
        logger.info("--limit: capping this run at %d frames", total)

    state = new_state(
        source_image=str(source_path),
        run_name=gen_cfg.run_name,
        width=width,
        height=height,
        total_frames=total,
        prompt=prompt,
        lora_fuse_scale=gen_cfg.lora_fuse_scale,
        randomize_seed=gen_cfg.randomize_seed,
        fixed_seed=gen_cfg.fixed_seed,
        config_hash=current_hash,
    )
    save_state(paths.state_path, state)
    _snapshot_config(paths=paths, config=config)
    return state, source, total


def _setup_resume(
    *,
    paths: RunPaths,
    current_hash: str,
    force: bool,
    logger,
) -> tuple[State, Image.Image, int]:
    state = load_state(paths.state_path)
    _verify_hash(state=state, current_hash=current_hash, force=force, logger=logger)

    last = state.last_completed_frame
    if last < 0:
        # Run was interrupted before the first frame finished. Fall back to source.
        starting_image = Image.open(Path(state.source_image)).convert("RGB")
        logger.info("Resume: no frames yet; starting from source image.")
    else:
        frame_path = paths.frame_path(last)
        if not frame_path.is_file():
            raise StateError(
                f"state.json says frame {last} is the last completed frame, "
                f"but {frame_path} does not exist"
            )
        starting_image = Image.open(frame_path).convert("RGB")
        logger.info("Resume: starting from %s (frame %d)", frame_path, last)
    return state, starting_image, state.total_frames


def _setup_fork(
    *,
    paths: RunPaths,
    from_frame: int,
    gen_cfg: GenerationConfig,
    current_hash: str,
    logger,
) -> tuple[State, Image.Image, int]:
    state = load_state(paths.state_path)
    _verify_hash(state=state, current_hash=current_hash, force=True, logger=logger)

    frame_path = paths.frame_path(from_frame)
    if not frame_path.is_file():
        raise StateError(f"Cannot fork from frame {from_frame}: {frame_path} not found")

    truncated = truncate_for_fork(state, from_frame)
    forked = dc_replace(
        truncated,
        randomize_seed=gen_cfg.randomize_seed,
        fixed_seed=gen_cfg.fixed_seed,
        config_hash=current_hash,
    )
    save_state(paths.state_path, forked)
    logger.info(
        "Fork: starting from frame %d; overwriting frames %d..%d",
        from_frame,
        from_frame + 1,
        state.total_frames - 1,
    )
    _delete_frames_after(paths, from_frame, state.total_frames)
    starting_image = Image.open(frame_path).convert("RGB")
    return forked, starting_image, state.total_frames


def _verify_hash(*, state: State, current_hash: str, force: bool, logger) -> None:
    if state.config_hash == current_hash or not state.config_hash:
        return
    msg = (
        f"config_hash mismatch:\n"
        f"  state.json: {state.config_hash}\n"
        f"  current:    {current_hash}\n"
        f"config.yaml has changed since this run started."
    )
    if not force:
        logger.error(msg)
        raise StateError(msg + "  Re-run with --force to override.")
    logger.warning(msg + "  Continuing due to --force.")


def _snapshot_config(*, paths: RunPaths, config: Config) -> None:
    paths.config_snapshot_path.write_text(config.raw_text, encoding="utf-8")


def _delete_frames_after(paths: RunPaths, from_frame: int, total: int) -> None:
    for i in range(from_frame + 1, total):
        candidate = paths.frame_path(i)
        if candidate.exists():
            candidate.unlink()


if __name__ == "__main__":
    sys.exit(main())
