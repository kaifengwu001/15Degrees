"""Stage 2 CLI: assemble generated frames into a video with an accelerating hold curve.

Usage examples:
    python assemble.py --run-name portrait_v1
    python assemble.py --run-name portrait_v1 --start-hold 5.0 --end-fps 25
    python assemble.py --run-name portrait_v1 --curve linear
    python assemble.py --run-name portrait_v1 --range 0 150
    python assemble.py --run-name portrait_v1 --preview
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Optional, Tuple

from orbit.concat import run_ffmpeg_concat, write_concat_file
from orbit.config import AssemblyConfig, ConfigError, load_config
from orbit.curves import (
    exponential_durations,
    linear_durations,
    load_custom_durations,
)
from orbit.logging_setup import configure_logger
from orbit.runs import RunPaths, ensure_output_dir, paths_for


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    try:
        config = load_config(Path(args.config))
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    run_name = args.run_name or config.generation.run_name
    paths = paths_for(run_name)
    if not paths.frames_dir.is_dir():
        print(f"ERROR: frames directory not found: {paths.frames_dir}", file=sys.stderr)
        return 2

    output_root = ensure_output_dir()
    logger = configure_logger(paths.log_path.with_name("assemble.log"), name="orbit.assemble")
    logger.info("=" * 72)
    logger.info("orbit assemble.py starting — run=%s", run_name)

    assembly = _apply_cli_overrides(config.assembly, args)
    try:
        frame_paths, durations = _collect_frames_and_durations(
            paths=paths,
            assembly=assembly,
            frame_range=_parse_range(args.range),
            logger=logger,
        )
    except (ValueError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 2

    concat_path = paths.run_dir / "concat.txt"
    write_concat_file(
        frame_paths=frame_paths,
        durations=durations,
        concat_path=concat_path,
    )
    logger.info("Wrote concat list: %s (%d frames)", concat_path, len(frame_paths))

    suffix = "_preview" if args.preview else ""
    output_path = output_root / f"{run_name}{suffix}.{assembly.output_format}"
    try:
        run_ffmpeg_concat(
            concat_path=concat_path,
            output_path=output_path,
            end_fps=assembly.end_fps,
            codec=assembly.output_codec,
            crf=assembly.output_crf,
            preview=args.preview,
            logger=logger,
        )
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 4

    logger.info("Wrote video: %s", output_path)
    return 0


def _parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orbit Decay — video assembly")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--run-name", help="Override run_name from config")
    parser.add_argument("--start-hold", type=float, help="Override start_hold_seconds")
    parser.add_argument("--end-fps", type=float, help="Override end_fps")
    parser.add_argument("--curve", choices=["exponential", "linear", "custom"], help="Override curve")
    parser.add_argument(
        "--range",
        nargs=2,
        metavar=("START", "END"),
        help="Assemble a subset of frames [START END] inclusive",
    )
    parser.add_argument("--preview", action="store_true", help="720p quick encode")
    return parser.parse_args(argv)


def _apply_cli_overrides(assembly: AssemblyConfig, args: argparse.Namespace) -> AssemblyConfig:
    overrides: dict = {}
    if args.start_hold is not None:
        overrides["start_hold_seconds"] = args.start_hold
    if args.end_fps is not None:
        overrides["end_fps"] = args.end_fps
    if args.curve is not None:
        overrides["curve"] = args.curve
    if not overrides:
        return assembly
    return dc_replace(assembly, **overrides)


def _parse_range(value: Optional[list[str]]) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    try:
        start = int(value[0])
        end = int(value[1])
    except ValueError as exc:
        raise ValueError(f"--range expects two integers, got {value}") from exc
    if start < 0 or end < start:
        raise ValueError(f"--range requires 0 <= START <= END; got {start} {end}")
    return start, end


def _collect_frames_and_durations(
    *,
    paths: RunPaths,
    assembly: AssemblyConfig,
    frame_range: Optional[Tuple[int, int]],
    logger: logging.Logger,
) -> Tuple[list[Path], Tuple[float, ...]]:
    all_frames = sorted(paths.frames_dir.glob("frame_*.png"))
    if not all_frames:
        raise FileNotFoundError(f"No frame_*.png files in {paths.frames_dir}")

    if frame_range is None:
        selected = all_frames
        logger.info("Using all %d frames.", len(selected))
    else:
        start, end = frame_range
        if end >= len(all_frames):
            raise ValueError(
                f"--range END ({end}) exceeds available frames ({len(all_frames) - 1})"
            )
        selected = all_frames[start : end + 1]
        logger.info("Using frames %d..%d (%d frames).", start, end, len(selected))

    durations = _durations_for(assembly, len(selected), frame_range, logger)
    return selected, durations


def _durations_for(
    assembly: AssemblyConfig,
    n_frames: int,
    frame_range: Optional[Tuple[int, int]],
    logger: logging.Logger,
) -> Tuple[float, ...]:
    end_hold = 1.0 / assembly.end_fps

    if assembly.curve == "custom":
        if not assembly.custom_durations_csv:
            raise ValueError("curve=custom requires custom_durations_csv in config")
        durations = load_custom_durations(Path(assembly.custom_durations_csv), n_frames)
        logger.info("Loaded %d custom durations.", len(durations))
        return durations

    start_hold = assembly.start_hold_seconds
    if assembly.curve == "linear":
        durations = linear_durations(n_frames=n_frames, start_hold=start_hold, end_hold=end_hold)
    else:
        durations = exponential_durations(
            n_frames=n_frames, start_hold=start_hold, end_hold=end_hold
        )
    logger.info(
        "Curve=%s start=%.3fs end=%.3fs (%.1f fps) over %d frames",
        assembly.curve,
        start_hold,
        end_hold,
        assembly.end_fps,
        n_frames,
    )
    _ = frame_range  # curve is computed over len(selected), not absolute indices
    return durations


if __name__ == "__main__":
    sys.exit(main())
