"""ffmpeg concat demuxer generation + invocation."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Sequence


def write_concat_file(
    *,
    frame_paths: Sequence[Path],
    durations: Sequence[float],
    concat_path: Path,
) -> None:
    """Write an ffmpeg concat demuxer file.

    The last frame is listed twice (with no trailing duration) because the concat
    demuxer's `duration` directive applies to the NEXT file — without the duplicate
    entry the final frame gets dropped.
    """
    if len(frame_paths) != len(durations):
        raise ValueError(
            f"frame_paths ({len(frame_paths)}) and durations ({len(durations)}) must match"
        )
    if not frame_paths:
        raise ValueError("At least one frame is required")

    concat_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for frame_path, duration in zip(frame_paths, durations):
        abs_path = frame_path.resolve().as_posix()
        lines.append(f"file '{_escape_single_quotes(abs_path)}'")
        lines.append(f"duration {duration:.6f}")

    # Re-list the final frame so it isn't dropped (ffmpeg concat demuxer quirk).
    last_abs = frame_paths[-1].resolve().as_posix()
    lines.append(f"file '{_escape_single_quotes(last_abs)}'")

    concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ffmpeg_concat(
    *,
    concat_path: Path,
    output_path: Path,
    end_fps: float,
    codec: str,
    crf: int,
    preview: bool,
    logger: logging.Logger,
) -> None:
    """Invoke ffmpeg to encode the concat demuxer stream into a video file.

    `fps=<end_fps>` resamples the variable-duration stream to a constant output rate.
    `preview=True` downscales to 720p for a quick check.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    video_filters = [f"fps={end_fps}"]
    if preview:
        video_filters.append("scale=-2:720")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-vf",
        ",".join(video_filters),
        "-c:v",
        codec,
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        str(output_path),
    ]
    logger.info("Running: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg was not found on PATH. Install it (e.g. `winget install ffmpeg` on Windows) "
            "and re-run."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed with exit code {exc.returncode}") from exc


def _escape_single_quotes(text: str) -> str:
    """ffmpeg concat demuxer needs single quotes in paths escaped as '\''."""
    return text.replace("'", "'\\''")
