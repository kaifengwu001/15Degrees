"""Hold-duration curves mapping frame index -> seconds for the concat demuxer."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence, Tuple


def exponential_durations(
    *,
    n_frames: int,
    start_hold: float,
    end_hold: float,
) -> Tuple[float, ...]:
    """d(i) = start_hold * (end_hold/start_hold) ** (i/(N-1))."""
    _validate_holds(n_frames, start_hold, end_hold)
    if n_frames == 1:
        return (start_hold,)
    ratio = end_hold / start_hold
    denominator = n_frames - 1
    return tuple(start_hold * (ratio ** (i / denominator)) for i in range(n_frames))


def linear_durations(
    *,
    n_frames: int,
    start_hold: float,
    end_hold: float,
) -> Tuple[float, ...]:
    """Linear interpolation between start_hold and end_hold."""
    _validate_holds(n_frames, start_hold, end_hold)
    if n_frames == 1:
        return (start_hold,)
    step = (end_hold - start_hold) / (n_frames - 1)
    return tuple(start_hold + i * step for i in range(n_frames))


def load_custom_durations(csv_path: Path, n_frames: int) -> Tuple[float, ...]:
    """Load per-frame durations from a CSV file.

    Accepts either a single-column file (one duration per row) or a two-column
    file where the second column is the duration. Row count must match n_frames.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"Custom durations CSV not found: {csv_path}")

    durations: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row_index, row in enumerate(reader):
            if not row or all(cell.strip() == "" for cell in row):
                continue
            try:
                value = float(row[-1])
            except ValueError as exc:
                raise ValueError(
                    f"{csv_path}:{row_index + 1}: expected a number, got {row[-1]!r}"
                ) from exc
            if value <= 0:
                raise ValueError(
                    f"{csv_path}:{row_index + 1}: duration must be positive, got {value}"
                )
            durations.append(value)

    if len(durations) != n_frames:
        raise ValueError(
            f"Custom durations CSV has {len(durations)} entries but {n_frames} frames were provided"
        )
    return tuple(durations)


def slice_durations(
    durations: Sequence[float],
    frame_range: Tuple[int, int],
) -> Tuple[float, ...]:
    """Return durations[start:end_inclusive+1], with validation."""
    start, end = frame_range
    if start < 0 or end < start or end >= len(durations):
        raise ValueError(
            f"Invalid frame range ({start}, {end}) for {len(durations)} frames"
        )
    return tuple(durations[start : end + 1])


def _validate_holds(n_frames: int, start_hold: float, end_hold: float) -> None:
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    if start_hold <= 0:
        raise ValueError("start_hold must be positive")
    if end_hold <= 0:
        raise ValueError("end_hold must be positive")
