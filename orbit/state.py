"""state.json read/write. All updates return new State objects — never mutate."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class State:
    source_image: str
    run_name: str
    width: int
    height: int
    total_frames: int
    last_completed_frame: int  # -1 before any frame has been saved
    prompt: str
    lora_fuse_scale: float
    randomize_seed: bool
    fixed_seed: Optional[int]
    frame_seeds: Tuple[int, ...] = field(default_factory=tuple)
    config_hash: str = ""


class StateError(ValueError):
    """Raised when state.json is malformed or inconsistent."""


def new_state(
    *,
    source_image: str,
    run_name: str,
    width: int,
    height: int,
    total_frames: int,
    prompt: str,
    lora_fuse_scale: float,
    randomize_seed: bool,
    fixed_seed: Optional[int],
    config_hash: str,
) -> State:
    return State(
        source_image=source_image,
        run_name=run_name,
        width=width,
        height=height,
        total_frames=total_frames,
        last_completed_frame=-1,
        prompt=prompt,
        lora_fuse_scale=lora_fuse_scale,
        randomize_seed=randomize_seed,
        fixed_seed=fixed_seed,
        frame_seeds=(),
        config_hash=config_hash,
    )


def load_state(path: Path) -> State:
    if not path.is_file():
        raise StateError(f"state.json not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StateError(f"state.json is not valid JSON: {exc}") from exc
    return _from_dict(data)


def save_state(path: Path, state: State) -> None:
    """Write atomically — temp file + replace — so interrupts never leave corrupt JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _to_dict(state)
    payload = json.dumps(data, indent=2, ensure_ascii=False)

    fd, tmp_name = tempfile.mkstemp(
        prefix=".state.", suffix=".json.tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


def record_frame(state: State, frame_index: int, seed: int) -> State:
    """Return a new State with the given frame recorded as completed.

    Enforces append-only for sequential runs and truncate-then-append for forks.
    """
    if frame_index < 0:
        raise StateError(f"frame_index must be >= 0, got {frame_index}")

    # Truncate any seeds beyond frame_index (happens on fork / --from-frame).
    preserved_seeds = state.frame_seeds[:frame_index]
    padding_length = frame_index - len(preserved_seeds)
    if padding_length > 0:
        # Shouldn't normally happen; guard against silent corruption.
        raise StateError(
            f"Cannot record frame {frame_index}: state has only "
            f"{len(state.frame_seeds)} seeds recorded (gap of {padding_length})"
        )

    new_seeds = preserved_seeds + (int(seed),)
    return replace(
        state,
        last_completed_frame=frame_index,
        frame_seeds=new_seeds,
    )


def truncate_for_fork(state: State, from_frame: int) -> State:
    """Return a new State representing a fork starting *from* frame `from_frame`.

    Frame `from_frame` itself is preserved as the input for the next step.
    Seeds and last_completed_frame are truncated so frame `from_frame + 1` will be
    the next generated frame.
    """
    if from_frame < 0:
        raise StateError(f"from_frame must be >= 0, got {from_frame}")
    if from_frame > state.last_completed_frame:
        raise StateError(
            f"Cannot fork from frame {from_frame}: only frames 0..{state.last_completed_frame} exist"
        )
    preserved_seeds = state.frame_seeds[: from_frame + 1]
    return replace(
        state,
        last_completed_frame=from_frame,
        frame_seeds=preserved_seeds,
    )


def _to_dict(state: State) -> dict:
    return {
        "source_image": state.source_image,
        "run_name": state.run_name,
        "width": state.width,
        "height": state.height,
        "total_frames": state.total_frames,
        "last_completed_frame": state.last_completed_frame,
        "prompt": state.prompt,
        "lora_fuse_scale": state.lora_fuse_scale,
        "randomize_seed": state.randomize_seed,
        "fixed_seed": state.fixed_seed,
        "frame_seeds": list(state.frame_seeds),
        "config_hash": state.config_hash,
    }


def _from_dict(data: dict) -> State:
    try:
        frame_seeds_raw: List[int] = data.get("frame_seeds", [])
        return State(
            source_image=str(data["source_image"]),
            run_name=str(data["run_name"]),
            width=int(data["width"]),
            height=int(data["height"]),
            total_frames=int(data["total_frames"]),
            last_completed_frame=int(data["last_completed_frame"]),
            prompt=str(data["prompt"]),
            lora_fuse_scale=float(data["lora_fuse_scale"]),
            randomize_seed=bool(data["randomize_seed"]),
            fixed_seed=data.get("fixed_seed"),
            frame_seeds=tuple(int(s) for s in frame_seeds_raw),
            config_hash=str(data.get("config_hash", "")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise StateError(f"state.json is missing or has invalid fields: {exc}") from exc
