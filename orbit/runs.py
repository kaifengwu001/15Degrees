"""Run directory layout and path helpers. No I/O side effects beyond mkdir."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

RUNS_ROOT = Path("runs")
OUTPUT_ROOT = Path("output")


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    frames_dir: Path
    state_path: Path
    config_snapshot_path: Path
    log_path: Path

    def frame_path(self, index: int) -> Path:
        return self.frames_dir / f"frame_{index:04d}.png"


def paths_for(run_name: str, runs_root: Path = RUNS_ROOT) -> RunPaths:
    """Return a RunPaths describing the canonical layout for a run (no mkdir)."""
    if not run_name or "/" in run_name or "\\" in run_name or ".." in run_name:
        raise ValueError(f"Invalid run_name: {run_name!r}")
    run_dir = runs_root / run_name
    return RunPaths(
        run_dir=run_dir,
        frames_dir=run_dir / "frames",
        state_path=run_dir / "state.json",
        config_snapshot_path=run_dir / "config.snapshot.yaml",
        log_path=run_dir / "generate.log",
    )


def ensure_run_dirs(paths: RunPaths) -> None:
    """Create the run and frames directories if they don't exist."""
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.frames_dir.mkdir(parents=True, exist_ok=True)


def ensure_output_dir(root: Path = OUTPUT_ROOT) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return root
