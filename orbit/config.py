"""Config loading, validation, hashing. Returns frozen dataclasses; never mutates."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

import yaml

VALID_CURVES = frozenset({"exponential", "linear", "custom"})


@dataclass(frozen=True)
class GenerationConfig:
    source_image: str
    run_name: str
    total_frames: int
    rotate_degrees: float
    lora_fuse_scale: float
    inference_steps: int
    true_cfg_scale: float
    longest_side: int
    randomize_seed: bool
    fixed_seed: Optional[int]
    enable_cpu_offload: bool
    quantize_4bit: bool


@dataclass(frozen=True)
class AssemblyConfig:
    start_hold_seconds: float
    end_fps: float
    curve: str
    custom_durations_csv: Optional[str]
    output_format: str
    output_codec: str
    output_crf: int


@dataclass(frozen=True)
class Config:
    generation: GenerationConfig
    assembly: AssemblyConfig
    raw_text: str  # original YAML text, used for hashing


class ConfigError(ValueError):
    """Raised when config is malformed or fails validation."""


def load_config(path: Path) -> Config:
    """Read YAML from disk and return a validated, frozen Config.

    Never mutates the source dict; validation is pure.
    """
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")

    raw_text = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"Top-level config must be a mapping, got {type(data).__name__}")

    generation = _build_generation(data)
    assembly = _build_assembly(data)
    return Config(generation=generation, assembly=assembly, raw_text=raw_text)


def _require(data: dict, key: str, expected_type: type) -> Any:
    if key not in data:
        raise ConfigError(f"Missing required config key: {key!r}")
    value = data[key]
    if not isinstance(value, expected_type):
        raise ConfigError(
            f"Config key {key!r} must be {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )
    return value


def _build_generation(data: dict) -> GenerationConfig:
    total_frames = _require(data, "total_frames", int)
    if total_frames < 1:
        raise ConfigError("total_frames must be >= 1")

    longest_side = _require(data, "longest_side", int)
    if longest_side < 64 or longest_side % 8 != 0:
        raise ConfigError("longest_side must be >= 64 and divisible by 8")

    inference_steps = _require(data, "inference_steps", int)
    if inference_steps < 1:
        raise ConfigError("inference_steps must be >= 1")

    fixed_seed_raw = data.get("fixed_seed", None)
    if fixed_seed_raw is not None and not isinstance(fixed_seed_raw, int):
        raise ConfigError("fixed_seed must be null or an integer")

    rotate = data.get("rotate_degrees", 15)
    if not isinstance(rotate, (int, float)):
        raise ConfigError("rotate_degrees must be a number")
    if rotate == 0:
        raise ConfigError("rotate_degrees must be nonzero (direction-less prompts unsupported)")

    return GenerationConfig(
        source_image=_require(data, "source_image", str),
        run_name=_require(data, "run_name", str),
        total_frames=total_frames,
        rotate_degrees=float(rotate),
        lora_fuse_scale=float(data.get("lora_fuse_scale", 1.25)),
        inference_steps=inference_steps,
        true_cfg_scale=float(data.get("true_cfg_scale", 1.0)),
        longest_side=longest_side,
        randomize_seed=bool(data.get("randomize_seed", True)),
        fixed_seed=fixed_seed_raw,
        enable_cpu_offload=bool(data.get("enable_cpu_offload", False)),
        quantize_4bit=bool(data.get("quantize_4bit", True)),
    )


def _build_assembly(data: dict) -> AssemblyConfig:
    curve = data.get("curve", "exponential")
    if curve not in VALID_CURVES:
        raise ConfigError(f"curve must be one of {sorted(VALID_CURVES)}, got {curve!r}")

    start_hold = data.get("start_hold_seconds", 5.0)
    end_fps = data.get("end_fps", 25)
    if not isinstance(start_hold, (int, float)) or start_hold <= 0:
        raise ConfigError("start_hold_seconds must be a positive number")
    if not isinstance(end_fps, (int, float)) or end_fps <= 0:
        raise ConfigError("end_fps must be a positive number")

    return AssemblyConfig(
        start_hold_seconds=float(start_hold),
        end_fps=float(end_fps),
        curve=curve,
        custom_durations_csv=data.get("custom_durations_csv"),
        output_format=str(data.get("output_format", "mp4")),
        output_codec=str(data.get("output_codec", "libx264")),
        output_crf=int(data.get("output_crf", 18)),
    )


def config_hash(config: Config) -> str:
    """SHA-256 of the raw YAML text, prefixed with 'sha256:'."""
    digest = hashlib.sha256(config.raw_text.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def with_overrides(
    generation: GenerationConfig,
    **overrides: Any,
) -> GenerationConfig:
    """Return a new GenerationConfig with fields replaced (immutable update)."""
    return replace(generation, **overrides)
