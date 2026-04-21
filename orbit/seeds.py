"""Seed selection policy. Mirrors the HF Space `randomize_seed` semantics."""

from __future__ import annotations

import random
from typing import Optional

MAX_SEED = (1 << 31) - 1  # numpy int32 max, same ceiling the Space uses


def pick_seed(*, randomize: bool, fixed_seed: Optional[int]) -> int:
    """Pick a seed for a single frame.

    - If `fixed_seed` is set, return it (deterministic branch).
    - Else if `randomize` is True, draw a fresh random seed (Space default).
    - Else return 0 (same behavior as the Space when both flags are off).
    """
    if fixed_seed is not None:
        return int(fixed_seed) & MAX_SEED
    if randomize:
        return random.randint(0, MAX_SEED)
    return 0
