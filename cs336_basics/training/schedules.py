from __future__ import annotations

import math


def learning_rate_schedule(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int) -> float:
    if t < T_w:
        return alpha_max * (t / max(1, T_w))
    if t < T_c:
        return alpha_min + 0.5 * (1 + math.cos(math.pi * (t - T_w) / max(1, T_c - T_w))) * (alpha_max - alpha_min)
    return alpha_min


def linear_warmup_decay(step: int, alpha_max: float, warmup_steps: int, total_steps: int) -> float:
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))
    if warmup_steps > 0 and step <= warmup_steps:
        return alpha_max * (step / warmup_steps)
    if step >= total_steps:
        return 0.0
    decay_steps = max(1, total_steps - warmup_steps)
    progress = (step - warmup_steps) / decay_steps
    progress = min(max(progress, 0.0), 1.0)
    return alpha_max * (1.0 - progress)


def trapezoid_schedule(
    step: int,
    *,
    warmup_steps: int,
    start: float,
    peak: float,
    hold_steps: int,
    cooldown_steps: int,
) -> float:
    """Trapezoidal LR schedule.

    - Warmup: linear from `start` to `peak` over `warmup_steps`.
    - Hold: constant `peak` for `hold_steps`.
    - Cooldown: linear from `peak` to `start` over `cooldown_steps`.
    """
    t = max(0, int(step))
    if t <= warmup_steps and warmup_steps > 0:
        frac = t / max(1, warmup_steps)
        return start + (peak - start) * frac
    t -= warmup_steps
    if t <= hold_steps:
        return peak
    t -= hold_steps
    if cooldown_steps > 0:
        frac = min(1.0, t / cooldown_steps)
        return peak + (start - peak) * frac
    return start

