from __future__ import annotations

import math


def learning_rate_schedule(t: float, alpha_max: float, alpha_min: float, T_w: float, T_c: float) -> float:
    t = float(t)
    T_w = max(0.0, float(T_w))
    T_c = max(T_w + 1.0, float(T_c))
    if t < T_w and T_w > 0.0:
        return alpha_max * (t / max(1.0, T_w))
    if t < T_c:
        span = max(1.0, T_c - max(T_w, 0.0))
        return alpha_min + 0.5 * (1 + math.cos(math.pi * (t - T_w) / span)) * (alpha_max - alpha_min)
    return alpha_min


def linear_warmup_decay(step: float, alpha_max: float, warmup_steps: float, total_steps: float) -> float:
    step = float(step)
    total_steps = max(1.0, float(total_steps))
    warmup_steps = max(0.0, float(warmup_steps))
    if warmup_steps > 0.0 and step <= warmup_steps:
        return alpha_max * (step / warmup_steps)
    if step >= total_steps:
        return 0.0
    decay_steps = max(1.0, total_steps - warmup_steps)
    progress = (step - warmup_steps) / decay_steps
    progress = min(max(progress, 0.0), 1.0)
    return alpha_max * (1.0 - progress)


def trapezoid_schedule(
    step: float,
    *,
    warmup_steps: float,
    start: float,
    peak: float,
    hold_steps: float,
    cooldown_steps: float,
) -> float:
    """Trapezoidal LR schedule.

    - Warmup: linear from `start` to `peak` over `warmup_steps`.
    - Hold: constant `peak` for `hold_steps`.
    - Cooldown: linear from `peak` to `start` over `cooldown_steps`.
    """
    t = max(0.0, float(step))
    warmup_steps = max(0.0, float(warmup_steps))
    hold_steps = max(0.0, float(hold_steps))
    cooldown_steps = max(0.0, float(cooldown_steps))
    if warmup_steps > 0.0 and t <= warmup_steps:
        frac = t / max(1.0, warmup_steps)
        return start + (peak - start) * frac
    t -= warmup_steps
    if t <= hold_steps:
        return peak
    t -= hold_steps
    if cooldown_steps > 0.0:
        frac = min(1.0, t / max(1.0, cooldown_steps))
        return peak + (start - peak) * frac
    return start
