from cs336_basics.training.schedules import trapezoid_schedule, learning_rate_schedule, linear_warmup_decay


def test_trapezoid_basic():
    start = 0.006
    peak = 0.06
    warm = 100
    hold = 50
    cool = 100
    # start
    assert abs(trapezoid_schedule(0, warmup_steps=warm, start=start, peak=peak, hold_steps=hold, cooldown_steps=cool) - start) < 1e-9
    # mid-warm
    mid = trapezoid_schedule(50, warmup_steps=warm, start=start, peak=peak, hold_steps=hold, cooldown_steps=cool)
    assert start < mid < peak
    # hold plateau
    val = trapezoid_schedule(120, warmup_steps=warm, start=start, peak=peak, hold_steps=hold, cooldown_steps=cool)
    assert abs(val - peak) < 1e-9
    # cooldown
    end = trapezoid_schedule(300, warmup_steps=warm, start=start, peak=peak, hold_steps=hold, cooldown_steps=cool)
    assert abs(end - start) < 1e-9 or (start <= end <= peak)


def test_cosine_and_linear():
    # cosine should start near 0 at t=0 and reach alpha_min later
    v0 = learning_rate_schedule(0, 0.01, 0.001, 10, 100)
    v_mid = learning_rate_schedule(50, 0.01, 0.001, 10, 100)
    v_end = learning_rate_schedule(110, 0.01, 0.001, 10, 100)
    assert v0 == 0.0
    assert v_end == 0.001
    assert v_mid > v_end

    # linear warmup then decay to zero
    lw0 = linear_warmup_decay(0, 0.01, 10, 100)
    lw10 = linear_warmup_decay(10, 0.01, 10, 100)
    lw_end = linear_warmup_decay(100, 0.01, 10, 100)
    assert lw0 == 0.0
    assert abs(lw10 - 0.01) < 1e-9
    assert lw_end == 0.0
