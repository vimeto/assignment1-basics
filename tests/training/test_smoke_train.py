import os
from pathlib import Path

import numpy as np

from cs336_basics.training.configs import load_config
from cs336_basics.training.engine import train


def _ensure_tiny_data(tmpdir: Path):
    tmpdir.mkdir(parents=True, exist_ok=True)
    for name in ("tiny_train.npy", "tiny_val.npy"):
        p = tmpdir / name
        if not p.exists():
            # simple repeating tokens 0..99
            arr = (np.arange(4096, dtype=np.uint16) % 100).astype(np.uint16)
            np.save(p, arr)


def test_train_smoke(tmp_path):
    cfg = load_config(Path("configs/sanity.json"))
    _ensure_tiny_data(Path("data"))
    # run a few steps on CPU without raising
    train(cfg)

