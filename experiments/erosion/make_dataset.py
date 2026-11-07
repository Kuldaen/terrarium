"""Generate Seaglass training chains and evaluation ensembles.

Small (CPU smoke test):
  python experiments/erosion/make_dataset.py --chains 128 --starts 16 --futures 16
GPU scale (example):
  python experiments/erosion/make_dataset.py --chains 50000 --shard 2048 --starts 512 --futures 128 --device cuda
"""

from __future__ import annotations

import argparse
import time

from terrarium.phenomena.erosion.cliff2d.dataset import generate_eval, generate_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--chains", type=int, default=128)
    ap.add_argument("--years", type=float, default=10.0)
    ap.add_argument("--shard", type=int, default=64)
    ap.add_argument("--starts", type=int, default=16)
    ap.add_argument("--futures", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    t = time.time()
    p = generate_training(f"{a.out}/train", a.chains, years=a.years, shard_size=a.shard, seed=a.seed, device=a.device)
    print(f"training chains -> {p} ({time.time() - t:.1f}s)")
    t = time.time()
    p = generate_eval(f"{a.out}/eval", a.starts, a.futures, seed=a.seed + 10_000, device=a.device)
    print(f"eval ensembles  -> {p} ({time.time() - t:.1f}s)")


if __name__ == "__main__":
    main()
