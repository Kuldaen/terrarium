"""Train the Seaglass flow-matching surrogate.

Runs on CUDA (fp16/bf16 autocast), Apple MPS (fp32) or CPU — auto-detected.

  # MacBook (M2): small model, quick iteration
  python experiments/erosion/train_surrogate.py --data data/train --out runs/mac --steps 20000 --batch 32
  # RTX 2060 (6 GB): default model, fp16
  python experiments/erosion/train_surrogate.py --data data/train --out runs/2060 --steps 100000 --batch 64 --workers 6
  # "survey-only" setting: geology hidden from the model
  python experiments/erosion/train_surrogate.py ... --no-strength

Then score it:
  python experiments/erosion/calibration_report.py --train data/train --eval data/eval \
      --surrogate runs/2060/last.pt --out outputs/calib_surrogate
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from terrarium.core.torch_utils import autocast, describe, make_scaler, pick_device
from terrarium.phenomena.erosion.cliff2d.dataset import PairDataset
from terrarium.phenomena.erosion.cliff2d.surrogate import FlowSurrogate, SurrogateConfig, cond_images, cond_vector, encode_target


class TorchPairs(Dataset):
    def __init__(self, pairs: PairDataset, use_strength: bool):
        self.p, self.use_strength = pairs, use_strength

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        it = self.p[i]
        r0, r1 = it["rock0"][None], it["rock1"][None]
        return {
            "cimg": cond_images(r0, it["strength"][None] if self.use_strength else None)[0],
            "cvec": cond_vector(it["summary"][None], np.array([it["dt_days"]]), np.array([it["rev_crest"]]), np.array([it["talus0"]]))[0],
            "x1": encode_target(r0, r1, np.array([it["talus1"]]))[0],
        }


def cond_stats(pairs: PairDataset, n: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    b = pairs.sample_stratified(max(1, n // len(pairs.dts)), np.random.default_rng(0))
    cv = cond_vector(b["summary"], b["dt_days"], b["rev_crest"], b["talus0"])
    return cv.mean(0), cv.std(0) + 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--out", default="runs/default")
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base-ch", type=int, default=32)
    ap.add_argument("--no-strength", action="store_true", help="hide the strength field (survey-only setting)")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = pick_device(a.device)

    pairs = PairDataset(a.data)
    cfg = SurrogateConfig(base_ch=a.base_ch, use_strength=not a.no_strength)
    mean, std = cond_stats(pairs)
    model = FlowSurrogate(cfg, mean, std, dev)
    ema = copy.deepcopy(model.net).eval()
    for p_ in ema.parameters():
        p_.requires_grad_(False)
    n_params = sum(p_.numel() for p_ in model.net.parameters())
    print(f"device {describe(dev)} | {len(pairs):,} pairs | {n_params / 1e6:.2f}M params | strength={'on' if cfg.use_strength else 'off'}")

    ds = TorchPairs(pairs, cfg.use_strength)
    dl = DataLoader(
        ds,
        batch_size=a.batch,
        sampler=RandomSampler(ds, replacement=True, num_samples=a.steps * a.batch),
        num_workers=a.workers,
        pin_memory=dev.type == "cuda",
        persistent_workers=a.workers > 0,
        drop_last=True,
    )
    opt = torch.optim.AdamW(model.net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.02)
    scaler = make_scaler(dev)
    (out / "args.json").write_text(json.dumps(vars(a), indent=2))

    def save(name):
        sd = model.state_dict()
        sd["ema"] = ema.state_dict()
        sd["step"] = step
        torch.save(sd, out / name)

    t0, run = time.time(), 0.0
    model.net.train()
    step = 0
    for step, b in enumerate(dl, 1):
        cimg = b["cimg"].to(dev, non_blocking=True)
        cvec = ((b["cvec"].to(dev) - model.cond_mean) / model.cond_std).float()
        x1 = b["x1"].to(dev, non_blocking=True)
        with autocast(dev):
            loss = model.loss(x1, cimg, cvec)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.net.parameters()):
                pe.lerp_(pm, 1 - a.ema)
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % a.log_every == 0:
            el = time.time() - t0
            print(f"step {step:>7} | loss {run:.4f} | {step * a.batch / el:,.0f} samples/s | {el / 60:.1f} min", flush=True)
        if step % a.save_every == 0:
            save("last.pt")
    save("last.pt")
    print(f"done: {out / 'last.pt'}")


if __name__ == "__main__":
    main()
