"""Time-budgeted fine-tuning of the detector (MPS if available, else CPU)."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import DFireDataset, collate, split_items
from .model import CKPT, build_model, pick_device


def train(
    minutes: float = 35.0,
    batch_size: int = 32,
    lr: float = 2e-3,
    seed: int = 0,
    resume: bool = False,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = pick_device()
    items = split_items("train")
    print(f"train images: {len(items)}  device: {device}")
    dl = DataLoader(
        DFireDataset(items, train=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        collate_fn=collate,
        drop_last=True,
        persistent_workers=True,
    )
    model = build_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    budget, t0, step, hist = minutes * 60, time.time(), 0, []
    if resume and CKPT.exists():
        ck = torch.load(CKPT, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, hist = ck["step"], ck["hist"]
        t0 -= ck["elapsed"]
    CKPT.parent.mkdir(exist_ok=True)
    model.train()
    done = False
    while not done:
        for imgs, tgts in dl:
            frac = (time.time() - t0) / budget
            if frac >= 1:
                done = True
                break
            warm = min(1.0, (step + 1) / 100)
            for g in opt.param_groups:
                g["lr"] = lr * warm * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * frac)))
            imgs = [i.to(device) for i in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
            losses = model(imgs, tgts)
            loss = sum(losses.values())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            step += 1
            if step % 20 == 0:
                hist.append({"step": step, "t": time.time() - t0, "loss": float(loss)})
                print(f"step {step} {frac:5.1%} loss {float(loss):.3f}", flush=True)
            if step % 200 == 0:
                _save(model, opt, step, hist, time.time() - t0)
    _save(model, opt, step, hist, time.time() - t0)
    Path("runs").mkdir(exist_ok=True)
    Path("runs/train_log.json").write_text(
        json.dumps(
            {
                "steps": step,
                "minutes": (time.time() - t0) / 60,
                "batch_size": batch_size,
                "images_seen": step * batch_size,
                "device": str(device),
                "hist": hist,
            }
        )
    )
    print("done:", step, "steps")


def _save(model, opt, step, hist, elapsed):
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "step": step,
            "hist": hist,
            "elapsed": elapsed,
        },
        CKPT,
    )
