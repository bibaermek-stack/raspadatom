"""Оқыту скрипті:  python -m spectra_ai.train --epochs 15"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import SpectralAttentionNet
from .nist import CSV_PATH, load_database
from .synth import T_MAX, T_MIN, SyntheticSpectra


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = 0
    exact = n = 0
    t_err = 0.0
    for x, y, t in loader:
        x, y, t = x.to(device), y.to(device), t.to(device)
        logits, t_pred = model(x)
        pred = (logits > 0).float()
        tp += (pred * y).sum().item()
        fp += (pred * (1 - y)).sum().item()
        fn += ((1 - pred) * y).sum().item()
        exact += (pred == y).all(1).sum().item()
        n += len(y)
        t_err += ((t_pred - t).abs() * (T_MAX - T_MIN)).sum().item()
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9),
            "exact_match": exact / n, "T_MAE_K": t_err / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(CSV_PATH))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--train-size", type=int, default=20000)
    ap.add_argument("--val-size", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="checkpoints/spectral_net.pt")
    a = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    db = load_database(a.data)
    train_ds = SyntheticSpectra(db, a.train_size, seed=1)
    val_ds = SyntheticSpectra(db, a.val_size, seed=999)  # тұрақты валидация
    species = train_ds.species
    print(f"Түрлер: {len(species)}, құрылғы: {device}")

    tl = DataLoader(train_ds, a.batch, shuffle=True, num_workers=a.workers, persistent_workers=a.workers > 0)
    vl = DataLoader(val_ds, 256, num_workers=a.workers)
    model = SpectralAttentionNet(len(species)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(tl))

    for ep in range(a.epochs):
        model.train()
        train_ds.set_epoch(ep)  # әр эпохада жаңа синтетикалық спектрлер
        t0, tot = time.time(), 0.0
        for x, y, t in tl:
            x, y, t = x.to(device), y.to(device), t.to(device)
            logits, t_pred = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y) + 5.0 * F.mse_loss(t_pred, t)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        m = evaluate(model, vl, device)
        print(f"ep {ep+1:2d}/{a.epochs} loss {tot/len(tl):.4f} | F1 {m['f1']:.3f} "
              f"P {m['precision']:.3f} R {m['recall']:.3f} exact {m['exact_match']:.3f} "
              f"T-MAE {m['T_MAE_K']:.0f} K | {time.time()-t0:.0f}s", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "species": species, "metrics": m}, a.out)
    print("Сақталды:", a.out)


if __name__ == "__main__":
    main()
