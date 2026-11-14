"""Оқыту скрипті.

  python -m spectra_ai.train --epochs 30                                  # лаборатория домені (200–900 нм)
  python -m spectra_ai.train --domain camera --epochs 30 --out checkpoints/camera_net.pt   # камера домені
  python -m spectra_ai.train --domain camera --epochs 8 --lr 1e-3 --init checkpoints/camera_net.pt   # жаңа домен/класқа дейін оқыту
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import SpectralAttentionNet
from .nist import CSV_PATH, load_database
from .synth import N_POINTS, T_MAX, T_MIN, SyntheticSpectra
from .synth_cam import CAM_N, CameraSpectra


def load_init(model, species, path):
    """Бұрынғы чекпойнттан бастау: түрлер аты бойынша сәйкестендіріліп, жаңа класс жолдары кездейсоқ қалады."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    old, sd = ck["species"], ck["state_dict"]
    cur = model.state_dict()
    for k, v in sd.items():
        if k.startswith("cls_head."):
            continue
        cur[k] = v
    w, b = cur["cls_head.weight"].clone(), cur["cls_head.bias"].clone()
    b[:] = -4.0                                       # жаңа класс: бастапқыда «жоқ»
    kept = 0
    for i, name in enumerate(species):
        if name in old:
            j = old.index(name)
            w[i], b[i] = sd["cls_head.weight"][j], sd["cls_head.bias"][j]
            kept += 1
    cur["cls_head.weight"], cur["cls_head.bias"] = w, b
    model.load_state_dict(cur)
    return kept


def save(path, model, species, m, a, n_points, epoch):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "species": species, "metrics": m, "domain": a.domain,
                "n_points": n_points, "num_layers": a.layers, "epoch": epoch}, path)


def unpack(batch):
    x, y, t, *rest = batch
    mask = rest[0] if rest else torch.ones_like(t)
    return x, y, t, mask


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = 0
    exact = n = 0
    t_err = t_n = 0.0
    empty_fp = n_empty = 0
    for batch in loader:
        x, y, t, mask = [b.to(device) for b in unpack(batch)]
        logits, t_pred = model(x)
        pred = (logits > 0).float()
        tp += (pred * y).sum().item()
        fp += (pred * (1 - y)).sum().item()
        fn += ((1 - pred) * y).sum().item()
        exact += (pred == y).all(1).sum().item()
        n += len(y)
        t_err += ((t_pred - t).abs() * mask).sum().item() * (T_MAX - T_MIN)
        t_n += mask.sum().item()
        empty = y.sum(1) == 0
        n_empty += empty.sum().item()
        empty_fp += ((pred.sum(1) > 0) & empty).sum().item()
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    out = {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9),
           "exact_match": exact / n, "T_MAE_K": t_err / max(t_n, 1)}
    if n_empty:
        out["empty_false_positive"] = empty_fp / n_empty
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["full", "camera"], default="full")
    ap.add_argument("--data", default=str(CSV_PATH))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--train-size", type=int, default=20000)
    ap.add_argument("--val-size", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--out", default="checkpoints/spectral_net.pt")
    ap.add_argument("--init", default=None, help="бастапқы чекпойнт (fine-tune)")
    a = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    db = load_database(a.data)
    cam = a.domain == "camera"
    ds_cls, n_points, t_weight = (CameraSpectra, CAM_N, 2.0) if cam else (SyntheticSpectra, N_POINTS, 5.0)
    train_ds = ds_cls(db, a.train_size, seed=1)
    val_ds = ds_cls(db, a.val_size, seed=999)  # тұрақты валидация
    species = train_ds.species
    print(f"Домен: {a.domain}, түрлер: {len(species)}, нүкте: {n_points}, құрылғы: {device}")

    tl = DataLoader(train_ds, a.batch, shuffle=True, num_workers=a.workers, persistent_workers=a.workers > 0)
    vl = DataLoader(val_ds, 256, num_workers=a.workers)
    model = SpectralAttentionNet(len(species), n_points=n_points, num_layers=a.layers).to(device)
    if a.init:
        print(f"Бастапқы чекпойнт: {a.init}, ортақ түрлер: {load_init(model, species, a.init)}/{len(species)}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(tl))

    for ep in range(a.epochs):
        model.train()
        train_ds.set_epoch(ep)  # әр эпохада жаңа синтетикалық спектрлер
        t0, tot = time.time(), 0.0
        for batch in tl:
            x, y, t, mask = [b.to(device) for b in unpack(batch)]
            logits, t_pred = model(x)
            t_loss = (((t_pred - t) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)
            loss = F.binary_cross_entropy_with_logits(logits, y) + t_weight * t_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        m = evaluate(model, vl, device)
        extra = f" empty-FP {m['empty_false_positive']:.3f}" if "empty_false_positive" in m else ""
        print(f"ep {ep+1:2d}/{a.epochs} loss {tot/len(tl):.4f} | F1 {m['f1']:.3f} "
              f"P {m['precision']:.3f} R {m['recall']:.3f} exact {m['exact_match']:.3f} "
              f"T-MAE {m['T_MAE_K']:.0f} K{extra} | {time.time()-t0:.0f}s", flush=True)
        save(a.out + ".last", model, species, m, a, n_points, ep + 1)   # әр эпохада аралық сақтау

    save(a.out, model, species, m, a, n_points, a.epochs)
    print("Сақталды:", a.out)


if __name__ == "__main__":
    main()
