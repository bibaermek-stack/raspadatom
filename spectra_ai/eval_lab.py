"""Лаборатория моделін бағалау: README хаттамасы + сенім аралықтары, түр саны / түр бойынша бөлу, калибровка, k > 3 стресс-тесті.

  python -m spectra_ai.eval_lab [--n 4000] [--seed 20261116] [--checkpoint checkpoints/spectral_net.pt]
                                [--previous алдыңғы_модель.pt] [--stress-max-species 8]

Хаттама: оқыту (seed 1 / 2) мен валидацияға (seed 999) кірмеген жаңа синтетикалық спектрлер, әр спектрде 1–3 түр.
`--previous` берілсе, екі модель бір таңдамамен өлшенеді де, айырманың жұптасқан ДИ-ы шығады.
Стресс-тест: 1..stress-max-species түрден тұратын қоспалар, нәтиже түр саны бойынша бөлінеді.
Синтетикалық сандар — оқыту генераторымен бір таралымда; нақты спектрлер үшін `spectra_ai.eval_libs_species`.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .metrics import bootstrap, counts, ece, prf, summarize
from .model import SpectralAttentionNet
from .nist import load_database
from .synth import N_POINTS, T_MAX, T_MIN, SyntheticSpectra, make_spectrum


def load(path, species):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck["species"] != species:
        raise ValueError(f"{path}: түрлер тізімі базадағыдан өзгеше")
    model = SpectralAttentionNet(len(species), n_points=ck.get("n_points", N_POINTS), num_layers=ck.get("num_layers", 2))
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck


@torch.no_grad()
def predict(model, X, batch=256):
    out = [model(X[i:i + batch]) for i in range(0, len(X), batch)]
    return torch.cat([o[0] for o in out]).numpy(), torch.cat([o[1] for o in out]).numpy()


def protocol_set(db, n, seed):
    """README хаттамасы: SyntheticSpectra, 0-эпоха (оқытуда seed 1/2 және эпоха нөмірімен тәуелсіз ағын)."""
    ds = SyntheticSpectra(db, n, seed=seed)
    ds.set_epoch(0)
    items = [ds[i] for i in range(n)]
    return (torch.stack([it[0] for it in items]), np.stack([it[1].numpy() for it in items]) > 0,
            np.array([float(it[2]) for it in items]))


def stress_set(db, species, n, seed, max_species):
    X, Y, T = [], [], []
    for i in range(n):
        s, y, t = make_spectrum(db, species, np.random.default_rng([seed + 1, i]), max_species=max_species)
        X.append(s), Y.append(y), T.append(t)
    return torch.from_numpy(np.stack(X))[:, None], np.stack(Y) > 0, np.array(T)


def run(model, X, Y, T):
    logits, t_pred = predict(model, X)
    c = counts(logits > 0, Y)
    return logits, c, np.abs(t_pred - T) * (T_MAX - T_MIN)


def by_k(c, terr, k_true, ks):
    out = {}
    for k in ks:
        idx = np.flatnonzero(k_true == k)
        if len(idx):
            out[str(k)] = {**summarize(c, idx), "T_MAE_K": float(terr[idx].mean())}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--checkpoint", default="checkpoints/spectral_net.pt")
    ap.add_argument("--previous", default=None, help="алдыңғы модель (жұптасқан салыстыру үшін)")
    ap.add_argument("--stress-max-species", type=int, default=8, help="0 — стресс-тестті өткізіп жіберу")
    ap.add_argument("--stress-n", type=int, default=2400)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    db = load_database()
    species = sorted(db)
    model, ck = load(a.checkpoint, species)
    X, Y, T = protocol_set(db, a.n, a.seed)
    logits, c, terr = run(model, X, Y, T)
    models = {"current": c}
    extras = {"current": {"T_MAE_K": terr}}
    out = {
        "protocol": f"{a.n} жаңа синтетикалық спектр, seed {a.seed}, k = 1..3 түр",
        "checkpoint": {"file": a.checkpoint, "epoch": ck.get("epoch"), "max_species_trained": ck.get("max_species", 3),
                       "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)},
        "test": {**summarize(c), "T_MAE_K": float(terr.mean()),
                 "T_MAE_K_if_predicting_mean": float(np.abs(T - T.mean()).mean() * (T_MAX - T_MIN))},
    }
    prob = 1.0 / (1.0 + np.exp(-logits))
    out["calibration"] = {"ece_15_bins": ece(prob, Y), "brier": float(((prob - Y) ** 2).mean())}
    if a.previous:
        prev, _ = load(a.previous, species)
        _, c_prev, terr_prev = run(prev, X, Y, T)
        models = {"previous": c_prev, "current": c}
        extras = {"previous": {"T_MAE_K": terr_prev}, "current": {"T_MAE_K": terr}}
        out["previous"] = {**summarize(c_prev), "T_MAE_K": float(terr_prev.mean())}
    out["ci95"] = bootstrap(models, extras, n_boot=a.n_boot)
    k_true = Y.sum(1).astype(int)
    out["by_k"] = by_k(c, terr, k_true, (1, 2, 3))

    pred = logits > 0
    tp, fp, fn = (pred & Y).sum(0), (pred & ~Y).sum(0), (~pred & Y).sum(0)
    per = []
    for j, sp in enumerate(species):
        p, r, f1 = prf(tp[j], fp[j], fn[j])
        per.append({"species": sp, "support": int(Y[:, j].sum()), "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)})
    out["per_species"] = sorted(per, key=lambda r: r["f1"])
    out["macro_f1"] = float(np.mean([r["f1"] for r in per]))

    if a.stress_max_species > 3:
        Xs, Ys, Ts = stress_set(db, species, a.stress_n, a.seed, a.stress_max_species)
        _, cs, terr_s = run(model, Xs, Ys, Ts)
        out["stress"] = {"max_species": a.stress_max_species, "n": a.stress_n,
                         "by_k": by_k(cs, terr_s, Ys.sum(1).astype(int), range(1, a.stress_max_species + 1))}
        if a.previous:
            _, cs_prev, terr_prev_s = run(prev, Xs, Ys, Ts)
            out["stress"]["previous_by_k"] = by_k(cs_prev, terr_prev_s, Ys.sum(1).astype(int), range(1, a.stress_max_species + 1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
