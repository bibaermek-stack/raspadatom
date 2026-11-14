"""Камера моделін бағалау: жалпы метрика, шарттар бойынша бөлу, лаборатория моделімен салыстыру, тәуелсіз қолмен жасалған көздер.

  python -m spectra_ai.eval_cam [--n 3000] [--checkpoint checkpoints/camera_net.pt] [--previous eski_camera_net.pt]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .model import SpectralAttentionNet
from .nist import load_database
from .synth import GRID, N_POINTS
from .synth_cam import CAM_EXCLUDE, CAM_GRID, CAM_N, make_camera_spectrum


def load(path, species=None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = SpectralAttentionNet(len(ck["species"]), n_points=ck.get("n_points", N_POINTS), num_layers=ck.get("num_layers", 2))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck["species"]


def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / max(p + r, 1e-9), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--checkpoint", default="checkpoints/camera_net.pt")
    ap.add_argument("--baseline", default="checkpoints/spectral_net.pt")
    ap.add_argument("--previous", default=None, help="алдыңғы камера моделі (ортақ атомдық түрлер бойынша салыстыру)")
    a = ap.parse_args()

    db = load_database()
    cam, species = load(a.checkpoint)
    base, base_species = load(a.baseline)
    base_cols = [j for j, s in enumerate(species) if s in base_species]       # екі модельдің ортақ атомдық түрлері («CFL» көз класы лабораториялық модельде жоқ)
    base_idx = [base_species.index(species[j]) for j in base_cols]

    X, Y, META = [], [], []
    for i in range(a.n):
        s, y, _, _, meta = make_camera_spectrum(db, species, np.random.default_rng([4242, i]), return_meta=True)
        X.append(s); Y.append(y); META.append(meta)
    X = torch.from_numpy(np.array(X))[:, None]
    Y = np.array(Y)
    with torch.no_grad():
        pc = (cam(X)[0] > 0).numpy().astype(float)
        full = np.stack([np.interp(GRID, CAM_GRID, x[0], left=0, right=0) for x in X.numpy()]).astype(np.float32)
        full /= full.max(1, keepdims=True) + 1e-8
        pb = np.zeros_like(pc)
        pb[:, base_cols] = (base(torch.from_numpy(full)[:, None])[0] > 0).numpy().astype(float)[:, base_idx]   # CFL үшін 0 (класы жоқ)
        if a.previous:
            prev, prev_species = load(a.previous)
            pp = np.zeros_like(pc)
            cols = [j for j, s in enumerate(species) if s in prev_species]
            pp[:, cols] = (prev(X)[0] > 0).numpy().astype(float)[:, [prev_species.index(species[j]) for j in cols]]

    def agg(mask, pred):
        t = pred[mask]; y = Y[mask]
        return prf((t * y).sum(), (t * (1 - y)).sum(), ((1 - t) * y).sum())

    fw = np.array([m["fwhm_nm"] for m in META]); ce = np.array([m["cal_err_nm"] for m in META]); k = np.array([m["k"] for m in META])
    has = k > 0
    out = {"n": a.n, "camera_model": agg(np.ones(a.n, bool), pc), "lab_model_on_camera_spectra": agg(np.ones(a.n, bool), pb),
           "empty_false_positive": round(float(((pc.sum(1) > 0) & ~has)[~has].mean()), 3)}
    if a.previous:
        out["previous_camera_model"] = agg(np.ones(a.n, bool), pp)
    cfl = np.array([m["cfl"] for m in META])
    if "CFL" in species:
        j = species.index("CFL")
        out["cfl"] = {"recall": round(float(pc[(Y[:, j] > 0), j].mean()), 3), "false_positive_rate": round(float(pc[~(Y[:, j] > 0), j].mean()), 4),
                      "n_labelled": int((Y[:, j] > 0).sum()), "n_cfl_samples": int(cfl.sum())}
    out["by_fwhm_nm"] = {f"{lo}-{hi}": agg((fw >= lo) & (fw < hi) & has, pc) for lo, hi in [(1.5, 3), (3, 6), (6, 12), (12, 30)]}
    out["by_calibration_error_nm"] = {f"{lo}-{hi}": agg((ce >= lo) & (ce < hi) & has, pc) for lo, hi in [(0, 2), (2, 5), (5, 100)]}
    out["by_species_count"] = {str(j): agg((k == j), pc) for j in (1, 2, 3)}
    # әр түр бойынша recall (бір түрлі үлгілерде)
    rec = {}
    for j, sp in enumerate(species):
        m = Y[:, j] > 0
        if m.sum() >= 20:
            rec[sp] = round(float(pc[m, j].mean()), 2)
    out["recall_by_species"] = dict(sorted(rec.items(), key=lambda kv: kv[1]))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
