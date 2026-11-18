"""Көп белгілі классификация метрикалары және bootstrap сенім аралықтары (eval_lab, eval_cam, eval_libs_species).

Барлық функция спектр бойынша есептейді де, содан соң қосады: bootstrap спектрлерді қайта таңдайды,
ал бірнеше модель БІР таңдамамен қайта өлшенгендіктен, олардың айырмасының ДИ-ы жұптасқан болады.
"""
from __future__ import annotations

import numpy as np


def counts(pred, y) -> dict[str, np.ndarray]:
    """pred, y: (N, C) bool. Әр спектр үшін tp / fp / fn және «барлық белгі дұрыс» (exact)."""
    pred, y = np.asarray(pred, bool), np.asarray(y, bool)
    return {"tp": (pred & y).sum(1), "fp": (pred & ~y).sum(1), "fn": (~pred & y).sum(1),
            "exact": (pred == y).all(1)}


def prf(tp, fp, fn) -> tuple[float, float, float]:
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return float(p), float(r), float(2 * p * r / max(p + r, 1e-12))


def summarize(c: dict, idx=None) -> dict:
    """micro P / R / F1 және exact үлесі (idx — спектрлер жиыны, әдепкіде барлығы)."""
    idx = np.arange(len(c["tp"])) if idx is None else idx
    p, r, f1 = prf(c["tp"][idx].sum(), c["fp"][idx].sum(), c["fn"][idx].sum())
    return {"precision": p, "recall": r, "f1": f1, "exact": float(c["exact"][idx].mean()), "n": int(len(idx))}


def _ci(a) -> list[float]:
    return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]


def bootstrap(counts_by_model: dict[str, dict], extras: dict[str, dict[str, np.ndarray]] | None = None,
              n_boot: int = 2000, seed: int = 0) -> dict:
    """95 % ДИ (2.5–97.5 процентиль): F1, exact және `extras[model][атау]` (әр спектрге бір сан, мыс. температура қателігі) орташасы.

    Екі модель берілсе, айырманың (екінші − бірінші) ДИ-ы да шығады: екеуі бір таңдамамен қайта өлшенеді.
    """
    extras = extras or {}
    rng = np.random.default_rng(seed)
    n = len(next(iter(counts_by_model.values()))["tp"])
    draws = {m: {k: np.empty(n_boot) for k in ("f1", "exact", *extras.get(m, {}))} for m in counts_by_model}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        for m, c in counts_by_model.items():
            s = summarize(c, idx)
            draws[m]["f1"][b], draws[m]["exact"][b] = s["f1"], s["exact"]
            for k, v in extras.get(m, {}).items():
                draws[m][k][b] = v[idx].mean()
    out = {m: {k: _ci(v) for k, v in d.items()} for m, d in draws.items()}
    names = list(counts_by_model)
    if len(names) == 2:
        a, b = names
        out[f"delta_{b}_minus_{a}"] = {k: _ci(draws[b][k] - draws[a][k]) for k in ("f1", "exact")}
    return out


def ece(prob, y, bins: int = 15) -> float:
    """Expected calibration error: болжам ықтималдығы мен байқалған жиілік айырмасының салмақталған орташасы."""
    p, t = np.asarray(prob, float).ravel(), np.asarray(y, float).ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1.0 else (p <= hi))
        if m.any():
            total += m.mean() * abs(p[m].mean() - t[m].mean())
    return float(total)
