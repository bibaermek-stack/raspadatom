"""Синтетикалық спектрометр: NIST сызықтары + Больцман интенсивтілігі + pseudo-Voigt профилі + шу.

Әр үлгі — 1–3 түрдің қоспасы (multi-label), плазма температурасы T, құрылғы ені (FWHM),
толқын ұзындығы калибровкасының ығысуы, базалық сызық және шу.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

KB_EV = 8.617333262e-5  # эВ/К
T_REF = 10_000.0        # NIST "Rel." шамасына сәйкес келетін шартты температура
T_MIN, T_MAX = 4_000.0, 15_000.0
LO_NM, HI_NM, N_POINTS = 200.0, 900.0, 2048
GRID = np.linspace(LO_NM, HI_NM, N_POINTS)
DX = GRID[1] - GRID[0]


def boltzmann_strengths(rel, ek_ev, T):
    """I ∝ Rel · exp(−E_k/kT) / exp(−E_k/kT_ref): температураға тәуелділік."""
    return rel * np.exp(-ek_ev * (1.0 / (KB_EV * T) - 1.0 / (KB_EV * T_REF)))


def pseudo_voigt_kernel(fwhm_pts: float, eta: float) -> np.ndarray:
    half = int(min(max(4 * fwhm_pts, 6), 200))
    x = np.arange(-half, half + 1, dtype=np.float64)
    sigma = fwhm_pts / 2.3548
    gamma = fwhm_pts / 2.0
    gauss = np.exp(-0.5 * (x / sigma) ** 2)
    lorentz = 1.0 / (1.0 + (x / gamma) ** 2)
    k = (1 - eta) * gauss + eta * lorentz
    return k / k.max()


def render_lines(wl_nm, strength, shift_nm=0.0) -> np.ndarray:
    """Сызықтарды торға (сызықтық интерполяциямен) жинайды -> дельта-спектр."""
    pos = (np.asarray(wl_nm) + shift_nm - LO_NM) / DX
    i0 = np.floor(pos).astype(int)
    frac = pos - i0
    out = np.zeros(N_POINTS)
    for idx, w in ((i0, 1 - frac), (i0 + 1, frac)):
        ok = (idx >= 0) & (idx < N_POINTS)
        np.add.at(out, idx[ok], (strength * w)[ok])
    return out


def make_spectrum(db: dict, species: list[str], rng: np.random.Generator, max_species=3):
    """Бір жаттығу үлгісі: (spectrum[2048], multi-hot[n], T_norm)."""
    k = int(rng.integers(1, max_species + 1))
    chosen = rng.choice(len(species), size=k, replace=False)
    T = rng.uniform(T_MIN, T_MAX)
    fwhm_pts = rng.uniform(0.6, 3.0) / DX
    kernel = pseudo_voigt_kernel(fwhm_pts, eta=rng.uniform(0, 1))
    shift = rng.normal(0, 0.3)

    delta = np.zeros(N_POINTS)
    for s in chosen:
        lines = db[species[s]]
        st = boltzmann_strengths(lines.rel.values, lines.ek_ev.values, T)
        st = st / st.max() * 10 ** rng.uniform(-1.0, 0.0)  # концентрация
        delta += render_lines(lines.wl_nm.values, st, shift)

    sig = np.convolve(delta, kernel, mode="same")
    sig /= sig.max() + 1e-12
    x = np.linspace(-1, 1, N_POINTS)
    base = rng.normal(0, 0.02) + rng.normal(0, 0.02) * x + rng.normal(0, 0.02) * x**2
    shot = rng.normal(0, 1, N_POINTS) * np.sqrt(sig) * rng.uniform(0, 0.02)
    read = rng.normal(0, rng.uniform(0.002, 0.03), N_POINTS)
    spec = np.clip(sig + base + shot + read, 0, None)
    spec /= spec.max() + 1e-8

    y = np.zeros(len(species), dtype=np.float32)
    y[chosen] = 1.0
    return spec.astype(np.float32), y, np.float32((T - T_MIN) / (T_MAX - T_MIN))


class SyntheticSpectra(Dataset):
    """Шексіз дерлік генератор: үлгі idx мен seed бойынша детерминді."""

    def __init__(self, db, n, seed=0, max_species=3):
        self.db, self.species = db, sorted(db)
        self.n, self.seed, self.max_species = n, seed, max_species
        self.epoch = 0

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        s, y, t = make_spectrum(self.db, self.species, rng, self.max_species)
        return torch.from_numpy(s)[None], torch.from_numpy(y), torch.tensor(t)
