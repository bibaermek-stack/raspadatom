"""Камера домені: веб-камера + дифракциялық тор арқылы алынған көрінетін спектр (380–780 нм).

Лабораториялық (`synth.py`) спектрден айырмасы — нақты камераның бұрмалаулары:
  * төмен ажыратымдылық (FWHM 1.5–12 нм), сызық ені мен пішіні өзгермелі;
  * калибровка қателігі: пиксель→нм сәйкестігінің ығысуы мен масштабы;
  * камера сезімталдығы (көк/қызыл шектері, IR-фильтр, толқындылық) — бейтаныс тегіс қисық;
  * қанығу (clip), жарқылдау (bloom), қалдық гамма;
  * үздіксіз фон (қыздыру шамы, LED/фосфор «төбелері»), қараңғы деңгей;
  * бөгде сызықтар (фосфор Eu/Tb, молекулалық жолақтар, шағылыс): олар жоқ белгілі түрлерге жатпайды;
  * NIST `Rel` мен нақты шам арасындағы интенсивтілік айырмасы: әр сызыққа лог-қалыпты ауытқу;
  * ешбір атомдық түрі жоқ спектрлер (қыздыру шамы, LED): белгі нөлдік.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .synth import T_MAX, T_MIN, boltzmann_strengths, pseudo_voigt_kernel

CAM_LO, CAM_HI, CAM_N = 380.0, 780.0, 1024
CAM_GRID = np.linspace(CAM_LO, CAM_HI, CAM_N)
CAM_DX = CAM_GRID[1] - CAM_GRID[0]
PIVOT = 580.0
CAM_EXCLUDE = ("Ca II",)  # көрінетін аймақта (>380 нм) танитындай сызықтары жоқ
CAM_SOURCES = ("CFL",)    # атомдық емес көз кластары: люминофорлы флуоресцентті шам (Hg + Eu/Tb)

# Фосфор (флуоресцентті шам) сызықтары: Tb³⁺, Eu³⁺ — бөгде сызық ретінде ғана (класс емес), шамамен
PHOSPHOR_LINES = [(487.7, 0.5), (543.5, 1.0), (585.0, 0.25), (620.0, 0.2),
                  (593.0, 0.35), (611.6, 1.0), (650.0, 0.2), (690.0, 0.3), (700.0, 0.25)]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def planck(wl_nm, T):
    l = wl_nm * 1e-9
    b = 1.0 / (l**5 * (np.exp(1.4388e-2 / (l * T)) - 1.0))
    return b / b.max()


def response_curve(rng, wl=CAM_GRID):
    """Бейтаныс камера сезімталдығы: көк шегі · IR-кесу · жасыл төбесі · толқындылық."""
    lb, wb = rng.uniform(385, 440), rng.uniform(6, 25)
    lr, wr = rng.uniform(640, 800), rng.uniform(8, 35)
    lm, sm, fl = rng.uniform(500, 600), rng.uniform(60, 160), rng.uniform(0.2, 0.6)
    r = _sigmoid((wl - lb) / wb) * (1 - _sigmoid((wl - lr) / wr)) * (fl + (1 - fl) * np.exp(-0.5 * ((wl - lm) / sm) ** 2))
    r *= 1 + rng.uniform(0, 0.15) * np.sin(2 * np.pi * wl / rng.uniform(150, 600) + rng.uniform(0, 6.3))
    return r / r.max()


def add_lines(delta, wl, st):
    pos = (wl - CAM_LO) / CAM_DX
    i0 = np.floor(pos).astype(int)
    frac = pos - i0
    for idx, w in ((i0, 1 - frac), (i0 + 1, frac)):
        ok = (idx >= 0) & (idx < CAM_N)
        np.add.at(delta, idx[ok], (st * w)[ok])


def make_camera_spectrum(db: dict, species: list[str], rng: np.random.Generator,
                         max_species: int = 3, p_empty: float = 0.10, p_cfl: float = 0.10, return_meta: bool = False):
    """-> (spectrum[1024] 0..1, multi-hot[n], T_norm, T_valid[, meta]).

    species: атомдық түрлер (db-да бар) + көз кластары (CAM_SOURCES, мыс. «CFL»).
    meta ішінде нақты калибровка қателігі (shift_nm, scale) бар: self-calibration оқытуына мақсат ретінде жарайды.
    """
    atomic = [i for i, n in enumerate(species) if n in db]
    src_idx = {n: i for i, n in enumerate(species) if n in CAM_SOURCES}
    u0 = rng.random()
    p_cfl = p_cfl if "CFL" in src_idx else 0.0          # көз класы жоқ болса, CFL үлесі бос спектрге айналмайды
    cfl = u0 < p_cfl
    k = 0 if (cfl or u0 < p_cfl + p_empty) else int(rng.choice([1, 2, 3][:max_species], p=[0.5, 0.3, 0.2][:max_species] if max_species == 3 else None))
    chosen = rng.choice(atomic, size=k, replace=False) if k else np.array([], dtype=int)
    T = rng.uniform(T_MIN, T_MAX)
    # аспап ені: көбінде 1.5–10 нм, 20% жағдайда кең (қанығудан жайылған, нашар фокус) — нақты суреттерде 20–30 нм сызықтар кездеседі
    if rng.random() < 0.8:
        fwhm_nm = float(np.exp(rng.uniform(np.log(1.5), np.log(10.0))))
    else:
        fwhm_nm = float(np.exp(rng.uniform(np.log(8.0), np.log(28.0))))
    kernel = pseudo_voigt_kernel(fwhm_nm / CAM_DX, eta=rng.uniform(0, 0.7))

    # калибровка қателігі: жақсы (CFL бойынша, RMS ~1 нм) 75%, дөрекі 25%; 50% жағдайда нелинейлік (объектив/перспектива)
    if rng.random() < 0.75:
        shift, scale = rng.normal(0, 1.0), 1 + rng.normal(0, 0.003)
    else:
        shift, scale = rng.normal(0, 5.0), 1 + rng.normal(0, 0.015)
    quad = rng.normal(0, 3.0) if rng.random() < 0.5 else 0.0
    jitter = rng.uniform(0.1, 0.5)

    def distort(w):
        return PIVOT + scale * (w - PIVOT) + shift + quad * ((w - PIVOT) / 200.0) ** 2

    delta = np.zeros(CAM_N)
    y = np.zeros(len(species), dtype=np.float32)
    own = {}                                  # класс -> оның жеке дельта-спектрі

    def add_atomic(s_idx, amp_scale):
        ln = db[species[s_idx]]
        wl = distort(ln.wl_nm.values)
        inwin = (wl >= CAM_LO) & (wl <= CAM_HI)
        if not inwin.any():
            return None
        st = boltzmann_strengths(ln.rel.values, ln.ek_ev.values, T) * np.exp(rng.normal(0, jitter, len(wl)))
        st = st / st[inwin].max() * amp_scale
        d_s = np.zeros(CAM_N)
        add_lines(d_s, wl[inwin], st[inwin])
        return d_s

    for s in chosen:
        d_s = add_atomic(int(s), 10 ** rng.uniform(-0.8, 0.0))
        if d_s is not None:
            delta += d_s
            own[int(s)] = d_s

    cont = np.zeros(CAM_N)
    if cfl:
        # флуоресцентті (трифосфорлы) шам: Hg (әлсіз) + Tb³⁺/Eu³⁺ сызықтары + көк Eu²⁺ жолағы; қатынастар кең ауытқиды
        d_c = np.zeros(CAM_N)
        for wl0, a in PHOSPHOR_LINES:
            if rng.random() < 0.9:
                add_lines(d_c, np.array([distort(wl0) + rng.normal(0, 0.8)]), np.array([a * np.exp(rng.normal(0, 0.35))]))
        if "Hg I" in db and "Hg I" in species:
            d_hg = add_atomic(species.index("Hg I"), rng.uniform(0.05, 0.5))
            if d_hg is not None:
                delta += d_hg
                own[species.index("Hg I")] = d_hg
        delta += d_c
        own[src_idx["CFL"]] = d_c + (own.get(species.index("Hg I"), 0) if "Hg I" in species else 0)
        c0, w0 = 450 + rng.normal(0, 6), rng.uniform(40, 60) / 2.3548
        cont += rng.uniform(0.05, 0.8) * np.exp(-0.5 * ((CAM_GRID - c0) / w0) ** 2)
    else:
        # бөгде сызықтар: жеке, не фосфор жиыны
        u = rng.random()
        if u < 0.08:
            sc = rng.uniform(0.2, 0.9)
            for wl0, a in PHOSPHOR_LINES:
                if rng.random() < 0.8:
                    add_lines(delta, np.array([distort(wl0) + rng.normal(0, 1.0)]), np.array([sc * a * rng.uniform(0.6, 1.4)]))
        else:
            nd = int(rng.choice([0, 1, 2, 3, 4], p=[0.5, 0.25, 0.15, 0.07, 0.03]))
            if nd:
                add_lines(delta, rng.uniform(395, 705, nd), rng.uniform(0.03, 0.7, nd))

    sig = np.convolve(delta, kernel, mode="same")
    m = sig.max()
    sig = sig / m if m > 0 else sig
    own_sig = {s_: np.convolve(d_, kernel, mode="same") / (m if m > 0 else 1.0) for s_, d_ in own.items()}

    # үздіксіз фон
    uc = rng.random()
    if not cfl:
        if uc < 0.35:
            cont += rng.uniform(0.05, 1.2) * planck(CAM_GRID, rng.uniform(2300, 6500))
        elif uc < 0.5:
            for _ in range(int(rng.integers(1, 4))):
                c, w = rng.uniform(420, 690), rng.uniform(25, 90) / 2.3548
                cont += rng.uniform(0.05, 1.0) * np.exp(-0.5 * ((CAM_GRID - c) / w) ** 2)
    elif uc < 0.25:
        cont += rng.uniform(0.02, 0.3) * planck(CAM_GRID, rng.uniform(3000, 6500))
    if k == 0 and not cfl and sig.max() == 0 and cont.max() == 0:
        cont += rng.uniform(0.1, 1.0) * planck(CAM_GRID, rng.uniform(2300, 6500))
    resp = response_curve(rng)
    total = (sig + cont) * resp

    # экспозиция, қанығу + «блуминг», гамма
    pk = total.max()
    read_sigma = float(np.exp(rng.uniform(np.log(0.002), np.log(0.025))))
    for s_, o_ in own_sig.items():
        # байқалу: класстың өз шыңы толық масштабтың ≥5% және шудан ≥3σ жоғары болса ғана белгі қойылады
        h = float((o_ * resp).max() / pk) if pk > 0 else 0.0
        if h >= max(0.05, 3.0 * read_sigma):
            y[s_] = 1.0
    total = total / pk if pk > 0 else total
    wing = np.convolve(total, pseudo_voigt_kernel(rng.uniform(30, 70) / CAM_DX, 1.0), mode="same")
    total = total + wing * rng.uniform(0.0, 0.06) / (wing.max() + 1e-9)
    gain = float(np.exp(rng.uniform(np.log(0.5), np.log(3.0)))) if rng.random() < 0.7 else float(np.exp(rng.uniform(np.log(2.0), np.log(12.0))))
    raw = total * gain
    over = np.clip(raw - 1.0, 0, None)
    if over.any():                              # қанығудан асқан энергия айналаға таралады (блуминг)
        kb = pseudo_voigt_kernel(rng.uniform(8, 40) / CAM_DX, 0.6)
        kb = kb / kb.sum()
        over = np.convolve(over, kb, mode="same") * rng.uniform(0.4, 1.2)
    total = np.clip(np.minimum(raw, 1.0) + over, 0, 1)
    total = total ** rng.uniform(0.85, 1.25)

    # қараңғы деңгей, шу, төмен жиілікті ақау
    x = np.linspace(0, 1, CAM_N)
    total = total + rng.uniform(0, 0.08) + rng.uniform(0, 0.02) * np.sin(2 * np.pi * x / rng.uniform(0.25, 1.0) + rng.uniform(0, 6.3))
    total = total + rng.normal(0, 1, CAM_N) * np.sqrt(np.clip(total, 0, None)) * rng.uniform(0, 0.03)
    total = total + rng.normal(0, read_sigma, CAM_N)
    total = np.clip(total, 0, None)
    spec = (total / (total.max() + 1e-8)).astype(np.float32)
    # T тек атомдық түр бар спектрде мағыналы (CFL/бос үшін маска 0)
    out = (spec, y, np.float32((T - T_MIN) / (T_MAX - T_MIN)), np.float32(1.0 if k else 0.0))
    if return_meta:
        meta = {"fwhm_nm": fwhm_nm, "shift_nm": float(shift), "scale": float(scale), "quad_nm": float(quad), "k": int(y.sum()), "cfl": bool(cfl),
                "cal_err_nm": float(abs(shift) + abs(scale - 1) * 200.0), "continuum": float(cont.max()), "gain": gain}
        return out + (meta,)
    return out


class CameraSpectra(Dataset):
    def __init__(self, db, n, seed=0, max_species=3, p_empty=0.10):
        self.db = db
        self.species = [s for s in sorted(db) if s not in CAM_EXCLUDE] + list(CAM_SOURCES)
        self.n, self.seed, self.max_species, self.p_empty, self.epoch = n, seed, max_species, p_empty, 0

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        s, y, t, m = make_camera_spectrum(self.db, self.species, rng, self.max_species, self.p_empty)
        return torch.from_numpy(s)[None], torch.from_numpy(y), torch.tensor(t), torch.tensor(m)
