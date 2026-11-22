"""Лаборатория моделін НАҚТЫ LIBS спектрлерінде (SuperLIBS 10K) бағалау: синтетикасыз.

  python -m spectra_ai.eval_libs_species [--data data/zenodo.10K_EARTH.2.4mJ.csv]
         [--checkpoint checkpoints/spectral_net.pt ...] [--smooth-fwhm 1.3]
         [--threshold 0.5] [--errors-csv libs-errors.csv]

SuperLIBS пеллеттері көп элементті (орташа 6–7 оксид ≥ 1 wt %), әрі негізгі элементтердің резонанс сызықтары
0.1 wt %-да да көрінеді (Ca II 393 нм — барлық спектрде). Сондықтан «≥ 1 wt % = бар, қалғаны = жоқ» деген
белгі precision өлшеуге жарамайды. Орнына екі таза өлшем алынады:

  1. 8 негізгі элементтің (Si, Ti, Al, Fe, Mg, Ca, Na, K) АНЫҚТАЛУ ЖИІЛІГІ (recall): оксид ≥ 1 және ≥ 5 wt % болғанда;
  2. пеллетте БОЛУЫ МҮМКІН ЕМЕС түрлер (He, Ne, Kr, Xe, Hg, Cd) бойынша ЖАЛҒАН ДАБЫЛ жиілігі.

Өңдеу алдын ала белгіленген (нақты белгілерге қарап таңдалмаған): спектр модель торына (200–900 нм, 2048 нүкте)
орташаланып қайта бөлінеді, спектрометр саңылаулары = 0, ажыратымдылық оқыту ауқымының ортасына (Гаусс FWHM 1.3 нм)
келтіріледі, max = 1. Оның қажеттігін in-silico тексеру көрсетті: тар сызықтарда (0.1–0.2 нм) модель онсыз нашар жұмыс істейді.
Модель SuperLIBS-та ОҚЫТЫЛМАҒАН, сондықтан барлық спектр — тест.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from .eval_lab import load, predict
from .nist import load_database
from .synth import GRID
from .train_libs import read_spectra

# train_libs.OXIDES тәртібімен: SiO2, TiO2, Al2O3, Fe2O3, MgO, CaO, Na2O, K2O
ELEMENTS = {"Si": ("Si I",), "Ti": ("Ti I",), "Al": ("Al I",), "Fe": ("Fe I",),
            "Mg": ("Mg I", "Mg II"), "Ca": ("Ca I", "Ca II"), "Na": ("Na I",), "K": ("K I",)}
# Ar (ауада ~1 %), N, O, H, C — ауа мен ылғалдан көрінуі мүмкін, сондықтан «жоқ» тізімінде емес
ABSENT = ("He I", "Ne I", "Kr I", "Xe I", "Hg I", "Cd I")
DEFAULT_FWHM_NM = 1.3


def rebin_to_grid(wl, spectra, grid=GRID, smooth_fwhm_nm: float | None = DEFAULT_FWHM_NM) -> np.ndarray:
    """(N, C) спектрді өсу ретіндегі `wl` бойынша `grid` нүктелеріне орташалап қайта бөледі.

    Деректері жоқ (спектрометр саңылауы, диапазоннан тыс) нүктелер 0 болады; теріс мәндер кесіледі;
    `smooth_fwhm_nm` берілсе Гаусспен тегістеледі; әр спектр max = 1-ге нормаланады.
    """
    wl, spectra = np.asarray(wl, float), np.asarray(spectra)
    if not np.all(np.diff(wl) > 0):
        order = np.argsort(wl)
        wl, spectra = wl[order], spectra[:, order]
    dx = grid[1] - grid[0]
    bin_of = np.floor((wl - (grid[0] - dx / 2)) / dx).astype(int)
    ok = (bin_of >= 0) & (bin_of < len(grid))
    bin_of, spectra = bin_of[ok], spectra[:, ok]
    out = np.zeros((spectra.shape[0], len(grid)))
    if ok.any():
        starts = np.r_[0, np.flatnonzero(np.diff(bin_of)) + 1]
        sums = np.add.reduceat(spectra.astype(np.float64), starts, axis=1)
        out[:, bin_of[starts]] = sums / np.diff(np.r_[starts, len(bin_of)])
    out = np.clip(out, 0.0, None)
    if smooth_fwhm_nm:
        sigma = smooth_fwhm_nm / 2.3548 / dx
        half = int(4 * sigma) + 1
        kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
        kernel /= kernel.sum()
        out = np.stack([np.convolve(row, kernel, mode="same") for row in out])
    out /= out.max(axis=1, keepdims=True) + 1e-12
    return out.astype(np.float32)


def _rate(flags):
    return round(float(np.mean(flags)), 4) if len(flags) else None


def detection_report(prob, species, oxide_wt, threshold: float = 0.5) -> dict:
    """prob: (N, түр) ықтималдықтар; oxide_wt: (N, 8) wt % (ELEMENTS тәртібімен)."""
    index = {s: i for i, s in enumerate(species)}
    prob = np.asarray(prob)
    detected = prob > threshold
    elements = {}
    for j, (el, names) in enumerate(ELEMENTS.items()):
        element_prob = prob[:, [index[s] for s in names]].max(axis=1)
        found = element_prob > threshold
        conc = oxide_wt[:, j]
        high, low = conc >= 1, conc < 1
        element = {"species": list(names),
                   "n_below_1wt": int(low.sum()), "detected_below_1wt": _rate(found[low]),
                   "mean_prob_ge_1wt": round(float(element_prob[high].mean()), 3) if high.any() else None}
        for minimum in (1, 5):
            qualified = found[conc >= minimum]
            suffix = f"ge_{minimum}wt"
            element[f"n_{suffix}"] = len(qualified)
            element[f"n_detected_{suffix}"] = int(qualified.sum())
            element[f"n_missed_{suffix}"] = int((~qualified).sum())
            element[f"recall_{suffix}"] = _rate(qualified)
            element[f"miss_rate_{suffix}"] = _rate(~qualified)
        elements[el] = element
    tp = sum(element["n_detected_ge_1wt"] for element in elements.values())
    support = sum(element["n_ge_1wt"] for element in elements.values())
    absent = {s: detected[:, index[s]] for s in ABSENT if s in index}
    absent_cols = [index[s] for s in absent]
    known = {s for names in ELEMENTS.values() for s in names} | set(ABSENT)
    return {
        "elements": elements,
        "micro_recall_ge_1wt": round(tp / max(support, 1), 4),
        "false_alarm_absent": {s: _rate(flags) for s, flags in absent.items()},
        "false_alarm_counts_absent": {s: int(flags.sum()) for s, flags in absent.items()},
        "false_alarm_any_absent": _rate(detected[:, absent_cols].any(axis=1)) if absent else None,
        "mean_prob_absent": round(float(prob[:, absent_cols].mean()), 4) if prob.size and absent else None,
        "other_species_detected_rate": {s: _rate(detected[:, i]) for s, i in index.items()
                                        if s not in known and detected[:, i].mean() > 0.05},
    }


def write_error_csv(path: Path, report: dict) -> None:
    """Save ranked misses by oxide threshold and false alarms for known-absent species."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["checkpoint", "probability_threshold", "error", "element", "species",
                         "minimum_oxide_wt_pct", "evaluated_spectra", "error_count", "error_rate"])
        for checkpoint, model in report["models"].items():
            prefix = (checkpoint, report["threshold"])
            for minimum in (1, 5):
                suffix = f"ge_{minimum}wt"
                ranked = sorted(model["elements"].items(), key=lambda item: (
                    -item[1][f"n_missed_{suffix}"] / item[1][f"n_{suffix}"]
                    if item[1][f"n_{suffix}"] else 1, item[0],
                ))
                for element, stats in ranked:
                    writer.writerow([*prefix, "miss", element, "/".join(stats["species"]), minimum,
                                     stats[f"n_{suffix}"], stats[f"n_missed_{suffix}"],
                                     stats[f"miss_rate_{suffix}"]])
            ranked = sorted(model["false_alarm_counts_absent"].items(),
                            key=lambda item: (-item[1], item[0]))
            for species, count in ranked:
                element = species.split()[0]
                writer.writerow([*prefix, "false_alarm", element, species, "", report["n_spectra"], count,
                                 model["false_alarm_absent"][species]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/zenodo.10K_EARTH.2.4mJ.csv"))
    ap.add_argument("--checkpoint", nargs="+", default=["checkpoints/spectral_net.pt"])
    ap.add_argument("--smooth-fwhm", type=float, default=DEFAULT_FWHM_NM, help="0 — тегістеусіз")
    ap.add_argument("--threshold", type=float, default=0.5, help="анықтау үшін модель ықтималдығы осы шектен жоғары болуы керек")
    ap.add_argument("--errors-csv", type=Path, help="әр элемент бойынша қателер кестесін сақтау")
    a = ap.parse_args()
    if not 0.0 <= a.threshold <= 1.0:
        ap.error("--threshold [0, 1] ауқымында болуы керек")
    if not a.data.is_file():
        ap.error(f"SuperLIBS CSV файлы табылмады: {a.data}")
    if a.errors_csv is not None and a.errors_csv.resolve() in {a.data.resolve(), *(Path(p).resolve() for p in a.checkpoint)}:
        ap.error("--errors-csv үшін кіріс файлдарынан бөлек жол көрсетіңіз")

    wl, spectra, oxides, _ = read_spectra(a.data, labeled=True)
    X = torch.from_numpy(rebin_to_grid(wl, spectra, smooth_fwhm_nm=a.smooth_fwhm or None))[:, None]
    species = sorted(load_database())
    out = {"data": a.data.name, "n_spectra": len(X), "smooth_fwhm_nm": a.smooth_fwhm,
           "threshold": a.threshold, "models": {}}
    for path in a.checkpoint:
        model, ck = load(path, species)
        logits, _ = predict(model, X)
        out["models"][path] = {"epoch": ck.get("epoch"), "max_species_trained": ck.get("max_species", 3),
                               **detection_report(expit(logits), species, oxides, a.threshold)}
    if a.errors_csv is not None:
        write_error_csv(a.errors_csv, out)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
