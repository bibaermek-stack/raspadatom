"""NIST Atomic Spectra Database-тен спектрлік сызықтарды жүктеу және кэштеу.

Түпнұсқа скрипттегі `Observed`/`Rel.` бағандары жиі бос немесе "457bl" сияқты
таңбалармен келеді, сондықтан мұнда оларды қатаң парсинг жасаймыз.
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .elements import HC_EV_NM, ionization_ev, vac_to_air

DEFAULT_SPECIES = [
    "H I", "He I", "Li I", "Na I", "K I", "Mg I", "Mg II", "Ca I", "Ca II",
    "Sr I", "Ba I", "Ne I", "Ar I", "Kr I", "Xe I", "Hg I", "Zn I", "Cd I",
    "Al I", "Si I", "Fe I", "Cu I", "Ti I", "Cr I", "Mn I", "Ni I",
    "C I", "N I", "O I",
]
CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "nist_lines.csv"
_FLOAT = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _first_float(x) -> float:
    m = _FLOAT.search(str(x))
    return float(m.group()) if m else np.nan


def _pair(x) -> tuple[float, float]:
    """'3.2657  -  6.3642' немесе '10.19 - [13.43]' -> (бірінші, соңғы) сан."""
    f = _FLOAT.findall(str(x))
    return (float(f[0]), float(f[-1])) if len(f) >= 2 else (np.nan, np.nan)


def _clean_level(x) -> str:
    return re.sub(r"\s+", " ", str(x)).replace(" | ", "|").strip(" |")


def fetch_species(name: str, lo_nm=200.0, hi_nm=900.0, max_lines=400, retries=3) -> pd.DataFrame:
    """Бір түрдің (мыс. 'Na I') байланысқан-байланысқан ауысуларын NIST-тен жүктейді.

    Сүзгілер: Ei < Ek < E_иондану (автоиондану деңгейлері алынбайды), Ek−Ei ≈ hc/λ_vac.
    wl_nm — ауадағы (спектрометр өлшейтін), wl_vac_nm — вакуумдағы толқын ұзындығы.
    """
    import astropy.units as u
    from astroquery.nist import Nist

    last_err = None
    for attempt in range(retries):
        try:
            # вакуумдық аралықты сәл кеңірек аламыз, ауаға ауыстырғаннан кейін қайта сүземіз
            table = Nist.query((lo_nm - 1) * u.nm, (hi_nm + 1) * u.nm, linename=name)
            break
        except Exception as e:  # желі/NIST қателері
            last_err = e
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"{name}: NIST сұрауы сәтсіз: {last_err}")

    raw = table.to_pandas()
    en_col = next(c for c in raw.columns if c.startswith("Ei"))
    g_col = next(c for c in raw.columns if c.startswith("gi"))
    ei_ek = raw[en_col].map(_pair)
    gi_gk = raw[g_col].map(_pair)
    wl_vac = pd.to_numeric(raw["Observed"], errors="coerce").fillna(
        pd.to_numeric(raw["Ritz"], errors="coerce"))
    df = pd.DataFrame({
        "species": name,
        "wl_nm": vac_to_air(wl_vac),
        "wl_vac_nm": wl_vac,
        "rel": raw["Rel."].map(_first_float),
        "aki": pd.to_numeric(raw["Aki"], errors="coerce"),
        "ei_ev": ei_ek.map(lambda p: p[0]),
        "ek_ev": ei_ek.map(lambda p: p[1]),
        "gi": gi_gk.map(lambda p: p[0]),
        "gk": gi_gk.map(lambda p: p[1]),
        "lower": raw["Lower level"].map(_clean_level),
        "upper": raw["Upper level"].map(_clean_level),
    })
    df = df.dropna(subset=["wl_nm", "rel", "ei_ev", "ek_ev"])
    e_ion = ionization_ev(name)
    de_err = (df.ek_ev - df.ei_ev - HC_EV_NM / df.wl_vac_nm).abs()
    keep = ((df.rel > 0) & df.wl_nm.between(lo_nm, hi_nm) & (df.ei_ev >= 0)
            & (df.ek_ev > df.ei_ev) & (df.ek_ev < e_ion) & (de_err < 0.01))
    return df[keep].nlargest(max_lines, "rel").reset_index(drop=True)


def build_database(species=DEFAULT_SPECIES, out=CSV_PATH, **kw) -> pd.DataFrame:
    frames = []
    for sp in species:
        try:
            df = fetch_species(sp, **kw)
            print(f"{sp:6s}: {len(df)} сызық")
            if len(df):
                frames.append(df)
        except Exception as e:
            print(f"{sp:6s}: ӨТКІЗІЛДІ ({e})")
    db = pd.concat(frames, ignore_index=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    db.to_csv(out, index=False)
    print(f"Сақталды: {out} ({len(db)} сызық, {db.species.nunique()} түр)")
    return db


def load_database(path=CSV_PATH) -> dict[str, pd.DataFrame]:
    db = pd.read_csv(path)
    return {sp: g.reset_index(drop=True) for sp, g in db.groupby("species", sort=True)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="NIST ASD жүктеу")
    ap.add_argument("--species", nargs="*", default=DEFAULT_SPECIES)
    ap.add_argument("--out", default=str(CSV_PATH))
    a = ap.parse_args()
    build_database(a.species, a.out)
