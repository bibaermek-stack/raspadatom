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


def _last_float(x) -> float:
    m = _FLOAT.findall(str(x).replace(" - ", " | ").split("|")[-1])
    return float(m[-1]) if m else np.nan


def fetch_species(name: str, lo_nm=200.0, hi_nm=900.0, max_lines=400, retries=3) -> pd.DataFrame:
    """Бір түрдің (мыс. 'Na I') сызықтарын NIST-тен жүктейді."""
    import astropy.units as u
    from astroquery.nist import Nist

    last_err = None
    for attempt in range(retries):
        try:
            table = Nist.query(lo_nm * u.nm, hi_nm * u.nm, linename=name)
            break
        except Exception as e:  # желі/NIST қателері
            last_err = e
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"{name}: NIST сұрауы сәтсіз: {last_err}")

    raw = table.to_pandas()
    ek_col = next(c for c in raw.columns if c.startswith("Ei"))
    gk_col = next(c for c in raw.columns if c.startswith("gi"))
    df = pd.DataFrame({
        "species": name,
        "wl_nm": pd.to_numeric(raw["Observed"], errors="coerce").fillna(
            pd.to_numeric(raw["Ritz"], errors="coerce")),
        "rel": raw["Rel."].map(_first_float),
        "aki": pd.to_numeric(raw["Aki"], errors="coerce"),
        "ek_ev": raw[ek_col].map(_last_float),
        "gk": raw[gk_col].map(_last_float),
    })
    df = df.dropna(subset=["wl_nm", "rel", "ek_ev"])
    df = df[(df.rel > 0) & (df.wl_nm.between(lo_nm, hi_nm))]
    return df.nlargest(max_lines, "rel").reset_index(drop=True)


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
