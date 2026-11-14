"""NIST деректерінің физикалық дұрыстығы: энергия сақталуы, иондану шегі, вакуум -> ауа түзетуі."""
import numpy as np
import pandas as pd
import pytest

from spectra_ai.elements import HC_EV_NM, SPECIES_INFO, ionization_ev, vac_to_air
from spectra_ai.nist import CSV_PATH, _pair, load_database


@pytest.fixture(scope="module")
def db():
    return pd.read_csv(CSV_PATH)


def test_all_default_species_present(db):
    assert set(db.species) == set(SPECIES_INFO)
    assert db.groupby("species").size().min() >= 20


def test_bound_bound_levels(db):
    """Ei < Ek < E_иондану: автоиондану деңгейлері (қалыпты плазмада көрінбейтін) алынбаған."""
    ion = db.species.map(ionization_ev)
    assert (db.ei_ev >= 0).all()
    assert (db.ei_ev < db.ek_ev).all()
    assert (db.ek_ev < ion).all()


def test_photon_energy_equals_level_difference(db):
    """ΔE = Ek − Ei = hc/λ (вакуумдағы толқын ұзындығы)."""
    err = (db.ek_ev - db.ei_ev - HC_EV_NM / db.wl_vac_nm).abs()
    assert err.max() < 0.01


def test_air_wavelengths_match_reference_lines(db):
    """Анықтамалық ауа толқын ұзындықтары: Na D2, H-alpha, Hg 253.652 нм, Cu 324.754 нм."""
    ref = {"Na I": 588.995, "H I": 656.279, "Hg I": 253.652, "Cu I": 324.754}
    for sp, wl in ref.items():
        g = db[db.species == sp]
        assert (g.wl_nm - wl).abs().min() < 0.005, sp
    assert abs(float(vac_to_air(589.158326)) - 588.9950) < 0.001


def test_wavelength_window(db):
    assert db.wl_nm.between(200, 900).all()


def test_pair_parser_handles_brackets_and_spacing():
    assert _pair("33.0956       -    36.3286") == (33.0956, 36.3286)
    assert _pair("10.19880615024     -   [13.430554096]") == (10.19880615024, 13.430554096)
    assert all(np.isnan(_pair("nan")))


def test_database_loader_groups_by_species():
    d = load_database()
    assert all((g.species == sp).all() for sp, g in d.items())
