"""Бағалау құралдары: метрикалар/bootstrap, lab-бағалау скрипті, нақты LIBS спектрін модель торына келтіру және анықтау есебі."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from spectra_ai.eval_libs_species import ABSENT, ELEMENTS, detection_report, rebin_to_grid
from spectra_ai.metrics import bootstrap, counts, ece, prf, summarize
from spectra_ai.synth import GRID
from spectra_ai.train_libs import OXIDES

ROOT = Path(__file__).resolve().parent.parent


# ---------- metrics ----------
def test_prf_and_summarize_match_hand_computation():
    pred = np.array([[1, 1, 0], [0, 1, 0], [1, 0, 0]], bool)
    y = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 0]], bool)
    c = counts(pred, y)
    assert c["tp"].tolist() == [1, 1, 1] and c["fp"].tolist() == [1, 0, 0] and c["fn"].tolist() == [0, 1, 0]
    assert c["exact"].tolist() == [False, False, True]
    s = summarize(c)
    assert s["precision"] == pytest.approx(3 / 4) and s["recall"] == pytest.approx(3 / 4) and s["exact"] == pytest.approx(1 / 3)
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)            # бос жиын NaN бермейді


def test_bootstrap_interval_contains_point_estimate_and_paired_delta():
    rng = np.random.default_rng(0)
    y = rng.random((400, 5)) < 0.3
    good = y ^ (rng.random(y.shape) < 0.02)           # 2 % қате
    bad = y ^ (rng.random(y.shape) < 0.10)            # 10 % қате
    cg, cb = counts(good, y), counts(bad, y)
    out = bootstrap({"bad": cb, "good": cg}, n_boot=300)
    lo, hi = out["good"]["f1"]
    assert lo <= summarize(cg)["f1"] <= hi and hi - lo < 0.06
    d_lo, d_hi = out["delta_good_minus_bad"]["f1"]
    assert d_lo > 0                                    # жақсы модель айқын жақсы: жұптасқан ДИ нөлден жоғары
    extra = bootstrap({"m": cg}, {"m": {"T": rng.normal(500, 50, 400)}}, n_boot=100)
    assert 450 < np.mean(extra["m"]["T"]) < 550


def test_ece_zero_for_confident_correct_and_large_for_confident_wrong():
    y = np.array([[1, 0], [0, 1], [1, 1]])
    assert ece(y.astype(float), y) == pytest.approx(0.0)
    assert ece(1.0 - y, y) == pytest.approx(1.0)


# ---------- real-data path ----------
def test_rebin_to_grid_gaps_range_and_normalisation():
    wl = np.arange(240.0, 860.0, 0.05)
    wl = wl[~(((wl > 351) & (wl < 370)) | ((wl > 479) & (wl < 508)))]          # екі спектрометр саңылауы
    line = np.exp(-0.5 * ((wl - 589.0) / (0.15 / 2.3548)) ** 2)
    sig = np.vstack([line, -line * 0.01 + 0.0])                                 # екінші жол: тек теріс мәндер
    out = rebin_to_grid(wl, sig, smooth_fwhm_nm=None)
    assert out.shape == (2, len(GRID)) and out.dtype == np.float32 and np.isfinite(out).all()
    assert abs(GRID[out[0].argmax()] - 589.0) < 0.4 and out[0].max() == pytest.approx(1.0, abs=1e-6)
    assert (out[0][GRID < 239] == 0).all() and (out[0][(GRID > 353) & (GRID < 368)] == 0).all()   # деректі жоқ жерде 0
    assert out[1].max() == 0.0                                                  # теріс мәндер кесіледі, бөлгіш нөл болмайды


def test_rebin_smoothing_widens_lines_and_handles_unsorted_input():
    wl = np.arange(240.0, 860.0, 0.05)
    line = np.exp(-0.5 * ((wl - 450.0) / (0.15 / 2.3548)) ** 2)[None]
    raw, smooth = rebin_to_grid(wl, line, smooth_fwhm_nm=None), rebin_to_grid(wl, line, smooth_fwhm_nm=1.3)
    assert (smooth > 0.5).sum() > (raw > 0.5).sum()
    perm = np.random.default_rng(1).permutation(len(wl))
    assert np.allclose(rebin_to_grid(wl[perm], line[:, perm], smooth_fwhm_nm=1.3), smooth, atol=1e-6)


def test_detection_report_counts_recall_and_false_alarms():
    species = ["Ca I", "Ca II", "Hg I", "Na I", "Ne I", "Ar I", "Si I", "Ti I", "Al I", "Fe I", "Mg I", "Mg II", "K I"]
    ix = {s: i for i, s in enumerate(species)}
    prob = np.full((4, len(species)), 0.01)
    prob[0, ix["Ca II"]] = prob[1, ix["Ca I"]] = 0.9            # Ca 4 спектрдің 2-інде табылды (екі иондық күйдің кез келгені)
    prob[2, ix["Hg I"]] = 0.8                                   # пеллетте жоқ түрге жалған дабыл
    prob[0, ix["Ne I"]] = 0.5                                   # шекке тең мән анықтау саналмайды
    prob[:, ix["Ar I"]] = 0.99                                  # Ar «жоқ» тізімінде емес: ауада болады
    oxide = np.zeros((4, 8))
    oxide[:, list(ELEMENTS).index("Ca")] = [10.0, 2.0, 0.5, 6.0]   # ≥1: 3 спектр, ≥5: 2 спектр
    rep = detection_report(prob, species, oxide)
    ca = rep["elements"]["Ca"]
    assert ca["n_ge_1wt"] == 3 and ca["recall_ge_1wt"] == pytest.approx(2 / 3, abs=1e-3)
    assert ca["n_ge_5wt"] == 2 and ca["recall_ge_5wt"] == pytest.approx(0.5)
    assert (ca["n_detected_ge_1wt"], ca["n_missed_ge_1wt"], ca["n_missed_ge_5wt"]) == (2, 1, 1)
    assert ca["miss_rate_ge_1wt"] == pytest.approx(1 / 3, abs=1e-3)
    assert ca["n_below_1wt"] == 1 and ca["detected_below_1wt"] == 0.0
    assert rep["elements"]["Si"]["miss_rate_ge_1wt"] is None
    assert rep["false_alarm_counts_absent"] == {"Ne I": 0, "Hg I": 1}
    assert rep["false_alarm_absent"]["Hg I"] == pytest.approx(0.25) and rep["false_alarm_any_absent"] == pytest.approx(0.25)
    assert "Ar I" not in rep["false_alarm_absent"] and rep["other_species_detected_rate"] == {"Ar I": 1.0}
    assert set(ABSENT) >= set(rep["false_alarm_absent"])


def test_eval_libs_cli_exports_ranked_errors(tmp_path):
    data, output = tmp_path / "spectra.csv", tmp_path / "errors.csv"
    with data.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["meta", *["comp"] * len(OXIDES), "wvl", "wvl", "wvl"])
        writer.writerow(["Pellet Name", *OXIDES, 390, 589, 770])
        writer.writerow(["P1", 50, 2, 5, 10, 10, 20, 3, 0, 0.2, 0.8, 0.1])
        writer.writerow(["P2", 45, 5, 15, 5, 10, 10, 6, 4, 0.8, 0.2, 1.0])
    result = subprocess.run([sys.executable, "-m", "spectra_ai.eval_libs_species", "--data", str(data),
                             "--threshold", "1", "--errors-csv", str(output)],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["n_spectra"] == 2 and report["threshold"] == 1
    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    misses = [row for row in rows if row["error"] == "miss" and row["minimum_oxide_wt_pct"] == "5"]
    assert len(misses) == len(OXIDES)
    assert [row["element"] for row in misses[:-1]] == sorted(set(ELEMENTS) - {"K"})
    assert misses[-1]["element"] == "K" and misses[-1]["error_rate"] == ""
    assert all(row["error_count"] == row["evaluated_spectra"] for row in misses)
    false_alarms = [row for row in rows if row["error"] == "false_alarm"]
    assert {row["species"] for row in false_alarms} == set(ABSENT)
    assert all(row["error_count"] == "0" and row["evaluated_spectra"] == "2" for row in false_alarms)


# ---------- eval_lab CLI ----------
@pytest.mark.skipif(not (ROOT / "checkpoints" / "spectral_net.pt").exists(), reason="lab checkpoint жоқ")
def test_eval_lab_cli_small_run_outputs_expected_structure():
    r = subprocess.run([sys.executable, "-m", "spectra_ai.eval_lab", "--n", "120", "--n-boot", "30",
                        "--stress-n", "80", "--stress-max-species", "5"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["test"]["n"] == 120 and 0.9 < out["test"]["f1"] <= 1.0
    assert out["test"]["T_MAE_K"] < out["test"]["T_MAE_K_if_predicting_mean"]
    assert set(out["by_k"]) == {"1", "2", "3"} and set(out["stress"]["by_k"]) == {"1", "2", "3", "4", "5"}
    lo, hi = out["ci95"]["current"]["f1"]
    assert lo <= out["test"]["f1"] <= hi
    assert len(out["per_species"]) == 29 and out["calibration"]["ece_15_bins"] < 0.05
