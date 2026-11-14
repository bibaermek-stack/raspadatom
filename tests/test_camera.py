"""Камера домені: генератор, экспорт, JS құбыры (бірлік тесттер, виртуал фото бенчмарк)."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from spectra_ai.nist import load_database
from spectra_ai.synth_cam import CAM_EXCLUDE, CAM_GRID, CAM_N, CameraSpectra, make_camera_spectrum

ROOT = Path(__file__).resolve().parent.parent
NODE = shutil.which("node")


@pytest.fixture(scope="module")
def db():
    return load_database()


def test_generator_shapes_and_ranges(db):
    ds = CameraSpectra(db, 50, seed=1)
    assert "Ca II" not in ds.species and set(CAM_EXCLUDE).isdisjoint(ds.species)
    for i in range(50):
        x, y, t, m = ds[i]
        assert x.shape == (1, CAM_N) and torch.isfinite(x).all()
        assert 0.0 <= x.min() and x.max() <= 1.0 + 1e-6
        assert set(np.unique(y.numpy())) <= {0.0, 1.0}
        assert m in (0.0, 1.0)


def test_cfl_source_class(db):
    """CFL — атомдық емес көз класы: белгі қойылады, ал T маскасы 0 (температура мағынасыз)."""
    ds = CameraSpectra(db, 10, seed=1)
    assert ds.species[-1] == "CFL" and "CFL" not in db
    j, n_cfl, hit, widths = ds.species.index("CFL"), 0, 0, []
    for i in range(600):
        _, y, _, m, meta = make_camera_spectrum(db, ds.species, np.random.default_rng([5, i]), return_meta=True)
        widths.append(meta["fwhm_nm"])
        if meta["cfl"]:
            n_cfl += 1
            hit += y[j] > 0
            assert m == 0.0
        else:
            assert y[j] == 0.0                          # CFL тек CFL үлгілерінде
    assert 0.05 < n_cfl / 600 < 0.16 and hit / n_cfl > 0.8       # үлесі ≈ 10%, көбі байқалады
    assert max(widths) > 20 and 0.1 < np.mean(np.array(widths) > 10) < 0.3     # кең (қаныққан) сызықтар режимі: ≈ 18%


def test_generator_is_deterministic(db):
    ds = CameraSpectra(db, 10, seed=7)
    assert torch.equal(ds[3][0], ds[3][0]) and torch.equal(ds[3][1], ds[3][1])


def test_empty_samples_have_no_labels(db):
    sp = [s for s in sorted(db) if s not in CAM_EXCLUDE]
    ys = [make_camera_spectrum(db, sp, np.random.default_rng(i), p_empty=1.0)[1].sum() for i in range(30)]
    assert max(ys) == 0
    ys = [make_camera_spectrum(db, sp, np.random.default_rng(i), p_empty=0.0)[1].sum() for i in range(200)]
    assert np.mean(np.array(ys) > 0) > 0.8       # көпшілігінде кемінде бір байқалатын түр бар


def test_invisible_species_not_labelled(db):
    """Байқалу сүзгісі: белгі саны таңдалған түр санынан артпайды және кейде кем болады."""
    sp = [s for s in sorted(db) if s not in CAM_EXCLUDE]
    diffs = []
    for i in range(300):
        _, y, _, _, meta = make_camera_spectrum(db, sp, np.random.default_rng([3, i]), p_empty=0.0, return_meta=True)
        diffs.append(y.sum())
    assert 1.0 < np.mean(diffs) < 2.0


@pytest.mark.skipif(NODE is None, reason="node жоқ")
def test_camera_model_export_matches_checkpoint():
    ck_path = ROOT / "checkpoints" / "camera_net.pt"
    if not ck_path.exists():
        pytest.skip("camera checkpoint жоқ")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    r = subprocess.run([NODE, "-e", "const M=require('./web/data/model_cam.js');console.log(JSON.stringify({s:M.species,w:M.window,n:M.nPoints}))"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    meta = json.loads(r.stdout)
    assert meta["s"] == ck["species"] and meta["n"] == CAM_N and meta["w"] == [float(CAM_GRID[0]), float(CAM_GRID[-1])]


@pytest.mark.skipif(NODE is None, reason="node жоқ")
def test_camera_js_unit():
    r = subprocess.run([NODE, str(ROOT / "web" / "tests" / "cam_unit.js")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1])["ok"]


@pytest.mark.skipif(NODE is None or not (ROOT / "web" / "data" / "model_cam.js").exists(), reason="node не камера моделі жоқ")
def test_camera_pipeline_on_virtual_photos():
    """Сурет -> жолақ -> профиль -> калибровка -> модель: виртуал фотоларда cәйкес сапа (нақты камера емес)."""
    r = subprocess.run([NODE, str(ROOT / "web" / "tests" / "cam_bench.js"), "120"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    res = json.loads(r.stdout.strip().splitlines()[-1])
    assert res["roi_fail"] == 0
    assert res["cfl_auto_cal_ok"] >= 0.9
    assert res["by_kind"]["all"]["f1"] > 0.7
    assert res["by_kind"]["nist1"]["f1"] > 0.75
    assert res["gate"]["all"]["f1"] >= res["by_kind"]["all"]["f1"] - 0.01       # қақпа сапаны төмендетпейді
    assert res["gate"]["continuum"]["exact"] >= 0.9                              # қыздыру шамы/LED үшін жалған түр жоқ
