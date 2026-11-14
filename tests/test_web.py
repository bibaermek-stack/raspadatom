"""Веб-зертхананың JS кодын (physics.js, ai.js) Python/PyTorch-пен салыстыру. Node керек (жоқ болса — өткізіледі)."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from spectra_ai.model import SpectralAttentionNet
from spectra_ai.nist import load_database
from spectra_ai.synth import (N_POINTS, boltzmann_strengths, make_spectrum, pseudo_voigt_kernel,
                              render_lines)

ROOT = Path(__file__).resolve().parent.parent
CKPT = Path(os.environ.get("SPECTRA_CKPT", ROOT / "checkpoints" / "spectral_net.pt"))

pytestmark = pytest.mark.skipif(shutil.which("node") is None or not CKPT.exists(), reason="node немесе checkpoint жоқ")


def clean_reference(db, names, conc, T, fwhm_nm, eta, shift):
    """make_spectrum-ның шусыз бөлігі (synth.py-дағы көмекші функциялармен)."""
    from spectra_ai.synth import DX
    delta = np.zeros(N_POINTS)
    for n, c in zip(names, conc):
        ln = db[n]
        # web/data/lines.js-те толқын ұзындығы 4, энергия 5 таңбаға дөңгелектелген
        st = boltzmann_strengths(ln.rel.values, np.array([round(float(x), 5) for x in ln.ek_ev]), T)
        delta += render_lines(np.array([round(float(x), 4) for x in ln.wl_nm]), st / st.max() * c, shift)
    sig = np.convolve(delta, pseudo_voigt_kernel(fwhm_nm / DX, eta), mode="same")
    return sig / (sig.max() + 1e-12)


def test_js_matches_python(tmp_path):
    db = load_database()
    species = sorted(db)
    rng = np.random.default_rng(7)
    physics = []
    for _ in range(8):
        k = int(rng.integers(1, 4))
        names = [species[i] for i in rng.choice(len(species), k, replace=False)]
        case = dict(species=names, conc=[float(10 ** rng.uniform(-1, 0)) for _ in names],
                    T=float(rng.uniform(4000, 15000)), fwhm=float(rng.uniform(0.6, 3.0)),
                    eta=float(rng.uniform(0, 1)), shift=float(rng.normal(0, 0.3)))
        case["clean"] = clean_reference(db, case["species"], case["conc"], case["T"], case["fwhm"],
                                        case["eta"], case["shift"]).tolist()
        physics.append(case)

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = SpectralAttentionNet(len(ck["species"]))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    ai = []
    for i in range(6):
        spec, _, _ = make_spectrum(db, species, np.random.default_rng(100 + i))
        with torch.no_grad():
            logits, t = model(torch.from_numpy(spec)[None, None])
        ai.append(dict(spec=spec.tolist(), logits=logits[0].tolist(), t=float(t[0])))

    cam = []
    cam_ck = ROOT / "checkpoints" / "camera_net.pt"
    if cam_ck.exists():
        from spectra_ai.synth_cam import CameraSpectra
        ck2 = torch.load(cam_ck, map_location="cpu", weights_only=False)
        m2 = SpectralAttentionNet(len(ck2["species"]), n_points=ck2["n_points"], num_layers=ck2.get("num_layers", 2))
        m2.load_state_dict(ck2["state_dict"])
        m2.eval()
        ds = CameraSpectra(db, 8, seed=5)
        for i in range(6):
            x = ds[i][0]
            with torch.no_grad():
                lg, _ = m2(x[None])
            cam.append(dict(spec=x[0].tolist(), logits=lg[0].tolist()))

    f = tmp_path / "vectors.json"
    f.write_text(json.dumps(dict(physics=physics, ai=ai, cam=cam)))
    r = subprocess.run(["node", str(ROOT / "web" / "tests" / "verify.js"), str(f)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["max_abs_diff_clean"] < 1e-9 and out["max_abs_diff_logit"] < 2e-3


def test_js_pipeline_quality_matches_validation():
    """JS-генератор + JS-модель оқыту таралымында Python-валидациямен бірдей сапа беруі керек."""
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    ref = ck["metrics"]["f1"]
    r = subprocess.run(["node", str(ROOT / "web" / "tests" / "bench.js"), "200"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    js = json.loads(r.stdout.strip().splitlines()[-1])
    assert js["f1"] > 0.85 and abs(js["f1"] - ref) < 0.06, (js, ref)


def test_browser_smoke():
    """Беттің Chromium-да қатесіз жүктелуі және негізгі сценарийлер (Playwright болса)."""
    probe = subprocess.run(["node", "-e", "require('playwright')"], capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip("Node Playwright жоқ")
    r = subprocess.run(["node", str(ROOT / "web" / "tests" / "smoke.js")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
