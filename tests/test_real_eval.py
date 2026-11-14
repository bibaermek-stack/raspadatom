"""tools/eval_real.py: суретті камера құбырынан өткізу (синтетикалық кемпірқосақ жолағымен; нақты фото емес)."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def bruton(l):
    r = g = b = 0.0
    if 380 <= l < 440: r, b = -(l - 440) / 60, 1
    elif l < 490: g, b = (l - 440) / 50, 1
    elif l < 510: g, b = 1, -(l - 510) / 20
    elif l < 580: r, g = (l - 510) / 70, 1
    elif l < 645: r, g = 1, -(l - 645) / 65
    elif l <= 780: r = 1
    return r, g, b


def make_band(path, W=720, H=360):
    """Қараңғы фонда көлденең спектр жолағы: 400–700 нм, CFL сызықтары + төмен үздіксіз фон."""
    img = np.full((H, W, 3), 6, np.float64)
    x = np.arange(W)
    lam = 400 + 300 * x / (W - 1)
    lines = [(435.8, 0.5), (487.7, 0.3), (544.5, 1.0), (578.0, 0.2), (611.6, 0.95), (650, 0.15), (690, 0.2)]
    prof = 0.05 + sum(a * np.exp(-0.5 * ((lam - c) / 4.0) ** 2) for c, a in lines)
    col = np.array([bruton(l) for l in lam])
    for y in range(H // 2 - 22, H // 2 + 22):
        img[y] = 6 + 240 * prof[:, None] * col
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)


@pytest.mark.skipif(shutil.which("node") is None, reason="node жоқ")
def test_eval_real_tool_outputs_json(tmp_path):
    p = tmp_path / "band.png"
    make_band(p)
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "eval_real.py"), str(p)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["roi"] is not None and abs(out["roi"]["theta_deg"]) < 5
    assert out["cfl_calibration"] is not None and out["cfl_calibration"]["rms_nm"] < 2.5     # CFL сызықтары табылып, калибровка жасалады
    assert out["calibration_used"] == "cfl" and len(out["top"]) == 5
