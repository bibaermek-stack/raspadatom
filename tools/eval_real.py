"""Нақты фотоны камера құбырынан өткізу: жолақ → профиль → калибровка (CFL / түс) → модель → растау.

  python tools/eval_real.py photo.jpg [--roi '{"cx":640,"cy":480,"theta":0.1,"hv":12,"t0":-300,"t1":300}']

Сурет ешқайда жіберілмейді (жергілікті Node құбыры). Нәтиже — JSON: ROI, калибровка, ЖИ болжамы, сызық сәйкестігімен расталғаны.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--roi", default=None, help="қолмен ROI (JSON, бұрыш радианмен)")
    ap.add_argument("--max-width", type=int, default=1280, help="веб-интерфейстегідей кішірейту")
    a = ap.parse_args()
    im = Image.open(a.image).convert("RGBA")
    if im.width > a.max_width:
        im = im.resize((a.max_width, round(im.height * a.max_width / im.width)), Image.LANCZOS)
    with tempfile.TemporaryDirectory() as d:
        base = Path(d) / Path(a.image).stem
        base.with_suffix(".rgba").write_bytes(im.tobytes())
        base.with_suffix(".json").write_text(json.dumps({"w": im.width, "h": im.height}))
        cmd = ["node", str(ROOT / "web" / "tests" / "real_eval.js"), str(base)] + ([a.roi] if a.roi else [])
        r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode:
        sys.stderr.write(r.stderr)
        sys.exit(r.returncode)


if __name__ == "__main__":
    main()
