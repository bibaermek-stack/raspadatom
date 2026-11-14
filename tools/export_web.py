"""Веб-зертханаға деректер экспорттау.

  python tools/export_web.py [--checkpoint checkpoints/spectral_net.pt]

Жасайды:
  web/data/lines.js  — NIST сызықтары (түр бойынша бағандар), деңгей белгілері, иондану энергиялары
  web/data/model.js  — оқытылған модель салмақтары (BatchNorm конволюцияға біріктірілген, float32, base64)
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spectra_ai.elements import SPECIES_INFO  # noqa: E402
from spectra_ai.model import SpectralAttentionNet  # noqa: E402
from spectra_ai.synth import HI_NM, LO_NM, N_POINTS, T_MAX, T_MIN  # noqa: E402

BN_EPS = 1e-5


def export_lines(csv: Path, out: Path, species_order: list[str]) -> None:
    db = pd.read_csv(csv)
    species = []
    for name in species_order:
        g = db[db.species == name].sort_values("wl_nm").reset_index(drop=True)
        sym, kk, z, charge, ion = SPECIES_INFO[name]
        levels: dict[str, int] = {}

        def lv(s: str) -> int:
            return levels.setdefault(s, len(levels))

        lo = [lv(s) for s in g.lower]
        up = [lv(s) for s in g.upper]
        species.append({
            "name": name, "sym": sym, "kk": kk, "z": z, "charge": charge, "ion": ion,
            "levels": list(levels),
            "wl": [round(float(x), 4) for x in g.wl_nm],
            "rel": [float(x) for x in g.rel],
            "ei": [round(float(x), 5) for x in g.ei_ev],
            "ek": [round(float(x), 5) for x in g.ek_ev],
            "aki": [None if pd.isna(x) else float(x) for x in g.aki],
            "gi": [int(x) for x in g.gi],
            "gk": [int(x) for x in g.gk],
            "lo": lo, "up": up,
        })
    data = {"grid": {"lo": LO_NM, "hi": HI_NM, "n": N_POINTS}, "tRange": [T_MIN, T_MAX],
            "tRef": 10000.0, "species": species}
    js = ("/* Автоматты түрде жасалған: python tools/export_web.py. NIST ASD деректері. */\n"
          "(function (root) {\n  var D = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n"
          "  if (typeof module === 'object' && module.exports) module.exports = D; else root.SL_DATA = D;\n"
          "})(typeof window !== 'undefined' ? window : globalThis);\n")
    out.write_text(js, encoding="utf-8")
    print(f"{out}: {out.stat().st_size/1024:.0f} КБ, {sum(len(s['wl']) for s in species)} сызық")


def fold_bn(conv: torch.nn.Conv1d, bn: torch.nn.BatchNorm1d):
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    w = conv.weight * scale[:, None, None]
    b = (conv.bias - bn.running_mean) * scale + bn.bias
    return w.detach(), b.detach()


def export_model(ckpt_path: Path, out: Path) -> list[str]:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    species = ck["species"]
    model = SpectralAttentionNet(len(species))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    assert abs(model.conv[0][1].eps - BN_EPS) < 1e-12

    tensors: list[tuple[str, torch.Tensor]] = []
    for i, block in enumerate(model.conv):
        w, b = fold_bn(block[0], block[1])
        tensors += [(f"conv{i+1}.w", w), (f"conv{i+1}.b", b)]
    for i, layer in enumerate(model.attention.layers):
        sa = layer.self_attn
        tensors += [
            (f"l{i}.ln1.w", layer.norm1.weight), (f"l{i}.ln1.b", layer.norm1.bias),
            (f"l{i}.qkv.w", sa.in_proj_weight), (f"l{i}.qkv.b", sa.in_proj_bias),
            (f"l{i}.out.w", sa.out_proj.weight), (f"l{i}.out.b", sa.out_proj.bias),
            (f"l{i}.ln2.w", layer.norm2.weight), (f"l{i}.ln2.b", layer.norm2.bias),
            (f"l{i}.ff1.w", layer.linear1.weight), (f"l{i}.ff1.b", layer.linear1.bias),
            (f"l{i}.ff2.w", layer.linear2.weight), (f"l{i}.ff2.b", layer.linear2.bias),
        ]
    tensors += [("cls.w", model.cls_head.weight), ("cls.b", model.cls_head.bias),
                ("temp.w", model.temp_head.weight), ("temp.b", model.temp_head.bias)]

    manifest, blobs, offset = [], [], 0
    for name, t in tensors:
        a = t.detach().cpu().numpy().astype("<f4")
        manifest.append({"name": name, "shape": list(a.shape), "offset": offset})
        blobs.append(a.tobytes())
        offset += a.size
    meta = {
        "species": species, "tRange": [T_MIN, T_MAX], "nPoints": N_POINTS,
        "dModel": 64, "heads": 4, "layers": len(model.attention.layers),
        "ln_eps": 1e-5, "metrics": ck.get("metrics", {}),
        "tensors": manifest, "weights": base64.b64encode(b"".join(blobs)).decode("ascii"),
    }
    js = ("/* Автоматты түрде жасалған: python tools/export_web.py. Модель салмақтары (float32, base64). */\n"
          "(function (root) {\n  var M = " + json.dumps(meta, ensure_ascii=False, separators=(",", ":")) + ";\n"
          "  if (typeof module === 'object' && module.exports) module.exports = M; else root.SL_MODEL = M;\n"
          "})(typeof window !== 'undefined' ? window : globalThis);\n")
    out.write_text(js, encoding="utf-8")
    print(f"{out}: {out.stat().st_size/1024:.0f} КБ, {offset} параметр, метрика: {ck.get('metrics')}")
    return species


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "spectral_net.pt"))
    ap.add_argument("--csv", default=str(ROOT / "data" / "nist_lines.csv"))
    ap.add_argument("--out", default=str(ROOT / "web" / "data"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    species = export_model(Path(a.checkpoint), out / "model.js")
    export_lines(Path(a.csv), out / "lines.js", species)


if __name__ == "__main__":
    main()
