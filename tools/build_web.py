"""Веб-зертхананы бір файлға жинау.

  python tools/build_web.py                      -> dist/spectra_lab.html   (толық, дербес бет; қосып ашуға болады)
  python tools/build_web.py --artifact PATH      -> claude.ai Artifact үшін нұсқа (doctype/head/body тегтерінсіз)

CSS пен барлық JS (деректер мен модель салмақтарын қоса) беттің ішіне енгізіледі, сыртқы тәуелділік тек Google Fonts.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"


def parts():
    src = (WEB / "index.html").read_text(encoding="utf-8")
    title = re.search(r"<title>(.*?)</title>", src, re.S).group(1).strip()
    fonts = re.findall(r'<link rel="stylesheet" href="https://fonts\.googleapis\.com[^>]*>', src)
    body = re.search(r"<body>(.*?)</body>", src, re.S).group(1)
    scripts = re.findall(r'<script src="([^"]+)"></script>', body)
    markup = re.sub(r'\s*<script src="[^"]+"></script>', "", body).strip()
    css = (WEB / "css" / "style.css").read_text(encoding="utf-8")
    tags = []
    for s in scripts:
        code = (WEB / s).read_text(encoding="utf-8").replace("</script", "<\\/script")
        tags.append(f"<script>\n/* {s} */\n{code}\n</script>")
    return title, fonts, css, markup, "\n".join(tags)


def standalone() -> str:
    title, fonts, css, markup, js = parts()
    return (
        '<!doctype html>\n<html lang="kk">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        f"<title>{title}</title>\n" + "\n".join(fonts) + f"\n<style>\n{css}\n</style>\n</head>\n<body>\n{markup}\n{js}\n</body>\n</html>\n"
    )


def artifact() -> str:
    title, fonts, css, markup, js = parts()
    return (
        f"<title>{title}</title>\n"
        '<meta name="claude:viewport-fit-sides" content="cover">\n' + "\n".join(fonts) +
        f"\n<style>\n{css}\n</style>\n{markup}\n{js}\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "dist" / "spectra_lab.html"))
    ap.add_argument("--artifact", default=None, help="Artifact нұсқасының жолы")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(standalone(), encoding="utf-8")
    print(f"{out}: {out.stat().st_size/1024:.0f} КБ")
    if a.artifact:
        art = Path(a.artifact)
        art.parent.mkdir(parents=True, exist_ok=True)
        art.write_text(artifact(), encoding="utf-8")
        print(f"{art}: {art.stat().st_size/1024:.0f} КБ")


if __name__ == "__main__":
    main()
