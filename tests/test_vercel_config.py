"""vercel.json: сайт статикалық. Репода requirements.txt болғандықтан, Vercel оны Python қолданбасы деп қабылдап
(PYTHON_ENTRYPOINT_NOT_FOUND) құлайтын: framework null және dist/ дайын шығыс ретінде көрсетілуі тиіс."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_vercel_serves_prebuilt_static_dist():
    cfg = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))
    assert "framework" in cfg and cfg["framework"] is None            # Python фреймворкін автоанықтауды өшіреді
    out = ROOT / cfg["outputDirectory"]
    assert out.is_dir() and (ROOT / "requirements.txt").exists()
    rw = {r["source"]: r["destination"] for r in cfg["rewrites"]}
    assert (out / rw["/"].lstrip("/")).is_file()                       # басты бет шын файлға апарады
    for key in ("installCommand", "buildCommand"):
        assert "pip" not in cfg[key] and "python" not in cfg[key]      # құрастыру кезінде ауыр Python орнатылмайды


def test_dist_is_standalone():
    """dist/spectra_lab.html сыртқы файлсыз: Vercel-де тек осы бір файл беріледі."""
    import re
    html = (ROOT / "dist" / "spectra_lab.html").read_text(encoding="utf-8")
    local = re.findall(r'<script[^>]+src="(?!https?:)[^"]+"|<link[^>]+href="(?!https?:)[^"]+"', html)
    assert not local, local                                            # жергілікті js/css сілтемелері жоқ (шрифт қана сыртқы)
    assert "SL_MODEL_CAM" in html and "SL_DATA" in html
