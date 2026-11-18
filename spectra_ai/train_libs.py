"""Train and run quantitative models for processed SuperLIBS spectra."""
import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

OXIDES = tuple(f"{oxide} (wt %)" for oxide in (
    "SiO2", "TiO2", "Al2O3", "Fe2O3", "MgO", "CaO", "Na2O", "K2O",
))
DEFAULT_MODEL = Path("checkpoints/superlibs_earth_2p4mj.joblib")
# Тор екі жағынан да үздіктің сыртына шығуы керек: бұрын 10–1000 болатын, ал валидация α=1000-ға дейін монотонды жақсарып,
# оптимум торға тірелген (шын оптимум ≈10 000)
RIDGE_ALPHAS = (10, 100, 300, 1000, 3000, 10000, 30000, 100000)


def read_spectra(path, labeled):
    spectra, labels, names = [], [], []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.reader(stream)
        kinds, fields = next(rows), next(rows)
        if len(kinds) != len(fields):
            raise ValueError("SuperLIBS header rows have different widths")
        channels = [i for i, kind in enumerate(kinds) if kind == "wvl"]
        if not channels:
            raise ValueError("No wvl columns found")
        wavelengths = np.array([float(fields[i]) for i in channels])
        if not np.isfinite(wavelengths).all():
            raise ValueError("Non-finite wavelength")
        pellet = fields.index("Pellet Name") if "Pellet Name" in fields else None
        if labeled and pellet is None:
            raise ValueError("Training requires Pellet Name")
        targets = [fields.index(oxide) for oxide in OXIDES] if labeled else []
        for number, row in enumerate(rows, start=3):
            if not row:
                continue
            if len(row) != len(fields):
                raise ValueError(f"CSV row {number} has an unexpected width")
            spectra.append(np.array([row[i] for i in channels], dtype=np.float32))
            names.append(row[pellet].strip().upper() if pellet is not None else str(number - 2))
            if labeled:
                labels.append([float(row[i]) for i in targets])
    x, y = np.asarray(spectra), np.asarray(labels)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected finite spectra")
    if labeled:
        if not all(names) or not np.isfinite(y).all() or np.any((y < 0) | (y > 100)):
            raise ValueError("Expected pellet names and finite oxide percentages in [0, 100]")
        compositions = {}
        for name, target in zip(names, y.tolist()):
            if compositions.setdefault(name, target) != target:
                raise ValueError(f"Inconsistent composition for pellet {name}")
    return wavelengths, x, y, names


def split_samples(chemistry, seed):
    # Keep pellets with identical reported chemistry in the same split.
    groups, indices = np.unique(chemistry, axis=0, return_inverse=True)
    order = np.random.default_rng(seed).permutation(len(groups))
    holdout = round(len(groups) * 0.15)
    if holdout < 1:
        raise ValueError("Too few reference compositions for three partitions")
    return tuple(
        np.flatnonzero(np.isin(indices, group))
        for group in (order[2 * holdout:], order[:holdout], order[holdout:2 * holdout])
    )


def predict(model, x):
    return np.clip(model.predict(x), 0, 100)


def errors(y, prediction):
    residuals = prediction - y
    return {
        "mean_mae_wt_pct_points": float(np.abs(residuals).mean()),
        "per_oxide": {
            oxide: {"mae": float(mae), "rmse": float(rmse)}
            for oxide, mae, rmse in zip(
                OXIDES, np.abs(residuals).mean(axis=0), np.sqrt((residuals**2).mean(axis=0))
            )
        },
    }


def build(estimator):
    return make_pipeline(StandardScaler(), TransformedTargetRegressor(
        regressor=estimator, transformer=StandardScaler(),
    ))


def split_spread(x, y, alpha, seeds=range(10)):
    """Таңдалған Ridge-ті басқа бөліну тұқымдарында қайта өлшеу: бір бөлінудің тест MAE-сі ± шамасыз берілмеуі үшін."""
    maes = []
    for seed in seeds:
        fit, _, test = split_samples(y, seed)
        maes.append(float(np.abs(predict(build(Ridge(alpha=alpha)).fit(x[fit], y[fit]), x[test]) - y[test]).mean()))
    return {"seeds": list(seeds), "test_mae_mean": float(np.mean(maes)), "test_mae_std": float(np.std(maes, ddof=1)),
            "test_mae_min": min(maes), "test_mae_max": max(maes)}


def train(source, output, seed):
    wavelengths, x, y, names = read_spectra(source, labeled=True)
    fit, validation, test = split_samples(y, seed)
    print(f"Spectra: train={len(fit)}, validation={len(validation)}, test={len(test)}", flush=True)
    scores, best = {}, None
    candidates = {f"ridge_{alpha}": Ridge(alpha=alpha) for alpha in RIDGE_ALPHAS}
    candidates["mlp_128_64"] = MLPRegressor(
        hidden_layer_sizes=(128, 64), alpha=0.03, max_iter=250,
        learning_rate_init=0.001, random_state=seed,
    )
    with threadpool_limits(limits=4):
        for name, estimator in candidates.items():
            model = build(estimator)
            model.fit(x[fit], y[fit])
            score = float(np.abs(predict(model, x[validation]) - y[validation]).mean())
            scores[name] = score
            print(f"{name}: validation MAE {score:.3f}", flush=True)
            if best is None or score < scores[best[0]]:
                best = name, model
        name, model = best
        test_prediction = predict(model, x[test])
        on_edge = name in (f"ridge_{RIDGE_ALPHAS[0]}", f"ridge_{RIDGE_ALPHAS[-1]}")
        if on_edge:
            print(f"ЕСКЕРТУ: үздік модель {name} α торының шетінде: тордан тыс оптимум болуы мүмкін, торды кеңейтіңіз", flush=True)
        spread = split_spread(x, y, int(name.split("_")[1])) if name.startswith("ridge_") else None
    report = {
        "source": source.name,
        "condition": "SuperLIBS 10K, Earth atmosphere, 2.4 mJ",
        "seed": seed,
        "split": {
            label: {"spectra": len(rows), "pellets": len({names[i] for i in rows}),
                    "unique_compositions": len(np.unique(y[rows], axis=0))}
            for label, rows in zip(("train", "validation", "test"), (fit, validation, test))
        },
        "channels": len(wavelengths),
        "wavelength_range_nm": [float(wavelengths.min()), float(wavelengths.max())],
        "ridge_alphas_searched": list(RIDGE_ALPHAS),
        "validation_mae_wt_pct_points": scores,
        "selected_model": name,
        "selected_on_grid_edge": bool(on_edge),
        "test": errors(y[test], test_prediction),
        "split_seed_spread": spread,
        "training_mean_baseline_test": errors(y[test], y[fit].mean(axis=0)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "wavelengths_nm": wavelengths, "oxides": OXIDES}, output, compress=3)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def run_prediction(source, checkpoint, output):
    wavelengths, x, _, names = read_spectra(source, labeled=False)
    bundle = joblib.load(checkpoint)
    if not np.array_equal(wavelengths, bundle["wavelengths_nm"]):
        raise ValueError("Prediction requires the trained SuperLIBS wavelength grid and channel order")
    values = predict(bundle["model"], x)
    with output.open("w", encoding="utf-8", newline="") as stream:
        rows = csv.writer(stream)
        rows.writerow(["Pellet Name", *bundle["oxides"]])
        rows.writerows(zip(names, *values.T))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fitting = commands.add_parser("train")
    fitting.add_argument("--data", type=Path, default=Path("data/zenodo.10K_EARTH.2.4mJ.csv"))
    fitting.add_argument("--out", type=Path, default=DEFAULT_MODEL)
    fitting.add_argument("--seed", type=int, default=42)
    inference = commands.add_parser("predict")
    inference.add_argument("data", type=Path)
    inference.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    inference.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args.data, args.out, args.seed)
    else:
        run_prediction(args.data, args.model, args.out)


if __name__ == "__main__":
    main()
