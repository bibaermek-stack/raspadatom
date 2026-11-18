import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spectra_ai.train_libs import OXIDES, RIDGE_ALPHAS, split_samples, train


class SplitTests(unittest.TestCase):
    def test_identical_composition_stays_in_one_partition(self):
        chemistry = np.repeat(np.arange(32)[:, None], 3, axis=0)
        partitions = split_samples(chemistry, seed=7)
        membership = [set(chemistry[indices, 0]) for indices in partitions]
        self.assertEqual(sum(map(len, partitions)), len(chemistry))
        self.assertEqual(sum(map(len, membership)), 32)
        self.assertEqual(len(set.union(*membership)), 32)
        self.assertTrue(all(membership))


def write_toy_superlibs(path, n=120, channels=30):
    """SuperLIBS пішіміндегі шағын CSV: 1-жол — баған түрлері, 2-жол — атаулар/толқын ұзындықтары."""
    rng = np.random.default_rng(0)
    comp = rng.uniform(0, 30, (n, len(OXIDES)))
    spectra = comp @ rng.normal(0, 1, (len(OXIDES), channels)) + rng.normal(0, 0.5, (n, channels))
    wavelengths = np.linspace(240, 860, channels)
    with path.open("w", newline="", encoding="utf-8") as stream:
        rows = csv.writer(stream)
        rows.writerow(["", *["comp"] * len(OXIDES), "meta", *["wvl"] * channels])
        rows.writerow(["", *OXIDES, "Pellet Name", *wavelengths])
        for i in range(n):
            rows.writerow([i, *comp[i], f"P{i}", *spectra[i]])


class RidgeGridTests(unittest.TestCase):
    def test_grid_brackets_old_grid_and_report_flags_edge_and_spread(self):
        # бұрынғы тор 10–1000 болатын және оптимум шетте қалған: енді екі жағынан да кеңірек
        self.assertLess(min(RIDGE_ALPHAS), 100)
        self.assertGreaterEqual(max(RIDGE_ALPHAS), 100_000)
        self.assertIn(10_000, RIDGE_ALPHAS)
        with tempfile.TemporaryDirectory() as tmp:
            data, out = Path(tmp) / "toy.csv", Path(tmp) / "model.joblib"
            write_toy_superlibs(data)
            train(data, out, seed=1)
            report = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertTrue(out.exists())
        self.assertEqual(report["ridge_alphas_searched"], list(RIDGE_ALPHAS))
        self.assertEqual(set(report["validation_mae_wt_pct_points"]), {f"ridge_{a}" for a in RIDGE_ALPHAS} | {"mlp_128_64"})
        name = report["selected_model"]
        edges = {f"ridge_{RIDGE_ALPHAS[0]}", f"ridge_{RIDGE_ALPHAS[-1]}"}
        self.assertEqual(report["selected_on_grid_edge"], name in edges)
        if name.startswith("ridge_"):
            spread = report["split_seed_spread"]
            self.assertEqual(len(spread["seeds"]), 10)
            self.assertLessEqual(spread["test_mae_min"], spread["test_mae_mean"])
            self.assertLessEqual(spread["test_mae_mean"], spread["test_mae_max"])
        else:
            self.assertIsNone(report["split_seed_spread"])
        self.assertLess(report["test"]["mean_mae_wt_pct_points"], report["training_mean_baseline_test"]["mean_mae_wt_pct_points"])
