"""Stage 11B tests (run before fitting): spline formula, fold-local preprocessing (held-out labels/values cannot change
preprocessing or held-out predictions), one OOF prediction per case, identical matched populations, metric/comparison
directions, explicit failures for degenerate inputs, and real feature dependencies (label / pair / mismatch fields
cannot change features)."""
import copy
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))
import numpy as np  # noqa: E402
from rv1 import e46_cv as E  # noqa: E402

R7A = os.path.join(HERE, "..", "runs", "xai_prod_7a", "20261003T173258Z_3b655f88_548b3a05")


def toyF(n=200, seed=0):
    rng = np.random.default_rng(seed)
    logit = rng.normal(0, 2, n)
    F = {"logit": logit, "prob": list(1 / (1 + np.exp(-logit)))}
    for c in E.XAI_COLS + ("d_baseline",):
        F[c] = list(rng.normal(size=n))
    F["ig_completeness_failure"] = list((rng.random(n) < 0.2).astype(int))
    F["gradcam_zero"] = list((rng.random(n) < 0.3).astype(int))
    F["gradcam_conc_top10"] = [None if z else v for z, v in zip(F["gradcam_zero"], F["gradcam_conc_top10"])]
    y = (rng.random(n) < 1 / (1 + np.exp(-np.abs(logit) + 1))).astype(int)
    folds = np.arange(n) % 5
    return F, y, folds


class TestSpline(unittest.TestCase):
    def test_formula(self):
        kn = (-2.0, -0.5, 0.5, 2.0)
        x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        B = E.rcs_basis(x, kn)
        t1, t2, t3, t4 = kn
        p = lambda z: max(z, 0) ** 3
        for i, xi in enumerate(x):
            for j, tj in enumerate((t1, t2)):
                v = (p(xi - tj) - p(xi - t3) * (t4 - tj) / (t4 - t3) + p(xi - t4) * (t3 - tj) / (t4 - t3)) / (t4 - t1) ** 2
                self.assertAlmostEqual(B[i, j + 1], v)
        lo, hi = E.rcs_basis(np.array([5.0, 6.0, 7.0]), kn), None
        d2 = np.diff(lo[:, 1:], 2, axis=0)
        self.assertTrue(np.allclose(d2, 0))                               # linear beyond the last knot
        with self.assertRaises(ValueError):
            E.rcs_basis(x, (0.0, 0.0, 1.0, 2.0))


class TestFoldLocal(unittest.TestCase):
    def test_heldout_labels_and_values_do_not_leak(self):
        F, y, folds = toyF()
        p1, meta1 = E.oof(F, y, folds, E.MODELS["M1"])
        y2 = y.copy()
        y2[folds == 0] = 1 - y2[folds == 0]                               # flip held-out labels of fold 0
        p2, meta2 = E.oof(F, y2, folds, E.MODELS["M1"])
        np.testing.assert_allclose(p1[folds == 0], p2[folds == 0])       # fold-0 predictions unchanged
        self.assertEqual(meta1[0]["knots"], meta2[0]["knots"])
        F3 = copy.deepcopy(F)
        F3["logit"] = F3["logit"].copy()
        F3["logit"][folds == 0] += 100.0                                  # extreme held-out values
        F3["ig_conc_top10"] = [v + 50 if f == 0 else v for v, f in zip(F3["ig_conc_top10"], folds)]
        _, meta3 = E.oof(F3, y, folds, E.MODELS["M1"])
        self.assertEqual((meta1[0]["knots"], meta1[0]["medians"], meta1[0]["mean"]), (meta3[0]["knots"], meta3[0]["medians"], meta3[0]["mean"]))

    def test_one_oof_per_case_and_matched_population(self):
        F, y, folds = toyF()
        preds = {m: E.oof(F, y, folds, c)[0] for m, c in E.MODELS.items()}
        for p in preds.values():
            self.assertEqual(p.shape, y.shape)
            self.assertFalse(np.isnan(p).any())
        self.assertEqual({len(p) for p in preds.values()}, {len(y)})       # identical population for all models

    def test_explicit_failures(self):
        F, y, folds = toyF()
        F["ig_conc_top10"] = [None] * len(y)
        with self.assertRaises(ValueError):
            E.oof(F, y, folds, E.MODELS["M1"])
        F, y, folds = toyF()
        F["gradcam_zero"] = [0] * len(y)
        with self.assertRaises(ValueError):                              # zero-variance training column
            E.oof(F, y, folds, E.MODELS["M1"])


class TestSolver(unittest.TestCase):
    def test_newton_matches_generic_optimiser(self):
        from scipy.optimize import minimize
        rng = np.random.default_rng(5)
        X = rng.normal(size=(300, 4))
        y = (rng.random(300) < 1 / (1 + np.exp(-(X @ [1.0, -0.5, 0.2, 0.0] + 0.3)))).astype(float)
        w, b, it, ok = E.fit_logistic(X, y, C=1.0)
        self.assertTrue(ok)

        def obj(t):
            eta = X @ t[:4] + t[4]
            return 0.5 * t[:4] @ t[:4] + np.sum(np.logaddexp(0, eta) - y * eta)
        r = minimize(obj, np.zeros(5), method="BFGS", options={"gtol": 1e-10})
        np.testing.assert_allclose(np.r_[w, b], r.x, atol=1e-5)


class TestMetrics(unittest.TestCase):
    def test_directions(self):
        y = np.array([0, 0, 1, 1])
        self.assertEqual(E.auroc(y, np.array([0.1, 0.2, 0.8, 0.9])), 1.0)
        self.assertEqual(E.auroc(y, np.array([0.9, 0.8, 0.2, 0.1])), 0.0)
        good, bad = np.array([0.1, 0.2, 0.8, 0.9]), np.array([0.5, 0.5, 0.5, 0.5])
        self.assertLess(E.brier(y, good), E.brier(y, bad))
        rng = np.random.default_rng(1)
        yy = (rng.random(300) < 0.4).astype(int)
        strong = yy + rng.normal(0, 0.5, 300)
        weak = yy + rng.normal(0, 3, 300)
        ci, und = E.paired_bootstrap(yy, {"A": strong, "B": weak}, [("A", "B")], B_=200, seed=3)
        self.assertGreater(ci["A-B"]["dauc"][0], 0)                       # A - B > 0 when A is better
        self.assertEqual(und, 0)
        ci2, _ = E.paired_bootstrap(yy, {"A": strong, "B": strong}, [("A", "B")], B_=50, seed=3)
        self.assertEqual(ci2["A-B"]["dauc"], [0.0, 0.0])

    def test_calibration_perfect(self):
        rng = np.random.default_rng(2)
        p = rng.uniform(0.05, 0.95, 20000)
        y = (rng.random(20000) < p).astype(int)
        c = E.calibration(y, p)
        self.assertAlmostEqual(c["slope"], 1.0, delta=0.1)
        self.assertAlmostEqual(c["intercept"], 0.0, delta=0.1)


@unittest.skipUnless(os.path.isdir(R7A), "Stage 7A run not present")
class TestRealFeatureDependencies(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(R7A, "chunk_00", "records.json"), encoding="utf-8") as f:
            recs = json.load(f)
        z = np.load(os.path.join(R7A, "chunk_00", "maps.npz"))
        self.i = next(i for i, r in enumerate(recs) if r["modality"] == "clinical")
        self.rec = recs[self.i]
        self.der = next(r for r in recs if r["pair_id"] == self.rec["pair_id"] and r["modality"] == "dermoscopy")
        self.ig, self.gc = z["ig512_primary"][self.i], z["gradcam"][self.i]

    def build(self, rec):
        return E.build_features({"logit": "0.3", "prob": "0.57"}, E.sanitize(rec), self.ig, self.gc)

    def test_label_pair_mismatch_do_not_change_features(self):
        base = self.build(self.rec)
        r2 = copy.deepcopy(self.rec)
        r2["label_mel"] = "0" if r2["label_mel"] == "1" else "1"
        r2["pair_id"] = "OTHER"
        r2["mismatch"] = {"ig512": {"auc_ins": 9.9, "donor": "x"}}
        r2["dermoscopy_twin"] = self.der
        self.assertEqual(self.build(r2), base)
        s = E.sanitize(r2)
        flat = json.dumps(s)
        for bad in ("label_mel", "pair_id", "mismatch", "donor", "dermoscopy"):
            self.assertNotIn(bad, flat)

    def test_features_depend_on_clinical_artifacts(self):
        base, _ = self.build(self.rec)
        r2 = copy.deepcopy(self.rec)
        r2["controls"]["random"]["auc_ins_each"] = [x + 0.1 for x in r2["controls"]["random"]["auc_ins_each"]]
        self.assertAlmostEqual(self.build(r2)[0]["ig_delta_ins"], base["ig_delta_ins"] - 0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
