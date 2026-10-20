"""Stage 7C tests: normalization boundaries/no clipping, normalized contrast == raw/d, matched filtering,
OLS on a known example, rank deficiency, bootstrap multiplicity on the same population."""
import os
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import numpy as np  # noqa: E402
from rv1 import rq4_secondary as S, rq4_stats as R4  # noqa: E402


class TestNormalization(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(S.norm_category(0.05, 0.05), "defined")
        self.assertEqual(S.norm_category(0.0499999, 0.05), "near_zero_denominator")
        self.assertEqual(S.norm_category(-0.05, 0.05), "baseline_more_melanoma_like")
        self.assertEqual(S.norm_category(-0.0499, 0.05), "near_zero_denominator")
        self.assertEqual(S.normalized(0.5, 0.2, 0.3, 0.4, 0.05)[1:], (None, None))     # negative d never normalized

    def test_no_clipping(self):
        cat, ni, nd = S.normalized(auc_ins=0.95, auc_del=0.0, p_full=0.6, p_base=0.5, delta=0.05)
        self.assertEqual(cat, "defined")
        self.assertAlmostEqual(ni, 4.5)          # (0.95-0.5)/0.1 > 1 kept
        self.assertAlmostEqual(nd, 6.0)

    def test_equals_raw_over_d(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            pf, pb = rng.uniform(0.3, 1), rng.uniform(0, 0.25)
            e, r = rng.uniform(0, 1, 2), rng.uniform(0, 1, 2)
            cat, ci, cd = S.norm_contrasts(tuple(e), tuple(r), pf, pb, 0.05)
            d = pf - pb
            self.assertAlmostEqual(ci, (e[0] - r[0]) / d, places=12)
            self.assertAlmostEqual(cd, (r[1] - e[1]) / d, places=12)


class TestMatched(unittest.TestCase):
    def test_both_modalities(self):
        rows = [{"pair_id": "a", "modality": "dermoscopy", "v": 1.0}, {"pair_id": "a", "modality": "clinical", "v": 2.0},
                {"pair_id": "b", "modality": "dermoscopy", "v": 1.0}, {"pair_id": "b", "modality": "clinical", "v": None}]
        self.assertEqual(sorted(S.matched(rows, ("v",))), ["a"])
        with self.assertRaises(ValueError):
            S.matched(rows[:3], ("v",))
        with self.assertRaises(ValueError):
            S.matched(rows + rows[:1], ("v",))


class TestOLS(unittest.TestCase):
    def test_known(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(200, 2))
        y = 0.3 - 2.0 * X[:, 0] + 0.5 * X[:, 1]
        f = S.ols_fit(y, X)
        for a, b in zip(f["coef"], [0.3, -2.0, 0.5]):
            self.assertAlmostEqual(a, b, places=10)
        self.assertAlmostEqual(f["r2"], 1.0)
        self.assertEqual(f["rank"], 3)

    def test_rank_deficient(self):
        x = np.arange(10.0)
        self.assertIsNone(S.ols_fit(x * 2, np.column_stack([x, 2 * x])))
        b = S.bootstrap_ols(np.array([1.0, 2, 3]), np.column_stack([[1.0, 1, 1], [2.0, 2, 2]]), 20, 3)
        self.assertEqual(b["n_undefined_rank_deficient"], 20)
        self.assertIsNone(b["ci95"])


class TestSamePopulation(unittest.TestCase):
    def test_shared_draws(self):
        raw = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        d = np.array([0.5, 0.5, 0.25, 0.25, 1.0])
        bs, _ = R4.bootstrap_stats({"raw": raw, "norm": raw / d}, 200, 9)
        rng = np.random.Generator(np.random.PCG64(9))
        idx = rng.integers(0, 5, 5)
        self.assertAlmostEqual(bs["raw"]["median"]["replicates"][0], float(np.median(raw[idx])))
        self.assertAlmostEqual(bs["norm"]["median"]["replicates"][0], float(np.median((raw / d)[idx])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
