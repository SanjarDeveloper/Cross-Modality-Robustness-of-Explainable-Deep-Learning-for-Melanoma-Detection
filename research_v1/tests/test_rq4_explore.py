"""Stage 7D tests: Spearman (identical/reversed/ties/constant), top-k exactness and tie handling,
class-contrast direction and stratified bootstrap, donor-multiplicity audit."""
import os
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import numpy as np  # noqa: E402
from rv1 import rq4_explore as X  # noqa: E402


class TestSpearman(unittest.TestCase):
    def test_cases(self):
        a = np.arange(16.0).reshape(4, 4)
        self.assertAlmostEqual(X.spearman(a, a), 1.0)
        self.assertAlmostEqual(X.spearman(a, -a), -1.0)
        t = np.array([0, 0, 0, 1, 2, 2], float)
        from scipy.stats import spearmanr
        self.assertAlmostEqual(X.spearman(t, t[::-1] * 0 + np.array([1, 2, 3, 4, 5, 6.0])),
                               spearmanr(t, [1, 2, 3, 4, 5, 6]).statistic, places=12)
        self.assertIsNone(X.spearman(np.zeros(9), a.reshape(-1)[:9]))      # constant -> unavailable, not 0


class TestTopK(unittest.TestCase):
    def test_exact_k_and_ties(self):
        m = np.zeros((10, 10))
        sel, k, tie = X.topk_indices(m)
        self.assertEqual(k, 10)
        self.assertEqual(sel.tolist(), list(range(10)))                     # ties -> ascending index
        self.assertTrue(tie)
        m2 = np.arange(100.0).reshape(10, 10)
        sel2, k2, tie2 = X.topk_indices(m2)
        self.assertEqual(sorted(sel2.tolist()), list(range(90, 100)))
        self.assertFalse(tie2)
        self.assertEqual(X.topk_indices(np.ones(101))[1], 11)               # ceil(10.1)

    def test_known_overlap(self):
        a = np.arange(100.0)
        b = a.copy()
        b[95:] = -1                                                          # 5 of a's top-10 removed from b's top
        ov, k, _, _ = X.topk_overlap(a, b)
        self.assertEqual(k, 10)
        self.assertAlmostEqual(ov, 0.5)                                      # |∩|/k, not Jaccard (would be 5/15)
        self.assertAlmostEqual(X.topk_overlap(a, a)[0], 1.0)


class TestStratified(unittest.TestCase):
    def test_direction_and_multiplicity(self):
        g = {"melanoma": np.array([0.3, 0.4, 0.5]), "non_melanoma": np.array([0.0, 0.1])}
        r = X.stratified_bootstrap(g, lambda v: float(np.median(v)), 200, 3)
        self.assertEqual(r["diff_label"], "melanoma - non_melanoma")
        self.assertGreater(r["diff_ci95"][0], 0)
        r2 = X.stratified_bootstrap(g, lambda v: float(np.median(v)), 200, 3)
        self.assertTrue(np.array_equal(r["replicates"]["diff"], r2["replicates"]["diff"]))
        rng = np.random.Generator(np.random.PCG64(3))
        i1 = rng.integers(0, 3, 3)
        self.assertAlmostEqual(r["replicates"]["melanoma"][0], float(np.median(g["melanoma"][i1])))


class TestDonorAudit(unittest.TestCase):
    def test_derangement(self):
        recs = ["a", "b", "c"]
        good = {f"{r}|m|e|none": {"record_uid": r, "donor": recs[(i + 1) % 3], "group": ["1", "m", "e", "none"], "shift": 1}
                for i, r in enumerate(recs)}
        self.assertTrue(X.donor_audit(good)["1|m|e|none"]["one_to_one_derangement"])
        bad = dict(good)
        bad["c|m|e|none"] = dict(good["c|m|e|none"], donor="b")             # b donates twice
        a = X.donor_audit(bad)["1|m|e|none"]
        self.assertFalse(a["one_to_one_derangement"])
        self.assertEqual(a["max_donor_multiplicity"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
