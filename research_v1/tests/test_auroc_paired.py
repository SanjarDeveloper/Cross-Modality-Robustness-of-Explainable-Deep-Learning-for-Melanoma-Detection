"""Stage 4B2 tests: AUROC with ties, paired DeLong vs direct placement calculation, symmetry,
zero variance, bootstrap behaviour for paired AUROC."""
import os
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import numpy as np  # noqa: E402
from rv1 import auroc_paired as AP, rq1_stats as RS  # noqa: E402


def psi(x, y):
    return 1.0 if x > y else (0.5 if x == y else 0.0)


def direct(a, b, y):
    """Independent O(mn) DeLong via explicit placement values and covariance sums."""
    P = [i for i in range(len(y)) if y[i] == 1]
    N = [j for j in range(len(y)) if y[j] == 0]
    m, n = len(P), len(N)
    V10 = [[sum(psi(s[i], s[j]) for j in N) / n for i in P] for s in (a, b)]
    V01 = [[sum(psi(s[i], s[j]) for i in P) / m for j in N] for s in (a, b)]
    A = [sum(v) / m for v in V10]

    def cov(u, v, k):
        mu, mv = sum(u) / k, sum(v) / k
        return sum((p - mu) * (q - mv) for p, q in zip(u, v)) / (k - 1)
    S = [[cov(V10[r], V10[s], m) / m + cov(V01[r], V01[s], n) / n for s in range(2)] for r in range(2)]
    return A, S[0][0] + S[1][1] - 2 * S[0][1], S[0][1]


class TestAUC(unittest.TestCase):
    def test_known_with_ties(self):
        self.assertEqual(AP.auc([0.9, 0.8], [0.1, 0.2]), 1.0)
        self.assertEqual(AP.auc([0.1], [0.9]), 0.0)
        self.assertEqual(AP.auc([0.5], [0.5]), 0.5)
        self.assertAlmostEqual(AP.auc([0.5, 0.7], [0.5, 0.2]), (0.5 + 1 + 1 + 1) / 4)
        from scipy.stats import mannwhitneyu
        rng = np.random.default_rng(0)
        p, q = rng.integers(0, 5, 30) / 4, rng.integers(0, 5, 40) / 4      # heavy ties
        self.assertAlmostEqual(AP.auc(p, q), mannwhitneyu(p, q).statistic / 1200, places=12)


class TestDeLong(unittest.TestCase):
    def data(self, seed=1, ties=False):
        rng = np.random.default_rng(seed)
        y = np.array([1] * 15 + [0] * 25)
        a = rng.normal(y * 1.0, 1)
        b = 0.6 * a + rng.normal(y * 0.3, 0.8)
        if ties:
            a, b = np.round(a, 1), np.round(b, 1)
        return a, b, y

    def test_against_direct(self):
        for ties in (False, True):
            a, b, y = self.data(ties=ties)
            r = AP.delong_paired(a, b, y)
            A, var, cov = direct(list(a), list(b), list(y))
            self.assertAlmostEqual(r["auc_a"], A[0], places=12)
            self.assertAlmostEqual(r["auc_b"], A[1], places=12)
            self.assertAlmostEqual(r["var_diff"], var, places=12)
            self.assertAlmostEqual(r["cov_ab"], cov, places=12)
            self.assertNotEqual(r["cov_ab"], 0.0)          # covariance retained

    def test_swap(self):
        a, b, y = self.data()
        r1, r2 = AP.delong_paired(a, b, y), AP.delong_paired(b, a, y)
        self.assertAlmostEqual(r1["diff"], -r2["diff"], places=15)
        self.assertAlmostEqual(r1["p_value"], r2["p_value"], places=15)

    def test_identical_zero_variance(self):
        a, _, y = self.data()
        r = AP.delong_paired(a, a.copy(), y)
        self.assertEqual(r["diff"], 0.0)
        self.assertEqual(r["var_diff"], 0.0)
        self.assertIsNone(r["p_value"])
        self.assertIn("zero variance", r["inference"])

    def test_one_class_undefined(self):
        self.assertEqual(AP.delong_paired([0.1, 0.2, 0.3], [0.1, 0.2, 0.4], [1, 1, 1])["status"], "undefined")

    def test_variance_vs_bootstrap(self):
        """Plan 03_rev2 §14: DeLong variance within 15% of a 5000-replicate bootstrap (synthetic)."""
        rng = np.random.default_rng(5)
        y = np.array([1] * 150 + [0] * 250)
        a = rng.normal(y * 1.0, 1)
        b = 0.5 * a + rng.normal(y * 0.5, 0.9)
        r = AP.delong_paired(a, b, y)
        d = []
        for _ in range(5000):
            i = rng.integers(0, len(y), len(y))
            if 0 < y[i].sum() < len(y):
                d.append(AP.auc(a[i][y[i] == 1], a[i][y[i] == 0]) - AP.auc(b[i][y[i] == 1], b[i][y[i] == 0]))
        self.assertLess(abs(np.var(d, ddof=1) / r["var_diff"] - 1), 0.15)


class TestPairedBootstrap(unittest.TestCase):
    def test_pairs_multiplicity_and_one_class(self):
        pairs = {f"p{i}": {"pair_id": f"p{i}", "label": int(i < 3), "prob_d": i / 10, "prob_c": (10 - i) / 10}
                 for i in range(10)}
        ids = sorted(pairs)

        def stat(ps):
            y = np.array([p["label"] for p in ps])
            if y.min() == y.max():
                return None
            r = AP.delong_paired([p["prob_d"] for p in ps], [p["prob_c"] for p in ps], y) \
                if min(y.sum(), len(y) - y.sum()) >= 2 else None
            ad = AP.auc([p["prob_d"] for p in ps if p["label"]], [p["prob_d"] for p in ps if not p["label"]])
            ac = AP.auc([p["prob_c"] for p in ps if p["label"]], [p["prob_c"] for p in ps if not p["label"]])
            return ad, ac, ad - ac
        a = RS.pair_bootstrap(pairs, ids, stat, 500, 11)
        b = RS.pair_bootstrap(pairs, ids, stat, 500, 11)
        self.assertGreater(a["n_undefined"], 0)                      # one-class replicates occur and are counted
        self.assertEqual(a["n_defined"] + a["n_undefined"], 500)
        self.assertTrue(np.array_equal(a["replicates"], b["replicates"]))
        rep = a["replicates"]
        self.assertTrue(np.allclose(rep[:, 2], rep[:, 0] - rep[:, 1]))   # both modalities from the same draw

    def test_row_order_invariance(self):
        def row(pid, mod, lab, p):
            return {"record_uid": f"{pid}:{mod}", "pair_id": pid, "modality": mod, "label_mel": str(lab),
                    "prob": repr(p), "pred": str(int(p >= 0.5)), "threshold": "0.5", "threshold_sha256": "t",
                    "checkpoint_sha256": "c", "in_derm7pt_primary": "1", "same_file_pair": "0", "status": "ok"}
        rows = [row(f"c{i}", m, int(i < 4), ((i * 7 + (m == "clinical") * 3) % 10) / 10)
                for i in range(10) for m in ("dermoscopy", "clinical")]
        r1 = RS.build_pairs(rows, 0.5, "t", "c")
        r2 = RS.build_pairs(list(reversed(rows)), 0.5, "t", "c")

        def res(pp):
            ps = [pp[k] for k in sorted(pp)]
            return AP.delong_paired([p["prob_d"] for p in ps], [p["prob_c"] for p in ps], [p["label"] for p in ps])
        self.assertEqual(res(r1), res(r2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
