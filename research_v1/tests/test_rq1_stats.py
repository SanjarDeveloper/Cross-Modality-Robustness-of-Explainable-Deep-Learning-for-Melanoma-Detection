"""Stage 4B1 tests: exact McNemar, pair reconstruction checks, pair bootstrap."""
import math
import os
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import numpy as np  # noqa: E402
from rv1 import rq1_stats as RS  # noqa: E402

T, TS, CS = 0.5, "th", "ck"


def row(pid, mod, label, prob, primary="1", **kw):
    r = {"record_uid": f"{pid}:{mod}", "pair_id": pid, "modality": mod, "label_mel": str(label),
         "prob": repr(prob), "pred": str(int(prob >= T)), "threshold": repr(T), "threshold_sha256": TS,
         "checkpoint_sha256": CS, "in_derm7pt_primary": primary, "same_file_pair": "0", "status": "ok"}
    r.update(kw)
    return r


def brute_mcnemar(b, c):
    """Independent calculation: enumerate all 2^n sign outcomes for small n; two-sided as the total
    probability of outcomes no more likely than the observed one (equal-tail for p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    probs = [math.comb(n, k) / 2 ** n for k in range(n + 1)]
    obs = probs[b]
    return min(1.0, sum(p for p in probs if p <= obs + 1e-15))


class TestMcNemar(unittest.TestCase):
    def test_against_independent(self):
        for b in range(0, 13):
            for c in range(0, 13):
                self.assertAlmostEqual(RS.mcnemar_exact(b, c)["p_value"], brute_mcnemar(b, c), places=12, msg=(b, c))

    def test_scipy_and_known(self):
        from scipy.stats import binomtest
        for b, c in ((117, 9), (10, 3), (0, 5)):
            self.assertAlmostEqual(RS.mcnemar_exact(b, c)["p_value"], binomtest(b, b + c, 0.5).pvalue,
                                   delta=1e-12 * max(1.0, binomtest(b, b + c, 0.5).pvalue))
        self.assertAlmostEqual(RS.mcnemar_exact(1, 0)["p_value"], 1.0)
        self.assertAlmostEqual(RS.mcnemar_exact(5, 0)["p_value"], 2 / 32)

    def test_zero_discordant(self):
        self.assertEqual(RS.mcnemar_exact(0, 0)["p_value"], 1.0)

    def test_swap_symmetry(self):
        self.assertEqual(RS.mcnemar_exact(117, 9)["p_exact"], RS.mcnemar_exact(9, 117)["p_exact"])
        self.assertGreater(RS.mcnemar_exact(117, 9)["p_value"], 0.0)


class TestPairs(unittest.TestCase):
    def rows(self):
        return [row("a", "dermoscopy", 1, 0.9), row("a", "clinical", 1, 0.1),
                row("b", "clinical", 0, 0.2), row("b", "dermoscopy", 0, 0.7)]

    def test_build_by_id_not_order(self):
        p = RS.build_pairs(list(reversed(self.rows())), T, TS, CS)
        self.assertEqual((p["a"]["pred_d"], p["a"]["pred_c"]), (1, 0))
        self.assertEqual((p["b"]["pred_d"], p["b"]["pred_c"]), (1, 0))

    def test_rejections(self):
        r = self.rows() + [row("a", "clinical", 1, 0.3, record_uid="a:clinical2")]
        with self.assertRaisesRegex(RS.PairError, "duplicate clinical"):
            RS.build_pairs(r, T, TS, CS)
        with self.assertRaisesRegex(RS.PairError, "duplicate record_uid"):
            RS.build_pairs(self.rows() + [self.rows()[0]], T, TS, CS)
        with self.assertRaisesRegex(RS.PairError, "incomplete"):
            RS.build_pairs(self.rows()[:3], T, TS, CS)
        bad = self.rows()
        bad[1] = row("a", "clinical", 0, 0.1)
        with self.assertRaisesRegex(RS.PairError, "label inconsistent"):
            RS.build_pairs(bad, T, TS, CS)
        bad = self.rows()
        bad[0]["pred"] = "0"
        with self.assertRaisesRegex(RS.PairError, "pred !="):
            RS.build_pairs(bad, T, TS, CS)
        bad = self.rows()
        bad[0]["prob"] = "nan"
        with self.assertRaisesRegex(RS.PairError, "non-finite"):
            RS.build_pairs(bad, T, TS, CS)
        with self.assertRaisesRegex(RS.PairError, "identity"):
            RS.build_pairs(self.rows(), T, "other", CS)

    def test_swap_reverses_direction(self):
        pairs = [{"label": 1, "pred_d": 1, "pred_c": 0}, {"label": 1, "pred_d": 1, "pred_c": 1}]
        swapped = [dict(p, pred_d=p["pred_c"], pred_c=p["pred_d"]) for p in pairs]
        self.assertEqual(RS.sens_diff(pairs)[2], -RS.sens_diff(swapped)[2])


class TestBootstrap(unittest.TestCase):
    def setUp(self):
        self.pairs = {f"p{i}": {"pair_id": f"p{i}", "label": 1, "pred_d": int(i % 3 != 0), "pred_c": int(i % 4 == 0)}
                      for i in range(40)}
        self.ids = sorted(self.pairs)

    def stat(self, ps):
        r = RS.sens_diff(ps)
        return None if r is None else r

    def test_pairs_kept_together_and_multiplicity(self):
        rng = np.random.Generator(np.random.PCG64(1))
        draw = RS.resample_ids(self.ids, rng)
        self.assertEqual(len(draw), len(self.ids))
        self.assertLess(len(set(draw)), len(draw))               # repeats present and retained
        reps = [self.pairs[i] for i in draw]
        for p in reps:                                            # each resampled unit carries both modalities
            self.assertIn("pred_d", p)
            self.assertIn("pred_c", p)
        m = sum(p["pred_d"] for p in reps) / len(reps)
        self.assertAlmostEqual(self.stat(reps)[0], m)            # multiplicity enters the estimate

    def test_seed_reproducible(self):
        a = RS.pair_bootstrap(self.pairs, self.ids, self.stat, 200, 7)
        b = RS.pair_bootstrap(self.pairs, self.ids, self.stat, 200, 7)
        self.assertTrue(np.array_equal(a["replicates"], b["replicates"]))
        self.assertEqual(a["ci"], b["ci"])
        c = RS.pair_bootstrap(self.pairs, self.ids, self.stat, 200, 8)
        self.assertFalse(np.array_equal(a["replicates"], c["replicates"]))

    def test_undefined_counted_not_redrawn(self):
        calls = {"n": 0}

        def flaky(ps):
            calls["n"] += 1
            return None if calls["n"] % 10 == 0 else (0.5,)
        r = RS.pair_bootstrap(self.pairs, self.ids, flaky, 100, 3)
        self.assertEqual(calls["n"], 100)
        self.assertEqual(r["n_undefined"], 10)
        self.assertTrue(r["unreliable"])                          # 10% > 5%


if __name__ == "__main__":
    unittest.main(verbosity=2)
