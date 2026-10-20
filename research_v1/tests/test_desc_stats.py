"""Stage 4B3 tests: correctness decomposition, ECE bins, lesion-cluster bootstrap."""
import os
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import numpy as np  # noqa: E402
from rv1 import desc_stats as DS, rq1_stats as RS  # noqa: E402


class TestCorrectness(unittest.TestCase):
    def pairs(self):
        return ([{"label": 1, "pred_d": 1, "pred_c": 0}] * 3 + [{"label": 1, "pred_d": 0, "pred_c": 1}] +
                [{"label": 0, "pred_d": 1, "pred_c": 0}] * 5 + [{"label": 0, "pred_d": 0, "pred_c": 1}] * 2 +
                [{"label": 0, "pred_d": 0, "pred_c": 0}])

    def test_decomposition(self):
        p = self.pairs()
        t = DS.correctness_table(p)
        m = DS.correctness_table([x for x in p if x["label"] == 1])
        n = DS.correctness_table([x for x in p if x["label"] == 0])
        self.assertEqual(m["dermoscopy_only_correct"], 3)        # melanoma: detected only under dermoscopy
        self.assertEqual(n["clinical_only_correct"], 5)          # non-melanoma: dermoscopy FP only
        for k in ("both_correct", "dermoscopy_only_correct", "clinical_only_correct", "both_incorrect"):
            self.assertEqual(t[k], m[k] + n[k])
        self.assertEqual((t["dermoscopy_only_correct"], t["clinical_only_correct"]), (5, 6))

    def test_swap(self):
        p = self.pairs()
        s = [dict(x, pred_d=x["pred_c"], pred_c=x["pred_d"]) for x in p]
        a, b = DS.correctness_table(p), DS.correctness_table(s)
        self.assertEqual(a["dermoscopy_only_correct"], b["clinical_only_correct"])
        self.assertAlmostEqual(a["accuracy_diff"], -b["accuracy_diff"])
        self.assertEqual(RS.mcnemar_exact(5, 6)["p_exact"], RS.mcnemar_exact(6, 5)["p_exact"])


class TestECE(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(DS.bin_index(0.0), 0)
        self.assertEqual(DS.bin_index(1.0), 14)                  # final bin includes 1
        # edges are the exact rationals b/15; the float nearest 1/15 lies just BELOW 1/15 -> bin 0,
        # the next float up lies above it -> bin 1 (left-closed); 0.2 (float) is just above 3/15 -> bin 3
        from fractions import Fraction
        self.assertLess(Fraction(1 / 15), Fraction(1, 15))
        self.assertEqual(DS.bin_index(1 / 15), 0)
        self.assertEqual(DS.bin_index(float(np.nextafter(1 / 15, 1))), 1)
        self.assertGreater(Fraction(0.2), Fraction(3, 15))
        self.assertEqual(DS.bin_index(0.2), 3)
        self.assertEqual(DS.bin_index(0.5), 7)
        self.assertEqual(DS.bin_index(14 / 15), 14)

    def test_hand_example_and_empty(self):
        p = [0.0, 0.02, 0.5, 1.0]
        y = [0, 0, 1, 1]
        r = DS.ece_bins(p, y)
        # bin0: {0,0.02} mean 0.01, obs 0 -> 2/4*0.01; bin7 [7/15,8/15): {0.5} gap .5 -> 1/4*.5; bin14: {1.0} gap 0
        self.assertAlmostEqual(r["ece"], 0.5 * 0.01 + 0.25 * 0.5, places=15)
        self.assertEqual(r["bins"][7]["n"], 1)
        self.assertEqual(r["n_empty_bins"], 12)
        e = r["bins"][3]
        self.assertTrue(e["empty"])
        self.assertEqual((e["weight"], e["contribution"]), (0.0, 0.0))
        self.assertIsNone(e["mean_prob"])

    def test_invalid(self):
        for p, y in (([1.1], [1]), ([-0.1], [0]), ([float("nan")], [0]), ([0.5], [2]), ([], [])):
            with self.assertRaises(ValueError):
                DS.ece_bins(p, y)


class TestLesionBootstrap(unittest.TestCase):
    def setUp(self):
        self.clusters = {f"L{i}": [{"lesion": f"L{i}", "label": int(i < 3), "prob": (i + k) / 12} for k in range(1 + i % 3)]
                         for i in range(10)}
        self.ids = sorted(self.clusters)

    def test_all_images_and_multiplicity(self):
        rng = np.random.Generator(np.random.PCG64(4))
        draw = RS.resample_ids(self.ids, rng)
        flat = DS.flatten_clusters([self.clusters[i] for i in draw])
        self.assertEqual(len(flat), sum(len(self.clusters[i]) for i in draw))   # every image of each draw
        for lid in set(draw):
            self.assertEqual(sum(r["lesion"] == lid for r in flat), draw.count(lid) * len(self.clusters[lid]))

    def test_reproducible_and_undefined(self):
        def stat(cl):
            s, _ = DS.binary_metrics(DS.flatten_clusters(cl), 0.3)
            return None if s is None else (s,)
        a = RS.pair_bootstrap(self.clusters, self.ids, stat, 300, 9)
        b = RS.pair_bootstrap(self.clusters, self.ids, stat, 300, 9)
        self.assertTrue(np.array_equal(a["replicates"], b["replicates"]))
        self.assertGreater(a["n_undefined"], 0)                  # no-melanoma replicates counted
        self.assertEqual(a["n_defined"] + a["n_undefined"], 300)


if __name__ == "__main__":
    unittest.main(verbosity=2)
