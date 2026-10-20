"""Stage 8C tests: worst-case loss, crossing (incl. censoring/undefined), censored quantiles, ranking ties,
lesion bootstrap multiplicity and shared draws, AUROC via ranks, join rejection."""
import os
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import numpy as np  # noqa: E402
from rv1 import rq2_stats as S  # noqa: E402

G = S.GRID


class TestWorst(unittest.TestCase):
    def test_zero_and_bounds(self):
        self.assertEqual(S.worst_case([0.9] * 9), (0.0, 0.0))                 # all equal -> 0 at s=0 (tie rule)
        w, lvl = S.worst_case([0.9, 0.95, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91])
        self.assertEqual((w, lvl), (0.0, 0.0))                                 # improvements never make it negative
        w, lvl = S.worst_case([0.9, 0.8, 0.85, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9])
        self.assertAlmostEqual(w, 0.1)
        self.assertEqual(lvl, 0.125)                                           # tie -> smallest severity
        self.assertEqual(S.worst_case([0.9, None] + [0.9] * 7), (None, None))

    def test_worst_level_changes_across_replicates(self):
        a = S.worst_case([0.9, 0.7, 0.8] + [0.9] * 6)[1]
        b = S.worst_case([0.9, 0.8, 0.6] + [0.9] * 6)[1]
        self.assertNotEqual(a, b)


class TestCrossing(unittest.TestCase):
    def test_cases(self):
        self.assertEqual(S.crossing([0.9] * 9), S.CENSORED)
        self.assertIsNone(S.crossing([0.9, None] + [0.9] * 7))
        self.assertAlmostEqual(S.crossing([0.9, 0.85, 0.75] + [0.9] * 6), 0.125 + 0.05 / 0.10 * 0.125)  # interpolated
        s0 = 157 / 170
        self.assertEqual(S.crossing([s0, 150 / 170, 140 / 170] + [s0] * 6), 0.25)   # exact boundary at a grid point
        # nonmonotone: first crossing counts despite later recovery
        self.assertAlmostEqual(S.crossing([0.9, 0.7, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]), 0.125 * 0.1 / 0.2)

    def test_censored_quantiles(self):
        q = S.censored_quantiles([0.2, 0.3, S.CENSORED, S.CENSORED])
        self.assertEqual(q[0.025], 0.2)
        self.assertEqual(q[0.5], 0.3)
        self.assertEqual(q[0.975], S.CENSORED)
        self.assertEqual(S.censored_quantiles([S.CENSORED] * 5)[0.5], S.CENSORED)
        self.assertIsNone(S.censored_quantiles([])[0.5])


class TestRanking(unittest.TestCase):
    def test_ties_and_sum(self):
        reps = [dict(zip(S.FACTORS, [0.1, 0.1, 0, 0, 0])), dict(zip(S.FACTORS, [0, 0, 0, 0.2, 0])),
                dict(zip(S.FACTORS, [None, 0, 0, 0, 0]))]
        r = S.ranking_shares(reps)
        self.assertEqual((r["valid"], r["excluded"], r["tie_replicates"]), (2, 1, 1))
        self.assertAlmostEqual(sum(r["shares"].values()), 1.0)
        self.assertAlmostEqual(r["shares"]["blur"], 0.25)
        self.assertAlmostEqual(r["shares"]["scale"], 0.5)


class TestBootstrapAndAuc(unittest.TestCase):
    def test_lesion_draw_multiplicity_and_shared(self):
        clusters = ["L1", "L1", "L2", "L3", "L3", "L3"]
        ids, groups = S.lesion_index(clusters)
        rng = np.random.Generator(np.random.PCG64(1))
        idx = S.draw_rows(groups, rng)
        rng2 = np.random.Generator(np.random.PCG64(1))
        pick = rng2.integers(0, 3, 3)
        self.assertEqual(len(idx), sum(len(groups[k]) for k in pick))      # all images of each draw, with repeats
        base, pert = np.array([.1, .9, .2, .8, .3, .7]), np.array([.2, .8, .1, .9, .4, .6])
        y = np.array([0, 1, 0, 1, 0, 1], bool)
        # one draw is applied to baseline and perturbed units: the paired difference uses identical images
        d = S.auc_ranks(pert[idx], y[idx]) if 0 < y[idx].sum() < len(idx) else None
        b = S.auc_ranks(base[idx], y[idx]) if 0 < y[idx].sum() < len(idx) else None
        self.assertEqual(d is None, b is None)

    def test_auc(self):
        from rv1 import auroc_paired as AP
        rng = np.random.default_rng(0)
        p, y = rng.integers(0, 5, 60) / 4, rng.random(60) < 0.3
        self.assertAlmostEqual(S.auc_ranks(p, y), AP.auc(p[y], p[~y]), places=12)
        self.assertIsNone(S.auc_ranks(p, np.zeros(60, bool)))


class TestJoins(unittest.TestCase):
    def test_wrong_files_rejected(self):
        with self.assertRaises(Exception):
            S.load_units(os.path.dirname(__file__), {}, "x", "y", 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
