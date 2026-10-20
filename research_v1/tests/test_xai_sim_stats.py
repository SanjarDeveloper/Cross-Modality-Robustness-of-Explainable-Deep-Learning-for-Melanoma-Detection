"""Stage 10D tests: lesion-cluster weights (all images of a lesion move together; multiplicity; shared draws),
summaries refuse unavailable values inside a population (no zero substitution), undefined replicates are counted,
population counts, contrast directions via control_scores, paired change on the same subset."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))
import numpy as np  # noqa: E402
from rv1 import xai_controls as XK, xai_sim_stats as XT  # noqa: E402


class TestBootstrap(unittest.TestCase):
    def test_lesion_weights(self):
        cl = ["A", "A", "B", "C", "C", "C"]
        W = XT.lesion_weights(cl, 200, 1)
        self.assertEqual(W.shape, (200, 6))
        self.assertTrue(np.all(W[:, 0] == W[:, 1]))                    # same lesion -> same multiplicity
        self.assertTrue(np.all(W[:, 3] == W[:, 5]))
        self.assertTrue(np.all(W[:, [0, 2, 3]].sum(1) == 3))           # K = 3 lesion draws per replicate
        np.testing.assert_array_equal(W, XT.lesion_weights(cl, 200, 1))  # deterministic shared draws

    def test_summarise_rules(self):
        v = np.array([1.0, 2.0, np.nan, 4.0])
        m = ~np.isnan(v)
        s = XT.summarise(v, m, XT.lesion_weights(["a", "b", "c", "d"], 100, 2))
        self.assertEqual((s["n"], s["median"]), (3, 2.0))
        with self.assertRaises(ValueError):                             # unavailable never treated as a value
            XT.summarise(v, np.ones(4, bool), np.ones((5, 4), int))
        W = np.zeros((4, 4), int)
        W[0] = 1
        self.assertEqual(XT.summarise(v, m, W)["undefined_replicates"], 3)

    def test_population(self):
        p = XT.population([True, True, False, True], ["1", "0", "1", "0"], ["A", "A", "B", "C"])
        self.assertEqual(p, {"images": 3, "lesions": 2, "mel": 1, "nonmel": 2})


class TestDirections(unittest.TestCase):
    def test_control_directions_and_mean_of_five(self):
        sc = XK.control_scores({"auc_ins": 0.6, "auc_del": 0.2},
                               {"status": "available", "auc_ins_each": [0.4, 0.5, 0.3, 0.4, 0.4], "auc_del_each": [0.5] * 5})
        self.assertAlmostEqual(sc["insertion_advantage"], 0.6 - 0.4)   # explainer - control
        self.assertAlmostEqual(sc["deletion_advantage"], 0.5 - 0.2)    # control - explainer

    def test_paired_change_same_subset(self):
        v0 = np.array([0.1, np.nan, 0.3, 0.2])
        v1 = np.array([0.0, 0.5, np.nan, 0.1])
        both = ~np.isnan(v0) & ~np.isnan(v1)
        self.assertEqual(both.tolist(), [True, False, False, True])
        W = XT.lesion_weights(list("abcd"), 50, 3)
        d = XT.summarise(v1 - v0, both, W, ("median",))
        self.assertAlmostEqual(d["median"], -0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
