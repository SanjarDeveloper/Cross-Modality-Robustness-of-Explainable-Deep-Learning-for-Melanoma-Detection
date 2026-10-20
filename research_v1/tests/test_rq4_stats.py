"""Stage 7B tests: HL vs brute force, median vs HL, Wilcoxon zeros/ties/all-zero, Holm known example,
shift direction and paired resampling, gate with fixed family, population separation, joins."""
import itertools
import os
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import numpy as np  # noqa: E402
from rv1 import rq4_stats as R4  # noqa: E402


class TestEstimators(unittest.TestCase):
    def test_hl_bruteforce(self):
        rng = np.random.default_rng(0)
        for n in (1, 2, 5, 8):
            x = rng.normal(size=n).round(2)
            walsh = [(x[i] + x[j]) / 2 for i, j in itertools.combinations_with_replacement(range(n), 2)]
            self.assertAlmostEqual(R4.hodges_lehmann(x), float(np.median(walsh)), places=12)

    def test_median_differs_from_hl(self):
        x = [0.0, 0.0, 1.0, 10.0]
        self.assertEqual(R4.sample_median(x), 0.5)
        # Walsh averages (i<=j): 0,0,0,0.5,0.5,1,5,5,5.5,10 -> median (0.5+1)/2 = 0.75; sample median 0.5
        self.assertEqual(R4.hodges_lehmann(x), 0.75)
        self.assertNotEqual(R4.hodges_lehmann(x), R4.sample_median(x))


class TestWilcoxon(unittest.TestCase):
    def test_zeros_ties_allzero(self):
        r = R4.wilcoxon_test([0, 0, 1, 1, -1, 2, 3, 3])
        self.assertEqual((r["n_zero"], r["n_nonzero"]), (2, 6))
        self.assertGreater(r["n_tied_abs_values"], 0)
        self.assertTrue(r["testable"])
        self.assertIn("not exact", r["method_label"])
        z = R4.wilcoxon_test([0.0, 0.0, 0.0])
        self.assertFalse(z["testable"])
        self.assertIsNone(z["p_value"])

    def test_symmetric_sign_flip(self):
        x = np.array([0.3, -0.1, 0.5, 0.2, 0.7, -0.4, 0.6, 0.15, 0.25, 0.9, 0.05, 0.35])
        self.assertAlmostEqual(R4.wilcoxon_test(x)["p_value"], R4.wilcoxon_test(-x)["p_value"], places=12)


class TestHolm(unittest.TestCase):
    def test_known_example(self):
        adj = R4.holm([0.01, 0.04, 0.03, 0.005])
        for a, b in zip(adj, [0.03, 0.06, 0.06, 0.02]):
            self.assertAlmostEqual(a, b, places=12)

    def test_untestable_keeps_family(self):
        adj = R4.holm([0.001] + [None] * 11)
        self.assertEqual(len(adj), 12)
        self.assertAlmostEqual(adj[0], 0.012)                           # m stays 12
        self.assertEqual(adj[1], 1.0)


class TestShiftAndBootstrap(unittest.TestCase):
    def test_shift_direction_and_pairing(self):
        derm = np.array([0.2, 0.1, 0.4, 0.3])
        clin = np.array([0.1, 0.0, 0.1, 0.5])
        D = clin - derm                                                  # D = clinical - dermoscopy
        self.assertTrue(np.allclose(D, [-0.1, -0.1, -0.3, 0.2]))
        b, meta = R4.bootstrap_stats({"derm": derm, "clin": clin, "D": D}, 300, 4)
        rd, rc, rD = (b[k]["median"]["replicates"] for k in ("derm", "clin", "D"))
        self.assertEqual(len(rd), 300)
        # same pair draws for every statistic: replicate D medians come from the same indices
        rng = np.random.Generator(np.random.PCG64(4))
        idx = rng.integers(0, 4, size=4)
        self.assertAlmostEqual(rD[0], float(np.median(D[idx])))
        self.assertAlmostEqual(rd[0], float(np.median(derm[idx])))
        self.assertEqual(meta["n_undefined"], 0)
        b2, _ = R4.bootstrap_stats({"derm": derm, "clin": clin, "D": D}, 300, 4)
        self.assertTrue(np.array_equal(b2["D"]["hl"]["replicates"], b["D"]["hl"]["replicates"]))
        with self.assertRaises(ValueError):
            R4.bootstrap_stats({"a": [1, 2], "b": [1]}, 10, 1)


class TestGate(unittest.TestCase):
    def test_gate(self):
        self.assertTrue(R4.gate(0.01, 0.02))
        self.assertFalse(R4.gate(0.01, -0.02))      # Holm-significant but wrong direction
        self.assertFalse(R4.gate(0.2, 0.02))
        self.assertFalse(R4.gate(None, 0.02))
        # gating never removes tests from the family
        adj = R4.holm([0.2, 0.001, 0.5, 0.04] + [0.3] * 8)
        self.assertEqual(len(adj), 12)


class TestPopulations(unittest.TestCase):
    def test_eligibility_and_join(self):
        from rv1 import rq4_analysis as RA
        rows = []
        for pid, ok_d, ok_c in (("a", True, True), ("b", True, False), ("c", False, False)):
            for mod, ok in (("dermoscopy", ok_d), ("clinical", ok_c)):
                rows.append({"ref_key": f"{pid}|{mod}", "pair_id": pid, "modality": mod, "label_mel": "1",
                             "status": "primary" if ok else "ineligible",
                             "delta_ins": "0.1" if ok else "", "delta_del": "0.2" if ok else "",
                             "as_ins": "0.05", "as_del": "0.1"})
        prim = RA.matched_pairs(rows, primary=True)
        sens = RA.matched_pairs(rows, primary=False)
        self.assertEqual(sorted(prim), ["a"])                            # both modalities required
        self.assertEqual(sorted(sens), ["a", "b", "c"])                  # as-computed separate population
        self.assertEqual(prim["a"]["dermoscopy"]["ins"], 0.1)
        self.assertEqual(sens["c"]["dermoscopy"]["ins"], 0.05)
        with self.assertRaises(ValueError):
            RA.matched_pairs(rows + rows[:1], primary=True)              # duplicate key
        with self.assertRaises(ValueError):
            RA.matched_pairs(rows[:-1], primary=False)                   # incomplete pair


if __name__ == "__main__":
    unittest.main(verbosity=2)
