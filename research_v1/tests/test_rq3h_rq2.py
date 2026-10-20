"""Stage 9G2 tests: heavy/standard matching and binding, identical perturbations across strategies, 18-cell
coverage arithmetic, per-replicate worst-loss contrast, benefit sign, equal-model zero, censored crossings ordinal."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))
import numpy as np  # noqa: E402
from rv1 import perturb as PT, rq2_stats as R2, rq3_rq2 as R, rq3h_rq2 as H  # noqa: E402
from rv1.rq2_predict import FACTORS, SEVERITIES  # noqa: E402


def m(i, a, s, t=0.3, sha="x"):
    return {"id": i, "arch": a, "seed": s, "threshold": t, "sha256": sha}


class TestBinding(unittest.TestCase):
    def test_match_and_threshold(self):
        hv = [m(f"{a}_heavy_s{s}", a, s) for a in ("resnet50", "densenet121") for s in (0, 1, 2)]
        st = [m(f"{a}_s{s}", a, s) for a in ("resnet50", "densenet121") for s in (0, 1, 2)]
        mp = H.match(hv, st)
        self.assertTrue(all(mp[h["id"]]["arch"] == h["arch"] and mp[h["id"]]["seed"] == h["seed"] for h in hv))
        with self.assertRaises(RuntimeError):
            H.match(hv, st[1:])
        reg = {"thresholds": {"r_heavy": {"model_sha256": "H", "threshold": 0.44}}}
        self.assertEqual(R.bind({"id": "r_heavy", "sha256": "H"}, reg), 0.44)
        with self.assertRaises(ValueError):
            R.bind({"id": "r_heavy", "sha256": "STD"}, reg)

    def test_identical_perturbation_independent_of_model(self):
        img = np.random.default_rng(1).integers(0, 255, (256, 256, 3), dtype=np.uint8)
        for f in FACTORS:
            a = PT.apply(img, f, 0.625, "ham10000:ISIC_X")[0]
            b = PT.apply(img.copy(), f, 0.625, "ham10000:ISIC_X")[0]
            self.assertEqual(R.digest(a), R.digest(b))

    def test_coverage_arithmetic(self):
        self.assertEqual(len(FACTORS) * len(SEVERITIES) * 1548, 69660)
        self.assertEqual(9 * 69660, 626940)
        self.assertEqual(18 * 69660, 1253880)


class TestContrasts(unittest.TestCase):
    def test_worst_per_replicate_and_benefit(self):
        # 2 replicates; standard loses 0.3 in rep0 and 0.1 in rep1; heavy loses 0.1 and 0.2
        std = [np.array([0.9, 0.8]), np.array([0.6, 0.75]), np.array([0.7, 0.7])]
        hv = [np.array([0.8, 0.9]), np.array([0.75, 0.8]), np.array([0.7, 0.7])]
        ws, ms = H.worst_reps(std)
        wh, mh = H.worst_reps(hv)
        np.testing.assert_allclose(ws, [0.3, 0.1])
        np.testing.assert_allclose(wh, [0.1, 0.2])
        np.testing.assert_allclose(H.benefit(ws, wh), [0.2, -0.1])          # positive = smaller heavy loss
        np.testing.assert_allclose(mh - ms, [0.1, 0.0])
        w0, _ = H.worst_reps([np.array([0.5]), np.array([0.6])])            # improvement -> worst loss 0 (s=0 included)
        self.assertEqual(w0[0], 0.0)

    def test_equal_models_zero_and_undefined(self):
        lv = [np.array([0.9, np.nan]), np.array([0.7, 0.6])]
        w, _ = H.worst_reps(lv)
        self.assertTrue(np.isnan(w[1]))
        self.assertEqual(H.benefit(w, w)[0], 0.0)

    def test_censored_crossing_ordinal(self):
        C = R2.CENSORED
        self.assertEqual(H.crossing_order(C, C), "both_censored")
        self.assertEqual(H.crossing_order(0.4, C), "heavy_later_censored")
        self.assertEqual(H.crossing_order(C, 0.9), "heavy_earlier_censored")
        self.assertEqual(H.crossing_order(0.3, 0.5), "heavy_later")
        self.assertEqual(H.crossing_order(0.5, 0.3), "heavy_earlier")
        self.assertEqual(H.crossing_order(None, 0.3), "undefined")
        with self.assertRaises(TypeError):
            _ = C - 0.5                                                       # marker is not numeric


if __name__ == "__main__":
    unittest.main(verbosity=2)
