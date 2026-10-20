"""Stage 9G1 tests: heavy registry, strategy/checkpoint binding, arch+seed matching, shared resampling and
difference direction, difference-of-gaps on a known example, join completeness."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))
import numpy as np  # noqa: E402
from rv1 import rq1_stats as RS, rq3_eval as E, rq3_heavy_eval as H  # noqa: E402


class TestRegistryAndBinding(unittest.TestCase):
    def test_nine_heavy_and_match(self):
        hv = H.heavy_models()
        self.assertEqual(len(hv), 9)
        std = E.models()
        m = H.match(hv, std)
        for h in hv:
            self.assertEqual((m[h["id"]]["arch"], m[h["id"]]["seed"]), (h["arch"], h["seed"]))
            self.assertTrue(h["run"].endswith("_9f") and h["strategy"] == "aug_heavy_v1")
        with self.assertRaises(RuntimeError):
            H.match(hv, [s for s in std if s["id"] != "densenet121_s1"])
        bad = [dict(s, seed=(s["seed"] + 1) % 3) if s["arch"] == "resnet50" else s for s in std]
        with self.assertRaises(RuntimeError):
            H.match(hv, bad)

    def test_strategy_and_threshold_binding(self):
        ck = {"config": {"versions": {"aug_version": "aug_heavy_v1"}}}
        self.assertTrue(H.check_strategy(ck, "aug_heavy_v1"))
        with self.assertRaises(RuntimeError):
            H.check_strategy({"config": {"versions": {"aug_version": "aug_std_v1"}}}, "aug_heavy_v1")
        with self.assertRaises(ValueError):
            E.threshold_value({"model_sha256": "STD", "status": "ESTIMATED", "threshold": 0.2}, "HEAVY")


class TestContrasts(unittest.TestCase):
    def test_binary_direction(self):
        cases = [{"y": True, "ps": .9, "ph": .4}, {"y": True, "ps": .6, "ph": .7}, {"y": False, "ps": .2, "ph": .1},
                 {"y": False, "ps": .7, "ph": .3}]
        v = H.binary_contrast(cases, 0.5, 0.5)
        self.assertEqual(v[:3], (1.0, 0.5, -0.5))                  # heavy minus standard
        self.assertEqual(v[3:6], (0.5, 1.0, 0.5))
        self.assertAlmostEqual(v[8], v[7] - v[6])

    def test_difference_of_gaps_known(self):
        P = lambda pd, pc, qd, qc: {"pred_d": pd, "pred_c": pc, "prob_d": qd, "prob_c": qc}
        pairs = [{"label": 1, "s": P(1, 0, .9, .2), "h": P(1, 1, .8, .7)},
                 {"label": 1, "s": P(1, 0, .8, .3), "h": P(0, 0, .3, .2)},
                 {"label": 0, "s": P(0, 0, .1, .1), "h": P(0, 1, .2, .9)},
                 {"label": 0, "s": P(0, 0, .2, .25), "h": P(0, 0, .1, .1)}]
        v = dict(zip(H.GAP_NAMES, H.gap_contrast(pairs)))
        # standard: sens_d 1.0, sens_c 0.0 -> gap 1.0; heavy: sens_d 0.5, sens_c 0.5 -> gap 0.0; d_gap = -1.0
        self.assertEqual((v["sens_derm_std"], v["sens_clin_std"], v["gap_sens_std"]), (1.0, 0.0, 1.0))
        self.assertEqual((v["sens_derm_heavy"], v["sens_clin_heavy"], v["gap_sens_heavy"]), (0.5, 0.5, 0.0))
        self.assertEqual(v["d_gap_sens"], -1.0)
        self.assertEqual((v["d_sens_derm"], v["d_sens_clin"]), (-0.5, 0.5))   # smaller gap from worse dermoscopy is visible
        self.assertEqual(v["spec_clin_heavy"], 0.5)
        self.assertAlmostEqual(v["d_gap_auroc"], v["gap_auroc_heavy"] - v["gap_auroc_std"])
        self.assertIsNone(H.gap_contrast([p for p in pairs if p["label"] == 1]))

    def test_shared_resampling(self):
        """Within a replicate both models see the same clusters: if heavy == standard, every replicate diff is 0."""
        rng = np.random.default_rng(0)
        cl = {f"L{i}": [{"y": bool(i % 3 == 0), "ps": float(p), "ph": float(p)} for p in rng.random(2)] for i in range(40)}
        b = RS.pair_bootstrap(cl, sorted(cl), lambda cs: H.binary_contrast([x for c in cs for x in c], .5, .5), 200, 7)
        self.assertTrue(np.all(b["replicates"][:, [2, 5, 8]] == 0))
        self.assertEqual(b["n_undefined"], 0)

    def test_join_complete(self):
        s = [{"record_uid": "a", "label_mel": "1", "cluster_id": "c1", "prob": ".5"},
             {"record_uid": "b", "label_mel": "0", "cluster_id": "c1", "prob": ".2"}]
        h = [dict(r, prob=".3") for r in s]
        self.assertEqual(len(H.join_binary(s, h, "cluster_id")["c1"]), 2)
        with self.assertRaises(RuntimeError):
            H.join_binary(s, h[:1], "cluster_id")
        with self.assertRaises(RuntimeError):
            H.join_binary(s, [h[0], dict(h[1], label_mel="1")], "cluster_id")


if __name__ == "__main__":
    unittest.main(verbosity=2)
