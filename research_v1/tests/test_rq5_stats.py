"""Stage 5B tests: threshold binding, joins, paired cluster bootstrap, swap, Secondary B, checkpoint identity."""
import os
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import numpy as np  # noqa: E402
from rv1 import rq1_stats as RS, rq5_stats as R5  # noqa: E402


def rows(sha, probs, labels, pred_t=0.5):
    return [{"record_uid": f"r{i}", "label_mel": str(y), "prob": repr(p), "pred": str(int(p >= pred_t)),
             "checkpoint_sha256": sha} for i, (p, y) in enumerate(zip(probs, labels))]


class TestThresholds(unittest.TestCase):
    def test_binding(self):
        tb = R5.ModelThreshold("BASE", 0.2, "tb", "base")
        ta = R5.ModelThreshold("ADAPT", 0.4, "ta", "adapted")
        rb = rows("BASE", [0.3], [1])[0]
        self.assertEqual(tb.predict_row(rb), 1)
        with self.assertRaisesRegex(ValueError, "another model"):
            ta.predict_row(rb)

    def test_secondary_b_not_estimable_at_9(self):
        r = R5.secondary_b([0.9] * 9 + [0.1] * 300, [1] * 9 + [0] * 300)
        self.assertEqual(r["status"], "NOT_ESTIMABLE")
        self.assertIsNone(r["threshold"])
        self.assertEqual(r["min_required"], 20)


class TestJoin(unittest.TestCase):
    def test_rejections(self):
        b = rows("B", [0.1, 0.9], [0, 1])
        a = rows("A", [0.2, 0.8], [0, 1])
        self.assertEqual(len(R5.join_models(b, a, "B", "A")), 2)
        with self.assertRaisesRegex(ValueError, "differ"):
            R5.join_models(b, a[:1], "B", "A")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            R5.join_models(b + b[:1], a, "B", "A")
        bad = [dict(a[0], label_mel="1"), a[1]]
        with self.assertRaisesRegex(ValueError, "label mismatch"):
            R5.join_models(b, bad, "B", "A")
        with self.assertRaisesRegex(ValueError, "wrong model"):
            R5.join_models(b, a, "B", "B")
        with self.assertRaisesRegex(ValueError, "invalid prob"):
            R5.join_models(b, [dict(a[0], prob="1.5"), a[1]], "B", "A")


class TestPairedBootstrap(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.cl = {}
        for i in range(30):
            k = 1 + i % 3
            self.cl[f"P{i}"] = [{"label": int(i < 6 and j == 0), "p_base": float(rng.random()),
                                 "p_adapt": float(rng.random()), "pid": f"P{i}"} for j in range(k)]
        self.ids = sorted(self.cl)

    def test_same_draws_and_multiplicity(self):
        rng = np.random.Generator(np.random.PCG64(3))
        draw = RS.resample_ids(self.ids, rng)
        flat = [r for i in draw for r in self.cl[i]]
        for pid in set(draw):                                     # all images, each repeat kept, both models
            self.assertEqual(sum(r["pid"] == pid for r in flat), draw.count(pid) * len(self.cl[pid]))
        st = R5.auc_pair(flat)
        self.assertAlmostEqual(st[2], st[1] - st[0])              # both models from the same draw
        a = RS.pair_bootstrap(self.cl, self.ids, R5.cluster_stat(R5.auc_pair), 200, 5)
        b = RS.pair_bootstrap(self.cl, self.ids, R5.cluster_stat(R5.sens_pair, 0.5, 0.5), 200, 5)
        self.assertTrue(np.allclose(a["replicates"][:, 2], a["replicates"][:, 1] - a["replicates"][:, 0]))
        self.assertEqual(a["n_defined"] + a["n_undefined"], 200)
        self.assertEqual(a["n_undefined"], b["n_undefined"])     # same seed -> same draws -> same empty replicates

    def test_swap_reverses(self):
        flat = [r for v in self.cl.values() for r in v]
        sw = [dict(r, p_base=r["p_adapt"], p_adapt=r["p_base"]) for r in flat]
        self.assertAlmostEqual(R5.auc_pair(flat)[2], -R5.auc_pair(sw)[2])
        self.assertAlmostEqual(R5.sens_pair(flat, 0.3, 0.6)[2], -R5.sens_pair(sw, 0.6, 0.3)[2])


class TestSelectedIdentity(unittest.TestCase):
    def test_wrong_sha_rejected(self):
        from rv1 import rq5_eval as RE
        with self.assertRaises(RuntimeError):
            RE.check_selected({"checkpoint": "ckpt_epoch5.pt", "epoch": 5, "sha256": "x"}, "x")
        with self.assertRaises(RuntimeError):
            RE.check_selected({"checkpoint": "ckpt_epoch4.pt", "epoch": 4, "sha256": "y"}, RE.ADAPT_SHA)
        RE.check_selected({"checkpoint": "ckpt_epoch4.pt", "epoch": 4, "sha256": RE.ADAPT_SHA}, RE.ADAPT_SHA)


if __name__ == "__main__":
    unittest.main(verbosity=2)
