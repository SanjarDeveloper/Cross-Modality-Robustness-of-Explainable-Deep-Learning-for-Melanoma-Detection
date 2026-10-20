"""Stage 8B tests: grid planning/keys, factors start from the original image, threshold binding,
baseline join/mismatch rejection, batch membership, chunk integrity (via xai_prod helpers)."""
import os
import shutil
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import numpy as np  # noqa: E402
from rv1 import rq2_predict as R  # noqa: E402


class TestPlan(unittest.TestCase):
    def test_grid_and_keys(self):
        u = R.plan_units()
        self.assertEqual(len(u), 45)
        self.assertEqual(len({x["unit"] for x in u}), 45)
        keys = {R.composite_key(f"r{i}", x["factor"], x["severity"]) for i in range(1548) for x in u}
        self.assertEqual(len(keys), 45 * 1548)
        self.assertEqual(R.SEVERITIES, (0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0))

    def test_batch_membership(self):
        b = R.batches(1548)
        self.assertEqual(len(b), 97)
        self.assertEqual(b[-1], (1536, 1548))
        self.assertEqual(b, R.batches(1548))


class TestOriginalImage(unittest.TestCase):
    def test_each_factor_from_original(self):
        rng = np.random.default_rng(0)
        imgs = [rng.integers(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(2)]
        copies = [i.copy() for i in imgs]
        a, _ = R.perturbed_inputs(imgs, ["u1", "u2"], "blur", 1.0)
        R.perturbed_inputs(imgs, ["u1", "u2"], "illumination", 1.0)
        b, _ = R.perturbed_inputs(imgs, ["u1", "u2"], "blur", 1.0)
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(imgs, copies)))   # originals untouched
        self.assertTrue(np.array_equal(a, b))                                         # no chaining


class TestBaseline(unittest.TestCase):
    def rows(self):
        old = [{"record_uid": f"r{i}", "label_mel": str(i % 2), "prob": repr(0.1 * i), "pred": str(int(0.1 * i >= 0.25))}
               for i in range(5)]
        new = [{"record_uid": r["record_uid"], "label_mel": r["label_mel"], "prob": r["prob"], "pred": int(r["pred"])} for r in old]
        return old, new

    def test_join_and_rejections(self):
        old, new = self.rows()
        self.assertEqual(R.check_baseline(old, new, 0.25), 0.0)
        with self.assertRaises(ValueError):
            R.check_baseline(old, list(reversed(new)), 0.25)                          # order differs
        bad = [dict(r) for r in new]
        bad[1]["label_mel"] = "0"
        with self.assertRaises(ValueError):
            R.check_baseline(old, bad, 0.25)
        bad = [dict(r) for r in new]
        bad[3]["prob"] = repr(0.3 + 1e-3)
        with self.assertRaises(ValueError):
            R.check_baseline(old, bad, 0.25)
        with self.assertRaises(ValueError):
            R.check_baseline(old, new, 0.15)                                         # wrong threshold -> pred mismatch


class TestThresholdBinding(unittest.TestCase):
    def test_bound(self):
        thr = {"model": {"sha256": R.MODEL_SHA}, "status": "ESTIMATED", "threshold": 0.1757}
        self.assertEqual(R.bind_threshold(thr), 0.1757)
        with self.assertRaises(ValueError):
            R.bind_threshold(dict(thr, model={"sha256": "a18bf969"}))
        with self.assertRaises(ValueError):
            R.bind_threshold(dict(thr, status="NOT_ESTIMABLE"))


class TestChunks(unittest.TestCase):
    def test_partial_not_complete(self):
        from rv1 import xai_prod as XP
        d = tempfile.mkdtemp()
        try:
            t = os.path.join(d, "u_blur__0.500.tmp-ab")
            os.makedirs(t)
            open(os.path.join(t, "predictions.csv"), "w").close()
            self.assertFalse(XP.verify_dir(os.path.join(d, "u_blur__0.500"), "S"))
            XP.quarantine_tmp(d)
            self.assertFalse(os.path.exists(t))
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
