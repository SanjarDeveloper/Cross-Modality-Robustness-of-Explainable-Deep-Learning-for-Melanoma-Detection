"""Stage 9C2 tests: bindings, own-baseline references, identical perturbations across models, coverage,
shared lesion draws, resume hash rejection."""
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))
import numpy as np  # noqa: E402
from rv1 import rq3_eval as E9, rq3_rq2 as R  # noqa: E402


class TestBindings(unittest.TestCase):
    def test_threshold_binding(self):
        reg = {"thresholds": {"m": {"model_sha256": "A", "threshold": 0.2}}}
        self.assertEqual(R.bind({"id": "m", "sha256": "A"}, reg), 0.2)
        with self.assertRaises(ValueError):
            R.bind({"id": "m", "sha256": "B"}, reg)

    def test_own_reference(self):
        ms = {m["id"]: m for m in E9.models()}
        ms = {k: dict(v, reuse={c: {"path": p} for c, p in v["reuse"].items()}) for k, v in ms.items()}
        self.assertIn("predictions_ham_test.csv", R.reference_path(ms["resnet50_s0"]))
        self.assertTrue(R.reference_path(ms["resnet50_s1"]).endswith("pred_resnet50_s1_ham_test.csv"))
        self.assertTrue(R.reference_path(ms["densenet121_s2"]).endswith("pred_densenet121_s2_ham_test.csv"))

    def test_coverage(self):
        self.assertEqual(len(E9.models()), 9)
        self.assertEqual(len(R.plan_units()) * 1548 * 9, 626940)


class TestPerturbIdentity(unittest.TestCase):
    def test_model_independent(self):
        from rv1 import perturb as PT
        img = np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        a = R.digest(PT.apply(img, "scale", 0.5, "ham10000:X")[0])
        b = R.digest(PT.apply(img, "scale", 0.5, "ham10000:X")[0])
        self.assertEqual(a, b)                                   # nothing model-specific enters the transform


class TestSharedDraws(unittest.TestCase):
    def test_same_seed_same_draws(self):
        from rv1 import rq2_stats as R2
        ids, groups = R2.lesion_index(["a", "a", "b", "c"])
        r1 = np.random.Generator(np.random.PCG64(20261050))
        r2 = np.random.Generator(np.random.PCG64(20261050))
        self.assertTrue(np.array_equal(R2.draw_rows(groups, r1), R2.draw_rows(groups, r2)))


class TestResume(unittest.TestCase):
    def test_incompatible_hash_rejected(self):
        from rv1 import xai_prod as XP
        d = tempfile.mkdtemp()
        try:
            t = os.path.join(d, "u_m__blur__0.500.tmp-a")
            os.makedirs(t)
            with open(os.path.join(t, "predictions.csv"), "w") as f:
                f.write("x")
            XP.publish_dir(t, os.path.join(d, "u_m__blur__0.500"), {"spec_sha256": "S1"})
            with self.assertRaises(RuntimeError):
                XP.verify_dir(os.path.join(d, "u_m__blur__0.500"), "S2")
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
