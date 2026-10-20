"""Stage 10A tests: sum|attr| vs |sum attr|, smoothing order, Gaussian kernel/impulse/boundary, shapes and no
mutation, completeness failures stay excluded, variant/common populations, contrast direction, shared bootstrap,
reconstruction against a saved Stage 7A chunk, random-control identity fields."""
import glob
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))
import numpy as np  # noqa: E402
from rv1 import rq4_stats as R4, xai_secondary as XS  # noqa: E402

R7A = os.path.join(HERE, "..", "runs", "xai_prod_7a", "20261003T173258Z_3b655f88_548b3a05")


class TestMaps(unittest.TestCase):
    def test_sum_abs_vs_abs_sum(self):
        a = np.zeros((3, 4, 4))
        a[0, 1, 1], a[1, 1, 1], a[2, 1, 1] = 2.0, -2.0, 0.5
        self.assertEqual(XS.abs_map(a)[1, 1], 4.5)                 # sum |attr|
        self.assertEqual(abs(a.sum(0))[1, 1], 0.5)                 # |sum| would differ
        self.assertEqual(XS.pos_map(a)[1, 1], 0.5)
        a[2, 1, 1] = -1.0
        self.assertEqual(XS.pos_map(a)[1, 1], 0.0)                 # ReLU of negative sum

    def test_smoothing_after_relu(self):
        a = np.zeros((3, 21, 21))
        a[0, 10, 10] = -5.0                                         # negative evidence only
        a[0, 10, 12] = 1.0
        b = XS.pos_smooth_map(a)
        wrong = np.maximum(XS.smooth(a.sum(0)), 0)                 # smoothing signed then ReLU
        self.assertTrue(np.all(b >= 0) and b[10, 12] > 0)
        self.assertFalse(np.allclose(b, wrong))
        np.testing.assert_allclose(b, XS.smooth(np.maximum(a.sum(0), 0)))

    def test_kernel_impulse_boundary(self):
        k = XS.gaussian_kernel()
        self.assertEqual(len(k), 17)
        self.assertAlmostEqual(k.sum(), 1.0, places=15)
        m = np.zeros((41, 41))
        m[20, 20] = 1.0
        s = XS.smooth(m)
        self.assertAlmostEqual(s.sum(), 1.0, places=12)
        yy, xx = np.mgrid[0:41, 0:41]
        var = (s * (xx - 20) ** 2).sum()
        self.assertAlmostEqual(var, 4.0, delta=0.01)               # sigma^2 = 4
        from scipy.ndimage import gaussian_filter
        r = np.random.default_rng(0).random((30, 25))
        np.testing.assert_allclose(XS.smooth(r), gaussian_filter(r, 2.0, mode="reflect", truncate=4.0), atol=1e-12)
        c = np.full((10, 10), 3.0)
        np.testing.assert_allclose(XS.smooth(c), c)                # reflection keeps constants at borders
        edge = np.zeros((20, 20))
        edge[0, 0] = 1.0
        self.assertAlmostEqual(XS.smooth(edge).sum(), 1.0, places=12)   # mass preserved at corner

    def test_shape_finite_no_mutation(self):
        a = np.random.default_rng(1).normal(size=(3, 224, 224)).astype(np.float32)
        a0 = a.copy()
        for f in (XS.abs_map, XS.pos_map, XS.pos_smooth_map):
            m = f(a)
            self.assertEqual(m.shape, (224, 224))
            self.assertTrue(np.isfinite(m).all())
        np.testing.assert_array_equal(a, a0)


class TestEligibilityAndPopulations(unittest.TestCase):
    def test_completeness_failure_excluded(self):
        self.assertEqual(XS.eligible_secondary(True, []), (False, ["ig_completeness_failure"]))
        self.assertEqual(XS.eligible_secondary(False, ["constant"]), (False, ["map:constant"]))
        self.assertEqual(XS.eligible_secondary(False, []), (True, []))

    def test_populations_and_direction(self):
        def row(p, m, ok, i, d):
            return {"pair_id": p, "modality": m, "ok": ok, "ins": i, "del": d}
        var = XS.pair_population([row("a", "dermoscopy", True, .3, .1), row("a", "clinical", True, .1, .0),
                                  row("b", "dermoscopy", True, .2, .1), row("b", "clinical", False, None, None)])
        orig = XS.pair_population([row("a", "dermoscopy", True, .2, .1), row("a", "clinical", True, .2, .1),
                                   row("b", "dermoscopy", True, .2, .1), row("b", "clinical", True, .2, .1)])
        self.assertEqual(sorted(var), ["a"])
        inter = sorted(set(var) & set(orig))
        v = XS.contrast_vectors(orig, var, inter)
        self.assertAlmostEqual(v["var_minus_orig|ins|dermoscopy"][0], 0.1)       # variant minus original
        self.assertAlmostEqual(v["var|ins|D"][0], -0.2)                          # clinical minus dermoscopy
        self.assertAlmostEqual(v["var_minus_orig|ins|D"][0], -0.2 - 0.0)
        with self.assertRaises(ValueError):
            XS.pair_population([row("c", "dermoscopy", True, .1, .1)])

    def test_shared_draws(self):
        x = np.arange(10, dtype=float)
        bs, meta = R4.bootstrap_stats({"a": x, "b": x * 2, "d": x * 2 - x}, 50, 7)
        np.testing.assert_allclose(bs["b"]["median"]["replicates"] - bs["a"]["median"]["replicates"], bs["d"]["median"]["replicates"])


@unittest.skipUnless(os.path.isdir(R7A), "Stage 7A run not present")
class TestStage7AReconstruction(unittest.TestCase):
    def test_reconstruct_saved_maps_chunk00(self):
        z = np.load(os.path.join(R7A, "chunk_00", "maps.npz"))
        for i in range(0, 48, 7):
            okp, _ = XS.recon_ok(XS.pos_map(z["ig512_attr_signed"][i]), z["ig512_primary"][i])
            oka, _ = XS.recon_ok(XS.abs_map(z["ig512_attr_signed"][i]), z["ig512_absolute"][i])
            self.assertTrue(okp and oka)
        self.assertFalse(XS.recon_ok(np.abs(z["ig512_attr_signed"][0].sum(0)), z["ig512_absolute"][0])[0])   # abs(sum) != stored

    def test_random_identity_fields(self):
        import json
        with open(os.path.join(R7A, "chunk_00", "records.json"), encoding="utf-8") as f:
            r = json.load(f)[0]
        rnd = r["controls"]["random"]
        self.assertEqual(len(rnd["auc_ins_each"]), 5)
        self.assertTrue(all(c["p_full"] == rnd["curves"][0]["p_full"] for c in rnd["curves"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
