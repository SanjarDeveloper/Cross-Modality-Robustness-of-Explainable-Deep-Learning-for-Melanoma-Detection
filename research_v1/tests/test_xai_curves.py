"""Stage 2D2 analytical tests: insertion/deletion curves v1, autocast policy, eligibility.

CPU, synthetic models. Tolerances fixed before running: 1e-6 for probabilities/AUCs of
analytic models; exact equality where the construction is exact.
No random-order expectation identity is used (E[g(X)] != g(E[X]) for nonlinear g).
"""
import hashlib
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
import numpy as np

TOL = 1e-6

if HAVE_TORCH:
    class Additive(nn.Module):
        """logit = sum V * x + c0 (exactly additive over pixels and channels)."""
        def __init__(self, V, c0=-0.2):
            super().__init__()
            self.V = nn.Parameter(V.clone())
            self.c0 = c0

        def forward(self, x):
            return ((x * self.V).sum(dim=(1, 2, 3)) + self.c0).view(-1, 1)

    class Spy(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner, self.seen = inner, []

        def forward(self, x):
            self.seen += [t.clone() for t in x]
            return self.inner(x)


def state_hash(m):
    h = hashlib.sha256()
    for k, v in m.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().numpy().tobytes())
    return h.hexdigest()


def analytic(V, x, b, smap, S=32):
    """Independent numpy computation of both curves for the additive model."""
    C, H, W = x.shape
    n = H * W
    order = np.lexsort((np.arange(n), -smap.reshape(-1)))       # desc value, then index
    e = [int(np.floor(j * n / S + 0.5)) for j in range(S + 1)]
    cx = (V * x).sum(0).reshape(-1)                              # per-pixel contribution from x
    cb = (V * b).sum(0).reshape(-1)
    ins, dele = [], []
    for k in e:
        top = np.zeros(n, bool)
        top[order[:k]] = True
        ins.append(1 / (1 + np.exp(-(np.where(top, cx, cb).sum() - 0.2))))
        dele.append(1 / (1 + np.exp(-(np.where(top, cb, cx).sum() - 0.2))))
    f = np.array(e) / n
    return np.array(ins), np.array(dele), f


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestCurves(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(3)
        self.V = torch.randn(3, 7, 9, generator=g) * 0.3
        self.x = torch.randn(3, 7, 9, generator=g)
        self.b = torch.randn(3, 7, 9, generator=g) * 0.1
        self.model = Additive(self.V).eval()

    def test_exact_curves_and_auc(self):
        from rv1 import xai_curves as XC
        smap = torch.randn(7, 9, generator=torch.Generator().manual_seed(4))
        r = XC.curves(self.model, self.x, self.b, smap, eval_batch=5)
        ins, dele, f = analytic(self.V.numpy(), self.x.numpy(), self.b.numpy(), smap.numpy())
        self.assertLess(np.abs(np.array(r["insertion"]) - ins).max(), TOL)
        self.assertLess(np.abs(np.array(r["deletion"]) - dele).max(), TOL)
        self.assertTrue(np.array_equal(np.array(r["fractions"]), f))
        tz = lambda y: float(((y[1:] + y[:-1]) / 2 * np.diff(f)).sum())   # hand trapezoid
        self.assertLess(abs(r["auc_insertion_v1"] - tz(ins)), TOL)
        self.assertLess(abs(r["auc_deletion_v1"] - tz(dele)), TOL)
        self.assertEqual(len(r["insertion"]), 33)
        for v in r["endpoint_prediction_diffs"].values():
            self.assertLess(abs(v), TOL)

    def test_directions_with_faithful_map(self):
        from rv1 import xai_curves as XC
        V = torch.rand(3, 7, 9) + 0.1
        x, b = torch.rand(3, 7, 9) + 1.0, torch.zeros(3, 7, 9)   # every pixel contributes positively
        m = Additive(V).eval()
        faithful = (V * (x - b)).sum(0)
        r = XC.curves(m, x, b, faithful)
        self.assertTrue(all(q >= p - 1e-12 for p, q in zip(r["insertion"], r["insertion"][1:])))
        self.assertTrue(all(q <= p + 1e-12 for p, q in zip(r["deletion"], r["deletion"][1:])))
        rev = XC.curves(m, x, b, -faithful)                        # anti-faithful ordering
        self.assertGreater(r["auc_insertion_v1"], rev["auc_insertion_v1"])
        self.assertLess(r["auc_deletion_v1"], rev["auc_deletion_v1"])

    def test_boundaries_and_coverage(self):
        from rv1 import xai_curves as XC
        for n in (224 * 224, 300 * 300, 49, 63, 31):
            e = XC.boundaries(n, 32)
            self.assertEqual((e[0], e[-1], len(e)), (0, n, 33))
            self.assertEqual(e, [int(np.floor(j * n / 32 + 0.5)) for j in range(33)])
            self.assertLessEqual(max(b - a for a, b in zip(e, e[1:])) - min(b - a for a, b in zip(e, e[1:])), 1)
        smap = torch.rand(7, 9)                                    # N=63, not divisible by 32
        XC.curves(self.model, self.x, self.b, smap, check=True)   # internal coverage asserts

    def test_tie_handling_deterministic(self):
        from rv1 import xai_curves as XC
        order, pos = XC.rank_positions(torch.zeros(4, 5))
        self.assertTrue(torch.equal(order, torch.arange(20)))      # constant map -> index order
        m = torch.tensor([[1.0, 2.0, 2.0], [0.0, 2.0, 1.0]])
        order, _ = XC.rank_positions(m)
        self.assertEqual(order.tolist(), [1, 2, 4, 0, 5, 3])

    def test_channels_together_and_endpoints(self):
        from rv1 import xai_curves as XC
        spy = Spy(self.model).eval()
        XC.curves(spy, self.x, self.b, torch.rand(7, 9), eval_batch=33, check=False)
        imgs = spy.seen
        self.assertEqual(len(imgs), 68)            # 66 curve images + 2 direct endpoint predictions
        self.assertTrue(torch.equal(imgs[66], self.x) and torch.equal(imgs[67], self.b))
        imgs = imgs[:66]
        for im in imgs:
            from_x = (im == self.x).all(0)
            from_b = (im == self.b).all(0)
            self.assertTrue(bool((from_x | from_b).all()))          # no mixed-channel pixel
        self.assertTrue(torch.equal(imgs[0], self.b) and torch.equal(imgs[32], self.x))    # insertion
        self.assertTrue(torch.equal(imgs[33], self.x) and torch.equal(imgs[65], self.b))   # deletion

    def test_immutability_and_chunk_equivalence(self):
        from rv1 import xai_curves as XC
        net = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.Tanh(),
                            nn.Flatten(), nn.Linear(4 * 7 * 9, 1)).eval()
        smap = torch.rand(7, 9)
        x0, b0, s0, h0 = self.x.clone(), self.b.clone(), smap.clone(), state_hash(net)
        rs = [XC.curves(net, self.x, self.b, smap, eval_batch=eb) for eb in (1, 5, 33)]
        for r in rs[1:]:
            self.assertLess(max(abs(a - b) for a, b in zip(r["insertion"] + r["deletion"],
                                                           rs[0]["insertion"] + rs[0]["deletion"])), TOL)
        self.assertTrue(torch.equal(self.x, x0) and torch.equal(self.b, b0) and torch.equal(smap, s0))
        self.assertEqual(state_hash(net), h0)
        with self.assertRaises(RuntimeError):
            XC.curves(net.train(), self.x, self.b, smap)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestAutocastAndEligibility(unittest.TestCase):
    def test_enclosing_autocast_does_not_change_attribution(self):
        from rv1 import xai_attr as XA, xai_curves as XC
        net = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Flatten(),
                            nn.Linear(4 * 8 * 8, 1)).eval()
        x = torch.rand(2, 3, 8, 8)
        b = XA.gaussian_baseline(x)
        ref_cam = XA.grad_cam(net, net[0], x)["map"]
        ref_ig = XA.integrated_gradients(net, x, b, 16)["attr"]
        ref_cv = XC.curves(net, x[0], b[0], ref_cam[0])["insertion"]
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with torch.no_grad():
                self.assertEqual(net(x).dtype, torch.bfloat16)   # the context really is active
            cam = XA.grad_cam(net, net[0], x)["map"]
            ig = XA.integrated_gradients(net, x, b, 16)["attr"]
            cv = XC.curves(net, x[0], b[0], ref_cam[0])["insertion"]
        self.assertEqual((cam.dtype, ig.dtype), (torch.float32, torch.float32))
        self.assertTrue(torch.equal(cam, ref_cam) and torch.equal(ig, ref_ig))
        self.assertEqual(cv, ref_cv)

    def test_eligibility_split(self):
        from rv1 import xai_attr as XA
        ok = {"valid": True, "flags": []}
        bad = {"valid": False, "flags": ["constant"]}
        e = XA.eligibility(ok, {"failure": True})
        self.assertEqual((e["map_shape_valid"], e["ig_completeness_pass"], e["eligible_for_primary_analysis"]),
                         (True, False, False))
        self.assertEqual(e["ineligibility_reasons"], ["ig_completeness_failure"])
        e = XA.eligibility(bad, {"failure": True})
        self.assertEqual(e["ineligibility_reasons"], ["map:constant", "ig_completeness_failure"])
        self.assertTrue(XA.eligibility(ok)["eligible_for_primary_analysis"])      # Grad-CAM path
        self.assertIsNone(XA.eligibility(ok)["ig_completeness_pass"])

    def test_precision_context_restores(self):
        from rv1 import xai_attr as XA
        before = XA.precision_settings()
        with XA.highest_fp32():
            s = XA.precision_settings()
            self.assertFalse(s["cudnn_allow_tf32"] or s["cuda_matmul_allow_tf32"])
            self.assertEqual(s["float32_matmul_precision"], "highest")
        self.assertEqual(XA.precision_settings(), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
