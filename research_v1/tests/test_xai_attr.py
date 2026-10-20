"""Stage 2D1 analytical tests for rv1.xai_attr. CPU, synthetic models only.

Tolerances fixed before running: float32 analytic identities 1e-5 absolute (1e-4 for sums
over >=10^4 terms); baseline normalisation 1e-6.
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

TOL, TOL_SUM = 1e-5, 1e-4


def state_hash(m):
    h = hashlib.sha256()
    for k, v in m.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


if HAVE_TORCH:
    class CamNet(nn.Module):
        """logit = w * mean_{h,w} A,  A = conv(x) (1 channel, fixed weights) + BN(eval)."""
        def __init__(self, w=2.0):
            super().__init__()
            self.conv = nn.Conv2d(3, 1, 1, bias=False)
            with torch.no_grad():
                self.conv.weight.fill_(1.0)          # A = R + G + B (1x1 conv)
            self.bn = nn.BatchNorm2d(1)             # identity in eval with default stats
            self.w = w

        def forward(self, x):
            a = self.bn(self.conv(x))
            return (self.w * a.mean(dim=(2, 3))).view(-1, 1)

    class LinNet(nn.Module):
        def __init__(self, W, b=0.3):
            super().__init__()
            self.W = nn.Parameter(W.clone())
            self.b = b

        def forward(self, x):
            return ((x * self.W).sum(dim=(1, 2, 3)) + self.b).view(-1, 1)

    class CubicNet(nn.Module):
        def forward(self, x):
            return (0.01 * (x ** 3).sum(dim=(1, 2, 3))).view(-1, 1)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestBaseline(unittest.TestCase):
    def test_params_and_properties(self):
        from rv1 import xai_attr as XA
        self.assertEqual(XA.blur_params(224, 224), (57, 9.5))
        c = torch.full((1, 3, 224, 224), 1.7)
        self.assertLess(float((XA.gaussian_baseline(c) - 1.7).abs().max()), 1e-5)   # reflect keeps constants
        imp = torch.zeros(1, 1, 224, 224)
        imp[0, 0, 112, 112] = 1.0
        y = XA.gaussian_baseline(imp)
        self.assertLess(abs(float(y.sum()) - 1.0), 1e-5)                             # mass preserved (interior)
        self.assertEqual((y.shape, y.dtype), (imp.shape, torch.float32))
        # analytic separable Gaussian at offset (dy,dx) = g(dy) g(dx)
        k, s = 57, 9.5
        g = torch.exp(-(torch.arange(k) - 28.0) ** 2 / (2 * s * s))
        g = g / g.sum()
        self.assertLess(abs(float(y[0, 0, 112 + 5, 112 - 3]) - float(g[28 + 5] * g[28 - 3])), 1e-7)
        with self.assertRaises(ValueError):
            XA.gaussian_baseline(torch.zeros(1, 3, 8, 8, dtype=torch.float64))


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestGradCam(unittest.TestCase):
    def test_known_region_and_formula(self):
        from rv1 import xai_attr as XA
        net = CamNet(2.0).eval()
        x = torch.zeros(1, 3, 16, 16)
        x[0, :, 4:8, 10:14] = 1.0                    # bright region
        r = XA.grad_cam(net, net.bn, x)
        A = x.sum(1)                                  # conv output
        expected = torch.relu((2.0 / 256) * A)       # w_c = w/(h*w) for GAP head
        self.assertLess(float((r["cam_raw"][0] - expected).abs().max()), TOL)
        self.assertEqual(r["map"].shape, (1, 16, 16))  # same size: upsampling is identity here
        m = r["map"][0]
        self.assertTrue(bool((m[4:8, 10:14] > 0).all()) and float(m.sum() - m[4:8, 10:14].sum()) == 0.0)
        self.assertAlmostEqual(float(r["logit"][0]), 2.0 * 3 * 16 / 256, places=5)

    def test_negative_evidence_gives_all_zero_flag(self):
        from rv1 import xai_attr as XA
        net = CamNet(-2.0).eval()
        x = torch.rand(2, 3, 16, 16)
        r = XA.grad_cam(net, net.bn, x)
        v = XA.validity(r["map"][0])
        self.assertIn("all_zero", v["flags"])
        self.assertIn("constant", v["flags"])
        self.assertFalse(v["valid"])

    def test_upsampling_mode(self):
        import torch.nn.functional as F
        from rv1 import xai_attr as XA
        net = nn.Sequential(nn.Conv2d(3, 1, 4, stride=4, bias=False), nn.Flatten(), nn.Linear(16, 1)).eval()
        x = torch.rand(1, 3, 16, 16)
        r = XA.grad_cam(net, net[0], x)
        ref = F.interpolate(r["cam_raw"][:, None], size=(16, 16), mode="bilinear", align_corners=False)[:, 0]
        self.assertTrue(torch.equal(r["map"], ref))
        self.assertEqual(r["cam_raw"].shape, (1, 4, 4))

    def test_state_and_hooks_unchanged(self):
        from rv1 import xai_attr as XA
        net = CamNet(2.0).eval()
        h0 = state_hash(net)
        x = torch.rand(3, 3, 16, 16)
        a = XA.grad_cam(net, net.bn, x)
        b = XA.grad_cam(net, net.bn, x)
        self.assertTrue(torch.equal(a["map"], b["map"]))
        self.assertEqual(state_hash(net), h0)
        self.assertEqual(len(net.bn._forward_hooks), 0)
        self.assertTrue(all(p.grad is None for p in net.parameters()))
        single = torch.cat([XA.grad_cam(net, net.bn, x[i:i + 1])["map"] for i in range(3)])
        self.assertLess(float((single - a["map"]).abs().max()), TOL)
        with self.assertRaises(RuntimeError):
            XA.grad_cam(net.train(), net.bn, x)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestIG(unittest.TestCase):
    def test_linear_exact_and_signed_aggregation(self):
        from rv1 import xai_attr as XA
        W = torch.zeros(3, 8, 8)
        W[0] = 1.5                                   # positive R
        W[1] = -2.0                                  # negative G
        W[2, :4] = 0.5                               # partial positive B
        net = LinNet(W).eval()
        x = torch.rand(2, 3, 8, 8)
        b = torch.rand(2, 3, 8, 8) * 0.1
        for m in (1, 16, 128):
            r = XA.integrated_gradients(net, x, b, m, internal_batch=7)
            exp = (x - b) * W
            self.assertLess(float((r["attr"] - exp).abs().max()), TOL)
            self.assertLess(float((r["primary"] - torch.relu(exp.sum(1))).abs().max()), TOL)
            self.assertLess(float((r["absolute"] - exp.abs().sum(1)).abs().max()), TOL)
            for c in r["completeness"]:
                self.assertLess(c["e_abs"], TOL_SUM)
                self.assertFalse(c["failure"])
        self.assertTrue(bool((r["primary"] >= 0).all()))

    def test_cubic_convergence_and_state(self):
        from rv1 import xai_attr as XA
        net = CubicNet().eval()
        x = torch.randn(1, 3, 16, 16, generator=torch.Generator().manual_seed(0)) * 2
        b = torch.zeros_like(x)
        errs = []
        for m in (16, 32, 64, 128):
            r = XA.integrated_gradients(net, x, b, m)
            exact = 0.01 * (x ** 3)                  # IG of sum a x^3 from 0 is exact a x^3
            errs.append(float((r["attr"] - exact).abs().max()))
        self.assertTrue(all(e2 < e1 for e1, e2 in zip(errs, errs[1:])))  # midpoint error falls with m
        self.assertLess(errs[-1], 1e-3)

    def test_batching_equivalence_and_no_state_change(self):
        from rv1 import xai_attr as XA
        net = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.Tanh(),
                            nn.Flatten(), nn.Linear(4 * 8 * 8, 1)).eval()
        h0 = state_hash(net)
        x = torch.rand(2, 3, 8, 8)
        b = XA.gaussian_baseline(x)
        a = XA.integrated_gradients(net, x, b, 32, internal_batch=32)
        c = XA.integrated_gradients(net, x, b, 32, internal_batch=5)
        self.assertLess(float((a["attr"] - c["attr"]).abs().max()), TOL)
        self.assertEqual(state_hash(net), h0)
        self.assertTrue(all(p.grad is None for p in net.parameters()))


class TestCompletenessRule(unittest.TestCase):
    def test_rule(self):
        from rv1 import xai_attr as XA
        ok = XA.completeness(1.02, 2.0, 1.0)             # delta 1.0, e_rel 0.02
        self.assertFalse(ok["failure"])
        self.assertAlmostEqual(ok["e_rel"], 0.02)
        bad = XA.completeness(1.2, 2.0, 1.0)              # e_rel 0.2
        self.assertTrue(bad["failure"])
        tiny = XA.completeness(0.03, 0.05, 0.0)           # |delta| < 0.1 -> relative undefined
        self.assertIsNone(tiny["e_rel"])
        self.assertFalse(tiny["failure"])                 # e_abs 0.02 <= 0.05
        tiny_bad = XA.completeness(0.2, 0.05, 0.0)
        self.assertTrue(tiny_bad["failure"])              # e_abs 0.15 > 0.05, not silently passed
        self.assertTrue(XA.completeness(float("nan"), 1.0, 0.0)["failure"])


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestValidity(unittest.TestCase):
    def test_flags(self):
        from rv1 import xai_attr as XA
        self.assertEqual(XA.validity(torch.tensor([[0.0, 1.0]]))["flags"], [])
        self.assertEqual(XA.validity(torch.zeros(4, 4))["flags"], ["all_zero", "constant"])
        self.assertEqual(XA.validity(torch.full((4, 4), 3.0))["flags"], ["constant"])
        self.assertEqual(XA.validity(torch.tensor([1.0, float("nan")]))["flags"], ["nonfinite"])
        self.assertEqual(XA.validity(torch.tensor([1.0, float("inf")]))["flags"], ["nonfinite"])
        m = torch.tensor([[0.0, 2.0]])
        d = XA.display_normalise(m.numpy())
        self.assertEqual(float(m[0, 1]), 2.0)             # stored map untouched
        self.assertEqual(float(d[0, 1]), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
