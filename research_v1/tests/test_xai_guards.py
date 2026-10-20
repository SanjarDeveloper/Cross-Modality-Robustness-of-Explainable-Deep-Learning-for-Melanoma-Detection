"""Stage 2D2.1 guard tests: input validation, eligibility boundary, endpoint status,
precision-policy scoping. CPU, synthetic models."""
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

if HAVE_TORCH:
    class Lin(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.linspace(-1, 1, 3 * 6 * 7).view(3, 6, 7))

        def forward(self, x):
            return (x * self.w).sum(dim=(1, 2, 3)).view(-1, 1)

    class TwoOut(Lin):
        def forward(self, x):
            return torch.cat([super().forward(x)] * 2, 1)

    class BatchShift(Lin):
        """Output depends on batch size -> endpoint predictions disagree with single passes."""
        def forward(self, x):
            return super().forward(x) + 0.01 * (x.shape[0] > 1)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestGuards(unittest.TestCase):
    def setUp(self):
        from rv1 import xai_attr as XA, xai_curves as XC
        self.XA, self.XC = XA, XC
        g = torch.Generator().manual_seed(1)
        self.x = torch.randn(3, 6, 7, generator=g)
        self.b = torch.zeros(3, 6, 7)
        self.m = torch.rand(6, 7, generator=g)
        self.model = Lin().eval()

    def test_nonfinite_rejected(self):
        for name in ("x", "b", "m"):
            t = {"x": self.x.clone(), "b": self.b.clone(), "m": self.m.clone()}
            t[name].view(-1)[3] = float("nan")
            with self.subTest(name), self.assertRaisesRegex(ValueError, "non-finite"):
                self.XC.curves(self.model, t["x"], t["b"], t["m"])
        m = self.m.clone()
        m[0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.XC.curves(self.model, self.x, self.b, m)

    def test_invalid_arguments_and_output_shape(self):
        for kw in ({"s": 0}, {"s": 2.0}, {"eval_batch": 0}, {"eval_batch": True}, {"s": -3}):
            with self.subTest(kw), self.assertRaises(ValueError):
                self.XC.curves(self.model, self.x, self.b, self.m, **kw)
        with self.assertRaisesRegex(ValueError, "exactly one logit"):
            self.XC.curves(TwoOut().eval(), self.x, self.b, self.m)
        with self.assertRaises(ValueError):
            self.XA.integrated_gradients(self.model, self.x[None], self.b[None], 0)
        with self.assertRaises(ValueError):
            self.XA.integrated_gradients(self.model, self.x[None], self.b[None], 8, internal_batch=0)

    def test_constant_map_needs_override_and_is_non_primary(self):
        const = torch.zeros(6, 7)
        with self.assertRaises(self.XC.IneligibleMap):
            self.XC.curves(self.model, self.x, self.b, const)
        elig = self.XA.eligibility(self.XA.validity(const))
        with self.assertRaises(self.XC.IneligibleMap):
            self.XC.analysis_curves(self.model, self.x, self.b, const, elig, "gradcam")
        r = self.XC.analysis_curves(self.model, self.x, self.b, const, elig, "gradcam", diagnostic_override=True)
        self.assertEqual(r["analysis_status"], "non_primary_diagnostic")
        self.assertEqual(r["ineligibility_reasons"], ["map:all_zero", "map:constant"])

    def test_failed_completeness_ig_blocked(self):
        v = self.XA.validity(self.m)
        failed = self.XA.eligibility(v, {"failure": True})
        with self.assertRaisesRegex(self.XC.IneligibleMap, "ig_completeness_failure"):
            self.XC.analysis_curves(self.model, self.x, self.b, self.m, failed, "ig")
        with self.assertRaises(self.XC.IneligibleMap):   # override does not rescue completeness failure
            self.XC.analysis_curves(self.model, self.x, self.b, self.m, failed, "ig", diagnostic_override=True)
        with self.assertRaisesRegex(self.XC.IneligibleMap, "completeness result"):
            self.XC.analysis_curves(self.model, self.x, self.b, self.m, self.XA.eligibility(v), "ig")
        with self.assertRaisesRegex(self.XC.IneligibleMap, "explicit eligibility"):
            self.XC.analysis_curves(self.model, self.x, self.b, self.m, None, "gradcam")
        ok = self.XC.analysis_curves(self.model, self.x, self.b, self.m,
                                     self.XA.eligibility(v, {"failure": False}), "ig")
        self.assertEqual(ok["analysis_status"], "primary")

    def test_endpoint_status(self):
        r = self.XC.curves(self.model, self.x, self.b, self.m)
        self.assertEqual((r["endpoint_status"], r["endpoint_images_exact"]), ("pass", True))
        bad = self.XC.curves(BatchShift().eval(), self.x, self.b, self.m, eval_batch=11)
        self.assertEqual(bad["endpoint_status"], "fail")
        self.assertGreater(bad["endpoint_prediction_max_abs"], self.XC.ENDPOINT_TOL)
        self.assertEqual(bad["insertion"][0], bad["insertion"][0])      # values kept, not replaced
        rr = self.XC.analysis_curves(BatchShift().eval(), self.x, self.b, self.m,
                                     self.XA.eligibility(self.XA.validity(self.m)), "gradcam")
        self.assertEqual(rr["analysis_status"], "endpoint_failure")

    def test_precision_scope_restores_caller_settings(self):
        b = torch.backends
        prev = (b.cuda.matmul.allow_tf32, b.cudnn.allow_tf32, torch.get_float32_matmul_precision(),
                b.cudnn.deterministic, b.cudnn.benchmark)
        try:
            b.cudnn.allow_tf32, b.cudnn.benchmark, b.cudnn.deterministic = True, True, False
            torch.set_float32_matmul_precision("medium")
            seen = {}

            class Probe(Lin):
                def forward(self, x):
                    seen.update(tf32=b.cudnn.allow_tf32, mm=torch.get_float32_matmul_precision(),
                                det=b.cudnn.deterministic, bench=b.cudnn.benchmark)
                    return super().forward(x)
            self.XC.curves(Probe().eval(), self.x, self.b, self.m)
            self.assertEqual(seen, {"tf32": False, "mm": "highest", "det": True, "bench": False})
            self.assertEqual((b.cudnn.allow_tf32, b.cudnn.benchmark, b.cudnn.deterministic,
                              torch.get_float32_matmul_precision()), (True, True, False, "medium"))
            seen.clear()
            self.XA.integrated_gradients(Probe().eval(), self.x[None], self.b[None], 4)
            self.assertEqual(seen["tf32"], False)
            self.assertEqual(torch.get_float32_matmul_precision(), "medium")
        finally:
            (b.cuda.matmul.allow_tf32, b.cudnn.allow_tf32, p, b.cudnn.deterministic, b.cudnn.benchmark) = prev
            torch.set_float32_matmul_precision(p)


if __name__ == "__main__":
    unittest.main(verbosity=2)
