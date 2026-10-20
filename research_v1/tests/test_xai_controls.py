"""Stage 2D3 tests: control maps (controls_v1), mismatch derangement, score aggregation and
signs, and the new curve/eligibility guards. Synthetic fixtures; CPU.
No expectation identity of the form E[g(X)] = g(E[X]) is used."""
import json
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
sys.dont_write_bytecode = True
sys.path.insert(0, SRC)

import numpy as np  # noqa: E402

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

from rv1 import preprocess as PP, xai_controls as XK  # noqa: E402


def to_norm(rgb):
    return (rgb - np.asarray(PP.MEAN)[:, None, None]) / np.asarray(PP.STD)[:, None, None]


class TestRandomCentreSobel(unittest.TestCase):
    def test_random_reproducible_across_processes_and_order(self):
        code = (f"import sys; sys.path.insert(0,{SRC!r}); from rv1 import xai_controls as XK;"
                "import hashlib; print(','.join(hashlib.sha256(m.tobytes()).hexdigest() "
                "for u in ('ham10000:B','ham10000:A') for m in XK.random_maps(u)))")
        outs = []
        for hs in ("0", "4242"):
            r = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True,
                               env=dict(os.environ, PYTHONHASHSEED=hs, PYTHONDONTWRITEBYTECODE="1"))
            self.assertEqual(r.returncode, 0, r.stderr)
            outs.append(r.stdout.strip().split(","))
        self.assertEqual(outs[0], outs[1])
        a = XK.random_maps("ham10000:A")
        import hashlib
        self.assertEqual([hashlib.sha256(m.tobytes()).hexdigest() for m in a], outs[0][5:])  # order-free
        self.assertEqual(len(a), 5)
        self.assertEqual(len({m.tobytes() for m in a}), 5)
        self.assertTrue(all(m.dtype == np.float64 and m.min() >= 0 and m.max() < 1 for m in a))
        state = np.random.get_state()[1].copy()
        XK.random_maps("x")
        self.assertTrue(np.array_equal(state, np.random.get_state()[1]))      # no global RNG use

    def test_centre_symmetry_and_peak(self):
        m = XK.centre_map(224, 224)
        self.assertTrue(np.array_equal(m, m[::-1, :]) and np.array_equal(m, m[:, ::-1]) and np.array_equal(m, m.T))
        peak = np.argwhere(m == m.max())
        self.assertEqual(sorted(map(tuple, peak)), [(111, 111), (111, 112), (112, 111), (112, 112)])
        self.assertAlmostEqual(float(m.max()), float(np.exp(-(0.25 + 0.25) / (2 * 56.0 ** 2))), places=15)
        self.assertAlmostEqual(float(m[111, 111 - 56]), float(np.exp(-(0.25 + 56.5 ** 2 - 0) / (2 * 56.0 ** 2))),
                               places=12)

    def test_sobel_known_edge_and_constant(self):
        rgb = np.zeros((3, 10, 12))
        rgb[:, :, 6:] = 1.0                                # vertical step between columns 5 and 6
        s = XK.sobel_map(to_norm(rgb))
        self.assertTrue(np.allclose(s[:, [5, 6]], 4.0))    # |gx| = (1+2+1)*1 at both sides
        self.assertTrue(np.allclose(np.delete(s, [5, 6], axis=1), 0.0))
        self.assertTrue(XK.map_valid(s)[0])
        flat = XK.sobel_map(to_norm(np.full((3, 10, 12), 0.4)))
        self.assertLess(float(flat.max()), 1e-12)
        self.assertEqual(XK.map_valid(flat), (False, ["constant"]))
        x0 = to_norm(rgb)
        x1 = x0.copy()
        XK.sobel_map(x1)
        self.assertTrue(np.array_equal(x0, x1))             # input untouched


class TestMismatch(unittest.TestCase):
    def recs(self, explainers=("gradcam",)):
        out = []
        for e in explainers:
            for i in range(9):
                out.append({"record_uid": f"u{i}", "case_id": f"c{i // 2}", "label": i % 2, "modality": "derm",
                            "explainer": e, "condition": "none"})
            out.append({"record_uid": "solo", "case_id": "cz", "label": 1, "modality": "clin",
                        "explainer": e, "condition": "none"})
        return out

    def test_valid_derangement_same_group(self):
        recs = self.recs()
        a = XK.mismatch_assignment(recs)
        by = {r["record_uid"]: r for r in recs}
        assigned = {k: v for k, v in a.items() if v["status"] == "assigned"}
        for k, v in assigned.items():
            u, d = v["record_uid"], v["donor"]
            self.assertEqual(k, XK.assignment_key(u, v["modality"], v["explainer"], v["condition"]))
            self.assertNotEqual(u, d)
            self.assertNotEqual(by[u]["case_id"], by[d]["case_id"])
            for f in ("label", "modality", "explainer", "condition"):
                self.assertEqual(by[u][f], by[d][f])
        for g in {tuple(v["group"]) for v in assigned.values()}:        # bijection within group
            us = [v["record_uid"] for v in assigned.values() if tuple(v["group"]) == g]
            self.assertEqual(sorted(us), sorted(v["donor"] for v in assigned.values() if tuple(v["group"]) == g))
        self.assertEqual(a[XK.assignment_key("solo", "clin", "gradcam", "none")]["status"], "unavailable")
        self.assertEqual(a, XK.mismatch_assignment(list(reversed(recs))))   # deterministic, order-free

    def test_composite_keys_do_not_overwrite(self):
        recs = self.recs(("gradcam", "ig"))
        recs += [dict(r, condition="blur@0.5") for r in self.recs(("ig",))]
        a = XK.mismatch_assignment(recs)
        self.assertEqual(len(a), len(recs))                                 # one entry per identity
        for e, c in (("gradcam", "none"), ("ig", "none"), ("ig", "blur@0.5")):
            v = a[XK.assignment_key("u0", "derm", e, c)]
            self.assertEqual((v["explainer"], v["condition"]), (e, c))
            if v["donor"]:
                self.assertEqual(v["donor_key"], XK.assignment_key(v["donor"], "derm", e, c))
        with self.assertRaisesRegex(ValueError, "duplicate mismatch assignment keys"):
            XK.mismatch_assignment(recs + [recs[0]])

    def test_no_valid_shift_and_invalid_donor(self):
        same = [{"record_uid": f"v{i}", "case_id": "one", "label": 0, "modality": "m", "explainer": "ig",
                 "condition": "none"} for i in range(3)]
        a = XK.mismatch_assignment(same)
        self.assertTrue(all(v["status"] == "unavailable" and v["reason"] == "no valid derangement shift"
                            for v in a.values()))
        a = XK.mismatch_assignment(self.recs())
        key = next(k for k, v in a.items() if v["status"] == "assigned")
        donor = a[key]["donor"]
        maps = {v["record_uid"]: np.full((2, 2), i, float) for i, v in enumerate(a.values())}
        m, st = XK.resolve_mismatch(key, a, maps, {donor: False})
        self.assertIsNone(m)
        self.assertEqual((st["status"], st["donor"]), ("unavailable", donor))    # planned donor kept
        m, st = XK.resolve_mismatch(key, a, maps, {donor: True})
        self.assertIs(m, maps[donor])


class TestScores(unittest.TestCase):
    def test_within_image_mean_and_signs(self):
        expl = {"auc_ins": 0.60, "auc_del": 0.20}
        r = XK.control_scores(expl, {"status": "available", "auc_ins_each": [0.1, 0.2, 0.3, 0.4, 0.5],
                                     "auc_del_each": [0.5, 0.4, 0.3, 0.2, 0.1]})
        self.assertAlmostEqual(r["control_auc_ins"], 0.3)
        self.assertAlmostEqual(r["insertion_advantage"], 0.30)        # explainer - control
        self.assertAlmostEqual(r["deletion_advantage"], 0.10)         # control - explainer
        worse = XK.control_scores({"auc_ins": 0.1, "auc_del": 0.9}, {"status": "available", "auc_ins": 0.5,
                                                                     "auc_del": 0.5})
        self.assertLess(worse["insertion_advantage"], 0)
        self.assertLess(worse["deletion_advantage"], 0)
        un = XK.control_scores(expl, {"status": "unavailable", "reason": "constant sobel"})
        self.assertEqual((un["status"], un["insertion_advantage"]), ("unavailable", None))

    def test_strict_random_and_nonfinite_validation(self):
        expl = {"auc_ins": 0.6, "auc_del": 0.2}
        good = [0.1, 0.2, 0.3, 0.4, 0.5]
        bad_sets = [good[:4], good + [0.6], good[:4] + [None], good[:4] + [float("nan")], good[:4] + [float("inf")]]
        for bad in bad_sets:
            for which in ("auc_ins_each", "auc_del_each"):
                ctrl = {"status": "available", "auc_ins_each": list(good), "auc_del_each": list(good)}
                ctrl[which] = bad
                r = XK.control_scores(expl, ctrl)
                self.assertEqual(r["status"], "unavailable", (which, bad))
                self.assertIsNone(r["insertion_advantage"])
        for e in ({"auc_ins": None, "auc_del": 0.2}, {"auc_ins": float("nan"), "auc_del": 0.2},
                  {"auc_ins": 0.5, "auc_del": float("inf")}, {}):
            self.assertEqual(XK.control_scores(e, {"status": "available", "auc_ins": 0.3, "auc_del": 0.3})["status"],
                             "unavailable")
        for c in ({"status": "available", "auc_ins": float("nan"), "auc_del": 0.3},
                  {"status": "available", "auc_ins": 0.3}):
            self.assertEqual(XK.control_scores(expl, c)["status"], "unavailable")


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestNewGuards(unittest.TestCase):
    def setUp(self):
        from rv1 import xai_attr as XA, xai_curves as XC
        self.XA, self.XC = XA, XC
        self.x = torch.rand(3, 6, 7)
        self.b = torch.zeros(3, 6, 7)
        self.m = torch.rand(6, 7)

    def test_nonfinite_curve_point_invalidated(self):
        class NanMid(nn.Module):
            def forward(self, x):
                o = x.sum(dim=(1, 2, 3)).view(-1, 1)
                frac = (x != 0).float().mean(dim=(1, 2, 3))
                return torch.where((frac > 0.4) & (frac < 0.6), torch.full_like(o[:, 0], float("nan")),
                                   o[:, 0]).view(-1, 1)
        r = self.XC.curves(NanMid().eval(), self.x, self.b, self.m)
        self.assertEqual(r["curve_status"], "nonfinite")
        self.assertIsNone(r["auc_insertion_v1"])
        self.assertGreater(r["n_nonfinite_points"], 0)
        rr = self.XC.analysis_curves(NanMid().eval(), self.x, self.b, self.m,
                                     self.XA.eligibility(self.XA.validity(self.m)), "gradcam")
        self.assertEqual(rr["analysis_status"], "invalid_nonfinite")

    def test_contradictory_records_rejected(self):
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 6 * 7, 1)).eval()
        v = self.XA.validity(self.m)
        liar = dict(self.XA.eligibility(v, {"failure": True}), eligible_for_primary_analysis=True,
                    ineligibility_reasons=[])
        with self.assertRaisesRegex(self.XC.IneligibleMap, "did not pass completeness"):
            self.XC.analysis_curves(model, self.x, self.b, self.m, liar, "ig")
        const = torch.zeros(6, 7)
        fake = dict(self.XA.eligibility(v))                     # claims a valid map
        with self.assertRaisesRegex(self.XC.IneligibleMap, "inconsistent eligibility metadata"):
            self.XC.analysis_curves(model, self.x, self.b, const, fake, "gradcam")
        inc = dict(self.XA.eligibility(v), eligible_for_primary_analysis=False)
        with self.assertRaisesRegex(self.XC.IneligibleMap, "inconsistent eligibility metadata"):
            self.XC.analysis_curves(model, self.x, self.b, self.m, inc, "gradcam")


if __name__ == "__main__":
    unittest.main(verbosity=2)
