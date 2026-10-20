"""Stage 10B tests: localisation definitions (known maps/masks, ties, constant maps, empty/full masks, unavailable
not zero), translation/scale alignment of transformed masks, subset/pilot selection, conditions, readiness rule,
mismatch donors from other lesions."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))
import numpy as np  # noqa: E402
from rv1 import perturb as PT, xai_controls as XK, xai_sim as XS  # noqa: E402


def sq(h=224, y0=50, y1=100, x0=60, x1=120):
    m = np.zeros((h, h), bool)
    m[y0:y1, x0:x1] = True
    return m


class TestLocalisation(unittest.TestCase):
    def test_perfect_and_disjoint(self):
        mk = sq()
        r = XS.localisation(mk.astype(float), mk)
        self.assertEqual((r["status"], r["iou_area"], r["iou_half"], r["pointing_hit"]), ("available", 1.0, 1.0, True))
        far = np.zeros_like(mk, float)
        far[150:200, 150:200] = 1.0
        r = XS.localisation(far, mk)
        self.assertEqual((r["iou_area"], r["pointing_hit"]), (0.0, False))

    def test_area_matched_and_half_max(self):
        mk = sq(y0=0, y1=10, x0=0, x1=10)                     # 100 px
        m = np.zeros(mk.shape)
        m[0:10, 0:5] = 1.0                                     # 50 inside at 1.0
        m[100:110, 100:105] = 0.4                              # 50 outside at 0.4
        r = XS.localisation(m, mk)
        self.assertAlmostEqual(r["iou_area"], 50 / 150)        # top-100: 50 in + 50 out
        self.assertAlmostEqual(r["iou_half"], 50 / 100)        # >= 0.5*max -> only the 50 inside

    def test_ties_deterministic(self):
        mk = np.zeros((224, 224), bool)
        mk[0, 1] = True
        m = np.zeros((224, 224))
        m[0, 0] = m[0, 1] = 1.0                                # tied maxima; lowest flat index (0,0) is the peak
        r = XS.localisation(m, mk)
        self.assertEqual((r["pointing_hit"], r["n_max_ties"], r["any_max_in_mask"]), (False, 2, True))
        self.assertEqual(r["iou_half"], 0.5)                   # ties included at the threshold
        self.assertEqual(XS.localisation(m, mk), r)

    def test_unavailable_not_zero(self):
        mk = sq()
        for m, reason in ((np.zeros((224, 224)), "map_invalid:all_zero"), (np.full((224, 224), 2.0), "map_invalid:constant"),
                          (np.full((224, 224), np.nan), "map_invalid:nonfinite")):
            r = XS.localisation(m, mk)
            self.assertEqual((r["status"], r["reason"], r["iou_area"], r["pointing_hit"]), ("unavailable", reason, None, None))
        m = np.random.default_rng(0).random((224, 224))
        self.assertEqual(XS.localisation(m, np.zeros((224, 224), bool))["reason"], "empty_mask")
        self.assertEqual(XS.localisation(m, np.ones((224, 224), bool))["reason"], "all_foreground")
        self.assertIsNone(XS.localisation(m, mk, "ig_completeness_failure")["iou_area"])

    def test_negative_map_half_max_undefined(self):
        m = -np.random.default_rng(1).random((224, 224))
        r = XS.localisation(m, sq())
        self.assertEqual(r["status"], "available")
        self.assertIsNone(r["iou_half"])


class TestMaskGeometry(unittest.TestCase):
    def test_photometric_keep_and_scale_moves_with_image(self):
        img = np.zeros((256, 256, 3), np.uint8)
        img[100:156, 90:170] = 255
        mk = np.zeros((256, 256), np.uint8)
        mk[100:156, 90:170] = 1
        for f in ("blur", "illumination", "compression", "nuisance"):
            _, m2, _ = PT.apply(img, f, 1.0, "ham10000:T", mask256=mk)
            self.assertTrue(np.array_equal(m2, mk))
        for s in (0.5, 1.0):
            out, m2, meta = PT.apply(img, "scale", s, "ham10000:T", mask256=mk)
            ys, xs = np.nonzero(m2)
            bright = out.mean(2) > 127.5                       # half intensity (INTER_AREA blends inset edges)
            by, bx = np.nonzero(bright)
            self.assertLessEqual(abs(by.mean() - ys.mean()), 1.0)          # 03_rev2 §13: centroid within 1 px
            self.assertLessEqual(abs(bx.mean() - xs.mean()), 1.0)
            self.assertGreater((bright & (m2 > 0)).sum() / (bright | (m2 > 0)).sum(), 0.95)
            z = 1 + 1.4 * s
            self.assertAlmostEqual(m2.sum() / mk.sum(), 1 / z ** 2, delta=0.05 / z ** 2 + 0.02)
            self.assertEqual(PT.mask_to_input(m2).shape, (224, 224))

    def test_identity_baseline_condition(self):
        self.assertEqual(XS.sim_factor("none"), "blur")
        img = np.random.default_rng(2).integers(0, 255, (256, 256, 3), dtype=np.uint8)
        self.assertTrue(np.array_equal(PT.apply(img, "blur", 0.0, "u")[0], img))


class TestDesign(unittest.TestCase):
    def rows(self):
        return [{"record_uid": f"ham10000:I{i:04d}", "label_mel": "1" if i < 170 else "0", "cluster_id": f"L{i // 2}"} for i in range(400)]

    def test_subset_pilot(self):
        sub = XS.select_subset(self.rows())
        self.assertEqual((sum(r["label_mel"] == "1" for r in sub), sum(r["label_mel"] == "0" for r in sub)), (170, 170))
        self.assertEqual(sub, XS.select_subset(list(reversed(self.rows()))))       # order-independent
        pil = XS.select_pilot(sub)
        self.assertEqual(len(pil), 20)
        for lab in ("1", "0"):
            cl = [r["cluster_id"] for r in pil if r["label_mel"] == lab]
            self.assertEqual((len(cl), len(set(cl))), (10, 10))

    def test_conditions_and_counts(self):
        c = XS.conditions()
        self.assertEqual(len(c), 11)
        self.assertEqual(c[0], ("none", 0.0))
        self.assertEqual(XS.planned_counts(340, 11)["references"], 3740)

    def test_readiness_rule(self):
        p = {"overall": "pass"}
        self.assertEqual(XS.readiness(p, {"a": {"overall": "pass"}})["status"], "READY")
        self.assertEqual(XS.readiness(p, {"a": {"overall": "pass"}, "b": {"overall": "fail"}})["status"], "NOT_READY")
        self.assertEqual(XS.readiness(p, {"a": {"overall": "UNRESOLVED"}})["status"], "NOT_READY")
        self.assertEqual(XS.readiness({"overall": "fail"}, {"a": {"overall": "pass"}})["status"], "NOT_READY")

    def test_mismatch_other_lesion_same_condition(self):
        recs = [{"record_uid": f"u{i}", "case_id": f"L{i}", "label": "1", "modality": "dermoscopy", "explainer": "ig512",
                 "condition": "blur|1.000"} for i in range(10)]
        a = XK.mismatch_assignment(recs)
        case = {r["record_uid"]: r["case_id"] for r in recs}
        for v in a.values():
            self.assertEqual((v["status"], v["condition"]), ("assigned", "blur|1.000"))
            self.assertNotEqual(case[v["donor"]], case[v["record_uid"]])
        self.assertEqual(list(XK.mismatch_assignment(recs[:1]).values())[0]["status"], "unavailable")


if __name__ == "__main__":
    unittest.main(verbosity=2)
