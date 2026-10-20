"""Stage 3B tests: pilot selection, IG convergence denominators/UNRESOLVED, batch equivalence,
paired eligibility (no modality dropped, no zero-filling)."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))

from rv1 import xai_pairs as XP  # noqa: E402
from rv1.seeds import stable_seed  # noqa: E402


def fake_manifests(n_mel=5, n_non=6):
    xai, derm = [], []
    for i in range(n_mel + n_non):
        lab = "1" if i < n_mel else "0"
        pid = f"case{i}"
        row = {"pair_id": pid, "label_mel": lab}
        for mod in ("dermoscopy", "clinical"):
            u, iu = f"derm7pt:{pid}:{mod}", f"img:{pid}:{mod}"
            row[f"record_uid_{mod}"], row[f"image_uid_{mod}"] = u, iu
            derm.append({"record_uid": u, "image_uid": iu, "modality": mod, "pair_id": pid, "label_mel": lab,
                         "in_derm7pt_primary": "1"})
        xai.append(row)
    return xai, derm


def rec(fail=False, e_rel=0.01, eligible=True, ins=0.5, dele=0.3, curve="ok", ep="pass"):
    return {"completeness_failure": fail, "e_rel": e_rel, "eligible": eligible, "curve_status": curve,
            "endpoint_status": ep, "auc_ins": ins, "auc_del": dele, "reasons": [] if eligible else ["map:all_zero"]}


class TestSelection(unittest.TestCase):
    def test_order_and_counts(self):
        xai, derm = fake_manifests()
        sel = XP.select_pilot_pairs(xai, derm, 3)
        self.assertEqual([s["label_mel"] for s in sel], ["1"] * 3 + ["0"] * 3)
        mel = sorted([r["pair_id"] for r in xai if r["label_mel"] == "1"],
                     key=lambda p: (stable_seed(XP.PILOT_SALT, p), p))[:3]
        self.assertEqual([s["pair_id"] for s in sel[:3]], mel)
        self.assertEqual(XP.batch_subset(sel, 2), [s["pair_id"] for s in sel[:2]] + [s["pair_id"] for s in sel[3:5]])

    def test_membership_violation_raises(self):
        xai, derm = fake_manifests()
        for d in derm:
            d["in_derm7pt_primary"] = "0"
        with self.assertRaisesRegex(ValueError, "primary cohort"):
            XP.select_pilot_pairs(xai, derm, 5)

    def test_too_few(self):
        xai, derm = fake_manifests(2, 2)
        with self.assertRaises(ValueError):
            XP.select_pilot_pairs(xai, derm, 3)


class TestConvergence(unittest.TestCase):
    def test_pass_and_denominators(self):
        refs = [f"r{i}" for i in range(20)]
        a = {r: rec() for r in refs}
        b = {r: rec(ins=0.501, dele=0.302) for r in refs}
        out = XP.assess_candidate(a, b, refs)
        self.assertEqual(out["overall"], "pass")
        self.assertEqual(out["c"]["n_eligible_both"], 20)

    def test_ineligible_reported_not_zero_filled(self):
        refs = [f"r{i}" for i in range(4)]
        a = {r: rec() for r in refs}
        b = {r: rec(ins=0.6) for r in refs}          # every comparison moves 0.1
        a["r0"] = rec(eligible=False)                # excluded with a reason, NOT treated as diff 0
        out = XP.assess_candidate(a, b, refs)
        self.assertEqual(out["c"]["n_eligible_both"], 3)
        self.assertIn("m:map:all_zero", out["c"]["not_eligible_both"]["r0"])
        self.assertAlmostEqual(out["c"]["ins"]["median_abs"], 0.1)
        self.assertEqual(out["overall"], "fail")

    def test_failure_rate_counts_missing(self):
        refs = [f"r{i}" for i in range(10)]
        a = {r: rec() for r in refs[1:]}             # r0 not computed -> counted as failure
        out = XP.assess_candidate(a, a, refs)
        self.assertEqual(out["a"]["failures"], 1)
        self.assertEqual(out["a"]["denominator"], 10)
        self.assertEqual(out["a"]["status"], "fail")   # 10% > 5%

    def test_unresolved(self):
        refs = ["r0", "r1"]
        a = {r: rec(e_rel=None, eligible=False) for r in refs}
        out = XP.assess_candidate(a, a, refs)
        self.assertEqual(out["b"]["status"], "UNRESOLVED")
        self.assertEqual(out["c"]["ins"]["status"], "UNRESOLVED")
        self.assertEqual(out["overall"], "UNRESOLVED")
        self.assertIsNone(XP.choose_m({128: out, 256: out}))

    def test_endpoint_failure_excluded(self):
        refs = ["r0", "r1"]
        a = {"r0": rec(ep="fail"), "r1": rec()}
        out = XP.assess_candidate(a, a, refs)
        self.assertEqual(out["c"]["n_eligible_both"], 1)
        self.assertEqual(out["c"]["not_eligible_both"]["r0"], ["m:endpoint_failure", "2m:endpoint_failure"])

    def test_choose_smallest(self):
        self.assertEqual(XP.choose_m({128: {"overall": "fail"}, 256: {"overall": "pass"}}), 256)
        self.assertEqual(XP.choose_m({128: {"overall": "pass"}, 256: {"overall": "pass"}}), 128)


class TestBatchEquivalence(unittest.TestCase):
    def c(self, ins, ep="pass"):
        return {"curve_status": "ok", "endpoint_status": ep, "auc_insertion_v1": ins, "auc_deletion_v1": 0.2,
                "insertion": [0.1, ins], "deletion": [0.3, 0.2]}

    def test_pass_fail(self):
        self.assertTrue(XP.batch_equivalence([("a", self.c(0.5), self.c(0.5005))])["pass"])
        self.assertFalse(XP.batch_equivalence([("a", self.c(0.5), self.c(0.502))])["pass"])
        self.assertFalse(XP.batch_equivalence([("a", self.c(0.5, "fail"), self.c(0.5))])["pass"])
        self.assertFalse(XP.batch_equivalence([])["pass"])


class TestPairTable(unittest.TestCase):
    def test_both_modalities_required(self):
        xai, derm = fake_manifests(1, 1)
        pairs = XP.select_pilot_pairs(xai, derm, 1)
        es, cs = {}, {}
        for p in pairs:
            for m in ("dermoscopy", "clinical"):
                u = p[f"record_uid_{m}"]
                es[(u, "ig")] = {"usable": True, "reasons": []}
                cs[(u, "ig", "random")] = {"available": True, "reason": None}
        u_bad = pairs[0]["record_uid_clinical"]
        es[(u_bad, "ig")] = {"usable": False, "reasons": ["ig_completeness_failure"]}
        cs[(u_bad, "ig", "random")] = {"available": False, "reason": "explainer AUC missing"}
        u2 = pairs[1]["record_uid_dermoscopy"]
        cs[(u2, "ig", "random")] = {"available": False, "reason": "planned donor map not eligible"}
        rows = XP.pair_table(pairs, es, cs, ["ig"], ["random"])
        d = {(r["pair_id"], r["contrast"]): r for r in rows}
        p0, p1 = pairs[0]["pair_id"], pairs[1]["pair_id"]
        self.assertFalse(d[(p0, "explainer")]["complete_pair"])
        self.assertEqual(d[(p0, "explainer")]["reasons"], ["clinical:ig_completeness_failure"])
        self.assertFalse(d[(p0, "random")]["complete_pair"])
        self.assertTrue(d[(p1, "explainer")]["complete_pair"])
        self.assertFalse(d[(p1, "random")]["complete_pair"])
        self.assertEqual(d[(p1, "random")]["reasons"], ["dermoscopy:control:planned donor map not eligible"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
