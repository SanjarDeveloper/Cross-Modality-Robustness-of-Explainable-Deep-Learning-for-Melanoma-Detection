"""Stage 7A tests (synthetic + frozen manifests): reference coverage/keys, frozen chunk membership,
full-cohort donor mapping, atomic chunk publication/resume, primary eligibility guard."""
import csv
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))
from rv1 import xai_prod as XP  # noqa: E402

MAN = os.path.join(HERE, "..", "manifests", "v1")


def load_refs():
    from rv1 import decode as DC
    xai = XP._read_csv_rows(os.path.join(MAN, "xai_derm7pt_v1.csv"))
    derm = DC.read_manifest(os.path.join(MAN, "derm7pt_v1.csv"), "derm7pt")[0]
    return xai, derm, XP.build_refs(xai, derm)


class TestRefsAndChunks(unittest.TestCase):
    def test_full_coverage_unique(self):
        xai, derm, refs = load_refs()
        self.assertEqual(len(refs), 996)
        self.assertEqual(len({r["ref_key"] for r in refs}), 996)
        self.assertEqual(sum(r["label_mel"] == "1" for r in refs), 498)
        self.assertEqual({r["modality"] for r in refs[:2]}, {"dermoscopy", "clinical"})
        bad = [dict(xai[0], label_mel="0" if xai[0]["label_mel"] == "1" else "1")] + xai[1:]
        with self.assertRaises(ValueError):
            XP.build_refs(bad, derm)
        with self.assertRaises(ValueError):
            XP.build_refs(xai + xai[:1], derm)

    def test_chunk_plan_frozen(self):
        p = XP.chunk_plan(996)
        self.assertEqual(len(p), 21)
        self.assertEqual([c["end"] - c["start"] for c in p], [48] * 20 + [36])
        self.assertEqual(p, XP.chunk_plan(996))                        # deterministic, no regrouping
        covered = [i for c in p for i in range(c["start"], c["end"])]
        self.assertEqual(covered, list(range(996)))


class TestDonors(unittest.TestCase):
    def test_full_cohort_mapping(self):
        _, _, refs = load_refs()
        a = XP.mismatch_records(refs)
        self.assertEqual(len(a), 996 * 2)
        by = {r["record_uid"]: r for r in refs}
        for k, v in a.items():
            self.assertEqual(v["status"], "assigned")
            d, r = by[v["donor"]], by[v["record_uid"]]
            self.assertNotEqual(d["pair_id"], r["pair_id"])             # no self-case donation
            self.assertEqual((d["label_mel"], d["modality"]), (r["label_mel"], r["modality"]))
            self.assertEqual(v["explainer"], k.split("|")[2])
        pilot = {r["pair_id"] for r in XP._read_csv_rows(os.path.join(HERE, "..", "configs", "xai_pilot_stage3b_v1_pairs.csv"))}
        self.assertEqual(len(pilot), 24)                                 # production mapping built on all 249 per group


class TestAtomicResume(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="rv1_prod_")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def _tmp(self, name="chunk_00"):
        t = os.path.join(self.d, f"{name}.tmp-abc")
        os.makedirs(t)
        with open(os.path.join(t, "records.json"), "w") as f:
            json.dump([{"x": 1}], f)
        return t, os.path.join(self.d, name)

    def test_publish_and_verify(self):
        t, final = self._tmp()
        self.assertFalse(XP.verify_dir(t, "S"))                           # an in-progress dir is never complete
        XP.publish_dir(t, final, {"spec_sha256": "S"})
        self.assertTrue(XP.verify_dir(final, "S"))
        with self.assertRaisesRegex(RuntimeError, "spec hash mismatch"):
            XP.verify_dir(final, "OTHER")
        with open(os.path.join(final, "records.json"), "w") as f:
            f.write("tampered")
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            XP.verify_dir(final, "S")

    def test_interrupted_write_quarantined(self):
        t, final = self._tmp("chunk_03")                                  # never published (simulated crash)
        self.assertFalse(XP.verify_dir(final, "S"))
        moved = XP.quarantine_tmp(self.d)
        self.assertEqual(len(moved), 1)
        self.assertFalse(os.path.exists(t))
        self.assertFalse(XP.verify_dir(final, "S"))
        t2, final2 = self._tmp("chunk_04")
        XP.publish_dir(t2, final2, {"spec_sha256": "S"})
        with self.assertRaises(FileExistsError):                          # completed results never overwritten
            XP.publish_dir(self._tmp("chunk_04x")[0], final2, {"spec_sha256": "S"})


class TestEligibilityGuard(unittest.TestCase):
    def test_invalid_maps_never_primary(self):
        import torch
        from rv1 import xai_attr as XA, xai_curves as XC
        self.assertFalse(XP.primary_ok({"analysis_status": "as_computed_non_primary"}))
        self.assertFalse(XP.primary_ok({"status": "ineligible"}))
        z = torch.zeros(8, 8)
        el = XA.eligibility(XA.validity(z))
        with self.assertRaises(XC.IneligibleMap):
            XC.analysis_curves(None, torch.zeros(3, 8, 8), torch.zeros(3, 8, 8), z, el, "gradcam")
        comp = {"failure": True}
        el2 = XA.eligibility(XA.validity(torch.rand(8, 8)), comp)
        self.assertFalse(el2["eligible_for_primary_analysis"])
        with self.assertRaises(XC.IneligibleMap):
            XC.analysis_curves(None, torch.zeros(3, 8, 8), torch.zeros(3, 8, 8), torch.rand(8, 8), el2, "ig")



class TestV2Fixes(unittest.TestCase):
    """Regression tests for the two Stage 7A v1 bugs (donor eligibility; result-dir filter)."""

    def test_donor_eligible_reads_explainer_record(self):
        rec = {"status": "primary", "primary_curve": {"analysis_status": "primary"}}
        self.assertTrue(XP.donor_eligible(rec))
        self.assertFalse(XP.primary_ok(rec))                              # the v1 bug: wrong field
        self.assertFalse(XP.donor_eligible({"status": "ineligible", "as_computed_curve": {"analysis_status": "as_computed_non_primary"}}))
        self.assertFalse(XP.donor_eligible({"status": "endpoint_failure", "primary_curve": {"analysis_status": "endpoint_failure"}}))

    def test_result_dir_filter(self):
        self.assertTrue(XP.is_result_dir("chunk_07"))
        self.assertTrue(XP.is_result_dir("mismatch2_20"))
        for n in ("mismatch_assignment.json", "mismatch_03", "chunk_07.tmp-ab12", "chunk_07.quarantined-x", "invalid_v1_donor_bug"):
            self.assertFalse(XP.is_result_dir(n), n)


if __name__ == "__main__":
    unittest.main(verbosity=2)
