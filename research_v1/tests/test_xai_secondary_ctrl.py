"""Stage 12A tests: variant maps reconstructed with the frozen 10A recipe match the 10A float64 hashes (and the
float32 smoothed copy does not); mismatch reuses the frozen ig512 assignment key; complete-pair population rule;
contrast directions for centre/Sobel/mismatch controls."""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))
import numpy as np  # noqa: E402
from rv1 import xai_controls as XK, xai_secondary_ctrl as XC  # noqa: E402

R7A = os.path.join(HERE, "..", "runs", "xai_prod_7a", "20261003T173258Z_3b655f88_548b3a05")
R10A = os.path.join(HERE, "..", "runs", "xai_secondary_10a", "20261004T171602Z_a97243d1_c22139d2")


@unittest.skipUnless(os.path.isdir(R10A), "runs not present")
class TestFidelity(unittest.TestCase):
    def test_variant_maps_match_10a_hashes(self):
        z = np.load(os.path.join(R7A, "chunk_00", "maps.npz"))
        zs = np.load(os.path.join(R10A, "var_00", "maps_smooth.npz"))
        with open(os.path.join(R10A, "var_00", "records.json"), encoding="utf-8") as f:
            r10 = json.load(f)
        for i in range(0, 48, 5):
            for v in XC.VARIANTS:
                m = XC.variant_map(v, z["ig512_primary"][i], z["ig512_absolute"][i])
                self.assertEqual(XC.sha64(m), r10[i]["variants"][v]["map_sha256_float64"])
            self.assertNotEqual(XC.sha64(zs["ig_pos_smooth_float32"][i]), r10[i]["variants"]["ig_pos_smooth"]["map_sha256_float64"])


class TestRules(unittest.TestCase):
    def test_donor_key_reuses_ig512_assignment(self):
        self.assertEqual(XC.donor_key("derm7pt:c1:clinical", "clinical"), "derm7pt:c1:clinical|clinical|ig512|none")

    def test_complete_pairs(self):
        rows = [{"pair_id": "a", "modality": "dermoscopy", "ins": .1, "del": .2}, {"pair_id": "a", "modality": "clinical", "ins": .0, "del": .1},
                {"pair_id": "b", "modality": "dermoscopy", "ins": .1, "del": .2}, {"pair_id": "b", "modality": "clinical", "ins": None, "del": None},
                {"pair_id": "c", "modality": "dermoscopy", "ins": .1, "del": .2}]
        self.assertEqual(sorted(XC.complete_pairs(rows)), ["a"])

    def test_control_directions(self):
        s = XK.control_scores({"auc_ins": 0.5, "auc_del": 0.2}, {"status": "available", "auc_ins": 0.3, "auc_del": 0.4})
        self.assertAlmostEqual(s["insertion_advantage"], 0.2)          # explainer - control
        self.assertAlmostEqual(s["deletion_advantage"], 0.2)           # control - explainer
        self.assertEqual(XK.control_scores({"auc_ins": 0.5, "auc_del": 0.2}, {"status": "unavailable", "reason": "x"})["status"], "unavailable")


if __name__ == "__main__":
    unittest.main(verbosity=2)
