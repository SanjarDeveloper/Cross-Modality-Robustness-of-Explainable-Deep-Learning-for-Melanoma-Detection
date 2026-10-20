"""Stage 4C tests: Wilson, role isolation, tone mapping, subgroup boundary, patient resampling with
mixed labels, undefined AUROC replicates, prediction-to-manifest join."""
import os
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import numpy as np  # noqa: E402
from rv1 import auroc_paired as AP, desc_stats as DS, ext_stats as ES, rq1_stats as RS  # noqa: E402


class TestWilson(unittest.TestCase):
    def test_reference(self):
        # reference values (Wilson 1927; Newcombe 1998 Table I, method 3): 0/10 -> [0, .2775]; 5/10 -> [.2366, .7634]
        lo, hi = ES.wilson(0, 10)
        self.assertAlmostEqual(lo, 0.0, places=4)
        self.assertAlmostEqual(hi, 0.2775, places=4)
        lo, hi = ES.wilson(5, 10)
        self.assertAlmostEqual(lo, 0.2366, places=4)
        self.assertAlmostEqual(hi, 0.7634, places=4)
        lo, hi = ES.wilson(81, 263)                               # Newcombe example: .2553-.3662
        self.assertAlmostEqual(lo, 0.2553, places=4)
        self.assertAlmostEqual(hi, 0.3662, places=4)
        self.assertIsNone(ES.wilson(0, 0))
        self.assertEqual(ES.wilson(7, 7)[1], 1.0)


class TestRolesTones(unittest.TestCase):
    def test_role_isolation(self):
        rows = [{"record_uid": "a", "cluster_id": "P1", "rq5_role": "eval"},
                {"record_uid": "b", "cluster_id": "P1", "rq5_role": "eval"},
                {"record_uid": "c", "cluster_id": "P2", "rq5_role": "adapt_val"}]
        self.assertEqual(ES.check_role_isolation(rows), {"eval": 1, "adapt_train": 0, "adapt_val": 1})
        with self.assertRaisesRegex(ValueError, "more than one role"):
            ES.check_role_isolation(rows + [{"record_uid": "d", "cluster_id": "P1", "rq5_role": "adapt_train"}])
        with self.assertRaisesRegex(ValueError, "unknown role"):
            ES.check_role_isolation([{"record_uid": "x", "cluster_id": "P9", "rq5_role": "test"}])

    def test_tones(self):
        self.assertEqual(ES.pad_tone("2.0"), "FST 2")
        self.assertEqual(ES.pad_tone(""), "missing")
        with self.assertRaises(ValueError):
            ES.pad_tone("7.0")
        self.assertEqual(ES.ddi_tone("56"), "FST V-VI")
        with self.assertRaises(ValueError):
            ES.ddi_tone("")

    def test_subgroup_boundary(self):
        self.assertFalse(ES.subgroup_eligible(4))
        self.assertTrue(ES.subgroup_eligible(5))


class TestPatientBootstrap(unittest.TestCase):
    def test_mixed_labels_multiplicity_and_undefined(self):
        # patient P0 has a melanoma AND a nevus image: labels stay per image
        pats = {"P0": [{"label": 1, "prob": 0.9}, {"label": 0, "prob": 0.4}],
                "P1": [{"label": 0, "prob": 0.1}], "P2": [{"label": 0, "prob": 0.2}, {"label": 0, "prob": 0.3}],
                "P3": [{"label": 0, "prob": 0.5}]}
        ids = sorted(pats)
        rng = np.random.Generator(np.random.PCG64(2))
        draw = RS.resample_ids(ids, rng)
        flat = DS.flatten_clusters([pats[i] for i in draw])
        self.assertEqual(sum(r["label"] for r in flat), draw.count("P0"))       # image labels preserved
        self.assertEqual(len(flat), sum(len(pats[i]) for i in draw))

        def m_auc(cl):
            f = DS.flatten_clusters(cl)
            pos = [r["prob"] for r in f if r["label"] == 1]
            neg = [r["prob"] for r in f if r["label"] == 0]
            return (AP.auc(pos, neg),) if pos and neg else None
        b = RS.pair_bootstrap(pats, ids, m_auc, 400, 1)
        self.assertGreater(b["n_undefined"], 0)               # replicates without P0 have no melanoma
        self.assertEqual(b["n_undefined"] + b["n_defined"], 400)
        self.assertTrue(b["unreliable"])


class TestJoin(unittest.TestCase):
    def test_complete_join(self):
        from rv1 import ext_eval as EE
        man = [{"record_uid": "a", "label_mel": "1"}, {"record_uid": "b", "label_mel": "0"}]
        EE.check_join(man, [{"record_uid": "b", "label_mel": "0"}, {"record_uid": "a", "label_mel": "1"}])
        with self.assertRaises(RuntimeError):
            EE.check_join(man, [{"record_uid": "a", "label_mel": "1"}])
        with self.assertRaises(RuntimeError):
            EE.check_join(man, [{"record_uid": "a", "label_mel": "1"}, {"record_uid": "a", "label_mel": "1"}])
        with self.assertRaises(RuntimeError):
            EE.check_join(man, [{"record_uid": "a", "label_mel": "0"}, {"record_uid": "b", "label_mel": "0"}])


if __name__ == "__main__":
    unittest.main(verbosity=2)
