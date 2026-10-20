"""Stage 9B tests: the six RQ3 standard-cell configs bind the right architecture, seed, sampler seed,
pinned weights and recipe; HAM train/val lists are lesion-separated and contain no test/external data."""
import csv
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, "src"))
CELLS = [(a, sh, s) for a, sh in (("efficientnet_b3", "effb3"), ("densenet121", "dn121")) for s in (0, 1, 2)]


class TestConfigs(unittest.TestCase):
    def test_bindings_and_recipe(self):
        from rv1 import config as C, provenance as P
        with open(os.path.join(ROOT, "runs", "stage9a", "weights", "weight_provenance.json"), encoding="utf-8") as f:
            prov = json.load(f)
        ref = C.load(os.path.join(ROOT, "configs", "train_main_resnet50_s0_v1.json"))["training"]
        for arch, sh, s in CELLS:
            cfg = C.load(os.path.join(ROOT, "configs", f"train_main_{sh}_s{s}_v1.json"))
            t = cfg["training"]
            self.assertEqual((t["arch"], cfg["seeds"]["train_seed"], t["sampler_seed"], t["epochs"]), (arch, s, 20260927 + s, 8))
            self.assertEqual(t["pretrained_weights_sha256"], prov[arch]["sha256"])
            self.assertEqual(P.sha256_file(C.resolve(cfg, t["pretrained_weights_file"])), prov[arch]["sha256"])
            self.assertNotIn("rq3_pilot", t["pretrained_weights_file"])               # never a pilot checkpoint
            same = {k: v for k, v in t.items() if k not in ("arch", "pretrained_model_id", "pretrained_weights_file",
                                                           "pretrained_weights_sha256", "sampler_seed")}
            self.assertEqual(same, {k: v for k, v in ref.items() if k in same})
            self.assertEqual(cfg["purpose"], "train_main")

    def test_lesion_separation_and_roles(self):
        from rv1 import decode as DC
        rows = DC.read_manifest(os.path.join(ROOT, "manifests", "v1", "ham10000_v1.csv"), "ham10000")[0]
        by = {r["record_uid"]: r for r in rows}
        lists = {}
        for name, want in (("pilot_ham_train_full_v1.csv", "train"), ("main_ham_val_full_v1.csv", "val")):
            with open(os.path.join(ROOT, "configs", name), encoding="utf-8", newline="") as f:
                u = [r["record_uid"] for r in csv.DictReader(f)]
            self.assertTrue(all(x.startswith("ham10000:") and by[x]["split"] == want for x in u))
            lists[want] = {by[x]["cluster_id"] for x in u}
        self.assertFalse(lists["train"] & lists["val"])                               # no lesion in both


if __name__ == "__main__":
    unittest.main(verbosity=2)
