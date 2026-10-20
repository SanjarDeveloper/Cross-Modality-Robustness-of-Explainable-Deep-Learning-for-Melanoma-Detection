"""Stage 5A tests for the PAD adaptation build function (real frozen inputs, CPU).
Selection/tie/best-pointer behaviour of the unchanged trainer is covered by test_best_selection."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))
CFG = os.path.abspath(os.path.join(HERE, "..", "configs", "train_adapt_pad_s0_v1.json"))

try:
    import torch
    HAVE = os.path.exists(CFG)
except ImportError:
    HAVE = False


@unittest.skipUnless(HAVE, "torch or config missing")
class TestAdaptBuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rv1 import adapt_train as AT, config as C
        cls.AT, cls.cfg = AT, C.load(CFG)
        cls.model, cls.tr, cls.va, cls.pw, cls.info = AT.build_resnet50_adapt(cls.cfg, "cpu")

    def test_base_weights_and_buffers(self):
        from rv1 import config as C, trainer as TR
        ck = TR._load_verified(C.resolve(self.cfg, self.cfg["training"]["pretrained_weights_file"]), self.AT.BASE_SHA)
        sd = self.model.state_dict()
        self.assertEqual(set(sd), set(ck["model"]))
        for k in ("conv1.weight", "fc.weight", "fc.bias", "bn1.running_mean", "bn1.running_var",
                  "layer4.2.bn3.num_batches_tracked"):
            self.assertTrue(torch.equal(sd[k], ck["model"][k]), k)
        self.assertIn("optimizer", ck)                        # present in the base file ...
        self.assertIn("fresh", self.info["optimizer_scheduler"])   # ... but never used by the build

    def test_no_eval_records_and_counts(self):
        from rv1 import config as C, decode as DC
        rows = DC.read_manifest(C.resolve(self.cfg, self.cfg["manifests"]["pad_ufes_20"]), "pad_ufes_20")[0]
        role = {r["record_uid"]: r["rq5_role"] for r in rows}
        self.assertTrue(all(role[r["record_uid"]] == "adapt_train" for r in self.tr.rows))
        self.assertTrue(all(role[r["record_uid"]] == "adapt_val" for r in self.va.rows))
        self.assertEqual((len(self.tr), len(self.va)), (805, 343))
        self.assertEqual(self.info["n_val_mel"], 9)
        self.assertEqual(self.tr.mode, "train")
        self.assertEqual(self.va.mode, "eval")

    def test_pos_weight_train_only(self):
        self.assertAlmostEqual(float(self.pw), 788 / 17, places=5)
        self.assertEqual(self.info["n_train_mel"], 17)

    def test_split_check_rejects_eval_record(self):
        from rv1 import config as C, decode as DC
        rows = DC.read_manifest(C.resolve(self.cfg, self.cfg["manifests"]["pad_ufes_20"]), "pad_ufes_20")[0]
        real = self.AT._uids
        ev = next(r for r in rows if r["rq5_role"] == "eval")

        def tampered(cfg, rel):
            lst = real(cfg, rel)
            if rel.endswith("adapt_val_v1.csv"):
                lst = lst[:-1] + [{"record_uid": ev["record_uid"], "label_mel": ev["label_mel"],
                                   "cluster_id": ev["cluster_id"]}]
            return lst
        self.AT._uids = tampered
        try:
            with self.assertRaisesRegex(RuntimeError, "differs from manifest role"):
                self.AT.check_split(self.cfg, rows)
        finally:
            self.AT._uids = real


if __name__ == "__main__":
    unittest.main(verbosity=2)
