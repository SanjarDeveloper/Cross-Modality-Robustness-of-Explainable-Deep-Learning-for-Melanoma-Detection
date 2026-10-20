"""Stage 6A tests: seed-replicate configs share membership/recipe; backbone from the pretrained
source; head init reproducible within a seed and different between seeds; only HAM train/val."""
import csv
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))
CFG = {s: os.path.abspath(os.path.join(HERE, "..", "configs", f"train_main_resnet50_s{s}_v1.json")) for s in (0, 1, 2)}

try:
    import torch  # noqa: F401
    HAVE = all(os.path.exists(p) for p in CFG.values())
except ImportError:
    HAVE = False


@unittest.skipUnless(HAVE, "torch or configs missing")
class TestSeedReplicates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rv1 import config as C
        cls.cfg = {s: C.load(p) for s, p in CFG.items()}

    def test_same_membership_and_recipe(self):
        t0 = self.cfg[0]["training"]
        for s in (1, 2):
            t = self.cfg[s]["training"]
            self.assertEqual({k: v for k, v in t.items() if k != "sampler_seed"},
                             {k: v for k, v in t0.items() if k != "sampler_seed"})
            self.assertEqual(t["sampler_seed"], 20260927 + s)
            self.assertEqual(self.cfg[s]["seeds"]["train_seed"], s)
            self.assertEqual(self.cfg[s]["seeds"]["split_seed"], 0)

    def test_lists_only_ham_train_val(self):
        from rv1 import config as C, decode as DC
        cfg = self.cfg[1]
        rows = DC.read_manifest(C.resolve(cfg, cfg["manifests"]["ham10000"]), "ham10000")[0]
        split = {r["record_uid"]: r["split"] for r in rows}
        for key, want in (("train_subset_file", "train"), ("probe_file", "val")):
            with open(C.resolve(cfg, cfg["training"][key]), encoding="utf-8", newline="") as f:
                u = [r["record_uid"] for r in csv.DictReader(f)]
            self.assertEqual(len(u), len(set(u)))
            self.assertTrue(all(x.startswith("ham10000:") and split[x] == want for x in u))
            self.assertEqual(sorted(u), sorted(k for k, v in split.items() if v == want))

    def test_head_init_and_backbone(self):
        from rv1 import smoke_train as ST
        from rv1.seeds import seed_everything
        heads = {}
        for s, rep in ((1, 0), (1, 1), (2, 0)):
            seed_everything(s, self.cfg[s]["determinism"])
            model, tr, pr, pw, info = ST.build_resnet50(self.cfg[s], "cpu")   # verifies all backbone tensors
            self.assertEqual(info["backbone_tensors_verified"], 318)
            heads[(s, rep)] = model.state_dict()["fc.weight"].clone()
        self.assertTrue(torch.equal(heads[(1, 0)], heads[(1, 1)]))             # reproducible within seed
        self.assertFalse(torch.equal(heads[(1, 0)], heads[(2, 0)]))            # differs between seeds
        self.assertAlmostEqual(info["pos_weight"], 6155 / 783, places=4)

    def test_fresh_start_semantics(self):
        """The build loads only the pretrained safetensors file; the trainer creates the optimizer/
        scheduler; a fresh run refuses a directory that already has a run (no accidental resume)."""
        from rv1 import trainer as TR
        import tempfile
        d = tempfile.mkdtemp(prefix="rv1_seed_")
        with open(os.path.join(d, "run_start.json"), "w") as f:
            json.dump({}, f)
        with self.assertRaises(FileExistsError):
            TR.train(self.cfg[1], d, None, "cpu")
        self.assertTrue(self.cfg[1]["training"]["pretrained_weights_file"].endswith("model.safetensors"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
