"""Stage 9E: heavy configs differ from standard only in strategy fields; the heavy recipe reaches the
training dataset through the architecture adapter (no silent fallback); heavy augmentation inputs are
reproducible for a fresh dataset instance (resume identity) and differ from aug_std_v1."""
import hashlib
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
CFG = os.path.abspath(os.path.join(HERE, "..", "configs"))
sys.dont_write_bytecode = True
sys.path.insert(0, SRC)

PAIRS = {"train_smoke_v1": "rq3h_smoke_r50_v1", "rq3_smoke_effb3_v1": "rq3h_smoke_effb3_v1",
         "rq3_smoke_dn121_v1": "rq3h_smoke_dn121_v1", "train_main_resnet50_s0_v1": "rq3h_pilot_r50_v1",
         "rq3_pilot_effb3_v1": "rq3h_pilot_effb3_v1", "rq3_pilot_dn121_v1": "rq3h_pilot_dn121_v1"}
ALLOWED = {"versions.aug_version", "description", "purpose", "training.epochs"}


def flat(d, p=""):
    o = {}
    for k, v in d.items():
        o.update(flat(v, p + k + ".") if isinstance(v, dict) else {p + k: v})
    return o


def load(n):
    with open(os.path.join(CFG, n + ".json"), encoding="utf-8") as f:
        return json.load(f)


class TestConfigs(unittest.TestCase):
    def test_only_strategy_fields_differ(self):
        for std, hv in PAIRS.items():
            a, b = flat(load(std)), flat(load(hv))
            diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
            bad = {k for k in diff if k not in ALLOWED and not k.startswith("supporting_inputs.")}
            self.assertFalse(bad, (hv, bad))
            self.assertEqual(b["versions.aug_version"], "aug_heavy_v1")
            self.assertEqual(a["versions.aug_version"], "aug_std_v1")
            for k in ("training.lr", "training.weight_decay", "training.batch_size", "training.num_workers",
                      "training.precision", "training.pretrained_weights_sha256", "seeds.train_seed",
                      "training.train_subset_file", "training.probe_file"):
                self.assertEqual(a.get(k), b.get(k), (hv, k))

    def test_pilots_two_epochs_seed0(self):
        for n in ("rq3h_pilot_r50_v1", "rq3h_pilot_effb3_v1", "rq3h_pilot_dn121_v1"):
            c = load(n)
            self.assertEqual((c["purpose"], c["training"]["epochs"], c["seeds"]["train_seed"]), ("train_pilot", 2, 0))


class TestAdapterPassesRecipe(unittest.TestCase):
    def test_heavy_reaches_train_dataset(self):
        from rv1 import arch_build as AB, config as C, torch_data as TD
        seen = []
        orig = TD.ManifestDataset

        def spy(*a, **k):
            seen.append(k)
            return orig(*a, **k)
        TD.ManifestDataset = spy
        try:
            cfg = C.load(os.path.join(CFG, "rq3h_smoke_r50_v1.json"))
            model, tr, pr, pw, info = AB.build_timm(cfg, "cpu")
        finally:
            TD.ManifestDataset = orig
        train_kw = [k for k in seen if k.get("mode") == "train"]
        self.assertEqual(len(train_kw), 1)
        self.assertEqual(train_kw[0]["recipe"], "aug_heavy_v1")
        self.assertEqual(tr.recipe, "aug_heavy_v1")
        self.assertEqual(pr.mode, "eval")
        self.assertGreater(info["backbone_tensors_verified"], 0)                 # pinned pretrained backbone


class TestHeavyInputs(unittest.TestCase):
    def test_reproducible_and_distinct(self):
        from rv1 import config as C, decode as DC, torch_data as TD
        cfg = C.load(os.path.join(CFG, "rq3h_smoke_r50_v1.json"))
        root = cfg["project_root"]
        rows = DC.read_manifest(C.resolve(cfg, cfg["manifests"]["ham10000"]), "ham10000")[0]
        uids = [r["record_uid"] for r in rows if r["split"] == "train"][:6]

        def dig(recipe, epoch):
            ds = TD.ManifestDataset(rows, "ham_train", os.path.join(root, "cache"), os.path.join(root, "DATASETS"),
                                    uids, mode="train", recipe=recipe, train_seed=0, epoch=epoch)
            return [hashlib.sha256(ds[i]["x"].numpy().tobytes()).hexdigest() for i in range(len(ds))]
        h1 = dig("aug_heavy_v1", 1)
        self.assertEqual(h1, dig("aug_heavy_v1", 1))                             # fresh instance = resume identity
        self.assertNotEqual(h1, dig("aug_heavy_v1", 0))
        self.assertTrue(all(a != b for a, b in zip(h1, dig("aug_std_v1", 1))))


if __name__ == "__main__":
    unittest.main(verbosity=2)
