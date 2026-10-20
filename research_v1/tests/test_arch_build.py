"""Stage 9A tests for the RQ3 architecture adapter (CPU, real pinned weights)."""
import io
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))
CFG = {a: os.path.join(HERE, "..", "configs", f"rq3_smoke_{s}_v1.json") for a, s in (("efficientnet_b3", "effb3"), ("densenet121", "dn121"))}

try:
    import torch
    HAVE = all(os.path.exists(p) for p in CFG.values())
except ImportError:
    HAVE = False


@unittest.skipUnless(HAVE, "torch/configs missing")
class TestAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rv1 import arch_build as AB, config as C
        from rv1.seeds import seed_everything
        cls.AB, cls.built = AB, {}
        for arch, p in CFG.items():
            cfg = C.load(p)
            seed_everything(0, cfg["determinism"])
            cls.built[arch] = (cfg,) + AB.build_timm(cfg, "cpu")

    def test_head_output_and_trainable(self):
        for arch, (cfg, model, tr, pr, pw, info) in self.built.items():
            pre = self.AB.CLASSIFIER_PREFIX[arch]
            self.assertEqual(info["head_tensors"][pre + "weight"][0], 1, arch)
            model.eval()
            with torch.no_grad():
                out = model(torch.zeros(2, 3, 224, 224))
            self.assertEqual(tuple(out.shape), (2, 1))
            self.assertTrue(torch.isfinite(out).all())
            self.assertEqual(info["trainable_params"], info["total_params"])
            self.assertGreater(info["backbone_tensors_verified"], 100)
            self.assertTrue(info["file_classifier_tensors_discarded"])

    def test_provenance_detects_mismatch(self):
        from safetensors.torch import load_file
        from rv1 import config as C
        cfg, model = self.built["densenet121"][:2]
        ref = load_file(C.resolve(cfg, cfg["training"]["pretrained_weights_file"]))
        k = next(k for k in ref if "norm" in k and k.endswith("running_mean"))
        ref[k] = ref[k] + 1
        with self.assertRaisesRegex(RuntimeError, "not loaded exactly"):
            self.AB.verify_backbone(model, ref, "densenet121")

    def test_role_isolation(self):
        for arch, (cfg, model, tr, pr, pw, info) in self.built.items():
            self.assertEqual(tr.mode, "train")
            self.assertEqual(pr.mode, "eval")
            self.assertTrue(set(r["record_uid"] for r in tr.rows).isdisjoint(r["record_uid"] for r in pr.rows))

    def test_checkpoint_roundtrip_with_bn_and_optimizer(self):
        import timm
        for arch, (cfg, model, tr, pr, pw, info) in self.built.items():
            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            x = torch.randn(2, 3, 224, 224)
            loss = torch.nn.BCEWithLogitsLoss(pos_weight=pw)(model(x).squeeze(1), torch.tensor([1.0, 0.0]))
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            before = {k: v.clone() for k, v in model.state_dict().items()}
            opt.step()
            changed = sum(not torch.equal(before[k], v) for k, v in model.state_dict().items())
            self.assertGreater(changed, 0)
            bn = [k for k in model.state_dict() if k.endswith("running_mean")]
            self.assertTrue(bn)
            buf = io.BytesIO()
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "arch": arch}, buf)
            buf.seek(0)
            ck = torch.load(buf, weights_only=False)
            m2 = timm.create_model(arch, pretrained=False, num_classes=1)
            m2.load_state_dict(ck["model"], strict=True)
            self.assertTrue(all(torch.equal(m2.state_dict()[k], v) for k, v in model.state_dict().items()))
            self.assertTrue(ck["optimizer"]["state"])
            self.assertTrue(all(k in ck["model"] for k in bn))


if __name__ == "__main__":
    unittest.main(verbosity=2)
