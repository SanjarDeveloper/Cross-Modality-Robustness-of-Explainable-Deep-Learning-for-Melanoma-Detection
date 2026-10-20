"""Stage 3A tests: best-checkpoint selection (highest val AUROC, earlier epoch on ties), latest vs
best, persistence across interruption/resume, undefined/non-finite validation rejection, and
validation-precision scoping. Toy fixtures; CPU. Subprocess arms may SCRIPT the validation AUROC
(test-only patch of trainer.evaluate_selection) so ties and resume behaviour are exact."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
sys.dont_write_bytecode = True
sys.path.insert(0, SRC)
sys.path.insert(0, HERE)

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def run_arm_scripted(cfg_path, run_dir, stop, resume, script):
    """Subprocess entry: script = comma list of AUROC values for epochs 1..E ('nan' allowed)."""
    from rv1 import config as C, trainer as TR
    import test_trainer as TT
    vals = [float(v) for v in script.split(",")] if script else None
    if vals is not None:
        real = TR.evaluate_selection

        def fake(model, val_ds, device, batch_size, eval_precision):
            uids, probs, labels, _ = real(model, val_ds, device, batch_size, eval_precision)
            a = vals[fake.calls]
            fake.calls += 1
            import math
            if not math.isfinite(a):
                raise TR.ValidationUndefined(f"validation AUROC non-finite: {a}")
            return uids, probs, labels, a
        fake.calls = 0
        TR.evaluate_selection = fake
    cfg = C.load(cfg_path)
    TR.train(cfg, run_dir, TT.build_toy, "cpu", stop, resume, log=lambda *a: None)


def make_sel_project(root, epochs=3):
    import test_trainer as TT
    cp = TT.make_project(root)
    with open(cp, encoding="utf-8") as f:
        raw = json.load(f)
    raw["training"].update(epochs=epochs, best_model_selection="val_auroc_max_earliest_tie",
                           train_backend="pytorch_default_tf32conv", eval_precision="xai_precision_v1")
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(raw, f)
    return cp


def sub(cfg, rd, stop=None, resume=False, script="", crash=None, epoch_offset=0):
    code = (f"import sys; sys.dont_write_bytecode=True; sys.path.insert(0,{SRC!r}); sys.path.insert(0,{HERE!r});"
            f"import test_best_selection as T; T.run_arm_scripted({cfg!r},{rd!r},{stop!r},{resume!r},{script!r})")
    return subprocess.run([sys.executable, "-B", "-W", "ignore", "-c", code], capture_output=True, text=True,
                          env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", RV1_CRASH_AT=crash or ""))


def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestBestSelection(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="rv1_best_")
        self.cfg = make_sel_project(self.root)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def read(self, rd, name):
        with open(os.path.join(rd, name), encoding="utf-8") as f:
            return json.load(f)

    def test_best_vs_latest_and_tie_earlier(self):
        rd = os.path.join(self.root, "A")
        r = sub(self.cfg, rd, script="0.60,0.70,0.70")
        self.assertEqual(r.returncode, 0, r.stderr)
        latest, best = self.read(rd, "latest.json"), self.read(rd, "best.json")
        self.assertEqual(latest["checkpoint"], "ckpt_epoch3.pt")
        self.assertEqual((best["checkpoint"], best["epoch"], best["val_auroc"]), ("ckpt_epoch2.pt", 2, 0.70))
        from rv1 import provenance as P
        self.assertEqual(best["sha256"], P.sha256_file(os.path.join(rd, "ckpt_epoch2.pt")))
        dec = jl(os.path.join(rd, "selection_decisions.jsonl"))
        self.assertEqual([d["improved"] for d in dec], [True, True, False])
        self.assertEqual(len([n for n in os.listdir(rd) if n.startswith("val_epoch")]), 3)   # per-epoch outputs
        v = self.read(rd, [n for n in os.listdir(rd) if n.startswith("val_epoch3_")][0])
        self.assertEqual(len(v["record_uids"]), len(v["probs"]))

    def test_best_state_survives_stop_and_crash(self):
        rd = os.path.join(self.root, "B")
        self.assertEqual(sub(self.cfg, rd, stop=2, script="0.80,0.70").returncode, 0)
        r = sub(self.cfg, rd, resume=True, script="0.80")       # epoch 3 ties epoch 1 -> keep epoch 1
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.read(rd, "best.json")["epoch"], 1)
        rd2 = os.path.join(self.root, "C")                        # crash after epoch-2 checkpoint, before pointer/best
        r = sub(self.cfg, rd2, script="0.60,0.90", crash="ckpt_final:2")
        self.assertEqual(r.returncode, 97, r.stderr)
        r = sub(self.cfg, rd2, resume=True, script="0.50")
        self.assertEqual(r.returncode, 0, r.stderr)
        b = self.read(rd2, "best.json")
        self.assertEqual((b["epoch"], b["val_auroc"]), (2, 0.90))  # restored from the adopted checkpoint

    def test_nonfinite_validation_stops(self):
        rd = os.path.join(self.root, "D")
        r = sub(self.cfg, rd, script="0.6,nan,0.7")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("ValidationUndefined", r.stderr)
        self.assertEqual(self.read(rd, "latest.json")["epoch_completed"], 1)   # epoch 2 not committed
        self.assertFalse(os.path.exists(os.path.join(rd, "ckpt_epoch2.pt")))


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestEvaluateSelection(unittest.TestCase):
    def test_undefined_and_nonfinite(self):
        from rv1 import trainer as TR
        import test_trainer as TT
        one = TT.ToyDS(6, "p", 0, False)
        one.y = [1] * 6
        one.rows = [dict(r, label_mel="1") for r in one.rows]
        net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 1)).eval()
        with self.assertRaisesRegex(TR.ValidationUndefined, "fewer than two classes"):
            TR.evaluate_selection(net, one, "cpu", 4, "xai_precision_v1")

        class NanNet(nn.Module):
            def forward(self, x):
                return torch.full((x.shape[0], 1), float("nan"))
        with self.assertRaisesRegex(TR.ValidationUndefined, "non-finite"):
            TR.evaluate_selection(NanNet().eval(), TT.ToyDS(6, "p", 0, False), "cpu", 4, "xai_precision_v1")

    def test_precision_scope_restores_training_settings(self):
        from rv1 import trainer as TR
        import test_trainer as TT
        b = torch.backends
        prev = TR._backend_state()
        try:
            TR._apply_train_backend("pytorch_default_tf32conv")
            before = TR._backend_state()
            self.assertTrue(before["cudnn_allow_tf32"])
            seen = {}

            class Probe(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.l = nn.Linear(3 * 8 * 8, 1)

                def forward(self, x):
                    seen["tf32"] = b.cudnn.allow_tf32
                    return self.l(x.flatten(1))
            TR.evaluate_selection(Probe().eval(), TT.ToyDS(6, "p", 0, False), "cpu", 4, "xai_precision_v1")
            self.assertFalse(seen["tf32"])                      # validation ran under xai_precision_v1
            self.assertEqual(TR._backend_state(), before)       # training settings restored
        finally:
            b.cudnn.allow_tf32, b.cuda.matmul.allow_tf32 = prev["cudnn_allow_tf32"], prev["cuda_matmul_allow_tf32"]
            torch.set_float32_matmul_precision(prev["float32_matmul_precision"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
