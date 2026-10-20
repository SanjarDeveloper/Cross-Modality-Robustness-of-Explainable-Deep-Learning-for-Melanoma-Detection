"""Stage 2C2 tests: epoch-boundary checkpoint/resume with a CPU toy model (dropout).

Needs torch (research_v1/.venv_model). Every training arm runs in a FRESH subprocess.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.dont_write_bytecode = True
sys.path.insert(0, SRC)
sys.path.insert(0, HERE)

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

N_TRAIN, N_PROBE = 24, 10


# ----------------------------------------------------------------- toy build (importable)
class ToyDS:
    """Dataset with seed/epoch-keyed 'augmentation' noise, same item format as torch_data."""

    def __init__(self, n, prefix, train_seed, noisy):
        import numpy as np
        rng = np.random.default_rng(123 if prefix == "t" else 456)
        self.x = rng.normal(size=(n, 3, 8, 8)).astype("float32")
        self.y = [(i % 2) for i in range(n)]
        self.rows = [{"record_uid": f"{prefix}{i}", "image_uid": f"{prefix}{i}",
                      "label_mel": str(self.y[i])} for i in range(n)]
        self.train_seed, self.noisy, self.epoch = train_seed, noisy, 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import numpy as np
        from rv1.seeds import stable_seed
        x = self.x[i].copy()
        if self.noisy:
            x += np.random.default_rng(stable_seed(self.train_seed, self.epoch, self.rows[i]["image_uid"])
                                       ).normal(scale=0.1, size=x.shape).astype("float32")
        return {"x": torch.from_numpy(x), "label": torch.tensor(self.y[i], dtype=torch.int64),
                "record": dict(self.rows[i]), "mask": None, "mask_flags": []}


def build_toy(cfg, device):
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.BatchNorm2d(4),
                              torch.nn.ReLU(), torch.nn.Flatten(), torch.nn.Dropout(0.5),
                              torch.nn.Linear(4 * 8 * 8, 1)).to(device)
    ts = cfg["seeds"]["train_seed"]
    return net, ToyDS(N_TRAIN, "t", ts, True), ToyDS(N_PROBE, "p", ts, False), \
        torch.tensor([1.0]), {"toy": True}


def run_arm(cfg_path, run_dir, stop=None, resume=False, break_rng_restore=False):
    """Entry point for subprocesses."""
    from rv1 import config as C, trainer as TR
    if break_rng_restore:                      # negative control (test only): skip restore
        TR._set_rng_state = lambda s: None
        if not resume or os.environ.get("RV1_TEST_PATCH_DIGEST") == "1":
            TR.rng_digest = lambda s: "patched"
    cfg = C.load(cfg_path)
    TR.train(cfg, run_dir, build_toy, "cpu", stop, resume, log=lambda *a: None)


def make_project(root, lr=0.05):
    os.makedirs(os.path.join(root, "m"), exist_ok=True)
    os.makedirs(os.path.join(root, "cfg"), exist_ok=True)
    for rel, txt in (("m/toy.csv", "a\n1\n"), ("m/env.json", "{}"),
                     ("m/subset.csv", "record_uid\n" + "\n".join(f"t{i}" for i in range(N_TRAIN)) + "\n"),
                     ("m/probe.csv", "record_uid\n" + "\n".join(f"p{i}" for i in range(N_PROBE)) + "\n")):
        p = os.path.join(root, *rel.split("/"))
        with open(p, "w", encoding="utf-8") as f:
            f.write(txt)
        if rel in ("m/toy.csv", "m/env.json"):
            from rv1 import provenance as P
            with open(p + ".sha256", "w", encoding="utf-8") as f:
                f.write(f"{P.sha256_file(p)}  {os.path.basename(p)}\n")
    with open(os.path.join(PROJECT, "research_v1", "configs", "train_smoke_v1.json"), encoding="utf-8") as f:
        raw = json.load(f)
    raw.update(project_root="..", manifests={"toy": "m/toy.csv"}, environment_inventory="m/env.json",
               legacy_sources=[], supporting_inputs={})
    raw["training"].update(arch="toy_cnn", pretrained_model_id="none", pretrained_weights_file="none",
                           pretrained_weights_sha256="none", batch_size=5, lr=lr, epochs=2,
                           train_subset_file="m/subset.csv", probe_file="m/probe.csv")
    cp = os.path.join(root, "cfg", "toy.json")
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(raw, f)
    return cp


def sub(cfg, rd, stop=None, resume=False, brk=False, patch_digest=False, crash=None):
    code = (f"import sys; sys.dont_write_bytecode=True; sys.path.insert(0,{SRC!r}); sys.path.insert(0,{HERE!r});"
            f"import test_trainer as T; T.run_arm({cfg!r},{rd!r},{stop!r},{resume!r},{brk!r})")
    r = subprocess.run([sys.executable, "-B", "-W", "ignore", "-c", code], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                                RV1_TEST_PATCH_DIGEST="1" if patch_digest else "0",
                                RV1_CRASH_AT=crash or ""))
    return r


def final(rd):
    with open(os.path.join(rd, "latest.json"), encoding="utf-8") as f:
        ptr = json.load(f)
    return torch.load(os.path.join(rd, ptr["checkpoint"]), weights_only=False), ptr


def steps(rd):
    with open(os.path.join(rd, "steps.jsonl"), encoding="utf-8") as f:
        return [json.loads(l) for l in f]


@unittest.skipUnless(HAVE_TORCH, "torch not installed in this interpreter")
class TestResume(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="rv1_tr_")
        cls.cfg = make_project(cls.root)
        cls.U, cls.R = os.path.join(cls.root, "U"), os.path.join(cls.root, "R")
        for args in ((cls.U,), (cls.R, 1)):
            r = sub(cls.cfg, *args)
            assert r.returncode == 0, r.stderr
        cls.mid = final(cls.R)
        r = sub(cls.cfg, cls.R, None, True)
        assert r.returncode == 0, r.stderr

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_uninterrupted_equals_fresh_process_resume(self):
        (u, pu), (r, pr) = final(self.U), final(self.R)
        self.assertEqual((pu["epoch_completed"], pu["global_step"]), (2, 10))
        self.assertEqual((pr["epoch_completed"], pr["global_step"]), (2, 10))
        md = max(float((u["model"][k].double() - r["model"][k].double()).abs().max()) for k in u["model"])
        self.assertLessEqual(md, 1e-6)                             # parameters AND buffers
        self.assertTrue(any("running_mean" in k for k in u["model"]))
        pd = max(abs(a - b) for a, b in zip(u["probe"]["probs"], r["probe"]["probs"]))
        self.assertLessEqual(pd, 1e-6)
        su, sr = steps(self.U), steps(self.R)
        self.assertEqual([(s["record_uids"], s["x_digest"], s["lr_used"], s["global_step"]) for s in su],
                         [(s["record_uids"], s["x_digest"], s["lr_used"], s["global_step"]) for s in sr])
        self.assertEqual(u["scheduler"], r["scheduler"])
        self.assertEqual(u["rng_digest"], r["rng_digest"])

    def test_epoch_one_checkpoint_state(self):
        ck, ptr = self.mid
        self.assertEqual((ptr["epoch_completed"], ptr["global_step"], ck["scheduler"]["last_epoch"]), (1, 5, 5))
        for k in ("model", "optimizer", "scheduler", "rng", "next_sampler_state", "fingerprint", "config"):
            self.assertIn(k, ck)
        with open(os.path.join(self.R, "resume_events.jsonl"), encoding="utf-8") as f:
            ev = json.loads(f.readline())
        self.assertEqual(ev["rng_digest_restored"], ck["rng_digest"])     # actual state restored
        self.assertEqual(ck["fingerprint"]["digest"], final(self.U)[0]["fingerprint"]["digest"])

    def test_rng_restore_is_exercised(self):
        # (a) the trainer's own check detects a skipped restore
        rd0 = os.path.join(self.root, "R_detect")
        self.assertEqual(sub(self.cfg, rd0, 1).returncode, 0)
        r = sub(self.cfg, rd0, None, True, True)
        self.assertIn("RNG state not restored exactly", r.stderr)
        # (b) with the check also disabled in both arms, skipping restore changes the result
        rd = os.path.join(self.root, "R_norng")
        self.assertEqual(sub(self.cfg, rd, 1, False, True).returncode, 0)
        self.assertEqual(sub(self.cfg, rd, None, True, True, patch_digest=True).returncode, 0)
        u, b = final(self.U)[0], final(rd)[0]
        md = max(float((u["model"][k].double() - b["model"][k].double()).abs().max()) for k in u["model"])
        self.assertGreater(md, 1e-6)          # dropout RNG matters: skipping restore diverges

    def test_mismatch_rejected_and_no_overwrite(self):
        root2 = tempfile.mkdtemp(prefix="rv1_tr2_")
        try:
            cfg2 = make_project(root2)
            rd = os.path.join(root2, "R")
            self.assertEqual(sub(cfg2, rd, 1).returncode, 0)
            r = sub(cfg2, rd)                                    # fresh start into used dir
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("already has a run", r.stderr)
            with open(os.path.join(root2, "m", "subset.csv"), "a", encoding="utf-8") as f:
                f.write("t0\n")                                  # input change
            r = sub(cfg2, rd, None, True)
            self.assertIn("ResumeMismatch", r.stderr)
            self.assertIn("input:m/subset.csv", r.stderr)
            make_project(root2, lr=0.01)                         # scientific config change
            r = sub(cfg2, rd, None, True)
            self.assertIn("ResumeMismatch", r.stderr)
            self.assertIn("config_sha256", r.stderr)
        finally:
            shutil.rmtree(root2, ignore_errors=True)

    def test_operational_args_not_in_fingerprint(self):
        from rv1 import config as C, trainer as TR
        cfg = C.load(self.cfg)
        self.assertEqual(TR.fingerprint(cfg)["digest"], final(self.R)[0]["fingerprint"]["digest"])


# ------------------------------------------------------------------ Stage 2C2.1
SPE = 5          # steps per epoch for the toy (24 samples, batch 5)
KEYS = ("epoch", "global_step", "record_uids", "x_digest", "lr_used", "loss")


def canon(rd):
    from rv1.trainer import canonical_steps
    return [tuple(json.dumps(s[k]) for k in KEYS) for s in canonical_steps(rd, SPE)]


def jl(rd, name):
    p = os.path.join(rd, name)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@unittest.skipUnless(HAVE_TORCH, "torch not installed in this interpreter")
class TestCrashRecovery(unittest.TestCase):
    """Simulated process crashes (os._exit via RV1_CRASH_AT); not power-loss tests."""

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="rv1_cr_")
        cls.cfg = make_project(cls.root)
        cls.U = os.path.join(cls.root, "U")
        r = sub(cls.cfg, cls.U)
        assert r.returncode == 0, r.stderr
        cls.u_final, _ = final(cls.U)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def assert_matches_U(self, rd):
        f, ptr = final(rd)
        u = self.u_final
        self.assertEqual((ptr["epoch_completed"], ptr["global_step"]), (2, 2 * SPE))
        md = max(float((u["model"][k].double() - f["model"][k].double()).abs().max()) for k in u["model"])
        self.assertLessEqual(md, 1e-6)
        om = max(float((v.double() - f["optimizer"]["state"][sid][n].double()).abs().max())
                 for sid, st in u["optimizer"]["state"].items() for n, v in st.items()
                 if isinstance(v, torch.Tensor))
        self.assertLessEqual(om, 1e-6)
        self.assertEqual(u["scheduler"], f["scheduler"])
        self.assertLessEqual(max(abs(a - b) for a, b in zip(u["probe"]["probs"], f["probe"]["probs"])), 1e-6)
        self.assertEqual(canon(rd), canon(self.U))

    def crash_then_resume(self, name, crash, stop=None):
        rd = os.path.join(self.root, name)
        r = sub(self.cfg, rd, stop, crash=crash)
        self.assertEqual(r.returncode, 97, r.stderr)                     # really crashed
        r = sub(self.cfg, rd, None, True)
        self.assertEqual(r.returncode, 0, r.stderr)
        return rd

    def test_crash_mid_epoch_after_steps(self):
        rd = self.crash_then_resume("mid", f"step:{SPE + 2}")            # 2 steps into epoch 2
        self.assert_matches_U(rd)
        self.assertEqual(len(jl(rd, "steps.jsonl")), 2 * SPE + 2)       # partial steps kept raw
        self.assertEqual(len(canon(rd)), 2 * SPE)                        # but not canonical
        self.assertEqual([a["mode"] for a in jl(rd, "attempts.jsonl")], ["fresh", "resume_from_checkpoint"])

    def test_crash_during_checkpoint_write(self):
        rd = self.crash_then_resume("tmp", "ckpt_tmp:2")
        self.assert_matches_U(rd)
        rec = jl(rd, "recovery.jsonl")
        self.assertTrue(any(e["action"] == "quarantine" and ".tmp-" in e["file"] for e in rec))
        self.assertTrue(os.listdir(os.path.join(rd, "quarantine")))      # evidence preserved

    def test_crash_after_finalize_before_pointer(self):
        rd = self.crash_then_resume("orphan", "ckpt_final:2")
        self.assertTrue(any(e["action"] == "adopt_orphan" for e in jl(rd, "recovery.jsonl")))
        self.assert_matches_U(rd)
        self.assertEqual([c["how"] for c in jl(rd, "commits.jsonl")], ["normal", "adopted"])

    def test_first_epoch_interruption_restarts_from_init(self):
        rd = self.crash_then_resume("first", "step:2")
        self.assert_matches_U(rd)
        self.assertEqual([a["mode"] for a in jl(rd, "attempts.jsonl")],
                         ["fresh", "restart_from_initialisation"])
        self.assertEqual(len(jl(rd, "steps.jsonl")), 2 + 2 * SPE)

    def test_first_epoch_checkpoint_write_crash(self):
        rd = self.crash_then_resume("first_tmp", "ckpt_tmp:1")
        self.assert_matches_U(rd)

    def test_stale_tmp_files_quarantined(self):
        rd = os.path.join(self.root, "stale")
        self.assertEqual(sub(self.cfg, rd, 1).returncode, 0)
        for n in ("ckpt_epoch2.pt.tmp-a9-deadbeef", "latest.json.tmp-a9-cafebabe"):
            with open(os.path.join(rd, n), "wb") as f:
                f.write(b"partial")
        self.assertEqual(sub(self.cfg, rd, None, True).returncode, 0)
        q = [e["file"] for e in jl(rd, "recovery.jsonl") if e["action"] == "quarantine"]
        self.assertEqual(sorted(q), ["ckpt_epoch2.pt.tmp-a9-deadbeef", "latest.json.tmp-a9-cafebabe"])
        self.assert_matches_U(rd)

    def test_invalid_orphan_quarantined_then_replayed(self):
        for variant in ("garbage_valid_sidecar", "bad_sidecar"):
            with self.subTest(variant):
                rd = os.path.join(self.root, variant)
                self.assertEqual(sub(self.cfg, rd, 1).returncode, 0)
                p = os.path.join(rd, "ckpt_epoch2.pt")
                with open(p, "wb") as f:
                    f.write(b"not a checkpoint")
                sha = (hashlib.sha256(b"not a checkpoint").hexdigest() if variant.startswith("garbage")
                       else "0" * 64)
                with open(p + ".sha256", "w", encoding="utf-8") as f:
                    f.write(sha + "  ckpt_epoch2.pt" + chr(10))
                r = sub(self.cfg, rd, None, True)
                self.assertEqual(r.returncode, 0, r.stderr)
                rec = jl(rd, "recovery.jsonl")
                self.assertTrue(any(e["file"] == "ckpt_epoch2.pt" and e["action"] == "quarantine" for e in rec))
                self.assert_matches_U(rd)                                # replayed, new file written

    def test_corrupted_pointer_checkpoint_rejected(self):
        rd = os.path.join(self.root, "corrupt")
        self.assertEqual(sub(self.cfg, rd, 1).returncode, 0)
        p = os.path.join(rd, "ckpt_epoch1.pt")
        with open(p, "r+b") as f:
            f.seek(2000)
            b = f.read(1)
            f.seek(2000)
            f.write(bytes([b[0] ^ 0xFF]))
        r = sub(self.cfg, rd, None, True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("CheckpointIntegrityError", r.stderr)
        self.assertFalse(os.path.exists(os.path.join(rd, "ckpt_epoch2.pt")))

    def test_pointer_inconsistent_with_checkpoint_rejected(self):
        rd = os.path.join(self.root, "ptr")
        self.assertEqual(sub(self.cfg, rd, 1).returncode, 0)
        lp = os.path.join(rd, "latest.json")
        with open(lp, encoding="utf-8") as f:
            ptr = json.load(f)
        ptr["global_step"] += 1
        with open(lp, "w", encoding="utf-8") as f:
            json.dump(ptr, f)
        r = sub(self.cfg, rd, None, True)
        self.assertIn("pointer epoch/global_step differ", r.stderr)


class TestRngDigest(unittest.TestCase):
    @unittest.skipUnless(HAVE_TORCH, "torch not installed")
    def test_all_numpy_fields_covered(self):
        import numpy as np
        from rv1 import trainer as TR
        np.random.seed(1)
        np.random.normal()                          # leaves has_gauss=1 with a cached value
        s = TR._rng_state()
        name, keys, pos, hg, cached = s["numpy"]
        self.assertEqual(hg, 1)
        base = TR.rng_digest(s)
        for i, new in ((0, "PCG64"), (2, pos + 1), (3, 0), (4, cached + 1.0)):
            t = list(s["numpy"])
            t[i] = new
            self.assertNotEqual(TR.rng_digest(dict(s, numpy=tuple(t))), base, i)
        k2 = keys.copy()
        k2[0] ^= 1
        self.assertNotEqual(TR.rng_digest(dict(s, numpy=(name, k2, pos, hg, cached))), base)
        TR._set_rng_state(s)                        # restore behaviour intact
        self.assertEqual(TR.rng_digest(TR._rng_state()), base)


if __name__ == "__main__":
    unittest.main(verbosity=2)
