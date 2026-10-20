"""Stage 2B1 foundation tests (stdlib unittest, CPU only, no images/models).

Run from the project root:
  set PYTHONDONTWRITEBYTECODE=1
  python -B -m unittest discover -s research_v1/tests -p "test_*.py" -v
"""
import copy
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))
CONFIG = os.path.join(PROJECT, "research_v1", "configs", "foundation_check_v1.json")
sys.dont_write_bytecode = True
sys.path.insert(0, SRC)

from rv1 import config as C, seeds as S, provenance as P  # noqa: E402

DET = {"cudnn_deterministic": True, "cudnn_benchmark": False,
       "use_deterministic_algorithms": True, "warn_only": True}


def child_env(**extra):
    env = {k: v for k, v in os.environ.items()}
    env.update(PYTHONDONTWRITEBYTECODE="1", **extra)
    return env


def run_py(code, **env):
    r = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True,
                       env=child_env(**env), cwd=tempfile.gettempdir())
    if r.returncode:
        raise AssertionError(r.stderr)
    return r.stdout.strip()


class T1_CrossProcess(unittest.TestCase):
    def test_stable_seed_independent_of_pythonhashseed(self):
        code = (f"import sys; sys.path.insert(0, {SRC!r}); from rv1.seeds import stable_seed;"
                "print(stable_seed(0,'ISIC_0000001','blur'), stable_seed('ctrl',3,'x'),"
                " hash('ISIC_0000001'))")
        a = run_py(code, PYTHONHASHSEED="0").split()
        b = run_py(code, PYTHONHASHSEED="12345").split()
        self.assertEqual(a[:2], b[:2])
        self.assertNotEqual(a[2], b[2], "built-in hash() should differ (sanity of the setup)")
        self.assertEqual(int(a[0]), S.stable_seed(0, "ISIC_0000001", "blur"))


class T2_Encoding(unittest.TestCase):
    def test_ambiguous_inputs_distinct(self):
        cases = [(1, "23"), ("1", 23), ("1", "23"), (1, 23), ("12", "3"), (12, 3),
                 ("a|b", "c"), ("a", "b|c"), ("a", "b", "c"), ("abc",), (True,), (1,),
                 (b"1",), ("1",)]
        enc = [S.encode_parts(*c) for c in cases]
        self.assertEqual(len(set(enc)), len(enc))
        seeds = [S.stable_seed(*c) for c in cases]
        self.assertEqual(len(set(seeds)), len(seeds))  # empirical; collisions not impossible
        for s in seeds:
            self.assertTrue(0 <= s <= S.SEED_MAX)

    def test_unsupported_types_rejected(self):
        for bad in (1.0, None, [1], {"a": 1}):
            with self.assertRaises(TypeError):
                S.stable_seed(bad)
        with self.assertRaises(ValueError):
            S.stable_seed()


class T3_RngReproducible(unittest.TestCase):
    def test_python_numpy_torch_cpu(self):
        import numpy as np
        import torch

        def draw():
            S.seed_everything(42, DET)
            return ([random.random() for _ in range(5)], np.random.rand(5).tolist(),
                    torch.rand(5).tolist())
        a, b = draw(), draw()
        self.assertEqual(a, b)
        S.seed_everything(43, DET)
        self.assertNotEqual(a[0], [random.random() for _ in range(5)])

    def test_record_contents(self):
        rec = S.seed_everything(7, DET)
        self.assertEqual(rec["base_seed"], 7)
        self.assertTrue(rec["torch"]["cudnn_deterministic"])
        self.assertFalse(rec["torch"]["cudnn_benchmark"])
        self.assertLess(rec["derived_seeds"]["numpy_legacy"], 2 ** 32)

    def test_bad_determinism_rejected(self):
        with self.assertRaises(ValueError):
            S.seed_everything(1, {"cudnn_deterministic": True})
        with self.assertRaises(ValueError):
            S.seed_everything(-1, DET)


def raw_config():
    with open(CONFIG, encoding="utf-8") as f:
        return json.load(f)


class T4_ConfigHash(unittest.TestCase):
    def test_key_order_invariant_and_setting_sensitive(self):
        raw = raw_config()
        cdir = os.path.dirname(CONFIG)
        h0 = C.config_hash(C.validate(raw, cdir))
        rev = json.loads(json.dumps(raw), object_pairs_hook=lambda kv: dict(reversed(kv)))
        self.assertNotEqual(list(rev), list(raw))
        self.assertEqual(C.config_hash(C.validate(rev, cdir)), h0)
        changed = copy.deepcopy(raw)
        changed["seeds"]["split_seed"] = 1
        self.assertNotEqual(C.config_hash(C.validate(changed, cdir)), h0)
        changed = copy.deepcopy(raw)
        changed["determinism"]["cudnn_benchmark"] = True
        self.assertNotEqual(C.config_hash(C.validate(changed, cdir)), h0)

    def test_paths_independent_of_cwd(self):
        old = os.getcwd()
        try:
            os.chdir(tempfile.gettempdir())
            a = C.load(CONFIG)
        finally:
            os.chdir(old)
        self.assertEqual(os.path.normcase(a["project_root"]), os.path.normcase(PROJECT))
        self.assertEqual(C.config_hash(a), C.config_hash(C.load(CONFIG)))


class T5_InvalidConfigs(unittest.TestCase):
    def check(self, mutate, fragment):
        raw = raw_config()
        mutate(raw)
        with self.assertRaises(C.ConfigError) as cm:
            C.validate(raw, os.path.dirname(CONFIG))
        self.assertIn(fragment, str(cm.exception))

    def test_cases(self):
        self.check(lambda r: r.update(extra=1), "unknown field(s) ['extra']")
        self.check(lambda r: r.pop("seeds"), "missing required field(s) ['seeds']")
        self.check(lambda r: r.update(run_kind="train"), ".run_kind")
        self.check(lambda r: r.update(schema_version="x"), ".schema_version")
        self.check(lambda r: r["seeds"].update(split_seed="0"), ".seeds.split_seed")
        self.check(lambda r: r["seeds"].update(split_seed=None), ".seeds.split_seed")
        self.check(lambda r: r["seeds"].update(train_seed=True), ".seeds.train_seed")
        self.check(lambda r: r["seeds"].update(epoch_seed=1), ".seeds")
        self.check(lambda r: r["versions"].update(prep_version="prep_v9"), ".versions.prep_version")
        self.check(lambda r: r["versions"].update(threshold_rule="legacy_first_roc_tpr95"),
                   ".versions.threshold_rule")
        self.check(lambda r: r["determinism"].update(warn_only="yes"), ".determinism.warn_only")
        self.check(lambda r: r["manifests"].update(ham10000="research_v1/manifests/v1/nope.csv"),
                   "file not found")
        self.check(lambda r: r["manifests"].update(ham10000="../outside.csv"), "escapes")
        self.check(lambda r: r["manifests"].update(ham10000="C:/abs.csv"), "absolute")


def write_sidecar(path, digest=None, raw=None):
    with open(path + ".sha256", "w", encoding="utf-8") as f:
        f.write(raw if raw is not None else
                f"{digest or P.sha256_file(path)}  {os.path.basename(path)}\n")


class T678_Receipts(unittest.TestCase):
    """Uses a temporary mini-project; real research inputs are never modified."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rv1_test_")
        os.makedirs(os.path.join(self.tmp, "m"))
        os.makedirs(os.path.join(self.tmp, "cfg"))
        self.man = os.path.join(self.tmp, "m", "toy.csv")
        with open(self.man, "w") as f:
            f.write("a,b\n1,2\n")
        self.env = os.path.join(self.tmp, "m", "env.json")
        with open(self.env, "w") as f:
            f.write("{}")
        write_sidecar(self.man)
        write_sidecar(self.env)
        self.src_copy = os.path.join(self.tmp, "legacy_src.py")
        with open(self.src_copy, "w") as f:
            f.write("x = 1\n")
        raw = raw_config()
        raw.update(project_root="..", manifests={"toy": "m/toy.csv"},
                   environment_inventory="m/env.json", legacy_sources=["legacy_src.py"])
        self.cfg_path = os.path.join(self.tmp, "cfg", "c.json")
        with open(self.cfg_path, "w") as f:
            json.dump(raw, f)
        # rv1 sources live outside this temp root, so build a receipt whose
        # source list is limited to the temp root's legacy_src.py
        self.cfg = C.load(self.cfg_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make(self):
        r = P.build_receipt(self.cfg, self.cfg_path, "completed", {"base_seed": 0})
        r["sources"] = {"legacy_src.py": P.sha256_file(self.src_copy)}
        return P.write_receipt(r, os.path.join(self.tmp, "runs", r["run_id"]))

    def test6_hashes_match(self):
        p = self.make()
        self.assertEqual(P.verify_receipt(p, self.tmp), [])
        with open(p) as f:
            r = json.load(f)
        self.assertEqual(r["manifests"]["toy"]["sha256"], P.sha256_file(self.man))
        with open(p, encoding="utf-8") as f:
            self.assertNotIn(P.sha256_file(p), f.read())  # own digest not inside
        self.assertEqual(set(r["runtime"]["env_allowlist"]), set(P.ENV_ALLOWLIST))

    def test7_modification_detected(self):
        p = self.make()
        with open(self.man, "a") as f:
            f.write("3,4\n")
        probs = P.verify_receipt(p, self.tmp)
        self.assertTrue(any("manifest toy" in x for x in probs), probs)
        with open(self.src_copy, "a") as f:
            f.write("y = 2\n")
        self.assertTrue(any("source" in x for x in P.verify_receipt(p, self.tmp)))
        with open(p, "a") as f:
            f.write(" ")
        self.assertIn("receipt file does not match its sidecar digest", P.verify_receipt(p, self.tmp))

    def test8_no_overwrite(self):
        p = self.make()
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        before = P.sha256_file(p)
        with self.assertRaises(FileExistsError):
            P.write_receipt(r, os.path.dirname(p))
        self.assertEqual(P.sha256_file(p), before)


class T10_SidecarsAndConfigConsistency(T678_Receipts):
    """Stage 2B1.1 checks. Fixtures are temporary; the outer receipt sidecar is always
    rewritten to be valid so the targeted internal check is what fails."""

    def rewrite(self, p, mutate):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        mutate(r)
        with open(p, "w", encoding="utf-8") as f:
            f.write(json.dumps(r, indent=2, sort_keys=True))
        write_sidecar(p)  # valid outer sidecar

    def test_valid_sidecars_accepted(self):
        self.assertEqual(P.verify_receipt(self.make(), self.tmp), [])

    def test_bad_input_sidecars_refused_at_build(self):
        nl = chr(10)
        cases = {"missing": None, "empty": "", "malformed": "xyz  toy.csv" + nl,
                 "zeros": "0" * 64 + "  toy.csv" + nl,
                 "wrong_name": P.sha256_file(self.man) + "  other.csv" + nl,
                 "uppercase": P.sha256_file(self.man).upper() + "  toy.csv" + nl}
        for label, raw in cases.items():
            with self.subTest(label):
                if raw is None:
                    os.remove(self.man + ".sha256")
                else:
                    write_sidecar(self.man, raw=raw)
                with self.assertRaises(P.ProvenanceError) as cm:
                    P.build_receipt(self.cfg, self.cfg_path, "completed", {})
                self.assertIn("sidecar", str(cm.exception))
                write_sidecar(self.man)
        write_sidecar(self.env, raw="")
        with self.assertRaises(P.ProvenanceError):
            P.build_receipt(self.cfg, self.cfg_path, "completed", {})

    def test_sidecar_changed_after_receipt(self):
        p = self.make()
        nl = chr(10)
        for raw, frag in (("0" * 64 + "  toy.csv" + nl, "current sidecar does not match"),
                          ("", "sidecar empty"), ("zz  toy.csv" + nl, "sidecar malformed")):
            with self.subTest(frag):
                write_sidecar(self.man, raw=raw)
                probs = P.verify_receipt(p, self.tmp)
                self.assertTrue(any("manifest toy" in x and frag in x for x in probs), probs)
        os.remove(self.man + ".sha256")
        self.assertTrue(any("sidecar missing" in x for x in P.verify_receipt(p, self.tmp)))
        write_sidecar(self.man)
        self.assertEqual(P.verify_receipt(p, self.tmp), [])

    def test_recorded_sidecar_inconsistent(self):
        p = self.make()
        self.rewrite(p, lambda r: r["manifests"]["toy"].update(sidecar_sha256="0" * 64))
        self.assertTrue(any("recorded sidecar digest" in x for x in P.verify_receipt(p, self.tmp)))

    def test_wrong_config_hash(self):
        p = self.make()
        self.rewrite(p, lambda r: r.update(config_sha256="0" * 64))
        self.assertIn("config: embedded configuration does not match config_sha256",
                      P.verify_receipt(p, self.tmp))

    def test_embedded_config_differs_from_file(self):
        p = self.make()

        def mut(r):
            r["config"]["seeds"]["split_seed"] = 99
            r["config_sha256"] = C.config_hash(r["config"])  # internally consistent
        self.rewrite(p, mut)
        probs = P.verify_receipt(p, self.tmp)
        self.assertIn("config: embedded configuration differs from config file", probs)
        self.assertNotIn("config: embedded configuration does not match config_sha256", probs)

    def test_run_kind_purpose_inconsistent(self):
        p = self.make()
        self.rewrite(p, lambda r: r.update(run_kind="lr", purpose="other"))
        probs = P.verify_receipt(p, self.tmp)
        for k in ("run_kind", "purpose"):
            self.assertTrue(any(f"receipt {k}" in x and "embedded" in x for x in probs), probs)
            self.assertTrue(any(f"receipt {k}" in x and "config file" in x for x in probs), probs)


class T9_ImportSideEffects(unittest.TestCase):
    def test_import_has_no_side_effects(self):
        code = f"""
import os, sys, random, json
import numpy as np, torch
sys.path.insert(0, {SRC!r})
before = (random.getstate(), np.random.get_state()[1].tolist(), torch.get_rng_state().tolist(),
          sorted(os.listdir('.')))
import rv1, rv1.config, rv1.seeds, rv1.provenance
after = (random.getstate(), np.random.get_state()[1].tolist(), torch.get_rng_state().tolist(),
         sorted(os.listdir('.')))
print(json.dumps({{"same": before == after, "cuda_init": torch.cuda.is_initialized(),
                   "pyc": sys.dont_write_bytecode}}))
"""
        d = tempfile.mkdtemp(prefix="rv1_import_")
        try:
            r = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True,
                               env=child_env(), cwd=d)
            self.assertEqual(r.returncode, 0, r.stderr)
            out = json.loads(r.stdout.strip().splitlines()[-1])
            self.assertTrue(out["same"])
            self.assertFalse(out["cuda_init"])
            self.assertEqual(os.listdir(d), [])
        finally:
            shutil.rmtree(d, ignore_errors=True)
        self.assertFalse(any(n == "__pycache__" for n in os.listdir(os.path.join(SRC, "rv1"))))


if __name__ == "__main__":
    unittest.main(verbosity=2)
