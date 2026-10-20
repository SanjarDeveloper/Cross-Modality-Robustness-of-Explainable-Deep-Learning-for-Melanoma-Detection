"""Stage 2B2D tests for rv1.augment. Synthetic fixtures only; CPU; no torch.

Run in the single-OpenCV project environment:
  research_v1\\.venv_prep\\Scripts\\python.exe -B -m unittest discover -s research_v1/tests -p "test_augment.py"
"""
import hashlib
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
sys.dont_write_bytecode = True
sys.path.insert(0, SRC)

import numpy as np  # noqa: E402
from rv1 import augment as AU, preprocess as PP  # noqa: E402


def fixture(k):
    rng = np.random.default_rng(1000 + k)
    img = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    m = np.zeros((256, 256), np.uint8)
    y, x = rng.integers(40, 150, 2)
    m[y:y + 60, x:x + 70] = 1
    return img, m


def digest(out):
    h = hashlib.sha256(out["x"].tobytes())
    if out["mask"] is not None:
        h.update(out["mask"].tobytes())
    return h.hexdigest()


class TestDeterminism(unittest.TestCase):
    def test_same_seed_identical(self):
        img, m = fixture(0)
        a = AU.augment(img, "aug_std_v1", 3, 7, "ham10000:X", m)
        b = AU.augment(img, "aug_std_v1", 3, 7, "ham10000:X", m)
        self.assertTrue(np.array_equal(a["x"], b["x"]) and np.array_equal(a["mask"], b["mask"]))
        self.assertEqual(a["aug_seed"], AU.stable_seed(3, 7, "ham10000:X"))

    def test_cross_process_pythonhashseed(self):
        code = (f"import sys; sys.path.insert(0, {SRC!r}); sys.path.insert(0, {HERE!r});"
                "from test_augment import fixture, digest; from rv1 import augment as AU;"
                "print(','.join(digest(AU.augment(*fixture(k)[:1], 'aug_std_v1', 1, e, f'u{k}',"
                " fixture(k)[1])) for k in range(4) for e in range(2)))")
        outs = []
        for hs in ("0", "999"):
            env = dict(os.environ, PYTHONHASHSEED=hs, PYTHONDONTWRITEBYTECODE="1")
            r = subprocess.run([sys.executable, "-B", "-W", "ignore", "-c", code],
                               capture_output=True, text=True, env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            outs.append(r.stdout.strip())
        self.assertEqual(outs[0], outs[1])
        local = ",".join(digest(AU.augment(fixture(k)[0], "aug_std_v1", 1, e, f"u{k}", fixture(k)[1]))
                         for k in range(4) for e in range(2))
        self.assertEqual(outs[0], local)

    def test_access_order_independent(self):
        keys = [(k, e) for k in range(5) for e in range(2)]
        f = {k: fixture(k) for k in range(5)}
        run = lambda order: {ke: digest(AU.augment(f[ke[0]][0], "aug_std_v1", 2, ke[1],
                                                   f"u{ke[0]}", f[ke[0]][1])) for ke in order}
        self.assertEqual(run(keys), run(list(reversed(keys))))

    def test_variation_across_seeds(self):
        diff = sum(digest(AU.augment(*fixture(k)[:1], "aug_std_v1", 0, 0, f"u{k}")) !=
                   digest(AU.augment(*fixture(k)[:1], "aug_std_v1", 1, 0, f"u{k}"))
                   for k in range(20))
        self.assertGreaterEqual(diff, 18)          # batch-level variation, not every draw

    def test_transform_decisions_not_coupled(self):
        """Regression for Albumentations' identical per-child seeding (see augment.py)."""
        combos = set()
        for s in range(200):
            p = AU._std_pipeline(s)
            p.save_applied_params = True
            combos.add(tuple(sorted(t[0] for t in p(image=fixture(0)[0])["applied_transforms"])))
        self.assertGreaterEqual(len(combos), 14)   # 16 possible; coupled seeding gave 3


class TestGeometry(unittest.TestCase):
    def test_image_mask_alignment(self):
        img = np.zeros((256, 256, 3), np.uint8)
        m = np.zeros((256, 256), np.uint8)
        img[80:170, 60:190] = 255
        m[80:170, 60:190] = 1
        n = 0
        for s in range(40):
            o = AU.augment(img, "aug_std_v1", s, 0, "sq", m)
            if not o["mask"].any():
                continue
            x = o["x"].transpose(1, 2, 0) * np.array(PP.STD) + np.array(PP.MEAN)
            bright = x.mean(2) > 0.5 * x.mean(2).max()
            iou = (bright & o["mask"]).sum() / (bright | o["mask"]).sum()
            self.assertGreaterEqual(iou, 0.9, s)
            n += 1
        self.assertGreater(n, 30)

    def test_photometric_leaves_mask_unchanged(self):
        import albumentations as A
        img, m = fixture(3)
        changed = 0
        for s in range(30):
            seed = AU.stable_seed(5, 1, f"k{s}")
            full = AU._std_pipeline(seed)
            geo = A.Compose(AU._std_pipeline(seed).transforms[:4], strict=True)  # fresh objects
            AU._seed_pipeline(geo, seed)           # same per-index sub-seeds as `full`
            a, b = full(image=img, mask=m), geo(image=img, mask=m)
            self.assertTrue(np.array_equal(a["mask"], b["mask"]))
            changed += not np.array_equal(a["image"], b["image"])
        self.assertGreater(changed, 0)             # ColorJitter did act on some images

    def test_mask_binary_and_empty_flag(self):
        img = np.zeros((256, 256, 3), np.uint8)
        m = np.zeros((256, 256), np.uint8)
        m[0:6, 0:6] = 1                            # tiny corner lesion
        empties = flagged = 0
        for s in range(60):
            o = AU.augment(img, "aug_std_v1", s, 0, "corner", m, mask_flags=("x",))
            self.assertEqual(o["mask"].dtype, bool)
            self.assertEqual(o["mask_flags"][0], "x")          # existing flags carried
            empties += not o["mask"].any()
            flagged += "empty_after_augmentation" in o["mask_flags"]
        self.assertGreater(empties, 0)
        self.assertEqual(empties, flagged)
        with self.assertRaises(ValueError):
            AU.augment(img, "aug_std_v1", 0, 0, "u", m * 255)   # non-{0,1} mask rejected


class TestContract(unittest.TestCase):
    def test_record_label_unchanged(self):
        img, m = fixture(1)
        rec = {"record_uid": "ham10000:A", "image_uid": "ham10000:A", "label_mel": "1",
               "split": "train", "rq5_role": "", "cluster_id": "L"}
        before = dict(rec)
        o = AU.augment_record(rec, img, "aug_std_v1", 0, 3, m)
        self.assertEqual(rec, before)
        self.assertEqual(o["record"], before)
        self.assertIsNot(o["record"], rec)
        self.assertEqual((o["train_seed"], o["epoch"], o["recipe"]), (0, 3, "aug_std_v1"))

    def test_aug_none_equals_evaluation(self):
        from rv1 import decode  # noqa: F401  (import check only)
        img, _ = fixture(2)
        les = np.zeros((450, 600), bool)
        les[100:300, 150:420] = True
        m256 = AU.mask_to_cache_grid(les)
        for ts, ep in ((0, 0), (7, 9)):
            o = AU.augment(img, "aug_none", ts, ep, "u", m256)
            self.assertTrue(np.array_equal(o["x"], PP.to_model_input(img)))
            self.assertTrue(np.array_equal(o["mask"], PP.mask_to_input_grid(les)))
            self.assertIsNone(o["aug_seed"])

    def test_evaluation_unaffected_and_input_not_mutated(self):
        img, m = fixture(4)
        ref, img0 = PP.to_model_input(img), img.copy()
        for s in range(5):
            AU.augment(img, "aug_std_v1", s, s, "u", m)
        self.assertTrue(np.array_equal(img, img0))
        self.assertTrue(np.array_equal(PP.to_model_input(img), ref))

    def test_deferred_and_unknown(self):
        img, _ = fixture(0)
        # aug_heavy_v1 was DEFERRED until Stage 9D (Addendum K); it now requires seeds like aug_std_v1
        with self.assertRaises(ValueError):
            AU.augment(img, "aug_heavy_v1", None, 0, "u")
        with self.assertRaises(ValueError):
            AU.augment(img, "aug_other", 0, 0, "u")
        with self.assertRaises(ValueError):
            AU.augment(img, "aug_std_v1", None, 0, "u")


if __name__ == "__main__":
    unittest.main(verbosity=2)
