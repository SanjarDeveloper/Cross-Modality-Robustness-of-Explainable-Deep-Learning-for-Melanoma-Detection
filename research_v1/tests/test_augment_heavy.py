"""Stage 9D tests for aug_heavy_v1 (Addendum K). Frequency tolerances are fixed in advance: |observed - expected|
<= 4 binomial standard deviations over N=2000 seeds."""
import hashlib
import math
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
sys.dont_write_bytecode = True
sys.path.insert(0, SRC)
import numpy as np  # noqa: E402
from rv1 import augment as AU  # noqa: E402

N = 2000


def synth():
    rng = np.random.default_rng(3)
    img = rng.integers(40, 220, (256, 256, 3), dtype=np.uint8)
    m = np.zeros((256, 256), np.uint8)
    m[96:160, 80:176] = 1
    return img, m


def applied(seed):
    p = AU._heavy_pipeline(seed)
    p.save_applied_params = True
    img, _ = synth()
    return [n for n, _ in p(image=img)["applied_transforms"]]


class TestFrequencies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = [applied(s) for s in range(N)]

    def check(self, frac, p):
        sd = math.sqrt(p * (1 - p) / N)
        self.assertLessEqual(abs(frac - p), 4 * sd, (frac, p))

    def test_activation_and_child_selection(self):
        cnt = lambda name: sum(name in r for r in self.runs) / N
        self.check(cnt("RandomResizedCrop"), 1.0 - 1e-12)
        for name, p in (("HorizontalFlip", .5), ("VerticalFlip", .5), ("ShiftScaleRotate", .8), ("ColorJitter", .8),
                        ("RandomBrightnessContrast", .6)):
            self.check(cnt(name), p)
        blur = ("GaussianBlur", "MotionBlur", "Defocus")
        for name in blur:
            self.check(cnt(name), .5 / 3)
        for name in ("GaussNoise", "ImageCompression"):
            self.check(cnt(name), .4 / 2)
        self.assertTrue(all(sum(n in r for n in blur) <= 1 for r in self.runs))           # OneOf: exactly one child
        self.assertTrue(all(sum(n in r for n in ("GaussNoise", "ImageCompression")) <= 1 for r in self.runs))
        self.check(sum(any(n in r for n in blur) for r in self.runs) / N, .5)

    def test_no_shared_rng_streams(self):
        both = sum(("HorizontalFlip" in r) and ("VerticalFlip" in r) for r in self.runs) / N
        self.check(both, .25)                                                              # independent decisions
        b_and_n = sum(any(n in r for n in ("GaussianBlur", "MotionBlur", "Defocus")) and
                      any(n in r for n in ("GaussNoise", "ImageCompression")) for r in self.runs) / N
        self.check(b_and_n, .5 * .4)


class TestDeterminism(unittest.TestCase):
    def out(self, ts=0, ep=0, uid="ham10000:X"):
        img, m = synth()
        return AU.augment(img, "aug_heavy_v1", ts, ep, uid, m)

    def test_same_and_epoch(self):
        a, b = self.out(), self.out()
        self.assertTrue(np.array_equal(a["x"], b["x"]) and np.array_equal(a["mask"], b["mask"]))
        c = self.out(ep=1)
        self.assertFalse(np.array_equal(a["x"], c["x"]))
        self.assertTrue(np.array_equal(c["x"], self.out(ep=1)["x"]))

    def test_order_and_hashseed(self):
        self.out(uid="ham10000:Y")
        a = self.out()
        self.assertTrue(np.array_equal(a["x"], self.out()["x"]))
        code = ("import sys,hashlib;sys.path.insert(0,%r);sys.dont_write_bytecode=True;import warnings;warnings.simplefilter('ignore');"
                "import test_augment_heavy as T;o=T.TestDeterminism().out();print(hashlib.sha256(o['x'].tobytes()+o['mask'].tobytes()).hexdigest())") % SRC
        res = []
        for hs in ("0", "999"):
            env = dict(os.environ, PYTHONHASHSEED=hs, PYTHONPATH=HERE, PYTHONDONTWRITEBYTECODE="1")
            res.append(subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, env=env).stdout.strip())
        self.assertTrue(res[0])
        self.assertEqual(res[0], res[1])


class TestContract(unittest.TestCase):
    def test_shapes_ranges_mask_geometry(self):
        img, m = synth()
        img_c = img.copy()
        for s in range(40):
            o = AU.augment(img, "aug_heavy_v1", 0, s, "ham10000:Z", m)
            self.assertEqual((o["x"].dtype, o["x"].shape), (np.float32, (3, 224, 224)))
            self.assertTrue(np.isfinite(o["x"]).all())
            self.assertEqual(o["mask"].shape, (224, 224))
            self.assertEqual(o["mask"].dtype, bool)
        self.assertTrue(np.array_equal(img, img_c))

    def test_photometric_only_keeps_mask(self):
        """With geometric transforms forced off, the mask equals the plain crop-free resize path is not
        applicable (RandomResizedCrop always runs); instead check mask and image geometry stay aligned:
        a white square on black moves with its mask under any seed."""
        img = np.zeros((256, 256, 3), np.uint8)
        m = np.zeros((256, 256), np.uint8)
        img[100:156, 100:156] = 255
        m[100:156, 100:156] = 1
        agree = []
        for s in range(30):
            o = AU.augment(img, "aug_heavy_v1", 1, s, "ham10000:W", m)
            x = o["x"].transpose(1, 2, 0) * np.asarray(AU.PP.STD) + np.asarray(AU.PP.MEAN)
            bright = x.mean(2) > 0.5
            if o["mask"].sum() > 50:
                agree.append((bright & o["mask"]).sum() / max(bright.sum(), 1))
        self.assertGreater(np.median(agree), 0.7)

    def test_label_record_unchanged_and_empty_flag(self):
        img, m = synth()
        rec = {"record_uid": "ham10000:Q", "image_uid": "ham10000:Q", "label_mel": "1", "split": "train"}
        o = AU.augment_record(rec, img, "aug_heavy_v1", 0, 0, m)
        self.assertEqual(o["record"], rec)
        tiny = np.zeros((256, 256), np.uint8)
        tiny[0, 0] = 1
        flags = [AU.augment(img, "aug_heavy_v1", 0, s, "ham10000:E", tiny)["mask_flags"] for s in range(20)]
        self.assertTrue(any("empty_after_augmentation" in f for f in flags))


if __name__ == "__main__":
    unittest.main(verbosity=2)
