"""Stage 2B2C tests for rv1.preprocess (prep_v1_cached256). Synthetic fixtures only.

Must run in the project environment with exactly one OpenCV distribution:
  research_v1\\.venv_prep\\Scripts\\python.exe -B -m unittest discover -s research_v1/tests -p "test_preprocess.py"
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from rv1 import preprocess as PP  # noqa: E402

MEAN, STD = np.array(PP.MEAN, np.float32), np.array(PP.STD, np.float32)


def make_cache(d, name, imgs, labels, groups):
    np.save(os.path.join(d, f"{name}_img.npy"), np.stack(imgs).astype(np.uint8))
    with open(os.path.join(d, f"{name}_meta.json"), "w") as f:
        json.dump({"n": len(imgs), "size": 256, "labels": labels, "groups": groups,
                   "has_mask": False, "mask_ok": [], "meta": [{}] * len(imgs)}, f)


def row(uid, idx, label=0, group="g", cache="c", **kw):
    return {"record_uid": uid, "cache_name": cache, "cache_index": str(idx),
            "label_mel": str(label), "cluster_id": group, **kw}


class Base(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="rv1_prep_")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)


class TestImagePath(Base):
    def test_channel_order_and_normalisation_exact(self):
        img = np.zeros((256, 256, 3), np.uint8)
        img[..., 0] = 255                       # pure red in RGB order
        x = PP.to_model_input(img)
        self.assertEqual((x.dtype, x.shape, x.flags["C_CONTIGUOUS"]), (np.float32, (3, 224, 224), True))
        want = (np.array([1.0, 0.0, 0.0], np.float32) - MEAN) / STD
        for c in range(3):
            self.assertTrue(np.all(x[c] == want[c]), c)   # constant image: exact equality
        grey = np.full((256, 256, 3), 128, np.uint8)
        xg = PP.to_model_input(grey)
        v = np.float32(128) / np.float32(255.0)
        for c in range(3):
            self.assertTrue(np.all(xg[c] == (v - MEAN[c]) / STD[c]))

    def test_rebuild_channel_order_from_file(self):
        for colour, ch in (((255, 0, 0), 0), ((0, 255, 0), 1), ((0, 0, 255), 2)):
            p = os.path.join(self.d, f"c{ch}.png")
            Image.new("RGB", (600, 450), colour).save(p)
            c = PP.rebuild_cache_entry(p)
            self.assertEqual(c.shape, (256, 256, 3))
            self.assertTrue(np.all(c[..., ch] == 255) and np.all(np.delete(c, ch, 2) == 0))

    def test_bilinear_resize_is_explicit_cv2_linear(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        x = PP.to_model_input(img)
        ref = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
        self.assertTrue(np.array_equal(x, ((ref - MEAN) / STD).transpose(2, 0, 1)))

    def test_rejects_wrong_input(self):
        with self.assertRaises(PP.CacheError):
            PP.to_model_input(np.zeros((224, 224, 3), np.uint8))
        with self.assertRaises(PP.CacheError):
            PP.to_model_input(np.zeros((256, 256, 3), np.float32))


class TestMaskPath(Base):
    def test_binary_and_alignment(self):
        les = np.zeros((450, 600), bool)
        les[90:180, 300:420] = True             # known block
        m = PP.mask_to_input_grid(les)
        self.assertEqual((m.dtype, m.shape), (bool, (224, 224)))
        # independent derivation of the documented mapping: nearest-exact takes the source
        # pixel under each destination pixel centre, applied 450x600 -> 256 -> 224
        def centre_idx(n_dst, n_src):
            return np.minimum(np.floor((np.arange(n_dst) + 0.5) * n_src / n_dst).astype(int),
                              n_src - 1)
        ry = centre_idx(256, 450)[centre_idx(224, 256)]
        rx = centre_idx(256, 600)[centre_idx(224, 256)]
        self.assertTrue(np.array_equal(m, les[np.ix_(ry, rx)]))
        # and within 1 output pixel of the continuous full-frame geometry
        ys, xs = np.nonzero(m)
        for got, want in ((ys.min(), 90 * 224 / 450), (ys.max() + 1, 180 * 224 / 450),
                          (xs.min(), 300 * 224 / 600), (xs.max() + 1, 420 * 224 / 600)):
            self.assertLessEqual(abs(got - want), 1.0)

    def test_equivalent_L_P_remain_equivalent(self):
        from rv1 import decode as DC
        truth = np.zeros((450, 600), bool)
        truth[100:300, 200:450] = True
        pl = os.path.join(self.d, "l.png")
        Image.fromarray(truth.astype(np.uint8) * 255, "L").save(pl)
        pp = os.path.join(self.d, "p.png")
        im = Image.fromarray(np.where(truth, 0, 255).astype(np.uint8), "P")
        pal = [0] * 768
        pal[0:3] = [255, 255, 255]
        im.putpalette(pal)
        im.save(pp)
        a = PP.mask_to_input_grid(DC.interpret_mask(pl)[0])
        b = PP.mask_to_input_grid(DC.interpret_mask(pp)[0])
        self.assertTrue(np.array_equal(a, b))


class TestReader(Base):
    def test_repeat_reads_identical_and_metadata_kept(self):
        rng = np.random.default_rng(1)
        imgs = [rng.integers(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(3)]
        make_cache(self.d, "c", imgs, [0, 1, 0], ["a", "b", "c"])
        rows = [row("r2", 2, 0, "c", image_uid="u2"), row("r0", 0, 0, "a", image_uid="u0"),
                row("r1", 1, 1, "b", image_uid="u1", pair_id="p1")]
        rd = PP.EvalReader(rows, "c", self.d)
        a, b = rd[2], rd[2]
        self.assertTrue(np.array_equal(a["x"], b["x"]))
        self.assertEqual(a["record"]["record_uid"], "r1")
        self.assertEqual((a["record"]["image_uid"], a["record"]["pair_id"]), ("u1", "p1"))
        self.assertEqual(a["prep_version"], "prep_v1_cached256")
        self.assertTrue(np.array_equal(rd[0]["x"], PP.to_model_input(imgs[2])))

    def test_index_and_metadata_mismatch_fail(self):
        make_cache(self.d, "c", [np.zeros((256, 256, 3), np.uint8)] * 2, [0, 1], ["a", "b"])
        cases = [([row("r0", 0, 0, "a"), row("r1", 2, 1, "b")], "out of range"),
                 ([row("r0", 0, 0, "a"), row("r1", 1, 0, "b")], "label mismatch"),
                 ([row("r0", 0, 0, "a"), row("r1", 1, 1, "zz")], "group mismatch"),
                 ([row("r0", 0, 0, "a"), row("r1", 0, 0, "a")], "exactly once"),
                 ([row("r0", 0, 0, "a")], "exactly once"),
                 ([row("r0", "x", 0, "a"), row("r1", 1, 1, "b")], "invalid cache_index")]
        for rows, frag in cases:
            with self.subTest(frag):
                with self.assertRaisesRegex(PP.CacheError, frag):
                    PP.EvalReader(rows, "c", self.d)

    def test_same_file_pairs_both_kept(self):
        same = np.full((256, 256, 3), 77, np.uint8)
        make_cache(self.d, "clin", [same], [1], ["case9"])
        make_cache(self.d, "derm", [same], [1], ["case9"])
        rows = [row("case9:clinical", 0, 1, "case9", cache="clin", pair_id="case9"),
                row("case9:dermoscopy", 0, 1, "case9", cache="derm", pair_id="case9")]
        c = PP.EvalReader(rows, "clin", self.d)[0]
        d = PP.EvalReader(rows, "derm", self.d)[0]
        self.assertEqual((c["record"]["record_uid"], d["record"]["record_uid"]),
                         ("case9:clinical", "case9:dermoscopy"))
        self.assertTrue(np.array_equal(c["x"], d["x"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
