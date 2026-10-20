"""Stage 2B2A tests: manifest reading, explicit decoding, masks, supporting inputs.

Synthetic temporary fixtures only; no dataset image is opened.
  python -B -m unittest discover -s research_v1/tests -p "test_*.py" -v
"""
import csv
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.dont_write_bytecode = True
sys.path.insert(0, SRC)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from rv1 import config as C, decode as DC, provenance as P  # noqa: E402

COLS = ["record_uid", "dataset", "image_uid", "rel_path", "sha256", "label_mel",
        "cluster_id", "pair_id", "mask_path"]


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rv1_dec_")
        self.ds = os.path.join(self.tmp, "DATASETS")
        os.makedirs(os.path.join(self.ds, "d"))
        # 4x3 image with a known pixel pattern
        self.rgb = np.zeros((3, 4, 3), np.uint8)
        self.rgb[0, 0] = (255, 0, 0)
        self.rgb[2, 3] = (0, 0, 255)
        Image.fromarray(self.rgb, "RGB").save(os.path.join(self.ds, "d", "a.png"))
        Image.fromarray(self.rgb, "RGB").convert("RGBA").save(os.path.join(self.ds, "d", "b.png"))
        m = np.zeros((3, 4), np.uint8)
        m[1, 1] = 255
        Image.fromarray(m, "L").save(os.path.join(self.ds, "d", "a_mask.png"))
        Image.fromarray(np.zeros((5, 5), np.uint8), "L").save(os.path.join(self.ds, "d", "bad_mask.png"))
        with open(os.path.join(self.ds, "d", "corrupt.jpg"), "wb") as f:
            f.write(b"\xff\xd8\xff\xe0 not really a jpeg")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def manifest(self, rows, name="m.csv", dataset="derm7pt"):
        p = os.path.join(self.tmp, name)
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            for r in rows:
                w.writerow({"dataset": dataset, "sha256": "", "pair_id": "", "mask_path": "",
                            "label_mel": "0", "cluster_id": "c", **r})
        return p

    def r(self, uid, rel, **kw):
        return {"record_uid": uid, "image_uid": f"x:{rel}", "rel_path": rel, **kw}


class TestRead(Fixture):
    def test_rgb_and_identity(self):
        p = self.manifest([self.r("r1", "d/a.png", label_mel="1", cluster_id="L1"),
                           self.r("r2", "d/b.png")])
        rows, _ = DC.read_manifest(p, "derm7pt")
        out = list(DC.iter_records(rows, self.ds))
        self.assertEqual([o[0]["record_uid"] for o in out], ["r1", "r2"])  # order kept
        a0, meta0 = out[0][2], out[0][3]
        self.assertEqual(a0.shape, (3, 4, 3))
        self.assertEqual(a0.dtype, np.uint8)
        self.assertTrue((a0 == self.rgb).all())
        self.assertEqual((meta0["width"], meta0["height"], meta0["mode"]), (4, 3, "RGB"))
        self.assertIsNone(meta0["converted_from"])
        self.assertEqual(out[1][3]["converted_from"], "RGBA")  # explicit, recorded
        self.assertTrue((out[1][2] == self.rgb).all())
        self.assertEqual(out[0][0]["cluster_id"], "L1")

    def test_missing_and_corrupt(self):
        p = self.manifest([self.r("r1", "d/nope.png"), self.r("r2", "d/corrupt.jpg")])
        rows, _ = DC.read_manifest(p, "derm7pt")
        out = list(DC.iter_records(rows, self.ds))
        self.assertIsNone(out[0][2])
        self.assertEqual(out[0][3]["error"], "file_missing")
        self.assertIsNone(out[1][2])
        self.assertTrue(out[1][3]["error"].startswith("decode_failed"))

    def test_duplicate_record_rejected(self):
        p = self.manifest([self.r("r1", "d/a.png"), self.r("r1", "d/b.png")])
        with self.assertRaisesRegex(DC.ManifestError, "duplicate record_uid"):
            DC.read_manifest(p, "derm7pt")

    def test_other_manifest_errors(self):
        for rows, frag in (([self.r("r1", "d/a.png", label_mel="2")], "unsupported label_mel"),
                           ([self.r("r1", "../x.png")], "invalid rel_path"),
                           ([self.r("", "d/a.png")], "empty record_uid")):
            with self.subTest(frag):
                with self.assertRaisesRegex(DC.ManifestError, frag):
                    DC.read_manifest(self.manifest(rows), "derm7pt")
        with self.assertRaisesRegex(DC.ManifestError, "dataset"):
            DC.read_manifest(self.manifest([self.r("r1", "d/a.png")]), "ddi")

    def test_paired_references_to_one_file(self):
        rows = [self.r("case9:clinical", "d/a.png", pair_id="case9"),
                self.r("case9:dermoscopy", "d/a.png", pair_id="case9")]
        rows, _ = DC.read_manifest(self.manifest(rows), "derm7pt")
        out = list(DC.iter_records(rows, self.ds))
        self.assertEqual(len(out), 2)
        self.assertEqual([o[0]["record_uid"] for o in out], ["case9:clinical", "case9:dermoscopy"])
        self.assertTrue((out[0][2] == out[1][2]).all())


class TestMasks(Fixture):
    def test_mask_ok(self):
        _, im = DC.decode_image(os.path.join(self.ds, "d", "a.png"))
        arr, mm = DC.decode_mask(os.path.join(self.ds, "d", "a_mask.png"))
        self.assertEqual(mm["mode"], "L")
        self.assertEqual(mm["values"], [0, 255])
        self.assertEqual(DC.mask_flags(mm, im), ([], []))

    def test_mask_missing_and_mismatch(self):
        with self.assertRaisesRegex(DC.DecodeError, "file_missing"):
            DC.decode_mask(os.path.join(self.ds, "d", "none.png"))
        _, im = DC.decode_image(os.path.join(self.ds, "d", "a.png"))
        _, mm = DC.decode_mask(os.path.join(self.ds, "d", "bad_mask.png"))
        crit, warn = DC.mask_flags(mm, im)
        self.assertIn("mask_image_size_mismatch", crit)
        self.assertIn("empty_foreground", warn)


class TestPaletteMask(Fixture):
    def test_palette_mask_not_interpreted(self):
        m = Image.fromarray(np.array([[0, 255], [255, 255]], np.uint8), "P")
        pal = [0] * 768
        pal[0:3] = [255, 255, 255]        # index 0 renders white (as observed in HAM P masks)
        m.putpalette(pal)
        p = os.path.join(self.ds, "d", "pmask.png")
        m.save(p)
        _, mm = DC.decode_mask(p)
        self.assertEqual(mm["mode"], "P")
        self.assertIsNone(mm["foreground_fraction"])
        self.assertEqual(mm["palette_rgb_of_values"][0], (255, 255, 255))
        crit, warn = DC.mask_flags(mm, {"width": 2, "height": 2})
        self.assertEqual(crit, [])
        self.assertIn("unexpected_mode:P", warn)
        self.assertNotIn("all_foreground", warn)


class TestSupportingInputs(unittest.TestCase):
    def test_supporting_input_change_detected(self):
        tmp = tempfile.mkdtemp(prefix="rv1_si_")
        try:
            os.makedirs(os.path.join(tmp, "m"))
            os.makedirs(os.path.join(tmp, "cfg"))
            files = {"m/toy.csv": "a\n1\n", "m/env.json": "{}", "m/card.json": '{"n": 1}'}
            for rel, txt in files.items():
                p = os.path.join(tmp, *rel.split("/"))
                with open(p, "w", encoding="utf-8") as f:
                    f.write(txt)
                if rel != "m/card.json":  # card deliberately has NO sidecar
                    with open(p + ".sha256", "w", encoding="utf-8") as f:
                        f.write(f"{P.sha256_file(p)}  {os.path.basename(p)}\n")
            with open(os.path.join(PROJECT, "research_v1", "configs", "data_audit_v1.json"),
                      encoding="utf-8") as f:
                raw = json.load(f)
            raw.update(project_root="..", manifests={"toy": "m/toy.csv"},
                       environment_inventory="m/env.json",
                       supporting_inputs={"card": "m/card.json"}, legacy_sources=[])
            cp = os.path.join(tmp, "cfg", "c.json")
            with open(cp, "w", encoding="utf-8") as f:
                json.dump(raw, f)
            cfg = C.load(cp)
            r = P.build_receipt(cfg, cp, "completed")
            r["sources"] = {}
            rp = P.write_receipt(r, os.path.join(tmp, "runs", "x"))
            self.assertEqual(P.verify_receipt(rp, tmp), [])
            with open(os.path.join(tmp, "m", "card.json"), "a", encoding="utf-8") as f:
                f.write(" ")
            self.assertIn("supporting_input card: hash mismatch m/card.json",
                          P.verify_receipt(rp, tmp))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_old_configs_still_valid_and_hash_unchanged(self):
        for name, h in (("foundation_check_v1.json", "6d95d9d3"),
                        ("foundation_check_v1_1.json", "cd39cf9c")):
            cfg = C.load(os.path.join(PROJECT, "research_v1", "configs", name))
            self.assertTrue(C.config_hash(cfg).startswith(h), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
