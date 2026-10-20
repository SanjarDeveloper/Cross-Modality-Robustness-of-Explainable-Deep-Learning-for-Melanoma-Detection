"""Stage 2B2B tests for decode.interpret_mask (mask_interp_v1). Synthetic masks only."""
import hashlib
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from rv1 import decode as DC  # noqa: E402

TRUTH = np.zeros((6, 8), bool)
TRUTH[1:4, 2:6] = True            # known spatial lesion


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class TestInterpretMask(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="rv1_mask_")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def save_L(self, arr, name="l.png"):
        p = os.path.join(self.d, name)
        Image.fromarray(arr.astype(np.uint8), "L").save(p)
        return p

    def save_P(self, idx, colours, name="p.png", transparency=None):
        """idx: index array; colours: {index: (r,g,b)}"""
        im = Image.fromarray(idx.astype(np.uint8), "P")
        pal = [7] * 768                      # unused entries: a non-B/W colour on purpose
        for i, rgb in colours.items():
            pal[3 * i:3 * i + 3] = list(rgb)
        im.putpalette(pal)
        p = os.path.join(self.d, name)
        if transparency is not None:
            im.save(p, transparency=transparency)
        else:
            im.save(p)
        return p

    def test_L_black_white(self):
        les, m = DC.interpret_mask(self.save_L(TRUTH * 255))
        self.assertTrue((les == TRUTH).all())
        self.assertEqual((m["original_mode"], m["rendered_values"], m["flags"]), ("L", [0, 255], []))
        self.assertEqual(m["version"], "mask_interp_v1")

    def test_reversed_palette_P(self):
        # HAM-like: index 0 -> white (lesion), index 255 -> black
        idx = np.where(TRUTH, 0, 255)
        les, m = DC.interpret_mask(self.save_P(idx, {0: (255, 255, 255), 255: (0, 0, 0)}))
        self.assertTrue((les == TRUTH).all())
        self.assertEqual(m["palette_map"], {0: [255, 255, 255], 255: [0, 0, 0]})

    def test_ordinary_palette_P(self):
        idx = np.where(TRUTH, 1, 0)
        les, _ = DC.interpret_mask(self.save_P(idx, {0: (0, 0, 0), 1: (255, 255, 255)}))
        self.assertTrue((les == TRUTH).all())

    def test_equivalent_L_and_P_identical(self):
        a, _ = DC.interpret_mask(self.save_L(TRUTH * 255))
        b, _ = DC.interpret_mask(self.save_P(np.where(TRUTH, 0, 255),
                                             {0: (255, 255, 255), 255: (0, 0, 0)}))
        c, _ = DC.interpret_mask(self.save_P(np.where(TRUTH, 1, 0), {0: (0, 0, 0), 1: (255, 255, 255)},
                                             name="p2.png"))
        self.assertTrue(np.array_equal(a, b) and np.array_equal(a, c))
        self.assertEqual(a.dtype, bool)

    def test_unsupported_rejected(self):
        grey = TRUTH * 255
        grey[0, 0] = 128
        with self.assertRaisesRegex(DC.MaskInterpretationError, "unsupported_rendered_values"):
            DC.interpret_mask(self.save_L(grey))
        with self.assertRaisesRegex(DC.MaskInterpretationError, "unsupported_palette_colour"):
            DC.interpret_mask(self.save_P(np.where(TRUTH, 1, 0), {0: (0, 0, 0), 1: (255, 0, 0)}))
        with self.assertRaisesRegex(DC.MaskInterpretationError, "transparency_present"):
            DC.interpret_mask(self.save_P(np.where(TRUTH, 1, 0), {0: (0, 0, 0), 1: (255, 255, 255)},
                                          transparency=0))
        rgb = os.path.join(self.d, "rgb.png")
        Image.fromarray(np.zeros((6, 8, 3), np.uint8), "RGB").save(rgb)
        with self.assertRaisesRegex(DC.MaskInterpretationError, "unsupported_mode:RGB"):
            DC.interpret_mask(rgb)
        la = os.path.join(self.d, "la.png")
        Image.fromarray(np.zeros((6, 8, 2), np.uint8), "LA").save(la)
        with self.assertRaisesRegex(DC.MaskInterpretationError, "unsupported_mode:LA"):
            DC.interpret_mask(la)

    def test_L_transparency_rejected(self):
        p = os.path.join(self.d, "lt.png")
        Image.fromarray((TRUTH * 255).astype(np.uint8), "L").save(p, transparency=0)
        with Image.open(p) as im:                                # fixture really carries tRNS
            self.assertIn("transparency", im.info)
        with self.assertRaisesRegex(DC.MaskInterpretationError, "transparency_present"):
            DC.interpret_mask(p)
        _, raw = DC.decode_mask(p)                               # raw inspection unaffected
        self.assertEqual(raw["values"], [0, 255])

    def test_empty_and_all_white_flagged(self):
        _, m = DC.interpret_mask(self.save_L(np.zeros((6, 8)), "e.png"))
        self.assertEqual(m["flags"], ["empty_foreground"])
        _, m = DC.interpret_mask(self.save_L(np.full((6, 8), 255), "w.png"))
        self.assertEqual(m["flags"], ["all_foreground"])
        _, m = DC.interpret_mask(self.save_P(np.zeros((6, 8)), {0: (255, 255, 255)}, "pw.png"))
        self.assertEqual(m["flags"], ["all_foreground"])

    def test_originals_unchanged_and_raw_inspection_kept(self):
        paths = [self.save_L(TRUTH * 255),
                 self.save_P(np.where(TRUTH, 0, 255), {0: (255, 255, 255), 255: (0, 0, 0)})]
        before = [sha(p) for p in paths]
        for p in paths:
            DC.interpret_mask(p)
        self.assertEqual([sha(p) for p in paths], before)
        _, raw = DC.decode_mask(paths[1])          # raw behaviour unchanged
        self.assertEqual(raw["mode"], "P")
        self.assertIsNone(raw["foreground_fraction"])
        self.assertEqual(raw["values"], [0, 255])


if __name__ == "__main__":
    unittest.main(verbosity=2)
