"""Stage 8A tests for rv1.perturb (perturb_msim_v1)."""
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
from rv1 import perturb as PT  # noqa: E402


def synth(seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((256, 256, 3), (200, 160, 140), np.uint8)
    yy, xx = np.mgrid[:256, :256]
    disk = (yy - 128) ** 2 + (xx - 128) ** 2 < 40 ** 2
    img[disk] = (90, 60, 50)
    img = np.clip(img.astype(int) + rng.integers(-8, 9, img.shape), 0, 255).astype(np.uint8)
    return img, disk.astype(np.uint8)


def digest(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


class TestIdentityAndValidity(unittest.TestCase):
    def test_severity_zero_identity(self):
        img, m = synth()
        from rv1 import preprocess as PP
        for f in PT.FACTORS:
            out, mo, meta = PT.apply(img, f, 0.0, "uid:a", mask256=m)
            self.assertTrue(np.array_equal(out, img), f)
            self.assertTrue(np.array_equal(mo, m), f)
            self.assertTrue(np.array_equal(PT.to_model_input(out), PP.to_model_input(img)), f)
        self.assertTrue(PT.apply(img, "compression", 0.0, "u")[2]["params"]["bypass"])

    def test_shapes_ranges_and_no_mutation(self):
        img, m = synth()
        img_c, m_c = img.copy(), m.copy()
        state = np.random.get_state()[1].copy()
        for f in PT.FACTORS:
            for s in PT.SEVERITY_GRID:
                out, mo, _ = PT.apply(img, f, s, "uid:b", mask256=m)
                self.assertEqual((out.dtype, out.shape), (np.uint8, (256, 256, 3)))
                x = PT.to_model_input(out)
                self.assertTrue(np.isfinite(x).all())
                self.assertEqual(set(np.unique(mo).tolist()) <= {0, 1}, True)
        self.assertTrue(np.array_equal(img, img_c) and np.array_equal(m, m_c))
        self.assertTrue(np.array_equal(np.random.get_state()[1], state))     # global RNG untouched

    def test_invalid_inputs(self):
        img, m = synth()
        for bad in (-0.1, 1.1, float("nan"), True, "0.5"):
            with self.assertRaises(PT.PerturbError):
                PT.apply(img, "blur", bad, "u")
        with self.assertRaises(PT.PerturbError):
            PT.apply(img, "fog", 0.5, "u")
        with self.assertRaises(PT.PerturbError):
            PT.apply(img[:224, :224], "blur", 0.5, "u")
        with self.assertRaises(PT.PerturbError):
            PT.apply(img, "scale", 0.5, "u", mask256=m * 255)
        with self.assertRaises(PT.PerturbError):
            PT.apply(img, "scale", 0.5, "u", mask256=m[:224, :224])


class TestDeterminism(unittest.TestCase):
    def test_order_independence_and_seed_dependence(self):
        img, _ = synth()
        a = PT.apply(img, "nuisance", 0.5, "uid:x")[0]
        PT.apply(img, "nuisance", 0.5, "uid:y")                              # different record in between
        b = PT.apply(img, "nuisance", 0.5, "uid:x")[0]
        self.assertTrue(np.array_equal(a, b))
        c = PT.apply(img, "nuisance", 0.5, "uid:y")[0]
        self.assertFalse(np.array_equal(a, c))

    def test_cross_process_hashseed(self):
        code = ("import sys,hashlib,numpy as np; sys.path.insert(0,%r); sys.dont_write_bytecode=True;"
                "from rv1 import perturb as PT; import test_perturb as T; img,m=T.synth();"
                "print(''.join(T.digest(PT.apply(img,f,0.75,'uid:p',mask256=m)[0]) for f in PT.FACTORS))") % SRC
        outs = []
        for hs in ("0", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=hs, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=HERE)
            outs.append(subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, env=env).stdout)
        self.assertTrue(outs[0])
        self.assertEqual(outs[0], outs[1])

    def test_same_realization_across_severity(self):
        img, _ = synth()
        m1 = PT.apply(img, "nuisance", 0.25, "uid:q")[2]["params"]
        m2 = PT.apply(img, "nuisance", 1.0, "uid:q")[2]["params"]
        self.assertEqual((m1["glare_center"], m1["glare_angle"]), (m2["glare_center"], m2["glare_angle"]))
        s1 = PT.apply(img, "illumination", 0.125, "uid:q")[2]["params"]["sign"]
        s2 = PT.apply(img, "illumination", 1.0, "uid:q")[2]["params"]["sign"]
        self.assertEqual(s1, s2)
        g1 = PT.apply(img, "scale", 0.25, "uid:q")[2]["params"]["offset_dir"]
        g2 = PT.apply(img, "scale", 1.0, "uid:q")[2]["params"]["offset_dir"]
        self.assertEqual(g1, g2)


class TestKnownBehaviour(unittest.TestCase):
    def test_blur_parameters_and_sharpness(self):
        import cv2
        img, _ = synth()
        lap = [cv2.Laplacian(cv2.cvtColor(PT.apply(img, "blur", s, "u")[0], cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
               for s in (0, 0.25, 0.5, 1.0)]
        self.assertTrue(all(b < a for a, b in zip(lap, lap[1:])))
        self.assertEqual(PT.apply(img, "blur", 1.0, "u")[2]["params"]["ksize"], 43)    # 2*round(21)+1
        self.assertTrue(PT.apply(img, "blur", 0.04, "u")[2]["params"]["identity"])     # r=0.28 < 0.3

    def test_illumination_formula(self):
        img = np.full((256, 256, 3), 128, np.uint8)
        out, _, meta = PT.apply(img, "illumination", 1.0, "uid:i")
        p = meta["params"]
        x = (128 / 255 - 0.5) * p["contrast"] + 0.5 + p["brightness"]
        exp = [x * (1 + p["temp"]), x, x * (1 - p["temp"])]
        for ch in range(3):
            self.assertEqual(int(out[0, 0, ch]), int(np.clip(np.float32(exp[ch]) * 255, 0, 255)))

    def test_jpeg_quality_progression(self):
        img, _ = synth()
        qs = [PT.apply(img, "compression", s, "u")[2]["params"]["quality"] for s in (0.125, 0.5, 1.0)]
        self.assertEqual(qs, [85, 56, 18])
        sizes = [PT.apply(img, "compression", s, "u")[2]["params"]["encoded_bytes"] for s in (0.125, 0.5, 1.0)]
        self.assertTrue(sizes[0] > sizes[1] > sizes[2])


class TestMaskGeometry(unittest.TestCase):
    def test_scale_alignment_centroid_area(self):
        img, m = synth()
        out, mo, meta = PT.apply(img, "scale", 1.0, "uid:s", mask256=m)
        g = meta["params"]
        self.assertTrue(meta["geometry_changed"])
        # area: inset of size nh x nw (unclipped; disk well inside frame) -> ratio ~ (nh*nw)/(256*256)
        ratio = mo.sum() / m.sum()
        self.assertAlmostEqual(ratio, g["nh"] * g["nw"] / 256 ** 2, delta=0.03)
        # centroid moves with the inset geometry
        ys, xs = np.nonzero(mo)
        cy_exp = g["y0"] + (127.5 + 0.5) * g["nh"] / 256 - 0.5
        cx_exp = g["x0"] + (127.5 + 0.5) * g["nw"] / 256 - 0.5
        self.assertAlmostEqual(ys.mean(), cy_exp, delta=1.0)
        self.assertAlmostEqual(xs.mean(), cx_exp, delta=1.0)
        # lesion (dark) pixels sit under the transformed mask
        dark = out.mean(axis=2) < 110
        self.assertGreater((dark & (mo == 1)).sum() / max(dark.sum(), 1), 0.9)

    def test_photometric_mask_invariance(self):
        img, m = synth()
        for f in PT.PHOTOMETRIC:
            self.assertTrue(np.array_equal(PT.apply(img, f, 1.0, "u", mask256=m)[1], m))

    def test_clipped_object_area_not_claimed(self):
        img = np.full((256, 256, 3), 180, np.uint8)
        m = np.zeros((256, 256), np.uint8)
        m[:, :128] = 1                                                      # touches the frame edge
        out, mo, meta = PT.apply(img, "scale", 1.0, "uid:c", mask256=m)
        g = meta["params"]
        self.assertTrue(mo[:, :g["x0"]].sum() == 0)                          # zero padding outside the inset
        self.assertEqual(int(mo.sum()), int(PT.__dict__["np"].count_nonzero(mo)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
