"""Stage 2C1 tests for rv1.torch_data (Dataset/DataLoader, workers, epochs). CPU only.

Needs torch: run in research_v1/.venv_model. Skipped (reported as skipped) elsewhere.
Synthetic cache fixture; the real HAM TRAIN check is rv1.loader_check.
"""
import hashlib
import json
import os
import pickle
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))

import numpy as np  # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

N = 11                                   # not divisible by the batch size (4)


def build_fixture(d):
    rng = np.random.default_rng(7)
    np.save(os.path.join(d, "c_img.npy"), rng.integers(0, 256, (N, 256, 256, 3), dtype=np.uint8))
    labels = [int(i % 3 == 0) for i in range(N)]
    with open(os.path.join(d, "c_meta.json"), "w") as f:
        json.dump({"n": N, "size": 256, "labels": labels, "groups": [f"g{i}" for i in range(N)],
                   "has_mask": False, "mask_ok": [], "meta": [{}] * N}, f)
    return [{"record_uid": f"ds:r{i}", "image_uid": f"ds:r{i}", "cache_name": "c",
             "cache_index": str(i), "label_mel": str(labels[i]), "cluster_id": f"g{i}",
             "split": "train"} for i in range(N)]


def run_epoch(ds, epoch, nw, shuffle=True, seed=123):
    from rv1 import torch_data as TD
    seq = []
    for b in TD.make_epoch_loader(ds, epoch, 4, shuffle, seed, nw):
        for x, y, r in zip(b["x"], b["label"], b["record"]):
            seq.append((r["record_uid"], int(y), int(r["label_mel"]),
                        hashlib.sha256(x.numpy().tobytes()).hexdigest()))
    return seq


@unittest.skipUnless(HAVE_TORCH, "torch not installed in this interpreter")
class TestTorchData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rv1 import torch_data as TD
        cls.d = tempfile.mkdtemp(prefix="rv1_td_")
        cls.rows = build_fixture(cls.d)
        cls.TD = TD

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, ignore_errors=True)

    def ds(self, **kw):
        kw.setdefault("mode", "train")
        kw.setdefault("train_seed", 0)
        return self.TD.ManifestDataset(self.rows, "c", self.d, self.d, **kw)

    def test_workers_0_vs_2_identical_sequence_and_records(self):
        ds = self.ds()
        a, b = run_epoch(ds, 0, 0), run_epoch(ds, 0, 2)
        self.assertEqual(a, b)                                    # order AND per-record tensors
        self.assertEqual(sorted(u for u, *_ in a), sorted(r["record_uid"] for r in self.rows))
        self.assertTrue(all(y == lab for _, y, lab, _ in a))      # label stays with its record

    def test_no_missing_or_duplicate(self):
        for nw in (0, 2):
            seq = run_epoch(self.ds(), 1, nw)
            uids = [u for u, *_ in seq]
            self.assertEqual(len(uids), N)
            self.assertEqual(len(set(uids)), N)

    def test_shuffle_reproducible_and_seeded(self):
        ds = self.ds()
        o1 = [u for u, *_ in run_epoch(ds, 0, 0, seed=5)]
        o2 = [u for u, *_ in run_epoch(ds, 0, 2, seed=5)]
        o3 = [u for u, *_ in run_epoch(ds, 0, 0, seed=6)]
        self.assertEqual(o1, o2)
        self.assertNotEqual(o1, o3)

    def test_epoch_reaches_workers_and_is_reversible(self):
        ds = self.ds()
        e0 = {u: h for u, _, _, h in run_epoch(ds, 0, 2)}
        e1_w = {u: h for u, _, _, h in run_epoch(ds, 1, 2)}
        e1_m = {u: h for u, _, _, h in run_epoch(ds, 1, 0)}
        self.assertEqual(e1_w, e1_m)                              # workers saw epoch 1
        self.assertGreaterEqual(sum(e0[u] != e1_w[u] for u in e0), N - 1)
        self.assertEqual({u: h for u, _, _, h in run_epoch(ds, 0, 2)}, e0)   # back to epoch 0
        # per-record content does not depend on the sampler seed
        self.assertEqual({u: h for u, _, _, h in run_epoch(ds, 0, 2, seed=999)}, e0)

    def test_eval_independent_of_seed_and_epoch(self):
        a = {u: h for u, _, _, h in run_epoch(self.ds(mode="eval", train_seed=None), 0, 0, False)}
        b = {u: h for u, _, _, h in run_epoch(self.ds(mode="eval", train_seed=None), 5, 2, False)}
        c = {u: h for u, _, _, h in run_epoch(self.ds(mode="eval", train_seed=3), 2, 2, True)}
        self.assertEqual(a, b)
        self.assertEqual(a, c)
        from rv1 import preprocess as PP
        img = np.load(os.path.join(self.d, "c_img.npy"), mmap_mode="r")
        self.assertEqual(a["ds:r4"], hashlib.sha256(PP.to_model_input(np.asarray(img[4])).tobytes()).hexdigest())

    def test_subset_and_pickling(self):
        sub = ["ds:r7", "ds:r2", "ds:r9"]
        ds = self.ds(record_uids=sub)
        self.assertEqual([r["record_uid"] for r in ds.rows], sub)
        _ = ds[0]                                                 # opens the memory map
        self.assertIsNotNone(ds._img)
        clone = pickle.loads(pickle.dumps(ds))
        self.assertIsNone(clone._img)
        self.assertLess(len(pickle.dumps(ds)), 20000)             # no array in the pickle
        self.assertTrue(torch.equal(clone[1]["x"], ds[1]["x"]))
        self.assertEqual(sorted(u for u, *_ in run_epoch(ds, 0, 2)), sorted(sub))
        with self.assertRaisesRegex(ValueError, "not in cache"):
            self.ds(record_uids=["ds:nope"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.ds(record_uids=["ds:r1", "ds:r1"])

    def test_item_types(self):
        it = self.ds()[3]
        self.assertEqual((it["x"].dtype, tuple(it["x"].shape)), (torch.float32, (3, 224, 224)))
        self.assertEqual(it["label"].dtype, torch.int64)
        self.assertIsNone(it["mask"])
        with self.assertRaises(ValueError):
            self.ds(train_seed=None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
