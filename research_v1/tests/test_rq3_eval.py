"""Stage 9C1 tests: nine-cell coverage, best-checkpoint identity, threshold binding, reuse verification."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))
from rv1 import rq3_eval as E  # noqa: E402


class TestRegistry(unittest.TestCase):
    def test_nine_cells(self):
        m = E.models()
        self.assertEqual(len(m), 9)
        self.assertEqual({(x["arch"], x["seed"]) for x in m},
                         {(a, s) for a in ("resnet50", "efficientnet_b3", "densenet121") for s in (0, 1, 2)})
        r0 = next(x for x in m if x["id"] == "resnet50_s0")
        self.assertEqual(set(r0["reuse"]), {"ham_test", "derm7pt", "pad_all"})
        r1 = next(x for x in m if x["id"] == "resnet50_s1")
        self.assertEqual(set(r1["reuse"]), {"derm7pt"})                     # HAM TEST / PAD not assumed complete

    def test_selected_identity(self):
        best = {"sha256": "X", "epoch": 5}
        ck = {"epoch_completed": 5, "config": {"training": {"arch": "densenet121"}, "seeds": {"train_seed": 1}}}
        self.assertTrue(E.check_selected(best, "X", ck, "densenet121", 1))
        with self.assertRaises(RuntimeError):
            E.check_selected(best, "Y", ck, "densenet121", 1)               # latest/other file
        with self.assertRaises(RuntimeError):
            E.check_selected(dict(best, epoch=8), "X", ck, "densenet121", 1)
        with self.assertRaises(RuntimeError):
            E.check_selected(best, "X", ck, "efficientnet_b3", 1)
        with self.assertRaises(RuntimeError):
            E.check_selected(best, "X", ck, "densenet121", 2)

    def test_threshold_binding(self):
        self.assertEqual(E.threshold_value({"model_sha256": "A", "status": "ESTIMATED", "threshold": 0.1}, "A"), 0.1)
        self.assertEqual(E.threshold_value({"model": {"sha256": "A"}, "status": "ESTIMATED", "threshold": 0.2}, "A"), 0.2)
        with self.assertRaises(ValueError):
            E.threshold_value({"model_sha256": "A", "status": "ESTIMATED", "threshold": 0.1}, "B")
        with self.assertRaises(ValueError):
            E.threshold_value({"model_sha256": "A", "status": "NOT_ESTIMABLE", "threshold": None}, "A")

    def test_reused_artifacts_exist_and_hash(self):
        from rv1 import provenance as P
        root = os.path.abspath(os.path.join(HERE, "..", ".."))
        known = {E.P4A + "predictions_ham_test.csv": None, E.P4A + "threshold.json":
                 "469a4d458276ca3be40193d58a86fc23089a7ab6642b10c348d39684c21f9ffb"}
        for rel, h in known.items():
            p = os.path.join(root, *rel.split("/"))
            self.assertTrue(os.path.exists(p))
            if h:
                self.assertEqual(P.sha256_file(p), h)


if __name__ == "__main__":
    unittest.main(verbosity=2)
