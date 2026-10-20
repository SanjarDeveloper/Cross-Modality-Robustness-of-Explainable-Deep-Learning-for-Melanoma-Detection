"""Stage 11A tests: outcome definition, leakage guard, deterministic case-grouped stratified folds with sufficiency,
and that the frozen feature set contains no forbidden predictors."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))
from rv1 import e46_design as E  # noqa: E402


class TestOutcome(unittest.TestCase):
    def test_error(self):
        t = 0.1757
        self.assertEqual(E.outcome("1", 0.10, t), 1)        # FN
        self.assertEqual(E.outcome("1", 0.1757, t), 0)      # at threshold -> melanoma
        self.assertEqual(E.outcome("0", 0.50, t), 1)        # FP
        self.assertEqual(E.outcome("0", 0.05, t), 0)


class TestLeakage(unittest.TestCase):
    def test_frozen_features_clean(self):
        self.assertTrue(E.leakage_check(list(E.FEATURES)))

    def test_forbidden_detected(self):
        for bad in ("label_mel", "error_flag", "dermoscopy_prob", "mismatch_delta", "mask_iou", "D_shift_ins"):
            with self.assertRaises(ValueError):
                E.leakage_check(["prob_basis", bad])


class TestFolds(unittest.TestCase):
    def setUp(self):
        self.cases = [f"c{i}" for i in range(200)]
        self.err = {c: int(i % 3 == 0) for i, c in enumerate(self.cases)}
        self.strata = {c: f"{i % 2}|{self.err[c]}" for i, c in enumerate(self.cases)}

    def test_deterministic_grouped_balanced(self):
        f = E.assign_folds(self.cases, self.strata)
        self.assertEqual(f, E.assign_folds(list(reversed(self.cases)), self.strata))   # order-independent
        self.assertEqual(set(f), set(self.cases))                                      # every case in exactly one fold
        ok, per = E.check_folds(f, self.err)
        self.assertTrue(ok)
        self.assertLessEqual(max(v["n"] for v in per.values()) - min(v["n"] for v in per.values()), 1)
        self.assertLessEqual(max(v["errors"] for v in per.values()) - min(v["errors"] for v in per.values()), 2)

    def test_duplicate_case_rejected(self):
        with self.assertRaises(ValueError):
            E.assign_folds(["a", "a"], {"a": "x"})

    def test_insufficient_detected(self):
        err = {c: 0 for c in self.cases}
        self.assertFalse(E.check_folds(E.assign_folds(self.cases, self.strata), err)[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
