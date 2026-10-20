"""Stage 4A tests: v1 threshold rule."""
import json
import os
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from rv1 import threshold as TH  # noqa: E402


class TestThreshold(unittest.TestCase):
    def test_required_positives_exact(self):
        self.assertEqual(TH.required_positives(160), 152)      # 0.95*160 = 152 exactly (float ceil risk)
        self.assertEqual(TH.required_positives(20), 19)
        self.assertEqual(TH.required_positives(21), 20)         # 19.95 -> 20
        self.assertEqual(TH.required_positives(100), 95)
        self.assertEqual([TH.required_positives(n) for n in range(20, 400)],
                         [-(-95 * n // 100) for n in range(20, 400)])

    def test_160_positive_case(self):
        pos = [1.0 - i / 1000 for i in range(160)]             # descending distinct
        neg = [0.01 * i / 10 for i in range(100)]
        r = TH.estimate(pos + neg, [1] * 160 + [0] * 100)
        self.assertEqual(r["k"], 152)
        self.assertEqual(r["threshold"], pos[151])
        self.assertEqual(r["TP"], 152)
        self.assertGreaterEqual(r["sensitivity"], 0.95)

    def test_highest_admissible(self):
        pos = [0.9 - i * 0.01 for i in range(20)]
        r = TH.estimate(pos + [0.0], [1] * 20 + [0])
        t = r["threshold"]
        self.assertGreaterEqual(TH.counts(pos, [1] * 20, t)["sensitivity"], 0.95)
        nxt = r["next_higher_pos_score"]                       # any higher threshold fails the target
        self.assertLess(TH.counts(pos, [1] * 20, nxt)["sensitivity"], 0.95)

    def test_ties(self):
        pos = [0.9] * 5 + [0.5] * 15 + [0.1] * 5               # k = 24 -> 0.1 ... make tie at k
        r = TH.estimate(pos, [1] * 25)
        self.assertEqual(r["k"], 24)
        self.assertEqual(r["threshold"], 0.1)
        self.assertEqual(r["ties_at_threshold_pos"], 5)
        self.assertEqual(r["TP"], 25)                          # ties at t classified positive

    def test_insufficient_and_nonfinite(self):
        self.assertEqual(TH.estimate([0.5] * 19, [1] * 19)["status"], "NOT_ESTIMABLE")
        self.assertEqual(TH.estimate([0.5] * 19 + [float("nan")], [1] * 20)["status"], "NOT_ESTIMABLE")
        self.assertEqual(TH.estimate([0.5] * 19 + [float("inf")], [1] * 20)["status"], "NOT_ESTIMABLE")
        self.assertIsNone(TH.estimate([0.5] * 10, [1] * 10)["threshold"])

    def test_boundary_and_serialization(self):
        t = 0.1 + 0.2                                          # 0.30000000000000004
        self.assertEqual(TH.predict(t, t), 1)
        self.assertEqual(TH.predict(0.3, t), 0)               # rounding to 0.3 would flip this
        back = json.loads(json.dumps({"t": t}))["t"]
        self.assertEqual(back, t)
        self.assertEqual(float(repr(t)), t)


if __name__ == "__main__":
    unittest.main(verbosity=2)
