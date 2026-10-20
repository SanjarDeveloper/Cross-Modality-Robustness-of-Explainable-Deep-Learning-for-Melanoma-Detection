"""Stage 10B.1 tests: gate-1 assessment of criteria (a)/(b) with the frozen xai_pairs thresholds, and gating logic
(a pooled pass with any per-condition failure stays NOT_READY)."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))
from rv1 import xai_sim_ext as XE  # noqa: E402


def recs(n, fails, e_rel=0.01):
    return {f"r{i}": {"completeness_failure": i < fails, "e_rel": (0.5 if i < fails else e_rel)} for i in range(n)}


class TestGate1(unittest.TestCase):
    def test_criteria_a_b(self):
        at = recs(20, 1)
        self.assertEqual(XE.assess_ab(at, list(at))["a"]["status"], "pass")          # 1/20 = 5% passes
        at = recs(20, 2)
        A = XE.assess_ab(at, list(at))
        self.assertEqual((A["a"]["status"], A["overall"]), ("fail", "fail"))         # 2/20 = 10% fails
        at = {f"r{i}": {"completeness_failure": False, "e_rel": 0.3} for i in range(20)}
        self.assertEqual(XE.assess_ab(at, list(at))["b"]["status"], "fail")          # p95 e_rel > 0.20
        at = recs(20, 0)
        self.assertEqual(XE.assess_ab(at, list(at))["overall"], "pass")

    def test_missing_counts_as_failure(self):
        at = recs(20, 0)
        keys = list(at) + ["r_missing_1", "r_missing_2"]
        self.assertEqual(XE.assess_ab(at, keys)["a"]["failures"], 2)

    def test_pooled_pass_with_condition_fail_is_not_ready(self):
        pooled = {"overall": "pass"}
        per = {f"c{k}": {"overall": "pass"} for k in range(10)}
        self.assertEqual(XE.gate1_decision(pooled, per)["status"], "READY")
        per["c3"] = {"overall": "fail"}
        d = XE.gate1_decision(pooled, per)
        self.assertEqual(d["status"], "NOT_READY")
        self.assertEqual(d["conditions_not_pass"], {"c3": "fail"})
        per["c3"] = {"overall": "UNRESOLVED"}
        self.assertEqual(XE.gate1_decision(pooled, per)["status"], "NOT_READY")

    def test_realistic_pooled_pass_condition_fail(self):
        """220 refs, 6 failures pooled (2.7% pass), but 2 in one condition of 20 (10%) -> NOT_READY."""
        pool = recs(220, 6)
        cond = {k: pool[k] for k in list(pool)[:20]}                                 # r0..r19 contains 6 failures
        P = XE.assess_ab(pool, list(pool))
        C = XE.assess_ab(cond, list(cond))
        self.assertEqual((P["overall"], C["overall"]), ("pass", "fail"))
        self.assertEqual(XE.gate1_decision(P, {"c": C})["status"], "NOT_READY")


if __name__ == "__main__":
    unittest.main(verbosity=2)
