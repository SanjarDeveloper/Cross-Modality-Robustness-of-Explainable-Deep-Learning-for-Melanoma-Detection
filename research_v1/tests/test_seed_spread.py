"""Stage 6B tests: threshold binding, selected-checkpoint identity, per_seed agreement with the
underlying statistics, identical membership requirement, seed-0 inputs unchanged."""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))
from rv1 import rq1_stats as RS, seed_spread as SS  # noqa: E402


def rows(seed_sha, th_sha, t, data):
    out = []
    for pid, lab, pd_, pc in data:
        for mod, p in (("dermoscopy", pd_), ("clinical", pc)):
            out.append({"record_uid": f"{pid}:{mod}", "pair_id": pid, "modality": mod, "label_mel": str(lab),
                        "prob": repr(p), "pred": str(int(p >= t)), "threshold": repr(t), "threshold_sha256": th_sha,
                        "checkpoint_sha256": seed_sha, "in_derm7pt_primary": "1", "same_file_pair": "0", "status": "ok"})
    return out


class TestBinding(unittest.TestCase):
    def test_threshold_binding(self):
        thr = {"model_sha256": SS.SELECTED[1][2], "threshold": 0.3}
        self.assertEqual(SS.bind_check(thr, SS.SELECTED[1][2]), 0.3)
        with self.assertRaises(ValueError):
            SS.bind_check(thr, SS.SELECTED[2][2])

    def test_selected_identity(self):
        ck, ep, sha = SS.SELECTED[1]
        SS.check_selected(1, {"checkpoint": ck, "epoch": ep, "sha256": sha}, sha)
        with self.assertRaises(RuntimeError):                        # latest instead of selected
            SS.check_selected(1, {"checkpoint": "ckpt_epoch8.pt", "epoch": 8, "sha256": sha}, sha)
        with self.assertRaises(RuntimeError):
            SS.check_selected(2, {"checkpoint": ck, "epoch": ep, "sha256": sha}, sha)


class TestPerSeed(unittest.TestCase):
    def test_summary_agrees_with_underlying(self):
        data = [(f"c{i}", 1, 0.9 if i % 2 else 0.1, 0.8 if i % 3 == 0 else 0.2) for i in range(24)] + \
               [(f"n{i}", 0, 0.2, 0.1) for i in range(10)]
        r = rows("S", "T", 0.5, data)
        res, reps = SS.per_seed(r, 0.5, "T", "S", {"PRIMARY": 7, "ALL": 9})
        pairs = RS.build_pairs(r, 0.5, "T", "S")
        mel = RS.paired_table(list(pairs.values()), 1)
        p = res["PRIMARY"]
        self.assertEqual((p["table"]["dermoscopy_only_b"], p["table"]["clinical_only_c"]), (mel["first_only"], mel["second_only"]))
        self.assertEqual(p["mcnemar_p"], RS.mcnemar_exact(mel["first_only"], mel["second_only"])["p_value"])
        sd, sc, dd = RS.sens_diff(list(pairs.values()))
        self.assertAlmostEqual(p["diff_pp"], 100 * dd)
        self.assertEqual(reps["PRIMARY"].shape, (2000, 3))
        with self.assertRaises(RS.PairError):                       # wrong threshold identity rejected
            SS.per_seed(r, 0.5, "OTHER", "S", {"PRIMARY": 7, "ALL": 9})


class TestSeed0Inputs(unittest.TestCase):
    def test_seed0_artifacts_unchanged(self):
        from rv1 import config as C, provenance as P
        cfgp = os.path.join(HERE, "..", "configs", "seed_spread_6b_v1.json")
        if not os.path.exists(cfgp):
            self.skipTest("config missing")
        cfg = C.load(cfgp)
        known = {"seed0_threshold": "469a4d458276ca3be40193d58a86fc23089a7ab6642b10c348d39684c21f9ffb",
                 "model_checkpoint": SS.SEED0_SHA}
        for k, h in known.items():
            self.assertEqual(P.sha256_file(C.resolve(cfg, cfg["supporting_inputs"][k])), h)


if __name__ == "__main__":
    unittest.main(verbosity=2)
