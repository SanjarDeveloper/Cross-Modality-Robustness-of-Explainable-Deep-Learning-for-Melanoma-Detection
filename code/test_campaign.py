"""Small regression tests; no training or writes to legacy artifacts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import campaign as C
from campaign_checks import validate_outputs


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {"smoke": True, "archs": ["resnet50"],
                       "strats": ["standard"], "seed": 0, "fairness": True}
        self.steps = C.steps(self.config)

    def test_pipeline_dependencies(self):
        seen = set()
        for step in self.steps:
            self.assertTrue(set(step.deps) <= seen)
            seen.add(step.key)
        self.assertEqual(len(self.steps), 9)
        self.assertIn("analyse_resnet50_standard_s0", seen)
        self.assertIn("figures_resnet50_standard_s0", seen)
        self.assertEqual(next(s for s in self.steps if s.kind == "fairness").deps,
                         ["train_resnet50_standard_s0"])

    def test_failed_train_blocks_consumers_and_returns_nonzero(self):
        with patch.object(C, "execute", side_effect=RuntimeError("injected failure")) as run:
            self.assertEqual(C.run_campaign(self.steps, self.config, self.root, "context"), 1)
        self.assertEqual(run.call_count, 1)

    def comparison_attempt(self):
        step = next(s for s in self.steps if s.kind == "compare")
        directory = self.root / "steps" / step.key / "attempt"
        (directory / "results").mkdir(parents=True)
        path = directory / step.outputs[0]
        C.write_json(path, [{"tag": "resnet50_standard_s0", "sens_retention": 0.9,
                            "faith_retention": 0.8, "baseline_auroc_ham": 0.9}])
        spec = {"config": self.config, "context": "A"}
        obj = {"spec": spec, "outputs": {step.outputs[0]: C.digest(path)}}
        C.write_json(directory / "complete.json", obj)
        return step, directory, spec, path

    def test_resume_requires_receipt_config_and_hash(self):
        step, directory, spec, path = self.comparison_attempt()
        self.assertIsNotNone(C.find_complete(self.root, step, spec))
        altered = copy.deepcopy(spec)
        altered["config"]["smoke"] = False
        self.assertIsNone(C.find_complete(self.root, step, altered))
        self.assertIsNone(C.find_complete(self.root, step, dict(spec, context="B")))
        path.write_text("[]", encoding="utf-8")
        self.assertIsNone(C.find_complete(self.root, step, spec))
        (directory / "complete.json").unlink()
        self.assertIsNone(C.find_complete(self.root, step, spec))

    def test_partial_comparison_rejected(self):
        step, directory, spec, path = self.comparison_attempt()
        cfg = dict(self.config, strats=["standard", "heavy_aug"])
        with self.assertRaisesRegex(ValueError, "Partial"):
            validate_outputs(step, directory, cfg)

    def test_zero_exit_without_outputs_fails(self):
        step = self.steps[0]
        with patch.object(C.subprocess, "run") as run:
            run.return_value.returncode = 0
            with self.assertRaisesRegex(ValueError, "Missing/empty"):
                C.execute(step, {"config": self.config}, self.root, {})
        self.assertFalse(list(self.root.rglob("complete.json")))
        self.assertEqual(len(list(self.root.rglob("failure.json"))), 1)

    def test_dependency_receipt_changes_invalidate_spec(self):
        step = self.steps[1]
        completed = {step.deps[0]: {"receipt": {"outputs": {"model": "a"}}}}
        before = C.expected_spec(step, self.config, "same", completed)
        completed[step.deps[0]]["receipt"]["outputs"]["model"] = "b"
        self.assertNotEqual(before, C.expected_spec(step, self.config, "same", completed))

    def test_smoke_sampling_has_two_classes(self):
        import data as D
        recs = [D.Rec("", int(i >= 900), "", "", str(i)) for i in range(1000)]
        idx = D.smoke_indices(recs, 200)
        self.assertEqual(len(idx), 200)
        self.assertEqual(sum(recs[i].label for i in idx), 20)
        self.assertEqual(idx.tolist(), D.smoke_indices(recs, 200).tolist())

    def test_labels_fail_closed_and_preserve_non_melanoma_cancers(self):
        import labels
        for mapper in labels.MAPPERS.values():
            with self.assertRaises(labels.LabelError):
                mapper("completely-unrecognised-diagnosis")
        self.assertEqual(labels.ddi("basal-cell-carcinoma"), labels.NON_MELANOMA)
        self.assertEqual(labels.ddi("melanoma-like-lesion"), labels.NON_MELANOMA)
        self.assertEqual(labels.derm7pt("melanoma metastasis"), labels.MELANOMA)
        self.assertEqual(labels.derm7pt("melanosis"), labels.NON_MELANOMA)

    def test_cache_rejects_different_split(self):
        import data as D
        import numpy as np
        recs = [D.Rec("", 0, "", "", "lesion1")]
        np.save(self.root / "fake_img.npy", np.zeros((1, 2, 2, 3), np.uint8))
        C.write_json(self.root / "fake_meta.json",
                     {"groups": ["lesion2"], "labels": [0], "meta": [{}], "n": 1})
        with self.assertRaisesRegex(ValueError, "does not match"):
            D.validate_cache("fake", recs, self.root)

    def test_threshold_preserves_absolute_not_relative_definition(self):
        from analyse import thresholds
        rows = [{"factor": "blur", "severity": s, "sensitivity": v, "auroc": 0.8}
                for s, v in [(0, 0.8), (1, 0.6)]]
        self.assertAlmostEqual(thresholds(rows)["blur"]["threshold_severity"], 0.5)

    def test_predict_subset_raw_and_cached_alignment(self):
        import data as D
        import run_eval
        import torch
        from torch.utils.data import Dataset
        recs = [D.Rec("", i % 2, "", "", str(i)) for i in range(6)]

        class DummyDataset(Dataset):
            def __init__(self, selected, *args, **kwargs):
                self.recs = selected

            def __len__(self):
                return len(self.recs)

            def __getitem__(self, i):
                return {"x": torch.tensor([float(self.recs[i].group)]),
                        "y": self.recs[i].label}

        with patch.object(D, "cache_exists", return_value=False), \
                patch.object(D, "LesionDataset", DummyDataset):
            y, p = run_eval.predict(torch.nn.Identity(), recs, "cpu", workers=0,
                                    indices=[1, 4], cache="missing")
        self.assertEqual(y.tolist(), [1, 0])
        self.assertEqual(len(p), 2)


if __name__ == "__main__":
    unittest.main()
