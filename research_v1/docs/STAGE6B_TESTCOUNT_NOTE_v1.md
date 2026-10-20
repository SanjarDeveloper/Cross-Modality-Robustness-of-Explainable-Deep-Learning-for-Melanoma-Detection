# Stage 6B Test-Count Correction Note v1 — 2026-10-03

`STAGE6B_REPORT.md` (frozen, unchanged) states that `tests/test_seed_spread.py` contains **5** tests. That is **incorrect**: executing the file discovers **4** tests.

| Class | Tests |
|---|---|
| `TestBinding` | `test_threshold_binding`, `test_selected_identity` |
| `TestPerSeed` | `test_summary_agrees_with_underlying` |
| `TestSeed0Inputs` | `test_seed0_artifacts_unchanged` |

**Where the error came from:** a repository search found no fifth seed-spread test anywhere else. The miscount came from the report text. The "14 tests" run in Stage 6B is consistent with this: 4 seed-spread tests plus 10 `rq1_stats` tests.

**Effect on results:** none.
