# Stage 14A Report: Methods Draft and Protocol History / Data Exposure

**Scope.** This stage was writing only. No new experiments, model runs or analyses were performed. No other thesis chapters were written.

## 1. Deliverables (all new files)

| File | Content |
|---|---|
| `docs/STAGE13B_CLARIFICATION_NOTE_v1.md` | Versioned clarification of the legacy-reproduction (LR) interpretation |
| `thesis_materials/METHODS_DRAFT_v1.md` | Methods chapter draft, §3.1–3.13, with [S..] audit tags and [CITATION NEEDED] markers |
| `thesis_materials/PROTOCOL_HISTORY_AND_DATA_EXPOSURE_v1.md` | Study status, the 13-row v0→v1 table with execution status, a dated chronology, pilot and analysis exposure, and precision choices |
| `thesis_materials/METHODS_EVIDENCE_LEDGER.md` | Tags S01–S61 mapped to project files. These are audit references only, not scholarly citations |
| `thesis_materials/UNRESOLVED_ITEMS_14A.md` | Items that could not be established from records (A1–A5, B1–B6, C1–C2) |
| `runs/stage14a/check_methods.py` | Tag/path and fact cross-check |
| `runs/stage14a/integrity_snapshot.py`, `integrity_now.json`, `integrity_comparison.json` | Integrity check |

## 2. Sources
The drafts were built from:
- the protocol and plan (rev2);
- the Stage 1.1 changelog;
- Addenda A–K;
- the stage reports and interpretation/clarification notes;
- frozen specifications, results JSON files, manifests and `env/env_inventory_v1.json`.

All the sources are listed in the ledger.

## 3. LR clarification (`STAGE13B_CLARIFICATION_NOTE_v1.md`)
- File timestamps do not prove that no earlier cache existed.
- Exact aggregate reproduction (30/30) supports the raw-JPEG decoding pathway for the reconstructed evaluation only.
- No claims are made about per-image identity.
- No inference is made about the legacy training pathway.
- The consistency of v1 preprocessing is not undermined. Invariance to other preprocessing is not established.
- The earlier wording "legacy model trained before the cache existed" is withdrawn.

## 4. Verification
`check_methods.py` ran **120 checks with 0 failures**. All 61 tags used in the drafts are defined in the ledger, and every ledger path exists. The fact checks cover:
- manifest counts;
- pos_weight;
- the threshold 0.17569 and epoch 8;
- PAD role counts;
- IG failures 138/996;
- the XAI and curve settings;
- the simulated arm (340 / 256 / 3,740 at m = 2,048);
- the E4.6 cohort (498/198);
- RQ5 Secondary B (n_pos 9 < 20);
- the simulator formulas;
- same-file pairs 9/9;
- output hashes 267/50;
- the Addendum K parameters.

**Corrections made to the drafts after verification**
- Masks are not used in training.
- The leakage checks are split into the RQ3 preflight and the E4.6 feature-dependency tests.
- The GPU is recorded in `env/env_inventory_v1.json` (RTX 5060 Laptop, 8,150 MiB, CUDA 12.8, cuDNN 9.19). It was added to the Methods draft, and item B3 is marked resolved.

## 5. Unresolved items (not invented)
- Ethics approval or exemption (A1).
- Dataset licences (A2).
- Protocol approval status (A3).
- University format and the AI-use declaration (A4).
- Scholarly citations (A5).
- The legacy training pathway (B1).
- Whether an earlier cache existed (B2).
- The overlap of the v0 400-pair subset with the v1 cohort (B4).
- Clinical image provenance (B5).
- v1 sensitivity to alternative preprocessing (B6).
- Two wording decisions for the author (C1–C2).

## 6. Integrity
No snapshot was taken at the start of 14A, so the reference is the Stage 13B after-snapshot (4,881 files). The present snapshot has 4,907 files:
- 0 files changed and 0 removed;
- the cache `.npy` files are unchanged (size and mtime);
- 26 files were added or newly brought into scope:
  - the `runs/lr_13b` and `runs/stage13b` directories, newly in scope;
  - the 13B report, note and appendix files;
  - the new 14A files.

No historical artifact was modified.

## 7. Stop point
Stage 14A is complete. No Results, Discussion or other chapters were written.
