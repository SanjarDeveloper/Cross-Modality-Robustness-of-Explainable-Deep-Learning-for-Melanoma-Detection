# Demo verification v2 — 2026-10-19

Supersedes `VERIFICATION.md` (2026-10-11, kept unchanged). The previous code is in `archive_v1/`.
Only files under `demo/` changed; research modules, checkpoints, thresholds and results are read-only.

## Defects found by browser review (as a user) and fixed
| # | Defect | Fix |
|---|---|---|
| 1 | README and CODE_GUIDE described a results page (RQ filters, search, charts); the frontend had none, although the backend served `/api/results` and the charts | Added a "Research results" tab: 4 key-finding cards computed from `MASTER_RESULTS_v2.csv` rows, 4 thesis figures, full 549-row table with RQ filter and search |
| 2 | Generic title "Model demo"; the page did not say what the model does | Header with study title, model and training data; model status pill |
| 3 | Unreadable upload showed a broken-image icon; errors looked like normal text | Preview hidden on decode failure; errors shown in red |
| 4 | Grad-CAM overlay (square 224 input) shown next to an aspect-correct preview, with no legend | Model input and Grad-CAM shown side by side at the same size, colour legend added |
| 5 | Per-image Grad-CAM scaling looked like strong evidence even at score 0.0002 | Note under the legend: colours are relative within each image |
| 6 | Live demo required the file dialog | Five one-click example images (HAM10000 TEST), dataset label shown after prediction |
| 7 | Large preview pushed "Run model" below the fold | Fixed-height preview |
| 8 | First prediction waited for checkpoint loading | Model hash-verified and loaded in a background thread at server start |
| 9 | On phones the result appeared off-screen | Auto-scroll to the output on narrow screens |
| 10 | No score context | Score bar with threshold marker; explanation that non-melanoma is not benign |

## Checks run
- `backend/verify_demo.py` (server open): **PASS**. 16 cache entries exact; batch-16 max difference 0.0; batch-1 difference 2.7e-7; 549 result rows; API equals direct prediction; invalid and transparent images rejected; foreign origin 403; traversal/unknown paths (`/../…`, `/checkpoint.pt`, `/samples/../app.py`, `/samples/06.jpg`, `/charts/../best.json`) 404. **New:** the five example images are byte-identical to the dataset files, are HAM TEST, have the labels shown, and their demo scores match Stage 4A saved probabilities (max difference 8.9e-7).
- `node --check frontend/app.js`: OK.
- Browser (built-in browser pane, desktop and 375 px mobile): example 01 → melanoma 0.2982 (Stage 4A 0.29818); example 05 → non-melanoma 0.0002; example 02 → melanoma 0.9876; invalid file → red error, no broken icon; results tab cards, figures, RQ5 filter (20 rows), search "PAD" (54 rows); no horizontal scroll on mobile; server log has no errors.

## Limits
Not a clinical validation. The example images are hand-picked for the interface, not a performance sample: with the current set all 5 are classified correctly (example 03 only narrowly: 0.127 vs threshold 0.176), whereas HAM TEST specificity is 66.1%. Native file-picker clicks were simulated with a programmatic file; real drag/dialog interaction was not automated.

## Update (same day): simplified for a 5-minute, non-specialist demo
- "Research results" tab hidden (code kept; `hidden` attribute on `#tab-results`), replaced by a plain-language "How it works" tab: five-step diagram, HAM10000 and library links.
- All visible text rewritten in plain English; footer: "For learning only — not a medical tool."
- Example 06 added: ISIC_0024352, HAM TEST, common mole (nv), Stage 4A score 0.94159 → deliberate false alarm; the page shows "AI was wrong: false alarm".
- `verify_demo.py` now checks six samples: **PASS**, max difference vs Stage 4A 8.9e-7. Browser: example 06 → 0.9416, false-alarm label shown; How-it-works tab checked at 800 px and 1280 px.
- The "all 5 correct" limit above no longer applies: with 6 examples, 5 are correct and 1 is a deliberate false alarm.
- The unknown-path check now uses `/samples/07.jpg` (06 is a real example now).

## Update 2: single-page demo
- Title: "Skin Lesion Classifier: Melanoma vs Non-melanoma"; subtitle, eyebrow and footer note removed.
- Tab bar (How it works, Research results) and model-status pill hidden with `hidden` (code kept).
- Small explanatory texts under the examples and under the result removed; card titles "Choose an image" / "Result".
- Browser: example 06 → 0.9416 false alarm shown; tabs and pill `display: none`; 375 px mobile has no horizontal scroll.
