# Demo verification — 2026-10-11

Implemented only in `demo/`; existing research modules and artifacts are reused read-only.

Executed `backend/verify_demo.py` in the existing model environment:
- 16 raw HAM TEST images reproduced their cached 256px entries exactly.
- Batch-16 inference reproduced the saved Stage 4A probabilities: max difference 0.0.
- First image with demo batch 1: absolute probability difference 2.682209014892578e-07.
- HTTP prediction equalled direct service prediction.
- Results API returned all 549 master-table rows.
- Invalid image bytes and transparent PNG were rejected.
- Foreign-origin POST was rejected (403); checkpoint/traversal routes returned 404.
- `node --check frontend/app.js` passed.
- Homepage, stylesheet, JavaScript, status API and all three aggregate chart URLs returned HTTP 200 after restarting the final server.

Limits: no automated browser screenshot or interactive browser inspection was performed.
Only the documented subset was compared against reference predictions. Batch-1
floating-point equivalence to every historical batch is not claimed. This is not
a clinical validation, nor an image-quality or out-of-distribution detector.

Run again with the server open:
`..\research_v1\.venv_model\Scripts\python.exe -B backend\verify_demo.py`

The server intentionally remains a localhost-only demonstration. No image uploads
are saved, no dataset images are bundled and no external CDN is used.
