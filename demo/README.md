# Derm Shift local demo

Double-click `start_demo.bat`, keep that window open, then open
http://127.0.0.1:8765 in your browser. Stop with Ctrl+C in the server window.
No package installation, network model download or frontend build is needed.
The existing `research_v1/.venv_model` and research artifacts must remain present.

This is a localhost-only research demonstration, not a clinical tool or a public
hosting server. Uploaded JPEG/PNG files are processed in memory and not persisted.
Use images you are allowed to use. Dataset images are not bundled or served.

The selected model is standard ResNet-50 seed 0, epoch 8. Checkpoint and threshold
hashes are verified before inference. Model score is not a calibrated cancer risk.
The Grad-CAM visualization uses the original research implementation; display
normalization is for visualization only. Invalid/constant maps are shown as unavailable.
The demo uses batch 1: tiny floating-point differences from batch-16 saved results
are possible. The "Research results" tab reads the original master table, not live predictions.

The demo shows a single page, **Skin Lesion Classifier: Melanoma vs Non-melanoma**: six example
HAM10000 TEST images (one click) or your own upload. Example 06 is a deliberate false alarm (a common
mole scored 0.94). Output: decision, score bar with the decision line, model input, Grad-CAM heatmap
and, for examples, the true diagnosis.

Hidden but kept in the code (remove `hidden` in `frontend/index.html` to show again):
- the tab bar (`<nav class="tabs">`) with the **How it works** and **Research results** tabs;
- the model-status pill (`#status`).

The model is loaded and hash-verified in the background when the server starts;
the server window prints "Model loaded and verified." when it is ready.

The API is implemented with Python's standard library to avoid changing the frozen
research environment. Do not expose this server on the internet.

## Files
- `frontend/index.html`: page structure and text.
- `frontend/styles.css`: responsive design and colours.
- `frontend/app.js`: tabs, example images, upload, API requests, results filtering.
- `backend/app.py`: HTTP routes and local access checks.
- `backend/model_service.py`: verified model and preprocessing adapter.
- `backend/verify_demo.py`: integration checks (run with the server open).
- `test_images/`: the six example images (HAM10000 TEST, CC BY-NC 4.0); see `LABELS.md`.
- `DEMO_CHEATSHEET_EN.md`: what to say in a 5-minute demo and likely questions.
- `archive_v1/`: the previous frontend/backend, kept unchanged for reference.
- `CODE_GUIDE_UZ.md`: beginner-friendly Uzbek walkthrough.

Alternative port: run `..\research_v1\.venv_model\Scripts\python.exe -B backend\app.py --port 8766` from this folder.
