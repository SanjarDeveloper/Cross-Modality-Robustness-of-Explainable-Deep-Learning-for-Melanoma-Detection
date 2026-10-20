"""Small localhost-only HTTP API using Python's standard library.

Routes: GET /api/status, GET /api/results, POST /api/predict.
This is a local research demo, not a production web server.
"""
import csv
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import threading
import traceback
from model_service import ModelService, ROOT, sha256
from PIL import UnidentifiedImageError

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
MASTER = ROOT / "research_v1/thesis_materials/MASTER_RESULTS_v2.csv"
MAX_UPLOAD = 10 * 1024 * 1024
SERVICE = ModelService()
SAMPLES_DIR = Path(__file__).resolve().parents[1] / "test_images"
# HAM10000 TEST images (CC BY-NC 4.0). Chosen for interface demonstration, not at random.
SAMPLES = [
    {"id": "01", "file": "01_ISIC_0024351.jpg", "image": "ISIC_0024351", "label": "Melanoma", "dx": "mel"},
    {"id": "02", "file": "02_ISIC_0024449.jpg", "image": "ISIC_0024449", "label": "Melanoma", "dx": "mel"},
    {"id": "03", "file": "03_ISIC_0024317.jpg", "image": "ISIC_0024317", "label": "Non-melanoma", "dx": "nv"},
    {"id": "04", "file": "04_ISIC_0024357.jpg", "image": "ISIC_0024357", "label": "Non-melanoma", "dx": "nv"},
    {"id": "05", "file": "05_ISIC_0024337.jpg", "image": "ISIC_0024337", "label": "Non-melanoma", "dx": "bkl"},
    # Deliberate false alarm (non-melanoma scored 0.94) so the demo shows a mistake.
    {"id": "06", "file": "06_ISIC_0024352.jpg", "image": "ISIC_0024352", "label": "Non-melanoma", "dx": "nv"},
]
SAMPLE_FILES = {f"/samples/{s['id']}.jpg": s["file"] for s in SAMPLES}


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, payload, kind="application/json; charset=utf-8"):
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8") if isinstance(payload, (dict, list)) else payload
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def trusted(self):
        """Reject foreign Host/Origin headers; uploads stay on this local server."""
        hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        return host in hosts and (origin is None or origin in {"http://" + h for h in hosts})

    def do_GET(self):
        if not self.trusted():
            return self.reply(403, {"error": "Local access only."})
        if self.path == "/api/status":
            return self.reply(200, {"model": "ResNet-50 · standard · seed 0 · epoch 8", "device": SERVICE.device, "loaded": SERVICE.model is not None, "threshold": SERVICE.threshold})
        if self.path == "/api/samples":
            public = [{k: s[k] for k in ("id", "image", "label", "dx")} | {"url": f"/samples/{s['id']}.jpg"} for s in SAMPLES]
            return self.reply(200, {"samples": public, "source": "HAM10000 test split (CC BY-NC 4.0)"})
        if self.path in SAMPLE_FILES:
            return self.reply(200, (SAMPLES_DIR / SAMPLE_FILES[self.path]).read_bytes(), "image/jpeg")
        if self.path == "/api/results":
            with MASTER.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            return self.reply(200, {"rows": rows, "source": "research_v1/thesis_materials/MASTER_RESULTS_v2.csv", "sha256": sha256(MASTER)})
        # An explicit allowlist prevents exposing checkpoints or patient images.
        files = {"/": "index.html", "/styles.css": "styles.css", "/app.js": "app.js"}
        charts = {"/charts/rq2.png": "fig4_1_rq2_sensitivity_change.png",
                  "/charts/rq3.png": "fig4_2_rq3_matched_worst_sens.png",
                  "/charts/rq4.png": "fig4_3_rq4_sim_faithfulness_ig.png",
                  "/charts/rq4-localisation.png": "fig4_4_rq4_sim_localisation.png"}
        if self.path in charts:
            chart = ROOT / "research_v1/thesis_materials/figures" / charts[self.path]
            return self.reply(200, chart.read_bytes(), "image/png")
        name = files.get(self.path)
        if name is None:
            return self.reply(404, {"error": "Not found"})
        path = FRONTEND / name
        return self.reply(200, path.read_bytes(), (mimetypes.guess_type(name)[0] or "text/plain") + "; charset=utf-8")

    def do_POST(self):
        if not self.trusted():
            return self.reply(403, {"error": "Local access only."})
        if self.path != "/api/predict":
            return self.reply(404, {"error": "Not found"})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_UPLOAD:
                return self.reply(413, {"error": "Please upload a non-empty image no larger than 10 MB."})
            self.connection.settimeout(30)
            data = self.rfile.read(size)
            if len(data) != size:
                raise ValueError("The upload was incomplete. Please try again.")
            result = SERVICE.predict(data)
            return self.reply(200, result)
        except UnidentifiedImageError:
            return self.reply(400, {"error": "This file is not a readable image. Please choose a valid JPEG or PNG."})
        except ValueError as error:
            return self.reply(400, {"error": str(error)})
        except OSError:
            return self.reply(400, {"error": "The image could not be read. Please try another JPEG or PNG."})
        except Exception:
            traceback.print_exc()
            return self.reply(500, {"error": "The model could not run. Check the server window for details."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Demo: http://127.0.0.1:{args.port} | Stop: Ctrl+C", flush=True)
    # Load the verified checkpoint in the background so the first click is fast.
    def warm_up():
        try:
            SERVICE.warm_up()
            print("Model loaded and verified.", flush=True)
        except Exception:
            traceback.print_exc()
    threading.Thread(target=warm_up, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
