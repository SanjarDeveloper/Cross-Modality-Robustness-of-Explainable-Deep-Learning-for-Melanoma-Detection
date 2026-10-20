"""Read-only adapter around the verified research model. No training occurs here."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research_v1/src"))
import cv2
import numpy as np
import torch
import timm
from PIL import Image
from rv1.preprocess import to_model_input
from rv1.xai_attr import grad_cam, xai_precision, validity

EXPECTED_SHA = "303d21e71986dc8f26dec84c6822a301f696060e5f08f613c90fc33c41a003dd"
THRESHOLD_SHA = "469a4d458276ca3be40193d58a86fc23089a7ab6642b10c348d39684c21f9ffb"
RUN = ROOT / "research_v1/runs/main_train/resnet50_s0_20260927T155924Z"
THRESHOLD_FILE = ROOT / "research_v1/runs/predict_v1/20261001T183557Z_20b3af67_4959f943/threshold.json"


def sha256(path):
    """Hash in blocks so large checkpoints do not require a second memory copy."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def decode_upload(data):
    """Validate dimensions before OpenCV decoding; match the research cache recipe."""
    with Image.open(io.BytesIO(data)) as image:
        if image.format not in ("JPEG", "PNG"):
            raise ValueError("Please upload a JPEG or PNG image.")
        if image.width * image.height > 20_000_000:
            raise ValueError("The image exceeds 20 megapixels. Please upload a smaller copy.")
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Animated images are not supported.")
        if image.getexif().get(274, 1) != 1:
            raise ValueError("Please rotate the image upright and save it without EXIF orientation.")
        if "A" in image.getbands() and image.getchannel("A").getextrema() != (255, 255):
            raise ValueError("Images with a transparent background are not supported.")
        if "transparency" in image.info:
            raise ValueError("Please save the image without transparency.")
        image.verify()
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("The image could not be opened.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cached = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
    return cached


def png_data(rgb):
    """Encode a small display image without writing the upload to disk."""
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


class ModelService:
    def __init__(self):
        self.lock = threading.Lock()
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if sha256(THRESHOLD_FILE) != THRESHOLD_SHA:
            raise RuntimeError("Threshold hash mismatch.")
        self.threshold = json.loads(THRESHOLD_FILE.read_text(encoding="utf-8"))["threshold"]

    def load(self):
        """Only deserialize the local checkpoint after verifying its trusted hash."""
        if self.model is not None:
            return
        checkpoint = RUN / "ckpt_epoch8.pt"
        best = json.loads((RUN / "best.json").read_text(encoding="utf-8"))
        if sha256(checkpoint) != EXPECTED_SHA or best["sha256"] != EXPECTED_SHA:
            raise RuntimeError("Checkpoint hash mismatch.")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = timm.create_model("resnet50", pretrained=False, num_classes=1)
        model.load_state_dict(state["model"], strict=True)
        self.model = model.to(self.device).eval()

    def warm_up(self):
        """Verify and load the checkpoint under the same lock used by predictions."""
        with self.lock:
            self.load()

    def predict(self, data):
        cached = decode_upload(data)
        # Serialize attribution calls: hooks and precision settings are process-wide.
        with self.lock:
            self.load()
            x = torch.from_numpy(to_model_input(cached)).unsqueeze(0).to(self.device)
            attribution = grad_cam(self.model, self.model.layer4, x)
            score = float(torch.sigmoid(attribution["logit"])[0].cpu())
            heat = attribution["map"][0].cpu().numpy()
            valid = bool(np.isfinite(heat).all() and heat.max() - heat.min() >= 1e-12)
            display = cv2.resize(cached, (224, 224), interpolation=cv2.INTER_LINEAR)
            overlay = None
            if valid:
                normal = (heat - heat.min()) / (heat.max() - heat.min())
                colours = cv2.cvtColor(cv2.applyColorMap((normal * 255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
                overlay = png_data(np.clip(display * .6 + colours * .4, 0, 255).astype(np.uint8))
            return {"score": score, "threshold": self.threshold,
                    "positive": score >= self.threshold, "device": self.device,
                    "preview": png_data(display), "overlay": overlay,
                    "cam_available": valid, "checkpoint_sha256": EXPECTED_SHA,
                    "preprocessing": "RGB → 256 INTER_AREA → 224 INTER_LINEAR → ImageNet",
                    "precision": "FP32 / TF32 off; batch=1"}
