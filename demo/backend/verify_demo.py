"""Integration checks against real, read-only research artifacts. No training."""
import csv
import json
import io
import urllib.request
import urllib.error
import numpy as np
import torch
from PIL import Image
from model_service import ROOT, ModelService, decode_upload, to_model_input, xai_precision


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    service = ModelService()
    predictions = read_csv(ROOT / "research_v1/runs/predict_v1/20261001T183557Z_20b3af67_4959f943/predictions_ham_test.csv")[:16]
    manifest = {r["record_uid"]: r for r in read_csv(ROOT / "research_v1/manifests/v1/ham10000_v1.csv")}
    cache = np.load(ROOT / "cache/ham_test_img.npy", mmap_mode="r")
    inputs = []
    for prediction in predictions:
        record = manifest[prediction["record_uid"]]
        data = (ROOT / "DATASETS" / record["rel_path"]).read_bytes()
        decoded = decode_upload(data)
        assert np.array_equal(decoded, cache[int(record["cache_index"])]), "Cache reconstruction mismatch"
        inputs.append(to_model_input(decoded))
    service.load()
    with xai_precision(), torch.no_grad():
        probs = torch.sigmoid(service.model(torch.from_numpy(np.stack(inputs)).to(service.device))).cpu().numpy().ravel()
    reference = np.array([float(p["prob"]) for p in predictions])
    batch_difference = float(np.max(np.abs(probs-reference)))
    assert batch_difference <= 1e-5, batch_difference
    first = manifest[predictions[0]["record_uid"]]
    data = (ROOT / "DATASETS" / first["rel_path"]).read_bytes()
    result = service.predict(data)
    assert abs(result["score"] - reference[0]) <= 1e-4
    assert result["positive"] == (result["score"] >= service.threshold)
    for bad in (b"not an image", b""):
        try:
            decode_upload(bad)
            raise AssertionError("Invalid image accepted")
        except (ValueError, OSError):
            pass
    transparent = io.BytesIO()
    Image.new("RGBA", (10, 10), (0, 0, 0, 0)).save(transparent, format="PNG")
    try:
        decode_upload(transparent.getvalue())
        raise AssertionError("Transparent image accepted")
    except ValueError:
        pass
    base = "http://127.0.0.1:8765"
    with urllib.request.urlopen(base + "/api/results") as response:
        results = json.load(response)
    assert len(results["rows"]) == 549
    with urllib.request.urlopen(urllib.request.Request(base + "/api/predict", data=data, headers={"Content-Type":"application/octet-stream"})) as response:
        api = json.load(response)
    assert api["score"] == result["score"]
    # Example images: served exactly, and their demo scores match saved Stage 4A predictions.
    all_predictions = {p["record_uid"]: float(p["prob"]) for p in read_csv(ROOT / "research_v1/runs/predict_v1/20261001T183557Z_20b3af67_4959f943/predictions_ham_test.csv")}
    with urllib.request.urlopen(base + "/api/samples") as response:
        samples = json.load(response)["samples"]
    assert len(samples) == 6
    sample_max_difference = 0.0
    for sample in samples:
        with urllib.request.urlopen(base + sample["url"]) as response:
            image = response.read()
        record = manifest["ham10000:" + sample["image"]]
        assert image == (ROOT / "DATASETS" / record["rel_path"]).read_bytes(), "Sample bytes differ from dataset file"
        assert (sample["label"] == "Melanoma") == (record["label_mel"] == "1") and sample["dx"] == record["dx_raw"]
        assert record["split"] == "test"
        score = service.predict(image)["score"]
        sample_max_difference = max(sample_max_difference, abs(score - all_predictions[record["record_uid"]]))
    assert sample_max_difference <= 1e-4, sample_max_difference
    for path in ("/../research_v1/configs", "/checkpoint.pt", "/samples/../app.py", "/samples/07.jpg", "/charts/../best.json"):
        try:
            urllib.request.urlopen(base + path)
            raise AssertionError("Unexpected file exposure")
        except urllib.error.HTTPError as error:
            assert error.code == 404
    try:
        urllib.request.urlopen(urllib.request.Request(base + "/api/predict", data=b"bad", headers={"Origin":"https://example.com"}))
        raise AssertionError("Foreign origin accepted")
    except urllib.error.HTTPError as error:
        assert error.code == 403
    print(json.dumps({"status":"PASS", "cache_entries_exact":16, "batch16_max_difference":batch_difference,
                      "batch1_difference":abs(result["score"]-reference[0]), "results_rows":len(results["rows"]),
                      "api_matches_direct":True, "invalid_images_rejected":True,
                      "samples_checked":len(samples), "sample_max_difference_vs_stage4a":sample_max_difference}, indent=2))


if __name__ == "__main__":
    main()
