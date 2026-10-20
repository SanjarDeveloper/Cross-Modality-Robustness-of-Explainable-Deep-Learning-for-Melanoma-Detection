"""Stage 2C1 real-data DataLoader check (HAM TRAIN subset) + tiny CUDA smoke test.

  cd research_v1/src
  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.loader_check --config ../configs/loader_check_v1.json --out-root ../runs/loader_check

Subset: 16 HAM-train records ordered by stable_seed("loader_check_v1", record_uid)
(the full ham_train cache mapping is validated first). train mode, aug_std_v1,
train_seed from config, masks on (diagnostic). No model; no performance metric.
"""
import argparse
import hashlib
import json
import os
import time

from . import config as C, decode as DC, provenance as P, torch_data as TD
from .seeds import stable_seed

N_SUBSET, BATCH, SAMPLER_SEED = 16, 5, 20260927


def _epoch(ds, epoch, nw, shuffle=True, seed=SAMPLER_SEED):
    t0 = time.time()
    seq = []
    for b in TD.make_epoch_loader(ds, epoch, BATCH, shuffle, seed, nw):
        for x, y, r, m in zip(b["x"], b["label"], b["record"], b["mask"]):
            h = hashlib.sha256(x.numpy().tobytes())
            if m is not None:
                h.update(m.numpy().tobytes())
            seq.append((r["record_uid"], int(y), int(r["label_mel"]), h.hexdigest()))
    return seq, round(time.time() - t0, 2)


def run(config_path, out_root):
    import torch
    t0 = time.time()
    cfg = C.load(config_path)
    if cfg["purpose"] != "loader_check":
        raise C.ConfigError("loader_check requires purpose 'loader_check'")
    root = cfg["project_root"]
    ds_root, cache_dir = os.path.join(root, "DATASETS"), os.path.join(root, "cache")
    rows = DC.read_manifest(C.resolve(cfg, cfg["manifests"]["ham10000"]), "ham10000")[0]
    train = sorted((r for r in rows if r["split"] == "train"),
                   key=lambda r: (stable_seed("loader_check_v1", r["record_uid"]), r["record_uid"]))
    sub = [r["record_uid"] for r in train[:N_SUBSET]]
    ts = cfg["seeds"]["train_seed"]
    mk = lambda **kw: TD.ManifestDataset(rows, "ham_train", cache_dir, ds_root, sub, want_mask=True, **kw)
    ds = mk(mode="train", train_seed=ts)
    runs, t = {}, {}
    for name, (ep, nw, seed) in {"e0_w0": (0, 0, SAMPLER_SEED), "e0_w2": (0, 2, SAMPLER_SEED),
                                 "e1_w0": (1, 0, SAMPLER_SEED), "e1_w2": (1, 2, SAMPLER_SEED),
                                 "e0_w2_again": (0, 2, SAMPLER_SEED),
                                 "e0_w2_otherseed": (0, 2, SAMPLER_SEED + 1)}.items():
        runs[name], t[name] = _epoch(ds, ep, nw, True, seed)
    ev = {}
    for name, (ep, nw, tseed) in {"eval_e0_w0": (0, 0, None), "eval_e3_w2": (3, 2, None),
                                  "eval_seed7_e1_w2": (1, 2, 7)}.items():
        ev[name], t[name] = _epoch(mk(mode="eval", train_seed=tseed), ep, nw, False)
    per = lambda s: {u: h for u, _, _, h in s}
    order = lambda s: [u for u, *_ in s]
    checks = {
        "same_records_and_labels_w0_w2": sorted((u, y) for u, y, _, _ in runs["e0_w0"])
        == sorted((u, y) for u, y, _, _ in runs["e0_w2"]),
        "labels_attached": all(y == lab for s in runs.values() for _, y, lab, _ in s),
        "per_record_identical_w0_w2_epoch0": per(runs["e0_w0"]) == per(runs["e0_w2"]),
        "per_record_identical_w0_w2_epoch1": per(runs["e1_w0"]) == per(runs["e1_w2"]),
        "ordered_sequence_identical_w0_w2": runs["e0_w0"] == runs["e0_w2"],
        "shuffle_reproducible_same_seed": order(runs["e0_w2"]) == order(runs["e0_w2_again"]),
        "shuffle_changes_with_sampler_seed": order(runs["e0_w2"]) != order(runs["e0_w2_otherseed"]),
        "per_record_independent_of_sampler_seed": per(runs["e0_w2"]) == per(runs["e0_w2_otherseed"]),
        "epoch_reaches_workers_n_records_changed": sum(per(runs["e0_w2"])[u] != per(runs["e1_w2"])[u]
                                                       for u in sub),
        "return_to_epoch0_reproduces": per(runs["e0_w2"]) == per(runs["e0_w2_again"]),
        "no_missing_or_duplicates": all(sorted(order(s)) == sorted(sub) for s in runs.values()),
        "eval_independent_of_seed_epoch_workers": per(ev["eval_e0_w0"]) == per(ev["eval_e3_w2"])
        == per(ev["eval_seed7_e1_w2"]),
    }
    ok = all(v is True for k, v in checks.items() if k != "epoch_reaches_workers_n_records_changed") \
        and checks["epoch_reaches_workers_n_records_changed"] >= N_SUBSET - 1
    cuda = {"available": torch.cuda.is_available()}
    try:
        if cuda["available"]:
            a = torch.arange(1_000_000, dtype=torch.int64)
            g = a.to("cuda")
            cuda.update(device=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                        torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                        op="sum of squares of arange(1e6) int64 on GPU vs CPU",
                        gpu_result=int((g * g).sum().item()), cpu_result=int((a * a).sum().item()))
            torch.cuda.synchronize()
            cuda["match"] = cuda["gpu_result"] == cuda["cpu_result"]
    except Exception as e:  # noqa: BLE001 - recorded, not hidden
        cuda["error"] = f"{type(e).__name__}: {e}"
    result = {"status": "PASS" if ok else "BLOCKED", "subset": sub, "batch_size": BATCH,
              "sampler_seed": SAMPLER_SEED, "train_seed": ts, "persistent_workers": False,
              "checks": checks, "timings_seconds": t, "cuda_smoke": cuda,
              "sequences": {k: v for k, v in {**runs, **ev}.items()},
              "elapsed_seconds": round(time.time() - t0, 1)}
    receipt = P.build_receipt(cfg, config_path, "completed",
                              details={"torch": torch.__version__, "loader_status": result["status"],
                                       "cuda_smoke": cuda})
    out_dir = os.path.join(out_root, receipt["run_id"])
    os.makedirs(out_dir)
    with open(os.path.join(out_dir, "loader_check.json"), "x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=1)
    receipt["outputs"] = {"loader_check.json": P.sha256_file(os.path.join(out_dir, "loader_check.json"))}
    return P.write_receipt(receipt, out_dir), result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()
    rp, r = run(a.config, a.out_root)
    print(rp)
    print(json.dumps({k: r[k] for k in ("status", "checks", "timings_seconds", "cuda_smoke",
                                        "elapsed_seconds")}, indent=1))
    print("verify:", P.verify_receipt(rp, C.load(a.config)["project_root"]) or "OK")
