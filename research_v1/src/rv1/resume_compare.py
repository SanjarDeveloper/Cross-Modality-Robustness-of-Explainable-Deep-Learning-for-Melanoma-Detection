"""Stage 2C2: compare uninterrupted (U) and interrupted/resumed (R) smoke arms.

  ..\\.venv_model\\Scripts\\python.exe -B -m rv1.resume_compare --u <U dir> --r <R dir> --out <json>

Pre-set tolerances (not changed after seeing results): max probe-probability
difference <= 1e-3; probe AUROC difference <= 1e-3; identical record order,
augmentation-input digests, learning rates, optimizer-step counts; parameters and
buffers are compared and every difference is reported.
"""
import argparse
import json
import os

TOL_PROB, TOL_AUROC = 1e-3, 1e-3


def _load(d, name):
    import torch
    return torch.load(os.path.join(d, name), map_location="cpu", weights_only=False)


def _jsonl(d, name):
    with open(os.path.join(d, name), encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _tensor_diff(a, b):
    out = {}
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            out[k] = "missing"
            continue
        x, y = a[k], b[k]
        if x.dtype.is_floating_point:
            out[k] = float((x.double() - y.double()).abs().max()) if x.numel() else 0.0
        else:
            out[k] = 0.0 if bool((x == y).all()) else float((x.double() - y.double()).abs().max())
    return out


def _opt_diff(a, b):
    import torch
    m = 0.0
    for sid in a["state"]:
        for k, v in a["state"][sid].items():
            w = b["state"][sid][k]
            if isinstance(v, torch.Tensor):
                m = max(m, float((v.double() - w.double()).abs().max()))
    return {"max_abs_state_diff": m, "param_groups_equal": a["param_groups"] == b["param_groups"]}


def compare(u, r):
    from .trainer import auroc
    from .trainer import canonical_steps
    spe = json.load(open(os.path.join(u, "run_start.json"), encoding="utf-8"))["steps_per_epoch"]
    su, sr = canonical_steps(u, spe), canonical_steps(r, spe)      # committed attempts only
    key = lambda s: (s["epoch"], s["global_step"], s["record_uids"], s["x_digest"], s["lr_used"])
    cu1, cr1 = _load(u, "ckpt_epoch1.pt"), _load(r, "ckpt_epoch1.pt")
    cu, cr = _load(u, "ckpt_epoch2.pt"), _load(r, "ckpt_epoch2.pt")
    ev = _jsonl(r, "resume_events.jsonl")
    pd = [abs(a - b) for a, b in zip(cu["probe"]["probs"], cr["probe"]["probs"])]
    au, ar = auroc(cu["probe"]["labels"], cu["probe"]["probs"]), auroc(cr["probe"]["labels"], cr["probe"]["probs"])
    td = _tensor_diff(cu["model"], cr["model"])
    res = {
        "steps_identical_order_digest_lr": [key(s) for s in su] == [key(s) for s in sr],
        "n_steps": [len(su), len(sr)],
        "loss_max_abs_diff": max(abs(a["loss"] - b["loss"]) for a, b in zip(su, sr)),
        "epoch1_checkpoint": {"model_max_abs_diff": max(_tensor_diff(cu1["model"], cr1["model"]).values()),
                              "optimizer": _opt_diff(cu1["optimizer"], cr1["optimizer"]),
                              "scheduler_equal": cu1["scheduler"] == cr1["scheduler"],
                              "rng_digest_equal": cu1["rng_digest"] == cr1["rng_digest"],
                              "global_step": [cu1["global_step"], cr1["global_step"]]},
        "resume_restored_rng_digest_equals_saved": ev[0]["rng_digest_restored"] == cr1["rng_digest"],
        "final": {"global_step": [cu["global_step"], cr["global_step"]],
                  "scheduler_equal": cu["scheduler"] == cr["scheduler"],
                  "final_lr": [cu["scheduler"]["_last_lr"], cr["scheduler"]["_last_lr"]],
                  "optimizer": _opt_diff(cu["optimizer"], cr["optimizer"]),
                  "rng_digest_equal": cu["rng_digest"] == cr["rng_digest"],
                  "params_buffers_max_abs_diff": max(v for v in td.values() if v != "missing"),
                  "n_tensors": len(td),
                  "n_tensors_nonzero_diff": sum(1 for v in td.values() if v != 0.0),
                  "missing_tensors": [k for k, v in td.items() if v == "missing"],
                  "probe_prob_max_abs_diff": max(pd), "probe_auroc": [au, ar],
                  "probe_auroc_abs_diff": abs(au - ar),
                  "probe_uids_equal": cu["probe"]["record_uids"] == cr["probe"]["record_uids"]},
        "fingerprints_equal": cu["fingerprint"]["digest"] == cr["fingerprint"]["digest"],
        "warnings": {"U": _jsonl(u, "warnings.jsonl"), "R": _jsonl(r, "warnings.jsonl")},
        "tolerances": {"probe_prob": TOL_PROB, "probe_auroc": TOL_AUROC},
    }
    f = res["final"]
    res["status"] = "PASS" if (res["steps_identical_order_digest_lr"] and f["global_step"][0] == f["global_step"][1]
                               and f["final_lr"][0] == f["final_lr"][1] and f["scheduler_equal"]
                               and res["resume_restored_rng_digest_equals_saved"] and not f["missing_tensors"]
                               and f["probe_prob_max_abs_diff"] <= TOL_PROB
                               and f["probe_auroc_abs_diff"] <= TOL_AUROC and f["probe_uids_equal"]
                               and res["fingerprints_equal"]) else "FAIL"
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--u", required=True)
    ap.add_argument("--r", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = compare(a.u, a.r)
    with open(a.out, "x", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "warnings"}, indent=1))
    print("warnings:", json.dumps(res["warnings"])[:1500])
