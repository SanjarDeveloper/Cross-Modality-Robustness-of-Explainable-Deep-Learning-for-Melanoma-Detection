"""Minimal FP32 training loop with epoch-boundary checkpoints and interruption-safe
recovery (Stage 2C2, hardened in 2C2.1). Software verification only; not the
main-training recipe.

Commit transaction for epoch k (all inside the run directory)
------------------------------------------------------------
  1. torch.save -> "ckpt_epoch<k>.pt.tmp-<attempt>-<uuid>"  (unique name, open "x", fsync)
  2. write "ckpt_epoch<k>.pt.sha256" with the SHA-256 of the temp file (open "x")
  3. os.rename(temp, "ckpt_epoch<k>.pt")   (never replaces an existing file)
  4. atomically replace "latest.json" {checkpoint, sha256, epoch_completed, global_step, attempt_id}
  5. append the commit to "commits.jsonl"; append the epoch summary to "epochs.jsonl".

Recovery policy (run only on --resume, bounded to the run directory)
--------------------------------------------------------------------
  * Stale temp files ("*.tmp-*", "latest.json.tmp-*") and sidecars without a checkpoint
    are MOVED to "quarantine/<recovery_id>/" (never deleted); each move is logged in
    "recovery.jsonl".
  * Orphan checkpoint (step 3 done, step 4 not): only "ckpt_epoch<pointer+1>.pt" may be
    adopted, and only if (a) its sidecar SHA-256 matches its bytes, (b) it loads, (c) its
    resume fingerprint equals the current one, (d) epoch_completed == pointer+1 and
    global_step == pointer_step + steps_per_epoch. Adoption rewrites latest.json and is
    logged. Any other/invalid checkpoint file beyond the pointer is quarantined; replay then
    proceeds from the last verified checkpoint and can write the filename again.
  * The pointer checkpoint is verified BEFORE torch.load: its SHA-256 must equal both
    latest.json and its sidecar; after loading, epoch_completed/global_step must equal the
    pointer. A mismatch raises CheckpointIntegrityError (no silent fallback).
  * Missing commits.jsonl entry for the pointer epoch (crash between steps 4 and 5) is
    re-appended from the pointer and logged.
  * No completed checkpoint yet (interruption during the first epoch): if run_start.json
    exists with the same fingerprint, training restarts from the SAME frozen
    initialisation (seed_everything -> build) at epoch 0 under a new attempt ID. Earlier
    attempt logs are preserved. There is no checkpoint to resume from, so nothing more is
    promised.
  * Interruption mid-epoch always replays that epoch from its start; exact mid-epoch
    recovery is not supported. These guarantees are for process crashes; they do not make
    the storage power-loss-proof and do not imply cross-machine determinism.

Logs
----
  Every step line in steps.jsonl carries its attempt_id. attempts.jsonl records each
  attempt. canonical_steps(run_dir) returns, for each committed epoch, only the steps of
  the attempt that committed it (from commits.jsonl), and checks their count; partial
  steps of interrupted attempts remain in the raw log but are not canonical.

Fault injection (tests only): env RV1_CRASH_AT = "step:<n>" | "ckpt_tmp:<k>" |
"ckpt_final:<k>" makes the process exit (os._exit(97)) at that point. Inert when unset.
"""
import glob
import hashlib
import json
import os
import random
import shutil
import time
import uuid
import warnings

from . import config as C, provenance as P, torch_data as TD
from .seeds import seed_everything, stable_seed

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
CRASH_CODE = 97


class ResumeMismatch(RuntimeError):
    pass


class CheckpointIntegrityError(RuntimeError):
    pass


class ValidationUndefined(RuntimeError):
    pass


TRAIN_BACKENDS = {"pytorch_default_tf32conv": {"cudnn_allow_tf32": True, "cuda_matmul_allow_tf32": False,
                                               "float32_matmul_precision": "highest"}}


def _backend_state():
    import torch
    b = torch.backends
    return {"cudnn_allow_tf32": b.cudnn.allow_tf32, "cuda_matmul_allow_tf32": b.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cudnn_deterministic": b.cudnn.deterministic, "cudnn_benchmark": b.cudnn.benchmark}


def _apply_train_backend(name):
    import torch
    c = TRAIN_BACKENDS[name]
    torch.backends.cudnn.allow_tf32 = c["cudnn_allow_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = c["cuda_matmul_allow_tf32"]
    torch.set_float32_matmul_precision(c["float32_matmul_precision"])
    return _backend_state()


def evaluate_selection(model, val_ds, device, batch_size, eval_precision):
    """Full validation pass (eval, no_grad). Under xai_precision_v1 the training backend settings
    are restored afterwards (checked). Returns uids, probs, labels, auroc; raises ValidationUndefined
    if the AUROC is undefined or non-finite."""
    import math
    before = _backend_state()
    if eval_precision == "xai_precision_v1":
        from .xai_attr import xai_precision
        with xai_precision():
            uids, probs, labels = probe_eval(model, val_ds, device, batch_size)
    else:
        uids, probs, labels = probe_eval(model, val_ds, device, batch_size)
    if _backend_state() != before:
        raise RuntimeError("training backend settings were not restored after validation")
    if not all(math.isfinite(p) for p in probs):
        raise ValidationUndefined("non-finite validation probabilities")
    if len(set(labels)) < 2:
        raise ValidationUndefined("validation AUROC undefined: fewer than two classes")
    a = auroc(labels, probs)
    if not math.isfinite(a):
        raise ValidationUndefined(f"validation AUROC non-finite: {a}")
    return uids, probs, labels, a


def _crash_point(tag):
    if os.environ.get("RV1_CRASH_AT") == tag:
        os._exit(CRASH_CODE)


def fingerprint(cfg, extra_files=()):
    """Scientific identity of a run; excludes operational/resume arguments."""
    fp = {"config_sha256": C.config_hash(cfg),
          "sources": {n: P.sha256_file(os.path.join(SRC_DIR, n))
                      for n in sorted(os.listdir(SRC_DIR)) if n.endswith(".py")},
          "inputs": {}}
    t = cfg["training"]
    rels = [t["train_subset_file"], t["probe_file"], cfg["environment_inventory"],
            *cfg["manifests"].values(), *cfg.get("supporting_inputs", {}).values(), *extra_files]
    if t["pretrained_weights_file"] != "none":
        rels.append(t["pretrained_weights_file"])
    for rel in sorted(set(rels)):
        fp["inputs"][rel] = P.sha256_file(C.resolve(cfg, rel))
    fp["digest"] = hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()
    return fp


def fp_diff(a, b):
    """Names of differing fingerprint parts (for clear mismatch messages)."""
    d = ["config_sha256"] if a["config_sha256"] != b["config_sha256"] else []
    for part in ("sources", "inputs"):
        for n in sorted(set(a[part]) | set(b[part])):
            if a[part].get(n) != b[part].get(n):
                d.append(f"{part[:-1] if part == 'sources' else 'input'}:{n}")
    return d


def _rng_state():
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(s):
    import numpy as np
    import torch
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch_cpu"])
    if s["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(s["torch_cuda"])


def rng_digest(s):
    """SHA-256 over ALL RNG state components, including the complete NumPy legacy state
    (generator name, key array, position, has_gauss flag, cached Gaussian)."""
    import torch
    h = hashlib.sha256()
    h.update(repr(s["python"]).encode())
    name, keys, pos, has_gauss, cached = s["numpy"]
    h.update(json.dumps([str(name), int(pos), int(has_gauss), repr(float(cached))]).encode())
    h.update(keys.tobytes())
    h.update(s["torch_cpu"].numpy().tobytes())
    for t in (s["torch_cuda"] or []):
        h.update(t.numpy().tobytes() if isinstance(t, torch.Tensor) else bytes(t))
    return h.hexdigest()


def _sync(device):
    if device == "cuda":
        import torch
        torch.cuda.synchronize()


def _sampler_gen(sampler_seed, epoch):
    import torch
    g = torch.Generator()
    g.manual_seed(stable_seed(sampler_seed, "sampler", epoch))
    return g


def probe_eval(model, probe_ds, device, batch_size):
    import torch
    model.eval()
    uids, probs, labels = [], [], []
    with torch.no_grad():
        for b in TD.make_epoch_loader(probe_ds, 0, batch_size, False, 0, 0):
            p = torch.sigmoid(model(b["x"].to(device)).squeeze(1).float()).cpu()
            probs += p.tolist()
            labels += b["label"].tolist()
            uids += [r["record_uid"] for r in b["record"]]
    return uids, probs, labels


def auroc(labels, probs):
    """Mann-Whitney U / (n1*n0) via scipy.stats (ties handled by scipy's midranks)."""
    from scipy.stats import mannwhitneyu
    pos = [p for p, y in zip(probs, labels) if y == 1]
    neg = [p for p, y in zip(probs, labels) if y == 0]
    return float(mannwhitneyu(pos, neg, alternative="two-sided").statistic / (len(pos) * len(neg)))


# ------------------------------------------------------------------ run-dir I/O
def _append(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:      # torn final line after a crash
                    out.append({"_unparseable": line[:200]})
    return out


def _atomic_json(obj, path, attempt):
    tmp = f"{path}.tmp-{attempt}-{uuid.uuid4().hex[:8]}"
    with open(tmp, "x", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_sidecar(ck_path):
    sc = ck_path + ".sha256"
    if not os.path.isfile(sc):
        return None
    with open(sc, encoding="utf-8") as f:
        tok = f.read().split()
    return tok[0] if tok else None


def _commit_checkpoint(ck, run_dir, epoch_k, gstep, attempt):
    import torch
    name = f"ckpt_epoch{epoch_k}.pt"
    final = os.path.join(run_dir, name)
    if os.path.exists(final):
        raise FileExistsError(f"{final} exists; refusing to overwrite a completed checkpoint")
    tmp = f"{final}.tmp-{attempt}-{uuid.uuid4().hex[:8]}"
    with open(tmp, "xb") as f:
        torch.save(ck, f)
        f.flush()
        os.fsync(f.fileno())
    digest = P.sha256_file(tmp)
    _crash_point(f"ckpt_tmp:{epoch_k}")
    with open(final + ".sha256", "x", encoding="utf-8") as f:
        f.write(f"{digest}  {name}\n")
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, final)                       # fails if final exists; never overwrites
    _crash_point(f"ckpt_final:{epoch_k}")
    ptr = {"checkpoint": name, "sha256": digest, "epoch_completed": epoch_k,
           "global_step": gstep, "attempt_id": attempt}
    _atomic_json(ptr, os.path.join(run_dir, "latest.json"), attempt)
    _append(os.path.join(run_dir, "commits.jsonl"), dict(ptr, time=time.time(), how="normal"))
    return ptr


def _quarantine(run_dir, path, rid, reason):
    qd = os.path.join(run_dir, "quarantine", rid)
    os.makedirs(qd, exist_ok=True)
    dst = os.path.join(qd, os.path.basename(path))
    os.rename(path, dst)
    _append(os.path.join(run_dir, "recovery.jsonl"),
            {"recovery_id": rid, "action": "quarantine", "file": os.path.basename(path),
             "to": os.path.relpath(dst, run_dir), "reason": reason, "time": time.time()})


def _load_verified(path, expected_sha):
    """SHA-256 check (pointer + sidecar) BEFORE torch.load."""
    import torch
    actual = P.sha256_file(path)
    side = _read_sidecar(path)
    if actual != expected_sha or side != actual:
        raise CheckpointIntegrityError(
            f"{os.path.basename(path)}: sha256 {actual[:12]} != expected {str(expected_sha)[:12]} "
            f"or sidecar {str(side)[:12]}")
    return torch.load(path, map_location="cpu", weights_only=False)


def recover(run_dir, fp, steps_per_epoch, rid):
    """Apply the recovery policy. Returns the (possibly adopted) pointer or None."""
    rec = os.path.join(run_dir, "recovery.jsonl")
    for p in sorted(glob.glob(os.path.join(run_dir, "*.tmp-*"))):
        _quarantine(run_dir, p, rid, "stale temporary file from an interrupted write")
    ptr_path = os.path.join(run_dir, "latest.json")
    ptr = None
    if os.path.exists(ptr_path):
        with open(ptr_path, encoding="utf-8") as f:
            ptr = json.load(f)
    done = ptr["epoch_completed"] if ptr else 0
    done_step = ptr["global_step"] if ptr else 0
    found = []
    for p in glob.glob(os.path.join(run_dir, "ckpt_epoch*.pt")):
        try:
            found.append((int(os.path.basename(p)[len("ckpt_epoch"):-3]), p))
        except ValueError:
            _quarantine(run_dir, p, rid, "unrecognised checkpoint name")
    for k, p in sorted(found):
        if k <= done:
            continue
        reason = None
        if k != done + 1:
            reason = f"unexpected checkpoint beyond pointer+1 (pointer epoch {done})"
        else:
            side = _read_sidecar(p)
            if side is None or side != P.sha256_file(p):
                reason = "orphan checkpoint fails sidecar SHA-256 check"
            else:
                try:
                    import torch
                    ck = torch.load(p, map_location="cpu", weights_only=False)
                except Exception as e:  # noqa: BLE001
                    ck, reason = None, f"orphan checkpoint does not load: {type(e).__name__}"
                if ck is not None:
                    if ck.get("fingerprint", {}).get("digest") != fp["digest"]:
                        reason = "orphan checkpoint fingerprint differs"
                    elif ck.get("epoch_completed") != k or \
                            ck.get("global_step") != done_step + steps_per_epoch:
                        reason = "orphan checkpoint epoch/global_step inconsistent"
        if reason:
            _quarantine(run_dir, p, rid, reason)
            if os.path.exists(p + ".sha256"):
                _quarantine(run_dir, p + ".sha256", rid, reason)
            continue
        ptr = {"checkpoint": os.path.basename(p), "sha256": side, "epoch_completed": k,
               "global_step": ck["global_step"], "attempt_id": ck.get("attempt_id")}
        _atomic_json(ptr, ptr_path, rid)
        _append(rec, {"recovery_id": rid, "action": "adopt_orphan", "checkpoint": ptr["checkpoint"],
                      "epoch_completed": k, "time": time.time()})
        _append(os.path.join(run_dir, "commits.jsonl"), dict(ptr, time=time.time(), how="adopted"))
        done, done_step = k, ck["global_step"]
    for sc in sorted(glob.glob(os.path.join(run_dir, "ckpt_epoch*.pt.sha256"))):
        if not os.path.exists(sc[:-len(".sha256")]):
            _quarantine(run_dir, sc, rid, "sidecar without checkpoint")
    if ptr is not None:
        commits = [c for c in _jsonl(os.path.join(run_dir, "commits.jsonl")) if "epoch_completed" in c]
        if not any(c["epoch_completed"] == ptr["epoch_completed"] for c in commits):
            _append(os.path.join(run_dir, "commits.jsonl"), dict(ptr, time=time.time(), how="pointer_recovered"))
            _append(rec, {"recovery_id": rid, "action": "commit_record_restored",
                          "epoch_completed": ptr["epoch_completed"], "time": time.time()})
    return ptr


def _sync_best_pointer(run_dir, best, attempt):
    """best.json = reference to the selected (immutable) checkpoint + its sha256; rewritten
    atomically whenever it differs from the best-selection state carried in the checkpoints."""
    if best["checkpoint"] is None:
        return
    p = os.path.join(run_dir, best["checkpoint"])
    want = {"checkpoint": best["checkpoint"], "epoch": best["epoch"], "val_auroc": best["val_auroc"],
            "sha256": P.sha256_file(p), "rule": "highest image-level HAM VAL AUROC; ties keep earlier epoch"}
    bp = os.path.join(run_dir, "best.json")
    if os.path.exists(bp):
        with open(bp, encoding="utf-8") as f:
            if json.load(f) == want:
                return
    _atomic_json(want, bp, attempt)


def canonical_steps(run_dir, steps_per_epoch):
    """Committed step sequence: for each committed epoch, steps of the committing attempt."""
    commits = {}
    for c in _jsonl(os.path.join(run_dir, "commits.jsonl")):
        if "epoch_completed" in c:
            commits[c["epoch_completed"]] = c["attempt_id"]
    steps = [s for s in _jsonl(os.path.join(run_dir, "steps.jsonl")) if "attempt_id" in s]
    out = []
    for k in sorted(commits):
        ep = [s for s in steps if s["epoch"] == k - 1 and s["attempt_id"] == commits[k]]
        if len(ep) != steps_per_epoch:
            raise RuntimeError(f"epoch {k}: {len(ep)} committed steps, expected {steps_per_epoch}")
        out += ep
    return out


# ------------------------------------------------------------------ training
def train(cfg, run_dir, build, device, stop_after_epoch=None, resume=False, log=print):
    """build(cfg, device) -> (model, train_ds, probe_ds, pos_weight_tensor, extra_info)."""
    import torch
    t = cfg["training"]
    fp = fingerprint(cfg)
    os.makedirs(run_dir, exist_ok=True)
    steps_log = os.path.join(run_dir, "steps.jsonl")
    n_prev = len(_jsonl(os.path.join(run_dir, "attempts.jsonl")))
    attempt = f"a{n_prev + 1}-{uuid.uuid4().hex[:8]}"
    seed = cfg["seeds"]["train_seed"]
    started = os.path.exists(os.path.join(run_dir, "run_start.json"))
    if not resume and (started or glob.glob(os.path.join(run_dir, "ckpt_epoch*"))
                       or os.path.exists(os.path.join(run_dir, "latest.json"))):
        raise FileExistsError(f"{run_dir} already has a run; use resume")
    if resume and not started:
        raise ResumeMismatch(f"{run_dir}: no run_start.json; nothing to resume")
    if resume:
        with open(os.path.join(run_dir, "run_start.json"), encoding="utf-8") as f:
            rs = json.load(f)["fingerprint"]
        if rs["digest"] != fp["digest"]:
            raise ResumeMismatch(f"run_start fingerprint differs: {fp_diff(rs, fp)}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        seed_rec = seed_everything(seed, cfg["determinism"])
        backend = _apply_train_backend(t["train_backend"]) if t.get("train_backend") else _backend_state()
        _sync(device)
        tb = time.perf_counter()
        model, train_ds, probe_ds, pos_weight, info = build(cfg, device)
        _sync(device)
        build_seconds = round(time.perf_counter() - tb, 3)
        opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
        steps_per_epoch = -(-len(train_ds) // t["batch_size"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"] * steps_per_epoch)
        crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        start, gstep = 0, 0
        mode = "fresh"
        selecting = t["best_model_selection"] != "disabled"
        best = {"val_auroc": None, "epoch": None, "checkpoint": None}
        if resume:
            ptr = recover(run_dir, fp, steps_per_epoch, attempt)
            if ptr is None:
                mode = "restart_from_initialisation"
                log("no committed checkpoint: restarting epoch 0 from the frozen initialisation")
            else:
                mode = "resume_from_checkpoint"
                ck = _load_verified(os.path.join(run_dir, ptr["checkpoint"]), ptr["sha256"])
                if (ck["epoch_completed"], ck["global_step"]) != (ptr["epoch_completed"], ptr["global_step"]):
                    raise CheckpointIntegrityError("pointer epoch/global_step differ from the checkpoint")
                if ck["fingerprint"]["digest"] != fp["digest"]:
                    raise ResumeMismatch(f"checkpoint fingerprint differs: {fp_diff(ck['fingerprint'], fp)}")
                model.load_state_dict(ck["model"], strict=True)
                opt.load_state_dict(ck["optimizer"])
                sched.load_state_dict(ck["scheduler"])
                start, gstep = ck["epoch_completed"], ck["global_step"]
                best = dict(ck.get("best") or best)
                if selecting:
                    _sync_best_pointer(run_dir, best, attempt)
                if _sampler_gen(t["sampler_seed"], start).get_state().tolist() != ck["next_sampler_state"]:
                    raise ResumeMismatch("sampler policy/state differs")
                _set_rng_state(ck["rng"])          # AFTER construction and state loading
                restored = rng_digest(_rng_state())
                if restored != ck["rng_digest"]:
                    raise ResumeMismatch("RNG state not restored exactly")
                log(f"resumed from {ptr['checkpoint']} epoch_completed={start} global_step={gstep}")
                _append(os.path.join(run_dir, "resume_events.jsonl"),
                        {"time": time.time(), "attempt_id": attempt, "from": ptr["checkpoint"],
                         "epoch_completed": start, "global_step": gstep, "rng_digest_restored": restored})
        else:
            with open(os.path.join(run_dir, "run_start.json"), "x", encoding="utf-8") as f:
                json.dump({"fingerprint": fp, "seed_record": seed_rec, "build_info": info,
                           "steps_per_epoch": steps_per_epoch, "config": cfg,
                           "train_backend_state": backend}, f, indent=1, default=str)
        _append(os.path.join(run_dir, "attempts.jsonl"),
                {"attempt_id": attempt, "mode": mode, "start_epoch": start, "start_global_step": gstep,
                 "time": time.time(), "stop_after_epoch": stop_after_epoch,
                 "build_seconds_model_init_weights_and_datasets": build_seconds,
                 "train_backend_state": backend})

        steps_in_attempt = 0
        for epoch in range(start, t["epochs"]):
            model.train()
            t0 = time.time()
            log_s = 0.0
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            _sync(device)
            tt = time.perf_counter()
            loader = TD.make_epoch_loader(train_ds, epoch, t["batch_size"], True, t["sampler_seed"],
                                          t["num_workers"])
            for b in loader:
                x = b["x"].to(device)
                y = b["label"].to(device=device, dtype=torch.float32)
                lr_used = sched.get_last_lr()[0]
                opt.zero_grad(set_to_none=True)
                logit = model(x).squeeze(1)
                loss = crit(logit, y)
                finite = bool(torch.isfinite(loss).item())
                if not finite:
                    raise FloatingPointError(f"non-finite loss at global_step {gstep}")
                loss.backward()
                opt.step()
                sched.step()
                gstep += 1
                steps_in_attempt += 1
                tl = time.perf_counter()
                _append(steps_log, {"attempt_id": attempt, "epoch": epoch, "global_step": gstep,
                                    "n": int(len(y)), "loss": float(loss.item()), "lr_used": lr_used,
                                    "finite": finite, "record_uids": [r["record_uid"] for r in b["record"]],
                                    "x_digest": hashlib.sha256(b["x"].numpy().tobytes()).hexdigest()})
                log_s += time.perf_counter() - tl
                _crash_point(f"step:{steps_in_attempt}")
            _sync(device)
            train_s = time.perf_counter() - tt
            tp = time.perf_counter()
            if selecting:
                uids, probs, labels, val_auc = evaluate_selection(
                    model, probe_ds, device, t["batch_size"], t.get("eval_precision", "training_default"))
                improved = best["val_auroc"] is None or val_auc > best["val_auroc"]   # ties keep earlier
                if improved:
                    best = {"val_auroc": val_auc, "epoch": epoch + 1, "checkpoint": f"ckpt_epoch{epoch + 1}.pt"}
                with open(os.path.join(run_dir, f"val_epoch{epoch + 1}_{attempt}.json"), "x", encoding="utf-8") as f:
                    json.dump({"epoch": epoch + 1, "attempt_id": attempt, "record_uids": uids, "labels": labels,
                               "probs": probs, "val_auroc": val_auc,
                               "eval_precision": t.get("eval_precision", "training_default")}, f)
            else:
                uids, probs, labels = probe_eval(model, probe_ds, device, t["batch_size"])
            _sync(device)
            probe_s = time.perf_counter() - tp
            rng = _rng_state()
            ck = {"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                  "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                  "epoch_completed": epoch + 1, "global_step": gstep, "attempt_id": attempt,
                  "rng": rng, "rng_digest": rng_digest(rng),
                  "sampler_policy": "per-epoch torch.Generator seeded stable_seed(sampler_seed,'sampler',epoch)",
                  "next_sampler_state": _sampler_gen(t["sampler_seed"], epoch + 1).get_state().tolist(),
                  "best_model_selection": t["best_model_selection"], "best": best,
                  "fingerprint": fp, "config": cfg,
                  "probe": {"record_uids": uids, "probs": probs, "labels": labels}}
            tc = time.perf_counter()
            _commit_checkpoint(ck, run_dir, epoch + 1, gstep, attempt)
            if selecting:
                _sync_best_pointer(run_dir, best, attempt)
                _append(os.path.join(run_dir, "selection_decisions.jsonl"),
                        {"epoch": epoch + 1, "attempt_id": attempt, "val_auroc": val_auc, "improved": improved,
                         "best_epoch": best["epoch"], "best_val_auroc": best["val_auroc"],
                         "rule": "highest val AUROC; ties keep the earlier epoch"})
            commit_s = time.perf_counter() - tc
            summ = {"attempt_id": attempt, "epoch": epoch + 1, "global_step": gstep,
                    "seconds": round(time.time() - t0, 2), "lr_after": sched.get_last_lr()[0],
                    "probe_auroc": auroc(labels, probs),
                    "peak_cuda_alloc_mib": (round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
                                            if device == "cuda" else None),
                    "peak_cuda_reserved_mib": (round(torch.cuda.max_memory_reserved() / 2 ** 20, 1)
                                               if device == "cuda" else None),
                    "timing": {"train_loop_s_incl_step_logging": round(train_s, 3),
                               "step_logging_s_input_sha256_and_fsync": round(log_s, 3),
                               "n_train_samples": len(train_ds),
                               "samples_per_s_incl_logging": round(len(train_ds) / train_s, 2),
                               "probe_eval_s": round(probe_s, 3),
                               "checkpoint_serialize_hash_commit_s": round(commit_s, 3),
                               "cuda_synchronized": device == "cuda"}}
            _append(os.path.join(run_dir, "epochs.jsonl"), summ)
            log(f"epoch {epoch + 1} done: {summ}")
            if stop_after_epoch is not None and epoch + 1 == stop_after_epoch and epoch + 1 < t["epochs"]:
                log(f"stopping after epoch {epoch + 1} as requested (operational)")
                break
    msgs = sorted({f"{w.category.__name__}: {w.message}" for w in caught})
    _append(os.path.join(run_dir, "warnings.jsonl"), {"time": time.time(), "attempt_id": attempt,
                                                      "warnings": msgs})
    return {"run_dir": run_dir, "global_step": gstep, "warnings": msgs, "attempt_id": attempt,
            "mode": mode}
