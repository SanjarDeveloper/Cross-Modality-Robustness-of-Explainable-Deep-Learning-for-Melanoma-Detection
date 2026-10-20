# Stage 2C2.1 Report — Interruption-Safe Checkpoint Recovery

Date: 2026-09-27. All work in this stage used `research_v1/.venv_model`.

## Correction to earlier wording

Stage 2C2 established **controlled epoch-boundary resume**: the process stopped cleanly after a committed epoch and was resumed in a fresh process.

It did **not** establish recovery from arbitrary interruptions. That is the subject of this stage, and even here it is shown only for *simulated process crashes*.

## Status

| Aspect | Result |
|---|---|
| Five fixes (checkpoint integrity, interrupted writes, attempt-aware logs, complete RNG digest, first-epoch path) | Implemented in `rv1/trainer.py`. `resume_compare.py` now uses the canonical step sequence |
| CPU toy tests with injected crashes (fresh processes) | **15 / 15 trainer tests pass** (5 earlier + 9 crash/recovery + 1 RNG-digest) |
| Full suite in `.venv_model` | **84 / 84 pass** |
| New GPU FP32 U/R smoke runs, same scientific settings | **PASS, bit-identical** (all differences 0.0). They are also bit-identical to the Stage 2C2 runs |
| Protected inputs | 0 changed, 0 removed |

No main training, AMP, other architectures or RQ runs were started.

**Limits of the evidence.** The tests use `os._exit` to simulate a process crash. They do **not** simulate power loss. There is no claim that storage is power-loss-proof, and no claim of determinism across machines.

## 1. Changes (`rv1/trainer.py`)

1. **Checkpoint integrity before loading.**
   - Before `torch.load`, the SHA-256 of the pointer's checkpoint must equal both `latest.json` and the checkpoint's `.sha256` sidecar. If not, `CheckpointIntegrityError` is raised; there is no fallback.
   - After loading, `epoch_completed` and `global_step` must equal the values in the pointer.
   - Fingerprint mismatches now name the parts that differ (`fp_diff`).

2. **Transactional commit.** For epoch *k*:
   1. `torch.save` to a **unique** temporary file `ckpt_epoch<k>.pt.tmp-<attempt>-<uuid>` (opened with `"x"`, then fsync);
   2. write the SHA-256 sidecar (`"x"`);
   3. `os.rename` to `ckpt_epoch<k>.pt` — this never replaces an existing file, and an existing target raises an error;
   4. atomically replace `latest.json`, using a unique temporary file and `os.replace`;
   5. append a line to `commits.jsonl` and to `epochs.jsonl`.

3. **Attempt-aware logs.**
   - Every invocation receives an `attempt_id`, recorded in `attempts.jsonl` together with its mode (`fresh`, `resume_from_checkpoint` or `restart_from_initialisation`).
   - Each step line carries its `attempt_id`.
   - `canonical_steps()` returns, for each committed epoch, only the steps of the attempt that committed it, and checks that their count equals `steps_per_epoch`.
   - Partial steps from interrupted attempts remain in the raw `steps.jsonl`, but they are **not** canonical.

4. **Complete RNG digest.** The NumPy legacy-state digest now covers the generator name, the key array, the position, the `has_gauss` flag and the cached Gaussian value, alongside the Python, Torch-CPU and Torch-CUDA states. Restoring the RNG works exactly as before.

5. **First-epoch interruption.** If no checkpoint has been committed yet, `--resume` requires `run_start.json` with the same fingerprint. Epoch 0 then restarts from the **same frozen initialisation** (`seed_everything`, then `build`) under a new attempt, and the logs of the failed attempt are kept. No resume from a checkpoint that does not exist is promised.

**Fault-injection hook.** `RV1_CRASH_AT=step:<n>|ckpt_tmp:<k>|ckpt_final:<k>` exits the process with code 97 at that point. It is inert unless set, and only the tests use it.

## 2. Recovery policy (runs only on `--resume`, and only inside the chosen run directory)

| Situation | Action |
|---|---|
| Stale temporary files (`*.tmp-*`, including `latest.json.tmp-*`) | **Moved** to `quarantine/<attempt>/`, never deleted. Each move is logged in `recovery.jsonl` |
| Completed checkpoint whose pointer was not updated (orphan `ckpt_epoch<pointer+1>.pt`) | **Adopted** only if (a) its sidecar SHA-256 matches its bytes, (b) it loads, (c) its fingerprint equals the current one, and (d) `epoch_completed = pointer + 1` and `global_step = pointer_step + steps_per_epoch`. `latest.json` is then rewritten, and the adoption is logged in `recovery.jsonl` and in `commits.jsonl` (`how: "adopted"`) |
| Invalid orphan, a checkpoint more than one epoch beyond the pointer, or an unrecognised checkpoint name | **Quarantined**, together with its sidecar. Training replays from the last verified checkpoint and can write that filename afresh without collision |
| Sidecar without a checkpoint | Quarantined |
| Missing `commits.jsonl` entry for the pointer's epoch | Restored from the pointer and logged |
| Pointer checkpoint corrupted, or inconsistent with the pointer | **Rejected** (`CheckpointIntegrityError`). No automatic fallback; a person must decide |
| No committed checkpoint (first-epoch interruption) | Restart from the frozen initialisation (§1.5) |
| Interruption mid-epoch | The interrupted epoch is replayed from its start, and the partial steps become non-canonical |
| Changed configuration, code or inputs | `ResumeMismatch`, naming the differing parts |

## 3. Tests

**CPU toy model:** Conv → BatchNorm → Dropout. Every arm runs in a fresh subprocess. Each crashed arm is checked to have exited with code 97, then resumed, and compared with an uninterrupted run U.

| Test | Result |
|---|---|
| Crash after 2 steps of epoch 2 → resume | Final model and buffers, optimizer, scheduler, step count, probe and canonical steps all match U (≤ 1e-6). The raw log has 12 steps; the canonical sequence has 10 |
| Crash during the checkpoint write (temporary file present, epoch 2) | The temporary file is quarantined and kept, epoch 2 is replayed, and the result matches U |
| Crash after finalising the checkpoint but before the pointer update (epoch 2) | The orphan is validated and adopted. `commits.jsonl` shows `normal`, then `adopted`. The result matches U |
| First-epoch crash (after 2 steps) | Restart from initialisation; the result matches U. Modes: `fresh`, then `restart_from_initialisation` |
| First-epoch crash during the checkpoint write | Temporary file quarantined, restart from initialisation; the result matches U |
| Stale temporary files (checkpoint and pointer) | Both quarantined; the result matches U |
| Invalid orphans (garbage with a valid sidecar; bad sidecar) | Quarantined; epoch 2 is replayed and a new file is written; the result matches U |
| One byte of the pointer checkpoint corrupted | `CheckpointIntegrityError`; nothing further is written |
| Pointer `global_step` inconsistent with the checkpoint | Rejected: "pointer epoch/global_step differ" |
| RNG digest: changing the NumPy generator name, position, `has_gauss`, cached Gaussian or a key word | Each change alters the digest; restoring the state reproduces it |
| Earlier 2C2 tests (resume equality, epoch-1 state, RNG negative controls, mismatch rejection, operational arguments) | Pass. One test was adapted to the new refusal message "already has a run" |

**Commands:**

```bash
research_v1\.venv_model\Scripts\python.exe -B -W ignore -m unittest discover -s research_v1/tests -p "test_trainer.py" -v
```

→ **15 tests OK** (100.6 s).

```bash
research_v1\.venv_model\Scripts\python.exe -B -W ignore -W error::ResourceWarning -m unittest discover -s research_v1/tests -p "test_*.py"
```

→ **84 tests OK** (155.5 s).

## 4. New GPU regression runs (same frozen config `train_smoke_v1.json`, new code)

The old Stage 2C2 runs were **not** resumed under the changed source fingerprint. They are kept unchanged.

- **U21:** `runs/train_smoke/U21_20260927T141619Z`, 2 epochs uninterrupted, 11.1 s.
- **R21:** `runs/train_smoke/R21_20260927T141619Z`. It stopped after epoch 1 and the process exited (7.8 s). It was then resumed in a fresh process under attempt `a2-…` (8.2 s).
- Peak CUDA allocation: about 1718 MiB.
- All 3 receipts pass `verify_receipt`.

**Comparison** (`runs/stage2c2_1/resume_comparison_gpu_fp32_v2.json`, using the canonical steps):

- The step order, augmentation digests, learning rates and losses are identical.
- At the epoch-1 checkpoint and at the final checkpoint, the model, optimizer, scheduler and RNG digests are identical.
- The restored RNG digest equals the saved one.
- Parameters and buffers: maximum difference **0.0** across all 320 tensors.
- Probe probabilities: maximum difference **0.0**.
- AUROC difference: 0.0.
- Fingerprints are equal.

**Status: PASS.**

**Against Stage 2C2** (`runs/stage2c2_1/old_vs_new_U.json`): the new U21 and the old U reach identical final parameters and buffers (0.0) and identical probe outputs (0.0). Their fingerprints differ only in `source:trainer.py` and `source:resume_compare.py`, which is the expected result of the code change. So the fixes did not change the training numerics.

**Not done:** crash-injection runs on the GPU. Crash recovery was verified on the CPU toy model. On the GPU, only the clean U/R regression check was run.

## 5. Files and integrity

**Changed:**
- `src/rv1/trainer.py`. The pre-change copy is preserved as `runs/stage2c2_1/trainer_before_2c2_1.py.txt`.
- `src/rv1/resume_compare.py`.
- `tests/test_trainer.py`.

**New:**
- the U21 and R21 run directories;
- in `runs/stage2c2_1/`: `resume_comparison_gpu_fp32_v2.json`, `old_vs_new_U.json` and the integrity evidence.

**Integrity:** 282 files before, 311 after.
- **0 changed, 0 removed.**
- The 29 added files are all inside the new U21 and R21 directories.
- The earlier `train_smoke` runs, configs, manifests and caches are unchanged. For the cache `.npy` files, only size and modification time were checked, and both are unchanged.
- No `__pycache__` was created.

**Receipts:** the earlier receipts and Stage 2C2 checkpoints were not rewritten. Verifying them against the current sources will report the expected source differences.

## 6. Remaining limitations

1. Only simulated process crashes were tested. There were no tests of power loss, OS-level write reordering, or full or failing disks.
2. The epoch is the only recovery granularity. Mid-epoch work is always repeated.
3. A corrupted pointer checkpoint is rejected but not repaired automatically. A person must decide, for example to keep the quarantined evidence and restart.
4. GPU crash-injection runs were not performed.
5. AMP, training with `num_workers > 0` combined with resume, other architectures and their weight provenance, the throughput pilot, and the main-training recipe are all **not yet tested or decided**.
6. Still carried forward:
   - the primary model is a newly trained v1 seed 0, and the legacy checkpoint stays separate;
   - `aug_heavy_v1` is deferred;
   - the Secondary B endpoint for adapt_val is not estimable;
   - the global OpenCV conflict remains.

**Stage 2C2.1 is complete. No main training was started.**
