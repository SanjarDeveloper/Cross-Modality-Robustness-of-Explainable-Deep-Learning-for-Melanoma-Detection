"""Isolated, dependency-aware campaign with immutable, verified attempts.

Legacy results are evidence, never implicit resume markers. Non-smoke execution
requires --execute; the current architecture/strategy matrix is not a protocol.
"""
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import uuid

from campaign_checks import validate_outputs

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
ARCHS = ["resnet50", "efficientnet_b3", "densenet121"]
STRATS = ["standard", "heavy_aug"]
BATCH = {"resnet50": 32, "efficientnet_b3": 16, "densenet121": 24}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    with open(path, "x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)


@dataclass
class Step:
    key: str
    kind: str
    tag: str
    args: list
    deps: list
    outputs: list


def steps(config):
    smoke = config["smoke"]
    out = []
    for arch in config["archs"]:
        for strat in config["strats"]:
            tag = f"{arch}_{strat}_s0"
            train = f"train_{tag}"
            out.append(Step(train, "train", tag,
                            ["train.py", "--arch", arch, "--strategy", strat,
                             "--epochs", "1" if smoke else "8", "--batch", str(BATCH[arch]),
                             "--subset", "300" if smoke else "0"],
                            [], [str(Path("checkpoints") / f"{tag}.pt"),
                                 str(Path("results") / f"baseline_{tag}.json")]))
            for suite in ("real", "sweep", "xai", "rq4"):
                if suite == "rq4":
                    args = ["rq4_test.py", "--n", "40" if smoke else "400",
                            "--steps", "8" if smoke else "32"]
                    names = [f"rq4_tests_{tag}.json"]
                else:
                    args = ["run_eval.py", "--suite", suite, "--batch", str(BATCH[arch]),
                            "--subset", "200" if smoke else "0",
                            "--levels", "3" if smoke else "6",
                            "--boot", "100" if smoke else "500",
                            "--xai_n", "40" if smoke else "400",
                            "--steps", "8" if smoke else "32", "--xai_batch", "8"]
                    names = [f"{suite}_{tag}.json"]
                    if suite == "real":
                        names.append(f"derm7pt_paired_{tag}.json")
                out.append(Step(f"{suite}_{tag}", suite, tag, args, [train],
                                [str(Path("results") / n) for n in names]))
            out.append(Step(f"analyse_{tag}", "analyse", tag,
                            ["analyse.py", "--tag", tag],
                            [f"{s}_{tag}" for s in ("real", "sweep", "xai")],
                            [str(Path("results") / f"report_{tag}.json")]))
            out.append(Step(f"figures_{tag}", "figures", tag,
                            ["figures.py", "--tag", tag],
                            [f"{s}_{tag}" for s in ("real", "sweep", "xai", "analyse")],
                            [str(Path("results") / "figures" / tag / n) for n in
                             ("fig1_sweep.png", "fig2_real_shift.png",
                              "fig3_faithfulness.png", "fig4_cmrs.png")]))
    tags = [s.tag for s in out if s.kind == "train"]
    # Keep the historical endpoint for a software rehearsal only, with warnings.
    # Its scientific inclusion requires an explicit methodology decision.
    if config["fairness"]:
        tag = tags[0]
        out.append(Step("fairness", "fairness", tag,
                        ["fairness.py"] + (["--smoke"] if smoke else []),
                        [f"train_{tag}"], [str(Path("results") / "fairness_report.json")]))
    out.append(Step("compare", "compare", "", ["compare_arch.py"],
                    [s.key for s in out if s.kind in ("real", "sweep", "xai")],
                    [str(Path("results") / "rq3_comparison.json")]))
    return out


def provenance():
    files = {str(p.relative_to(ROOT)): digest(p) for p in sorted(CODE.rglob("*.py"))}
    # Hash metadata AND image/cache contents: a same-size edit is not a valid resume.
    for folder in ("DATASETS", "cache"):
        for p in sorted((ROOT / folder).rglob("*")):
            if p.is_file():
                files[str(p.relative_to(ROOT))] = digest(p)
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                if d.metadata["Name"]}
    return {"files": files, "python": sys.version, "executable": sys.executable,
            "packages": packages}


def expected_spec(step, config, context, completed):
    return {"schema": 1, "step": asdict(step), "config": config,
            "context": context, "dependencies": {
                dep: fingerprint(completed[dep]["receipt"]) for dep in step.deps}}


def find_complete(run_dir, step, spec):
    for receipt in sorted((run_dir / "steps" / step.key).glob("*\\complete.json"),
                          reverse=True):
        try:
            obj = json.loads(receipt.read_text(encoding="utf-8"))
            if obj["spec"] != spec:
                print(f"STALE {receipt}: configuration, provenance or dependency changed")
                continue
            if set(obj["outputs"]) != set(step.outputs):
                raise ValueError("Incomplete output inventory")
            for name in step.outputs:
                if digest(receipt.parent / name) != obj["outputs"][name]:
                    raise ValueError(f"Output checksum mismatch: {name}")
            validate_outputs(step, receipt.parent, spec["config"])
            return {"directory": receipt.parent, "receipt": obj}
        except (OSError, ValueError, KeyError, TypeError) as e:
            print(f"INVALID {receipt}: {e}", file=sys.stderr)
    return None


def execute(step, spec, run_dir, completed):
    attempt = run_dir / "steps" / step.key / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8])
    attempt.mkdir(parents=True)
    (attempt / "results").mkdir()
    (attempt / "checkpoints").mkdir()
    write_json(attempt / "request.json", spec)
    cmd = [sys.executable, str(CODE / step.args[0]), *step.args[1:]]
    for dep in step.deps:
        prior = completed[dep]
        for name in prior["receipt"]["outputs"]:
            source = prior["directory"] / name
            if source.suffix == ".pt":
                cmd.extend(["--ckpt", str(source)])
            elif source.suffix == ".json":
                target = attempt / "results" / source.name
                if target.exists():
                    raise ValueError(f"Conflicting dependency output: {target}")
                shutil.copyfile(source, target)
    env = dict(os.environ, MELANOMA_RUN_ROOT=str(attempt),
               MELANOMA_RUN_MODE="smoke" if spec["config"]["smoke"] else "experiment",
               PYTHONUNBUFFERED="1",
               PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    log = attempt / "process.log"
    print(f"RUN {step.key} -> {attempt}", flush=True)
    try:
        with log.open("x", encoding="utf-8") as f:
            f.write(subprocess.list2cmdline(cmd) + "\n")
            f.flush()
            p = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        if p.returncode:
            raise RuntimeError(f"{step.key}: exit {p.returncode}; see {log}")
        validate_outputs(step, attempt, spec["config"])
    except Exception as e:
        write_json(attempt / "failure.json",
                   {"error": str(e), "traceback": traceback.format_exc()})
        raise
    receipt = {"spec": spec, "outputs": {n: digest(attempt / n) for n in step.outputs}}
    # Rename is the commit point; an interrupted write is never a completion.
    write_json(attempt / "complete.pending.json", receipt)
    (attempt / "complete.pending.json").rename(attempt / "complete.json")
    return {"directory": attempt, "receipt": receipt}


def run_campaign(todo, config, run_dir, context, status=False, force=False):
    completed, states = {}, {}
    for step in todo:
        missing = [d for d in step.deps if d not in completed]
        if missing:
            states[step.key] = "blocked"
            print(f"BLOCKED {step.key}: {', '.join(missing)}")
            continue
        spec = expected_spec(step, config, context, completed)
        cached = None if force else find_complete(run_dir, step, spec)
        if cached:
            completed[step.key] = cached
            states[step.key] = "verified"
            print(f"VERIFIED {step.key}")
        elif status:
            states[step.key] = "pending"
            print(f"PENDING {step.key}")
        else:
            try:
                completed[step.key] = execute(step, spec, run_dir, completed)
                states[step.key] = "done"
            except Exception as e:
                # Process boundary: preserve traceback, fail the step, block consumers.
                traceback.print_exc()
                states[step.key] = "failed"
                print(f"FAILED {step.key}: {e}", file=sys.stderr)
    print(f"{len(completed)}/{len(todo)} verified complete; states={states}")
    if not status:
        write_json(run_dir / f"summary-{uuid.uuid4().hex}.json",
                   {"states": states, "outputs": {
                       k: str(v["directory"]) for k, v in completed.items()}})
    return 0 if len(completed) == len(todo) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--force", action="store_true", help="new attempts; never overwrite")
    ap.add_argument("--execute", action="store_true", help="explicitly permit non-smoke work")
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--archs", nargs="+", choices=ARCHS, default=ARCHS)
    ap.add_argument("--strats", nargs="+", choices=STRATS, default=STRATS)
    ap.add_argument("--include-legacy-fairness", action="store_true",
                    help="off-target malignancy analysis; NOT protocol approval")
    a = ap.parse_args()
    if not (a.smoke or a.status or a.execute):
        ap.error("No training started. Non-smoke execution requires --execute after protocol approval.")
    config = {"smoke": a.smoke, "archs": sorted(set(a.archs)),
              "strats": sorted(set(a.strats)), "seed": 0,
              "fairness": a.smoke or a.include_legacy_fairness}
    mode = "smoke" if a.smoke else "experiments"
    run_dir = (a.run_dir or ROOT / "runs" / mode / "default").resolve()
    if run_dir == ROOT or any(run_dir == ROOT / n or ROOT / n in run_dir.parents
                             for n in ("results", "checkpoints", "logs", "code", "DATASETS", "cache")):
        ap.error("Use a separate run directory, never a legacy/source/data directory.")
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        if json.loads(cfg_path.read_text(encoding="utf-8")) != config:
            ap.error("Run configuration/mode differs; use a NEW --run-dir.")
    elif run_dir.exists() and any(run_dir.iterdir()):
        ap.error("Unmanaged nonempty run directory; use a new directory.")
    elif not a.status:
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(cfg_path, config)
    print("Hashing source, datasets and cache for exact resume provenance...", flush=True)
    prov = provenance()
    context = fingerprint(prov)
    if not a.status:
        p = run_dir / f"provenance-{context}.json"
        if not p.exists():
            write_json(p, prov)
    return run_campaign(steps(config), config, run_dir, context, a.status, a.force)


if __name__ == "__main__":
    sys.exit(main())
