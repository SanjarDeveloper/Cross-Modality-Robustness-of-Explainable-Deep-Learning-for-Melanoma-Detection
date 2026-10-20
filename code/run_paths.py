"""Output isolation shared by campaign subprocesses; datasets stay read-only."""
import os
from pathlib import Path


def output_root():
    return os.path.abspath(os.environ.get(
        "MELANOMA_RUN_ROOT", Path(__file__).resolve().parent.parent))


def paired_path(results, tag):
    path = Path(results) / f"derm7pt_paired_{tag}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: tagged paired result required; the legacy untagged file "
            "has no reliable model provenance and is not used automatically.")
    return path
