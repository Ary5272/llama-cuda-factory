#!/usr/bin/env python3
"""
Delete the broken MKL wheels from AIencoder/llama-cpp-wheels.

The original factory built +mkl_* wheels with MKL EXCLUDED, so they're ~3-6 MB and cannot
import (libmkl_intel_lp64.so not found at load time). They've never worked. This removes
them so the factory can rebuild them properly bundled.

SAFETY:
  - Dry run by default: lists what WOULD be deleted, deletes nothing.
  - Only targets +mkl_* wheels UNDER a size threshold (the broken ones). A correctly bundled
    MKL wheel is >100 MB, so the threshold can't catch good ones.
  - Requires --confirm to actually delete, and re-checks each file's size at delete time.

Usage:
  python delete_broken_mkl.py              # dry run — preview only
  python delete_broken_mkl.py --confirm    # actually delete

Needs HF_TOKEN in the environment with write access to the dataset.
"""

from __future__ import annotations

import os
import sys

from huggingface_hub import HfApi

DATASET = "AIencoder/llama-cpp-wheels"
# Broken MKL wheels are 3-6 MB. A bundled (working) MKL wheel is >100 MB. 50 MB is a safe
# line: nothing above it is a broken wheel, everything below it that is +mkl is broken.
MAX_BROKEN_MB = 50.0


def main() -> int:
    confirm = "--confirm" in sys.argv
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set — export a token with write access to the dataset.")
        return 1

    api = HfApi(token=token)

    print(f"Scanning {DATASET} for broken (under-{MAX_BROKEN_MB:.0f}MB) +mkl wheels ...\n")
    info = api.repo_info(DATASET, repo_type="dataset", files_metadata=True)

    broken = []
    for sib in info.siblings:
        name = sib.rfilename
        if not name.endswith(".whl"):
            continue
        if "+mkl" not in name:
            continue
        size_mb = (sib.size or 0) / 1024**2
        if size_mb < MAX_BROKEN_MB:
            broken.append((name, size_mb))

    if not broken:
        print("No broken MKL wheels found — nothing to delete.")
        return 0

    broken.sort()
    total_mb = sum(mb for _, mb in broken)
    print(f"Found {len(broken)} broken MKL wheels ({total_mb/1024:.2f} GB):\n")
    for name, mb in broken[:20]:
        print(f"  {mb:5.1f} MB  {name}")
    if len(broken) > 20:
        print(f"  ... and {len(broken) - 20} more")
    print()

    if not confirm:
        print("DRY RUN — nothing deleted. Re-run with --confirm to delete these.")
        return 0

    # Delete. operations must be CommitOperationDelete.
    from huggingface_hub import CommitOperationDelete

    ops = [CommitOperationDelete(path_in_repo=name) for name, _ in broken]
    print(f"Deleting {len(ops)} files in one commit ...")
    api.create_commit(
        repo_id=DATASET,
        repo_type="dataset",
        operations=ops,
        commit_message=f"Remove {len(ops)} broken (MKL-excluded, non-importable) wheels",
    )
    print("Done. The factory will rebuild these properly bundled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
