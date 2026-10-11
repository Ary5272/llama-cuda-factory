#!/usr/bin/env python3
"""
sorter.py — reorganize the AIencoder/llama-cpp-wheels dataset into a clean
            OS / version / type / <wheel> hierarchy.

    linux/0.3.19/cu124/llama_cpp_python-0.3.19+cu124-cp311-cp311-manylinux_2_31_x86_64.whl
    windows/0.3.19/cu126/llama_cpp_python-0.3.19+cu126-cp311-cp311-win_amd64.whl
    macos/0.1.30/base/llama_cpp_python-0.1.30-cp310-cp310-macosx_10_9_x86_64.whl
    │     │      │
    │     │      └── type  = build variant / backend  (the +tag; "base" if none)
    │     └── version                                  (0.3.19, 0.1.30, ...)
    └── os  = linux / windows / macos                  (from the platform tag)

Why: HuggingFace caps any single folder at 10,000 files. Flat root hit that wall.
This nests wheels so every leaf folder stays tiny — and it's just nicer to browse.

HOW IT WORKS
    - Lists every .whl in the dataset (wherever it currently lives: root, a version
      subfolder, or already sorted).
    - Computes each wheel's target path and MOVES it there via server-side copy+delete
      (HfApi CommitOperationCopy/Delete) — LFS pointers are copied on the Hub, so the
      big multi-GB CUDA wheels are NOT re-downloaded or re-uploaded.
    - Idempotent: wheels already at their target are skipped, so you can re-run anytime
      (e.g. after the factory/mirror add more wheels).
    - Non-wheel files (README.md, .gitattributes) are left in the root.

USAGE
    pip install "huggingface_hub>=0.28"
    export HF_TOKEN=hf_...            # a WRITE token (or run `huggingface-cli login`)

    python sorter.py                 # DRY RUN — shows the plan, moves nothing
    python sorter.py --apply         # actually move the files
    python sorter.py --apply --limit 200   # move only the first 200 (handy first test)

Safe by default: without --apply it only prints what it would do and writes the full
plan to sort_plan.txt.
"""
import argparse
import os
import re
import sys
import time

from huggingface_hub import HfApi, CommitOperationCopy, CommitOperationDelete
from huggingface_hub.errors import HfHubHTTPError

REPO = "AIencoder/llama-cpp-wheels"
MOVES_PER_COMMIT = 100          # each move = 1 copy + 1 delete op

_VERSION = re.compile(r"^llama_cpp_python-(\d+\.\d+\.\d+)")
_TYPE = re.compile(r"^llama_cpp_python-\d+\.\d+\.\d+\+([^-]+)-")


def classify(fname):
    """filename -> (os, version, type). Returns None if it can't be parsed."""
    vm = _VERSION.match(fname)
    if not vm:
        return None
    version = vm.group(1)

    low = fname.lower()
    if "win_amd64" in low or "win32" in low or "-win" in low:
        os_ = "windows"
    elif "macosx" in low or "-macos" in low:
        os_ = "macos"
    elif "linux" in low:                      # manylinux / musllinux / linux
        os_ = "linux"
    else:
        os_ = "any"                           # e.g. py3-none-any

    tm = _TYPE.match(fname)
    type_ = tm.group(1) if tm else "base"     # no +tag => upstream base build
    return os_, version, type_


def target_path(fname):
    c = classify(fname)
    if c is None:
        return None
    os_, version, type_ = c
    return f"{os_}/{version}/{type_}/{fname}"


def main():
    ap = argparse.ArgumentParser(description="Sort the llama-cpp-wheels dataset into os/version/type/")
    ap.add_argument("--apply", action="store_true", help="actually move files (default: dry run)")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N moves (0 = all)")
    args = ap.parse_args()

    api = HfApi(token=os.environ.get("HF_TOKEN"))  # falls back to cached `huggingface-cli login`

    print("listing dataset...", flush=True)
    files = [f for f in api.list_repo_files(REPO, repo_type="dataset") if f.endswith(".whl")]
    print(f"{len(files)} wheels found", flush=True)

    moves = []          # (src, dst)
    already = 0
    unparsed = []
    seen_targets = {}
    for src in files:
        fname = src.split("/")[-1]
        dst = target_path(fname)
        if dst is None:
            unparsed.append(src)
            continue
        if src == dst:
            already += 1
            continue
        # guard against two sources colliding on one target (duplicate basenames)
        if dst in seen_targets:
            # someone else already claims this target; just remove this stray copy
            moves.append((src, None))
        else:
            seen_targets[dst] = src
            moves.append((src, dst))

    # summarize
    from collections import Counter
    by_os = Counter((target_path(m[0].split("/")[-1]) or "??/").split("/")[0] for m in moves if m[1])
    print(f"\nalready sorted: {already}")
    print(f"to move:        {sum(1 for m in moves if m[1])}")
    print(f"stray dupes to drop: {sum(1 for m in moves if not m[1])}")
    print(f"unparseable (left alone): {len(unparsed)}")
    if by_os:
        print("moves by OS:", dict(by_os))

    # write the full plan for inspection
    with open("sort_plan.txt", "w") as fh:
        for src, dst in moves:
            fh.write(f"{src}  ->  {dst or '(delete stray dupe)'}\n")
    print("full plan written to sort_plan.txt", flush=True)

    if not moves:
        print("\nnothing to do — dataset already organized.", flush=True)
        return

    if not args.apply:
        print("\nDRY RUN — nothing moved. Re-run with --apply to perform these moves.", flush=True)
        return

    # apply, batched
    todo = moves[: args.limit] if args.limit else moves
    done = 0
    batch = []
    for src, dst in todo:
        if dst is None:
            # stray duplicate: delete it in its own small op-group via a move-less commit
            batch.append((src, None))
        else:
            batch.append((src, dst))
        if len(batch) >= MOVES_PER_COMMIT:
            _flush(api, batch)
            done += len(batch)
            print(f"  moved {done}/{len(todo)}", flush=True)
            batch = []
    if batch:
        _flush(api, batch)
        done += len(batch)
        print(f"  moved {done}/{len(todo)}", flush=True)

    print(f"\ndone: {done} operations committed."
          + ("" if not args.limit else "  (--limit hit; re-run to continue)"), flush=True)


def _flush(api, batch):
    """Build one commit for a batch that may contain moves and stray-dupe deletes."""
    ops = []
    for src, dst in batch:
        if dst is None:
            ops.append(CommitOperationDelete(path_in_repo=src))
        else:
            ops.append(CommitOperationCopy(src_path_in_repo=src, path_in_repo=dst))
            ops.append(CommitOperationDelete(path_in_repo=src))
    for attempt in range(1, 11):
        try:
            api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops,
                              commit_message=f"sort: {len(batch)} wheels -> os/version/type")
            return
        except HfHubHTTPError as e:
            if "429" in str(e) and attempt < 10:
                wait = min(180, 20 * attempt)
                print(f"    rate limited, waiting {wait}s (attempt {attempt})", flush=True)
                time.sleep(wait)
            else:
                raise


if __name__ == "__main__":
    main()
