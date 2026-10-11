#!/usr/bin/env python3
"""
sorter.py — organize the AIencoder/llama-cpp-wheels dataset into a clean, considered
            os / arch / backend / version / <wheel> hierarchy.

    linux/x86_64/cuda12.4/0.3.19/llama_cpp_python-0.3.19+cu124-cp311-cp311-manylinux_2_31_x86_64.whl
    linux/x86_64/cpu-avx512-vnni-vbmi/0.3.19/llama_cpp_python-0.3.19+basic_avx512_fma_f16c_vnni_vbmi-...whl
    windows/x86_64/cuda12.6/0.3.19/llama_cpp_python-0.3.19+cu126-cp311-cp311-win_amd64.whl
    macos/arm64/metal/0.3.4/llama_cpp_python-0.3.4-cp312-cp312-macosx_11_0_arm64.whl
    │     │      │        │
    │     │      │        └── llama-cpp-python version
    │     │      └── backend / build variant  (readable: cuda12.4, cpu-avx2, openblas-avx512, vulkan, metal, …)
    │     └── CPU architecture  (x86_64 / arm64 / aarch64 / i686 / …)
    └── operating system  (linux / windows / macos)

WHY THIS ORDER
    os     — the hardest constraint (a Windows user can never use a Linux wheel). 3 clean entries at the root.
    arch   — the next hard constraint; previously missing, so arm64/i686 wheels got mixed in with x86_64.
    backend— the capability you actually choose (CUDA version, CPU instruction tier, BLAS, Metal…). Readable names.
    version— interchangeable-ish; the leaf holds every Python build for that exact combo (~7 files), so no folder
             ever approaches HuggingFace's 10,000-files-per-directory cap.

HOW IT WORKS
    Lists every .whl in the dataset (wherever it currently sits), computes its target path, and MOVES it there via
    server-side copy+delete (HfApi CommitOperationCopy/Delete) so multi-GB CUDA wheels are NOT re-downloaded.
    Idempotent (skips wheels already at their target) and re-runnable. Non-wheel files (README, .gitattributes)
    stay in the root.

USAGE
    pip install "huggingface_hub>=0.28"
    export HF_TOKEN=hf_...                  # a WRITE token (or run `huggingface-cli login`)

    python sorter.py                        # DRY RUN — prints the plan + writes sort_plan.txt, moves nothing
    python sorter.py --apply                # perform the moves
    python sorter.py --apply --limit 200    # move only the first 200 (good first test)
"""
import argparse
import os
import re
import time
from collections import Counter

from huggingface_hub import HfApi, CommitOperationCopy, CommitOperationDelete
from huggingface_hub.errors import HfHubHTTPError

REPO = "AIencoder/llama-cpp-wheels"
MOVES_PER_COMMIT = 100            # each move = 1 copy + 1 delete op

_VERSION = re.compile(r"^llama_cpp_python-(\d+\.\d+\.\d+)")
_TAG = re.compile(r"^llama_cpp_python-\d+\.\d+\.\d+\+([^-]+)-")
_CUDA = re.compile(r"^cu(\d{2})(\d+)$")

# arch tokens to look for in the platform tag, longest/most-specific first; value = normalized name
_ARCH = [
    ("x86_64", "x86_64"), ("amd64", "x86_64"), ("aarch64", "aarch64"), ("arm64", "arm64"),
    ("universal2", "universal2"), ("i686", "i686"), ("ppc64le", "ppc64le"),
    ("s390x", "s390x"), ("win32", "x86"),
]


def backend_of(tag):
    """Readable backend/variant folder name from the +local tag ('base' build when untagged)."""
    if not tag:
        return "cpu"                                  # upstream base build (no +tag)
    t = tag.lower()
    m = _CUDA.match(t)
    if m:
        return f"cuda{m.group(1)}.{m.group(2)}"        # cu124 -> cuda12.4
    if t in {"vulkan", "opencl", "clblast", "sycl", "rpc", "metal"}:
        return t
    for fam in ("openblas", "mkl", "clblast", "basic", "cpu"):
        if t.startswith(fam):
            tail = t[len(fam):].strip("_").replace("_", "-")
            name = "cpu" if fam in ("basic", "cpu") else fam
            return f"{name}-{tail}" if tail else name
    return t.replace("_", "-")                         # anything unexpected, cleaned up


def classify(fname):
    """filename -> (os, arch, backend, version), or None if unparseable."""
    vm = _VERSION.match(fname)
    if not vm:
        return None
    version = vm.group(1)

    low = fname.lower()
    if "win_amd64" in low or "win32" in low or "-win" in low:
        os_ = "windows"
    elif "macosx" in low or "-macos" in low:
        os_ = "macos"
    elif "linux" in low:                               # manylinux / musllinux / linux
        os_ = "linux"
    else:
        os_ = "any"                                    # py3-none-any etc.

    arch = "any"
    for token, norm in _ARCH:
        if token in low:
            arch = norm
            break

    tm = _TAG.match(fname)
    tag = tm.group(1) if tm else None
    backend = backend_of(tag)
    if tag is None and os_ == "macos":
        backend = "metal"            # abetlen's default macOS builds are Metal-enabled
    return os_, arch, backend, version


def target_path(fname):
    c = classify(fname)
    if c is None:
        return None
    os_, arch, backend, version = c
    return f"{os_}/{arch}/{backend}/{version}/{fname}"


def _flush(api, batch):
    """One atomic commit for a batch of (src, dst) moves; dst=None means delete a stray duplicate."""
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
                              commit_message=f"sort: {len(batch)} wheels -> os/arch/backend/version")
            return
        except HfHubHTTPError as e:
            if "429" in str(e) and attempt < 10:
                wait = min(180, 20 * attempt)
                print(f"    rate limited, waiting {wait}s (attempt {attempt})", flush=True)
                time.sleep(wait)
            else:
                raise


def main():
    ap = argparse.ArgumentParser(description="Sort llama-cpp-wheels into os/arch/backend/version/")
    ap.add_argument("--apply", action="store_true", help="actually move files (default: dry run)")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N moves (0 = all)")
    args = ap.parse_args()

    api = HfApi(token=os.environ.get("HF_TOKEN"))      # falls back to cached `huggingface-cli login`

    print("listing dataset...", flush=True)
    files = [f for f in api.list_repo_files(REPO, repo_type="dataset") if f.endswith(".whl")]
    print(f"{len(files)} wheels found", flush=True)

    moves, already, unparsed, seen = [], 0, [], {}
    for src in files:
        fname = src.split("/")[-1]
        dst = target_path(fname)
        if dst is None:
            unparsed.append(src); continue
        if src == dst:
            already += 1; continue
        if dst in seen:
            moves.append((src, None))                  # duplicate basename -> drop the stray copy
        else:
            seen[dst] = src
            moves.append((src, dst))

    real = [m for m in moves if m[1]]
    by_os = Counter(m[1].split("/")[0] for m in real)
    by_backend = Counter(m[1].split("/")[2] for m in real)
    print(f"\nalready in place:       {already}")
    print(f"to move:                {len(real)}")
    print(f"stray duplicates to drop:{sum(1 for m in moves if not m[1])}")
    print(f"unparseable (untouched): {len(unparsed)}")
    if by_os:
        print("  by os:     ", dict(by_os))
        print("  by backend:", dict(sorted(by_backend.items(), key=lambda kv: -kv[1])[:15]))

    with open("sort_plan.txt", "w") as fh:
        for src, dst in moves:
            fh.write(f"{src}  ->  {dst or '(drop stray duplicate)'}\n")
    print("full plan written to sort_plan.txt", flush=True)

    if not moves:
        print("\nnothing to do — dataset already organized.", flush=True); return
    if not args.apply:
        print("\nDRY RUN — nothing moved. Re-run with --apply to perform these moves.", flush=True); return

    todo = moves[: args.limit] if args.limit else moves
    done, batch = 0, []
    for mv in todo:
        batch.append(mv)
        if len(batch) >= MOVES_PER_COMMIT:
            _flush(api, batch); done += len(batch)
            print(f"  committed {done}/{len(todo)}", flush=True); batch = []
    if batch:
        _flush(api, batch); done += len(batch)
        print(f"  committed {done}/{len(todo)}", flush=True)

    print(f"\ndone: {done} wheels sorted." + ("  (--limit hit; re-run to continue)" if args.limit else ""), flush=True)


if __name__ == "__main__":
    main()
