#!/usr/bin/env python3
"""
sorter.py — organize the AIencoder/llama-cpp-wheels dataset into a clean, considered
            os / arch / backend / version / <wheel> hierarchy.

    linux/x86_64/cuda12.4/0.3.36/llama_cpp_python-0.3.36+cu124_basic-py3-none-manylinux_2_31_x86_64.whl
    linux/x86_64/cpu-avx512-fma-f16c-vnni-vbmi/0.3.19/llama_cpp_python-0.3.19+basic_avx512_fma_f16c_vnni_vbmi-...whl
    windows/x86_64/cuda12.6/0.3.36/llama_cpp_python-0.3.36+cu126-py3-none-win_amd64.whl
    macos/arm64/metal/0.3.4/llama_cpp_python-0.3.4-cp312-cp312-macosx_11_0_arm64.whl
    │     │      │        │
    │     │      │        └── llama-cpp-python version
    │     │      └── backend / build variant  (cuda12.4, cpu-avx2-fma-f16c, openblas-avx512-…, vulkan, metal, …)
    │     └── CPU architecture  (x86_64 / arm64 / aarch64 / i686 / …)
    └── operating system  (linux / windows / macos / any)

WHY THIS ORDER
    os     — the hardest constraint (a Windows user can never use a Linux wheel). A clean split at the root.
    arch   — the next hard constraint; keeps arm64 / aarch64 / i686 from being mixed in with x86_64.
    backend— the capability you actually choose. CPU wheels split by instruction tier (their primary axis);
             CUDA wheels group by toolkit version, cuda12.4/ etc. (their primary axis), with the CPU
             baseline carried in the filename — so e.g. +cu124_basic and +cu124_avx2_fma_f16c sit together.
    version— the leaf: a handful of wheels per combo, so no folder nears HuggingFace's 10k-files-per-dir cap.

HOW IT WORKS
    Lists every .whl in the dataset (wherever it currently sits), computes its target path, and MOVES it via
    server-side copy+delete (one atomic HfApi commit per batch) so multi-GB CUDA wheels are NEVER
    re-downloaded. Idempotent (skips wheels already at their target) and safely re-runnable: an interrupted
    or rate-limited run just picks up where it left off next time.

    ⚠️  Run it when the factory / mirror workflows are idle. They upload into version/ folders; sorting while
        they run is not dangerous (atomic commits, basename dedup) but leaves fresh stragglers for next time.

USAGE
    pip install "huggingface_hub>=0.28"
    export HF_TOKEN=hf_...                   # a WRITE token (or run `huggingface-cli login`)

    python sorter.py                         # DRY RUN — print the plan + write sort_plan.txt, move nothing
    python sorter.py --apply                 # perform the moves
    python sorter.py --apply --limit 200     # move only the first 200 (good first test)
    python sorter.py --verify                # report any wheel not at its canonical path (exit 1 if any)
    python sorter.py --apply --moves-per-commit 80
"""
import argparse
import os
import re
import sys
import time
from collections import Counter

try:
    from huggingface_hub import HfApi, CommitOperationCopy, CommitOperationDelete
    from huggingface_hub.errors import HfHubHTTPError
except ImportError:
    sys.exit("huggingface_hub is required:  pip install 'huggingface_hub>=0.28'")

REPO_DEFAULT = "AIencoder/llama-cpp-wheels"

_VERSION = re.compile(r"^llama_cpp_python-(\d+\.\d+\.\d+)")
_TAG = re.compile(r"^llama_cpp_python-\d+\.\d+\.\d+\+([^-]+)-")
_CUDA = re.compile(r"^cu(\d{2})(\d+)$")   # cu124 -> 12.4 ; cu1210 -> 12.10 (major = first 2 digits)

# platform-tag arch tokens, most-specific first; value = normalized folder name
_ARCH = [
    ("x86_64", "x86_64"), ("amd64", "x86_64"), ("aarch64", "aarch64"), ("arm64", "arm64"),
    ("universal2", "universal2"), ("i686", "i686"), ("ppc64le", "ppc64le"),
    ("s390x", "s390x"), ("riscv64", "riscv64"), ("armv7l", "armv7l"), ("wasm32", "wasm32"),
    ("win32", "x86"), ("x86", "x86"),
]
_STANDALONE = {"vulkan", "opencl", "clblast", "sycl", "rpc", "metal"}
_CPU_FAMILIES = ("openblas", "mkl", "basic", "cpu")


def backend_of(tag):
    """Readable backend/variant folder name from the +local tag (plain 'cpu' when untagged)."""
    if not tag:
        return "cpu"                                   # upstream base build (no +tag)
    t = tag.lower()
    # CUDA: cu124, cu124_basic, cu126_avx2_fma_f16c all group under cudaXX.Y — the CPU baseline,
    # if any, stays in the filename. (The bug this fixes: cu124_basic used to become "cu124-basic".)
    m = _CUDA.match(t.split("_", 1)[0])
    if m:
        return f"cuda{m.group(1)}.{m.group(2)}"
    if t in _STANDALONE:                               # vulkan / opencl / clblast / sycl / rpc / metal
        return t
    for fam in _CPU_FAMILIES:                          # openblas_avx2… / mkl_basic / basic_avx512… / cpu
        if t == fam or t.startswith(fam + "_"):
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
    elif "emscripten" in low or "wasm" in low:         # Pyodide / WebAssembly builds
        os_ = "emscripten"
    elif "linux" in low:                               # manylinux / musllinux / linux
        os_ = "linux"
    else:
        os_ = "any"                                    # py3-none-any etc.

    arch = "any"
    for token, norm in _ARCH:
        if token in low:
            arch = norm
            break
    if os_ == "macos" and arch == "aarch64":
        arch = "arm64"                                 # Apple Silicon: aarch64 and arm64 tags are the same arch

    tm = _TAG.match(fname)
    tag = tm.group(1) if tm else None
    backend = backend_of(tag)
    if tag is None and os_ == "macos":
        backend = "metal"                              # abetlen's default macOS builds are Metal-enabled
    return os_, arch, backend, version


def target_path(fname):
    c = classify(fname)
    if c is None:
        return None
    os_, arch, backend, version = c
    return f"{os_}/{arch}/{backend}/{version}/{fname}"


def plan_moves(files):
    """Return (moves, already, unparsed). moves is a list of (src, dst); dst=None drops a stray duplicate."""
    moves, already, unparsed, seen = [], 0, [], {}
    for src in files:
        fname = src.split("/")[-1]
        dst = target_path(fname)
        if dst is None:
            unparsed.append(src); continue
        if src == dst:
            already += 1; seen[dst] = src; continue
        if dst in seen:
            moves.append((src, None))                  # duplicate basename already placed -> drop this one
        else:
            seen[dst] = src
            moves.append((src, dst))
    return moves, already, unparsed


def _flush(api, repo, batch):
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
            api.create_commit(repo_id=repo, repo_type="dataset", operations=ops,
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
    ap.add_argument("--verify", action="store_true",
                    help="only check every wheel is at its canonical path; exit 1 if any are not")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N moves (0 = all)")
    ap.add_argument("--repo", default=REPO_DEFAULT, help=f"dataset repo id (default {REPO_DEFAULT})")
    ap.add_argument("--moves-per-commit", type=int, default=100,
                    help="moves bundled into each atomic commit (default 100; each move = 1 copy + 1 delete)")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)                            # falls back to cached `huggingface-cli login`
    if (args.apply) and not token:
        try:
            api.whoami()                               # confirm a cached login exists before moving anything
        except Exception:
            sys.exit("--apply needs write auth: set HF_TOKEN=hf_... (a WRITE token) or run huggingface-cli login")

    print(f"listing {args.repo} ...", flush=True)
    files = [f for f in api.list_repo_files(args.repo, repo_type="dataset") if f.endswith(".whl")]
    print(f"{len(files)} wheels found", flush=True)

    moves, already, unparsed = plan_moves(files)
    real = [m for m in moves if m[1]]
    strays = [m for m in moves if not m[1]]
    by_os = Counter(m[1].split("/")[0] for m in real)
    by_backend = Counter(m[1].split("/")[2] for m in real)

    print(f"\nalready in place:        {already}")
    print(f"to move:                 {len(real)}")
    print(f"stray duplicates to drop:{len(strays)}")
    print(f"unparseable (untouched): {len(unparsed)}")
    if by_os:
        print("  by os:     ", dict(by_os))
        print("  by backend:", dict(sorted(by_backend.items(), key=lambda kv: -kv[1])[:15]))
    if unparsed[:5]:
        print("  e.g. unparseable:", unparsed[:5])

    # --verify: report-only gate (useful after a run, or in CI)
    if args.verify:
        with open("sort_unsorted.txt", "w") as fh:
            for src, dst in real:
                fh.write(f"{src}  ->  {dst}\n")
        if real or strays:
            print(f"\nVERIFY: NOT fully sorted — {len(real)} wheel(s) off-target, {len(strays)} stray "
                  f"duplicate(s). See sort_unsorted.txt.", flush=True)
            sys.exit(1)
        print("\nVERIFY: every wheel is at its canonical path. ✓", flush=True)
        return

    with open("sort_plan.txt", "w") as fh:
        for src, dst in moves:
            fh.write(f"{src}  ->  {dst or '(drop stray duplicate)'}\n")
    print("full plan written to sort_plan.txt", flush=True)

    if not moves:
        print("\nnothing to do — dataset already organized.", flush=True); return
    if not args.apply:
        print("\nDRY RUN — nothing moved. Re-run with --apply to perform these moves.", flush=True); return

    todo = moves[: args.limit] if args.limit else moves
    per = max(1, args.moves_per_commit)
    done, batch, t0 = 0, [], time.time()
    for mv in todo:
        batch.append(mv)
        if len(batch) >= per:
            _flush(api, args.repo, batch); done += len(batch); batch = []
            rate = done / max(1e-9, time.time() - t0)
            eta = (len(todo) - done) / rate if rate else 0
            print(f"  committed {done}/{len(todo)}  (~{rate*60:.0f}/min, ETA {eta/60:.0f} min)", flush=True)
    if batch:
        _flush(api, args.repo, batch); done += len(batch)
        print(f"  committed {done}/{len(todo)}", flush=True)

    print(f"\ndone: {done} wheels sorted." + ("  (--limit hit; re-run to continue)" if args.limit else ""),
          flush=True)
    if not args.limit:
        print("tip: `python sorter.py --verify` confirms nothing was left behind.", flush=True)


if __name__ == "__main__":
    main()
