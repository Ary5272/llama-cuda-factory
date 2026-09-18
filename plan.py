#!/usr/bin/env python3
"""
Factory dispatcher.

Computes every wheel in range, subtracts what's already in the dataset, hands the next
batch to the build matrix. The queue is DERIVED from the dataset every run — never stored
as state, so a failed build simply reappears next time and nothing can get stuck the way
the old Spaces job files did.

Two kinds of wheel, built differently:

  CPU  — an explicit list of 29 variants (config "cpu_variants"). These are NOT a product:
         `openblas` pairs with 8 CPU levels, but `vulkan`/`sycl`/`clblast`/`opencl`/`rpc`
         are baseline-only. Listing them explicitly avoids generating thousands of
         combinations that never existed (e.g. vulkan_avx512), which would fail the name
         check and re-queue forever.

  CUDA — a product of (cuda_version x cpu_baseline), because those genuinely all exist.
         Local tag is cu<major><minor>_<baseline>, e.g. cu124_basic.

Both are crossed with (llama_version x python_version). Writes QUEUE.md + queue.json.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import sys

from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.abspath(__file__))
PLATFORM = "manylinux_2_31_x86_64"


def load_json(name, default):
    path = os.path.join(ROOT, name)
    if not os.path.exists(path):
        return default
    with open(path) as fh:
        return json.load(fh)


def cuda_tag(version: str) -> str:
    major, minor = version.split(".")[:2]
    return f"cu{major}{minor}"


def py_tag(version: str) -> str:
    return "cp" + version.replace(".", "")


def wheel_name(llama_ver: str, local: str, python: str) -> str:
    base = llama_ver.lstrip("v")
    tag = py_tag(python)
    return f"llama_cpp_python-{base}+{local}-{tag}-{tag}-{PLATFORM}.whl"


def version_key(v: str):
    nums = re.findall(r"\d+", v)
    return tuple(int(n) for n in nums[:3]) if nums else (0, 0, 0)


def build_targets(cfg):
    """Return {wheel_name: job_dict} for the whole configured range."""
    targets = {}
    llamas = cfg["llama_versions"]
    pythons = cfg["python_versions"]

    # ---- CPU wheels: explicit variant list, NOT a product --------------
    for variant, llama, python in itertools.product(cfg.get("cpu_variants", []), llamas, pythons):
        name = wheel_name(llama, variant, python)
        targets[name] = {
            "name": name, "kind": "cpu", "variant": variant,
            "llama": llama, "python": python,
            # CPU builds run in the plain manylinux image, no CUDA.
            "cuda": "", "cpu": variant,
        }

    # ---- CUDA wheels: product of (cuda_version x cpu_baseline) ----------
    for cuda_v, baseline, llama, python in itertools.product(
        cfg.get("cuda_versions", []), cfg.get("cuda_cpu_baselines", []), llamas, pythons
    ):
        local = f"{cuda_tag(cuda_v)}_{baseline}"
        name = wheel_name(llama, local, python)
        targets[name] = {
            "name": name, "kind": "cuda", "variant": local,
            "llama": llama, "python": python,
            "cuda": cuda_v, "cpu": baseline,
        }

    return targets


def sort_key(job):
    """Newest llama first (0.3.19 before 0.3.16 — most-wanted), then CPU before CUDA
    (CPU is cheap and fills the biggest gap), then by variant for stable grouping."""
    return (
        [-n for n in version_key(job["llama"])],   # newest version first
        0 if job["kind"] == "cpu" else 1,           # CPU before CUDA
        job["variant"], job["python"],
    )


def main() -> int:
    cfg = load_json("config.json", {})
    blocked = set(load_json("blocklist.json", []))

    if not cfg.get("enabled", True):
        with open(os.path.join(ROOT, "QUEUE.md"), "w") as fh:
            fh.write("# Factory queue\n\n**STOPPED** — `enabled` is false in config.json.\n")
        out = os.environ.get("GITHUB_OUTPUT")
        if out:
            with open(out, "a") as fh:
                fh.write('matrix={"include":[]}\n')
                fh.write("count=0\n")
                fh.write("remaining=0\n")
        print("factory disabled via config.json")
        return 0

    dataset = cfg.get("dataset", "AIencoder/llama-cpp-wheels")
    batch_size = int(cfg.get("batch_size", 20))

    target = build_targets(cfg)

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    existing = {f.split("/")[-1] for f in api.list_repo_files(dataset, repo_type="dataset")
                if f.endswith(".whl")}

    done = [n for n in target if n in existing]
    todo = [n for n in target if n not in existing and n not in blocked]
    skipped = [n for n in target if n in blocked and n not in existing]

    todo.sort(key=lambda n: sort_key(target[n]))
    batch = todo[:batch_size]

    matrix = {"include": [
        {"name": target[n]["name"], "kind": target[n]["kind"],
         "variant": target[n]["variant"],
         "llama": target[n]["llama"], "python": target[n]["python"],
         "cuda": target[n]["cuda"], "cpu": target[n]["cpu"]}
        for n in batch
    ]}

    # ---- board ---------------------------------------------------------
    total = len(target)
    pct = 100 * len(done) / total if total else 0
    filled = int(pct // 4)
    n_cpu = sum(1 for n in target if target[n]["kind"] == "cpu")
    n_cuda = total - n_cpu
    todo_cpu = sum(1 for n in todo if target[n]["kind"] == "cpu")
    todo_cuda = len(todo) - todo_cpu

    lines = [
        "# Factory queue",
        "",
        "Regenerated every run. Do not edit — change `config.json`.",
        "",
        f"`[{'#' * filled}{'.' * (25 - filled)}]` **{len(done)} / {total}** ({pct:.1f}%)",
        "",
        f"- built: **{len(done)}**",
        f"- remaining: **{len(todo)}**  ({todo_cpu} CPU, {todo_cuda} CUDA)",
        f"- blocked: **{len(skipped)}**" + ("  (see `blocklist.json`)" if skipped else ""),
        "",
        "## Range",
        "",
        f"- llama-cpp-python: {', '.join(cfg['llama_versions'])}",
        f"- Python: {', '.join(cfg['python_versions'])}",
        f"- CPU variants: {n_cpu} wheels across {len(cfg.get('cpu_variants', []))} variants",
        f"- CUDA: {len(cfg.get('cuda_versions', []))} toolkit(s) x "
        f"{len(cfg.get('cuda_cpu_baselines', []))} baseline(s) = {n_cuda} wheels",
        "",
        f"## Next batch ({len(batch)})",
        "",
    ]
    lines += [f"- `{n}`" for n in batch] or ["_nothing left to build_"]

    rest = todo[batch_size:]
    if rest:
        lines += ["", f"## Still queued ({len(rest)})", "", "<details><summary>show</summary>", ""]
        lines += [f"- `{n}`" for n in rest[:400]]
        if len(rest) > 400:
            lines.append(f"- _...and {len(rest) - 400} more_")
        lines += ["", "</details>"]

    if skipped:
        lines += ["", f"## Blocked ({len(skipped)})", ""] + [f"- `{n}`" for n in skipped]

    with open(os.path.join(ROOT, "QUEUE.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    with open(os.path.join(ROOT, "queue.json"), "w") as fh:
        json.dump({"total": total, "built": len(done), "remaining": len(todo),
                   "remaining_cpu": todo_cpu, "remaining_cuda": todo_cuda,
                   "blocked": sorted(skipped), "next_batch": batch,
                   "queued": rest}, fh, indent=2)

    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as fh:
            fh.write(f"matrix={json.dumps(matrix)}\n")
            fh.write(f"count={len(batch)}\n")
            fh.write(f"remaining={len(todo)}\n")

    print(f"target {total} ({n_cpu} CPU + {n_cuda} CUDA) | built {len(done)} | "
          f"remaining {len(todo)} | blocked {len(skipped)}")
    print(f"dispatching {len(batch)}:")
    for n in batch:
        print("   ", target[n]["kind"], n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
