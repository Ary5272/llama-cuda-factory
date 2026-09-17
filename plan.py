#!/usr/bin/env python3
"""
Factory dispatcher.

Works out every CUDA wheel in the configured range, subtracts what already exists in
the dataset, and hands the next batch to the build matrix.

The queue is DERIVED, never stored as state. The dataset is the only source of truth:
a wheel exists or it doesn't. That's deliberate — the old Spaces factory kept job files
with "claimed" and "skipped" statuses, and four MKL wheels got permanently stuck in one
because the file disagreed with reality. Here a failed build just reappears in the next
batch. Anything genuinely unbuildable goes in blocklist.json by hand, where you can see it.

Writes QUEUE.md and queue.json as a readable board. Those are outputs, not inputs.
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
    """12.4.1 -> cu124"""
    major, minor = version.split(".")[:2]
    return f"cu{major}{minor}"


def py_tag(version: str) -> str:
    """3.11 -> cp311"""
    return "cp" + version.replace(".", "")


def wheel_name(llama_ver, cuda, cpu, python) -> str:
    base = llama_ver.lstrip("v")
    tag = py_tag(python)
    return f"llama_cpp_python-{base}+{cuda_tag(cuda)}_{cpu}-{tag}-{tag}-{PLATFORM}.whl"


def version_key(v: str):
    nums = re.findall(r"\d+", v)
    return tuple(int(n) for n in nums[:3]) if nums else (0, 0, 0)


def main() -> int:
    cfg = load_json("config.json", {})
    blocked = set(load_json("blocklist.json", []))

    dataset = cfg.get("dataset", "AIencoder/llama-cpp-wheels")
    batch_size = int(cfg.get("batch_size", 20))

    # ---- the full target space -----------------------------------------
    combos = list(itertools.product(
        cfg["llama_versions"], cfg["cuda_versions"],
        cfg["cpu_variants"], cfg["python_versions"],
    ))
    target = {wheel_name(l, c, f, p): dict(llama=l, cuda=c, cpu=f, python=p)
              for l, c, f, p in combos}

    # ---- what already exists -------------------------------------------
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    existing = {f.split("/")[-1] for f in api.list_repo_files(dataset, repo_type="dataset")
                if f.endswith(".whl")}

    done = [n for n in target if n in existing]
    todo = [n for n in target if n not in existing and n not in blocked]
    skipped = [n for n in target if n in blocked and n not in existing]

    # Oldest llama version first, then cheapest CPU variant: finish whole
    # versions rather than scattering half-built ones everywhere.
    cpu_order = {name: i for i, name in enumerate(cfg["cpu_variants"])}
    todo.sort(key=lambda n: (
        version_key(target[n]["llama"]),
        cpu_order.get(target[n]["cpu"], 99),
        target[n]["cuda"],
        target[n]["python"],
    ))

    batch = todo[:batch_size]
    matrix = {"include": [
        {"name": n, "llama": target[n]["llama"], "cuda": target[n]["cuda"],
         "cpu": target[n]["cpu"], "python": target[n]["python"]}
        for n in batch
    ]}

    # ---- the board ------------------------------------------------------
    total = len(target)
    pct = 100 * len(done) / total if total else 0
    filled = int(pct // 4)

    lines = [
        "# Factory queue",
        "",
        "Regenerated on every run. Do not edit — change `config.json` instead.",
        "",
        f"`[{'#' * filled}{'.' * (25 - filled)}]` **{len(done)} / {total}** ({pct:.1f}%)",
        "",
        f"- built: **{len(done)}**",
        f"- remaining: **{len(todo)}**",
        f"- blocked: **{len(skipped)}**" + ("  (see `blocklist.json`)" if skipped else ""),
        "",
        "## Range",
        "",
        f"- llama-cpp-python: {', '.join(cfg['llama_versions'])}",
        f"- CUDA: {', '.join(cfg['cuda_versions'])}",
        f"- CPU baselines: {', '.join(cfg['cpu_variants'])}",
        f"- Python: {', '.join(cfg['python_versions'])}",
        "",
        f"## Next batch ({len(batch)})",
        "",
    ]
    lines += [f"- `{n}`" for n in batch] or ["_nothing left to build_"]

    if todo[batch_size:]:
        rest = todo[batch_size:]
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
                   "blocked": sorted(skipped), "next_batch": batch,
                   "queued": todo[batch_size:]}, fh, indent=2)

    # ---- hand off to Actions -------------------------------------------
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as fh:
            fh.write(f"matrix={json.dumps(matrix)}\n")
            fh.write(f"count={len(batch)}\n")
            fh.write(f"remaining={len(todo)}\n")

    print(f"target {total} | built {len(done)} | remaining {len(todo)} | blocked {len(skipped)}")
    print(f"dispatching {len(batch)}:")
    for n in batch:
        print("   ", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
