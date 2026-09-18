#!/usr/bin/env python3
"""
Restore underscores in a wheel's local-version tag.

PEP 440 normalization turns every `_` in a local version into `.`, so a wheel stamped
`0.3.16+cu124_basic` is written to disk as `...+cu124.basic-...whl`, and
`0.3.16+openblas_avx2_fma_f16c` becomes `...+openblas.avx2.fma.f16c-...`. The existing
4,382 wheels in the dataset use underscores, so to stay consistent every new wheel has to
be converted back.

This can't be fixed in __version__ — packaging normalizes it no matter what. The only
route is to rewrite the built wheel: rename its .dist-info, fix the version recorded in
METADATA and RECORD, and rename the file. `wheel pack` regenerates RECORD hashes from the
tree, so we unpack, edit, and repack.

Usage:  python fix_wheel_name.py <wheel> <output_dir>
Prints the final path on success. Idempotent: a wheel already using underscores is copied
through unchanged.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path


def underscore_local(filename: str) -> str:
    """In a wheel filename, replace dots with underscores INSIDE the local-version tag only.

    Wheel name: {distribution}-{version}[+{local}]-{pytag}-{abi}-{platform}.whl
    The local tag sits between the first '+' and the '-' that starts the Python tag.
    Everything else (the release version's own dots, the platform tag) is untouched.
    """
    if "+" not in filename:
        return filename
    head, rest = filename.split("+", 1)
    # rest looks like:  cu124.basic-cp311-cp311-manylinux_2_31_x86_64.whl
    local, _, tail = rest.partition("-")
    return f"{head}+{local.replace('.', '_')}-{tail}"


def fix_wheel(wheel: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target_name = underscore_local(wheel.name)

    # Already correct — nothing to rewrite.
    if target_name == wheel.name:
        dest = out_dir / wheel.name
        if wheel.resolve() != dest.resolve():
            shutil.copy2(wheel, dest)
        return dest

    work = out_dir / "_wheelfix"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    # Unpack.
    subprocess.run([sys.executable, "-m", "wheel", "unpack", str(wheel), "-d", str(work)],
                   check=True, capture_output=True, text=True)
    unpacked = next(work.iterdir())

    # The version as it appears normalized (with dots) and as we want it (underscores).
    # e.g. 0.3.16+cu124.basic  ->  0.3.16+cu124_basic
    m = re.match(r"[^-]+-([^-]+)-", wheel.name)   # distribution-VERSION-...
    norm_version = m.group(1)
    fixed_version = norm_version.split("+")[0] + "+" + norm_version.split("+", 1)[1].replace(".", "_") \
        if "+" in norm_version else norm_version

    # Rename the .dist-info directory (it's named {distribution}-{version}.dist-info).
    distinfo = next(p for p in unpacked.iterdir() if p.name.endswith(".dist-info"))
    new_distinfo = distinfo.parent / distinfo.name.replace(norm_version, fixed_version)
    if new_distinfo != distinfo:
        distinfo.rename(new_distinfo)
    distinfo = new_distinfo

    # Fix the Version field in METADATA and the local tag anywhere it's recorded.
    meta = distinfo / "METADATA"
    text = meta.read_text()
    text = re.sub(rf"^Version: {re.escape(norm_version)}$",
                  f"Version: {fixed_version}", text, flags=re.MULTILINE)
    meta.write_text(text)

    # The unpacked top-level dir is also named with the version; wheel pack reads the
    # dist-info to name the output, but rename the tree dir too so pack is unambiguous.
    fixed_tree = unpacked.parent / unpacked.name.replace(norm_version, fixed_version)
    if fixed_tree != unpacked:
        unpacked.rename(fixed_tree)
        unpacked = fixed_tree

    # Repack — regenerates RECORD (including corrected hashes) from the tree.
    subprocess.run([sys.executable, "-m", "wheel", "pack", str(unpacked), "-d", str(out_dir)],
                   check=True, capture_output=True, text=True)

    produced = next(p for p in out_dir.glob("*.whl") if p.name != wheel.name)
    final = out_dir / target_name
    if produced != final:
        if final.exists():
            final.unlink()
        produced.rename(final)

    shutil.rmtree(work)
    return final


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python fix_wheel_name.py <wheel> <output_dir>", file=sys.stderr)
        sys.exit(2)
    wheel_in = Path(sys.argv[1])
    out = fix_wheel(wheel_in, Path(sys.argv[2]))
    print(out)
