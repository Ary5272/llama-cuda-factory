#!/usr/bin/env python3
"""
wheel_layout.py — the single source of truth for where a wheel lives in the dataset:

    os / arch / backend / version / <filename>

Used by BOTH the upload steps (factory.yml, factory_windows.yml, fill_missing.py), so every new
wheel is born in the right folder, AND by sorter.py, so the one-time migration of older wheels
computes exactly the same paths. If you change the layout, change it here only.

    linux/x86_64/cuda12.4/0.3.36/llama_cpp_python-0.3.36+cu124_basic-py3-none-manylinux_2_31_x86_64.whl
    macos/arm64/metal/0.3.4/llama_cpp_python-0.3.4-cp312-cp312-macosx_11_0_arm64.whl
    windows/x86_64/cuda12.6/0.3.36/llama_cpp_python-0.3.36+cu126-py3-none-win_amd64.whl
"""
import re

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
    # if any, stays in the filename (so cu124_basic and cu124_avx2_fma_f16c sit together).
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
    """filename -> (os, arch, backend, version), or None if it isn't a parseable llama-cpp wheel."""
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
        arch = "arm64"                                 # Apple Silicon: aarch64 and arm64 tags are one arch

    tm = _TAG.match(fname)
    tag = tm.group(1) if tm else None
    backend = backend_of(tag)
    if tag is None and os_ == "macos":
        backend = "metal"                              # abetlen's default macOS builds are Metal-enabled
    return os_, arch, backend, version


def target_path(fname):
    """Canonical path for a wheel, or None if the name is unparseable (sorter leaves those alone)."""
    c = classify(fname)
    if c is None:
        return None
    os_, arch, backend, version = c
    return f"{os_}/{arch}/{backend}/{version}/{fname}"


def dest_path(fname):
    """Upload path: canonical if parseable, else the bare name (root) so an odd wheel still lands."""
    return target_path(fname) or fname
