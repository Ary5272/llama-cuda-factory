---
license: mit
pretty_name: llama-cpp-python Wheels — Complete Collection
tags:
  - llama-cpp-python
  - llama.cpp
  - wheels
  - cuda
  - metal
  - prebuilt
viewer: false
---

# llama-cpp-python Wheels — the complete collection

Prebuilt [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python) wheels, assembled to be
the most complete single place to grab one — **10,000+ wheels** spanning every published upstream
version and every backend this project can build, for **Linux, Windows, and macOS**.

It contains two things:

1. **A full mirror of the official upstream wheels** — abetlen's `cpu`, `metal`, and `cu121`–`cu124`
   releases, across all historical versions (`0.1.x` → `0.3.x`) and all three OSes.
2. **Thousands of additional builds that upstream never shipped** — produced here by an automated
   GitHub Actions "wheel factory": extra CUDA versions, extra CPU instruction-set variants, and
   extra backends (see the table below).

> These are self-contained builds. CUDA wheels bundle the CUDA runtime, so you only need an NVIDIA
> **driver** installed — not the full CUDA toolkit.

## Install a wheel

Every wheel is a direct download. Install one by pointing `pip` at its URL:

```bash
pip install https://huggingface.co/datasets/AIencoder/llama-cpp-wheels/resolve/main/<path-to-wheel>
```

For example, llama-cpp-python 0.3.19 with CUDA 12.4 on Python 3.11 / Windows:

```bash
pip install https://huggingface.co/datasets/AIencoder/llama-cpp-wheels/resolve/main/llama_cpp_python-0.3.19+cu124-cp311-cp311-win_amd64.whl
```

## Reading a filename

```
llama_cpp_python-0.3.19+cu124-cp311-cp311-win_amd64.whl
                 │      │     │     │     └── platform  (win_amd64 / manylinux… / macosx…)
                 │      │     └─────┴──────── Python tag (cp311 = CPython 3.11)
                 │      └── build variant / backend  (see table)
                 └── llama-cpp-python version
```

The part after `+` (the "local version") is the **backend / build variant**. Upstream base wheels
have no `+` tag (plain CPU). Factory builds encode the backend there, e.g. `+cu126`,
`+basic_avx2_fma_f16c`, `+openblas_avx512_fma_f16c_vnni_vbmi`, `+vulkan`, `+metal`.

## What's covered

| OS | Backends | Notes |
|----|----------|-------|
| **Linux** (`manylinux`) | CPU (plain + AVX / AVX2 / AVX-512 up to VNNI·VBMI·BF16·AMX), OpenBLAS, MKL, CLBlast, OpenCL, Vulkan, SYCL, RPC, and **CUDA 12.4 / 12.6 / 12.8 / 12.9** | The broadest coverage — every CPU instruction-set tier so you can match your exact chip |
| **Windows** (`win_amd64`) | Full upstream mirror + **CUDA 12.4 / 12.6 / 12.8** | CUDA wheels self-contained (driver only) |
| **macOS** (`macosx`) | Metal + CPU (upstream mirror) | Apple-Silicon and Intel |

- **Python:** cp38 – cp314 for factory builds; upstream mirror covers whatever upstream shipped (back to cp37 / PyPy on old versions).
- **Versions:** factory builds target the latest releases (0.3.16 – 0.3.19); the mirror covers everything from 0.1.x onward.

## Repository layout

Because HuggingFace caps any single folder at 10,000 files, wheels are organised into **per-version
subfolders** (`0.3.19/`, `0.1.30/`, …). Earlier wheels live in the repository root; both are equally
valid — use the Hub file browser or the filename to find what you need.

## How these are built

The additional wheels are compiled automatically by a GitHub Actions factory (no GPU required — `nvcc`
cross-compiles the CUDA kernels). Each wheel is built from the official `abetlen/llama-cpp-python`
source at the tagged version, repaired to be self-contained, import/structure-checked, and uploaded.
The queue is derived by diffing this dataset against the target matrix, so the collection fills in
and stays current on its own.

## Caveats

- **CUDA wheels need a matching NVIDIA driver** on the machine that runs them (the CUDA runtime is
  bundled; the driver is not).
- **CUDA wheels are large** (~0.6–1.3 GB) because they bundle the runtime and fat CUDA kernels — this
  is expected and is what makes them work without a separate CUDA install.
- These are community builds, provided as-is, under the same MIT license as llama-cpp-python.

---

*Collection maintained by [AIencoder](https://huggingface.co/AIencoder). Wheels are builds of
[abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python).*
