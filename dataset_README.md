---
license: mit
pretty_name: llama-cpp-python Wheels — Complete Collection
tags:
  - llama-cpp-python
  - llama.cpp
  - wheels
  - prebuilt
  - cuda
  - metal
  - vulkan
viewer: false
---

# llama-cpp-python Wheels — the complete collection

[![Model Pulse](https://modelpulse.ifsp.dev/badge/dataset/AIencoder/llama-cpp-wheels.svg)](https://huggingface.co/spaces/tardellirs/model-pulse?dataset=AIencoder/llama-cpp-wheels)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Prebuilt [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python) wheels, gathered to be the
single most complete place to grab one: **over 11,400 wheels and growing**, covering **Linux, Windows,
and macOS**, every published release from `0.1.x` through `0.3.36`, and **every backend the project can
build** — plain CPU up through AVX-512 / AMX, OpenBLAS, MKL, Vulkan, OpenCL, CLBlast, SYCL, RPC, Metal,
and **CUDA 12.1 → 12.9**.

It is two things in one repo:

1. **A full mirror of the official upstream wheels** — everything abetlen publishes (`cpu`, `metal`,
   and `cu121`–`cu125`), across every version and OS, back to the early `0.1.x` releases.
2. **Thousands of builds upstream never shipped** — produced here by an automated GitHub Actions "wheel
   factory": newer CUDA toolkits (**12.6 / 12.8 / 12.9**), every CPU instruction-set tier, and the
   OpenBLAS / MKL / Vulkan / OpenCL / CLBlast / SYCL / RPC backends, built for the **whole 0.3.x series**.

> **Self-contained.** CUDA wheels bundle the CUDA runtime — you only need an NVIDIA **driver** installed,
> not the full CUDA toolkit.

---

## At a glance

| | |
|---|---|
| **Wheels** | 11,400+ and climbing |
| **OSes** | Linux (`manylinux`/`musllinux`), Windows (`win_amd64`), macOS (`arm64` + `x86_64`) |
| **Versions** | full history `0.1.x` → `0.3.36` |
| **Python** | `cp37`–`cp314` + PyPy on older releases; a single `py3-none` wheel per platform from `0.3.20` on (see below) |
| **CPU** | `basic` → `avx` → `avx2` → `avx512` → `…vnni/vbmi/bf16/amx`, plus OpenBLAS & MKL at every tier |
| **GPU** | CUDA **12.1 – 12.9**, Vulkan, OpenCL, CLBlast, SYCL, Metal |
| **License** | MIT (same as llama-cpp-python) |

---

## ⚠️ One thing that trips people up: `py3-none` wheels (v0.3.20+)

Starting at **v0.3.20**, llama-cpp-python's bindings load the compiled library through `ctypes` — there is
no CPython ABI to vary — so upstream (and this repo) ship **one version-agnostic wheel per platform**
instead of one per Python version. That wheel is tagged **`py3-none`** and works on **any** CPython 3.x
(and PyPy):

```
llama_cpp_python-0.3.36+cu124_basic-py3-none-manylinux_2_31_x86_64.whl   ← one wheel, every Python
```

For releases **≤ 0.3.19** you still pick the wheel whose `cpXY` tag matches your Python:

```
llama_cpp_python-0.3.19+cu124-cp311-cp311-manylinux_2_31_x86_64.whl      ← cp311 = CPython 3.11
```

So: **new versions → ignore Python, just match OS + backend. Old versions → also match `cpXY`.**

---

## Quick start

Every wheel is a plain file you can `pip install` from its URL. The most robust way to grab the right one
(works no matter how the folders are arranged) is to let `huggingface_hub` find it by a substring and
install it:

```python
# pip install huggingface_hub
from huggingface_hub import HfApi, hf_hub_download
import subprocess, sys

REPO = "AIencoder/llama-cpp-wheels"
WANT = "0.3.36+cu124"          # version + backend you want
PLAT = "manylinux"             # "manylinux" | "win_amd64" | "macosx"

f = next(p for p in HfApi().list_repo_files(REPO, repo_type="dataset")
         if p.endswith(".whl") and WANT in p and PLAT in p)
whl = hf_hub_download(REPO, f, repo_type="dataset")
subprocess.run([sys.executable, "-m", "pip", "install", whl])
```

Prefer a direct `pip install`? Open the **Files** tab, find your wheel, copy its download link, and:

```bash
pip install "https://huggingface.co/datasets/AIencoder/llama-cpp-wheels/resolve/main/<path-to>/<wheel>.whl"
```

---

## Choosing the right wheel

Match **four** things — OS, (Python), backend, version:

### 1. OS + architecture
`linux` (manylinux/musllinux), `windows` (`win_amd64`), or `macos`; then your CPU architecture —
`x86_64` (most desktops/servers, incl. Windows and Intel Macs), `arm64` (Apple Silicon), or
`aarch64` (ARM Linux).

### 2. Python — only for releases ≤ 0.3.19
Releases `0.3.20+` are `py3-none` (one wheel, any Python — skip this step). For older releases, match the
`cpXY` tag to your interpreter: `cp311` = CPython 3.11 (`python --version`).

### 3. Backend — the important choice

**CPU (no GPU).** Pick the highest instruction tier your CPU supports. Too high → `Illegal instruction`
on import; too low → just slower.

| Backend tag | Needs | Who it's for |
|---|---|---|
| `basic` | nothing | oldest CPUs / VMs / max compatibility |
| `basic_avx`, `basic_avx_f16c` | AVX | pre-2013 CPUs |
| `basic_avx2_fma_f16c` | AVX2 | **most modern desktops/laptops** |
| `basic_avx2_fma_f16c_avxvnni` | AVX2 + VNNI | Alder Lake+ consumer Intel |
| `basic_avx512_fma_f16c` | AVX-512 | HEDT / server |
| `…_avx512_fma_f16c_vnni_vbmi`, `…_bf16_amx` | AVX-512 VNNI/VBMI/BF16/AMX | Sapphire Rapids & newer servers |

Check your CPU: Linux `lscpu | grep -o 'avx[^ ]*'`; any OS `pip install py-cpuinfo && python -c "import cpuinfo;print(cpuinfo.get_cpu_info()['flags'])"`.

**NVIDIA GPU.** Use a `cu1XY` wheel. Run `nvidia-smi`; the "CUDA Version: 12.x" shown top-right is the
**maximum** your driver supports — pick that minor **or lower** (CUDA 12.x is forward-compatible within the
12 series, so a 12.6 driver runs `cu121`…`cu126`). You need only the **driver**, not the toolkit.

**Apple GPU.** `metal` (Apple Silicon and Intel Macs with a Metal GPU).

**Other accelerators / BLAS.**

| Tag | What it is |
|---|---|
| `openblas_*` | OpenBLAS CPU BLAS (faster prompt processing) |
| `mkl_*` | Intel MKL BLAS (fastest on Intel CPUs) |
| `vulkan` | cross-vendor GPU (AMD / Intel / NVIDIA via Vulkan) |
| `opencl` / `clblast` | OpenCL GPU acceleration (incl. older/mobile GPUs) |
| `sycl` | Intel GPUs (oneAPI / SYCL) |
| `rpc` | distributed inference across machines |

### 4. Version
`0.3.36` is the newest built here; everything older is kept for reproducibility. Newer is usually better
unless you're pinning for compatibility.

---

## Filename anatomy

```
llama_cpp_python-0.3.36+cu124_basic-py3-none-manylinux_2_31_x86_64.whl
                 │      │            │        └── platform (manylinux… / win_amd64 / macosx…)
                 │      │            └── Python: py3-none (any Python, v0.3.20+) or cpXY-cpXY (≤0.3.19)
                 │      └── backend tag (+cu124_basic, +basic_avx2_fma_f16c, +openblas_…, +vulkan, +metal; absent = plain CPU/upstream base)
                 └── llama-cpp-python version
```

The backend tag is normalised into the readable folder names below (`cu124` → `cuda12.4`,
`basic_avx512_fma_f16c_vnni_vbmi` → `cpu-avx512-fma-f16c-vnni-vbmi`).

---

## How the collection is organised

Wheels are being filed under **`os / arch / backend / version /`**:

```
linux/
  x86_64/
    cpu/                         ← plain CPU (max compatibility)
    cpu-avx2-fma-f16c/           ← AVX2 tier (most modern x86)
    cpu-avx512-fma-f16c-vnni-vbmi-bf16-amx/   ← newest Intel/AMD servers
    openblas-avx2-fma-f16c/   mkl-avx2-fma-f16c/
    vulkan/  opencl/  clblast/  sycl/  rpc/
    cuda12.4/  cuda12.6/  cuda12.8/  cuda12.9/
      0.3.36/
        llama_cpp_python-0.3.36+cu124_basic-py3-none-manylinux_2_31_x86_64.whl
        …+cu124_avx2_fma_f16c…, …+cu124_avx512_fma_f16c…   (CPU baseline is in the filename)
    aarch64/ …     i686/ …
windows/
  x86_64/
    cpu/  cuda12.4/  cuda12.6/  cuda12.8/   (+ upstream-mirrored variants)
      0.3.36/ …
macos/
  arm64/   metal/ …   (Apple Silicon)
  x86_64/  metal/ …   (Intel Macs)
```

**Why this order:** `os` is the hardest constraint (a clean 3-way split); `arch` keeps `arm64`/`aarch64`/
`i686` from mixing with `x86_64`; `backend` is the capability you actually choose; `version` is the leaf,
so each leaf folder holds only a handful of wheels and never approaches HuggingFace's 10,000-files-per-dir
limit. CPU wheels split by instruction tier (their primary axis); CUDA wheels group by toolkit version
(their primary axis) with the CPU baseline carried in the filename.

> **Fully organised.** Every wheel lives under `os/arch/backend/version/`, and new builds are written there
> automatically as they're produced. If you ever can't spot one by browsing, the `huggingface_hub` snippet
> above and the **Files** tab search locate it by name regardless of folder.

### Programmatic listing

```python
from huggingface_hub import HfApi
files = HfApi().list_repo_files("AIencoder/llama-cpp-wheels", repo_type="dataset")
cu124 = [f for f in files if f.endswith(".whl") and "+cu124" in f]
```

---

## Coverage

| | Linux (`x86_64`, some `aarch64`/`i686`) | Windows (`x86_64`) | macOS (`arm64` + `x86_64`) |
|---|---|---|---|
| **CPU** | plain + every AVX/AVX2/AVX-512 tier to VNNI·VBMI·BF16·AMX | upstream mirror | upstream mirror |
| **BLAS** | OpenBLAS, MKL (every tier) | upstream mirror | — |
| **GPU** | CUDA 12.4/12.6/12.8/12.9, Vulkan, OpenCL, CLBlast, SYCL | CUDA 12.4/12.6/12.8 | Metal |
| **Other** | RPC | — | — |

- **Python:** `py3-none` (one wheel, any Python) for `0.3.20+`; `cp38`–`cp314` for the factory's `≤0.3.19`
  builds; the upstream mirror reaches back to `cp37` / PyPy on old releases.
- **Versions:** the factory builds the **entire `0.3.0`–`0.3.36` series**; the mirror covers the full
  history (`0.1.x` → `0.3.x`).
- **CUDA span:** `cu121`–`cu125` from the upstream mirror, plus `cu124`/`cu126`/`cu128`/`cu129` built here.

---

## How these wheels are built

The non-mirror wheels are compiled automatically by a GitHub Actions "factory":

- **No GPU required to build** — `nvcc` cross-compiles the CUDA kernels on ordinary CPU runners, for GPU
  architectures `sm_50`–`sm_90` (Maxwell through Hopper).
- Built from the official `abetlen/llama-cpp-python` source at the exact tagged version.
- **Self-contained:** Linux wheels are `auditwheel`-repaired; Windows CUDA wheels bundle the CUDA runtime
  DLLs next to `llama.dll`. You need only the GPU driver.
- Each wheel is installed and its import is exercised (with the CUDA driver stub, so a GPU-less runner still
  loads it) before upload — catching missing runtime deps and bad symbols, not just a successful compile.
- The build queue is **derived** by diffing this dataset against the target matrix every run, so the
  collection fills in and stays current on its own.

---

## Troubleshooting

- **`Could not find module 'llama.dll' (or one of its dependencies)` (Windows CUDA)** — no NVIDIA driver, or
  it's too old for the CUDA minor you picked. Install/update the driver, or drop to a lower `cu1XY` wheel.
- **`libcuda.so.1: cannot open shared object file` (Linux CUDA)** — same thing: install the NVIDIA driver
  (the wheel bundles the *runtime* but never the driver), or pick a lower CUDA minor.
- **`Illegal instruction` on import (CPU)** — the wheel's instruction tier is above what your CPU supports.
  Use a lower `cpu-…` tier (when unsure, plain `cpu`).
- **CUDA wheels are large (~0.5 – 3 GB)** — expected: they bundle the runtime and fat multi-arch kernels,
  which is exactly what lets them run with no separate CUDA install.
- **GPU wheels can't be import-tested on a GPU-less machine beyond structure** — normal; a real GPU load is
  only exercised at model-load time on your hardware.
- Community builds, provided **as-is** under the same **MIT** license as llama-cpp-python.

---

*Maintained by [AIencoder](https://huggingface.co/AIencoder). Wheels are builds of
[abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python); all credit for llama-cpp-python and
llama.cpp to their authors.*
