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

Prebuilt [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python) wheels, gathered to be the
single most complete place to grab one: **10,000+ wheels** covering **Linux, Windows, and macOS**,
every published version from `0.1.x` to `0.3.x`, Python **3.7 – 3.14**, and **every backend this
project can build** — from plain CPU up through AVX-512/AMX, OpenBLAS, MKL, Vulkan, OpenCL, SYCL,
CLBlast, Metal, and **CUDA 12.1 → 12.9**.

It combines two things:

1. **A full mirror of the official upstream wheels** — everything abetlen publishes (`cpu`, `metal`,
   `cu121`–`cu124`) across every version and OS.
2. **Thousands of builds upstream never shipped** — produced here by an automated GitHub Actions
   "wheel factory": newer CUDA versions (12.6 / 12.8 / 12.9), every CPU instruction-set tier, and the
   OpenBLAS / MKL / Vulkan / OpenCL / SYCL / CLBlast / RPC backends, for the latest releases.

> **These are self-contained.** CUDA wheels bundle the CUDA runtime, so you only need an NVIDIA
> **driver** — not the full CUDA toolkit — to use them.

---

## Quick start

Every wheel is a direct download. Point `pip` at its URL:

```bash
pip install "https://huggingface.co/datasets/AIencoder/llama-cpp-wheels/resolve/main/linux/x86_64/cuda12.4/0.3.19/llama_cpp_python-0.3.19+cu124-cp311-cp311-manylinux_2_31_x86_64.whl"
```

Browse the folders (below) to find the one matching your OS, CPU, GPU, Python, and version.

---

## How the collection is organised

Wheels are nested **`os / arch / backend / version /`**:

```
linux/
  x86_64/
    cpu/                         ← plain CPU, no special instructions (max compatibility)
    cpu-avx2-fma-f16c/           ← CPU, AVX2 tier (most modern x86 CPUs)
    cpu-avx512-fma-f16c-vnni-vbmi-bf16-amx/   ← newest Intel/AMD server CPUs
    openblas-avx2-fma-f16c/
    mkl-avx2-fma-f16c/
    vulkan/  opencl/  clblast/  sycl/  rpc/
    cuda12.4/  cuda12.6/  cuda12.8/  cuda12.9/
      0.3.19/
        llama_cpp_python-0.3.19+cu124-cp311-cp311-manylinux_2_31_x86_64.whl
        …one per Python version…
    aarch64/ …     i686/ …
windows/
  x86_64/
    cpu/  cuda12.4/  cuda12.6/  cuda12.8/   (+ upstream-mirrored variants)
      0.3.19/ …
macos/
  arm64/   metal/ …   (Apple Silicon)
  x86_64/  metal/ …   (Intel Macs)
```

**Why this order:** `os` is the hardest constraint (and a clean 3-way split at the root); `arch`
keeps `arm64`/`aarch64`/`i686` wheels from being mixed in with `x86_64`; `backend` is the capability
you actually choose; `version` sits at the leaf, which then holds just the per-Python builds for that
exact combination. Every leaf folder stays tiny, which also keeps the repo under HuggingFace's hard
limit of 10,000 files per directory.

*(Some early wheels may still sit in the repository root during migration — both are valid; use the
file browser or search.)*

---

## Choosing the right wheel

You need to match **four** things. Work down the folder tree:

### 1. OS + architecture
`linux` (manylinux/musllinux), `windows`, or `macos`; then your CPU architecture — `x86_64` (most
desktops/servers, incl. Windows and Intel Macs), `arm64` (Apple Silicon), or `aarch64` (ARM Linux).

### 2. Python version
The `cpXY` tag in the filename is your Python: `cp311` = CPython 3.11. Check with `python --version`.
Pick the wheel whose `cpXY` matches exactly.

### 3. Backend — the important choice

**CPU (no GPU):** pick the highest instruction tier your CPU supports. Going too high causes an
`Illegal instruction` crash; too low just runs slower.

| Folder | Needs | Who it's for |
|--------|-------|--------------|
| `cpu` | nothing | oldest CPUs / max compatibility / VMs |
| `cpu-avx` , `cpu-avx-f16c` | AVX | pre-2013 CPUs |
| `cpu-avx2-fma-f16c` | AVX2 | **most modern desktops/laptops** |
| `cpu-avx2-fma-f16c-avxvnni` | AVX2 + VNNI | Alder Lake+ consumer Intel |
| `cpu-avx512-fma-f16c` | AVX-512 | HEDT / server |
| `cpu-avx512-…-vnni-vbmi` , `…-bf16-amx` | AVX-512 VNNI/VBMI/BF16/AMX | Sapphire Rapids & newer servers |

Check your CPU: Linux `lscpu | grep -o 'avx[^ ]*'` · or `pip install py-cpuinfo && python -c "import cpuinfo,pprint; pprint.pp(cpuinfo.get_cpu_info()['flags'])"` on any OS.

**NVIDIA GPU:** use a `cudaXX.Y` folder. Run `nvidia-smi` — the top-right "CUDA Version: 12.x" is the
**maximum** your driver supports. Pick a folder with that minor version **or lower** (e.g. driver shows
12.6 → `cuda12.4` or `cuda12.6` both work; CUDA 12.x is forward-compatible within the 12 series). You
do **not** need the CUDA toolkit installed — only the driver — because the runtime is bundled.

**Apple GPU:** `metal` (Apple Silicon and Intel Macs with a Metal GPU).

**Other accelerators / BLAS:**

| Folder | What it is |
|--------|-----------|
| `openblas-*` | OpenBLAS-accelerated CPU BLAS (faster prompt processing on CPU) |
| `mkl-*` | Intel MKL BLAS (fastest on Intel CPUs) |
| `vulkan` | cross-vendor GPU (AMD / Intel / NVIDIA via Vulkan) |
| `opencl` / `clblast` | OpenCL GPU acceleration (incl. older/mobile GPUs) |
| `sycl` | Intel GPUs (oneAPI / SYCL) |
| `rpc` | distributed inference across machines |

### 4. Version
`0.3.19` is the newest release built here; older ones are kept for reproducibility. Newer is usually
better unless you're pinning for compatibility.

---

## Filename anatomy

```
llama_cpp_python-0.3.19+cu124-cp311-cp311-manylinux_2_31_x86_64.whl
                 │      │     │     │     └── platform (manylinux… / win_amd64 / macosx…)
                 │      │     └─────┴──────── Python (cp311 = CPython 3.11)
                 │      └── backend tag  (+cu124, +basic_avx2_fma_f16c, +openblas_…, +vulkan; absent = base build)
                 └── llama-cpp-python version
```

The folder name normalises that backend tag into something readable (`cu124` → `cuda12.4`,
`basic_avx512_fma_f16c_vnni_vbmi` → `cpu-avx512-fma-f16c-vnni-vbmi`).

---

## More install examples

```bash
# Linux, modern CPU, no GPU (AVX2):
pip install "https://huggingface.co/datasets/AIencoder/llama-cpp-wheels/resolve/main/linux/x86_64/cpu-avx2-fma-f16c/0.3.19/llama_cpp_python-0.3.19+basic_avx2_fma_f16c-cp311-cp311-manylinux_2_31_x86_64.whl"

# Windows, NVIDIA GPU, CUDA 12.6, Python 3.11:
pip install "https://huggingface.co/datasets/AIencoder/llama-cpp-wheels/resolve/main/windows/x86_64/cuda12.6/0.3.19/llama_cpp_python-0.3.19+cu126-cp311-cp311-win_amd64.whl"

# macOS Apple Silicon, Metal, Python 3.12:
pip install "https://huggingface.co/datasets/AIencoder/llama-cpp-wheels/resolve/main/macos/arm64/metal/0.3.4/llama_cpp_python-0.3.4-cp312-cp312-macosx_11_0_arm64.whl"
```

### Programmatic download

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="AIencoder/llama-cpp-wheels", repo_type="dataset",
    filename="linux/x86_64/cuda12.4/0.3.19/llama_cpp_python-0.3.19+cu124-cp311-cp311-manylinux_2_31_x86_64.whl",
)
# then: pip install <path>
```

List everything with `HfApi().list_repo_files("AIencoder/llama-cpp-wheels", repo_type="dataset")`.

---

## Coverage

| | Linux (`x86_64`, plus some `aarch64`/`i686`) | Windows (`x86_64`) | macOS (`arm64` + `x86_64`) |
|---|---|---|---|
| **CPU** | plain + every AVX/AVX2/AVX-512 tier up to VNNI·VBMI·BF16·AMX | upstream mirror | upstream mirror |
| **BLAS** | OpenBLAS, MKL (every tier) | upstream mirror | — |
| **GPU** | CUDA 12.4 / 12.6 / 12.8 / 12.9, Vulkan, OpenCL, CLBlast, SYCL | CUDA 12.4 / 12.6 / 12.8 | Metal |
| **Other** | RPC | — | — |

- **Python:** cp38 – cp314 for factory builds; the upstream mirror goes back to cp37 / PyPy on old versions.
- **Versions:** factory builds target 0.3.16 – 0.3.19; the mirror covers the full history (0.1.x → 0.3.x).
- **Totals:** 10,000+ wheels and growing. The upstream mirror targets all ~6,800 official wheels; the
  factory adds several thousand more that upstream never produced.

---

## How these wheels are built

The non-mirror wheels are compiled automatically by a GitHub Actions "factory":

- **No GPU required to build** — `nvcc` cross-compiles the CUDA kernels on ordinary CPU runners.
- Built from the official `abetlen/llama-cpp-python` source at the exact tagged version.
- **Self-contained:** Linux wheels are `auditwheel`-repaired; Windows CUDA wheels bundle the CUDA
  runtime DLLs (`cudart`, `cublas`, `cublasLt`) next to `llama.dll`. You need only the GPU driver.
- Each wheel is installed and structurally verified (dependency tree checked) before upload.
- The build queue is **derived** by diffing this dataset against the target matrix, so the collection
  fills in and stays current on its own.

---

## Caveats & troubleshooting

- **`Could not find module 'llama.dll' (or one of its dependencies)` (Windows CUDA):** you don't have an
  NVIDIA driver installed, or it's too old for the CUDA minor version you picked. Install/update the
  driver, or drop to a lower `cudaXX.Y` folder.
- **`Illegal instruction` on import (CPU):** the wheel's instruction tier is higher than your CPU
  supports. Use a lower `cpu-…` folder (when unsure, plain `cpu`).
- **CUDA wheels are large** (~0.6 – 1.3 GB): expected — they bundle the runtime and the fat multi-arch
  CUDA kernels, which is exactly what lets them run without a separate CUDA install.
- **GPU wheels can't be import-tested without a GPU** — that's normal; the build verifies structure, not
  a live GPU load.
- Community builds, provided **as-is** under the same **MIT** license as llama-cpp-python.

---

*Maintained by [AIencoder](https://huggingface.co/AIencoder). Wheels are builds of
[abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python); all credit for llama-cpp-python
and llama.cpp to their authors.*
