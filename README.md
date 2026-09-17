# llama-cpp-wheels CUDA factory

GPU edition of the wheel factory. Builds CUDA `llama-cpp-python` wheels on free
GitHub Actions runners and pushes them to
[`AIencoder/llama-cpp-wheels`](https://huggingface.co/datasets/AIencoder/llama-cpp-wheels).

**Keep this repository public.** Standard GitHub-hosted runners are free and unmetered
on public repos; private repos burn a 2,000 minute/month quota.

## Setup (once)

1. Create a public repo and commit `.github/workflows/build-cuda-wheels.yml`.
2. Make a **fresh** HF token with write access to the dataset.
3. Repo → Settings → Secrets and variables → Actions → New repository secret,
   named `HF_TOKEN`. The token never appears in the workflow file or in logs.

## Running

Actions tab → *Build CUDA wheels* → Run workflow. Inputs:

| Input | Default | Notes |
|:---|:---|:---|
| `llama_version` | `v0.3.19` | git tag in abetlen/llama-cpp-python |
| `cuda` | `12.4.1` | 12.4.1 / 12.6.3 / 12.8.1 / 12.9.1 — all have ubuntu20.04 devel images |
| `cpu_flags` | `avx2_fma_f16c` | CPU baseline compiled alongside CUDA |
| `upload` | true | uncheck for a dry run; wheels still land as artifacts |

Four wheels per run (Python 3.11–3.14), built in parallel. Output looks like:

```
llama_cpp_python-0.3.19+cu124_avx2_fma_f16c-cp311-cp311-manylinux_2_31_x86_64.whl
```

## Why this works without a GPU

`nvcc` is a compiler, not a driver. It emits PTX/SASS for whatever architectures you
name. The build runs inside `nvidia/cuda:<ver>-devel-ubuntu20.04`:

- **`devel`**, not `runtime` — only the devel images ship `nvcc`. The old factory used
  `python:3.11-slim`, which had no CUDA compiler at all. That was the failure.
- **ubuntu20.04** → glibc 2.31 → `manylinux_2_31_x86_64`, matching the existing wheels.
- **`-DCMAKE_CUDA_ARCHITECTURES=61;70;75;80;86;89`** passed explicitly. Leave it out and
  cmake tries to interrogate a physical GPU and dies with
  "Failed to detect a default CUDA architecture."

CUDA libraries are excluded from the wheel by `auditwheel`, so it stays a few MB and uses
the installing machine's CUDA runtime. Bundling cuBLAS is what makes upstream's Windows
CUDA wheels 428 MB.

## The gap this fills

Upstream's CUDA CI matrix is pinned to Python 3.9–3.12. Nobody currently publishes a
cp313 or cp314 CUDA wheel for this library.

## Not yet wired up

- A ZeroGPU Space to smoke-test each wheel on a real card before publishing. Free accounts
  can host 2 ZeroGPU Spaces and get 5 GPU-minutes a day, which is plenty for
  load-a-tiny-GGUF-and-run-one-token. Upstream shipped segfaulting Windows CUDA wheels in
  2025 precisely because nothing tested them on hardware.
- `cu124` and friends aren't in the wheel-finder's backend table yet.
