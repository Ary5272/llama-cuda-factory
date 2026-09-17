# llama-cpp-wheels CUDA factory

Builds CUDA `llama-cpp-python` wheels on free GitHub Actions runners and pushes them to
[`AIencoder/llama-cpp-wheels`](https://huggingface.co/datasets/AIencoder/llama-cpp-wheels).

**Keep this repo public.** Standard runners are unmetered on public repos; private repos
burn a 2,000 minute/month quota.

Current queue: **[QUEUE.md](QUEUE.md)** — regenerated on every run.

## How it runs itself

`Factory` fires every six hours (and on demand):

1. **plan** — `plan.py` expands `config.json` into every wheel in range, subtracts what's
   already in the dataset, writes `QUEUE.md` / `queue.json`, and hands the next 20 to the
   build matrix.
2. **build** — those 20 compile in parallel inside `nvidia/cuda:<ver>-devel-ubuntu20.04`,
   get repaired to `manylinux_2_31_x86_64`, and upload.

Leave it alone and it grinds the queue down on its own.

## The queue is derived, not stored

`plan.py` asks the dataset what exists and treats the difference as the work. There is no
job file, no "claimed" status, no "skipped" status. A build that fails simply shows up in
the next batch.

That's a direct fix for how the old Spaces factory died: it kept per-worker job files, and
four MKL wheels got stuck in a state the files disagreed with, looping forever on
`permanently skipped`. Nothing here can enter that state.

Anything genuinely unbuildable goes in `blocklist.json` by hand — visible, and only ever
put there by you.

## Configuring the range

Everything lives in `config.json`:

```json
{
  "llama_versions": ["v0.3.16", "v0.3.17", "v0.3.18", "v0.3.19"],
  "cuda_versions":  ["12.4.1", "12.6.3", "12.8.1", "12.9.1"],
  "cpu_variants":   ["basic", "avx2_fma_f16c", "avx512_fma_f16c"],
  "python_versions": ["3.11", "3.12", "3.13", "3.14"],
  "batch_size": 20
}
```

That's 4 x 4 x 3 x 4 = **192 wheels**. Widen any list and the next run picks up the
difference automatically.

CUDA versions must have an `ubuntu20.04` devel image — 12.4 through 12.9 do, CUDA 13
does not. Ubuntu 20.04 is what pins glibc to 2.31 so wheels tag `manylinux_2_31_x86_64`,
matching the 4,382 CPU wheels already in the dataset.

A word of restraint: 192 wheels is roughly 130 runner-hours. That's ordinary CI usage for
an open-source project. Expanding to all 158 llama versions would be tens of thousands of
hours and would fairly be read as abuse of the free tier.

## Setup

1. Public repo, commit everything here.
2. Settings → Secrets and variables → Actions → new secret `HF_TOKEN`, with write access
   to the dataset. It never appears in a file and Actions masks it in logs.
3. Actions tab → enable workflows.
4. First run: **Run workflow** with `dry_run` checked. You get `QUEUE.md` and no builds.

Scheduled workflows get disabled after 60 days without repository activity, so if the
factory ever goes quiet, that's the first thing to check.

## Why no GPU is needed

`nvcc` is a compiler. It emits PTX and SASS for whatever architectures you name; a card is
only needed to *run* the result. The old factory used `python:3.11-slim`, which had no CUDA
compiler at all — that was the failure, not nvcc itself.

Two things make it work:

- **`devel`**, not `runtime` — only devel images ship `nvcc`.
- **`-DCMAKE_CUDA_ARCHITECTURES=60;61;70;75;80;86;89`** passed explicitly. Omit it and
  cmake interrogates a physical GPU and dies with
  "Failed to detect a default CUDA architecture." `60` is in there so Kaggle's P100 can
  run the output; without it that card gives
  "no kernel image is available for execution on the device."

## Testing what comes out

Building can't tell you a wheel runs. `cuda-wheel-prover.ipynb` on Kaggle installs the
newest wheel, loads a 105 MB GGUF with `n_gpu_layers=-1`, and checks CUDA actually engages
and beats the CPU. Kaggle gives 30 GPU-hours a week; HF ZeroGPU gives 5 minutes a day.

Upstream shipped segfaulting Windows CUDA wheels in 2025 because nothing tested them on
hardware.

## The gap this fills

Upstream's CUDA CI is pinned to Python 3.9–3.12. Nobody currently publishes a cp313 or
cp314 CUDA wheel for this library.
