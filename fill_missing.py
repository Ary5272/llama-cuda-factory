#!/usr/bin/env python3
"""
Mirror EVERY upstream llama-cpp-python wheel into AIencoder/llama-cpp-wheels.

Derived queue (same idea as plan.py): each run rebuilds the upstream universe from
abetlen's package indexes, diffs it against what's already in the dataset, and uploads
whatever is missing. Nothing is stored between runs — re-running (or the cron) just
picks up whatever is still missing, so it is safe to run repeatedly until done.

Universe = abetlen indexes {cpu, metal, cu121, cu122, cu123, cu124}. (PyPI publishes no
wheels for this project, only sdists.) This is the full official catalog across all three
OSes and all historical versions (0.1.x -> 0.3.x).

Streaming + size-aware batching keeps runner disk bounded: download a batch, upload it as
ONE commit, delete it, repeat. Commits (not files) are what HF rate-limits (128/hour), so
batching many wheels per commit keeps us well under the limit; a 429 still backs off and
retries. A per-run time budget stops cleanly before the job timeout so the next run can
continue.
"""
import os, re, sys, time, tempfile, shutil
import requests
from huggingface_hub import HfApi, CommitOperationAdd
from huggingface_hub.errors import HfHubHTTPError

REPO = "AIencoder/llama-cpp-wheels"
BACKENDS = ["cpu", "metal", "cu121", "cu122", "cu123", "cu124"]
INDEX = "https://abetlen.github.io/llama-cpp-python/whl/{be}/llama-cpp-python/"

BATCH_FILES = int(os.environ.get("MIRROR_BATCH_FILES", "50"))      # max wheels per commit
BATCH_BYTES = int(os.environ.get("MIRROR_BATCH_BYTES", str(3 * 2**30)))  # or ~3 GB per commit
MAX_WHEELS  = int(os.environ.get("MIRROR_MAX", "100000"))          # per-run cap (optional)
TIME_BUDGET = int(os.environ.get("MIRROR_TIME_BUDGET", "18000"))   # 5h; job timeout is 6h
HREF = re.compile(r'href="([^"]+?\.whl)(?:#[^"]*)?"')

start = time.time()
token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit("HF_TOKEN not set")
api = HfApi(token=token)


def build_universe():
    """filename -> download url, across all backend indexes."""
    uni = {}
    for be in BACKENDS:
        html = requests.get(INDEX.format(be=be), timeout=60).text
        n0 = len(uni)
        for m in HREF.finditer(html):
            url = m.group(1)
            fname = url.split("/")[-1].split("#")[0]
            uni.setdefault(fname, url)
        print(f"  index {be}: +{len(uni)-n0} (running total {len(uni)})", flush=True)
    return uni


def download(url, dest):
    for attempt in range(1, 4):
        try:
            with requests.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(dest, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            return True
        except Exception as e:
            print(f"    download retry {attempt} for {os.path.basename(dest)}: {e}", flush=True)
            time.sleep(3 * attempt)
    return False


def commit(paths):
    if not paths:
        return
    ops = [CommitOperationAdd(path_in_repo=os.path.basename(p), path_or_fileobj=p) for p in paths]
    for attempt in range(1, 11):
        try:
            api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops,
                              commit_message=f"mirror: {len(ops)} upstream wheels")
            return
        except HfHubHTTPError as e:
            if "429" in str(e) and attempt < 10:
                wait = min(180, 20 * attempt)
                print(f"    429 on commit, waiting {wait}s (attempt {attempt})", flush=True)
                time.sleep(wait)
            else:
                raise


def main():
    print("building upstream universe...", flush=True)
    universe = build_universe()
    print(f"universe: {len(universe)} wheels", flush=True)

    print("listing dataset...", flush=True)
    existing = {f for f in api.list_repo_files(REPO, repo_type="dataset") if f.endswith(".whl")}
    print(f"dataset already has: {len(existing)} wheels", flush=True)

    missing = sorted(f for f in universe if f not in existing)
    print(f"missing (to mirror): {len(missing)}", flush=True)
    if not missing:
        print("nothing to do — dataset already mirrors the full upstream catalog.", flush=True)
        return

    tmp = tempfile.mkdtemp(prefix="mirror_")
    uploaded = 0
    batch, batch_bytes = [], 0
    try:
        for fname in missing:
            if uploaded >= MAX_WHEELS:
                print("hit per-run MIRROR_MAX cap", flush=True); break
            if time.time() - start > TIME_BUDGET:
                print("hit per-run time budget — stopping cleanly, next run continues", flush=True); break

            dest = os.path.join(tmp, fname)
            if not download(universe[fname], dest):
                print(f"  skip (download failed): {fname}", flush=True)
                continue
            sz = os.path.getsize(dest)
            batch.append(dest); batch_bytes += sz

            if len(batch) >= BATCH_FILES or batch_bytes >= BATCH_BYTES:
                commit(batch)
                uploaded += len(batch)
                print(f"  committed {len(batch)} wheels ({batch_bytes/2**20:.0f} MB) — {uploaded} this run", flush=True)
                for p in batch:
                    os.remove(p)
                batch, batch_bytes = [], 0

        if batch:
            commit(batch)
            uploaded += len(batch)
            print(f"  committed final {len(batch)} wheels — {uploaded} this run", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    remaining = len(missing) - uploaded
    print(f"\nrun done: uploaded {uploaded} this run, ~{remaining} still missing (cron/next run continues).", flush=True)


if __name__ == "__main__":
    main()
