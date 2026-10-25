import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
UA = {"User-Agent": "benchmark-research-bot/1.0 (+https://github.com/Konayukiw/bench)"}

FILES = {
    "epoch": [
        "https://epoch.ai/data/eci_scores.csv",
        "https://epoch.ai/data/eci_benchmarks.csv",
        "https://epoch.ai/data/edi_scores.csv",
        "https://epoch.ai/data/benchmark_data.zip",
    ],
    "swebench": ["https://raw.githubusercontent.com/SWE-bench/swe-bench.github.io/master/data/leaderboards.json"],
    "aider": ["https://raw.githubusercontent.com/Aider-AI/aider/main/aider/website/_data/polyglot_leaderboard.yml"],
    "evalplus": ["https://evalplus.github.io/results.json"],
    "bigcodebench": ["https://bigcode-bench.github.io/results.json"],
    "livecodebench": ["https://livecodebench.github.io/performances_generation.json"],
}

AA_BASE = "https://artificialanalysis.ai/api/v2"
SWEREBENCH_URL = "https://swe-rebench.com/"
VALS_BASE = "https://www.vals.ai"
VALS_SLUGS = ["terminal-bench-4", "terminal-bench-2-1", "terminal-bench-2", "swebench", "lcb", "vibe-code", "vcb-1-100", "programbench", "ioi", "code-migration", "skillsbench"]


def aa_key():
    key = os.environ.get("AA_API_KEY")
    if key:
        return key
    if sys.platform == "win32":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                return winreg.QueryValueEx(k, "AA_API_KEY")[0]
        except OSError:
            return None
    return None


def save(dest_dir, name, content, url):
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / name
    path.write_bytes(content)
    return {"file": name, "url": url, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def fetch_files(source, urls, dest_dir):
    entries = []
    for url in urls:
        r = requests.get(url, headers=UA, timeout=120)
        r.raise_for_status()
        entries.append(save(dest_dir, url.rsplit("/", 1)[-1], r.content, url))
        print(f"  {source}: {entries[-1]['file']} {entries[-1]['bytes']:,} bytes")
    return entries


def fetch_aa(dest_dir):
    key = aa_key()
    if not key:
        print("  aa: AA_API_KEY not found, skipped")
        return None
    headers = {**UA, "x-api-key": key}
    for path in ("/language/models", "/language/models/free"):
        pages, page = [], 1
        while True:
            r = requests.get(AA_BASE + path, headers=headers, params={"page": page}, timeout=60)
            if r.status_code in (401, 403) and page == 1:
                break
            if r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", "30")))
                continue
            r.raise_for_status()
            body = r.json()
            pages.append(body)
            if not body.get("pagination", {}).get("has_more"):
                break
            page += 1
        if pages:
            merged = {k: v for k, v in pages[0].items() if k != "data"}
            merged["endpoint"] = path
            merged["data"] = [m for p in pages for m in p.get("data", [])]
            content = json.dumps(merged, ensure_ascii=False, indent=1).encode("utf-8")
            entry = save(dest_dir, "aa_models.json", content, AA_BASE + path)
            print(f"  aa: {len(merged['data'])} models via {path} (remaining quota {r.headers.get('X-RateLimit-Remaining', '?')})")
            return [entry]
    print("  aa: key rejected by both endpoints, skipped")
    return None


def fetch_swerebench(dest_dir):
    r = requests.get(SWEREBENCH_URL, headers=UA, timeout=120)
    r.raise_for_status()
    entry = save(dest_dir, "index.html", r.content, SWEREBENCH_URL)
    print(f"  swerebench: index.html {entry['bytes']:,} bytes")
    return [entry]


def fetch_vals(dest_dir):
    entries = []
    for slug in VALS_SLUGS:
        url = f"{VALS_BASE}/benchmarks/{slug}"
        r = requests.get(url, headers=UA, timeout=60)
        if r.status_code == 404:
            print(f"  vals: {slug} not found, skipped")
            continue
        r.raise_for_status()
        entries.append(save(dest_dir, f"{slug}.html", r.content, url))
        m = re.search(r"/_astro/benchmark_view_[A-Za-z0-9_]+\.[A-Za-z0-9_-]+\.json", r.text)
        if not m:
            print(f"  vals: {slug} has no benchmark view json, skipped")
            continue
        jr = requests.get(VALS_BASE + m.group(0), headers=UA, timeout=60)
        jr.raise_for_status()
        entries.append(save(dest_dir, f"{slug}.json", jr.content, VALS_BASE + m.group(0)))
        print(f"  vals: {slug} {len(jr.content):,} bytes")
        time.sleep(0.5)
    return entries


def write_manifest(dest, entries):
    (dest / "_fetch.json").write_text(json.dumps({"fetched_at": dt.datetime.now().isoformat(timespec="seconds"), "files": entries}, indent=1), encoding="utf-8")


def main():
    only = set(sys.argv[1:])
    stamp = dt.date.today().isoformat()
    for source, urls in FILES.items():
        if only and source not in only:
            continue
        dest = RAW / source / stamp
        write_manifest(dest, fetch_files(source, urls, dest))
    for source, fn in (("aa", fetch_aa), ("swerebench", fetch_swerebench), ("vals", fetch_vals)):
        if only and source not in only:
            continue
        dest = RAW / source / stamp
        entries = fn(dest)
        if entries:
            write_manifest(dest, entries)


if __name__ == "__main__":
    main()
