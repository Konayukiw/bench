"""Parse the public Artificial Analysis site into a stable JSON snapshot.

The free API requires AA_API_KEY. Model pages and the Progress Index page embed
the same published numbers in the Next.js flight payload, which is what a browser
receives. This module only reads that public payload.
"""

import json
import re

FLIGHT_RE = re.compile(r'self\.__next_f\.push\(\[1,("(?:[^"\\]|\\.)*")\]\)')
VERSION_RE = re.compile(r"Intelligence Index v(\d+\.\d+(?:\.\d+)?)")
FORMAT = "aa_public_v1"


def flight_payload(html):
    chunks = FLIGHT_RE.findall(html)
    if not chunks:
        raise ValueError("page has no Next.js flight payload")
    return "".join(json.loads(c) for c in chunks)


def _arrays(payload, key):
    dec = json.JSONDecoder()
    token = f'"{key}":'
    out = []
    start = 0
    while True:
        i = payload.find(token, start)
        if i < 0:
            break
        j = i + len(token)
        if j < len(payload) and payload[j] == "[":
            try:
                arr, _ = dec.raw_decode(payload, j)
            except json.JSONDecodeError:
                start = j
                continue
            if isinstance(arr, list):
                out.append([x for x in arr if isinstance(x, dict)])
        start = j
    return out


def model_table(payload):
    arrays = _arrays(payload, "models")
    ranked = [a for a in arrays if any("terminalBench40" in m or "intelligenceIndex" in m for m in a)]
    if not ranked:
        raise ValueError("model page has no benchmark table")
    return max(ranked, key=len)


def progress_table(payload):
    arrays = _arrays(payload, "models")
    ranked = [a for a in arrays if any(isinstance(m.get("score"), (int, float)) and "terminalBench40" not in m for m in a)]
    if not ranked:
        raise ValueError("progress page has no score table")
    return max(ranked, key=len)


def index_version(payload):
    m = VERSION_RE.search(payload)
    if not m:
        return None
    return m.group(1)


def _version_number(text):
    if not text:
        return None
    major, minor = text.split(".")[:2]
    return float(f"{int(major)}.{int(minor)}")


def _round(v, n):
    return None if v is None else round(float(v), n)


def build_document(models_html, progress_html, fetched_at, models_url, progress_url):
    model_payload = flight_payload(models_html)
    progress_payload = flight_payload(progress_html)
    version = index_version(model_payload) or index_version(progress_payload)
    models = []
    for m in model_table(model_payload):
        scores = {}
        for field in (
            "livecodebench", "scicode", "terminalbenchHard", "terminalBench21", "terminalBench40",
            "frontierCode", "frontierCode20",
        ):
            v = m.get(field)
            if isinstance(v, (int, float)):
                scores[field] = _round(v, 6)
        models.append({
            "slug": m.get("slug") or "",
            "name": m.get("name") or "",
            "release_date": m.get("releaseDate") or "",
            "intelligence_index": _round(m.get("intelligenceIndex"), 4) if isinstance(m.get("intelligenceIndex"), (int, float)) else None,
            "scores": scores,
        })
    progress = []
    for m in progress_table(progress_payload):
        if not isinstance(m.get("score"), (int, float)):
            continue
        progress.append({
            "slug": m.get("slug") or "",
            "name": m.get("name") or "",
            "release_date": m.get("releaseDate") or "",
            "score": _round(m["score"], 4),
        })
    if len(models) < 100 or len(progress) < 50:
        raise ValueError(f"public tables look truncated (models={len(models)}, progress={len(progress)})")
    return {
        "format": FORMAT,
        "fetched_at": fetched_at,
        "intelligence_index_version": version,
        "intelligence_index_version_number": _version_number(version),
        "coding_index_field": None,
        "sources": [
            {"file": "models.html", "url": models_url},
            {"file": "progress.html", "url": progress_url},
        ],
        "note": (
            "Built from the public website because /api/v2/language/models/free requires an API key. "
            "intelligence_index is the v5 composite on its published point scale, not a 0-1 accuracy. "
            "scores are pass rates or rubric means already on 0-1. Progress Index v1 is a separate "
            "IRT scale anchored at o3-mini (high)=220 and GPT-5.5 (xhigh)=320."
        ),
        "models": models,
        "progress": progress,
    }
