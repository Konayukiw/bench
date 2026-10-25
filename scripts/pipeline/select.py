import math
import statistics
from collections import defaultdict

from . import config


def logit(p):
    p = min(max(p, 0.01), 0.99)
    return math.log(p / (1.0 - p))


def is_vendor_source(vendor, text):
    text = text.lower()
    return any(d in text for d in config.VENDOR_DOMAINS.get(vendor, []))


def attach_models(rows, matcher, models_by_name):
    out = []
    for r in rows:
        model, stage = matcher.match(r["source"], r["name"])
        if not model:
            continue
        r = dict(r, model=model, match=stage)
        if r["priority"] != config.P_SELF and is_vendor_source(models_by_name[model]["vendor"], r["url"] + " " + r["source_text"]):
            r["priority"] = config.P_SELF
        r["group"] = config.GROUP_OF_PRIORITY[r["priority"]]
        out.append(r)
    return out


def pick(cands):
    best = min(c["priority"] for c in cands)
    return max((c for c in cands if c["priority"] == best), key=lambda c: c["value"])


def select(rows):
    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["bench"])].append(r)
    chosen = {}
    for key, cands in by.items():
        c = pick(cands)
        chosen[key] = dict(c, n_candidates=len(cands), sources=sorted({x["source"] for x in cands}))
    return chosen, by


def estimate_offsets(by):
    diffs = defaultdict(list)
    has_standard = set()
    for (model, bench), cands in by.items():
        if config.spec(bench).get("kind") == "elo":
            continue
        standard = [c for c in cands if c["group"] == "standard"]
        measured = [c for c in cands if c["group"] in config.MEASURED_GROUPS]
        if standard:
            has_standard.add(bench)
            ref = logit(max(c["value"] for c in standard))
            board = [c["value"] for c in cands if c["group"] == "leaderboard"]
            if board:
                diffs[(bench, "leaderboard")].append(logit(max(board)) - ref)
        if not measured:
            continue
        ref = logit(pick(measured)["value"])
        for g in ("unknown", "self"):
            vals = [c["value"] for c in cands if c["group"] == g]
            if vals:
                diffs[(bench, g)].append(logit(max(vals)) - ref)
    pooled = {"leaderboard": 0.0}
    for g in ("unknown", "self"):
        allv = [d for (b, gg), v in diffs.items() if gg == g for d in v]
        pooled[g] = statistics.median(allv) if allv else 0.0
    offsets = {"standard": defaultdict(float)}
    detail = []
    for g in ("leaderboard", "unknown", "self"):
        table = defaultdict(lambda g=g: pooled[g])
        for (b, gg), v in diffs.items():
            if gg != g:
                continue
            n = len(v)
            med = statistics.median(v)
            table[b] = (n * med + config.OFFSET_SHRINK_K * pooled[g]) / (n + config.OFFSET_SHRINK_K)
            detail.append({"benchmark": b, "group": g, "n_pairs": n, "median_logit_gap": round(med, 4), "applied": round(table[b], 4)})
        offsets[g] = table
    return offsets, pooled, sorted(detail, key=lambda d: (d["group"], -d["n_pairs"]))
