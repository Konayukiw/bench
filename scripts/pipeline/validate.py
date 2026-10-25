import csv
from itertools import combinations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from scipy.stats import spearmanr

from . import config, sources
from .select import pick

GAP_FRACTION = 0.10
Z_SIGNIFICANT = 1.96
TARGET_RATE = 0.05


def separated(va, vb, sa, sb, thr):
    if thr <= 0 or abs(va - vb) < thr:
        return False
    if sa is not None and sb is not None and (sa > 0 or sb > 0):
        return abs(va - vb) > Z_SIGNIFICANT * (sa ** 2 + sb ** 2) ** 0.5
    return True


def inversions(values, scores, ses=None, limit=3):
    ses = ses or {}
    items = [(m, v) for m, v in values.items() if m in scores]
    if len(items) < config.MIN_MODELS_PER_BENCH:
        return None
    vals = [v for _, v in items]
    thr = GAP_FRACTION * (max(vals) - min(vals))
    pairs = inv = 0
    worst = []
    for (a, va), (b, vb) in combinations(items, 2):
        if not separated(va, vb, ses.get(a), ses.get(b), thr):
            continue
        pairs += 1
        if (va - vb) * (scores[a] - scores[b]) < 0:
            inv += 1
            worst.append((abs(va - vb), a, va, b, vb))
    worst.sort(reverse=True)
    return {"n_models": len(items), "pairs": pairs, "inversions": inv, "rate": inv / pairs if pairs else 0.0,
            "examples": [{"higher_on_benchmark": a if va > vb else b, "lower_on_benchmark": b if va > vb else a,
                          "values": [round(max(va, vb), 4), round(min(va, vb), 4)]} for _, a, va, b, vb in worst[:limit]]}


def load_old_theta():
    if not config.PREVIOUS_CSV.exists():
        return {}
    with config.PREVIOUS_CSV.open(encoding="utf-8") as f:
        return {r["model"]: float(r["theta"]) for r in csv.DictReader(f) if r.get("theta")}


def epoch_swe_refit(processed, edi):
    anchors = {spec["anchor"] for spec in config.BENCHMARKS.values() if spec.get("anchor")}
    out = {}
    sub = processed[processed.benchmark.isin(anchors)]
    for model, g in sub.groupby("Model"):
        D = np.array([edi[b]["edi"] for b in g.benchmark])
        a = np.array([edi[b]["slope"] for b in g.benchmark])
        y = g.performance.to_numpy()
        f = lambda c: float(np.sum((y - expit(a * (c - D))) ** 2))
        out[model] = (minimize_scalar(f, bounds=(30.0, 280.0), method="bounded").x, len(g))
    return out


def swerebench_common_windows(matcher, scores, min_days=60):
    _, dates, items = sources.swerebench_items()
    best = {}
    for it in items:
        m, _ = matcher.match("swerebench", sources.swerebench_name(it["modelName"]))
        if not m or m not in scores:
            continue
        rank = (it.get("agentVersion") == "tools", it["taskRangeTimestamp"]["to"] - it["taskRangeTimestamp"]["from"])
        if m not in best or rank > best[m][0]:
            best[m] = (rank, it)
    comps = []
    for a, b in combinations(sorted(best), 2):
        ia, ib = best[a][1], best[b][1]
        if ia.get("agentVersion") != ib.get("agentVersion"):
            continue
        lo = max(ia["taskRangeTimestamp"]["from"], ib["taskRangeTimestamp"]["from"])
        hi = min(ia["taskRangeTimestamp"]["to"], ib["taskRangeTimestamp"]["to"])
        lo = min((t for t in dates if t >= lo), default=None)
        hi = max((t for t in dates if t <= hi), default=None)
        if lo is None or hi is None or hi - lo < min_days * 86400000:
            continue
        sa, sb = ia["rangeStats"]["all"].get(f"{lo}:{hi}"), ib["rangeStats"]["all"].get(f"{lo}:{hi}")
        if sa and sb and sa.get("resolvedRate") is not None and sb.get("resolvedRate") is not None:
            comps.append((a, sa["resolvedRate"], b, sb["resolvedRate"], sa.get("sem"), sb.get("sem")))
    if not comps:
        return None
    rates = [x for c in comps for x in (c[1], c[3])]
    thr = GAP_FRACTION * (max(rates) - min(rates))
    pairs = inv = 0
    worst = []
    for a, ra, b, rb, sa, sb in comps:
        if not separated(ra, rb, sa, sb, thr):
            continue
        pairs += 1
        if (ra - rb) * (scores[a] - scores[b]) < 0:
            inv += 1
            worst.append((abs(ra - rb), a, ra, b, rb))
    worst.sort(reverse=True)
    return {"n_models": len(best), "pairs": pairs, "inversions": inv, "rate": inv / pairs if pairs else 0.0,
            "examples": [{"higher_on_benchmark": a if ra > rb else b, "lower_on_benchmark": b if ra > rb else a,
                          "values": [round(max(ra, rb) / 100, 4), round(min(ra, rb) / 100, 4)]} for _, a, ra, b, rb in worst[:3]]}


def run(result):
    prob, x_hat, by = result["prob"], result["x_hat"], result["by"]
    C = dict(zip(prob.models, x_hat[: prob.nM]))
    old = load_old_theta()
    per_bench = {}
    tot = {"new": [0, 0], "old": [0, 0]}
    benches = sorted({b for (_, b) in by if not config.spec(b).get("share_group")})
    for bench in benches:
        values, ses = {}, {}
        for (m, b), cands in by.items():
            if b != bench:
                continue
            measured = [c for c in cands if c["group"] in config.MEASURED_GROUPS]
            if measured:
                p = pick(measured)
                values[m], ses[m] = p["value"], p.get("se")
        new = inversions(values, C, ses)
        if not new:
            continue
        common = {m: v for m, v in values.items() if m in C and m in old}
        entry = {"role": config.spec(bench).get("role", "fit"), "new": new}
        o = inversions(common, old, ses) if old else None
        n = inversions(common, C, ses) if old else None
        if o and n:
            entry["same_pairs"] = {"n_models": o["n_models"], "pairs": o["pairs"], "old_rate": o["rate"], "new_rate": n["rate"]}
            tot["old"][0] += o["inversions"]
            tot["old"][1] += o["pairs"]
            tot["new"][0] += n["inversions"]
            tot["new"][1] += n["pairs"]
        per_bench[bench] = entry
    swr = swerebench_common_windows(result["matcher"], C)
    if swr:
        entry = {"role": "fit", "new": swr, "note": "pairwise on each pair's common task window (>= 60 days), as the leaderboard shows when that window is selected"}
        if old:
            o = swerebench_common_windows(result["matcher"], {m: v for m, v in old.items() if m in C})
            n = swerebench_common_windows(result["matcher"], {m: C[m] for m in old if m in C})
            if o and n:
                entry["same_pairs"] = {"n_models": o["n_models"], "pairs": o["pairs"], "old_rate": o["rate"], "new_rate": n["rate"]}
                tot["old"][0] += o["inversions"]
                tot["old"][1] += o["pairs"]
                tot["new"][0] += n["inversions"]
                tot["new"][1] += n["pairs"]
        per_bench["SWE-rebench (common windows)"] = entry
    overall_new = [0, 0]
    for e in per_bench.values():
        overall_new[0] += e["new"]["inversions"]
        overall_new[1] += e["new"]["pairs"]

    matcher = result["matcher"]
    eci_scores = result["eci_scores"]
    eci_map = {}
    for r in eci_scores.itertuples():
        m, _ = matcher.match("epoch", r.Model)
        if m and m in C:
            eci_map.setdefault(m, []).append(r.eci)
    common = sorted(eci_map)
    rho_eci = spearmanr([max(eci_map[m]) for m in common], [C[m] for m in common]).statistic if len(common) > 5 else None

    swe = epoch_swe_refit(result["processed"], result["edi"])
    swe_map = {}
    for model, (c, n) in swe.items():
        m, _ = matcher.match("epoch", model)
        if m and m in C and n >= 2:
            swe_map[m] = max(swe_map.get(m, -1e9), c)
    common_swe = sorted(swe_map)
    rho_swe = spearmanr([swe_map[m] for m in common_swe], [C[m] for m in common_swe]).statistic if len(common_swe) > 5 else None
    mad_swe = float(np.mean([abs(swe_map[m] - C[m]) for m in common_swe])) if common_swe else None

    failing = sorted([(b, e["new"]["rate"]) for b, e in per_bench.items() if e["new"]["rate"] > TARGET_RATE and e["new"]["pairs"] >= 20], key=lambda x: -x[1])
    return {
        "definition": f"For each leaderboard, take every pair of models whose scores on that leaderboard differ by at least {int(GAP_FRACTION * 100)}% of its score range and, when the leaderboard publishes standard errors, by more than {Z_SIGNIFICANT} x the combined standard error and count the pairs whose order is reversed by the unified score. Target: below {int(TARGET_RATE * 100)}% per leaderboard.",
        "overall": {"pairs": overall_new[1], "inversions": overall_new[0], "rate": overall_new[0] / overall_new[1] if overall_new[1] else 0.0},
        "same_pairs_vs_previous_method": {
            "pairs": tot["new"][1],
            "new_rate": tot["new"][0] / tot["new"][1] if tot["new"][1] else None,
            "old_rate": tot["old"][0] / tot["old"][1] if tot["old"][1] else None,
        },
        "per_benchmark": per_bench,
        "above_target": [{"benchmark": b, "rate": r} for b, r in failing],
        "epoch_eci_spearman": {"models": len(common), "rho": rho_eci},
        "epoch_swe_eci_recomputed": {"models": len(common_swe), "spearman": rho_swe, "mean_abs_diff_eci": mad_swe},
    }


def print_summary(v):
    o = v["overall"]
    print(f"validation: {o['inversions']}/{o['pairs']} pairs reversed ({o['rate'] * 100:.1f}%)")
    s = v["same_pairs_vs_previous_method"]
    if s["old_rate"] is not None:
        print(f"  same pairs vs previous method: new {s['new_rate'] * 100:.1f}% / old {s['old_rate'] * 100:.1f}% ({s['pairs']} pairs)")
    for b, e in sorted(v["per_benchmark"].items(), key=lambda x: -x[1]["new"]["rate"]):
        sp = e.get("same_pairs")
        cmp = f" | same pairs old {sp['old_rate'] * 100:5.1f}% -> new {sp['new_rate'] * 100:5.1f}%" if sp else ""
        print(f"  {e['new']['rate'] * 100:5.1f}% {b:26} [{e['role']}] models={e['new']['n_models']:3} pairs={e['new']['pairs']:5}{cmp}")
    print(f"  Epoch ECI spearman: {v['epoch_eci_spearman']}")
    print(f"  Epoch SWE ECI (recomputed) : {v['epoch_swe_eci_recomputed']}")
