"""Order checks and the Artificial Analysis scale diagnostics.

The adopted score does not use the Intelligence Index as a probability. The
counterfactual below refits once with that old treatment so the build can show
what it does to established pairs.
"""

import math

import numpy as np
from scipy.stats import spearmanr

from . import config, sources
from .fit import Problem, connected_subset, uncertainty

V5_CODING_WEIGHTS = (("frontierCode20", 0.15), ("terminalBench40", 0.10))


def _logit(p):
    p = min(max(float(p), 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


def _score_table(result):
    prob = result["prob"]
    C, _, lo, hi = uncertainty(prob, result["x_hat"], result["draws"])
    low = result.get("low_info") or {}
    to = result["to_score"]
    table = {}
    for i, m in enumerate(prob.models):
        table[m] = {
            "score": float(to(C[i])), "lo": float(to(lo[i])), "hi": float(to(hi[i])),
            "theta": float(C[i]), "low_info": m in low, "interpolated": False,
        }
    for e in result.get("estimates") or []:
        if e["model"] in table:
            continue
        se = float(e["se_theta"])
        table[e["model"]] = {
            "score": float(to(e["theta"])), "lo": float(to(e["theta"] - 1.96 * se)), "hi": float(to(e["theta"] + 1.96 * se)),
            "theta": float(e["theta"]), "low_info": False, "interpolated": True,
        }
    return table


def _rank_map(table):
    ordered = sorted(((v["score"], m) for m, v in table.items() if not v["low_info"] and not v.get("interpolated")), reverse=True)
    return {m: k for k, (_, m) in enumerate(ordered, 1)}


def established_pairs(result):
    table = _score_table(result)
    ranks = _rank_map(table)
    rows = []
    for stronger, weaker, severity, why in config.ESTABLISHED_PAIRS:
        a, b = table.get(stronger), table.get(weaker)
        if not a or not b:
            rows.append({"stronger": stronger, "weaker": weaker, "severity": severity, "why": why, "status": "missing"})
            continue
        gap = a["score"] - b["score"]
        overlap = a["lo"] <= b["hi"] and b["lo"] <= a["hi"]
        clear = b["lo"] > a["hi"]
        if a.get("interpolated") or b.get("interpolated"):
            status = "unmeasured"
        elif a["low_info"] or b["low_info"]:
            status = "pass" if gap > 0 else "low_information"
        else:
            status = "pass" if gap > 0 else "fail"
        rows.append({
            "stronger": stronger, "weaker": weaker, "severity": severity, "why": why, "status": status,
            "stronger_score": round(a["score"]), "weaker_score": round(b["score"]),
            "stronger_rank": ranks.get(stronger), "weaker_rank": ranks.get(weaker),
            "stronger_low_info": a["low_info"], "weaker_low_info": b["low_info"],
            "stronger_interpolated": a.get("interpolated", False), "weaker_interpolated": b.get("interpolated", False),
            "gap": round(gap), "ci_overlap": overlap, "clear_inversion": bool(status == "fail" and clear),
        })
    failed = [r for r in rows if r["status"] == "fail"]
    low = [r for r in rows if r["status"] == "low_information"]
    return {
        "definition": "Point estimate of the coding score among models with a measured fit. Interpolated models are reported as unmeasured. A clear inversion means the weaker model's entire 95% interval sits above the stronger model's.",
        "checked": sum(1 for r in rows if r["status"] in ("pass", "fail")),
        "passed": sum(1 for r in rows if r["status"] == "pass"),
        "failed": len(failed),
        "low_information": len(low),
        "unmeasured": sum(1 for r in rows if r["status"] == "unmeasured"),
        "missing": sum(1 for r in rows if r["status"] == "missing"),
        "hard_failed": sum(1 for r in failed if r["severity"] == "hard"),
        "clear_inversions": sum(1 for r in failed if r["clear_inversion"]),
        "pairs": rows,
    }


def _best_progress(doc, matcher):
    best = {}
    raw = {}
    for row in doc.get("progress") or []:
        model, _ = matcher.match("aa", row.get("name") or "")
        if not model or not isinstance(row.get("score"), (int, float)):
            continue
        score = float(row["score"])
        raw.setdefault(model, score)
        if model not in best or score > best[model]:
            best[model] = score
    return best


def _aa_by_model(doc, matcher):
    out = {}
    matched = unmatched = 0
    seen = set()
    for row in doc.get("models") or []:
        name = row.get("name") or ""
        model, stage = matcher.match("aa", name)
        key = name
        if key not in seen:
            seen.add(key)
            if model:
                matched += 1
            elif stage != "ignored":
                unmatched += 1
        if not model:
            continue
        slot = out.setdefault(model, {"scores": {}, "intelligence": None, "names": []})
        slot["names"].append(name)
        ii = row.get("intelligence_index")
        if isinstance(ii, (int, float)):
            slot["intelligence"] = max(ii, slot["intelligence"] if slot["intelligence"] is not None else ii)
        for field, value in (row.get("scores") or {}).items():
            if isinstance(value, (int, float)):
                prev = slot["scores"].get(field)
                if prev is None or value > prev:
                    slot["scores"][field] = float(value)
    return out, {"matched_names": matched, "unmatched_names": unmatched, "canonical_models": len(out)}


def _spearman(xs, ys):
    if len(xs) < 6:
        return None
    rho = spearmanr(xs, ys).statistic
    return None if rho is None or not np.isfinite(rho) else float(rho)


def _pair_order(values, ours):
    """Share of well-separated pairs whose order disagrees with the coding score."""
    from .validate import inversions
    return inversions(values, ours)


def aa_external(result):
    doc = sources.aa_document()
    table = _score_table(result)
    ours = {m: v["theta"] for m, v in table.items()}
    if not doc:
        return {"status": "no_public_snapshot", "version": None}
    by_model, coverage = _aa_by_model(doc, result["matcher"])
    progress = _best_progress(doc, result["matcher"])
    common_p = sorted(set(progress) & set(ours))
    intel = {m: slot["intelligence"] for m, slot in by_model.items() if slot["intelligence"] is not None and m in ours}
    common_i = sorted(intel)
    coding = {}
    for m, slot in by_model.items():
        if m not in ours:
            continue
        if all(field in slot["scores"] for field, _ in V5_CODING_WEIGHTS):
            coding[m] = sum(w * slot["scores"][field] for field, w in V5_CODING_WEIGHTS) / sum(w for _, w in V5_CODING_WEIGHTS)
    common_c = sorted(coding)
    gpt4o = progress.get("GPT-4o")
    above = []
    tb = result["edi"].get("Terminal Bench")
    fc = result["edi"].get("FrontierCode")
    gpt4o_theta = ours.get("GPT-4o")
    if tb and fc and gpt4o is not None and gpt4o_theta is not None:
        for m, slot in by_model.items():
            if progress.get(m, -1e9) <= gpt4o:
                continue
            implied = []
            if "terminalBench40" in slot["scores"]:
                implied.append(tb["edi"] + _logit(slot["scores"]["terminalBench40"]) / tb["slope"])
            if "frontierCode20" in slot["scores"]:
                implied.append(fc["edi"] + _logit(slot["scores"]["frontierCode20"]) / fc["slope"])
            if not implied:
                continue
            c = min(implied)
            above.append({"model": m, "progress": round(progress[m], 1), "naive_eci": round(c, 1), "below_gpt4o": c < gpt4o_theta})
    below = [r for r in above if r["below_gpt4o"]]
    below.sort(key=lambda r: r["naive_eci"])
    return {
        "status": "public_snapshot",
        "version": doc.get("intelligence_index_version"),
        "version_number": doc.get("intelligence_index_version_number"),
        "coding_index_field": doc.get("coding_index_field"),
        "coverage": coverage,
        "progress_index": {
            "scale": "o3-mini (high) = 220, GPT-5.5 (xhigh) = 320; general capability, not a coding probability",
            "models": len(common_p),
            "spearman_vs_coding_eci": _spearman([progress[m] for m in common_p], [ours[m] for m in common_p]),
            "order": _pair_order(progress, ours),
            "gpt4o": None if gpt4o is None else round(gpt4o, 1),
            "gpt55": None if progress.get("GPT-5.5") is None else round(progress["GPT-5.5"], 1),
        },
        "intelligence_index_v5": {
            "used_in_fit": False,
            "reason": "v5 is a weighted mix of pass rates and Elo-normalised components. Scores are not comparable with v4.3 and are missing for models that were not rerun, including GPT-4o and GPT-5.5 in this snapshot.",
            "models": len(common_i),
            "spearman_vs_coding_eci": _spearman([intel[m] for m in common_i], [ours[m] for m in common_i]),
            "order": _pair_order(intel, ours) if intel else None,
            "present": {
                "GPT-4o": "GPT-4o" in intel,
                "GPT-5.5": "GPT-5.5" in intel,
                "GPT-6 Astra": "GPT-6 Astra" in intel,
            },
        },
        "coding_category_v5": {
            "definition": "0.60 * FrontierCode-AA 2.0 + 0.40 * Terminal-Bench 4.0, the v5 coding category (15% and 10% of the index). Not an independent check: those benchmarks are fit inputs.",
            "models": len(common_c),
            "spearman_vs_coding_eci": _spearman([coding[m] for m in common_c], [ours[m] for m in common_c]),
            "order": _pair_order(coding, ours) if coding else None,
        },
        "naive_anchor_if_new_scores_replace_old_benchmarks": {
            "definition": (
                "Single-benchmark capability if Terminal-Bench 4.0 were scored on Epoch's Terminal Bench anchor "
                "(the 2.x leaderboard) or FrontierCode 2.0 on Epoch's FrontierCode anchor, for models the Progress Index "
                "places above GPT-4o. min of the two implied capabilities."
            ),
            "gpt4o_theta": None if gpt4o_theta is None else round(gpt4o_theta, 2),
            "models_above_gpt4o_on_progress_with_a_v5_coding_score": len(above),
            "implied_below_gpt4o": len(below),
            "examples": below[:12],
        },
    }


def _naive_observations(result, by_model):
    """Replace versioned terminal / FrontierCode rows with the new AA numbers on the old anchors, and add v5/100."""
    drop_terminal = {m for m, slot in by_model.items() if "terminalBench40" in slot["scores"]}
    drop_fc = {m for m, slot in by_model.items() if "frontierCode20" in slot["scores"]}
    kept = []
    for o in result["fit_obs"]:
        if o["model"] in drop_terminal and str(o["bench"]).startswith("Terminal-Bench"):
            continue
        if o["model"] in drop_fc and (o["bench"] == "FrontierCode" or str(o["bench"]).startswith("FrontierCode")):
            continue
        kept.append(o)
    extra = []
    for m, slot in by_model.items():
        if "terminalBench40" in slot["scores"]:
            extra.append({"model": m, "bench": "Terminal-Bench 2.0", "value": slot["scores"]["terminalBench40"], "weight": 1.0, "offset": 0.0})
        if "frontierCode20" in slot["scores"]:
            extra.append({"model": m, "bench": "FrontierCode", "value": slot["scores"]["frontierCode20"], "weight": 1.0, "offset": 0.0})
        if slot["intelligence"] is not None:
            extra.append({"model": m, "bench": "AA Intelligence Index v5", "value": float(slot["intelligence"]) / 100.0, "weight": 1.0, "offset": 0.0})
    obs, dropped = connected_subset(kept + extra)
    return obs, dropped


def naive_refit(result):
    doc = sources.aa_document()
    if not doc:
        return {"status": "no_public_snapshot"}
    by_model, _ = _aa_by_model(doc, result["matcher"])
    obs, dropped = _naive_observations(result, by_model)
    try:
        prob = Problem(obs, result["edi"])
        x, res = prob.solve()
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}
    C = dict(zip(prob.models, x[: prob.nM]))
    base = dict(zip(result["prob"].models, result["x_hat"][: result["prob"].nM]))

    def row(name):
        if name not in C or name not in base:
            return None
        return {"model": name, "adopted_score": round(float(result["to_score"](base[name]))), "naive_score": round(float(result["to_score"](C[name]))),
                "adopted_theta": round(float(base[name]), 2), "naive_theta": round(float(C[name]), 2)}

    focus = ["GPT-5.5", "GPT-4o", "GPT-4", "GPT-5", "OpenAI o3", "OpenAI o1", "Claude 3.5 Sonnet", "Claude Opus 4.5", "Claude Opus 4", "GPT-6 Astra"]
    failures = []
    for stronger, weaker, severity, why in config.ESTABLISHED_PAIRS:
        if stronger not in C or weaker not in C:
            continue
        if C[stronger] < C[weaker]:
            failures.append({"stronger": stronger, "weaker": weaker, "severity": severity,
                             "naive_scores": [round(float(result["to_score"](C[stronger]))), round(float(result["to_score"](C[weaker])))]})
    hard = [f for f in failures if f["severity"] == "hard"]
    return {
        "status": "ok",
        "treatment": (
            "For models with a Terminal-Bench 4.0 score, delete every Terminal-Bench observation and write that 4.0 "
            "rate onto Terminal-Bench 2.0, which is anchored to Epoch's Terminal Bench difficulty. Do the same for "
            "FrontierCode 2.0 onto the anchored FrontierCode series. Also add Intelligence Index v5 divided by 100 "
            "as if it were a success probability. One refit, no bootstrap."
        ),
        "cost": round(float(res.cost), 3),
        "observations": len(obs),
        "benchmarks_dropped": dropped,
        "hard_pairs_inverted": len(hard),
        "pairs_inverted": len(failures),
        "inverted": failures[:20],
        "focus": [r for r in (row(n) for n in focus) if r],
    }


def extra(result):
    saved = result["matcher"].log
    result["matcher"].log = []
    try:
        pairs = established_pairs(result)
        external = aa_external(result)
        refit = naive_refit(result) if external.get("status") == "public_snapshot" else {"status": "skipped"}
        external["naive_refit"] = refit
        return {"established_pairs": pairs, "aa_scale": external}
    finally:
        result["matcher"].log = saved


def print_extra(validation):
    pairs = validation.get("established_pairs") or {}
    if pairs:
        print(f"established pairs: {pairs.get('passed')}/{pairs.get('checked')} measured pairs ordered, "
              f"{pairs.get('failed')} inverted ({pairs.get('hard_failed')} hard, {pairs.get('clear_inversions')} clear), "
              f"{pairs.get('unmeasured')} unmeasured, {pairs.get('low_information')} low-information, {pairs.get('missing')} missing")
        for r in pairs.get("pairs") or []:
            if r["status"] == "pass" and r["severity"] != "hard":
                continue
            if r["status"] == "missing":
                print(f"  {r['severity']:8} {r['stronger']} > {r['weaker']}: missing")
                continue
            flag = "CLEAR INVERSION" if r.get("clear_inversion") else r["status"]
            print(f"  {r['severity']:8} {r['stronger']} > {r['weaker']}: {flag} {r.get('stronger_score')} vs {r.get('weaker_score')} "
                  f"(ranks {r.get('stronger_rank')}, {r.get('weaker_rank')})")
    aa = validation.get("aa_scale") or {}
    if not aa:
        return
    prog = aa.get("progress_index") or {}
    intel = aa.get("intelligence_index_v5") or {}
    naive = (aa.get("naive_anchor_if_new_scores_replace_old_benchmarks") or {})
    refit = aa.get("naive_refit") or {}
    rho = prog.get("spearman_vs_coding_eci")
    print(f"AA Intelligence Index {aa.get('version')}: coding-index field {aa.get('coding_index_field')}; "
          f"v5 index on {intel.get('models')} matched models, used in the fit: {intel.get('used_in_fit')}")
    if rho is not None:
        print(f"  Progress Index spearman vs coding score: {rho:.3f} ({prog.get('models')} models); "
              f"GPT-5.5 {prog.get('gpt55')} vs GPT-4o {prog.get('gpt4o')}")
    if naive:
        print(f"  naive anchor (TB 4.0 or FC 2.0 on the old Epoch difficulty): "
              f"{naive.get('implied_below_gpt4o')}/{naive.get('models_above_gpt4o_on_progress_with_a_v5_coding_score')} "
              f"models above GPT-4o on the Progress Index would fall below GPT-4o's ECI")
    if refit.get("status") == "ok":
        print(f"  naive refit inversions: {refit.get('hard_pairs_inverted')} hard / {refit.get('pairs_inverted')} total")
        for row in refit.get("focus") or []:
            if row["model"] in ("GPT-5.5", "GPT-4o", "GPT-4", "Claude 3.5 Sonnet"):
                print(f"    {row['model']}: adopted {row['adopted_score']} -> naive {row['naive_score']}")

