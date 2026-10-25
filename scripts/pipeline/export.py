import csv
import datetime as dt
import json
from collections import Counter, defaultdict

import numpy as np

from . import config
from .fit import uncertainty as fit_uncertainty

WIDE_CI_SCORE = 600


def score(c):
    return config.SCORE_SLOPE * c + config.SCORE_INTERCEPT


def fmt_date(s):
    return s.replace("-", "/") if s else ""


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def uncertainty(result):
    return fit_uncertainty(result["prob"], result["x_hat"], result["draws"])


def build_ranking(result):
    prob, x_hat, draws = result["prob"], result["x_hat"], result["draws"]
    C, se, lo, hi = uncertainty(result)
    obs_by_model = defaultdict(list)
    for o in result["fit_obs"]:
        obs_by_model[o["model"]].append(o)
    out = []
    for i, m in enumerate(prob.models):
        info = result["by_name"][m]
        groups = Counter(o["group"] for o in obs_by_model[m])
        n = len(obs_by_model[m])
        n_meas = groups["standard"] + groups["leaderboard"]
        flags = []
        if n_meas == 0:
            flags.append("no_measured")
        if groups["self"] / n >= 0.5:
            flags.append("self_heavy")
        if n == 1:
            flags.append("single_obs")
        if score(hi[i]) - score(lo[i]) > WIDE_CI_SCORE:
            flags.append("wide_ci")
        if not info.get("release_date"):
            flags.append("no_release_date")
        fam = {f: (round(score(result["family_c"][f][i])) if i in result["family_c"].get(f, {}) else None) for f in config.FAMILIES}
        out.append({
            "model": m, "vendor": info["vendor"], "series": info["series"], "theta": float(C[i]), "score": float(score(C[i])),
            "se": float(se[i] * config.SCORE_SLOPE), "ci_lo": float(score(lo[i])), "ci_hi": float(score(hi[i])),
            "n_obs": n, "n_measured": n_meas, "n_standard": groups["standard"], "n_leaderboard": groups["leaderboard"],
            "n_unknown": groups["unknown"], "n_self": groups["self"],
            "evidence": "measured" if n_meas == n else ("self_reported" if n_meas == 0 else "mixed"),
            "families": fam, "flags": flags, "release_date": info.get("release_date", ""),
            "observations": sorted(({"benchmark": o["bench"], "value": round(o["value"], 4), "group": o["group"], "source": o["source"],
                                     "source_name": o["name"], "config": o["config"][:160], "url": o["url"], "data_date": o["date"],
                                     "offset_logit": round(o["offset"], 4), "candidates": o["n_candidates"]} for o in obs_by_model[m]),
                                    key=lambda d: d["benchmark"]),
        })
    out.sort(key=lambda r: -r["score"])
    low_info = result.get("low_info", {})
    ranking = [r for r in out if r["model"] not in low_info]
    for k, r in enumerate(ranking, 1):
        r["rank"] = k
    return ranking, [r for r in out if r["model"] in low_info]


def benchmark_diagnostics(result):
    prob, x_hat = result["prob"], result["x_hat"]
    _, D, alpha = prob.unpack(x_hat)
    pred = prob.predict(x_hat)
    out = []
    for j, b in enumerate(prob.benches):
        sel = prob.bi == j
        groups = Counter(result["fit_obs"][i]["group"] for i in np.where(sel)[0])
        out.append({"benchmark": b, "family": config.spec(b)["family"], "anchored_to_epoch": bool(config.spec(b).get("anchor")),
                    "epoch_name": config.spec(b).get("anchor"), "difficulty_eci": round(float(D[j]), 2), "slope": round(float(alpha[j]), 4),
                    "n_obs": int(sel.sum()), "by_group": dict(groups), "rmse": round(float(np.sqrt(np.mean((prob.y[sel] - pred[sel]) ** 2))), 4)})
    return sorted(out, key=lambda d: (not d["anchored_to_epoch"], d["family"], d["benchmark"]))


def estimated_rows(result, low_rows):
    rows = []
    for e in result["estimates"]:
        sc, se = score(e["theta"]), e["se_theta"] * config.SCORE_SLOPE
        flags = ["interpolated", "no_observations"] + (["wide_ci"] if 2 * 1.96 * se > WIDE_CI_SCORE else [])
        rows.append(dict(e, score=sc, se=se, ci_lo=sc - 1.96 * se, ci_hi=sc + 1.96 * se, n_obs=0, basis="series_interpolation", flags=flags))
    for r in low_rows:
        info = result["low_info"][r["model"]]
        flags = ["low_information"] + [f for f in r["flags"] if f != "low_information"]
        rows.append({"model": r["model"], "vendor": r["vendor"], "series": r["series"], "release_date": r["release_date"], "theta": r["theta"],
                     "score": r["score"], "se": r["se"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "n_obs": r["n_obs"],
                     "basis": "self_reported_low_information", "method": info["method"], "flags": flags})
    return rows


def write(result):
    ranking, low_rows = build_ranking(result)
    est = result["estimates"]
    fitted_names = {r["model"] for r in ranking} | {r["model"] for r in low_rows}
    est_rows = estimated_rows(result, low_rows)
    merged = sorted([(r["score"], r["model"], False) for r in ranking] + [(e["score"], e["model"], True) for e in est_rows], reverse=True)
    est_rank = {name: k for k, (_, name, is_est) in enumerate(merged, 1) if is_est}

    header = ["rank", "model", "vendor", "series", "score", "theta", "se", "ci_lo", "ci_hi", "n_obs", "n_measured", "n_unknown", "n_self", "evidence",
              *config.FAMILIES, "flags", "release_date"]
    rows = []
    for r in ranking:
        rows.append([r["rank"], r["model"], r["vendor"], r["series"], round(r["score"]), round(r["theta"], 2), round(r["se"]), round(r["ci_lo"]), round(r["ci_hi"]),
                     r["n_obs"], r["n_measured"], r["n_unknown"], r["n_self"], r["evidence"],
                     *[("" if r["families"][f] is None else r["families"][f]) for f in config.FAMILIES], "|".join(r["flags"]), fmt_date(r["release_date"])])
    write_csv(config.OUT_CSV, header, rows)

    est_header = ["est_rank", "rank", "model", "vendor", "series", "score", "theta_est", "theta", "se", "ci_lo", "ci_hi", "n_obs", "basis", "method", "series_used", "flags", "release_date"]
    est_out = []
    for e in sorted(est_rows, key=lambda e: -e["score"]):
        est_out.append([est_rank[e["model"]], est_rank[e["model"]], e["model"], e["vendor"], e["series"], round(e["score"]), round(e["theta"], 2), round(e["theta"], 2),
                        round(e["se"]), round(e["ci_lo"]), round(e["ci_hi"]), e["n_obs"], e["basis"], e["method"], e["series"], "|".join(e["flags"]), fmt_date(e["release_date"])])
    write_csv(config.OUT_EST_CSV, est_header, est_out)

    est_names = {e["model"] for e in est}
    no_data = [{"model": m["model"], "vendor": m["vendor"], "series": m["series"], "release_date": m["release_date"],
                "reason": "no coding observation and no measured neighbours on both sides in its series" if m["release_date"] else "no coding observation and no release date"}
               for m in result["models"] if m["model"] not in fitted_names and m["model"] not in est_names]
    unmatched = Counter((e["source"], e["name"]) for e in result["matcher"].log if e["stage"] == "unmatched")
    prob = result["prob"]
    payload = {
        "generated": dt.date.today().isoformat(),
        "method": {
            "model": "performance(m,b) = sigmoid(slope_b * (C_m - D_b) + offset_{b,group}), fitted by ridge-regularised non-linear least squares (same functional form as Epoch AI's ECI)",
            "scale": "C is on the Epoch Capabilities Index (ECI) scale because difficulty D_b and slope of every Epoch-tracked coding benchmark are fixed to the values published in Epoch AI's edi_scores.csv; other benchmarks are linked through models measured on both",
            "score": f"score = {config.SCORE_SLOPE:g} * C {config.SCORE_INTERCEPT:+g} (ECI 130 -> 4000, ECI 150 -> 5000)",
            "benchmarks": "coding benchmarks only; reasoning, knowledge, general chat and tool-use benchmarks are excluded; composite indices (AA Coding Index) and preference Elo (Arena WebDev) are used for validation only",
            "selection": "one value per model and benchmark: third-party standard harness > third-party leaderboard with participant-built agents > unknown provenance > vendor self-report; within the best available tier, the highest value (highest reasoning effort)",
            "offsets": "leaderboard, unknown and self-reported values are shifted by the per-benchmark median logit gap against standard-harness values of the same models, shrunk toward the pooled gap (k = 5); leaderboard values on benchmarks without any standard-harness run are used as-is",
            "weights": config.GROUP_WEIGHT,
            "uncertainty": f"bootstrap over each model's observations (Epoch-style), B = {0 if result['draws'] is None else len(result['draws'])}, seed = {config.SEED}; CI = 2.5/97.5 percentiles",
            "family_scores": "C refitted per family with all benchmark parameters fixed; security benchmarks (ExploitBench, Cybench) only feed the security sub-score and are excluded from the overall coding score",
            "low_information": f"models with only self-reported observations and a 95% CI wider than {config.LOW_INFO_CI_WIDTH:g} score points are moved to the estimated table and are not used as interpolation anchors",
            "missing_models": "models without coding observations are linearly interpolated by release date between the nearest measured models before and after them in the same series; no extrapolation",
            "benchmark_rule": f"a benchmark needs >= {config.MIN_MODELS_PER_BENCH} models and must be connected to the Epoch-anchored benchmarks through shared models",
        },
        "sources": result["load_counts"],
        "counts": {"models_in_list": len(result["models"]), "models_ranked": len(ranking), "models_interpolated": len(est), "models_low_information": len(low_rows), "models_without_data": len(no_data),
                   "observations_used": len(prob.y), "benchmarks_used": len(prob.benches), "benchmarks_dropped": result["dropped_benches"]},
        "offsets": {"pooled": result["pooled"], "per_benchmark": result["offset_detail"]},
        "validation": result["validation"],
        "benchmark_diagnostics": benchmark_diagnostics(result),
        "ranking": ranking,
        "estimated": [{"model": e["model"], "basis": e["basis"], "score": round(e["score"]), "theta": round(e["theta"], 2), "se": round(e["se"]), "method": e["method"], "release_date": e["release_date"]} for e in est_rows],
        "no_data": no_data,
        "unmatched_source_names": [{"source": s, "name": n, "rows": k} for (s, n), k in unmatched.most_common(120)],
    }
    config.BUILD.mkdir(parents=True, exist_ok=True)
    (config.BUILD / "observations_by_model.json").write_text(
        json.dumps({r["model"]: r["observations"] for r in ranking + low_rows}, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    for r in payload["ranking"]:
        r["benchmarks"] = [{"benchmark": o["benchmark"], "group": o["group"], "source": o["source"]} for o in r.pop("observations")]
    for b in payload["validation"]["per_benchmark"].values():
        for e in b["new"].get("examples", []):
            e.pop("values", None)
    config.OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    v = result["validation"]
    summary = {
        "generated": payload["generated"],
        "counts": payload["counts"],
        "inversion_rate": round(v["overall"]["rate"], 4),
        "inversion_pairs": v["overall"]["pairs"],
        "previous_method_same_pairs": {k: (None if x is None else round(x, 4)) if k != "pairs" else x for k, x in v["same_pairs_vs_previous_method"].items()},
        "epoch_swe_eci_spearman": None if v["epoch_swe_eci_recomputed"]["spearman"] is None else round(float(v["epoch_swe_eci_recomputed"]["spearman"]), 4),
        "per_benchmark": sorted(({"benchmark": b, "role": e["role"], "rate": round(e["new"]["rate"], 4), "pairs": e["new"]["pairs"], "models": e["new"]["n_models"]}
                                 for b, e in v["per_benchmark"].items()), key=lambda d: d["benchmark"]),
    }
    config.OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    config.BUILD.mkdir(parents=True, exist_ok=True)
    write_csv(config.BUILD / "observations_selected.csv", ["model", "benchmark", "value", "group", "source", "source_name", "config", "url", "candidates"],
              [[o["model"], o["bench"], round(o["value"], 4), o["group"], o["source"], o["name"], o["config"][:200], o["url"], o["n_candidates"]]
               for o in sorted(result["fit_obs"], key=lambda o: (o["model"], o["bench"]))])
    seen = set()
    review = []
    for e in result["matcher"].log:
        k = (e["source"], e["name"])
        if k in seen:
            continue
        seen.add(k)
        review.append([e["stage"], e["source"], e["name"], e["model"]])
    write_csv(config.BUILD / "match_review.csv", ["stage", "source", "source_name", "matched_model"], sorted(review))
    print(f"wrote {config.OUT_CSV.name} ({len(ranking)}), {config.OUT_EST_CSV.name} ({len(est_rows)}), {config.OUT_JSON.name}, {config.OUT_SUMMARY.name}, data/build/*.csv")
