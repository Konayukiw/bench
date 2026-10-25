import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from pipeline import config, estimate, export, registry, sources, validate
from pipeline.fit import Problem, connected_subset, refit_external
from pipeline.select import attach_models, estimate_offsets, select


def to_score(c):
    return config.SCORE_SLOPE * np.asarray(c) + config.SCORE_INTERCEPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=config.BOOTSTRAP)
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    models = registry.parse_model_list()
    by_name = {m["model"]: m for m in models}
    dates = registry.load_release_dates()
    for m in models:
        m["release_date"] = dates.get(m["model"], "")
    matcher = registry.Matcher(models, registry.load_aliases())

    raw, load_counts = sources.load_all()
    rows = attach_models(raw, matcher, by_name)
    chosen, by = select(rows)
    offsets, pooled, offset_detail = estimate_offsets(by)

    fit_obs, sub_obs = [], []
    for (model, bench), c in chosen.items():
        role = config.spec(bench).get("role", "fit")
        o = dict(c, weight=config.GROUP_WEIGHT[c["group"]], offset=offsets[c["group"]][bench])
        if role == "fit":
            fit_obs.append(o)
        elif role == "subscore":
            sub_obs.append(o)
    fit_obs, dropped_benches = connected_subset(fit_obs)
    shared = Counter((o["model"], config.spec(o["bench"]).get("share_group")) for o in fit_obs if config.spec(o["bench"]).get("share_group"))
    for o in fit_obs:
        g = config.spec(o["bench"]).get("share_group")
        if g:
            o["weight"] /= shared[(o["model"], g)]

    eci_scores, edi_df, processed = sources.load_epoch_eci()
    edi = {r.benchmark_name: {"edi": r.edi, "slope": r.estimated_slope_scaled} for r in edi_df.itertuples()}

    prob = Problem(fit_obs, edi)
    x_hat, res = prob.solve()
    print(f"fit: {prob.nM} models, {len(prob.benches)} benchmarks ({prob.nF} free), {len(prob.y)} obs, cost={res.cost:.3f}, {time.time() - t0:.1f}s")

    draws = prob.bootstrap(x_hat, args.bootstrap, config.SEED) if args.bootstrap > 0 else None
    print(f"bootstrap: {args.bootstrap} replicates, {time.time() - t0:.1f}s")

    fam_of = {b: config.spec(b)["family"] for b in prob.benches}
    family_c = {}
    for fam in config.FAMILIES:
        mask = np.array([fam_of[prob.benches[j]] == fam for j in prob.bi])
        if mask.any():
            family_c[fam] = prob.refit_subset(x_hat, mask)
        external = refit_external([o for o in sub_obs if config.spec(o["bench"])["family"] == fam], edi, {m: i for i, m in enumerate(prob.models)})
        if external:
            family_c.setdefault(fam, {}).update(external)

    result = {
        "models": models, "by_name": by_name, "matcher": matcher, "rows": rows, "chosen": chosen, "by": by,
        "offsets": offsets, "pooled": pooled, "offset_detail": offset_detail, "fit_obs": fit_obs,
        "dropped_benches": dropped_benches, "prob": prob, "x_hat": x_hat, "draws": draws, "family_c": family_c,
        "processed": processed, "edi": edi, "eci_scores": eci_scores, "load_counts": load_counts, "to_score": to_score,
    }
    result["low_info"] = estimate.low_information(result)
    result["estimates"] = estimate.interpolate(result)
    result["validation"] = validate.run(result)
    validate.print_summary(result["validation"])
    if not args.no_export:
        export.write(result)
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
