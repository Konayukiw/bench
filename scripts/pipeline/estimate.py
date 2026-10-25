import datetime as dt
from collections import Counter, defaultdict

import numpy as np

from . import config
from .fit import uncertainty

INTERP_SE_FLOOR = 2.0


def parse_date(s):
    try:
        return dt.date.fromisoformat(s.replace("/", "-"))
    except (ValueError, AttributeError):
        return None


def side(points, d, before):
    cands = [p for p in points if (p[0] <= d if before else p[0] >= d)]
    if not cands:
        return None
    edge = max(p[0] for p in cands) if before else min(p[0] for p in cands)
    same = [p for p in cands if p[0] == edge]
    return edge, float(np.mean([p[2] for p in same])), float(np.sqrt(np.mean([p[3] ** 2 for p in same]))), [p[1] for p in same]


def low_information(result):
    prob = result["prob"]
    C, se, lo, hi = uncertainty(prob, result["x_hat"], result["draws"])
    groups = defaultdict(Counter)
    for o in result["fit_obs"]:
        groups[o["model"]][o["group"]] += 1
    out = {}
    for i, m in enumerate(prob.models):
        g = groups[m]
        width = (hi[i] - lo[i]) * config.SCORE_SLOPE
        if g["standard"] + g["leaderboard"] == 0 and width > config.LOW_INFO_CI_WIDTH:
            out[m] = {"theta": float(C[i]), "se_theta": float(se[i]), "lo_theta": float(lo[i]), "hi_theta": float(hi[i]), "n_obs": sum(g.values()),
                      "method": f"自己申告値のみ（{sum(g.values())}件）で信頼区間の幅が {width:.0f} 点あるため参考値として表示"}
    return out


def interpolate(result):
    C, se, _, _ = uncertainty(result["prob"], result["x_hat"], result["draws"])
    low_info = result.get("low_info", {})
    fitted = {m: (C[i], se[i]) for i, m in enumerate(result["prob"].models) if m not in low_info}
    has_data = set(result["prob"].models)
    by_series = defaultdict(list)
    for m in result["models"]:
        by_series[(m["vendor"], m["series"])].append(m)
    out = []
    for (vendor, series), members in by_series.items():
        points = []
        for m in members:
            d = parse_date(m["release_date"])
            if m["model"] in fitted and d:
                points.append((d, m["model"], *fitted[m["model"]]))
        for m in members:
            d = parse_date(m["release_date"])
            if m["model"] in has_data or not d:
                continue
            prev, nxt = side(points, d, True), side(points, d, False)
            if not prev or not nxt:
                continue
            (d0, c0, s0, n0), (d1, c1, s1, n1) = prev, nxt
            t = 0.5 if d1 == d0 else (d - d0).days / (d1 - d0).days
            c = c0 + t * (c1 - c0)
            s = max(float(np.sqrt((s0 ** 2 + s1 ** 2) / 2 + ((c1 - c0) / 2) ** 2)), INTERP_SE_FLOOR)
            if d0 == d1:
                method = f"{series} 内の同日リリース {' / '.join(n0)}（{d0}）の値を流用"
            else:
                method = f"{series} 内の {' / '.join(n0)}（{d0}）と {' / '.join(n1)}（{d1}）の間をリリース日で線形補間"
            out.append({"model": m["model"], "vendor": vendor, "series": series, "release_date": m["release_date"], "theta": c, "se_theta": s, "method": method})
    return out
