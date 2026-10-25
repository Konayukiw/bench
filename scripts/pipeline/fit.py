import math
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import expit

from . import config
from .select import logit

C_PRIOR_MU = 130.0
C_RIDGE = 0.002
D_PRIOR_MU = 145.0
D_RIDGE = 0.002
A_RIDGE = 0.05


class Problem:
    def __init__(self, observations, edi):
        self.models = sorted({o["model"] for o in observations})
        self.benches = sorted({o["bench"] for o in observations})
        mpos = {m: i for i, m in enumerate(self.models)}
        bpos = {b: j for j, b in enumerate(self.benches)}
        self.mi = np.array([mpos[o["model"]] for o in observations])
        self.bi = np.array([bpos[o["bench"]] for o in observations])
        self.y = np.array([o["value"] for o in observations], dtype=float)
        self.w = np.array([o["weight"] for o in observations], dtype=float)
        self.off = np.array([o["offset"] for o in observations], dtype=float)
        self.D_fixed = np.full(len(self.benches), np.nan)
        self.alpha_fixed = np.full(len(self.benches), np.nan)
        for j, b in enumerate(self.benches):
            anchor = config.spec(b).get("anchor")
            if anchor and anchor in edi:
                self.D_fixed[j] = edi[anchor]["edi"]
                self.alpha_fixed[j] = edi[anchor]["slope"]
        self.free = np.where(np.isnan(self.D_fixed))[0]
        self.free_k = np.full(len(self.benches), -1)
        self.free_k[self.free] = np.arange(len(self.free))
        self.nM, self.nF = len(self.models), len(self.free)
        anchored = self.alpha_fixed[~np.isnan(self.alpha_fixed)]
        self.a0 = math.log(float(np.median(anchored))) if len(anchored) else math.log(0.12)

    def unpack(self, x):
        C = x[: self.nM]
        D = self.D_fixed.copy()
        alpha = self.alpha_fixed.copy()
        D[self.free] = x[self.nM : self.nM + self.nF]
        alpha[self.free] = np.exp(x[self.nM + self.nF :])
        return C, D, alpha

    def residuals(self, x, mi, bi, y, sw, off):
        C, D, alpha = self.unpack(x)
        s = expit(alpha[bi] * (C[mi] - D[bi]) + off)
        return np.concatenate([
            sw * (y - s),
            C_RIDGE * (C - C_PRIOR_MU),
            D_RIDGE * (x[self.nM : self.nM + self.nF] - D_PRIOR_MU),
            A_RIDGE * (x[self.nM + self.nF :] - self.a0),
        ])

    def jacobian(self, x, mi, bi, y, sw, off):
        C, D, alpha = self.unpack(x)
        n = len(y)
        s = expit(alpha[bi] * (C[mi] - D[bi]) + off)
        g = -sw * s * (1.0 - s)
        ga = g * alpha[bi]
        rows = [np.arange(n)]
        cols = [mi]
        vals = [ga]
        k = self.free_k[bi]
        free = k >= 0
        idx = np.arange(n)[free]
        rows += [idx, idx]
        cols += [self.nM + k[free], self.nM + self.nF + k[free]]
        vals += [-ga[free], ga[free] * (C[mi[free]] - D[bi[free]])]
        p = self.nM + 2 * self.nF
        diag = np.arange(p)
        rows.append(n + diag)
        cols.append(diag)
        vals.append(np.concatenate([np.full(self.nM, C_RIDGE), np.full(self.nF, D_RIDGE), np.full(self.nF, A_RIDGE)]))
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n + p, p))

    def initial(self):
        alpha0 = math.exp(self.a0)
        lg = np.array([logit(v) for v in self.y])
        fixed = ~np.isnan(self.D_fixed[self.bi])
        C = np.full(self.nM, C_PRIOR_MU)
        acc = defaultdict(list)
        for i in np.where(fixed)[0]:
            acc[self.mi[i]].append(self.D_fixed[self.bi[i]] + (lg[i] - self.off[i]) / self.alpha_fixed[self.bi[i]])
        for m, v in acc.items():
            C[m] = float(np.mean(v))
        D_free = np.full(self.nF, D_PRIOR_MU)
        for j in self.free:
            sel = np.where((self.bi == j) & np.isin(self.mi, list(acc)))[0]
            if len(sel):
                D_free[self.free_k[j]] = float(np.median(C[self.mi[sel]] - (lg[sel] - self.off[sel]) / alpha0))
        D = self.D_fixed.copy()
        D[self.free] = D_free
        alpha = np.where(np.isnan(self.alpha_fixed), alpha0, self.alpha_fixed)
        acc2 = defaultdict(list)
        for i in range(len(self.y)):
            acc2[self.mi[i]].append(D[self.bi[i]] + (lg[i] - self.off[i]) / alpha[self.bi[i]])
        for m, v in acc2.items():
            if m not in acc:
                C[m] = float(np.mean(v))
        return np.concatenate([C, D_free, np.full(self.nF, self.a0)])

    def solve(self, x0=None, mi=None, bi=None, y=None, w=None, off=None, max_nfev=None):
        mi = self.mi if mi is None else mi
        bi = self.bi if bi is None else bi
        y = self.y if y is None else y
        w = self.w if w is None else w
        off = self.off if off is None else off
        args = (mi, bi, y, np.sqrt(w), off)
        res = least_squares(self.residuals, self.initial() if x0 is None else x0, jac=self.jacobian, args=args,
                            method="trf", tr_solver="lsmr", x_scale="jac", max_nfev=max_nfev)
        return res.x, res

    def bootstrap(self, x_hat, n, seed):
        rng = np.random.default_rng(seed)
        by_model = defaultdict(list)
        for i, m in enumerate(self.mi):
            by_model[m].append(i)
        groups = [np.array(v) for v in by_model.values()]
        draws = np.empty((n, self.nM))
        for b in range(n):
            counts = np.zeros(len(self.y))
            for g in groups:
                np.add.at(counts, rng.choice(g, size=len(g), replace=True), 1)
            keep = counts > 0
            x, _ = self.solve(x_hat, self.mi[keep], self.bi[keep], self.y[keep], self.w[keep] * counts[keep], self.off[keep], max_nfev=200)
            draws[b] = x[: self.nM]
        return draws

    def analytic_se(self, x, noise_floor=0.03):
        C, D, alpha = self.unpack(x)
        s = expit(alpha[self.bi] * (C[self.mi] - D[self.bi]) + self.off)
        resid = self.y - s
        rmse = np.array([np.sqrt(np.mean(resid[self.bi == j] ** 2)) for j in range(len(self.benches))])
        sigma = np.maximum(rmse, noise_floor)[self.bi]
        grad = alpha[self.bi] * s * (1.0 - s)
        info = np.bincount(self.mi, weights=self.w * grad ** 2 / sigma ** 2, minlength=self.nM)
        return 1.0 / np.sqrt(np.maximum(info, 1e-12))

    def predict(self, x):
        C, D, alpha = self.unpack(x)
        return expit(alpha[self.bi] * (C[self.mi] - D[self.bi]) + self.off)

    def refit_subset(self, x, mask):
        C, D, alpha = self.unpack(x)
        out = {}
        for m in np.unique(self.mi[mask]):
            sel = mask & (self.mi == m)
            yy, ww, oo, dd, aa = self.y[sel], self.w[sel], self.off[sel], D[self.bi[sel]], alpha[self.bi[sel]]
            f = lambda c: float(np.sum(ww * (yy - expit(aa * (c - dd) + oo)) ** 2) + (C_RIDGE * (c - C_PRIOR_MU)) ** 2)
            out[m] = minimize_scalar(f, bounds=(30.0, 280.0), method="bounded").x
        return out


def refit_external(observations, edi, model_index):
    groups = defaultdict(list)
    for o in observations:
        anchor = config.spec(o["bench"]).get("anchor")
        if anchor and anchor in edi and o["model"] in model_index:
            groups[model_index[o["model"]]].append((o["value"], o["weight"], o["offset"], edi[anchor]["edi"], edi[anchor]["slope"]))
    out = {}
    for i, rows in groups.items():
        y, w, off, D, a = (np.array(v) for v in zip(*rows))
        f = lambda c: float(np.sum(w * (y - expit(a * (c - D) + off)) ** 2) + (C_RIDGE * (c - C_PRIOR_MU)) ** 2)
        out[i] = minimize_scalar(f, bounds=(30.0, 280.0), method="bounded").x
    return out


def uncertainty(prob, x_hat, draws):
    C = x_hat[: prob.nM]
    se_a = prob.analytic_se(x_hat)
    if draws is None:
        return C, se_a, C - 1.96 * se_a, C + 1.96 * se_a
    se = np.maximum(draws.std(axis=0), se_a)
    lo = np.minimum(np.percentile(draws, 2.5, axis=0), C - 1.96 * se_a)
    hi = np.maximum(np.percentile(draws, 97.5, axis=0), C + 1.96 * se_a)
    return C, se, lo, hi


def connected_subset(observations):
    bench_models = defaultdict(set)
    for o in observations:
        bench_models[o["bench"]].add(o["model"])
    keep_b = {b for b, ms in bench_models.items() if len(ms) >= config.MIN_MODELS_PER_BENCH}
    obs = [o for o in observations if o["bench"] in keep_b]
    adj = defaultdict(set)
    for o in obs:
        adj["m:" + o["model"]].add("b:" + o["bench"])
        adj["b:" + o["bench"]].add("m:" + o["model"])
    start = ["b:" + b for b in keep_b if config.spec(b).get("anchor")]
    seen = set(start)
    stack = list(start)
    while stack:
        for nxt in adj[stack.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    kept = [o for o in obs if "m:" + o["model"] in seen]
    dropped_b = sorted(set(bench_models) - {o["bench"] for o in kept})
    return kept, dropped_b
