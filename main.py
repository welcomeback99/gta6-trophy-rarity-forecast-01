#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster — make.fun gta6-trophy-rarity-forecast-01.

Reads one case (schema makefun/gta6-trophies-case/v1) from the input directory
and writes, for each of the game's 55 trophies, five quantiles (5/25/50/75/95%)
of that trophy's Exophase earn rate at the case target.

Model — structural earners/players decomposition.  The earn rate is
r = E / N, earned count over players tracked.  Two engines, chosen by how much
history the case carries:

* Rich history (several distinct snapshots): a saturating trend fit plus a
  mechanistic earners/players ODE.  The trend fits
  r(t) = L / (1 + exp(-(ln(1+h) - mu) / s)) — a logistic in log-time — so the
  forecast levels off at the trophy's eventual share L instead of running
  away linearly.  The ODE integrates dE = rho*dN + g*max(0, L*N - E)*dt,
  where rho is the marginal earn rate per newly tracked player and g the
  per-hour hazard for not-yet-earners, both fitted from the intervals
  between distinct snapshots; the L cap keeps progress from overshooting
  the trophy's eventual share.  Rising trophies blend the two engines
  50/50 in logit space; falling trophies (dilution regime) use the ODE.

* Flat history (one distinct snapshot, e.g. launch night before the first
  refresh): the case carries no trend, so a dilution prior takes over.  The
  tracked-player count is assumed to grow ~1.37x per hour since the last
  refresh (the launch-night refresh grew the population ~6x in ~5.5h), new
  players earn each trophy at a small fraction of the early rate
  (rho prior = r * min(1, r/10)), and progress is a small hazard
  (g prior = 0.0003/hr, scaled down for rare trophies).  The ODE then
  dilutes every rate toward its refresh value.

Player-count forecast: log-log OLS (log N vs ln(1 + hours since release),
recency-weighted, 24h half-life) when the history has several distinct N
values, else the dilution prior above.

Spread: sigma = sqrt(0.35^2 + (0.5*|logit move|)^2 + resid_std^2); the
quantiles are inv_logit(logit(median) + z_tau * sigma), z the standard
normal quantiles, rounded to 4 decimals half-up, clamped to [0.01, 99.99],
non-decreasing.

Deterministic, standard library only, no network, no randomness.
"""

import json
import math
import os
import sys
from datetime import datetime

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Tuned constants (see module docstring).
N_GROWTH = 1.37        # per-hour growth of tracked players, flat-history prior
N_WINDOW_CAP = 5.5     # max accumulation window (hours) for the N prior
KAPPA_DIV = 10.0       # rho prior divisor: rho = r * min(1, r / KAPPA_DIV)
G_PRIOR = 0.0003       # per-hour progress hazard prior (rare trophies scale down)
SIG0 = 0.35            # base logit spread
SIG_MOVE = 0.5         # spread per unit of logit move from the last read


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1 - p))


def inv_logit(lg):
    if lg > 20:
        return 99.99
    if lg < -20:
        return 0.01
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    v = whole / 10000
    return int(v) if float(v).is_integer() else v


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def wls(x, y, w):
    """Weighted least squares y = a + b*x. Returns (a, b, resid_std) or None."""
    sw = sum(w)
    if sw <= 0 or len(x) < 2:
        return None
    mx = sum(wi * xi for wi, xi in zip(w, x)) / sw
    my = sum(wi * yi for wi, yi in zip(w, y)) / sw
    sxx = sum(wi * (xi - mx) ** 2 for wi, xi in zip(w, x))
    if sxx <= 1e-12:
        return None
    sxy = sum(wi * (xi - mx) * (yi - my) for wi, xi, yi in zip(w, x, y))
    b = sxy / sxx
    a = my - b * mx
    resid = [yi - (a + b * xi) for xi, yi in zip(x, y)]
    var = sum(wi * r * r for wi, r in zip(w, resid)) / sw
    return a, b, math.sqrt(var)


def median(vals):
    v = sorted(vals)
    n = len(v)
    if n == 0:
        return None
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def distinct_snapshots(case):
    """Collapse consecutive identical reads into one snapshot each."""
    snaps = []
    prev = None
    for h in case["history"]:
        key = (h["players_tracked"], tuple(h["rates"]))
        t = parse_ts(h["at"])
        if key == prev:
            snaps[-1]["last_seen"] = t
            continue
        prev = key
        snaps.append({"t": t, "last_seen": t, "N": h["players_tracked"],
                      "rates": list(h["rates"])})
    return snaps


def saturating_fit(xs, ys, wts):
    """Fit r = L / (1 + exp(-(x - mu) / s)) with x = ln(1 + hours).

    Grid-searches L; for each L, logit(r / L) is linear in x, so the fit is
    weighted least squares. Returns (L, a, b, resid_std) or None. Picks the
    smallest L whose residual is within 10% of the best, so the asymptote
    stays conservative.
    """
    rmax = max(ys)
    if rmax <= 0:
        return None
    lo = math.log(max(rmax * 1.02, 0.05))
    hi = math.log(99.9)
    cand = [math.exp(lo + (hi - lo) * k / 25.0) for k in range(26)]
    best_sd = None
    for L in cand:
        yy = []
        ok = True
        for r in ys:
            p = r / L
            if p <= 0.001 or p >= 0.999:
                ok = False
                break
            yy.append(math.log(p / (1 - p)))
        if not ok:
            continue
        fit = wls(xs, yy, wts)
        if not fit or fit[1] <= 0:
            continue
        if best_sd is None or fit[2] < best_sd:
            best_sd = fit[2]
    if best_sd is None:
        return None
    for L in cand:
        yy = []
        ok = True
        for r in ys:
            p = r / L
            if p <= 0.001 or p >= 0.999:
                ok = False
                break
            yy.append(math.log(p / (1 - p)))
        if not ok:
            continue
        fit = wls(xs, yy, wts)
        if not fit or fit[1] <= 0:
            continue
        if fit[2] <= best_sd * 1.10:
            return (L, fit[0], fit[1], fit[2])
    return None


def forecast_case(case):
    trophies = case["trophies"]
    ntr = len(trophies)
    snaps = distinct_snapshots(case)
    if not snaps:
        # No reads at all: a neutral prior, wide spread, still a valid file.
        return {"trophies": [{"id": t["id"], "q": [0.01, 0.05, 0.5, 5.0, 50.0]}
                             for t in trophies]}
    t_target = parse_ts(case["target"])
    release = datetime.fromisoformat(case["game"]["release_date"] + "T00:00:00+00:00")
    last = snaps[-1]
    N_last = last["N"]
    r_last = [last["rates"][i] if last["rates"][i] is not None else 0.0
              for i in range(ntr)]
    E_last = [r_last[i] * N_last / 100.0 for i in range(ntr)]

    def hrs(t):
        return max(0.0, (t - release).total_seconds() / 3600.0)

    xs = [math.log1p(hrs(s["t"])) for s in snaps]
    wts = [0.5 ** ((last["last_seen"] - s["t"]).total_seconds() / 3600.0 / 24.0)
           for s in snaps]
    Ns = [s["N"] for s in snaps]
    rich = len(snaps) >= 3 and len(set(Ns)) >= 2

    # ---- players-tracked forecast ----
    N_target = None
    if rich:
        fitN = wls(xs, [math.log(max(1, n)) for n in Ns], wts)
        if fitN:
            aN, bN, _ = fitN
            N_target = math.exp(aN + bN * math.log1p(hrs(t_target)))
    if N_target is None:
        window_h = min((t_target - snaps[0]["t"]).total_seconds() / 3600.0 + 1.0,
                       N_WINDOW_CAP)
        N_target = N_last * (N_GROWTH ** window_h)
    N_target = max(N_last, N_target)

    # ---- rho / g from the intervals between distinct snapshots ----
    rho = [None] * ntr
    g = [None] * ntr
    if len(snaps) >= 2:
        for i in range(ntr):
            rhos, gs = [], []
            for k in range(1, len(snaps)):
                dN = snaps[k]["N"] - snaps[k - 1]["N"]
                dt = (snaps[k]["t"] - snaps[k - 1]["t"]).total_seconds() / 3600.0
                if dt <= 0:
                    continue
                E0 = (snaps[k - 1]["rates"][i] or 0.0) * snaps[k - 1]["N"] / 100.0
                E1 = (snaps[k]["rates"][i] or 0.0) * snaps[k]["N"] / 100.0
                dE = E1 - E0
                rho_k = None
                if dN > 0:
                    rho_k = max(0.0, min(1.0, dE / dN))
                    rhos.append(rho_k)
                rem = max(0.0, snaps[k - 1]["N"] - E0)
                if rem > 0:
                    prog = dE - (rho_k if rho_k is not None else 0.0) * max(0, dN)
                    gs.append(max(0.0, min(0.5, prog / rem / dt)))
            if rhos:
                rho[i] = median(rhos[-4:])
            if gs:
                g[i] = median(gs[-4:])
    for i in range(ntr):
        r = r_last[i]
        if rho[i] is None:
            rho[i] = (r / 100.0) * min(1.0, r / KAPPA_DIV)
        if g[i] is None:
            g[i] = G_PRIOR * min(1.0, r / 2.0)

    # ---- saturating trend fit (rich history only) ----
    Lsat = [None] * ntr
    r_sat = [None] * ntr
    resid_std = [0.0] * ntr
    slope = [0.0] * ntr
    if rich:
        for i in range(ntr):
            ys, xx, ww = [], [], []
            for k, s in enumerate(snaps):
                r = s["rates"][i]
                if r is None:
                    continue
                ys.append(r)
                xx.append(xs[k])
                ww.append(wts[k])
            if len(xx) >= 3:
                fit = wls(xx, [logit(r) for r in ys], ww)
                if fit:
                    slope[i] = fit[1]
                sf = saturating_fit(xx, ys, ww)
                if sf:
                    L, a, b, sd = sf
                    Lsat[i] = L
                    xt = math.log1p(hrs(t_target))
                    p = 1.0 / (1.0 + math.exp(-(a + b * xt)))
                    r_sat[i] = L * p
                    resid_std[i] = sd

    # ---- mechanistic ODE with saturation ----
    dt_h = max(0.0, (t_target - last["last_seen"]).total_seconds() / 3600.0)
    steps = max(1, int(math.ceil(dt_h / 0.5)))
    dt_step = dt_h / steps if steps else 0.0
    logN0 = math.log(max(1, N_last))
    logN1 = math.log(max(1, N_target))
    E = list(E_last)
    N = float(N_last)
    for s in range(steps):
        N_next = math.exp(logN0 + (logN1 - logN0) * (s + 1) / steps)
        dN = N_next - N
        for i in range(ntr):
            if Lsat[i] is not None:
                cap = Lsat[i] / 100.0 * N_next
            else:
                cap = min(N_next, max(E[i], 3.0 * r_last[i] / 100.0 * N_next))
            E[i] += rho[i] * dN + g[i] * max(0.0, cap - E[i]) * dt_step
            E[i] = min(E[i], N_next)
        N = N_next
    r_mech = [E[i] / N * 100.0 if N > 0 else r_last[i] for i in range(ntr)]

    rows = []
    for i in range(ntr):
        lg_m = logit(r_mech[i])
        if rich and r_sat[i] is not None and slope[i] >= 0:
            lg = 0.5 * lg_m + 0.5 * logit(r_sat[i])
        else:
            lg = lg_m
        move = abs(lg - logit(r_last[i]))
        sig = math.sqrt(SIG0 ** 2 + (SIG_MOVE * move) ** 2 + resid_std[i] ** 2)
        qs = [round4(clamp(inv_logit(lg + z * sig))) for z in Z]
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophies[i]["id"], "q": qs})
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        out = forecast_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as f:
            json.dump(out, f, separators=(",", ":"))
            f.write("\n")


if __name__ == "__main__":
    main()
