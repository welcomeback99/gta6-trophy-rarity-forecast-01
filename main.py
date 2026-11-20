#!/usr/bin/env python3
"""
GTA VI trophy-rarity forecaster (v2) for the make.fun bounty
"gta6-trophy-rarity-forecast-01".

Models each trophy's Exophase earn rate at the case target as earners /
players, from three engines blended in logit space:

  1. A saturating log-time trend: each rate is a logistic in log-time
     since release, so it levels off at its eventual share.
  2. A mechanistic earners/players ODE: a marginal earn rate per newly
     tracked player plus a progress hazard, capped at the eventual share.
  3. A cohort-age model (new in v2): players join over time (from the
     tracked-players trajectory) and each trophy's earn probability is a
     logistic in log player-age, fitted per trophy from the observed
     snapshots with weak category priors. This catches the week-one
     turnaround: slow trophies first dilute as the crowd floods in, then
     rise as the early cohorts age into them. It only fires on a rich
     history; on a flat launch-night history the dilution prior takes over.

The tracked-players forecast is a recency-weighted log-log fit over the
last 36 hours of snapshots (the launch transient otherwise inflates the
power-law slope several-fold). Quantiles come from a logit spread that
grows with the forecast's move from the last read. Python 3.13, standard
library only, offline, deterministic.
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
W_COHORT = 0.30        # blend weight of the cohort-age median (logit space)


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


# ---------------- v2: cohort-age model ----------------
TAU_GRID = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0]
L_GRID = [0.005, 0.01, 0.02, 0.04, 0.07, 0.12, 0.20, 0.35, 0.55, 0.80, 0.99]
F_SCALE = 0.5  # logistic scale in log-age
# category priors for tau (hours) and L (fraction), from trophy research:
# story is finished in tens of hours, grind takes days of play.
TAU_PRIOR = {"gimme": 2.0, "story": 15.0, "mission": 15.0, "side": 30.0,
             "grind": 120.0}
# L prior by CATEGORY (eventual completion share): story is finished by most
# players, grind by few. Tier alone misleads (a bronze chapter trophy has
# L ~ 0.7). Weak tie-breaker only; the history trajectory dominates.
L_PRIOR_CAT = {"gimme": 0.99, "story": 0.55, "mission": 0.35, "side": 0.25,
               "grind": 0.08}
LAM_TAU = 0.02  # weak prior (mean-sq-logit-residual units)
LAM_L = 0.02


def trophy_cat(t):
    d = t["description"]
    if t["id"] == 2:
        return "gimme"
    if "Chapter" in d or "final mission" in d:
        return "story"
    if "Complete " in d and ("mission" in d.lower() or "Heist" in d
                             or "Play" in d or "Run" in d or "Takedown" in d
                             or "Glory" in d):
        return "mission"
    if (t["tier"] in ("silver", "gold") or "Complete 250" in d
            or "100 locations" in d or "Card sets" in d or "gold medal" in d
            or "50 Challenges" in d or "100% Completion" in d):
        return "grind"
    return "side"


def _f_age(age, tau):
    # logistic in log-age, 0..1
    x = (math.log1p(max(0.0, age)) - math.log(tau)) / F_SCALE
    if x > 30:
        return 1.0
    if x < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def cohort_shapes(snaps, t_target, N_fit, release):
    """Return (times_k, S[k][tau_idx], S_target[tau_idx], N_target_from_grid).

    Builds the join-time distribution on a 1h grid from the first snapshot
    to the target: observed N (step function at refreshes) up to the last
    read, the fitted power law after it. S[k][j] = sum over cohorts of
    joins * f(age at snapshot k; tau_j); E(t_k) = L * S[k][j].
    """
    def hrs(t):
        return max(0.0, (t - release).total_seconds() / 3600.0)

    t0 = hrs(snaps[0]["t"])
    tT = hrs(t_target)
    nsteps = int(math.ceil(tT - t0)) + 1
    grid_t = [t0 + i for i in range(nsteps)]
    # N on the grid
    N_grid = []
    for gt in grid_t:
        # last snapshot at or before this grid time (by its first-seen time)
        nval = snaps[0]["N"]
        for s in snaps:
            if hrs(s["t"]) <= gt + 1e-9:
                nval = s["N"]
            else:
                break
        if gt > hrs(snaps[-1]["last_seen"]) and N_fit is not None:
            aN, bN = N_fit[0], N_fit[1]
            nval = math.exp(aN + bN * math.log1p(gt))
        N_grid.append(max(1.0, float(nval)))
    joins = [0.0] * nsteps
    for i in range(1, nsteps):
        joins[i] = max(0.0, N_grid[i] - N_grid[i - 1])
    # snapshot times (use last_seen as the read time)
    snap_t = [hrs(s["last_seen"]) for s in snaps]
    # f(age) table per tau: ages 0..ceil(tT) hours
    max_age = int(math.ceil(tT)) + 1
    ftab = [[_f_age(a, tau) for a in range(max_age)] for tau in TAU_GRID]
    S = []
    for tk in snap_t:
        row = []
        for j, tau in enumerate(TAU_GRID):
            tot = 0.0
            for i, gt in enumerate(grid_t):
                if gt > tk + 1e-9:
                    break
                age = int(tk - gt)
                if age < len(ftab[j]):
                    tot += joins[i] * ftab[j][age]
            row.append(tot)
        S.append(row)
    S_targ = []
    for j, tau in enumerate(TAU_GRID):
        tot = 0.0
        for i, gt in enumerate(grid_t):
            age = int(tT - gt)
            if age < len(ftab[j]):
                tot += joins[i] * ftab[j][age]
        S_targ.append(tot)
    N_targ_grid = N_grid[-1]
    return snap_t, S, S_targ, N_targ_grid


def cohort_fit(snaps, S, trophies):
    """Fit (L, tau) per trophy by logit-space residual + priors.

    Returns list of (L, tau, resid_std, S_target_value) per trophy (L as a
    fraction; resid_std is the weighted std of logit residuals at the fit).
    """
    out = []
    for i, t in enumerate(trophies):
        cat = trophy_cat(t)
        tau0 = TAU_PRIOR[cat]
        L0 = L_PRIOR_CAT[cat]
        obs = []
        for k, s in enumerate(snaps):
            r = s["rates"][i]
            if r is None:
                continue
            obs.append((k, r, s["N"]))
        if not obs:
            out.append((L0, tau0, 0.5, None))
            continue
        wts = [0.5 ** ((snaps[-1]["last_seen"] - snaps[k]["t"]).total_seconds()
                       / 3600.0 / 24.0) for k, _, _ in obs]
        best = None
        for j, tau in enumerate(TAU_GRID):
            for L in L_GRID:
                se = 0.0
                sw = 0.0
                for (k, r, Nk), w in zip(obs, wts):
                    pred = L * S[k][j] / max(1.0, Nk) * 100.0
                    d = logit(pred) - logit(r)
                    se += w * d * d
                    sw += w
                if sw <= 0:
                    continue
                pen = (LAM_TAU * (math.log(tau) - math.log(tau0)) ** 2
                       + LAM_L * (math.log(L) - math.log(L0)) ** 2)
                obj = se / sw + pen
                if best is None or obj < best[0]:
                    best = (obj, L, tau)
        _, L, tau = best
        # residual std at the fit (logit space, weighted)
        j = TAU_GRID.index(tau)
        se = sw = 0.0
        for (k, r, Nk), w in zip(obs, wts):
            pred = L * S[k][j] / max(1.0, Nk) * 100.0
            d = logit(pred) - logit(r)
            se += w * d * d
            sw += w
        resid = math.sqrt(se / sw) if sw > 0 else 0.5
        out.append((L, tau, resid, None))
    return out


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

    # ---- players-tracked forecast (v2: recent-window log-log fit) ----
    N_target = None
    N_fit = None
    if rich:
        idx36 = [k for k, s in enumerate(snaps)
                 if (last["last_seen"] - s["t"]).total_seconds() / 3600.0 <= 36.0]
        use = None
        if len(set(Ns[k] for k in idx36)) >= 3:
            use = idx36
        else:
            idx100 = [k for k, n in enumerate(Ns) if n >= 100]
            if len(set(Ns[k] for k in idx100)) >= 3:
                use = idx100
        if use is not None:
            fitN = wls([xs[k] for k in use],
                       [math.log(max(1, Ns[k])) for k in use],
                       [wts[k] for k in use])
        else:
            fitN = wls(xs, [math.log(max(1, n)) for n in Ns], wts)
        if fitN:
            aN, bN, _ = fitN
            N_fit = (aN, bN)
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

    # ---- v2 cohort-age model ----
    r_coh = [None] * ntr
    if rich and N_fit is not None:
        try:
            _, S, S_targ, _ = cohort_shapes(snaps, t_target, N_fit, release)
            fits = cohort_fit(snaps, S, trophies)
            for i in range(ntr):
                L, tau, _resid, _ = fits[i]
                E_t = L * S_targ[TAU_GRID.index(tau)]
                r_coh[i] = E_t / max(1.0, N_target) * 100.0
        except Exception:
            r_coh = [None] * ntr

    rows = []
    for i in range(ntr):
        lg_m = logit(r_mech[i])
        if rich and r_sat[i] is not None and slope[i] >= 0:
            lg = 0.5 * lg_m + 0.5 * logit(r_sat[i])
        else:
            lg = lg_m
        # blend in the cohort median when available (v2). Weight 0.30 was
        # the flat optimum on ten generative sim worlds (0.25-0.35 all
        # within noise); heavier weights over-trust the cohort fit.
        # Saturated trophies are exempt: when nearly every tracked player
        # already has the trophy (last rate >= 98%), there is no
        # age-turnaround left for the cohort model to find. The fit is
        # unidentifiable at the ceiling (its logit residual runs > 1.2 on
        # the gimme) and the tau grid bottoms out at 2h, so it cannot
        # represent a near-instant earn; the blend then only drags the
        # median down. Those trophies stay with the trend/ODE engines.
        if r_coh[i] is not None and r_coh[i] > 0 and r_last[i] < 98.0:
            lg = (1.0 - W_COHORT) * lg + W_COHORT * logit(r_coh[i])
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
