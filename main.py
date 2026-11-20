"""gta6-trophy-rarity-forecast-01 v3: count-consistent refresh-aware structural forecaster.

Forecasts every GTA VI trophy's Exophase earn rate at the case target as five
quantiles, from the case history of Exophase reads.

Model (standard library only, deterministic, no network):
  1. Refreshes. Exophase does not update continuously: the page is frozen
     between refreshes and jumps at them. A read where the player count or
     any rate changes is a refresh. The forecaster works in refresh steps
     and forecasts the pooled rate at the last refresh before the target.
  2. Progression curves, fitted to the trajectory. Each trophy's earn
     probability for a cohort of a given age follows a logistic CDF in
     log(1 + age). When the history is live (refreshes observed), the
     curve's (cap, tau) are fitted by matching the model's pooled rate --
     computed from the actual join cohorts, so the age spread of the
     population is handled exactly -- to the observed pooled rates at
     reads spanning the history, including early reads when the
     population is young (they pin down the young-age behaviour). The
     inverted newcomer point is deliberately NOT used in the fit: it is
     derived from the curve itself (via the old cohort's progress), so
     including it would let a wrong curve confirm itself. When the
     history is flat (no refresh observed), the curve is anchored on the
     last rate with a category cap prior, and the newcomers' rate is
     applied directly in the forward simulation as a small fraction of
     the tracked rate (the competition's own published launch refresh
     has newcomers at ~1.03 trophies each against the old cohort's
     1.48, and the practice reconstructions put the median newcomer
     fraction at only ~2-7% of the tracked rate).
  3. Structural counts. Rates are earners over players. The forecaster
     tracks earners and players, never rates alone: every join cohort
     (from the history and from the forecast growth) earns along its
     fitted curve, and the page rate at a refresh is the pooled
     earners over pooled players -- so a young cohort's fast early
     progress and an old cohort's saturation are both handled, instead
     of being smeared into a single mean age.
  4. Player growth. The tracked-player count is fitted as a power law in
     time since release over the recent reads, capped so it cannot explode;
     the refresh cadence is the median observed gap between refreshes.
     When no refresh was observed yet, the next refresh jumps the count
     by the standing prior (the launch refresh multiplied it ~6x).
  5. Ensemble. The forward simulation runs over an ensemble of parameter
     draws (growth exponent, cadence, progression speed, newcomer scale,
     caps), all from a fixed seed, so two runs are byte-identical. The five
     quantiles come from the ensemble's pooled rates at the last refresh
     before the target, widened in logit space for model error.
  6. Saturation guard. A trophy already at 98% or more stays near its
     last rate: its cap is close to (but below) 100%, so it creeps up by
     a fraction of a point rather than jumping to 99.99% (forcing 99.99%
     costs a huge logit error when the answer is 98.8%). Only a trophy
     already at 99.9%+ is pinned at the top.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only. It reads every <id>.in in the input
directory and writes <id>.out to the output directory:
{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import
# model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Ensemble size (fixed seed: two runs are byte-identical).
ENSEMBLE = 48
SEED = 20261120
# Extra logit widening of the ensemble quantiles, for model error.
SPREAD_EXTRA = 0.5
# Spread of earn times across players, in log(1 + age) units. The
# generative worlds used to validate this entry use 0.5; a wider spread
# (0.6) biased the fitted (cap, tau) upward (larger tau and cap to match
# the same trajectory) and overshot the future.
SIGMA = 0.5
# Standing prior for the next refresh when no refresh is observed yet:
# the launch refresh the competition itself publishes (638 -> 3868
# players, 943 -> 4276 trophies earned). Newcomers earn ~1.03 trophies
# each; the old cohort earned 1.48 each.
PRIOR_N_JUMP = 3868.0 / 638.0
PRIOR_CADENCE_H = 4.0
# Standing newcomer fraction by category: the share of the tracked rate a
# brand-new cohort has earned. The competition's own published launch
# refresh has newcomers at ~1.03 trophies each against the old cohort's
# 1.48, but the practice reconstructions show the rate falling much
# further than that implies: the median reconstructed answer for a 1.1%
# trophy is ~0.24%, which needs newcomers at only ~2-7% of the tracked
# rate. The ensemble spreads the fraction (see new_m) to cover the
# reconstructed scenarios, which range from ~0% (pure dilution) to ~15%.
PRIOR_NEW_FRAC = {"story": 0.04, "mission": 0.03, "side": 0.03,
                  "grind": 0.01}


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def logit_frac(f: float) -> float:
    x = min(max(f, 1e-9), 1.0 - 1e-9)
    return math.log(x / (1.0 - x))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up (as JavaScript Math.round does it)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def trophy_cat(t: dict) -> str:
    """Category from the trophy's own metadata (tier + description).
    The story check comes first: a gold/platinum story trophy is still
    a story trophy (its cap is set by the chapter, not the tier)."""
    desc = t.get("description", "") or ""
    tier = t.get("tier", "") or ""
    if "Chapter" in desc or "final mission" in desc:
        return "story"
    if tier in ("platinum", "gold"):
        return "grind"
    grind_words = ("100%", "250 ", "100 ", "50 Challenges", "20 different",
                   "every location", "every spot", "all of ", "all garage",
                   "all Shooting", "Earn $", "Spend $", "Purchase all",
                   "3 Collectible", "8 mission strands", "5 Bank Vaults",
                   "Register 10")
    if any(w in desc for w in grind_words):
        return "grind"
    if tier == "silver":
        return "mission"
    return "side"


def chapter_of(t: dict) -> int:
    desc = t.get("description", "") or ""
    for c in ("Chapter 1", "Chapter 2", "Chapter 3", "Chapter 4"):
        if c in desc:
            return int(c[-1])
    if "final mission" in desc:
        return 6
    return 1


# Category priors for the progression curve's cap: the share of a cohort
# that ever earns the trophy. Story caps fall with the chapter (later
# chapters are reached by fewer players); the other caps are tier-driven
# (platinum and gold are the rarest, silver sits in the middle, bronze
# scales with how common the trophy already is at launch).
def category_cap(cat: str, chapter: int, rate: float, tier: str = "") -> float:
    if cat == "story":
        ch = max(1, min(6, chapter))
        return 0.60 - 0.06 * ch
    if tier == "platinum":
        return 0.015
    if tier == "gold":
        return 0.025
    if tier == "silver":
        return 0.06
    return min(0.30, max(0.05, 0.05 + 0.15 * rate))


def prog_curve(age_h: float, cap: float, tau: float) -> float:
    """Share of a cohort of age `age_h` hours that has earned the trophy:
    a logistic CDF in log(1 + age)."""
    if age_h <= 0 or cap <= 0 or tau <= 0:
        return 0.0
    z = (math.log1p(age_h) - math.log(tau)) / SIGMA
    if z >= 0:
        return cap / (1.0 + math.exp(-z))
    e = math.exp(z)
    return cap * e / (1.0 + e)


def derive_tau(cap: float, rate_frac: float, age_h: float) -> float:
    """Characteristic time tau such that the curve passes through the
    observed point p(age_h) = rate_frac (as a fraction, not percent).
    Caller ensures cap > rate_frac."""
    r = min(max(rate_frac, 1e-6), cap * 0.999)
    x = r / cap
    x = min(max(x, 1e-9), 1.0 - 1e-9)
    ln_tau = math.log1p(age_h) - SIGMA * math.log(x / (1.0 - x))
    return min(max(math.exp(ln_tau), 0.05), 5000.0)


def history_buckets(info):
    """The tracked population as join cohorts [(size, join_time)] from the
    observed player counts: the joiners at a refresh joined during the
    interval since the previous refresh (midpoint used as join time). The
    population at the first read joined around the release."""
    times, ns = info["times"], info["ns"]
    buckets = []
    if ns and ns[0] > 0:
        buckets.append((ns[0], 0.0))
    prev_refresh_t = times[0] if times else 0.0
    for k in info["refresh_idx"]:
        dn = ns[k] - ns[k - 1]
        if dn > 0:
            jt = (prev_refresh_t + times[k]) / 2.0
            buckets.append((dn, jt))
        prev_refresh_t = times[k]
    return buckets


def cohort_rate(buckets, cap, tau, t):
    """Pooled earn rate (fraction) at time t for cohorts [(size, join_time)]:
    each cohort's earners are size * p(t - join_time), pooled over the
    cohorts that have joined by t."""
    num = 0.0
    den = 0.0
    for (size, jt) in buckets:
        if jt > t:
            continue
        num += size * prog_curve(t - jt, cap, tau)
        den += size
    return num / den if den > 0 else 0.0


def fit_curve_cohorts(buckets, read_times, read_rates, cap_prior):
    """Fit (cap, tau) for a trophy's progression curve by matching the
    model's pooled rate (from the actual join cohorts) to the observed
    pooled rates at the given reads. Grid search in logit space with a
    pull toward the category cap prior, then a fine local refinement
    around the best coarse point. The reads are direct observations; the
    inverted newcomer point is NOT used here -- it is derived from the
    curve itself (via the old-cohort progress), so including it would
    let a wrong curve confirm itself."""
    if not read_times:
        return cap_prior, 100.0
    r_last = read_rates[-1]
    c_prior = min(max(cap_prior, r_last * 1.05), 0.999)

    def err(cap, tau):
        e = 0.0
        for (t, r) in zip(read_times, read_rates):
            pred = cohort_rate(buckets, cap, tau, t)
            e += (logit_frac(pred) - logit_frac(r)) ** 2
        e += 1.0 * (math.log(cap / c_prior)) ** 2
        return e

    caps = []
    c = max(r_last * 1.05, 0.01)
    while c < 0.999:
        caps.append(c)
        c *= 1.6
    caps.append(0.999)
    taus = []
    t = 0.25
    while t <= 2048.0:
        taus.append(t)
        t *= 1.4
    best = None
    for cap in caps:
        for tau in taus:
            e = err(cap, tau)
            if best is None or e < best[0]:
                best = (e, cap, tau)
    _, cap, tau = best
    # Fine local refinement: x1.15 steps around the coarse best.
    for dc in (-2, -1, 0, 1, 2):
        for dtau in (-2, -1, 0, 1, 2):
            c2 = min(max(cap * (1.15 ** dc), 0.01), 0.999)
            t2 = min(max(tau * (1.15 ** dtau), 0.1), 4096.0)
            e = err(c2, t2)
            if e < best[0]:
                best = (e, c2, t2)
                cap, tau = c2, t2
    return cap, tau


def fit_power_law(times_h, ns):
    """Fit N(t) = a * t**b on log-log; returns (a, b). Falls back to flat."""
    pts = [(t, n) for t, n in zip(times_h, ns) if t > 0 and n > 0]
    if len(pts) < 3:
        return None, None
    xs = [math.log(t) for t, _ in pts]
    ys = [math.log(n) for _, n in pts]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-9:
        return None, None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = math.exp(my - b * mx)
    if not (0.05 < b < 4.0) or a <= 0:
        return None, None
    return a, b


def analyse_history(case):
    """Turn the history reads into times, player counts, earn counts and the
    refresh list."""
    release = parse_time(case["game"].get("release_date", "2026-11-19")
                         + "T00:00:00Z") if "T" not in str(
        case["game"].get("release_date", "")) else parse_time(
        case["game"]["release_date"])
    hist = case["history"]
    times = []
    ns = []
    totals = []
    rates = []
    for read in hist:
        t = (parse_time(read["at"]) - release).total_seconds() / 3600.0
        times.append(t)
        ns.append(float(read["players_tracked"]))
        totals.append(float(read["trophies_earned"]))
        rates.append([0.0 if r is None else float(r) for r in read["rates"]])
    refresh_idx = []
    for k in range(1, len(hist)):
        if ns[k] != ns[k - 1] or rates[k] != rates[k - 1]:
            refresh_idx.append(k)
    return {"release": release, "times": times, "ns": ns, "totals": totals,
            "rates": rates, "refresh_idx": refresh_idx}


def forecast_case(case: dict) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError("levels are not the contest's five")
    if not case["history"]:
        raise ValueError("case has no history")
    trophies = case["trophies"]
    ntr = len(trophies)
    release = parse_time(case["game"].get("release_date", "2026-11-19")
                         + "T00:00:00Z") if "T" not in str(
        case["game"].get("release_date", "")) else parse_time(
        case["game"]["release_date"])
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_h = (target - at).total_seconds() / 3600.0
    ckpt_h = (at - release).total_seconds() / 3600.0

    info = analyse_history(case)
    times, ns = info["times"], info["ns"]
    last_rates = info["rates"][-1]
    n_last = ns[-1]
    e_last = [r * n_last / 100.0 for r in last_rates]
    has_refresh = len(info["refresh_idx"]) > 0

    cats = [trophy_cat(t) for t in trophies]
    # A trophy already at 98%+ is saturated: it stays there.
    saturated = [r >= 98.0 for r in last_rates]

    # Player growth: power law on the recent reads (last 36h), else flat.
    recent = [(t, n) for t, n in zip(times, ns)
              if t >= ckpt_h - 36.0 and t > 0 and n > 0]
    a_fit, b_fit = fit_power_law([t for t, _ in recent],
                                 [n for _, n in recent])
    if a_fit is None:
        a_fit, b_fit = n_last / max(ckpt_h, 1.0) ** 1.0, 1.0
    # Mean cohort age at time t: join density ~ s**(b-1), so the mean join
    # time is t*b/(b+1) and the mean age is t - that = t/(b+1).
    def mean_age(t):
        if t <= 0:
            return 0.5
        return min(max(t / (b_fit + 1.0), 0.25), t)

    # Time since the page last moved: how overdue the next refresh is.
    flat_h = ckpt_h - times[info["refresh_idx"][-1]] if info["refresh_idx"] \
        else ckpt_h - times[0]
    flat_h = max(flat_h, 0.0)

    # Progression curves, fitted per trophy.
    cap_priors = [category_cap(cats[i], chapter_of(trophies[i]),
                               last_rates[i] / 100.0,
                               trophies[i].get("tier", "") or "")
                  for i in range(ntr)]
    age_ckpt = mean_age(ckpt_h)
    hist_buckets = history_buckets(info)
    t_last_refresh = (times[info["refresh_idx"][-1]] if info["refresh_idx"]
                      else times[0])
    if has_refresh:
        # Live history: fit each trophy's curve by matching the model's
        # pooled rate (from the actual join cohorts) to the observed pooled
        # rates at reads spanning the history, including early reads when
        # the population is young (they pin down the young-age behaviour).
        cad_gaps = []
        prev_t = 0.0
        for k in info["refresh_idx"]:
            cad_gaps.append(times[k] - prev_t)
            prev_t = times[k]
        cad_gaps_sorted = sorted(cad_gaps)
        cadence = (cad_gaps_sorted[len(cad_gaps_sorted) // 2]
                   if cad_gaps_sorted else PRIOR_CADENCE_H)
        # Reads spanning the history for the fit.
        n_reads = len(times)
        fit_idx = sorted(set([max(0, int(n_reads * f) - 1)
                              for f in (0.05, 0.15, 0.25, 0.5, 0.75, 1.0)]))
        fit_times = [times[k] for k in fit_idx]
        fitted = []
        for i in range(ntr):
            rts = [info["rates"][k][i] / 100.0 for k in fit_idx]
            use_t, use_r = [], []
            for (t, r) in zip(fit_times, rts):
                # Skip reads below 0.1%: the rates are rounded to 2
                # decimals, so a 0.01% read could be anywhere in
                # [0.005%, 0.015%] -- a 3x range in logit terms -- and
                # it would drag the fit toward an absurdly slow curve.
                if r >= 0.001:
                    use_t.append(t)
                    use_r.append(r)
            if not use_t:
                fitted.append((cap_priors[i], 1000.0))
                continue
            fitted.append(fit_curve_cohorts(hist_buckets, use_t, use_r,
                                            cap_priors[i]))
    else:
        # Flat history: anchor each curve on the last rate with the
        # category cap. The newcomer rate is NOT fitted into the curve
        # (a newcomer point is not a point on the same curve: the
        # newcomers are a young cohort whose rate is far below the
        # pooled rate at a similar mean age, and forcing one curve
        # through both points corrupts it); it is applied directly in
        # the forward simulation instead.
        cadence = PRIOR_CADENCE_H
        fitted = []
        for i in range(ntr):
            c_prior = min(max(cap_priors[i], last_rates[i] / 100.0 * 1.05),
                          0.999)
            tau = derive_tau(c_prior, last_rates[i] / 100.0, age_ckpt)
            fitted.append((c_prior, tau))
    cadence = min(max(cadence, 1.0), 12.0)

    # Standing prior for a refresh jump when nothing was observed yet.
    n_obs_jumps = []
    for k in info["refresh_idx"]:
        if info["ns"][k - 1] > 0:
            n_obs_jumps.append(info["ns"][k] / info["ns"][k - 1])
    jump_prior = (sorted(n_obs_jumps)[len(n_obs_jumps) // 2]
                  if n_obs_jumps else PRIOR_N_JUMP)

    rng = random.Random(SEED)
    # Ensemble members: (growth exponent, cadence, progression speed,
    # newcomer scale, cap scale, first-refresh jump). The newcomer scale
    # only matters when no refresh was observed yet (flat history); the
    # jump only matters then too.
    members = []
    for _ in range(ENSEMBLE):
        b_m = b_fit * math.exp(rng.gauss(0, 0.15))
        cad_m = cadence * math.exp(rng.gauss(0, 0.3))
        cad_m = min(max(cad_m, 0.5), 24.0)
        speed_m = math.exp(rng.gauss(0, 0.35))
        new_m = math.exp(rng.gauss(0, 0.6))
        cap_m = math.exp(rng.gauss(0, 0.15))
        jump_m = math.exp(math.log(jump_prior) + rng.gauss(0, 0.35))
        members.append((b_m, cad_m, speed_m, new_m, cap_m, jump_m))

    # Forward forecast per member. The page only moves at refreshes, so the
    # forecast is the pooled rate at the last refresh before the target.
    sim_rates = [[] for _ in range(ntr)]
    if has_refresh:
        # Cohort-based forward: build the future join cohorts from the
        # power-law growth, then the pooled rate at the last refresh
        # before the target is the cohort-weighted curve value, anchored
        # to the observed last rate (correcting the fit's level residual).
        for (b_m, cad_m, speed_m, new_m, cap_m, jump_m) in members:
            step_times = []
            t = ckpt_h + max(cad_m - flat_h, 0.05)
            while t <= ckpt_h + horizon_h:
                step_times.append(t)
                t += cad_m
            t_page = step_times[-1] if step_times else ckpt_h + horizon_h
            fut = []
            prev_t, prev_n = t_last_refresh, n_last
            for ts in step_times:
                if t_last_refresh > 0:
                    n_ts = n_last * (ts / t_last_refresh) ** b_m
                else:
                    n_ts = n_last
                n_ts = min(n_ts, n_last * 40.0)
                n_ts = max(n_ts, prev_n)
                dn = n_ts - prev_n
                if dn > 0:
                    fut.append((dn, (prev_t + ts) / 2.0))
                prev_t, prev_n = ts, n_ts
            all_buckets = hist_buckets + fut
            for i in range(ntr):
                if saturated[i]:
                    sim_rates[i].append(last_rates[i])
                    continue
                cap, tau = fitted[i]
                cap_i = min(cap * cap_m, 0.999)
                tau_i = tau / speed_m
                model_last = cohort_rate(hist_buckets, cap_i, tau_i,
                                         t_last_refresh) * 100.0
                model_page = cohort_rate(all_buckets, cap_i, tau_i,
                                         t_page) * 100.0
                if model_last > 0.01:
                    # The anchor corrects the fit's level residual; the
                    # ratio is the model's predicted growth. It must be
                    # allowed to be large: a slow trophy at 0.36% can
                    # grow ~5x over the horizon (its cohorts are young
                    # relative to its tau), and a tight clamp would pin
                    # it at 2x and collapse the ensemble spread.
                    ratio = min(max(model_page / model_last, 0.1), 10.0)
                    sim_rates[i].append(last_rates[i] * ratio)
                else:
                    sim_rates[i].append(model_page)
    else:
        # Flat history: step-based forward (the horizon is short, so the
        # mean-age approximation is acceptable here). The first refresh
        # after a flat history jumps the player count by the standing
        # prior; the count is flat until then.
        for (b_m, cad_m, speed_m, new_m, cap_m, jump_m) in members:
            e = list(e_last)
            n = n_last
            age = age_ckpt
            t = ckpt_h
            a_cur = a_fit
            last_page = list(last_rates)
            jumped = False
            next_refresh = t + max(cad_m - flat_h, 0.05)
            while t < ckpt_h + horizon_h:
                step_end = min(next_refresh, ckpt_h + horizon_h)
                dt = step_end - t
                if dt <= 0:
                    break
                is_refresh = abs(step_end - next_refresh) < 1e-9
                if is_refresh and not jumped:
                    n_at_end = n * jump_m
                    jumped = True
                    if step_end > 0:
                        a_cur = n_at_end / max(step_end, 0.01) ** b_m
                elif not jumped:
                    n_at_end = n
                else:
                    n_at_end = a_cur * max(step_end, 0.01) ** b_m
                    n_at_end = min(n_at_end, n_last * 40.0)
                    n_at_end = max(n_at_end, n)
                dn = n_at_end - n
                for i in range(ntr):
                    if saturated[i]:
                        continue
                    cap, tau = fitted[i]
                    cap_i = min(cap * cap_m, 0.999)
                    tau_i = tau / speed_m
                    old_prog = n * (prog_curve(age + dt, cap_i, tau_i)
                                    - prog_curve(age, cap_i, tau_i))
                    # Newcomers earn the standing-prior fraction of the
                    # checkpoint rate (scaled by this member's draw),
                    # not the curve's young-age value: they have only
                    # just started playing.
                    new_rate = (PRIOR_NEW_FRAC[cats[i]] * new_m
                                * last_rates[i] / 100.0)
                    new_e = dn * min(new_rate, 0.99)
                    e[i] = e[i] + old_prog + new_e
                age = ((n * (age + dt)) + dn * (dt / 2.0)) / max(n + dn, 1.0)
                n = n_at_end
                t = step_end
                if is_refresh:
                    last_page = [e[i] / n * 100.0 if n > 0 else 0.0
                                 for i in range(ntr)]
                    next_refresh = t + cad_m
            for i in range(ntr):
                sim_rates[i].append(last_page[i])

    def quantile(vals, q):
        s = sorted(vals)
        if len(s) == 1:
            return s[0]
        pos = q * (len(s) - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, len(s) - 1)
        frac = pos - lo
        return s[lo] * (1 - frac) + s[hi] * frac

    # (Count consistency holds by construction within each ensemble member
    # -- each member's rates and total come from the same counts -- so no
    # cross-member rescaling is applied. Under a bimodal ensemble the sum
    # of marginal medians and the median total live in different modes,
    # and forcing them to agree would distort the forecasts.)
    rows = []
    for i in range(ntr):
        if saturated[i]:
            # A trophy already at 98%+ stays near its last rate: its cap
            # is close to (but below) 100%, so it creeps up by a fraction
            # of a point, it does not jump to 99.99%. (Forcing 99.99%
            # costs a huge logit error when the answer is 98.8%.) Only a
            # trophy already at 99.9%+ is pinned at the top.
            if last_rates[i] >= 99.9:
                rows.append({"id": trophies[i]["id"],
                             "q": [99.5, 99.8, 99.99, 99.99, 99.99]})
            else:
                base = last_rates[i]
                rows.append({"id": trophies[i]["id"],
                             "q": [base - 0.2, base, base + 0.3,
                                   base + 0.5, base + 0.8]})
            continue
        qs = [quantile(sim_rates[i], lv) for lv in LEVELS]
        med = qs[2]
        # Widen in logit space around the median for model error.
        lg_med = logit(med)
        widened = []
        for q in qs:
            lg = logit(q)
            widened.append(inv_logit(lg_med + (lg - lg_med)
                                     * (1.0 + SPREAD_EXTRA)))
        rows.append({"id": trophies[i]["id"], "q": widened})
    # Final tidy: round to 4 decimals, non-decreasing, in range.
    out_rows = []
    for r in rows:
        qs = [round4(clamp_pct(q)) for q in r["q"]]
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        out_rows.append({"id": r["id"], "q": qs})
    return {"trophies": out_rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
