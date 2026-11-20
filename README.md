# GTA VI Trophy Rarity Forecast

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, the
percent of the players Exophase tracks for the game who will have earned it
at 2026-11-26T00:00:00Z, a week after launch, as five quantiles per trophy.

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/welcomeback99/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/welcomeback99/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Entry for the [make.fun](https://make.fun/) bounty
`gta6-trophy-rarity-forecast-01` ($250 pool).

## Model

`main.py` is a count-consistent refresh-aware structural forecaster
(Python 3, standard library only, deterministic, no network):

- **Refreshes.** Exophase updates in jumps, not continuously: the page is
  frozen between refreshes. The forecaster works in refresh steps and
  forecasts the pooled rate at the last refresh before the target.
- **Progression curves, fitted to the trajectory.** Each trophy's earn
  probability for a cohort of a given age follows a logistic CDF in
  log(1 + age). When the history is live, the curve's (cap, tau) are
  fitted by matching the model's pooled rate -- computed from the actual
  join cohorts, so the age spread of the population is handled exactly --
  to the observed pooled rates at reads spanning the history, including
  early reads when the population is young. The inverted newcomer point
  is deliberately not used in the fit: it is derived from the curve
  itself, so including it would let a wrong curve confirm itself.
- **Structural counts.** It tracks earners and players, never rates
  alone: every join cohort (from the history and from the forecast
  growth) earns along its fitted curve, and the page rate is the pooled
  earners over pooled players -- so a young cohort's fast early progress
  and an old cohort's saturation are both handled, instead of being
  smeared into a single mean age. The forecast is anchored to the
  observed last rate, correcting the fit's level residual.
- **Player growth.** The tracked-player count is fitted as a power law in
  time since release over the recent reads; the refresh cadence is the
  median observed gap between refreshes. With no refresh observed yet,
  the next refresh jumps the count by a standing prior (the launch
  refresh multiplied it ~6x) and newcomers earn a small standing-prior
  fraction of the tracked rate.
- **Ensemble.** The forward simulation runs over an ensemble of parameter
  draws (growth exponent, cadence, progression speed, newcomer scale,
  cap scale, first-refresh jump) from a fixed seed. The five quantiles
  come from the ensemble, widened in logit space for model error.
- **Saturation guard.** A trophy already at 98% or more stays near its
  last rate: its cap is close to (but below) 100%, so it creeps up by a
  fraction of a point rather than jumping to 99.99%.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory and writes `<id>.out` to the
output directory:
`{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}`.
