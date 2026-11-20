# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/welcomeback99/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/welcomeback99/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each GTA VI trophy's Exophase earn rate at the case target. v2 adds a cohort-age model to the v1 engines: players join over time (from the tracked-players trajectory) and each trophy's earn probability is a logistic in log player-age, fitted per trophy from the observed snapshots with weak category priors. That catches the week-one turnaround — slow trophies first dilute as the crowd floods in, then rise as the early cohorts age into them. The tracked-players forecast is a recency-weighted log-log fit over the last 36 hours of snapshots, so the launch transient can't inflate the growth slope. The other engines are unchanged from v1: a saturating log-time trend blended with a mechanistic earners/players ODE, and a dilution prior on a flat launch-night history. Quantiles come from a logit spread that grows with the forecast's move from the last read. Python 3.13, standard library only, offline, deterministic.

## Run it

```sh
python3 /bot/main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=8esqycgc&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
