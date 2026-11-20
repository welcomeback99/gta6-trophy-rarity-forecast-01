# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/welcomeback99/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/welcomeback99/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each GTA VI trophy's Exophase earn rate at the case target. A saturating log-time trend (logistic in log-time, so each rate levels off at its eventual share) is blended with a mechanistic earners/players ODE (marginal earn rate per newly tracked player plus a progress hazard, capped at the eventual share); on a flat launch-night history a dilution prior takes over (tracked players grow ~1.37x/hour, new players earn at a small fraction of the early rate). Quantiles come from a logit spread that grows with the forecast's move from the last read. Python 3.13, standard library only, offline, deterministic.

## Run it

```sh
python3 /bot/main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=8esqycgc&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
