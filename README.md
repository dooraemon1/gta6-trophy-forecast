# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/dooraemon1/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/dooraemon1/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

An entry for make.fun's *Forecast how many GTA VI players earn each trophy in week one* bounty.

## What it does

`main.py` reads each case file in `$MAKEFUN_INPUT_DIR` (Python 3.13, standard library only,
no network) and writes one forecast per case to `$MAKEFUN_OUTPUT_DIR`: for every trophy of the
case, five quantiles of its earn rate at the case's target, on the logit scale the grader scores.

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## The idea

A trophy's Exophase earn rate never falls: nobody un-earns a trophy, and the page only ever
adds players to the numerator. The house baseline treats the rate as a symmetric random walk, so
half of its band sits below a value the rate cannot go under. This entry puts every quantile at or
above the last read and spends the whole spread upward.

Where the history carries signal the centre comes from a weighted least-squares fit of
`logit(rate)` against `log1p(hours since release)`: newer reads weigh more, a short series shrinks
its slope toward the pooled slope of all trophies, and the residual spread widens the band. Where
the history is stale (consecutive reads identical, as Exophase's cache makes them) the fit is blind,
so a floor on the drift and on the spread keeps the forecast honest about the coming refresh.

Quantiles are `max(0.02, drift + sigma * z_tau)` logit units above the last read, sigma combining
the fit residual, the horizon's random walk and an extrapolation term, then rounded to four
decimals half up and written non-decreasing, one row per trophy id.

## License

MIT
