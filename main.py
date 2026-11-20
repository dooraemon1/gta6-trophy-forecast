"""Forecast GTA VI trophy earn rates: log-log growth fit per trophy with pooled shrinkage.

Reads every <id>.in in MAKEFUN_INPUT_DIR, writes <id>.out (five logit-scale quantiles per trophy)
to MAKEFUN_OUTPUT_DIR. Python 3.13, standard library only, deterministic.
"""

import json
import math
import os
import sys
from pathlib import Path

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6449, -0.6745, 0.0, 0.6745, 1.6449]
RELEASE = 1794528000.0  # 2026-11-19T00:00:00Z
DAY = 86400.0
SIGMA_FLOOR = 0.055      # logit units, guards a stale read against a small refresh
SIGMA_RW_DAY = 0.42      # random-walk scale per sqrt(day), below the house 0.5
MIN_SIGMA = 0.085        # never pretend to be certain


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(z: float) -> float:
    if z > 30:
        return 99.99
    if z < -30:
        return 0.01
    return 100.0 / (1.0 + math.exp(-z))


def round4(x: float):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value.is_integer() else value


def hours_of(read: dict) -> float:
    ts = read.get("at") or read.get("slot")
    if isinstance(ts, (int, float)):
        t = float(ts)
        if t > 1e11:  # milliseconds
            t /= 1000.0
    else:
        from datetime import datetime, timezone
        t = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()
    return max(0.02, (t - RELEASE) / 3600.0)


def wls(xs, ys, ws):
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    sse = sum(w * r * r for w, r in zip(ws, resid))
    var = sse / max(1e-9, sw - 2.0)
    return a, b, math.sqrt(max(0.0, var))


def forecast_case(case: dict) -> dict:
    trophies = case["trophies"]
    history = [h for h in case["history"] if h]
    if not history:
        raise ValueError("no history")
    target_h = hours_of({"at": case["target"]}) if case.get("target") else hours_of(history[-1]) + case["horizon_minutes"] / 60.0
    u_target = math.log1p(target_h)
    last = history[-1]
    n_tr = len(trophies)

    # per-trophy series
    series = []
    for i in range(n_tr):
        pts = []
        for h in history:
            r = h["rates"][i] if i < len(h["rates"]) else None
            if r is None:
                continue
            pts.append((hours_of(h), logit(float(r))))
        series.append(pts)

    pool_bs = []
    fits = []
    for i, pts in enumerate(series):
        if len(pts) >= 3:
            us = [math.log1p(t) for t, _ in pts]
            ys = [y for _, y in pts]
            # newer reads weigh more, but keep early signal
            tmax = max(t for t, _ in pts)
            ws = [math.exp((t - tmax) / 18.0) for t, _ in pts]
            fit = wls(us, ys, ws)
        else:
            fit = None
        fits.append(fit)
        if fit and len(pts) >= 4:
            pool_bs.append(fit[1])

    pool_b = sorted(pool_bs)[len(pool_bs) // 2] if pool_bs else 0.0
    # a hair of global growth when every trophy is still flat: the site's rates drift up
    pool_b = max(pool_b, 0.0) if pool_bs else 0.25

    rows = []
    for i, t in enumerate(trophies):
        r_last = last["rates"][i] if i < len(last["rates"]) else None
        y_last = logit(float(r_last)) if r_last is not None else logit(0.5)
        pts = series[i]
        u_last = math.log1p(hours_of(last))

        if fits[i] and len(pts) >= 3:
            a, b, sd = fits[i]
            # shrink the slope toward the pooled one; short series lean on the pool
            w_own = min(1.0, (len(pts) - 2) / 5.0)
            b_hat = w_own * b + (1.0 - w_own) * pool_b
            a_hat = y_last - b_hat * u_last
            sig_fit = sd
        else:
            b_hat = pool_b
            a_hat = y_last - b_hat * u_last
            sig_fit = 0.22

        # stale-read guard: identical consecutive reads make the fit blind, so add slack
        if len(history) >= 2:
            prev = history[-2]["rates"][i] if i < len(history[-2]["rates"]) else None
            if prev is not None and r_last is not None and abs(float(prev) - float(r_last)) < 1e-9:
                sig_fit = max(sig_fit, 0.20)

        d_days = case["horizon_minutes"] / 1440.0
        extra = math.log1p(max(0.0, target_h - hours_of(last))) / math.log1p(max(1.0, target_h))
        sigma = math.sqrt(
            max(SIGMA_FLOOR, sig_fit) ** 2
            + (SIGMA_RW_DAY ** 2) * d_days
            + (0.16 ** 2) * extra
        )
        sigma = max(MIN_SIGMA, sigma)

        # A trophy's earn rate never falls: anchor every quantile on the last read and only
        # put mass above it. The fitted curve sets how far above; a stale read keeps a floor.
        centre = a_hat + b_hat * u_target
        drift = max(0.0, centre - y_last)
        if len(history) >= 2:
            prev = history[-2]["rates"][i] if i < len(history[-2]["rates"]) else None
            if prev is not None and r_last is not None and abs(float(prev) - float(r_last)) < 1e-9:
                drift = max(drift, 0.10)  # stale read: the next refresh only moves it up
        qs = []
        for z in Z:
            delta = max(0.02, drift + sigma * z)
            qs.append(round4(inv_logit(y_last + delta)))
        qs = sorted(qs)
        rows.append({"id": t["id"], "q": qs})

    return {"trophies": rows}


def main() -> None:
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(in_dir.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if case.get("levels") != LEVELS:
            raise ValueError("unexpected levels")
        out = forecast_case(case)
        (out_dir / f"{path.stem}.out").write_text(
            json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
