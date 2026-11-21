"""Track record of the recommendations.

1. Live log (docs/history.json): every time a stock's verdict changes we append an event
   {d: date, s: symbol, v: verdict, p: price, tg: target, sl: stop}. That also gives the
   "what changed since yesterday" list for the morning summary.
2. Live evaluation: for every past "buy" (and "avoid") event, how the stock did afterwards
   versus its benchmark, and whether it hit the target or the stop first.
3. Simulated backtest: the price-based part of the verdict (trend + entry timing) replayed on
   the last ~2 years of prices. Fundamentals can't be replayed (no point-in-time history),
   so this tests the "when" half of the method, not the "what" half.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from verdict import BAD, GOOD, check_timing, check_trend

MAX_EVENTS = 6000


def load(path: Path) -> dict:
    try:
        h = json.loads(path.read_text(encoding="utf-8"))
        h.setdefault("events", []); h.setdefault("last", {})
        return h
    except Exception:  # noqa: BLE001
        return {"events": [], "last": {}}


def update(hist: dict, stocks: list[dict], today: str) -> list[dict]:
    """Append verdict changes; return today's changes (only for stocks we already knew)."""
    changes = []
    for s in stocks:
        c = s.get("consult") or {}
        v = c.get("verdict")
        if not v:
            continue
        prev = hist["last"].get(s["symbol"])
        prev_v = prev["v"] if prev else None
        if prev_v == v:
            continue
        ex = c.get("exit") or {}
        ev = {"d": today, "s": s["symbol"], "v": v, "p": s["price"], "tg": ex.get("target"), "sl": ex.get("stop")}
        if prev and prev.get("d") == today:
            # re-run on the same day: overwrite instead of logging a flip-flop
            hist["events"] = [e for e in hist["events"] if not (e["s"] == s["symbol"] and e["d"] == today)]
            prev_v = prev.get("from")
            if prev_v == v:  # flipped back to yesterday's verdict: nothing changed after all
                hist["last"][s["symbol"]] = {"v": v, "d": prev.get("d0") or today, "from": None}
                continue
        hist["events"].append(ev)
        hist["last"][s["symbol"]] = {"v": v, "d": today, "from": prev_v, "d0": prev.get("d") if prev else None}
        if prev_v is not None and prev_v != v:
            changes.append({"symbol": s["symbol"], "name": s.get("name_he") or s["name"], "from": prev_v, "to": v,
                            "price": s["price"], "summary": c.get("summary"), "score": c.get("score")})
    hist["events"] = hist["events"][-MAX_EVENTS:]
    return changes


def _forward(close: pd.Series, i: int, n: int):
    return float(close.iloc[i + n] / close.iloc[i] - 1) if i + n < len(close) else None


def _first_hit(close: pd.Series, i: int, target, stop, horizon=63):
    """'target' / 'stop' / None (neither within horizon, or still open)."""
    if not target or not stop:
        return None
    w = close.iloc[i + 1:i + 1 + horizon]
    for v in w:
        if v >= target:
            return "target"
        if v <= stop:
            return "stop"
    return None


def _summ(rows: list[dict]) -> dict:
    def avg(k):
        xs = [r[k] for r in rows if r.get(k) is not None]
        return (round(float(np.mean(xs)), 4), len(xs)) if xs else (None, 0)

    def winrate(k):
        xs = [r[k] for r in rows if r.get(k) is not None]
        return round(sum(1 for x in xs if x > 0) / len(xs), 3) if xs else None

    hits = [r["hit"] for r in rows if r.get("hit")]
    out = {"n": len(rows)}
    for k in ("r1m", "r3m", "x1m", "x3m", "r_now"):
        out[k], out["n_" + k] = avg(k)
    out["win1m"], out["win3m"] = winrate("r1m"), winrate("r3m")
    out["beat3m"] = winrate("x3m")
    out["target_first"] = round(hits.count("target") / len(hits), 3) if hits else None
    out["n_hits"] = len(hits)
    return out


def evaluate_live(hist: dict, closes: pd.DataFrame, bench_of: dict) -> dict:
    out = {}
    for verdict in ("buy", "wait", "avoid"):
        rows = []
        for e in hist["events"]:
            if e["v"] != verdict or e["s"] not in closes:
                continue
            c = closes[e["s"]].dropna()
            i = int(c.index.searchsorted(pd.Timestamp(e["d"]).tz_localize(c.index.tz) if c.index.tz else pd.Timestamp(e["d"])))
            if i >= len(c):
                continue
            b = closes.get(bench_of.get(e["s"]))
            b = b.dropna().reindex(c.index).ffill() if b is not None else None
            r1, r3 = _forward(c, i, 21), _forward(c, i, 63)
            row = {"s": e["s"], "d": e["d"], "r1m": r1, "r3m": r3,
                   "r_now": float(c.iloc[-1] / c.iloc[i] - 1),
                   "hit": _first_hit(c, i, e.get("tg"), e.get("sl")) if verdict == "buy" else None}
            if b is not None:
                b1, b3 = _forward(b, i, 21), _forward(b, i, 63)
                row["x1m"] = r1 - b1 if r1 is not None and b1 is not None else None
                row["x3m"] = r3 - b3 if r3 is not None and b3 is not None else None
            rows.append(row)
        out[verdict] = _summ(rows)
        if verdict == "buy":
            out["recent_buys"] = sorted(rows, key=lambda r: r["d"], reverse=True)[:15]
    first = min((e["d"] for e in hist["events"]), default=None)
    out["since"] = first
    return out


def simulate(closes: pd.DataFrame, symbols: list[str], bench_of: dict, price_metrics, exit_plan,
             step: int = 10) -> dict:
    """Replay trend + timing on history. Signal 'buy' = trend good and timing not bad."""
    groups = {"buy": [], "avoid": [], "all": []}
    for s in symbols:
        if s not in closes:
            continue
        c = closes[s].dropna()
        b = closes.get(bench_of.get(s))
        b = b.dropna().reindex(c.index).ffill() if b is not None else None
        for i in range(210, len(c) - 21, step):
            m = price_metrics(c.iloc[:i + 1], None)
            if not m:
                continue
            m["vol"] = m.get("vol") or 0.3
            tr, tm = check_trend(m, {}, {}), check_timing(m, {}, {})
            r1, r3 = _forward(c, i, 21), _forward(c, i, 63)
            row = {"r1m": r1, "r3m": r3}
            if b is not None:
                b1, b3 = _forward(b, i, 21), _forward(b, i, 63)
                row["x1m"] = r1 - b1 if r1 is not None and b1 is not None else None
                row["x3m"] = r3 - b3 if r3 is not None and b3 is not None else None
            groups["all"].append(row)
            if tr["status"] == GOOD and tm["status"] != BAD:
                ex = exit_plan(m, {}, None)
                row = {**row, "hit": _first_hit(c, i, ex.get("target"), ex.get("stop"))}
                groups["buy"].append(row)
            elif tr["status"] == BAD:
                groups["avoid"].append(row)
    out = {k: _summ(v) for k, v in groups.items()}
    idx = closes.dropna(how="all").index
    out["from"] = str(idx[min(210, len(idx) - 1)].date()) if len(idx) else None
    out["to"] = str(idx[-1].date()) if len(idx) else None
    out["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="minutes")
    return out
