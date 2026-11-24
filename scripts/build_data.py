"""Fetch market data + news, compute per-stock factor scores and risk, write docs/data.json.

Runs daily in GitHub Actions. Sources (free, no API keys):
  * Yahoo Finance via the `yfinance` package: price history + company fundamentals
  * Google News RSS: recent headlines per stock and for the market overall

Scores are cross-sectional percentiles (0-100) within the universe, so "80" means
"better than 80% of the stocks in the list on this factor". Risk is an absolute
1-5 level based on volatility, drawdown, beta and leverage.
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verdict import consult, earnings_context, exit_plan  # noqa: E402
import history  # noqa: E402

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "data.json"
HIST = ROOT / "docs" / "history.json"
WATCH_FILE = ROOT / "watchlist.txt"
TICKERS_FILE = ROOT / "tickers.txt"

BENCH = {"US": "SPY", "TA": "^TA125.TA"}
MACRO = {"VIX": "^VIX", "TNX": "^TNX", "SPY": "SPY", "QQQ": "QQQ", "TA125": "^TA125.TA",
         "OIL": "CL=F", "GOLD": "GC=F", "USDILS": "ILS=X"}

# --------------------------------------------------------------------------------------
# Small finance sentiment lexicon (inspired by Loughran-McDonald word lists)
POS = """beat beats beating surge surges surged soar soars soared jump jumps jumped rally rallies
rallied gain gains gained rise rises rising rose record upgrade upgraded upgrades outperform
strong stronger strongest growth grow grows boost boosts boosted bullish buy profit profits
profitable exceed exceeds exceeded raise raises raised higher top tops win wins won approval
approved breakthrough expand expands expansion partnership deal optimistic rebound rebounds
recover recovers recovery dividend buyback upbeat positive momentum robust accelerate""".split()
NEG = """miss misses missed plunge plunges plunged drop drops dropped fall falls fell sink sinks
sank slump slumps slumped tumble tumbles tumbled decline declines declined downgrade downgraded
downgrades underperform weak weaker weakest loss losses lose loses losing bearish sell cut cuts
lower lawsuit sue sued probe investigation fraud recall layoff layoffs warning warns warned
risk risks concern concerns fear fears crash default bankruptcy delay delayed halt halted
fine fined penalty tariff tariffs sanction sanctions war slowdown recession negative disappoint
disappoints disappointing struggle struggles volatile volatility uncertainty antitrust""".split()
POS_SET, NEG_SET = set(POS), set(NEG)
NEGATORS = {"not", "no", "never", "without", "fails", "failed"}


def headline_sentiment(text: str) -> float:
    words = re.findall(r"[a-z']+", text.lower())
    score = 0
    for i, w in enumerate(words):
        s = 1 if w in POS_SET else -1 if w in NEG_SET else 0
        if s and i > 0 and words[i - 1] in NEGATORS:
            s = -s
        score += s
    return max(-1.0, min(1.0, score / 2))


def fetch_news(query: str, days: int = 14, limit: int = 12) -> list[dict]:
    url = ("https://news.google.com/rss/search?q=" + urllib.parse.quote(query + f" when:{days}d")
           + "&hl=en-US&gl=US&ceid=US:en")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 stock-picker"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            root = ET.fromstring(r.read())
    except Exception as e:  # noqa: BLE001
        print(f"  news failed for {query!r}: {e}", file=sys.stderr)
        return []
    items = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        link = it.findtext("link") or ""
        src = it.findtext("source") or ""
        try:
            dt = parsedate_to_datetime(it.findtext("pubDate"))
        except Exception:  # noqa: BLE001
            dt = None
        if dt and dt < cutoff:
            continue
        # Google appends " - Source" to titles
        if src and title.endswith(" - " + src):
            title = title[: -len(src) - 3]
        items.append({"title": title, "link": link, "source": src,
                      "date": dt.isoformat() if dt else None,
                      "sentiment": round(headline_sentiment(title), 2)})
        if len(items) >= limit:
            break
    return items


def news_score(items: list[dict]) -> float | None:
    if not items:
        return None
    now = datetime.now(timezone.utc)
    num = den = 0.0
    for it in items:
        age = 7.0
        if it["date"]:
            age = max(0.0, (now - datetime.fromisoformat(it["date"])).total_seconds() / 86400)
        w = math.exp(-age / 7)  # recent headlines count more
        num += w * it["sentiment"]
        den += w
    return num / den if den else None


# --------------------------------------------------------------------------------------
def load_universe() -> list[dict]:
    """tickers.txt: `SYMBOL | Hebrew name` lines, grouped under `# --- group name` headers."""
    out = []
    group = ""
    seen = set()
    for raw in TICKERS_FILE.read_text(encoding="utf-8").splitlines():
        if raw.strip().startswith("# ---"):
            group = raw.strip()[5:].strip()
            continue
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        sym = parts[0].upper()
        if sym in seen:
            continue
        seen.add(sym)
        out.append({"symbol": sym, "name_he": parts[1] if len(parts) > 1 else "", "group": group,
                    "market": "TA" if sym.endswith(".TA") else "US"})
    # stocks on the watchlist are always analysed, even if they are not in tickers.txt
    have = {u["symbol"] for u in out}
    for w in load_watchlist():
        if w["symbol"] not in have:
            have.add(w["symbol"])
            out.append({"symbol": w["symbol"], "name_he": "", "group": "", "market": "TA" if w["symbol"].endswith(".TA") else "US"})
    return out


def price_metrics(close: pd.Series, bench: pd.Series | None, volume: pd.Series | None = None) -> dict:
    close = close.dropna()
    if len(close) < 130:
        return {}
    r = close.pct_change().dropna()
    last = float(close.iloc[-1])

    def ret(n):
        return float(close.iloc[-1] / close.iloc[-n - 1] - 1) if len(close) > n else None

    y = close.iloc[-252:]
    ry = r.iloc[-252:]
    vol = float(ry.std() * np.sqrt(252))
    down = ry[ry < 0]
    downside = float(np.sqrt((down ** 2).mean()) * np.sqrt(252)) if len(down) else 0.0
    dd = float((y / y.cummax() - 1).min())
    ma50 = float(close.iloc[-50:].mean())
    ma200 = float(close.iloc[-200:].mean()) if len(close) >= 200 else None
    hi52 = float(y.max())
    beta = None
    if bench is not None:
        b = bench.pct_change().dropna()
        j = pd.concat([ry, b], axis=1, join="inner").dropna().iloc[-252:]
        if len(j) > 100 and j.iloc[:, 1].var() > 0:
            beta = float(j.iloc[:, 0].cov(j.iloc[:, 1]) / j.iloc[:, 1].var())
    m12 = ret(252)
    m1 = ret(21)
    mom_12_1 = ((1 + m12) / (1 + m1) - 1) if (m12 is not None and m1 is not None) else None
    var95 = float(-np.percentile(ry, 5))  # one-day 95% historical VaR
    # Wilder RSI(14)
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi_s = (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(100.0)
    rsi = float(rsi_s.iloc[-1])
    rsi_5ago = float(rsi_s.iloc[-6])
    ma20 = float(close.iloc[-20:].mean())
    ma20_5ago = float(close.iloc[-25:-5].mean())
    hi20 = float(close.iloc[-20:].max())
    # relative strength: 3 months vs the market index
    rs_3m = None
    if bench is not None:
        bb = bench.dropna()
        if len(bb) > 64 and ret(63) is not None:
            rs_3m = ret(63) - float(bb.iloc[-1] / bb.iloc[-64] - 1)
    # volume: unusual activity and whether it comes on up days (buying) or down days (selling)
    vol_ratio = updown_vol = None
    if volume is not None:
        v = volume.reindex(close.index).fillna(0)
        v50 = float(v.iloc[-55:-5].mean())
        if v50 > 0:
            vol_ratio = float(v.iloc[-5:].mean()) / v50
        rr20, vv20 = r.iloc[-20:], v.reindex(r.index).iloc[-20:]
        dn_v = float(vv20[rr20 < 0].sum())
        if dn_v > 0:
            updown_vol = float(vv20[rr20 > 0].sum()) / dn_v
    dvol20 = float(r.iloc[-20:].std())
    spark = [round(float(v), 2) for v in close.iloc[-252::5]]
    return {"price": round(last, 2), "ret_1m": m1, "ret_3m": ret(63), "ret_6m": ret(126),
            "ret_1y": m12, "mom_12_1": mom_12_1, "vol": vol, "downside_vol": downside,
            "max_dd": dd, "beta": beta, "var95": var95,
            "above_ma200": (last > ma200) if ma200 else None,
            "ma50_vs_200": (ma50 / ma200 - 1) if ma200 else None,
            "from_high": last / hi52 - 1, "rsi14": rsi, "ret_5d": ret(5), "ret_1d": ret(1),
            "ma20": round(ma20, 2), "dvol20": dvol20, "ma50": round(ma50, 2), "ma200": round(ma200, 2) if ma200 else None,
            "low_3m": round(float(close.iloc[-63:].min()), 2), "spark": spark,
            "rsi_5ago": rsi_5ago, "ma20_rising": ma20 > ma20_5ago,
            "crossed_ma20": bool(last > ma20 and float(close.iloc[-6]) < ma20_5ago),
            "from_hi20": last / hi20 - 1, "rs_3m": rs_3m, "vol_ratio": vol_ratio, "updown_vol": updown_vol}


INFO_KEYS = ["shortName", "longName", "sector", "industry", "marketCap", "currency",
             "trailingPE", "forwardPE", "pegRatio", "trailingPegRatio", "priceToBook",
             "enterpriseToEbitda", "profitMargins", "operatingMargins", "returnOnEquity",
             "debtToEquity", "currentRatio", "revenueGrowth", "earningsGrowth",
             "freeCashflow", "dividendYield", "targetMeanPrice", "recommendationMean",
             "numberOfAnalystOpinions", "currentPrice", "grossMargins", "totalCash", "totalDebt",
             "operatingCashflow", "recommendationKey", "earningsTimestamp", "earningsTimestampStart",
             "shortPercentOfFloat", "longBusinessSummary", "quoteType", "totalAssets"]


def fetch_earnings(t) -> list[dict]:
    """Recent and upcoming earnings dates with EPS surprise, newest first."""
    try:
        df = t.get_earnings_dates(limit=8)
    except Exception as e:  # noqa: BLE001
        print(f"  earnings dates failed: {e}", file=sys.stderr)
        return []
    if df is None or df.empty:
        return []
    out = []
    for idx, row in df.sort_index(ascending=False).iterrows():
        est, rep = clean(row.get("EPS Estimate")), clean(row.get("Reported EPS"))
        sur = (rep - est) / abs(est) if est not in (None, 0) and rep is not None else None
        out.append({"date": pd.Timestamp(idx).isoformat(), "estimate": est, "reported": rep,
                    "surprise": round(sur, 4) if sur is not None else None})
    return out


def fetch_all(universe, yf):
    syms = [u["symbol"] for u in universe]
    extra = sorted((set(BENCH.values()) | set(MACRO.values())) - set(syms))
    print(f"Downloading prices for {len(syms) + len(extra)} symbols...")
    hist = yf.download(syms + extra, period="3y", interval="1d", auto_adjust=True,
                       progress=False, group_by="column", threads=True)
    closes = hist["Close"]
    volumes = hist["Volume"] if "Volume" in hist else None
    infos = {}
    for i, s in enumerate(syms):
        for attempt in range(3):
            try:
                t = yf.Ticker(s)
                info = t.info or {}
                infos[s] = {k: info.get(k) for k in INFO_KEYS}
                is_fund = (info.get("quoteType") or "").upper() in ("ETF", "MUTUALFUND")
                infos[s]["_earnings"] = fetch_earnings(t) if not (s.endswith(".TA") or is_fund) else []
                break
            except Exception as e:  # noqa: BLE001
                print(f"  info {s} attempt {attempt + 1}: {e}", file=sys.stderr)
                time.sleep(2 + attempt * 3)
        else:
            infos[s] = {}
        if i % 10 == 9:
            print(f"  fundamentals {i + 1}/{len(syms)}")
        time.sleep(0.25)
    return closes, infos, volumes


# --------------------------------------------------------------------------------------
def pct_rank(values: dict, higher_better=True) -> dict:
    s = pd.Series({k: v for k, v in values.items() if v is not None and np.isfinite(v)}, dtype=float)
    if s.empty:
        return {}
    r = s.rank(pct=True, ascending=higher_better) * 100
    return r.round(1).to_dict()


def combine(parts: list[tuple[dict, float]], keys) -> dict:
    out = {}
    for k in keys:
        num = den = 0.0
        for d, w in parts:
            if k in d:
                num += d[k] * w
                den += w
        out[k] = round(num / den, 1) if den >= 0.5 * sum(w for _, w in parts) else None
    return out


def clean(x):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def risk_level(m: dict, f: dict) -> tuple[int, float, list[str]]:
    """Absolute risk 1 (low) .. 5 (very high) plus a 0-100 continuous score."""
    vol = m.get("vol") or 0.3
    dd = abs(m.get("max_dd") or 0.2)
    beta = m.get("beta")
    de = clean(f.get("debtToEquity"))
    mcap = clean(f.get("marketCap"))
    pts = 0.0
    pts += np.interp(vol, [0.12, 0.25, 0.40, 0.65], [0, 25, 45, 65])
    pts += np.interp(dd, [0.08, 0.20, 0.35, 0.60], [0, 6, 12, 18])
    if beta is not None:
        pts += np.interp(beta, [0.6, 1.0, 1.6, 2.2], [-3, 0, 4, 7])
    if de is not None:
        pts += np.interp(de, [50, 150, 300], [0, 3, 6])
    if mcap is not None:
        pts += np.interp(math.log10(max(mcap, 1)), [9, 10, 11], [6, 3, 0])
    pts = float(np.clip(pts, 0, 100))
    level = 1 if pts < 20 else 2 if pts < 32 else 3 if pts < 45 else 4 if pts < 58 else 5
    why = []
    why.append(f"תנודתיות שנתית {vol * 100:.0f}%")
    why.append(f"ירידה מקסימלית בשנה האחרונה {dd * 100:.0f}%")
    if beta is not None:
        why.append(f"בטא {beta:.2f} (רגישות לשוק)")
    if de is not None and de > 150:
        why.append(f"מינוף גבוה (חוב/הון {de:.0f}%)")
    if mcap is not None and mcap < 1e10:
        why.append("חברה קטנה יחסית, נזילות ותנודות גבוהות יותר")
    return level, round(pts, 1), why


def explain(row: dict) -> tuple[list[str], list[str]]:
    pros, cons = [], []
    m, f, sc = row["m"], row["f"], row["scores"]

    def p(x):
        return f"{x * 100:+.0f}%"

    if sc.get("momentum") is not None:
        if sc["momentum"] >= 70:
            pros.append(f"מומנטום חזק: {p(m['ret_6m'])} בחצי שנה" + (", מעל ממוצע 200 יום" if m.get("above_ma200") else ""))
        elif sc["momentum"] <= 30:
            cons.append(f"מומנטום חלש: {p(m['ret_6m'])} בחצי שנה" + (", מתחת לממוצע 200 יום" if m.get("above_ma200") is False else ""))
    fpe = clean(f.get("forwardPE"))
    if sc.get("value") is not None:
        if sc["value"] >= 70:
            pros.append("תמחור זול ביחס לשאר המניות" + (f" (מכפיל רווח עתידי {fpe:.1f})" if fpe else ""))
        elif sc["value"] <= 25:
            cons.append("תמחור יקר ביחס לשאר המניות" + (f" (מכפיל רווח עתידי {fpe:.1f})" if fpe else ""))
    roe, pm = clean(f.get("returnOnEquity")), clean(f.get("profitMargins"))
    if sc.get("quality") is not None:
        if sc["quality"] >= 70:
            pros.append("עסק איכותי: רווחיות גבוהה" + (f", תשואה על ההון {roe * 100:.0f}%" if roe else ""))
        elif sc["quality"] <= 25:
            cons.append("איכות נמוכה: רווחיות חלשה או מינוף גבוה" + (f" (שולי רווח {pm * 100:.0f}%)" if pm is not None else ""))
    rg = clean(f.get("revenueGrowth"))
    if sc.get("growth") is not None:
        if sc["growth"] >= 70 and rg is not None:
            pros.append(f"צמיחה מהירה בהכנסות ({rg * 100:+.0f}% בשנה)")
        elif sc["growth"] <= 25 and rg is not None:
            cons.append(f"צמיחה חלשה בהכנסות ({rg * 100:+.0f}% בשנה)")
    up = row.get("upside")
    if up is not None and clean(f.get("numberOfAnalystOpinions")) and f["numberOfAnalystOpinions"] >= 3:
        if up > 0.15:
            pros.append(f"אנליסטים רואים פוטנציאל של {up * 100:.0f}% עד מחיר היעד הממוצע")
        elif up < -0.03:
            cons.append(f"המחיר כבר מעל מחיר היעד הממוצע של האנליסטים ({up * 100:.0f}%)")
    ns = row.get("news_raw")
    if ns is not None and row["news_count"] >= 3:
        if ns > 0.2:
            pros.append("טון החדשות בשבועיים האחרונים חיובי")
        elif ns < -0.15:
            cons.append("טון החדשות בשבועיים האחרונים שלילי, כדאי לקרוא את הכותרות")
    if m.get("from_high") is not None and m["from_high"] < -0.35:
        cons.append(f"נסחרת {abs(m['from_high']) * 100:.0f}% מתחת לשיא השנתי")
    return pros, cons


def market_regime(closes: pd.DataFrame, market_news: list[dict]) -> dict:
    def last(sym):
        s = closes.get(sym)
        return float(s.dropna().iloc[-1]) if s is not None and len(s.dropna()) else None

    def trend(sym):
        s = closes.get(sym)
        if s is None:
            return None
        s = s.dropna()
        if len(s) < 200:
            return None
        return {"last": round(float(s.iloc[-1]), 2), "ma200": round(float(s.iloc[-200:].mean()), 2),
                "above_ma200": bool(s.iloc[-1] > s.iloc[-200:].mean()),
                "ret_1d": float(s.iloc[-1] / s.iloc[-2] - 1),
                "ret_1m": float(s.iloc[-1] / s.iloc[-22] - 1), "ret_3m": float(s.iloc[-1] / s.iloc[-64] - 1),
                "dd": float(s.iloc[-1] / s.iloc[-252:].max() - 1)}

    spy, ta, qqq = trend("SPY"), trend("^TA125.TA"), trend("QQQ")
    vix = last("^VIX")
    tnx_s = closes.get("^TNX")
    tnx = last("^TNX")
    tnx_chg = None
    if tnx_s is not None and len(tnx_s.dropna()) > 64:
        t = tnx_s.dropna()
        tnx_chg = float(t.iloc[-1] - t.iloc[-64])
    mnews = news_score(market_news)

    pts, notes = 0, []
    if spy:
        if spy["above_ma200"]:
            pts += 1; notes.append("המניות הגדולות בארה\"ב בעלייה לאורך החודשים האחרונים")
        else:
            pts -= 1; notes.append("המניות הגדולות בארה\"ב בירידה לאורך החודשים האחרונים")
    if vix is not None:
        if vix < 16:
            pts += 1; notes.append(f"המשקיעים רגועים (מדד הפחד נמוך: {vix:.0f})")
        elif vix > 25:
            pts -= 1; notes.append(f"יש פחד בשוק, והמחירים קופצים חזק (מדד הפחד גבוה: {vix:.0f})")
        else:
            notes.append(f"רמת לחץ רגילה בשוק (מדד הפחד: {vix:.0f})")
    if tnx_chg is not None and tnx_chg > 0.4:
        pts -= 0.5; notes.append("הריבית בארה\"ב עלתה בחודשים האחרונים, וזה בדרך כלל לוחץ על מחירי המניות")
    if mnews is not None:
        if mnews < -0.15:
            pts -= 0.5; notes.append("רוב הכותרות על הבורסה בשבועיים האחרונים שליליות")
        elif mnews > 0.2:
            pts += 0.5; notes.append("רוב הכותרות על הבורסה בשבועיים האחרונים חיוביות")
    if ta:
        notes.append("בורסת תל אביב " + ("בעלייה" if ta["above_ma200"] else "בירידה") + " לאורך החודשים האחרונים")
    state = "positive" if pts >= 1.5 else "negative" if pts <= -1 else "neutral"
    return {"state": state, "points": pts, "notes": notes, "spy": spy, "qqq": qqq, "ta125": ta,
            "vix": vix, "tnx": tnx, "tnx_chg_3m": tnx_chg, "news_score": mnews,
            "oil": last("CL=F"), "gold": last("GC=F"), "usdils": last("ILS=X")}


def load_watchlist() -> list[dict]:
    """watchlist.txt lines: SYMBOL | target price (optional) | note (optional)"""
    out = []
    if not WATCH_FILE.exists():
        return out
    for line in WATCH_FILE.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        out.append({"symbol": parts[0].upper(), "target": clean(parts[1]) if len(parts) > 1 and parts[1] else None,
                    "note": parts[2] if len(parts) > 2 else ""})
    return out


def watch_alerts(watch: list[dict], stocks: list[dict], changes: list[dict]) -> list[dict]:
    by = {s["symbol"]: s for s in stocks}
    ch = {c["symbol"]: c for c in changes}
    out = []
    for w in watch:
        s = by.get(w["symbol"])
        item = {**w, "found": bool(s), "alerts": []}
        if s:
            v = s["consult"]["verdict"]
            item.update(name=s.get("name_he") or s["name"], price=s["price"], verdict=v)
            c = ch.get(w["symbol"])
            if c and v == "buy":
                item["alerts"].append(f"עברה ל\"לקנות עכשיו\" (הייתה {VERDICT_HE[c['from']]})")
            elif c:
                item["alerts"].append(f"ההמלצה השתנתה: {VERDICT_HE[c['from']]} \u2190 {VERDICT_HE[v]}")
            if w["target"]:
                prev = (s["metrics"].get("ret_1d") is not None) and s["price"] / (1 + s["metrics"]["ret_1d"])
                if s["price"] <= w["target"] and (not prev or prev > w["target"]):
                    item["alerts"].append(f"ירדה למחיר היעד שלך ({w['target']:g})")
                item["at_target"] = s["price"] <= w["target"]
        out.append(item)
    return out


VERDICT_HE = {"buy": "לקנות עכשיו", "wait": "לחכות", "avoid": "לא לקנות"}


def build(universe, closes: pd.DataFrame, infos: dict, news: dict, market_news: list[dict],
          volumes: pd.DataFrame | None = None) -> dict:
    rows = []
    for u in universe:
        s = u["symbol"]
        if s not in closes:
            print(f"  no prices for {s}", file=sys.stderr)
            continue
        bench = closes.get(BENCH[u["market"]])
        vser = volumes[s] if volumes is not None and s in volumes else None
        m = price_metrics(closes[s], bench, vser)
        if not m:
            continue
        f = infos.get(s, {})
        tgt = clean(f.get("targetMeanPrice"))
        cur = clean(f.get("currentPrice")) or m["price"]
        items = news.get(s, [])
        rows.append({**u, "m": m, "f": f, "news": items, "news_count": len(items),
                     "news_raw": news_score(items),
                     "upside": (tgt / cur - 1) if tgt and cur else None})

    keys = [r["symbol"] for r in rows]
    g = lambda fn: {r["symbol"]: fn(r) for r in rows}  # noqa: E731
    fi = lambda k: g(lambda r: clean(r["f"].get(k)))  # noqa: E731

    momentum = combine([(pct_rank(g(lambda r: r["m"]["mom_12_1"])), 0.4),
                        (pct_rank(g(lambda r: r["m"]["ret_6m"])), 0.3),
                        (pct_rank(g(lambda r: r["m"]["ma50_vs_200"])), 0.3)], keys)
    pe = {k: v for k, v in fi("forwardPE").items() if v and v > 0}
    tpe = {k: v for k, v in fi("trailingPE").items() if v and v > 0}
    peg = {k: v for k, v in {k: clean(r["f"].get("trailingPegRatio") or r["f"].get("pegRatio"))
                              for k, r in zip(keys, rows)}.items() if v and v > 0}
    ev = {k: v for k, v in fi("enterpriseToEbitda").items() if v and v > 0}
    value = combine([(pct_rank(pe, False), 0.35), (pct_rank(tpe, False), 0.2),
                     (pct_rank(peg, False), 0.2), (pct_rank(ev, False), 0.25)], keys)
    quality = combine([(pct_rank(fi("returnOnEquity")), 0.35), (pct_rank(fi("profitMargins")), 0.3),
                       (pct_rank(fi("operatingMargins")), 0.15), (pct_rank(fi("debtToEquity"), False), 0.2)], keys)
    growth = combine([(pct_rank(fi("revenueGrowth")), 0.6), (pct_rank(fi("earningsGrowth")), 0.4)], keys)
    analyst = combine([(pct_rank(g(lambda r: r["upside"])), 0.5),
                       (pct_rank(fi("recommendationMean"), False), 0.5)], keys)
    sent = {k: (round(50 + 50 * v, 1) if v is not None else None)
            for k, v in g(lambda r: r["news_raw"]).items()}
    lowrisk_rank = pct_rank(g(lambda r: r["m"]["vol"]), False)

    regime = market_regime(closes, market_news)
    # sector rotation: which sectors are attracting money this month (average 1-month move per sector)
    sec_of = lambda r: r["f"].get("sector") or r.get("group") or ""  # noqa: E731
    by_sec: dict[str, list[float]] = {}
    for r in rows:
        if r["m"].get("ret_1m") is not None and sec_of(r):
            by_sec.setdefault(sec_of(r), []).append(r["m"]["ret_1m"])
    sec_avg = {k: float(np.median(v)) for k, v in by_sec.items() if len(v) >= 3}
    sec_rank = pct_rank(sec_avg)
    stocks = []
    for r in rows:
        k = r["symbol"]
        r["scores"] = {"momentum": momentum.get(k), "value": value.get(k), "quality": quality.get(k),
                       "growth": growth.get(k), "analysts": analyst.get(k), "news": sent.get(k),
                       "low_vol": lowrisk_rank.get(k)}
        lvl, rpts, rwhy = risk_level(r["m"], r["f"])
        pros, cons = explain(r)
        f = r["f"]
        ectx = earnings_context(f, f.get("_earnings"))
        neg_titles = [n["title"] for n in r["news"] if n["sentiment"] < 0][:2]
        consultation = consult(r["m"], f, {**ectx, "upside": r["upside"], "news_score": r["news_raw"],
                                           "news_count": r["news_count"], "neg_headlines": neg_titles,
                                           "sector_rank": (sec_rank.get(sec_of(r)) or 0) / 100 if sec_of(r) in sec_rank else None,
                                           "sector_ret_1m": sec_avg.get(sec_of(r)),
                                           "risk_level": lvl, "market": regime})
        is_etf = (f.get("quoteType") or "").upper() in ("ETF", "MUTUALFUND")
        if is_etf:  # a fund is not a company: keep the wording right
            consultation["summary"] = consultation["summary"].replace("חברה טובה", "קרן טובה").replace("חברה במצב טוב", "קרן במצב טוב")
        stocks.append({
            "symbol": k, "market": r["market"], "group": r.get("group", ""),
            "type": "etf" if is_etf else "stock",
            "name": f.get("shortName") or f.get("longName") or k, "name_he": r["name_he"],
            "sector": f.get("sector") or "", "industry": f.get("industry") or "",
            "currency": f.get("currency") or ("ILA" if r["market"] == "TA" else "USD"),
            "price": r["m"]["price"], "market_cap": clean(f.get("marketCap")),
            "scores": r["scores"], "risk": {"level": lvl, "score": rpts, "why": rwhy},
            "metrics": {k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in r["m"].items() if k2 != "spark"},
            "spark": r["m"]["spark"],
            "fund": {k2: clean(f.get(k2)) for k2 in ["trailingPE", "forwardPE", "priceToBook", "profitMargins",
                                                     "returnOnEquity", "debtToEquity", "revenueGrowth",
                                                     "earningsGrowth", "dividendYield", "targetMeanPrice",
                                                     "recommendationMean", "numberOfAnalystOpinions"]},
            "upside": r["upside"], "news_score": r["news_raw"], "news": r["news"][:8],
            "pros": pros, "cons": cons, "consult": consultation,
            "about": (f.get("longBusinessSummary") or "")[:600],
            "earnings": {"next": ectx.get("next_earnings"), "days": ectx.get("days_to_earnings"),
                         "history": [e for e in (f.get("_earnings") or []) if e["surprise"] is not None][:4]},
        })
    # ---- track record, daily changes, watchlist alerts ----------------------------------
    today = datetime.now(timezone.utc).date().isoformat()
    bench_of = {u["symbol"]: BENCH[u["market"]] for u in universe}
    hist = history.load(HIST)
    changes = history.update(hist, stocks, today)
    HIST.write_text(json.dumps(hist, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    track = {"live": history.evaluate_live(hist, closes, bench_of),
             "sim": history.simulate(closes, [s["symbol"] for s in stocks if s["type"] == "stock"], bench_of,
                                     price_metrics, exit_plan, step=10 if len(stocks) <= 150 else 20)}
    watch = watch_alerts(load_watchlist(), stocks, changes)
    return {"generated_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
            "changes": changes, "track": track, "watchlist": watch,
            "count": len(stocks), "market": regime, "market_news": market_news[:12], "stocks": stocks}


def main():
    import yfinance as yf

    universe = load_universe()
    closes, infos, volumes = fetch_all(universe, yf)
    def news_for(u):
        nm = (infos.get(u["symbol"]) or {}).get("shortName") or u["symbol"].replace(".TA", "")
        nm = re.sub(r"\b(Inc|Corp|Corporation|Ltd|Limited|plc|Co|Holdings?|Group|Class [A-C])\b\.?,?", "", nm).strip(" ,.")
        q = f'"{nm}" stock' if u["market"] == "US" else f'"{nm}" Israel'
        items = fetch_news(q)
        time.sleep(0.3)
        return u["symbol"], items

    # a few parallel requests keep the run short now that the list has hundreds of stocks
    from concurrent.futures import ThreadPoolExecutor
    news = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, (sym, items) in enumerate(pool.map(news_for, universe)):
            news[sym] = items
            if i % 50 == 49:
                print(f"  news {i + 1}/{len(universe)}")
    market_news = fetch_news("stock market OR Wall Street OR Federal Reserve", days=5, limit=15)
    market_news += fetch_news("Tel Aviv Stock Exchange OR Bank of Israel", days=7, limit=6)
    data = build(universe, closes, infos, news, market_news, volumes)
    if data["count"] < len(universe) * 0.5:
        sys.exit("Too few stocks processed, refusing to publish a half-empty dataset")
    write_split(data)
    print(f"Wrote {OUT} with {data['count']} stocks")


SUMMARY_KEYS = ["symbol", "market", "group", "type", "name", "name_he", "sector", "currency", "price", "upside",
                "market_cap"]


def write_split(data: dict) -> None:
    """data.json = the small list the home page needs; docs/s/<SYMBOL>.json = one stock's full check.
    Keeps the first load on a phone fast."""
    sdir = OUT.parent / "s"
    sdir.mkdir(parents=True, exist_ok=True)
    keep = set()
    small = []
    for st in data["stocks"]:
        c = st["consult"]
        small.append({**{k: st.get(k) for k in SUMMARY_KEYS},
                      "risk": {"level": st["risk"]["level"]},
                      "consult": {"verdict": c["verdict"], "score": c["score"], "summary": c["summary"],
                                  "opp": c.get("opp"), "entry": c.get("entry"), "signals": (c.get("signals") or [])[:2],
                                  "exit": c.get("exit") or {}},
                      "metrics": {k: st["metrics"].get(k) for k in ("ret_1d", "ret_5d", "ret_1m", "ret_3m", "ret_6m",
                                                                      "ret_1y", "above_ma200", "from_high")},
                      "spark": st["spark"][::3]})
        fn = st["symbol"] + ".json"
        keep.add(fn)
        (sdir / fn).write_text(json.dumps(st, ensure_ascii=False, separators=(",", ":"), default=str), encoding="utf-8")
    for old in sdir.glob("*.json"):
        if old.name not in keep:
            old.unlink()
    out = {**data, "stocks": small}
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":"), default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
