"""Per-stock consultation: run a battery of checks and give a verdict.

Verdicts:
  buy   = "לקנות עכשיו"   good business + healthy uptrend + no timing red flags
  wait  = "לחכות"         the stock is worth owning but the timing is wrong right now
                           (earnings in a few days, overheated after a run-up, trend not confirmed,
                           market under stress), or the picture is mixed
  avoid = "לא לקנות"      weak fundamentals and/or a broken trend, or serious red flags

Every check uses ABSOLUTE thresholds (not ranks within the list), so a stock gets the same verdict
whatever else is in tickers.txt. Each check returns a status (good / ok / bad / na), a short Hebrew
title and an explanation with the actual numbers, so the verdict can always be traced back.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

GOOD, OK, BAD, NA = "good", "ok", "bad", "na"
VAL = {GOOD: 1.0, OK: 0.0, BAD: -1.0}


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _pct(x, d=0):
    # LRI/PDI isolates keep "+12%" readable inside right-to-left Hebrew text
    return f"\u2066{x * 100:+.{d}f}%\u2069"


def _chk(key, group, title, status, text, weight=1.0):
    return {"key": key, "group": group, "title": title, "status": status, "text": text, "weight": weight}


# --------------------------------------------------------------------------------------------
# Individual checks. Each takes (m, f, ctx) and returns a check dict.
#   m   = price metrics from build_data.price_metrics
#   f   = Yahoo fundamentals dict
#   ctx = extra context: risk level, news score, earnings info, market regime


def check_trend(m, f, ctx):
    p, ma50, ma200 = m.get("price"), m.get("ma50"), m.get("ma200")
    r3 = m.get("ret_3m")
    if not p or not ma50:
        return _chk("trend", "tech", "מגמת מחיר", NA, "אין מספיק היסטוריית מחירים", 2.0)
    pts = 0
    parts = []
    if ma200:
        if p > ma200:
            pts += 1; parts.append("מעל ממוצע 200 יום")
        else:
            pts -= 1; parts.append("מתחת לממוצע 200 יום")
        if ma50 > ma200:
            pts += 1; parts.append("ממוצע 50 מעל ממוצע 200 (מגמה עולה ארוכה)")
        else:
            pts -= 1; parts.append("ממוצע 50 מתחת לממוצע 200 (מגמה יורדת ארוכה)")
    if p > ma50:
        pts += 1; parts.append("מעל ממוצע 50 יום")
    else:
        pts -= 1; parts.append("מתחת לממוצע 50 יום")
    if r3 is not None:
        parts.append(f"{_pct(r3)} ב-3 חודשים")
        pts += 0.5 if r3 > 0.03 else -0.5 if r3 < -0.05 else 0
    status = GOOD if pts >= 2.5 else BAD if pts <= -1.5 else OK
    title = {GOOD: "מגמה עולה", OK: "מגמה לא ברורה", BAD: "מגמה יורדת"}[status]
    return _chk("trend", "tech", title, status, ", ".join(parts), 2.0)


def check_timing(m, f, ctx):
    """Is it a good moment to enter, or is the stock overheated / in free fall?"""
    rsi, p, ma50 = m.get("rsi14"), m.get("price"), m.get("ma50")
    r1 = m.get("ret_1m")
    if rsi is None or not ma50:
        return _chk("timing", "tech", "נקודת כניסה", NA, "אין מספיק נתונים", 1.5)
    stretch = p / ma50 - 1
    txt = f"RSI {rsi:.0f}, המחיר {_pct(stretch)} מממוצע 50 יום"
    if rsi >= 75 or stretch > 0.18:
        return _chk("timing", "tech", "מתוחה אחרי עלייה חדה", BAD,
                    txt + ". אחרי ריצה כזו תיקון קצר שכיח. עדיף לחכות להתקררות או להיכנס בחלקים", 1.5)
    if rsi <= 28 and (r1 or 0) < -0.12:
        return _chk("timing", "tech", "בנפילה חדה", BAD,
                    txt + f", {_pct(r1)} בחודש. קשה לתפוס תחתית, עדיף לחכות לייצוב", 1.5)
    if 40 <= rsi <= 68 and -0.03 <= stretch <= 0.10:
        return _chk("timing", "tech", "נקודת כניסה סבירה", GOOD, txt + ", לא מתוחה ולא בנפילה", 1.5)
    return _chk("timing", "tech", "נקודת כניסה ניטרלית", OK, txt, 1.5)


def check_earnings_date(m, f, ctx):
    d = ctx.get("days_to_earnings")
    if d is None:
        return _chk("earn_date", "event", "תאריך הדוח הבא", NA, "לא ידוע מתי הדוח הבא", 0)
    when = ctx.get("next_earnings")
    if 0 <= d <= 7:
        return _chk("earn_date", "event", "דוח רבעוני בעוד ימים ספורים", BAD,
                    f"הדוח צפוי ב-{when} (בעוד {d} ימים). סביב דוחות המניה יכולה לזוז 5%–15% ביום, לכאן או לכאן", 0)
    if 0 <= d <= 21:
        return _chk("earn_date", "event", "דוח רבעוני מתקרב", OK, f"הדוח צפוי ב-{when} (בעוד {d} ימים)", 0)
    return _chk("earn_date", "event", "אין דוח קרוב", GOOD, f"הדוח הבא צפוי ב-{when}" if when else "אין דוח בשלושת השבועות הקרובים", 0)


def check_last_earnings(m, f, ctx):
    s = ctx.get("last_surprise")
    hist = ctx.get("surprises") or []
    if s is None:
        return _chk("earn_last", "fund", "הדוח האחרון", NA, "אין נתוני הפתעה ברווח", 0.8)
    beats = sum(1 for x in hist if x > 0)
    tail = f" (עקפה תחזיות ב-{beats} מתוך {len(hist)} הדוחות האחרונים)" if len(hist) >= 2 else ""
    if s > 0.02:
        return _chk("earn_last", "fund", "הדוח האחרון עקף תחזיות", GOOD, f"רווח למניה גבוה ב-{s * 100:.0f}% מהצפי" + tail, 0.8)
    if s < -0.02:
        return _chk("earn_last", "fund", "הדוח האחרון החטיא תחזיות", BAD, f"רווח למניה נמוך ב-{abs(s) * 100:.0f}% מהצפי" + tail, 0.8)
    return _chk("earn_last", "fund", "הדוח האחרון בקו התחזיות", OK, "הרווח היה קרוב לצפי" + tail, 0.8)


def check_valuation(m, f, ctx):
    fpe, tpe = _f(f.get("forwardPE")), _f(f.get("trailingPE"))
    peg = _f(f.get("trailingPegRatio")) or _f(f.get("pegRatio"))
    eg = _f(f.get("earningsGrowth"))
    sector = f.get("sector") or ""
    pe = fpe if fpe and fpe > 0 else tpe if tpe and tpe > 0 else None
    if pe is None:
        if (tpe is not None and tpe < 0) or (_f(f.get("profitMargins")) or 0) < 0:
            return _chk("value", "fund", "החברה מפסידה, אין מכפיל רווח", BAD,
                        "המחיר נשען על ציפייה לרווחים עתידיים, לא על רווח קיים", 1.2)
        return _chk("value", "fund", "תמחור", NA, "אין נתוני מכפיל", 1.2)
    # Rough "fair" P/E band by sector (long-run norms)
    cheap, rich = {"Financial Services": (10, 18), "Energy": (9, 16), "Utilities": (14, 22),
                   "Real Estate": (14, 30), "Basic Materials": (11, 20), "Consumer Defensive": (17, 26),
                   "Technology": (20, 38), "Communication Services": (16, 30), "Healthcare": (15, 28)
                   }.get(sector, (14, 26))
    txt = f"מכפיל רווח {'עתידי' if pe == fpe else 'נוכחי'} {pe:.1f}"
    if peg:
        txt += f", PEG {peg:.1f}"
    if peg and 0 < peg < 1.2 or pe < cheap:
        return _chk("value", "fund", "תמחור אטרקטיבי", GOOD, txt + " (זול ביחס לסקטור או לצמיחה)", 1.2)
    growth_ok = (eg or 0) > 0.25 or (peg and peg < 2)
    if pe > rich * 1.6 and not growth_ok:
        return _chk("value", "fund", "תמחור יקר מאוד", BAD, txt + ". כמעט אין מקום לאכזבה", 1.2)
    if pe > rich and not growth_ok:
        return _chk("value", "fund", "תמחור יקר", BAD, txt + " (גבוה מהמקובל בסקטור בלי צמיחה שמצדיקה)", 1.2)
    if pe > rich:
        return _chk("value", "fund", "יקרה אבל עם צמיחה", OK, txt + ". המכפיל גבוה, אבל הצמיחה מצדיקה חלק ממנו", 1.2)
    return _chk("value", "fund", "תמחור סביר", OK, txt, 1.2)


def check_profitability(m, f, ctx):
    pm, roe, om = _f(f.get("profitMargins")), _f(f.get("returnOnEquity")), _f(f.get("operatingMargins"))
    if pm is None and roe is None:
        return _chk("quality", "fund", "רווחיות", NA, "אין נתונים", 1.2)
    parts = []
    if pm is not None:
        parts.append(f"שולי רווח נקי {pm * 100:.0f}%")
    if roe is not None:
        parts.append(f"תשואה על ההון {roe * 100:.0f}%")
    if om is not None:
        parts.append(f"שולי רווח תפעולי {om * 100:.0f}%")
    txt = ", ".join(parts)
    if (pm is not None and pm < 0) or (roe is not None and roe < 0):
        return _chk("quality", "fund", "החברה מפסידה כסף", BAD, txt, 1.2)
    if (pm or 0) >= 0.15 or (roe or 0) >= 0.18:
        return _chk("quality", "fund", "עסק רווחי מאוד", GOOD, txt, 1.2)
    if (pm is not None and pm < 0.04) and (roe is None or roe < 0.08):
        return _chk("quality", "fund", "רווחיות נמוכה", BAD, txt, 1.2)
    return _chk("quality", "fund", "רווחיות סבירה", OK, txt, 1.2)


def check_balance(m, f, ctx):
    de, cr = _f(f.get("debtToEquity")), _f(f.get("currentRatio"))
    fcf = _f(f.get("freeCashflow"))
    cash, debt = _f(f.get("totalCash")), _f(f.get("totalDebt"))
    fin = (f.get("sector") or "") in ("Financial Services", "Real Estate", "Utilities")
    parts, bad, good = [], 0, 0
    if de is not None:
        parts.append(f"חוב/הון {de:.0f}%")
        if not fin:
            bad += de > 200
            good += de < 60
    if cash is not None and debt is not None and debt > 0:
        parts.append("מזומן " + ("גדול מ" if cash > debt else "קטן מ") + "החוב")
        good += cash > debt
    if cr is not None and not fin:
        parts.append(f"יחס שוטף {cr:.1f}")
        bad += cr < 0.8
    if fcf is not None:
        parts.append("תזרים מזומנים חופשי " + ("חיובי" if fcf > 0 else "שלילי"))
        bad += fcf < 0
        good += fcf > 0
    if not parts:
        return _chk("balance", "fund", "איתנות פיננסית", NA, "אין נתונים", 1.0)
    txt = ", ".join(parts)
    if fin:
        txt += " (בבנקים, ביטוח ונדל\"ן מינוף גבוה הוא חלק מהמודל העסקי)"
    if bad >= 2:
        return _chk("balance", "fund", "מאזן חלש", BAD, txt, 1.0)
    if bad == 0 and good >= 2:
        return _chk("balance", "fund", "מאזן חזק", GOOD, txt, 1.0)
    return _chk("balance", "fund", "מאזן סביר", OK if bad == 0 else BAD if good == 0 else OK, txt, 1.0)


def check_growth(m, f, ctx):
    rg, eg = _f(f.get("revenueGrowth")), _f(f.get("earningsGrowth"))
    if rg is None and eg is None:
        return _chk("growth", "fund", "צמיחה", NA, "אין נתונים", 1.0)
    parts = []
    if rg is not None:
        parts.append(f"הכנסות {_pct(rg)} בשנה")
    if eg is not None:
        parts.append(f"רווח {_pct(eg)} בשנה")
    txt = ", ".join(parts)
    if (rg or 0) >= 0.12 or ((rg or 0) >= 0.05 and (eg or 0) >= 0.15):
        return _chk("growth", "fund", "צמיחה חזקה", GOOD, txt, 1.0)
    if (rg is not None and rg < -0.03) or (eg is not None and eg < -0.20):
        return _chk("growth", "fund", "העסק מתכווץ", BAD, txt, 1.0)
    return _chk("growth", "fund", "צמיחה מתונה", OK, txt, 1.0)


def check_analysts(m, f, ctx):
    rec, n = _f(f.get("recommendationMean")), _f(f.get("numberOfAnalystOpinions"))
    up = ctx.get("upside")
    if not n or n < 3 or rec is None:
        return _chk("analysts", "sent", "אנליסטים", NA, "מעט מדי אנליסטים מכסים את המניה", 0.8)
    word = "קנייה חזקה" if rec <= 1.6 else "קנייה" if rec <= 2.3 else "החזק" if rec <= 3.0 else "מכירה"
    txt = f"{n:.0f} אנליסטים, המלצה ממוצעת: {word} ({rec:.1f} מתוך 5, נמוך = חיובי)"
    if up is not None:
        txt += f", מחיר יעד ממוצע {_pct(up)} מהמחיר"
    if rec <= 2.2 and (up is None or up > 0.08):
        return _chk("analysts", "sent", "אנליסטים חיוביים", GOOD, txt, 0.8)
    if rec > 3.0 or (up is not None and up < -0.05):
        return _chk("analysts", "sent", "אנליסטים שליליים או שהמחיר עבר את היעד", BAD, txt, 0.8)
    return _chk("analysts", "sent", "אנליסטים ניטרליים", OK, txt, 0.8)


def check_news(m, f, ctx):
    s, n = ctx.get("news_score"), ctx.get("news_count", 0)
    if s is None or n < 3:
        return _chk("news", "sent", "חדשות", NA, "מעט כותרות בשבועיים האחרונים", 0.6)
    neg = ctx.get("neg_headlines") or []
    if s > 0.2:
        return _chk("news", "sent", "חדשות חיוביות", GOOD, f"{n} כותרות בשבועיים האחרונים, הטון חיובי", 0.6)
    if s < -0.15:
        t = f"{n} כותרות בשבועיים האחרונים, הטון שלילי"
        if neg:
            t += ". לדוגמה: “" + neg[0] + "”"
        return _chk("news", "sent", "חדשות שליליות", BAD, t, 0.6)
    return _chk("news", "sent", "חדשות מעורבות", OK, f"{n} כותרות בשבועיים האחרונים, בלי טון ברור", 0.6)


def check_risk(m, f, ctx):
    lvl = ctx.get("risk_level")
    dd, vol = m.get("max_dd"), m.get("vol")
    txt = f"רמת סיכון {lvl} מתוך 5"
    if vol is not None:
        txt += f", תנודתיות שנתית {vol * 100:.0f}%"
    if dd is not None:
        txt += f", ירידה מקסימלית בשנה {abs(dd) * 100:.0f}%"
    st = GOOD if lvl <= 2 else OK if lvl == 3 else BAD
    title = {GOOD: "סיכון נמוך", OK: "סיכון בינוני", BAD: "סיכון גבוה"}[st]
    return _chk("risk", "risk", title, st, txt, 0.8)


def check_market(m, f, ctx):
    mk = ctx.get("market") or {}
    state = mk.get("state")
    if not state:
        return _chk("market", "risk", "מצב השוק", NA, "אין נתונים", 0)
    notes = "; ".join((mk.get("notes") or [])[:2])
    if state == "negative":
        return _chk("market", "risk", "השוק הכללי בלחץ", BAD, notes + ". כשהשוק יורד, רוב המניות יורדות איתו", 0)
    if state == "positive":
        return _chk("market", "risk", "השוק הכללי תומך", GOOD, notes, 0)
    return _chk("market", "risk", "השוק הכללי מעורב", OK, notes, 0)


CHECKS = [check_trend, check_timing, check_earnings_date, check_last_earnings, check_valuation,
          check_profitability, check_balance, check_growth, check_analysts, check_news, check_risk, check_market]

GROUPS = {"tech": "מגמה ותזמון", "fund": "העסק עצמו", "sent": "אנליסטים וחדשות", "event": "אירועים קרובים",
          "risk": "סיכון ושוק"}


def _score(checks, group=None):
    num = den = 0.0
    for c in checks:
        if c["status"] == NA or c["weight"] == 0 or (group and c["group"] != group):
            continue
        num += VAL[c["status"]] * c["weight"]
        den += c["weight"]
    return (num / den if den else None), den


def exit_plan(m: dict, f: dict, upside) -> dict:
    """Stop-loss from recent volatility, target from analysts or a 2:1 reward/risk."""
    p, dv = m.get("price"), m.get("dvol20")
    if not p:
        return {}
    dv = dv if dv else (m.get("vol") or 0.3) / 15.9
    atr = p * dv * 1.25  # close-to-close proxy for the average true range
    stop = min(max(p - 2.5 * atr, p * 0.85), p * 0.95)
    basis_stop = "2.5 פעמים התנודה היומית הממוצעת מתחת למחיר"
    ma50 = m.get("ma50")
    if ma50 and stop < ma50 * 0.98 < p * 0.96:
        stop, basis_stop = ma50 * 0.98, "קצת מתחת לממוצע 50 יום (שבירה שלו מחלישה את המגמה)"
    risk = p - stop
    target, basis_t = p + 2 * risk, "פי 2 מהסיכון (יחס סיכוי/סיכון 1:2)"
    if upside is not None and 1.5 * risk / p <= upside <= 0.6:
        target, basis_t = p * (1 + upside), "מחיר היעד הממוצע של האנליסטים"
    return {"target": round(target, 2), "stop": round(stop, 2), "target_pct": round(target / p - 1, 4),
            "stop_pct": round(stop / p - 1, 4), "rr": round((target - p) / risk, 1),
            "basis_target": basis_t, "basis_stop": basis_stop}


def _r(c):
    return {"status": c["status"], "title": c["title"], "text": c["text"], "plain": c.get("plain"), "key": c["key"]}


SIMPLE_NAME = {"trend": "כיוון המחיר", "timing": "האם זה זמן טוב להיכנס", "earn_date": "דוח רווחים קרוב",
               "earn_last": "הדוח האחרון", "value": "האם המחיר הוגן", "quality": "האם החברה מרוויחה",
               "balance": "חובות ומזומן", "growth": "האם החברה גדלה", "analysts": "מה חושבים המומחים",
               "news": "מה כותבים בחדשות", "risk": "כמה המניה קופצת", "market": "מצב הבורסה כולה"}


def plain(c, m, f, ctx) -> str:
    """The same finding, in words for someone who has never invested."""
    k, st, t = c["key"], c["status"], c["title"]
    pc = lambda x: _pct(x) if x is not None else ""  # noqa: E731
    if st == NA:
        return "אין מספיק מידע לבדיקה הזו"
    if k == "trend":
        r3 = m.get("ret_3m")
        tail = f" ({pc(r3)} בשלושה חודשים)" if r3 is not None else ""
        return {GOOD: "המחיר עולה בחודשים האחרונים" + tail, OK: "המחיר לא בכיוון ברור, לא עולה ולא יורד בעקביות" + tail,
                BAD: "המחיר יורד בחודשים האחרונים" + tail}[st]
    if k == "timing":
        if st == GOOD:
            return "המחיר לא קפץ ולא צנח לאחרונה. זו נקודה סבירה לקנות"
        if st == OK:
            return "אין סימן מיוחד לגבי התזמון"
        if t.startswith("מתוחה"):
            return "המניה עלתה מהר מאוד לאחרונה. אחרי קפיצה כזו היא לרוב יורדת קצת"
        return f"המניה צונחת עכשיו ({pc(m.get('ret_1m'))} בחודש). קשה לדעת איפה היא תעצור"
    if k == "earn_date":
        d = ctx.get("days_to_earnings")
        return {GOOD: "אין דוח רווחים בשבועות הקרובים",
                OK: f"החברה תפרסם דוח רווחים בעוד {d} ימים",
                BAD: f"בעוד {d} ימים החברה מפרסמת דוח רווחים, ואז המחיר יכול לקפוץ חזק לכל כיוון"}[st]
    if k == "earn_last":
        return {GOOD: "בדוח האחרון החברה הרוויחה יותר ממה שציפו", OK: "בדוח האחרון החברה הרוויחה בערך כמו שציפו",
                BAD: "בדוח האחרון החברה הרוויחה פחות ממה שציפו"}[st]
    if k == "value":
        if st == GOOD:
            return "המחיר זול יחסית לכמה שהחברה מרוויחה"
        if st == OK:
            return "המניה יקרה, אבל החברה גדלה מהר וזה מצדיק חלק מהמחיר" if t.startswith("יקרה") else "המחיר סביר ביחס לרווחים"
        if "מפסידה" in t:
            return "החברה עדיין מפסידה כסף, אז מי שקונה משלם בעיקר על תקווה לעתיד"
        return "המניה יקרה ביחס לכמה שהחברה מרוויחה. אם החברה תאכזב, המחיר עלול לרדת חזק"
    if k == "quality":
        pm = _f(f.get("profitMargins"))
        if st == GOOD:
            return "החברה מרוויחה יפה" + (f": מכל 100 ₪ של מכירות נשארים לה בערך {pm * 100:.0f} ₪ רווח" if pm and pm > 0 else "")
        return {OK: "החברה רווחית במידה סבירה", BAD: "החברה מרוויחה מעט מאוד או מפסידה כסף"}[st]
    if k == "balance":
        return {GOOD: "לחברה יש מספיק מזומן, והחובות שלה לא מסוכנים", OK: "מצב החובות של החברה סביר",
                BAD: "לחברה הרבה חובות, או שיוצא ממנה יותר כסף ממה שנכנס"}[st]
    if k == "growth":
        rg = _f(f.get("revenueGrowth"))
        tail = f" ({pc(rg)} בשנה)" if rg is not None else ""
        return {GOOD: "המכירות של החברה גדלות מהר" + tail, OK: "המכירות של החברה גדלות בקצב מתון" + tail,
                BAD: "המכירות או הרווחים של החברה יורדים" + tail}[st]
    if k == "analysts":
        return {GOOD: "רוב המומחים שעוקבים אחרי החברה ממליצים לקנות", OK: "המומחים חלוקים בדעתם",
                BAD: "המומחים לא ממליצים, או שהמחיר כבר גבוה ממה שהם צופים"}[st]
    if k == "news":
        return {GOOD: "רוב הכותרות מהשבועיים האחרונים חיוביות", OK: "בחדשות האחרונות אין כיוון ברור",
                BAD: "הרבה כותרות שליליות בשבועיים האחרונים"}[st]
    if k == "risk":
        dd = m.get("max_dd")
        return {GOOD: "מניה רגועה יחסית, בדרך כלל בלי קפיצות חזקות", OK: "המניה קופצת בצורה בינונית",
                BAD: "מניה שקופצת חזק" + (f": בשנה האחרונה היא ירדה בשלב מסוים {abs(dd) * 100:.0f}% מהשיא" if dd else "")}[st]
    if k == "market":
        return {GOOD: "הבורסה כולה בעלייה, וזה עוזר לרוב המניות", OK: "לבורסה כולה אין כרגע כיוון ברור",
                BAD: "הבורסה כולה בירידה. בתקופה כזו גם מניות טובות יורדות"}[st]
    return c["text"]


def consult(m: dict, f: dict, ctx: dict) -> dict:
    checks = [fn(m, f, ctx) for fn in CHECKS]
    for c in checks:
        c["name"] = SIMPLE_NAME.get(c["key"], c["title"])
        c["plain"] = plain(c, m, f, ctx)
    by = {c["key"]: c for c in checks}
    overall, w_all = _score(checks)
    fund, w_fund = _score(checks, "fund")
    trend = by["trend"]["status"]
    timing = by["timing"]["status"]
    earn_soon = by["earn_date"]["status"] == BAD
    market_bad = by["market"]["status"] == BAD
    risk_lvl = ctx.get("risk_level") or 3
    reds = [c for c in checks if c["status"] == BAD]
    greens = [c for c in checks if c["status"] == GOOD]
    possible = sum(c["weight"] for c in checks if c["weight"] > 0)
    coverage = w_all / possible if possible else 0

    fund = fund if fund is not None else 0.0
    overall = overall if overall is not None else 0.0
    p = m.get("price")

    # ---- decision tree -----------------------------------------------------------------
    waits = []  # concrete things to wait for
    if earn_soon:
        waits.append(f"לחכות לדוח הרווחים ב-{ctx.get('next_earnings')}. סביב הדוח המחיר יכול לקפוץ חזק לכל כיוון")
    if timing == BAD and by["timing"]["title"].startswith("מתוחה") and m.get("ma50"):
        waits.append(f"המניה עלתה מהר לאחרונה. עדיף לחכות שתרד קצת, לבערך {m['ma50']:.2f}, או לקנות בכמה פעמים קטנות")
    if timing == BAD and by["timing"]["title"].startswith("בנפילה"):
        waits.append("לחכות שהמחיר יפסיק לרדת" + (f" ויחזור מעל {m['ma20']:.2f}" if m.get("ma20") else ""))
    if trend != GOOD and m.get("ma200") and p and p < m["ma200"]:
        waits.append(f"לחכות שהמחיר יעלה מעל {m['ma200']:.2f}. זה יהיה סימן שהירידה הארוכה נגמרה")
    elif trend != GOOD and m.get("ma50") and p and p < m["ma50"]:
        waits.append(f"לחכות שהמחיר יעלה מעל {m['ma50']:.2f}. זה יהיה סימן שהמניה חזרה לעלות")
    if market_bad:
        waits.append("הבורסה כולה בירידה. אם קונים, עדיף בכמה פעמים קטנות ולא בבת אחת")

    if fund <= -0.35 or (trend == BAD and fund < 0.1) or overall <= -0.3:
        verdict = "avoid"
    elif fund >= 0.0 and trend == GOOD and overall >= 0.25 and not earn_soon and timing != BAD \
            and not (market_bad and risk_lvl >= 4):
        verdict = "buy"
    elif fund >= -0.1 and overall >= -0.05:
        verdict = "wait"
    else:
        verdict = "avoid"

    confidence = "גבוהה" if coverage > 0.8 and abs(overall) > 0.35 else "בינונית" if coverage > 0.55 else "נמוכה"

    # ---- headline reasons ---------------------------------------------------------------
    def top(lst, n):
        return sorted(lst, key=lambda c: -c["weight"])[:n]

    if verdict == "buy":
        summary = "חברה במצב טוב, והמחיר בעלייה. אין כרגע סיבה מיוחדת לחכות."
        reasons = [_r(c) for c in top(greens, 4)]
        cautions = [c["title"] for c in reds]
    elif verdict == "wait":
        if fund >= 0.2 and (earn_soon or timing == BAD or trend != GOOD or market_bad):
            summary = "חברה טובה, אבל זה לא הרגע הכי טוב לקנות."
        else:
            summary = "יש דברים טובים ויש דברים פחות טובים. עדיף לחכות שהתמונה תתבהר."
        reasons = [_r(c) for c in top(reds, 3)] + [_r(c) for c in top(greens, 2)]
        cautions = []
    else:
        summary = "יותר מדי סימני אזהרה כרגע. יש היום מניות במצב טוב יותר."
        reasons = [_r(c) for c in top(reds, 4)]
        cautions = [c["title"] for c in greens]

    ex = exit_plan(m, f, ctx.get("upside")) if verdict != "avoid" else {}
    return {"verdict": verdict, "summary": summary, "exit": ex, "reasons": reasons, "cautions": cautions,
            "waits": waits if verdict != "avoid" else [w for w in waits if "200" in w][:1],
            "score": round(overall * 100), "fund_score": round(fund * 100),
            "confidence": confidence, "checks": checks,
            "counts": {"good": len(greens), "bad": len(reds),
                       "ok": sum(1 for c in checks if c["status"] == OK)}}


def earnings_context(info: dict, edates: list[dict] | None, today: datetime | None = None) -> dict:
    """Days to next earnings + recent EPS surprises (fractions)."""
    today = today or datetime.now(timezone.utc)
    nxt = None
    for k in ("earningsTimestampStart", "earningsTimestamp"):
        ts = _f(info.get(k))
        if ts:
            d = datetime.fromtimestamp(ts, timezone.utc)
            if d.date() >= today.date():
                nxt = d
                break
    surprises = []
    for e in edates or []:
        if e.get("date") and datetime.fromisoformat(e["date"]).date() >= today.date() and e.get("surprise") is None:
            d = datetime.fromisoformat(e["date"])
            if nxt is None or d < nxt:
                nxt = d
        elif e.get("surprise") is not None:
            surprises.append(e["surprise"])
    out = {"surprises": surprises[:4], "last_surprise": surprises[0] if surprises else None}
    if nxt:
        out["days_to_earnings"] = (nxt.date() - today.date()).days
        out["next_earnings"] = nxt.strftime("%d/%m/%Y")
    return out
