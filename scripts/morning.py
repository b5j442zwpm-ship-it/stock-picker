"""Write the morning summary (Markdown) from docs/data.json. Used by the morning workflow,
which posts it as a GitHub issue so it arrives as a notification on the phone / by email."""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = json.loads((ROOT / "docs" / "data.json").read_text(encoding="utf-8"))
VH = {"buy": "🟢 לקנות עכשיו", "wait": "🟡 לחכות", "avoid": "🔴 לא לקנות"}
MK = {"positive": "חיובי 🟢", "neutral": "מעורב 🟡", "negative": "זהיר 🔴"}
site = os.environ.get("SITE_URL", "")
by = {s["symbol"]: s for s in D["stocks"]}


def link(sym, name):
    return f"[{name} ({sym})]({site}#{sym})" if site else f"{name} ({sym})"


def pct(x):
    return "—" if x is None else f"{x * 100:+.1f}%"


L = []
gen = datetime.fromisoformat(D["generated_at"])
age_h = (datetime.now(timezone.utc) - gen).total_seconds() / 3600
if age_h > 36:
    L.append(f"> ⚠️ הנתונים לא עודכנו כבר {age_h / 24:.0f} ימים. ייתכן שהעדכון האוטומטי נכשל (לשונית Actions).\n")

M = D["market"]
L.append(f"**מצב השוק: {MK.get(M['state'], M['state'])}**")
for n in M["notes"][:4]:
    L.append(f"- {n}")
L.append("")

alerts = [w for w in D.get("watchlist", []) if w.get("alerts")]
if alerts:
    L.append("## 🔔 התראות מרשימת המעקב")
    for w in alerts:
        for a in w["alerts"]:
            L.append(f"- **{link(w['symbol'], w.get('name') or w['symbol'])}**: {a}. מחיר {w.get('price')}")
    L.append("")

ch = D.get("changes", [])
order = {"buy": 0, "wait": 1, "avoid": 2}
if ch:
    L.append(f"## מה השתנה מאתמול ({len(ch)})")
    for c in sorted(ch, key=lambda c: order[c["to"]]):
        s = by.get(c["symbol"], {})
        ex = (s.get("consult") or {}).get("exit") or {}
        extra = f" · יעד {ex['target']} / סטופ {ex['stop']}" if c["to"] == "buy" and ex else ""
        L.append(f"- {link(c['symbol'], c['name'])}: {VH[c['from']]} ← **{VH[c['to']]}**{extra}. {c.get('summary') or ''}")
    L.append("")
else:
    L.append("אין שינויים בהמלצות מאתמול.\n")

buys = sorted([s for s in D["stocks"] if s["consult"]["verdict"] == "buy"], key=lambda s: -s["consult"]["score"])[:10]
if buys:
    L.append(f"## במצב \"לקנות עכשיו\" ({sum(1 for s in D['stocks'] if s['consult']['verdict'] == 'buy')} מניות, 10 הבולטות)")
    L.append("| מניה | מחיר | יעד | סטופ | סיכון |")
    L.append("|---|---|---|---|---|")
    for s in buys:
        ex = s["consult"].get("exit") or {}
        L.append(f"| {link(s['symbol'], s.get('name_he') or s['name'])} | {s['price']} | {ex.get('target', '—')} ({pct(ex.get('target_pct'))}) "
                 f"| {ex.get('stop', '—')} ({pct(ex.get('stop_pct'))}) | {s['risk']['level']}/5 |")
    L.append("")

t = (D.get("track") or {}).get("live", {}).get("buy") or {}
if t.get("n_r1m"):
    L.append(f"**מעקב הצלחה:** מתוך {t['n_r1m']} המלצות קנייה שעבר עליהן חודש, {t['win1m'] * 100:.0f}% עלו. "
             f"תשואה ממוצעת אחרי חודש {pct(t['r1m'])}.")
L.append("\n---\n_כלי עזר בלבד, לא ייעוץ השקעות._" + (f" [ללוח המלא]({site})" if site else ""))
if os.environ.get("OWNER"):
    L.append(f"\n@{os.environ['OWNER']}")

(ROOT / "summary.md").write_text("\n".join(L), encoding="utf-8")
flag = "🔔 " if alerts else ""
n_buy_new = sum(1 for c in ch if c["to"] == "buy")
title = f"{flag}סיכום בוקר {datetime.now(timezone.utc):%d/%m}: {len(ch)} שינויים" + (f", {n_buy_new} חדשות ב\"לקנות\"" if n_buy_new else "")
print(title)
sys.exit(0)
