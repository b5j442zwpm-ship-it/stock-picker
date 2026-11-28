"""Sanity checks for the buy / wait / avoid logic. Run: python -m pytest tests"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from verdict import consult  # noqa: E402

UP = {"price": 110, "ma20": 108, "ma50": 104, "ma200": 92, "ret_1m": 0.04, "ret_3m": 0.12, "rsi14": 58,
      "vol": 0.24, "max_dd": -0.15}
DOWN = {"price": 70, "ma20": 74, "ma50": 80, "ma200": 95, "ret_1m": -0.08, "ret_3m": -0.2, "rsi14": 38,
        "vol": 0.45, "max_dd": -0.4}
GOODF = {"sector": "Technology", "forwardPE": 24, "trailingPegRatio": 1.3, "profitMargins": 0.25, "returnOnEquity": 0.3,
         "debtToEquity": 40, "currentRatio": 1.8, "freeCashflow": 5e9, "totalCash": 2e10, "totalDebt": 1e10,
         "revenueGrowth": 0.15, "earningsGrowth": 0.2, "recommendationMean": 1.9, "numberOfAnalystOpinions": 30}
BADF = {"sector": "Technology", "trailingPE": -5, "profitMargins": -0.2, "returnOnEquity": -0.3, "debtToEquity": 250,
        "currentRatio": 0.7, "freeCashflow": -1e9, "revenueGrowth": -0.08, "earningsGrowth": -0.5,
        "recommendationMean": 3.2, "numberOfAnalystOpinions": 10}
CTX = {"risk_level": 2, "market": {"state": "positive", "notes": []}, "upside": 0.15, "days_to_earnings": 40,
       "next_earnings": "01/01/2027", "last_surprise": 0.05, "surprises": [0.05, 0.03]}


def test_buy():
    assert consult(UP, GOODF, CTX)["verdict"] == "buy"


def test_wait_before_earnings():
    c = consult(UP, GOODF, {**CTX, "days_to_earnings": 3})
    assert c["verdict"] == "wait" and any("הדוח" in w for w in c["waits"])


def test_wait_when_overheated():
    hot = {**UP, "price": 130, "rsi14": 81}
    assert consult(hot, GOODF, CTX)["verdict"] == "wait"


def test_wait_good_business_in_downtrend():
    assert consult(DOWN, GOODF, CTX)["verdict"] == "wait"


def test_avoid():
    assert consult(DOWN, BADF, {**CTX, "risk_level": 5, "upside": -0.1})["verdict"] == "avoid"


# pullback to the 50-day line inside an uptrend, with buying volume and a turn up
DIP = {**UP, "price": 105, "ma20": 106, "ma50": 104, "rsi14": 50, "rsi_5ago": 40, "ret_1m": -0.03, "ret_5d": 0.02,
       "from_hi20": -0.07, "vol_ratio": 1.7, "updown_vol": 1.6, "rs_3m": 0.08, "crossed_ma20": False}


def test_pullback_is_a_good_entry_with_signals():
    c = consult(DIP, GOODF, CTX)
    by = {k["key"]: k for k in c["checks"]}
    assert by["entry"]["status"] == "good"
    assert by["catalyst"]["status"] == "good" and len(c["signals"]) >= 3
    assert c["verdict"] == "buy" and "מחיר כניסה טוב" in c["summary"]
    assert c["opp"] > consult(UP, GOODF, CTX)["opp"]


def test_selling_pressure_blocks_buy():
    dist = {**UP, "ret_5d": -0.03, "vol_ratio": 2.0, "updown_vol": 0.5, "rs_3m": -0.15}
    c = consult(dist, GOODF, {**CTX, "sector_rank": 0.1})
    assert c["verdict"] == "wait" and any("מוכרים" in w for w in c["waits"])


def test_chasing_far_above_support_waits():
    far = {**UP, "price": 121, "ma50": 104, "rsi14": 66}
    assert consult(far, GOODF, CTX)["verdict"] == "wait"
