"""Append $SYMBOL (validated against Yahoo Finance) to tickers.txt. Used by the add-stock workflow."""
import os
import re
import sys
from pathlib import Path

TICKERS = Path(__file__).resolve().parent.parent / "tickers.txt"

sym = os.environ.get("SYMBOL", "").strip().upper()
name = re.sub(r"[|\n\r#]", " ", os.environ.get("NAME_HE", "")).strip()
if not re.fullmatch(r"[A-Z0-9^][A-Z0-9.\-=^]{0,14}", sym):
    sys.exit(f"Invalid symbol: {sym!r}")

existing = {line.split("#", 1)[0].split("|", 1)[0].strip().upper()
            for line in TICKERS.read_text(encoding="utf-8").splitlines()}
if sym in existing:
    print(f"{sym} is already in the list")
    sys.exit(0)

import yfinance as yf  # noqa: E402

hist = yf.Ticker(sym).history(period="1y")
if hist is None or len(hist) < 130:
    sys.exit(f"Yahoo Finance has no (or too little) price history for {sym}. "
             "Check the symbol; Tel Aviv stocks end with .TA")

with TICKERS.open("a", encoding="utf-8") as fh:
    fh.write(f"\n{sym}" + (f" | {name}" if name else ""))
print(f"Added {sym}")
