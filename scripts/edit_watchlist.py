"""Add/remove $SYMBOL in watchlist.txt (used by the watchlist workflow)."""
import os
import re
import sys
from pathlib import Path

F = Path(__file__).resolve().parent.parent / "watchlist.txt"
sym = os.environ.get("SYMBOL", "").strip().upper()
action = os.environ.get("ACTION", "add")
target = os.environ.get("TARGET", "").strip().replace(",", "")
note = re.sub(r"[|\n\r#]", " ", os.environ.get("NOTE", "")).strip()
if not re.fullmatch(r"[A-Z0-9^][A-Z0-9.\-=^]{0,14}", sym):
    sys.exit(f"Invalid symbol: {sym!r}")
if target and not re.fullmatch(r"\d+(\.\d+)?", target):
    sys.exit(f"Target must be a number, got {target!r}")

lines = F.read_text(encoding="utf-8").splitlines() if F.exists() else []
keep = [ln for ln in lines if ln.split("#", 1)[0].split("|", 1)[0].strip().upper() != sym]
if action == "add":
    keep.append(" | ".join([sym, target, note]).rstrip(" |"))
F.write_text("\n".join(keep) + "\n", encoding="utf-8")
print(f"{action} {sym}")
