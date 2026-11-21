import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import history  # noqa: E402
from verdict import exit_plan  # noqa: E402


def S(sym, v):
    return {"symbol": sym, "name": sym, "price": 10, "consult": {"verdict": v, "exit": {}}}


def test_changes_only_after_first_run_and_same_day_flip_back():
    h = {"events": [], "last": {}}
    assert history.update(h, [S("A", "wait")], "2026-01-01") == []
    ch = history.update(h, [S("A", "buy")], "2026-01-02")
    assert [(c["from"], c["to"]) for c in ch] == [("wait", "buy")]
    assert history.update(h, [S("A", "wait")], "2026-01-02") == []  # same-day flip back
    assert [e["v"] for e in h["events"]] == ["wait"]


def test_exit_plan_bounds():
    e = exit_plan({"price": 100, "dvol20": 0.02, "ma50": 95}, {}, 0.3)
    assert 85 <= e["stop"] <= 95 and e["target"] > 100 and e["rr"] > 1
