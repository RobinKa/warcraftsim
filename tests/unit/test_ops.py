"""scripts/ops.py: the routine checks' pure parts (rows, scores, tiers, the race-balanced mean, names, the budget's
commands)."""

import importlib.util
import json
import math
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("ops", ROOT / "scripts" / "ops.py")
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)


def _row(t, outcome=1.0, race="human", opponent="script:ai-insane (real)", game_time=700.0, **kw):
    return {"time": t, "outcome": outcome, "race": race, "opponent": opponent, "game_time": game_time,
            "prod": kw.pop("prod", {"built": {}, "research": {}, "workers": 8, "heroes": 1, "expansions": 0}), **kw}


def test_rows_skip_nul_blank_and_broken_lines(tmp_path):
    p = tmp_path / "episodes.jsonl"
    p.write_bytes(json.dumps(_row(1.0)).encode() + b"\n" + b"\x00" * 300 + b"\n\n{broken\n"
                  + b"\x00\x00" + json.dumps(_row(2.0)).encode() + b"\n")
    assert [r["time"] for r in ops.rows(p)] == [1.0, 2.0]
    assert list(ops.rows(tmp_path / "missing.jsonl")) == []


def test_rows_since_seeks_but_keeps_every_later_row(tmp_path):
    p = tmp_path / "episodes.jsonl"
    with p.open("w") as fh:
        for i in range(6000):  # (~1 MB: several binary search steps)
            fh.write(json.dumps(_row(1000.0 + i, notes="x" * 100)) + "\n")
            if i % 997 == 0:
                fh.write("\x00" * 200 + "\n")
    since = 1000.0 + 4321
    got = [r["time"] for r in ops.rows(p, since) if r["time"] >= since]
    assert got == [1000.0 + i for i in range(4321, 6000)]
    first = next(ops.rows(p, since))["time"]
    assert first < since  # (from a little before: the rows are only roughly in time order)


def test_tail_rows(tmp_path):
    p = tmp_path / "train.jsonl"
    p.write_text("".join(json.dumps({"agent_steps": i, "SPS": 10.0 * i}) + "\n" for i in range(50)) + '{"half')
    assert [r["agent_steps"] for r in ops.tail_rows(p, 3)] == [47, 48, 49]
    assert len(ops.tail_rows(p, 100)) == 50


def test_score_and_tier():
    assert [ops.score(o) for o in (1.0, 0.0, -1.0)] == [1.0, 0.5, 0.0]
    assert ops.tier({"built": {"hkee": 1}}, "human") == 2
    assert ops.tier({"built": {"hkee": 1, "hcas": 1}}, "human") == 3
    assert ops.tier({"built": {"unp2": 1}}, "undead") == 3
    assert ops.tier({"built": {"ostr": 0}}, "orc") == 1
    assert ops.tier({"built": {"hkee": 1}}, "orc") == 1  # (another race's hall)
    assert ops.tier({}, "nightelf") == 1


def test_development_means():
    rs = [_row(1, 1.0, "human", prod={"built": {"hkee": 1}, "research": {"Rhme": 2, "Rhar": 1}, "workers": 10,
                                      "heroes": 2, "expansions": 1}),
          _row(2, -1.0, "orc", prod={"built": {}, "research": {}, "workers": 6, "heroes": 0, "expansions": 0})]
    d = ops.development(rs)
    assert d == {"n": 2, "heroes": 1.0, "tier": 1.5, "research": 1.5, "workers": 8.0, "expanded": 0.5, "won": 0.5}
    assert ops.development([]) is None


def test_balanced_matches_burst_267_current():
    rs = ([_row(0, 1.0, "human")] * 19 + [_row(0, 1.0, "nightelf")] * 23 + [_row(0, -1.0, "nightelf")] * 3
          + [_row(0, 1.0, "orc")] * 18 + [_row(0, 1.0, "undead")] * 20 + [_row(0, 0.0, "undead")]
          + [_row(0, -1.0, "undead")] * 3)
    m, ci, per = ops.balanced(rs)
    assert per["nightelf"] == (23 / 26, 26) and per["undead"] == (20.5 / 24, 24)
    p_n, p_u = 23 / 26, 20.5 / 24
    assert m == pytest.approx((1 + p_n + 1 + p_u) / 4)
    assert ci == pytest.approx(1.96 * math.sqrt(p_n * (1 - p_n) / 26 + p_u * (1 - p_u) / 24) / 4)
    assert round(m, 3) == 0.935 and round(ci, 3) == 0.047
    assert ops.races_str(per) == "hu 1.00(19) ni 0.88(26) or 1.00(18) un 0.85(24)"
    m, ci, per = ops.balanced([])
    assert math.isnan(m) and per == {}


def test_training_filters_and_quarter_bins():
    since = 10_000.0
    rs = [_row(since + 60, 1.0), _row(since + 120, -1.0), _row(since + 16 * 60, 0.0),
          _row(since + 70, 1.0, eval=True), _row(since + 80, 1.0, z=[1, 2]),
          _row(since + 90, 1.0, opponent="script:ai-insane (real, agent -25%)"), _row(since + 95, 1.0, opponent="past:3")]
    ai = [r for r in rs if ops.training_ai(r)]
    assert len(ai) == 3
    assert ops.quarter_bins(ai, since) == (15, [(0, 0.5, 2), (15, 0.5, 1)])
    dev = [r for r in rs if ops.training_dev(r)]
    assert len(dev) == 5 and not any(r.get("eval") or r.get("z") for r in dev)
    assert not ops.training_dev(_row(since, game_time=599.0))
    long = [_row(since + h * 3600, 1.0) for h in range(10)]
    width, bins = ops.quarter_bins(long, since)
    assert width == 45 and len(bins) <= 16  # (9 h: 15-minute bins widened to at most 16)


def test_burst_policy_keys():
    cur = "runs/fgself-267/averaged/ema-0000000026967703.pt"
    evals = ["runs/fullgame/soups/soup-2026-11-26.pt", "runs/fgself-267/averaged/swa-17-27.pt",
             "runs/fgself-252/averaged/swa-17-27.pt"]
    f = lambda k: ops.burst_policy(k, cur, evals)  # noqa: E731
    assert f("current") == (cur, "ai") and f("current @-25%") == (cur, "ai")
    assert f("runs/fgself-252/averaged/swa-17-27.pt") == (evals[2], "ai")
    assert f("/home/x/warcraftsim/runs/fgself-267/averaged/swa-17-27.pt @-50%") == (evals[1], "ai")
    assert f("vs fgself-252/swa-17-27.pt") == (evals[2], "h2h")
    assert f("vs fgself-267/swa-17-27.pt") == (evals[1], "h2h")
    assert f("vs fullgame/soup-2026-11-26.pt") == (evals[0], "h2h")
    assert f("vs fgself-999/swa-1-2.pt") is None and f("runs/other.pt") is None


def test_burst_table_splits_ai_rungs_and_h2h():
    cur, ev = "runs/fgself-1/averaged/ema-0000000000000005.pt", ["runs/fgself-2/averaged/swa-1-2.pt"]
    rs = [_row(1, eval=True, eval_key="current"), _row(2, eval=True, eval_key="current @-25%", handicap=25,
                                                        opponent="script:ai-insane (real, agent -25%)"),
          _row(3, eval=True, eval_key="vs fgself-2/swa-1-2.pt", opponent="eval-vs:runs/fgself-2/averaged/swa-1-2.pt"),
          _row(4, eval=True, eval_key=ev[0]), _row(5, eval=True, eval_key="mystery")]
    t, unmapped = ops.burst_table(rs, cur, ev)
    assert [len(t[cur]["ai"]), len(t[cur]["rungs"][25]), len(t[cur]["h2h"])] == [1, 1, 0]
    assert [len(t[ev[0]]["ai"]), len(t[ev[0]]["h2h"])] == [1, 1]
    assert unmapped == ["mystery"]


def test_names():
    assert ops.runs_rel("/home/x/warcraftsim/runs/fgself-252/averaged/swa-17-27.pt") == \
        "runs/fgself-252/averaged/swa-17-27.pt"
    assert ops.vs_tag("runs/fullgame/soups/soup-2026-11-26.pt") == "fullgame/soup-2026-11-26.pt"
    assert ops.vs_tag("/abs/runs/fgself-252/averaged/swa-17-27.pt") == "fgself-252/swa-17-27.pt"
    assert ops.rating_name("runs/fgself-252/averaged/swa-17-27.pt") == "fgself-252 avg swa-17-27"
    assert ops.rating_name("runs/fullgame/soups/soup-2026-11-26.pt") == "soup avg soup-2026-11-26"
    assert ops.rating_name("runs/fgself-267/averaged/ema-0000000026967703.pt") == "fgself-267 avg ema-0000000026967703"
    assert ops.short_name("runs/fgself-267/averaged/ema-0000000026967703.pt") == "fgself-267 avg ema-26967703"
    assert ops.short_name("runs/fgself-212/checkpoints/0000000118808740.pt") == "fgself-212/118808740"
    assert ops.short_name("runs/fullgame/soups/soup-2026-11-26.pt") == "soup avg soup-2026-11-26"


def test_parse_steps_and_since():
    assert ops.parse_steps("54e6") == ops.parse_steps("54M") == ops.parse_steps("54000000") == 54_000_000
    assert ops.parse_steps("26.5m") == 26_500_000
    now = time.mktime((2026, 11, 27, 10, 30, 0, 0, 0, -1))
    assert ops.parse_since("09:15", now) == now - 75 * 60
    assert ops.parse_since("21:27", now) == time.mktime((2026, 11, 26, 21, 27, 0, 0, 0, -1))  # (still ahead: yesterday)
    assert ops.parse_since("1795728420.5", now) == 1795728420.5
    assert ops.parse_since("2026-11-26 21:27", now) == time.mktime((2026, 11, 26, 21, 27, 0, 0, 0, -1))
    assert ops.default_since({"created": 100.0, "resumes": [{"time": 300.0}, {"time": 200.0}]}) == 300.0
    assert ops.default_since({"created": 100.0}) == 100.0


def test_log_refused_counts_this_launch(tmp_path):
    log = tmp_path / "fgself-1.log"
    log.write_text("map frame: a\ngateway: x refused: its code (a) is not the learner's (b)\n" * 2
                   + "map frame: b\nother\ngateway: y refused: its code (c) is not the learner's (d)\n")
    total, this, last = ops.log_refused(log)
    assert (total, this) == (3, 1) and "(c)" in last


def _budget_root(tmp_path, steps=(26_000_000, 54_100_000, 54_400_000)):
    rd = tmp_path / "runs" / "fgself-9"
    (rd / "checkpoints").mkdir(parents=True)
    (rd / "run.json").write_text("{}")
    for s in steps:
        (rd / "checkpoints" / f"{s:016d}.pt").write_bytes(b"x")
    (rd / "checkpoints" / f"{55_000_000:016d}.pt.tmp").write_bytes(b"x")
    (tmp_path / "runs" / "fullgame" / "soups").mkdir(parents=True)
    (tmp_path / "runs" / "fullgame" / "soups" / "soup-a.pt").write_bytes(b"x")
    return rd


def test_budget_plan_swa_and_ema():
    p = ops.budget_plan("fgself-9", "burst-9", 54_000_000, 27_000_000, 54_100_000, ["runs/fullgame/soups/soup-a.pt"])
    assert p["swa"] == "runs/fgself-9/averaged/swa-27-54.pt" and p["current"] == p["swa"]
    labels = [x[0] for x in p["steps"]]
    assert labels == ["stop the learner", "stop the box's actors", "average", "prepare the burst"]
    argv = {x[0]: x[1] for x in p["steps"]}
    assert argv["stop the learner"] == [".venv/bin/python", "scripts/selfplay_restart.py", "fgself-9", "--stop"]
    assert argv["stop the box's actors"][-2:] == ["tora-ms-a2@tora-ms-a2",
                                                  "systemctl --user stop warcraftsim-remoteactor.service"]
    assert argv["average"] == ["nice", "-n", "5", ".venv/bin/python", "scripts/average_checkpoints.py",
                               "runs/fgself-9/averaged/swa-27-54.pt", "--run", "fgself-9", "--from", "27000000",
                               "--to", "54100000"]
    assert argv["prepare the burst"] == [".venv/bin/python", "scripts/burst_prep.py", "burst-9", "fgself-9",
                                         "runs/fgself-9/averaged/swa-27-54.pt", "runs/fullgame/soups/soup-a.pt"]
    assert p["worker"] == ["setsid", "nohup", ".venv/bin/python", "-m", "warcraftsim.fullgame.evalworker", "--run",
                           "burst-9", "--games", "64", "--processes", "16", "--host", "tora-burst", "--hours", "0.75"]
    assert p["log"] == "runs/burst-9.log"
    e = ops.budget_plan("fgself-9", "burst-9", 54_000_000, 27_000_000, 54_100_000, ["runs/fullgame/soups/soup-a.pt"],
                        current="ema")
    assert e["current"] == "runs/fgself-9/averaged/ema-0000000054100000.pt"
    assert e["evals"] == ["runs/fgself-9/averaged/swa-27-54.pt", "runs/fullgame/soups/soup-a.pt"]
    assert e["steps"][3][1][4:] == [e["current"], *e["evals"]]


def test_budget_dry_run_prints_and_touches_nothing(tmp_path, capsys):
    rd = _budget_root(tmp_path)
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    rc = ops.main(["--root", str(tmp_path), "budget", "fgself-9", "--at", "54e6", "--burst", "burst-9",
                   "--avg-from", "27e6", "--eval", "runs/fullgame/soups/soup-a.pt", "--current", "ema", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "found 0000000054100000.pt" in out  # (the first at or past the budget, not the .tmp)
    assert ("burst_prep.py burst-9 fgself-9 runs/fgself-9/averaged/ema-0000000054100000.pt "
            "runs/fgself-9/averaged/swa-27-54.pt runs/fullgame/soups/soup-a.pt") in out
    assert "--from 27000000 --to 54100000" in out
    assert "> runs/burst-9.log 2>&1 < /dev/null &" in out
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before
    assert ops.checkpoints(rd) == [26_000_000, 54_100_000, 54_400_000]


def test_budget_dry_run_reports_problems(tmp_path, capsys):
    _budget_root(tmp_path)
    (tmp_path / "runs" / "burst-9").mkdir()
    rc = ops.main(["--root", str(tmp_path), "budget", "fgself-9", "--at", "54e6", "--burst", "burst-9",
                   "--avg-from", "60e6", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "PROBLEM: runs/burst-9 exists" in out and "PROBLEM: no evals" in out and "--avg-from must be below" in out


def test_budget_eval_repeated_keeps_all(tmp_path, capsys):
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("ops_cli", Path(__file__).resolve().parents[2] / "scripts" / "ops.py")
    ops = importlib.util.module_from_spec(spec)
    sys.modules["ops_cli"] = ops
    spec.loader.exec_module(ops)
    (tmp_path / "runs" / "r" / "checkpoints").mkdir(parents=True)
    try:
        ops.main(["--root", str(tmp_path), "budget", "r", "--at", "1e6", "--burst", "b", "--avg-from", "0",
                  "--eval", "x.pt", "--eval", "y.pt", "z.pt", "--dry-run"])
    except SystemExit:
        pass
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if "burst_prep.py" in ln)
    assert "x.pt" in line and "y.pt" in line and "z.pt" in line
