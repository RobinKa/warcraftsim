"""The verdict scripts (scripts/head_to_head.py, scripts/scorecard.py) on hand-made episode rows: the rows they count
(evaluation rows of both machines' files, --since), ties as half, and the races weighed alike over the races both
policies played."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REF = "runs/start/checkpoints/0000000000000100.pt"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(t: float, key: str, race: str, outcome: int, opponent: str = "script:ai-insane (real)") -> dict:
    return {"time": t, "eval": True, "eval_key": key, "opponent": opponent, "race": race, "outcome": outcome,
            "game_time": 900.0, "prod": {"trained": {"hfoo": 12}}, "opp_prod": {}}


def _write(tmp_path: Path, main: list[dict], remote: list[dict]) -> Path:
    run = tmp_path / "runs" / "r"
    run.mkdir(parents=True)
    (run / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in main))
    (run / "episodes-remote.jsonl").write_text("".join(json.dumps(r) + "\n" for r in remote))
    return tmp_path / "runs"


def test_head_to_head_races_alike_only_races_both_played(tmp_path, capsys):
    """The start played no orc in the window: orc is left out of the races-alike difference (it had counted the
    current policy's orc score, 1.0, as a difference of +1 against the start's empty 0)."""
    rows = ([_row(10 + i, "current", "human", 1 if i % 2 else -1) for i in range(4)]
            + [_row(20 + i, REF, "human", 1 if i % 2 else -1) for i in range(4)]
            + [_row(30 + i, "current", "orc", 1) for i in range(4)])
    runs = _write(tmp_path, rows[:6], rows[6:])
    assert _script("head_to_head").main(["r", "--runs", str(runs), "--vs", "start/0000000000000100"]) == 0
    out = capsys.readouterr().out
    line = next(x for x in out.splitlines() if "races weighed alike" in x)
    assert "= +0.000" in line and "left out" in line and "orc" in line


def test_head_to_head_counts_both_files_ties_half_and_since(tmp_path, capsys):
    rows = [_row(1, "vs start/0000000000000100.pt", "human", 1, "eval-vs:x"),  # (before --since: out)
            _row(5, "vs start/0000000000000100.pt", "human", 0, "eval-vs:x"),
            _row(6, "vs start/0000000000000100.pt", "orc", 1, "eval-vs:x"),
            {**_row(7, "vs start/0000000000000100.pt", "orc", 1, "eval-vs:x"), "eval": False}]  # (training: out)
    runs = _write(tmp_path, rows[:2], rows[2:])
    _script("head_to_head").main(["r", "--runs", str(runs), "--since", "2"])
    line = next(x for x in capsys.readouterr().out.splitlines() if x.startswith("vs start"))
    assert line.split()[2:5] == ["2", "0.750", "(1/1/0)"]


def test_scorecard_counts_both_files(tmp_path, capsys):
    rows = ([_row(10 + i, "current", "human", 1) for i in range(3)] + [_row(20 + i, REF, "human", -1) for i in range(3)])
    runs = _write(tmp_path, rows[:3], rows[3:])
    assert _script("scorecard").main(["r", "--runs", str(runs), "--vs", "start/0000000000000100", "--last", "0"]) == 0
    out = capsys.readouterr().out
    assert "current (3 games)" in out and "(3)" in out
    win = next(x for x in out.splitlines() if x.startswith("win "))
    assert win.split()[1:4] == ["1.00", "0.00", "1.00*"]
