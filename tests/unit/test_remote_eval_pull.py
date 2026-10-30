"""scripts/remote_eval.py's pull counts only its own host's rows: a second worker taking over mid-run (the games box from
the laptop, 2026-10-30) had its rows held back until their count passed the first's, and then lost that many."""

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("remote_eval", ROOT / "scripts" / "remote_eval.py")
remote_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote_eval)


def _row(host: str, i: int) -> str:
    return json.dumps({"host": host, "time": float(i), "i": i}) + "\n"


def _pull(mine: Path, theirs: list[str], name: str = "box", clock: float = 0.0) -> None:
    def run(cmd, timeout=120, binary=False):  # (the command's `tail -n +K | head -n N` over the host's file)
        k = int(cmd.split("tail -n +")[1].split()[0])
        n = int(cmd.split("head -n ")[1].split()[0])
        return subprocess.CompletedProcess(cmd, 0, "".join(theirs[k - 1:k - 1 + n]).encode(), b"")
    m = SimpleNamespace(a=SimpleNamespace(name=name), h=SimpleNamespace(run=run), remote=lambda f: f)
    remote_eval.Manager._pull(m, {"lines": len(theirs), "clock": clock}, mine)


def test_second_host_rows_are_taken_from_its_first(tmp_path):
    mine = tmp_path / "episodes-remote.jsonl"
    mine.write_text("".join(_row("laptop", i) for i in range(5)))
    box = [_row("box", i) for i in range(3)]
    _pull(mine, box)
    rows = [json.loads(x) for x in mine.read_text().splitlines()]
    assert [r["host"] for r in rows] == ["laptop"] * 5 + ["box"] * 3
    box.append(_row("box", 3))
    _pull(mine, box)
    assert [json.loads(x)["i"] for x in mine.read_text().splitlines()][5:] == [0, 1, 2, 3]  # (each once)
    _pull(mine, box)
    assert len(mine.read_text().splitlines()) == 9


def test_damaged_row_is_kept_once_and_counted(tmp_path):
    mine = tmp_path / "episodes-remote.jsonl"
    box = [_row("box", 0), '{"host": "bo\n', _row("box", 2)]
    _pull(mine, box)
    assert len(mine.read_text().splitlines()) == 3
    _pull(mine, box)
    assert len(mine.read_text().splitlines()) == 3
