"""scripts/host_stats.py's rows (what the remote command prints -> a row; WSL has no sensors; a host that does not answer)
and the dashboard's machines: the games split by machine, the readings shown."""

import importlib.util
import json
import subprocess
import time
from pathlib import Path

from warcraftsim.dashboard.server import _JsonlCache, _machine_split, _machines

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("host_stats", ROOT / "scripts" / "host_stats.py")
host_stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host_stats)

BOX = """stat1 cpu  1000 0 500 8000 100 0 0 0 0 0
stat2 cpu  1100 0 550 8050 100 0 0 0 0 0
MemTotal:       29960000 kB
MemAvailable:    7340000 kB
load 9.05 9.41 12.10
uptime 2825.01
mhz 4750.3 4806.0
games 42
temp Tctl:         +94.4°C
profile balanced
"""


def test_row_from_the_remote_output():
    r = host_stats.parse_stats(BOX, 100.0, "box")
    assert r["up"] and r["host"] == "box" and r["time"] == 100.0
    assert r["cpu_busy"] == 75.0  # (150 of 200 ticks not idle)
    assert abs(r["mem_total"] - 28.57) < 0.01 and abs(r["mem_avail"] - 7.0) < 0.01
    assert r["load"] == [9.05, 9.41, 12.1] and r["uptime"] == 2825.01
    assert (r["mhz_mean"], r["mhz_max"]) == (4750, 4806) and r["games"] == 42
    assert r["temp"] == 94.4 and r["profile"] == "balanced"


def test_wsl_has_no_sensors_nor_profile():
    wsl = "\n".join(l for l in BOX.splitlines() if not l.startswith(("temp", "profile"))) + "\nprofile \n"
    r = host_stats.parse_stats(wsl, 1.0, "wsl")
    assert r["temp"] is None and r["profile"] is None and r["cpu_busy"] == 75.0
    r = host_stats.parse_stats("stat1 cpu 1 2 3 4\nstat2 cpu 1 2 3 4\n", 1.0, "bare")  # (nothing else readable)
    assert r["cpu_busy"] is None and r["mem_total"] is None and r["load"] is None and r["games"] is None


def test_unreachable_host_is_a_row_with_up_false(monkeypatch):
    def down(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 255, "", "ssh: connect to host x port 22: No route to host\n")
    monkeypatch.setattr(host_stats.subprocess, "run", down)
    r = host_stats.look("u@x", "x")
    assert r["up"] is False and r["host"] == "x" and "No route" in r["error"]

    def hang(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])
    monkeypatch.setattr(host_stats.subprocess, "run", hang)
    assert host_stats.look("u@x", "x")["error"] == "timeout"


def test_the_remote_command_runs_here():
    """The same command through sh on this machine: the live run's games are counted (an int), whatever number."""
    out = subprocess.run(["sh", "-c", host_stats.REMOTE], capture_output=True, text=True, timeout=30).stdout
    r = host_stats.parse_stats(out, 1.0, "here")
    assert r["cpu_busy"] is not None and r["mem_total"] > 0 and isinstance(r["games"], int) and len(r["load"]) == 3


def test_trim_drops_old_rows(tmp_path):
    f = tmp_path / "box.jsonl"
    now = time.time()
    f.write_text("".join(json.dumps({"time": t}) + "\n" for t in (now - 9 * 86400, now - 86400, now)) + "not json\n")
    host_stats.trim(f, 7)
    assert [json.loads(l)["time"] for l in f.read_text().splitlines()] == [now - 86400, now]


def test_split_by_machine():
    now = 10000.0
    eps = [{"time": now - 3000 + i, "length": 100, "host": "box"} for i in range(10)]  # (training, a while ago)
    eps += [{"time": now - 60 + i, "length": 300, "host": "box"} for i in range(4)]
    eps += [{"time": now - 20 + i, "length": 500, "eval": True, "host": "wsl"} for i in range(3)]
    eps += [{"time": now - 10, "length": 200}, {"time": now - 5, "length": 200, "eval": True}]  # (this machine's)
    by = {m["machine"]: m for m in _machine_split(eps, now)}
    assert len(by) == 3 and {"box", "wsl"} < set(by)
    box = by["box"]
    assert (box["train"], box["eval"], box["train_recent"], box["eval_recent"]) == (14, 0, 4, 0)
    assert abs(box["steps_per_s_recent"] - 1200 / 1800) < 1e-9
    assert abs(box["steps_per_s"] - 2200 / (now - 5 - (now - 3000))) < 1e-9  # (to the run's newest game)
    assert (by["wsl"]["train"], by["wsl"]["eval"], by["wsl"]["eval_recent"]) == (0, 3, 3)
    assert by["wsl"]["steps_per_s"] is None  # (no training games)
    here = next(m for k, m in by.items() if k not in ("box", "wsl"))
    assert (here["train"], here["eval"]) == (1, 1) and _machine_split([], now) == []


def test_machines_shown_with_a_recent_row_only(tmp_path):
    d = tmp_path / "hosts"
    assert _machines(d, _JsonlCache(), 5000.0) == []  # (no directory)
    d.mkdir()
    up = {"up": True, "cpu_busy": 50.0, "mem_total": 30.0, "mem_avail": 10.0, "temp": None}
    (d / "box.jsonl").write_text("".join(json.dumps({**up, "time": 4900.0 + i}) + "\n" for i in range(5))
                                 + json.dumps({"time": 4990.0, "up": False, "error": "timeout"}) + "\n")
    (d / "old.jsonl").write_text(json.dumps({**up, "time": 10.0}) + "\n")
    m = _machines(d, _JsonlCache(), 5000.0)
    assert [x["name"] for x in m] == ["box"]
    assert m[0]["last"]["up"] is False and m[0]["reading"]["time"] == 4904.0  # (down now; the last answer kept)
    assert m[0]["series"][0] == {"time": 4900.0, "temp": None, "cpu": 50.0, "mem": 20.0}
