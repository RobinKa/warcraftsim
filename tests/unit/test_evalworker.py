"""The remote evaluation worker (fullgame/evalworker.py): its settings from a run's run.json, its rows, and the
readers counting them (scripts/real_games.py) with the run's own."""

import importlib.util
import json
import sys
from pathlib import Path

from warcraftsim import paths
from warcraftsim.fullgame import evalworker

SCRIPTS = Path(__file__).parents[2] / "scripts"


def _script(name: str):
    sys.path.insert(0, str(SCRIPTS))  # (real_games imports throughput)
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(tmp_path: Path) -> Path:
    """A run directory as selfplay writes it: its config, vars(args) as strings."""
    run = tmp_path / "fgself-x"
    run.mkdir()
    config = {"runs": "/home/someone/warcraftsim/runs", "map": "echo100", "handicap": "100", "step_seconds": "0.5",
              "max_minutes": "40.0", "game_cuts": "1529", "features": "29", "eval_ai": "insane", "races": "all",
              "eval_checkpoints": "[PosixPath('runs/fgself-73/checkpoints/0000000008386478.pt'), "
                                  "PosixPath('/home/someone/warcraftsim/runs/fgself-70/checkpoints/0000000000000001.pt')]",
              "wine": "/home/someone/wc3/ge/GE-Proton10-34/files/bin", "fsync": "1", "ntsync": "0", "seed": "0",
              "games_per_process": "2", "game_nice": "10", "native_obs": "1", "mirror": "0", "shaping": "0.5",
              "gamma": "0.9997", "float_penalty": "0.5", "infer_entities": "32,48,64,96", "exploiter_share": "0.2"}
    (run / "run.json").write_text(json.dumps({"config": config, "launch": {"argv": []}}))
    return run


def test_run_config_from_run_json(tmp_path):
    cfg, keys = evalworker.run_config(_run(tmp_path), "laptop", 12, "120+3")
    assert cfg["map"] == "echo100" and cfg["game_cuts"] == 1529 and cfg["max_steps"] == 4820
    assert cfg["eval_ai"] == ["insane"] and cfg["features"] == 29 and len(cfg["races"]) == 4
    # every slot evaluates, nothing trains, the draws are this worker's
    assert cfg["games_per_actor"] == 0 and cfg["eval_games_per_actor"] == 12 and cfg["exploiter_share"] == 0.0
    assert cfg["seed"] == "0/laptop" and cfg["start_update"] == "120+3"
    # the evaluation checkpoints here, by the run's own names (a path under the run machine's repository too)
    root = paths.REPO_ROOT
    assert keys == {str(root / "runs/fgself-73/checkpoints/0000000008386478.pt"): "runs/fgself-73/checkpoints/0000000008386478.pt",
                    str(root / "runs/fgself-70/checkpoints/0000000000000001.pt"):
                        "/home/someone/warcraftsim/runs/fgself-70/checkpoints/0000000000000001.pt"}
    assert cfg["eval_checkpoints"] == list(keys)
    assert cfg["_wine"] == str(Path.home() / "wc3/ge/GE-Proton10-34/files/bin") and cfg["_fsync"] and not cfg["_ntsync"]


class _Q(list):
    put = list.append


def test_rows_by_the_runs_own_names():
    infer = type("I", (), {"nets": type("N", (), {"version": 7})()})()
    q = _Q()
    rows = evalworker.Rows(q, "laptop", {"/here/a.pt": "runs/a.pt"}, infer)
    rows.put({"steps": []})  # (no trajectories)
    rows.put({"episode": {"eval": True, "eval_key": "/here/a.pt"}})
    rows.put({"episode": {"eval": True, "eval_key": "current"}})
    assert q == [{"eval": True, "eval_key": "runs/a.pt", "host": "laptop"},
                 {"eval": True, "eval_key": "current", "host": "laptop", "version": 7}]


def test_real_games_count_remote_rows(tmp_path, monkeypatch):
    rg = _script("real_games")
    run = tmp_path / "fgself-x"
    run.mkdir()

    def row(t, key, outcome, **kw):
        return json.dumps({"time": t, "opponent": "script:ai-insane (real)", "race": "human", "opponent_race": "orc",
                           "outcome": outcome, "eval": True, "eval_key": key, **kw}) + "\n"
    (run / "episodes.jsonl").write_text(row(3, "current", 1.0) + row(1, "old", -1.0))
    (run / "episodes-remote.jsonl").write_text(row(2, "current", 1.0, host="laptop") + row(4, "old", 1.0, host="laptop")
                                               + '{"time": 5, "opp')  # (a line being written)
    monkeypatch.setattr(rg, "RUNS", tmp_path)
    eps = rg.episodes("fgself-x")
    assert [e["time"] for e in eps] == [1, 2, 3, 4]
    assert len(rg.episodes("fgself-x", remote=False)) == 2
    assert [rg.value(e, "host") for e in eps] == ["main", "laptop", "main", "laptop"]
    rates = {("human", "orc", "insane"): 0.5}
    by_key, n = rg.standardised(eps, "eval_key", rates)
    assert n == 4 and by_key == {"current": 200.0, "old": 100.0}
    by_host, _ = rg.standardised(eps, "host", rates)
    assert by_host == {"laptop": 200.0, "main": 100.0}


def test_remote_eval_manager_follows_the_run(tmp_path, monkeypatch):
    """scripts/remote_eval.py's looks: the worker started once, left be over a restart of the run with the same
    evaluation settings, drained (its games in flight finish) and started again for other settings, started after
    the host was away, drained for the next run (stopped when the drain takes too long), and drained when the run
    ended (after the grace)."""
    rm = _script("remote_eval")
    monkeypatch.setattr(rm, "ROOT", tmp_path)
    monkeypatch.setattr(rm, "RUNS", tmp_path / "runs")
    (tmp_path / "runs/x/checkpoints").mkdir(parents=True)
    (tmp_path / "runs/x/checkpoints/1.pt").write_bytes(b"")
    for run in ("r1", "r2"):
        (tmp_path / "runs" / run).mkdir()
    monkeypatch.setattr(rm.Manager, "push_code", lambda self, rev="HEAD": True)
    monkeypatch.setattr(rm, "newest_checkpoint", lambda run: "0000000000000100.pt")
    configs = {"r1": {"map": "echo100", "eval_checkpoints": "None"}, "r2": {"map": "echo100", "eval_checkpoints": "None"}}
    monkeypatch.setattr(rm, "run_info", lambda run: {"config": configs[run]})
    live = ["r1"]
    monkeypatch.setattr(rm, "live_runs", lambda: list(live))
    there = {"active": "inactive", "unit": f"{rm.UNIT}.service", "away": False}
    calls = []

    class Host:
        def run(self, cmd, timeout=120, binary=False):
            if "systemd-run" in cmd:
                calls.append(("start", cmd.split("--description=")[1].split(" -p")[0].strip("'")))
                there.update(active="active", unit=calls[-1][1])
            elif "SIGUSR1" in cmd:
                calls.append(("drain", there["unit"]))
            elif "systemctl --user stop" in cmd:
                if there["active"] == "active":
                    calls.append(("stop", there["unit"]))
                there.update(active="inactive", unit=f"{rm.UNIT}.service")
            return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        def rsync(self, files, timeout=900):
            return True
    a = type("A", (), dict(host="h", name="h", run=None, games=4, processes=0, remote_dir="warcraftsim", every=60.0,
                           push_every=240.0, keep=3, grace=10.0, nice=10, gauntlet_games=0, gauntlet_size=5))()
    m = rm.Manager(a)
    m.h = Host()
    monkeypatch.setattr(m, "state", lambda: None if there["away"] else
                        {"active": there["active"], "unit": there["unit"], "lines": 0, "clock": 0.0, "status": {}})
    key1 = rm.launch_key({"config": configs["r1"]})

    def starts():
        return sum(c[0] == "start" for c in calls)
    assert m.cycle() and calls == [("start", f"warcraftsim evalworker r1 {key1} 4x0")]
    live.clear()  # (the run restarting: the gap)
    assert m.cycle() and m.cycle() and calls == [("start", f"warcraftsim evalworker r1 {key1} 4x0")]
    live.append("r1")
    configs["r1"] = {**configs["r1"], "lr": "1e-5"}  # (a restart with another learning rate: the same games)
    assert m.cycle() and starts() == 1
    configs["r1"] = {**configs["r1"], "eval_checkpoints": "[PosixPath('runs/x/checkpoints/1.pt')]"}
    assert m.cycle() and calls[-1] == ("drain", f"warcraftsim evalworker r1 {key1} 4x0")
    assert m.cycle() and calls[-1][0] == "drain" and len(calls) == 2  # (draining: its games in flight finish)
    there.update(active="inactive", unit=f"{rm.UNIT}.service")  # (they did: it exited)
    assert m.cycle() and calls[-1] == ("start", f"warcraftsim evalworker r1 {rm.launch_key({'config': configs['r1']})} 4x0")
    there["away"] = True
    assert m.cycle() and starts() == 2
    there.update(away=False, active="inactive", unit=f"{rm.UNIT}.service")  # (WSL powered off meanwhile)
    assert m.cycle() and calls[-1][0] == "start" and starts() == 3
    live[:] = ["r1", "r2"]  # (the next run: r1's worker drained, r2's own evaluation games held meanwhile)
    assert m.cycle() and calls[-1][0] == "drain" and (tmp_path / "runs/r2/remote-eval.alive").exists()
    m.draining -= 3600  # (a drain that takes too long: stopped)
    assert m.cycle() and [c[0] for c in calls[-2:]] == ["stop", "start"] and "r2" in calls[-1][1]
    live.clear()
    assert m.cycle() and calls[-1][0] == "start"  # (within the grace)
    m.gone_since -= 3600
    assert m.cycle() and calls[-1] == ("drain", f"warcraftsim evalworker r2 {rm.launch_key({'config': configs['r2']})} 4x0")
    there.update(active="inactive", unit=f"{rm.UNIT}.service")
    a.run = "r2"  # (with --run: done when it ended)
    m.gone_since = None
    assert m.cycle()
    m.gone_since -= 3600
    assert not m.cycle()


def test_run_config_has_what_the_games_read(tmp_path):
    """Every cfg["..."] that selfplay's game_loop and play_one read is set by run_config or the worker's main (a
    key the learner's cfg gained since, missing here, would fail every game of the worker)."""
    import re

    src = (paths.REPO_ROOT / "warcraftsim" / "fullgame" / "selfplay.py").read_text()
    body = src[src.index("def game_loop("):src.index("def film_game(")]
    read = set(re.findall(r'cfg\["([a-z_]+)"\]', body))
    cfg, _ = evalworker.run_config(_run(tmp_path), "laptop", 2, 0)
    main_sets = {"values", "costs", "costs_array", "map_geometry", "device", "slot", "learner_pid", "actors"}
    assert read - set(cfg) - main_sets == set()
    (tmp_path / "b").mkdir()
    run = _run(tmp_path / "b")
    data = json.loads((run / "run.json").read_text())
    data["config"]["z_build"] = "2"  # (the builder a teacher only: a number, not a flag)
    (run / "run.json").write_text(json.dumps(data))
    assert evalworker.run_config(run, "laptop", 2, 0)[0]["z_build"] == 2
