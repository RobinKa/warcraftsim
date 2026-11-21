"""scripts/host_stats.py's rows (what the remote command prints -> a row; WSL has no sensors; a host that does not answer;
this machine read in place from /proc, /sys and NVML, all fed here: no GPU needed) and the dashboard's machines: the
games split by machine, the readings shown."""

import importlib.util
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

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


def test_host_status_for_the_header(tmp_path):
    from warcraftsim.dashboard.server import HOST_FRESH, _host_status
    d = tmp_path / "hosts"
    assert _host_status(d, _JsonlCache(), 5000.0) == []  # (no directory)
    d.mkdir()
    now = 5000.0
    def write(name, *rows):
        (d / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    write("up", {"time": now - 10, "up": True, "games": 56, "temp": 79.0})
    write("off", {"time": now - 600, "up": True, "games": 56}, {"time": now - 10, "up": False})
    write("stale", {"time": now - HOST_FRESH - 1, "up": True, "games": 56})
    by = {h["name"]: h for h in _host_status(d, _JsonlCache(), now)}
    assert by["up"]["state"] == "up" and by["up"]["games"] == 56 and by["up"]["temp"] == 79.0
    assert by["off"]["state"] == "down" and by["off"]["last_up"] == now - 600 and "games" not in by["off"]
    assert by["stale"]["state"] == "unknown" and "games" not in by["stale"]  # (nothing is known now)


# ---- this machine, read in place (host_stats.Local, host_stats.Gpu) ------------------------------------------------

class FakeNVML:
    """pynvml's calls host_stats.Gpu makes, answering canned readings (an RTX 3090 under a run)."""
    NVML_CLOCK_SM, NVML_CLOCK_MEM = 1, 2
    NVML_GPU_UTILIZATION_SAMPLES, NVML_MEMORY_UTILIZATION_SAMPLES = 1, 2
    NVML_TEMPERATURE_GPU = 0

    def __init__(self):
        self.samples = {1: [(1, 80), (2, 90)], 2: [(1, 40), (2, 50)]}  # kind -> [(time µs, value)]
        self.energy, self.reasons = 1_000_000, 0x4
        self.since = []

    def nvmlInit(self):
        pass

    def nvmlDeviceGetHandleByIndex(self, i):
        return "h"

    def nvmlDeviceGetName(self, h):
        return b"NVIDIA GeForce RTX 3090"

    def nvmlDeviceGetMaxClockInfo(self, h, clock):
        return 2100

    def nvmlGpmQueryDeviceSupport(self, h):
        return SimpleNamespace(isSupportedDevice=0)  # (GPM: Hopper and newer)

    def nvmlDeviceGetSamples(self, h, kind, since):
        got = [SimpleNamespace(timeStamp=t, sampleValue=SimpleNamespace(uiVal=v)) for t, v in self.samples[kind] if t > since]
        if not got:
            raise RuntimeError("NVML_ERROR_NOT_FOUND")  # (as NVML: nothing newer)
        return 1, got

    def nvmlDeviceGetCurrentClocksEventReasons(self, h):
        return self.reasons

    def nvmlDeviceGetMemoryInfo(self, h):
        return SimpleNamespace(used=16 * 2**30, total=24 * 2**30)

    def nvmlDeviceGetTotalEnergyConsumption(self, h):
        return self.energy

    def nvmlDeviceGetPowerUsage(self, h):
        return 300_000

    def nvmlDeviceGetEnforcedPowerLimit(self, h):
        return 350_000

    def nvmlDeviceGetTemperature(self, h, sensor):
        return 64

    def nvmlDeviceGetFanSpeed(self, h):
        raise RuntimeError("NVML_ERROR_NOT_SUPPORTED")

    def nvmlDeviceGetClockInfo(self, h, clock):
        return {1: 1845, 2: 9501}[clock]

    def nvmlDeviceGetPerformanceState(self, h):
        return 2

    def nvmlDeviceGetComputeRunningProcesses(self, h):
        return [SimpleNamespace(pid=9, usedGpuMemory=2**30), SimpleNamespace(pid=7, usedGpuMemory=15 * 2**30),
                SimpleNamespace(pid=8, usedGpuMemory=None)]

    def nvmlDeviceGetProcessUtilization(self, h, since):
        self.since.append(since)
        return [SimpleNamespace(pid=7, smUtil=80), SimpleNamespace(pid=7, smUtil=90), SimpleNamespace(pid=9, smUtil=5)]

    def nvmlDeviceGetUtilizationRates(self, h):
        return SimpleNamespace(gpu=55, memory=20)


def test_profiling_permission_from_the_driver(tmp_path):
    p = tmp_path / "params"
    p.write_text("ResmanDebugLevel: 4294967295\nRmProfilingAdminOnly: 1\nRmLogonRC: 1\n")
    assert host_stats.profiling_admin_only(p) is True
    p.write_text("RmProfilingAdminOnly: 0\n")
    assert host_stats.profiling_admin_only(p) is False
    assert host_stats.profiling_admin_only(tmp_path / "none") is None  # (no NVIDIA driver: it does not say)


def test_gpu_readings_are_the_intervals_means(tmp_path):
    (tmp_path / "params").write_text("RmProfilingAdminOnly: 1\n")
    nv = FakeNVML()
    g = host_stats.Gpu(nv, params=tmp_path / "params")
    assert g.name == "NVIDIA GeForce RTX 3090" and g.gpm is False and g.admin_only is True
    g.gather()
    nv.samples[1].append((3, 100))  # (a new sample; the two old ones are not counted again)
    r = g.read(now=100.0)
    assert r["util"] == 90.0 and r["mem_util"] == 45.0
    assert r["watts"] == 300.0 and r["watts_limit"] == 350  # (first look: the moment's power, no energy difference yet)
    assert r["capped"] == 100 and r["capped_n"] == 2  # (the gathering and the look's own: both at the power limit)
    assert r["throttle"] == ["power cap"] and r["pstate"] == "P2" and r["sm_mhz"] == 1845 and r["mem_mhz"] == 9501
    assert r["sm_mhz_max"] == 2100 and r["temp"] == 64 and r["fan"] is None  # (a reading NVML refuses: null)
    assert r["mem_used"] == 16.0 and r["mem_total"] == 24.0 and r["gpm"] is False and r["prof_admin_only"] is True
    assert [p["pid"] for p in r["procs"]] == [7, 9, 8]  # (the most memory first)
    assert r["procs"][0] == {"pid": 7, "mem": 15.0, "sm": 85} and r["procs"][2]["mem"] is None and r["procs"][2]["sm"] is None
    # the next interval: no new samples (NVML's moment's figure), 310 W from the energy counter, never at the limit
    nv.energy += 3_100_000  # (mJ over 10 s)
    nv.reasons = 0x1
    r = g.read(now=110.0)
    assert r["util"] == 55 and r["mem_util"] == 20 and r["watts"] == 310.0
    assert r["capped"] == 0 and r["capped_n"] == 1 and r["throttle"] == ["idle"]
    assert nv.since[-1] == 100_000_000  # (the per-process samples since the last look, µs)


def test_which_run_a_process_is():
    assert host_stats.run_of(["python", "-m", "warcraftsim.fullgame.selfplay", "--name", "fgself-9", "--bf16", "1", ""]) \
        == ("fgself-9", "learner")
    assert host_stats.run_of(["python", "-m", "warcraftsim.fullgame.evalworker", "--run=burst-1"]) == ("burst-1", "evaluation")
    assert host_stats.run_of(["python", "-c", "from multiprocessing.spawn import spawn_main; spawn_main()"]) is None
    assert host_stats.run_of(["python", "-m", "warcraftsim", "dashboard"]) is None
    assert host_stats.run_of(["python", "-m"]) is None and host_stats.run_of([]) is None


def _train(path, times, steps_each=8000):
    rows = [{"epoch": i, "time": t, "agent_steps": i * steps_each, "perf/train": 1.5, "perf/rollout": 0.5, "x": "y" * 300}
            for i, t in enumerate(times)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_a_runs_speed_from_train_jsonls_tail(tmp_path):
    f = tmp_path / "train.jsonl"
    _train(f, [1000.0 + 10 * i for i in range(11)])  # (800 steps a second; 75% of each cycle in the update)
    r = host_stats.run_speed(f, now=1105.0)
    assert r == {"steps": 80000, "age": 5.0, "steps_per_s": 800.0, "update_share": 0.75}
    assert host_stats.run_speed(f, 1105.0, window=30)["steps_per_s"] == 800.0  # (the newest 4 rows)
    small = host_stats.run_speed(f, 1105.0, tail=1000)  # (a cut first line skipped: only whole rows)
    assert small["steps"] == 80000 and small["steps_per_s"] == 800.0
    f.write_text(f.read_text() + '{"time": 1200.0, "eval": 1}\n' + "null\n")  # (rows without steps are not its steps)
    assert host_stats.run_speed(f, 1205.0)["steps"] == 80000
    assert host_stats.run_speed(tmp_path / "none.jsonl", 1.0) == {}
    _train(f, [1000.0])
    assert host_stats.run_speed(f, 1000.0) == {"steps": 0, "age": 0.0, "steps_per_s": None, "update_share": None}


def _fake_machine(tmp_path):
    """A /proc and /sys as files: a learner (its actor and Xvfb below it), two games, an evaluation worker and its
    process; k10temp; a run's train.jsonl and run.json."""
    proc, sysd, runs = tmp_path / "proc", tmp_path / "sys", tmp_path / "runs"
    proc.mkdir()
    (proc / "stat").write_text("cpu  1000 0 500 8000 100 0 0 0 0 0\ncpu0 1 1 1 1\n")
    (proc / "meminfo").write_text("MemTotal:       62914560 kB\nMemFree:          100 kB\nMemAvailable:    9437184 kB\n")
    (proc / "loadavg").write_text("63.19 69.42 62.02 70/3000 12345\n")
    (proc / "uptime").write_text("2257.90 1000.00\n")
    (proc / "cpuinfo").write_text("processor\t: 0\ncpu MHz\t\t: 4400.000\n\nprocessor\t: 1\ncpu MHz\t\t: 4300.000\n")
    (proc / "self").mkdir()  # (not a process: not a number)

    def p(pid, comm, ppid, anon_kb, *cmd):
        d = proc / str(pid)
        d.mkdir()
        (d / "stat").write_text(f"{pid} ({comm}) S {ppid} {pid} {pid} 0 -1 4194560 0 0 0 0 10 5 0 0 20 0 1 0 100\n")
        (d / "status").write_text(f"Name:\t{comm}\nRssAnon:\t{anon_kb} kB\nRssFile:\t999999 kB\n")
        (d / "cmdline").write_bytes(("\0".join(cmd or (comm,)) + "\0").encode())
    p(1, "systemd", 0, 1024)
    p(100, "python", 1, 4 * 1048576, "python", "-m", "warcraftsim.fullgame.selfplay", "--name", "fgself-9")
    p(101, "python", 100, 1048576, "python", "-c", "from multiprocessing.spawn import spawn_main")  # (an actor)
    p(102, "Xvfb", 100, 524288)
    p(200, "Warcraft III.ex", 1, 2 * 1048576)
    p(201, "Warcraft III.ex", 1, 2 * 1048576)
    p(300, "python3.14", 1, 524288, "python", "-m", "warcraftsim.fullgame.evalworker", "--run", "burst-1")
    p(301, "python3.14", 300, 524288, "python", "-c", "from multiprocessing.spawn import spawn_main")
    hw = sysd / "class" / "hwmon"
    for n, (name, labels) in enumerate([("nvme", {"temp1": "Composite"}), ("k10temp", {"temp1": "Tctl", "temp3": "Tccd1"})]):
        d = hw / f"hwmon{n}"
        d.mkdir(parents=True)
        (d / "name").write_text(name + "\n")
        for t, lab in labels.items():
            (d / f"{t}_label").write_text(lab + "\n")
            (d / f"{t}_input").write_text("70875\n" if lab == "Tctl" else "40000\n")
    (sysd / "firmware" / "acpi").mkdir(parents=True)
    (sysd / "firmware" / "acpi" / "platform_profile").write_text("performance\n")
    (runs / "fgself-9").mkdir(parents=True)
    _train(runs / "fgself-9" / "train.jsonl", [time.time() - 100 + 10 * i for i in range(11)])
    (runs / "fgself-9" / "run.json").write_text(json.dumps({"config": {"bf16": 1, "infer_precision": "bf16-mixed"}}))
    return proc, sysd, runs


def test_this_machine_read_in_place(tmp_path, monkeypatch):
    proc, sysd, runs = _fake_machine(tmp_path)

    def no_daemon(cmd, **kw):  # (powerprofilesctl: none here; the ACPI profile instead)
        raise FileNotFoundError(cmd[0])
    monkeypatch.setattr(host_stats.subprocess, "run", no_daemon)
    nv = FakeNVML()
    nv.nvmlDeviceGetComputeRunningProcesses = lambda h: [SimpleNamespace(pid=100, usedGpuMemory=15 * 2**30),
                                                         SimpleNamespace(pid=301, usedGpuMemory=2**30),
                                                         SimpleNamespace(pid=999, usedGpuMemory=2**20)]
    local = host_stats.Local("tora", runs, proc=proc, sys_dir=sysd, gpu=host_stats.Gpu(nv, params=tmp_path / "x"),
                             first_wait=0)
    r = local.look()
    assert r["host"] == "tora" and r["up"] and r["local"] is True
    assert r["cpu_busy"] is None  # (the first look: the same /proc/stat twice here)
    assert r["mem_total"] == 60.0 and r["mem_avail"] == 9.0 and r["load"] == [63.19, 69.42, 62.02]
    assert r["uptime"] == 2257.9 and (r["mhz_mean"], r["mhz_max"]) == (4350, 4400) and r["games"] == 2
    assert r["temp"] == 70.875 and r["profile"] == "performance"
    assert (r["mem_learner"], r["mem_others"], r["mem_games"]) == (4.0, 2.5, 4.0)  # (actor, Xvfb, evaluation: 1+.5+.5+.5)
    learner, ev = r["live"]
    assert learner["run"] == "fgself-9" and learner["kind"] == "learner" and learner["steps_per_s"] == 800.0
    assert learner["update_share"] == 0.75 and learner["bf16"] is True and learner["infer_precision"] == "bf16-mixed"
    assert ev == {"run": "burst-1", "kind": "evaluation"}
    names = {p["pid"]: p["name"] for p in r["gpu"]["procs"]}
    assert names == {100: "fgself-9 learner", 301: "burst-1 evaluation", 999: "999"}  # (a process gone: its pid)
    assert r["gpu"]["prof_admin_only"] is None  # (no driver params file here)
    (proc / "stat").write_text("cpu  1100 0 550 8050 100 0 0 0 0 0\n")
    assert local.look()["cpu_busy"] == 75.0  # (since the last look: 150 of 200 ticks busy)


def test_this_machine_without_gpu_nor_sensors(tmp_path, monkeypatch):
    proc, sysd, runs = _fake_machine(tmp_path)
    monkeypatch.setattr(host_stats.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "balanced\n", ""))
    for d in (sysd / "class" / "hwmon").iterdir():
        (d / "name").write_text("nvme\n")
    r = host_stats.Local("tora", runs, proc=proc, sys_dir=sysd, first_wait=0).look()
    assert "gpu" not in r and r["temp"] is None and r["profile"] == "balanced" and r["games"] == 2


def test_the_local_look_runs_here(tmp_path):
    """The real /proc and /sys of this machine, no GPU: the readings are there (whatever their numbers)."""
    r = host_stats.Local("here", tmp_path, gpu=None, first_wait=0.2).look()
    assert r["mem_total"] > 0 and isinstance(r["games"], int) and len(r["load"]) == 3 and isinstance(r["live"], list)
    assert r["mem_games"] >= 0 and r["uptime"] > 0


def test_this_machine_first_in_the_system_tab_and_not_in_the_header(tmp_path):
    from warcraftsim.dashboard.server import _host_status
    d = tmp_path / "hosts"
    d.mkdir()
    now = 5000.0
    box = {"up": True, "cpu_busy": 50.0, "mem_total": 30.0, "mem_avail": 10.0, "temp": 80.0, "games": 64}
    here = {**box, "local": True, "gpu": {"util": 88.0, "watts": 322.3, "watts_limit": 350}}
    (d / "box.jsonl").write_text("".join(json.dumps({**box, "time": now - 60 + i}) + "\n" for i in range(3)))
    (d / "tora.jsonl").write_text("".join(json.dumps({**here, "time": now - 60 + i}) + "\n" for i in range(3)))
    m = _machines(d, _JsonlCache(), now)
    assert [x["name"] for x in m] == ["tora", "box"]
    assert m[0]["series"][-1] == {"time": now - 58, "temp": 80.0, "cpu": 50.0, "mem": 20.0, "gpu": 88.0, "gpu_w": 322.3}
    assert "gpu" not in m[1]["series"][-1]
    assert [h["name"] for h in _host_status(d, _JsonlCache(), now)] == ["box"]  # (the header: the other machines)
