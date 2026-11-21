"""How the machines of a run are doing (the dashboard's System tab, "Machines"): every --every seconds, for each --host
one ssh call that reads /proc (CPU busy over ~1 s, memory, load, clocks, the game processes, uptime), `sensors` (the
CPU's Tctl: WSL has none, null) and the power profile, as a row in runs/hosts/<name>.jsonl. A host that does not
answer gets a row with up: false and the others carry on. Light: no torch, one short ssh call a host. Rows older than
--keep days are dropped at the start (and every few hours).

This machine (the learner's: --local, on by default, named by its hostname) is read in this process instead (Local:
no ssh, no shell): the same readings (CPU busy over the time since the last look), the memory of the learners, their
other processes and the games, the GPU through NVML (Gpu: utilization and power as the interval's means, memory,
temperature, fan, clocks, what holds the clocks down, the processes on it), and the live runs (a learner's run, its
agent steps a second from train.jsonl's newest rows, the share of its cycle in its BF16 update). Its rows carry
"local": true.

Tensor cores: no counter of theirs is readable here, and none is made up. NVML's tensor metrics (GPM) need a Hopper or
newer GPU (nvmlGpmQueryDeviceSupport: not on the RTX 3090); the GA102's own counter (sm__pipe_tensor_cycles_active,
CUPTI's PM sampling) is there, but reading it is admin-only on this driver (/proc/driver/nvidia/params:
RmProfilingAdminOnly: 1; CUPTI answers CUPTI_ERROR_INSUFFICIENT_PRIVILEGES). The row says which (gpu.gpm,
gpu.prof_admin_only); the dashboard shows the proxies (the GPU's busy time, an upper bound; the learner's BF16 share).

    setsid nohup .venv/bin/python scripts/host_stats.py --host robin@laptop-wsl --host tora-ms-a2@tora-ms-a2 \\
        > runs/host_stats.log 2>&1 &
    .venv/bin/python scripts/host_stats.py --host robin@laptop-wsl --once     # one look, printed, nothing written
    .venv/bin/python scripts/host_stats.py --once                             # this machine only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4"]
# What runs there (sh): a key and its value a line. The games are counted by their comm in /proc (awk reads them all in
# one process; pgrep would match its own shell, whose command line holds the name).
REMOTE = r"""
echo "stat1 $(head -1 /proc/stat)"; sleep 1; echo "stat2 $(head -1 /proc/stat)"
grep -E '^(MemTotal|MemAvailable):' /proc/meminfo
echo "load $(cut -d' ' -f1-3 /proc/loadavg)"
echo "uptime $(cut -d' ' -f1 /proc/uptime)"
echo "mhz $(awk '/^cpu MHz/{s+=$4;n++;if($4>m)m=$4} END{if(n)print s/n, m}' /proc/cpuinfo)"
echo "games $(awk 'FNR==1&&$0=="Warcraft III.ex"{n++} END{print n+0}' /proc/[0-9]*/comm 2>/dev/null)"
if command -v sensors >/dev/null 2>&1; then sensors 2>/dev/null | grep -m1 -E 'Tctl|Package id 0' | sed 's/^/temp /'; fi
p=$(powerprofilesctl get 2>/dev/null || cat /sys/firmware/acpi/platform_profile 2>/dev/null); echo "profile $p"
"""


def log(*a) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def _cpu_busy(line1: str, line2: str) -> float | None:
    """Busy percent between two `cpu ...` lines of /proc/stat (idle and iowait count as idle)."""
    def parse(s: str) -> tuple[float, float]:
        v = [float(x) for x in s.split()[1:9]]
        return sum(v), v[3] + v[4]
    try:
        (t1, i1), (t2, i2) = parse(line1), parse(line2)
    except (ValueError, IndexError):
        return None
    return round(100 * (1 - (i2 - i1) / (t2 - t1)), 1) if t2 > t1 else None


def parse_stats(text: str, now: float, name: str) -> dict:
    """The row for what REMOTE printed. A reading missing there (no sensors, no clocks) is null."""
    f: dict[str, str] = {}
    for line in text.splitlines():
        k, _, v = line.partition(" ")
        f.setdefault(k.rstrip(":"), v.strip())
    row: dict = {"time": now, "host": name, "up": True}
    row["cpu_busy"] = _cpu_busy(f["stat1"], f["stat2"]) if f.get("stat1") and f.get("stat2") else None
    for key, out in (("MemTotal", "mem_total"), ("MemAvailable", "mem_avail")):
        m = re.match(r"(\d+)", f.get(key, ""))
        row[out] = round(int(m.group(1)) / 1048576, 2) if m else None  # (GiB)
    try:
        row["load"] = [float(x) for x in f["load"].split()[:3]]
    except (KeyError, ValueError):
        row["load"] = None
    m = re.match(r"\d+(?:\.\d+)?", f.get("uptime", ""))
    row["uptime"] = float(m.group()) if m else None
    mhz = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", f.get("mhz", ""))]
    row["mhz_mean"], row["mhz_max"] = (round(mhz[0]), round(mhz[1])) if len(mhz) >= 2 else (None, None)
    row["games"] = int(f["games"]) if f.get("games", "").isdigit() else None
    m = re.search(r"(?:Tctl|Package id 0):\s*\+?(-?\d+(?:\.\d+)?)", f.get("temp", ""))
    row["temp"] = float(m.group(1)) if m else None
    row["profile"] = f.get("profile") or None
    return row


def look(host: str, name: str, timeout: float = 25.0) -> dict:
    """One host's row (up: false when it did not answer in time or the call failed)."""
    now = time.time()
    try:
        r = subprocess.run(SSH + [host, "sh -c " + shlex.quote(REMOTE)], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:  # (Tailscale SSH waiting for an approval in a browser)
        return {"time": now, "host": name, "up": False, "error": "timeout"}
    if r.returncode != 0 or "stat2" not in r.stdout:
        return {"time": now, "host": name, "up": False,
                "error": (r.stderr.strip().splitlines() or [f"exit {r.returncode}"])[-1][:200]}
    return parse_stats(r.stdout, now, name)


GAME = "Warcraft III.ex"  # (a game's comm: its name cut to 15 characters)
GATHER = 5.0  # seconds between gathering the GPU's utilization samples between looks (NVML keeps ~14 s of them)
PROFILE_EVERY = 600.0  # seconds between asking powerprofilesctl (a Python script: not at every look)
RUN_WINDOW = 120.0  # seconds of train.jsonl's newest rows behind a run's steps a second
RUN_TAIL = 262144  # bytes of train.jsonl read for them (its rows are ~3 KB, an update every ~2 s)
GPU_PROCS = 5  # the processes on the GPU listed (the most memory first)
# NVML's clock event reasons (nvmlClocksEventReasons; "throttle reasons" before): what holds the clocks below their most
THROTTLE = {0x1: "idle", 0x2: "application clocks", 0x4: "power cap", 0x8: "hardware slowdown", 0x10: "sync boost",
            0x20: "software thermal", 0x40: "hardware thermal", 0x80: "power brake", 0x100: "display clocks"}


def _try(f, *a, default=None):
    """f(*a), or `default` when it fails (a reading this GPU or driver does not have: NVML raises)."""
    try:
        return f(*a)
    except Exception:  # noqa: BLE001 (pynvml's NVMLError and its many subclasses; a fake's errors in the tests)
        return default


def _text(p: Path | str) -> str | None:
    try:
        with open(p, errors="replace") as f:
            return f.read()
    except OSError:
        return None


def profiling_admin_only(params: Path = Path("/proc/driver/nvidia/params")) -> bool | None:
    """Whether the GPU's performance counters (CUPTI, Nsight: a tensor-core counter) are for root only (the driver's
    NVreg_RestrictProfilingToAdminUsers); None when the driver does not say."""
    m = re.search(r"^RmProfilingAdminOnly:\s*(\d+)", _text(params) or "", re.M)
    return None if m is None else m.group(1) != "0"


class Gpu:
    """The GPU's readings through NVML (pynvml, in the venv: no CUDA context, nothing runs on the GPU; ~2 ms of CPU a
    look). Utilization is NVML's own samples (one every ~0.2 s, ~14 s kept) gathered between looks, so a look's figure
    is the mean over the time since the last one, not a moment's; power is the energy counter's difference (the
    interval's mean); "capped" the share of the gatherings (~every 5 s) at which the power limit held the clocks."""

    def __init__(self, nv, index: int = 0, params: Path = Path("/proc/driver/nvidia/params")):
        self.nv = nv
        nv.nvmlInit()
        self.h = nv.nvmlDeviceGetHandleByIndex(index)
        name = nv.nvmlDeviceGetName(self.h)
        self.name = name.decode() if isinstance(name, bytes) else str(name)
        self.sm_max = _try(nv.nvmlDeviceGetMaxClockInfo, self.h, nv.NVML_CLOCK_SM)
        support = _try(nv.nvmlGpmQueryDeviceSupport, self.h)  # (the tensor-core metrics' API: Hopper and newer)
        self.gpm = bool(getattr(support, "isSupportedDevice", 0))
        self.admin_only = profiling_admin_only(params)
        self.reasons = getattr(nv, "nvmlDeviceGetCurrentClocksEventReasons", None) \
            or getattr(nv, "nvmlDeviceGetCurrentClocksThrottleReasons", None)
        self.seen = {"util": 0, "mem_util": 0}  # the newest sample's time (µs) taken so far, by kind
        self.sums = {"util": [0.0, 0], "mem_util": [0.0, 0]}
        self.capped = [0, 0]  # the gatherings since the last look with the power limit holding the clocks, all of them
        self.last: dict[str, tuple[float, float]] = {}  # counter -> (time, value) at the last look
        self.procs_since = 0  # (µs) the per-process samples' start: the last look

    @classmethod
    def open(cls) -> "Gpu | None":
        """The first GPU, or None (no pynvml, no driver, no GPU)."""
        try:
            import pynvml
            return cls(pynvml)
        except Exception:  # noqa: BLE001
            return None

    def gather(self) -> None:
        """The utilization samples NVML took since the last gathering, added to the interval's means; whether the power
        limit holds the clocks now (NVML's counter of that time, NVML_FI_DEV_PERF_POLICY_POWER, stands still on this
        card and driver: the reason is sampled instead, at every gathering)."""
        reasons = _try(self.reasons, self.h) if self.reasons else None
        if isinstance(reasons, int):
            self.capped[0] += bool(reasons & 0x4)
            self.capped[1] += 1
        for key, kind in (("util", self.nv.NVML_GPU_UTILIZATION_SAMPLES), ("mem_util", self.nv.NVML_MEMORY_UTILIZATION_SAMPLES)):
            got = _try(self.nv.nvmlDeviceGetSamples, self.h, kind, self.seen[key])  # (none newer: NVML raises)
            for s in (got[1] if got else []):
                if s.timeStamp > self.seen[key]:
                    v = s.sampleValue.uiVal
                    if 0 <= v <= 100:
                        self.sums[key][0] += v
                        self.sums[key][1] += 1
            if got and got[1]:
                self.seen[key] = max(self.seen[key], max(s.timeStamp for s in got[1]))

    def _rate(self, key: str, value: float | None, now: float) -> float | None:
        """The counter's change a second since the last look (None at the first look, or when it did not grow)."""
        if value is None:
            return None
        prev = self.last.get(key)
        self.last[key] = (now, value)
        if prev is None or now <= prev[0] or value < prev[1]:
            return None
        return (value - prev[1]) / (now - prev[0])

    def read(self, now: float | None = None) -> dict:
        nv, h = self.nv, self.h
        now = time.time() if now is None else now
        self.gather()
        means = {k: (round(s / n, 1) if n else None) for k, (s, n) in self.sums.items()}
        self.sums = {k: [0.0, 0] for k in self.sums}
        if means["util"] is None:  # (no samples: NVML's moment's figure)
            u = _try(nv.nvmlDeviceGetUtilizationRates, h)
            means = {"util": getattr(u, "gpu", None), "mem_util": getattr(u, "memory", None)} if u else means
        mem = _try(nv.nvmlDeviceGetMemoryInfo, h)
        mw = self._rate("energy", _try(nv.nvmlDeviceGetTotalEnergyConsumption, h), now)  # (mJ a second: mW)
        if mw is None:
            mw = _try(nv.nvmlDeviceGetPowerUsage, h)
        limit = _try(nv.nvmlDeviceGetEnforcedPowerLimit, h)
        capped, self.capped = self.capped, [0, 0]
        reasons = _try(self.reasons, h) if self.reasons else None
        pstate = _try(nv.nvmlDeviceGetPerformanceState, h)
        row = {
            "name": self.name,
            "util": means["util"], "mem_util": means["mem_util"],
            "mem_used": round(mem.used / 2**30, 2) if mem else None,
            "mem_total": round(mem.total / 2**30, 2) if mem else None,
            "watts": round(mw / 1000, 1) if mw is not None else None,
            "watts_limit": round(limit / 1000) if limit else None,
            "capped": round(100 * capped[0] / capped[1]) if capped[1] else None,  # (% of the gatherings: ~every 5 s)
            "capped_n": capped[1],
            "temp": _try(nv.nvmlDeviceGetTemperature, h, getattr(nv, "NVML_TEMPERATURE_GPU", 0)),
            "fan": _try(nv.nvmlDeviceGetFanSpeed, h),
            "sm_mhz": _try(nv.nvmlDeviceGetClockInfo, h, nv.NVML_CLOCK_SM),
            "sm_mhz_max": self.sm_max,
            "mem_mhz": _try(nv.nvmlDeviceGetClockInfo, h, nv.NVML_CLOCK_MEM),
            "pstate": f"P{pstate}" if isinstance(pstate, int) and 0 <= pstate < 32 else None,
            "throttle": [v for k, v in THROTTLE.items() if reasons & k] if isinstance(reasons, int) else None,
            "procs": self._procs(now),
            "gpm": self.gpm, "prof_admin_only": self.admin_only,
        }
        return row

    def _procs(self, now: float) -> list[dict]:
        """The compute processes on the GPU: their memory, and their share of its time (NVML's per-process samples
        since the last look, the mean; null when it has none)."""
        procs = _try(self.nv.nvmlDeviceGetComputeRunningProcesses, self.h) or []
        samples = _try(self.nv.nvmlDeviceGetProcessUtilization, self.h, self.procs_since) or []
        self.procs_since = int(now * 1e6)
        sm: dict[int, list[int]] = {}
        for s in samples:
            sm.setdefault(s.pid, []).append(s.smUtil)
        out = [{"pid": p.pid, "mem": round(p.usedGpuMemory / 2**30, 2) if p.usedGpuMemory else None,
                "sm": round(sum(sm[p.pid]) / len(sm[p.pid])) if sm.get(p.pid) else None} for p in procs]
        out.sort(key=lambda p: -(p["mem"] or 0))
        return out[:GPU_PROCS]


def run_of(args: list[str]) -> tuple[str, str] | None:
    """(the run, what the process is) for a learner's command line (python -m warcraftsim.fullgame.selfplay --name RUN)
    or an evaluation worker's (python -m warcraftsim.fullgame.evalworker --run RUN); None for anything else (an actor:
    multiprocessing's spawn_main, no -m)."""
    if "-m" not in args[:-1]:
        return None
    mod = args[args.index("-m") + 1]
    key, kind = {"warcraftsim.fullgame.selfplay": ("--name", "learner"),
                 "warcraftsim.fullgame.evalworker": ("--run", "evaluation")}.get(mod, (None, None))
    if key is None:
        return None
    for i, a in enumerate(args):
        if a == key and i + 1 < len(args):
            return args[i + 1], kind
        if a.startswith(key + "="):
            return a.split("=", 1)[1], kind
    return None


def run_speed(train: Path, now: float, window: float = RUN_WINDOW, tail: int = RUN_TAIL) -> dict:
    """A run's last agent steps, its steps a second over train.jsonl's newest `window` seconds of rows, how old its
    newest row is, and the share of the learner's cycle in its update (perf/train over perf/train + perf/rollout, the
    rest is waiting for the games: the BF16 work's share of its time). Reads the file's last `tail` bytes only."""
    try:
        with open(train, "rb") as f:
            size = f.seek(0, 2)
            f.seek(max(0, size - tail))
            data = f.read()
    except OSError:
        return {}
    lines = data.split(b"\n")[1 if size > tail else 0:]  # (the first: cut, unless the file is all here)
    rows: list[dict] = []
    for line in reversed(lines):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict) and isinstance(r.get("agent_steps"), (int, float)) and isinstance(r.get("time"), (int, float)):
            if rows and rows[0]["time"] - r["time"] > window:
                break
            rows.append(r)
    if not rows:
        return {}
    rows.reverse()
    first, last = rows[0], rows[-1]
    out: dict = {"steps": int(last["agent_steps"]), "age": round(now - last["time"], 1), "steps_per_s": None,
                 "update_share": None}
    if last["time"] - first["time"] > 0:
        out["steps_per_s"] = round((last["agent_steps"] - first["agent_steps"]) / (last["time"] - first["time"]), 1)
    cyc = [(r["perf/train"], r["perf/rollout"]) for r in rows[1:]
           if isinstance(r.get("perf/train"), (int, float)) and isinstance(r.get("perf/rollout"), (int, float))]
    tr, ro = sum(c[0] for c in cyc), sum(c[1] for c in cyc)
    if tr + ro > 0:
        out["update_share"] = round(tr / (tr + ro), 3)
    return out


def scan(proc: Path) -> dict[int, tuple[str, int]]:
    """Every process: pid -> (its comm, its parent)."""
    out = {}
    for p in os.listdir(proc):
        if not p.isdigit():
            continue
        st = _text(f"{proc}/{p}/stat")
        if not st:
            continue
        i, j = st.find("("), st.rfind(")")
        try:
            out[int(p)] = (st[i + 1:j], int(st[j + 2:].split()[1]))
        except (ValueError, IndexError):
            pass
    return out


def rss_anon(proc: Path, pid: int) -> float:
    """A process's private resident memory (RssAnon, GiB): not its mapped files, which the games' Wine processes
    share."""
    m = re.search(r"^RssAnon:\s*(\d+)", _text(f"{proc}/{pid}/status") or "", re.M)
    return int(m.group(1)) / 1048576 if m else 0.0


class Local:
    """This machine, read in this process (no ssh, no shell): REMOTE's readings from /proc and /sys (CPU busy over the
    time since the last look; at the first look over a second, as REMOTE's), the memory of the learners, their other
    processes and the games, the GPU (Gpu) and the live runs. ~70 ms of CPU a look, most of it /proc's ~1,300
    processes; ~1 ms a gathering of the GPU's samples between looks."""

    def __init__(self, name: str, runs: Path = RUNS, proc: Path = Path("/proc"), sys_dir: Path = Path("/sys"),
                 gpu: Gpu | None = None, first_wait: float = 1.0):
        self.name, self.runs, self.proc, self.sys, self.gpu = name, runs, proc, sys_dir, gpu
        self.first_wait = first_wait
        self.stat: str | None = None  # /proc/stat's cpu line at the last look
        self.temp_file: Path | None = None  # (found at the first look)
        self.profile: tuple[float, str | None] = (float("-inf"), None)  # (when asked, the answer)
        self.configs: dict[Path, tuple[float, dict]] = {}  # run.json -> (its mtime, its config)
        self.roots: dict[int, tuple[str, str]] = {}  # the learners and evaluation workers at the last look

    def gather(self) -> None:
        if self.gpu:
            self.gpu.gather()

    def _cpu_line(self) -> str:
        return ((_text(self.proc / "stat") or "").splitlines() or [""])[0]

    def _temp(self) -> float | None:
        """The CPU's Tctl (AMD's k10temp) or Package id 0 (Intel's coretemp), from /sys/class/hwmon."""
        if self.temp_file is None:
            for d in sorted((self.sys / "class" / "hwmon").glob("hwmon*")):
                if (_text(d / "name") or "").strip() not in ("k10temp", "coretemp"):
                    continue
                for lab in sorted(d.glob("temp*_label")):
                    if (_text(lab) or "").strip() in ("Tctl", "Package id 0"):
                        self.temp_file = lab.with_name(lab.name.replace("_label", "_input"))
                        break
                if self.temp_file:
                    break
        v = _text(self.temp_file) if self.temp_file else None
        return int(v) / 1000 if v and v.strip().lstrip("-").isdigit() else None

    def _profile(self, now: float) -> str | None:
        """The power profile (powerprofilesctl, asked every PROFILE_EVERY s; or the ACPI platform profile)."""
        if now - self.profile[0] >= PROFILE_EVERY:
            p = None
            try:
                r = subprocess.run(["powerprofilesctl", "get"], capture_output=True, text=True, timeout=10)
                p = r.stdout.strip() if r.returncode == 0 else None
            except (OSError, subprocess.SubprocessError):
                pass
            p = p or (_text(self.sys / "firmware" / "acpi" / "platform_profile") or "").strip() or None
            self.profile = (now, p)
        return self.profile[1]

    def _config(self, run: str) -> dict:
        """A run's settings (run.json's "config": vars(args) of its last launch), read again when the file changes."""
        f = self.runs / run / "run.json"
        try:
            mtime = f.stat().st_mtime
            if f not in self.configs or self.configs[f][0] != mtime:
                self.configs[f] = (mtime, json.loads(f.read_text()).get("config") or {})
            return self.configs[f][1]
        except (OSError, ValueError, AttributeError):
            return {}

    def look(self) -> dict:
        """The row: REMOTE's lines made here (parse_stats reads them the same), and this machine's own readings."""
        if self.stat is None:  # (the first look: two samples a second apart, as REMOTE's)
            self.stat = self._cpu_line()
            time.sleep(self.first_wait)
        now = time.time()
        prev, self.stat = self.stat, self._cpu_line()
        procs = scan(self.proc)
        mhz = [float(x) for x in re.findall(r"^cpu MHz\s*:\s*([\d.]+)", _text(self.proc / "cpuinfo") or "", re.M)]
        meminfo = [x for x in (_text(self.proc / "meminfo") or "").splitlines() if x.startswith(("MemTotal:", "MemAvailable:"))]
        temp = self._temp()
        lines = [f"stat1 {prev}", f"stat2 {self.stat}", *meminfo,
                 f"load {' '.join((_text(self.proc / 'loadavg') or '').split()[:3])}",
                 f"uptime {((_text(self.proc / 'uptime') or '').split() or [''])[0]}",
                 f"mhz {sum(mhz) / len(mhz)} {max(mhz)}" if mhz else "",
                 f"games {sum(1 for c, _ in procs.values() if c == GAME)}",
                 f"temp Tctl: {temp}" if temp is not None else "", f"profile {self._profile(now) or ''}"]
        row = parse_stats("\n".join(lines), now, self.name)
        row["local"] = True
        row.update(self._runs(procs, now))
        if self.gpu:
            row["gpu"] = g = self.gpu.read(now)
            for p in g["procs"]:
                p["name"] = self._label(p["pid"], procs)
        return row

    def _roots(self, procs: dict[int, tuple[str, int]]) -> dict[int, tuple[str, str]]:
        """The learners and evaluation workers running here: pid -> (run, "learner" or "evaluation"). Only python
        processes' command lines are read."""
        out = {}
        for pid, (comm, _) in procs.items():
            if comm.startswith("python"):
                raw = _text(f"{self.proc}/{pid}/cmdline")
                got = run_of(raw.split("\0")) if raw else None
                if got:
                    out[pid] = got
        self.roots = out
        return out

    def _label(self, pid: int, procs: dict[int, tuple[str, int]]) -> str:
        """What a GPU process is: "<run> learner", "<run> evaluation" (its processes too), or its comm."""
        roots, q = self.roots, pid
        for _ in range(6):
            if q in roots:
                return f"{roots[q][0]} {roots[q][1]}"
            if q not in procs:
                break
            q = procs[q][1]
        return procs[pid][0] if pid in procs else str(pid)

    def _runs(self, procs: dict[int, tuple[str, int]], now: float) -> dict:
        """The memory of the learners, of the other processes below them and the evaluation workers (actors, Xvfb,
        launchers, evaluation), and of the games (private resident memory, GiB); the live runs."""
        roots = self._roots(procs)
        kids: dict[int, list[int]] = {}
        for pid, (_, ppid) in procs.items():
            kids.setdefault(ppid, []).append(pid)
        below, todo = set(), list(roots)
        while todo:
            q = todo.pop()
            if q not in below:
                below.add(q)
                todo += kids.get(q, [])
        learners = {p for p, (_, kind) in roots.items() if kind == "learner"}
        games = {p for p, (c, _) in procs.items() if c == GAME}
        out: dict = {"mem_learner": round(sum(rss_anon(self.proc, p) for p in learners), 2),
                     "mem_others": round(sum(rss_anon(self.proc, p) for p in below - learners - games), 2),
                     "mem_games": round(sum(rss_anon(self.proc, p) for p in games), 2)}
        live = []
        for pid, (run, kind) in sorted(roots.items(), key=lambda x: (x[1][1] != "learner", x[1][0])):
            if any(r["run"] == run and r["kind"] == kind for r in live):
                continue
            r: dict = {"run": run, "kind": kind}
            if kind == "learner":
                r.update(run_speed(self.runs / run / "train.jsonl", now))
                cfg = self._config(run)
                r["bf16"] = bool(cfg["bf16"]) if "bf16" in cfg else None
                r["infer_precision"] = cfg.get("infer_precision")
            live.append(r)
        out["live"] = live
        return out


def trim(path: Path, keep_days: float) -> None:
    """Rows older than keep_days dropped (the file written again through a temporary one)."""
    if not path.exists():
        return
    cut, kept = time.time() - keep_days * 86400, []
    for line in path.read_text().splitlines():
        try:
            if json.loads(line).get("time", 0) >= cut:
                kept.append(line)
        except json.JSONDecodeError:
            pass
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(x + "\n" for x in kept))
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", action="append", default=[],
                    help="ssh destination user@name (repeatable); its rows are called name, or name=user@host to call it otherwise")
    ap.add_argument("--local", default=socket.gethostname(),
                    help="this machine's rows' name (default: its hostname, as the dashboard calls it); read here, no ssh")
    ap.add_argument("--no-local", action="store_true", help="not this machine")
    ap.add_argument("--runs", type=Path, default=RUNS,
                    help="the runs directory: rows go to <runs>/hosts, the live runs' train.jsonl are read there")
    ap.add_argument("--every", type=float, default=30.0, help="seconds between looks")
    ap.add_argument("--keep", type=float, default=7.0, help="days of rows kept")
    ap.add_argument("--once", action="store_true", help="one look at each host, printed; nothing written")
    a = ap.parse_args()
    hosts = []
    for h in a.host:
        name, _, dest = h.partition("=")
        hosts.append((dest, name) if dest else (h, h.split("@")[-1]))
    local = None if a.no_local else Local(a.local, a.runs, gpu=Gpu.open())
    if not hosts and not local:
        ap.error("nothing to look at: give --host, or leave out --no-local")
    looks = [(n, lambda h=h, n=n: look(h, n)) for h, n in hosts] + ([(local.name, local.look)] if local else [])
    if a.once:
        for _, f in looks:
            print(json.dumps(f()))
        return 0
    out = a.runs / "hosts"
    out.mkdir(parents=True, exist_ok=True)
    last_trim = 0.0
    log("looking at", ", ".join(n for n, _ in looks), f"every {a.every:.0f} s",
        f"(this machine's GPU: {local.gpu.name})" if local and local.gpu else "")
    with ThreadPoolExecutor(len(looks)) as pool:
        try:
            while True:
                t0 = time.time()
                if t0 - last_trim > 6 * 3600:
                    for n, _ in looks:
                        trim(out / f"{n}.jsonl", a.keep)
                    last_trim = t0
                for row in pool.map(lambda nf: nf[1](), looks):  # (all at once: a host that hangs holds only its own)
                    with open(out / f"{row['host']}.jsonl", "a") as f:
                        f.write(json.dumps(row, separators=(",", ":")) + "\n")
                end = max(t0 + a.every, time.time() + 1.0)  # (meanwhile this machine's GPU samples gathered: NVML keeps ~14 s)
                while (left := end - time.time()) > 0:
                    time.sleep(min(left, GATHER))
                    if local:
                        local.gather()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
