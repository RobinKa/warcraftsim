#!/usr/bin/env python3
"""The routine checks of a live run, a burst and a run's budget, each one short command with compact output
(2026-11-27: the checks the ~50-minute status reads had run by hand).

    scripts/ops.py status fgself-267 [--since HH:MM] [--vs runs/fullgame/soups/soup-2026-11-26.pt ...]
    scripts/ops.py burst burst-267 [--no-ratings]
    scripts/ops.py budget fgself-267 --at 54e6 --burst burst-267b --avg-from 27e6 [--eval CKPT ...] [--current ema] \\
        [--dry-run]          # (run it in the background: it waits for the checkpoint, stops the learner and the box,
                             # averages, prepares the burst and starts its evaluation worker)
    scripts/ops.py fork-check fgself-268

status: the machine (disk, RAM, GPU memory, kernel Xid/MCE/OOM lines this boot), the learner and the box's service,
the box's refusals of wrong code in the run's log, train.jsonl's last rows (steps, age, SPS, the losses' means), the
training games against the real AI without a target by quarter hour since the run's last launch (a win 1, a tie
half), the development in training games past 10 minutes, and scripts/head_to_head.py's lines for each --vs.
burst: the evaluation worker, each policy's untaxed insane score as a race-balanced mean +- 1.96 SE (each race
weighed alike), the handicap rungs, the current policy's head-to-head with each, long games against insane, the
dashboard's ratings. budget: what the end of a run's step budget takes, in order (--dry-run prints the commands).
fork-check: after a fork or a resume, its log's notes, its launch's code and the command that points the box at it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TIER2 = {"human": "hkee", "orc": "ostr", "undead": "unp1", "nightelf": "etoa"}
TIER3 = {"human": "hcas", "orc": "ofrt", "undead": "unp2", "nightelf": "etoe"}
RACE_ABBR = {"human": "hu", "nightelf": "ni", "orc": "or", "undead": "un"}
INSANE = "script:ai-insane (real)"  # (the untaxed insane AI in the real game: the ratings' anchor)
BOX = "tora-ms-a2@tora-ms-a2"
BOX_SERVICE = "warcraftsim-remoteactor.service"
RATINGS_URL = "http://100.120.68.79:8765/api/ratings"
LAUNCH_MARK = "map frame:"  # (the first line each launch of a learner writes to runs/<run>.log)
CKPT = re.compile(r"\d{16}\.pt")
FORK_LINES = re.compile(r"Adam's state|Traceback|error: argument|act-ema:|note:")


# ---------------------------------------------------------------------------------------------------- rows and scores

def _row_time(line: bytes) -> float | None:
    m = re.search(rb'"time": ?([0-9.eE+-]+)', line)
    try:
        return float(m.group(1)) if m else None
    except ValueError:
        return None


def _seek(fh, size: int, t: float, margin: float = 600.0) -> int:
    """An offset in a time-ordered jsonl file at or before its first row of time >= t - margin (binary search on
    the rows' "time"; the rows are written as games end, a few seconds out of order at most)."""
    lo, hi = 0, size
    while hi - lo > 1 << 16:
        mid = (lo + hi) // 2
        fh.seek(mid)
        fh.readline()  # (the rest of the line mid fell in)
        found = None
        for _ in range(8):  # (a NUL block or a half line: the next few)
            line = fh.readline()
            if not line:
                break
            found = _row_time(line)
            if found is not None:
                break
        if found is not None and found < t - margin:
            lo = mid
        else:
            hi = mid
    return lo


def rows(path: Path, since: float | None = None):
    """The rows of an episodes jsonl file, streamed: NUL-padded, blank and broken lines skipped; with `since`, from
    a binary search's offset on (rows before it may still come: filter on "time")."""
    path = Path(path)
    if not path.exists():
        return
    with path.open("rb") as fh:
        if since:
            fh.seek(_seek(fh, path.stat().st_size, since))
            if fh.tell():
                fh.readline()
        for line in fh:
            s = line.strip(b"\x00 \n\r")
            if not s:
                continue
            try:
                yield json.loads(s)
            except ValueError:
                continue


def tail_rows(path: Path, n: int) -> list[dict]:
    """The last n complete rows of a jsonl file (read from its end)."""
    size, chunk = path.stat().st_size, 1 << 20
    while True:
        with path.open("rb") as fh:
            fh.seek(max(0, size - chunk))
            lines = fh.read().split(b"\n")
        if size > chunk:
            lines = lines[1:]  # (a partial first line)
        out = []
        for line in lines:
            s = line.strip(b"\x00 \r")
            if s:
                try:
                    out.append(json.loads(s))
                except ValueError:
                    pass
        if len(out) >= n or chunk >= size:
            return out[-n:]
        chunk *= 4


def score(outcome: float) -> float:
    """A win 1, a tie half, a loss 0."""
    return 1.0 if outcome > 0 else 0.5 if outcome == 0 else 0.0


def tier(prod: dict, race: str) -> int:
    built = (prod or {}).get("built") or {}
    if built.get(TIER3.get(race, "?"), 0) > 0:
        return 3
    return 2 if built.get(TIER2.get(race, "?"), 0) > 0 else 1


def development(rs: list[dict]) -> dict | None:
    """Means over the rows: heroes, tier, research (the sum of prod.research), workers, the share expanded."""
    if not rs:
        return None
    n = len(rs)
    p = [r.get("prod") or {} for r in rs]
    return {"n": n, "heroes": sum(x.get("heroes", 0) for x in p) / n,
            "tier": sum(tier(x, r.get("race", "")) for x, r in zip(p, rs)) / n,
            "research": sum(sum((x.get("research") or {}).values()) for x in p) / n,
            "workers": sum(x.get("workers", 0) for x in p) / n,
            "expanded": sum((x.get("expansions") or 0) > 0 for x in p) / n,
            "won": sum(score(r["outcome"]) for r in rs) / n}


def balanced(rs: list[dict]) -> tuple[float, float, dict]:
    """The race-balanced score: the mean of the per-race means, its 1.96 SE half-width
    (sqrt(sum_r p_r (1 - p_r) / n_r) / R) and {race: (p, n)}. nan without rows."""
    by = defaultdict(list)
    for r in rs:
        by[r.get("race", "?")].append(score(r["outcome"]))
    per = {k: (sum(v) / len(v), len(v)) for k, v in by.items()}
    if not per:
        return math.nan, math.nan, {}
    R = len(per)
    mean = sum(p for p, _ in per.values()) / R
    se = math.sqrt(sum(p * (1 - p) / n for p, n in per.values())) / R
    return mean, 1.96 * se, per


def mean_ci(scores: list[float]) -> tuple[float, float]:
    """A plain mean and its 1.96 SE half-width."""
    n = len(scores)
    if not n:
        return math.nan, math.nan
    m = sum(scores) / n
    return m, 1.96 * math.sqrt(sum((s - m) ** 2 for s in scores) / n / n)


def races_str(per: dict) -> str:
    return " ".join(f"{RACE_ABBR.get(k, k[:2])} {p:.2f}({n})" for k, (p, n) in sorted(per.items(),
                                                                                    key=lambda kv: RACE_ABBR.get(kv[0], kv[0])))


def training_ai(r: dict) -> bool:
    """A training game against the real AI without a target (status's quarter hours)."""
    return not r.get("eval") and not r.get("z") and "(real)" in str(r.get("opponent", ""))


def training_dev(r: dict) -> bool:
    """A training game without a target past 10 minutes (status's development)."""
    return not r.get("eval") and not r.get("z") and r.get("game_time", 0) >= 600


def quarter_bins(rs: list[dict], since: float, max_bins: int = 16) -> tuple[int, list[tuple[int, float, int]]]:
    """(the bins' width in minutes, [(minutes since `since`, score, games)] of the bins with games): 15-minute bins,
    widened by multiples of 15 to at most max_bins over the rows' span."""
    if not rs:
        return 15, []
    span = max(r["time"] for r in rs) - since
    width = 15 * max(1, math.ceil(span / (15 * 60) / max_bins))
    by = defaultdict(list)
    for r in rs:
        by[int((r["time"] - since) // (width * 60)) * width].append(score(r["outcome"]))
    return width, [(k, sum(v) / len(v), len(v)) for k, v in sorted(by.items())]


# ------------------------------------------------------------------------------------------------- names and paths

def runs_rel(path: str) -> str:
    """'/home/x/warcraftsim/runs/fgself-252/averaged/swa-17-27.pt' -> 'runs/fgself-252/averaged/swa-17-27.pt'."""
    parts = Path(path).parts
    if "runs" in parts:
        i = len(parts) - 1 - parts[::-1].index("runs")
        return "/".join(parts[i:])
    return str(path)


def vs_tag(path: str) -> str:
    """The head-to-head key's part selfplay writes for a checkpoint: 'vs fgself-252/swa-17-27.pt' (the folder under
    runs/ and the file's name)."""
    parts = runs_rel(path).split("/")
    return f"{parts[1]}/{parts[-1]}" if len(parts) > 2 and parts[0] == "runs" else Path(path).name


def burst_policy(key: str, current: str, evals: list[str]) -> tuple[str, str] | None:
    """A burst row's eval_key -> (the policy's path, 'ai' or 'h2h'): 'current' and 'current @-25%' are burst.json's
    current; a path (relative or absolute, with or without ' @-25%') an eval; 'vs <run>/<file>.pt' the current
    policy's head-to-head game against that eval. None: none of the burst's."""
    k = re.sub(r" @-\d+%$", "", key or "current")
    if k == "current":
        return current, "ai"
    if k.startswith("vs "):
        tag = k[3:]
        return next(((e, "h2h") for e in evals if vs_tag(e) == tag), None)
    rel = runs_rel(k)
    return next(((e, "ai") for e in evals if runs_rel(e) == rel), None)


def rating_name(path: str) -> str:
    """The ratings' name of a checkpoint ('fgself-252 avg swa-17-27', 'soup avg soup-2026-11-26', 'fgself-252/123')."""
    from warcraftsim.dashboard.ratings import checkpoint_name
    return checkpoint_name(str(path)) or Path(path).stem


def short_name(path: str) -> str:
    """rating_name without the leading zeros of a step count ('ema-0000000026967703' -> 'ema-26967703')."""
    return re.sub(r"(?<=[-/])0+(?=\d)", "", rating_name(path))


def parse_steps(s: str) -> int:
    """'54e6', '54M', '54000000' -> 54000000."""
    s = s.strip()
    if s[-1:] in "mM":
        return int(round(float(s[:-1]) * 1e6))
    return int(round(float(s)))


def parse_since(s: str, now: float | None = None) -> float:
    """'HH:MM' (today, or yesterday if that is still ahead), 'YYYY-MM-DD HH:MM' or a unix time -> a unix time."""
    now = time.time() if now is None else now
    try:
        return float(s)
    except ValueError:
        pass
    if re.fullmatch(r"\d{1,2}:\d{2}", s):
        h, m = map(int, s.split(":"))
        lt = time.localtime(now)
        t = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, h, m, 0, 0, 0, -1))
        return t - 86400 if t > now else t
    return time.mktime(time.strptime(s, "%Y-%m-%d %H:%M"))


def default_since(meta: dict) -> float:
    """The run's last launch: run.json "created", or its latest "resumes" entry if newer."""
    t = float(meta.get("created") or 0.0)
    for r in meta.get("resumes") or []:
        t = max(t, float(r.get("time") or 0.0))
    return t


def hm(t: float) -> str:
    return time.strftime("%m-%d %H:%M", time.localtime(t))


# -------------------------------------------------------------------------------------------------------- processes

def python_procs(pattern: str) -> list[tuple[int, int, str]]:
    """(pid, seconds up, args) of the python processes whose args match `pattern` (a bash -c wrapper is no python),
    without those whose parent matches too (a forked child keeps its parent's args)."""
    out = subprocess.run(["ps", "-eo", "pid=,ppid=,etimes=,args="], capture_output=True, text=True).stdout
    rx = re.compile(pattern)
    found = {}
    for line in out.splitlines():
        f = line.split(None, 3)
        if len(f) < 4 or "python" not in Path(f[3].split(" ")[0]).name:
            continue
        if rx.search(f[3]) and int(f[0]) != os.getpid():
            found[int(f[0])] = (int(f[1]), int(f[2]), f[3])
    return [(pid, up, args) for pid, (ppid, up, args) in found.items() if ppid not in found]


def learner(run: str) -> list[tuple[int, int, str]]:
    return python_procs(rf"fullgame\.selfplay --name {re.escape(run)}(?:\s|$)")


def evalworker(burst: str) -> list[tuple[int, int, str]]:
    return python_procs(rf"fullgame\.evalworker\b.*--run {re.escape(burst)}(?:\s|$)")


def uptime(sec: int) -> str:
    h, m = divmod(sec // 60, 60)
    return f"{h}:{m:02d}h" if h else f"{m}m"


def sh(cmd: list[str], timeout: float = 15.0) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except (subprocess.TimeoutExpired, OSError) as e:
        return f"? ({type(e).__name__})"


def box_state() -> str:
    try:
        p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", BOX,
                            f"systemctl --user is-active {BOX_SERVICE}"], capture_output=True, text=True, timeout=12)
        return p.stdout.strip() or p.stderr.strip()[:40] or "?"
    except subprocess.TimeoutExpired:
        return "ssh timeout"


def ram() -> str:
    for line in sh(["free", "-g"]).splitlines():
        if line.startswith("Mem:"):
            f = line.split()
            return f"RAM used {f[2]}G avail {f[6]}G"
    return "RAM ?"


def gpu() -> str:
    out = sh(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"])
    try:
        used, total = (int(x) for x in out.splitlines()[0].split(","))
        return f"GPU {used}/{total} MiB"
    except (ValueError, IndexError):
        return f"GPU {out[:30]}"


def kernel() -> list[str]:
    out = sh(["bash", "-c", "journalctl -k -b 0 2>/dev/null | grep -iE 'NVRM: Xid|mce: \\[hardware error|out of memory'"
                            " | tail -2"])
    return [x[:140] for x in out.splitlines() if x.strip()]


def log_refused(log: Path) -> tuple[int, int, str]:
    """The box's refusals of wrong code in a run's log: (all, since its last launch, the last such line)."""
    total = this = 0
    last = ""
    if log.exists():
        with log.open(errors="replace") as fh:
            for line in fh:
                if line.startswith(LAUNCH_MARK):
                    this = 0
                elif "refused: its code" in line:
                    total, this, last = total + 1, this + 1, line.strip()
    return total, this, last


# ----------------------------------------------------------------------------------------------------------- status

LOSS_KEYS = (("loss/ref_kl", "ref_kl"), ("loss/value", "value"), ("loss/clipfrac", "clipfrac"), ("loss/kl", "kl"),
             ("act_ema/distance", "ema_dist"))


def train_lines(path: Path, now: float) -> list[str]:
    if not path.exists():
        return [f"train: no {path.name}"]
    tail = tail_rows(path, 100)
    if not tail:
        return ["train: no rows"]
    r = tail[-1]
    age = max(0.0, now - r.get("time", 0))
    sps = [x["SPS"] for x in tail[-30:] if "SPS" in x]
    out = [f"train {r.get('agent_steps', 0) / 1e6:.2f}M steps | last row {age:.0f} s ago"
           f"{' STALE' if age > 120 else ''} | SPS {sum(sps) / max(len(sps), 1):.0f} (last {len(sps)})"]
    means = []
    for k, label in LOSS_KEYS:
        v = [x[k] for x in tail if k in x]
        if v:
            means.append(f"{label} {sum(v) / len(v):.3g}")
    league = [f"{k[7:]} {v:.3g}" if isinstance(v, float) else f"{k[7:]} {v}" for k, v in r.items()
              if k.startswith(("league/exploiter", "league/lexploiter"))]
    out.append(f"last {len(tail)}: " + " ".join(means) + (" | " + " ".join(league) if league else ""))
    return out


def h2h_lines(root: Path, run: str, ckpt: str) -> str:
    """scripts/head_to_head.py RUN --vs CKPT: its head-to-head line for CKPT and its races-weighed-alike line."""
    try:
        p = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "head_to_head.py"), run,
                            "--vs", ckpt], capture_output=True, text=True, timeout=300, cwd=root)
    except subprocess.TimeoutExpired:
        return f"--vs {ckpt}: head_to_head timed out"
    lines = p.stdout.splitlines()
    tag = ("vs " + vs_tag(ckpt))[:45]
    vs = [x for x in lines if x.startswith(tag)] or [x for x in lines if x.startswith("vs ")]
    rest = [x.strip() for x in lines if "races weighed alike" in x or x.startswith("--vs ")]
    return "  " + " | ".join([" ".join(x.split()) for x in vs] + rest) if vs or rest else f"--vs {ckpt}: no output"


def cmd_status(a) -> int:
    root, run = a.root, a.run
    rd = root / "runs" / run
    meta = json.loads((rd / "run.json").read_text())
    now = time.time()
    since = parse_since(a.since, now) if a.since else default_since(meta)
    with ThreadPoolExecutor(4 + len(a.vs)) as ex:  # (the slow ones beside the files' reads)
        f_box, f_ram, f_gpu, f_kern = ex.submit(box_state), ex.submit(ram), ex.submit(gpu), ex.submit(kernel)
        f_h2h = [ex.submit(h2h_lines, root, run, c) for c in a.vs]
        disk = shutil.disk_usage(root)
        L = learner(run)
        total, this, _ = log_refused(root / "runs" / f"{run}.log")
        ai, dev = [], []
        for r in rows(rd / "episodes.jsonl", since):
            if r.get("time", 0) < since:
                continue
            if training_ai(r):
                ai.append(r)
            if training_dev(r):
                dev.append(r)
        tl = train_lines(rd / "train.jsonl", now)
        print(f"{run} @ {time.strftime('%H:%M:%S')} | disk {disk.free / 2**30:.0f}G free | {f_ram.result()} | "
              f"{f_gpu.result()}")
        lrn = f"learner PID {L[0][0]} up {uptime(L[0][1])}" + (f" (+{len(L) - 1} more)" if len(L) > 1 else "") \
            if L else "learner GONE"
        kern = f_kern.result()
        print(f"{lrn} | box {f_box.result()} | refused-code {total} (this launch {this}) | kernel "
              + ("clean" if not kern else f"{len(kern)} line(s):"))
        for x in kern:
            print("  " + x)
        for x in tl:
            print(x)
        width, bins = quarter_bins(ai, since)
        cells = [f"{m}:{s:.2f}({n})" for m, s, n in bins]
        print(f"vs AI (real, no target) since {hm(since)}, {width}-min bins:" + ("" if cells else " no games yet"))
        for i in range(0, len(cells), 8):
            print("  " + " ".join(cells[i:i + 8]))
        if ai:
            m, ci, per = balanced(ai)
            print(f"  all {sum(score(r['outcome']) for r in ai) / len(ai):.3f}({len(ai)}) race-balanced "
                  f"{m:.3f}±{ci:.3f}: {races_str(per)}")
        d = development(dev)
        print(f"past 10 min (no target, n={d['n']}): heroes {d['heroes']:.2f} tier {d['tier']:.2f} research "
              f"{d['research']:.2f} workers {d['workers']:.1f} expanded {100 * d['expanded']:.0f}%"
              if d else "past 10 min (no target): no games")
        for c, f in zip(a.vs, f_h2h):
            print(f"--vs {short_name(c)}:")
            print(f.result())
    return 0


# ------------------------------------------------------------------------------------------------------------ burst

def fetch_ratings(url: str, timeout: float = 120.0) -> dict | str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return {p["name"]: p for p in json.load(resp).get("players", [])}
    except (OSError, ValueError) as e:
        return f"ratings: {type(e).__name__}: {str(e)[:80]}"


def burst_table(rs: list[dict], current: str, evals: list[str]) -> tuple[dict, list[str]]:
    """{policy path: {"ai": untaxed insane rows, "rungs": {handicap: rows}, "h2h": the current policy's head-to-head
    rows against it}} and the eval_keys none of the burst's policies have."""
    t = {p: {"ai": [], "rungs": defaultdict(list), "h2h": []} for p in [current, *evals]}
    unmapped = set()
    for r in rs:
        key = r.get("eval_key") or "current"
        got = burst_policy(key, current, evals)
        if got is None:
            unmapped.add(key)
            continue
        path, kind = got
        if kind == "h2h":
            t[path]["h2h"].append(r)
        elif r.get("handicap"):
            t[path]["rungs"][r["handicap"]].append(r)
        elif r.get("opponent") == INSANE:
            t[path]["ai"].append(r)
    return t, sorted(unmapped)


def cmd_burst(a) -> int:
    root, name = a.root, a.burst
    bd = root / "runs" / name
    b = json.loads((bd / "burst.json").read_text())
    current, evals = b["current"], list(b.get("evals") or [])
    with ThreadPoolExecutor(1) as ex:
        f_rat = ex.submit(fetch_ratings, a.ratings_url) if not a.no_ratings else None
        rs = [r for f in ("episodes-remote.jsonl", "episodes.jsonl") for r in rows(bd / f)]
        t, unmapped = burst_table(rs, current, evals)
        W = evalworker(name)
        if W:
            state = f"worker RUNNING PID {W[0][0]} up {uptime(W[0][1])}"
        else:
            try:
                w = json.loads((bd / "remote-worker.json").read_text())
                state = f"worker gone (stopped {hm(w.get('stopped') or 0)}, {w.get('episodes')} episodes)"
            except (OSError, ValueError):
                state = "worker gone"
        print(f"{name} | {state} | rows {len(rs)}" + (f" | unmapped keys: {', '.join(unmapped)}" if unmapped else ""))
        rungs = sorted({h for v in t.values() for h in v["rungs"]})
        ratings = f_rat.result() if f_rat else None
        nw = max(len(short_name(p)) for p in t) + 1
        head = f"{'policy (* current)':{nw}} insane       " + " ".join(f"{x:8}" for x in ("hu", "ni", "or", "un")) \
            + "".join(f" -{h}%".ljust(9) for h in rungs) + " h2h(cur vs)       rating"
        print(head)
        for p, v in t.items():
            m, ci, per = balanced(v["ai"])
            race = " ".join(f"{per[r][0]:.2f}({per[r][1]})".ljust(8) if r in per else f"{'-':8}"
                            for r in ("human", "nightelf", "orc", "undead"))
            rung = "".join(f" {sum(score(r['outcome']) for r in v['rungs'][h]) / len(v['rungs'][h]):.2f}"
                           f"({len(v['rungs'][h])})".ljust(9) if v["rungs"].get(h) else f" {'-':8}" for h in rungs)
            hs = [score(r["outcome"]) for r in v["h2h"]]
            if hs:
                hm_, hci = mean_ci(hs)
                h2h = f"{hm_:.3f}±{hci:.3f}({len(hs)})"
            else:
                h2h = "-"
            rat = "-"
            if isinstance(ratings, dict) and (q := ratings.get(rating_name(p))):
                rat = f"{q['rating']:.0f}±{q['se']:.0f} ({q['games']})"
            star = "*" if p == current else " "
            print(f"{star}{short_name(p):{nw - 1}} {m:.3f}±{ci:.3f}  {race}{rung} {h2h:17} {rat}")
        print("long games (>15 min) vs untaxed insane: n tier research workers won")
        for p, v in t.items():
            d = development([r for r in v["ai"] if r.get("game_time", 0) > 900])
            print(f" {short_name(p):{nw - 1}} " + (f"n {d['n']:3} tier {d['tier']:.2f} research {d['research']:.2f} "
                                                   f"workers {d['workers']:.1f} won {d['won']:.2f}" if d else "none"))
        if isinstance(ratings, str):
            print(ratings)
        elif isinstance(ratings, dict):
            rung_s = " ".join(f"{k[len('insane AI vs '):]} {q['rating']:.0f}±{q['se']:.0f} ({q['games']})"
                              for k in ("insane AI vs -25%", "insane AI vs -50%") if (q := ratings.get(k)))
            top = sorted((q for q in ratings.values() if not q["name"].startswith("insane AI")),
                         key=lambda q: -q["rating"])[:3]
            print(f"ratings: rungs {rung_s} | top " + ", ".join(f"{q['rating']:.0f}±{q['se']:.0f} {q['name']}"
                                                              for q in top))
    return 0


# ----------------------------------------------------------------------------------------------------------- budget

def checkpoints(run_dir: Path) -> list[int]:
    d = run_dir / "checkpoints"
    return sorted(int(p.name[:16]) for p in d.iterdir() if CKPT.fullmatch(p.name)) if d.exists() else []


def swa_path(run: str, avg_from: int, at: int) -> str:
    return f"runs/{run}/averaged/swa-{round(avg_from / 1e6)}-{round(at / 1e6)}.pt"


def budget_plan(run: str, burst: str, at: int, avg_from: int, ckpt_steps: int, evals: list[str], current: str = "swa",
                games: int = 64, processes: int = 16, hours: float = 0.75, py: str = ".venv/bin/python",
                box: str = BOX) -> dict:
    """The budget's commands (run from the repository's root), in order: steps [(label, argv, fatal)], the evaluation
    worker's argv and its log."""
    swa = swa_path(run, avg_from, at)
    if current == "ema":
        cur, ev = f"runs/{run}/averaged/ema-{ckpt_steps:016d}.pt", [swa, *[e for e in evals if e != swa]]
    else:
        cur, ev = swa, list(evals)
    steps = [("stop the learner", [py, "scripts/selfplay_restart.py", run, "--stop"], True),
             ("stop the box's actors", ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", box,
                                        f"systemctl --user stop {BOX_SERVICE}"], False),
             ("average", ["nice", "-n", "5", py, "scripts/average_checkpoints.py", swa, "--run", run,
                          "--from", str(avg_from), "--to", str(ckpt_steps)], True),
             ("prepare the burst", [py, "scripts/burst_prep.py", burst, run, cur, *ev], True)]
    worker = ["setsid", "nohup", py, "-m", "warcraftsim.fullgame.evalworker", "--run", burst, "--games", str(games),
              "--processes", str(processes), "--host", "tora-burst", "--hours", f"{hours:g}"]
    return {"swa": swa, "current": cur, "evals": ev, "steps": steps, "worker": worker, "log": f"runs/{burst}.log"}


def budget_problems(root: Path, run: str, burst: str, plan: dict, avg_from: int, at: int) -> list[str]:
    p = []
    if not (root / "runs" / run / "run.json").exists():
        p.append(f"runs/{run}/run.json missing")
    if (root / "runs" / burst).exists():
        p.append(f"runs/{burst} exists (burst_prep makes it)")
    if (root / plan["swa"]).exists():
        p.append(f"{plan['swa']} exists (average_checkpoints will not overwrite it)")
    for e in plan["evals"]:
        if e != plan["swa"] and not (root / e).exists():
            p.append(f"eval {e} missing")
    if not plan["evals"]:
        p.append("no evals: burst_prep needs at least one (--eval, or --current ema)")
    if avg_from >= at:
        p.append("--avg-from must be below --at")
    return p


def log_tail(log: Path, n: int = 15) -> list[str]:
    if not log.exists():
        return [f"(no {log})"]
    with log.open("rb") as fh:
        fh.seek(max(0, log.stat().st_size - 64 * 1024))
        return [x[:200] for x in fh.read().decode(errors="replace").splitlines()[-n:]]


def stamp() -> str:
    return time.strftime("%H:%M:%S")


def cmd_budget(a) -> int:
    root, run = a.root, a.run
    at, avg_from = parse_steps(a.at), parse_steps(a.avg_from)
    rd = root / "runs" / run
    found = [s for s in checkpoints(rd) if s >= at]
    plan = budget_plan(run, a.burst, at, avg_from, found[0] if found else at, a.eval, a.current, a.games, a.processes,
                       a.hours)
    problems = budget_problems(root, run, a.burst, plan, avg_from, at)
    if a.dry_run:
        print(f"cd {root}")
        print(f"# wait (every {a.poll:g} s) for runs/{run}/checkpoints/<16 digits>.pt >= {at}"
              + (f" and its averaged/ema-<steps>.pt" if a.current == "ema" else "") + " or the learner gone; "
              + (f"found {found[0]:016d}.pt" if found else f"assumed {at:016d}.pt below"))
        for label, argv, _ in plan["steps"]:
            print(f"# {label}\n{shlex.join(argv)}")
        print(f"# the evaluation worker, detached\n{shlex.join(plan['worker'])} > {plan['log']} 2>&1 < /dev/null &")
        for x in problems:
            print("PROBLEM:", x)
        return 1 if problems else 0
    if problems:
        for x in problems:
            print("PROBLEM:", x, flush=True)
        return 2
    print(f"{stamp()} waiting for {run} >= {at} steps (avg from {avg_from}, burst {a.burst}, current {a.current})",
          flush=True)
    while True:
        found = [s for s in checkpoints(rd) if s >= at]
        ema_ok = a.current != "ema" or (found and (rd / "averaged" / f"ema-{found[0]:016d}.pt").exists())
        if found and ema_ok:
            break
        if not learner(run):
            time.sleep(5)  # (a checkpoint renamed just as the learner left)
            found = [s for s in checkpoints(rd) if s >= at]
            if not found or not (a.current != "ema" or (rd / "averaged" / f"ema-{found[0]:016d}.pt").exists()):
                print(f"{stamp()} LEARNER GONE before {at} steps; the log's last lines:", flush=True)
                for x in log_tail(root / "runs" / f"{run}.log"):
                    print("  " + x)
                return 1
            break
        time.sleep(a.poll)
    ck = found[0]
    plan = budget_plan(run, a.burst, at, avg_from, ck, a.eval, a.current, a.games, a.processes, a.hours)
    print(f"{stamp()} checkpoint {ck:016d}.pt", flush=True)
    problems = budget_problems(root, run, a.burst, plan, avg_from, at)  # (again: hours may have passed; nothing stopped)
    if problems:
        for x in problems:
            print("PROBLEM:", x, flush=True)
        return 2
    for label, argv, fatal in plan["steps"]:
        print(f"{stamp()} {label}: {shlex.join(argv)}", flush=True)
        try:
            p = subprocess.run(argv, cwd=root, timeout=1800, capture_output=True, text=True)
            rc, out = p.returncode, (p.stdout + p.stderr).strip()
        except subprocess.TimeoutExpired:
            rc, out = -1, "timed out"
        for x in out.splitlines()[-4:]:
            print("  " + x[:200], flush=True)
        if rc != 0:
            print(f"{stamp()} {label} FAILED (exit {rc})" + ("" if fatal else ": going on"), flush=True)
            if fatal:
                return 1
    log = root / plan["log"]
    with log.open("ab") as fh:  # (detached: its own session, no terminal, survives this script)
        subprocess.Popen(plan["worker"], cwd=root, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         close_fds=True)
    time.sleep(5)
    W = evalworker(a.burst)
    if W:
        print(f"{stamp()} evalworker PID {W[0][0]} ({a.burst}, log {plan['log']})", flush=True)
        return 0
    print(f"{stamp()} the evalworker is NOT running; its log:", flush=True)
    for x in log_tail(log):
        print("  " + x)
    return 1


# ------------------------------------------------------------------------------------------------------- fork-check

def cmd_fork_check(a) -> int:
    root, run = a.root, a.run
    log = root / "runs" / f"{run}.log"
    hits, launches = [], 0  # (the log is appended to by each launch: the notes since the last one's first line)
    if log.exists():
        with log.open(errors="replace") as fh:
            for x in fh:
                if x.startswith(LAUNCH_MARK):
                    hits, launches = [], launches + 1
                elif FORK_LINES.search(x):
                    hits.append(x.rstrip()[:220])
    print(f"{log.name}: launch {launches}, its notes:" if launches else f"{log.name}: no launch marker yet, notes:")
    for x in hits[-a.lines:] or ["(none)"]:
        print("  " + x)
    total, this, last = log_refused(log)
    print(f"refused-code {total} (this launch {this})" + (f": {last[:160]}" if this else ""))
    L = learner(run)
    print(f"learner PID {L[0][0]} up {uptime(L[0][1])}" if L else "learner GONE")
    meta = json.loads((root / "runs" / run / "run.json").read_text())
    code = (meta.get("launch") or {}).get("code") or ""
    commit = Path(code).name if code else ""
    if not commit:
        print("run.json has no launch code")
        return 1
    subject = ((meta.get("launch") or {}).get("git") or {}).get("subject", "")
    print(f"launch code {commit}: {subject[:100]}")
    print(f"box: RUN={run} scripts/box_games.sh {commit} 8 8")
    return 0


# ------------------------------------------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--root", type=Path, default=ROOT, help="the repository whose runs/ to read (default: this one)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status", help="a live run's health and training read")
    s.add_argument("run")
    s.add_argument("--since", help="HH:MM, 'YYYY-MM-DD HH:MM' or unix time (default: the run's last launch)")
    s.add_argument("--vs", nargs="*", default=[], help="checkpoints: head_to_head.py's lines for each")
    s.set_defaults(fn=cmd_status)
    b = sub.add_parser("burst", help="an evaluation burst's read")
    b.add_argument("burst")
    b.add_argument("--no-ratings", action="store_true", help="skip the dashboard's ratings")
    b.add_argument("--ratings-url", default=RATINGS_URL)
    b.set_defaults(fn=cmd_burst)
    g = sub.add_parser("budget", help="the end of a run's budget: wait, stop, average, burst (run in the background)")
    g.add_argument("run")
    g.add_argument("--at", required=True, help="the budget in agent steps (54e6, 54M)")
    g.add_argument("--burst", required=True, help="the burst's name (runs/<name> must not exist)")
    g.add_argument("--avg-from", required=True, help="the average's first checkpoint's steps (inclusive; 27e6)")
    g.add_argument("--eval", nargs="*", action="extend", default=[], help="the burst's other policies (checkpoint paths; several after one --eval, or --eval repeated: all are kept)")
    g.add_argument("--current", choices=("swa", "ema"), default="swa",
                   help="the burst's current: the new average (swa) or the checkpoint's EMA (the average joins the evals)")
    g.add_argument("--games", type=int, default=64)
    g.add_argument("--processes", type=int, default=16)
    g.add_argument("--hours", type=float, default=0.75)
    g.add_argument("--poll", type=float, default=15.0, help="seconds between checks while waiting")
    g.add_argument("--dry-run", action="store_true", help="print the commands; nothing waited for, stopped or started")
    g.set_defaults(fn=cmd_budget)
    f = sub.add_parser("fork-check", help="after a fork or resume: log notes, launch code, the box's command")
    f.add_argument("run")
    f.add_argument("--lines", type=int, default=8)
    f.set_defaults(fn=cmd_fork_check)
    a = ap.parse_args(argv)
    a.root = a.root.resolve()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
