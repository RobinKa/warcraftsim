"""Evaluation games on another machine for a self-play run here (warcraftsim/fullgame/evalworker.py there):
starts the worker over ssh (again when it died or the machine came back), sends it what it plays with, and
takes its games back into runs/<run>/episodes-remote.jsonl, which real_games.py and the dashboard read with
episodes.jsonl. Light: no torch, an ssh call or two a minute and a ~13 MB checkpoint every few minutes.

    scripts/remote_eval.py --host robin@laptop-wsl [--run fgself-76] [--games 16]   # (in the background)
    scripts/remote_eval.py --host robin@laptop-wsl --status
    scripts/remote_eval.py --host robin@laptop-wsl --stop

Every --every seconds, when the host answers:
* the run: --run, else the live one (its learner's process here). When it ends (no learner for --grace
  minutes), the worker stops (with --run this exits too); following the live run, a new run gets the worker.
* a launch of the run with other settings for its evaluation games (SETTINGS: other evaluation checkpoints, map,
  rules; a restart with the same ones leaves the worker be), or a worker that is not running: this tree's code is
  mirrored there (the tracked and new files of warcraftsim/, scripts/, native/ and the built shim: what a run
  launched now would use), run.json, vocab.json and the evaluation checkpoints sent (at their paths relative to
  the repository), and the worker (re)started as a systemd user service (systemd-run --user, niced: it outlives
  the ssh session, `systemctl --user stop` ends all of it). New code for the worker: --stop, and the manager
  starts it again with it.
* the run's newest checkpoint, at most every --push-every seconds: the worker's "current"; the host keeps the
  newest --keep it was sent.
* the worker's new rows (complete lines, by count), their times moved by the host's clock offset when it is off
  by more than 2 s (a WSL clock drifts after the laptop slept).
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
UNIT = "warcraftsim-evalworker"  # the systemd user service there (one worker a host)
EPISODES = "episodes-remote.jsonl"  # (fullgame/evalworker.py's file names)
STATUS = "remote-worker.json"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4"]


def log(*a) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def notify(title: str, body: str, urgent: bool = False) -> None:
    """A desktop notification on this machine (the user's session; urgent ones stay until dismissed): the
    worker's host going away or coming back. Best effort: no session, no notification."""
    try:
        subprocess.run(["notify-send", "-a", "warcraftsim", "-u", "critical" if urgent else "normal", title, body],
                       timeout=10, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        pass


class Host:
    def __init__(self, host: str, repo: str):
        self.host, self.repo = host, repo  # repo: the repository there, relative to its home

    def run(self, cmd: str, timeout: float = 120, binary: bool = False) -> subprocess.CompletedProcess | None:
        """A shell command there (None: the host did not answer)."""
        try:
            r = subprocess.run(SSH + [self.host, cmd], capture_output=True, text=not binary, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        return None if r.returncode == 255 else r

    def rsync(self, files: list[str], timeout: float = 900) -> bool:
        """Files (paths relative to this repository) to the same paths there."""
        if not files:
            return True
        cmd = ["rsync", "-a", "-R", "--timeout=60", "-e", " ".join(SSH)] + files + [f"{self.host}:{self.repo}/"]
        try:
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        if r.returncode:
            log(f"rsync failed ({r.returncode}): {r.stderr.strip()[-300:]}")
        return r.returncode == 0


def live_runs() -> list[str]:
    """The self-play runs whose learner runs here, the newest last (by process start)."""
    out = subprocess.run(["ps", "-eo", "lstart,args", "--sort=start_time"], capture_output=True, text=True).stdout
    runs = []
    for line in out.splitlines():
        m = re.search(r"-m warcraftsim\.fullgame\.selfplay .*--name (\S+)", line)
        if m and "spawn_main" not in line and m.group(1) not in runs:
            runs.append(m.group(1))
    return runs


def run_info(run: str) -> dict | None:
    try:
        return json.loads((RUNS / run / "run.json").read_text())
    except (OSError, ValueError):
        return None


# the run's settings its evaluation games depend on (fullgame/evalworker.run_config reads them): a restart of the
# run that changed none of them (the main session restarts runs for infrastructure often) leaves the worker be
SETTINGS = ("map", "handicap", "step_seconds", "max_minutes", "wait_floor_ms", "games_per_process", "agent_games_factor",
            "pairs", "scripted_reset", "native_obs", "game_nice", "game_cuts", "features", "eval_ai", "eval_checkpoints",
            "races", "mirror", "opd_difficulty", "shared_noise", "float_penalty", "hero_level_value", "shaping",
            "shaping_scale", "gamma", "tie_break", "tie_value", "avail_mask", "infer_entities", "seed", "wine", "fsync", "ntsync",
            "eval_z", "z_pool", "z_src", "z_enforce", "z_time_scale")


def launch_key(info: dict) -> str:
    """The run's evaluation games' settings, as a key."""
    c = info.get("config", {})
    return hashlib.sha1(json.dumps([c.get(k) for k in SETTINGS]).encode()).hexdigest()[:10]


def eval_checkpoints(info: dict) -> list[str]:
    """The run's --eval-checkpoints relative to this repository (others can't be sent: left out, the worker
    then refuses to start)."""
    out = []
    for p in re.findall(r"PosixPath\('((?:[^'\\]|\\.)*)'\)", info.get("config", {}).get("eval_checkpoints") or ""):
        q = Path(p)
        if q.is_absolute():
            if not q.is_relative_to(ROOT):
                log(f"evaluation checkpoint {p} is outside the repository: not sent")
                continue
            q = q.relative_to(ROOT)
        out.append(str(q))
    return out


def newest_checkpoint(run: str) -> str | None:
    cks = sorted(p.name for p in (RUNS / run / "checkpoints").glob("[0-9]*.pt") if re.fullmatch(r"\d+\.pt", p.name))
    return cks[-1] if cks else None


def code_files() -> list[str]:
    """What the worker runs: this tree's tracked and new (not ignored) files of the package, the scripts and the
    native sources, and the built shim."""
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "--", "warcraftsim", "scripts", "native",
                          "pyproject.toml"], cwd=ROOT, capture_output=True, text=True).stdout.split("\n")
    files = [f for f in out if f and (ROOT / f).is_file()]
    return files + [f"build/shim/{f}" for f in ("w3shim.dll", "w3launch.exe") if (ROOT / "build" / "shim" / f).exists()]


class Manager:
    def __init__(self, a):
        self.a = a
        self.h = Host(a.host, a.remote_dir)
        self.run: str | None = None
        self.key: str | None = None  # the launch the worker plays for
        self.pushed: list[str] = []  # the run's checkpoints sent, oldest first
        self.t_push = 0.0
        self.gone_since: float | None = None  # the run's learner not seen since
        self.away = False
        self.misses = 0  # the host's unanswered looks in a row
        self.notified = False  # the user was told it is down (then also that it is back)

    def remote(self, rel: str = "") -> str:
        return f"{self.a.remote_dir}/runs/{self.run}/{rel}"

    def description(self, key: str) -> str:
        """The worker's service's description there: what it plays (the run, its settings' key, games x processes)."""
        return f"warcraftsim evalworker {self.run} {key} {self.a.games}x{self.a.processes}"

    def state(self) -> dict | None:
        """The worker there: {"active", "unit" (its description: run and launch), "lines", "clock", "status"}."""
        run = self.run or "-"
        cmd = (f"cd {shlex.quote(self.a.remote_dir)} 2>/dev/null; date +%s.%N; systemctl --user is-active {UNIT}; "
               f"systemctl --user show -p Description --value {UNIT}; "
               f"wc -l < runs/{run}/{EPISODES} 2>/dev/null || echo 0; cat runs/{run}/{STATUS} 2>/dev/null")
        t0 = time.time()
        r = self.h.run(cmd, timeout=60)
        t1 = time.time()
        if r is None:
            return None
        lines = r.stdout.split("\n")
        try:
            status = json.loads("\n".join(lines[4:])) if "".join(lines[4:]).strip() else {}
        except ValueError:
            status = {}
        try:
            offset = float(lines[0]) - (t0 + t1) / 2
        except ValueError:
            offset = 0.0
        return {"active": lines[1].strip(), "unit": lines[2].strip(), "lines": int(lines[3] or 0) if lines[3].strip().isdigit() else 0,
                "clock": offset, "status": status}

    def stop_worker(self, why: str) -> None:
        r = self.h.run(f"systemctl --user stop {UNIT}; systemctl --user reset-failed {UNIT} 2>/dev/null; true", timeout=120)
        log(f"worker stopped ({why})" if r is not None else f"worker not reached to stop it ({why})")
        self.key = None

    def push_checkpoint(self, force: bool = False) -> None:
        ck = newest_checkpoint(self.run)
        if ck is None or (self.pushed and ck == self.pushed[-1]):
            return
        if not force and time.time() - self.t_push < self.a.push_every:
            return
        if not self.h.rsync([f"runs/{self.run}/checkpoints/{ck}"]):
            return
        self.t_push = time.time()
        self.pushed = (self.pushed + [ck])[-self.a.keep:]
        # the older ones there go (an evaluation checkpoint of this run's own stays)
        keep = " ".join(f"-e {shlex.quote(Path(p).name)}" for p in eval_checkpoints(run_info(self.run) or {})
                        if Path(p).parent.parent.name == self.run)
        self.h.run(f"cd {shlex.quote(self.remote('checkpoints'))} && ls | grep -E '^[0-9]+[.]pt$' | sort | "
                   f"head -n -{self.a.keep} " + (f"| grep -vxF {keep} " if keep else "") + "| xargs -r rm -f --", timeout=60)

    def start_worker(self, info: dict) -> bool:
        """The code, the run's files and its newest checkpoint there, then the worker (as a fresh service)."""
        key = launch_key(info)
        log(f"starting the worker for {self.run} (launch {key}): code, run files, checkpoints")
        self.h.run(f"systemctl --user stop {UNIT} 2>/dev/null; systemctl --user reset-failed {UNIT} 2>/dev/null; "
                   f"mkdir -p {shlex.quote(self.remote('checkpoints'))}", timeout=120)
        if not self.h.rsync(code_files()):
            return False
        evals = eval_checkpoints(info)
        missing = [p for p in evals if not (ROOT / p).exists()]
        if missing:
            log(f"evaluation checkpoints missing here: {missing}")
            return False
        pool = (info.get("config") or {}).get("z_pool") or ""  # (--eval-z: the strategies, as the run's)
        pool = [pool] if pool and not Path(pool).is_absolute() and (ROOT / pool).exists() else []
        if not self.h.rsync([f"runs/{self.run}/run.json", f"runs/{self.run}/vocab.json"] + evals + pool):
            return False
        self.pushed = []
        self.push_checkpoint(force=True)
        if not self.pushed:
            log("no checkpoint sent: the worker waits for one")
        home = "$HOME/" + self.a.remote_dir
        logf = f"{home}/runs/{self.run}/remote-worker.log"
        args = ["--run", self.run, "--games", str(self.a.games), "--host", self.a.name]
        if self.a.processes:
            args += ["--processes", str(self.a.processes)]
        cmd = (f"systemd-run --user --unit={UNIT} --collect --nice={self.a.nice} --working-directory={home} "
               f"--description={shlex.quote(self.description(key))} "
               f"-p KillMode=control-group -p TimeoutStopSec=60 "
               f"-p StandardOutput=append:{logf} -p StandardError=append:{logf} "
               f"{home}/.venv/bin/python -m warcraftsim.fullgame.evalworker {' '.join(shlex.quote(x) for x in args)}")
        r = self.h.run(cmd, timeout=120)
        if r is None or r.returncode:
            log(f"worker not started: {(r.stderr if r else 'no answer').strip()[-300:]}")
            return False
        self.key = key
        log(f"worker started: {self.a.games} games for {self.run}")
        return True

    def pull(self, st: dict) -> None:
        """The host's new rows, appended here (by count: complete lines only; one manager at a time, as --stop
        pulls too)."""
        mine = RUNS / self.run / EPISODES
        with open(mine.with_suffix(".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._pull(st, mine)

    def _pull(self, st: dict, mine: Path) -> None:
        have = sum(1 for _ in mine.open()) if mine.exists() else 0
        if st["lines"] < have:
            if st["lines"]:
                log(f"the host's {EPISODES} has {st['lines']} rows, {have} here: not pulled")
            return
        if st["lines"] == have:
            return
        r = self.h.run(f"tail -n +{have + 1} {shlex.quote(self.remote(EPISODES))} | head -n {st['lines'] - have}",
                       timeout=120, binary=True)
        if r is None or r.returncode:
            return
        rows = []
        for line in r.stdout.split(b"\n")[:st["lines"] - have]:
            try:
                e = json.loads(line)
            except ValueError:  # (kept as it is, the readers skip it: the counts here and there stay the same)
                log("a damaged row from the host")
                rows.append(line.decode(errors="replace") + "\n")
                continue
            if abs(st["clock"]) > 2.0:  # (the host's clock: its rows on this machine's time)
                e["time"] = round(e["time"] - st["clock"], 3)
            rows.append(json.dumps(e) + "\n")
        if rows:
            with mine.open("a") as f:
                f.writelines(rows)
            (mine.parent / "remote-eval.alive").touch()  # (the run's own evaluation games wait while it is fresh: --eval-fallback)
            log(f"{len(rows)} games from {self.a.name} ({have + len(rows)} in all)")

    def cycle(self) -> bool:
        """One look; False: done (--run's run ended)."""
        a = self.a
        live = live_runs()
        target = a.run or (live[-1] if live else self.run)  # (no live run: the last one's grace, a restart's gap)
        if target is not None and target in live:
            self.gone_since = None
        elif self.gone_since is None:
            self.gone_since = time.time()
        ended = target is None or (self.gone_since is not None and time.time() - self.gone_since > 60 * a.grace)
        st = self.state()
        if st is None:
            if not self.away:
                log(f"{a.host} does not answer: trying again every {a.every:.0f} s")
            self.away = True
            self.misses += 1
            # (the user is told only of an outage: 3 looks in a row unanswered, ~3 minutes; a slow answer while the
            # laptop is busy, or the worker restarting for a new run, popped the warning up before)
            if self.misses >= 3 and not self.notified:
                notify(f"Evaluation worker down ({a.host})", f"No answer from the laptop for {self.misses} minutes. The "
                       "run's own evaluation games take over (--eval-fallback). If WSL is up, run `ssh {a.host}` once: "
                       "Tailscale SSH may want its periodic browser check.", urgent=True)
                self.notified = True
            return True
        self.misses = 0
        if self.away:
            log(f"{a.host} is back (worker {st['active']})")
            if self.notified:
                notify(f"Evaluation worker back ({a.host})", "The laptop answers again: its evaluation games resume.")
                self.notified = False
            self.away = False
        if self.run is not None:  # (its rows first: a stop or a switch below must not lose them)
            self.pull(st)
        unit = st["unit"].split()
        playing = unit[2] if st["active"] == "active" and unit[:2] == ["warcraftsim", "evalworker"] else None
        if self.run is not None and (self.run != target or ended):
            if playing == self.run:
                self.stop_worker(f"{self.run} ended" if ended else f"{target} is the live run now")
                st = self.state() or st
                self.pull(st)  # (its last games)
                playing = None
            self.run, self.key, self.pushed = None, None, []
        if playing is not None and (playing != target or ended):  # (a worker left from before, for another run)
            self.stop_worker(f"it played for {playing}")
        if ended:
            return not a.run
        if self.run is None:
            self.run = target
            log(f"run {target}")
            st = self.state() or st
        info = run_info(self.run)
        if info is None:
            log(f"no runs/{self.run}/run.json here")
            return True
        key = launch_key(info)
        if not (st["active"] == "active" and st["unit"] == self.description(key)):
            if target in live:  # (a new launch of the run, or no worker: started, with this tree's code)
                if st["active"] == "active":
                    log(f"the worker plays for another launch ({st['unit']})")
                self.start_worker(info)
            return True
        self.key = key
        if not self.pushed:  # (a worker found running: what it has is unknown, the newest is sent)
            self.push_checkpoint(force=True)
        self.push_checkpoint()
        return True

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True, help="ssh destination, e.g. robin@laptop-wsl")
    ap.add_argument("--name", help="the host's name in the rows (default: the destination's host part)")
    ap.add_argument("--run", help="the run (default: the live one, followed when another starts)")
    ap.add_argument("--games", type=int, default=16, help="games at a time there (laptop-wsl: 12 made ~710 an hour, 16 ~790)")
    ap.add_argument("--processes", type=int, default=0, help="worker processes there (default: evalworker's)")
    ap.add_argument("--remote-dir", default="warcraftsim", help="the repository there, relative to its home")
    ap.add_argument("--every", type=float, default=60.0, help="seconds between looks")
    ap.add_argument("--push-every", type=float, default=240.0, help="seconds between checkpoints sent")
    ap.add_argument("--keep", type=int, default=3, help="checkpoints kept there")
    ap.add_argument("--grace", type=float, default=10.0, help="minutes without the run's learner before the worker stops")
    ap.add_argument("--nice", type=int, default=10)
    ap.add_argument("--status", action="store_true", help="the worker's state, then exit (nothing pulled)")
    ap.add_argument("--stop", action="store_true", help="stop the worker (after taking its last games), then exit")
    a = ap.parse_args()
    a.name = a.name or a.host.split("@")[-1]
    m = Manager(a)
    if a.status or a.stop:
        m.run = a.run or (live_runs() or [None])[-1]
        st = m.state()
        if st is None:
            print(f"{a.host} does not answer")
            return 1
        print(json.dumps({"run": m.run, **st}, indent=1))
        if a.stop:
            m.stop_worker("--stop")
            st = m.state()
            if m.run and st:  # (its last games)
                m.pull(st)
        return 0
    log(f"managing the evaluation worker on {a.host} ({a.games} games) for {a.run or 'the live run'}")
    try:
        while m.cycle():
            time.sleep(a.every)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
