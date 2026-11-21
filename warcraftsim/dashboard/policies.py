"""A run's policies as things of their own (the user, 2026-11-21: "better tracking of the ckpts and when clicking
them, bringing us to the correct section of that training run where we can inspect that specific ckpt"), a run's
launches, and the evaluation bursts' own identity in the run list.

listing(): what the run page's Checkpoints section lists: each of the run's policies that was ever rated (a checkpoint,
a weight average, the run's policy in one of its hours: ratings.py's players), was started from, is another run's KL
anchor or a burst's policy, and each weight average on disk. For each: where it lies on the run's timeline (steps, when
it was saved, which launch wrote it), and from the ratings' view (ratings.Ratings.view, made once per scan: nothing here
reads an episode file) its rating, its record against every opponent (the insane AI by race, the handicap rungs, head
to head) and the folders whose games those were.

launches(): the run's start and each restart (run.json "resumes") as a stretch of its own: what it was (its title and
note), when and how long it ran, its steps, its code, the flags that differ from the launch before, and its own
evaluation record.

burst(), own(): an evaluation burst's run.json is a copy of the run's whose policy it plays (scripts/burst_prep.py),
so the run list showed that run's name, notes and start once more for every burst of it (fgself-236: four entries with
one card, each opening fgself-236). own() gives a burst its folder's name, the policy it plays as what it came from,
its own time and state; the run list draws it as a burst."""

from __future__ import annotations

import bisect
import json
import pickletools
import re
import time
import zipfile
from pathlib import Path

from . import ratings

WORKER = "remote-worker.json"  # (fullgame/evalworker.py's state: "pid" while it plays, "stopped" after)
_FILES: dict[Path, tuple[int, object]] = {}  # a small file's content by its modification time (_small)
_AVERAGED: dict[Path, tuple[int, dict | None]] = {}
_SWA = re.compile(r"swa-(\d+)-(\d+)")
# flags that say nothing about how a launch trained (its words are shown beside it; every restart has --resume)
QUIET_FLAGS = ("--note", "--title", "--resume", "--name")


def _small(path: Path):
    """A small JSON file's content, read again only when the file changed; None: none, or not JSON."""
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        _FILES.pop(path, None)
        return None
    got = _FILES.get(path)
    if got is None or got[0] != stamp:
        try:
            got = (stamp, json.loads(path.read_text()))
        except (OSError, ValueError):
            got = (stamp, None)
        _FILES[path] = got
        while len(_FILES) > 512:
            _FILES.pop(next(iter(_FILES)))
    return got[1]


def _ref(name: str | None) -> dict:
    """A policy's name with its run and its id there (what a link to it needs)."""
    pol = ratings.policy_of(name) if name else None
    return {"name": name, "run": pol and pol["run"], "id": pol and pol["id"], "kind": pol and pol["kind"]}


def burst(d: Path, now: float | None = None) -> dict | None:
    """An evaluation burst's folder (its burst.json): {"current": the checkpoint it plays as "current", "policy",
    "run", "id", "kind": that policy by name, "evals": the listed checkpoints, "note", "created": when the folder was
    made, "status": "prepared", "playing" or "finished"}; None: the folder is no burst."""
    b = _small(d / ratings.BURST)
    if not isinstance(b, dict):
        return None
    cur = str(b.get("current") or "")
    made = []
    for f in (ratings.BURST, "vocab.json", "checkpoints"):  # (burst_prep.py writes them at once; the worker none of them)
        try:
            made.append((d / f).stat().st_mtime)
        except OSError:
            pass
    w = _small(d / WORKER)
    w = w if isinstance(w, dict) else None
    alive = w is not None and w.get("pid") and (now or time.time()) - float(w.get("time") or 0) < 150  # (written every 30 s)
    return {"current": cur, **_ref(ratings.checkpoint_name(cur)),
            "policy": ratings.checkpoint_name(cur), "note": str(b.get("note") or ""),
            "evals": [_ref(n) for n in map(ratings.checkpoint_name, map(str, b.get("evals") or [])) if n],
            "created": min(made) if made else None, "host": (w or {}).get("host"),
            "status": "playing" if alive else "finished" if w is not None else "prepared"}


def own(d: Path, info: dict, now: float | None = None) -> dict:
    """run.json as its own folder's. A burst's is a copy of another run's: here it has the folder's name, the policy
    it plays as what it came from, no restarts, start or end of that run, and its own time and state ("burst")."""
    b = burst(d, now)
    if b is None:
        return info
    info = {k: v for k, v in info.items() if k not in ("resumes", "inputs", "first_launch", "started", "finished")}
    info.update(name=d.name, burst={k: v for k, v in b.items() if k != "name"}, status=b["status"],
                init_from=b["current"] or None, init_steps=None,
                description=f"An evaluation burst: {b['policy'] or b['current']} against the insane AI"
                            + (f" and {', '.join(e['name'] for e in b['evals'])}" if b["evals"] else ""))
    if b["created"]:
        info["created"] = b["created"]
    return info


# ---- launches ------------------------------------------------------------------------------------------------------
def flags_of(argv) -> dict[str, str]:
    """A command line's flags and their values ("" for a bare one; the last of a flag given twice)."""
    out: dict[str, str] = {}
    key = None
    for a in map(str, argv or []):
        if a.startswith("--") and len(a) > 2 and not a[2].isdigit():
            key, eq, v = a.partition("=")
            out[key] = v if eq else ""
        elif key is not None:
            out[key] = (out[key] + " " + a).strip()
    return out


def flag_changes(before, after) -> list[dict] | None:
    """The flags of one launch that differ from the launch before: [{"flag", "from", "to"}] (None: not given there;
    "": a bare flag); None when either command line was not recorded."""
    if not before or not after:
        return None
    a, b = flags_of(before), flags_of(after)
    return [{"flag": k, "from": a.get(k), "to": b.get(k)} for k in sorted(set(a) | set(b))
            if a.get(k) != b.get(k) and k not in QUIET_FLAGS]


def _words(text: str, n: int = 400) -> str:
    """A note's first paragraph, at most n characters."""
    para = " ".join(str(text or "").strip().split("\n\n")[0].split())
    return para if len(para) <= n else para[:n - 1].rsplit(" ", 1)[0] + "…"


def launches(info: dict, train: list[dict], episodes: list[dict] | None = None) -> list[dict]:
    """A run's launches in order, the first its start, then each restart (run.json "resumes"): {"index", "title",
    "note", "time", "end" (when its last trainer line was written), "trained" (seconds), "steps", "to_steps", "commit",
    "subject", "dirty", "flags" (flag_changes against the launch before; None: a command line was not recorded: a
    restart overwrote the first launch's until run.json kept it in "first_launch"), "checkpoint" (the one a restart
    went on from), "rolled_back_to"}. With `episodes` (the run's rows, by time) each has its own numbers: "games"
    (training games), "insane" (the current policy's evaluation games against the untaxed insane AI in that time,
    [won, tied, lost]) and "vs" (head to head: opponent -> [won, tied, lost])."""
    resumes = [r for r in info.get("resumes") or [] if isinstance(r, dict)]
    last = info.get("launch") or {}
    first = info.get("first_launch") or ({} if resumes else last)
    out = [{"index": 0, "title": str(info.get("title") or ""), "note": _words(info.get("notes") or ""),
            "time": info.get("created"), "steps": 0, "git": first.get("git"), "argv": first.get("argv")}]
    for k, r in enumerate(resumes, 1):
        newest = k == len(resumes)  # (run.json "launch" is the latest launch's)
        out.append({"index": k, "title": str(r.get("title") or ""), "note": _words(r.get("note") or ""),
                    "time": r.get("time"), "steps": int(r.get("steps") or 0),
                    "git": r.get("git") or (last.get("git") if newest else None),
                    "argv": r.get("argv") or (last.get("argv") if newest else None),
                    "checkpoint": r.get("checkpoint"), "rolled_back_to": r.get("rolled_back_to")})
    starts = [float(s["time"] or 0.0) for s in out]
    rows = [r for r in train if "time" in r]
    times = [r["time"] for r in rows]
    for k, s in enumerate(out):
        nxt = out[k + 1] if k + 1 < len(out) else None
        lo = bisect.bisect_left(times, starts[k])
        hi = bisect.bisect_left(times, starts[k + 1]) if nxt else len(rows)
        mine = rows[lo:hi]
        s["end"] = mine[-1]["time"] if mine else None
        # (the learner's uptime starts anew with each launch: its last line's is how long this one trained)
        s["trained"] = (mine[-1].get("uptime") or mine[-1]["time"] - (s["time"] or mine[0]["time"])) if mine else 0.0
        s["to_steps"] = nxt["steps"] if nxt else (mine[-1].get("agent_steps") if mine else s["steps"])
        g = s.pop("git") or {}
        s.update(commit=str(g.get("commit") or "")[:12], subject=str(g.get("subject") or ""), dirty=bool(g.get("dirty")))
        s["flags"] = flag_changes(out[k - 1]["argv_"], s["argv"]) if k else None
        s["argv_"] = s.pop("argv")
        if episodes is not None:
            s.update(games=0, insane=[0, 0, 0], vs={})
    for s in out:
        del s["argv_"]
    if episodes is not None:
        for e in episodes:
            s = out[max(0, bisect.bisect_right(starts, e.get("time", 0.0)) - 1)]
            if not e.get("eval"):
                s["games"] += 1
                continue
            key, opp, x = str(e.get("eval_key") or "current"), str(e.get("opponent") or ""), e.get("outcome", 0)
            i = 0 if x > 0 else 2 if x < 0 else 1
            if key == "current" and "ai-insane" in opp and "(real)" in opp:
                s["insane"][i] += 1
            elif key.startswith("vs "):
                s["vs"].setdefault(ratings.checkpoint_name(key) or key[3:], [0, 0, 0])[i] += 1
    return out


# ---- a run's policies ------------------------------------------------------------------------------------------------
def averaged(path: Path) -> dict | None:
    """A weight average's {"n", "from", "to"} (scripts/average_checkpoints.py's "averaged" entry: how many checkpoints,
    the first's and the last's steps), read from the file's pickle as text (no torch, nothing unpickled); None: none."""
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        return None
    got = _AVERAGED.get(path)
    if got is None or got[0] != stamp:
        entry = None
        try:
            with zipfile.ZipFile(path) as z:
                data = z.read(next(n for n in z.namelist() if n.endswith("data.pkl")))
            found, key, vals = False, None, {}
            memo: dict[int, object] = {}  # (a string written before is a reference to it after: "n", "from", "to")
            text, puts = None, 0  # the string the last opcode pushed; how many objects were memoized
            for op, arg, _ in pickletools.genops(data):
                name = op.name
                if name in ("BINPUT", "LONG_BINPUT", "PUT", "MEMOIZE"):
                    memo[puts if name == "MEMOIZE" else arg] = text
                    puts += 1
                    continue
                text = arg if name.endswith("UNICODE") else memo.get(arg) if name in ("BINGET", "LONG_BINGET", "GET") else None
                if not found:
                    found = text == "averaged"
                elif isinstance(text, str):
                    if key in ("from", "to"):
                        vals[key], key = text, None
                    else:
                        key = text
                elif name.startswith(("BININT", "INT", "LONG")) and key == "n":
                    vals[key], key = arg, None
                elif name in ("SETITEMS", "SETITEM", "STOP", "DICT"):
                    break
            steps = {k: int(Path(str(vals[k])).stem) for k in ("from", "to") if Path(str(vals.get(k, ""))).stem.isdigit()}
            if "n" in vals and len(steps) == 2:
                entry = {"n": int(vals["n"]), **steps}
        except (OSError, ValueError, KeyError, StopIteration, zipfile.BadZipFile, IndexError):
            entry = None
        _AVERAGED[path] = got = (stamp, entry)
    return got[1]


def _span_of_name(name: str) -> dict | None:
    """A weight average's span from its name ("swa-43-53": millions of steps; "swa-118808740-144699411": steps)."""
    m = _SWA.fullmatch(name)
    if not m:
        return None
    lo, hi = int(m.group(1)), int(m.group(2))
    return {"from": lo * 1_000_000, "to": (hi + 1) * 1_000_000 - 1, "rounded": True} if hi < 100_000 else {"from": lo, "to": hi}


def listing(runs_dir: Path, d: Path, info: dict, view: dict | None, train: list[dict], replays=None) -> dict:
    """A run's policies for its Checkpoints section, the newest on the run's timeline first: {"run", "rated" (whether
    the ratings were there), "updated", "launches": how many, "policies": [{"id", "kind" ("checkpoint", "average",
    "hour"), "name", "label", "steps" / "from_steps", "to_steps", "n" (a weight average's checkpoints), "saved" (when;
    "saved_by": "file" or "trainer log"), "launch" (index), "file", "path", "rated": [...] (ratings.Ratings._view),
    "forked_by", "anchor_of", "bursts", "listed_in", "checkpoints" (an hour's: those saved in it), "aliases",
    "replays": {"q": the Replays page's search for its games, "count"}}], "burst": the folder's own burst (the
    ratings' view of it), if it is one}.
    `replays(names)`: name -> how many evaluation replays name that policy (None: not counted)."""
    run = d.name
    segs = launches(info, train)
    pts = [(r["time"], r.get("agent_steps") or 0) for r in train if "time" in r]
    times, steps = [p[0] for p in pts], [p[1] for p in pts]

    def steps_at(t: float) -> int | None:
        if not pts:
            return None
        return int(steps[min(len(pts) - 1, max(0, bisect.bisect_right(times, t) - 1))]) if t >= times[0] else 0

    def time_at(s: int) -> float | None:  # (when the trainer first counted that many steps)
        i = bisect.bisect_left(steps, s)
        return times[i] if i < len(pts) else None

    def launch_by_steps(s: int) -> int:  # (a checkpoint at a restart's step was written before it)
        return max([k for k, g in enumerate(segs) if g["steps"] < s] or [0])

    def launch_by_time(t: float) -> int:
        return max([k for k, g in enumerate(segs) if (g["time"] or 0) <= t] or [0])
    files: dict[int, tuple[str, float]] = {}  # steps -> (file, when saved)
    try:
        for f in (d / "checkpoints").iterdir():
            if f.suffix == ".pt" and f.stem.isdigit():
                files[int(f.stem)] = (f"checkpoints/{f.name}", f.stat().st_mtime)
    except OSError:
        pass
    avgs: dict[str, Path] = {}
    try:
        avgs = {f.stem: f for f in (d / "averaged").iterdir() if f.suffix == ".pt"}
    except OSError:
        pass
    known = dict((view or {}).get("runs", {}).get(run, {}))
    for name in avgs:
        known.setdefault(name, {"id": name, "kind": "average", "name": f"{run} avg {name}", "rated": []})
    created = float(info.get("created") or 0.0)
    try:
        rel = d.resolve().relative_to(Path(runs_dir).resolve().parent).as_posix()
    except ValueError:
        rel = str(d)
    out = []
    for pid, p in known.items():
        row = {**p, "label": pid}
        if p["kind"] == "checkpoint":
            s = int(pid)
            f = files.get(s)
            when = f[1] if f else time_at(s)
            row.update(label=f"checkpoint {s:,}", steps=s, saved=when, saved_by="file" if f else "trainer log" if when else None,
                       launch=launch_by_steps(s), file=f[0] if f else None, at=(s, 1, 0.0))
        elif p["kind"] == "average":
            f = avgs.get(pid)
            span = (averaged(f) if f else None) or _span_of_name(pid) or {}
            lo, hi = span.get("from"), span.get("to")
            if span.get("rounded") and files:  # (a span from the name alone: the checkpoints on disk within it)
                inside = sorted(s for s in files if lo <= s <= hi)
                lo, hi, span = (inside[0], inside[-1], {**span, "n": len(inside)}) if inside else (lo, hi, span)
            row.update(label=f"average {pid}", from_steps=lo, to_steps=hi, n=span.get("n"),
                       saved=f.stat().st_mtime if f else None, saved_by="file" if f else None,
                       launch=launch_by_steps(hi) if hi is not None else None, file=f"averaged/{f.name}" if f else None,
                       at=(hi if hi is not None else -1, 2, 0.0))
        else:  # the run's policy in one of its hours: no checkpoint of its own
            h = int(pid[1:])
            t0, t1 = created + 3600.0 * h, created + 3600.0 * (h + 1)
            lo, hi = steps_at(t0), steps_at(t1)
            inside = sorted(s for s in files if lo is not None and lo < s <= hi) if hi is not None else []
            # (its launch: the one running as the hour ended; a restart within it: its games are the new launch's)
            row.update(label=f"hour {h}", from_time=t0, to_time=t1, from_steps=lo, to_steps=hi, launch=launch_by_time(t1),
                       checkpoints=[{"steps": s, "saved": files[s][1], "rated": str(s) in known} for s in inside],
                       aliases=[_ref(a) for a in sorted((view or {}).get("aliases", {}).get(p["name"], []))],
                       at=(hi if hi is not None else -1, 0, t0))
        if row.get("file"):
            row["path"] = f"{rel}/{row['file']}"
        out.append(row)
    # an evaluation replay's title names a listed checkpoint by its name (selfplay.played_by); the run's own policy
    # is "policy v<update>" there: an hour's link is the run's replays
    counts = replays([r["name"] for r in out if r["kind"] != "hour"]) if replays is not None else {}
    for r in out:
        r["replays"] = {"q": run, "count": None} if r["kind"] == "hour" else {"q": r["name"], "count": counts.get(r["name"])}
    # the newest on the run's timeline first, by the steps each ends at (an average above the checkpoint it ends at,
    # an hour above the checkpoints saved in it; hours without a trainer's log: last, by their time)
    out.sort(key=lambda r: r["at"], reverse=True)
    for r in out:
        del r["at"]
    return {"run": run, "rated": view is not None, "updated": (view or {}).get("updated"), "launches": len(segs),
            "policies": out, "burst": (view or {}).get("bursts", {}).get(run)}
