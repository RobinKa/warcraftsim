"""Pinned code for self-play launches: a launch runs from a checkout of its commit, so the main tree can change
(merges, edits) while a run is live.

A launch (scripts/selfplay_fork.py, scripts/selfplay_restart.py) prepares the checkout of the commit it launches
(HEAD by default: `prepare`) and starts the learner in it (its working directory and PYTHONPATH: `launch_env`).
Every module the learner imports later, its actor processes' (spawned: the learner's directory and sys.path) and
its games' shim are that commit's; a merge in the main tree meanwhile reaches none of them. Before, a module that
changed under a live learner was imported new on first use (fgself-155's learner crashed when bc.py changed).

A checkout is a detached git worktree in paths.CODE_DIR/<commit[:12]> (git works there: run.json's launch.git
names the commit that runs; instance.recorded_map reads the history), reused by later launches of the commit:
  runs                         -> the main tree's runs: the run's data (run dirs, league, checkpoints, episodes);
                                  selfplay.RUNS resolves it, so the paths it writes are the main tree's
  build/stormlib, build/pjass  -> the main tree's (third-party tools, the same for every commit)
  build/shim                   the commit's shim (make -C shim, minhook from the main tree's submodule)
  build/native                 the commit's C libraries (native.lib_path: built here; build_lib deletes the other
                               hashes in its folder, so the folders are not shared)
The C++ inference loop builds in torch's cache by its source's hash (native_infer.build_source: shared by the
checkouts with the same source) and the venv is the main tree's. `prune` removes the checkouts that no process
uses (its working directory, command line or a mapped file in one), beyond the KEEP last prepared.

    python -m warcraftsim.codepin prepare [REV]    # the checkout's folder, built (on stdout)
    python -m warcraftsim.codepin list             # the checkouts, which processes use them
    python -m warcraftsim.codepin prune
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from pathlib import Path

from . import paths

KEEP = 4  # checkouts kept unused (the last prepared): a restart on the same commit reuses its build
STAMP = ".pinned.json"  # (written once a checkout is ready: one without it is unfinished)
NAME = re.compile(r"[0-9a-f]{12}")
# what a launch runs: uncommitted changes there are not in its checkout (dirty)
CODE_PATHS = ("warcraftsim", "native", "shim", "scripts", "pyproject.toml")
LINKS = ("runs", "build/stormlib", "build/pjass")
# (this script in the checkout: the commit's native libraries built before the launch, not by 8 actors at once; a
# module an older commit lacks is passed over, and the inference loop only where it builds by source hash
# (native_infer.build_source): torch's default folder is the one a live learner of an older commit has loaded)
PREBUILD = """
import importlib, sys
done = []
for mod in ("warcraftsim.native", "warcraftsim.fullgame.featnative", "warcraftsim.fullgame.rulesnative"):
    try:
        m = importlib.import_module(mod)
    except ImportError:
        continue
    if hasattr(m, "lib"):
        m.lib()
        done.append(mod.rsplit(".", 1)[-1])
if "--infer" in sys.argv:
    try:
        from warcraftsim.fullgame import native_infer
        if hasattr(native_infer, "build_source"):
            native_infer.extension()
            done.append("infer_loop")
    except Exception as e:  # (the learner builds it, or fails, itself)
        print(f"warning: the C++ inference loop not built: {e!r}"[:600], file=sys.stderr)
print(" ".join(done))
"""


def _git(root: Path, *args: str, check: bool = True) -> str:
    r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=300)
    if check and r.returncode:
        raise RuntimeError(f"git {' '.join(args)} (in {root}): {r.stderr.strip()[-600:]}")
    return r.stdout.rstrip() if r.returncode == 0 else ""  # (rstrip: status lines start with a space)


def commit_of(root: Path, rev: str = "HEAD") -> str:
    return _git(Path(root), "rev-parse", "--verify", f"{rev}^{{commit}}")


def dirty(root: Path) -> list[str]:
    """The code's uncommitted changes in `root` (git status lines; untracked files too): not in a launch's checkout."""
    return [line for line in _git(Path(root), "status", "--porcelain", "--", *CODE_PATHS).splitlines() if line.strip()]


def code_dir(commit: str) -> Path:
    return paths.CODE_DIR / commit[:12]


def stamp(d: Path) -> dict:
    try:
        return json.loads((Path(d) / STAMP).read_text())
    except (OSError, ValueError):
        return {}


def _shared(root: Path) -> dict[str, Path]:
    """Each link's target: the main tree's (an environment override of the tool's file: its folder)."""
    def tool(env: str, default: Path) -> Path:
        v = os.environ.get(env)
        return Path(v).expanduser().resolve().parent if v else default
    return {"runs": root / "runs", "build/stormlib": tool("WARCRAFTSIM_STORMLIB", root / "build" / "stormlib"),
            "build/pjass": tool("WARCRAFTSIM_PJASS", root / "build" / "pjass")}


def minhook_dir(root: Path) -> Path:
    v = os.environ.get("WARCRAFTSIM_MINHOOK")
    return Path(v).expanduser().resolve() if v else root / "third_party" / "minhook"


@contextmanager
def _locked():
    """One launch's preparing or pruning at a time."""
    paths.CODE_DIR.mkdir(parents=True, exist_ok=True)
    with open(paths.CODE_DIR / ".lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def _link(root: Path, d: Path) -> None:
    for rel, target in _shared(root).items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.is_symlink():
            if os.readlink(p) == str(target):
                continue
            p.unlink()
        elif p.exists():
            raise RuntimeError(f"{p} exists and is no link (the commit tracks it?): the run's data would go there")
        p.symlink_to(target, target_is_directory=True)


def _remove(root: Path, d: Path) -> None:
    """A checkout gone: its links first (unlinked, never followed: the main tree's runs stays), then git's worktree."""
    if d.is_symlink():
        d.unlink()
        return
    for rel in LINKS:
        if (d / rel).is_symlink():
            (d / rel).unlink()
    _git(root, "worktree", "remove", "--force", str(d), check=False)
    if d.exists():
        shutil.rmtree(d)  # (an unfinished checkout git doesn't know; rmtree unlinks links, never follows them)


def build(root: Path, d: Path, commit: str, infer: bool = True, log: Callable[[str], None] = print) -> list[str]:
    """The checkout's shim (make -C shim: build/shim) and native libraries (build/native; the inference loop: torch's
    cache) -> what was built. Each is a no-op when built."""
    minhook = minhook_dir(root)
    if not (minhook / "include" / "MinHook.h").exists():
        raise RuntimeError(f"no minhook in {minhook} (git submodule update --init third_party/minhook)")
    want = _git(root, "rev-parse", f"{commit}:third_party/minhook", check=False)  # (the commit's gitlink)
    have = _git(minhook, "rev-parse", "HEAD", check=False) if (minhook / ".git").exists() else ""
    if want and have and want != have:
        raise RuntimeError(f"{commit[:12]} builds the shim with minhook {want[:12]}, {minhook} has {have[:12]}")
    if not (want and have):
        log(f"warning: minhook's commit not checked ({minhook})")
    r = subprocess.run(["make", "-s", "-C", str(d / "shim"), f"MINHOOK={minhook}", f"OUT={d / 'build' / 'shim'}"],
                       capture_output=True, text=True, timeout=900)
    if r.returncode:
        raise RuntimeError(f"the shim's build failed in {d}:\n{(r.stdout + r.stderr)[-2000:]}")
    r = subprocess.run([sys.executable, "-c", PREBUILD, *(["--infer"] if infer else [])], cwd=d, env=launch_env(d),
                       capture_output=True, text=True, timeout=1800)
    if r.stderr.strip():
        log(r.stderr.strip()[-2000:])
    if r.returncode:
        raise RuntimeError(f"the native libraries' build failed in {d}")
    return ["shim", *r.stdout.split()]


def prepare(root: Path, rev: str = "HEAD", *, builds: bool = True, infer: bool = True,
            log: Callable[[str], None] = print) -> Path:
    """The checkout of `rev` (its folder in paths.CODE_DIR), made or reused, linked and built."""
    root = Path(root).resolve()
    commit = commit_of(root, rev)
    d = code_dir(commit)
    with _locked():
        if stamp(d).get("commit") != commit:
            t0 = time.time()
            if d.exists() or d.is_symlink():
                log(f"{d}: unfinished, made again")
                _remove(root, d)
            _git(root, "worktree", "prune")
            _git(root, "worktree", "add", "--detach", str(d), commit)
            log(f"checked out {commit[:12]} in {d} ({time.time() - t0:.1f} s)")
        _link(root, d)
        if builds:
            t0 = time.time()
            done = build(root, d, commit, infer=infer, log=log)
            log(f"built in {d}: {', '.join(done)} ({time.time() - t0:.1f} s)")
        info = {"commit": commit, "subject": _git(root, "log", "-1", "--format=%s", commit), "root": str(root),
                "prepared": time.time()}
        tmp = d / f"{STAMP}.tmp"
        tmp.write_text(json.dumps(info))
        tmp.replace(d / STAMP)
    return d


def launch_env(code: Path, env: dict | None = None) -> dict:
    """A launch's environment: the checkout first on PYTHONPATH (a script started by its path finds the commit's
    package, not the venv's editable install of the main tree; `-m` in the checkout finds it anyway)."""
    env = dict(os.environ if env is None else env)
    env["PYTHONPATH"] = os.pathsep.join([str(code), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])])
    return env


def users(dirs: Iterable[Path], proc: Path = Path("/proc")) -> dict[Path, list[int]]:
    """The processes that use each checkout: a working directory, command line or mapped file (a native library,
    the shim's DLL in a game) in it."""
    dirs = {Path(d): f"{Path(d).resolve()}/" for d in dirs}  # (as the kernel names them)
    out: dict[Path, list[int]] = {d: [] for d in dirs}
    if not dirs:
        return out
    for p in proc.iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        texts = []
        try:
            texts.append(os.readlink(p / "cwd") + "/")
        except OSError:
            pass
        for f in ("cmdline", "maps"):
            try:
                texts.append((p / f).read_bytes().replace(b"\0", b" ").decode(errors="replace"))
            except OSError:
                pass
        text = "\n".join(texts)
        for d, prefix in dirs.items():
            if prefix in text:
                out[d].append(int(p.name))
    return out


def checkouts() -> list[Path]:
    """The checkouts in paths.CODE_DIR, the last prepared first."""
    if not paths.CODE_DIR.is_dir():
        return []
    ds = [d for d in paths.CODE_DIR.iterdir() if NAME.fullmatch(d.name) and d.is_dir() and not d.is_symlink()]
    return sorted(ds, key=lambda d: stamp(d).get("prepared", d.stat().st_mtime), reverse=True)


def prune(root: Path, keep: int = KEEP, protect: Iterable[Path] = (), log: Callable[[str], None] = print) -> list[Path]:
    """The checkouts no process uses removed, beyond the `keep` last prepared and `protect` -> the removed."""
    root = Path(root).resolve()
    protect = {Path(p).resolve() for p in protect}
    with _locked():
        old = [d for d in checkouts()[keep:] if d.resolve() not in protect]
        used = users(old)
        gone = [d for d in old if not used[d]]
        for d in gone:
            subject = stamp(d).get("subject", "unfinished")
            _remove(root, d)
            log(f"removed {d} ({subject[:70]})")
        if gone:
            _git(root, "worktree", "prune", check=False)
    return gone


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="the checkout of REV (HEAD), built: its folder on stdout")
    p.add_argument("rev", nargs="?", default="HEAD")
    p.add_argument("--no-infer", action="store_true", help="not the C++ inference loop")
    sub.add_parser("list", help="the checkouts and the processes that use them")
    p = sub.add_parser("prune", help="remove the checkouts no process uses, beyond the last prepared")
    p.add_argument("--keep", type=int, default=KEEP)
    a = ap.parse_args(argv)
    root = paths.REPO_ROOT
    err = lambda m: print(m, file=sys.stderr)  # noqa: E731
    if a.cmd == "prepare":
        print(prepare(root, a.rev, infer=not a.no_infer, log=err))
    elif a.cmd == "list":
        ds = checkouts()
        used = users(ds)
        for d in ds:
            s = stamp(d)
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(s["prepared"])) if s.get("prepared") else "unfinished"
            print(f"{d}  {when}  {'in use by ' + str(len(used[d])) + ' processes' if used[d] else 'unused':<24}  "
                  f"{s.get('subject', '')[:70]}")
    else:
        prune(root, a.keep, log=err)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
