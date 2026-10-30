"""Old runs' heavy files to another machine: the checkpoints nothing needs and the videos of fgself-<= --upto move there
(to ~/run-archive/runs/...), checked by checksum before they go from here. Their logs stay, so the dashboard's charts,
lineage and ratings are as they were. Stays too: each run's last checkpoint, every checkpoint a later run's settings or
league names (the anchor, --league-add, evaluation checkpoints), and the top rated (the evaluation worker's gauntlet).
Without --go it only says what it would do. The moved files are listed in runs/archived-<date>.txt; to bring one back:

    rsync -a <host>:run-archive/runs/fgself-12/checkpoints/0000000000123.pt runs/fgself-12/checkpoints/
    scripts/archive_runs.py --host tora-ms-a2@tora-ms-a2 --upto 150 [--go]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
CKPT = re.compile(r"runs/fgself-\d+/checkpoints/\d+\.pt")
SSH = "ssh -o BatchMode=yes -o ConnectTimeout=15"


def named(root: Path, upto: int) -> set[str]:
    """The checkpoints (paths from the repository) that the files of the runs past --upto name: settings, league."""
    out: set[str] = set()
    for d in root.glob("fgself-*"):
        m = re.fullmatch(r"fgself-(\d+)", d.name)
        if not m or int(m.group(1)) <= upto:
            continue
        for f in ("run.json", "league.json", "league_spec.json", "gauntlet.json"):
            try:
                out.update(CKPT.findall((d / f).read_text()))
            except OSError:
                pass
    return out


def plan(root: Path, upto: int, keep: set[str]) -> list[str]:
    """What moves (paths from root's parent): the old runs' checkpoints but the last and `keep`, and their videos."""
    out = []
    for d in sorted(root.glob("fgself-*")):
        m = re.fullmatch(r"fgself-(\d+)", d.name)
        if not m or int(m.group(1)) > upto:
            continue
        cks = sorted(p for p in (d / "checkpoints").glob("*") if p.is_file())
        pts = [p for p in cks if p.suffix == ".pt"]
        last = pts[-1] if pts else None
        for p in cks:
            rel = f"{root.name}/{p.relative_to(root)}"
            if p != last and rel not in keep:
                out.append(rel)
        out += [f"{root.name}/{p.relative_to(root)}" for p in sorted((d / "videos").rglob("*.mp4"))]  # (their small
        # .info.json stay: the dashboard's Videos page searches them)
    return out


def top_rated(size: int) -> set[str]:
    spec = importlib.util.spec_from_file_location("remote_eval", ROOT / "scripts" / "remote_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return set(mod.gauntlet({"config": {}}, "", size))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", required=True)
    ap.add_argument("--upto", type=int, required=True, help="the runs fgself-<= this")
    ap.add_argument("--top", type=int, default=60, help="the top rated checkpoints that stay")
    ap.add_argument("--bwlimit", default="60m", help="rsync's limit: the live run's games share the link")
    ap.add_argument("--go", action="store_true", help="do it (without: only say what it would do)")
    a = ap.parse_args()
    rated = top_rated(a.top)
    if not rated:
        raise SystemExit("no ratings from the dashboard: not moving anything (the top rated must stay)")
    keep = named(RUNS, a.upto) | rated
    files = plan(RUNS, a.upto, keep)
    gb = sum((ROOT / f).stat().st_size for f in files) / 2**30
    print(f"{len(files)} files, {gb:.1f} GB to {a.host}; staying by name: {len(keep)} checkpoints ({len(rated)} top rated)")
    print("  e.g. staying:", sorted(keep)[:6])
    if not a.go:
        return 0
    lst = RUNS / f"archived-{time.strftime('%Y%m%d-%H%M')}.txt"
    lst.write_text("".join(f + "\n" for f in files))
    base = ["nice", "-n", "15", "ionice", "-c3", "rsync", "-a", "--files-from", str(lst), "-e", SSH]
    dest = f"{a.host}:run-archive/"
    subprocess.run([*base, f"--bwlimit={a.bwlimit}", str(ROOT), dest], check=True)
    r = subprocess.run([*base, "--checksum", "--dry-run", "--itemize-changes", str(ROOT), dest], check=True,
                       capture_output=True, text=True)
    differ = [x for x in r.stdout.splitlines() if x.strip() and not x.startswith(".d")]
    if differ:
        print(f"{len(differ)} files differ there, nothing removed here:", differ[:5], file=sys.stderr)
        return 1
    for f in files:
        (ROOT / f).unlink()
    print(f"moved and checked: {len(files)} files, {gb:.1f} GB; list in {lst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
