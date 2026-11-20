"""An evaluation burst's run dir (2026-11-13): runs/NAME with SOURCE_RUN's vocab.json and run.json, CURRENT as its one
checkpoint (the policy "current") and the EVAL checkpoints as its opponents and fellow policies against insane, with
eval_vs 0.4 and no targets, heuristic or human games. Then, the learner and the box's service stopped:

    python scripts/burst_prep.py burst-220 fgself-220 runs/fgself-220/checkpoints/0000000025981195.pt \\
        runs/fgself-212/checkpoints/<the control's>.pt runs/fgself-212/checkpoints/0000000118808740.pt
    setsid nohup .venv/bin/python -m warcraftsim.fullgame.evalworker --run burst-220 --games 64 --processes 16 \\
        --host tora-burst --hours 0.75 > runs/burst-220.log 2>&1 < /dev/null &
    python scripts/head_to_head.py burst-220 --vs <an eval checkpoint>
"""
import json
import shutil
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) < 4:
        print(__doc__)
        return 2
    name, src, cur, *evals = argv
    d = Path("runs") / name
    (d / "checkpoints").mkdir(parents=True, exist_ok=False)
    shutil.copy(f"runs/{src}/vocab.json", d / "vocab.json")
    dest = Path(cur).name if Path(cur).stem.isdigit() else f"{0:015d}1.pt"  # (the worker reads <steps>.pt only)
    shutil.copy(cur, d / "checkpoints" / dest)
    r = json.load(open(f"runs/{src}/run.json"))
    c = r["config"]
    c["eval_checkpoints"] = "[" + ", ".join(f"PosixPath('{e}')" for e in evals) + "]"
    c.update(eval_vs="0.4", eval_z="0", eval_heuristic="0", eval_human="0",
             eval_handicap="0.25,0.5", eval_handicap_share="0.4")  # (the handicap ladder: 40% of the AI matchups)
    (d / "run.json").write_text(json.dumps(r, indent=1))
    (d / "burst.json").write_text(json.dumps({"current": cur, "evals": evals}, indent=1))  # (the ratings' name for
    # "current": the checkpoint it is, dashboard/ratings.BURST)
    print(name, "ready:", cur, "as", dest, c["eval_checkpoints"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
