#!/usr/bin/env bash
# The run's training games on the games box (selfplay --remote-games there, fullgame/remoteactor.py here): the commit's
# code and built shim mirrored there, checked against the learner's (wire.code_version), and the actors (re)started as
# a systemd user service (`systemctl --user stop warcraftsim-remoteactor` there ends them).
#     scripts/box_games.sh <commit> [actors] [games a actor] [ssh destination] [the learner's address there]
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
COMMIT=$(git rev-parse --short=12 "${1:?the commit of the run}")
ACTORS=${2:-8} GAMES=${3:-8} BOX=${4:-tora-ms-a2@tora-ms-a2} LEARNER=${5:-10.77.0.1}
CHECKOUT=$HOME/wc3/code/$COMMIT
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15)
git archive "$COMMIT" -- warcraftsim scripts native pyproject.toml | "${SSH[@]}" "$BOX" 'cd ~/warcraftsim && tar -x -m -f -'
(cd "$CHECKOUT" && rsync -a -R -e "${SSH[*]}" build/shim/w3shim.dll build/shim/w3launch.exe "$BOX:warcraftsim/")
HERE=$(cd "$CHECKOUT" && PYTHONPATH=. "$ROOT/.venv/bin/python" -c 'from warcraftsim.fullgame import wire; print(wire.code_version())')
"${SSH[@]}" "$BOX" bash -s -- "$HERE" "$ACTORS" "$GAMES" "$LEARNER" <<'REMOTE'
set -euo pipefail
cd ~/warcraftsim
systemctl --user stop warcraftsim-remoteactor 2>/dev/null || true
THERE=$(.venv/bin/python -c 'from warcraftsim.fullgame import wire; print(wire.code_version())')
[ "$THERE" = "$1" ] || { echo "code differs: here $1, there $THERE" >&2; exit 1; }
LOG=$HOME/warcraftsim/runs/remoteactor.log
systemd-run --user --unit=warcraftsim-remoteactor --collect --working-directory="$HOME/warcraftsim" \
    -p KillMode=control-group -p TimeoutStopSec=60 -p "StandardOutput=append:$LOG" -p "StandardError=append:$LOG" \
    "$HOME/warcraftsim/.venv/bin/python" -m warcraftsim.fullgame.remoteactor --host "$4" --actors "$2" \
    --games-per-actor "$3" --name "$(hostname)" --ksm 1
REMOTE
