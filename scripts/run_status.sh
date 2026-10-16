#!/bin/bash
# A self-play run's status at a glance: alive, errors since its graphs were captured, the last update,
# steps/s over the last 30 minutes, the head-to-head evaluation games since SINCE (each policy's wins,
# ties and AI-record share by 20-minute windows; the last line of real_games.py: the share on one mix).
#   scripts/run_status.sh fgself-65 "2026-10-16 09:07"
cd "$(dirname "$0")/.."
RUN=$1; SINCE=$2
date
pgrep -f "selfplay --name $RUN " | head -1 | xargs -I{} ps -o pid,etime -p {}
L=runs/$RUN.log; n=$(grep -n "graphs captured" $L | tail -1 | cut -d: -f1)
echo "errors since graphs captured:"; tail -n +${n:-1} $L | grep -E "Traceback|GameCrashed|failed|Assertion|terminate" | cut -c1-60 | sort | uniq -c | head
tail -1 $L | cut -c1-100
.venv/bin/python scripts/throughput.py $RUN --since "$(date -d '30 min ago' '+%Y-%m-%d %H:%M')" 2>&1 | tail -4 | head -2
.venv/bin/python scripts/real_games.py $RUN --since "$SINCE" --split eval_key | grep -E "^== eval_key|by difficulty|one matchup"
.venv/bin/python - "$RUN" "$SINCE" <<'PY'
import json, sys
sys.path.insert(0, 'scripts')
from collections import Counter
from throughput import when
from real_games import ai_rates, difficulty
rates = ai_rates(); lo = when(sys.argv[2])
eps = [json.loads(l) for l in open(f'runs/{sys.argv[1]}/episodes.jsonl')]
eps = [e for e in eps if e['time'] > lo and e.get('eval')]
for k in sorted({e['eval_key'] for e in eps}):
    X = [e for e in eps if e['eval_key'] == k]; c = Counter(e['outcome'] for e in X); win = []
    for a in range(0, 240, 20):
        Y = [e for e in X if lo + a * 60 < e['time'] <= lo + (a + 20) * 60]
        R = [(e, rates.get((e['race'], e['opponent_race'], difficulty(e)))) for e in Y]; R = [(e, p) for e, p in R if p]
        if R:
            win.append(f"{100 * sum(e['outcome'] > 0 for e, _ in R) / max(1e-9, sum(p for _, p in R)):.0f}%")
    heroes = sum(e['prod']['heroes'] for e in X) / len(X)
    print(f"{k[-14:]:14s} n={len(X)} won {100 * c[1.0] / len(X):.0f}% ties {100 * c[0.0] / len(X):.0f}% heroes {heroes:.2f}  by 20 min: {' '.join(win)}")
PY
