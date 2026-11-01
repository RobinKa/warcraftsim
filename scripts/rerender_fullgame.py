"""Re-render a whole-game self-play video from its replay (e.g. after changing the camera or the overlay).

    python scripts/rerender_fullgame.py fgself-64 game-1792106881-a4 [--speed 4] [--out file.mp4]

warcraftsim/fullgame/rerender.py: uses runs/RUN/replays/STEM.w3g (with its .commands.json and .trace.json), the run's
launch flags and the run's GE Wine. Default output: the run's videos/STEM.mp4 (replacing it). Renders with sound, as
fast as the machine draws.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


if __name__ == "__main__":
    from warcraftsim.fullgame.rerender import main
    raise SystemExit(main())
