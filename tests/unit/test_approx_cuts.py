"""The approximation bits of W3SIM_CUTS (2^44 and up, shim/approx.c): evaluation games never get them."""

import json
from pathlib import Path

from warcraftsim.fullgame import evalworker
from warcraftsim.runtime.instance import EXACT_CUTS_MASK, exact_cuts

EXACT = 17592186041849  # (round 7's exact cuts)
APPROX = (1 << 44) | (1 << 45) | (1 << 46)  # (shim/approx.c's three)


def test_exact_cuts_drops_the_approximations():
    assert EXACT_CUTS_MASK == (1 << 44) - 1
    assert exact_cuts(EXACT) == EXACT
    assert exact_cuts(EXACT | APPROX) == EXACT
    assert exact_cuts(None) == 0 and exact_cuts(0) == 0


def test_remote_evaluation_plays_the_exact_game(tmp_path: Path):
    run = tmp_path / "fgself-x"
    run.mkdir()
    config = {"map": "echofast", "eval_map": "(2)EchoIsles", "game_cuts": str(EXACT | APPROX), "features": "29",
              "eval_ai": "insane", "races": "all", "eval_checkpoints": "[]"}
    (run / "run.json").write_text(json.dumps({"config": config, "launch": {"argv": []}}))
    cfg, _ = evalworker.run_config(run, "laptop", 4, "1")
    assert cfg["game_cuts"] == EXACT
