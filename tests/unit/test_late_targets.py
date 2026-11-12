"""A late start (the scenario "late", late.py) has no strategy target without --late-z: the target drawn for the game
before the start was picked goes from the side's view too (its input), and the win balance is told the game has none.
Until 2026-11-10 only --late-z cleared the views: a late start kept the first drawn target's input, seen and never
paid (zs, the reward's, was cleared), and the balance's tax counted it (840 of fgself-201's first 1,524 late starts)."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

pytest.importorskip("torch")

from warcraftsim.fullgame import play, selfplay  # noqa: E402


class _Bot:
    made: list = []

    def __init__(self, *_a, **_k):
        self.view = NS(z=None, zmask=None, forbid=None, mix_target={}, levels={}, max_own=0, max_ent=0)
        self.enc = NS()
        self.h, self.trees, self.issued = None, {}, 0
        _Bot.made.append(self)

    def begin(self, obs, races):
        pass

    def observe(self, obs, t, rows=None):  # (no side acts in a late start's first steps anyway)
        return None


def _player(x: int, y: int) -> NS:
    return NS(start_x=x, start_y=y, gold=500, lumber=150, gold_gathered=0, lumber_gathered=0, food_used=5, food_cap=10,
              upkeep=0, result=NS(name="TIE"))


def _obs(over: bool) -> NS:
    return NS(players={0: _player(-3000, 0), 1: _player(3000, 0)}, units=[], unit_array=None, events=[], issued=[],
              command_results=[], game_time=0.5 if over else 0.0, game_over=over, destructables=None)


class _Game:
    def __init__(self):
        self.setup = NS(slots=[NS(kind="agent", difficulty="normal", handicap=100) for _ in range(2)])

    def step(self, cmds):
        return _obs(True)


def _play(monkeypatch, late_z: int) -> tuple[dict, list]:
    _Bot.made = []
    monkeypatch.setattr(play, "BCAgent", _Bot)

    def target(cfg, view, st, pool, race, vocab, modes=None):
        view.z, view.mix_target = np.ones(4, np.float32), {"hfoo": 1.0}
        return st
    monkeypatch.setattr(selfplay, "target", target)
    monkeypatch.setattr(selfplay, "strategy_pool", lambda cfg: {"items": {"human": ["hbar"], "orc": ["obar"]}})
    monkeypatch.setattr(selfplay.strat, "sample", lambda *a, **k: {"race": "human", "t": {"hbar": 100.0}, "src": "ai",
                                                                   "won": True})
    monkeypatch.setattr(selfplay.strat, "scaled", lambda st, f: st)
    monkeypatch.setattr(selfplay.late_game, "states", lambda files, from_minute=0.0: NS(pick=lambda *a: {"minute": 9.0, "sides": [{}, {}]}))
    monkeypatch.setattr(selfplay.late_game, "commands", lambda state, rows: [])
    monkeypatch.setattr(selfplay.late_game, "levels", lambda state, s: {})
    monkeypatch.setattr(selfplay.late_game, "plan_target", lambda *a, **k: None)  # (--late-z: no plan in this state)
    cfg = {"map": "echofast2", "values": {}, "shaping_scale": 2000.0, "shaping": 0.5, "gamma": 0.99, "lam": 0.95,
           "chunk": 64, "max_steps": 10, "step_seconds": 0.5, "tie_break": 0.1, "tie_value": -0.5, "float_penalty": 0.5,
           "z_share": 1.0, "scenarios": ["late"], "scenario_share": 1.0, "late_states": ["states.jsonl"],
           "win_balance": 0.5, "pfsp_data": 1e-9, "late_z": late_z, "run_dir": "."}
    opp = {"kind": "past", "name": "past:1", "path": "/x/1.pt"}
    out = []
    ep = selfplay.play_one(_Game(), _obs(False), cfg, {"orders": []}, ["human", "orc"], 0, opp, None,
                           NS(put=out.append), 0)
    return ep, _Bot.made


@pytest.mark.parametrize("late_z", [0, 1])
def test_late_start_drops_the_drawn_target(monkeypatch, late_z):
    ep, bots = _play(monkeypatch, late_z)
    assert ep["scenario"] == "late" and "z" not in ep  # (no target in the game: no reward for one)
    assert len(bots) == 2
    for b in bots:  # (nor in its views: the policy's input)
        assert b.view.z is None and b.view.mix_target == {} and b.view.zmask is None and b.view.forbid is None
    assert ep["tb_x"][4] == 0.0  # (the balance's features: neither side plays toward a target)
