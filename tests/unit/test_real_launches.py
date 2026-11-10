"""--real-share's training launches ("script:ai-X (real)") play the real game when the run evaluates on another map
(--eval-map): its map, order costs and strategies, as the evaluation games. Until 2026-11-10 they played the run's own
map (echofast2) with the AI untaxed: every training instance of fgself-200 (--real-share 0.5) loaded echofast2's map."""

import json
import os
import threading
from types import SimpleNamespace

import numpy as np

from warcraftsim.fullgame import remoteactor, selfplay


class _Game:
    _ended = False

    def __init__(self, setup, name="", timeout=0):
        self.setup = setup

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def start(self):
        return SimpleNamespace(unit_array=np.zeros((0, 32), np.int32), units=[], players={}, destructables=None,
                               ground=[], issued=[])

    def restart(self, relaunch=False):
        return self.start()


def _cfg(tmp_path, **extra) -> dict:
    (tmp_path / "vocab.json").write_text(json.dumps({"orders": []}))
    return {"seed": 0, "start_update": 0, "video_every": 0.0, "actors": 1, "vocab": str(tmp_path / "vocab.json"),
            "run_dir": str(tmp_path), "slot": 0, "games_per_actor": 1, "learner_pid": os.getppid(),
            "races": ["human", "orc"], "mirror": False, "map": "echofast2", "handicap": 100,
            "step_seconds": 0.5, "max_minutes": 1.0, "wait_floor_ms": 5, "scripted_reset": False, "native_obs": True,
            "games_per_process": 1, "agent_games_factor": 1, "avail_mask": True,
            "costs": [[0, 0, 0], [67, 0, 2]], "z_pool": "fast.json",
            "eval_map": "(2)EchoIsles", "eval_z_pool": "real.json", "eval_costs": [[0, 0, 0], [135, 0, 2]], **extra}


def _launches(tmp_path, monkeypatch, cfg: dict, n: int = 12) -> list[dict]:
    from warcraftsim.runtime import instance
    (tmp_path / "league_spec.json").write_text(json.dumps({
        "launch": [{"kind": "ai", "difficulty": "insane", "real": True, "p": 0.5},
                   {"kind": "ai", "difficulty": "insane", "p": 0.5,
                    "by_race": {k: {"handicap": 100, "delay": 0.0, "tax": 0.3, "level": 0.33}
                                for k in ("human/orc", "orc/human", "human/human", "orc/orc")}}],
        "agents": [{"kind": "self", "name": "self", "p": 1.0}]}))
    seen, stop = [], threading.Event()

    def play_one(g, obs, cfg_, vocab, races, side, opp, infer, out_q, wid, record=False, train=True, grpo=None):
        seen.append({"name": opp["name"], "tax": opp.get("tax"), "setup_map": g.setup.map, "map": cfg_["map"],
                     "z_pool": cfg_.get("z_pool"), "costs": cfg_["costs_array"].tolist()})
        if len(seen) >= n:
            stop.set()
        return {}

    monkeypatch.setattr(instance, "GameInstance", _Game)
    monkeypatch.setattr(selfplay, "play_one", play_one)
    selfplay.game_loop(0, 0, cfg, None, None, stop, None)
    return seen


def test_real_launches_play_the_evaluation_map(tmp_path, monkeypatch):
    seen = _launches(tmp_path, monkeypatch, _cfg(tmp_path))
    real = [s for s in seen if s["name"].endswith("(real)")]
    taxed = [s for s in seen if not s["name"].endswith("(real)")]
    assert real and taxed  # (both kinds drawn, in turn: the settings never carry from one launch to the next)
    for s in real:
        assert s["setup_map"] == s["map"] == "(2)EchoIsles" and s["z_pool"] == "real.json"
        assert s["costs"] == [[0, 0, 0], [135, 0, 2]] and s["tax"] is None
    for s in taxed:
        assert s["setup_map"] == s["map"] == "echofast2" and s["z_pool"] == "fast.json"
        assert s["costs"] == [[0, 0, 0], [67, 0, 2]] and s["tax"] == 0.3


def test_real_map_off_or_no_eval_map_plays_the_runs_map(tmp_path, monkeypatch):
    for extra in ({"real_map": 0}, {"eval_map": "", "eval_costs": None}):
        seen = _launches(tmp_path, monkeypatch, _cfg(tmp_path, **extra), n=6)
        assert all(s["setup_map"] == "echofast2" and s["z_pool"] == "fast.json" for s in seen)


def test_real_game_cfg():
    cfg = {"map": "echofast2", "eval_map": "(2)EchoIsles", "z_pool": "a", "eval_z_pool": "b", "eval_costs": [[1, 2, 3]]}
    got = selfplay.real_game_cfg(cfg)
    assert got["map"] == "(2)EchoIsles" and got["z_pool"] == "b" and got["costs_array"].tolist() == [[1, 2, 3]]
    assert got["real_launch"] and "real_launch" not in selfplay.eval_map_cfg(cfg)  # (its episode rows name the map)
    assert selfplay.real_game_cfg({**cfg, "eval_map": "echofast2"}) is None  # (the run's own map: nothing to switch)
    assert selfplay.real_game_cfg({**cfg, "real_map": 0}) is None
    assert selfplay.real_game_cfg({**cfg, "eval_z_pool": ""})["z_pool"] == "a"


def test_remote_actors_get_the_real_games_strategies(tmp_path):
    hello = {"cfg": {"seed": 0, "eval_map": "(2)EchoIsles"}, "files": {"vocab": "{}", "z_pool": "{\"x\": 1}",
                                                                         "eval_z_pool": "{\"y\": 2}"}}
    remoteactor.write_files(hello, tmp_path)
    a = SimpleNamespace(games_per_actor=4, actors=2, host="h", port=1, name="box", wait=1.0)
    cfg = remoteactor.actor_settings(hello, a, tmp_path, b"k", 0)
    assert json.loads((tmp_path / "eval_z_pool.json").read_text()) == {"y": 2}
    assert cfg["eval_z_pool"] == str(tmp_path / "eval_z_pool.json") and cfg["z_pool"] == str(tmp_path / "z_pool.json")
    hello["files"].pop("eval_z_pool")
    assert remoteactor.actor_settings(hello, a, tmp_path, b"k", 0)["eval_z_pool"] == ""
