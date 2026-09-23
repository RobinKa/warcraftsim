"""Real-game tests: launch Warcraft III under Wine. Run with `pytest -m wine`."""

import pytest

from warcraftsim import Agent, BuiltinAI, GameInstance, GameSetup, Scenario, Scripted
from warcraftsim.protocol import PointOrder, Result

pytestmark = pytest.mark.wine


def test_melee_step_and_orders(game_dir):
    setup = GameSetup(slots=[Agent("human"), BuiltinAI("orc", "easy")], step_seconds=0.25)
    with GameInstance(setup, name="it_melee") as g:
        obs = g.start()
        assert obs.is_first and obs.game_time < 0.5
        assert set(obs.players) == {0, 1} and obs.players[0].is_agent and not obs.players[1].is_agent
        mine = obs.units_of(0)
        assert sum(u.type == "hpea" for u in mine) == 5 and any(u.type == "htow" for u in mine)
        worker = next(u for u in mine if u.type == "hpea")
        target = (worker.x, worker.y - 600)
        obs = g.step([PointOrder(worker.id, obs.orders["move"], *target)])
        assert obs.command_results == [True]
        times = [obs.game_ms]
        for _ in range(24):
            obs = g.step()
            times.append(obs.game_ms)
        assert all(b - a == 250 for a, b in zip(times, times[1:])), "every step is exactly one step of game time"
        assert obs.unit(worker.id).dist(*target) < 120
        # the built-in AI plays: its workers mine and it builds
        assert obs.players[1].gold_gathered > 0 or obs.players[1].food_used > 5


def test_scenario_elimination_and_reset(game_dir):
    sc = Scenario.skirmish(["hfoo"] * 2, ["ogru"] * 3, max_game_seconds=120)
    with GameInstance(GameSetup(slots=[Agent("human"), Scripted("orc")], scenario=sc), name="it_scen") as g:
        obs = g.start()
        assert sorted(u.type for u in obs.units) == ["hfoo", "hfoo", "ogru", "ogru", "ogru"]
        while not obs.game_over:
            obs = g.step()
        assert obs.players[1].result == Result.VICTORY and obs.players[0].result == Result.DEFEAT
        obs = g.restart()
        assert obs.seq == 0 and not obs.game_over and len(obs.units) == 5
        assert all(u.hp == u.max_hp for u in obs.units)


def test_navigate_env(game_dir):
    import numpy as np

    from warcraftsim.env import NavigateEnv

    env = NavigateEnv(name="it_nav")
    try:
        obs, _ = env.reset()
        for _ in range(100):
            a = 1 + int(round(np.degrees(np.arctan2(obs[1], obs[0])) % 360 / 45)) % 8
            obs, reward, terminated, truncated, info = env.step(a)
            if terminated or truncated:
                break
        assert terminated and info["distance"] <= 100
    finally:
        env.close()
