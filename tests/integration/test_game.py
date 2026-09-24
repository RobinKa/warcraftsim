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


def test_melee_restart_uses_warm_spare(game_dir):
    import time

    setup = GameSetup(slots=[Agent("human"), BuiltinAI("orc", "easy")], step_seconds=0.5)
    with GameInstance(setup, name="it_spare") as g:
        g.start()
        first = g.name
        deadline = time.time() + 120
        while g._spare_thread is not None and g._spare_thread.is_alive() and time.time() < deadline:
            g.step()  # the spare loads in the background meanwhile
        t = time.time()
        obs = g.restart()
        assert time.time() - t < 3.0, "a loaded spare makes restart nearly instant"
        assert g.name != first and obs.seq == 0 and obs.game_time < 0.5
        assert len(obs.units_of(0)) == 6
        obs = g.step()
        assert obs.game_ms > 0


def test_selfplay_env(game_dir):
    import numpy as np

    from warcraftsim.env import MicroSelfPlayEnv

    env = MicroSelfPlayEnv(Scenario.skirmish(["hfoo"] * 4, ["hfoo"] * 2, max_game_seconds=120), name="it_selfplay")
    try:
        obs, _ = env.reset()
        assert obs[0]["own_mask"].sum() == 4 and obs[1]["enemy_mask"].sum() == 4 and obs[1]["own_mask"].sum() == 2
        attack = np.zeros((env.max_units, 3), np.int64)
        attack[:, 0] = 3  # player 0 focus-fires; player 1 only auto-acquires
        total = {0: 0.0, 1: 0.0}
        for n in range(600):
            obs, rewards, terminated, truncated, infos = env.step({0: attack if n % 8 == 0 else attack * 0,
                                                                  1: attack * 0})
            total = {p: total[p] + rewards[p] for p in total}
            if any(terminated.values()) or any(truncated.values()):
                break
        assert terminated[0] and terminated[1]
        assert infos[0]["obs"].players[0].result == Result.VICTORY
        assert abs(total[0] + total[1]) < 1e-6 and total[0] > 1
    finally:
        env.close()


def test_replay_reproduces_agent_game(game_dir, tmp_path):
    sc = Scenario.skirmish(["hfoo"] * 3, ["ogru"] * 2, max_game_seconds=60)
    setup = GameSetup(slots=[Agent("human"), Scripted("orc")], scenario=sc)
    from warcraftsim import Wc3Game

    live = {}
    with Wc3Game(setup, name="it_replay") as g:
        obs = g.reset()
        n = 0
        while not obs.game_over:
            live[obs.seq] = {u.id: (u.x, u.y, u.hp) for u in obs.units if u.alive}
            enemies = g.enemies()
            if n % 4 == 0 and enemies:
                for u in g.my_units():
                    g.attack(u, min(enemies, key=lambda e: e.hp))
            obs = g.step()
            n += 1
        replay = g.instance.save_replay(tmp_path / "ep.w3g")
        assert (tmp_path / "ep.commands.json").exists()
        inst = g.instance
        obs = inst.play_replay(replay)
        played = {}
        while True:
            played[obs.seq] = {u.id: (u.x, u.y, u.hp) for u in obs.units if u.alive}
            if obs.game_over:
                break
            obs = inst.step()
    assert len(live) > 50 and live == {s: played[s] for s in live}


def test_replay_video_with_audio(game_dir, tmp_path):
    """A replay rendered to MP4: 40 fps video, the game's audio, the overlay's panel."""
    import json
    import subprocess

    import numpy as np

    from warcraftsim import Wc3Game
    from warcraftsim.overlay import EpisodeOverlay
    from warcraftsim.puffer.tasks import get_task
    from warcraftsim.video import render_replay

    task = get_task("micro_mirror")
    setup = GameSetup(slots=[Agent("human"), Scripted("orc")], scenario=Scenario.skirmish(
        ["hfoo"] * 2, ["hfoo"] * 2, max_game_seconds=12))
    with Wc3Game(setup, name="it_video") as g:
        obs, steps = g.reset(), 0
        while not obs.game_over:
            obs = g.step()
            steps += 1
        replay = g.instance.save_replay(tmp_path / "ep.w3g")
    trace = {"obs": np.zeros((steps, 1, task.obs_size), np.float32),
             "actions": np.zeros((steps, 1, task.num_atns), np.float32), "rewards": np.zeros((steps, 1), np.float32)}
    out = render_replay(setup, replay, tmp_path / "ep.mp4", name="it_video_render",
                        overlay=EpisodeOverlay(task, trace, None, title="test"))
    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(out)],
                                      capture_output=True, text=True, check=True).stdout)["streams"]
    video = next(s for s in probe if s["codec_type"] == "video")
    audio = next(s for s in probe if s["codec_type"] == "audio")
    assert video["r_frame_rate"] == "40/1" and int(video["width"]) == 960 + EpisodeOverlay.PANEL_W
    assert abs(float(video["duration"]) - float(audio["duration"])) < 0.1
    level = subprocess.run(["ffmpeg", "-i", str(out), "-vn", "-af", "volumedetect", "-f", "null", "-"],
                           capture_output=True, text=True).stderr
    assert float(level.split("max_volume:")[1].split("dB")[0]) > -40  # not silent


def test_scenario_spare_parks_the_running_game(game_dir, tmp_path):
    """Video episodes (relaunch=True, then save_replay) swap processes without a game load."""
    import time

    from warcraftsim import Wc3Game

    sc = Scenario.skirmish(["hfoo"] * 2, ["hfoo"] * 2, max_hp=60, max_game_seconds=20)
    setup = GameSetup(slots=[Agent("human"), Scripted("orc")], scenario=sc, scenario_spare=True)

    def play(g, obs):
        seen = {}
        while not obs.game_over:
            seen[obs.seq] = sorted((u.id, u.x, u.y, u.hp) for u in obs.units if u.alive)
            obs = g.step()
        return seen

    with Wc3Game(setup, name="it_spare_sc") as g:
        play(g, g.reset())  # episode 1, and the spare loads in the background
        g.instance._spare_thread.join()
        t = time.time()
        live = play(g, g.reset(relaunch=True))  # the video episode, in the (fresh) spare
        swap_in = time.time() - t
        replay = g.instance.save_replay(tmp_path / "ep.w3g")
        t = time.time()
        obs = g.reset()  # back to the parked game: an in-game restart
        resume = time.time() - t
        assert obs.game_time < 1.0 and not obs.game_over
        play(g, obs)
    assert swap_in < 3.0 and resume < 1.0, (swap_in, resume)
    with GameInstance(setup, name="it_spare_sc_play") as inst:
        obs = inst.play_replay(replay)
        played = {}
        while True:
            played[obs.seq] = sorted((u.id, u.x, u.y, u.hp) for u in obs.units if u.alive)
            if obs.game_over:
                break
            obs = inst.step()
    assert len(live) > 10 and live == {k: played[k] for k in live}
