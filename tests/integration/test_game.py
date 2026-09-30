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


def test_melee_engine_restart(game_dir):
    """restart() reloads the map in the running game: same process, a new game, the AI plays again."""
    setup = GameSetup(slots=[Agent("human"), BuiltinAI("orc", "easy")], step_seconds=0.5)
    with GameInstance(setup, name="it_reload") as g:
        obs = g.start()
        proc = g.proc
        for _ in range(60):
            obs = g.step()
        obs = g.restart()
        assert g.proc is proc and obs.seq == 0 and obs.game_time < 0.5
        assert len(obs.units_of(0)) == 6 and obs.players[1].gold_gathered == 0
        for _ in range(120):
            obs = g.step()
        assert obs.players[1].food_used > 5, "the built-in AI trains workers after the reload"


def test_melee_restart_uses_warm_spare(game_dir):
    import time

    setup = GameSetup(slots=[Agent("human"), BuiltinAI("orc", "easy")], step_seconds=0.5, engine_restart=False)
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


def test_fresh_process_episodes_never_continue_a_parked_game(game_dir):
    """A video episode needs a process that ran nothing else (its replay is re-simulated from the
    start). Parking must not leak: an interrupted video episode goes back to the parked game, two
    video episodes in a row relaunch, and a recycled process is retired, not parked."""
    from warcraftsim import Wc3Game

    sc = Scenario.skirmish(["hfoo"] * 2, ["hfoo"] * 2, max_hp=60, max_game_seconds=20)
    setup = GameSetup(slots=[Agent("human"), Scripted("orc")], scenario=sc, scenario_spare=True, window=(320, 240))

    def play(g, obs):
        while not obs.game_over:
            obs = g.step()

    with Wc3Game(setup, name="it_spare_leak") as g:
        inst = g.instance
        play(g, g.reset())
        play(g, g.reset())
        long_pid = inst.pid
        inst._spare_thread.join()
        g.reset(relaunch=True)  # a video episode in the fresh spare; the long game parks
        assert inst.pid != long_pid and inst._spare._parked and inst._proc_episode == 0
        g.step()  # ... cut short (the trainer went away)
        obs = g.reset()  # the next normal episode continues the parked game
        assert inst.pid == long_pid and inst._proc_episode > 1
        play(g, obs)
        inst._spare_thread.join()
        g.reset(relaunch=True)  # video episode
        first_video = inst.pid
        g.reset(relaunch=True)  # and another one right away: a fresh process, the long game stays parked
        assert inst.pid not in (long_pid, first_video) and inst._proc_episode == 0
        assert inst._spare._parked and inst._spare.pid == long_pid
        play(g, g.reset())  # back to the long game
        assert inst.pid == long_pid
        inst._spare_thread.join()
        inst.setup.recycle_steps = 1
        g.reset()  # recycle: the long game's process retires
        inst.setup.recycle_steps = 0
        assert inst.pid != long_pid and inst._proc_episode == 0
        assert inst._spare is None or not inst._spare._parked


def test_hero_skills_and_casting(game_dir):
    """Queued heroes learn their skills; observations report each ability slot's level and
    cooldown; a cast (Storm Bolt) hits and starts the cooldown."""
    from warcraftsim import Wc3Game
    from warcraftsim.data.abilities import ability_info, hero_abilities
    from warcraftsim.protocol import QueueSpawn

    sc = Scenario(units=(), victory="elimination", max_game_seconds=40, name="abilities")
    setup = GameSetup(slots=[Agent("human"), Scripted("orc")], scenario=sc, step_seconds=0.5, window=(320, 240))
    slots = hero_abilities()["Hmkg"]
    with Wc3Game(setup, name="it_abilities") as g:
        g.reset()
        obs = g.reset(spawns=[QueueSpawn(0, "Hmkg", -300, 0, 0, hero_level=3, skills=("AHtb", "AHtb", "AHtc")),
                              QueueSpawn(1, "hfoo", 200, 0, 180)])
        mk = next(u for u in obs.units if u.type == "Hmkg")
        levels = dict(zip(slots, (lvl for lvl, _ in mk.abilities)))
        assert levels["AHtb"] == 2 and levels["AHtc"] == 1 and levels["AHbh"] == 0
        assert mk.skill_points == 0 and all(cd == 0 for _, cd in mk.abilities)
        foe = next(u for u in obs.units if u.type == "hfoo")
        g.cast(mk, ability_info()["AHtb"].order, target=foe)
        hp0 = foe.hp
        for _ in range(4):
            obs = g.step()
        mk, foe = obs.unit(mk.id), obs.unit(foe.id)
        cd = dict(zip(slots, (c for _, c in mk.abilities)))["AHtb"]
        assert 0 < cd < ability_info()["AHtb"].cooldown[1]
        assert foe.hp < hp0 - 100 or not foe.alive  # Storm Bolt level 2: 225 damage
        assert mk.mana < mk.max_mana


def test_replay_of_a_respawned_episode(game_dir, tmp_path):
    """Episodes that start with queued spawns (random compositions) replay exactly."""
    import numpy as np

    from warcraftsim import Wc3Game
    from warcraftsim.puffer.tasks import mirror_spawner

    sc = Scenario(units=(), victory="elimination", max_game_seconds=40, name="respawn_replay")
    setup = GameSetup(slots=[Agent("human"), Scripted("orc")], scenario=sc, step_seconds=0.5)
    spawns = mirror_spawner(units=(2, 3))(np.random.default_rng(3))
    live = {}
    with Wc3Game(setup, name="it_respawn") as g:
        g.reset()
        obs = g.reset(spawns=spawns)
        n = 0
        while not obs.game_over:
            live[obs.seq] = sorted((u.id, u.x, u.y, u.hp) for u in obs.units if u.alive)
            enemies = g.enemies()
            if n % 3 == 0 and enemies:
                for u in g.my_units():
                    g.attack(u, min(enemies, key=lambda e: e.hp))
            obs = g.step()
            n += 1
        live[obs.seq] = sorted((u.id, u.x, u.y, u.hp) for u in obs.units if u.alive)
        replay = g.instance.save_replay(tmp_path / "ep.w3g")
    with GameInstance(setup, name="it_respawn_play") as inst:
        obs = inst.play_replay(replay)
        played = {}
        for _ in range(500):
            if obs.game_over:
                break
            obs = inst.step()
            played[obs.seq] = sorted((u.id, u.x, u.y, u.hp) for u in obs.units if u.alive)
    assert len(live) > 10 and all(played.get(k) == v for k, v in live.items() if k > 0)
    assert obs.game_over


def test_mirror_selfplay_env(game_dir):
    """Both sides of MirrorSelfPlayEnv act with the full action set: zero-sum rewards, opposite
    outcomes, each side's masks allow what the casting script does."""
    from warcraftsim.agents.micro import micro_action
    from warcraftsim.puffer.tasks import get_task

    task = get_task("mirror_mix_abil_self_hp400")
    env = task.make_env("it_selfplay")
    env.setup.window, env.setup.step_seconds = (320, 240), 0.5
    try:
        task.reset(env)
        states, done, total = [{}, {}], False, [0.0, 0.0]
        while not done:
            acts = [micro_action("castnoop", env.sides[a], states[a], 5).ravel() for a in range(2)]
            for a, m in enumerate(task.action_mask(env)):
                m, act = m.reshape(5, -1), acts[a].reshape(5, 4)
                assert all(m[i, act[i, 0]] for i in range(5))
            _, rewards, done, _, outcomes = task.step(env, acts)
            total = [t + r for t, r in zip(total, rewards)]
        assert abs(total[0] + total[1]) < 1e-6 and outcomes[0] == -outcomes[1]
    finally:
        env.close()


def test_builtin_ai_takes_over_an_agent(game_dir):
    """StartAI: the built-in AI takes over an agent's player mid-game and carries on from its state;
    its orders are recorded like any AI player's, and agent commands to it are refused."""
    from warcraftsim.protocol import ImmediateOrder, StartAI
    setup = GameSetup(map="duelrush", slots=[Agent("human", handicap=50), BuiltinAI("orc", "normal", handicap=50)],
                      step_seconds=0.5, record_ai_orders=True)
    with GameInstance(setup, name="it_takeover") as g:
        obs = g.start()
        for _ in range(20):  # the agent does nothing for 10 s
            obs = g.step()
        food = obs.players[0].food_used
        assert not any(o.unit in {u.id for u in obs.units_of(0)} for o in obs.issued)
        obs = g.step([StartAI(0)])
        assert obs.command_results == [True] and not obs.players[0].is_agent
        mine, issued = {u.id for u in obs.units_of(0)}, 0
        for _ in range(60):
            obs = g.step()
            mine |= {u.id for u in obs.units_of(0)}
            issued += sum(o.unit in mine for o in obs.issued)
        assert issued > 10, "the AI's orders for the player it took over are recorded"
        assert obs.players[0].food_used > food, "it trains"
        hall = next(u for u in obs.units_of(0) if u.type_id == int.from_bytes(b"htow", "big"))
        obs = g.step([ImmediateOrder(hall.id, int.from_bytes(b"hpea", "big"))])
        assert obs.command_results == [False], "agent commands to a player the AI plays are refused"


def test_shim_unit_pass_matches_the_harness(game_dir, tmp_path, monkeypatch):
    """The pass over the units made by the shim (shim/units.c) writes what the harness's JASS
    wrote: a game played with the harness's pass, its replay played back with the shim's, every
    field of every unit and player the same on every step (and the units dropped, and the result)."""
    import dataclasses

    from warcraftsim.runtime.instance import BuiltinAI, GameInstance

    def snap(obs):
        return ({u.id: dataclasses.astuple(u) for u in obs.units},
                {i: dataclasses.astuple(p) for i, p in obs.players.items()}, obs.game_over)

    setup = GameSetup(map="duelrush", slots=[BuiltinAI("undead", "normal", handicap=50), BuiltinAI("human", "normal", handicap=50)],
                      step_seconds=0.5, max_game_seconds=120, victory="decisive", window=(320, 240), wait_floor_ms=5,
                      record_ai_orders=True)
    monkeypatch.setenv("W3SIM_UNITS", "0")
    live, played = {}, {}
    with GameInstance(setup, name="it_unitpass") as g:
        obs = g.start()
        while True:
            live[obs.seq] = snap(obs)
            if obs.game_over:
                break
            obs = g.step([])
        replay = g.save_replay(tmp_path / "game.w3g")
        assert "written by the harness" in (g.inst_dir / "shim.log").read_text()
        monkeypatch.setenv("W3SIM_UNITS", "1")
        obs = g.play_replay(replay)
        while True:
            played[obs.seq] = snap(obs)
            if obs.game_over:
                break
            obs = g.step([])
        assert "written by the shim" in (g.inst_dir / "shim.log").read_text()
    assert len(live) > 100 and sum(len(s[0]) for s in live.values()) > 5000
    assert [s for s in live if live[s] != played.get(s)] == []
