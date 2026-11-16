"""The tinker's skills in the real game (2026-11-16, the user: "tinker only skilled passive"). Engineering Upgrade
('ANeg') replaces Pocket Factory, Cluster Rockets and Robo-Goblin with their level's versions ('ANs1'.., 'ANc1'..,
'ANg1'..); the harness learns and reads a hero skill through W3S_Skill, so a learn of the base code reaches the
replaced ability and its level reads. Places a level-10 tinker, learns the passive first and then the others, and
prints the levels per slot after each step and the game's answer to each learn (accepted only when the level rose:
Robo-Goblin learned again, at its last level, refused). `--cuts`: the runs' W3SIM_CUTS (with bit 2^31 the shim
applies a step's learns itself, shim/units.c cmd_apply, as in the runs).

    python scripts/tinker_probe.py [--harness path/to/w3sim.j] [--shim build/shim] [--cuts 17592186041849]
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--harness", type=Path, default=None, help="a harness source to use (default: the package's)")
    ap.add_argument("--shim", type=Path, default=None)
    ap.add_argument("--cuts", type=int, default=None, help="W3SIM_CUTS (the runs': 17592186041849)")
    ap.add_argument("--wine", default=str(Path.home() / "wc3/ge/GE-Proton10-34/files/bin"))
    a = ap.parse_args()
    from warcraftsim.runtime import wine
    wine.use_wine(a.wine, True)
    wine.ensure_template()
    from warcraftsim.data import mapbuild
    if a.harness is not None:
        src = a.harness.read_text()
        mapbuild.harness_source = lambda: src
    from warcraftsim.protocol import LearnSkill, Place
    from warcraftsim.runtime.instance import Agent, GameInstance, GameSetup
    slots = [Agent("human", handicap=100, difficulty="normal"), Agent("orc", handicap=100, difficulty="normal")]
    gs = GameSetup(map="(2)EchoIsles", slots=slots, step_seconds=0.5, draw=False, native_obs=False, d3d_thread=False,
                   render_threads=0, nice=19, warm_spare=False, window=(320, 240), max_game_seconds=0, victory="none",
                   cuts=a.cuts, **({"shim_dir": a.shim} if a.shim else {}))
    fail = 0
    with GameInstance(gs, name="tinkerprobe", timeout=600) as g:
        obs = g.start()
        p = obs.players[0]
        obs = g.step([Place(0, "Ntin", p.start_x + 300, p.start_y, hero_level=10)])
        tink = [u for u in obs.units if u.owner == 0 and u.type == "Ntin"]
        if not tink:
            print("RESULT no tinker placed")
            return 1
        tid = tink[0].id

        def levels(o):
            u = next(u for u in o.units if u.id == tid)
            return [lv for lv, _ in u.abilities], u.skill_points

        print("placed:", levels(obs))
        for code, want, took in (("ANeg", (0, 0, 1, 0), True), ("ANsy", (1, 0, 1, 0), True),
                                 ("ANcs", (1, 1, 1, 0), True), ("ANrg", (1, 1, 1, 1), True),
                                 ("ANeg", (1, 1, 2, 1), True), ("ANsy", (2, 1, 2, 1), True),
                                 ("ANrg", (2, 1, 2, 1), False)):
            obs = g.step([LearnSkill(tid, code)])
            answer = list(obs.command_results)[:1]
            obs = g.step([])
            got, pts = levels(obs)
            ok = tuple(got[:4]) == want and answer == [took]
            fail += not ok
            print(f"learn {code}: levels {got} points {pts} answer {answer} {'ok' if ok else f'WANT {want} {[took]}'}")
    print("RESULT", "ok" if not fail else f"{fail} wrong")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
