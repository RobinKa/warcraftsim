"""A hero placed in a timed form (late-game starts, late.capture): a state captured while the demon hunter was in
Metamorphosis ('Edmm', 60 s) or the alchemist in Chemical Rage ('Nalm', 'Nal2', 'Nal3') placed the form itself
(CreateUnit), which has no duration to end it. Places each form (and the tinker's Robo-Goblin, a toggle, and a
bear-form druid, a unit's toggle) with its skills, plays 90 s of game time and prints each one's type then: a timed form
that is still the form never turns back.

    python scripts/hero_form_probe.py [--harness path/to/w3sim.j] [--shim build/shim] [--cuts N]
"""
from __future__ import annotations

import argparse
from pathlib import Path

FORMS = (("Edmm", 6, ("AEme", "AEmb", "AEim", "AEev", "AEmb", "AEim")), ("Nalm", 5, ("ANcr", "ANhs", "ANab")),
         ("Nal2", 5, ("ANcr", "ANcr", "ANhs")), ("Nrob", 6, ("ANrg", "ANsy", "ANcs")), ("edcm", 0, ()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--harness", type=Path, default=None)
    ap.add_argument("--shim", type=Path, default=None)
    ap.add_argument("--cuts", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--wine", default=str(Path.home() / "wc3/ge/GE-Proton10-34/files/bin"))
    a = ap.parse_args()
    from warcraftsim.runtime import wine
    wine.use_wine(a.wine, True)
    wine.ensure_template()
    from warcraftsim.data import mapbuild
    if a.harness is not None:
        src = a.harness.read_text()
        mapbuild.harness_source = lambda: src
    from warcraftsim.protocol import Place
    from warcraftsim.runtime.instance import Agent, GameInstance, GameSetup
    slots = [Agent("nightelf", handicap=100, difficulty="normal"), Agent("orc", handicap=100, difficulty="normal")]
    gs = GameSetup(map="(2)EchoIsles", slots=slots, step_seconds=0.5, draw=False, native_obs=False, d3d_thread=False,
                   render_threads=0, nice=19, warm_spare=False, window=(320, 240), max_game_seconds=0, victory="none",
                   cuts=a.cuts, **({"shim_dir": a.shim} if a.shim else {}))
    with GameInstance(gs, name="formprobe", timeout=900) as g:
        obs = g.start()
        p = obs.players[0]
        before = {u.id for u in obs.units}
        cmds = [Place(0, t, p.start_x + 250 + 120 * k, p.start_y - 300, 1000, 1000, lv, sk)
                for k, (t, lv, sk) in enumerate(FORMS)]
        obs = g.step(cmds)
        placed = [u for u in obs.units if u.id not in before and u.owner == 0]
        print("placed:", [(u.id, u.type, getattr(u, "hero_level", None), [lv for lv, _ in (u.abilities or ())])
                          for u in placed])
        ids = {u.id: u.type for u in placed}
        for _ in range(int(a.seconds / 0.5)):
            obs = g.step([])
        now = {u.id: u.type for u in obs.units if u.id in ids}
        for i, t in ids.items():
            print(f"{t} after {a.seconds:.0f} s: {now.get(i)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
