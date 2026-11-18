"""--order-masks in the real game (2026-11-17, bug finder 10's refusals): what the game takes and refuses where the masks
(fullgame/ordermasks.py) say it does. Places heroes and units of both players on Echo Isles (obs-full on, the runs'
cuts) and sends each order once, printing the game's answer against the mask's:

  * bit 1, spell targets: frost nova on the lich itself, its own ghoul and ziggurat and an enemy building (refused), an
    enemy grunt (taken); storm bolt and mana burn on the hero itself (refused). And what the mask leaves, printed: a
    peon inside an orc burrow (loaded) or a gold mine (hidden: inside ~1 s at a time) as a target (taken in 2 of 3)
  * bit 2, toggles: immolation and mana shield off while off and on while on (refused), and the buff obs-full's events
    give each while on (the mask's codes: BEim, BNms); the tinker's way back unmorphed and the robo-goblin's morph again
    (refused). Defend's buffs are printed: none shows (no mask for it)
  * bit 4, learns: a ghoul's learn (refused); a level-6 archmage's mass teleport (taken)

    python scripts/order_masks_probe.py [--shim build/shim] [--cuts 17592186041849]
"""
from __future__ import annotations

import argparse
from pathlib import Path

IDS = {"frostnova": 852226, "thunderbolt": 852095, "manaburn": 852179, "immolation": 852177, "unimmolation": 852178,
       "defend": 852055, "undefend": 852056, "manashieldon": 852589, "manashieldoff": 852590, "robogoblin": 852656,
       "unrobogoblin": 852657, "harvest": 852018, "battlestations": 852099}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shim", type=Path, default=None)
    ap.add_argument("--cuts", type=int, default=None, help="W3SIM_CUTS (the runs': 17592186041849)")
    ap.add_argument("--wine", default=str(Path.home() / "wc3/ge/GE-Proton10-34/files/bin"))
    a = ap.parse_args()
    from warcraftsim.runtime import wine
    wine.use_wine(a.wine, True)
    wine.ensure_template()
    from warcraftsim.protocol import ImmediateOrder, LearnSkill, Place, SetResearch, TargetOrder
    from warcraftsim.runtime.instance import Agent, GameInstance, GameSetup
    slots = [Agent("undead", handicap=100, difficulty="normal"), Agent("orc", handicap=100, difficulty="normal")]
    gs = GameSetup(map="(2)EchoIsles", slots=slots, step_seconds=0.5, draw=False, native_obs=False, d3d_thread=False,
                   render_threads=0, nice=19, warm_spare=False, window=(320, 240), max_game_seconds=0, victory="none",
                   obs_full=True, cuts=a.cuts, **({"shim_dir": a.shim} if a.shim else {}))
    fail = []
    buffs: dict[int, set] = {}  # (obs-full's events: a unit's state anew, 101, then its buffs, 102s)

    def check(what: str, got, want) -> None:
        ok = got == want
        if not ok:
            fail.append(what)
        print(f"{what}: {got} {'ok' if ok else f'WANT {want}'}", flush=True)

    with GameInstance(gs, name="ordermaskprobe", timeout=900) as g:
        def step(cmds):
            o = g.step(cmds)
            for e in o.events:
                if int(e.kind) == 101:
                    buffs[e.a] = set()
                elif int(e.kind) == 102 and e.a in buffs:
                    buffs[e.a].add(e.b)
            return o

        obs = g.start()
        p1 = obs.players[1]
        mine = min((u for u in obs.units if u.type == "ngol"),
                   key=lambda u: (u.x - p1.start_x) ** 2 + (u.y - p1.start_y) ** 2)
        peons = [u for u in obs.units if u.owner == 1 and u.type == "opeo"]
        hall = next(u for u in obs.units if u.owner == 1 and u.type == "ogre")
        # the enemy's peons to its mine (one inside it at a time: hidden), a burrow beside it; our heroes and units
        obs = step([TargetOrder(u.id, IDS["harvest"], mine.id) for u in peons] + [SetResearch(0, "Rhde", 1)])
        x, y = (mine.x + p1.start_x) / 2, (mine.y + p1.start_y) / 2
        before = {u.id for u in obs.units}
        placed = [("Ulic", 5, ("AUfn", "AUfn", "AUfn")), ("Ulic", 5, ("AUfn", "AUfn", "AUfn")), ("Hmkg", 3, ("AHtb", "AHtb")),
                  ("Edem", 3, ("AEmb", "AEim")), ("Nngs", 3, ("ANms", "ANms")),
                  ("Hamg", 6, ("AHbz", "AHbz", "AHwe", "AHab", "AHbz")), ("Ntin", 6, ("ANsy", "ANcs", "ANrg")),
                  ("ugho", 0, ()), ("hfoo", 0, ()), ("uzig", 0, ())]
        cmds = [Place(0, t, x - 200 + 60 * (k % 5), y - 150 - 80 * (k // 5), 1000, 1000, lv, sk)
                for k, (t, lv, sk) in enumerate(placed)]
        cmds += [Place(1, "ogru", x + 150, y + 150), Place(1, "otrb", (mine.x + x) / 2 + 150, (mine.y + y) / 2)]
        obs = step(cmds)
        new: dict[str, list] = {}
        for u in obs.units:
            if u.id not in before:
                new.setdefault(u.type, []).append(u.id)
        print("placed:", sorted((t, len(v)) for t, v in new.items()), flush=True)
        (lich, lich2), (mk,), (dh,), (naga,), (amg,), (tink,) = (new[t] for t in ("Ulic", "Hmkg", "Edem", "Nngs", "Hamg",
                                                                                 "Ntin"))
        ghoul, foot, zig, grunt, burrow = new["ugho"][0], new["hfoo"][0], new["uzig"][0], new["ogru"][0], new["otrb"][0]

        def answer(cmd) -> bool:
            nonlocal obs
            obs = step([cmd])
            got = list(obs.command_results)[:1]
            obs = step([])
            return got[0] if got else None

        def buff_names(i: int) -> list:
            return sorted(b.to_bytes(4, "big").decode("latin-1", "replace") for b in buffs.get(i, ()))

        # bit 1: frost nova's targets (the grunt first, before our heroes kill it, by the second lich; the refused by the
        # first: a cast taken starts its cooldown)
        nova = IDS["frostnova"]
        check("frost nova on an enemy grunt", answer(TargetOrder(lich2, nova, grunt)), True)
        for what, t in (("itself", lich), ("own ghoul", ghoul), ("own ziggurat", zig), ("enemy great hall", hall.id)):
            check(f"frost nova on {what}", answer(TargetOrder(lich, nova, t)), False)
        check("storm bolt on the mountain king itself", answer(TargetOrder(mk, IDS["thunderbolt"], mk)), False)
        check("mana burn on the demon hunter itself", answer(TargetOrder(dh, IDS["manaburn"], dh)), False)
        for _ in range(10):  # (5 s: a peon inside the mine)
            obs = step([])
        obs = step([ImmediateOrder(burrow, IDS["battlestations"])])  # (peons near it board it: hidden and loaded)
        for _ in range(6):
            obs = step([])
        enemy_peons = [(u.id, int(u.flags)) for u in obs.units if u.owner == 1 and u.type == "opeo"]
        print("enemy peons' flags:", enemy_peons, flush=True)
        loaded = [i for i, f in enemy_peons if f & 32]
        in_mine = [i for i, f in enemy_peons if f & 16 and not f & 32]
        # (the mask leaves these: the game took frost nova on a burrowed peon, and on one in the mine but refused storm
        # bolt on another there; a miner is inside for ~1 s at a time, so the answer depends on when: printed, no check)
        if loaded:
            print("  frost nova on an enemy peon in a burrow (loaded):", answer(TargetOrder(lich, nova, loaded[0])),
                  flush=True)
        if in_mine:
            print("  storm bolt on an enemy peon in the gold mine (hidden):",
                  answer(TargetOrder(mk, IDS["thunderbolt"], in_mine[0])), flush=True)
        # bit 2: toggles, and their buffs in obs-full's events
        check("defend on (its research placed)", answer(ImmediateOrder(foot, IDS["defend"])), True)
        obs = step([])
        print(f"  defend on: the footman's buffs {buff_names(foot) if foot in buffs else 'unknown'}", flush=True)
        check("defend off", answer(ImmediateOrder(foot, IDS["undefend"])), True)
        for name, unit, on, off, buff in (("immolation", dh, "immolation", "unimmolation", "BEim"),
                                          ("mana shield", naga, "manashieldon", "manashieldoff", "BNms")):
            print(f"  {name}: buffs before {buff_names(unit) if unit in buffs else 'unknown'}", flush=True)
            check(f"{name} off while off", answer(ImmediateOrder(unit, IDS[off])), False)
            check(f"{name} on", answer(ImmediateOrder(unit, IDS[on])), True)
            obs = step([])
            print(f"  {name} on: buffs {buff_names(unit)}", flush=True)
            check(f"{name} on: its buff {buff} in the events", buff in buff_names(unit), True)
            check(f"{name} on while on", answer(ImmediateOrder(unit, IDS[on])), False)
            check(f"{name} off", answer(ImmediateOrder(unit, IDS[off])), True)
            obs = step([])
            check(f"{name} off: no {buff}", buff in buff_names(unit), False)
        check("tinker's way back, unmorphed", answer(ImmediateOrder(tink, IDS["unrobogoblin"])), False)
        check("tinker's robo-goblin", answer(ImmediateOrder(tink, IDS["robogoblin"])), True)
        for _ in range(4):
            obs = step([])
        print("  the tinker's form:", next((u.type for u in obs.units if u.id == tink), None), flush=True)
        check("robo-goblin's morph again", answer(ImmediateOrder(tink, IDS["robogoblin"])), False)
        # bit 4: learns
        check("a ghoul's learn", answer(LearnSkill(ghoul, "AUfn")), False)
        h = next(u for u in obs.units if u.id == amg)
        print(f"  archmage: level {h.hero_level}, points {h.skill_points}, abilities {[lv for lv, _ in h.abilities]}",
              flush=True)
        check("a level-6 archmage's mass teleport", answer(LearnSkill(amg, "AHmt")), True)
        h = next(u for u in obs.units if u.id == amg)
        print(f"  archmage: points {h.skill_points}, abilities {[lv for lv, _ in h.abilities]}", flush=True)
    print("RESULT", "ok" if not fail else f"{len(fail)} wrong: {fail}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
