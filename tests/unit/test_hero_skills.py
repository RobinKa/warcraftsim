"""A hero's skills as it has them now (the tinker's Engineering Upgrade replaces four of its abilities with their level's
versions): every learn and every read of a skill's level goes through the code the hero has now, in the harness
(W3S_Skill) and in the shim (skill_now), and a learn is answered accepted only when the skill's level rose."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HARNESS = (ROOT / "warcraftsim" / "harness" / "w3sim.j").read_text()
SHIM = (ROOT / "shim" / "units.c").read_text()


def _function(src: str, name: str) -> str:
    at = src.index(f"function {name} takes")
    return src[at:src.index("endfunction", at)]


def _c_function(src: str, head: str) -> str:
    at = src.index(head)
    end = src.index("\n}\n", at)
    return src[at:end]


def test_every_learn_goes_through_the_current_code():
    """SelectHeroSkill of the base code did nothing once Engineering Upgrade replaced it (the user, 2026-11-16): each call
    in the harness and in the shim takes W3S_Skill's / skill_now's code. The shim applies the agents' learns itself in
    every run (W3SIM_CUTS bit 2^31, cmd_apply): its learn had kept the base code (scripts/tinker_probe.py --cuts: 6 of
    7 steps wrong)."""
    calls = re.findall(r"SelectHeroSkill\(([^\n]*)\)", HARNESS)
    assert calls, "no learn in the harness"
    for c in calls:
        assert "W3S_Skill(" in c or re.fullmatch(r"\w+, (a|id)", c), c
    placed = _function(HARNESS, "W3S_Placed")
    assert "set id = W3S_Skill(w3s_placed, id)" in placed
    apply = _function(HARNESS, "W3S_ApplyOne")
    op5 = apply[apply.index("elseif op == 5 then"):apply.index("elseif op == 6 then")]
    assert "set a = W3S_Skill(u, w3s_cmd[at + 2])" in op5 and "SelectHeroSkill(u, a)" in op5
    shim_calls = re.findall(r"SelectHeroSkill_, u, (\w+)\)", SHIM)
    assert shim_calls == ["a"], shim_calls
    cmd = _c_function(SHIM, "static int cmd_apply(void)")
    assert "int a = skill_now(u, c[2])" in cmd


def test_a_learn_is_accepted_only_when_its_level_rose():
    """op 5 had answered ok whenever the hero had a point (a learn the game ignored counted as accepted: the tinker's 224
    learns of Robo-Goblin, none refused). Now ok: the level after above the level before, in both."""
    apply = _function(HARNESS, "W3S_ApplyOne")
    op5 = apply[apply.index("elseif op == 5 then"):apply.index("elseif op == 6 then")]
    assert "set was = GetUnitAbilityLevel(u, a)" in op5
    assert "set ok = GetUnitAbilityLevel(u, a) > was" in op5
    assert "set ok = true" not in op5
    cmd = _c_function(SHIM, "static int cmd_apply(void)")
    assert "before = n2(GetUnitAbilityLevel, u, a)" in cmd
    assert "ok = n2(GetUnitAbilityLevel, u, a) > before" in cmd


def test_the_levels_read_through_the_current_code():
    ser = _function(HARNESS, "W3S_SerUnit")
    assert ser.index("set a = W3S_Skill(u, a)") < ser.index("set alevel[i] = GetUnitAbilityLevel(u, a)")
    rec = _c_function(SHIM, "static void unit_record(int u)")
    assert rec.index("a = skill_now(u, a);") < rec.index("alevel[i] = n2(GetUnitAbilityLevel, u, a);")


def test_the_replacements_are_the_games(game_dir):
    """W3S_Skill's and skill_now's table is Engineering Upgrade's (AbilityData.slk: DataC-F name each replaced ability and
    its level-1 version, the level-L one ending in L), and no other ability in the game replaces abilities so."""
    from warcraftsim.data.mpq import GameArchives
    from warcraftsim.data.objects import parse_slk
    with GameArchives() as g:
        rows = parse_slk(g.read("Units\\AbilityData.slk").decode("latin-1"))
    by = {r.get("alias"): r for r in rows}
    eg = by["ANeg"]
    pairs = [str(eg.get(f"Data{c}1")).split(",") for c in "CDEF"]
    assert [p[0] for p in pairs] == ["ANsy", "ANcs", "ANrg", "ANde"]
    levels = int(eg["levels"])
    for base, first in pairs:
        assert first == base[:2] + first[2] + "1"
        for k in range(1, levels + 1):
            assert f"{first[:3]}{k}" in by and by[f"{first[:3]}{k}"].get("code") == base
    skill = _function(HARNESS, "W3S_Skill")
    for base, first in pairs:
        assert f"'{base}'" in skill and f"'{first[:3]}0' + e" in skill
    now = _c_function(SHIM, "static int skill_now(int u, int a)")
    for base, first in pairs:
        assert "RAW('A', 'N', '{}', '{}')".format(*base[2:]) in now and f"'{first[2]}'" in now
    assert [r.get("alias") for r in rows if r.get("code") == "ANeg"] == ["ANeg"]
