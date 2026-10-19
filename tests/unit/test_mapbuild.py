import pytest

from warcraftsim.data.mapbuild import HarnessConfig, MapBuildError, SpawnSpec, inject_harness, pjass_check
from warcraftsim.data.mpq import MpqArchive
from warcraftsim import paths


@pytest.fixture(scope="module")
def echo_script(game_dir):
    with MpqArchive(game_dir / "Maps" / "FrozenThrone" / "(2)EchoIsles.w3x") as m:
        return m.read("war3map.j").decode("latin-1")


def _pjass_available():
    if not paths.PJASS_PATH.exists():
        pytest.skip("pjass not built")


def test_melee_injection_validates(echo_script):
    _pjass_available()
    out = inject_harness(echo_script, HarnessConfig(step_seconds=0.5, agent_players=(0, 1)))
    assert "call W3S_StartingAI()" in out and "call MeleeStartingAI(" not in out
    assert "call W3S_InitVictoryDefeat()" in out
    assert "constant integer W3S_CFG_AGENT_MASK = 3" in out
    assert "constant real W3S_CFG_STEP_S = 0.5000" in out
    pjass_check(out)


def test_scenario_injection_validates(echo_script):
    _pjass_available()
    cfg = HarnessConfig(scenario=True, victory="elimination", scripted_players=(1,),
                        spawn=(SpawnSpec(0, "hfoo", -100, 0), SpawnSpec(1, "ogru", 100, 0, 180, max_hp=100)),
                        resources=((0, 100, 50),), clear_area=(0, 0, 500))
    out = inject_harness(echo_script, cfg)
    assert "call W3S_SpawnUnit(0, 'hfoo', -100.0, 0.0, 270.0, 0)" in out
    assert "call W3S_SpawnUnit(1, 'ogru', 100.0, 0.0, 180.0, 100)" in out
    assert "constant integer W3S_CFG_SCRIPTED_MASK = 2" in out
    pjass_check(out)


def test_bad_inputs():
    with pytest.raises(MapBuildError):
        inject_harness("globals\nendglobals\nfunction main takes nothing returns nothing\nendfunction\n", HarnessConfig())
    with pytest.raises(ValueError):
        HarnessConfig(agent_players=(12,)).agent_mask()
    with pytest.raises(ValueError):
        HarnessConfig(spawn=(SpawnSpec(0, "footman", 0, 0),)).spawn_code()


def test_scenario_spawns_keep_their_settings():
    from warcraftsim.scenario import Scenario

    sc = Scenario.skirmish(["hfoo"], ["hfoo"], max_hp=100, center=(64.0, -32.0))
    units = sc.absolute_units()
    assert [u.max_hp for u in units] == [100, 100] and units[0].x == 64.0 - 350.0 and units[0].y == -32.0


def test_hero_ability_table_in_the_harness():
    from warcraftsim.data.mapbuild import HarnessConfig, _split_harness, harness_source

    cfg = HarnessConfig(hero_abilities={"Hmkg": ("AHtc", "AHtb", "AHbh", "AHav")}, order_names=("stop",))
    _, body = _split_harness(harness_source(), cfg)
    assert "call SaveInteger(w3s_abil, 'Hmkg', 1, 'AHtb')" in body


def test_pairs_of_players_per_game(echo_script):
    """Games with two players of their own each (HarnessConfig.pairs): the map's main makes only
    the first game's starting units (the harness's own call of MeleeStartingUnits stays)."""
    _pjass_available()
    out = inject_harness(echo_script, HarnessConfig(step_seconds=0.5, agent_players=(0, 2, 4), melee_reset="W3S_DuelCreeps", pairs=3))
    assert "constant integer W3S_CFG_PAIRS = 3" in out and "constant integer W3S_CFG_AGENT_MASK = 21" in out
    main = out[out.index("function Trig_Melee_Initialization_Actions"):]
    assert "call W3S_StartingUnits()" in main and "call MeleeStartingUnits(" not in main
    assert out.count("call MeleeStartingUnits()") == 1  # (the harness's reset of the same players)
    one = inject_harness(echo_script, HarnessConfig(step_seconds=0.5, agent_players=(0,)))
    assert "constant integer W3S_CFG_PAIRS = 1" in one and "call W3S_StartingUnits()" not in one[one.index("function Trig_Melee_Initialization_Actions"):]


def test_observer_injection(echo_script):
    """The observing harness (people's replays): the map's own melee code stays as it is."""
    from warcraftsim.data.mapbuild import inject_observer
    _pjass_available()
    out = inject_observer(echo_script, HarnessConfig(step_seconds=0.5), upgrades=("Rhme", "Rome"))
    assert "call MeleeStartingAI(" in out and "call MeleeInitVictoryDefeat(" in out
    assert "W3S_StartingAI" not in out and "call W3S_Upgrade('Rome')" in out
    main = out[out.index("function main takes nothing returns nothing"):]
    assert main.index("call RunInitializationTriggers(") < main.index("call W3S_Init()") < main.index("endfunction")
    pjass_check(out)


def test_observer_makes_nothing_in_a_step():
    """w3obs.j: a replay's later units keep their object ids only if a step makes no object (every one
    is made in W3S_Init, between its probes) and nothing changes the game: no unit or player natives,
    no randomness, no event triggers (each firing makes objects)."""
    import re
    from warcraftsim.data.mapbuild import observer_source
    src = observer_source()
    bodies = dict(re.findall(r"^function (\w+) takes .*?$(.*?)^endfunction", src, re.M | re.S))
    makes = re.compile(r"\b(Create\w+|InitHashtable|Condition|Filter|GetWorldBounds|Rect|Location|TimerStart|"
                       r"TriggerRegister\w+|TriggerAdd\w+|EnumDestructablesInRect)\s*\(")
    changes = re.compile(r"\b(IssueOrder|Issue\w+Order|SetPlayerState|SetUnit\w+|KillUnit|RemoveUnit|UnitAdd\w+|"
                         r"GetRandom\w+|PickMeleeAI|RestartGame|EndGame|DestroyTrigger|CreateUnit\w*)\s*\(")
    called = {"W3S_Step"}
    todo = ["W3S_Step"]
    while todo:  # every function a step calls (directly or as a callback)
        for name in re.findall(r"\b(W3S_\w+)\b", bodies[todo.pop()]):
            if name in bodies and name not in called:
                called.add(name)
                todo.append(name)
    assert "W3S_Init" not in called and {"W3S_CPass", "W3S_WriteObs", "W3S_PollResearch"} <= called
    for name in called:
        assert not makes.search(bodies[name]), name
    for name, body in bodies.items():
        assert not changes.search(body), name
    init = bodies["W3S_Init"]
    assert init.strip().startswith("set w3s_probe0 = CreateGroup()")
    last = max(m.start() for m in makes.finditer(init))  # (the last object made: the second probe)
    assert last == init.index("set w3s_probe1 = CreateGroup()") + len("set w3s_probe1 = ")


def test_seed_first_in_main(echo_script):
    """A fixed seed goes in before Blizzard's and the melee initialization (the built-in AI draws at once)."""
    _pjass_available()
    out = inject_harness(echo_script, HarnessConfig(step_seconds=0.5, agent_players=(0, 1), seed=1234))
    main = out[out.index("function main takes nothing returns nothing"):]
    assert main.index("call SetRandomSeed(1234)") < main.index("call InitBlizzard(")
    assert "SetRandomSeed" not in inject_harness(echo_script, HarnessConfig(step_seconds=0.5, agent_players=(0, 1)))
    pjass_check(out)
