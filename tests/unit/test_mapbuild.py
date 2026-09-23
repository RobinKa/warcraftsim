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
                        spawn=(SpawnSpec(0, "hfoo", -100, 0), SpawnSpec(1, "ogru", 100, 0, 180)),
                        resources=((0, 100, 50),), clear_area=(0, 0, 500))
    out = inject_harness(echo_script, cfg)
    assert "call W3S_SpawnUnit(1, 'ogru', 100.0, 0.0, 180.0)" in out
    assert "constant integer W3S_CFG_SCRIPTED_MASK = 2" in out
    pjass_check(out)


def test_bad_inputs():
    with pytest.raises(MapBuildError):
        inject_harness("globals\nendglobals\nfunction main takes nothing returns nothing\nendfunction\n", HarnessConfig())
    with pytest.raises(ValueError):
        HarnessConfig(agent_players=(12,)).agent_mask()
    with pytest.raises(ValueError):
        HarnessConfig(spawn=(SpawnSpec(0, "footman", 0, 0),)).spawn_code()
