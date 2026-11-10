"""The shaping's material values (trace.unit_values): what each unit type cost, from the game's tables."""

from types import SimpleNamespace

import pytest

from warcraftsim.fullgame import trace


def _code(c: str) -> str:
    return str(int.from_bytes(c.encode("latin-1"), "big"))


@pytest.fixture(scope="module")
def values():
    try:
        return trace.unit_values()
    except Exception as e:  # noqa: BLE001 (the game's tables: not on every machine)
        pytest.skip(f"no game data: {e}")


def test_a_shade_is_worth_its_acolyte(values):
    """The tables price a shade at 0 (an acolyte sacrificed for nothing): it counted 0, so each sacrifice read as 75 of
    material lost and an enemy shade as nothing to kill."""
    assert values[_code("ushd")] == values[_code("uaco")] == 75
    assert _code("uske") not in values and _code("hwat") not in values  # (summons stay 0: they come and go)
    assert values[_code("hkee")] == 1120 and values[_code("htow")] == 590  # (an upgraded hall: the tables' sum)


def test_material_counts_a_shade(values):
    acolyte = SimpleNamespace(owner=0, type_id=int(_code("uaco")), flags=0, hp=100, max_hp=100)
    shade = SimpleNamespace(owner=1, type_id=int(_code("ushd")), flags=0, hp=50, max_hp=100)
    m = trace.material(SimpleNamespace(units=[acolyte, shade]), values)
    assert m == {0: 75.0, 1: 37.5}
