import pytest

from warcraftsim.data.objects import parse_slk, patch_slk, unit_table
from warcraftsim.data.terrain import PATH_NO_WALK, load_terrain, open_area_center

SLK = """ID;PWXL;N;E
B;X3;Y3;D0
C;X1;Y1;K"unitID"
C;X2;K"goldcost"
C;X1;Y2;K"hfoo"
C;X2;K135
C;X1;Y3;K"hpea"
C;X2;K75
E
"""


def test_parse_slk():
    assert parse_slk(SLK) == [{"unitID": "hfoo", "goldcost": "135"}, {"unitID": "hpea", "goldcost": "75"}]


def test_patch_slk():
    """Cells replaced in a table's text (the duel maps' rules: the game's tables, changed, in the map)."""
    out = patch_slk(SLK.replace("\n", "\r\n"), {"hpea": {"goldcost": 38}, "hfoo": {"goldcost": 67.5}})
    assert out.count("\r\n") == SLK.count("\n")  # the same lines, the same line ends
    assert parse_slk(out) == [{"unitID": "hfoo", "goldcost": "67.5"}, {"unitID": "hpea", "goldcost": "38"}]
    assert patch_slk(SLK, {}) == SLK
    for missing in ({"hkni": {"goldcost": 1}}, {"hfoo": {"lumbercost": 1}}):
        with pytest.raises(KeyError):
            patch_slk(SLK, missing)


def test_unit_table(game_dir):
    t = unit_table()
    assert t["hfoo"].gold == 135 and t["hfoo"].hp == 420 and t["hfoo"].race == "human"
    assert t["htow"].is_building and not t["hpea"].is_building


def test_terrain(game_dir):
    t = load_terrain(game_dir / "Maps" / "FrozenThrone" / "(2)EchoIsles.w3x")
    assert (t.width, t.height) == (129, 97) and t.pathing.shape == (384, 512)
    assert (t.offset_x, t.offset_y) == (-8192.0, -6144.0)
    x, y, clearance = open_area_center(t)
    r, c = t.cell(x, y)
    assert not t.pathing[r, c] & PATH_NO_WALK and clearance > 500


def test_flat_map(game_dir, tmp_path):
    import numpy as np

    from warcraftsim.data.flatmap import make_flat_map
    from warcraftsim.data.mpq import MpqArchive
    from warcraftsim.data.w3i import parse_w3i

    src = game_dir / "Maps" / "FrozenThrone" / "(2)EchoIsles.w3x"
    # full size: same outline, flat, no obstacles inside the border
    out = make_flat_map(src, tmp_path / "flat.w3x")
    t = load_terrain(out)
    assert (t.width, t.height) == (129, 97)
    assert np.all(t.corner_height == 0) and not t.corner_water.any()
    inner = t.pathing[16:-32, 24:-24]  # border: 4 bottom, 8 top, 6 left/right tiles
    assert np.all(inner == 0x40) and np.all(t.pathing[:16] == 0xCE)
    with MpqArchive(out) as m:
        assert m.read("war3map.doo")[12:16] == b"\0\0\0\0"  # no doodads
    # resized: 32 x 32 playable tiles around (0, 0)
    small = make_flat_map(src, tmp_path / "flat32.w3x", size=32)
    t = load_terrain(small)
    assert (t.width, t.height) == (45, 45) and (t.offset_x, t.offset_y) == (-2816.0, -2560.0)
    r, c = t.cell(0, 0)
    assert t.pathing[r, c] == 0x40 and t.pathing[r, t.cell(2100, 0)[1]] == 0xCE
    with MpqArchive(small) as m:
        info = parse_w3i(m.read("war3map.w3i"))
        script = m.read("war3map.j").decode("latin-1")
    assert info.playable_width == 32 and [(p.start_x, p.start_y) for p in info.players] == [(-1024, 0), (1024, 0)]
    assert "call DefineStartLocation( 1, 1024.0, 0.0 )" in script and "call CreateAllUnits" not in script.split(
        "function main")[1]
