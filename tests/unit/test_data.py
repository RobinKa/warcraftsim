from warcraftsim.data.objects import parse_slk, unit_table
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

    src = game_dir / "Maps" / "FrozenThrone" / "(2)EchoIsles.w3x"
    out = make_flat_map(src, tmp_path / "flat.w3x")
    t = load_terrain(out)
    assert (t.width, t.height) == (129, 97)
    assert np.all(t.corner_height == 0) and not t.corner_water.any()
    border = (t.pathing & 0x80) != 0
    assert np.all(t.pathing[~border] == 0x40) and border.any()
    with MpqArchive(out) as m:
        assert m.read("war3map.doo")[12:16] == b"\0\0\0\0"  # no doodads
        assert m.read("war3map.j") == MpqArchive(src).read("war3map.j")
