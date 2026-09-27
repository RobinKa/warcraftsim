import math
import struct

from warcraftsim.data.duelmap import FAST, Rules, _object_mods, duel_layout, parse_duel_name


def test_duel_layout():
    lay = duel_layout()
    (ax, ay), (bx, by) = lay["starts"]
    assert math.hypot(ax - bx, ay - by) == 4600
    for sx, sy in lay["starts"]:
        mine = min(lay["mines"], key=lambda m: math.hypot(m[0] - sx, m[1] - sy))
        assert 600 < math.hypot(mine[0] - sx, mine[1] - sy) < 800
        # melee initialization removes creeps within 1500 of a start location
        assert all(math.hypot(cx - sx, cy - sy) > 1500 for cx, cy, _ in lay["camps"])
        near = min(math.hypot(x - sx, y - sy) for x, y, _, _ in lay["trees"])
        assert 400 < near < 800  # lumber close by, room for the base
    for mx, my in lay["mines"]:  # a mine's footprint is clear of trees
        assert min(math.hypot(x - mx, y - my) for x, y, _, _ in lay["trees"]) > 250
    assert duel_layout()["trees"] == lay["trees"]  # the same map every time


def test_duel_names_and_object_data():
    assert parse_duel_name("duel") == (48, Rules()) and parse_duel_name("duelfast64") == (64, FAST)
    assert parse_duel_name("flat") is None
    data = _object_mods([("hfoo", [("uhpm", 105), ("ubld", 7)])], levels=False)
    version, n = struct.unpack_from("<ii", data)
    assert (version, n) == (2, 1) and data[8:12] == b"hfoo" and data[12:16] == b"\0\0\0\0"
    assert struct.unpack_from("<i", data, 16)[0] == 2
    assert data[20:24] == b"uhpm" and struct.unpack_from("<iii", data, 24) == (0, 105, 0)
    leveled = _object_mods([("Rhme", [("gtib", 20)])], levels=True)
    assert leveled[20:24] == b"gtib" and struct.unpack_from("<iiiii", leveled, 24) == (0, 0, 0, 20, 0)
    assert struct.unpack_from("<i", data, len(data) - 4)[0] == 0  # no new objects
