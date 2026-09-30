import math
import struct

from warcraftsim.data.duelmap import FAST, Rules, _object_mods, duel_layout, parse_duel_name


def test_duel_layout():
    small = duel_layout(40, base_x=1500.0)
    for sx, sy in small["starts"]:
        assert all(math.hypot(cx - sx, cy - sy) > 2000 for cx, cy, _ in small["camps"])
    assert all(abs(cy) < 40 * 64 - 3 * 128 for _, cy, _ in small["camps"])
    lay = duel_layout()
    (ax, ay), (bx, by) = lay["starts"]
    assert math.hypot(ax - bx, ay - by) == 4600
    for sx, sy in lay["starts"]:
        mine = min(lay["mines"], key=lambda m: math.hypot(m[0] - sx, m[1] - sy))
        assert 600 < math.hypot(mine[0] - sx, mine[1] - sy) < 800
        # melee initialization removes creeps within 1500 of a start location; closer than 2000
        # they walked into a base
        assert all(math.hypot(cx - sx, cy - sy) > 2000 for cx, cy, _ in lay["camps"])
        near = min(math.hypot(x - sx, y - sy) for x, y, _, _ in lay["trees"])
        assert 400 < near < 800  # lumber close by, room for the base
    for mx, my in lay["mines"]:  # a mine's footprint is clear of trees
        assert min(math.hypot(x - mx, y - my) for x, y, _, _ in lay["trees"]) > 250
    assert duel_layout()["trees"] == lay["trees"]  # the same map every time


def test_duel_names_and_object_data():
    assert parse_duel_name("duel") == (48, Rules(), 2300.0) and parse_duel_name("duelfast64") == (64, FAST, 2300.0)
    assert parse_duel_name("duelrush")[0] == 40 and parse_duel_name("duelrush")[2] == 1500.0
    assert parse_duel_name("duelrush")[1].speed == 7.0 and parse_duel_name("duelrushx5")[1].speed == 5.0
    assert parse_duel_name("flat") is None
    data = _object_mods([("hfoo", [("uhpm", 105), ("ubld", 7)])], levels=False)
    version, n = struct.unpack_from("<ii", data)
    assert (version, n) == (2, 1) and data[8:12] == b"hfoo" and data[12:16] == b"\0\0\0\0"
    assert struct.unpack_from("<i", data, 16)[0] == 2
    assert data[20:24] == b"uhpm" and struct.unpack_from("<iii", data, 24) == (0, 105, 0)
    leveled = _object_mods([("Rhme", [("gtib", 20)])], levels=True)
    assert leveled[20:24] == b"gtib" and struct.unpack_from("<iiiii", leveled, 24) == (0, 0, 0, 20, 0)
    assert struct.unpack_from("<i", data, len(data) - 4)[0] == 0  # no new objects


def test_rules_as_tables_match_the_object_data(game_dir):
    """The rush rules written as the game's tables say what the object data said: every change of
    the object files is the value in its table's cell, and nothing else in the tables changed."""
    from warcraftsim.data.duelmap import ABILITY_FIELDS, RUSH, UNIT_FIELDS, UPGRADE_FIELDS, rules_files
    from warcraftsim.data.mpq import GameArchives
    from warcraftsim.data.objects import parse_slk
    tables, objects = rules_files(RUSH, True), rules_files(RUSH, False)
    assert not any(name.startswith("war3map.w3") for name in tables)
    assert {k: v for k, v in objects.items() if not k.startswith("war3map.w3")} == \
        {k: v for k, v in tables.items() if not k.startswith("Units\\")}  # the constants and AI scripts: the same

    def read(data: bytes, levels: bool) -> dict:
        """{(object, field[, level, column]): value} of an object data file."""
        n, at, out = struct.unpack_from("<i", data, 4)[0], 8, {}
        for _ in range(n):
            oid, count = data[at:at + 4].decode(), struct.unpack_from("<i", data, at + 8)[0]
            at += 12
            for _ in range(count):
                field, kind = data[at:at + 4].decode(), struct.unpack_from("<i", data, at + 4)[0]
                at += 8
                where = ()
                if levels:
                    where = struct.unpack_from("<ii", data, at)
                    at += 8
                out[(oid, field) + where] = struct.unpack_from("<f" if kind == 2 else "<i", data, at)[0]
                at += 8
        return out

    ids = {"UnitBalance": "unitBalanceID", "UnitWeapons": "serpent", "UnitData": "unitID", "UpgradeData": "upgradeid",
           "AbilityData": "alias"}
    new = {n: {r[c]: r for r in parse_slk(tables[f"Units\\{n}.slk"].decode("latin-1")) if c in r} for n, c in ids.items()}
    with GameArchives() as g:
        old = {n: {r[c]: r for r in parse_slk(g.read(f"Units\\{n}.slk").decode("latin-1")) if c in r} for n, c in ids.items()}
    want: dict[tuple, float] = {}
    for (oid, field), v in read(objects["war3map.w3u"], False).items():
        want[(*UNIT_FIELDS[field], oid)] = v
    for (oid, field, _, _), v in read(objects["war3map.w3q"], True).items():
        want[("UpgradeData", UPGRADE_FIELDS[field], oid)] = v
    for (oid, field, level, col), v in read(objects["war3map.w3a"], True).items():
        want[("AbilityData", f"{ABILITY_FIELDS[field]}{level}" if col == 0 else f"Data{'ABCD'[col - 1]}{level}", oid)] = v
    assert len(want) > 2500
    for (table, col, oid), v in want.items():
        assert abs(float(new[table][oid][col]) - v) <= 1e-4 * max(1.0, abs(v)), (table, oid, col)
    changed = {(t, c, o) for t in old for o in old[t] for c in old[t][o] if old[t][o][c] != new[t][o].get(c)}
    assert changed <= set(want) and all(old[t].keys() == new[t].keys() for t in old)
