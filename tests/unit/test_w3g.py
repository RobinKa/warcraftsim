"""The replay parser (data/w3g.py) and the human replays' converter's pure parts (fullgame/replays.py) on a
tiny 1.29 replay: a 63-second 1v1 on Echo Isles from warcraft3.info (its chat removed)."""

import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from warcraftsim.data import w3g

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "echo_1v1_6060.w3g"


@pytest.fixture(scope="module")
def data():
    return FIXTURE.read_bytes()


@pytest.fixture(scope="module")
def rep(data):
    return w3g.parse_replay(data)


def test_header_and_game_info(rep):
    h = rep.header
    assert (h.container, h.expansion, h.version, h.build, h.patch) == ("w3g", "W3XP", 29, 6060, "1.29")
    assert h.multiplayer and not h.wide_blocks and h.length_ms == rep.length_ms == 62880
    assert rep.map_path == "Maps\\FrozenThrone\\(2)EchoIsles.w3x"
    assert rep.map_checksum == 0xCCD6B5CA and rep.map_sha1 == "02c4ac24de1de084360bd4ddd1c848bf58743dfb"
    assert rep.settings["speed"] == "fast" and rep.settings["observers"] == "full"
    assert rep.creator == "GHost++" and rep.map_size == (116, 84)


def test_players_and_slots(rep):
    playing = rep.playing
    assert [(p.slot, p.team, p.race, p.random) for p in playing] == [(1, 1, "random", True), (2, 0, "random", True)]
    assert [p.id for p in playing] == [3, 4] and playing[1].host
    observers = [p for p in rep.players.values() if p.observer]
    assert len(observers) == 1 and observers[0].team == 24  # (1.29: the observers' team is 24)
    assert rep.matchup == "random vs random"
    assert len(rep.slots) == 12 and sum(s.status == "used" for s in rep.slots) == 3


def test_actions(rep):
    assert len(rep.actions) == 941 and rep.unknown_actions == 0
    times = [a.time for a in rep.actions]
    assert times == sorted(times) and times[-1] <= rep.length_ms
    first = rep.actions[0]
    assert first.kind == w3g.SELECT and first.flags == 1 and len(first.units) == 1
    builds = [(a.player, a.order, int(a.x), int(a.y)) for a in rep.actions if a.kind == w3g.ABILITY_POINT
              and a.item >= w3g.TYPE_CODE]
    assert (3, "oalt", -4576, 3104) in builds and (4, "obar", 4288, 2496) in builds
    smart = next(a for a in rep.actions if a.kind == w3g.ABILITY_TARGET)
    assert smart.order == "851971" and smart.target and smart.source is None  # (851971: smart)
    assert any(a.kind == w3g.ABILITY_TARGET and a.target is None for a in rep.actions)  # (on the ground)
    assert [(lv.player, lv.result) for lv in rep.leaves] == [(2, "left"), (3, "left"), (4, "left")]


def test_summary(rep):
    s = w3g.summarize(rep)
    assert (s["winner_team"], s["winner_how"]) == (0, "the others left")  # (player 3 left first)
    by = {p["player"]: p for p in s["players"]}
    assert [by[3]["race_played"], by[4]["race_played"]] == ["orc", "orc"]  # (random: from the starting units)
    assert by[4]["won"] and not by[3]["won"] and by[3]["left_ms"] == 62240
    assert (1600, "build", "oalt", -4576, 3104) in by[3]["build"]
    assert by[3]["counts"]["train"]["opeo"] >= 2 and 100 < by[3]["apm"] < by[4]["apm"] < 400
    z = w3g.strategy(by[4])
    assert z["build_order"][:3] == ["oalt", "otrb", "obar"] and z["made"]["opeo"] >= 2


def test_rewrite_round_trip_and_checksums(data, rep):
    assert w3g.rewrite(data) == data  # (the same blocks, header CRC and block checksums)
    out = w3g.rewrite(data, map_path="Maps\\x.w3x", checksum=0x12345678, sha1=bytes(range(20)),
                      objects=lambda t: (t[0] + 7, t[1] + 9))
    r2 = w3g.parse_replay(out)
    assert (r2.map_path, r2.map_checksum, r2.map_sha1) == ("Maps\\x.w3x", 0x12345678, bytes(range(20)).hex())
    assert r2.creator == rep.creator and len(r2.actions) == len(rep.actions)
    for a, b in zip(rep.actions, r2.actions):
        assert b.units == tuple((i + 7, s + 9) for i, s in a.units)
        assert b.target == (None if a.target is None else (a.target[0] + 7, a.target[1] + 9))
    head = bytearray(out[:68])
    crc = struct.unpack_from("<I", head, 64)[0]
    struct.pack_into("<I", head, 64, 0)
    assert zlib.crc32(bytes(head)) == crc
    size, _, check = struct.unpack_from("<HHI", out, 68)
    block = out[76:76 + size]
    fold = lambda c: (c ^ (c >> 16)) & 0xFFFF  # noqa: E731
    assert check == fold(zlib.crc32(out[68:72] + b"\0" * 4)) | fold(zlib.crc32(block)) << 16


def test_settings_encoding_round_trip():
    raw = bytes(range(256)) + b"\x00\x01\xfe\xff" * 9
    enc = w3g.encode_settings(raw)
    assert all(b % 2 for b in enc) and w3g.decode_settings(enc) == raw


def test_containers(data, rep):
    # Reforged 2.0's GRBN: a 128-byte header (the metadata's size at 12), zlib'd metadata, then the replay
    meta = zlib.compress(b"\x08\x02metadata")
    grbn = b"GRBN" + struct.pack("<3I", 2, 11, len(meta)) + bytes(112) + meta + data
    r = w3g.parse_replay(grbn)
    assert r.header.container == "GRBN" and len(r.actions) == len(rep.actions)
    # Reforged's 12-byte block headers (32-bit sizes)
    header, d = w3g.decompress(data)
    pos, blocks = 68, []
    for _ in range(header.blocks):
        size, full, check = struct.unpack_from("<HHI", data, pos)
        blocks.append(struct.pack("<III", size, full, check) + data[pos + 8:pos + 8 + size])
        pos += 8 + size
    wide = w3g.parse_replay(data[:68] + b"".join(blocks))
    assert wide.header.wide_blocks and len(wide.actions) == len(rep.actions)
    with pytest.raises(w3g.ReplayError):
        w3g.parse_replay(b"<!DOCTYPE html>" + bytes(100))


def test_map_checksums():
    assert w3g._xoro(b"") == 0 and w3g._xoro(b"\x01") == 8 and w3g._xoro(b"\x01\0\0\0\x01") == ((8 ^ 1) << 3)
    files = {"war3map.j": b"function main takes nothing returns nothing\nendfunction\n", "war3map.w3e": b"W3E!"}
    a = w3g.map_checksums(files, b"common", b"blizzard")
    assert a == w3g.map_checksums({**files, "scripts\\war3map.j": b"ignored"}, b"common", b"blizzard")
    assert a != w3g.map_checksums({**files, "war3map.w3e": b"W3E?"}, b"common", b"blizzard")


def test_map_checksums_of_the_stock_map(game_dir, rep):
    from warcraftsim.fullgame.replays import stock_checksums
    xoro, sha1 = stock_checksums()
    assert (xoro, sha1.hex()) == (rep.map_checksum, rep.map_sha1)


def test_object_kinds():
    assert w3g.object_kinds()["oalt"] == ("building", "Altar of Storms")
    assert w3g.object_kind("hfoo") == "unit" and w3g.object_kind("Hpal") == "hero"
    assert w3g.object_kind("Rhme") == "upgrade" and w3g.object_kind("AHbz") == "skill"
    assert w3g.object_kind("hslv") == "item"  # (healing salve: an item, though it looks like a unit)
    assert w3g.object_kind("Xzzz") == "hero" and w3g.object_kind("AXzz") == "skill"  # (not in the table: by shape)
    assert w3g.object_kind("zz") == ""
    assert w3g.fourcc(0x68666F6F) == "hfoo" and w3g.item_name(851971) == "851971"


# ---- fullgame/replays.py ---------------------------------------------------------------------------

def test_skip_reasons(rep):
    from warcraftsim.fullgame.replays import skip_reason
    stock = (rep.map_checksum, bytes.fromhex(rep.map_sha1))
    assert skip_reason(rep, stock) is None
    assert "checksum" in skip_reason(rep, (1, bytes(20)))
    assert "version" in skip_reason(w3g.Replay(w3g.Header("w3g", 1, "W3XP", 30, 6061, True, 0, 0, False)), stock)


def test_stock_map_match(rep):
    import copy

    from warcraftsim.fullgame.replays import match_map, skip_reason
    sums = (rep.map_checksum, bytes.fromhex(rep.map_sha1))
    stock = {"(2)Other": (1, bytes(20)), "(2)EchoIsles": sums}
    assert match_map(rep, stock) == "(2)EchoIsles" and skip_reason(rep, stock) is None
    blank = copy.copy(rep)
    blank.map_checksum = 0xFFFFFFFF  # (some replays carry no classic checksum: the SHA-1 decides)
    assert match_map(blank, stock) == "(2)EchoIsles"
    other = copy.copy(rep)
    other.map_sha1 = "00" * 20  # (another version of the map, or another game version's scripts)
    assert match_map(other, stock) is None and "checksum" in skip_reason(other, stock)
    assert match_map(other, stock, any_version=True) == "(2)EchoIsles"
    newer = copy.copy(other)
    newer.header = w3g.Header("w3g", 1, "W3XP", 30, 6061, True, 0, 0, False)
    assert "version" in skip_reason(newer, stock) and skip_reason(newer, stock, any_version=True) is None
    assert skip_reason(other, stock, other_maps=True) is None  # (NetEase's map versions: by the file name)
    assert "version" in skip_reason(newer, stock, other_maps=True)


def test_version_shift():
    from types import SimpleNamespace as NS

    from warcraftsim.fullgame.replays import version_shift
    ogre = int.from_bytes(b"ogre", "big")
    rep = w3g.Replay(w3g.Header("w3g", 1, "W3XP", 28, 6059, True, 0, 0, False))
    rep.players = {3: w3g.Player(3, "a", slot=1)}
    rep.actions = [w3g.Action(500, 3, w3g.SUBGROUP, item=ogre, target=(14907, 14907))]
    obs = NS(units=[NS(id=7, owner=1, type_id=ogre, x=0.0, y=0.0), NS(id=8, owner=24, type_id=0, x=0.0, y=0.0)])
    tags = [["T", "1", "7", "15387", "15387"], ["T", "1", "8", "14000", "14000"]]
    fn, offsets = version_shift(rep, obs, tags)
    assert offsets == (0, 480)
    assert fn((14907, 14907)) == (15387, 15387)  # (the 1.29 engine's 480 more objects before the start)
    assert fn((12000, 15500)) == (12000, 15980)  # a slot from before the preplaced units' last: only its count
    tags[0][3:] = ["14907", "14907"]
    assert version_shift(rep, obs, tags) == (None, (0, 0))  # (the same ids: nothing to shift)


def test_rewrite_version(data):
    out = w3g.rewrite(data, version=(30, 6061))
    h = w3g.parse_replay(out).header
    assert (h.version, h.build) == (30, 6061)
    assert w3g.parse_replay(w3g.rewrite(out, version=(29, 6060))).actions == w3g.parse_replay(data).actions


def test_object_id_shift():
    from warcraftsim.fullgame.replays import shift_of
    fn, n = shift_of([(15874, 15880), (15889, 15895)])
    assert n == 16
    assert fn((12292, 12292)) == (12292, 12292)  # made before the harness: as it was
    assert fn((15291, 16638)) == (15291, 16654)  # a slot used before, made after: only its count shifts
    assert fn((15900, 16000)) == (15916, 16016)  # a new slot, made after
    with pytest.raises(ValueError):
        shift_of([(15874, 15880), (15890, 15895)])  # (the harness took a used slot: no plain shift)


def test_remap_players():
    from warcraftsim.fullgame.replays import remap_players
    unit = lambda owner, vis: (0, 1, 2, owner, 0, 0, 0, 1, 1, 0, 0, 0, 0, vis, 0, 0, 0, 0)  # noqa: E731
    rec = {"units": [unit(2, 0b101), unit(0, 0b001), unit(15, 0b100)],
           "players": [(0, 0, 500), (0, 2, 400)], "events": [(3, 9, -3, 1, 1)]}
    remap_players(rec, [2, 0])  # (the replay's player in slot 2 first)
    assert [(r[3], r[13]) for r in rec["units"]] == [(0, 0b11), (1, 0b10), (15, 0b01)]
    assert rec["players"] == [(0, 1, 500), (0, 0, 400)] and rec["events"] == [(3, 9, -1, 1, 1)]


def _row(t, uid, typ, owner, flags, x=0, y=0, level=0):
    return (t, uid, typ, owner, x, y, 0, 100, 100, 0, 0, 0, flags, 0, 0, level, 0, 0)


def test_derived_events():
    from warcraftsim.fullgame.replays import PRODUCING, derive_events
    barracks, footman, peasant = w3g.TYPE_CODE + 1, 0x68666F6F, 0x68706561
    units = np.asarray([
        _row(0, 10, peasant, 0, 4),
        _row(1, 10, peasant, 0, 4), _row(1, 20, barracks, 0, 2 | 8),  # construction starts
        _row(2, 10, peasant, 0, 4), _row(2, 20, barracks, 0, 2),  # and ends
        _row(3, 10, peasant, 0, 4), _row(3, 20, barracks, 0, 2 | PRODUCING),  # training
        _row(4, 10, peasant, 0, 4 | 1024), _row(4, 20, barracks, 0, 2), _row(4, 30, footman, 0, 0, 50, 0),
    ], np.int64)
    heroes = np.zeros((0, 16), np.int64)
    orders = [(3, 20, footman, 0, 0, 0, 0)]  # the barracks' train order (a label)
    ev = derive_events(units, heroes, 4, [(4, 9, -1, 0x52686D65, 1)], orders, set())
    assert (1, 2, 20, barracks, 0) in ev and (2, 3, 20, barracks, 0) in ev
    assert (3, 5, 20, footman, 0) in ev and (4, 6, 20, 30, footman) in ev
    assert (4, 1, 10, 0, peasant) in ev  # (the killer unknown)
    assert (4, 9, 20, 0x52686D65, 1) in ev  # research done: at the player's building
    assert not any(e[1] == 7 for e in ev)  # (the unit came out: no cancel)


def test_derived_orders():
    from warcraftsim.fullgame.replays import derive_orders
    a, b, mine = (100, 100), (101, 101), (50, 50)
    actions = [w3g.Action(0, 1, w3g.SELECT, flags=1, units=(a, b)), w3g.Action(0, 1, w3g.SUBGROUP, 0x68706561, target=a),
               w3g.Action(600, 1, w3g.ABILITY_TARGET, 851971, x=10.0, y=20.0, target=mine),
               w3g.Action(1100, 1, w3g.ABILITY_POINT, 0x68626172, x=300.0, y=400.0)]  # build a barracks ('hbar')
    rep = w3g.Replay(w3g.Header("w3g", 1, "W3XP", 29, 6060, True, 0, 0, False), actions=actions)
    tags = {a: [(0, 7)], b: [(0, 8)], mine: [(0, 3)]}
    here = {7: (0x68706561, 0, 0, 0, 4, 0), 8: (0x68706561, 0, 0, 0, 4, 0), 3: (0x6E676F6C, 15, 0, 0, 0, 0)}
    later = {**here, 8: (0x68706561, 0, 0, 0, 4, 0x68626172)}  # (8 is on its way to build)
    states = [here, here, here, later]
    rows = derive_orders(rep, [1], [0, 500, 1000, 1500], tags, {}, lambda t: states[t], w3g.object_kinds())
    assert (1, 7, 851971, 2, 10, 20, 3) in rows and (1, 8, 851971, 2, 10, 20, 3) in rows  # smart: every unit
    assert rows[-1] == (2, 8, 0x68626172, 1, 300, 400, 0)  # the build: the worker that went


def test_reforged_records(data, rep):
    """Reforged's protobuf records before the slots (0x38, 0x39) and 2.0's action 0x7A (the unit, the
    ability and the order before an order)."""
    header, d = w3g.decompress(data)
    at = d.index(b"\x19", d.index(b"PlayerTwo"))  # (the game start record follows the last player)
    extra = b"\x38\x04" + struct.pack("<I", 2) + b"\x08\x01" + b"\x39\x03" + struct.pack("<I", 3) + b"abc"
    r = w3g.Replay(header)
    w3g._records(r, d[:at] + extra + d[at:], w3g._parse_info(r, d[:at] + extra + d[at:]), True)
    assert len(r.slots) == len(rep.slots) and len(r.actions) == len(rep.actions)
    block = (b"\x7a" + struct.pack("<II", 12249, 12249) + b"euqA" + b"aeph"
             + b"\x10\x42\x00" + b"aeph" + b"\xff" * 8)
    out = []
    w3g._actions(r, block, 1, 500, 99, out)
    assert [(a.kind, a.order, a.target) for a in out] == [(0x7A, "hpea", (12249, 12249)), (w3g.ABILITY, "hpea", None)]
    assert w3g.fourcc(out[0].item2) == "Aque"
    assert w3g.Header("w3g", 1, "W3XP", 10200, 7000, True, 0, 0, True).patch == "2.0"
    assert w3g.Header("w3g", 1, "W3XP", 10036, 6116, True, 0, 0, True).patch == "1.36"


def test_other_map_version_ids():
    from warcraftsim.fullgame.replays import id_map, preplaced_offsets
    # NetEase's Last Refuge: its trees as ours, 3 more objects before the preplaced units, starting units as those
    f = id_map(first_unit=11736, preplaced_off=-3, first_player=15505, player_off=-3)
    assert f((4013, 4013)) == (4013, 4013) and f((11739, 11739)) == (11736, 11736)
    assert f((15508, 15508)) == (15505, 15505) and f((15720, 15726)) == (15717, 15723)
    # NetEase's Echo Isles: 96 more objects before the starting units
    g = id_map(first_unit=11700, preplaced_off=-3, first_player=15387, player_off=-99)
    assert g((12295, 12295)) == (12292, 12292) and g((15486, 15486)) == (15387, 15387)
    assert id_map(11736, 0, 15505, 0) is None
    rep = w3g.Replay(w3g.Header("w3g", 1, "W3XP", 29, 6060, True, 0, 0, False))
    rep.actions = [w3g.Action(0, 1, w3g.ABILITY_TARGET, x=100.0, y=200.0, target=(11763, 11763)),
                   w3g.Action(10, 1, w3g.ABILITY_TARGET, x=100.0, y=200.0, target=(11763, 11763)),
                   w3g.Action(20, 1, w3g.ABILITY_TARGET, x=-500.0, y=0.0, target=(11786, 11786)),
                   w3g.Action(30, 1, w3g.ABILITY_TARGET, x=900.0, y=900.0, target=(15000, 15000))]
    assert preplaced_offsets(rep, [(110.0, 190.0, 11760), (-500.0, 10.0, 11783)]) == {-3: 2}
