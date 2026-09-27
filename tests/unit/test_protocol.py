from pathlib import Path

import pytest

from warcraftsim.protocol import (MAILBOX_BASE, PROTOCOL_VERSION, Build, Camera, ImmediateOrder, PointOrder,
                                  ProtocolError, Restart, Result, UnitFlags, VisArea, VisClear, VisLine, VisMark,
                                  action_file_text, command_ops, encode_commands, fourcc, parse_observation, rawcode)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "obs_echoisles.txt"
V = str(PROTOCOL_VERSION)


def _obs_text(*tokens: str) -> str:
    lines = ["function PreloadFiles takes nothing returns nothing", ""]
    lines += [f'\tcall Preload( "{t}" )' for t in tokens]
    lines += ["\tcall PreloadEnd( 0.0 )", "", "endfunction"]
    return "\r\n".join(lines)


def _rec(tag: str, *fields: int) -> list[str]:
    total = (sum(fields) + 2 ** 31) % 2 ** 32 - 2 ** 31
    return [tag, *map(str, fields), str(total)]


def _unit(hid: int, x: int) -> list[str]:
    return _rec("U", hid, fourcc("hfoo"), 0, x, 5, 90, 420, 420, 0, 0, 0, 0, 1, 0)


def test_parse_real_observation():
    obs = parse_observation(FIXTURE.read_text(encoding="latin-1"))
    assert obs.full and obs.is_first and not obs.game_over
    assert obs.orders["move"] == 851986 and obs.orders["attack"] == 851983
    assert set(obs.players) == {0, 1}
    assert obs.players[0].is_agent and obs.players[0].result == Result.PLAYING
    mine = obs.units_of(0)
    assert {u.type for u in mine} == {"htow", "hpea"}
    hall = next(u for u in mine if u.type == "htow")
    assert hall.is_structure and hall.hp == hall.max_hp == 1500
    assert any(u.type == "ngol" and u.resource > 0 for u in obs.units)


def test_parse_minimal_and_hero_record():
    # hero extra: level, xp, skill points, 6 items, then (level, cooldown in 0.1 s) per ability slot
    hero = _rec("U", 1049, fourcc("Hpal"), 0, 100, -200, 270, 650, 650, 255, 255, 0, int(UnitFlags.HERO), 1, 0,
                3, 120, 1, fourcc("ankh"), 0, 0, 0, 0, 0, 2, 37, 1, 0, 0, 0, 0, 0)
    text = _obs_text("V", V, *_rec("T", 7, 250, 0, 1), *hero, *_rec("E", 1, 5, 6, fourcc("hfoo")), *_rec("C", 1),
                     *_rec("C", 0), "X")
    obs = parse_observation(text)
    u = obs.units[0]
    assert u.type == "Hpal" and u.is_hero and u.hero_level == 3 and rawcode(u.items[0]) == "ankh"
    assert u.abilities == ((2, 3.7), (1, 0.0), (0, 0.0), (0, 0.0))
    assert (u.x, u.y) == (100, -200)
    assert obs.events[0].kind == 1 and obs.command_results == [True, False]


def test_foreign_preload_lines_are_ignored():
    text = _obs_text("V", V, "Sound\\\\Buildings\\\\Orc\\\\OrcBuildingBirthWhat1.wav", *_rec("T", 0, 63, 0, 1), "X")
    assert parse_observation(text).game_ms == 63


def test_truncated_observation_is_rejected():
    with pytest.raises(ProtocolError):
        parse_observation(_obs_text("V", V, *_rec("T", 0, 63, 0, 1)))
    with pytest.raises(ProtocolError):
        parse_observation(_obs_text("V", "2", *_rec("T", 0, 63, 0, 1), "X"))


def test_command_encoding_and_action_file():
    ints = encode_commands([PointOrder(10, 851986, -100.4, 200), ImmediateOrder(11, fourcc("hfoo")),
                            Build(12, "hhou", 0, 0), Restart()])
    assert ints[:5] == [1, 10, 851986, 65536 - 100, 65536 + 200]
    assert ints[5:8] == [3, 11, fourcc("hfoo")]
    assert ints[8:13] == [4, 12, fourcc("hhou"), 65536, 65536]
    assert ints[13:] == [99]
    text = action_file_text(ints)
    assert f"Player(PLAYER_NEUTRAL_PASSIVE), {MAILBOX_BASE}, {len(ints)})" in text
    assert "PreloadEnd" not in text  # PreloadEnd would wait for every recorded preload
    assert text.count("SetPlayerTechMaxAllowed") == len(ints) + 1


def test_video_marker_commands():
    marks = [VisClear(), VisMark(7, (70, 255, 120), "A3"), VisMark(8, (215, 215, 220), "E12", ring=False),
             VisLine(7, (80, 190, 245), x=-10, y=20), VisLine(7, (245, 80, 60), target=8),
             VisLine(7, (245, 110, 210), ability="AOws", radius=250.4), VisArea(0, 5, 150, (255, 215, 0))]
    ints = encode_commands(marks)
    assert ints[1:7] == [81, 7, (70 << 16) | (255 << 8) | 120, 1, 0, 3]
    assert ints[7:13] == [81, 8, (215 << 16) | (215 << 8) | 220, 0, 2, 12]
    assert ints[13:22] == [82, 7, (80 << 16) | (190 << 8) | 245, 0, 65536 - 10, 65536 + 20, 0, 0, 0]
    assert ints[22:31][3] == 1 and ints[22:31][6] == 8  # to a unit
    assert ints[31:40][3] == 2 and ints[31:40][7:] == [fourcc("AOws"), 250]  # at the caster
    assert command_ops(ints) == [80, 81, 81, 82, 82, 82, 83]
    # only commands for watching are sent during replay playback
    assert all(c.playback for c in marks + [Camera(0, 0)])
    assert not PointOrder(1, 2, 3, 4).playback and not Restart().playback


def test_rawcode_roundtrip():
    assert rawcode(fourcc("hfoo")) == "hfoo" and rawcode(0) == ""


def test_checksummed_records_and_repair():
    head = ["V", V, *_rec("T", 3, 750, 0, 1)]
    good = _obs_text(*head, *_unit(1, 100), *_unit(2, 200), "X")
    obs = parse_observation(good)
    assert [u.x for u in obs.units] == [100, 200] and obs.damaged_records == 0
    # a stray numeric entry inside a record is removed
    u1 = _unit(1, 100)
    stray = u1[:5] + ["-3663"] + u1[5:]
    obs = parse_observation(_obs_text(*head, *stray, *_unit(2, 200), "X"))
    assert [u.x for u in obs.units] == [100, 200] and obs.damaged_records == 0
    # a stray entry between records is skipped
    obs = parse_observation(_obs_text(*head, *u1, "77", *_unit(2, 200), "X"))
    assert [u.x for u in obs.units] == [100, 200]
    # an unrepairable record is dropped and counted, the rest survives
    broken = u1[:3] + ["1", "2"] + u1[3:]
    obs = parse_observation(_obs_text(*head, *broken, *_unit(2, 200), "X"))
    assert [u.x for u in obs.units] == [200] and obs.damaged_records == 1


def test_delta_merge():
    from warcraftsim.protocol import merge_observation

    table = {}
    full = parse_observation(_obs_text("V", V, *_rec("T", 0, 0, 0, 1), *_unit(1, 100), *_unit(2, 200), "X"))
    assert [u.id for u in merge_observation(table, full).units] == [1, 2]
    delta = parse_observation(_obs_text("V", V, *_rec("T", 1, 250, 0, 0), *_unit(2, 250), *_unit(3, 300),
                                        *_rec("R", 1), "X"))
    assert not delta.full and delta.removed == [1]
    merged = merge_observation(table, delta)
    assert [(u.id, u.x) for u in merged.units] == [(2, 250), (3, 300)]


def test_decode_commands_roundtrip():
    from warcraftsim.protocol import (Camera, ImmediateOrder, PointOrder, Restart, SetResources, TargetOrder,
                                      decode_commands, encode_commands)

    cmds = [PointOrder(12, 3, -250.0, 400.0), Restart(), TargetOrder(12, 2, 40), SetResources(0, 5, 6),
            ImmediateOrder(7, 1), Camera(10.0, -20.0)]
    assert decode_commands(encode_commands(cmds)) == [c for c in cmds if not isinstance(c, (Restart, SetResources))]


def test_queue_spawn_encoding():
    from warcraftsim.protocol import QueueSpawn, decode_commands, encode_commands

    ints = encode_commands([QueueSpawn(1, "Hmkg", -300, 90, 180, 250, 3)])
    assert ints[0] == 92 and len(ints) == 8 and ints[6:] == [250, 3]
    assert decode_commands(ints) == []  # not a unit order


def test_queue_spawn_with_skills():
    from warcraftsim.protocol import QueueSpawn, command_ops, encode_commands

    ints = encode_commands([QueueSpawn(1, "Hmkg", -300, 90, hero_level=3, skills=("AHtb", "AHtb", "AHtc")),
                            Restart()])
    assert command_ops(ints) == [92, 93, 93, 93, 99]
    assert ints[8:10] == [93, fourcc("AHtb")] and ints[12:14] == [93, fourcc("AHtc")]
