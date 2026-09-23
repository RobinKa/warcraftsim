from pathlib import Path

import pytest

from warcraftsim.protocol import (MAILBOX_BASE, Build, ImmediateOrder, PointOrder, ProtocolError, Restart, Result,
                                  UnitFlags, action_file_text, encode_commands, fourcc, parse_observation, rawcode)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "obs_echoisles.txt"


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
    hero = _rec("U", 1049, fourcc("Hpal"), 0, 100, -200, 270, 650, 650, 255, 255, 0, int(UnitFlags.HERO), 1, 0,
                3, 120, 1, fourcc("ankh"), 0, 0, 0, 0, 0)
    text = _obs_text("V", "3", *_rec("T", 7, 250, 0, 1), *hero, *_rec("E", 1, 5, 6, fourcc("hfoo")), *_rec("C", 1),
                     *_rec("C", 0), "X")
    obs = parse_observation(text)
    u = obs.units[0]
    assert u.type == "Hpal" and u.is_hero and u.hero_level == 3 and rawcode(u.items[0]) == "ankh"
    assert (u.x, u.y) == (100, -200)
    assert obs.events[0].kind == 1 and obs.command_results == [True, False]


def test_foreign_preload_lines_are_ignored():
    text = _obs_text("V", "3", "Sound\\\\Buildings\\\\Orc\\\\OrcBuildingBirthWhat1.wav", *_rec("T", 0, 63, 0, 1), "X")
    assert parse_observation(text).game_ms == 63


def test_truncated_observation_is_rejected():
    with pytest.raises(ProtocolError):
        parse_observation(_obs_text("V", "3", *_rec("T", 0, 63, 0, 1)))
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


def test_rawcode_roundtrip():
    assert rawcode(fourcc("hfoo")) == "hfoo" and rawcode(0) == ""


def test_checksummed_records_and_repair():
    head = ["V", "3", *_rec("T", 3, 750, 0, 1)]
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
    full = parse_observation(_obs_text("V", "3", *_rec("T", 0, 0, 0, 1), *_unit(1, 100), *_unit(2, 200), "X"))
    assert [u.id for u in merge_observation(table, full).units] == [1, 2]
    delta = parse_observation(_obs_text("V", "3", *_rec("T", 1, 250, 0, 0), *_unit(2, 250), *_unit(3, 300),
                                        *_rec("R", 1), "X"))
    assert not delta.full and delta.removed == [1]
    merged = merge_observation(table, delta)
    assert [(u.id, u.x) for u in merged.units] == [(2, 250), (3, 300)]
