import struct

from warcraftsim.data.wgc import CTRL_OBSERVER, CTRL_USER, Difficulty, Race, Wgc, WgcSlot


def test_layout_matches_spec():
    w = Wgc(
        "Maps\\w3sim\\echo.w3x",
        [WgcSlot.observer(2), WgcSlot.computer(0, "human", "insane"), WgcSlot.computer(1, "orc", 0)],
        game_speed=64,
        flags=0,
    )
    data = w.to_bytes()
    assert data[:12] == struct.pack("<III", 1, 0, 64)
    path_end = data.index(b"\0", 12)
    assert data[12:path_end] == b"Maps\\w3sim\\echo.w3x"
    assert struct.unpack_from("<I", data, path_end + 1) == (3,)
    first = struct.unpack_from("<7I", data, path_end + 5)
    assert first == (2, 0, Race.HUMAN, 0, 100, CTRL_USER | CTRL_OBSERVER, 1)
    assert Wgc.from_bytes(data) == w


def test_custom_ai_flags():
    s = WgcSlot.computer(1, "undead", Difficulty.EASY, ai_script="Scripts\\null.ai")
    assert s.controller == 0xC and s.ai_script == "Scripts\\null.ai"
    assert not s.is_user and not s.is_observer
    assert Race.parse("elf") is Race.NIGHTELF and Difficulty.parse("hard") is Difficulty.INSANE
