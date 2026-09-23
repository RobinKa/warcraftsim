import shutil

from warcraftsim.data.mpq import GameArchives, MpqArchive


def test_game_archives_layering(game_dir):
    with GameArchives(game_dir) as g:
        common = g.read("Scripts\\common.j")
        assert b"native Preloader" in common
        assert g.source_of("Scripts\\common.j") == "War3x.mpq"  # TFT overrides RoC
        assert "Units/UnitData.slk" in g  # forward slashes accepted


def test_map_rewrite_roundtrip(game_dir, tmp_path):
    src = game_dir / "Maps" / "FrozenThrone" / "(2)EchoIsles.w3x"
    dst = tmp_path / "echo.w3x"
    shutil.copyfile(src, dst)
    with MpqArchive(dst, writable=True) as m:
        script = m.read("war3map.j")
        m.write("war3map.j", script + b"\n// edited\n")
        m.write("w3sim\\extra.txt", b"hello")
    with MpqArchive(dst) as m:
        assert m.read("war3map.j").endswith(b"// edited\n")
        assert m.read("w3sim\\extra.txt") == b"hello"
        assert "war3map.w3e" in m
