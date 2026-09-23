import pytest

from warcraftsim import paths


@pytest.fixture(scope="session")
def game_dir():
    if not (paths.GAME_DIR / "War3.mpq").exists():
        pytest.skip(f"Warcraft III 1.29 install not found at {paths.GAME_DIR}")
    if not paths.STORMLIB_PATH.exists():
        pytest.skip("StormLib not built (scripts/build_native.sh)")
    return paths.GAME_DIR
