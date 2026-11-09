"""The actors' game threads share features' and featnative's module caches: one thread's first build of a cache must not
show another a half-made one, and a view's C arguments must point into the tables it keeps alive, whatever another thread
rebuilds meanwhile (ctypes releases the GIL during the C call, and Python may switch threads between any two lines)."""

import linecache
import sys

import numpy as np
import pytest

from warcraftsim.fullgame import featnative, features as fx


def _rows(n: int = 40, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = np.zeros((n, fx.ROW_COLS), np.int64)
    rows[:, fx.C_ID] = np.arange(1, n + 1)
    rows[:, fx.C_TYPE] = rng.choice([int.from_bytes(c.encode(), "big") for c in ("hfoo", "ugho", "uaco", "hpea", "Hpal")], n)
    rows[:, fx.C_OWNER] = rng.integers(0, 2, n)
    rows[:, fx.C_HP] = rng.integers(1, 400, n)
    rows[:, fx.C_MAXHP] = 400
    rows[:, fx.C_VIS] = 3
    return rows


@pytest.fixture()
def kinds():
    try:
        fx._kinds()
    except Exception as e:  # noqa: BLE001 (the game's tables: not on every machine)
        pytest.skip(f"no game data: {e}")
    return fx._KINDS


def test_kinds_published_whole(kinds, monkeypatch):
    """_kinds(): the dict goes into features._KINDS once it holds every kind ("immune" too). Published before its last
    entry, another thread's _kinds() returned it without "immune": spell_any's tables then raised KeyError."""
    real = fx.codes
    seen = []

    def codes(values):
        seen.append(fx._KINDS)  # (what another thread's _kinds() would get now)
        return real(values)
    monkeypatch.setattr(fx, "_KINDS", None)
    monkeypatch.setattr(fx, "codes", codes)
    got = fx._kinds()
    assert seen and all(k is None for k in seen)
    assert {"undead", "mechanical", "ancient", "immune"} <= set(got) and fx._KINDS is got


def test_spell_any_keeps_the_tables_it_points_to(kinds, monkeypatch):
    """Another game thread rebuilding spell_any's tables (its first call, or new kinds) while this view makes its argument
    array: the view keeps the tables its addresses point into (they had been the other thread's: this view's own freed
    while its C read them at every later step), and its answers stay the Python's."""
    if featnative.lib() is None:
        pytest.skip("no features library")
    rows = _rows()

    class Owner:
        pass
    first, other = Owner(), Owner()
    featnative.spell_any(fx, rows, 0, other)  # (the tables made once)
    path = featnative.__file__
    fired = []

    def rebuild():  # (another thread: new kinds, the same codes in a new dict, so new tables)
        monkeypatch.setattr(fx, "_KINDS", dict(fx._KINDS))
        featnative.spell_any(fx, rows, 1, other)

    def tracer(frame, event, arg):
        if frame.f_code.co_name != "spell_any" or frame.f_code.co_filename != path:
            return None

        def local(frame, event, arg):
            if event == "line" and not fired and "owner._spell_buf =" in linecache.getline(path, frame.f_lineno):
                fired.append(1)
                sys.settrace(None)
                rebuild()
                sys.settrace(tracer)
            return local
        return local
    sys.settrace(tracer)
    try:
        featnative.spell_any(fx, rows, 0, first)
    finally:
        sys.settrace(None)
    assert fired
    buf = first._spell_buf
    a, kept = buf[0], buf[5]
    assert list(a[4:14]) == list(kept[2][:10])  # (its C arguments point into the tables it keeps alive)
    want = fx.spell_any(rows, 0)  # (the Python pass)
    assert featnative.spell_any(fx, rows, 0, first) == want
