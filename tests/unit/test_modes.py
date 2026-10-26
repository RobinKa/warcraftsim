"""More kinds of strategy conditioning (modes.py): draws, the input, and each kind's violations."""

import random
from types import SimpleNamespace

import numpy as np
import pytest

from warcraftsim.fullgame import modes


def code(s: str) -> int:
    return int.from_bytes(s.encode(), "big")


def _row(i, owner, x, y, flags=0, typ="hfoo", level=0):
    r = np.zeros(32, np.int64)
    r[1], r[modes.C_TYPE], r[modes.C_OWNER], r[modes.C_X], r[modes.C_Y] = i, code(typ), owner, x, y
    r[modes.C_FLAGS], r[modes.C_HLEVEL] = flags, level
    return r


def test_draw_and_vector():
    rng = random.Random(0)
    for _ in range(200):
        m = modes.draw(rng, 0.6)
        assert not any(a in m and b in m for a, b in modes.EXCLUSIVE)
        for k, v in m.items():
            (lo, hi), _ = modes.PARAMS[k]
            assert lo <= v <= hi
    z = modes.vector({"no_rush": 450.0, "workers": 24.0})
    i, j = modes.KINDS.index("no_rush"), modes.KINDS.index("workers")
    assert z.shape == (modes.MDIM,) and z[2 * i] == 1 and z[2 * i + 1] == pytest.approx(0.5) and z[2 * j + 1] == pytest.approx(0.8)
    assert z.sum() == pytest.approx(1 + 0.5 + 1 + 0.8) and not modes.vector(None).any()


def test_violations():
    food = {code("hfoo"): 2}
    tk = modes.Tracker({"no_rush": 300.0, "expand": 240.0, "scout": 0.0, "workers": 4.0, "army": 4.0, "harass": 2.0},
                       0, (3000.0, 0.0), food, {code("hpea")})
    home = [_row(1, 0, -3000, 0, modes.STRUCTURE, "htow"), _row(2, 0, -2900, 0, modes.WORKER, "hpea"),
            _row(3, 0, -2800, 0)]
    assert tk.step(np.stack(home), [], 10, 60.0) == 0.0
    there = home[:2] + [_row(3, 0, 2000, 0)]  # an army unit at the enemy's base for 60 s of the 300
    assert tk.step(np.stack(there), [], 10, 120.0) == pytest.approx(0.2)  # (also near its start: scouted)
    enemy_peon = _row(9, 1, 2900, 0, modes.WORKER, "hpea")
    kill = SimpleNamespace(kind=modes.DEATH, a=9, b=3)
    tk.step(np.stack(there + [enemy_peon]), [kill], 10, 180.0)
    assert tk.kills == 1
    v = tk.step(np.stack(home), [], 10, 250.0)  # no second hall by 240: a whole violation
    assert tk.v["expand"] == 1.0 and tk.v["no_rush"] == pytest.approx(0.4)
    v = tk.step(np.stack(home + [_row(4, 0, -2700, 0, modes.WORKER, "hpea")]), [], 10, 600.0)  # 2 of 4 workers at 480
    assert tk.v["workers"] == pytest.approx(0.5) and tk.v["army"] == pytest.approx(0.5) and tk.v["harass"] == pytest.approx(0.5)
    assert v == pytest.approx(sum(tk.v.values())) and tk.report()["workers"] == [4.0, 0.5]


def test_expand_needs_a_hall_at_another_mine():
    """With its own start known, "expand" is met by a hall far from it at a gold mine, not by a second one at home."""
    tk = modes.Tracker({"expand": 240.0}, 0, (3000.0, 0.0), {}, {code("hpea")}, own_start=(-3000.0, 0.0))
    home = [_row(1, 0, -3000, 0, modes.STRUCTURE, "htow"), _row(2, 15, -3300, 0, modes.STRUCTURE, "ngol"),
            _row(3, 15, 0, 1500, modes.STRUCTURE, "ngol")]
    tk.step(np.stack(home + [_row(4, 0, -2600, 300, modes.STRUCTURE, "htow")]), [], 10, 200.0)
    assert "expand" not in tk.done
    tk.step(np.stack(home + [_row(5, 0, 300, 1400, modes.STRUCTURE, "htow")]), [], 10, 220.0)
    assert "expand" in tk.done


def test_comp_draw_and_vector():
    """--z-comp: the composition kinds drawn after the others (none: the generator's draws as before), their block."""
    a, b = random.Random(3), random.Random(3)
    for _ in range(50):
        assert modes.draw(a, 0.3) == modes.draw(b, 0.3, 0.0)
    rng, seen = random.Random(1), {k: set() for k in modes.COMP_KINDS}
    for _ in range(300):
        m = modes.draw(rng, 0.0, 0.5)
        assert not set(m) - set(modes.COMP_KINDS)
        for k, v in m.items():
            (lo, hi), _ = modes.PARAMS[k]
            assert lo <= v <= hi
            seen[k].add(v)
    assert seen["forbid"] == {0.0, 1.0} and seen["variety"] == {3.0, 4.0} and len(seen["cap"]) > 5
    z = modes.comp_vector({"cap": 0.45, "forbid": 1.0, "variety": 3.0, "no_rush": 300.0})
    assert z.shape == (modes.CDIM,) and z.tolist() == pytest.approx([1, 0.45, 1, 1, 1, 0.6])
    assert not modes.comp_vector(None).any() and not modes.comp_vector({"rush": 200.0}).any()


def test_comp_adapts_the_target():
    """A forbidden unit leaves the target's items and mix; a cap caps the mix (idempotent; a target unchanged
    without them is the same object)."""
    st = {"race": "human", "t": {"hfoo": 60.0, "hrif": 200.0, "Hpal": 90.0}, "n": {"hfoo": 12, "hrif": 4, "hkni": 4}}
    assert modes.adapt(st, {"no_rush": 300.0}, "human") is st and modes.adapt(st, {}, "human") is st
    out = modes.adapt(st, {"forbid": 0.0}, "human")
    assert "hfoo" not in out["t"] and "hfoo" not in out["n"] and st["t"]["hfoo"] == 60.0  # (a copy)
    assert modes.adapt(st, {"forbid": 0.0}, "orc") is st  # (no grunt in it)
    capped = modes.adapt(st, {"cap": 0.4}, "human")["n"]
    assert sum(capped.values()) == pytest.approx(20) and capped["hfoo"] == pytest.approx(8)
    assert capped["hrif"] == pytest.approx(6) and capped["hkni"] == pytest.approx(6)
    again = modes.adapt({**st, "n": capped}, {"cap": 0.4}, "human")
    assert again["n"] == capped
    assert modes.capped({"a": 9, "b": 1}, 0.4) == {"a": 5.0, "b": 5.0}  # (fewer than 1 / cap types: alike)
    assert modes.capped({"a": 6, "b": 3, "c": 1}, 0.45) == pytest.approx({"a": 4.5, "b": 4.125, "c": 1.375})
    with pytest.raises(ValueError):
        modes.Tracker({"forbid": 0.0}, 0, (0.0, 0.0), {}, set())


def _army(owner, types, start=10, x=-2500):
    return [_row(start + i, owner, x, 0, 0, t) for i, t in enumerate(types)]


@pytest.mark.parametrize("native", [True, False])
def test_comp_violations(native, monkeypatch):
    """cap: the time above it after CAP_FROM, by how far; forbid: the most alive beyond the allowed; variety: the
    types at once by 600 s (both paths: C and numpy)."""
    from warcraftsim.fullgame import rulesnative
    if not native:
        monkeypatch.setattr(rulesnative, "modes", lambda *a: None)
    elif rulesnative.lib() is None:
        pytest.skip("no C compiler")
    food = {code("hfoo"): 2, code("hrif"): 3, code("hkni"): 4, code("hmtm"): 4, code("hwat"): 0}
    tk = modes.Tracker({"cap": 0.5, "forbid": 1.0, "variety": 3.0}, 0, (3000.0, 0.0), food, {code("hpea")},
                       race="human")
    home = [_row(1, 0, -3000, 0, modes.STRUCTURE, "htow"), _row(2, 0, -2900, 0, modes.WORKER, "hpea"),
            _row(3, 0, -2900, 0, modes.HERO, "Hpal", 3), _row(4, 1, 3000, 0, 0, "hrif")]  # (the enemy's rifleman)
    spam = _army(0, ["hfoo"] * 8 + ["hwat"] * 3)  # (16 food of footmen; water elementals: no food)
    assert tk.step(np.stack(home + spam), [], 30, 240.0) == 0.0  # (before CAP_FROM)
    v = tk.step(np.stack(home + spam), [], 30, 360.0)  # 60 s of the 600 all footmen: (1 - 0.5) / 0.5 = 1
    assert tk.v["cap"] == pytest.approx(0.1) and v == pytest.approx(0.1)
    mixed = _army(0, ["hfoo"] * 4 + ["hkni"] * 2 + ["hrif"] * 2, 40)  # 8 / 22 food: below the cap
    tk.step(np.stack(home + mixed), [], 30, 420.0)
    assert tk.v["cap"] == pytest.approx(0.1) and tk.v["forbid"] == pytest.approx(0.5)  # (two riflemen)
    few = _army(0, ["hfoo"] * 5, 60)  # 10 food: too small an army to count
    tk.step(np.stack(home + few), [], 30, 480.0)
    assert tk.v["cap"] == pytest.approx(0.1) and tk.v["forbid"] == pytest.approx(0.5)  # (the most at once)
    assert "variety" in tk.done and tk.v["variety"] == 0.0  # (three types at 420)
    tk.step(np.stack(home + _army(0, ["hrif"] * 6, 80)), [], 30, 500.0)
    assert tk.v["forbid"] == 1.0 and tk.report()["forbid"] == [1.0, 1.0]
    late = modes.Tracker({"variety": 4.0}, 0, (3000.0, 0.0), food, {code("hpea")}, race="human")
    late.step(np.stack(home + mixed), [], 30, 300.0)
    late.step(np.stack(home + mixed), [], 30, 610.0)
    assert late.v["variety"] == pytest.approx(0.25) and "variety" in late.done


def test_forbid_allows_undead_choppers():
    tk = modes.Tracker({"forbid": 0.0}, 0, (3000.0, 0.0), {code("ugho"): 2}, {code("uaco")}, race="undead")
    tk.step(np.stack(_army(0, ["ugho"] * 3)), [], 10, 100.0)
    assert tk.v["forbid"] == 0.0
    tk.step(np.stack(_army(0, ["ugho"] * 5)), [], 10, 200.0)
    assert tk.v["forbid"] == pytest.approx(0.5)


def test_measures_c_and_numpy_agree():
    """rulesnative.modes and Tracker._measure_np: the same measures for any rows and needs (all bits)."""
    from warcraftsim.fullgame import rulesnative
    if rulesnative.lib() is None:
        pytest.skip("no C compiler")
    types = ["hfoo", "hrif", "hkni", "hpea", "htow", "ngol", "Hpal", "hwat", "ugho", "ogru"]
    food = {code("hfoo"): 2, code("hrif"): 3, code("hkni"): 4, code("ugho"): 2, code("ogru"): 3, code("Hpal"): 5}
    flags = {"hpea": modes.WORKER, "htow": modes.STRUCTURE, "ngol": modes.STRUCTURE, "Hpal": modes.HERO}
    rng = np.random.default_rng(0)
    for trial in range(60):
        n = int(rng.integers(1, 80))
        rows = []
        for i in range(n):
            t = types[int(rng.integers(len(types)))]
            fl = flags.get(t, 0) | (modes.DEAD if rng.random() < 0.1 else 0)
            rows.append(_row(i + 1, int(rng.integers(0, 2)) if t != "ngol" else 15, int(rng.integers(-4000, 4000)),
                             int(rng.integers(-4000, 4000)), fl, t, int(rng.integers(0, 6))))
        rows = np.stack(rows)
        tk = modes.Tracker({"cap": 0.5, "forbid": float(trial % 2)}, trial % 2, (3000.0, 100.0), food,
                           {code("hpea")}, own_start=(-3000.0, 0.0) if trial % 3 else None,
                           race="human" if trial % 4 else "undead")
        need = 511 if trial < 20 else int(rng.integers(0, 512))
        got, ref = rulesnative.modes(tk, rows, need, tk._constants), tk._measure_np(rows, need)
        asked = [bool(need & bit) for bit in BITS] + [True] * (len(ref) - modes.MH)  # (the C measures some anyway)
        assert len(got) == len(ref) and [x for x, k in zip(got, asked) if k] == [x for x, k in zip(ref, asked) if k], \
            (trial, need)


BITS = (64, 8, 16, 32, 4 | 128, 1, 2, 2, 128, 128, 256)  # (each measure's need bits, w3r_modes' out[0..10])


def test_target_with_the_composition_kinds():
    """selfplay.target: the block at the strategy block's end, the target adapted, --z-forbid-mask's order."""
    from warcraftsim.fullgame import features as fx, strategy
    from warcraftsim.fullgame.selfplay import target
    items = ["hfoo", "hrif", "hkni", "Hpal"]
    pool = {"items": {"human": items}}
    vocab = {"orders": [[code("hrif"), fx.IMMEDIATE], [code("hfoo"), fx.IMMEDIATE]]}
    st = {"race": "human", "t": {"hfoo": 60.0, "hrif": 200.0}, "n": {"hfoo": 12, "hrif": 4}}
    m = {"forbid": 0.0, "variety": 3.0}
    cfg = {"z_mix": 1, "z_forbid_mask": 1, "step_seconds": 0.5}
    fx.set_z(strategy.ZDIM + modes.MDIM + strategy.SLOTS + modes.CDIM)
    try:
        view = SimpleNamespace(z=None, forbid=None)
        out = target(cfg, view, st, pool, "human", vocab, m)
        assert "hfoo" not in out["t"] and "hfoo" in st["t"] and len(view.z) == fx.ZDIM
        assert view.z[-modes.CDIM:].tolist() == pytest.approx(modes.comp_vector(m).tolist())
        assert view.z[strategy.ZDIM + modes.MDIM:][:2].tolist() == [0.0, 1.0]  # (the mix: riflemen only)
        assert view.forbid == (2, code("hfoo"), 1800, 0)
        fx.set_z(strategy.ZDIM + modes.MDIM + strategy.SLOTS)  # (a network without the block: as before)
        view = SimpleNamespace(z=None)
        out = target({"z_mix": 1, "step_seconds": 0.5}, view, st, pool, "human", vocab, {"no_rush": 300.0})
        assert out is st and len(view.z) == fx.ZDIM and not hasattr(view, "forbid")
    finally:
        fx.set_z(0)
