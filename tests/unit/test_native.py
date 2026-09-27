"""The C observation parser (warcraftsim.native) against protocol.parse_tokens + merge_observation."""

import random
import shutil

import pytest

from warcraftsim.protocol import PROTOCOL_VERSION, merge_observation, parse_token_lines

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")


def _i32(v):
    return (v + 2 ** 31) % 2 ** 32 - 2 ** 31


def rec(tag, fields):
    return [tag, *[str(f) for f in fields], str(_i32(sum(fields)))]


def unit(uid, hero=False, rng=random):
    flags = (1 if hero else 0) | rng.choice([0, 2, 4, 1024])
    base = [uid, 1751674721 + uid % 7, rng.randrange(3), rng.randrange(-3000, 3000), rng.randrange(-3000, 3000),
            rng.randrange(360), rng.randrange(1, 900), 900, rng.randrange(300), 300, rng.choice([0, 851983, 852018]),
            flags, rng.randrange(4096), rng.randrange(20000)]
    if flags & 1:
        base += [rng.randrange(1, 10), rng.randrange(5000), rng.randrange(3)] + [rng.randrange(1000000) for _ in range(6)] \
            + [v for _ in range(4) for v in (rng.randrange(4), rng.randrange(600))]
    return base


def observation(seq, full, units, removed=(), stray=False, damage=False, orders=None, rng=random):
    toks = ["V", str(PROTOCOL_VERSION)] + rec("T", [seq, seq * 500, 0, int(full)])
    for p in (0, 1):
        toks += rec("P", [p, 1 + p, 1, 500, 200, 5, 10, 0, 1000, 200, 3, 0, -2300 + 4600 * p, 0])
    if orders:
        for o in orders:
            toks += rec("O", [o])
    for k, u in enumerate(units):
        r = rec("U", u)
        if stray and k == 1:  # a stray number inside a record: repaired
            r.insert(5, "12345")
        if damage and k == 2:  # two stray numbers: the record is lost
            r.insert(3, "7")
            r.insert(8, "9")
        toks += r
    for uid in removed:
        toks += rec("R", [uid])
    toks += rec("E", [1, 1048600, 1048601, 1751674721]) + rec("E", [99, 1, 2, 3])
    toks += rec("C", [1]) + rec("C", [0]) + rec("I", [1048600, 851983, 1, -100, 200, 0])
    toks += ["sound\\foo.wav", "3.5", "Q"] + rec("K", [100, -200]) + ["X"]  # strays: dropped
    return ("\n".join(toks) + "\n").encode("latin-1")


def _same(a, b):
    assert (a.seq, a.game_ms, a.game_over, a.version, a.damaged_records, a.camera) == \
           (b.seq, b.game_ms, b.game_over, b.version, b.damaged_records, b.camera)
    assert a.players == b.players and a.events == b.events and a.command_results == b.command_results
    assert a.issued == b.issued and a.orders == b.orders and a.destructables == b.destructables
    assert list(a.units) == list(b.units), "the merged unit tables differ"


def test_native_matches_python():
    from warcraftsim.native import NativeObs
    rng = random.Random(1)
    native, table = NativeObs(), {}
    ids = list(range(1048576, 1048576 + 40))
    for step in range(30):
        full = step in (0, 17)  # a new episode clears the table
        if full:
            units = [unit(i, hero=i % 9 == 0, rng=rng) for i in ids]
            removed = ()
        else:
            units = [unit(i, hero=i % 9 == 0, rng=rng) for i in rng.sample(ids, 8)]
            units += [unit(ids[-1] + 1 + step, rng=rng)]  # a new unit
            removed = rng.sample(ids, 2)
        data = observation(step, full, units, removed, stray=step % 3 == 1, damage=step % 5 == 2,
                           orders=[851983, 0, 852018] if full else None, rng=rng)
        py = merge_observation(table, parse_token_lines(data, ("attack", "missing", "harvest")))
        c = native.merge(native.parse(data, ("attack", "missing", "harvest")))
        _same(py, c)
        assert c.unit_array.shape[0] == len(py.units)
    assert py.damaged_records == 0 and any(True for _ in py.units)


def test_native_rejects_partial():
    from warcraftsim.native import NativeObs
    from warcraftsim.protocol import ProtocolError
    data = observation(0, True, [unit(1048576)])
    with pytest.raises(ProtocolError):
        NativeObs().parse(data[:-3], ())  # no end marker


def test_material_vectorized_matches_loop():
    from types import SimpleNamespace

    from warcraftsim.fullgame.trace import material
    from warcraftsim.native import NativeObs
    rng = random.Random(2)
    n = NativeObs()
    obs = n.merge(n.parse(observation(0, True, [unit(1048576 + i, hero=i % 5 == 0, rng=rng) for i in range(60)], rng=rng), ()))
    values = {str(1751674721 + k): 50 * (k + 1) for k in range(7)}
    loop = material(SimpleNamespace(units=list(obs.units)), values)
    fast = material(obs, values)
    assert loop.keys() == fast.keys() and all(abs(loop[p] - fast[p]) < 1e-6 for p in loop)
