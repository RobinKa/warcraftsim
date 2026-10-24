"""The entity features in C (featnative.py, native/w3feat.c) against the numpy reference: the same selection, types,
current orders and features (the sines and cosines within float32's last bit), and the same privileged block."""

import numpy as np
import pytest

from warcraftsim.fullgame import featnative
from warcraftsim.fullgame import features as fx


def _rows(rng, n):
    r = np.zeros((n, fx.ROW_COLS), np.int64)
    r[:, fx.C_ID] = np.arange(1, n + 1) + 1000
    r[:, fx.C_TYPE] = rng.choice([int.from_bytes(c.encode(), "big") for c in ("hfoo", "hpea", "htow", "hhou", "Hpal",
                                                                            "ogru", "ntav", "nkob", "ngol")], n)
    r[:, fx.C_OWNER] = rng.choice([0, 1, 12, 15], n)
    r[:, fx.C_X], r[:, fx.C_Y] = rng.integers(-6000, 6000, n), rng.integers(-6000, 6000, n)
    r[:, fx.C_FACING] = rng.integers(0, 360, n)
    r[:, fx.C_MAXHP] = rng.integers(0, 2000, n)
    r[:, fx.C_HP] = rng.integers(0, 2000, n)
    r[:, fx.C_MAXMANA] = rng.integers(0, 3, n) * 100
    r[:, fx.C_MANA] = rng.integers(0, 300, n)
    r[:, fx.C_ORDER] = rng.choice([0, 851983, 852018, int.from_bytes(b"hbar", "big")], n)
    r[:, fx.C_FLAGS] = rng.choice([0, 1, 2, 4, 8, 16, 512, 1024, 2 | 8, 4 | 16], n)
    r[:, fx.C_VIS] = rng.integers(0, 4, n)
    r[:, fx.C_RESOURCE] = rng.integers(0, 12500, n)
    r[:, fx.C_HLEVEL] = rng.integers(0, 10, n)
    r[:, fx.C_SKILLPTS] = rng.integers(0, 3, n)
    return r


@pytest.mark.skipif(featnative.lib() is None, reason="no C compiler")
def test_native_entities_match_numpy(monkeypatch):
    vocab = {"types": [int.from_bytes(c.encode(), "big") for c in ("hfoo", "hpea", "htow", "Hpal", "ogru")],
             "current_orders": [851983, 852018], "upgrades": [], "orders": [[851983, fx.POINT]]}
    enc = fx.Encoder(vocab)
    rng = np.random.default_rng(0)
    for trial in range(60):
        rows = _rows(rng, int(rng.integers(0, 260)))
        player, sign = int(rng.integers(0, 2)), float(rng.choice([1.0, -1.0]))
        ids = rows[:, fx.C_ID]
        production = ({int(i): int(rng.integers(1, 7)) for i in ids[:5]}, {int(i) for i in ids[3:9]}, {})
        lumber = {int(i) for i in ids[::7]}
        native = enc.entities(rows, player, sign, production, lumber)
        monkeypatch.setattr(featnative, "_lib", False)
        py = enc.entities(rows, player, sign, production, lumber)
        monkeypatch.setattr(featnative, "_lib", None)
        assert np.array_equal(native[0], py[0]) and native[1] == py[1]
        assert np.allclose(native[2], py[2], rtol=0, atol=1e-6)
        assert np.array_equal(native[3], py[3]) and np.array_equal(native[4], py[4])
        view = enc.view(player, sign, [0, 1])
        a = view.priv(rows, None)
        monkeypatch.setattr(featnative, "_lib", False)
        b = view.priv(rows, None)
        monkeypatch.setattr(featnative, "_lib", None)
        assert np.allclose(a, b, rtol=0, atol=1e-6)


def test_to_f16_matches_numpy():
    """The F16C conversion gives numpy's float16 values: rounding to nearest even, halves' subnormals, overflow to
    infinity, signs and zeros."""
    import numpy as np

    from warcraftsim.fullgame.featnative import to_f16
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 3, 20000), rng.uniform(-1e-4, 1e-4, 5000), rng.uniform(-1e5, 1e5, 2000),
                        [0.0, -0.0, 65504, 65519.99, 65520, -65520, 6e-8, 3e-8, 2.9e-8, 1e-9, np.inf, -np.inf]])
    x = x.astype(np.float32)
    # (float32 values exactly halfway between two float16 values: the rounding's ties)
    ties = ((np.arange(-30000, 30000, dtype=np.int64) * 8192 + 4096) & 0x7FFFFFFF).astype(np.uint32).view(np.float32)
    for a in (x, ties[np.isfinite(ties)], x[:7].reshape(7, 1), x[:0]):
        got = to_f16(a)
        assert got.dtype == np.float16 and got.shape == a.shape
        assert np.array_equal(got.view(np.uint16), a.astype(np.float16).view(np.uint16))
