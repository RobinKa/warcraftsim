"""--worker-value W (trace.worker_valued): the shaping's values' table with each worker type W times its value, every
other type as it was; W 1 the table itself; one object a table and weight (the native material code caches tables by id)."""

from types import SimpleNamespace

from warcraftsim.fullgame import trace


def code(s: str) -> str:
    return str(int.from_bytes(s.encode(), "big"))


def test_workers_only_and_cached():
    values = {code("hpea"): 75, code("hfoo"): 135, code("opeo"): 75, code("htow"): 385}
    assert trace.worker_valued(values, 1.0) is values and trace.worker_valued(values, 0) is values
    w = trace.worker_valued(values, 3.0)
    assert w[code("hpea")] == 225 and w[code("opeo")] == 225 and w[code("hfoo")] == 135 and w[code("htow")] == 385
    assert trace.worker_valued(values, 3.0) is w and values[code("hpea")] == 75


def test_material_counts_them():
    values = {code("hpea"): 75, code("hfoo"): 135}
    units = [SimpleNamespace(owner=1, type_id=int(code("hpea")), flags=0, hp=220, max_hp=220, hero_level=0),
             SimpleNamespace(owner=1, type_id=int(code("hfoo")), flags=0, hp=420, max_hp=420, hero_level=0)]
    obs = SimpleNamespace(units=units, unit_array=None)
    assert trace.material_units(obs, values)[1] == 210
    assert trace.material_units(obs, trace.worker_valued(values, 3.0))[1] == 225 + 135


def test_eval_handicap_draw():
    """--eval-handicap: levels parsed; a matchup's level drawn with the share, none without levels or at share 0."""
    import random
    from warcraftsim.fullgame.selfplay import eval_handicap, handicap_levels
    assert handicap_levels("0.25,0.5") == [0.25, 0.5] and handicap_levels("") == [] and handicap_levels("1.5,0.25") == [0.25]
    rng = random.Random(0)
    assert eval_handicap({"eval_handicap": [], "eval_handicap_share": 1.0}, rng) == 0.0
    assert eval_handicap({"eval_handicap": [0.25], "eval_handicap_share": 0.0}, rng) == 0.0
    got = [eval_handicap({"eval_handicap": [0.25, 0.5], "eval_handicap_share": 0.4}, rng) for _ in range(4000)]
    assert abs(sum(g > 0 for g in got) / 4000 - 0.4) < 0.03 and set(got) == {0.0, 0.25, 0.5}
