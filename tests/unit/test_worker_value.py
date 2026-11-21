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


def test_rules_line_for_the_videos():
    """A filmed game's ruleset in words (the panel's line under the title, the videos' tag)."""
    from warcraftsim.fullgame.selfplay import rules_line
    assert rules_line("(2)EchoIsles") == "Echo Isles · the game's own rules"
    assert rules_line("echofast2").startswith("Echo Isles · echofast2 rules (costs ×0.5")
    assert rules_line("(2)EchoIsles", "late", {"minute": 11.6}).endswith("· late start (minute 12)")
    assert rules_line("echofast2", "base2").endswith("· scenario base2") and len(rules_line("echofast2", "base2")) <= 72


def test_shaped_extras_xp_and_whole_buildings():
    """--hero-xp-value: a living hero's experience counts; --building-hp-value 0: a finished building counts whole while
    it stands (one under construction as before); both off: the material itself."""
    import numpy as np
    from warcraftsim.protocol import UnitFlags
    assert (trace.STRUCTURE_FLAG, trace.CONSTRUCTING_FLAG, trace.HERO_FLAG, trace.DEAD_FLAG) == (
        int(UnitFlags.STRUCTURE), int(UnitFlags.CONSTRUCTING), int(UnitFlags.HERO), int(UnitFlags.DEAD))
    values = {code("Udea"): 425, code("unpl"): 255, code("uzig"): 150}
    def unit(owner, typ, flags, hp, max_hp, xp=0):
        return SimpleNamespace(owner=owner, type_id=int(code(typ)), flags=flags, hp=hp, max_hp=max_hp, hero_xp=xp, hero_level=1)
    units = [unit(0, "Udea", trace.HERO_FLAG, 600, 600, xp=300), unit(0, "Udea", trace.HERO_FLAG | trace.DEAD_FLAG, 0, 600, xp=900),
             unit(1, "unpl", trace.STRUCTURE_FLAG, 750, 1500), unit(1, "uzig", trace.STRUCTURE_FLAG | trace.CONSTRUCTING_FLAG, 100, 400)]
    obs = SimpleNamespace(units=units, unit_array=None)
    mat = trace.material_units(obs, values)
    assert mat == {0: 425.0, 1: 255 * 0.5 + 150 * 0.25}
    assert trace.shaped_extras(obs, values, mat) is mat
    got = trace.shaped_extras(obs, values, mat, hero_xp=0.5, building_hp=0.0)
    assert got[0] == 425.0 + 150.0 and got[1] == 255.0 + 150 * 0.25 and mat[0] == 425.0
    # the native table's path: the same numbers (columns 1 type, 2 owner, 6 hp, 7 max hp, 11 flags, 15 xp)
    ua = np.zeros((4, 31), np.int64)
    for i, u in enumerate(units):
        ua[i, [1, 2, 6, 7, 11, 15]] = [u.type_id, u.owner, u.hp, u.max_hp, u.flags, u.hero_xp]
    got2 = trace.shaped_extras(SimpleNamespace(unit_array=ua, units=[]), values, mat, hero_xp=0.5, building_hp=0.0)
    assert got2 == got


def test_replays_have_a_kind_and_evaluation_titles_name_the_policy(tmp_path):
    """The Replays page's filter: a replay's kind from its trace ("eval": selfplay --eval-replay-every; older traces:
    training games); an evaluation game's title names the policy that played it."""
    from warcraftsim.dashboard import replays
    from warcraftsim.fullgame.selfplay import played_by
    tr = {"sides": [{"player": 0, "name": "learner (orc)"}, {"player": 1, "name": "built-in AI insane (human)"}],
          "steps": [{"game_time": 600.0}], "outcome": {"0": 1.0}, "title": "burst-x · fgself-236 avg swa-43-53 · vs built-in AI"}
    assert replays.summarize("game-1795000000-a0", tr)["kind"] == "train"
    assert replays.summarize("game-1795000000-a0", {**tr, "kind": "eval"})["kind"] == "eval"
    assert played_by({"eval_key": "runs/fgself-236/averaged/swa-43-53.pt"}, 7) == "fgself-236 avg swa-43-53"
    assert played_by({"eval_key": "/x/runs/fgself-238/checkpoints/0000000041109888.pt"}, 7) == "fgself-238/41109888"
    assert played_by({}, 7) == "policy v7" and played_by({"eval_key": "current+z"}, 7) == "policy v7"
    ix = replays.ReplayIndex(tmp_path, tmp_path / "cache.json")
    ix._loaded = True
    ix.runs = {"r": {"mtime": 0, "rows": {"a": {**replays.summarize("game-1795000000-a0", tr)},
                                           "b": {**replays.summarize("game-1795000001-a0", {**tr, "kind": "eval"})},
                                           "c": {k: v for k, v in replays.summarize("game-1795000002-a0", tr).items() if k != "kind"}}}}
    assert ix.listing(kind="eval")["matched"] == 1 and ix.listing(kind="train")["matched"] == 2 and ix.listing()["matched"] == 3
