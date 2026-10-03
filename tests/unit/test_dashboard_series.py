"""The dashboard's episode series: kept between requests, extended with new episodes only."""

from pathlib import Path

from warcraftsim.dashboard.server import EPISODE_SERIES, _binned, _EpisodeSeries, _interp_steps


def _episode(i: int, race: str, opp: str, opponent: str, outcome: float) -> dict:
    return {"time": 100.0 + i + (0.5 if i % 3 == 0 else 0.0),  # (not quite in order, as games arrive)
            "outcome": outcome, "return": outcome, "length": 10 + i, "race": race, "opponent_race": opp,
            "opponent": opponent, "game_time": 60.0}


def test_series_grow_with_new_episodes_and_match_a_full_read():
    f = Path("episodes.jsonl")
    rows = [_episode(i, "human", "orc", "script:ai-easy", 1.0 if i % 2 else -1.0) for i in range(50)]
    train = [{"time": 90.0, "agent_steps": 0}, {"time": 200.0, "agent_steps": 1100}]
    s = _EpisodeSeries()
    s.update([(f, rows)])
    rows += [_episode(i, "orc", "human", "self", 1.0) for i in range(50, 80)]  # (the same list grows)
    s.update([(f, rows)])
    binned, sparse = s.binned(train)
    ordered = sorted(rows, key=lambda r: r["time"])
    want = _binned(_interp_steps([r["time"] for r in ordered], train),
                   [{k: float(v) for k, g in EPISODE_SERIES.items() if (v := g(e)) is not None} for e in ordered])
    assert len(binned) == len(want)
    for a, b in zip(binned, want):
        for k, v in b.items():
            assert abs(a[k] - v) < 1e-9, k
    assert sum(p[2] for p in sparse["cwin/ai-easy human/orc"]) == 50  # every curriculum game, 40+ a point
    assert abs(sum(p[1] * p[2] for p in sparse["cwin/ai-easy human/orc"]) - 25) < 1e-9
    m = s.matchups()
    assert m["curriculum"]["all"]["human"]["orc"] == [25, 0, 25]
    # against itself each game counts from both sides
    assert m["self"]["all"]["orc"]["human"] == [30, 0, 0] and m["self"]["all"]["human"]["orc"] == [0, 0, 30]
    s.update([(f, list(rows[:10]))])  # a file replaced: read anew
    assert s.matchups()["curriculum"]["all"]["human"]["orc"] == [5, 0, 5]


def test_docs_list_read_save(tmp_path):
    """The Docs tab: docs/*.md listed (the reader's order first), read, saved; names checked."""
    from warcraftsim.dashboard.server import Docs
    (tmp_path / "zeta.md").write_text("# Zeta\n\ntext\n")
    (tmp_path / "overview.md").write_text("# Overview\n")
    d = Docs(tmp_path)
    assert [x["name"] for x in d.list()] == ["overview", "zeta"] and d.list()[1]["title"] == "Zeta"
    assert d.read("zeta")["text"].startswith("# Zeta") and d.read("../zeta") is None and d.read("nope") is None
    assert d.save("new-doc", "# New\n") and d.read("new-doc")["title"] == "New"
    assert d.save("reference/deep", "# Deep\n") and d.list()[-1]["name"] == "reference/deep"
    assert not d.save("../evil", "x") and not d.save("Upper", "x")


def test_lineage_links_runs_fits_and_collections(tmp_path):
    """The Lineage view's graph: a run's parent, a fit's start and data, a collection's policy."""
    import json as j

    from warcraftsim.dashboard.server import Dashboard
    runs = tmp_path / "runs"
    (runs / "a").mkdir(parents=True)
    (runs / "a" / "run.json").write_text(j.dumps({"name": "a", "kind": "run", "task": "fullgame_duelfast", "status": "stopped"}))
    (runs / "b").mkdir()
    (runs / "b" / "run.json").write_text(j.dumps({"name": "b", "kind": "run", "task": "fullgame_duelfast", "status": "training",
                                                  "init_from": "runs/a/checkpoints/1.pt"}))
    d = Dashboard(runs)
    d.runs = lambda: [{"name": "a", "kind": "run", "task": "fullgame_duelfast"},
                      {"name": "b", "kind": "run", "task": "fullgame_duelfast", "parent": {"kind": "run", "name": "a"}},
                      {"name": "bc/c", "kind": "bc", "task": "fullgame", "init_from": "runs/b/checkpoints/2.pt",
                       "data": "runs/fullgame/demos-x"},
                      {"name": "fullgame/demos-x", "kind": "collect", "task": "fullgame"},
                      {"name": "m", "kind": "match", "task": "micro", "players": {"A": {"name": "a@1.0M", "spec": "runs/a/checkpoints/1.pt"},
                                                                                  "B": {"name": "script:amove"}}}]
    (runs / "fullgame" / "demos-x").mkdir(parents=True)
    (runs / "fullgame" / "demos-x" / "collect.json").write_text(j.dumps({"policy": "runs/a/checkpoints/1.pt"}))
    L = d.lineage()
    assert {(e["from"], e["to"], e["why"]) for e in L["edges"]} == {
        ("a", "b", "start"), ("b", "bc/c", "start"), ("fullgame/demos-x", "bc/c", "data"), ("a", "fullgame/demos-x", "policy"),
        ("a", "m", "played")}
    assert {n["name"]: n["kind"] for n in L["nodes"]}["a"] == "selfplay"
