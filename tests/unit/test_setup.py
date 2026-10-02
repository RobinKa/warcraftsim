def test_every_game_setup_call_names_real_fields():
    """The game setups written around the package (self-play's actors, the collectors, the video
    renderer) only name fields GameSetup has: a wrong one fails in an actor's thread, not at import
    (a run once sat without games for 40 minutes on one)."""
    import ast
    import dataclasses
    from pathlib import Path

    import warcraftsim
    from warcraftsim.runtime.instance import GameSetup
    fields = {f.name for f in dataclasses.fields(GameSetup)}
    root = Path(warcraftsim.__file__).parent
    calls = 0
    for path in list(root.rglob("*.py")) + list((root.parent / "scripts").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", "")) == "GameSetup":
                calls += 1
                names = {k.arg for k in node.keywords if k.arg}
                for k in node.keywords:  # GameSetup(**{**setup.__dict__, "field": value, ...})
                    if k.arg is None and isinstance(k.value, ast.Dict):
                        names |= {key.value for key in k.value.keys if isinstance(key, ast.Constant)}
                assert names <= fields, f"{path.name}:{node.lineno}: {sorted(names - fields)}"
    assert calls > 5


def test_a_human_slot_is_the_local_player():
    """versus.py: a person plays one slot (the user), the policy the other; no observer slot."""
    import pytest

    from warcraftsim.runtime.instance import Agent, GameSetup, Human
    setup = GameSetup(map="duelrush", slots=[Human("orc", handicap=50), Agent("human", handicap=50)])
    w = setup.wgc("Maps\\x.w3x")
    assert [s.is_user for s in w.slots] == [True, False] and not any(s.is_observer for s in w.slots)
    assert w.slots[0].handicap == 50 and setup.agent_players == (1,)
    with pytest.raises(ValueError):
        GameSetup(map="duelrush", slots=[Human("orc"), Human("human")]).wgc("Maps\\x.w3x")
