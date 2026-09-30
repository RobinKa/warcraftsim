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
