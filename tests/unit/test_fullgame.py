import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from warcraftsim.fullgame import features as fx  # noqa: E402
from warcraftsim.fullgame.selfplay import League, Trajectory  # noqa: E402


class _Q(list):
    def put(self, x):
        self.append(x)


def _step(value: float) -> tuple[dict, dict]:
    st = {"n": 2, "n_own": 1, "ent": np.zeros((2, fx.F), np.float32), "type": np.zeros(2, np.int64),
          "cur": np.zeros(2, np.int64), "glob": np.zeros(4, np.float32)}
    res = {"order": np.zeros(1, np.int64), "tgt": np.zeros(1, np.int64), "bx": np.zeros(1, np.int64),
           "by": np.zeros(1, np.int64), "logp": np.zeros(1, np.float32), "value": value, "version": 0}
    return st, res


def test_trajectory_gae_chunks_and_terminal_reward():
    q = _Q()
    tr = Trajectory({"chunk": 2, "gamma": 0.5, "lam": 1.0}, q, {})
    for v in (0.0, 0.0, 0.8, 0.0):
        tr.add(*_step(v))
    tr.end(1.0)
    steps = [s for msg in q for s in msg["steps"]]
    assert len(steps) == 4  # every step once, the chunk boundary's bootstrap step not twice
    # the first chunk bootstraps from the value at its end (0.8), the last one ends in the win
    assert [round(s["ret"], 4) for s in steps] == [0.2, 0.4, 0.5, 1.0]
    assert steps[-1]["done"] and not steps[0]["done"]


def test_league_spec(tmp_path):
    lg = League(tmp_path, ["normal"], {"ai": 0.25, "self": 0.5, "past": 0.5}, max_past=2, pfsp="hard")
    spec = lg.spec()
    assert sum(o["p"] for o in spec["launch"]) == pytest.approx(1.0)
    assert spec["agents"] == [{"name": "self", "kind": "self", "p": 1.0}]  # no snapshots yet
    for steps in (0, 10, 20):
        lg.add_snapshot(tmp_path / f"{steps}.pt", steps)
    assert [m.name for m in lg.past] == ["past:0", "past:20"]  # the first and the newest
    lg.member("past:0").record(1.0)
    lg.member("script:ai-normal").record(-1.0)
    spec = lg.spec()
    assert sum(o["p"] for o in spec["agents"]) == pytest.approx(1.0)
    lg.write()
    summary = json.loads((tmp_path / "league.json").read_text())
    assert {m["name"] for m in summary["members"]} == {"script:ai-normal", "past:0", "past:20"}
    assert lg.train_keys()["league/script:ai-normal"] == 0.0


def test_dashboard_collection_and_plays(tmp_path):
    from warcraftsim.dashboard.server import Dashboard

    d = tmp_path / "fullgame" / "demos-x"
    d.mkdir(parents=True)
    (d / "collect.json").write_text(json.dumps({"games": 4, "map": "duelrush", "status": "finished", "time": 1.0}))
    rows = [{"time": 100.0 + i, "game": i, "races": ["human", "orc"], "winner": w, "minutes": 2.0, "orders": 10}
            for i, w in enumerate((0, 1, None))]
    (d / "games.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    dash = Dashboard(tmp_path)
    (c,) = dash.collections()
    assert c["name"] == "fullgame/demos-x" and c["summary"]["games"] == 3
    detail = dash.run("fullgame/demos-x")
    assert detail["matrix"]["human"]["orc"] == [1, 1, 1] and detail["matrix"]["orc"]["human"] == [1, 1, 1]
    b = tmp_path / "bc" / "fit-x"
    b.mkdir(parents=True)
    (b / "bc.json").write_text(json.dumps({"task": "fullgame", "status": "fitted", "data": str(d), "games": {"train": 3},
                                               "created": 1.0}))
    plays = [{"eval": "e1", "outcome": o, "race": "human", "ai_race": "orc", "minutes": 2, "orders": 5, "failed": 1,
              "time": 5.0} for o in ("VICTORY", "DEFEAT")]
    (b / "play.jsonl").write_text("".join(json.dumps(r) + "\n" for r in plays))
    fit = dash.run("bc/fit-x")
    assert fit["info"]["datasets"] == ["fullgame/demos-x"]
    (p,) = fit["plays"]
    assert (p["games"], p["wins"], p["losses"], p["win_rate"]) == (2, 1, 1, 0.5)
    assert p["matrix"]["human"]["orc"] == [1, 0, 1]


def test_describe_spaces_names_orders():
    code = lambda s: int.from_bytes(s.encode(), "big")  # noqa: E731
    vocab = {"types": [code("hpea")], "current_orders": [851986], "upgrades": [code("Rhma")],
             "orders": [[851983, fx.UNIT], [code("hpea"), fx.IMMEDIATE], [code("Rhma"), fx.IMMEDIATE],
                        [code("hbar"), fx.POINT]],
             "order_names": {"851983": "attack"}}
    sp = fx.describe_spaces(vocab)
    order = sp["actions"]["heads"][0]
    assert order["options"] == ["none", "attack (unit)", "train hpea", "research Rhma", "build hbar"]
    assert order["size"] == 5 and [h["name"] for h in sp["actions"]["heads"]][1:] == ["target unit (pointer)", "point x", "point y (given x)"]
    glob = sp["observation"]["blocks"][0]
    assert len(glob["features"]) == fx.Encoder(vocab).G  # every global feature named
    assert len(sp["observation"]["blocks"][1]["features"]) == fx.F


def test_material_potential():
    from types import SimpleNamespace as U

    from warcraftsim.fullgame.selfplay import potential
    foot, hall = int.from_bytes(b"hfoo", "big"), int.from_bytes(b"htow", "big")
    values = {str(foot): 135, str(hall): 590}
    obs = U(units=[U(flags=0, owner=0, type_id=foot, hp=210, max_hp=420),       # half a footman: 67.5
                   U(flags=2, owner=1, type_id=hall, hp=1500, max_hp=1500),     # a whole town hall: 590
                   U(flags=1024 | 2, owner=0, type_id=hall, hp=0, max_hp=1500),  # dead: nothing
                   U(flags=0, owner=12, type_id=foot, hp=420, max_hp=420)])     # neutral: nothing
    assert potential(obs, 0, values, 100.0) == pytest.approx((67.5 - 590) / 100)
    assert potential(obs, 1, values, 100.0) == pytest.approx(-(67.5 - 590) / 100)  # zero-sum


def test_avail_mask_in_act_and_evaluate():
    from warcraftsim.fullgame.model import FullGameNet, act, evaluate
    torch.manual_seed(0)
    net = FullGameNet(n_types=20, n_cur=10, n_orders=30, G=30, d=64, layers=1).eval()
    net.allowed[:] = True
    net.order_kind.copy_(torch.randint(0, 5, (30,)))
    x = [torch.randn(3, 12, fx.F), torch.randint(0, 20, (3, 12)), torch.randint(0, 10, (3, 12)),
         torch.ones(3, 12, dtype=torch.bool), torch.randn(3, 30), torch.tensor([5, 3, 12])]
    avail = torch.rand(3, 30) < 0.3
    avail[:, 0] = True
    with torch.no_grad():
        for _ in range(20):
            a = act(net, *x, avail)
            own = torch.arange(a["order"].shape[1])[None] < x[5][:, None]
            chosen = avail.gather(1, a["order"])  # every sampled order is one the player can pay for
            assert bool((chosen | ~own).all())
        ev = evaluate(net, *x, a["order"], a["tgt"], a["bx"], a["by"], avail)
    assert float(((ev["logp"] - a["logp"]) * own).abs().max()) < 1e-5


def test_view_avail():
    code = lambda s: int.from_bytes(s.encode(), "big")  # noqa: E731
    vocab = {"types": [code("hpea")], "current_orders": [], "upgrades": [],
             "orders": [[851983, fx.UNIT], [code("hpea"), fx.IMMEDIATE], [code("hbar"), fx.POINT]]}
    costs = np.array([[0, 0, 0], [0, 0, 0], [38, 0, 1], [80, 30, 0]])
    enc = fx.Encoder(vocab, costs)
    view = enc.view(0, 1.0, [0, 1])
    rows = np.zeros((1, 18), np.int64)
    rows[0, fx.C_ID], rows[0, fx.C_TYPE], rows[0, fx.C_MAXHP], rows[0, fx.C_HP] = 1048576, code("hpea"), 220, 220
    me = np.array([0, 0, 50, 10, 10, 10, 0, 0, 0, 0, 0])  # 50 gold, 10 lumber, food 10/10
    st = view.step(rows, me, np.zeros((0, 5), np.int64), 0)
    assert st["avail"].tolist() == [True, True, False, False]  # supply-blocked worker; the barracks too dear
    me[5] = 12
    assert view.step(rows, me, np.zeros((0, 5), np.int64), 1)["avail"].tolist() == [True, True, True, False]


def _net(memory: bool, seed: int = 0):
    from warcraftsim.fullgame.model import FullGameNet
    torch.manual_seed(seed)
    net = FullGameNet(n_types=20, n_cur=10, n_orders=30, G=30, d=64, layers=1, memory=memory).eval()
    net.allowed[:] = True
    net.order_kind.copy_(torch.randint(0, 5, (30,)))
    net.order_kind[0] = fx.IMMEDIATE  # (no order: no target)
    return net


def test_memory_scan_matches_steps():
    """The core stepped as the actors run it and scanned over a sequence (training) agree, with a
    carried state and an episode's start inside the sequence."""
    from warcraftsim.fullgame.model import act, evaluate
    net = _net(True)
    torch.nn.init.normal_(net.mem_out.weight, std=0.3)  # (zero at first: the memory would do nothing)
    B, T, E = 3, 11, 12
    x = [torch.randn(B * T, E, fx.F), torch.randint(0, 20, (B * T, E)), torch.randint(0, 10, (B * T, E)),
         torch.ones(B * T, E, dtype=torch.bool), torch.randn(B * T, 30), torch.full((B * T,), 5)]
    starts = torch.zeros(B, T, dtype=torch.bool)
    starts[1, 4] = starts[2, 0] = True
    h0 = torch.randn(B, 64)
    h, acts = h0, []
    with torch.no_grad():
        xs = [v.view(B, T, *v.shape[1:]) for v in x]
        for t in range(T):
            a = act(net, *[v[:, t] for v in xs], h=torch.where(starts[:, t, None], torch.zeros_like(h), h))
            h = a["h"]
            acts.append(a)
        cat = lambda k: torch.stack([a[k] for a in acts], 1).reshape(B * T, -1)  # noqa: E731
        ev = evaluate(net, *x, cat("order"), cat("tgt"), cat("bx"), cat("by"), seq=(B, T, h0, starts))
    own = torch.arange(E)[None] < 5
    assert float(((ev["logp"] - cat("logp")) * own).abs().max()) < 1e-4
    assert float((ev["value"] - cat("value")[:, 0]).abs().max()) < 1e-4


def test_fresh_memory_is_a_no_op():
    net, plain = _net(True, 1), _net(False, 1)
    plain.load_state_dict({k: v for k, v in net.state_dict().items() if not k.startswith("mem_")})
    g = torch.randn(4, 64)
    c, h = net.context(g, torch.randn(4, 64))
    assert torch.equal(c, plain.context(g)[0]) and h.shape == (4, 64)


def test_learner_sequences_from_stored_states():
    """collate_seq: chunks cut into sequences from the states the actors stored, padded; evaluated
    as one batch they give the actors' log-probabilities."""
    from warcraftsim.fullgame.model import act, evaluate
    from warcraftsim.fullgame.selfplay import collate_seq, pieces
    net = _net(True, 2)
    torch.nn.init.normal_(net.mem_out.weight, std=0.3)
    rng = np.random.default_rng(0)
    chunks = []
    for length in (13, 7):
        h, steps = None, []
        for _ in range(length):
            n = int(rng.integers(3, 9))
            st = {"ent": rng.standard_normal((n, fx.F)).astype(np.float16), "type": rng.integers(0, 20, n).astype(np.int16),
                  "cur": rng.integers(0, 10, n).astype(np.int16), "glob": rng.standard_normal(30).astype(np.float32),
                  "n": n, "n_own": 2, "avail": None}
            with torch.no_grad():
                out = act(net, torch.as_tensor(st["ent"]).float()[None], torch.as_tensor(st["type"]).long()[None],
                          torch.as_tensor(st["cur"]).long()[None], torch.ones(1, n, dtype=torch.bool),
                          torch.as_tensor(st["glob"])[None], torch.tensor([2]), h=None if h is None else torch.as_tensor(h)[None])
            steps.append({**st, **{k: out[k][0, :2].numpy().astype(np.int16) for k in ("order", "tgt", "bx", "by")},
                          "logp": out["logp"][0, :2].numpy(), "h": h, "adv": 0.0, "ret": 0.0})  # (own units: as the actors)
            h = out["h"][0].numpy()
        chunks.append(steps)
    seqs = pieces(chunks, 5)
    assert [len(q) for q in seqs] == [5, 5, 3, 5, 2]
    mb = collate_seq(seqs, "cpu")
    assert mb["seq"][:2] == (5, 5) and int(mb["valid"].sum()) == 20
    with torch.no_grad():
        ev = evaluate(net, mb["ent"], mb["type"], mb["cur"], mb["mask"], mb["glob"], mb["n_own"], mb["order"],
                      mb["tgt"], mb["bx"], mb["by"], mb["avail"], seq=mb["seq"])
    assert float(((ev["logp"] - mb["logp"]) * mb["own"]).abs().max()) < 1e-4


def test_demo_returns():
    """BC's value targets: self-play's rewards on a recorded game (zero-sum; the winner's return
    at the last step is its outcome minus its material lead)."""
    from warcraftsim.fullgame.bc import returns
    foot = int.from_bytes(b"hfoo", "big")
    values = {str(foot): 100}
    # step, id, type, owner, x, y, facing, hp, max hp, mana, max mana, order, flags, ...
    rows = []
    for t, (a, b) in enumerate([(2, 2), (2, 1), (2, 0)]):  # player 1's footmen die
        rows += [[t, 10 + k, foot, 0, 0, 0, 0, 420, 420, 0, 0, 0, 0] + [0] * 5 for k in range(a)]
        rows += [[t, 20 + k, foot, 1, 0, 0, 0, 420, 420, 0, 0, 0, 0] + [0] * 5 for k in range(b)]
    game = {"units": np.array(rows, np.int64), "meta": {"steps": 2, "result": {"0": "VICTORY", "1": "DEFEAT"}}}
    reward = {"gamma": 0.5, "shaping": 1.0, "shaping_scale": 100.0, "tie_break": 0.5}
    r0, r1 = returns(game, 0, values, reward), returns(game, 1, values, reward)
    assert np.allclose(r0, -r1)
    # phi = 0, 1, 2; rewards: 0.5*1 - 0 = 0.5, 0.5*2 - 1 = 0, then 1 - 2 = -1
    assert np.allclose(r0, [0.5 + 0.5 * 0 + 0.25 * -1, 0 + 0.5 * -1, -1])


def test_demo_orders_the_player_could_not_pay_for_are_dropped():
    """Encoder.encode with costs: the AI's train orders beyond what it could pay for at the step
    (costs deducted in order) and ones that started nothing are no labels and queue nothing."""
    code = lambda s: int.from_bytes(s.encode(), "big")  # noqa: E731
    hall, peon = code("ogre"), code("opeo")
    vocab = {"types": [hall, peon], "current_orders": [], "upgrades": [],
             "orders": [[peon, fx.IMMEDIATE]]}
    costs = np.array([[0, 0, 0], [75, 0, 1]])
    B = 1048576

    def unit(t, uid, typ, flags):  # step, id, type, owner, x, y, facing, hp, max hp, ..., order, flags, vis
        r = [t, uid, typ, 0, -1000, 0, 0, 100, 100, 0, 0, 0, flags, 3] + [0] * 4
        return r
    units = [r for t in range(3) for r in (unit(t, B, hall, fx.STRUCTURE), unit(t, B + 1, peon, 4))]  # (in step order)
    players = [r for t in range(3) for r in ([t, 0, 100, 0, 1, 10, 0, 0, 0, 1, 0], [t, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])]
    # step 0: two peons ordered with 100 gold (the second: refused); step 2: one ordered, nothing starts
    orders = [[0, B, peon, 0, 0, 0, 0], [0, B, peon, 0, 0, 0, 0], [2, B, peon, 0, 0, 0, 0]]
    events = [[0, 5, B, peon, 0], [1, 6, B, B + 2, peon]]  # a peon started at the hall in step 0, done in step 1
    game = {"units": np.array(units, np.int64), "players": np.array(players, np.int64),
            "events": np.array(events, np.int64).reshape(-1, 5), "orders": np.array(orders, np.int64),
            "trees": np.zeros((0, 5), np.int64), "heroes": np.zeros((0, 16), np.int64),
            "meta": {"steps": 2, "races": ["orc", "orc"], "result": {}}}
    plain = fx.Encoder(vocab).encode(game, 0)
    paid = fx.Encoder(vocab, costs).encode(game, 0)
    assert plain["y_order"][:, 0].tolist() == [1, 0, 1]  # (the hall is the first own entity)
    assert paid["y_order"][:, 0].tolist() == [1, 0, 0]  # step 2's order started nothing (the hall not busy)
    # the queue after the peon is done: without costs the refused order stays queued (stale)
    assert plain["ent"][1, 0, 26] == pytest.approx(1 / 5) and paid["ent"][1, 0, 26] == 0
    assert paid["avail"][0].tolist() == [True, True] and plain["avail"].all()


def test_takeover_side_starts_at_the_takeover():
    """bc.side_data: in a takeover game the side a policy played until the built-in AI took over
    starts at the takeover step (the steps before have the policy's orders, not the AI's)."""
    from warcraftsim.fullgame.bc import side_data
    code = lambda s: int.from_bytes(s.encode(), "big")  # noqa: E731
    hall = code("ogre")
    vocab = {"types": [hall], "current_orders": [], "upgrades": [], "orders": [[code("opeo"), fx.IMMEDIATE]]}
    units = [[t, 1048576 + p, hall, p, -1000 + 2000 * p, 0, 0, 100, 100, 0, 0, 0, fx.STRUCTURE, 3] + [0] * 4
             for t in range(4) for p in (0, 1)]
    players = [[t, p, 100, 0, 1, 10, 0, 0, 0, 1, 0] for t in range(4) for p in (0, 1)]
    game = {"units": np.array(units, np.int64), "players": np.array(players, np.int64),
            "events": np.zeros((0, 5), np.int64), "orders": np.zeros((0, 7), np.int64), "trees": np.zeros((0, 5), np.int64),
            "meta": {"steps": 3, "races": ["orc", "orc"], "result": {"0": "VICTORY", "1": "DEFEAT"},
                     "takeover": {"player": 0, "step": 2, "policy": "p.pt"}}}
    enc = fx.Encoder(vocab)
    a, b = side_data(enc, game, 0, {}), side_data(enc, game, 1, {})
    assert len(a["n_own"]) == len(a["ret"]) == 2 and len(b["n_own"]) == len(b["ret"]) == 4
    assert a["ret"][-1] == pytest.approx(1.0) and b["ret"][-1] == pytest.approx(-1.0)


def test_curriculum_against_the_builtin_ai(tmp_path):
    """The curriculum: a level per built-in AI that a loss raises and a win lowers (towards a 50%
    score); from 0 to 0.5 the learner's hit points rise to twice the AI's, then the AI starts late."""
    lg = League(tmp_path, ["easy"], {"ai": 0.5, "self": 0.5, "past": 0.5}, max_past=2, pfsp="hard",
                curriculum=(0.75, 0.25, 120.0, 50))
    n = "script:ai-easy"
    hd = lambda: tuple(lg.knobs(n, "")[k] for k in ("handicap", "delay"))  # noqa: E731
    assert hd() == (100, 60.0)
    launch = [x for x in lg.spec()["launch"] if x["kind"] == "ai"][0]["by_race"][""]
    assert (launch["handicap"], launch["delay"], launch["level"]) == (100, 60.0, 0.75)
    lg.curriculum(n, 1.0)  # a win: harder
    assert hd() == (100, 0.0)
    lg.curriculum(n, 0.0)  # a tie: unchanged
    lg.curriculum(n, 1.0)
    assert hd() == (80, 0.0)  # (handicaps in steps of 10)
    for _ in range(5):
        lg.curriculum(n, 1.0)
    assert lg.level[(n, "")] == 0.0 and hd() == (50, 0.0)  # the real game
    for _ in range(9):
        lg.curriculum(n, -1.0)
    assert lg.level[(n, "")] == 1.0 and hd() == (100, 120.0)
    lg.write()
    again = League(tmp_path, ["easy"], {"ai": 0.5, "self": 0.5, "past": 0.5}, max_past=2, pfsp="hard",
                   curriculum=(0.75, 0.25, 120.0, 50))
    again.restore(json.loads((tmp_path / "league.json").read_text()))
    assert again.level[(n, "")] == 1.0  # (resumed runs keep their levels)


def test_curriculum_tie_moves_the_level(tmp_path):
    lg = League(tmp_path, ["normal"], {"ai": 0.5, "self": 0.5, "past": 0.5}, max_past=2, pfsp="hard",
                curriculum=(0.5, 0.1, 0.9, 50), mode="tax", tie=0.5)
    lg.curriculum("script:ai-normal", 0.0)
    assert abs(lg.level[("script:ai-normal", "")] - 0.55) < 1e-9  # (half a loss's step)
    still = League(tmp_path, ["normal"], {"ai": 0.5, "self": 0.5, "past": 0.5}, max_past=2, pfsp="hard",
                   curriculum=(0.5, 0.1, 0.9, 50), mode="tax")
    still.curriculum("script:ai-normal", 0.0)
    assert still.level[("script:ai-normal", "")] == 0.5


def test_curriculum_late_start_only(tmp_path):
    lg = League(tmp_path, ["normal"], {"ai": 0.5, "self": 0.5, "past": 0.5}, max_past=2, pfsp="hard",
                curriculum=(0.5, 0.1, 180.0, 50), mode="delay")
    assert lg.knobs("script:ai-normal") == {"handicap": 50, "delay": 90.0, "tax": 0.0}  # the AI 90 s late
    lg.curriculum("script:ai-normal", 1.0)
    assert lg.knobs("script:ai-normal")["delay"] == 72.0
    tax = League(tmp_path, ["normal"], {"ai": 0.5, "self": 0.5, "past": 0.5}, max_past=2, pfsp="hard",
                 curriculum=(0.5, 0.1, 0.9, 50), mode="tax")
    assert tax.knobs("script:ai-normal") == {"handicap": 50, "delay": 0.0, "tax": 0.45}  # the AI keeps 55% of its income


def test_curriculum_real_game_yardstick(tmp_path):
    """With a curriculum some launches play the real game: their results go to "script:ai-X (real)"
    (charted with the others) and leave the curriculum alone."""
    lg = League(tmp_path, ["normal"], {"ai": 0.5, "self": 0.5, "past": 0.5}, max_past=2, pfsp="hard",
                curriculum=(0.5, 0.1, 180.0, 50), mode="delay")
    m = lg.member("script:ai-normal (real)")
    m.record(1.0)
    assert lg.train_keys()["league/script:ai-normal (real)"] == 1.0
    assert all(x["kind"] != "ai" or "(real)" not in x["difficulty"] for x in lg.spec()["launch"])
    lg.write()
    again = League(tmp_path, ["normal"], {"ai": 0.5, "self": 0.5, "past": 0.5}, max_past=2, pfsp="hard",
                   curriculum=(0.5, 0.1, 180.0, 50), mode="delay")
    again.restore(json.loads((tmp_path / "league.json").read_text()))
    assert again.member("script:ai-normal (real)").wins == 1


def test_curriculum_by_race(tmp_path):
    """A level per difficulty and learner's race; levels from before races go to every race."""
    lg = League(tmp_path, ["normal"], {"ai": 0.5, "self": 0.5, "past": 0.5}, max_past=2, pfsp="hard",
                curriculum=(0.5, 0.1, 0.9, 50), mode="tax", races=("human", "nightelf"))
    lg.curriculum("script:ai-normal", -1.0, "human")
    lg.curriculum("script:ai-normal", 1.0, "nightelf")
    by = [x for x in lg.spec()["launch"] if x["kind"] == "ai"][0]["by_race"]
    assert by["human"]["tax"] == 0.54 and by["nightelf"]["tax"] == 0.36
    assert "curriculum/ai-normal human tax" in lg.train_keys()
    old = {"level": {"script:ai-normal": 0.7}}
    lg.restore(old)
    assert lg.level[("script:ai-normal", "human")] == lg.level[("script:ai-normal", "nightelf")] == 0.7


def test_mixed_matchups_curriculum_and_balance(tmp_path):
    """Without mirror matchups: a curriculum level per matchup, and between agents of two races a
    tax on the stronger one's income that the learner's games against itself move towards 50%."""
    from warcraftsim.fullgame.selfplay import balance_tax
    lg = League(tmp_path, ["normal"], {"ai": 0.5, "self": 0.5, "past": 0.5}, max_past=2, pfsp="hard",
                curriculum=(0.5, 0.1, 0.9, 50), mode="tax", races=("human", "nightelf"), mirror=False,
                balance=(0.25, 0.8))
    n = "script:ai-normal"
    assert set(k for _, k in lg.level) == {"human/human", "human/nightelf", "nightelf/human", "nightelf/nightelf"}
    lg.curriculum(n, -1.0, lg.matchup("human", "nightelf"))  # the learner human, the AI night elf
    by = [x for x in lg.spec()["launch"] if x["kind"] == "ai"][0]["by_race"]
    assert by["human/nightelf"]["tax"] == 0.54 and by["nightelf/human"]["tax"] == 0.45
    assert lg.balance == {"human/nightelf": 0.0}
    assert balance_tax(lg.spec(), ["human", "nightelf"])[1] == 0.0  # (nothing taken yet)
    lg.balanced("nightelf", "human", 1.0)  # night elf won: its income taxed
    lg.balanced("human", "nightelf", -1.0)
    lg.balanced("human", "nightelf", 0.0)  # (a tie: unchanged)
    assert lg.balance["human/nightelf"] == -0.5 and lg.spec()["balance"] == {"human/nightelf": -0.4}
    assert balance_tax(lg.spec(), ["nightelf", "human"]) == (0, 0.4)  # player 0 is the night elf
    assert balance_tax(lg.spec(), ["human", "nightelf"]) == (1, 0.4)
    assert balance_tax(lg.spec(), ["human", "human"])[1] == 0.0
    for _ in range(9):
        lg.balanced("human", "nightelf", 1.0)
    assert lg.balance["human/nightelf"] == 1.0 and lg.train_keys()["balance/human/nightelf tax"] == 0.8
    lg.write()
    again = League(tmp_path, ["normal"], {"ai": 0.5, "self": 0.5, "past": 0.5}, max_past=2, pfsp="hard",
                   curriculum=(0.5, 0.1, 0.9, 50), mode="tax", races=("human", "nightelf"), mirror=False,
                   balance=(0.25, 0.8))
    again.restore(json.loads((tmp_path / "league.json").read_text()))
    assert again.balance == lg.balance and again.level == lg.level
    again.restore({"level": {f"{n}|human": 0.3}})  # a run of mirror matchups goes on with mixed ones
    assert again.level[(n, "human/nightelf")] == again.level[(n, "human/human")] == 0.3
    mirror = League(tmp_path, ["normal"], {"ai": 0.5, "self": 0.5, "past": 0.5}, max_past=2, pfsp="hard",
                    curriculum=(0.5, 0.1, 0.9, 50), mode="tax", races=("human", "nightelf"), balance=(0.25, 0.8))
    assert mirror.balance == {} and "balance" not in mirror.spec() and mirror.matchup("human", "human") == "human"


def test_harvest_switch_is_a_decision():
    """A harvest order is redundant only for a worker already harvesting that resource: a miner
    sent to the trees is a decision (the clone had learned that harvesting workers never switch)."""
    tree, mine = fx.harvest_resource(fx.SMART, fx.TREE, None), fx.harvest_resource(fx.HARVEST, fx.UNIT, None)
    assert (tree, mine) == ("lumber", "gold")
    assert fx.harvest_resource(fx.SMART, fx.UNIT, fx.GOLD_MINES[0]) == "gold"
    assert fx.harvest_resource(fx.SMART, fx.UNIT, 12345) is None  # smart on something else: no harvest
    h = fx.HARVEST
    assert fx.redundant(fx.SMART, fx.TREE, h, "lumber", "lumber")
    assert not fx.redundant(fx.SMART, fx.TREE, h, "lumber", "gold")  # the switch
    assert fx.redundant(fx.HARVEST, fx.UNIT, h, "gold", None)  # unknown: mining gold (melee start, rally point)
    assert not fx.redundant(fx.SMART, fx.TREE, h, "lumber", None)  # so to the trees: a switch
    assert not fx.redundant(fx.SMART, fx.TREE, 0, "lumber", "gold")  # not harvesting


def test_view_tracks_workers_on_lumber():
    code = lambda s: int.from_bytes(s.encode(), "big")  # noqa: E731
    peon, B = code("opeo"), 1048576
    vocab = {"types": [peon], "current_orders": [fx.HARVEST], "upgrades": [], "orders": [[fx.SMART, fx.TREE], [fx.HARVEST, fx.UNIT]]}
    enc = fx.Encoder(vocab)
    view = enc.view(0, 1.0, [1, 1])
    rows = np.zeros((1, 18), np.int64)
    rows[0, fx.C_ID], rows[0, fx.C_TYPE], rows[0, fx.C_MAXHP], rows[0, fx.C_HP] = B, peon, 220, 220
    rows[0, fx.C_ORDER], rows[0, fx.C_FLAGS], rows[0, fx.C_VIS] = fx.HARVEST, 4, 3
    me = np.array([0, 0, 50, 10, 1, 10, 0, 0, 0, 0, 0])
    st = view.step(rows, me, np.zeros((0, 5), np.int64), 0)
    assert st["ent"][0, fx.F - 1] == 0
    view.track_harvest(np.array([[0, B, fx.SMART, 2, 0, 0, 777]]), st, {777: (100, 100)})  # smart on tree 777
    assert view.assign[B] == "lumber"
    st = view.step(rows, me, np.zeros((0, 5), np.int64), 1)
    assert st["ent"][0, fx.F - 1] == 1  # the lumber feature
    view.track_harvest(np.array([[1, B, 851986, 1, 0, 0, 0]]), st, {777: (100, 100)})  # a move: no longer harvesting
    assert B not in view.assign


def test_real_game_launches_in_the_spec(tmp_path):
    """With a curriculum the real game is a launch kind of its own (an extra draw after the races'
    had made night elf rare among the real games)."""
    lg = League(tmp_path, ["easy", "normal"], {"ai": 0.8, "self": 0.5, "past": 0.5}, max_past=2, pfsp="hard",
                curriculum=(0.5, 0.1, 0.9, 50), mode="tax", races=("human", "nightelf"))
    launch = lg.spec(0.1)["launch"]
    real = [x for x in launch if x.get("real")]
    curr = [x for x in launch if x["kind"] == "ai" and not x.get("real")]
    assert len(real) == 2 and all("by_race" not in x for x in real)
    assert abs(sum(x["p"] for x in real) - 0.08) < 1e-9 and abs(sum(x["p"] for x in curr) - 0.72) < 1e-9


def test_steps_shuffle_buffer_pays_out_every_step_once(monkeypatch):
    """The cloning batches come from a buffer kept full (a side read pays out what it brings): every
    step of every side once, the last partial batch aside, and batches mix sides."""
    from warcraftsim.fullgame import bc
    sides = {}

    def side(enc, game, player, values):
        n = 30 + 7 * ((game + player) % 3)
        out = {k: np.zeros((n, 2), np.float32) for k in bc.KEYS}
        out["n_own"] = (1000 * (2 * game + player) + np.arange(n)).astype(np.int64)  # (which step of which side)
        sides[2 * game + player] = n
        return out

    monkeypatch.setattr(bc.fx, "load_game", lambda p: p)
    monkeypatch.setattr(bc.fx, "Encoder", lambda vocab, costs: None)
    monkeypatch.setattr(bc, "side_data", side)
    batches = list(bc.Steps(list(range(9)), {}, 8, {}, buffer_steps=64))
    seen = np.concatenate([b["n_own"].numpy() for b in batches])
    total = sum(sides.values())
    assert all(len(b["n_own"]) == 8 for b in batches) and len(seen) == total - total % 8
    assert len(set(seen.tolist())) == len(seen)  # no step twice
    assert np.mean([len(set((b["n_own"].numpy() // 1000).tolist())) for b in batches]) > 1.5


def test_past_snapshots_share_one_network_object():
    """The inference server's past snapshots: one network whose weights are swapped (no compiling
    per league member), giving each snapshot's own outputs."""
    from warcraftsim.fullgame.model import FullGameNet
    from warcraftsim.fullgame.selfplay import PastNet
    torch.manual_seed(0)
    a, b = (FullGameNet(n_types=20, n_cur=10, n_orders=30, G=30, d=64, layers=1).eval() for _ in range(2))
    b.order_kind.fill_(2)
    x = [torch.randn(3, 12, fx.F), torch.randint(0, 20, (3, 12)), torch.randint(0, 10, (3, 12)),
         torch.ones(3, 12, dtype=torch.bool), torch.randn(3, 30)]
    swap = PastNet(torch.device("cpu"))
    with torch.no_grad():
        want = {"a": a.encode(*x)[0], "b": b.encode(*x)[0]}
        assert float((want["a"] - want["b"]).abs().max()) > 1e-3
        hosts = []
        for key in ("a", "b", "b", "a"):
            host = swap.get(key, {"a": a, "b": b}[key])
            hosts.append(host)
            assert float((host.encode(*x)[0] - want[key]).abs().max()) < 1e-6
            assert int(host.order_kind[0]) == (2 if key == "b" else 0)
    assert all(h is hosts[0] for h in hosts) and hosts[0] is not a


def test_inference_server_answers_each_game_over_its_pipe(tmp_path):
    """The inference server (its own process) and an actor's side of it: a game's request for the
    current network and a past snapshot comes back on the game's own pipe, one answer per side."""
    import os
    import torch.multiprocessing as mp
    from warcraftsim.fullgame.model import FullGameNet
    from warcraftsim.fullgame.selfplay import RemoteInference, inference_main, publish
    net = FullGameNet(n_types=20, n_cur=10, n_orders=30, G=30, d=64, layers=1).eval()
    net.allowed[:] = True
    publish(net, 3, tmp_path)
    past = tmp_path / "past.pt"
    torch.save({"model": net.state_dict(), "config": net.config}, past)
    ctx = mp.get_context("spawn")
    pipes = [ctx.Pipe() for _ in range(2)]
    stop = ctx.Event()
    cfg = {"run_dir": str(tmp_path), "device": "cpu", "learner_pid": os.getpid(), "infer_batch": 8, "compile": False,
           "max_past": 2}
    server = ctx.Process(target=inference_main, args=(cfg, [b for _, b in pipes], stop), daemon=True)
    server.start()
    try:
        infer = RemoteInference([a for a, _ in pipes])
        rng = np.random.default_rng(0)

        def st(n, own):
            return {"n": n, "n_own": own, "ent": rng.normal(size=(n, fx.F)).astype(np.float32),
                    "type": rng.integers(0, 20, n), "cur": rng.integers(0, 10, n),
                    "glob": rng.normal(size=30).astype(np.float32), "avail": None, "h": None}
        for _ in range(3):
            out = infer.request([("current", st(9, 4)), (str(past), st(5, 2))])
            assert [len(r["order"]) for r in out] == [4, 2] and [r["version"] for r in out] == [3, -1]
            assert all(len(r["logp"]) == len(r["order"]) and np.isfinite(r["value"]) for r in out)
        assert infer.nets.version == 3
    finally:
        stop.set()
        server.join(timeout=20)
        if server.is_alive():
            server.terminate()


@pytest.mark.parametrize("memory", [False, True])
def test_packed_network_call_matches_evaluate(memory):
    """The actors' network call packs a batch into one tensor (and its outputs into one): the
    orders it samples have the log-probabilities and values evaluate() gives for the same steps."""
    from warcraftsim.fullgame.model import evaluate
    from warcraftsim.fullgame.selfplay import Inference
    net = _net(memory)
    if memory:
        torch.nn.init.normal_(net.mem_out.weight, std=0.3)
    rng = np.random.default_rng(1)
    sts = []
    for n, own in ((9, 4), (5, 5), (12, 1)):
        avail = rng.random(30) < 0.5
        avail[0] = True
        sts.append({"n": n, "n_own": own, "ent": rng.normal(size=(n, fx.F)).astype(np.float32),
                    "type": rng.integers(0, 20, n), "cur": rng.integers(0, 10, n),
                    "glob": rng.normal(size=30).astype(np.float32), "avail": avail,
                    "h": rng.normal(size=64).astype(np.float32) if memory and n != 5 else None})
    infer = Inference(None, torch.device("cpu"), 8, compile_=False, thread=False)
    calls = [infer._launch(net, sts[:2]), infer._launch(net, sts[2:])]  # (two calls, then one wait)
    infer.wait()
    res = infer._finish(calls[0]) + infer._finish(calls[1])
    assert [len(r["order"]) for r in res] == [4, 5, 1]
    E = 12
    ent, typ, cur = torch.zeros(3, E, fx.F), torch.zeros(3, E, dtype=torch.long), torch.zeros(3, E, dtype=torch.long)
    mask, order = torch.zeros(3, E, dtype=torch.bool), torch.zeros(4, 3, E, dtype=torch.long)
    for i, (st, r) in enumerate(zip(sts, res)):
        n, o = st["n"], st["n_own"]
        ent[i, :n], typ[i, :n], cur[i, :n], mask[i, :n] = (torch.from_numpy(st["ent"]), torch.from_numpy(st["type"]),
                                                          torch.from_numpy(st["cur"]), True)
        for k, f in enumerate(("order", "tgt", "bx", "by")):
            order[k, i, :o] = torch.from_numpy(r[f])
        assert bool(st["avail"][r["order"]].all())  # only orders it could pay for
    glob = torch.from_numpy(np.stack([st["glob"] for st in sts]))
    avail = torch.from_numpy(np.stack([st["avail"] for st in sts]))
    n_own = torch.tensor([st["n_own"] for st in sts])
    h0 = torch.from_numpy(np.stack([st["h"] if st["h"] is not None else np.zeros(64, np.float32) for st in sts]))
    with torch.no_grad():
        ev = evaluate(net, ent, typ, cur, mask, glob, n_own, *order[:, :, :fx.MAX_OWN], avail,
                      seq=(3, 1, h0, torch.zeros(3, 1, dtype=torch.bool)) if memory else None)
    for i, r in enumerate(res):
        o = sts[i]["n_own"]
        assert np.abs(ev["logp"][i, :o].numpy() - r["logp"]).max() < 1e-4
        assert abs(float(ev["value"][i]) - r["value"]) < 1e-4
        assert ("h" in r) == memory


def test_minibatches_of_similar_sizes():
    """The learner's minibatches: every step once an epoch, and steps of similar entity counts
    together (a minibatch pads to its widest step)."""
    from warcraftsim.fullgame.selfplay import minibatches
    rng = np.random.default_rng(0)
    sizes = rng.integers(5, 110, 8192)
    padded = {}
    for group in (1, 8):
        mbs = minibatches(sizes, 256, group, np.random.RandomState(0))
        assert len(mbs) == 32 and sorted(np.concatenate(mbs).tolist()) == list(range(8192))
        padded[group] = np.mean([sizes[m].max() for m in mbs])
    assert padded[8] < 0.65 * padded[1]
    odd = minibatches(sizes[:1000], 256, 8, np.random.RandomState(0))
    assert sorted(np.concatenate(odd).tolist()) == list(range(1000)) and sorted(len(m) for m in odd) == [232, 256, 256, 256]


@pytest.mark.parametrize("memory", [False, True])
def test_ppo_update_runs_on_minibatches_by_size(memory):
    """One PPO update over steps of different sizes (minibatches cut by entity count, the clone's KL
    term, the statistics read once at the end; with memory over sequences of them): finite losses,
    and the network moved."""
    import types

    from warcraftsim.fullgame.model import act
    from warcraftsim.fullgame.selfplay import ppo_update
    net, ref = _net(memory), _net(memory)
    rng = np.random.default_rng(0)
    steps = []
    for i in range(24):
        n, own = int(rng.integers(3, 13)), int(rng.integers(1, 4))
        st = {"n": n, "n_own": own, "ent": rng.normal(size=(n, fx.F)).astype(np.float32), "type": rng.integers(0, 20, n),
              "cur": rng.integers(0, 10, n), "glob": rng.normal(size=30).astype(np.float32)}
        with torch.no_grad():
            a = act(net, torch.from_numpy(st["ent"])[None], torch.from_numpy(st["type"])[None], torch.from_numpy(st["cur"])[None],
                    torch.ones(1, n, dtype=torch.bool), torch.from_numpy(st["glob"])[None], torch.tensor([own]))
        steps.append({**st, **{k: a[k][0, :own].numpy() for k in ("order", "tgt", "bx", "by", "logp")}, "avail": None,
                      "adv": float(rng.normal()), "ret": float(rng.normal())})
    args = types.SimpleNamespace(epochs=2, minibatch=8, pad_groups=2, seq_len=4 if memory else 1, bf16=0, clip=0.2,
                                 vf_coef=0.5, ent_coef=0.01, ref_kl=0.2, bc_coef=0.0, max_grad_norm=0.5)
    before = net.value_head[0].weight.detach().clone()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    chunks = [steps[a:a + 8] for a in range(0, 24, 8)] if memory else None  # (three games' pieces)
    ref.allowed[:, 1:] = False  # (orders only the learner may give: the cloning loss allows more as it goes)
    out = ppo_update(net, ref, opt, steps, args, False, torch.device("cpu"), chunks)
    assert {"loss/policy", "loss/value", "loss/entropy", "loss/kl", "loss/clipfrac", "loss/ref_kl", "grad_norm"} <= set(out)
    assert all(np.isfinite(v) for v in out.values()) and 0 <= out["loss/ref_kl"] < 10
    assert float((net.value_head[0].weight - before).abs().max()) > 0


def test_cloning_loss_on_shuffled_steps_with_memory():
    """Self-play's auxiliary cloning loss draws shuffled steps (no lanes): a network with memory
    takes each as a game's first step."""
    from warcraftsim.fullgame.bc import losses
    net = _net(True)
    rng = np.random.default_rng(0)
    B, E, O = 4, 6, 3
    b = {"ent": torch.from_numpy(rng.normal(size=(B, E, fx.F)).astype(np.float32)), "type": torch.from_numpy(rng.integers(0, 20, (B, E))),
         "cur": torch.from_numpy(rng.integers(0, 10, (B, E))), "mask": torch.ones(B, E, dtype=torch.bool),
         "glob": torch.from_numpy(rng.normal(size=(B, 30)).astype(np.float32)), "n_own": torch.full((B,), O),
         "y_order": torch.from_numpy(rng.integers(0, 5, (B, E))), "y_ptr": torch.full((B, E), -1),  # (labels as wide as the view)
         "y_x": torch.from_numpy(rng.integers(0, fx.BINS, (B, E))), "y_y": torch.from_numpy(rng.integers(0, fx.BINS, (B, E))),
         "avail": torch.ones(B, 30, dtype=torch.bool), "ret": torch.zeros(B)}
    loss, _ = losses(net, b, torch.device("cpu"), None, value_coef=0.0, stats=False)
    loss.backward()
    assert torch.isfinite(loss) and net.mem_in.weight.grad is not None


def test_league_with_an_exploiter(tmp_path):
    """--exploiter-share: games between the learner and its exploiter are a launch kind of their
    own (out of the others' shares alike); the learner's results against it are kept and restored,
    and its snapshots join the league under their own name."""
    lg = League(tmp_path, ["normal"], {"ai": 0.5, "self": 0.5, "past": 0.5}, max_past=4, pfsp="hard", exploiter_share=0.2)
    spec = lg.spec()
    p = {x["kind"]: x["p"] for x in spec["launch"]}
    assert abs(p["exploit"] - 0.2) < 1e-9 and abs(sum(x["p"] for x in spec["launch"]) - 1.0) < 1e-9
    assert abs(p["agents"] - 0.4) < 1e-9
    for o in (1.0, -1.0, -1.0, 0.0):
        lg.member("exploiter").record(o)
    assert abs(lg.train_keys()["league/exploiter_vs_main"] - (1 - 1.5 / 4)) < 1e-9
    lg.add_snapshot(tmp_path / "x.pt", 100, name="exploiter:100")
    lg.exploiter_resets = 2
    lg.write()
    again = League(tmp_path, ["normal"], {"ai": 0.5, "self": 0.5, "past": 0.5}, max_past=4, pfsp="hard", exploiter_share=0.2)
    (tmp_path / "x.pt").write_bytes(b"")
    again.restore(json.loads((tmp_path / "league.json").read_text()))
    assert again.exploiter.recent == lg.exploiter.recent and again.exploiter_resets == 2
    assert any(m.name == "exploiter:100" for m in again.past)
    plain = League(tmp_path, ["normal"], {"ai": 0.5, "self": 0.5, "past": 0.5}, max_past=4, pfsp="hard")
    assert plain.member("exploiter") is None and all(x["kind"] != "exploit" for x in plain.spec()["launch"])


def test_a_restarted_run_draws_other_launches():
    from warcraftsim.fullgame.selfplay import game_rng
    draws = lambda cfg, w, k: [game_rng(cfg, w, k).random() for _ in range(3)]  # noqa: E731
    a = {"seed": 0, "start_update": 100}
    assert draws(a, 1, 2) == draws(dict(a), 1, 2)  # (a thread's stream is its own, and repeatable)
    assert draws(a, 1, 2) != draws(a, 2, 1) and draws(a, 0, 16) != draws(a, 1, 0)
    assert draws(a, 1, 2) != draws({"seed": 0, "start_update": 101}, 1, 2)
    assert draws({"seed": 0}, 1, 2) == draws({"seed": 0, "start_update": 0}, 1, 2)
