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
