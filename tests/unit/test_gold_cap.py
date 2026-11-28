"""--econ-masks bit 512 (features.GOLD_CAP; docs/proposals/weakness-analysis-2026-11-28.md, section 1): human's and orc's
gold mines take 5 gold workers (6 while one walks there from afar: teacher.SATURATED). A gold harvest or a worker's
right-click on a full mine is not sent and the worker harvests the tree nearest the mine's hall instead; the harvest
class is masked while every worked mine is full; a worker the computer slot's engine sends to a full mine by itself goes
to the trees too. Other races and the bit off: as before."""

import numpy as np
import pytest

from warcraftsim import paths
from warcraftsim.fullgame import features as fx
from warcraftsim.protocol import TargetDestructable, TargetOrder


def code(s: str) -> int:
    return int.from_bytes(s.encode(), "big")


MOVE_ID, RETURN, RESUME = 851986, 852020, 852017
TYPES = ("htow", "hpea", "ngol", "ogre", "opeo", "unpl", "uaco", "ugol", "etol", "ewsp", "egol", "Hpal", "hfoo", "ogru")
VOCAB = {"types": [code(c) for c in TYPES], "current_orders": [], "upgrades": [],
         "orders": [[fx.HARVEST, fx.UNIT], [fx.SMART, fx.UNIT], [fx.HARVEST, fx.TREE], [MOVE_ID, fx.POINT]]}
HARVEST, SMART_UNIT, HARVEST_TREE, MOVE = 1, 2, 3, 4
RACE = {"human": 0, "orc": 1, "undead": 2, "nightelf": 3}
HALL = {"human": "htow", "orc": "ogre", "undead": "unpl", "nightelf": "etol"}
WORKER_OF = {"human": "hpea", "orc": "opeo", "undead": "uaco", "nightelf": "ewsp"}
MINE_OF = {"human": "ngol", "orc": "ngol", "undead": "ugol", "nightelf": "egol"}
HX, HY = -2000, 0  # (the hall; its mine 780 north, as Echo Isles' ~781)
MX, MY = -2000, 780
MINE, IDLE = 2, 30
TREE_HALL, TREE_WORKER = 900, 901  # (one by the hall, one by the idle worker)
TREES = {TREE_HALL: (-2350, -100), TREE_WORKER: (-3050, 100)}


@pytest.fixture(autouse=True)
def masks_off_after():
    yield
    fx.set_econ_masks(0)


def _unit(i: int, typ: str, flags: int = 0, order: int = 0, x: float = -2000, y: float = 400, owner: int = 0,
          hp: int = 300, maxhp: int = 300, items=()) -> np.ndarray:
    r = np.zeros(fx.ROW_COLS, np.int64)
    r[fx.C_ID], r[fx.C_TYPE], r[fx.C_OWNER], r[fx.C_X], r[fx.C_Y] = i, code(typ), owner, x, y
    r[fx.C_HP], r[fx.C_MAXHP], r[fx.C_ORDER], r[fx.C_FLAGS], r[fx.C_VIS] = hp, maxhp, order, flags, 3
    for k, it in enumerate(items):
        r[fx.C_ITEM0 + k] = code(it)
    return r


def _base(race: str, miners: int = 0, extra=(), order: int = fx.HARVEST):
    """The player's hall and mine (a plain one for human and orc; its own haunted or entangled one), `miners` workers in
    a harvest cycle between them, and an idle worker by the trees west of the hall."""
    owner = 15 if MINE_OF[race] == "ngol" else 0
    rows = [_unit(1, HALL[race], fx.STRUCTURE, x=HX, y=HY), _unit(MINE, MINE_OF[race], fx.STRUCTURE, x=MX, y=MY, owner=owner)]
    rows += [_unit(10 + k, WORKER_OF[race], fx.WORKER, order=order, x=-2100 + 40 * k, y=500) for k in range(miners)]
    rows += [_unit(IDLE, WORKER_OF[race], fx.WORKER, x=-3000, y=100)]
    return np.stack(rows + list(extra))


def _bot(race: str, rows, assign=None, mine_of=None, t: int = 0, bot=None, last=None):
    from warcraftsim.fullgame.play import BCAgent
    if bot is None:
        bot = BCAgent(None, VOCAB, 0, None)
        bot.enc = fx.Encoder(VOCAB, np.zeros((len(VOCAB["orders"]) + 1, 3), np.int64))
        bot.view = bot.enc.view(0, 1.0, [RACE[race], 2])
        bot.view.assign.update(assign or {})
        bot.view.mine_of.update(mine_of or {})
        bot.view.last_order.update(last or {})  # (the agent's accepted orders: unit -> (kind, x, y, target, step))
        bot.trees, bot.building_since = dict(TREES), {}
    bot.t = t
    bot.me = me = np.asarray([t, 0, 500, 500, 5, 30, 0])
    bot.rows = rows
    st = bot.view.step(rows, me, np.zeros((0, 5), np.int64), t)
    return bot, st


def _send(bot, st, units: dict) -> list:
    """{unit: (order class, target unit id)} -> the step's commands."""
    n = st["n_own"]
    order, tgt = [0] * n, [0] * n
    for u, (cls, target) in units.items():
        order[st["index"][u]], tgt[st["index"][u]] = cls, st["index"].get(target, 0)
    return bot.commands(st, order, tgt, [0] * n, [0] * n)


def test_the_flag_is_bit_512_alone():
    from warcraftsim.fullgame import teacher
    for v in (0, 1, 256, 511):
        fx.set_econ_masks(v)
        assert not fx.GOLD_CAP
    fx.set_econ_masks(512)
    assert fx.GOLD_CAP and not (fx.ECON_MASK or fx.FUTILE_MASK or fx.TREE_HARVEST or fx.BURROW_MASK)
    fx.set_econ_masks(1023)
    assert fx.GOLD_CAP and fx.TREE_HARVEST and fx.ECON_MASK and fx.FUTILE_MASK
    assert fx.GOLD_SATURATED == teacher.SATURATED and fx.GOLD_FULL == 5


def test_gold_full_at():
    assert not fx.gold_full_at(4, 0) and fx.gold_full_at(5, 0) and not fx.gold_full_at(5, 1)
    assert fx.gold_full_at(6, 1) and fx.gold_full_at(6, 2) and fx.gold_full_at(7, 0)


@pytest.mark.parametrize("race", ["human", "orc"])
@pytest.mark.parametrize("masks", [0, 511])
def test_off_as_before(race, masks):
    """Without the bit (every run before it): a 6th, 8th gold worker sent, the harvest class open, no count made."""
    fx.set_econ_masks(masks)
    bot, st = _bot(race, _base(race, 7))
    assert st["avail"][HARVEST] and "gold" not in bot.view.__dict__
    assert _send(bot, st, {IDLE: (HARVEST, MINE)}) == [TargetOrder(IDLE, fx.HARVEST, MINE)]
    assert _send(bot, st, {IDLE: (SMART_UNIT, MINE)}) == [TargetOrder(IDLE, fx.SMART, MINE)]
    assert bot.sent_resources == ["gold"]
    assert not getattr(bot, "gold_full_masked", 0) and not getattr(bot, "gold_full_engine", 0)


@pytest.mark.parametrize("race", ["human", "orc"])
@pytest.mark.parametrize("masks", [512, 1023])
def test_a_sixth_gold_worker_goes_to_the_trees(race, masks):
    fx.set_econ_masks(masks)
    bot, st = _bot(race, _base(race, 5))
    assert not st["avail"][HARVEST]  # (its one worked mine full: the class masked)
    assert bot.view.gold_full_steps == 1
    for cls in (HARVEST, SMART_UNIT):  # (a right-click on the mine likewise; the class stays open for other targets)
        cmds = _send(bot, st, {IDLE: (cls, MINE)})
        assert cmds == [TargetDestructable(IDLE, fx.HARVEST, TREE_HALL)]  # (the tree nearest the hall, not the worker)
        assert bot.sent_resources == ["lumber"]
    assert bot.gold_full_masked == 2 and bot.gold_full_lumber == 2
    bot.accepted(cmds, [True])
    assert bot.view.assign == {IDLE: "lumber"}


@pytest.mark.parametrize("race", ["human", "orc"])
def test_the_fifth_allowed_the_sixth_of_the_step_not(race):
    fx.set_econ_masks(512)
    extra = [_unit(31, WORKER_OF[race], fx.WORKER, x=-3000, y=-100)]
    bot, st = _bot(race, _base(race, 4, extra))
    assert st["avail"][HARVEST]
    assert _send(bot, st, {IDLE: (HARVEST, MINE)}) == [TargetOrder(IDLE, fx.HARVEST, MINE)]
    assert not getattr(bot, "gold_full_masked", 0)
    cmds = _send(bot, st, {IDLE: (HARVEST, MINE), 31: (HARVEST, MINE)})  # (two in one step: the second finds it full)
    got = dict(zip(cmds, bot.sent_resources))
    (to_mine,) = [c for c in cmds if isinstance(c, TargetOrder)]
    (to_tree,) = [c for c in cmds if isinstance(c, TargetDestructable)]
    assert {to_mine.unit, to_tree.unit} == {IDLE, 31} and (to_mine.order, to_mine.target) == (fx.HARVEST, MINE)
    assert (to_tree.order, to_tree.destructable) == (fx.HARVEST, TREE_HALL)
    assert got == {to_mine: "gold", to_tree: "lumber"} and len(cmds) == 2
    assert bot.gold_full_masked == 1


def test_a_sixth_while_one_walks_from_afar():
    """teacher.SATURATED: 5 mine, one walking. One of the mine's gold workers GOLD_WALK_FAR or farther from it (sent there
    from the natural): a 6th allowed, a 7th not."""
    fx.set_econ_masks(512)
    walker = _unit(40, "hpea", fx.WORKER, order=fx.HARVEST, x=-2000, y=-900)  # (1,680 from the mine, past the hall)
    bot, st = _bot("human", _base("human", 4, [walker]), assign={40: "gold"}, mine_of={40: MINE})
    g = bot.view.gold
    assert sorted(g.on[MINE]) == [10, 11, 12, 13, 40] and g.far[MINE] == [40]
    assert st["avail"][HARVEST]
    assert _send(bot, st, {IDLE: (HARVEST, MINE)}) == [TargetOrder(IDLE, fx.HARVEST, MINE)]
    bot, st = _bot("human", _base("human", 5, [walker]), assign={40: "gold"}, mine_of={40: MINE})
    assert not st["avail"][HARVEST]
    assert _send(bot, st, {IDLE: (HARVEST, MINE)}) == [TargetDestructable(IDLE, fx.HARVEST, TREE_HALL)]


def test_who_counts_on_the_mine():
    """Inside the mine (hidden, its current order 0), carrying gold back, on its way, right-clicked onto it by the agent:
    on it. Lumber workers in a harvest cycle by the mine, an unassigned worker the agent right-clicked to a point, a
    builder, a loaded one (a peon in a burrow's kind): not."""
    fx.set_econ_masks(512)
    extra = [_unit(41, "hpea", fx.WORKER | fx.HIDDEN_FLAG, x=MX, y=MY),  # (inside)
             _unit(42, "hpea", fx.WORKER, order=RETURN, x=-2000, y=200),  # (carrying gold back: the start's)
             _unit(43, "hpea", fx.WORKER, order=RESUME, x=-2000, y=600),
             _unit(44, "hpea", fx.WORKER, order=RETURN, x=-1900, y=700),  # (carrying lumber: the view's on lumber)
             _unit(45, "hpea", fx.WORKER, order=fx.SMART, x=-1900, y=600),  # (right-clicked to a point by the agent)
             _unit(46, "hpea", fx.WORKER, order=code("hhou"), x=-1800, y=600),  # (on its way to build)
             _unit(47, "hpea", fx.WORKER, order=fx.SMART, x=-1950, y=650),  # (right-clicked onto the mine: gold)
             _unit(48, "hpea", fx.WORKER | fx.LOADED_FLAG, order=fx.HARVEST, x=-1950, y=650)]
    bot, st = _bot("human", _base("human", 1, extra), assign={44: "lumber", 47: "gold"}, mine_of={47: MINE},
                   last={45: (1, -1900.0, 600.0, 0, 4)}, t=5)  # (45's right-click: the agent's, sent at step 4)
    g = bot.view.gold
    assert sorted(g.on[MINE]) == [10, 41, 42, 43, 47] and g.hidden == {41} and not g.far[MINE]
    assert not st["avail"][HARVEST]
    assert 45 not in bot.view.auto_gold and 42 in bot.view.auto_gold  # (the start's: no order of the agent's)
    bot, st = _bot("human", _base("human", 0, extra), t=6, bot=bot)  # (one fewer: room again)
    assert sorted(bot.view.gold.on[MINE]) == [41, 42, 43, 47] and st["avail"][HARVEST]


def test_the_mines_own_count_not_the_others():
    """Two bases: the main's mine full, the natural's (its hall a frame) open: the class open, the main's harvest to the
    trees by the main's hall, the natural's sent."""
    fx.set_econ_masks(512)
    nat = [_unit(3, "ngol", fx.STRUCTURE, x=2000, y=780, owner=15), _unit(4, "htow", fx.STRUCTURE | fx.CONSTRUCTING, x=2000, y=0)]
    bot, st = _bot("human", _base("human", 5, nat))
    assert st["avail"][HARVEST]
    assert bot.view.gold.hall == {MINE: (float(HX), float(HY)), 3: (2000.0, 0.0)}
    assert _send(bot, st, {IDLE: (HARVEST, MINE)}) == [TargetDestructable(IDLE, fx.HARVEST, TREE_HALL)]
    assert _send(bot, st, {IDLE: (HARVEST, 3)}) == [TargetOrder(IDLE, fx.HARVEST, 3)]


def test_a_chopper_kept_off_gets_nothing():
    fx.set_econ_masks(512)
    chopper = _unit(31, "hpea", fx.WORKER, order=fx.HARVEST, x=-3000, y=-100)
    bot, st = _bot("human", _base("human", 5, [chopper]), assign={31: "lumber"})
    assert _send(bot, st, {31: (HARVEST, MINE)}) == []
    assert bot.gold_full_masked == 1 and not getattr(bot, "gold_full_lumber", 0)


@pytest.mark.parametrize("race", ["undead", "nightelf"])
@pytest.mark.parametrize("miners", [3, 7])
def test_other_races_as_before(race, miners):
    """Undead and night elf mine their own mines (their 5: _mine's count, bit 64's mask); bit 512 changes nothing of
    theirs."""
    got = {}
    for masks in (0, 512, 255, 767):
        fx.set_econ_masks(masks)
        bot, st = _bot(race, _base(race, miners))
        got[masks] = (bool(st["avail"][HARVEST]), _send(bot, st, {IDLE: (HARVEST, MINE)}), list(bot.sent_resources))
        assert "gold" not in bot.view.__dict__ and not getattr(bot, "gold_full_masked", 0)
    assert got[0] == got[512] and got[255] == got[767]
    assert got[0][1] == ([TargetOrder(IDLE, fx.HARVEST, MINE)] if miners < 5 else [])


@pytest.mark.parametrize("race", ["human", "orc"])
def test_the_engines_new_worker(race):
    """The computer slot sends a newly trained worker to the mine by itself (its first order not the agent's): past the
    mine's 5 it goes to the trees by the hall; the fifth stays; one the agent gives an order this step is the agent's."""
    fx.set_econ_masks(512)
    w = WORKER_OF[race]
    bot, st = _bot(race, _base(race, 5), t=0)  # (the melee start's 5: no assignment, mining)
    assert _send(bot, st, {}) == []
    new = _unit(50, w, fx.WORKER, order=fx.HARVEST, x=HX + 150, y=HY + 150)  # (by the hall, on its way)
    bot, st = _bot(race, _base(race, 5, [new]), t=1, bot=bot)
    assert bot.view.auto_gold.get(50) == 1 and 50 in bot.view.gold.on[MINE]
    cmds = _send(bot, st, {})
    assert cmds == [TargetDestructable(50, fx.HARVEST, TREE_HALL)] and bot.sent_resources == ["lumber"]
    assert bot.gold_full_engine == 1
    bot.accepted(cmds, [True])
    assert bot.view.assign[50] == "lumber"
    bot, st = _bot(race, _base(race, 5, [new]), t=2, bot=bot)  # (the view has it on lumber now: off the mine's count)
    assert 50 not in bot.view.auto_gold and 50 not in bot.view.gold.of and _send(bot, st, {}) == []
    # (4 on the mine: the new one is the fifth)
    bot, st = _bot(race, _base(race, 4), t=0)
    bot, st = _bot(race, _base(race, 4, [new]), t=1, bot=bot)
    assert _send(bot, st, {}) == [] and not getattr(bot, "gold_full_engine", 0)
    # (the agent's own order to it this step: not the guard's)
    bot, st = _bot(race, _base(race, 5), t=0)
    bot, st = _bot(race, _base(race, 5, [new]), t=1, bot=bot)
    assert _send(bot, st, {50: (MOVE, 0)})[0].__class__.__name__ == "PointOrder"
    assert not getattr(bot, "gold_full_engine", 0)
    # (an idle worker the engine puts to work later, a builder's after its building: the same; past AUTO_GOLD_STEPS
    # since its start it is left alone)
    bot, st = _bot(race, _base(race, 5), t=10)
    rows = _base(race, 5)
    rows[-1, fx.C_ORDER] = fx.HARVEST  # (IDLE, idle at step 10, harvesting at 11 with no order of the agent's)
    rows[-1, fx.C_X] = -2200  # (by the hall: from where it stood, 1,209 from the mine, it would be one walking from afar)
    bot, st = _bot(race, rows, t=11, bot=bot)
    assert bot.view.auto_gold[IDLE] == 11  # (and the start's five at 10: a first step has no step before)
    bot.t = 11 + fx.AUTO_GOLD_STEPS + 1
    assert _send(bot, st, {}) == []
    bot.t = 11
    assert _send(bot, st, {}) == [TargetDestructable(IDLE, fx.HARVEST, TREE_HALL)]


def test_the_agents_own_harvest_is_no_engine_start():
    """A worker the agent sent to gold (its order the step before) is the agent's, checked when sent: no engine start."""
    fx.set_econ_masks(512)
    bot, st = _bot("human", _base("human", 4), t=0)
    cmds = _send(bot, st, {IDLE: (HARVEST, MINE)})
    bot.accepted(cmds, [True])
    moved = _base("human", 4, [])
    moved[-1, fx.C_ORDER] = fx.HARVEST  # (the idle one on its way to the mine now)
    bot, st = _bot("human", moved, t=1, bot=bot)
    assert IDLE not in bot.view.auto_gold and IDLE in bot.view.gold.on[MINE]


def test_the_episode_row_and_the_evaluation_games():
    """The rows count it; the actors and the evaluation worker set the masks from the run's --econ-masks (one key)."""
    src = (paths.REPO_ROOT / "warcraftsim" / "fullgame" / "selfplay.py").read_text()
    body = src[src.index("def play_one("):src.index("def film_game(")]
    for k in ("gold_full_masked", "gold_full_engine", "gold_full_lumber", "gold_full_steps"):
        assert f'"{k}"' in body
    assert src.count('fx.set_econ_masks(cfg.get("econ_masks", 0))') >= 2  # (the actors' processes)
    assert '"econ_masks": args.econ_masks' in src
    worker = (paths.REPO_ROOT / "warcraftsim" / "fullgame" / "evalworker.py").read_text()
    assert '"econ_masks": num("econ_masks", 0, int)' in worker
    assert worker.count('fx.set_econ_masks(cfg.get("econ_masks", 0))') == 2  # (its main and its game processes)
    for f in ("evalworker.py", "render_request.py", "bc.py"):
        assert "set_econ_masks(" in (paths.REPO_ROOT / "warcraftsim" / "fullgame" / f).read_text()


def test_run_config_carries_the_bit(tmp_path):
    import json
    from warcraftsim.fullgame import evalworker
    (tmp_path / "run.json").write_text(json.dumps({"config": {"map": "(2)EchoIsles", "econ_masks": "1023",
                                                               "wine": None, "runs": None}}))
    cfg, _ = evalworker.run_config(tmp_path, "box", 4, None)
    assert cfg["econ_masks"] == 1023
    fx.set_econ_masks(cfg["econ_masks"])
    assert fx.GOLD_CAP
