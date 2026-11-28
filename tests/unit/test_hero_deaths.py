"""The episode row's hero deaths (play.BCAgent._watch_heroes, _hero_death; docs/proposals/weakness-analysis-2026-11-28.md,
section 2: the policy's hero died first in 49 of 58 losses, 57 of 141 deaths at level 1, and the traces could not say
why). Logging only: each own hero's death with its hit points' trend, the armies around it, its distance from home and
from its army, the town portal and healing items it died with, the killer."""

import json
from types import SimpleNamespace

import numpy as np

from warcraftsim import paths
from warcraftsim.fullgame import features as fx

from test_gold_cap import HX, HY, VOCAB, _unit, code


def _hero_bot():
    from warcraftsim.fullgame.play import BCAgent
    bot = BCAgent(None, VOCAB, 0, None)
    bot.enc = fx.Encoder(VOCAB, np.zeros((len(VOCAB["orders"]) + 1, 3), np.int64))
    bot.view = bot.enc.view(0, 1.0, [0, 1])
    bot.unit_values = {str(code("hfoo")): 135, str(code("ogru")): 200, str(code("Hpal")): 425}
    return bot


def test_a_hero_death_recorded():
    fx.set_econ_masks(0)
    bot = _hero_bot()
    hall = _unit(1, "htow", fx.STRUCTURE, x=HX, y=HY)
    feet = [_unit(60 + k, "hfoo", x=-500 + 10 * k, y=0) for k in range(3)]  # (the army's centre ~500 from the hero)
    grunts = [_unit(70 + k, "ogru", owner=1, x=-100, y=50 * k) for k in range(4)]
    far = _unit(80, "ogru", owner=1, x=3000, y=0)  # (not near it)
    for t in range(25):  # (12.5 s alive, its hit points falling from 100%)
        hp = max(1, 500 - 20 * t)
        pal = _unit(5, "Hpal", fx.HERO, x=0, y=0, hp=hp, maxhp=500, items=("stwp", "phea"))
        pal[fx.C_HLEVEL] = 2
        bot._watch_heroes(np.stack([hall, pal, *feet, *grunts, far]), [], 0.5 * t)
    assert not getattr(bot, "hero_deaths", None)
    dead = _unit(5, "Hpal", fx.HERO | fx.DEAD, x=0, y=0, hp=0, maxhp=500, items=("stwp", "phea"))
    death = SimpleNamespace(kind=1, a=5, b=70, c=code("Hpal"))
    bot._watch_heroes(np.stack([hall, dead, *feet, *grunts, far]), [death], 12.5)
    (rec,) = bot.hero_deaths
    assert rec["s"] == 12.5 and rec["hero"] == "Hpal" and rec["lvl"] == 2
    # (every 2 s over the last 10, oldest first: steps 4, 8, ..., 24 -> 500 - 20 t of 500)
    assert rec["hp"] == [round((500 - 20 * t) / 500, 2) for t in (4, 8, 12, 16, 20, 24)]
    assert rec["own"] == [3, 405] and rec["enemy"] == [4, 800] and "creeps" not in rec
    assert rec["hall_d"] == 2000 and rec["near_hall"] is False
    assert rec["army_n"] == 3 and 480 <= rec["army_d"] <= 500
    assert rec["tp"] is True and rec["heal"] is True and rec["killer"] == "ogru"
    # (dead, it stays dead: no second record; revived, a new life's history)
    bot._watch_heroes(np.stack([hall, dead, *feet]), [], 13.0)
    assert len(bot.hero_deaths) == 1
    json.dumps(bot.hero_deaths)  # (the episode row: plain numbers)


def test_hero_death_near_home_with_creeps():
    bot = _hero_bot()
    hall = _unit(1, "htow", fx.STRUCTURE, x=HX, y=HY)
    creep = _unit(90, "ogru", owner=12, x=-1500, y=0)
    pal = _unit(5, "Hpal", fx.HERO, x=-1000, y=0, hp=50, maxhp=500)
    bot._watch_heroes(np.stack([hall, pal, creep]), [], 100.0)
    bot._watch_heroes(np.stack([hall, creep]), [], 100.5)  # (gone from the rows, no death: no record)
    assert not bot.hero_deaths and not bot._hero_rows
    bot._watch_heroes(np.stack([hall, pal, creep]), [], 101.0)
    dead = pal.copy()
    dead[fx.C_FLAGS] |= fx.DEAD
    bot._watch_heroes(np.stack([hall, dead, creep]), [], 101.5)  # (its row dead, no event seen: died)
    (rec,) = bot.hero_deaths
    assert rec["hp"] == [0.1] and rec["near_hall"] is True and rec["hall_d"] == 1000
    assert rec["creeps"] == [1, 200] and rec["enemy"] == [0, 0] and rec["own"] == [0, 0]
    assert rec["army_d"] is None and rec["army_n"] == 0 and rec["tp"] is False and rec["killer"] is None
    assert not bot._hero_rows


def test_the_episode_row_lists_them():
    """play_one gives each bot the units' costs, and the learner's row its deaths (only when it had any)."""
    src = (paths.REPO_ROOT / "warcraftsim" / "fullgame" / "selfplay.py").read_text()
    body = src[src.index("def play_one("):src.index("def film_game(")]
    assert 'bot.unit_values = cfg.get("values")' in body
    assert ('**({"hero_deaths": bots[side].hero_deaths} if side in bots and getattr(bots[side], "hero_deaths", None)'
            in body)
