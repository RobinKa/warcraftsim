"""The whole-game camera director: a fight needs blows struck; units idle beside enemies are no fight."""
from types import SimpleNamespace as NS

from warcraftsim.video import Director


def _u(i, owner, x, y, hp, structure=False):
    return NS(id=i, owner=owner, x=x, y=y, hp=hp, alive=True, is_structure=structure, is_worker=False)


def test_idle_units_beside_enemies_are_no_fight():
    d = Director(1, [1, 0])
    base = [_u(1, 0, 4000, 3000, 900, structure=True), _u(2, 0, 4100, 3000, 60)]  # (its building, a wisp)
    ghouls = [_u(10, 1, 4200, 3000, 70), _u(11, 1, 4250, 3000, 7)]  # (idle there: never hurt)
    army = [_u(20, 1, -5000, 3000, 400), _u(21, 1, -5100, 3000, 400), _u(22, 1, -5050, 3100, 400)]  # (its army at home)
    obs = NS(units=base + ghouls + army, players={0: None, 1: None})
    for _ in range(3):
        (cx, _), group = d.focus(obs, (0.0, 0.0))
    assert cx < 0 and {u.id for u in group} >= {20, 21}  # (the shots: the learner's army, not the "fight")
    obs.units[2].hp = 50  # (a ghoul hit, 70 -> 50: blows struck, a fight)
    (cx, _), group = d.focus(obs, (0.0, 0.0))
    assert cx > 3000 and {10, 11} <= {u.id for u in group}
