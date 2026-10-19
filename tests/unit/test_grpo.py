"""GRPO from mid-game branches (selfplay.run_branch): held trajectories and the group-relative advantage."""
from warcraftsim.fullgame import selfplay as sp


class _Q(list):
    def put(self, x):
        self.append(x)


def _traj(rewards, values, hold=True):
    q = _Q()
    tr = sp.Trajectory({"gamma": 1.0, "lam": 1.0, "chunk": 2}, q, {"worker": 0}, hold=hold)
    tr.steps = [{"reward": r, "value": v, "done": False, "version": 0} for r, v in zip(rewards, values)]
    return tr, q


def test_held_trajectory_releases_with_the_bonus_in_pieces():
    tr, q = _traj([0.0, 0.0, 0.0], [0.1, 0.2, 0.3])
    tr.end(1.0)
    assert q == []  # (held: nothing sent at the end)
    tr.release(0.5)
    steps = [s for m in q for s in m["steps"]]
    assert [len(m["steps"]) for m in q] == [2, 1]  # (in pieces of --chunk)
    assert [round(s["adv"], 6) for s in steps] == [1.4, 1.3, 1.2]  # (GAE with gamma = lambda = 1: 1 - v, plus 0.5)
    assert [round(s["ret"], 6) for s in steps] == [1.0, 1.0, 1.0]  # (the value's target without the bonus)


def test_group_relative_returns_per_side():
    got = {}

    class T:
        def __init__(self, name):
            self.name = name

        def release(self, bonus):
            got[self.name] = bonus

    games = [({0: T("a0"), 1: T("a1")}, {0: 1.0, 1: -1.0}), ({0: T("b0"), 1: T("b1")}, {0: -1.0, 1: 1.0}),
             ({0: T("c0")}, {0: 0.0})]
    sp.release_group(games, 2.0)
    assert got == {"a0": 2.0, "b0": -2.0, "c0": 0.0, "a1": -2.0, "b1": 2.0}
    sp.release_group([({0: T("alone")}, {0: 3.0})], 1.0)
    assert got["alone"] == 0.0
