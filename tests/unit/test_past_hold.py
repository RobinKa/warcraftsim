"""--infer-past-period-ms: csrc/past_hold.h (the C++ loop's rule, compiled with g++ and played through arrivals; the loop
around it needs the GPU: scripts/check_native_infer.py) and the Python server's path end to end on the CPU: every row
answered within its bound, a network's calls spaced by the period, a mixed request answered whole with the right values."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from warcraftsim.fullgame import features as fx, selfplay as sp
from warcraftsim.fullgame.model import FullGameNet

HEADER = Path(sp.__file__).with_name("csrc") / "past_hold.h"
DRIVER = r"""
// reads "time key id" (times rising), plays the loop's use of PastHold: at each arrival what is due leaves; between
// arrivals the loop wakes at next_due(). Prints "out time key id" for each row leaving, "call time key rows" for each
// network's batch, then rows_out, waited and the networks still remembered.
#include "past_hold.h"
#include <cstdio>
#include <iostream>
int main(int argc, char** argv) {
    PastHold<int> h;
    h.period = atof(argv[1]);
    h.full = atoi(argv[2]);
    auto flush = [&](double t) {
        std::map<std::string, std::vector<int>> groups;
        h.take_due(t, groups, [](const int&) {});
        for (auto& [k, v] : groups) {
            printf("call %.6f %s %zu\n", t, k.c_str(), v.size());
            for (int i : v) printf("out %.6f %s %d\n", t, k.c_str(), i);
        }
    };
    double t, end = 0;
    std::string key;
    int id;
    while (std::cin >> t >> key >> id) {
        while (h.holding() && h.next_due() <= t) flush(h.next_due());
        h.add(key, std::vector<int>{id}, t);
        flush(t);
        end = t;
    }
    while (h.holding()) flush(h.next_due());
    flush(end + 10 * h.period + 1);  // (everything forgotten by now)
    printf("stats %ld %.6f %zu\n", h.rows_out, h.waited, h.networks());
}
"""


@pytest.fixture(scope="module")
def driver(tmp_path_factory):
    if shutil.which("g++") is None:
        pytest.skip("no g++")
    d = tmp_path_factory.mktemp("hold")
    (d / "driver.cpp").write_text(DRIVER)
    subprocess.run(["g++", "-std=c++17", "-O1", f"-I{HEADER.parent}", str(d / "driver.cpp"), "-o", str(d / "driver")], check=True)
    return d / "driver"


def play(driver, arrivals: list[tuple[float, str, int]], period: float, full: int = 16):
    text = "".join(f"{t:.6f} {k} {i}\n" for t, k, i in arrivals)
    out = subprocess.run([str(driver), str(period), str(full)], input=text, capture_output=True, text=True, check=True).stdout
    calls, rows, stats = [], {}, None
    for line in out.splitlines():
        kind, *rest = line.split()
        if kind == "call":
            calls.append((float(rest[0]), rest[1], int(rest[2])))
        elif kind == "out":
            assert int(rest[2]) not in rows, "a row left twice"
            rows[int(rest[2])] = float(rest[0])
        else:
            stats = (int(rest[0]), float(rest[1]), int(rest[2]))
    return calls, rows, stats


def test_hold_bounds_every_row_and_spaces_a_networks_calls(driver):
    rng = np.random.default_rng(0)
    t, arrivals = 0.0, []
    for i in range(4000):  # (bursts and quiet spells over a few networks)
        t += float(rng.exponential(0.003 if rng.random() < 0.99 else 0.2))
        arrivals.append((t, f"net{int(rng.integers(5))}", i))
    period = 0.03
    calls, rows, stats = play(driver, arrivals, period)
    assert sorted(rows) == list(range(4000)), "a row lost"
    for tt, _, i in arrivals:
        assert tt <= rows[i] + 1e-6 and rows[i] - tt <= period + 1e-5, "beyond the bound"
    for net in {k for _, k, _ in calls}:
        times = [c[0] for c in calls if c[1] == net]
        assert np.diff(times).min() >= period - 1e-6, "a network called within its period"
    plain = len(arrivals)  # (each row a call of its own, at the worst)
    assert len(calls) < plain / 2
    assert stats[0] == 4000 and stats[2] == 0, "all left, all forgotten"
    assert abs(stats[1] - sum(rows[i] - tt for tt, _, i in arrivals)) < 1e-2


def test_hold_a_quiet_networks_row_goes_at_once(driver):
    calls, rows, _ = play(driver, [(1.0, "a", 0), (5.0, "a", 1), (5.01, "a", 2)], 0.05)
    assert rows[0] == 1.0 and rows[1] == 5.0, "after a quiet spell: no wait"
    assert abs(rows[2] - 5.05) < 1e-6, "within the period: held to its end"


def test_hold_a_full_call_goes_at_once(driver):
    arrivals = [(1.0 + i * 1e-4, "a", i) for i in range(9)]
    calls, rows, _ = play(driver, arrivals, 10.0, full=4)
    assert [c[2] for c in calls] == [1, 4, 4]  # (the first at once, then full ones, none waiting the 10 s)
    assert max(rows.values()) < 1.01


def snapshot_net(tmp_path, seed: int) -> tuple[FullGameNet, Path]:
    torch.manual_seed(seed)
    net = FullGameNet(n_types=20, n_cur=10, n_orders=30, G=30, d=32, layers=1, memory=True).eval()
    net.allowed[:] = True
    path = tmp_path / f"past{seed}.pt"
    torch.save({"model": net.state_dict(), "config": net.config}, path)
    return net, path


def test_python_server_holds_past_rows_and_answers_every_game(tmp_path, monkeypatch):
    """Four games asking at once, each request a current row and a past one (or two past ones): each answered whole,
    values a direct call's, within the bound; a past network's calls spaced by the period, with more rows a call
    than without. No request starves when all the games wait on held rows."""
    import torch.multiprocessing as mp

    net, _ = snapshot_net(tmp_path, 0)
    net_a, past_a = snapshot_net(tmp_path, 1)
    net_b, past_b = snapshot_net(tmp_path, 2)
    sp.publish(net, 4, tmp_path)
    spec = sp.SharedRows(str(tmp_path / "rows"), 8, 30, 30, 32, create=True).spec
    period = 0.05

    rounds: list = []
    launch = sp.launch_round

    def recording(groups, *a, **k):
        rounds.append((time.time(), {key: len(items) for key, items in groups.items()}))
        return launch(groups, *a, **k)

    monkeypatch.setattr(sp, "launch_round", recording)
    direct = sp.Inference(None, torch.device("cpu"), 8, compile_=False, thread=False)
    pipes = [mp.get_context("spawn").Pipe() for _ in range(4)]
    stop = threading.Event()
    cfg = {"run_dir": str(tmp_path), "device": "cpu", "learner_pid": os.getpid(), "infer_batch": 8, "compile": False, "max_past": 4,
           "shared_rows": spec, "infer_pipeline": 1, "infer_past_period": period}
    server = threading.Thread(target=sp.inference_main, args=(cfg, [b for _, b in pipes], stop, True), daemon=True)
    server.start()
    infer = sp.RemoteInference([a for a, _ in pipes], sp.SharedRows(*spec), [0, 2, 4, 6])
    errors: list = []
    waits: list = []

    def game(seed: int, until: float):
        rng = np.random.default_rng(seed)
        while time.time() < until:
            n = int(rng.integers(3, 12))
            st = {"n": n, "n_own": 2, "ent": rng.normal(size=(n, fx.F)).astype(np.float32), "type": rng.integers(1, 20, n),
                  "cur": rng.integers(0, 10, n), "glob": rng.normal(size=30).astype(np.float32), "avail": None, "h": None}
            pair = [("current", net), (str(past_a), net_a)] if seed % 2 else [(str(past_a), net_a), (str(past_b), net_b)]
            t = time.time()
            got = infer.request([(k, dict(st)) for k, _ in pair])
            waits.append(time.time() - t)
            for (key, which), res in zip(pair, got):
                want = direct._finish(direct._launch(which, [dict(st)]))[0]
                if res is None or abs(res["value"] - want["value"]) > 1e-4 or res["version"] != (4 if key == "current" else -1):
                    errors.append((key, None if res is None else res["value"], want["value"]))

    try:
        until = time.time() + 2.0
        games = [threading.Thread(target=game, args=(s, until), daemon=True) for s in range(4)]
        [g.start() for g in games]
        [g.join(timeout=60) for g in games]
        assert not any(g.is_alive() for g in games), "a game waits for its answer"
    finally:
        stop.set()
        server.join(timeout=20)
    assert not errors, errors[:3]
    assert len(waits) >= 8
    past_rounds = [(t, r[str(past_a)]) for t, r in rounds if str(past_a) in r]
    gaps = np.diff([t for t, _ in past_rounds])
    assert gaps.min() >= period * 0.9, "a past network's calls within its period"
    assert np.mean([k for _, k in past_rounds]) > 1, "its rows batched"
