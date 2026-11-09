"""--learner-intake (selfplay.Intake): the learner reads the actors' queue ahead during the update's pauses (and, with 2,
while it waits for the GPU) instead of only between updates. The collect loop takes those messages first and in the
order they arrived, so the batches, their steps' staleness and every minibatch the update makes from them are the same,
bit for bit, as without reading ahead, however the reads fall between the batches and the minibatches."""

from __future__ import annotations

import multiprocessing as mp
import queue
import random
import threading
import time
import types

import numpy as np
import pytest
import torch

from warcraftsim.fullgame import features as fx
from warcraftsim.fullgame import selfplay as sp

BATCH, CHUNK, SEQ, MB = 512, 32, 8, 64


def _messages(seed: int = 0, n: int = 120) -> list[dict]:
    """A stream as the actors send it: trajectory chunks (some short: games ending), episodes, an exploiter's chunk."""
    rng = np.random.default_rng(seed)
    out, sid = [], 0
    for k in range(n):
        if k % 9 == 4:
            out.append({"episode": {"outcome": float(rng.choice([-1, 0, 1])), "id": k}})
            continue
        steps = []
        for t in range(int(rng.choice([CHUNK, CHUNK, CHUNK, 5, 17]))):
            e, o = int(rng.integers(3, 40)), int(rng.integers(1, 6))
            steps.append({"ent": rng.normal(size=(e, fx.F)).astype(np.float32), "type": rng.integers(0, 20, e),
                          "cur": rng.integers(0, 10, e), "glob": rng.normal(size=12).astype(np.float32), "n": e,
                          "n_own": o, "order": rng.integers(0, 30, o), "tgt": rng.integers(0, e, o),
                          "bx": rng.integers(0, 50, o), "by": rng.integers(0, 50, o),
                          "logp": rng.normal(size=o).astype(np.float32), "value": float(rng.normal()), "reward": 0.0,
                          "avail": rng.random(30) < 0.6, "h": rng.normal(size=8).astype(np.float32) if t else None,
                          "done": False, "version": int(rng.integers(0, 4)), "adv": float(rng.normal()),
                          "ret": float(rng.normal()), "uadv": float(rng.normal()), "sid": sid})
            sid += 1
        info = {"learner": "exploiter"} if k % 13 == 7 else {"worker": 0}
        out.append({"steps": sp.PackedSteps(steps, raw=True), "info": info})
    return out


def _producer(q, msgs, gap: float = 0.0):
    for m in msgs:
        q.put(m)
        if gap:
            time.sleep(gap)


def _collect(intake: sp.Intake, update: int, between=None) -> tuple[list, list, list, list]:
    """The learner's collect loop's routing (selfplay.main): episodes apart, the exploiter's steps apart, the rest into
    the batch until it holds BATCH steps; `between` called after each message (reads ahead may fall anywhere)."""
    buf, chunks, stale, other = [], [], [], []
    while len(buf) < BATCH:
        msg = intake.get(timeout=30)
        if "episode" in msg:
            other.append(("episode", msg["episode"]["id"]))
        elif (msg.get("info") or {}).get("learner") == "exploiter":
            other.append(("exploiter", [s["sid"] for s in msg["steps"]]))
        else:
            buf += msg["steps"]
            chunks.append(msg["steps"])
            stale += [update - s["version"] for s in msg["steps"] if s["version"] >= 0]
        if between is not None:
            between()
    return buf, chunks, stale, other


def _updater() -> sp.GraphedUpdate:
    upd = sp.GraphedUpdate.__new__(sp.GraphedUpdate)  # (its minibatches alone: _batches, on the CPU)
    upd.args = types.SimpleNamespace(seq_len=SEQ, minibatch=MB, pad_groups=4, opd_coef=0.0, bc_coef=0.0, epochs=1)
    upd.device, upd.widths = torch.device("cpu"), sp.update_widths("8,16,24,32,40")
    return upd


def _run(msgs, ahead: int, seed: int, reads: bool) -> tuple[list, int]:
    """Three updates' batches and minibatches from `msgs` through a multiprocessing queue (pickled as the actors' are),
    an actor thread putting them; with `reads`, Intake.fill at random moments: between the collect loop's messages,
    between the minibatches (the pauses) and with wait (the waits for the GPU)."""
    q = mp.get_context("spawn").Queue(maxsize=8)  # (small: the actor's puts block while nobody reads)
    actor = threading.Thread(target=_producer, args=(q, msgs, 0.0005), daemon=True)
    actor.start()
    intake, upd, rng = sp.Intake(q, BATCH, ahead), _updater(), random.Random(seed)
    fill = (lambda: intake.fill(rng.choice([0.0, 0.001, 0.004]), wait=rng.random() < 0.3)) if reads else None
    np.random.seed(5)  # (the minibatches' order: np.random, as the learner's)
    out, ahead_read = [], 0
    for update in range(3):
        buf, chunks, stale, other = _collect(intake, update, fill if reads and rng.random() < 0.5 else None)
        mbs = []
        for key, mb, _ in upd._batches(chunks, None, 1):
            mbs.append((key, {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in mb.items()}))
            if fill is not None:  # (the update's pause after each minibatch)
                fill()
        ahead_read += intake.stats()["intake/messages"]
        out.append((buf, stale, other, mbs))
    while True:  # (the rest of the stream read off: the actor's puts and the queue's feeder thread end)
        try:
            q.get(timeout=0.2)
        except queue.Empty:
            if not actor.is_alive():
                break
    q.close()
    q.join_thread()
    return out, ahead_read


def _same_step(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and x.dtype == y.dtype
                    and np.array_equal(x, y)):
                return False
        elif x != y:
            return False
    return True


@pytest.mark.parametrize("ahead", [1, 2])
def test_same_batches_and_minibatches_bit_for_bit(ahead):
    msgs = _messages()
    base, none = _run(msgs, 0, seed=1, reads=False)
    assert none == 0
    for seed in (1, 2):
        got, read = _run(msgs, ahead, seed=seed, reads=True)
        assert read > 10  # (messages did come through the reads ahead)
        for (b0, s0, o0, m0), (b1, s1, o1, m1) in zip(base, got):
            assert [s["sid"] for s in b0] == [s["sid"] for s in b1]
            assert all(_same_step(x, y) for x, y in zip(b0, b1))
            assert s0 == s1 and o0 == o1
            assert len(m0) == len(m1)
            for (k0, x0), (k1, x1) in zip(m0, m1):
                assert k0 == k1 and x0.keys() == x1.keys()
                for k in x0:
                    if isinstance(x0[k], torch.Tensor):
                        assert x0[k].dtype == x1[k].dtype and torch.equal(x0[k], x1[k]), k
                    elif k == "seq":
                        assert x0[k][:2] == x1[k][:2] and torch.equal(x0[k][2], x1[k][2])
                    else:
                        assert x0[k] == x1[k], k


def test_reads_ahead_in_order_and_up_to_a_batch():
    q = queue.Queue()
    msgs = _messages(3, n=40)
    for m in msgs:
        q.put(m)
    intake = sp.Intake(q, BATCH, ahead=1)
    t = time.monotonic()
    intake.fill(0.05)  # (everything is there: it reads until a batch's steps wait, then sleeps the rest)
    assert time.monotonic() - t >= 0.045
    waiting = sum(len(m["steps"]) for m in intake.pending if "steps" in m)
    assert intake.steps == waiting and waiting >= BATCH
    assert waiting - max(len(m.get("steps", ())) for m in intake.pending) < BATCH  # (it stopped right after)
    st = intake.stats()
    assert st["intake/messages"] == len(intake.pending) and st["intake/steps"] == waiting
    assert st["intake/waiting_steps"] == waiting and intake.stats()["intake/messages"] == 0
    got = [intake.get(timeout=1) for _ in msgs]  # (those read ahead, then the queue's: all in order)
    assert [id(m) for m in got] == [id(m) for m in msgs] and intake.steps == 0
    with pytest.raises(queue.Empty):
        intake.get(timeout=0.01)


def test_a_failed_read_ahead_meets_the_collect_loop():
    """A read ahead the queue fails (a message it can't give): not raised inside the update's pause (there it had read as
    a CUDA-graph failure: the eager update over the same batch again, the graphs gone for the run), but by get() after
    the messages read before it, where the collect loop had met it without reading ahead; the queue's next after it."""
    class Q:
        def __init__(self, items):
            self.items = list(items)

        def get(self, timeout=None):
            if not self.items:
                raise queue.Empty
            x = self.items.pop(0)
            if isinstance(x, BaseException):
                raise x
            return x
    a, b, c = {"steps": [1, 2]}, {"episode": {"outcome": 1.0}}, {"steps": [3]}
    err = EOFError("a message cut short")
    intake = sp.Intake(Q([a, b, err, c]), BATCH, ahead=1)
    intake.fill(0.01)  # (no exception here)
    intake.fill(0.01, wait=True)
    assert list(intake.pending) == [a, b] and intake.steps == 2
    assert intake.get(timeout=1) is a and intake.get(timeout=1) is b
    with pytest.raises(EOFError):
        intake.get(timeout=1)
    assert intake.get(timeout=1) is c


def test_off_and_waits():
    q = queue.Queue()
    q.put({"steps": [1, 2, 3]})
    off = sp.Intake(q, BATCH, ahead=0)
    t = time.monotonic()
    off.fill(0.02)  # (off: a sleep, nothing read)
    assert time.monotonic() - t >= 0.019 and not off.pending and q.qsize() == 1
    on = sp.Intake(q, BATCH, ahead=1)
    q.put({"steps": [4]})
    t = time.monotonic()
    on.fill(0.5, wait=True)  # (waiting for the GPU: one message, then back to its poll, nothing slept)
    assert len(on.pending) == 1 and time.monotonic() - t < 0.25
    on.fill(0.01, wait=True)
    assert len(on.pending) == 2 and on.steps == 4
    t = time.monotonic()
    on.fill(0.01, wait=True)  # (nothing there: its timeout)
    assert 0.009 <= time.monotonic() - t < 0.2 and len(on.pending) == 2
    capped = sp.Intake(q, 1, ahead=1)
    q.put({"steps": [5]})
    capped.fill(0.01, wait=True)
    t = time.monotonic()
    capped.fill(0.01, wait=True)  # (a batch waits: no read, asleep, no spinning)
    assert 0.009 <= time.monotonic() - t and len(capped.pending) == 1


def test_update_pause_reads_instead_of_sleeping(monkeypatch):
    """GraphedUpdate._pause: without an intake it waits for the GPU and sleeps (as before); with one it reads instead
    of sleeping, and with --learner-intake 2 also until the minibatch's GPU work is done."""
    calls = []

    class Stream:
        def synchronize(self):
            calls.append("sync")

    class Event:
        n = 0

        def record(self, stream):
            calls.append("record")

        def query(self):
            Event.n += 1
            return Event.n > 3
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: Stream())
    monkeypatch.setattr(torch.cuda, "Event", Event)
    slept = []
    monkeypatch.setattr(sp.time, "sleep", lambda s: slept.append(s))
    upd = sp.GraphedUpdate.__new__(sp.GraphedUpdate)
    upd.idle, upd.idle_waits, upd._done_event = None, False, None
    upd._pause(0.008)
    assert calls == ["sync"] and slept == [0.008]
    calls.clear()
    upd.idle = lambda s, wait=False: calls.append(("idle", s, wait))
    upd._pause(0.008)
    assert calls == ["sync", ("idle", 0.008, False)] and slept == [0.008]
    calls.clear()
    upd.idle_waits = True
    upd._pause(0.008)
    assert calls == ["record"] + [("idle", 0.001, True)] * 3 + ["sync", ("idle", 0.008, False)]


def test_needs_the_pauses(capsys, tmp_path):
    with pytest.raises(SystemExit):
        sp.main(["--name", "intake-test", "--init", str(tmp_path / "none.pt"), "--learner-intake", "1"])
    assert "--learner-intake reads the actors' queue in the update's pauses" in capsys.readouterr().err
