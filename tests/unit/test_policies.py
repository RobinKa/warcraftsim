"""A run's policies as things of their own (dashboard/policies.py): a run's launches with their flags and records,
the Checkpoints section's listing (where a policy lies on the run's timeline, its records from the ratings' view), and
an evaluation burst's own identity (its run.json is a copy of the run's whose policy it plays)."""

import json
import os
import pickle
import zipfile

from warcraftsim.dashboard import policies, ratings

ECHO = {"eval_map": "(2)EchoIsles"}
INSANE = "script:ai-insane (real)"


def _row(key, outcome, t, opp=INSANE, race="orc", opp_race="human", **more):
    return json.dumps({"eval": True, "eval_key": key, "outcome": outcome, "time": t, "opponent": opp, "race": race,
                       "opponent_race": opp_race, **more})


def _run(root, name, rows=(), created=0.0, file="episodes.jsonl", **info):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps({"name": name, "created": created, "config": ECHO, **info}))
    if rows:
        (d / file).write_text("\n".join(rows) + "\n")
    return d


def _checkpoint(d, steps, when):
    (d / "checkpoints").mkdir(exist_ok=True)
    f = d / "checkpoints" / f"{steps:016d}.pt"
    f.write_bytes(b"x")
    os.utime(f, (when, when))
    return f


def test_flag_changes_between_launches():
    a = ["--name", "r", "--lr", "3e-5", "--tie-value", "-0.5", "--resume", "--note", "why --not a flag", "--mirror", "0"]
    b = ["--name", "r", "--lr", "2e-5", "--tie-value", "-0.5", "--first-hero-weights", "Ofar:4,Hamg:4", "--fused",
         "--title", "a new title"]
    assert policies.flags_of(a)["--tie-value"] == "-0.5" and policies.flags_of(b)["--fused"] == ""
    assert policies.flag_changes(a, b) == [
        {"flag": "--first-hero-weights", "from": None, "to": "Ofar:4,Hamg:4"}, {"flag": "--fused", "from": None, "to": ""},
        {"flag": "--lr", "from": "3e-5", "to": "2e-5"}, {"flag": "--mirror", "from": "0", "to": None}]
    assert policies.flag_changes(None, b) is None and policies.flag_changes(a, a) == []


def test_launches_are_stretches_with_their_own_numbers():
    """The start and each restart: its words, its time and steps, its code, its flags against the launch before, and
    the evaluation record of its own time."""
    info = {"created": 1000.0, "title": "first hero weighted", "notes": "Audit 20: the first hero decides.\n\nLater: kept.",
            "first_launch": {"git": {"commit": "a" * 40, "subject": "one"}, "argv": ["--lr", "3e-5"]},
            "launch": {"git": {"commit": "c" * 40, "subject": "three", "dirty": True}, "argv": ["--lr", "1e-5", "--resume"]},
            "resumes": [{"time": 5000.0, "steps": 400, "title": "control for 237", "note": "the control", "checkpoint": "400.pt",
                         "git": {"commit": "b" * 40, "subject": "two"}, "argv": ["--lr", "2e-5", "--resume", "--note", "x"]},
                        {"time": 9000.0, "steps": 700}]}
    train = [{"time": 1100.0, "agent_steps": 100, "uptime": 100.0}, {"time": 3000.0, "agent_steps": 400, "uptime": 2000.0},
             {"time": 5100.0, "agent_steps": 500, "uptime": 90.0}, {"time": 6000.0, "agent_steps": 700, "uptime": 990.0},
             {"time": 9100.0, "agent_steps": 800, "uptime": 60.0}]
    ev = lambda t, key, x, opp=INSANE: {"time": t, "eval": True, "eval_key": key, "outcome": x, "opponent": opp}  # noqa: E731
    episodes = [{"time": 1200.0, "outcome": 1.0}, ev(1300.0, "current", 1.0), ev(1400.0, "current", -1.0),
                ev(1500.0, "current", 1.0, "script:ai-insane (real, agent -25%)"),  # (a handicap rung: not the yardstick)
                ev(5200.0, "current", 0.0), ev(5300.0, "vs fgself-9/0000000000000042.pt", 1.0, "eval-vs:x"),
                ev(5400.0, "vs fgself-9/swa-1-2.pt", -1.0, "eval-vs:y"), {"time": 5500.0, "outcome": -1.0}]
    a, b, c = policies.launches(info, train, episodes)
    assert (a["title"], a["note"], a["steps"], a["to_steps"], a["trained"]) == \
        ("first hero weighted", "Audit 20: the first hero decides.", 0, 400, 2000.0)
    assert a["commit"] == "a" * 12 and a["flags"] is None and a["games"] == 1 and a["insane"] == [1, 0, 1] and a["vs"] == {}
    assert (b["title"], b["note"], b["time"], b["end"], b["steps"], b["to_steps"], b["trained"], b["checkpoint"]) == \
        ("control for 237", "the control", 5000.0, 6000.0, 400, 700, 990.0, "400.pt")
    assert b["flags"] == [{"flag": "--lr", "from": "3e-5", "to": "2e-5"}] and b["commit"] == "b" * 12
    assert b["insane"] == [0, 1, 0] and b["vs"] == {"fgself-9/42": [1, 0, 0], "fgself-9 avg swa-1-2": [0, 0, 1]} and b["games"] == 1
    # the newest restart without its own record: run.json "launch" is its launch
    assert c["commit"] == "c" * 12 and c["dirty"] and c["flags"] == [{"flag": "--lr", "from": "2e-5", "to": "1e-5"}]
    assert c["to_steps"] == 800 and c["insane"] == [0, 0, 0]
    # a run restarted before run.json kept the first launch's command line: the first restart's flags are not known
    old = {**info, "first_launch": {"git": {"commit": "a" * 40}}}
    assert policies.launches(old, train)[1]["flags"] is None and "insane" not in policies.launches(old, train)[1]
    # a run never restarted: one launch, run.json "launch" its own
    one = policies.launches({"created": 1000.0, "launch": info["launch"]}, train)
    assert len(one) == 1 and one[0]["commit"] == "c" * 12 and one[0]["to_steps"] == 800


def test_averaged_entry_read_without_torch(tmp_path):
    f = tmp_path / "swa-43-53.pt"
    entry = {"n": 31, "from": "runs/fgself-236/checkpoints/0000000043410057.pt",
             "to": "runs/fgself-236/checkpoints/0000000053275744.pt"}
    with zipfile.ZipFile(f, "w") as z:
        z.writestr("archive/data.pkl", pickle.dumps({"model": {"w": [1.0, 2.0]}, "config": {"n": 7, "from": "elsewhere"},
                                                     "averaged": entry}, protocol=2))
    assert policies.averaged(f) == {"n": 31, "from": 43410057, "to": 53275744}
    (tmp_path / "plain.pt").write_bytes(b"not a zip")
    assert policies.averaged(tmp_path / "plain.pt") is None and policies.averaged(tmp_path / "none.pt") is None
    assert policies._span_of_name("swa-118808740-144699411") == {"from": 118808740, "to": 144699411}
    assert policies._span_of_name("swa-43-53") == {"from": 43_000_000, "to": 53_999_999, "rounded": True}


def test_a_burst_is_its_own_run(tmp_path):
    """Its run.json is a copy of the policy's run's (burst_prep.py): its folder's name, the policy as what it came
    from, none of that run's restarts, its own time and state."""
    src = {"name": "fgself-236", "created": 5.0, "status": "training", "init_from": "runs/fgself-235/checkpoints/0000000000000007.pt",
           "resumes": [{"steps": 3, "title": "control"}], "inputs": {"start_weights": "x"}, "finished": 9.0}
    d = _run(tmp_path, "burst-236b", **{k: v for k, v in src.items() if k not in ("name", "created")})
    (d / "run.json").write_text(json.dumps({**src, "config": ECHO}))
    assert policies.burst(d) is None and policies.own(d, src) is src  # (no burst.json: a run like any other)
    (d / "burst.json").write_text(json.dumps({"current": "runs/fgself-236/checkpoints/0000000027296274.pt",
                                              "evals": ["runs/fgself-235/averaged/swa-21-31.pt"], "note": "why"}))
    os.utime(d / "burst.json", (2000.0, 2000.0))
    b = policies.burst(d, now=3000.0)
    assert (b["policy"], b["run"], b["id"], b["kind"]) == ("fgself-236/27296274", "fgself-236", "27296274", "checkpoint")
    assert b["evals"] == [{"name": "fgself-235 avg swa-21-31", "run": "fgself-235", "id": "swa-21-31", "kind": "average"}]
    assert b["created"] == 2000.0 and b["status"] == "prepared"
    info = policies.own(d, src, now=3000.0)
    assert info["name"] == "burst-236b" and info["created"] == 2000.0 and info["status"] == "prepared"
    assert info["init_from"] == "runs/fgself-236/checkpoints/0000000027296274.pt" and "resumes" not in info and "finished" not in info
    assert src["name"] == "fgself-236" and "resumes" in src  # (the copy it was given: left alone)
    (d / "remote-worker.json").write_text(json.dumps({"pid": 7, "time": 2990.0, "host": "tora-burst"}))
    assert policies.burst(d, now=3000.0)["status"] == "playing" and policies.burst(d, now=4000.0)["status"] == "finished"
    (d / "remote-worker.json").write_text(json.dumps({"pid": None, "stopped": 2995.0, "time": 2995.0}))
    os.utime(d / "remote-worker.json", (2996.0, 2996.0))
    assert policies.burst(d, now=3000.0)["status"] == "finished"


def test_listing_places_each_policy_on_the_runs_timeline(tmp_path):
    """A rated checkpoint, a weight average and an hour of the run: steps, when saved, which launch; the records from
    the ratings' view; who started from it and which burst plays it."""
    created = 100000.0
    d = _run(tmp_path, "fgself-5", created=created, rows=[
        _row("current", 1, created + 3 * 3600 + 10), _row("current", -1, created + 3 * 3600 + 20, race="human", opp_race="orc"),
        _row("vs fgself-4/0000000000000900.pt", 0, created + 3 * 3600 + 30, "eval-vs:x")],
        resumes=[{"time": created + 2 * 3600, "steps": 3000, "title": "second"}])
    _checkpoint(d, 2000, created + 3000.0)
    _checkpoint(d, 3500, created + 3 * 3600 + 600.0)
    _checkpoint(d, 4000, created + 3 * 3600 + 1800.0)
    (d / "averaged").mkdir()
    with zipfile.ZipFile(d / "averaged" / "swa-2-4.pt", "w") as z:
        z.writestr("a/data.pkl", pickle.dumps({"averaged": {"n": 2, "from": "runs/fgself-5/checkpoints/0000000000002000.pt",
                                                             "to": "runs/fgself-5/checkpoints/0000000000004000.pt"}}, protocol=2))
    (d / "averaged" / "swa-7-9.pt").write_bytes(b"x")  # (unreadable, never rated: listed by its name)
    # a burst plays the 4000 checkpoint against the average; a later run started from the average
    b = _run(tmp_path, "burst-5", file="episodes-remote.jsonl", rows=[
        _row("current", 1, created + 90000.0), _row("current", 1, created + 90010.0, "script:ai-insane (real, agent -25%)"),
        _row("runs/fgself-5/averaged/swa-2-4.pt", -1, created + 90020.0),
        _row("vs fgself-5/swa-2-4.pt", 1, created + 90030.0, "eval-vs:x")])
    (b / "burst.json").write_text(json.dumps({"current": "runs/fgself-5/checkpoints/0000000000004000.pt",
                                              "evals": ["runs/fgself-5/averaged/swa-2-4.pt"]}))
    _run(tmp_path, "fgself-6", init_from="runs/fgself-5/averaged/swa-2-4.pt",
         inputs={"init": "runs/fgself-5/checkpoints/0000000000002000.pt"})
    r = ratings.Ratings(tmp_path)
    assert r.scan() and r.view is not None
    train = [{"time": created + 60.0 * k, "agent_steps": 20 * k} for k in range(1, 241)]  # (20 steps a minute for 4 hours)
    info = policies.own(d, json.loads((d / "run.json").read_text()))
    L = policies.listing(tmp_path, d, info, r.view, train, replays=lambda names: {n: 3 for n in names})
    P = {p["id"]: p for p in L["policies"]}
    assert L["run"] == "fgself-5" and L["rated"] and L["launches"] == 2 and L["burst"] is None
    assert set(P) == {"h3", "4000", "2000", "swa-2-4", "swa-7-9"}  # (3500: saved, never rated or used: the files' table)
    ck = P["4000"]
    assert (ck["kind"], ck["label"], ck["steps"], ck["saved_by"], ck["launch"], ck["file"]) == \
        ("checkpoint", "checkpoint 4,000", 4000, "file", 1, "checkpoints/0000000000004000.pt")
    assert ck["path"].endswith("fgself-5/checkpoints/0000000000004000.pt") and ck["bursts"] == ["burst-5"]
    rated = ck["rated"][0]
    assert rated["name"] == "fgself-5/4000" and rated["games"] == 3 and rated["played_by"][0]["run"] == "burst-5"
    assert rated["played_by"][0]["burst"] and rated["played_by"][0]["games"] == 3
    opp = {o["name"]: o for o in rated["opponents"]}
    assert list(opp) == ["insane AI", "insane AI vs -25%", "fgself-5 avg swa-2-4"]  # (the anchor, the rungs, head to head)
    assert opp["insane AI"]["record"] == [1, 0, 0] and opp["insane AI"]["races"]["orc"]["human"] == [1, 0, 0]
    assert opp["fgself-5 avg swa-2-4"]["record"] == [1, 0, 0] and opp["fgself-5 avg swa-2-4"]["policy"] == "swa-2-4"
    assert ck["replays"] == {"q": "fgself-5/4000", "count": 3}
    avg = P["swa-2-4"]
    assert (avg["kind"], avg["n"], avg["from_steps"], avg["to_steps"], avg["launch"]) == ("average", 2, 2000, 4000, 1)
    assert avg["forked_by"] == ["fgself-6"] and avg["listed_in"] == ["burst-5"] and avg["file"] == "averaged/swa-2-4.pt"
    assert {o["name"]: o["record"] for o in avg["rated"][0]["opponents"]} == {"insane AI": [0, 0, 1], "fgself-5/4000": [0, 0, 1]}
    assert P["swa-7-9"]["rated"] == [] and (P["swa-7-9"]["from_steps"], P["swa-7-9"]["to_steps"]) == (7_000_000, 9_999_999)
    # a checkpoint at a restart's step belongs to the launch before it; one only named as another run's anchor is listed
    old = P["2000"]
    assert old["launch"] == 0 and old["anchor_of"] == ["fgself-6"] and old["rated"] == [] and old["saved"] == created + 3000.0
    hour = P["h3"]
    assert (hour["kind"], hour["label"], hour["launch"]) == ("hour", "hour 3", 1)
    assert (hour["from_time"], hour["to_time"]) == (created + 3 * 3600, created + 4 * 3600)
    assert (hour["from_steps"], hour["to_steps"]) == (3600, 4800)  # (from the trainer's log)
    assert [c["steps"] for c in hour["checkpoints"]] == [4000] and hour["checkpoints"][0]["rated"]
    h = hour["rated"][0]
    assert h["name"] == "fgself-5 h3" and {o["name"]: o["record"] for o in h["opponents"]} == \
        {"insane AI": [1, 0, 1], "fgself-4/900": [0, 1, 0]}
    assert h["opponents"][0]["races"]["human"]["orc"] == [0, 0, 1] and hour["replays"] == {"q": "fgself-5", "count": None}
    # the newest on the timeline first, by the steps each ends at: an average above the checkpoint it ends at
    assert [p["id"] for p in L["policies"]] == ["swa-7-9", "h3", "swa-2-4", "4000", "2000"]
    # the burst's own listing: what was played there, its own policy first
    B = policies.listing(tmp_path, b, policies.own(b, json.loads((b / "run.json").read_text())), r.view, [])
    assert B["policies"] == [] and B["burst"]["policy"] == "fgself-5/4000" and B["burst"]["games"] == 4
    assert [row["name"] for row in B["burst"]["rows"]] == ["fgself-5/4000", "fgself-5 avg swa-2-4"]
    assert B["burst"]["rows"][0]["opponents"][0] == {"name": "insane AI", "kind": "anchor", "record": [1, 0, 0],
                                                     "rating": 1500.0}
    # a checkpoint whose file is gone (archived): when the trainer's log reached its steps
    (d / "checkpoints" / f"{4000:016d}.pt").unlink()
    gone = {p["id"]: p for p in policies.listing(tmp_path, d, info, r.view, train)["policies"]}["4000"]
    assert gone["file"] is None and gone["saved_by"] == "trainer log" and gone["saved"] == created + 60.0 * 200
    assert gone["replays"] == {"q": "fgself-5/4000", "count": None}
