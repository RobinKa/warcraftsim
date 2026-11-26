"""selfplay's flags reach what reads them (bug finder 11, 2026-11-23). --creep-lesson-cap and --creep-lesson-until were
parsed and never put into the actors' cfg: play_one read cfg.get("creep_lesson_cap", 0), always 0, and fgself-246, 247
and 248 ("the creep lesson capped") ran uncapped: 12-24 creep lessons a game, up to 211 in one, where a cap of 4 creeps
and 480 s allows at most 48. A flag nothing reads, or a cfg key nothing writes, fails here."""

import re

from warcraftsim import paths

SRC = (paths.REPO_ROOT / "warcraftsim" / "fullgame" / "selfplay.py").read_text()
# cfg keys other code writes: the evaluation worker's own (evalworker.py), a launch's copy (real_cfg), a remote actor's
ELSEWHERE = {"eval_map_pool", "guide_capable", "gauntlet_file", "real_launch", "remote"}


def _cfg_written() -> set[str]:
    lines = SRC.split("\n")
    start = next(i for i, l in enumerate(lines) if l.strip().startswith('cfg = {"run_dir"'))
    depth, end = 0, start
    for i in range(start, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if depth <= 0:
            end = i
            break
    block = "\n".join(lines[start:end + 1])
    return set(re.findall(r'"([a-z_0-9]+)":', block)) | set(re.findall(r'cfg\["([a-z_0-9]+)"\]\s*=', SRC))


def test_every_flag_is_read():
    flags = {f.replace("-", "_") for f in re.findall(r'add_argument\("--([a-z0-9-]+)"', SRC)}
    read = set(re.findall(r"args\.([a-z_0-9]+)", SRC)) | set(re.findall(r'getattr\(args, "([a-z_0-9]+)"', SRC))
    assert flags - read == set()


def test_every_cfg_key_read_is_written():
    read = set(re.findall(r'cfg(?:\.get\(|\[)"([a-z_0-9]+)"', SRC))
    assert read - _cfg_written() - ELSEWHERE == set()


def test_the_creep_lessons_cap_reaches_the_games():
    written = _cfg_written()
    assert {"creep_lesson", "creep_lesson_cap", "creep_lesson_until"} <= written
    assert '"creep_lesson_cap": args.creep_lesson_cap' in SRC and '"creep_lesson_until": args.creep_lesson_until' in SRC


def test_play_one_hands_the_cap_to_the_teacher():
    """play_one's teachers take the cap from the cfg's keys (the teacher's own side of it:
    tests/unit/test_long_game_lessons.py)."""
    body = SRC[SRC.index("def play_one("):SRC.index("def film_game(")]
    assert 'creep_cap=int(cfg.get("creep_lesson_cap", 0) or 0)' in body
    assert "creep_until=creep_lesson_until(cfg, late_start)" in body  # (a late start: what is left of it)


def test_the_early_expansion_lessons_reach_the_games():
    """--expand-lesson-from and --expand-lesson-share: in the actors' cfg, and from there in play_one's teachers."""
    assert {"expand_lesson_from", "expand_lesson_share", "expand_lesson_reserve"} <= _cfg_written()
    assert '"expand_lesson_from": args.expand_lesson_from' in SRC and '"expand_lesson_share": args.expand_lesson_share' in SRC
    assert '"expand_lesson_reserve": args.expand_lesson_reserve' in SRC
    body = SRC[SRC.index("def play_one("):SRC.index("def film_game(")]
    assert "expand_from=expand_lesson_from(cfg, late_start)" in body
    assert 'expand_share=float(cfg.get("expand_lesson_share", 1.0))' in body
    assert 'expand_reserve=int(cfg.get("expand_lesson_reserve", 0) or 0)' in body
    assert "bots[s].view.reserve = tc.hall_reserve(bots[s].enc)" in body  # (--expand-lesson-reserve: before the step's view)
    assert "tc.saving(rows, obs.players.get(s), obs.game_time, t)" in body  # (the sides without lessons: the same rule)
    # (and the evaluation worker's games: its cfg is made from run.json, key by key)
    worker = (paths.REPO_ROOT / "warcraftsim" / "fullgame" / "evalworker.py").read_text()
    for key in ("expand_lesson_from", "expand_lesson_share", "expand_lesson_reserve", "lesson_time_scale"):
        assert f'"{key}": num("{key}"' in worker
    from warcraftsim.fullgame.selfplay import expand_lesson_from
    cfg = {"map": "(2)EchoIsles", "expand_lesson_from": 360.0}
    assert expand_lesson_from({"map": "(2)EchoIsles"}, None) == 0.0 and expand_lesson_from(cfg, None) == 360.0
    assert expand_lesson_from(cfg, {"sides": []}) == 360.0  # (a two-base start's state: no minute)
    assert expand_lesson_from(cfg, {"minute": 2.0}) == 240.0  # (a late start: its clock starts at 0 at minute 2)
    assert expand_lesson_from(cfg, {"minute": 12.0}) == 1.0  # (past S: from its first second, not off)
    assert expand_lesson_from({"map": "(2)EchoIsles", "expand_lesson_from": 0.0}, {"minute": 12.0}) == 0.0
    assert expand_lesson_from({**cfg, "lesson_time_scale": 0.5}, {"minute": 2.0}) == 120.0  # (S x 0.5 = 180 s, 120 s in: 60 s x 2)


def test_the_harvest_floor_reaches_the_loss():
    """--teach-floor-pointer and the "harvest" group set selfplay.TEACH_FLOOR_POINTER in main (the learner's own: no cfg)."""
    assert "TEACH_FLOOR_POINTER = (hm.to(device), math.log(pointer))" in SRC
    assert "pointer = args.teach_floor_pointer or args.teach_floor_place or args.teach_floor" in SRC
    assert "TEACH_FLOOR_LESSONS = (lm.to(device), math.log(args.teach_floor))" in SRC  # (the "workers" group)


def test_late_starts_take_the_lessons_seconds_from_the_captured_minute():
    """--hunt-press and --creep-lesson-until are game seconds of the game a late start's state is from, as
    --expand-lesson-from/-until (bug finder 12): a late start's clock starts at 0 at the captured minute, and taken as
    they were, fgself-255's late starts (a quarter of its games) pressed only from S seconds of their own clock (game
    minute 20-28 at 720), and fgself-257's had creep lessons for 480 s of theirs, all past its minute 8."""
    from warcraftsim.fullgame.selfplay import creep_lesson_until, hunt_press_from
    body = SRC[SRC.index("def play_one("):SRC.index("def film_game(")]
    assert "hunt_press=hunt_press_from(cfg, late_start)" in body
    assert "creep_until=creep_lesson_until(cfg, late_start)" in body
    cfg = {"map": "(2)EchoIsles", "hunt_press": 720.0, "creep_lesson_until": 480.0}
    assert hunt_press_from(cfg, None) == 720.0 and creep_lesson_until(cfg, None) == 480.0
    assert hunt_press_from(cfg, {"sides": []}) == 720.0  # (a two-base start's state: no minute, S as it is)
    assert hunt_press_from(cfg, {"minute": 8.0}) == 240.0  # (12 minutes, 8 of them before the state)
    assert hunt_press_from(cfg, {"minute": 16.0}) == 1.0  # (past S: it presses from its first second)
    assert creep_lesson_until(cfg, {"minute": 2.0}) == 360.0
    assert creep_lesson_until(cfg, {"minute": 8.0}) == 1.0  # (past S: no creep lessons, not 480 s more)
    assert hunt_press_from({"map": "(2)EchoIsles"}, {"minute": 8.0}) == 0.0  # (0: none, and no limit)
    assert creep_lesson_until({"map": "(2)EchoIsles", "creep_lesson_until": 0.0}, {"minute": 8.0}) == 0.0
    # (the creep limit in the teacher's paced seconds, as expand_lesson_until's; the press is the teacher's own second)
    assert creep_lesson_until({**cfg, "lesson_time_scale": 0.5}, {"minute": 2.0}) == 240.0
    assert hunt_press_from({**cfg, "lesson_time_scale": 0.5}, {"minute": 2.0}) == 600.0
