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
    assert 'creep_until=float(cfg.get("creep_lesson_until", 0.0) or 0.0)' in body


def test_the_early_expansion_lessons_reach_the_games():
    """--expand-lesson-from and --expand-lesson-share: in the actors' cfg, and from there in play_one's teachers."""
    assert {"expand_lesson_from", "expand_lesson_share"} <= _cfg_written()
    assert '"expand_lesson_from": args.expand_lesson_from' in SRC and '"expand_lesson_share": args.expand_lesson_share' in SRC
    body = SRC[SRC.index("def play_one("):SRC.index("def film_game(")]
    assert 'expand_from=float(cfg.get("expand_lesson_from", 0.0) or 0.0)' in body
    assert 'expand_share=float(cfg.get("expand_lesson_share", 1.0))' in body


def test_the_harvest_floor_reaches_the_loss():
    """--teach-floor-pointer and the "harvest" group set selfplay.TEACH_FLOOR_POINTER in main (the learner's own: no cfg)."""
    assert "TEACH_FLOOR_POINTER = (hm.to(device), math.log(pointer))" in SRC
    assert "pointer = args.teach_floor_pointer or args.teach_floor_place or args.teach_floor" in SRC
