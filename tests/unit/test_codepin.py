"""Pinned launches (warcraftsim/codepin.py; scripts/selfplay_restart.py, selfplay_fork.py): a checkout per commit,
linked to the main tree's runs, built, reused, pruned when no process uses it; the launch scripts' dry run."""

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from warcraftsim import codepin, paths

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
MAKEFILE = "OUT ?= ../build/shim\nall:\n\tmkdir -p $(OUT) && test -f $(MINHOOK)/include/MinHook.h && touch $(OUT)/w3shim.dll $(OUT)/w3launch.exe\n"
NATIVE = "from pathlib import Path\ndef lib():\n    (Path(__file__).parent.parent / 'built.txt').write_text(__file__)\n"


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def commit(root, msg):
    git(root, "add", "-A")
    git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", msg)
    return git(root, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A main tree: a package, a shim whose build only touches its outputs, runs/ (ignored) with a run's file."""
    root = tmp_path / "main"
    (root / "warcraftsim").mkdir(parents=True)
    (root / "warcraftsim" / "__init__.py").write_text("")
    (root / "warcraftsim" / "native.py").write_text(NATIVE)
    (root / "warcraftsim" / "fullgame").mkdir()  # (else the venv's editable install finds the main tree's)
    (root / "warcraftsim" / "fullgame" / "__init__.py").write_text("")
    (root / "shim").mkdir()
    (root / "shim" / "Makefile").write_text(MAKEFILE)
    (root / ".gitignore").write_text("runs/\n/runs\nbuild/\n__pycache__/\n")
    git(root, "init", "-q")
    commit(root, "first")
    (root / "runs" / "fgtest").mkdir(parents=True)
    (root / "runs" / "fgtest" / "keep.txt").write_text("the run's data")
    minhook = tmp_path / "minhook"
    (minhook / "include").mkdir(parents=True)
    (minhook / "include" / "MinHook.h").write_text("")
    monkeypatch.setenv("WARCRAFTSIM_MINHOOK", str(minhook))
    monkeypatch.setattr(paths, "CODE_DIR", tmp_path / "code")
    return root


def test_prepare_links_builds_and_reuses(repo):
    c = git(repo, "rev-parse", "HEAD")
    d = codepin.prepare(repo, log=lambda m: None)
    assert d == paths.CODE_DIR / c[:12] and git(d, "rev-parse", "HEAD") == c  # (git works there: launch.git)
    assert (d / "runs").is_symlink() and (d / "runs" / "fgtest" / "keep.txt").read_text() == "the run's data"
    assert (d / "runs").resolve() == (repo / "runs").resolve()
    assert (d / "build" / "shim" / "w3shim.dll").exists() and not (repo / "build").exists()  # (built there)
    assert Path((d / "built.txt").read_text()) == d / "warcraftsim" / "native.py"  # (the checkout's code built)
    assert codepin.stamp(d)["commit"] == c
    (d / "marker").write_text("")
    assert codepin.prepare(repo, builds=False, log=lambda m: None) == d and (d / "marker").exists()  # reused


def test_a_new_commit_gets_its_own_checkout_and_the_old_keeps_its_code(repo):
    old = codepin.prepare(repo, builds=False, log=lambda m: None)
    (repo / "warcraftsim" / "new.py").write_text("X = 1\n")
    commit(repo, "second")
    new = codepin.prepare(repo, builds=False, log=lambda m: None)
    assert new != old and (new / "warcraftsim" / "new.py").exists() and not (old / "warcraftsim" / "new.py").exists()
    first = codepin.prepare(repo, "HEAD~1", builds=False, log=lambda m: None)
    assert first == old


def test_unfinished_checkout_is_made_again(repo):
    d = codepin.prepare(repo, builds=False, log=lambda m: None)
    (d / codepin.STAMP).unlink()
    (d / "leftover").write_text("")
    assert codepin.prepare(repo, builds=False, log=lambda m: None) == d
    assert not (d / "leftover").exists() and codepin.stamp(d)


def test_dirty_names_uncommitted_code_only(repo):
    assert codepin.dirty(repo) == []
    (repo / "notes.md").write_text("docs")  # (not code)
    assert codepin.dirty(repo) == []
    (repo / "warcraftsim" / "native.py").write_text(NATIVE + "# edited\n")
    (repo / "warcraftsim" / "fresh.py").write_text("")
    assert sorted(line[3:] for line in codepin.dirty(repo)) == ["warcraftsim/fresh.py", "warcraftsim/native.py"]


def test_launch_env_puts_the_checkout_first(tmp_path):
    env = codepin.launch_env(tmp_path, {"PYTHONPATH": "/x", "A": "1"})
    assert env["PYTHONPATH"] == f"{tmp_path}{os.pathsep}/x" and env["A"] == "1"
    assert codepin.launch_env(tmp_path, {})["PYTHONPATH"] == str(tmp_path)


def test_prune_spares_used_newest_and_the_runs(repo):
    ds = []
    for i in range(4):
        (repo / "warcraftsim" / f"m{i}.py").write_text("")
        commit(repo, f"c{i}")
        ds.append(codepin.prepare(repo, builds=False, log=lambda m: None))
        time.sleep(0.01)
    busy = subprocess.Popen(["sleep", "30"], cwd=ds[0])  # (a live run's process in the oldest)
    try:
        assert codepin.users([ds[0], ds[1]])[ds[0]] == [busy.pid]
        gone = codepin.prune(repo, keep=1, protect=[ds[2]], log=lambda m: None)
    finally:
        busy.kill()
        busy.wait()
    assert gone == [ds[1]] and not ds[1].exists()
    assert all(d.exists() for d in (ds[0], ds[2], ds[3]))
    assert (repo / "runs" / "fgtest" / "keep.txt").read_text() == "the run's data"  # (the link unlinked, not followed)
    assert str(ds[1]) not in git(repo, "worktree", "list")
    assert codepin.prune(repo, keep=1, log=lambda m: None) == [ds[2], ds[0]]
    assert (repo / "runs" / "fgtest" / "keep.txt").exists()


def load_script(name):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def restart(repo, monkeypatch):
    mod = load_script("selfplay_restart")
    monkeypatch.setattr(mod, "ROOT", repo)
    run = repo / "runs" / "fgtest"
    (run / "checkpoints").mkdir()
    (run / "checkpoints" / "0000000000000100.pt").write_text("")
    (run / "run.json").write_text(json.dumps({"launch": {"argv": ["--name", "fgtest", "--init", "runs/a.pt",
                                                                  "--note=old"], "git": {"commit": "HEAD"}}}))

    def no(*a, **k):
        raise AssertionError("a dry run stops or launches nothing")
    monkeypatch.setattr(mod, "stop", no)
    monkeypatch.setattr(mod, "subprocess", SimpleNamespace(run=subprocess.run, Popen=no, STDOUT=subprocess.STDOUT))
    return mod


def test_restart_dry_run_prepares_and_prints(restart, repo, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["selfplay_restart.py", "fgtest", "--note", "why", "--set", "x=2", "--dry-run"])
    restart.main()
    out = capsys.readouterr().out
    d = paths.CODE_DIR / git(repo, "rev-parse", "HEAD")[:12]
    assert f"cd {d} && PYTHONPATH={d}" in out and "-m warcraftsim.fullgame.selfplay --name fgtest --init runs/a.pt" in out
    assert "--x 2 --resume --note=why" in out and f"imports from {d}/warcraftsim/__init__.py" in out
    assert (d / "build" / "shim" / "w3shim.dll").exists() and not (repo / "runs" / "fgtest.log").exists()


def test_restart_refuses_a_dirty_tree(restart, repo, monkeypatch):
    (repo / "warcraftsim" / "native.py").write_text(NATIVE + "# not committed\n")
    monkeypatch.setattr(sys, "argv", ["selfplay_restart.py", "fgtest", "--note", "why"])
    with pytest.raises(SystemExit, match="uncommitted"):
        restart.main()
    assert not paths.CODE_DIR.exists()


def test_fork_dry_run(restart, repo, monkeypatch, capsys):
    fork = load_script("selfplay_fork")
    monkeypatch.setattr(fork, "ROOT", repo)
    monkeypatch.setattr(fork, "stop", restart.stop)
    monkeypatch.setattr(fork, "start", restart.start)
    monkeypatch.setattr(sys, "argv", ["selfplay_fork.py", "fgtest", "fgnew", "--note", "n", "--dry-run"])
    fork.main()
    out = capsys.readouterr().out
    assert "--name fgnew --init runs/a.pt --start-weights runs/fgtest/checkpoints/0000000000000100.pt" in out
    assert "--league-from fgtest" in out and not (repo / "runs" / "fgnew").exists()


def test_remote_eval_sends_the_checkouts_shim(tmp_path):
    re_ = load_script("remote_eval")
    assert re_.shim_root({"launch": {"code": str(tmp_path)}}) == re_.ROOT  # (no shim built there)
    for f in re_.SHIM:
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text("")
    assert re_.shim_root({"launch": {"code": str(tmp_path)}}) == tmp_path
    assert re_.shim_root({}) == re_.ROOT
