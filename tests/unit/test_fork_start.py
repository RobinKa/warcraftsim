"""scripts/selfplay_fork.py --set: "start" in a value is the fork's start weights only in a flag that takes paths
(bug finder 11, 2026-11-23). It was replaced in any flag's value: --set exploiter-adam=start went out as
--exploiter-adam <the checkpoint>, and the value had to be renamed "saved"."""

import importlib.util
import sys
from pathlib import Path

from warcraftsim import paths

SCRIPTS = paths.REPO_ROOT / "scripts"
START = "runs/fgself-1/checkpoints/0000000000000100.pt"


def load_script(name):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_selfplays_path_flags():
    fork = load_script("selfplay_fork")
    flags = fork.path_flags((paths.REPO_ROOT / "warcraftsim" / "fullgame" / "selfplay.py").read_text())
    assert {"eval-checkpoints", "league-add", "init", "ref2", "start-weights"} <= flags
    assert not flags & {"exploiter-adam", "label-norm", "econ-masks", "note", "title"}


def test_start_is_the_weights_only_in_a_path_flag(capsys):
    fork = load_script("selfplay_fork")
    flags = {"eval-checkpoints", "league-add"}
    assert fork.set_words("eval-checkpoints", "start runs/fgself-130/x.pt", START, flags) == [START, "runs/fgself-130/x.pt"]
    assert fork.set_words("league-add", "start", START, flags) == [START]
    assert capsys.readouterr().out == ""
    # any other flag: the word itself, and a note (a value named "start" reaches the parser as it was written)
    assert fork.set_words("exploiter-adam", "start", START, flags) == ["start"]
    assert '"start" kept as the word' in capsys.readouterr().out
    assert fork.set_words("econ-masks", "511", START, flags) == ["511"]
    assert fork.set_words("first-hero-weights", "Ofar:4,Hamg:4", START, flags) == ["Ofar:4,Hamg:4"]
    assert capsys.readouterr().out == ""
    assert fork.set_words("eval-checkpoints", "restart starts", START, flags) == ["restart", "starts"]  # (whole words only)
