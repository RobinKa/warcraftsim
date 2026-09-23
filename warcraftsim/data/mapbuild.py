"""Build harness maps: a stock melee map with the w3sim JASS harness injected.

The World Editor generates melee map scripts from one template, so the edits are
textual and checked:

* harness globals are merged into the map's ``globals`` block (JASS allows
  global declarations only before the first function);
* harness functions go right after it, before every map function;
* ``MeleeStartingAI()`` becomes ``W3S_StartingAI()`` so agent-controlled
  computer slots get no built-in AI;
* ``MeleeInitVictoryDefeat()`` becomes a no-op: the harness decides results
  without removing players, which would end the local session;
* ``W3S_Init()`` is called at the end of ``main``.

The result is validated with pjass against the game's common.j/Blizzard.j.
"""

from __future__ import annotations

import os
import re
import shutil
import threading
import subprocess
import tempfile
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from .. import paths
from .mpq import GameArchives, MpqArchive


class MapBuildError(RuntimeError):
    pass


VICTORY_MODES = {"melee": 0, "elimination": 1, "none": 2}


@dataclass(frozen=True)
class SpawnSpec:
    player: int
    unit: str  # four-character unit code, e.g. "hfoo"
    x: float
    y: float
    facing: float = 270.0


@dataclass
class HarnessConfig:
    step_seconds: float = 0.25
    agent_players: tuple[int, ...] = (0,)
    max_game_seconds: float = 0.0  # 0 = no limit; otherwise the game ends in a tie
    scripted_players: tuple[int, ...] = ()  # scenario opponents: idle units attack-move to the nearest enemy
    scenario: bool = False  # remove every pre-placed unit and spawn `spawn`; restart = respawn in-game
    victory: str = "melee"  # melee (structures) | elimination (units) | none
    send_destructables: bool = True
    spawn: tuple[SpawnSpec, ...] = ()
    resources: tuple[tuple[int, int, int], ...] = ()  # (player, gold, lumber)
    clear_area: tuple[float, float, float] | None = None  # (x, y, radius): remove trees there
    order_names: tuple[str, ...] | None = None  # order strings resolved in the first observation

    def resolved_order_names(self) -> tuple[str, ...]:
        from ..protocol import ORDER_NAMES, all_order_names

        if self.order_names is not None:
            return self.order_names
        try:
            return all_order_names()
        except (FileNotFoundError, OSError):
            return ORDER_NAMES

    @staticmethod
    def _mask(players) -> int:
        mask = 0
        for p in players:
            if not 0 <= p < 12:
                raise ValueError(f"player {p} out of range")
            mask |= 1 << p
        return mask

    def agent_mask(self) -> int:
        return self._mask(self.agent_players)

    def spawn_code(self) -> str:
        lines = []
        for sp in self.spawn:
            if len(sp.unit) != 4:
                raise ValueError(f"unit code must have four characters: {sp.unit!r}")
            lines.append(f"    call W3S_SpawnUnit({sp.player}, '{sp.unit}', {sp.x:.1f}, {sp.y:.1f}, {sp.facing:.1f})")
        for player, gold, lumber in self.resources:
            lines.append(f"    call SetPlayerState(Player({player}), PLAYER_STATE_RESOURCE_GOLD, {int(gold)})")
            lines.append(f"    call SetPlayerState(Player({player}), PLAYER_STATE_RESOURCE_LUMBER, {int(lumber)})")
        return "\n".join(lines) if lines else "    // no units"


def harness_source() -> str:
    return resources.files("warcraftsim.harness").joinpath("w3sim.j").read_text()


def _split_harness(src: str, cfg: HarnessConfig) -> tuple[str, str]:
    m = re.search(r"^globals\n(.*?)^endglobals\n", src, re.M | re.S)
    if not m:
        raise MapBuildError("harness has no globals block")
    glob = m.group(1)
    cx, cy, cr = cfg.clear_area or (0.0, 0.0, 0.0)
    jbool = lambda b: "true" if b else "false"  # noqa: E731
    replacements = {
        "W3S_CFG_STEP_S": f"constant real W3S_CFG_STEP_S = {cfg.step_seconds:.4f}",
        "W3S_CFG_MAX_GAME_S": f"constant real W3S_CFG_MAX_GAME_S = {cfg.max_game_seconds:.1f}",
        "W3S_CFG_AGENT_MASK": f"constant integer W3S_CFG_AGENT_MASK = {cfg.agent_mask()}",
        "W3S_CFG_SCENARIO": f"constant boolean W3S_CFG_SCENARIO = {jbool(cfg.scenario)}",
        "W3S_CFG_VICTORY": f"constant integer W3S_CFG_VICTORY = {VICTORY_MODES[cfg.victory]}",
        "W3S_CFG_SCRIPTED_MASK": f"constant integer W3S_CFG_SCRIPTED_MASK = {cfg._mask(cfg.scripted_players)}",
        "W3S_CFG_SEND_DESTRUCTABLES":
            f"constant boolean W3S_CFG_SEND_DESTRUCTABLES = {jbool(cfg.send_destructables)}",
        "W3S_CFG_CLEAR_X": f"constant real W3S_CFG_CLEAR_X = {cx:.1f}",
        "W3S_CFG_CLEAR_Y": f"constant real W3S_CFG_CLEAR_Y = {cy:.1f}",
        "W3S_CFG_CLEAR_R": f"constant real W3S_CFG_CLEAR_R = {cr:.1f}",
    }
    for name, decl in replacements.items():
        glob, n = re.subn(rf"^\s*constant \w+ {name} = .*$", "    " + decl, glob, flags=re.M)
        if n != 1:
            raise MapBuildError(f"harness config {name} not found")
    body = src[m.end():]
    orders = "\n".join(f'    call W3S_Order("{name}")' for name in cfg.resolved_order_names())
    body, n = re.subn(r"^\s*// @ORDERS@\s*$", lambda _m: orders, body, flags=re.M)
    if n != 1:
        raise MapBuildError("harness order table marker not found")
    spawn = cfg.spawn_code()
    body, n = re.subn(r"^\s*// @SCENARIO_SPAWN@\s*$", lambda _m: spawn, body, flags=re.M)
    if n != 1:
        raise MapBuildError("harness scenario marker not found")
    return glob, body


def inject_harness(script: str, cfg: HarnessConfig, harness: str | None = None) -> str:
    """Return the map script with the harness injected (input may use CRLF)."""
    j = script.replace("\r\n", "\n")
    glob, body = _split_harness(harness or harness_source(), cfg)

    def sub_once(pattern: str, repl: str, text: str, what: str) -> str:
        out, n = re.subn(pattern, repl, text, count=1, flags=re.M)
        if n != 1:
            raise MapBuildError(f"map script: {what} not found")
        return out

    # the harness only depends on common.j/Blizzard.j, so it goes before every map function
    j = sub_once(r"^endglobals\s*$", lambda_repl(glob + "endglobals\n" + body), j, "globals block")
    if not re.search(r"^function main takes nothing returns nothing\s*$", j, re.M):
        raise MapBuildError("map script: function main not found")
    j = sub_once(r"call MeleeStartingAI\(\s*\)", "call W3S_StartingAI()", j, "MeleeStartingAI call")
    j = sub_once(r"call MeleeInitVictoryDefeat\(\s*\)", "call W3S_InitVictoryDefeat()", j,
                 "MeleeInitVictoryDefeat call")
    # end of main: the line after RunInitializationTriggers
    j = sub_once(r"^(\s*call RunInitializationTriggers\(\s*\)\s*)$", r"\1\n    call W3S_Init()", j,
                 "RunInitializationTriggers call in main")
    return j


def lambda_repl(text: str):
    # re.sub treats backslashes in string replacements specially; the harness contains "\\".
    return lambda _m: text


def game_scripts() -> tuple[Path, Path]:
    """common.j and Blizzard.j from the game archives (cached)."""
    out = paths.CACHE_DIR / "game" / "Scripts"
    common, blizzard = out / "common.j", out / "Blizzard.j"
    if not (common.exists() and blizzard.exists()):
        out.mkdir(parents=True, exist_ok=True)
        with GameArchives() as g:
            common.write_bytes(g.read("Scripts\\common.j"))
            blizzard.write_bytes(g.read("Scripts\\Blizzard.j"))
    return common, blizzard


def pjass_check(script: str) -> None:
    if not paths.PJASS_PATH.exists():
        raise MapBuildError(f"pjass not built at {paths.PJASS_PATH}; run scripts/build_native.sh")
    common, blizzard = game_scripts()
    with tempfile.NamedTemporaryFile("w", suffix=".j", delete=False, encoding="latin-1") as f:
        f.write(script)
        tmp = f.name
    try:
        r = subprocess.run([str(paths.PJASS_PATH), str(common), str(blizzard), tmp],
                           capture_output=True, text=True)
    finally:
        Path(tmp).unlink(missing_ok=True)
    if r.returncode != 0:
        errors = "\n".join(line for line in r.stdout.splitlines() if "Parse successful" not in line)
        raise MapBuildError(f"pjass rejected the generated script:\n{errors}")


def stock_map_path(name: str) -> Path:
    """Resolve a map name to a file: a path, a stock map ("(2)EchoIsles" or "EchoIsles"), or
    "flat" / "flat:<stock map>" for a flat, empty version of a stock map (default Echo Isles)."""
    if name == "flat" or name.startswith("flat:"):
        from .flatmap import flat_map_path

        return flat_map_path(name[5:] or "(2)EchoIsles")
    candidates = []
    for sub in ("Maps/FrozenThrone", "Maps"):
        d = paths.GAME_DIR / sub
        if not d.exists():
            continue
        for f in d.iterdir():
            stem = f.stem.lower()
            if stem == name.lower() or re.sub(r"^\(\d+\)", "", stem) == name.lower():
                candidates.append(f)
    if Path(name).exists():
        return Path(name)
    if not candidates:
        raise FileNotFoundError(f"map {name!r} not found under {paths.GAME_DIR}/Maps")
    candidates.sort(key=lambda f: (f.suffix != ".w3x", "FrozenThrone" not in str(f)))
    return candidates[0]


def build_map(source: str | Path, dest: str | Path, cfg: HarnessConfig, check: bool = True) -> Path:
    """Copy `source` to `dest` with the harness injected into war3map.j."""
    src = stock_map_path(str(source)) if not Path(source).exists() else Path(source)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    shutil.copyfile(src, tmp)
    try:
        with MpqArchive(tmp, writable=True) as m:
            script = m.read("war3map.j").decode("latin-1")
            out = inject_harness(script, cfg)
            if check:
                pjass_check(out)
            m.write("war3map.j", out.replace("\n", "\r\n").encode("latin-1"))
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest
