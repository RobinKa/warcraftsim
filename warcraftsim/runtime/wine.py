"""Wine prefixes for game instances.

One template prefix is created once (``wineboot``, registry settings, a link to
the game install). Every instance gets its own prefix so that instances never
share a wineserver, a named mutex or a Documents folder. Instance prefixes are
hard-link clones of the template (the 1.4 GB of Wine system files cost no extra
disk); files that Wine or the game write are copied instead.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .. import paths

TEMPLATE_NAME = "prefix-template"
GAME_LINK = "Warcraft III"  # C:\Warcraft III -> the read-only game install
WORK_DIR = "w3sim"  # C:\w3sim: per-instance map and .wgc

# HKCU\Software\Blizzard Entertainment\Warcraft III\Video: cheapest rendering.
VIDEO_SETTINGS = {
    "reswidth": 800, "resheight": 600, "colordepth": 32, "refreshrate": 60,
    "maxfps": 10000, "lockfb": 0, "modeldetail": 0, "animquality": 0, "texquality": 0,
    "particles": 0, "lights": 0, "unitshadows": 0, "occlusion": 0, "spellfilter": 0,
}
GAME_SETTINGS = {"Allow Local Files": 1}
SOUND_SETTINGS = {"sfx": 0, "music": 0, "ambient": 0, "movement": 0, "positional": 0}


def wine_bin_dir() -> str | None:
    """Wine to use: $WARCRAFTSIM_WINE, else WineHQ stable if installed, else `wine` on PATH.

    WineHQ stable 11.0 measured ~35% more parallel throughput than staging 11.18 here (less
    wineserver CPU per game). GE-Proton 10's Wine (10.0) with fsync: 28% less CPU a step than stable
    11.0 (wineserver 2.6 -> 0.7 ms), the same without fsync (see use_wine).
    """
    configured = os.environ.get("WARCRAFTSIM_WINE")
    if configured:
        return configured
    stable = Path("/opt/wine-stable/bin")
    return str(stable) if (stable / "wine").exists() else None


def proton_files(wine_bin: str | None) -> Path | None:
    """A Proton build's `files` folder (GE-Proton: Wine with fsync), if `wine_bin` is its `files/bin`."""
    if not wine_bin:
        return None
    files = Path(wine_bin).resolve().parent
    return files if (files / "lib" / "vkd3d").is_dir() and (files / "share" / "default_pfx").is_dir() else None


def use_wine(wine_bin: str, fsync: bool) -> None:
    """This process and its children use another Wine: its own prefixes (in a runtime folder named after
    it: a prefix belongs to the Wine that made it) and, with `fsync`, its futex-based synchronization
    (WINEFSYNC: Wine's events, mutexes and waits without a wineserver round trip; Proton builds only)."""
    wine_bin = str(Path(wine_bin).expanduser().resolve())
    files = proton_files(wine_bin)
    tag = files.parent.name if files else Path(wine_bin).parent.name
    os.environ["WARCRAFTSIM_WINE"] = wine_bin
    runtime = paths.RUNTIME_DIR if paths.RUNTIME_DIR.name.endswith(f"-{tag}") else \
        paths.RUNTIME_DIR.with_name(f"{paths.RUNTIME_DIR.name}-{tag}")
    os.environ["WARCRAFTSIM_RUNTIME"] = str(runtime)
    paths.RUNTIME_DIR = runtime
    os.environ["WINEFSYNC"] = "1" if fsync else "0"


def wine_env(prefix: Path, **extra: str) -> dict[str, str]:
    env = dict(os.environ)
    wine_bin = wine_bin_dir()
    if wine_bin:
        env["PATH"] = f"{wine_bin}{os.pathsep}{env.get('PATH', '')}"
    files = proton_files(wine_bin)
    if files:  # what Proton's own launcher sets (its Wine's libraries, vkd3d for wined3d)
        lib = files / "lib"
        env["WINEDLLPATH"] = f"{lib / 'vkd3d'}:{lib / 'wine'}"
        env["LD_LIBRARY_PATH"] = ":".join([str(lib / "x86_64-linux-gnu"), str(lib / "i386-linux-gnu")]
                                          + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
    env.update(
        WINEPREFIX=str(prefix),
        WINEDEBUG=os.environ.get("W3SIM_WINEDEBUG", "-all"),  # (e.g. +server: every wineserver request, to profile)
        # no Mono/Gecko prompts, no menu entries, no audio device (sound is off anyway)
        WINEDLLOVERRIDES="mscoree,mshtml=;winemenubuilder.exe=d;winealsa.drv,winepulse.drv,wineoss.drv=d",
        WINEARCH="win64",
    )
    from .display import SOFTWARE_GL
    env.update(SOFTWARE_GL)  # never the GPU (see display.SOFTWARE_GL)
    env.pop("W3SIM_PORT", None)
    env.update(extra)
    return env


def documents_dir(prefix: Path) -> Path:
    users = prefix / "drive_c" / "users"
    # (Proton's Wine names the Windows user "steamuser" whoever runs it)
    user = "steamuser" if (users / "steamuser").is_dir() else os.environ.get("USER", "user")
    return users / user / "Documents" / "Warcraft III"


def _reg_file(settings: dict[str, dict[str, int]]) -> str:
    lines = ["Windows Registry Editor Version 5.00", ""]
    for key, values in settings.items():
        lines.append(f"[HKEY_CURRENT_USER\\Software\\Blizzard Entertainment\\Warcraft III{key}]")
        for name, value in values.items():
            lines.append(f'"{name}"=dword:{value:08x}')
        lines.append("")
    return "\r\n".join(lines)


def template_prefix() -> Path:
    return paths.RUNTIME_DIR / TEMPLATE_NAME


def ensure_template(display: str | None = None, force_registry: bool = False) -> Path:
    """Create (once) the template prefix with the game linked in and registry settings applied."""
    prefix = template_prefix()
    marker = prefix / ".warcraftsim-template"
    if marker.exists() and not force_registry:
        return prefix
    if not (paths.GAME_DIR / "Warcraft III.exe").exists():
        raise FileNotFoundError(f"Warcraft III 1.29 not found at {paths.GAME_DIR} (set WC3_GAME_DIR)")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    env = wine_env(prefix)
    if display:
        env["DISPLAY"] = display
    if not (prefix / "system.reg").exists():
        # (`wine wineboot`: a Proton build has no wineboot script, and another Wine's would make the prefix)
        subprocess.run(["wine", "wineboot", "-i"], env=env, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        subprocess.run(["wineserver", "-w"], env=env, check=False)
    files = proton_files(wine_bin_dir())
    if files:  # Proton copies these into its prefixes: wined3d (Direct3D 9) needs them
        for arch in ("system32", "syswow64"):
            for dll in ("libvkd3d-1.dll", "libvkd3d-shader-1.dll"):
                shutil.copy2(files / "share" / "default_pfx" / "drive_c" / "windows" / arch / dll,
                             prefix / "drive_c" / "windows" / arch / dll)
    game_link = prefix / "drive_c" / GAME_LINK
    if game_link.is_symlink() or game_link.exists():
        game_link.unlink()
    game_link.symlink_to(paths.GAME_DIR, target_is_directory=True)
    reg = prefix / "drive_c" / "warcraftsim.reg"
    reg.write_text(_reg_file({"": GAME_SETTINGS, "\\Video": VIDEO_SETTINGS, "\\Sound": SOUND_SETTINGS}),
                   encoding="utf-16")
    subprocess.run(["wine", "regedit", "/S", "C:\\warcraftsim.reg"], env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["wineserver", "-w"], env=env, check=False)
    docs = documents_dir(prefix)
    for sub in ("CustomMapData", "Maps", "Replay", "Logs", "Errors"):
        (docs / sub).mkdir(parents=True, exist_ok=True)
    marker.write_text("ok\n")
    return prefix


def clone_prefix(name: str, fresh: bool = False) -> Path:
    """An instance prefix: hard links to the template, private copies of everything writable."""
    template = template_prefix()
    if not (template / ".warcraftsim-template").exists():
        raise RuntimeError("template prefix missing; run `python -m warcraftsim setup` first")
    dest = paths.RUNTIME_DIR / "instances" / name / "prefix"
    if dest.exists():
        if not fresh and (dest / ".warcraftsim-instance").exists():
            return dest
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Hard-link everything under drive_c/windows (read-only Wine system files); copy the rest.
    subprocess.run(["cp", "-al", str(template), str(dest)], check=True)
    for rel in ("system.reg", "user.reg", "userdef.reg"):
        f = dest / rel
        if f.exists():
            f.unlink()
            shutil.copy2(template / rel, f)
    users = dest / "drive_c" / "users"
    shutil.rmtree(users)
    shutil.copytree(template / "drive_c" / "users", users, symlinks=True)
    for f in (dest / "drive_c" / "windows").glob("*.ini"):
        f.unlink()
        shutil.copy2(template / "drive_c" / "windows" / f.name, f)
    work = dest / "drive_c" / WORK_DIR
    if work.exists() or work.is_symlink():
        shutil.rmtree(work) if work.is_dir() and not work.is_symlink() else work.unlink()
    work.mkdir()
    (dest / ".warcraftsim-instance").write_text("ok\n")
    return dest


def kill_prefix(prefix: Path) -> None:
    subprocess.run(["wineserver", "-k"], env=wine_env(prefix), check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
