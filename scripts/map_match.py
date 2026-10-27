"""Whether candidate map files are the versions a set of 1.29 replays was played on (e.g. NetEase's ladder maps, which
the converter skips: "another version of <map>"): each candidate's classic checksum against the replays', and, as the
game's scripts may differ (NetEase's client), the candidate peeled back out of each replay checksum of its map name: two
maps leaving the same remainder (the scripts' part) are the replays' versions. The stock 1.29 scripts leave c3aa7b5f.

    python -I scripts/map_match.py <maps...>     # (candidates: untrusted files; nothing runs them)
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from warcraftsim.data import w3g  # noqa: E402
from warcraftsim.data.mapbuild import game_scripts  # noqa: E402
from warcraftsim.data.mpq import MpqArchive  # noqa: E402

# NetEase's 1.29 versions (docs/reference/human-replays.md): map name -> the replays' classic checksums
NETEASE_129 = {"lastrefuge": (0xa39562b3, 0x1853563b), "amazonia": (0x23d68f0b,), "echoisles": (0x94b4de8e,),
               "twistedmeadows": (0xc43206f3,), "terenasstand_lv": (0x1a181b47,), "turtlerock": (0x6a9d1f28,)}


def _rotr3(v: int) -> int:
    return ((v >> 3) | (v << 29)) & 0xFFFFFFFF


def checksum_files(path: Path) -> dict[str, bytes]:
    with MpqArchive(path) as m:
        return {n: m.read(n) for n in w3g.CHECKSUM_FILES if n in m}


def peel(target: int, files: dict[str, bytes]) -> int:
    """The scripts' part of `target` if `files` are the map it was computed over (w3g.map_checksums run backwards)."""
    names, script = [], False
    for name in w3g.CHECKSUM_FILES:
        if files.get(name) is None or (script and name == "scripts\\war3map.j"):
            continue
        script = script or name.endswith("war3map.j")
        names.append(name)
    v = target
    for name in reversed(names):
        v = _rotr3(v) ^ w3g._xoro(files[name])
    return v


def key(path: Path) -> str:
    return re.sub(r"^\(\d+\)", "", path.stem.split("_", 1)[-1] if re.match(r"^\d+_", path.stem) else path.stem).lower()


def main(argv: list[str]) -> int:
    common, blizzard = (p.read_bytes() for p in game_scripts())
    stock = w3g._rotl3(w3g._rotl3(w3g._xoro(common) ^ w3g._xoro(blizzard)) ^ 0x03F1379E)
    targets = {v: k for k, vs in NETEASE_129.items() for v in vs}
    rest: dict[int, list[str]] = collections.defaultdict(list)
    for arg in argv:
        p = Path(arg)
        try:
            files = checksum_files(p)
        except Exception as e:  # noqa: BLE001
            print(f"{p.name}: unreadable ({type(e).__name__})")
            continue
        xoro, _ = w3g.map_checksums(files, common, blizzard)
        hit = targets.get(xoro)
        print(f"{p.name}: checksum {xoro:08x}" + (f"  EXACT MATCH: NetEase 1.29 {hit}" if hit else ""))
        for t in NETEASE_129.get(key(p), ()):
            r = peel(t, files)
            rest[r].append(f"{p.name}->{t:08x}")
            if r == stock:
                print(f"   peels {t:08x} to the stock scripts' part: the replays' version")
    shared = {f"{k:08x}": v for k, v in rest.items() if len(v) > 1}
    print("remainders shared by two or more candidates (their versions, under the same scripts):", json.dumps(shared) if shared else "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
