"""warcraft3.info's replay archive (public API, no login): an index of its replays' metadata, and
downloads of the ones an index filter picks; and W3Champions' ladder 1v1s (public API: the matches, with
the players' MMR, and their replays). Politely: one request at a time, ~1 per second.

    python scripts/replay_fetch.py index --out runs/fullgame/replay-survey/w3info-index.jsonl \
        --version 1.28 --version 1.30                      # every page of these versions (resumes)
    python scripts/replay_fetch.py download --index runs/fullgame/replay-survey/w3info-index.jsonl \
        --out runs/fullgame/human-replays-128 --version 1.28 --type 1on1 [--sha1 <map sha1> ...] [--limit N]
    python scripts/replay_fetch.py w3c --out runs/fullgame/replays-w3c-s25 --season 25 --min-mmr 1800 --limit 500

The list (POST /api/v1/replays {"page": N, "version": "1.29"}) gives 25 replays a page, newest first:
version, build, map (path, checksums, the site's name for it), type, length, players and races, origin
(e.g. Netease). A download (GET /api/v1/replays/<id>/download) is the replay file (NetEase's in the GRBN
container: data/w3g.py reads it). Files are untrusted data: written as <id>.w3g into a directory of
their own, never run; parse them with `python -I` (scripts/replay_stats.py).
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://warcraft3.info/api/v1/replays"
AGENT = "warcraftsim-research/1.0 (replay survey; ~1 request/s)"


def request(url: str, body: dict | None = None, tries: int = 5, pause: float = 1.1) -> bytes:
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "User-Agent": AGENT})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            time.sleep(pause)
            return data
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if isinstance(e, urllib.error.HTTPError) and 400 <= e.code < 500 and e.code != 429:
                raise
            time.sleep(min(300, 10 * 2 ** k))
    raise RuntimeError(f"{url}: no answer after {tries} tries")


def row_of(r: dict) -> dict:
    alias = (r.get("map_alias") or {}).get("map") or {}
    return {"id": r["id"], "version": r.get("version"), "build": r.get("build_number"), "map": r.get("map"),
            "map_name": alias.get("name"), "checksum": r.get("map_checksum"), "sha1": r.get("map_checksum_sha1"),
            "type": r.get("type"), "duration": r.get("duration"), "origin": r.get("origin"),
            "filetype": r.get("filetype"), "winner": r.get("winner") or r.get("detected_winner"),
            "created_at": r.get("created_at"), "event": bool(r.get("stats_event_id")),
            "players": [{"name": p.get("player"), "race": p.get("race"), "team": p.get("team"),
                         "known": ((p.get("stats_player") or {}).get("name"))} for p in r.get("players") or []]}


def index(args) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    state_path = out.with_suffix(".state.json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    seen = set()
    if out.exists():
        seen = {json.loads(line)["id"] for line in out.read_text().splitlines() if line.strip()}
    for version in args.version:
        st = state.setdefault(version, {"next": 1, "last": None})
        while st["last"] is None or st["next"] <= st["last"]:
            d = json.loads(request(API, {"page": st["next"], "version": version}))
            st["last"] = d["last_page"]
            new = [row_of(r) for r in d["data"] if r["id"] not in seen]
            with open(out, "a") as f:
                for r in new:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    seen.add(r["id"])
            st["next"] += 1
            st["total"] = d["total"]
            state_path.write_text(json.dumps(state, indent=1))
            if st["next"] % 50 == 0:
                print(f"{version}: page {st['next'] - 1} of {st['last']} ({len(seen)} rows)", flush=True)
        print(f"{version}: done, {st['total']} replays", flush=True)


def download(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in Path(args.index).read_text().splitlines() if line.strip()]
    want = [r for r in rows
            if (not args.version or r["version"] in args.version) and (not args.type or r["type"] in args.type)
            and (not args.sha1 or (r.get("sha1") or "").lower() in {s.lower() for s in args.sha1})
            and (not args.map or any(m.lower() in (r.get("map") or "").lower() for m in args.map))]
    want.sort(key=lambda r: -r["id"])
    if args.limit:
        want = want[:args.limit]
    have = {p.stem for p in out.glob("*.w3g")}
    todo = [r for r in want if str(r["id"]) not in have]
    print(f"{len(want)} replays match, {len(todo)} to download", flush=True)
    with open(out / "index.jsonl", "a") as idx:
        for k, r in enumerate(todo):
            try:
                data = request(f"{API}/{r['id']}/download", pause=args.pause)
            except (urllib.error.HTTPError, RuntimeError) as e:
                print(f"{r['id']}: {e}", flush=True)
                continue
            tmp = out / f"{r['id']}.part"
            tmp.write_bytes(data)
            tmp.replace(out / f"{r['id']}.w3g")
            idx.write(json.dumps(r, ensure_ascii=False) + "\n")
            idx.flush()
            if (k + 1) % 100 == 0:
                print(f"{k + 1} of {len(todo)}", flush=True)


W3C = "https://website-backend.w3champions.com/api"
W3C_RACES = {0: "random", 1: "human", 2: "orc", 4: "nightelf", 8: "undead"}


def w3c(args) -> None:
    """W3Champions' ladder 1v1s (Reforged, current patch: build orders only) of a season, both players at
    --min-mmr or more: an index row each (map, length, races, MMR, winner) and the replay (the site keeps
    them a few months: older ones answer 404/410)."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in out.glob("*.w3g")}
    done = set()
    if (out / "index.jsonl").exists():
        done = {json.loads(line)["id"] for line in (out / "index.jsonl").read_text().splitlines() if line.strip()}
    got = 0
    offset = args.offset
    with open(out / "index.jsonl", "a") as idx:
        while got < args.limit:
            page = json.loads(request(f"{W3C}/matches?offset={offset}&pageSize=100&gameMode=1&season={args.season}",
                                      pause=args.pause))
            matches = page.get("matches") or []
            if not matches:
                break
            offset += len(matches)
            for m in matches:
                players = [p for t in m.get("teams") or [] for p in t.get("players") or []]
                if m["id"] in done or len(players) != 2 or min(p.get("oldMmr") or 0 for p in players) < args.min_mmr:
                    continue
                row = {"id": m["id"], "map": m.get("map"), "map_name": m.get("mapName"), "start": m.get("startTime"),
                       "duration": m.get("durationInSeconds"), "season": args.season,
                       "players": [{"name": p.get("name"), "race": W3C_RACES.get(p.get("race"), p.get("race")),
                                    "mmr": p.get("oldMmr"), "won": p.get("won")} for p in players]}
                if m["id"] not in have:
                    try:
                        data = request(f"{W3C}/replays/{m['id']}", pause=args.pause, tries=2)
                    except (urllib.error.HTTPError, RuntimeError) as e:
                        row["error"] = str(getattr(e, "code", e))
                        data = None
                    if data:
                        (out / f"{m['id']}.part").write_bytes(data)
                        (out / f"{m['id']}.part").replace(out / f"{m['id']}.w3g")
                        got += 1
                idx.write(json.dumps(row, ensure_ascii=False) + "\n")
                idx.flush()
                done.add(m["id"])
                if got >= args.limit:
                    break
            print(f"offset {offset}: {got} downloaded", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("index")
    a.add_argument("--out", required=True)
    a.add_argument("--version", action="append", required=True, help="e.g. 1.29 (repeat for more)")
    b = sub.add_parser("download")
    b.add_argument("--index", required=True)
    b.add_argument("--out", required=True, help="a directory for the files (its own, new)")
    b.add_argument("--version", action="append", default=[])
    b.add_argument("--type", action="append", default=[], help="e.g. 1on1")
    b.add_argument("--sha1", action="append", default=[], help="only maps with this checksum SHA-1")
    b.add_argument("--map", action="append", default=[], help="only map paths containing this")
    b.add_argument("--limit", type=int, default=0)
    b.add_argument("--pause", type=float, default=1.1, help="seconds between requests")
    c = sub.add_parser("w3c", help="W3Champions' ladder 1v1s (Reforged)")
    c.add_argument("--out", required=True, help="a directory for the files (its own, new)")
    c.add_argument("--season", type=int, required=True)
    c.add_argument("--min-mmr", type=int, default=1800)
    c.add_argument("--offset", type=int, default=0, help="where to start in the season's list (newest first)")
    c.add_argument("--limit", type=int, default=100, help="this many downloads")
    c.add_argument("--pause", type=float, default=1.1, help="seconds between requests")
    args = ap.parse_args()
    {"index": index, "download": download, "w3c": w3c}[args.cmd](args)


if __name__ == "__main__":
    main()
