"""How the machines of a run are doing (the dashboard's System tab, "Machines"): every --every seconds, for each --host
one ssh call that reads /proc (CPU busy over ~1 s, memory, load, clocks, the game processes, uptime), `sensors` (the
CPU's Tctl: WSL has none, null) and the power profile, as a row in runs/hosts/<name>.jsonl. A host that does not
answer gets a row with up: false and the others carry on. Light: no torch, one short ssh call a host. Rows older than
--keep days are dropped at the start (and every few hours).

    setsid nohup .venv/bin/python scripts/host_stats.py --host robin@laptop-wsl --host tora-ms-a2@tora-ms-a2 \\
        > runs/host_stats.log 2>&1 &
    .venv/bin/python scripts/host_stats.py --host robin@laptop-wsl --once     # one look, printed, nothing written
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOSTS = ROOT / "runs" / "hosts"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4"]
# What runs there (sh): a key and its value a line. The games are counted by their comm in /proc (awk reads them all in
# one process; pgrep would match its own shell, whose command line holds the name).
REMOTE = r"""
echo "stat1 $(head -1 /proc/stat)"; sleep 1; echo "stat2 $(head -1 /proc/stat)"
grep -E '^(MemTotal|MemAvailable):' /proc/meminfo
echo "load $(cut -d' ' -f1-3 /proc/loadavg)"
echo "uptime $(cut -d' ' -f1 /proc/uptime)"
echo "mhz $(awk '/^cpu MHz/{s+=$4;n++;if($4>m)m=$4} END{if(n)print s/n, m}' /proc/cpuinfo)"
echo "games $(awk 'FNR==1&&$0=="Warcraft III.ex"{n++} END{print n+0}' /proc/[0-9]*/comm 2>/dev/null)"
if command -v sensors >/dev/null 2>&1; then sensors 2>/dev/null | grep -m1 -E 'Tctl|Package id 0' | sed 's/^/temp /'; fi
p=$(powerprofilesctl get 2>/dev/null || cat /sys/firmware/acpi/platform_profile 2>/dev/null); echo "profile $p"
"""


def log(*a) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def _cpu_busy(line1: str, line2: str) -> float | None:
    """Busy percent between two `cpu ...` lines of /proc/stat (idle and iowait count as idle)."""
    def parse(s: str) -> tuple[float, float]:
        v = [float(x) for x in s.split()[1:9]]
        return sum(v), v[3] + v[4]
    try:
        (t1, i1), (t2, i2) = parse(line1), parse(line2)
    except (ValueError, IndexError):
        return None
    return round(100 * (1 - (i2 - i1) / (t2 - t1)), 1) if t2 > t1 else None


def parse_stats(text: str, now: float, name: str) -> dict:
    """The row for what REMOTE printed. A reading missing there (no sensors, no clocks) is null."""
    f: dict[str, str] = {}
    for line in text.splitlines():
        k, _, v = line.partition(" ")
        f.setdefault(k.rstrip(":"), v.strip())
    row: dict = {"time": now, "host": name, "up": True}
    row["cpu_busy"] = _cpu_busy(f["stat1"], f["stat2"]) if f.get("stat1") and f.get("stat2") else None
    for key, out in (("MemTotal", "mem_total"), ("MemAvailable", "mem_avail")):
        m = re.match(r"(\d+)", f.get(key, ""))
        row[out] = round(int(m.group(1)) / 1048576, 2) if m else None  # (GiB)
    try:
        row["load"] = [float(x) for x in f["load"].split()[:3]]
    except (KeyError, ValueError):
        row["load"] = None
    m = re.match(r"\d+(?:\.\d+)?", f.get("uptime", ""))
    row["uptime"] = float(m.group()) if m else None
    mhz = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", f.get("mhz", ""))]
    row["mhz_mean"], row["mhz_max"] = (round(mhz[0]), round(mhz[1])) if len(mhz) >= 2 else (None, None)
    row["games"] = int(f["games"]) if f.get("games", "").isdigit() else None
    m = re.search(r"(?:Tctl|Package id 0):\s*\+?(-?\d+(?:\.\d+)?)", f.get("temp", ""))
    row["temp"] = float(m.group(1)) if m else None
    row["profile"] = f.get("profile") or None
    return row


def look(host: str, name: str, timeout: float = 25.0) -> dict:
    """One host's row (up: false when it did not answer in time or the call failed)."""
    now = time.time()
    try:
        r = subprocess.run(SSH + [host, "sh -c " + shlex.quote(REMOTE)], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:  # (Tailscale SSH waiting for an approval in a browser)
        return {"time": now, "host": name, "up": False, "error": "timeout"}
    if r.returncode != 0 or "stat2" not in r.stdout:
        return {"time": now, "host": name, "up": False,
                "error": (r.stderr.strip().splitlines() or [f"exit {r.returncode}"])[-1][:200]}
    return parse_stats(r.stdout, now, name)


def trim(path: Path, keep_days: float) -> None:
    """Rows older than keep_days dropped (the file written again through a temporary one)."""
    if not path.exists():
        return
    cut, kept = time.time() - keep_days * 86400, []
    for line in path.read_text().splitlines():
        try:
            if json.loads(line).get("time", 0) >= cut:
                kept.append(line)
        except json.JSONDecodeError:
            pass
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(x + "\n" for x in kept))
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", action="append", required=True,
                    help="ssh destination user@name (repeatable); its rows are called name, or name=user@host to call it otherwise")
    ap.add_argument("--every", type=float, default=30.0, help="seconds between looks")
    ap.add_argument("--keep", type=float, default=7.0, help="days of rows kept")
    ap.add_argument("--once", action="store_true", help="one look at each host, printed; nothing written")
    a = ap.parse_args()
    hosts = []
    for h in a.host:
        name, _, dest = h.partition("=")
        hosts.append((dest, name) if dest else (h, h.split("@")[-1]))
    if a.once:
        for h, n in hosts:
            print(json.dumps(look(h, n)))
        return 0
    HOSTS.mkdir(parents=True, exist_ok=True)
    last_trim = 0.0
    log("looking at", ", ".join(n for _, n in hosts), f"every {a.every:.0f} s")
    with ThreadPoolExecutor(len(hosts)) as pool:
        try:
            while True:
                t0 = time.time()
                if t0 - last_trim > 6 * 3600:
                    for _, n in hosts:
                        trim(HOSTS / f"{n}.jsonl", a.keep)
                    last_trim = t0
                for row in pool.map(lambda hn: look(*hn), hosts):  # (all at once: a host that hangs holds only its own)
                    with open(HOSTS / f"{row['host']}.jsonl", "a") as f:
                        f.write(json.dumps(row, separators=(",", ":")) + "\n")
                time.sleep(max(1.0, a.every - (time.time() - t0)))
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
