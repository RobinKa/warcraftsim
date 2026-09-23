# Architecture and engine notes

## Components

| Path | Role |
|---|---|
| `warcraftsim/data/mpq.py` | StormLib (ctypes). Reads the game archives and reads/writes map archives. |
| `warcraftsim/data/mapbuild.py` | Injects the harness into a stock map script, then validates it with pjass. |
| `warcraftsim/data/wgc.py` | Writes `.wgc` game configs (format: Luashine/wc3-file-formats). |
| `warcraftsim/data/terrain.py` | Parses `war3map.w3e` and `war3map.wpm`, and finds open areas. |
| `warcraftsim/data/objects.py` | SLK unit table and the unit-type vocabulary. |
| `warcraftsim/harness/w3sim.j` | In-map controller (JASS, 1.29 has no Lua). |
| `warcraftsim/protocol.py` | Observation parsing (checksummed token records, deltas) and command encoding. |
| `warcraftsim/runtime/wine.py` | Template prefix plus hard-link-cloned prefixes, one per game. |
| `warcraftsim/runtime/display.py` | One private Xvfb per game. |
| `warcraftsim/runtime/instance.py` | Process lifecycle, the TCP step protocol and IPC files in `/dev/shm`. |
| `warcraftsim/client.py`, `env.py`, `vec.py`, `scenario.py` | User-facing API. |
| `shim/` | `w3shim.dll` (clock, sync, turbo, profiler) and the `w3launch.exe` injector. |

## Step protocol

1. The harness timer fires every `step_seconds` of game time.
2. The harness writes the observation with `PreloadGenStart`, then `Preload(token)`×N, then `PreloadGenEnd("w3sim\\obs.txt")`.
3. The harness calls `Preloader("w3sim\\act.txt")`, which makes the game open the file with `CreateFileW`. The shim intercepts that call, freezes the virtual clock and sends `OBS n`, then blocks on the socket.
4. Python reads `obs.txt` and parses and merges it. It writes `act.txt` (lines of `call SetPlayerTechMaxAllowed(Player(PLAYER_NEUTRAL_PASSIVE), 1048576+i, v)`) and answers `GO [speed=..] [turbo=..]`.
5. The game runs the action file. The Preloader interpreter executes natives with nested calls and constants, but not map functions, `set`, or `BlzSetAbilityTooltip`. The harness then reads the mailbox and issues the orders.

## Things learned about the 1.29 engine (keep these in mind)

* **Preload buffer**
  * The file written by `PreloadGenEnd` records *everything* preloaded while generating, including the engine's own resource loads from other threads (sound files, occasionally a number).
  * Hence every record carries an int32 checksum, and the parser repairs or skips damaged records.
  * Never put `PreloadEnd(0.0)` in the action file: it waits for all recorded preloads (about 1 s per step).
* **JASS strings are never freed.** Observation tokens are short integers, so the set of distinct strings stays bounded.
* **Delta observations.** A unit is written only when a rolling hash of its fields changed. Removed units get `R` records, and `Snapshot()` forces a full observation. The Python side keeps the table (`merge_observation`).
* **Hidden units.** Area enumerations skip hidden units, such as workers inside a gold mine. The harness therefore tracks units that enter the map (plus an initial enumeration) itself.
* **Resets.** `RestartGame`, `ChangeLevel` and `LoadGame` all drop a `.wgc` game back to the main menu. `RestartGame` does work when the map is launched with `-loadfile map.w3x`, but then slots and AI difficulty cannot be set.
  * Melee resets therefore relaunch the process (about 8 s).
  * Scenarios reset inside the game.
* **Melee AI start.** In 1.29 the melee start sends the starting workers to the mine automatically for every player. The AI is started by the map's `MeleeStartingAI`, which the harness replaces with one that skips agent slots. The AI reads its level through the native `MeleeDifficulty()`, which comes from the lobby or `.wgc`.
* **Old-format maps** use players 12-15 as the neutral players, so the harness loops over `bj_MAX_PLAYERS` (12).
* **Speed.** The simulation itself costs about 0.6 ms of wall time per 25 ms turn (a 25-minute AI-vs-AI game in 39 s). Per 0.25 s step:
  * the harness plus the sync cost roughly 3-6 ms;
  * serializing about 120 units costs about 3 ms (with deltas);
  * Python parsing costs about 0.5-1 ms.
  * Rendering is only a few frames per second.
  * `turbo` (engine turn pacing bypass: `W3SIM_TURBO_MS`) exists but did not help at 0.25 s steps.
* **Launching.** `-graphicsapi Null` crashes 1.29. The minimum working window is 800x600 (400x300 hangs). The game must be started with a proper working directory: the launcher uses `C:\Warcraft III`, since `.wgc` map paths are relative to it.
* **WSLg.** `/tmp/.X11-unix` is read-only, so Xvfb listens only on the abstract socket. Readiness is checked in `/proc/net/unix`.

## Offsets used by the shim (1.29.2.9231, sha256 3f2ed012…0eed)

| RVA | What |
|---|---|
| `0x396c80` | `rdtsc; ret` precise-timer helper (patched to the virtual clock) |
| `0x1aefd0` | `GameUpdate(this, now_ms)` (turbo) |
| `0x54c960` / `0x557db2` | Turn-time writer and its return address in the offline host (turbo) |
| `0x3d1070` | `GxPresent` (profiling only) |

The layout of these was documented by the MIT-licensed `pwang724/wc3env` project.

## Not done yet / next steps

* **Melee resets without relaunching.** Options: keep a warm spare process per slot, or use `-loadfile map` plus a difficulty override.
* **Faster steps:**
  * move serialization into the shim by hooking natives;
  * skip rendering entirely;
  * profile the JASS VM cost.
* **Ability order strings** (`stormbolt`, ...) in the order table, so `cast()` accepts any ability.
* **A generated flat scenario map**, instead of a cleared patch of a stock map.
