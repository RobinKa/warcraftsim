# Architecture and engine notes

## Components

| Path | Role |
|---|---|
| `warcraftsim/data/mpq.py` | StormLib (ctypes). Reads the game archives and reads/writes map archives. |
| `warcraftsim/data/mapbuild.py` | Injects the harness into a stock map script, then validates it with pjass. |
| `warcraftsim/data/wgc.py` | Writes `.wgc` game configs (format: Luashine/wc3-file-formats). |
| `warcraftsim/data/terrain.py` | Parses `war3map.w3e` and `war3map.wpm`, and finds open areas. |
| `warcraftsim/data/flatmap.py` | Flat, empty maps: a stock map flattened, or resized to N×N tiles (the default scenario map). |
| `warcraftsim/data/w3i.py` | Map-info (`war3map.w3i`) reader/writer: camera bounds, border tiles, playable size, start locations. |
| `warcraftsim/data/objects.py` | SLK unit table and the unit-type vocabulary. |
| `warcraftsim/harness/w3sim.j` | In-map controller (JASS, 1.29 has no Lua). |
| `warcraftsim/protocol.py` | Observation parsing (checksummed token records, deltas) and command encoding. |
| `warcraftsim/runtime/wine.py` | Template prefix plus hard-link-cloned prefixes, one per game. |
| `warcraftsim/runtime/display.py` | One private Xvfb per game. |
| `warcraftsim/runtime/instance.py` | Process lifecycle, the TCP step protocol and IPC files in `/dev/shm`. |
| `warcraftsim/client.py`, `env.py`, `vec.py`, `scenario.py` | User-facing API. |
| `shim/` | `w3shim.dll` (clock, sync, the pass over the units, turbo, profiler) and the `w3launch.exe` injector. |
| `warcraftsim/video.py` | Replay playback to MP4: frame-stepped clock, every frame grabbed from Xvfb (XGetImage) into ffmpeg; audio from the shim's virtual sound card (`shim/audio.c`), muxed with its latency removed. |
| `puffer/wc3_bridge.h`, `warcraftsim/puffer/` | PufferLib 5.0: C bridge environment, tasks, bridge server, trainer build, `train` orchestrator. |
| `warcraftsim/dashboard/` | Training dashboard: a standard-library HTTP server plus one self-contained page. |

## Step protocol

1. The harness timer fires every `step_seconds` of game time.
2. The harness writes the observation with `PreloadGenStart`, then `Preload(token)`×N, then `PreloadGenEnd("w3sim\\obs.txt")`. The shim hooks `Preload`/`PreloadGenEnd` (`shim/obs.c`) and keeps the tokens in memory instead of checking the disk per token and writing a file.
   * The units are the shim's part (`shim/units.c`). The harness hands it each unit's handle id (`ForGroup` over all units, a call of the mailbox native per unit with the negated id), and the shim makes the native calls itself: it drops units that are gone or were reported dead, counts the living ones for the result, and keeps the records of those that changed, which go into the observation where the harness used to write them.
   * The natives are plain cdecl functions in the executable: handles and integers by value, a real returned as its bits in `eax`, and "handles" like `UNIT_TYPE_HERO` are the integers themselves. The game registers each with `push signature; push name; push function; call`, so the shim finds them by name (the first registration of a name is the real function; a second table registers every name with one stub).
   * In JASS this was three passes with ~35 native calls and a few hundred instructions per unit: 9-13 ms a step with 150 units and 32 games running, against 4 now (4.0 and 1.5 ms on an idle machine).
   * `W3SIM_UNITS=0` leaves it all to the harness (also what happens without the shim or with a native missing); `W3SIM_UNITS=2` has the harness write the records and the shim check each against its own. An integration test plays a game with the harness's pass and its replay with the shim's and compares every field of every unit and player on every step.
3. The harness calls `GetPlayerTechMaxAllowed(Player(PLAYER_NEUTRAL_PASSIVE), 1048575)`. The shim hooks that native (`shim/sync.c`): it freezes the virtual clock, sends `OBS n len` followed by the tokens, then blocks on the socket.
4. Python parses and merges the observation and answers `GO [speed=..] [turbo=..] [frame=..] [capture=..] A n v1 .. vn` with the encoded commands.
5. The native returns n; the harness reads the command integers with the same native (keys 1048577+i, answered from the `A` list) and issues the orders.

Before the mailbox, actions went through a file the harness loaded with `Preloader`. The engine compiled it as JASS on every step (~7 ms under Wine, and ~20 KB of compiler memory that was never freed: "Not enough memory" after ~20k steps, hence game recycling). Games no longer need recycling (`recycle_steps` is a safety net at 200k steps). The `CreateFileW` hook on `act.txt` still exists as a fallback sync point.

## Things learned about the 1.29 engine (keep these in mind)

* **Preload buffer**
  * The file written by `PreloadGenEnd` records *everything* preloaded while generating, including the engine's own resource loads from other threads (sound files, occasionally a number).
  * Hence every record carries an int32 checksum, and the parser repairs or skips damaged records.
  * Never put `PreloadEnd(0.0)` in the action file: it waits for all recorded preloads (about 1 s per step).
* **JASS strings are never freed.** Observation tokens are short integers, so the set of distinct strings stays bounded.
* **Delta observations.** A unit is written only when a rolling hash of its fields changed. Removed units get `R` records, and `Snapshot()` forces a full observation. The Python side keeps the table (`merge_observation`).
* **Hidden units.** Area enumerations skip hidden units, such as workers inside a gold mine. The harness therefore tracks units that enter the map (plus an initial enumeration) itself.
* **Replays.** A replay stores the map as `..\w3sim\map.w3x` and resolves it from `Documents\Warcraft III`, so each instance links `Documents\Warcraft III\w3sim` to `C:\w3sim`. Playback runs the harness again (it writes observations and syncs), which is how `play_replay` feeds back the command log.
* **Resets.** `RestartGame(false)` called by the harness reloads the map inside the running `.wgc` game: the loading screen runs again, and the new game has the same slots and a fresh built-in AI. (Earlier notes here said it drops a `.wgc` game back to the main menu; re-tested, it does not.) `ChangeLevel` and `LoadGame` were not re-tested.
  * Melee restarts use it by default (`GameSetup.engine_restart`). A reload takes 1.0 s on the duel maps, against about 4 s for a new launch; over 20 reloads in one process the time stayed the same and memory grew by about 1 MB per reload. With 32 games running it takes 4-5 s (and costs 1.4 s of CPU; 2.1 s before the loading screen was drawn in the game's thread: `GameSetup.render_threads`, `d3d_thread`).
  * What made reloads slow, and no longer does:
    * **Map object data** (`w3u`, `w3q`, `w3a`) is applied change by change at every load: the rush rules took 2 s (every modified unit counts). The same values as changed copies of the game's tables (`Units\UnitBalance.slk`, `UnitWeapons.slk`, `UnitData.slk`, `UpgradeData.slk`, `AbilityData.slk`) in the map cost nothing: the game reads a map's copies of its files first (`data.objects.patch_slk`, `duelmap.rules_files`).
    * **"Allow Local Files"** (registry) makes the game look in its folder for every file before its archives; under Wine each miss scans the directory. `GameSetup.local_files` is off: nothing needs it since the actions stopped coming from a file.
    * The overriding AI scripts cost nothing. Melee's preload lists (`Scripts\*Melee.pld`) cost nothing either: the shim's `Preload` hook does not pass them on.
  * The loading screen waits about two seconds of game time while drawing frames, so a slow clock during a reload makes it cost more, not less.
  * With `engine_restart=False`, melee resets use a **warm spare**: a second process with its own prefix, display and IPC directory. It loads in the background and waits frozen at game time 0 (the harness is blocked in its first sync, so it uses no CPU).
  * On `restart()` the instance then swaps process fields with the spare (`_PROCESS_ATTRS`). The retired process is shut down in the background, and the next spare loads under its name. A spare restarts in about 1 s, but every game costs a full launch, which competes for the CPU when many games run.
  * A scripted melee reset (`GameSetup.melee_reset`: remove every unit and respawn the start) takes 0.1 s, but it is **not a new game**:
    * the built-in AI does not survive it (its engine state outlives the reset, with its scripts started anew too);
    * the engine keeps counting a removed hero: the type is refused from then on and the next hero needs the second tier, although `GetPlayerTechCount`, the limits and the hero tokens read as in a new game (handing the hero to the neutral player first does not help);
    * in self-play's games between agents, research all but stopped and the food count drifted (down to -345).
  * **A pair of players per game** (`GameSetup.pairs`, duel maps, up to 5) is the fast reset that is a new game: the map has 2·pairs players, game k of a load is played by players 2k and 2k+1, and only the last game's restart reloads the map. The engine's leftovers are per player, and these players have not played.
    * Python sees players 0 and 1 in every game: the harness and the shim map the ids in observations and commands (`W3S_Pid`, `W3S_Real`).
    * The switch takes three ticks (`W3S_PairClear`, `W3S_PairMake`, `W3S_StepAgain`), as a load does. Tick 1 removes everything (units, corpses, items), regrows the trees, clears the blight and pauses the old built-in AI. Tick 2 removes what the engine made in between (a removed haunted or entangled gold mine leaves a plain one behind), then makes the map's mines and creeps and the new players' starting units and starts their AI. Tick 3 collects the units by enumeration, as after a load (`W3S_Retrack`: enter events would also have recorded the hidden mine under an entangled one), and steps at once, so the first observation is a load's.
    * The map's `config` gives every pair its own team and the two start locations to every pair; `MeleeStartingUnits` is replaced by one for the pair in play (`W3S_StartingUnits`).
    * 12 players crash the game while loading; 10 and the observer work.
  * Scenarios reset inside the game.
* **Melee AI start.** In 1.29 the melee start sends the starting workers to the mine automatically for every player. The AI is started by the map's `MeleeStartingAI`, which the harness replaces with one that skips agent slots. The AI reads its level through the native `MeleeDifficulty()`, which comes from the lobby or `.wgc`.
* **Old-format maps** use players 12-15 as the neutral players, so the harness loops over `bj_MAX_PLAYERS` (12).
* **Clock speed and Wine overhead.** Scaled waits make the game's background threads wake `speed` times as often, and under Wine every wake-up is a wineserver round trip. At 64x the wineserver cost about 0.35 cores per game; at 128x it was worse overall.
  * The default adaptive clock keeps `speed` at about 2.5x the rate the game actually reaches.
  * WineHQ stable 11.0 needed less wineserver CPU than staging 11.18. Esync made no difference.
* **Speed.** The simulation itself costs about 0.6 ms of wall time per 25 ms turn (a 25-minute AI-vs-AI game in 39 s). Per 0.25 s step:
  * the harness plus the sync cost roughly 3-6 ms;
  * serializing about 120 units cost about 3 ms in JASS (with deltas; now done by the shim, see the step protocol);
  * Python parsing costs about 0.5-1 ms.
  * Rendering is only a few frames per second.
  * A `duelrush` step (0.5 s, built-in AI on both sides) costs 14.6 ms of CPU in a game alone (68 steps/s) and 35-40 ms with 32 games running; about 8 ms of it is the harness (from steps of 0.25, 0.5 and 2 s). The game makes about 500 wineserver requests a step (its threads set and wait for events at the virtual clock's pace): a tenth of the CPU.
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

* **Faster whole-game training** (the machine's CPU is its limit; see `experiment-log.md`):
  * the learner (a compiled update, fewer passes): it is the limit since the games stopped reloading the map each time;
  * a cheaper video renderer (3.4 cores while it renders).
* **Replays in the stock client.** Agent orders are not in the `.w3g`; they are in the command log next to it.
  * Recording agent orders as engine actions would need the command-packet path (wc3env's approach): selection plus order packets sent as the local player's network actions.
  * Game-cache syncs (`SyncStoredInteger`) are not recorded in single-player replays; tested and ruled out.
