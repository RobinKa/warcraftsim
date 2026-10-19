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
* **Replays of stock-map games** ([human replays](human-replays.md)). A replay finds its map only with the replay's checksums (a modified map: rewrite them into the replay). Actions name units by object ids (a slot and the count of objects made before, at `+0xc`/`+0x10` of every engine object), so a script that makes any object (a trigger, a timer's handler, an event's data each time it fires) gives every later unit another id, and the replay's orders reach nobody. The observing harness makes its objects at once, and the replay's ids are shifted by their count.
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
* **A WSL2 laptop as a second machine** (2026-10-17, `laptop-wsl`: Ubuntu 24.04, Ryzen 9 5900HX, RTX 3050 Ti; the [remote evaluation worker](../architecture.md#a-remote-evaluation-worker)).
  * GE-Proton's Wine runs the 32-bit game with 32-bit Unix libraries (the old WoW64): a plain Ubuntu needs them from i386 (`libc6`, Mesa's GL and EGL, X11, freetype, fontconfig, glib, gnutls, dbus, cups, krb5, mpg123, wayland). No WineHQ package and no MinGW: the shim's DLL is copied.
  * The games draw with llvmpipe (`display.SOFTWARE_GL`), never through `/dev/dxg`. The worker's inference is CUDA next to them (torch cu130, the driver's `/usr/lib/wsl/lib`).
  * A game step costs 6.97 ms of CPU there (`scripts/wine_sync_ab.py`: 4 games of the built-in AI on duelfast, fsync, no cuts, the machine otherwise idle). WSL's own cost does not show: the main machine measured 9.2 ms under WSL and 7.4 natively (with other work running).
  * Tailscale SSH holds a call open while any process it started runs, `setsid nohup ... &` too. The worker runs as a systemd user service (`systemd-run --user`; the user lingers), which returns at once and outlives the session.
  * WSL powers off from the Windows side: "Clock change detected" (the laptop slept or woke), then the Plan 9 server's "Operation canceled" and "The system will power off now!": twice in its first hour (once before any game ran), with a terminal open. It came back 5-10 minutes later each time, and the manager started the worker again.
  * Its clock can be off after a sleep: the manager measures the offset at each look and moves the rows' times when it is over 2 s.
* **Rules in a stock map** (2026-10-15). The game's unit, upgrade and ability tables (`Units\UnitBalance.slk` and the others) and its gameplay constants (`war3mapMisc.txt`), written into a copy of Echo Isles, change the rules there as on the duel maps (`echofast`: duelfast's hit points, times and costs in the real map). The built-in AI plays them as usual.
* **Pairs of players** (`--pairs`, several games a map load) need extra start locations, which only the generated duel maps have. Echo Isles has two: one pair a load.
* **Building placement on Echo Isles.** Its bases are ringed by trees and cliffs. A build order at a spot where the building does not fit is refused. The harness then tries rings of spots up to 768 away (`W3S_Build`: 8 spots a ring up to 256, 16 beyond). Up to 256 it took only 25 of undead's 100 build orders in a test game (its ziggurats 8 of 67, off the blight) and 11 of night elf's 54. Up to 768: 26 of 33 and 14 of 19. A policy's learned spots from the flat duel map mostly fit only when the frame kept the duel maps' scale around the base (in a frame scaled to the map's size: 5 of 67 barracks orders taken).
* **A game the built-in AI never starts.** About 1 game in 28,000 ends within its first second with the learner the winner: the AI's player never comes in. Its row has no food at one minute.
* **No automatic camera in 1.29 replays.** The game's replay viewer has none: the renderer moves the camera itself (`video.Director`: the fight, else the players' armies and bases in turn).

## Offsets used by the shim (1.29.2.9231, sha256 3f2ed012…0eed)

| RVA | What |
|---|---|
| `0x396c80` | `rdtsc; ret` precise-timer helper (patched to the virtual clock) |
| `0x1aefd0` | `GameUpdate(this, now_ms)` (turbo) |
| `0x54c960` / `0x557db2` | Turn-time writer and its return address in the offline host (turbo) |
| `0x3d1070` | `GxPresent` (profiling only) |

The layout of these was documented by the MIT-licensed `pwang724/wc3env` project.

## Not done yet / next steps

* **Faster whole-game training** (the machine's CPU is its limit, see `optimization-log.md`):
  * the games: they set the speed again since 2026-10-14 (the learner waits ~1 s for each batch). Left: the harness's JASS, its native calls, A*'s node lookup and open list.
  * a cheaper video renderer (3.4 cores while it renders).
* **Human replays** (2026-10-17, in progress): ~3,000 replays of 1.29 from warcraft3.info. A converter reads their build orders (any game version) and replays 1.29 games in the real game to make demonstrations ([human replays](human-replays.md)).
* **Replays in the stock client.** Agent orders are not in the `.w3g`; they are in the command log next to it.
  * Recording agent orders as engine actions would need the command-packet path (wc3env's approach): selection plus order packets sent as the local player's network actions.
  * Game-cache syncs (`SyncStoredInteger`) are not recorded in single-player replays; tested and ruled out.

## The insane AI gathers twice as much

The built-in AI on insane gets double income from the engine; nothing in its scripts (`common.ai`, `human.ai`, ...) or the game's data tables says so. Measured on Echo Isles, insane against normal of the same race, 5 game minutes (2026-10-18): orc insane 5,600 gold gathered while its mine lost 2,820 (ratio 1.99), normal 2,820 from 2,840 (0.99); human insane 5,600 gold and 2,222 lumber, normal 2,790 and 1,129 with the same 16 workers; night elf insane 5,700 gold and 1,830 lumber, normal 2,860 and 890 with 17 workers. The evaluation games keep this: the "untaxed insane AI" is the game's own insane AI, with its double income. The curriculum's tax takes its share of the doubled income.

## Determinism (2026-10-19)

Measured on `echofast` (`GameSetup.seed`: `SetRandomSeed` first thing in the map's `main`; games side by side, the units' types, owners, positions and hit points compared every step):

* Without a seed, two identical games (an idle agent against the normal AI) differ from step 4.
* The start locations on Echo Isles are drawn before the map's script runs: the seed alone does not fix them (a game's spawns match another's half the time). With a seed the map's `config` now also sets `MAP_PLACEMENT_FIXED` and `ForcePlayerStartLocation(Player(i), i)`: 4 games of 4 with the same spawns, identical for 300 steps.
* With a seed and the same spawns, **games between agents with the same orders are identical** (3 games, 300 steps of harvesting and walking across the map). The engine is deterministic.
* **The built-in AI is not**: with a seed, AI against AI differed from step 26, an idle agent against the AI from step 36. Its script threads seem to run on the wall clock.

So games can be replayed and branched mid-game only between agents (self-play, past snapshots).

