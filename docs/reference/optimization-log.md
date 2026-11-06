# Optimizations

How a step of the real game became cheap enough to train on, what each fix bought, what limits throughput now, and what did not work. The numbers are from this machine: Ryzen 5950X (16 cores, 32 threads), 47 GB, RTX 3090, WSL2, WineHQ stable 11.0, llvmpipe; from 2026-10-11 the same machine under native Ubuntu 26.04 (60 GB, GE-Proton 10-34's Wine; see the section on the move).

## A game step

| fix | what it saved |
|---|---|
| Actions through a hooked native (`GetPlayerTechMaxAllowed` on mailbox keys) instead of a file the harness loaded with `Preloader` | ~7 ms a step of JASS compilation, and a ~20 KB leak per step that forced game recycling |
| Observations captured by hooking `Preload` / `PreloadGenEnd` instead of written to a file | a wineserver round trip per token |
| The harness's pass over the units done by the shim in C (`shim/units.c`: the game's natives called directly) | 9–13 → 4 ms a step with 150 units under load (4.0 → 1.5 ms on an idle machine); self-play 644 → 719 steps/s |
| `wait_floor_ms`: the virtual clock turns short waits into at least 1–5 ms | game threads polling with scaled timeouts spun at 5,000–11,000 wakeups a second, each a wineserver call |
| Observations parsed in C (`native/w3obs.c`, `GameSetup.native_obs`) | a Python object per unit, and the GIL |
| Training games on a 320×240 virtual screen, drawn in the game's own thread (`render_threads=0`, Wine's csmt off) | llvmpipe 5.3 → 1.7 cores for 16 games; a reload's CPU 2.1 → 1.4 s |
| WineHQ stable instead of staging | staging was ~25% slower in parallel |

A `duelrush` step (0.5 s of game time, built-in AI on both sides) costs 14.6 ms of CPU in a game alone (68 steps/s) and 35–40 ms with 32 games running (two threads a core, the all-core clock, contention); about 8 ms of it is the harness and the sync, ~10% is wineserver (~500 requests a step).

## Starting the next game

| fix | effect |
|---|---|
| `RestartGame` from the harness: the map reloads inside the running game | ~1 s instead of a ~4 s launch |
| "Allow Local Files" off (every file lookup scanned the game folder under Wine) | a reload under load 13.2 → 9.9 s |
| The rules as the game's own tables in the map instead of object data (`w3u`/`w3q`/`w3a`, applied change by change at every load) | a reload 3.0 → 1.0 s on an idle machine |
| Five games per map load, each with two players of its own (`GameSetup.pairs`) | a reload was a quarter of a game thread's time: games 2.1 → 2.55 a second; demonstrations 1.55× faster |

A reset by script for the same players (remove everything, respawn) took 0.1 s but was not a new game: the engine kept counting the removed heroes (see [experiments](experiments.md)).

## Whole-game self-play (`fullgame/selfplay.py`)

32 games in 4 actor processes, one inference server process, one learner. Agent steps per second on `fgself-9`:

| change | steps/s | why |
|---|---|---|
| start | 560 | the inference server 90% busy at 3.4 rows a call |
| a pipe per game thread instead of queues | 545 | a request took ~32 ms of a 59 ms step through feeder threads and locks |
| games niced (+10) | 595 | the server and the learner get the CPU first |
| a round's network calls launched together, one GPU wait per round | 606 | every wait costs a turn of the GPU next to the learner |
| CUDA graphs captured by hand instead of `torch.compile` | 607 | 0.4 s a shape instead of 10–20 s stalls; past snapshots through one network with swapped weights |
| every game reloads the map (the reset fix) | 540 | the agents' games became real games |
| the loading screen drawn in the game's thread, a video every 10 minutes | 644 | |
| the units' pass in C | 719 | |
| minibatches of similar entity counts, statistics kept on the GPU | ≈800 | padding 91 → 49 entities; the update 10.5 → 8.9 s |
| five games per map load | 822 | |
| the trajectory queue bounded to a batch, an inference round at most every 4 ms | ≈840 | the batches 1.3 updates old instead of 3 and growing |

Now the learner (an update of 8,192 steps in ~8.5–9 s, bound by its own Python; the GPU ~36% busy) and the games' CPU are about even: speeding one up alone buys a few percent.

## Micro training

* **PufferLib 5** (`micro_mirror`, 24 games): 620 → 3,000 agent steps/s. OpenMP threads spun at the barrier while the slowest game finished (`OMP_WAIT_POLICY=passive`), one buffer per two games.
* **The torch trainer** (league self-play): 160–180 → ~800 agent steps/s. The league's snapshots shared one compiled network with swapped weights (128 → 5 ms a step); the policy step compiled whole (Gumbel-max sampling, the GRU as plain ops: 10 → 0.6 ms); one pinned upload and one download a step; the PPO update in a thread during the next rollout.

## The dashboard

A run's page re-read and re-binned all its episodes on every refresh: 4.5 s for 135k episodes. Its series are now kept between requests as numpy columns, extended with new episodes only and binned with numpy: ~1 s.

## Where the time goes with 32 whole games (2026-10-03, `fgself-12`)

| part | measured | what it means |
|---|---|---|
| the games | 13.3 cores of 16 (lifetime averages over 68 game processes), wineserver 3.5, Wine's services 1.0 | the machine's limit: a 0.5 s step is 8-12 ms of the game's own thread, mostly the 20 turns of simulation; the harness's JASS (result check, observation) is under 1 ms, drawing ~10% (one frame for ~8 steps) |
| the learner's update | 3.6-3.8 s for 8,192 steps alone (`scripts/bench_update.py`), ~10 s in the run | bound by its own Python: 117-135k kernel launches (2.1 s of CPU), 1.8 s of kernels; in the run the games take the CPU it needs. The cloning loss's extra pass on 64 demonstration steps is a third of it (2.4-2.6 s without). |
| the inference server | ~0.7 cores | a third of its time waiting for the GPU it shares with the learner |
| an actor process (8 games) | ~0.45 cores, the GIL held 40% of the time | features 30%, receiving observations 15%, sending requests 14%, the advisor's labels 9% (most of it one `np.isin` on a few events: now a set) |
| videos | 0.5-3.4 cores while one renders | |

* The learner and the games are balanced (the learner waits ~0.7 s an update for data), so a faster learner alone gives at most the CPU it frees (~1 core, ~5%).
* One autocast region per minibatch and losses without GPU syncs (masked means): the same losses to the digit; no measurable change in seconds per update (3.64 and 4.54 s against 3.79 and 3.81 s: the noise of the demonstration loader).
* bf16 against fp32 on one real batch (CPU): KL between the order distributions 3e-6 (max 8e-4), log-probability differences 7e-4 (max 0.09), the gradient's cosine 1.000.

## The learner as CUDA graphs (2026-10-03, `fgself-12`)

`GraphedUpdate` captures a minibatch's whole step once per shape (entities padded to 32, 48, 64, 96, 128 or 160; the cloning batch alike; minibatches of 16 sequences of 16 steps) and replays it. First the update lost its GPU syncs (masked means, not boolean indexing; one autocast region per minibatch); a capture warms up on a side stream and puts the weights and Adam's state back.

| | eager | graphed |
|---|---|---|
| alone, no cloning loss (dropout off: the same losses to 3-4 digits) | 2.35 s | 1.53 s |
| alone, the run's settings | 3.35 s | ~2.3 s |
| in the run, next to 32 games | ~9.5 s | ~5.0 s |
| the run's steps/s | ~800 | ~950-1,100 |

The steps/s rose more than the learner's freed CPU: the trajectory queue holds one batch, so the actors had waited for the slow updates. Now the learner waits 2-3 s an update for data.

## Games that draw nothing, 40 of them (2026-10-03)

With drawing on, `OPENGL32.dll` (Wine's Direct3D 9 on Mesa's software rasterizer) took 18% of a training game's main thread (`W3SIM_PROFILE=2`). The shim now makes the Direct3D device's draw calls, clears and presents return at once (`W3SIM_DRAW=0`, `GameSetup.draw`, `shim/render.c`); the game still builds every frame. `OPENGL32.dll` fell to 1%.

* Six duelfast games against six alongside: 22.6 → 18.7 ms of CPU a step (game 18.5 → 15.0, wineserver 4.0 → 3.7).
* The simulation does not change: a replay played back with and without drawing agrees with the live game in every unit's position, hit points, mana and order at every step (`scripts/draw_parity.py`, 300 steps).
* Mesa's own no-op driver (`GALLIUM_NOOP=1`) left the game without a first frame.

In the run the freed CPU first went idle (~24%): the games waited for the inference server, which shares the GPU with the graphed learner (busy 73% → 85-90%, rounds 5.1 → 6 ms). More games give it bigger batches:

| games | steps/s (12 updates after the restart) | inference busy | CPU idle |
|---|---|---|---|
| 32 | ~990 | 85% | ~24% |
| 40 | **~1,110** | 87% | ~21% |
| 48 | ~1,010 | 86% | ~16% |

The run stays at 40. The game's main thread without drawing: the game's own code 58% (no page above 3.5%), Wine's system layer 38% (about 15 points of it the wait for Python at each step's sync).

## Where a game's CPU goes without drawing (2026-10-03)

* Over 10 s with 40 games: the games' own code 9.7 cores, their kernel time 2.4, wineserver 2.3 (80% of it in the kernel). Wine's overhead is a third of the games' CPU.
* The main thread spends ~95% of its wall time in `GameUpdate` (the turns; the step sync's wait inside it): with nothing drawn, little is left outside the simulation.
* Wineserver requests (`W3SIM_WINEDEBUG=+server`): ~300 a step, most of them events (`event_op`, `select`, `create_event`, `close_handle`).
* The game's own synchronization calls (`W3SIM_PROFILE=4`, shim/syncstat.c): ~3,800 a second a game, 85% from two call sites, one `SetEvent` (`exe+0x3c2add`) and one `ResetEvent` (`exe+0x3c1e46`) per 25 ms turn: the hand-off between the main thread and the thread that paces the turns. Only ~4.5% of the calls set an event already set or reset one already reset: skipping them saves nothing worth having.
* What would remove most of the requests: Wine's ntsync (events in the kernel, no wineserver round trip), which needs `/dev/ntsync` (Linux 6.14; WSL runs 6.6), or the turn hand-off's events replaced in the shim. Proton's fsync does it on WSL's kernel (below). (Tried on native Linux on 2026-10-11: no gain over fsync, below.)

## Where a step's time goes with 40 games (2026-10-03, `fgself-12` at 47.9M)

~1,010 agent steps/s are ~776 game steps/s (a self-play game makes two agent steps a step): each of the 40 games makes ~19 steps a second, ~51 ms a step.

| a step's wall time (an actor's 10 games, `py-spy`) | share | ms |
|---|---|---|
| waiting for the game's observation | 39% | ~20 |
| waiting for the inference server (rounds of 7-9 ms, 3-4.7 ms of each waiting for the GPU it shares with the learner; 92% busy) | 37% | ~19 |
| the actor's Python (features, parsing, sending: 3.6 ms of CPU; the rest waiting for the GIL its 10 games share) | 24% | ~12 |

The same kind of game with replies at once, under the same load (`BuiltinAI` on both sides): 10-15 ms a step, 8-11 ms of the game's CPU and 1.7-2.2 ms of wineserver's.

| CPU over 30 s (32 threads, 24.6 busy, a video rendering) | cores | ms a game step |
|---|---|---|
| the games' main threads | 8.3 (kernel 0.8) | 10.7 |
| the games' other threads | 3.0 | 3.8 |
| wineserver | 3.3 (kernel 2.75) | 4.3 |
| Wine's services (`winedevice.exe` polling every ~1 ms) | 0.84 | 1.1 |
| the actors (4 processes) | 2.8 | 3.6 |
| the inference server | 1.2 | 1.5 |
| the learner and its data loaders | 1.2 | 1.5 |
| the video renderer (game, ffmpeg) | 2.7 | |

* Kernel time is 8 of the 24.6 busy cores: the scheduler, pipes and wakeups of the games' wineserver round trips, with the kernel's Spectre/SRSO mitigations on every entry and context switch (`srso_alias_safe_ret`, MSR writes).
* While a game waits for its orders, the shim freezes the clock, and the background threads' timed waits become 1-5 ms polls: ~2,550 wakeups a second a game (~130 a step), each a wineserver round trip. With replies at once a step costs half the background and wineserver CPU.
* Nine `winedevice.exe` from a finished collection were still polling after 10 hours (0.14 cores; killed).

## Background threads asleep while the clock is frozen; no gamepad drivers (2026-10-03)

* While the clock is frozen (the step sync), a background thread's timed wait now blocks until the clock runs again or what it waits for is signalled, then times out (`shim/clock.c`, `W3SIM_PARK=0` for the old polling). A frozen clock cannot reach a timeout, so the game sees what it saw before: a wait that timed out.
* Wine's HID, USB and Bluetooth bus drivers are not loaded (`GameSetup.device_drivers`, a DLL override; with their services disabled Plug and Play still loaded them). The HID bus's SDL event loop woke every millisecond in every game.

Four games against four, each waiting 35 ms for every step's orders as in the run (`BuiltinAI` on both sides, 300 steps each):

| | the game | wineserver | Wine's services | CPU a step | wall a step |
|---|---|---|---|---|---|
| before | 11.6-12.2 ms | 3.9-4.3 ms | 0.7 ms | 16.2-17.2 ms | 49-50 ms |
| asleep while frozen, no drivers | 9.8-10.3 ms | 2.4-2.5 ms | 0.0 ms | 12.8-12.9 ms | 49-50 ms |

A replay played back with the threads polling and asleep agrees with the live game at every step (`scripts/draw_parity.py --env W3SIM_PARK=0,1`, 388 steps). The drawing check is now really one: `GameSetup.draw` had overridden the script's environment, so its "not drawing" playback drew; drawing on and off agree over 400 steps.

## Wine's synchronization on futexes: GE-Proton's Wine with fsync (2026-10-03)

GE-Proton 10-34's Wine (10.0 with Proton's patches) with `WINEFSYNC=1` keeps Wine's events, mutexes and waits in shared memory with futexes (`futex_waitv`, in WSL's 6.6 kernel) instead of a wineserver round trip each. Four games against four, each waiting 35 ms for every step's orders, at the same time:

| Wine | the game | wineserver | CPU a step | wall a step |
|---|---|---|---|---|
| WineHQ stable 11.0 (until now) | 10.20 ms | 2.57 ms | 12.77 ms | 49.3 ms |
| GE-Proton 10-34, no fsync | 10.26 ms | 2.57 ms | 12.82 ms | 49.8 ms |
| GE-Proton 10-34, fsync | 8.47 ms | 0.74 ms | **9.21 ms** | **47.6 ms** |

28% less CPU a step, and the game's part of a step 12.6 instead of 14.3 ms. A replay plays back identically with fsync off and on (`scripts/draw_parity.py --wine ... --env WINEFSYNC=0,1`, 400 steps).

* Proton 11 builds need glibc 2.38 (Ubuntu 22.04 has 2.35). 10-34 runs here: `scripts/setup_ge_wine.sh`.
* Outside Steam its Wine needs what the proton script sets: `WINEDLLPATH` with its vkd3d, `LD_LIBRARY_PATH` with its libraries, and vkd3d's DLLs in the prefix (`wined3d.dll` imports them: without them "unable to initialize DirectX"). `runtime.wine` does all three for a Proton build.
* Its Windows user is "steamuser" (the Documents folder: replays).
* `selfplay --wine <files/bin> --fsync 1`: prefixes in a runtime folder of the Wine's own (`~/wc3/runtime-GE-Proton10-34`).

## Actors, and the inference server's transport (2026-10-03/04)

* **8 actor processes of 5 games** instead of 4 of 10 (40 games either way): an actor's games had each waited ~8 ms a step for its GIL (py-spy: 3.6 ms of Python a step, ~12 ms in it). 1,053 → ~1,200 steps/s (10-20 minute averages).
* **A round's networks on CUDA streams of their own** (the current network's; past snapshots in four `PastNet` slots, a network copy each): `scripts/bench_inference.py` gives identical values and entropies; alone a round only went 4.4 → 4.2 ms (1.6 ms of launching, ~3 ms of the GPU), so in the run the wait is mostly for the learner's kernels.
* **View steps and answers through shared memory** (`SharedRows`: a /dev/shm file mapped by every process; a game writes its sides' rows and sends their numbers; the server gathers a call's rows in a few index operations): pickling and the row-by-row copy had been ~40% of the server's time.

| inference server (run averages) | round | of it waiting for the GPU | busy |
|---|---|---|---|
| before (GE Wine, 23:21-23:36) | 12.5 ms | 6.8 ms | 96% |
| CUDA streams (6 minutes) | 9.9 ms | 4.6 ms | 95% |
| shared memory as well (14 minutes) | 7.4 ms | 3.5 ms | 91% |

The run's steps/s for the last two are not measured yet: another project's headless Chrome took ~10 cores in bursts in both windows (the run fell to ~1,000 steps/s while it ran). With it the live run's CPU: wineserver 0.6 cores (3.3 at the start of the evening), the games' other threads 0.8 (3.0), kernel time 3.0 (8.0).

Measured a week later (2026-10-10, 15-minute windows, nothing else on the machine: `scripts/throughput.py`):

| inference server | steps/s | round | of it waiting for the GPU | busy | the learner's update | its wait for data |
|---|---|---|---|---|---|---|
| without streams and shared memory | 1,364 | 12.4 ms | 6.8 ms | 98% | 5.08 s | 0.96 s |
| with them | 1,347 | 9.3 ms | 5.2 ms | 94% | 5.45 s | 0.65 s |

The same steps/s: the inference server was no longer the limit but the learner (an update of 8,192 steps 5.1-5.5 s of a ~6 s cycle). They stay on. Without the cloning loss (`--bc-coef 0`, an experiment on the learning side) an update takes 3.8-3.9 s and the learner waits ~2 s for data: the games and actors set the pace again (~1,350-1,500 steps/s).

## The inference server's calls fused, the round period, bf16 (2026-10-11)

The learner and the inference server are processes of their own on one GPU: on WSL they take turns on it (no MPS, no time-slice setting), so in the run an update took 5.7 s (2.24 s alone) and a round ~11 ms (~4.4 ms alone).

| change (`scripts/bench_inference.py --fuse`, alone; then the run, 15 minutes) | a round alone | of it on the GPU | the run |
|---|---|---|---|
| fp32, as before | 4.86 ms | 2.94 ms | ~1,300-1,340 steps/s, rounds ~11 ms |
| **calls compiled (`torch.compile`: fused kernels) inside their CUDA graphs** (`--infer-fuse`) | **3.77 ms** | **2.05 ms** | **1,431 steps/s**, rounds 7.4 ms, the update 5.1 s |
| bf16 matmuls | 4.49 ms | 2.21 ms | (values off by up to 0.014: not used) |

The fused calls give the same values to 1e-7 and draw new sampling noise at every replay; compiling takes ~1 minute at a start. A round at most every 8 ms instead of 4 (fewer, larger rounds: more GPU for the learner) gave 1,246 steps/s: the update fell 5.7 → 5.3 s, but the games waited twice as long for their orders. The pipeline now keeps the period too (it had launched rounds back to back). The KL term's clone with its weights in bf16: no faster (2.30 against 2.24 s an update).

## The inference server inside the learner's process, a prototype (2026-10-11)

`--infer-in-learner 1`: the server as a thread of the learner's process (its own high-priority streams and random generator registered with its graphs, every graph captured before its loop, thread-local captures on both sides). One GPU context: the learner's kernels and the server's run side by side instead of taking turns.

| 15 minutes | steps/s | the learner's update | its wait for data | a round | of it the GPU |
|---|---|---|---|---|---|
| the server a process of its own (fused calls) | **1,431** | 5.1 s | 0.66 s | 7.4 ms | 1.6 ms |
| the server a thread of the learner's | 1,300 | **3.55 s** | 2.77 s | 12.7 ms | 0.4 ms |

The turn-taking cost the learner 1.5 s an update; in the thread the server's Python waits for the learner's GIL instead (rounds 12.7 ms, the games waiting). Back to its own process; the next step is the server's loop in C++ in the learner's process, without the GIL.

## The inference server's loop in C++ in the learner's process (2026-10-11)

`--infer-native` (`fullgame/native_infer.py`, `fullgame/csrc/infer_loop.cpp`): Python captures every call's graph (fused, the server's own random generator) and the past-snapshot slots, and a C++ thread runs the rounds without the GIL: it reads the games' binary requests (`pack_request`), gathers their shared rows into a call's input, copies a past snapshot's weights into its slot when it changes, replays the graphs (`at::cuda::CUDAGraph::replay` from C++, so the generator's state advances), writes the answers back and replies (`pack_reply`). Python is called only to reload the current network (every 5 s), to load a new past snapshot, and for the statistics. `scripts/check_native_infer.py`: 604 calls equal to direct calls (values, entropies, versions), the sampling redrawn. Only the rows asked for are written and copied (the call's other rows keep earlier valid inputs whose answers nobody reads: ~1 MB less a call).

| 15 minutes | steps/s | the learner's update | its wait for data | rows/s | a round |
|---|---|---|---|---|---|
| the server a process of its own (fused) | 1,431 | 5.1 s | 0.66 s | ~1,780 | 7.4 ms |
| **the loop in C++ in the learner's process** | **1,487** | **3.5 s** | 2.05 s | 1,835 | 6.5 ms |

The learner no longer waits for GPU turns; the loop (97% busy, half of a round waiting for its GPU work) is now the limit. Next: an event-driven loop that reads and launches new requests while earlier rounds are on the GPU.

## The game's own CPU with Ghidra and REA (2026-10-11)

A subagent explored REA (https://github.com/morluto/rea, an MCP server driving Ghidra) to find what the game's main thread spends its time on (report: `~/tools/w3analysis/report.md`, outside the repo). A raw stack sampler in the shim (`W3SIM_PROFILE=5`, in a worktree) and Ghidra exports named the time:

| part of the main thread | share |
|---|---|
| pathing (path search, regions, proximity) | ~37% |
| unit, ability and attack timers | ~17% |
| the built-in AI (JASS and C++) | ~8-10% |
| **our harness**: the per-unit JASS callback feeding `shim/units.c` (9.7%), the rest of `W3S_Step`'s JASS (5.4%) | **~15%** |
| frame paint (with drawing off) | ~6% |
| fog of war | ~2% |

* Hooks that keep the simulation identical (every unit's state over 400 steps, 22 playbacks): a memo of the map-version check behind `GetPlayerId` (a 14 KB copy, 20-40k times a second: -5.6% of the main thread's instructions) and no JPEG decoding of textures created during play (-5.1%); together ~6% less CPU a step. Skipping paint left the simulation identical but made the fog pass grow over a game (to 17% by step 700): not safe as is.
* The harness's own share is the larger target and needs no engine patch: units.c walking the units itself, `GetPlayerId` cached per player, `UnitIsSleeping` only for neutral units, the rest of the step's serialization in C.
* These pay off once the games are the limit again (now the GPU's turns between the learner and the inference server are).

## Native Linux instead of WSL (2026-10-11)

The same machine under native Ubuntu 26.04.1 (kernel 7.0, Python 3.14, torch 2.14 with CUDA 13, GE-Proton 10-34's Wine with fsync as before), `fgself-12` resumed from the same checkpoint with the same recipe and 48 games:

| | steps/s | the learner's update | its wait for data | a round of the inference loop |
|---|---|---|---|---|
| WSL, the last 40 minutes (18:30-19:09) | 1,387 | 4.23 s | 1.69 s | 9.7 ms |
| native Linux (21:25-21:55) | **1,854** | 3.01 s | 1.43 s | 9.9 ms |

A game step costs less CPU: four games against four, each waiting 35 ms for its orders (`scripts/wine_sync_ab.py`), 7.4 ms of CPU a step against 9.2 ms in the same test on WSL. With 48 games, 29% of the CPU was idle and the learner waited for data: 60 games next (10 actors of 6).

| Wine's synchronization (`scripts/wine_sync_ab.py`, 4 games each, 300 steps) | the game | wineserver | CPU a step | wall a step |
|---|---|---|---|---|
| fsync (`WINEFSYNC=1`) | 6.92 ms | 0.49 ms | 7.42 ms | 45.2 ms |
| ntsync (`WINENTSYNC=1`, `/dev/ntsync` after `modprobe ntsync`) | 6.79 ms | 0.58 ms | 7.37 ms | 45.0 ms |

ntsync is no better than fsync: fsync already keeps the waits out of the wineserver. `selfplay --ntsync 1` stays off.

What the move broke, all fixed:

* **CUDA graphs replayed side by side hung the GPU** (100% busy, ~170 W, forever). Graphs captured on one stream keep that stream's cuBLAS workspace, and CUDA 13's cuBLAS picks split-K kernels that count in it: two networks' graphs at once and one waited forever. Every network's graphs are now captured on a stream of their own (`Inference._capture`); `CUBLAS_WORKSPACE_CONFIG=:0:0` also avoids it.
* **Python 3.14 starts processes by forkserver by default**: the cloning loss's data workers (a local function as their init) could not be pickled and the learner died at its first update. They are forked as before (`multiprocessing_context="fork"`).
* **dynamo's limit of 8 compiles a function** (also on WSL): the fused calls past the 8th shape had run unfused. The limit is raised for the fused calls: rounds of `scripts/check_native_infer.py` 4.2 → 3.6 ms, the start's capture ~20-50 s.
* The C++ loop's build finds CUDA 13's runtime (`nvidia/cu13/lib`; the `nvidia-cuda-crt` package gives the headers torch's extension build needs).

## The GPU shared by the learner and the inference loop (2026-10-11/12, native Linux)

With 48 or 60 games, with or without a GPU split, the run stayed at ~1,800-1,850 steps/s, the inference loop at ~2,300 rows/s, ~155 rounds/s. py-spy on an actor (it reads Python 3.14 with sudo): the games' threads waited for answers ~2/3 of their time. The loop's statistics, split by the learner's phase (`lat_/gpu_/launch_/tail_` `_idle`/`_update` in `inference.jsonl`), showed where the time went:

| rounds launched | latency | of it on the GPU | enqueueing the round |
|---|---|---|---|
| the learner waiting for data | 4.5 ms | 1.3 ms | 0.4 ms |
| the learner in its update | 13.5 ms | 2.3 ms | 4.2 ms |

* **Creating and destroying CUDA events takes the CUDA context's lock exclusively** (gdb: the loop blocked in `cuEventCreate`/`cuEventDestroy`): the loop's events are now made once per stream and bank.
* **A copy from pageable memory holds that lock while the CUDA driver stages it** (gdb: the loop blocked in `cuEventRecord` while the learner was in `cuMemcpyHtoDAsync`): the learner's copies to the GPU now go through pinned memory (`selfplay.to_device`). Its update fell 3.0 → 2.4 s, and the GPU became the contention: rounds during the update 39 ms, 16.5 of them GPU work.
* **The learner pauses after each minibatch** (`--update-pause-ms`: the next minibatch made, the GPU waited for, then a few ms for the inference loop alone): rounds during the update ~6 ms.

| 5 minutes each | steps/s | the update | its wait for data | rounds during the update |
|---|---|---|---|---|
| before (events, pageable copies) | ~1,800 | 3.0 s | 1.5 s | 13.5 ms |
| pinned copies | 1,770 | 2.3 s | 2.3 s | 39 ms |
| pauses 15 ms, waiting before making the next minibatch | 1,687 | 4.3 s | 0.5 s | 5 ms |
| pauses 15 ms, the next minibatch made first | 1,902 | 3.8 s | 0.6 s | 5.6 ms |
| **pauses 8 ms** | **2,155** | 3.3 s | 0.5 s | 5.8 ms |
| pauses 4 ms | 2,106 | 3.3 s | 0.6 s | 6.0 ms |

Staleness rose from ~1.3 to ~2 updates (the learner is near its limit again).

Tried and dropped:
* **SMs split between the two (CUDA green contexts, `--gpu-split`)**: 16 or 32 SMs for inference changed nothing while the driver lock was the limit; after it, 24 SMs made rounds during the update worse (42 ms of GPU work, 1,657 steps/s): the two parts take turns on this GPU.
* `CUDA_DEVICE_MAX_CONNECTIONS=32` (streams sharing hardware queues): no change.
* Finer entity buckets for the inference calls (32/48/64/96): 12% less GPU a round alone, no change in the run; kept.

**A bug found on the way**: the loop copied only a row's own entities into a call's input and left the padding as the pinned buffer held it. With recycled pinned memory a stale unit type failed a device-side assert in a fused call and killed the run; the padding is zeroed now.

## The game's own code: equivalent speedups (2026-10-12, `--game-cuts`)

A subagent profiled the game's main thread (Ghidra, a raw stack sampler in the shim, `perf stat -t`) and replaced hot code with faster code that gives the same results: `shim/cuts.c`, one `W3SIM_CUTS` bit each, checked with `scripts/cut_ab.py` (a replay played back once per variant at the same time; every unit, player, event, AI order and command result compared at every step with the live game and between variants).

| bit | what | main thread, M instructions a step (with the others on) |
|---|---|---|
| 16 | the harness's units pass without a JASS thread per unit (`ForGroup` calls the pass directly) | -2.05 |
| 64 | A*: each cell's ring check remembered for one search (90% were repeats; nothing in a search changes a cell's answer) | -1.72 |
| 1 | the map-version check behind `GetPlayerId` memoised | -1.32 |
| 256 | paint at most every 50 steps (drawing off) | -1.0 |
| 1024 | A*'s heuristic computed only when used | -0.87 |
| 32 | `units.c`: pure getters once per unit, player ids by handle | -0.21 |
| 128 | soft-float helpers: clz as `lzcnt`, isqrt as `floor(sqrt(double))` (checked over all 2^32 inputs) | ai -0.82 |
| 8 | textures without pixels (drawing off) | -0.29 |

All on (`--game-cuts 1529`): a self-play replay 28.4 → 19.3 M instructions a step (-32%), main-thread CPU 5.92 → 4.24 ms (-28%). Parity: 21 replays (~22,700 steps: seven race pairs, several games per map load, scripted agents over 20 minutes, eight fgself-12 self-play replays) and, on master's build, three more (the old shim, the new one off, and 1529 alike at every step). Not covered: the 5th game's full reload, drawing on, many hours. Left out: 2048 (a type-check cache: little gain, delicate), 4 (an event pool: nothing), 512 (nothing), 2 (paint off entirely: the fog's list grows).

In the run: **2,276 steps/s** (10:00-10:10, 48 games) against ~2,100; the learner is the limit now (an update 3.1 s, its wait 0.5 s), with a quarter of the CPU idle.

Rejected by the study: remembering paths between searches (moving units change the search's inputs), faster soft-float arithmetic (the cost is the calls), the agent-id lookup (already an array), `UnitIsSleeping` only for neutral units (a charmed creep could differ). Left: the harness's JASS (~11% of the main thread), its native calls (~8%), the node lookup and open list of A* (~5%).

## The game's own code, round 2 (2026-10-24, `--game-cuts 259577`; branch `game-cuts-2`)

A subagent profiled today's self-play games (fgself-139's replays played back with `--game-cuts 1529`, the shim's raw sampler `W3SIM_PROFILE=5`, Ghidra) and replaced more of the main thread's hot code with code that gives the same results. perf's counters were not available (`kernel.perf_event_paranoid` is 4 again after the reboot), so the gains below are CPU time (`/proc/<pid>/task/<tid>/schedstat`), each variant played back at the same time as its base on the loaded machine (pairs of games; a pair's noise ~1-2%).

Where the run's CPU went (2026-10-24 12:05, 56 training games): the games 16.6 cores (52% of the machine), their main threads 91% of that; wineserver 0.5 cores (~3% of the games' CPU, not ~10% any more); the video renderer's game ~2 cores. The main thread with 1529 (raw samples):

| part | short game vs the AI | 15-minute agents' game | 40-minute agents' game |
|---|---|---|---|
| pathing (the system tick, other pathing timers) | 32% | 43% | 54% |
| other agent timers (abilities, harvest, buffs, the AI's C++) | 21% | 13% | 15% |
| **the harness**: units pass (units.c's natives; UnitIsSleeping a third of it) | 11% | 13% | 12% |
| **the harness**: the rest of W3S_Step's JASS, its orders, EnumItemsInRect over the whole map, event triggers | 10% | 12% | 9% |
| attack (target acquisition) | 7% | 8% | 2% |
| fog | 4% | 5% | 5% |
| paint (every 50 steps), turn other, unattributed | 15% | 7% | 3% |

Timing marks in a live game's harness (wall ms a step, loaded machine): the units pass 0.7, W3S_CheckResult 0.17, the K/T/players' records 0.33 (a JASS call, I2S and Preload per token), EnumItemsInRect 0.2-0.37, the commands 0.05-0.16; the simulation between steps 4.6-5.9.

wineserver: ~16 requests per observation: ~7 `get_thread_info` (the game's GetThreadPriority around its timing code, on the main thread and one other) and ~9 for the socket of the step sync (two sends, a recv, their async results and waits).

| bit | what | main thread CPU a step (paired playbacks) |
|---|---|---|
| 4096 | the long-range path search's integer square root (0x99b740: Newton's method, two divisions a round, twice per node relaxed) as floor(sqrt(double)): the executable's own code run natively over all 2^32 inputs gives floor(sqrt) for every one | -1.4% (40-minute game, 2 x 2,900 steps) |
| 8192 | a unit's ability by id (0x6d6570; UnitIsSleeping asks it for 'Asla' for every unit every step, 0.28 ms a step) in C: the list's agent lookups inlined, IsA(type, id) remembered per game on the main thread | -2.5% |
| 16384 | the widget grid's walk (0x8092a0 -> 0x809170 -> 0x809030; EnumItemsInRect walks ~4,000 cells and ~5,000 entries a step) in C, a cell's widgets prefetched (the walk is bound by their cache misses; without the prefetch ~-1%) | -2.7% |
| 32768 | a unit's handle converted to its object once per unit of the pass (the natives' 0x483940 / 0x483ba0, ~30 a unit) | -1.3% |
| 65536 | Wine: an observation and its header in one send; the calling thread's priority remembered | wineserver requests 15.7 -> 7.9 an observation, its CPU 0.22 -> 0.16 ms a step |
| 131072 | Wine: the step sync's TCP connection made and used with Linux system calls (`int 0x80`: the game is a 32-bit Linux process) instead of Winsock | -4.4 to -6% (the client side of each wineserver round trip was ~50 µs); wineserver 0.16 -> 0.03-0.09 ms |
| 262144 | the harness's players' records (W3S_SerPlayers) and its check of the result (W3S_CheckResult: loops over 24 x 24 players, MeleeGetAllyStructureCount for each) made by the shim with the same natives; a harness change (W3S_CSerInit asks for it; melee or decisive victory, no pairs, scenarios or resets by script) | -6.4% (8 new recordings, against the same harness's JASS path) |

All on (`--game-cuts 259577` = 1529 + the six), against 1529, both played back at the same time: the 33 echofast self-play replays of fgself-135/137/138/139 (all race pairs, games against the AI and between agents, 400 to 4,800 steps) **main thread -10.1%, the game's processes and wineserver -12.2%** (steps-weighted: 7.87 -> 7.08 ms and 8.34 -> 7.33 ms a step; per replay -8 to -15%); fgself-139's alone -9.8% and -11.9%; over all 54 replays -9.7% and -12.1% (wineserver 0.19 -> 0.04 ms a step). Parity: 54 replays, 60,772 steps, identical at every step (every unit, player, event, AI order, command result and ground item, and the raw observation bytes, against 1529; the 13 old live recordings also against their live games): the 33 self-play replays above, the 13 duelfast recordings of 2026-10-11 (seven race pairs, insane and normal AI, scripted agents, five games per load) and 8 fgself-12 replays (duelfast, five pairs a load).

Everything with the harness change (`--game-cuts 521721` = 1529 + the seven) against 1529, on 4 of the new recordings: main thread 7.72 -> 6.29 ms a step (**-18.4%**), the game's processes and wineserver -20.7% (agents' game -31%, games between AIs -15 to -18%). The new harness with the old shim takes the JASS path (identical to the shim's path and to the live game); 262144 with an old map does nothing. Tried and not kept: paint every 200 steps instead of 50 (no change).

262144 changes the map (the harness), and a replay plays back only on the map it was recorded on, so the run's replays cannot check it. Instead: 8 live echofast games with the new harness (AI against AI with all races, scripted agents, agents against the AI; won, lost, a tie at the time limit; 10,997 steps) recorded with `W3SIM_COBS_CHECK=1`, where the harness writes its own records and the shim compares its records and results with them at every step (no mismatch), then each replay played back on the JASS path (259577) and on the shim's (521721): identical at every step, raw observation bytes included, and identical to the live games. What the new path leaves out are getters (slot states, player states, alliances, start locations, I2S and Preload): nothing in the simulation reads them.

## The game's own code, round 3: pathing (2026-10-24, branch `game-cuts-3`)

perf's counters were back (`kernel.perf_event_paranoid` = 1), so this round measured instructions a step (identical from run to run: a playback is deterministic) and recorded the main thread with call chains (`perf record --call-graph fp`, the shim built with frame pointers; `scripts/cut_ab.py` takes `CUTAB_PERF_RECORD` and `CUTAB_PERF_EVENTS`).

Where the main thread's cycles went with `--game-cuts 521721`:

| part | short game vs the AI (fgself-139) | fgself-140 game (agents) | 40-minute agents' game |
|---|---|---|---|
| pathing (the system tick 0x8b5220) | 32% | 50% | 56% |
| of it: the local A* search (0x8bf8b0) | 5% | 11% | 17% |
| of it: the long-range search (0x8a4800 -> 0x8a3bf0) | 7% | 18% | 3% |
| of it: the pathing map's update (0x8b7b30) | 6% | 4% | 4% |
| of it: collision / proximity (0x8bbc50) | 3% | 3% | 5% |
| the harness's JASS (units pass ~11%) | 24% | 19% | 22% |
| attack / fog | 7% / 4% | 7% / 4% | 2% / 4% |

Pathing's cost is spread over dozens of functions and much of it waits on memory (IPC 1.2-1.6): the soft float's calls alone are ~25% of pathing (multiply 0x99ffe0, add 0x9a01c0, ...: called from everywhere, no single caller worth rewriting), the cell lookups of the long-range search (0x8a20a0) and the grid's cell lists. Results must stay bit for bit: the open lists are binary heaps whose pop order on equal keys is part of the result, and the searches leave state in the shared widget grid (type-2 node entries, stamps) and node arrays that later searches read.

| bit | what | M instructions a step, the 40-minute game / a fgself-140 game (alone, on 521721) | main-thread CPU (one pair each) |
|---|---|---|---|
| 524288 | the long-range search's relaxation (0x8a3d00) in C: the neighbour's square root after the early return, the heap push inline | -0.5% / -2.1% | -0.7% / -1.7% |
| 1048576 | the local A* search's loop (0x8bf8b0) in C: the open list's pop and push inline (the same binary heap, the same sift steps), the ring of cells and each cell's node remembered for the run in one array, the relaxation with its heuristic after the early return; new nodes still made by the game's code (0x8bcf30 -> 0x8c22c0, 0x8a3480) | -1.7% / -1.3% | -3.6% / 0% |
| 2097152 | (with 128) the soft float's integer square root (0x9a2150) as an inline sqrtsd | -0.2% / -0.1% | 0% / -0.8% |
| 4194304 | the pathing map's cell update (0x8b78b0: four sub-cell queries 0x8be7a0 -> 0x8be420) in C, the same stores in the same order | -1.4% / -1.4% | -1.9% / -2.8% |
| 8388608 | the units pass's per-unit values of the harness's table kept by the shim (the record's hash, "reported dead"; "under construction" mirrored from the table's natives): no hashtable lookups per unit | -0.6% / -0.8% | -2.8% / -4.2% |
| 16777216 | (with 8192) a unit without an ability of the sleep's type remembered while its ability list's head is the same (CUnit::AddAbility 0x6d4560 links every new ability at the head; W3SIM_SLEEPCHECK=1 checks every answer) | -0.2% / -0.2% | -1.8% / -2.6% |
| 33554432 | (with 131072; `GameInstance` also listens on a Unix socket) the step sync over a Unix socket (W3SIM_SOCK) instead of TCP loopback | 0 | -1.8% the game's main thread; the controller thread's sync -6% (0.33 -> 0.31 ms a step) |

Pathing's own instructions (perf samples, 0x8b5220 inclusive): 40.9 -> 38.9 M a step on the 40-minute game (-4.9%), 18.1 -> 16.9 M on a fgself-140 game (-6.8%): A* -9 to -11%, the long-range search -8 to -13%, the map update -22 to -28%.

Everything (`--game-cuts 33551865` against 521721, both played back at the same time): 45 replays, 48,766 steps, identical at every step (raw observation bytes included): main thread **-4.6% instructions, -6.6% CPU** (steps-weighted; fgself-140's 5 replays -4.7% and -7.1%, fgself-139's 22 -4.8% and -6.5%, the 8 new-harness recordings -4.3% and -6.3%, fgself-137's 6 and four old duelfast recordings alike). The Unix socket (33554432, on top): 4 more playbacks identical (two with the full digest and against the live recording).

Tried and dropped: huge pages (the shim's madvise(MADV_HUGEPAGE) and MADV_COLLAPSE through `int 0x80` on the private regions or the whole address space: ~35 MB in huge pages, dTLB misses -4 to -6% (~13,000 a step), no change in CPU: the TLB's misses are at most ~2% of a step); a binary observation format (the shim's text tokens are 0.2% of its main thread, the actor's parse and merge 31 µs an observation, much of it the Observation's objects).

What is left in pathing: the long-range search as a whole (0x8a3bf0 with its ~20 expansion helpers, 0x8a20a0's four-level cell lookups: up to 18% of a fgself-140 game), the soft float's calls (~25% of pathing, spread over many callers), collision and proximity (0x8bbc50, 3-5%). Beside it: the harness's command application in JASS (~3.5%), EnumItemsInRect's whole-map walk (~2%, memory-bound), fog (~4%).

## The game's own code, round 4: the harness's JASS, soft float, the name hash (2026-10-24, branch `game-cuts-4`)

`W3SIM_CUTS` is read as 64 bits. Each bit on top of 67106297, checked with `scripts/cut_ab.py` (every digest and the raw observation bytes at every step).

| bit | what | gain (main thread) |
|---|---|---|
| 67108864 | the soft float's multiply (0x99ffe0) as the exact double product truncated to 24 bits (IEEE single round-toward-zero) when both exponent fields are nonzero and 2^-125 <= \|p\| < 2^128; else the original | -2.5% instructions, CPU 0 |
| 134217728 | float to int (0x9a07b0): 0 below 1, cvttss2si below 2^31 | -0.7% instructions, -0.4% CPU |
| 268435456, 536870912 | floor (0x9a1520): roundss for 1 <= \|x\| < 2^23 (SSE4.1); int to float (0x9a1620): cvtsi2ss for \|v\| <= 2^24 | -1.3% instructions, -0.6% CPU (both) |
| 4294967296 | the name hash (0x433890: upper-case into a 1 KB buffer, then Jenkins' lookup2 0x4347f0) in one pass with a table; the interpreter resolves every variable by name at run time (0x8cdca0), so it is 1-3% of the main thread | -1.9% instructions, -2.4% cycles (four fgself-141/142 replays); -3.2% instructions and -3.1% cycles in a game against the AI |
| 1073741824 | (harness handshake MBOX - 29..31, 39) `W3S_Tok`'s integers as `GetPlayerTechMaxAllowed(the neutral victim player, v)`, written by the shim as the token `Preload(I2S(v))` would have been (the checksum still summed in JASS); the "V", "K" and "T" records made by the shim (the camera's target with the same natives and R2I; the step, the time and the flags from the harness) | -0.6 to -5% CPU (I2S, the string table's insert and release, was 1-2% alone) |
| 2147483648 | (MBOX - 32..38) the agents' commands applied by the shim after `W3S_BuyPending`: `W3S_ApplyOne`'s natives that act or decide, in its order (LoadUnitHandle, GetOwningPlayer, GetPlayerId, the Issue*OrderById natives with I2R'd coordinates, GetHeroSkillPoints and SelectHeroSkill, Player, GetPlayerState, SetPlayerState) for ops 1, 2, 3, 5, 6, 87, 90; a step with any other op (a build: its ring of tries uses the interpreter's soft-float arithmetic), a command cut short, a pending purchase or an advised player is the harness's, which then reads as many integers as `W3S_ApplyOne` may (past the step's count the mailbox answers what the harness's array would hold; zeroed at a load's first step, MBOX - 37); the "C" records written by the shim | 0 to -2% with the 0.5-3 commands a step of training (fgself-141/142: ~1.7, 85-90% of those steps the shim's); -13% with 50 |
| 8589934592 | (with 1073741824; MBOX - 43, 44) `EnumItemsInRect`'s per-item callback (0x49b200) runs the shim's `W3S_SerItem` in place of a JASS thread per item (the same natives: IsItemOwned, IsItemVisible, GetWidgetLife, GetItemX / Y, IsVisibleToPlayer per seer, SaveInteger of the step under key 9, GetItemTypeId, R2I, GetItemType) | -0.6% against the AI, 0 without items |

Parity: 63 replays (fgself-137/139/140/141/142, old recordings; 63,792 steps) with everything but the harness bits' effect (their maps' harness never asks): identical, -6.3% instructions, -1.5% CPU. 17 live recordings with the new harness (one of 5 pairs on duelfast, one recording the AI's orders) (scripted agents, `cut_ab.py record --rich` for every kind of command including pending purchases, unknown units and commands cut short, `--rich=train` for training's proportions): identical with the bits off and on, against the live game, and (6 of them) played back with round 3's shim (master's: the harness's JASS paths). Per recording, all bits: -5.5/-6.2% (train-like agents), -4.3/-5.3%, -3.2/-4.8% (an agent against the AI), -2.3/-3.4% (AI against AI), -14.5/-15.9% (50 commands a step); -11% with the AI's orders recorded (their "I" records' tokens).

The soft float, routine by routine (read in the disassembly, then the executable's own bytes run natively: ~/tools/wt-cuts2-out/sf):

| routine | what it computes | the same as IEEE? |
|---|---|---|
| multiply 0x99ffe0 | sign xor; the exponent fields added; the 24-bit mantissas' exact 48-bit product truncated to 24 bits; result exponent < 1: +0 | **yes**, IEEE single round-toward-zero, when both exponent fields are nonzero and the result's exponent is normal; not for zero or denormal operands (a denormal is taken as 1.m · 2^-127; any zero gives +0, never -0), on underflow (+0, the sign dropped, no denormals), overflow (the exponent wraps into the sign bit: no infinity), or infinities and NaNs (taken as finite numbers of exponent 128). 2^32 random and boundary pairs: 0 mismatches on that domain |
| float to int 0x9a07b0 | truncation by shifts | **yes**, cvttss2si, for \|x\| < 2^31; from 2^31 on the shift wraps mod 32 (no 0x80000000 "indefinite"). All 2^32 inputs |
| floor 0x9a1520 | integer masking; \|x\| < 1 returns one of two constants of its own (0x112d804 for negative x, else 0x112d88c: -0.0 gives the latter); \|x\| >= 2^23 returns x | **yes**, roundss (floor), for 1 <= \|x\| < 2^23. All 2^32 inputs |
| int to float 0x9a1620 | count leading zeros, shift, truncate | **yes**, cvtsi2ss, for \|v\| <= 2^24; above, it truncates (IEEE rounds to nearest). All 2^32 inputs |
| add 0x9a01c0, subtract 0x9a00b0 (the add with b's sign flipped) | the smaller operand aligned with one guard bit by an arithmetic shift (rounding toward minus infinity), the sum's magnitude truncated | **no**: an exponent difference of 23 or more returns the larger operand unchanged (IEEE round-toward-zero changes it by one ulp when the smaller has the other sign), an exponent field of 0 returns the other operand, an exact cancellation gives +0, no overflow handling |
| reciprocal 0x9a1d00, division 0x9a02d0 | a table and linear interpolation; the division is that reciprocal times the multiply | **no** (approximations) |
| square root 0x9a20a0 | an integer square root of the mantissa (the integer part was already replaced, 128/2097152) | **no** |
| 0x9a05f0, 0x9a1da0 | table functions (an arc tangent-like, sine/cosine) | **no** |

The SSE paths use XMM0-1 only, and no MXCSR change (every operation on them is exact); the callers keep nothing in XMM across these calls (the 32-bit ABI; a scan of the ~3,400 call sites, following branches up to the next other call, finds no XMM register read before it is written). They cut instructions but not time: a soft-float call waits on the operands its caller has just stored, and the SSE path's latency is the integer path's (a dependent chain of multiplies: 17.1 against 16.8 cycles a call; independent ones 18.3 against 8.3).

Not done: the long-range search as a whole. At 33551865 it is 11.3% of the 40-minute game's main thread; its self time is the four-level cell lookup 0x8a20a0 4.6% (half of it on one load: the cell of a node far from the last), the relaxation (already C) 3.0%, the open list's pop 1.2%, everything else (the loop, the two expansions and their ~12 edge and corner helpers, node creation) 2.5%. A C rewrite with the same heap order and writes could only remove a part of that 2.5%. Prefetching each expansion's neighbourhood and the next node's (reads only) changed nothing (59.4 against 60.4 M cycles a step, ±1.5). The items' walk: it stamps every widget it passes and advances the grid's counter per cell (~4,000-5,000 a step), state a maintained list would leave behind (harmless until the 32-bit counter wraps, ~860,000 steps into a load): kept. A remembered name hash (by address, checked against the bytes): instructions -5.4%, cycles -0.7% (its table's misses): dropped.

What is left in the harness: ~90 JASS statements a step at ~0.8 µs (W3S_Step's frame, CheckResult's 8, W3S_ReadCommands' 16, the event and removed-unit loops), the commands of other ops, the units pass's ~30 natives a unit (~7-11%), and the items' walk (~2%).

## The game's own code, round 5: the units pass (2026-10-25, branch `game-cuts-5`)

Round 4 left the units pass (`shim/units.c`: every unit's record each step) at 7-13% of the main thread: ~25 natives a unit, each turning the unit's handle back into its object, checking its agent and reading a field or calling one of the engine's functions. `perf record --call-graph fp` (the shim built with frame pointers), each sample of the pass put on the native the pass had called, on a fgself-147 game between agents at `17179866617`: the pass 9% of the main thread, 2.05 M cycles a step, of it `GetWidgetLife` 1.2%, `IsUnitVisible` 1.0%, the push of `ForGroup`'s callback 1.0% (the handle map's hash chain 0x4ebd40: 0.9% alone), `UnitIsSleeping` 0.8%, `GetUnitX` 0.7% and `GetUnitY` 0.3% (the same position computed twice), `GetUnitState` 0.7%, `IsUnitType` 0.5%, each unit's first native (the handle's conversion) 0.2-0.3%. Most of it waits on memory: the agent table's lookups (0x4707f0, 1.2% as a leaf: each unit's life, mana, position, order and ability agents), the handle map, each unit's object.

Six bits, all in `units.c`, none in the harness (the shim's pass serves every harness version, so old replays check them too). Their gains as the pass's own cycles a step (M), each bit added to the ones above it, the variants played back side by side under `perf record` (fgself-147 games: agents, an agent against the AI; and a 40-minute game of scripted agents recorded today, its first 2,500 steps):

| bit | what | agents | against the AI | 40 minutes |
|---|---|---|---|---|
| | the pass at `17179866617` | 2.05 | 1.90 | 2.25 |
| 17179869184 | the enum unit's handle remembered: `ForGroup`'s callback (0x5026a0) pushes each unit's handle on the JASS state's enum stack (0x4ebf00: the handle map's lookup 0x4e72c0 -> 0x4ebd40, the count + 1) and pops it (0x4ebce0: - 1, freed at 0). For a live handle the pair leaves nothing behind (a count reaching 0 frees the handle at once, 0x4e8180 -> 0x4ecb50, which clears its entry's object and drops its map record: the map's only removal, and 0x4e72c0 its only insert outside a saved game's load), so a handle remembered with the object's agent ids is used while the handle table's entry holds the object with a count >= 1 and the stack has room; a unit the pass drops (whose saved handle `FlushChildHashtable` releases) gets its push before the drop, as in the native order | -0.18 | -0.17 | -0.20 |
| 34359738368 | the record from the unit's object: the natives' reads and engine calls made directly on the callback's object, in the record's order (`GetUnitTypeId` +0x34; `IsUnitType` +0x164's bits, +0x408, 0x48a660's constant masks & +0x454; life and mana 0x800d80 / 0x800e20 on +0x1a0 / +0x1c0; the position 0x7fcc50 on +0x26c once for x and y; facing the position's vtable +0x1c and the soft float's multiply; the order 0x7fc7e0 on +0x3a8; visibility the vtable's +0xfc with the seer's player number; resources 0x6929c0; sleeping 0x6d6570 for 'Asla'). Only objects with CUnit's vtable (0xe9c504); the agent checks the natives make are the callback's | -0.41 | -0.35 | -0.56 |
| 68719476736 | a hero's experience, skill points and items read once a record (they were read for the hash and again for the tokens; `UnitItemInSlot`'s second call only finds the handle its first found or made) | +0.06 (load) | -0.05 | -0.01 |
| 274877906944 | (with 34359738368) the owner's handle remembered per player object, checked against the handle table as for the enum handles (and the player's agent, as 0x4e72c0 does) | -0.07 | +0.02 | -0.04 |
| 549755813888 | (with 34359738368) all seers' visibility at once: `IsUnitVisible`'s 0x66d280 asks the game's option 0xd (a TLS lookup) and the unit's fog cell (0x7ffbe0: reads only) for each seer; once a unit now, the seers' detection (0x4758d0) and fog bits (0x4757a0) in their order | -0.07 | -0.01 (one seer) | -0.09 |
| 137438953472 | the pass prefetches the units ahead: `ForGroup`'s loop walks a snapshot of the group (0x4f5b20, hooked to note it), so four units ahead their objects' lines are prefetched, three ahead the agent table's entries they name (and their slots of the remembered handles and of the sleep cache), two ahead those agents and the handle's entry, one ahead the position's box and the objects behind the order's and first ability's agents. Hints only | -0.49 | -0.46 | -0.59 |
| | all six (`1099511625209`) | 0.83 | 0.85 | 0.85 |

The pass as a part of the main thread (the round 4 breakdown's rules, each variant's own run): 9.6% -> 4.3% (agents), 7.5% -> 3.5% (against the AI), 12.9% -> 5.4% (40 minutes, ~230 units). The hero fields, the owner and the visibility batch are 0.1-0.5% each, inside the CPU noise between playbacks; kept for what the profile shows they remove (`UnitItemInSlot`'s inventory lookup halved, `GetOwningPlayer` 0.32% -> 0.11%, `IsUnitVisible` 0.66% -> 0.54% with two seers) at no risk the check modes did not cover.

Parity and the combined gain, on master after audit 11 (`1099511625209` against `17179866617`, round 4's shim; a third game ran all six bits with both check modes; every unit, player, event, AI order, command result and ground item, and the raw observation bytes, at every step): **67 replays, 83,550 steps, identical**, the check game too. Main thread, steps-weighted: **-2.2% instructions, -5.2% CPU** (fgself-146/147/148's 10 replays -2.1% / -4.7%; 10 recordings made today with the current harness, AI against AI, agents, mixed, 40-minute and paired games, rich commands, recorded AI orders: -2.6% / -4.6%; round 4's 17 recordings -2.3% / -5.4%; 30 replays of fgself-137 to -142 -2.0% / -5.5%). The live digests of round 4's recordings differ from every variant alike (round 4's shim as well) at the step a unit left without dying: the "R" records since audit 9. 33 older replays (fgself-137 to -142) did not load in any variant, round 4's shim included (no observation within 300 s on a machine at load 80-100), and were not retried. Before the rebase, 20 of the same replays with all six bits and with each left out in turn: identical, and all six -2.5% instructions, -5.0% CPU.

Check modes: `W3SIM_UOBJ_CHECK=1` reads every unit's fields both ways (the object's and the natives') and compares them, the owner and the visibility bits included; `W3SIM_EH_CHECK=1` makes the push anyway and compares its handle with the remembered one. Over the 100 playbacks of the final suite: at least 17.9 million units' fields and 17.5 million enum handles checked, no mismatch.

Tried and dropped: prefetching five units ahead with the next unit's fog cell (its address needs the cell, so the position agent's box read for real: an unchecked agent faulted at step 157 of a playback; the gain would have been ~0.04 M cycles a step); the objects' prefetches issued after the record instead of before (the pass 9% slower). Not done: a positive cache of the creeps' sleep ability (a creep's sleep ability is the head of its list, so the cache of units without one never helps them; the walk finds it at once, and remembering it would need the agent's liveness checked: the loads it would save).

What is left in the pass (all bits, the agents' game): 0.83 M cycles a step, ~4% of the main thread: `UnitIsSleeping`'s walk for creeps 0.65% of the main thread, the prefetching's own loads 0.6%, the visibility 0.5% (the fog query 0.23%, the cell 0.14%), life and mana 0.33% (the soft float's arithmetic of the regenerating values), the position 0.27%, a hero's ability levels, cooldowns and the harness's ability table ~0.4%, the tokens' formatting 0.16%.

## The game's own code, round 6: the local player's fog display, the grid walk (2026-10-26, branch `worktree-agent-a6b23bb9cea678cff`)

Profiled at round 5's cuts (`1099511625209`) on three of the live run's own games (fgself-153: every agent slot at normal difficulty), each played back once under `perf record --call-graph fp -e cycles` (the shim built with frame pointers), the round 3 breakdown's rules; shares of the main thread:

| part | agents (a1, 717 steps) | an agent against insane (a5, 1,308 steps) | 40 minutes of agents (a6, steps 300-2,500) |
|---|---|---|---|
| pathing, all of it | 46.8% | 45.7% | 43.8% |
| – long-range search / local A* / planning and following | 11.3 / 5.9 / 7.1% | 6.2 / 6.8 / 7.8% | 7.7 / 6.3 / 6.7% |
| – other: events, timers, small passes / the relations timer | 11.4 / 5.1% | 12.1 / 5.6% | 10.0 / 8.0% |
| – collision and proximity / the pathing map's updates | 3.7 / 2.3% | 3.9 / 3.3% | 3.5 / 1.7% |
| unit, ability and buff timers | 12.3% | 11.7% | 11.9% |
| our harness | 15.7% | 11.9% | 17.6% |
| – W3S_Step's JASS, records, commands (of it the agents' orders themselves, Issue*OrderById, ~3.5%) | 7.6% | 4.2% | 7.6% |
| – the units pass / EnumItemsInRect's whole-map walk / the step sync | 4.1 / 2.7 / 0.9% | 4.1 / 2.4 / 0.8% | 5.4 / 3.3 / 0.9% |
| target acquisition | 6.6% | 5.1% | 6.7% |
| fog of war | 5.8% | 5.2% | 6.6% |
| the built-in AI (scripts and C++) | – | 8.9% | – |
| Wine and the kernel / Storm's allocator / paint | 4.4 / 2.4 / 1.6% | 4.7 / 2.2 / 1.2% | 4.1 / 2.3 / 2.2% |

As leaves: the soft float 9.3-10.3% (spread over its callers), the grid walk (16384's C) 3.7-5.4%, the agent table's lookup 0x4707f0 3.2-4.1% (called from everywhere), the long-range search's cell lookup 0x8a20a0 2.6-4.9%, the ability walk (8192's C) 2.3-2.5%, the terrain's 0x733830 1.3-1.9% and the fog's 0x4c5470 0.9-1.2%. Two of these were work nobody needs or work that waited on memory for no reason:

* **The fog's last two calls draw.** The fog pass (0x4c5a80, from the units' timer 0x4ebb00) ends with 0x4c5470, which sets bit 31 of every cell of the fog's map (+0x34) from the local player's vision mask (+0x3c) over +0x30 (what the local player has seen), and 0x706e50, which hands the map with the mask | bit 31 to the terrain's 0x733830: a scan of every terrain vertex that re-tiles the ones whose shroud changed (0x732060: the vertex's display texture and, at random, its neighbours', with the C runtime's `rand()` 0xa7dd2d, never the game's synchronized random numbers). The pair is called together at all six of their call sites (the pass and the fog natives' refreshes). Both are state of one client, built from the local player's vision: a multiplayer game cannot let the simulation read them, and they differ between a live game and its replay's playback (an observer), whose simulations `cut_ab.py` finds identical.
* **The grid walk's prefetch had no lead.** 16384's walk prefetched a cell's widgets and then read them at once: most cells hold one to three entries, so half of the walk's samples sat on the first widget's stamp load, and an eighth on the entries' links. Over the whole map (the harness's `EnumItemsInRect` every step: ~4,000 cells, ~5,000 entries) that was 2.4-3.3% of the main thread, the rest of it target acquisition's smaller queries.

| bit | what | instructions | cycles |
|---|---|---|---|
| 1099511627776 | (drawing off) 0x4c5470 and 0x733830 return at once (each head and the shroud's `ret 0x14` checked before hooking); bit 31 stays as the map was loaded, the terrain's display as it was drawn last | -4.2% | -2.3% |
| 2199023255552 | (with 16384) the walk's entries gathered across cells into a ring 32 ahead of the one processed, each widget prefetched as it is gathered and the first entry of the cell four ahead too; processing is `grid_entry`'s, entry by entry in the walk's order, each cell's stamp counted (`++` +0xb4) at its first entry: the gathering reads only the grid's heads and links, which processing never changes, so the stores and their order are the original's | 0 | -2.6% |
| | both (`4398046508537`) | -4.1% | **-4.2%** (CPU -4.1%) |

Parity and the gain, all four variants played back side by side (`1099511625209`, each bit, both; every unit, player, event, AI order, command result and ground item, and the raw observation bytes, at every step): **29 games, 37,275 measured steps (38,725 compared a variant), identical**: the 8 replays of fgself-153 (two between agents, one of them 40 minutes; six against insane), 8 of fgself-150 to -152, round 5's 10 recordings of the current harness (against their live digests too) and three new live recordings made with both bits on (agents with training's mix of commands; an agent against insane and two insane AIs, with the AI's orders recorded: there the local player is a real slot whose display was skipped live), played back with round 5's cuts and with round 6's: identical to the live game. fgself-153's replays alone: -4.9% instructions, **-4.6% cycles and CPU**; the training replays of fgself-150 to -153 -4.5% / -4.2%; the recordings -3.7% / -4.2%. With both bits the fog of war is 5.8% → 3.3% of a game between agents (6.6% → 3.9% in the 40-minute one) and `EnumItemsInRect` 2.7% → 1.5% (3.3% → 1.4%).

Not usable: `cut_ab.py live` (seeded live games side by side): two games with the same cuts already differed at step 100 on echofast with scripted agents, so the live check was made through recordings instead.

Not done: the agent table's lookup (0x4707f0, 3-4% as a leaf, its callers everywhere: misses on the table), the long-range search's cell lookup (round 4: one miss per node), the proximity events of unit movement (0x806ad0 posts events that Storm's dispatcher hands to ability and order handlers synchronously: 9% inclusive in the 40-minute game, game logic), the fog pass's every-fourth-round display tail (0x7057a0 and 0x706e70: 0.3%), paint (1.2-2.2%: it rebuilds the list the fog pass walks).

## The game's own code, round 7: pathing (2026-10-26, branch `worktree-agent-a581d2110c9018588`)

Profiled at round 6's cuts (`4398046508537`) on two of the live runs' games, each played back under `perf record --call-graph fp -e cycles` (the shim with frame pointers; the round 3 breakdown's rules): fgself-154's first game (an agent against insane on `echofast2`, 2,501 steps) and fgself-153's 40-minute game between agents (steps 300-2,500). Pathing as a whole: 44-48% of the main thread.

| part of pathing | agent vs insane (fgself-154) | 40 minutes of agents (fgself-153) |
|---|---|---|
| the local A* search (1048576's C: of it the memo's and the cells' lists' misses ~2.5%) | 11.0% | 6.5% |
| the long-range search | 6.7% | 7.7% |
| path following's own work (0x8afee0 outside the searches: 0x8a5760 -> 0x8a59c0, 0x8a6ed0, 0x8a5f30, ...; half of it soft float leaves) | 7.5% | 5.9% |
| the proximity events of moving units (0x806ad0 -> 0x7f2b20 -> 0x83ee30: the abilities', the AI's and the orders' handlers) | 4.3% | 5.4% |
| collision and proximity (0x8ac620) | 4.3% | 3.6% |
| the per-agent update's rest (0x8ac110) | 3.6% | 3.4% |
| range sensors (the 'sig^' timers 0x7fae00 -> 0x8b8ae0 -> 0x8bb120): their query of the grid (0x8bb0b0 -> 0x8bbf20 -> 0x8bbe10 per cell -> 0x8bad90, a distance test) | 2.9% | 2.7% |
| range sensors: the enter/leave diff and its events (0x8bb420 -> 0x8bb4a0) | 2.3% | 1.8% |
| the pathing grid's dirty cells' pass (a timer 0x7faec0 -> 0x7f9740 -> 0x8c28c0: a bitmap scan, 0x8c2c20 per set cell drops the searches' node entries and dead widgets' from the cell's list) | 2.3% | 2.1% |
| the timer wheel (0x7fae00 / 0x7f9660), the pathing tick's rest (mostly soft float), 0x8aa740's rest | 1.7 / 1.8 / 1.2% | 2.0 / 1.2 / 0.9% |
| the pathing map's update | 1.3% | 1.8% |

Round 6's "other pathing" (the relations timer, events, small passes, planning and following: 26-28%) is these rows from path following down. The largest single functions by their own time in it: the range query's cell walk 0x8bbe10 (1.7%), the dirty cell's list walk 0x8c2c20 (1.5%) and its bitmap scan 0x8c28c0 (0.8%), the soft float's add and subtract (1.7 / 1.2%), the range test 0x8bad90 (0.5%), the timer heap's pop 0x7f76e0 (0.5%). All of it waits on memory or is game logic (the events' handlers).

| bit | what | its part's cycles a step (agent vs insane / 40 minutes), M |
|---|---|---|
| 4398046511104 | the pathing grid's two walks pipelined, the same stores and calls in the same order: the range sensors' query in C as round 6's walk (entries gathered 24 ahead across cells, two cursors 4 and 8 cells ahead in the walk's order: across the rect's short rows, the next entries and widgets prefetched; the units under the widgets 12 entries ahead; 0x8bad90 and the array's growth the game's); the dirty cells' pass in C (the bitmap's zero words skipped four at a time, the set cells' heads, first entries and widgets prefetched one and two set cells ahead, 0x8c2c20 the game's) | query 0.81 → 0.71 / 0.43 → 0.36; dirty pass 0.62 → 0.62 / 0.36 → 0.29 |
| 8796093022208 | (with 1048576) the local A* search's expansions prefetched: before the ring's walk, the first list entries of the ring's cells and the eight neighbours that the run has not remembered (each head read, the loads overlapping, the entry prefetched), and after a node is made the free list's head entry (0x8c22c0 takes it next) | A* 3.02 → 2.83 / 1.16 → 1.06 |
| | over the suite, main thread, steps-weighted: 4398046511104 alone / both (`17592186041849`) | cycles -1.0% / **-1.4%**, CPU -1.1% / -1.3%, instructions 0 / +0.2% |

Parity and the gain, the three variants played back side by side (`4398046508537`, `8796093019641`, `17592186041849`; every unit, player, event, AI order, command result and ground item, and the raw observation bytes, at every step): **33 games, 43,707 measured steps (45,357 compared a variant), identical**: round 5's 10 recordings and round 6's 3 (against their live digests too), 6 replays of fgself-150 to -152, the 8 of fgself-153 (one of them 40 minutes) and the first 4 of fgself-154 (`echofast2`). By group (both bits): the recordings -1.3% cycles, fgself-150 to -152 -1.4%, fgself-153 -1.7% (the grid walks alone -2.0%), fgself-154 -1.1%. The A* bit adds ~0.4% to the grid walks' ~1.0%: below the round's 1% bar on its own, kept because it is exact and small (`8796093019641` is the grid walks alone).

Tried and dropped (each identical, none faster):

* **The long-range search's loop in C with the next node's cells prefetched** (the open list's top after a pop: its level-0 3x4 cells or its level's 3x3, before the popped node's expansion; also the next candidates' node entries and the popped node's neighbours' nodes). The first-load stall in 0x8a20a0 halved (1.3-1.5% → 0.8% of the main thread), but the search's cycles a step did not move (2.01 → 2.03 M; the loop alone in C 2.14): the stall moves to the relaxation's node loads, and the loop's own work (the pop inline: 0.77% against the game's 0.62%) takes the rest. Round 4 found the same with a simpler prefetch.
* **The A* memo in 4x4 tiles of a cache line** (the shim's own array): +1.9% instructions (the index arithmetic of ~3.8 M lookups a second), the A* search +6% cycles.
* **The A* memo in a 32 KB hash table** (a run's ~500 cells in the core's caches instead of a grid-sized array): the A* search +19% cycles (the probes' branches); its misses were not the array's.

What an approximation could save (not done; each changes the simulation, so a run would have to accept it), as rough shares of the main thread:

| where | now | how | could save |
|---|---|---|---|
| the long-range search | 7-11% | one search for the units of one order with one target (a group now searches once per unit), or recent (start region, goal) paths reused; a coarser grid | 3-6% |
| the local A* search | 6.5-11% | a lower node budget (+0x68), a blocked unit's last local path kept while its blockers have not moved, blocked units replanned every other tick | 3-5% |
| path following's own work | 6-7.5% | settled or idle units steered less often; SSE for its soft-float add and subtract (not IEEE: their rounding differs) | 2-3% |
| range sensors | 4.5-5% | queried every other tick, or only when a unit crossed a cell | 2-3% |
| proximity events of moving units | 4.3-5.4% | posted only on cell changes (their handlers are acquisition and abilities: behaviour) | ~2% |
| the dirty cells' pass | 2-2.3% | every fourth tick (the searches' node entries stay longer, walks skip them; dead widgets go later) | ~1.5% |
| the soft float's add and subtract everywhere | ~4.4% as leaves | SSE (round toward zero is not their rounding) | ≤1% (the multiply's SSE gave -2.5% instructions and no CPU: the callers wait on memory) |
| outside pathing: target acquisition, the units' timers | 5-7%, ~12% | idle or far units' acquisition less often | ~3% |

## The learner's intake (2026-10-12)

With 8 evaluation games, the last-order features and the opponent-aware value, the run fell to ~1,550-1,600 steps/s and the learner's update to ~4.4-4.7 s (3.3 s in the morning). py-spy on the learner's main thread (20 s): ~33% receiving and unpickling the actors' trajectories, of which the answers' arrays were `np.memmap` subclasses (`astype` keeps the subclass of the shared rows; unpickling them rebuilds memmaps); ~18% waiting for the cloning loss's batches from its 2 data workers; ~13% in the update's pauses for the inference loop (by design); and before that, the last-order features' per-unit loop ran in those workers even when `--features 50` zeroed it (an update alone 3.2 → 2.7 s once skipped).

| `fgself-21` | steps/s | the update | its wait for data |
|---|---|---|---|
| before (20:28-20:32) | ~1,600 | 4.4 s | 0.65 s |
| plain arrays from the shared rows, 4 cloning workers (20:48-21:00) | **2,001** | 3.48 s | 0.63 s |

## Recipe changes that changed the speed (2026-10-14)

The policy came to beat the built-in AI, so its teachers went. Each was a recipe experiment ([experiment log](experiment-log.md)), and each made the run faster:

| change | steps/s | the learner's update |
|---|---|---|
| before (`fgself-47`, 8 evaluation games) | ~1,870 | 3.7 s |
| no advice from the built-in AI (`--opd-share 0`: no shadow AI in the games, no labels) | ~2,030 | 3.4 s |
| no cloning loss (`--bc-coef 0`: no cloning batches in the updates) | **~2,220** | 2.4 s |

The learner then waited ~1.3 s for each batch: the games set the speed again.

## The real game on Echo Isles (2026-10-15/17)

| setting | steps/s | the learner's update |
|---|---|---|
| `echofast` (duelfast's rules on Echo Isles, 352 point bins, 5 layers) | ~1,600 | 3.8 s |
| the real rules (`echo100`, games up to 40 minutes) | ~1,500-1,650 | 3.9-4.1 s |
| the cloning loss on the AI's Echo Isles games (×0.02; tried and dropped) | ~1,220 | 6.2 s |
| the cloning loss on the AI's undead games only (`fgself-70`, dropped) | 16% fewer | |
| `fgself-75` (2026-10-17 10:01-10:31: 8 actors of 6 games, 8 evaluation games) | 1,610 | 4.0 s, waiting 1.1 s for data, staleness 1.5 |

Echo Isles has more units in view than the duel maps (creeps, expansions, larger armies), so a step costs the actors and the learner more. ~3,400 games an hour (2026-10-17), ~15 game minutes each on average: the wins are short (~10 minutes), the losses long (~26).

## The afternoon of 2026-10-24: the learner compiled, the actors, the bottleneck view

* **The update compiled** (`--update-fuse default`): the update's forward passes and losses compiled by torch.compile (inductor: fused kernels) before the minibatch step's CUDA graph is captured, as the inference loop's calls already were: an update of 8,192 steps alone 1.38 → 0.64 s (`scripts/bench_update.py --run`), in the run ~2.8 → ~1.5 s. Each minibatch shape (entity width, labels or not) compiles once, 25-45 s the first time; the caches are kept in `~/.cache/torchinductor` (under `/tmp`, a RAM disk here, they were lost at every reboot).
* **The noise scale's spikes**: its measurement (an extra pass over the batch) ran eager every 10 updates, ~4 s; now every 50, through captured graphs of the compiled pass.
* **The bottleneck view** (the System tab; `fullgame/sysstat.py` samples the machine every 30 s into `system.jsonl`): the CPU by part of the run per process (ended threads counted, new processes from their start), a game step's wall time (simulating, waiting for its orders), the learner's updates one by one. It found the video renderer at ~2.2 cores and the relaunches' churn at ~2.2 (4 games a launch: ~1.1).
* **The actors' per-step Python**, a second round (`scripts/bench_actor.py` replays `play_one` on recorded games: the same commands, chunks and episode rows): 5.88 → 3.87 M instructions a side-step; in the run the actors' CPU a step 3.36 → 2.67 ms and ~2,010 → ~2,170 steps/s.
* **12 actors of 6 games** instead of 8 of 9 (each actor had been near its GIL's one core): level, kept.

| `fgself-140` → `fgself-141` | steps/s | games' CPU a step | actors' CPU a step |
|---|---|---|---|
| `fgself-140`, round 2 of the game's code, 15:20-16:00 | 1,786 | 9.20 ms | 3.50 ms |
| `fgself-141`, round 3, 16:12-16:31 | 1,936 | 8.83 ms | 3.35 ms |
| 12 actors of 6, 16:37-16:52 | ~2,010 | 9.5 ms | 3.36 ms |
| the actors' Python, 17:27-17:45 | **~2,170** | 9.4 ms | **2.67 ms** |

## The actors' per-step Python, rounds 3 and 4 (2026-10-24/25)

Round 3 (2026-10-24 evening, merged then) took the old recordings from 3.87 to 3.17 M instructions a side-step: material from a cached table of the types seen, the strategy's counts once per strategy, the view's masks shared, the policy side's builders and miners by masks, one copy of a side's living rows for the modes and the teacher, the parser's tables by memmove, Python's numbers before arithmetic.

Round 4 (2026-10-25, branch `worktree-agent-ac6ba6c870d34f712`; two sub-agents' branches merged into it, then master). The old recordings no longer replay (a cancelled frame now leaves the unit table), so 12 new ones of `fgself-148` (`runs/bench/actor4-*.pkl`, references `ref4-*.pkl` made with the code before): all four races, sides with and without a target, a night elf mirror (`nn`), a 4,800-step game to the time limit (`uol`, made with `record --tax 0.05`: the bench's untaxed self-play games ended in 5-12 minutes), and two games against the built-in insane AI (`hai`, `uai`: `record --ai insane`; such games are ~46% of the run's training steps). Master's code gives 0 differences against the references (its `--macros-off` is off in these games).

Where the time went: `bench_actor.py replay --functions` reads a perf_event_open counter around chosen functions (perf's call graphs lost numpy's share: numpy is built without frame pointers). On these ~100-400-row tables a numpy call costs ~3-15k instructions (a mask `fl & X == 0` ~16k, `rows[mask]` 20-50k), an import statement inside a function ~5k, `arr.ctypes.data` ~10k (a Python object each time; `ctypes.addressof(c_char.from_buffer(arr))` ~3k), and a ctypes call ~2.3k plus ~1.2k for each argument it converts.

| recording | before (master): M instructions, µs CPU a side-step | after | |
|---|---|---|---|
| hu (human, undead: both with targets) | 3.044, 970 | 1.123, 399 | -63% |
| on (orc, night elf: no targets) | 2.450, 821 | 0.884, 298 | -64% |
| nh (night elf, human) | 3.485, 1,237 | 1.260, 446 | -64% |
| uo (undead, orc) | 3.300, 1,090 | 1.155, 375 | -65% |
| ho (human, orc) | 3.005, 989 | 1.103, 387 | -63% |
| un (undead, night elf: no targets) | 2.471, 795 | 0.886, 320 | -64% |
| no (night elf, orc) | 3.368, 1,048 | 1.117, 405 | -67% |
| on2 (orc with a target, night elf without) | 2.743, 936 | 0.967, 337 | -65% |
| nn (night elf mirror) | 3.001, 914 | 1.074, 371 | -64% |
| hai (human against the insane AI) | 3.659, 1,122 | 1.414, 487 | -61% |
| uai (undead against the insane AI) | 3.434, 1,079 | 1.334, 468 | -61% |
| uol (undead, orc with a target: 4,800 steps) | 3.401, 1,134 | 1.274, 441 | -63% |
| **all 12 (28,130 side-steps)** | **3.175, 1,038** | **1.161, 403** | **-63%, -61% CPU** |

(CPU: the best of 3 replays, the two trees replayed side by side on the loaded machine.) By part on `uo`, k instructions a side-step: View.step 1,289 → 458, the teacher 412 → 51, the target builder 364 → 194, the modes 262 → 23, the policy's commands 234 → 108, the trajectory with its chunks 132 → 60, the parser 135 → 45 (a step's: per side-step half), the tech mask 107 → 50, material 93 → 10, Production 71 → 22, mismatch 42 → 7.

What changed, all exact:

* **View.step in two C calls** (`native/w3feat.c`, `featnative.side`, `view_step`): the side's masks of its rows (buildings, units, dead heroes, heroes, finished buildings, army, the mix's types, heroes with a skill point, frames), then the entities with the hero block, what each building makes, the hit points' change and its sorted ids, the zeroed columns (the positions' sines and cosines not made when `--features` zeroes them), the privileged block, the own workers on a building order and the frames for the policy's commands. Arguments in one array kept on the view. The affordable mask in C. macros' natural expansion (`natural`, `site`, `camp`) in C, a tie in distance left to numpy's argsort. The view forgets gone units' entries by the units gone since its last step (`View.added` keeps the invariant that makes it exact). The tech mask's needs resolved once per item.
* **The teacher and the modes' tracker** (a sub-agent; `native/w3rules.c`, `rulesnative.py`): a step's facts in C, one argument array per teacher or tracker (the teacher's heroes, tavern and merchant, finished buildings, army, altars and the worker nearest the tavern in one pass; its expansion lesson's natural and what stands around it, two mines as near left to numpy's argsort; the tracker's counts, measured only for its undecided modes).
* **The target builder**: on the step's rows and featnative.side's lists (no Unit objects: light tuples with Unit's names and values), the owners' map once per observation, its frames and makers by type ids, its imports at the top.
* **The policy's commands**: bins_world in Python's floats (numpy's calls on numbers: ~80k an order), the gold harvests' mines from the step's candidates once, the nearest tree's table made once a game with the fallen trees marked, `macros._makers` once (a mix order had made it: ~400k), the auto skills' heroes and night elves' entangle from C, `accepted()` at once without orders.
* **Bookkeeping** (a sub-agent): the parser's tables in one C export with a one-pass tokenizer, material, Production and unit_rows in C (material's sums as numpy's pairwise summation), the trajectory's arrays converted a chunk at a time when it is pickled, the merge and the table's copy in one call.
* **strategy.mismatch and mix_gap** keep their last answer while `made` / `trained` are equal and the time lies between the same two thresholds.
* **SharedRows.answer** (the live actors' read of the inference server's answers; outside the bench): 46.5k → 31.1k instructions an answer.

Parity: 0 differences (every step's commands, every chunk the learner gets, the episode row) on all 12 recordings after each change and after the merges; again on all 12 replayed with the live run's recipe (`replay --settings fgself-151`: `--teach 3`, `--macros-off mix,creep,expand`; its own references `ref5-*.pkl` from master's code, which differ from `ref4-*` in every chunk), 3.193 → 1.187 M instructions and 1,048 → 411 µs a side-step there; the numpy fallback (`WARCRAFTSIM_PY_FEATURES=1`; trace's numpy path) too; `tests/unit/test_viewnative.py` compares the new C paths with the code they replaced on random steps, an evaluation game's view with every feature kept (the hit points' change is zeroed by the run's `--features 29`, so the recordings do not check it).

Not done (not exact, or not measured): a view's dicts' cleanup without the `View.added` bookkeeping; reusing view_step's output buffers between steps (the trajectory now keeps the step's arrays until its chunk is pickled); SharedRows.write in C (~26k a side-step, live only); a CPython extension instead of ctypes (the arguments' conversions are now a large share of the C calls' cost); numpy's per-array cost in the chunks' concatenation.

## Dead ends

* 8 actor processes of 4 games instead of 4 of 8 (627 against 644 steps/s).
* A CUDA-graph inference thread inside the learner process (capturing fails while another thread draws random numbers).
* Pinning the learner and the server to cores of their own (no faster; the games lost 4 threads).
* Emptying melee's preload lists (the shim's `Preload` hook already drops them).
* A slower clock while reloading (the loading screen waits ~2 s of game time drawing frames: slower costs more).
* Wine's csmt off alone; esync; clock speeds far above what the game reaches (more wineserver wakeups).
* 36 or 48 micro games instead of 24 (1,871 and 1,751 steps/s against 1,830).
* Slower waits for the game's background threads (2026-10-03, shim `W3SIM_BG_SPEED`, `W3SIM_SLOW_WAITS`): four threads wake ~1,300 times a second a game (the shim divides their timeouts by the clock's speed). One of them paces the turns (call site `exe+0x3c2ede`, ~720 wakeups a second, mostly polls): waiting in real time, a game took 4x as long. Capping the other three (`0x352ce`, `0x3c2f8f`, `0x3b80f1`) cut wineserver from 4.2 to 3.6 ms a step and the whole game's CPU by ~3%, with the games no faster. Off.

## What is left

* **Longer steps on longer games.** Half the step's CPU is per step (harness, sync, Python), half is simulation; 1 s steps would give ~1.4× the game time per CPU, at the cost of half the decisions.
* **The learner**: the anchor's forward for the KL term (~5% of an update), a compiled update. (The cloning loss is off since 2026-10-14.)
* **The video renderer**: 3.4 cores while it renders.
* **wineserver**: ~3% of a game's CPU (2026-10-24); with 131072 the step sync makes no requests.
* **The games' main thread after `17592186041849`** (round 7; round 6's numbers): pathing ~45% (broken down in round 7: the searches, path following, sensors and the proximity events, nearly all of it memory-bound or game logic; round 7's table of approximations says what each could save) (the long-range search waits on a cell's cache miss per node it expands; the rest of it, the local search and planning, spread over many small functions and the soft float's add and subtract, which are not IEEE), the unit, ability and buff timers ~12%, target acquisition ~7%, fog ~3.5%, the units pass ~4-5.5% (creeps' sleep-ability walks, the prefetching's own loads, the fog query, the soft float of life, mana and position), EnumItemsInRect's walk ~1.5% (it stamps every widget), ~90 JASS statements a step of W3S_Step and the agents' orders themselves (~3.5%: the game's order code).

## Approximations for training games (2026-10-26, `shim/approx.c`)

Round 7's candidates measured (the whole account in [speeding up the game itself](../game-speedups.md#approximations-disabled)): counters in the shim (`W3SIM_APPROX_STATS=1`) put the relation timers at 12-15% of the main thread, the local searches at 7-13% (stuck units' budget-exhausting A* runs 2-5%), the long-range searches at 9-10% (repeats of a goal ~1.2%). Three opt-in bits, all off: 17592186044416 and 70368744177664 (relation timers of 1 s and more, and of 0.25-1 s, working on every other firing: -9.8% and -6.6% instructions in paired windows, but the hand-written player's gold a minute -48% and -29%: their periods are the owners' predictions of the next event, a worker at the mine waits one more) and 35184372088832 (after a search out of nodes a unit waits 20 pathing ticks: the game unchanged over 48 games, no speed measurable, ≤2.5% by the counters). Doubling the timers' periods where they are made changed nothing (the owners re-arm them). Evaluation games drop bits from 2⁴⁴ on (`runtime.instance.exact_cuts`).

## Measuring

* `py-spy record --threads --idle` on a process (it pauses the process while sampling: an update measured during a profile is slower).
* `perf record -a` (`/usr/lib/linux-tools/5.15.0-194-generic/perf`; counters need `kernel.perf_event_paranoid` <= 2, Ubuntu's default 4 refuses them: `scripts/cut_ab.py` then reports CPU time from schedstat).
* `scripts/bench_env.py` (env steps/s, CPU per process and thread kind), `scripts/bench_train.py`, `W3SIM_PROFILE=1..3` (the shim's per-second profile in each game's `shim.log`).
* `scripts/bench_actor.py`: `record` (a game of a run's policy; `--ai`, `--tax`), `replay` with `--compare` (the actors' outputs against a reference), `--instructions` (perf stat over the warm replays) and `--functions` (inclusive instructions of chosen functions).
* `runs/<run>/inference.jsonl` (the server's rounds, rows, waits); `scripts/selfplay_status.py <run>`.
* `scripts/production_probe.py` and `scripts/order_probability.py`: what a policy does at its buildings, and how likely it is to give the built-in AI's orders.
* Benchmarks with many games must share one measurement window (games launched at different times never overlap).

## Training games on a second machine (2026-10-30)

The games box (MS-A2, 8945HX, capped at 4.0 GHz) plays training games for the learner over a direct 2.5 GbE cable (`--remote-games`, `fullgame/remoteactor.py`). Measured over ~5 minutes each, on the same run; 72 games stay on the main machine, evaluation games on the laptop.

| on the box | steps/s | the box's side-steps/s | a box game's step | the box's memory | what held it back |
|---|---|---|---|---|---|
| none | 3,280 | | | | the main machine's CPU |
| 42 games | 4,030-4,240 | ~1,250 | 33-36 ms | 22 of 28 GB | the box's memory |
| 42 games, same-page merging (`--ksm 1`) | as above | as above | 35.6 ms | 13-17 GB, ksmd 1-2% of a core | |
| 64 games, one gateway | 4,210 | 1,440 | 42.8 ms | | the gateway's GIL, ~95% of a core |
| 64 games, 4 gateways (`--remote-gateways 4`) | 4,450 | 1,800 | 33.0 ms | full: 2.2 GB available, 3.1 GB of swap, ~5 GB merged, ksmd 5% | the box's memory, the GPU |
| 56 games, 4 gateways, merging's scan 2.5 times as fast (2,500 pages each 20 ms) | **4,625** | 1,660 | 31.9 ms | 19.7 GB, ~9.6 GB merged, ksmd 8%, no swap | **the GPU** |

With the box's games at the pace of the main machine's, the RTX 3090 is the limit: the inference server is busy 99% of the time, 55% of its rounds overlap a learner update (latency 8.3 ms, 17.9 ms in an update), and the update's ~35% of the GPU and inference's ~65% fill it. The box's CPU is still ~50% idle. Steps a second rose 41%, with 78% more games. More games won't help until a step costs the GPU less: 66% of inference calls are past snapshots', each a small batch of its own.

The learner's pause after each minibatch, with 56 games on the box, four minutes' mean each (one reading each, so a few percent of noise): 2 ms 4,330 steps/s, **4 ms 4,625**, 8 ms 4,320. It stays at 4 ms.

Where the GPU goes next: the current network's calls carry ~3,600 rows a second in ~190 calls, past snapshots' ~1,000 rows in ~330 calls (about 3 rows a call). So a fifth of the rows take about two thirds of the calls, at about 1 ms of GPU a call whatever its size. Two ways to spend less: serve a past network's calls less often than every round, so they batch (games against past snapshots wait a little, and they are ~20% of the launches), or answer past snapshots on the box's idle CPU for the box's games (weights sent there, more work).

### The GPU's next steps, opt-in (2026-10-30, not yet measured on the GPU)

Written beside a live run, on the CPU; `scripts/bench_gpu_round.py` measures them in one sitting with the GPU to itself. All off by default.

* **`--update-grad-fuse 1`** (with `--update-fuse`): the gradients inside the compiled function (`torch.func.grad_and_value` over the detached weights). **Today's backward is already compiled**: `loss.grad_fn` is AOTAutograd's `CompiledFunctionBackward` and the only other autograd nodes are the 100 `AccumulateGrad`s (the compile's first `backward()` takes seconds), so this only joins the two compiled halves into one program; on the CPU, the real network, 0.33 s a call against today's 0.30 s. dynamo traces it whole (one graph, no break, 1,635 captured calls against 982), inductor compiles it (47 s on the CPU), its gradients within 1e-7 of an eager `backward()`. Weights the loss doesn't reach stay `None`, not `torch.func`'s zeros: Adam would move them by their momentum. Under bf16 the two differ by rounding (0.3% on the mean, bf16 against fp32 0.5-0.9%). The capture in a CUDA graph is unverified.
* **`--adam-fused 1`**: `Adam(fused=True, capturable=True)`. From the headers in the venv (`ATen/native/cuda/fused_adam_utils.cuh`, `MultiTensorApply.cuh`) a captured graph can replay it: the step count is read on the GPU, the addresses go to the kernel by value, `_foreach_add_` advances the step inside the graph, and `lr` and the betas are numbers fixed at the capture, as in the foreach path. `load_optimizer` keeps this run's setting (a saved state's group would have turned it off without a word).
* **`--update-fuse max-autotune-no-cudagraphs`**: passes through as it is (`torch.compile` knows the mode). The tuning happens in `_capture`'s warm-up, on the first run of each kernel: expect minutes a shape the first time, noisy next to a live inference loop; its results stay in the inductor's cache, so tune in the bench, alone, first.
* **`--update-widths`**, and the existing `--infer-entities`, `--infer-past-entities`. What finer buckets save, from the entity counts of 60 Echo Isles games scaled to the run's 41 a row (no run logs its rows' counts; the simulation's mean widest of a call, 57-59, is a little above the run's 53, so these are a little high):

| | today | finer | padded entities | rows x E | rows x E² |
|---|---|---|---|---|---|
| the current network's calls (~19 rows) | 32,48,64,96 | 32,40,48,56,64,80,96 | 88.5 → 77.6 | -12% | -24% |
| | | with 72 too, and 128 | 74.1 | -16% | -31% |
| past snapshots' calls (~3 rows) | 32,48,64,96 | 32,40,48,56,64,80,96 | 63.0 → 57.7 | -9% | -18% |
| the update's minibatches | 32,48,64,96,128,160 | 32,40,48,56,64,80,96,128 | 55.0 → 51.8 | -6% | -11% |

  76% of the current network's calls pad to 96 for a widest row of ~70: 72 and 80 are where its savings are. The cost: the graphs (at the start, 60 s today) go 55 → 88 with one architecture's slots (per entity count, 3 for the current network, 2 x `--infer-past-slots` for past ones), each with a private pool: at most ~1 GB per new count for the current network, ~0.6 GB for past ones, adding every allocation of an eager call, so ≤ ~5 GB and probably a third of it, next to the run's ~13 of 24 GB. The update's graphs share one pool (no more than their widest needs, plus ~13 MB of gradients a graph) and compile lazily, ~1 minute each, cached. Calls' rows also pad, to 8, 24 or 64: a current call of 25 rows pays for 64, likely more than the entities' padding (`calls_by_rows` will tell).
* **`--infer-past-period-ms P`** (`csrc/past_hold.h`, as `PastHold`; the Python server's path alike): each past network's calls at most once every P ms from its last, so a row after a quiet spell goes at once, a full call (16) too, and no row waits longer than P (plus the round's pace). A request's current row goes in its own round; the reply waits for the request's last item, answered when the round holding it finishes. It binds only above a network's own spacing: fgself-169's ~530 past calls/s over ~9 networks is one per ~17 ms each, so P of 33 to 50 ms, not 8. Past-opponent and evaluation games' steps slow by about P/2; games block on their answers, so only the mix of steps changes, a little towards other games. `inference.jsonl` gains `past_rows_per_s`, `past_rows_per_call`, `past_held_ms`, `past_swaps_per_s`, `calls_by_entities` and `calls_by_rows` (8 a bin; current, past).

### The GPU, measured (2026-10-30, `scripts/bench_gpu_round.py`, the GPU alone)

An update of 8,192 steps, seconds after its first: baseline 0.623; forward and backward merged in one compiled program (`--update-grad-fuse`, the backward was compiled already) 0.637, +2.3%, its first update 128 s; `--adam-fused 1` 0.608, -2.4%; finer `--update-widths 32,40,48,56,64,80,96,128` 0.589, -5.5%; **both 0.571, -8.2%, kept**, the losses within 0.02 of the baseline's; `--update-fuse max-autotune-no-cudagraphs` not finished in 10 minutes, dropped. In the run an update fell from ~1.3-1.4 s to ~1.2 s.

An inference round with finer buckets: 5.27 to 5.28 ms (GPU 4.26 to 4.15), no gain, not kept. Padding itself changes nothing: on the CPU with a real checkpoint, values and entropies were identical at widths 64, 80 and 96, with zeros or junk in the padded places.

`--infer-past-period-ms`, past snapshots' calls batched, in the run over four minutes each: off 4,480 steps/s, 576 calls a second, 4.12 ms of GPU a round, a step of a game against a past snapshot 33 ms; 33 ms 4,450, 376 calls, 3.66 ms, 42 ms; 50 ms 4,340, 342 calls, 3.83 ms, 56 ms. The GPU it frees is lost again in those games' wait, and it would change the training's mix: off.

With those two kept, 64 games on the box again: 4,620 steps/s against 4,480 at 56 (within noise), the box down to 1.8 GB available and ksmd at 23% of a core: back to 56. The limit stays the GPU: the server busy 99% of the time, half its rounds beside an update. What's left is a smaller network, or a second GPU.

### The GPU at 350 W, for the fan's noise (2026-10-31, in a live run, kept)

The GPU's fan is the machine's loudest. 350 W against 400 W in fgself-170: 4,099 steps/s against 4,272 (-4%, 9 checkpoints against 45), 67 °C against 70, 1,720 MHz against 1,860, fan 91-94% against 100%. The fan's own curve runs it near full even at 67 °C, well below the card's 83 °C target, so the fan does not slow much, yet the difference can be heard: kept at 350 W, on startup too (`nvidia-power-limit.service`). Not tried: setting the fan's speed directly and letting the GPU run warmer.

### An undervolt (2026-11-02, the user's question; not kept, it crashed a run)

NVIDIA's Linux driver has no voltage control. What it has is a clock offset that shifts the voltage/frequency curve (NVML's `nvmlDeviceSetGpcClkVfOffset`), together with a locked top clock (`nvmlDeviceSetGpuLockedClocks`), so the card holds a clock at the voltage it used to need for a lower one. `scripts/gpu_undervolt.py` does this (`status`, `apply`, `reset`, and `test`, which runs a bf16 GEMM load with an fp32 GEMM checked bit for bit against its first result every ~0.5 s, and watches the kernel log for Xid lines).

The test, the GPU alone at 350 W, the top clock locked at 1,725 MHz (training's clock at the limit):

| Load | Stock | +100 | +150 | +200 | +250 |
|---|---|---|---|---|---|
| bf16 8192³ (power-limited) | 112.2 TFLOPS, 1,412 MHz | 115.1, 1,448 | 117.8, 1,486 | 119.8, 1,514 | |
| bf16 65536×256×1024 (power-limited) | 85.6, 1,432 MHz | 87.6, 1,475 | 89.7, 1,504 | 91.3, 1,534 | 92.6, 1,558 |
| bf16 16384×256×256 (reaches 1,725) | 61.0, 1,646 MHz, 345 W | | 62.9, 1,725, 345 W | 63.0, 1,725, 338 W | 62.9, 1,725, 327 W |

No mismatches and no Xid lines up to +250. Under a heavy load the offset buys speed at the same 350 W (+7% at +200); at training's clock it saves little power (-7 W at +200, -18 W at +250).

Then `fgself-178` at 1,725 MHz +200: the learner died within the minute with `CUDA error: unspecified launch failure`, and the kernel logged `NVRM: Xid 109 ... CTX SWITCH TIMEOUT` (12:26:44). Training mixes the learner's and the inference processes' contexts and CUDA graphs, which the GEMM test does not. Whether the offset itself or changing the clocks under live contexts caused it is not known. The card went back to stock (`reset`), ran a GEMM correctly, and the run resumed from its checkpoint at 328,833 steps (~3 minutes lost). For a few percent of power, not worth another crash: **the GPU stays at stock with the 350 W limit**.

### Why a game step's wall time rises in a run (2026-11-01, the user's question)

It is not a leak: the game's own part stays ~6.4 ms a step and only the wait for orders rises, and a run's rise levels off. `fgself-170` over 6 hours: 35 ms in its first half hour, ~38 from its second hour, up to 39 late; inference rounds 17 ms to ~20-21 ms; entities flat at ~40; training steps/s flat at ~4,400 (+-3%). The GPU is at 100% throughout, so a game step waits for its share of a saturated GPU, and the wait follows how much it has to serve. What grows is the past snapshots' traffic: 714 rows/s to ~900, in ~550 calls/s of ~700 (1.6 rows a call). `fgself-168` shows the same step: past calls 351/s to 570/s with rounds 13.8 ms to 18.9 ms. The first figures also read low because long games have not ended yet. `fgself-174`'s larger rise (rounds 17.6 to 27.9 ms in an hour) is not clean: it starts from fewer past games, and an agent's probes and browser tests were running beside it (load 40-58): re-read once they end. Possible gain, not built: past snapshots' inference on the box's CPU, where it has spare cores (80% of the GPU's calls for ~17% of its rows; `--infer-past-period-ms` batching them gave no net gain).

### The inference calls' precision and padding (2026-11-05, opt-in, not yet measured on the GPU)

Written beside a live run on the CPU (`scripts/infer_cpu_study.py`). With the GPU the limit, inference is ~2/3 of it, and its matrix multiplies run in **FP32**: the BF16 unlock never reached them (the gpu-unlock page had said otherwise). From fgself-182's calls (`infer_cpu_study.py flops`: FLOPs per call shape, weighted by `inference.jsonl`'s calls by rows and by widest row) and this GPU's rates at our layer shapes (fp32 ~18, tf32 ~40, bf16 ~80 TFLOPS), the inference calls' matmuls alone take ~250 ms of GPU a second, ~a quarter of it. 61% of the current network's calls pad to 64 rows (a call of 25-32 rows pays for 64). A call of 64 rows x 96 entities is 34 GFLOP: 1.9 ms in fp32, 0.85 in tf32, 0.43 in bf16.

| setting | the current network's matmul FLOPs | inference matmuls, GPU-ms a second |
|---|---|---|
| today: fp32, rows 8,24,64, entities 32,48,64,96 | 3.9 TFLOP/s | ~250 |
| `--infer-rows 8,16,24,32,48,64` | -26% | ~195 |
| and `--infer-entities 32,48,64,80,96,112` | -33% | ~180 |
| `--infer-precision tf32` (today's buckets) | | ~115 |
| tf32 and both bucket changes | | ~80 |
| `--infer-precision bf16` (today's buckets) | | ~55 |

Matmuls only: the small kernels' latency (past snapshots' calls are ~1-2 rows) is not in it, so these are upper bounds on what each frees.

* **`--infer-rows`** (default 8,24,64, as before): the rows the current network's calls pad to. A row's answer does not depend on the others (`tests/unit/test_infer_speed.py`: value, entropy and memory state the same with 0 to 27 rows beside it); only the sampling's noise is drawn for another shape. More graphs at the start (each row count x each entity count, compiled once, then from the inductor cache).
* **`--infer-precision`** fp32 (default) / tf32 / bf16, both networks' calls. tf32 is set only while the inference calls are compiled and their graphs captured (`selfplay.matmul_precision`; a graph keeps its cuBLAS kernels), so the learner's own numbers stay as they were. What it changes, on 6,372 recorded steps of fgself-148 (`infer_cpu_study.py precision`; TF32 emulated by rounding every matmul's operands, bf16 by the CPU's autocast):

| against fp32 | a unit's log-probability, mean / p99 / max | the order distribution's KL, mean | value, mean / max (values ~0.5) | most likely order the same |
|---|---|---|---|---|
| tf32 | 4.6e-5 / 1.3e-3 / 1.5e-2 | 2.6e-8 | 1.9e-4 / 9.8e-4 | 100% |
| bf16 (also: the learner's `--bf16 1` evaluation against today's fp32 acting) | 8.0e-4 / 2.2e-2 / 0.25 | 1.1e-5 | 1.9e-3 / 1.3e-2 | 99.97% |

The bf16 row is the gap PPO's ratio already carries between the actors' fp32 and the learner's bf16; tf32 adds about a twentieth of it, and moves the values by ~0.4% of the advantages' spread (adv/std ~0.044). bf16 doubles the ratio's rounding and moves the values by ~4% of it.

Not built (the next ones): **the past snapshots' calls in one call per round.** They are ~520 of ~670 calls a second at 1.4 rows each, ~3.7 a round, each a chain of ~100 small kernels, so their cost is per call, not per row: the 2026-10-30 estimate put them near two thirds of the calls' GPU. The slots' weights stacked ([slots, ...]) and one graph per shape running every slot's rows through batched matmuls (`torch.func.vmap` over `functional_call`, the sampling's noise drawn outside it) would make a round's 3-4 past calls one. It needs the C++ loop to gather a round's past rows by slot into one input; the network's answers can be checked against today's on the CPU, the speed only on the GPU.

### Recording videos faster than real time (2026-11-01, the user's question)

A 40-minute game took ~48 minutes on the main machine, and a quarter of the frames (`--speed 4`) saved nothing: the game side cost ~0.8 wall seconds per game second whatever the frame count. What paced it, all of it in the virtual sound card and in Python, none in the game's clock:

| what | how it paced | now |
|---|---|---|
| `audio_frame` slept until real time caught up with the audio played | one game second per real second, at any frame rate | `W3SIM_AUDIO_FREE=1` (videos only): no such sleep |
| Miles's mixer is a timer on its own real-time clock (its timer thread wakes every 5 ms, mixes every 50 ms of it, Mss32 imports QueryPerformanceCounter and timeGetTime itself), and the frame waited for it in 1 ms sleeps, up to 50 | at 4x most frames stalled ~55 ms with the sleep above gone, and the audio came out with silence in it | Miles's imports hooked: its clock is the audio played, each 25 ms piece wakes its timer and stream threads and waits for the timer's pass (0.15 ms) |
| the panel, the composition (BGRX to RGB per frame) and the pipe into ffmpeg ran between the game's steps | ~6 ms a frame at 1x, the game waiting | a bounded thread beside the game (`video._Writer`), and frames composed in the X window's own format (`FullGameOverlay.pix_fmt`) |

Measured with `scripts/rerender_fullgame.py` on real replays, wall time including the ~20 s load; the box at load 10-28 (its 56 training games), the main machine at load 50-77 (the run), both niced 10:

| game, machine | before 1x | before 4x | now 1x | 2x | now 4x | 8x |
|---|---|---|---|---|---|---|
| 78 s, box | 101 s | 100 s | 77 s | 49 s | 36 s | 30 s |
| 78 s, main | 137 s | 105 s | 132 s | | 53 s | |
| 40 min, box | | | 25 m 32 s (1.57x) | | 8 m 57 s (4.5x; 9 m 0 s with the copies there and back, as a `render_on_host`) | |
| 40 min, main | ~48 m | | 40 m 11 s (1.0x; its rate went from ~0.3x to ~2x with the run's load) | | 13 m 19 s (3.0x) | |

What is left is the game's own frame: ~14 ms of its main thread (47% the executable, spread thin, 12% wined3d, the rest system calls), per frame rather than per game second, and the grab ~1.8 ms. Tried and not kept: `LP_NUM_THREADS` 4 and 8 (21 and 16.5 ms a frame against 15), Direct3D's thread off (22.7 ms), XShmGetImage (0.3 ms less a frame, not worth its code), the exact game cuts (about 3%; they are in the run's flags, so a replay gets them anyway). The load's ~20 s: ~16 s of it comes with the music (4.7 s to the first frame with music off: half a game second of something at the load's 0.03x clock). A clock of 0.1x brings it to 9 s on the box but loses the game's first 0.4 s on the main machine under load (876 frames instead of 880), so it stays.

The output: the same frames (3,220 against 3,220 at 1x, 880 against 880 at 4x; ±1 or 2 between any two recordings, the old way too), PSNR 30.8 dB against an older recording where two older recordings differ by 29.0 dB (water and particles), the same units at all 157 steps of the 78 s game in every mode (a hash of each unit's position and hit points, old sound card against new, 1x against 4x). The sound: loudness over time correlates 0.955 with an older recording where two older ones reach 0.905; music off, sounds land 28 ms later than the old way (median of 5 s windows, p10 -67 to p90 +26 ms; two old recordings differ by 28 ms, 0 to 38), no silence added (the shim logged no padding in these recordings). At 2x, 4x and 8x the sound is the 1x sound sped up by atempo, in steps of 2 so that no samples are dropped: its loudness correlates 0.88, 0.90 and 0.85 with the 1x recording at the same tempo, at lag 0 within 20 ms. Not compared: a 40-minute recording the old way on the same machine (against the run's video from the main machine under load the new sound is ~70 ms later on average; that recording's own wander is not known).

**Measured on the free GPU (2026-11-06, between fgself-194's launches; `scripts/bench_inference.py --run runs/fgself-194 --data runs/fullgame/demos-echo-1 --setup --live-rows 10,45 --entities 32,48,64,96 --past-entities 32,48,64,96`):** fused fp32 with rows 8,24,64 took 7.38 ms a round (GPU 6.20 ms); rows 8,16,24,32,48,64 took 6.09 ms (GPU 4.96, -18%); TF32 4.83 ms (GPU 3.79, -35%); bf16 3.88 ms (GPU 2.82, -47%). TF32's value differences against fp32: mean 4.6e-3, 99th percentile 5.3e-4; bf16's 99th percentile 1.5e-2. A few rows differ by up to 2.7 in every comparison, fp32 against fp32 with the same settings too (0.79), so that is the benchmark's (or the inference loop's) own nondeterminism, under investigation. `fgself-194` relaunched with `--infer-rows 8,16,24,32,48,64 --infer-precision tf32` at 6.75M steps and crashed at startup: the native loop's past-snapshot graph capture with TF32 gave "CUDA error: misaligned address" (the benchmark, without NativeServer's slots, ran). It runs with the finer rows and fp32 (graphs captured in 105 s, from 64); TF32 stays off until that is understood.

**Both explained and fixed (582d25d, a bug finder, CPU tests).** The outliers were a bug in `Inference._launch`: a call's pinned answer buffer was keyed by the answers' width, and a call padded to 96 entities and one padded to 160 both answer 5 × 96 + 2 numbers a row, so two calls of a round shared one buffer and the first read the second's rows. This hit the benchmark and the Python inference server (`--infer-native 0`); the C++ loop the runs use keeps a buffer per shape, bank and call, and the laptop's evaluation worker waits after each call, so neither was affected. The TF32 crash was alignment: a past-snapshot slot's weights were views packed end to end in one flat tensor, and after the order head's 519-wide bias a fifth of them sat 12 bytes off a 16-byte boundary, which TF32's kernels refuse. Each weight now starts at a 512-byte boundary, the same layout for the slots and the C++ loop's copies. To confirm on the GPU before the next run: the benchmark's original command at ≤1e-3, `--twice --only past` and `--twice --only current` with 0 rows off, `--precisions tf32` without outliers, then a launch with `--infer-precision tf32`.

### Inference in bf16 without hurting learning: bf16-mixed (2026-11-06, CPU study; GPU to measure)

The question: which precision split keeps PPO's numbers intact while getting most of bf16's speed (fused rounds: fp32 7.38 ms, tf32 4.83, bf16 3.88)? Measured on the CPU beside fgself-194 with `scripts/infer_cpu_study.py mixed` (fgself-182's network, 12 game sides of `runs/fullgame/demos-echo-1` through the actors' `View.step`, 256 steps each after 32 to warm the memory: 3,072 steps, ~110k units). CUDA's bf16 autocast is emulated, not the CPU's (whose layer norms and sums stay bf16): every bf16 matrix multiply as a tensor core's (bf16 products summed in fp32, rounded to bf16 once), layer norms, softmax and sums in fp32, the attention as the memory-efficient kernel. The inference side as the server calls it (a step's rows in one call, padded to the buckets, each candidate carrying its own memory state, `model.act`); the fp32 call draws the actions every candidate is scored on. The learner's side as the update does it: `model.evaluate` on 16-step sequences from the candidate's recorded state, entities float16 (as `raw_step_value` stores them), minibatches of 16 sequences padded to `--update-widths`.

Against fp32 acting (a unit's |d log-probability| mean / p99 / max; KL per head, mean; the value):

| candidate | log-prob | KL order | KL pointer | KL x | KL y | value mean / p99 / max | top order the same |
|---|---|---|---|---|---|---|---|
| tf32 | 2.1e-4 / 3.6e-3 / 0.018 | 1.9e-7 | 2.1e-6 | 5.4e-7 | 9.7e-7 | 3.6e-4 / 1.4e-3 / 2.3e-3 | 99.998% |
| bf16 (the learner's autocast) | 3.2e-3 / 0.058 / 0.27 | 5.1e-5 | 2.1e-4 | 9.7e-5 | 3.6e-4 | 3.3e-3 / 0.015 / 0.023 | 99.79% |
| bf16, value head fp32 | as bf16 | | | | | 1.1e-3 / 4.0e-3 / 6.9e-3 | |
| bf16, heads' last layers fp32 | 1.6e-3 / 0.026 / 0.21 | 1.2e-5 | 1.3e-4 | 2.1e-5 | 4.3e-5 | 2.7e-3 / 0.012 / 0.021 | 99.95% |
| bf16, all heads fp32 | 1.3e-3 / 0.021 / 0.22 | 1.0e-5 | 1.0e-4 | 1.6e-5 | 1.0e-5 | 1.1e-3 / 4.0e-3 / 6.9e-3 | 99.95% |
| **bf16-mixed** (all heads fp32, their multiplies TF32) | 1.3e-3 / 0.021 / 0.22 | 1.0e-5 | 1.0e-4 | 1.6e-5 | 1.1e-5 | 1.2e-3 / 4.2e-3 / 6.8e-3 | 99.95% |

Why bf16 moves them: the heads' logits are large and bf16 rounds its outputs. The order's allowed logits are ~19 in the median (up to 111), the point's x bins 24-32: bf16's spacing there is 0.125-0.25, so each logit moves by up to half of that. The value head's hidden layer and output are rounded alike. The trunk's rounding (the encoders, the transformer, the memory core: 82% of the calls' FLOPs, `act`'s heads 18%, their last layers 9%) costs far less; TF32 in the heads is as good as fp32 there.

PPO's log ratio at the update's first minibatch from the numbers alone (no lag), the live learner (bf16, float16 entities) against each candidate: |log r| mean / p99 / max, the k3 KL estimate (`loss/kl`'s), the share past the clip (`loss/clipfrac`'s). fgself-194's last 200 updates for scale: `loss/kl` median 5.5e-4, `loss/clipfrac` 0.56%, staleness 3.15 updates.

| learner | candidate | log ratio | KL estimate | \|r-1\| > 0.2 | value, learner - candidate |
|---|---|---|---|---|---|
| bf16 (live) | fp32 (today) | 3.2e-3 / 0.058 / 0.27 | 7.8e-5 | 0.019% | 3.3e-3 |
| bf16 (live) | tf32 | 3.2e-3 / 0.059 / 0.26 | 7.7e-5 | 0.019% | 3.3e-3 |
| bf16 (live) | bf16 | 2.7e-3 / 0.052 / 0.38 | 8.0e-5 | 0.064% | 8.1e-4 |
| bf16 (live) | bf16-mixed | 2.9e-3 / 0.055 / 0.29 | 7.2e-5 | 0.019% | 3.3e-3 |
| bf16, float32 entities (diagnostic) | bf16 | 2.3e-5 / 0 / 0.12 | 4.2e-7 | 0% | 1.0e-5 |
| bf16 (live) | bf16, the actors' entities rounded to float16 first | 2.6e-5 / 1e-7 / 0.11 | 4.7e-7 | 0% | 1.9e-5 |
| bf16, its fp32 sums moved by ≤2e-7 (another kernel's order) | fp32 | 3.2e-3 / 0.058 / 0.27 | 7.6e-5 | 0.022% | 3.3e-3 |
| the same | bf16-mixed | 2.9e-3 / 0.056 / 0.31 | 6.8e-5 | 0.012% | 3.4e-3 |
| the same | bf16, entities float16 first | 2.2e-3 / 0.046 / 0.30 | 6.0e-5 | 0.050% | 6.8e-4 |
| bf16, heads fp32 (hypothetical learner) | fp32 | 1.3e-3 / 0.020 / 0.19 | 1.2e-5 | 0.001% | 9.1e-4 |
| bf16, heads fp32 (hypothetical learner) | bf16-mixed | 7.0e-4 / 0.011 / 0.15 | 3.6e-6 | 0% | 5.2e-4 |

* **bf16 that computes as the learner's autocast does not bring the ratio closer to 1.** Its mean is a little lower (the order head's especially), its tails are longer (max 0.38, three times the share past the clip) and the KL estimate the same as fp32's. The learner never computes the same thing: it reads the entities as float16, and that rounding (2^-11) flips bf16 roundings downstream, which cascade. With the learner given the actors' float32 entities, or the actors the learner's float16 ones, bf16 acting matched it nearly bit for bit (KL 4e-7) on the CPU, where both sum in the same order. But moving the learner's fp32 sums by at most 2e-7 before their rounding (less than another cuBLAS kernel, split-K or inductor fusion changes them; the GPU's learner and actors run different graphs at different shapes) undid it: KL 6.0e-5 and 0.050% past the clip, worse tails than fp32's. So an exact match is not to be had on the GPU, and float16 entities for the actors were not built. Two roundings of the large point logits, the learner's and the actors', are independent and add up.
* **The numbers are ~14% of the live first-minibatch KL** (7.8e-5 of 5.5e-4) and ~3% of its clipped share; the rest is the ~3 updates of lag. bf16-mixed keeps both as they are today (KL 7.2e-5, 0.019%), plain bf16 triples the clipped share.
* **The value**, the advantages' baseline: plain bf16 moves it by 3.3e-3 on average (~7% of the advantages' spread, 0.044), bf16-mixed by 1.2e-3 (~3%), tf32 by 3.6e-4.
* **The learner, not the actors, sets the floor.** A learner with its heads in fp32 too would cut the numbers' KL ~20 times against bf16-mixed acting (3.6e-6) and ~6 times against today's fp32 (1.2e-5). Not built: the learner's loss math is another agent's audit.
* Not masked on one side only in any unit (0 of ~110k): the float16 entities did not flip `unit_valid`'s thresholds here. (fgself-194's `loss/kl` mean has outliers in the thousands in 93 of 2,577 updates: a unit the learner gives ~-1e9 that the actor sampled. Their cause is not the precision; watch the median.)

Built: **`--infer-precision bf16-mixed`**: the calls under bf16 autocast for the trunk, `model.act(heads_fp32=True)` runs every head (order; the target's conditioning, query, key and pointer; x and y; the value) with autocast off in fp32, their multiplies TF32 (set while the calls are compiled and captured, `matmul_precision`). The C++ loop only replays the graphs and copies fp32 weights into the past-snapshot slots; autocast without its cache casts them inside each graph, so nothing changes there. The default (fp32) is the same calls as before: `act` bit for bit against master's on fgself-182's network, fp32 and the old bf16 path alike; dynamo traces the mixed act in one graph. The study's second round ran `bf16-mixed` through this code (`act(heads_fp32=True)`): the same numbers as the prototype to the digit. Tests: `tests/unit/test_infer_speed.py` (flags; the default unchanged; bf16-mixed within the bounds above with CUDA's autocast emulated; a slot answers with the weights copied into it).

To measure on the GPU (between runs, the GPU free), from the main tree:

1. Speed and accuracy: `python3 scripts/bench_inference.py --run runs/fgself-194 --data runs/fullgame/demos-echo-1 --setup --live-rows 10,45 --rows 8,16,24,32,48,64 --entities 32,48,64,96 --past-entities 32,48,64,96 --precisions tf32,bf16,bf16-mixed`. Passes: bf16-mixed's ms a round at most 0.9 times tf32's (else tf32, which is more accurate, is the better buy) and near bf16's; its value difference p99 ≤ 5e-3 and max ≤ 2e-2 (CPU: 4.2e-3, 6.8e-3; plain bf16 ~1.5e-2 p99), entropy p99 ≤ 1e-2.
2. The C++ loop and the past slots: `python3 scripts/check_native_infer.py --run runs/fgself-194 --data runs/fullgame/demos-echo-1 --setup --rows 8,16,24,32,48,64 --entities 32,48,64,96 --past-entities 32,48,64,96 --precision tf32`, then the same with `--precision bf16-mixed` (the heads' TF32 in the slots: the old crash's path). Passes: exit 0 ("0 wrong" within the precision's tolerances against direct fp32 calls, value and entropy: tf32 5e-3 and 0.025, bf16-mixed 0.02 and 0.2, a row's entropy being one unit's at worst; the slots' rows alike), "different samples", no CUDA error while the graphs are captured; against direct bf16-mixed calls (eager) the value p99 ≤ 5e-3 (inductor's fusions); requests a second not below an fp32 run of the same command.
3. A run forked from fgself-194 with only `--infer-precision bf16-mixed` changed, its first 20 minutes against fgself-194's last 200 updates: "graphs captured in" near 105 s and no CUDA error; `inference.jsonl` `gpu_idle` / `gpu_update` below 4.6 / 8.3 ms and `round_ms` below 14.9; `train.jsonl` SPS above 4,160 (median; p10 3,300, p90 5,170); medians, not means, of `loss/clipfrac` within 0.44-0.71% and `loss/kl` within 4.2e-4-8.8e-4 (the numbers are ~14% of the KL, so no visible change is expected: these guard against a broken path), `loss/value` ~1.0e-3, `adv/std` ~0.044, `loss/entropy` ~0.11. The ratio itself is not logged beyond those two.


**Confirmed on the GPU after the fixes (2026-11-06 14:55, between `fgself-194` and `fgself-195`; rows 8,16,24,32,48,64):** fused fp32 6.10 ms a round, TF32 3.97 (value difference p99 1.5e-4, max 2.4e-4: the outliers are gone), bf16 3.37 (p99 1.6e-2), **bf16-mixed 3.41** (p99 3.9e-3, max 7.1e-3, entropy p99 9.8e-4). `scripts/check_native_infer.py --precision tf32` and `--precision bf16-mixed` both pass in the C++ loop with the past-snapshot slots (no capture error, no Xid). `fgself-195` runs with `--infer-precision bf16-mixed --infer-rows 8,16,24,32,48,64`.

## After bf16-mixed: the learner's Python is the limit (2026-11-06, `fgself-195`)

`--infer-precision bf16-mixed` with `--infer-rows 8,16,24,32,48,64` took the run from ~4,300 to ~5,300 agent steps/s. Then, 15-minute windows each:

| change | steps/s | inference rounds/s | staleness |
|---|---|---|---|
| bf16-mixed, finer rows (box 56 games) | 5,309 | 215 | 3.28-3.57 |
| box 64 games (+~10% games from the box) | 5,263 | 216 | 3.59 |
| `--infer-past-slots 10` (swaps 615/s → 0), `--infer-past-period-ms 4` | 5,345 | 216 | 3.63 |
| `--update-pause-ms 8`, `--infer-period-ms 3` | 5,236 | 278 | 3.71 |

More games and a faster inference loop gave nothing while staleness rose: data waits for the learner. The learner works in turns, collecting 8,192 steps (~0.43 s, `perf/rollout`) and then updating (~1.1 s). In 8 py-spy dumps of its main thread, 3 were in `collate`/`to_device` (building minibatches on the CPU inside the update), 2 in the queue's `get`/`recv` (unpickling the games' steps), and 3 waiting on the GPU. The next gain is overlapping that Python with the GPU's work: intake in a thread or process, the next minibatch collated while the current one runs. Past-snapshot batching (~620 calls/s of 1.3 rows) stays a smaller, later gain. The pause and period changes go back to 4 and 4 at the next launch.

### The learner reads ahead during its update: `--learner-intake` (2026-11-06, opt-in, CPU-tested; run to measure)

fgself-195 after the bf16-mixed speed-up: the inference loop given more room (rounds 211 to 278 a second) and 8 more box games gave nothing (5,240-5,345 steps/s) while the staleness rose (3.28 to 3.71). The learner's cycle is strictly sequential, collect (`perf/rollout` ~0.43 s) then update (`perf/train` ~1.1 s), and py-spy put its main thread in collate (3 of 8 dumps), in `out_q.get` (2) and in the update's GPU wait or pause (3).

Measured on the CPU, synthetic data of fgself-195's shapes (8,192 steps in chunks of 64, entities 8-150 around 45, F 106, G 318, 519 orders, d 192, labels on ~15% of steps; `--seq-len 16 --minibatch 256 --pad-groups 8`, widths 32-128):

| the learner's work a cycle | ms |
|---|---|
| receiving and unpickling a batch (128 messages, ~100 MB, 8 actor processes waiting with theirs) | 257 (2.0 a message) |
| of which unpickling alone | 76 |
| the collect loop's own handling | 2 |
| collating the update's 32 minibatches | 228 (7.1 each, p90 10.8) |
| a minibatch's copy to pinned memory (6.4 MB, a memcpy) | ~0.3 |

* **Collate is not on the update's critical path.** Each minibatch is collated while the GPU runs the one before (~25 ms a minibatch, from 1.1 s / 32 less the 8 ms pause); the dumps found the thread there because that is where it spends the GPU's time. Vectorising it would not shorten the update.
* **The intake is.** It runs only between updates, and `out_q` holds at most a batch's messages (`batch_steps // chunk` = 128, episodes and games' short last chunks among them), so during the update the actors' and gateways' puts block once it is full, their games stop, and the collect loop then receives and unpickles (~0.26 s) and waits for the rest. More games or faster inference only filled the queue sooner.
* **The update idles anyway**: 31 pauses of 8 ms (`--update-pause-ms 8`, the GPU to the inference loop) are 0.25 s a cycle of `time.sleep`, and its waits for the GPU before them another ~0.3 s.

Built: **`--learner-intake`** (`selfplay.Intake`; default 0, as before). 1: the update's pauses read the actors' queue ahead instead of sleeping (each pause still ends 8 ms after the GPU finished, plus at most one message's read, ~2 ms); 2: also while it waits for the GPU (polling the minibatch's event between messages: late by at most one message). The collect loop takes the messages read ahead first and handles each exactly as before (episodes, exploiters' chunks, staleness at the same update), so the batches, their order and every minibatch are the same; reading ahead stops while a batch's steps wait, and the queue's own bound then holds the actors back as before. It needs `--update-pause-ms` above 0 (the CUDA-graph update; the eager fallback reads as before). `train.jsonl` gains `intake/messages`, `intake/steps`, `intake/seconds` (read ahead during the update) and `intake/waiting_steps`. Tests: `tests/unit/test_learner_intake.py` (three updates' batches, staleness and every minibatch tensor equal bit for bit with reads ahead at random moments, through a multiprocessing queue as the actors' messages come; the order kept; the cap; the pause's sleep, read and GPU poll).

Expected: the cycle 0.43 + 1.1 = 1.53 s (5,350 steps/s). Mode 1 moves the ~0.26 s of receiving and unpickling into the 0.25 s of pauses: ≥ 1.27 s, ~6,450 steps/s (+20%). The rest of the collect phase was the actors blocked on the full queue; with it drained during the update they keep playing, and the cycle tends to the update alone, ~1.13 s: ~7,250 steps/s (+35%) when the games make that many (now the extra box games count). The update may lengthen by up to ~2 ms a minibatch (a read overrunning a pause or, with 2, the GPU's end): ≤ 60 ms.

In the run (a fork of fgself-195 with only `--learner-intake 1`, then 2): `train.jsonl` medians over ≥ 100 updates against fgself-195's last 100: `perf/rollout` 0.43 → ≤ 0.2 s, `perf/train` ≤ 1.16 s, `SPS` 5,300 → ≥ 6,300, `intake/steps` ~5,000-8,000 an update, `staleness` not above ~4; the losses (`loss/kl`, `loss/clipfrac`, `loss/value`) in their usual ranges.

