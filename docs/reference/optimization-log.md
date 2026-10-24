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

Parity: 63 replays (fgself-137/139/140/141/142, old recordings; 63,792 steps) with everything but the harness bits' effect (their maps' harness never asks): identical, -6.3% instructions, -1.5% CPU. 15 live recordings with the new harness (scripted agents, `cut_ab.py record --rich` for every kind of command including pending purchases, unknown units and commands cut short, `--rich=train` for training's proportions): identical with the bits off and on, against the live game, and played back with round 3's shim (master's: the harness's JASS paths). Per recording, all bits: -5.5/-6.2% (train-like agents), -4.3/-5.3%, -3.2/-4.8% (an agent against the AI), -2.3/-3.4% (AI against AI), -14.5/-15.9% (50 commands a step).

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
* **The games' main thread after `17179866617`** (round 4): pathing ~38-60% (the long-range search waits on a cell's cache miss per node it expands; the rest of it, the local search and planning, spread over many small functions and the soft float's add and subtract, which are not IEEE), the other agent timers, attack and fog (~20-25%), the units pass's natives (~7-11%: their bodies; the conversions are memoised), EnumItemsInRect's walk (~2%: it stamps every widget) and ~90 JASS statements a step of W3S_Step.

## Measuring

* `py-spy record --threads --idle` on a process (it pauses the process while sampling: an update measured during a profile is slower).
* `perf record -a` (`/usr/lib/linux-tools/5.15.0-194-generic/perf`; counters need `kernel.perf_event_paranoid` <= 2, Ubuntu's default 4 refuses them: `scripts/cut_ab.py` then reports CPU time from schedstat).
* `scripts/bench_env.py` (env steps/s, CPU per process and thread kind), `scripts/bench_train.py`, `W3SIM_PROFILE=1..3` (the shim's per-second profile in each game's `shim.log`).
* `runs/<run>/inference.jsonl` (the server's rounds, rows, waits); `scripts/selfplay_status.py <run>`.
* `scripts/production_probe.py` and `scripts/order_probability.py`: what a policy does at its buildings, and how likely it is to give the built-in AI's orders.
* Benchmarks with many games must share one measurement window (games launched at different times never overlap).
