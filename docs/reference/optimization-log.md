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
* **The learner**: the cloning loss's calls (~18% of an update: many small batches), the clone's forward for the KL term (~5%), a compiled update.
* **The video renderer**: 3.4 cores while it renders.
* **wineserver**: ~10% of a step.

## Measuring

* `py-spy record --threads --idle` on a process (it pauses the process while sampling: an update measured during a profile is slower).
* `perf record -a` (`/usr/lib/linux-tools/5.15.0-194-generic/perf`).
* `scripts/bench_env.py` (env steps/s, CPU per process and thread kind), `scripts/bench_train.py`, `W3SIM_PROFILE=1..3` (the shim's per-second profile in each game's `shim.log`).
* `runs/<run>/inference.jsonl` (the server's rounds, rows, waits); `scripts/selfplay_status.py <run>`.
* `scripts/production_probe.py` and `scripts/order_probability.py`: what a policy does at its buildings, and how likely it is to give the built-in AI's orders.
* Benchmarks with many games must share one measurement window (games launched at different times never overlap).
