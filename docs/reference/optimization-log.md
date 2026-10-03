# Optimizations

How a step of the real game became cheap enough to train on, what each fix bought, what limits throughput now, and what did not work. The numbers are from this machine: Ryzen 5950X (16 cores, 32 threads), 47 GB, RTX 3090, WSL2, WineHQ stable 11.0, llvmpipe.

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
* What would remove most of the requests: Wine's ntsync (events in the kernel, no wineserver round trip), which needs `/dev/ntsync` (Linux 6.14; WSL runs 6.6), or the turn hand-off's events replaced in the shim. Proton's fsync does it on WSL's kernel (below).

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
