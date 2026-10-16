# Optimizations

The game's CPU sets the speed. The [optimization log](reference/optimization-log.md) has every fix with its numbers.

## Self-play throughput

```mermaid
xychart-beta horizontal
  title "Whole-game self-play, agent steps per second"
  x-axis ["start", "pipes not queues", "games niced", "one GPU wait", "CUDA graphs", "real resets", "cheaper render", "units in C", "padded less", "5 games a load", "bounded queue", "learner as CUDA graphs", "no drawing, 40 games", "8 actors of 5 games", "Wine with fsync", "fused inference", "inference loop in C++", "native Linux", "GPU shared better", "game code faster", "learner intake", "no AI advice", "no cloning loss"]
  y-axis "steps/s" 0 --> 2400
  bar [560, 545, 595, 606, 607, 540, 644, 719, 800, 822, 840, 1000, 1110, 1200, 1280, 1430, 1490, 1850, 2150, 2280, 2000, 2030, 2220]
```

## Where the time goes

The bars from "learner intake" on are with 8 evaluation games running beside the 48 training games (the learner intake made it ~25% faster in the same setting). The real game on Echo Isles runs at ~1,500-1,650 steps/s: its games have more units.

```mermaid
flowchart LR
  G["48 games + 8 evaluation games<br/>drawing nothing"] -- "observations" --> I["Inference loop<br/>C++ in the learner, CUDA graphs"]
  I -- "orders" --> G
  G -- "trajectories" --> L["Learner<br/>8,192 steps in ~2.4-4 s<br/>CUDA graphs"]
  L -- "weights" --> I
```

* The games set the speed again: the learner waits ~1 s for each batch of 8,192 steps.
* The learner's update takes ~2.4 s on duelfast and ~4 s in the real game (5 layers, more units). The cloning loss had cost a third of it.
* A game step costs ~7 ms of CPU on native Linux, less with the equivalent speedups of the game's own code.

## The largest fixes

1. Orders and observations travel with the step sync, not through files.
2. The shim reads the units in C, not the map script in JASS.
3. Five games per map load, each with new players.
4. One inference server with CUDA graphs and one GPU wait per round.
5. The learner's minibatch steps as CUDA graphs: one launch where eager PyTorch made ~2,000.
6. The games draw nothing: the Direct3D device's draw calls return at once (17% less CPU a step, the same simulation).
7. While a game waits for its orders its background threads sleep, and Wine's gamepad drivers are not loaded (~22% less CPU a step).
8. GE-Proton's Wine with fsync: Wine's synchronization on futexes, not wineserver round trips (28% less CPU a step).
9. 8 actor processes of 5 games (13% more steps/s: each actor's games had waited for its GIL). The inference server's view steps through shared memory and its calls on CUDA streams (rounds 12.5 → 7.4 ms).
10. The inference server's calls compiled into fused kernels inside their CUDA graphs (30% less GPU a round, ~1,430 steps/s).
11. The inference server's loop in C++ in the learner's process: one GPU context, no GIL (the learner's update 5.1 → 3.5 s, ~1,490 steps/s).
12. Native Linux instead of WSL on the same machine (2026-10-11): a game step 7.4 instead of 9.2 ms of CPU, the learner's update 4.2 → 3.0 s, ~1,850 steps/s with 48 games (1,387 in WSL's last hour). Wine's ntsync gave nothing over fsync.
13. The learner and the inference loop share the GPU better (2026-10-12): the loop's CUDA events made once, the learner's copies to the GPU through pinned memory (both had held the CUDA driver's lock), and the learner pausing a few ms after each minibatch: rounds during the update 39 → 6 ms, ~2,150 steps/s.
14. The game's own code faster with the same results (2026-10-12, `--game-cuts 1529`: the harness's units pass without a JASS thread per unit, A* cell checks remembered within a search, a memoised map-version check and more): -16 to -28% CPU a game step, the same simulation at every step on 24 replays, ~2,280 steps/s.
15. The learner's intake (2026-10-12): plain arrays from the shared rows (a memmap subclass was slow to unpickle), 4 workers for the cloning batches: ~1,600 → ~2,000 steps/s with evaluation games.
16. No advice and no cloning loss from the built-in AI (2026-10-14, a recipe change that the policy no longer needed): no shadow AI in the games, no labels or cloning batches in the updates: ~1,900 → ~2,220 steps/s.

## What remains

* The games: the harness's JASS (~11% of the main thread after the cuts), its native calls (~8%), A*'s node lookup and open list.
* The actors' features in Python (~40% of an actor's CPU).
* The video renderer: 3.4 cores while it renders.
* Other work on the machine: a headless Chrome from another project takes ~10 cores in bursts.
* Longer steps on long games: about 1.4× more game time per CPU, at the cost of half the decisions.
