# Optimizations

The game's CPU sets the speed. The [optimization log](reference/optimization-log.md) has every fix with its numbers.

## Self-play throughput

```mermaid
xychart-beta horizontal
  title "Whole-game self-play, agent steps per second"
  x-axis ["start", "pipes not queues", "games niced", "one GPU wait", "CUDA graphs", "real resets", "cheaper render", "units in C", "padded less", "5 games a load", "bounded queue", "learner as CUDA graphs", "no drawing, 40 games"]
  y-axis "steps/s" 0 --> 1200
  bar [560, 545, 595, 606, 607, 540, 644, 719, 800, 822, 840, 1000, 1110]
```

## Where the time goes

```mermaid
flowchart LR
  G["40 games, drawing nothing"] -- "observations" --> I["Inference server<br/>CUDA graphs, ~87% busy"]
  I -- "orders" --> G
  G -- "trajectories" --> L["Learner<br/>8,192 steps in ~5 s<br/>CUDA graphs"]
  L -- "weights" --> I
```

* The games fill all 32 CPU threads. A step costs 14.6 ms alone and 35–40 ms with 32 games running.
* The learner is faster than the games: an update of 8,192 steps takes ~5 s (9.5 s before its CUDA graphs), and it waits 2–3 s for the next batch.

## The largest fixes

1. Orders and observations travel with the step sync, not through files.
2. The shim reads the units in C, not the map script in JASS.
3. Five games per map load, each with new players.
4. One inference server with CUDA graphs and one GPU wait per round.
5. The learner's minibatch steps as CUDA graphs: one launch where eager PyTorch made ~2,000.
6. The games draw nothing: the Direct3D device's draw calls return at once (17% less CPU a step, the same simulation).

## What remains

* Longer steps on long games: about 1.4× more game time per CPU.
* The video renderer: 3.4 cores while it renders.
