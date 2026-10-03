# Optimizations

The game's CPU sets the speed. The [optimization log](reference/optimization-log.md) has every fix with its numbers.

## Self-play throughput

```mermaid
xychart-beta horizontal
  title "Whole-game self-play, agent steps per second"
  x-axis ["start", "pipes not queues", "games niced", "one GPU wait", "CUDA graphs", "real resets", "cheaper render", "units in C", "padded less", "5 games a load", "bounded queue"]
  y-axis "steps/s" 0 --> 900
  bar [560, 545, 595, 606, 607, 540, 644, 719, 800, 822, 840]
```

## Where the time goes

```mermaid
flowchart LR
  G["32 games<br/>CPU: 35–40 ms a step"] -- "observations" --> I["Inference server<br/>CUDA graphs"]
  I -- "orders" --> G
  G -- "trajectories" --> L["Learner<br/>8,192 steps in ~9 s"]
  L -- "weights" --> I
```

* The games fill all 32 CPU threads. A step costs 14.6 ms alone and 35–40 ms with 32 games running.
* The learner is about as fast as the games. Each update of 8,192 steps takes 8.5–9 s.

## The largest fixes

1. Orders and observations travel with the step sync, not through files.
2. The shim reads the units in C, not the map script in JASS.
3. Five games per map load, each with new players.
4. One inference server with CUDA graphs and one GPU wait per round.

## What remains

* Longer steps on long games: about 1.4× more game time per CPU.
* A faster learner: fewer small calls for the cloning loss.
* The video renderer: 3.4 cores while it renders.
