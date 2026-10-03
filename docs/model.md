# The whole-game model

One network plays one side. It sees the game from its own player's view, as a set of tokens. Code: `fullgame/features.py` and `fullgame/model.py`.

## The full network

E is the number of units in view (at most 160). O is the number of own units (at most 96). Width d = 192.

```mermaid
flowchart TB
  subgraph IN["Inputs, one step"]
    F["unit numbers<br/>[E, 29]"]
    TY["unit type ids<br/>[E] of 147"]
    CO["current order ids<br/>[E] of 156"]
    PL["player numbers<br/>[80]"]
  end
  F --> LF["Linear 29→192"]
  TY --> ET["Embedding 147×192"]
  CO --> EC["Embedding 156×192"]
  LF --> SUM(("+"))
  ET --> SUM
  EC --> SUM
  SUM --> UT["unit tokens [E, 192]"]
  PL --> MP["Linear 80→192 · ReLU · Linear 192→192"] --> PT["player token [192]"]
  PT --> SEQ["tokens [1+E, 192]<br/>padding masked"]
  UT --> SEQ
  SEQ --> TR["Transformer encoder × 3<br/>pre-norm · 4 heads · feed-forward 768 · dropout 0.1<br/>then a final LayerNorm"]
  TR --> G["g: player token out [192]"]
  TR --> U["u: unit tokens out [E, 192]"]
  G --> MEM["minGRU (optional)<br/>a, c = Linear 192→384 of g · z = sigmoid of a<br/>h = (1−z)·h + z·c, carried to the next step<br/>context = g + Linear 192→192 of h (starts at 0)"]
  MEM --> C["context c [192]"]
  C --> VH["Value head<br/>Linear 192→192 · ReLU · Linear 192→1"] --> V["value"]
  U --> OI["for each own unit i: concat u_i, c [384]"]
  C --> OI
  OI --> OH["Order head<br/>Linear 384→192 · ReLU · Linear 192→399"] --> MSK["mask: the unit type's orders,<br/>what the player can pay for · class 0 = none"] --> O1["order o_i [O]"]
  O1 --> OE["Embedding 399×192 of o_i"]
  OE --> CD["concat u_i + emb, c [384]<br/>Linear 384→192 · ReLU · LayerNorm"]
  U --> CD
  C --> CD
  CD --> Z["z_i [192]"]
  subgraph PTRS["Pointer head: a target unit"]
    Q["query: Linear 192→192 of z_i"] --> PTR["q·k / √192 over the E units<br/>→ target unit"]
    K["keys: Linear 192→192 of every u_j"] --> PTR
  end
  subgraph PNT["Point head: x, then y given x"]
    PX["x: Linear 192→128<br/>→ x bin"] --> XE["Embedding 128×192 of the x bin"] --> PY["y: z_i + it · Linear 192→192 · ReLU · Linear 192→128<br/>→ y bin"]
  end
  Z --> Q
  U --> K
  Z --> PX
  Z --> PY
```

* The tokens are a set. There is no position encoding: a unit's position is its x, y numbers.
* The order decides which target heads count: a unit (pointer), a point (x then y), or none.
* Each head samples with the Gumbel-max trick.

## Token channels

A unit token is the sum of three parts:

* a linear layer on 29 numbers (the table below),
* a learned vector for its unit type (147 types),
* a learned vector for its current order (156 orders).

| index | channel |
|---|---|
| 0, 1, 2 | side: own, enemy, neutral (as this player sees it) |
| 3, 4 | position x, y ÷ 3072 (x mirrored, so the own base is always on the left) |
| 5, 6 | hit points ÷ maximum, maximum ÷ 1000 |
| 7, 8 | mana ÷ maximum, maximum ÷ 1000 |
| 9 – 19 | flags: hero, building, worker, under construction, hidden, loaded, sleeping, paused, summoned, illusion, flying |
| 20 | hero level ÷ 10 |
| 21 | gold left in it (gold mines) ÷ 12,500 |
| 22 | unspent skill points ÷ 3 |
| 23, 24 | facing: sine, cosine (mirrored) |
| 25 | it has an order now |
| 26 | own building: units queued ÷ 5 |
| 27 | own building: it makes something now |
| 28 | own worker: sent to cut lumber |

The **player token** has 80 numbers:

| index | channel |
|---|---|
| 0 – 5 | gold ÷ 1000, lumber ÷ 1000, supply used ÷ 100, supply cap ÷ 100, upkeep ÷ 2, step number ÷ 1800 (steps of half a second) |
| 6 – 9 | own race: human, orc, undead, night elf (one-hot) |
| 10 – 13 | enemy race, the same |
| 14 – 79 | the player's level of each of 66 upgrades |

**How players differ:** only by the side channels. The view is always from "my" side, so "own" means this player. The two races sit in the player token. Position is the x, y numbers, not a token index. The transformer treats the units as a set, so their order does not matter.

## Memory (optional): a minGRU across steps

The transformer looks across units within one step. The minGRU looks across steps, on one vector: the player token's output `g`.

```mermaid
flowchart LR
  subgraph S1["step t-1"]
    T1["Transformer<br/>over units"] --> G1["g(t-1)"]
  end
  subgraph S2["step t"]
    T2["Transformer<br/>over units"] --> G2["g(t)"]
  end
  G1 --> H1(("h(t-1)"))
  H1 --> H2(("h(t)"))
  G2 --> H2
  H2 --> C["context = g(t) + W·h(t)"]
  G2 --> C
  C --> HEADS["Order, target and value heads"]
```

* Gate and candidate come from `g(t)` only: `z = sigmoid(A·g)`, `c = B·g`.
* The new state is `h(t) = (1 − z)·h(t−1) + z·c`.
* The gate does not read `h`, so training computes 16 steps at once with a parallel scan. The actors start each sequence from the state they stored.
* `W` starts at zero, so a new core changes nothing at first. In `fgself-12` its size grew from 0 to 2.2.

There are no tokens per time step and no time encoding. The order of the steps is the order of the recurrence. AlphaStar and OpenAI Five use the same split: a network over units at each step, then a recurrent core.

## Actions

* Every own unit gets its own order at every step (half a second).
* Masks allow only orders that the unit's type gave in the demonstrations, and that the player can pay for now.
* The chosen order goes into the target heads, so a target depends on its order.
* The action's log-probability is the order's plus the target heads that the order uses.
