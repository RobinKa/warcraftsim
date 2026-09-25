# warcraftsim

Headless Warcraft III for reinforcement learning: the **real game** (Legacy TFT 1.29.2, Blizzard's
offline client) under Wine in Ubuntu/WSL2, driven step by step from Python. There is no renderer
window, no Battle.net and no human input, and it runs much faster than real time.

```python
from warcraftsim import Wc3Game, GameSetup, Agent, BuiltinAI

with Wc3Game(GameSetup(map="(2)EchoIsles", slots=[Agent("human"), BuiltinAI("orc", "insane")])) as game:
    obs = game.reset()                       # game time 0
    while not obs.game_over:
        for worker in game.idle_workers():
            game.harvest(worker, game.nearest_mine(worker))
        obs = game.step()                    # 0.25 s of game time
    print(obs.players[0].result)
```

## How it works

```
Python (warcraftsim)                 Wine prefix per game                 Warcraft III 1.29 process
------------------------------       -------------------------------      -------------------------------------
Wc3Game / Wc3Env / Wc3VecEnv          .wgc game config (slots, AI)  --->   map = stock map + injected JASS harness
GameInstance ── TCP 127.0.0.1 ──────────────────────────────────────────── w3shim.dll (injected by w3launch.exe)
     │  observations and commands travel with the step sync:              - virtual clock (game runs N x faster)
     │  "OBS n len" + tokens  ->   <- "GO ... A n ints"                    - blocks the game at every step
     └──────────────────────────────────────────────────────────────────── harness: Preload() observation tokens
                                                                              (captured by the shim), mailbox native
                                                                              for commands
```

* **Harness** (`warcraftsim/harness/w3sim.j`): JASS injected into the map script by
  `data/mapbuild.py`. On every step (a game-time timer) it:
  1. writes an observation: players, changed units, events and command results;
  2. reads the next commands through a mailbox native the shim answers;
  3. issues them with ordinary order natives (`IssuePointOrderById`, ...).

  Its other jobs:
  * It suppresses the built-in AI for agent slots.
  * It decides wins and losses without ending the session.
  * It runs scenarios.
* **w3shim.dll** (`shim/`, mingw, 32-bit). It is injected into the game by `w3launch.exe`, and all its hooks are in the game's import table or a few engine functions:
  * **Virtual clock:** replaces the game's timers (QPC, GetTickCount, FILETIME, rdtsc helper) and scales its waits, so the game runs as fast as the CPU allows.
  * **Step sync:** when the harness calls its mailbox native, the DLL freezes the clock, sends the observation over TCP ("OBS") and waits for "GO" with the next commands. That makes stepping exactly synchronous and deterministic in game time.
  * **Frame capture and audio** for videos (see Watching games).
* **`.wgc` game configs** start a local game directly, with no menus or Battle.net. They set the map, the slots (agents are computer slots with no AI; built-in AIs get easy/normal/insane) and an observer as the local player.

## Setup (once)

1. **Install Warcraft III - Legacy TFT 1.29** from the Battle.net app (Warcraft III → Game Version dropdown), then copy it into WSL:
   ```bash
   rsync -a --exclude .battle.net --exclude Data "/mnt/d/Warcraft III (Legacy)/" ~/wc3/legacy-1.29/
   ```
   To use another location, set `WC3_GAME_DIR`.
2. **Install system packages** (WineHQ, Xvfb, mingw, ...):
   ```bash
   sudo bash scripts/setup_system.sh        # WineHQ stable (11.0); staging measured slower here
   ```
3. **Build the native helpers and create the Python venv:**
   ```bash
   git submodule update --init
   python3 -m venv .venv && .venv/bin/pip install cmake pytest numpy "gymnasium>=1.0" && .venv/bin/pip install -e .
   scripts/build_native.sh     # StormLib (MPQ), pjass (JASS checker)
   make -C shim                # w3shim.dll, w3launch.exe
   .venv/bin/python -m warcraftsim setup
   ```

## Usage

| | |
|---|---|
| `python -m warcraftsim ai-vs-ai --difficulty insane` | Two built-in AIs play a full game headless. |
| `python -m warcraftsim play --difficulty easy` | The scripted Human bot plays the built-in AI. |
| `python -m warcraftsim scenario` | Micro skirmish against the scripted opponent. |
| `python -m warcraftsim bench -n 8` | Parallel throughput. |

### Python API

* **`GameSetup`:** map, slots, `step_seconds`, `speed` (clock multiplier; adaptive by default), `max_game_seconds`, `fog`, `scenario`, `warm_spare` (melee: keep a loaded spare game so `restart()` takes about 1 s).
* **Slots:**
  * `Agent(race)`: controlled from Python.
  * `BuiltinAI(race, "easy"|"normal"|"insane")`: Blizzard's melee AI.
  * `Scripted(race)`: scenario opponent that attack-moves to the nearest enemy.
  * `Idle(race)`: no controller.
* **`Wc3Game`:**
  * `reset()` and `step(commands)`.
  * Queued order helpers: `move`, `attack`, `attack_move`, `smart`, `stop`, `hold`, `harvest`, `harvest_tree`, `train` / `research` / `upgrade`, `build`, `learn`, `cast`, `use_item`.
  * `cast()` accepts an order string (`"thunderbolt"`) or an ability code (`"AHtb"`). All ability order strings in the game data are resolved in the first observation.
  * Heroes report each ability slot's level and cooldown left (`Unit.abilities`, in the order of `data.abilities.hero_abilities()`). `data.abilities.ability_info()` has each hero ability's name, order, how it is cast (unit / point / instant / passive), whom it is for, and its mana, cooldown, range and area per level. Queued heroes learn skills with `QueueSpawn(..., skills=("AHtb", "AHtb", "AHtc"))`.
  * Debug commands: `spawn`, `set_resources`.
  * Queries: `my_units`, `idle_workers`, `mines`, `enemies`.
* **`Observation`:**
  * `players` (gold, lumber, food, upkeep, result, ...).
  * `units`: every unit with id, type, owner, position, facing, hp/mana, current order, flags (hero, structure, worker, constructing, hidden, ...) and per-agent visibility.
  * `events`: deaths, training, research, construction, spells, tree deaths.
  * `command_results`.
  * On the first observation only: `destructables` and the `orders` table.
* **Gymnasium** (`warcraftsim.env`):
  * `Wc3Env`: feature arrays; actions are lists of commands.
  * `MicroEnv`: scenario fights, with a MultiDiscrete action per unit.
  * `NavigateEnv`: move a unit to a target.
  * `MicroSelfPlayEnv`: two policies fight each other (PettingZoo-style dicts per player, zero-sum reward).
* **`warcraftsim.vec.Wc3VecEnv`:** N games in parallel with auto-reset.
* **Self-play in full games:** use two `Agent` slots. `game.as_player(1)` returns a handle whose helpers and queries act as player 1; orders from all handles go out with the next `step()`.

### Watching games

The game runs headless, so there are two ways to see what happened:

* **Trajectories (any game).** `warcraftsim.record.TrajectoryRecorder` saves the observation stream to a `.jsonl` file. `python -m warcraftsim view ep.jsonl` renders it as a self-contained HTML animation: units over the map's walkable area, with HP bars, player stats, play/pause and a time slider.
  `play` and `scenario` take `--record DIR`.
* **Replays and videos.** `GameInstance.save_replay(path)` ends the game normally and saves two files: the `.w3g` replay the engine writes, and `path.commands.json`, the agent orders per step.
  * Agent orders are issued by the map script, not through the engine's recorded command stream. So the stock 1.29 client shows the built-in AI's play correctly but not the agents'.
  * `GameInstance.play_replay(path)` plays the replay back through the harness and feeds the logged orders in at the same steps, reproducing the game exactly (verified step by step).
  * `warcraftsim.video.render_replay(setup, path, "ep.mp4")` records that playback as real game footage with the game's sound: MP4 at 40 fps in real time by default, optionally following a player's units.
    * The shim switches the game clock to frame-stepped mode, so every rendered frame advances the game by exactly 25 ms, one engine turn.
    * Each frame is grabbed from the virtual display before the game continues.
    * The result is smooth and independent of machine load. A 60 s episode renders in about 60–75 s next to a training run.
    * **Audio** comes from a virtual sound card in the shim (`shim/audio.c`, `GameSetup.audio`):
      * The game mixes its sound with Miles, which outputs through `waveOut`; the shim implements those functions.
      * Each captured frame takes exactly one frame's worth (25 ms) of mixed audio from Miles's queue, so sound follows the picture however fast the render runs.
      * Miles mixes ahead of playback. The shim lowers its buffering (preferences 11 and 45: about 55 ms instead of about 180 ms) and reports the remaining queue; the soundtrack is shifted earlier by that amount.
      * Audio renders run at no more than real time, because Miles's mixer runs on real time. If it falls behind, the gap is padded with silence and the late audio is dropped, so the soundtrack stays in sync.
      * `music_volume` sets the music level (default 40, 0 for none). Training games have no sound device.
* **Screenshots.** `instance.screenshot(path)` saves the virtual display. In scenarios the camera is centered on the action.

### Scenarios (fast RL iteration)

```python
from warcraftsim import GameSetup, Agent, Scripted, Scenario
Scenario.skirmish(["hfoo"] * 4, ["ogru"] * 3)            # last side standing wins
Scenario.move_to_target("hfoo", distance=1200)           # navigation, success judged in Python
Scenario(units=(SpawnSpec(0, "hfoo", -300, 0), ...), victory="elimination", max_game_seconds=60)
```

A scenario:
* removes every pre-placed unit;
* clears trees around its center;
* spawns its units.

The default scenario map is `"flat"`: a generated 32×32-tile (4096×4096) level plane with no water, cliffs or trees, centered at (0, 0), with the camera on the action.
* `"flatN"` gives an N×N-tile plane.
* `"flat:(2)EchoIsles"` gives a full-size flat version of a stock map.
* Any stock map name also works; its most open walkable spot becomes the center.

`reset()` re-spawns them inside the running game in about 10 ms, so there's no reload.

## Training with PufferLib 5.0 and the dashboard

PufferLib 5.0 (`third_party/PufferLib`, the current `5.0` branch) is a native CUDA trainer, and its environments are C code compiled into the `puffer` binary. The pieces:
* **C environment** (`puffer/wc3_bridge.h`): each environment is one Warcraft III game, reached through a bridge.
* **Bridge** (`warcraftsim/puffer/bridge.py`): a Python server that runs the games and serves them to the trainer over a Unix socket. Every trainer environment gets its own game thread.
* **Tasks** (`warcraftsim/puffer/tasks.py`): fixed-size observations and actions for PufferLib. The C header is generated from them.

```bash
# needs the CUDA toolkit (nvcc), clang, ccache, NCCL, libomp:
#   sudo apt-get install ccache libnccl2 libnccl-dev libomp-14-dev libomp5-14 libgl-dev libx11-dev
python -m warcraftsim.puffer.train --task nav --timesteps 400000                  # navigation
python -m warcraftsim.puffer.train --task footmen2 --step-seconds 0.5 --timesteps 400000 \
    --lr 0.01 --minibatch 192 --replay-ratio 4                                     # 2 v 2 footmen: 95% wins in ~1.5 min
python -m warcraftsim.puffer.train --task micro_mirror --timesteps 3000000        # 4 v 4 footmen
python -m warcraftsim.puffer.train --task selfplay_micro --envs 8 --timesteps 3000000 # both sides learn
# a sweep: the games launch once, then one run per --sweep (runs NAME-1, NAME-2, ... in the dashboard)
python -m warcraftsim.puffer.train --task footmen2 --name f2 --timesteps 1e6 --sweep "--lr 0.01" --sweep "--lr 0.02"
python -m warcraftsim dashboard                                                    # http://localhost:8765
```

Tasks (add more in `tasks.py`; `scripts/baselines.py` measures scripted policies on any of them):
* `nav`: reach a point.
* `footmen2`: 2 vs 2 footmen with 100 hit points against the scripted opponent (episodes ~17 s).
  * Scripted baselines win 0% (random), 30% (noop), 65–70% (focus fire), 90% (focus fire, and pulling a footman back while it is low and being hit).
  * Tuned by sweeps (`f2-sweep1`..`5`, `f2-step*` in the dashboard): 0.5 s steps, lr 0.01, minibatch 192, replay ratio 4, the learning rate annealed over 400k steps → 95% wins after ~0.2M steps (~1.5 min), 98-100% soon after; 1.0 s steps: 100% for both seeds.
  * Most of the speed came from more updates per sample (1 → 32 per epoch), then from longer steps and a shorter annealing schedule. Horizon 32 learns fastest at first but ends lower.
  * Early policies learned focus fire plus pulling a hurt footman back. The 100% policy instead holds position until the enemies arrive: the scripted opponent then splits its damage over both footmen, while ours focus one enemy.
* `mirror_mix[_sem][_abil][_hp<P>]`: a mirror match with a new random composition every episode: a hero (level 1-3) and 2-4 units from all races (footman, rifleman, knight, grunt, headhunter, tauren, ghoul, crypt fiend, abomination, archer, huntress), at P‰ of their hit points (default 250). Unit features include the type's range, DPS, armor, speed and cooldown (`data.objects.combat_stats`); episodes are spawned through `QueueSpawn` + `Restart` (`Wc3Game.reset(spawns=...)`).
  * Hit points decide whether micro matters (scripted baselines, 120-150 episodes each; the time limit scales with hit points):

    | hit points | episode | noop | focus + pull back (`pull35`) |
    |---|---|---|---|
    | 25% | 21 s | 55% | 44% |
    | 35% | 28 s | 53% | 64% |
    | 40% | 31 s | 54% | 80% |
    | 50% | 36-40 s | 53% | 79% |

    At 25% units die within a few hits and every order costs more than it gains (attacking the weakest enemy in range 48-50%, pulling back without focus 19%); RL at 25% converged to noop (lr 0.003: 52%). `mirror_mix_hp400` is the training setting.
  * `_sem`: the target head picks a rule instead of an enemy slot (weakest in range, nearest, weakest, hero, threat = DPS per hit point left) and stop becomes retreat (straight away from the nearest enemy), so an order means the same whatever the composition (`MicroEnv(targeting="semantic")`).
  * `_abil` (implies `_sem`): heroes fight with their abilities. Each hero gets a random skill build for its level (fighting abilities only: no summons, far sight, blink or sacrifices), the same on both sides. Heroes can also cast:
    * each unit has a fourth action head, the ability slot, and a fifth kind, cast;
    * the target rules pick an enemy within the ability's cast range, or an own unit for heals and buffs (the most hurt, the nearest, a hero, the strongest);
    * instant abilities need no target;
    * a cast that isn't possible (not learned, cooling down, too little mana, no target) does nothing and is counted as `cast_invalid`.

    Units get 10 more features per ability slot: level, ready, cooldown, how it is cast, for whom, range, area. The scripted opponent's heroes cast what `MicroEnv.scripted_cast` picks. Scripts: `cast<policy>`, e.g. `castpull35`.

    Results on `mirror_mix_abil_hp400` (0.5 s steps; runs `abil40-*`):

    | policy | win rate |
    |---|---|
    | noop (no casting) | 31% |
    | pull35 (no casting) | 29% |
    | castnoop | 50% |
    | castpull35 | 50% |
    | PPO from scratch, lr 0.001, 2.5M steps | 42% |
    | PPO from the fitted castpull35, lr 0.001, 2.5M steps | 52% |

    Not casting against a caster loses badly. PPO from scratch found a shortcut: 60-66% of its actions are casts and 94% of those aren't possible. An impossible cast does nothing, so this means "do nothing, and cast each ability the moment it's ready".

    Sweeps for learning speed from scratch (with action masks; 600k steps, ~5 min each; runs `abil6-*`, `abil6b-*`). At this budget the win rate is still rising, so faster runs show as higher numbers:

    | round 1 (base: lr 0.001, minibatch 192, replay ratio 4, horizon 64, γ 0.99, λ 0.9, entropy 0.001) | win @0.3M | win @0.6M |
    |---|---|---|
    | lr 0.003 | 21% | 29% (peak 34%) |
    | replay ratio 8 | 9% | 27% |
    | base | 4% | 25% |
    | lr 0.002 | 7% | 23% |
    | minibatch 384 | 8% | 22% |
    | hidden 256 | 4% | 21% |
    | λ 0.95 / entropy 0.0003 | 1% | 15% |
    | γ 0.97 | 0% | 0% (never learned) |

    | round 2 (base: lr 0.003) | win @0.3M | @0.45M | @0.6M |
    |---|---|---|---|
    | horizon 32 | 27% | 35% | 33% (peak 40%) |
    | lr 0.005 | 26% | 28% | 33% |
    | λ 0.8 | 24% | 31% | 33% (peak 38%) |
    | γ 0.995 | 24% | 31% | 33% |
    | clip 0.3 | 20% | 33% | 34% |
    | base (lr 0.003; round 1: 21% / 29%) | 19% | 28% | 33% |
    | replay ratio 8 | 10% | 26% | 30% |
    | entropy 0.003 | 7% | 16% | 24% |

    | round 3 (base: lr 0.003, horizon 32) | 30% reached at | win @0.3M | @0.45M | @0.6M |
    |---|---|---|---|---|
    | horizon 16 | 0.19M (1.8 min) | 33% | 39% | 39% |
    | clip 0.3 | 0.25M | 34% | 33% | 40% |
    | λ 0.8 | 0.27M | 31% | 37% | 36% (peak 42%) |
    | lr 0.005 + λ 0.8 + γ 0.995 + clip 0.3 | 0.30M | 29% | 38% | 38% |
    | lr 0.005 | 0.31M | 29% | 27% | 35% |
    | γ 0.995 | 0.33M | 27% | 34% | 32% |
    | base (horizon 32; round 2: 27% / 35% / 33%) | 0.40M | 21% | 31% | 36% |
    | minibatch 96 | 0.45M | 27% | 30% | 35% |

    | round 4 (base: lr 0.003, horizon 16) | 30% reached at | 35% reached at | win @0.3M | @0.6M |
    |---|---|---|---|---|
    | **λ 0.8 + clip 0.3** | **0.10M (1.0 min)** | **0.15M (1.5 min)** | 34% | **45%** (peak 47%) |
    | horizon 8 | 0.13M | 0.15M | 37% | 40% |
    | horizon 8 + λ 0.8 | 0.12M | 0.15M | 33% | 41% |
    | clip 0.3 | 0.13M | 0.24M | 42% | 41% |
    | λ 0.8 | 0.15M | 0.25M | 35% | 40% |
    | base (horizon 16; round 3: 0.19M) | 0.23M | 0.29M | 35% | 38% |
    | lr 0.005 | 0.24M | 0.37M | 32% | 39% |

    Shorter horizons (64 → 32 → 16) give more, smaller updates per sample and learned fastest; horizon 8 costs throughput (the trainer updates twice as often). The winner, horizon 16 with λ 0.8 and clip 0.3, is the task's default now (`Task.train_defaults`, used where the command line doesn't set an option). It reaches 30% wins after 0.1M steps; the settings this started from needed 0.66M.

    Seed noise is about ±4 points at 0.6M, where the learning rate has annealed to zero and most configs end alike; the earlier columns separate them better. The minibatch must be a multiple of the horizon (PufferLib asserts it).

    The plateau: from scratch the tuned settings reach 40-45% after 0.6M steps, and 42-44% after 2.5M (runs `abilbest-*`). What was tried at 600k steps (runs `plat-*`; base 40%, other seeds 45% and 38-45%):

    | change | win @0.6M |
    |---|---|
    | hidden 256 / 3 layers | 39% / 39% |
    | entropy 0.003 annealed to 0 | 43% |
    | γ 0.995 | 40% |
    | replay ratio 2 / 8 | 39% / 40% |
    | relational features (`_rel`: nearest opponent, in range, threatened, weakest, time to die) | 37-39% |
    | tactical masks (`_tac`: retreat only while losing hit points, no plain moves) | 38-40% (faster early; retreat still unused) |
    | 0.25 s steps (horizon 32) | 33-35% |
    | **start from the fitted `castpull35` (masked BC), lr 0.001** | **52%** (50% after 0.27M steps) |
    | start from the fitted `castpull35`, lr 0.003 | 48% |
    | value-loss weight 0.5 / 4 (default 2) | 42% / 38% |
    | value clipping off / gradient norm 0.5 / lr floor 20% | 40% / 40% / 41% |
    | V-trace / momentum 0.9 / momentum 0.98 | 38% / 37% / 36% |
    | one agent per unit, one shared policy (`_units`, 3M agent steps = 0.6M game steps) | 43% / 45% (30% after 54k game steps, 0.6 min) |

    Scripts that cast more carefully don't do better either (`smartcast`: area spells only with two enemies inside, targeted spells on the biggest threat, heals below half; 200 episodes each): `castnoop` 53%, `smartcastnoop` 48%, `smartcastpull35` 55%. Against a casting opponent, reasonable strategies all end near a coin flip: the headroom here may be small.

    The trained policy plays as well greedily as sampled (43% / 42%), so that is its level. With the scripted casting rule in place of its own casts it wins 45%; `castnoop` (no attack orders at all, the same casting) wins 50%. So the gap to the scripts is partly casting, and partly attack orders that do worse than letting units auto-acquire. Only a better starting point moved the plateau.

    Longer from the fitted script (`bclong`, 2.5M steps, lr 0.001): 47% at 0.3M, 50% at 2.5M, flat. It keeps the script's style: 91% attack orders, 6% casts.

    Self-play (`mirror_mix_abil_self_hp400`, `MirrorSelfPlayEnv`: the policy plays both sides of the same game with the full action set; the observations and actions are the single-agent task's, so a checkpoint also plays the scripted opponent with `bc eval`). Starting from `bclong`, 3M agent steps (run `selfplay1`), against the scripted opponent over training: 50% → 51% (0.6M) → 44% (1.2M) → 42% (1.8M) → 41% (2.4M) → 49% (3.0M, lr at 0). Plain self-play against the latest self drifts away from what beats the script and doesn't improve it. PufferLib's self-play pool (older checkpoints as opponents in part of the games: `--selfplay.enabled=1 --vec.num_policies=2 --vec.hist_policy_percent=0.5`) works with this env; the second agent carries policy tag 1. With it (run `selfpool1`: half the games against a past checkpoint, resampled every 100k steps, from `bclong`, 3M agent steps), the results against the scripted opponent stay at 45-52% (47%, 48%, 52%, 48%, 45%, 49% over training): no drift, but no gain either.
  * Without abilities the plateau is an exploration problem. `pull35` (68%) is focus fire plus pulling hurt units back, and neither half works alone: focus fire 53% (noop 51-54%), pulling back without focus 19% (at 25% hit points). PPO from scratch with every improvement above ends at noop's level (runs `sem6-*`: 47-49%; with tactical masks `semtac6-*`: 42-48%, 97% noop and no retreats): trying either half alone is punished, so it never finds the pair.
  * Tactical mode (`_tac`) tries to make the pull-back discoverable:
    * no plain moves;
    * a retreat only for a unit below half its hit points that is losing them;
    * a chosen retreat goes on, re-ordered every step, while the unit keeps losing hit points (at most 3 s), so one decision is the whole pull-back.

    With the same mechanics `pull35` still wins 68% (a first version with a fixed 1.5 s commitment kept units out too long: 51%). PPO still doesn't learn it. From a fitted focus-fire script (`focus`, 49%, no retreats; runs `focusrl*`) it holds ~50% and drops its retreats from 1-2% to 0%, even with the commitment. A reward for kills and losses (`_kill`, ±0.2 per unit) doesn't change that (48% from the focus clone, 44% from scratch). From the fitted `pull35` (`pullrl2`), PPO keeps its retreats (3% → 2% of unit decisions) and goes from 58% to 65-66% in 3M steps. PPO can value retreats when they come with the rest of the strategy. Pulling back pays because the attack-moving opponent keeps switching targets and chasing, and that happens only when the whole team pulls hurt units consistently. A lone retreat while the rest fight just loses that unit's damage. Isolated exploratory retreats therefore look useless, and the strategy has to come from somewhere else (a script, replays).
  * lr 0.01 (tuned on `footmen2`) is far too high with 15 action heads: the KL per update was 1.0-1.5 (clip fraction 0.9) and the win rate peaked at 29%. The KL at a given lr grows with the number of heads (≈0.15 with 6, 0.3 with 9, 1.0+ with 15); lr 0.003 keeps it at 0.03-0.14.
* `footmen<N>v<M>[_hp<HP>][_ehp<EHP>]`: N agent footmen against M scripted ones with HP hit points each (default 100), the enemies EHP (a handicap).
* `micro`: 4 footmen vs 3 scripted grunts. This is hard: scripted baselines win about 1 game in 3.
* `micro_mirror`: 4 vs 4 footmen against the scripted opponent.
* `selfplay_micro`: 4 vs 4 footmen with both sides served to the trainer as agents of the same policy. `--envs` counts games, so each game gives two agents. The dashboard's win rate is side 0's.

Warm start from a script (behavior cloning, `puffer/bc.py`):
```bash
python -m warcraftsim.puffer.bc collect mirror_mix_sem_hp400 --policy pull35 --episodes 2000 --games 8  # ~5 min
python -m warcraftsim.puffer.bc fit runs/bc/mirror_mix_sem_hp400-pull35        # torch; ~2 min on the GPU
python -m warcraftsim.puffer.bc eval mirror_mix_sem_hp400 runs/bc/mirror_mix_sem_hp400-pull35/policy.bin
python -m warcraftsim.puffer.train --task mirror_mix_sem_hp400 --lr 0.001 \
    --init-from runs/bc/mirror_mix_sem_hp400-pull35/policy.bin ...
```
* `collect` plays a scripted policy (`agents/micro.py`) in games set up like the trainer's and saves the observations, actions, scaled rewards and which unit slots were alive.
* `fit` trains PufferLib's network (linear encoder, MinGRU layers, a linear decoder with the value as its last output) in torch and writes its weight file. It needs torch, which is not in the venv: it runs with `WC3_TORCH_PYTHON` or the first Python that has it.
  * Direction and target heads count only on steps where the unit moved or attacked.
  * Label smoothing (0.1) keeps every choice possible, so PPO can still try what the script never does.
  * A small value weight (0.005) matters. The returns are noisy, and at 0.05 the value took over the shared layers: retreat recall was 0.11 instead of 0.82.
* Results on `mirror_mix_sem_hp400` (noop 54%, the `pull35` script 67-68% at 0.5 s steps; runs `mix40-*` in the dashboard):

    | start | lr, entropy | win rate |
    |---|---|---|
    | random (PPO from scratch) | 0.003, 0.001 | 44% after 3M steps; never learned to retreat |
    | random (PPO from scratch) | 0.001, 0.001 | 47-48% after 2.5M (92% noop, no retreats) |
    | `pull35` fitted exactly (62% sampled, 64% greedy) | 0.001, 0.001 | ~70% throughout (stopped at 0.8M) |
    | `pull35` label-smoothed (33% sampled) | 0.001, 0.001 | 67-69% after 1M, 70% (best 73%) after 2M |
    | `pull35` label-smoothed | 0.001, 0.003 | 62% after 2M (more randomness, no new behavior) |
    | noop label-smoothed (~18% sampled) | 0.003, 0.001 | 42-47% at 1-1.5M (stopped): no better than from scratch, still 25% harmful attacks |
    | noop label-smoothed | 0.001, 0.001 | 49-50% after 2.5M (90% noop, no retreats) |

  Without a good script to start from, PPO converges to letting the units fight on their own (≈ noop); it never discovers pulling hurt units back. From the script, it converges onto the script's behavior (99% of attacks on "weakest", 3-4% retreats) rather than beyond it. Random actions are very costly here: 7.5% of them turn noop's 54% into ~18%, since a random retreat or retarget takes a unit out of the fight for a second or two. Exploration is therefore punished hard.

Notes:
* Action masks: a task can say which options of each action head are possible right now (`Task.action_mask`). The bridge sends them with every observation, and PufferLib samples and trains with them. Micro tasks mask:
  * attacks on empty enemy slots (slot targeting);
  * with abilities, "cast" for a unit that can't cast anything now, and the ability slots that can't be cast.

  Before masks, 94% of the from-scratch policy's casts on `mirror_mix_abil_hp400` were impossible. With them, it reached 39% at 0.9M steps against 35% without.
* Updates per epoch are `replay_ratio × batch / minibatch`. With the minibatch equal to the batch (the old default), there was one update per epoch and learning was slow: `footmen2` reached 61% wins in 1M steps, against 97% with 16 updates.
* PufferLib 5.0 does not normalize advantages, and the micro rewards per step are small. Two settings keep the entropy bonus of the 18 action heads from outweighing the reward and pushing the policy to uniform:
  * `--ent-coef` defaults to 0.001;
  * micro tasks scale rewards by 10 (`Task.reward_scale`; logged returns are scaled too).
* Runs can train concurrently. Each run claims a machine-wide training slot, and its games are named after that slot, so consecutive runs reuse their Wine prefixes.

Each run writes `runs/<name>/`, which the dashboard shows live:
* `run.json`: configuration and status.
* `train.jsonl`: one line per trainer epoch (SPS, losses, win rate), from a small patch applied to the build copy.
* `episodes.jsonl`: every finished episode.
* `renders/`: trajectory animations.
* `replays/` and `videos/`: single-episode replays rendered to real game footage in the background (40 fps).
  * The footage shows which units are the agent's (rings and slot labels A0, A1, ...; enemies E0, ...) and each step's orders: move arrows, attack lines with a crosshair on the target, stop markers, and casts (the ability's name, a line to its target and its area).
  * Every unit has a hit point bar, and a mana bar if it has mana. Under a hero there is a square per learned ability: green when ready, grey filling up during the cooldown, a blue outline when mana is short, a dot for passives.
  * A side panel shows what the policy thought. It evaluates the latest checkpoint before the episode (`puffer/policy.py`, numpy) on the observations the agent saw, and plots:
    * the value V(s) against the discounted return that actually followed;
    * reward and TD error per step;
    * team hit points;
    * each hero's level, mana and learned abilities (level, and ready / cooldown seconds / no mana / passive);
    * per unit, the action probabilities and the sampled action;
    * the policy entropy;
    * an outcome card at the end.
  * In self-play both sides are shown.
  * `scripts/calibrate_camera.py` measures the camera projection the drawing uses.
* `checkpoints/`.

The dashboard shows:
* the run list, with comparison;
* progress cards;
* **Outcomes**: win rate, win/draw/loss, return, episode length;
* **Behaviour** (from the actions the policy sent):
  * the action mix (noop/stop/retreat/move/attack);
  * targeting: focus fire, attacks on the weakest enemy, invalid targets;
  * damage dealt and taken, kills and losses;
* **Learning**:
  * value calibration: predicted V(s₀) against the actual return of video episodes;
  * PPO losses, entropy and clip fraction;
* **System**: throughput, and the trainer's time per epoch split into rollout (waiting for games, model) and training;
* a gallery of replay videos and trajectory renders;
* the recent episodes with their combat and action statistics.

Example: `nav` with 16 games went from a 3% to a 62% success rate within 100k steps (about 3 minutes, at about 550 env steps/s).

## Performance (Ryzen 5950X, 32 threads, WSL2, llvmpipe)

| workload | throughput |
|---|---|
| **Melee, AI vs AI, 2 s steps** | 39x real time (25-minute game in 39 s) |
| **Melee, agent vs AI, 0.25 s steps, full observations** | ≈35x per game (≈140 steps/s) |
| **16 melee games in parallel, 0.25 s steps** | ≈250x real time combined (1000 steps/s) |
| **Scenario skirmish (4 v 4), 0.25 s steps** | ≈160x per game (1.5 ms per step); reset in 11 ms |
| **24 skirmish games in parallel (training setup: 320x240 screens)** | ≈4600 env steps/s (≈1160x real time) |
| **PufferLib training, 24 games (micro_mirror)** | ≈3000 agent steps/s (was ≈620 before this round of work) |
| **Game start** | ≈8 s (map load); 16 games ≈2 min (4 load at a time) |
| **Melee reset** | ≈1 s with the warm spare (≈8 s relaunch without one, or if the episode was shorter than a load) |

Where the time went, and what fixed it (`scripts/bench_env.py` measures env steps/s and the CPU per process
and thread kind; `W3SIM_PROFILE=1..3` adds the shim's per-second profile to each game's `shim.log`):
* **Actions** used to go through a file that the harness loaded with `Preloader`. The engine compiled
  that file as JASS on every step, which cost ~7 ms per step and leaked memory. They now travel in the
  shim's `GO` message; the harness reads them through a hooked native (`GetPlayerTechMaxAllowed` on
  mailbox keys).
* **Observations** used to be written by `PreloadGenEnd` to a file, with file-system calls for every
  token (each a wineserver round trip under Wine, plus a registry lookup of the Documents folder). The
  shim now hooks `Preload`/`PreloadGenEnd` and sends the tokens with the step sync.
* **Polling threads**: the virtual clock divides wait timeouts by its speed, so game threads that poll
  with 100–1000 ms timeouts spun at 5,000–11,000 wakeups per second, each a wineserver call.
  `GameSetup.wait_floor_ms` (1 ms) stops that.
* **Rendering**: training games run on a 320x240 virtual screen (llvmpipe CPU 5.3 → 1.7 cores for 16
  games); replay videos are still rendered at 960x540.
* **Trainer** (`scripts/bench_train.py` runs the trainer in several configurations on the same games):
  * With 2 buffers, the trainer itself burned 10.7 cores, more than 24 games together. OpenMP threads
    spin at the barrier while the slowest game of their buffer finishes; `OMP_WAIT_POLICY=passive` fixes that.
  * Each buffer waits for its slowest game before its next model call, so train.py now uses one buffer
    per two games.
  * Together: 2150 → 2970 agent steps/s at 24 games.

These numbers are for WineHQ **stable** 11.0, which is picked automatically from `/opt/wine-stable` (override with `WARCRAFTSIM_WINE`). Staging 11.18 was about 25% slower in parallel runs.


Training throughput (`mirror_mix_abil_hp400`, 24 games): the games alone step 3,550 times per second (`scripts/bench_env.py`, random actions), training about 1,830. Measured, and not the limit:
* the CPU: 25 of 32 cores are busy;
* the number of games: 36 give 1,871 steps/s, 48 give 1,751;
* trainer buffers: 12 is best (24: 1,758; 6: 1,743; 4: 1,663; 3: 1,559);
* blocking CUDA sync: 1,756;
* the window: 160x120 is 3% faster than 320x240; rendering takes 1-2 cores in all.

The limit is PufferLib's synchronous rollout. Each buffer thread runs its horizon step by step (inference, then its games), and training waits until every buffer is done. So each epoch lasts as long as the slowest buffer: steps take 6.8 ms (p90 9.6), and an episode reset takes 28 ms (max 52).

## Tests

```bash
.venv/bin/pytest                 # unit tests (need the game files, not Wine)
.venv/bin/pytest -m wine         # integration tests: launch real games
```

See `docs/architecture.md` for design notes, limits and what is known about the engine.
