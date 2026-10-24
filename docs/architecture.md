# Architecture

Python drives the real game through a script in the map (the harness) and a DLL in the game process (the shim). The [engine notes](reference/engine-notes.md) have the details.

## The parts

```mermaid
flowchart LR
  subgraph PY["Python"]
    API["GameInstance<br/>env · self-play · BC"]
  end
  subgraph WINE["Wine: one per game"]
    SHIM["w3shim.dll<br/>virtual clock · step sync"]
    GAME["Warcraft III 1.29"]
    HAR["Harness (JASS)<br/>in the map"]
    GAME --- HAR
    SHIM --- GAME
  end
  API <-- "TCP" --> SHIM
```

* **The harness** writes each observation and issues the orders with the game's own functions.
* **The shim** runs the game's clock as fast as the CPU allows and stops the game at each step.
* **The `.wgc` file** starts a game with no menus: the map, the slots and the built-in AI levels.
* **The game's own code, faster** (`shim/cuts.c`, `--game-cuts`): the shim replaces some of the game's functions with equivalent faster ones. Replays prove the same simulation at every step. With `--game-cuts 521721` (2026-10-24) the shim also talks to Python through Linux system calls instead of Wine's sockets, and writes the players' records and checks the game's result itself instead of the harness's JASS.
* **Wine** runs each game in a prefix of its own, with no audio, gamepad or system D-Bus (each prefix's device manager had polled the disk service), drawing into an Xvfb screen of 320×240 with nothing drawn.
* **Other rules** go into a copy of the map as the game's unit and upgrade tables (`data/duelmap.py`): the duel maps, and Echo Isles with duelfast's rules (`echofast`).

## Self-play

```mermaid
flowchart LR
  A["8 actor processes<br/>8 games + 1 evaluation game each"] -- "views (shared memory)" --> I["Inference loop<br/>C++ thread in the learner, CUDA graphs"]
  I -- "orders" --> A
  A -- "trajectories" --> L["Learner<br/>PPO + UPGO, CUDA graphs"]
  L -- "weights" --> I
  A -- "replays" --> R["Renderer<br/>videos with the policy's panel"]
```

* One GPU context holds the learner and the inference loop. The loop has no Python in its hot path.
* An actor turns each game's observation into the policy's view: the observation parsed in C (`native/w3obs.c`), the units' features in C (`native/w3feat.c`, 2026-10-24), the masks and the rest in Python. Its games share one Python process, so its threads take turns on the GIL; an actor uses ~0.8 of a core.
* The video director (`video.Director`) moves the camera in whole-game replays, because 1.29's replays have no automatic camera. It follows the largest fight, raids on buildings too. Between fights it rotates: the learner's army, its base, the opponent's army, its base.

## A remote evaluation worker

A verdict needs ~300 evaluation games a policy. The run's own 8 make ~400 an hour, so another machine plays more of them (`fullgame/evalworker.py`, managed from the run's machine by `scripts/remote_eval.py`). The laptop's 16 make ~790 an hour: a verdict in ~30 minutes.

```mermaid
flowchart LR
  subgraph MAIN["The run's machine"]
    L["Learner<br/>a checkpoint every ~2 min"]
    M["scripts/remote_eval.py<br/>a look every minute"]
    E["the run's episodes.jsonl<br/>+ episodes-remote.jsonl"]
  end
  subgraph LAP["laptop-wsl (WSL2)"]
    W["evalworker: 8 processes of 2 games<br/>their own inference (GPU)"]
  end
  L --> M
  M -- "code, run.json, vocab.json, checkpoints (rsync)" --> W
  W -- "new rows (ssh)" --> M
  M --> E
```

* The worker plays the run's own evaluation games: `selfplay.game_loop`'s evaluation slots and `play_one`, set up from `run.json`'s config. Every policy plays each matchup in turn, as in the run.
* Its "current" is the run's newest checkpoint, a few minutes behind the learner's live weights. The other policies are the run's `--eval-checkpoints`.
* Its draws are seeded by the run's name, the worker's name and where it started: other matchups than the run machine's.
* Its rows are the learner's evaluation rows plus `host` and `version` (the update "current" played with). `real_games.py`, `run_status.sh` and the dashboard read them with the run's own; `real_games.py --split host` shows them apart, `--no-remote` leaves them out.
* The manager runs the worker as a systemd user service there. It starts it again when it died or the machine came back, and stops it when the run ends or another run is live. A restart of the run with other settings for its evaluation games, or a worker started anew, gets this tree's code.
* A machine for it: the repository in `~/warcraftsim` with a venv (torch), GE-Proton's Wine in `~/wc3/ge` and the game in `~/wc3/legacy-1.29` (copied), `scripts/build_native.sh` (StormLib, pjass), and the 32-bit libraries the [engine notes](reference/engine-notes.md) list. The manager copies the shim.

```
scripts/remote_eval.py --host robin@laptop-wsl [--run fgself-76] [--games 16]   # in the background
scripts/remote_eval.py --host robin@laptop-wsl --status | --stop
```

## One step

```mermaid
sequenceDiagram
  participant H as Harness
  participant S as Shim
  participant P as Python
  H->>S: observation (units, players, events)
  S->>P: OBS
  Note over S: game clock stops
  P->>S: GO + orders
  S->>H: orders
  H->>H: issue orders, game runs 0.5 s
```

## Rules to remember

* A reload by `RestartGame` gives a new game. A reset by script does not: the engine keeps counting removed heroes.
* Several pairs of players in one map load (`--pairs`) work only on the duel maps. Echo Isles has two start locations: one pair.
* A build order that does not fit is tried again on rings of spots up to 768 units away, in the same step (up to 256 until 2026-10-16: Echo Isles refused most of undead's builds).
* The game runs only as the 32-bit Legacy 1.29 build. The shim's offsets are for that build.
