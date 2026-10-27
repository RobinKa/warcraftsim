# People's replays

Two tools for replays (`.w3g`) of games people played:

* **A parser** (`warcraftsim/data/w3g.py`, `scripts/replay_stats.py`) reads replays of any version: the players, the map and every action. From the actions it makes each player's build order and APM. This part needs no game.
* **A converter** (`warcraftsim/fullgame/replays.py`) plays a 1.29 replay on a stock map back in our client and records the built-in AI's kind of demonstrations from it. `bc.py` reads them unchanged.
* **A fetcher** (`scripts/replay_fetch.py`): warcraft3.info's index and downloads, W3Champions' ladder 1v1s ([sources](#sources-and-versions)).

```mermaid
flowchart LR
  W["replay .w3g<br/>stock map"] --> P["w3g.py<br/>build orders, APM, winner"]
  W --> R["rewrite: map checksums,<br/>object ids shifted"]
  M["Echo Isles +<br/>w3obs.j (observes only)"] --> G
  R --> G["our client<br/>replay build of w3shim"]
  G --> D["demonstration .npz<br/>units, players, orders, events"]
```

## Run it

```
python scripts/replay_fetch.py index --out runs/fullgame/replay-survey/w3info-index.jsonl --version 1.29
python scripts/replay_fetch.py download --index runs/fullgame/replay-survey/w3info-index.jsonl \
    --out runs/fullgame/human-replays-128 --version 1.28 --type 1on1    # files: untrusted data, own directory
python -I scripts/replay_stats.py runs/fullgame/human-replays-129       # rows in <dir>/stats.jsonl, a summary
make -C shim replay                                                     # the replay build of w3shim (build/shim-replay)
python -m warcraftsim.fullgame.replays --replays runs/fullgame/human-replays-129 \
    --out runs/fullgame/demos-human-129 --parallel 2                    # 1v1s on Echo Isles at build 6060
python -m warcraftsim.fullgame.replays --replays runs/fullgame/human-replays-129 \
    --out runs/fullgame/demos-human-129-maps --map "(2)LastRefuge-1.3,(2)Amazonia,(2)TerenasStand_LV"  # other stock maps (or --map all)
python -m warcraftsim.fullgame.bc --data runs/fullgame/demos-human-129 ...
```

* The converter skips what it can't play and gives the reason in `<out>/skipped.jsonl`. It resumes where it stopped (`--retry-failed` tries the failures again).
* A game takes about 66 s for both playbacks, two at a time beside the training (niced): the 95 games took 52 minutes. The games are named `replayconv*`.
* A game whose playback is out of step with the replay goes to `<out>/desync` (`--min-sync`).

## The parser

* **Formats:** the classic layout (16-bit block sizes), Reforged's (32-bit block sizes, protobuf records before the slots, 2.0's action `0x7A` before an order) and the GRBN container (protobuf metadata, then a classic replay; warcraft3.info serves NetEase's replays in it). It also reads the older action layouts (before 1.07, 1.13 and 1.14b). Checked on 1.26, 1.28, 1.29, 1.30 and Reforged 2.0 replays: every action known (1.30-1.31 have an action 0x77 of 12 bytes, twice a game, that Reforged uses otherwise).
* **What it reads:** the header, the game info (host, game name, settings, map path, the map's checksums), players and slots (team, colour, race, computer and level, handicap, observers), chat, leave records, and every action with its time. Actions name units by object ids: a slot in the engine's table of objects and the count of objects made before it.
* **Build orders** (`w3g.summarize`): per player, the buildings placed (with the spot), units and heroes trained, upgrades, researches, hero skills, items bought and cancels, with their game time. `w3ids.json` (from the 1.29 game data, `replay_stats.py --ids`) tells which code is which. A random player's race is the one its starting units show.
* **Strategy statistics** (`w3g.strategy`, in `stats.jsonl`): as AlphaStar's conditioning, the first 20 buildings, units, upgrades and researches (workers left out) and the counts of everything made.
* **Noise:** the same order again within 1.5 s counts once (a double click, a spammed key), and so does a building placed again at the same spot. A replay does not say whether the game took an order. So a train click refused for gold, food or a full queue still counts, and a hero's revival looks like a second hero of the type.
* **APM:** orders, adding to the selection, hotkeys, queue cancels, ESC and the skill and build menus, per minute until the player left.
* **Winner:** a "won" leave record, else the side still there when every other player left. It agreed with warcraft3.info's winner in 168 of the 176 games that have one.

## The downloaded replays

2,591 replays from warcraft3.info, all 1.29 build 6060, all parsed (2026-10-17).

| | replays |
|---|---|
| containers | 1,857 GRBN (NetEase), 734 classic |
| types | 2,543 1v1, 47 2v2, 1 6v6 |
| maps | Last Refuge 464, Amazonia 411, Echo Isles 350, Turtle Rock 311, Twisted Meadows 294, Terenas Stand 278, Ancient Isles 218, ... |
| 1v1 on Echo Isles | 350: 95 on the stock map (convertible), 249 on NetEase's own version (map checksum 94b4de8e), 6 others |
| 1v1 length | median 15 min (10-90%: 9-24) |
| 1v1 APM | median 260 (10-90%: 188-331) |

## Playing a replay back

A replay holds only the players' actions. The game re-simulates it from the map, so the map must be the one the game was played on, and nothing may change the simulation.

**The map.** The replay names its map (`Maps\FrozenThrone\(2)EchoIsles.w3x`), and 1.29 looks for it in `Documents\Warcraft III`. It finds it only if its checksums are the replay's: a modified map is "not found". The checksums are the classic one and a SHA-1 over `common.j`, `Blizzard.j` and the map's script and terrain files (GHost++'s algorithm, `w3g.map_checksums`). `w3g.rewrite` writes the observer map's checksums into a copy of the replay. 1.29 checks the replay's own checksums too: the header's CRC-32, and per block the CRC-32s of its header and its data, each folded to 16 bits.

**Object ids.** Actions name units by object ids: a slot and the count of objects made before (at `+0xc` and `+0x10` of every engine object). Every object the map's script makes takes the next id, so units made later get other ids than in the recorded game, and the replay's orders reach nobody. One `TimerStart` with a callback in an otherwise stock map was enough: the bases stopped growing after a minute.

* What makes objects: triggers, timers, groups, hashtables, regions, rects, events, a timer's handler (its first `TimerStart` with code) and a callback's wrapper (its first use). A trigger's event makes objects each time it fires, even with no condition or action.
* What doesn't: reading units and players, `Preload` and the mailbox sync, enumerating into a group made before.
* The observing harness (`harness/w3obs.j`) makes all of its objects at the end of `main`, after the map's own melee initialization, between two probe groups. It registers no events. A short first playback reads the probes' ids (16 objects). Every later object id in the replay is then shifted by their count: the count always, the slot if it is a new one (`replays.shift_of`). The 8 bytes after an order's id are an object id too (the item a "useslot" uses): a Rod of Necromancy raised no skeletons until they were shifted as well.

**The replay build of w3shim** (`shim/replay.c`, `make -C shim replay`) reports each unit's object ids, the probes' and the trees'. It also adds the engine's own state of buildings to the unit records, found by comparing the bytes of an altar, a barracks, a ziggurat and a great hall over their construction, training and upgrade:

| flag | what | where |
|---|---|---|
| 8 | under construction | `+0x164 & 0x2000000` |
| 4096 | producing (training or researching) | `+0x3c4` |
| 8192 | being upgraded | `+0x164 & 0x4000000` |

The default build has none of this (`#ifdef W3SIM_REPLAY`).

## The demonstrations

The arrays and files of `collect.py`: units, heroes, players, events, orders and trees per game, plus `games.jsonl` and `collect.json`. Differences:

* **Players** are 0 and 1 in slot order (in the game they are their slots), also in the visibility bits. `difficulties` are `["human", "human"]`. `meta.source` names the replay and has the players' APM.
* **Orders** (the labels) come from the replay's actions. Each player's selection is followed through clicks, groups and subgroups, and its units are found by their object ids. Orders every unit takes (smart, move, attack, patrol, hold, stop) go to every selected unit. A build order goes to one worker: the one walking to that building the next step. Train, research and upgrade orders go to one building of the active subgroup, one not yet producing if there is one. Other abilities go to the subgroup's units, and hero skills to its hero (kind 3). Targets are units and trees; a tree out of sight is found by its type and spot.
* **Events** are derived from the units' states step by step: construction start, finish and cancel; production start, finish and cancel at a building (from its producing flag and the units of its owner that appear beside it); upgrades; deaths (no killer); hero levels; summons (no summoner); items picked up. Research levels are polled by the harness, and the research is put at the building last ordered to research it.
* **Result:** from the leave records (above).

## Verified

All 95 Echo Isles 1v1s on the stock map converted (2026-10-17): 23.7 hours of play, median 13.8 minutes, APM median 226. None was out of step.

| check | result |
|---|---|
| units the players' subgroup actions name (type and object ids) that are that unit in the playback, at that time (`meta.sync`) | **9,508 of 9,508** |
| buildings the replay places that came up within 64 of the spot | 2,892 of 2,982 (97%; the rest: orders the game refused) |
| how far the playback ran | to the first player's leave, give or take a second; when a side's defeat ended the game for the viewer, up to 72 s before |
| researches the harness saw done, against the replay's research clicks (8 games) | 59 of 63 (the rest cancelled or unfinished) |
| build labels: the worker went to build (or the building came up at once) / another selected worker went / none | 374 / 7 / 14 of 395 (10 games) |
| `bc.py`'s encoder (`features.load_game`, both sides, `side_data`, `sides_of`) | every game reads: 341,220 side-steps, 48% with labels (the AI's: 23%) |

* Before the probes, one passive `TimerStart` was enough to stop every base after a minute. With the harness's objects shifted, a Rod of Necromancy's skeletons were the first miss: their object ids came up in the stock playback (no harness, read from the engine) and not in ours, until the item ids in orders were shifted too.
* The players' food, gold and build orders go as in a real game. People against the built-in AI on Echo Isles (`demos-echo-1`, normal and insane), medians:

| race | workers at 5 / 10 min | food at 5 / 10 min | first hero | tier 2 |
|---|---|---|---|---|
| human | 13 / 14 (AI 16 / 16) | 32 / 44 (AI 40 / 49) | 123 s (121) | 468 s (506) |
| orc | 12 / 12 (14 / 14) | 29 / 46 (34 / 43) | 123 s (128) | 283 s (610) |
| undead | 5 / 5 (5 / 7) | 28 / 44 (32 / 53) | 136 s (150) | 332 s (352) |
| night elf | 12 / 15 (17 / 18) | 27 / 44 (35 / 49) | 122 s (125) | 328 s (344) |

## Limits

* Only replays of our client's build (1.29.2, build 6060) play back, and only on a stock map of the game (`Maps/FrozenThrone`, its `Community` folder too) whose checksums the replay has ([other versions](#other-versions)). NetEase's 1.29 games are on NetEase's own map versions ([sources](#sources-and-versions)).
* `bc.py` takes the map frame (`features.set_geometry`) from its first game: demonstrations of several maps need it per game. On a four-player map (Turtle Rock, Twisted Meadows) a 1v1's two bases are any two of the four starts, and `map_geometry` takes the first two.
* The labels are the players' commands spread over their selection, not the game's issued orders: which worker builds and which building trains is a choice, and a command the game refused is still a label.
* The events are reconstructions: the building a unit came from is the nearest producing one, research and train starts come from the last order a building got, and there are no spell, tree-death or killer events.
* Units the harness never saw are missing: it adds units by enumerating the map each step, which skips units that are hidden from the moment they are made.
* Each replay plays twice (the probe playback is short: about 10 s).
* The game's own events and issued orders would need the shim to hook the engine's event dispatch (no script objects): a next step if the derived labels and events fall short.

## Sources and versions

Surveyed 2026-10-26 (`runs/fullgame/replay-survey/`: `w3info-index.jsonl`, warcraft3.info's metadata of every replay of 1.26-1.31).

**warcraft3.info** (public API, no login; robots.txt allows `/api`): 140,250 replays. The list takes a `version` filter (`{"page": N, "version": "1.30"}`); a row has the map's checksums (the classic one byte-swapped against the replay's) and the origin (Battle.net or NetEase).

| version | replays | 1v1 | origin of the 1v1s | 1v1s on a map file identical to ours |
|---|---|---|---|---|
| 1.26 | 6,692 | 6,359 | Battle.net (W3Arena's map versions and stock maps) | 1,535 (Turtle Rock 565, Echo Isles 381, Terenas Stand 228, Secret Valley 194, Twisted Meadows 126, ...) |
| 1.27 | 546 | 331 | NetEase 317 | 5 |
| 1.28 | 4,032 | 4,030 | NetEase 4,006 | 5 |
| **1.29** | 2,605 | 2,557 | NetEase 1,857, Battle.net 700 | **608 convertible** (checksums match): Last Refuge 158, Amazonia 124, Echo Isles 98, Turtle Rock 87, Terenas Stand LV 61, Twisted Meadows 57, others 23 |
| 1.30 | 4,036 | 4,004 | NetEase 2,764 | 15 (1.30 changed most ladder maps) |
| 1.31 | 9,998 | 9,309 | Battle.net 4,576, NetEase 4,733 | 0 |
| 1.32-1.36 (Reforged) | 75,615 | | | (another engine) |
| 2.0 / 3.0 | 29,802 / 2,585 | | NetEase, Battle.net | |

"Identical": the replay's classic checksum with our map's files peeled off leaves the same value for several maps of a version (the version's `common.j` and `Blizzard.j`), so those map files are byte for byte ours. The 1.29 NetEase games (1,816 1v1s) are on NetEase's map versions (Maps/1.29: Last Refuge a39562b3 291, Amazonia 23d68f0b 276, Echo Isles 94b4de8e 249, Twisted Meadows c43206f3 218, Terenas Stand LV 1a181b47 214, Ancient Isles 56381717 209, Turtle Rock 6a9d1f28 203, ...): convertible only with those map files (not served; not stock). Ancient Isles is not a 1.29 stock map at all.

**W3Champions** (Reforged ladder, public backend API, robots.txt allows all): `GET website-backend.w3champions.com/api/matches?offset=&pageSize=100&gameMode=1&season=N` lists the 1v1s with both players' MMR (season 0 2020: 139k; seasons 20-25 2025-26: 330k-690k each; about 9.3 million 1v1s in all). `GET /api/replays/<match id>` serves the replay for some months (2026 games yes, most 2021-2025 ones 404/410). The replay endpoint answers 429 after ~20 downloads in a row: bulk downloads need W3Champions' blessing. Reforged 2.0, build 7003: parse-only.

**Others:** wc3replays.com is an analyzer, not an archive (and disallows `/api/`); gamereplays.org has old-patch replays (not checked); wcreplays.com and replays.net no longer answer. No public replay dataset (Kaggle, GitHub, Zenodo) turned up.

## Other maps

A pilot of 54 1.29 games on Last Refuge (28), Amazonia (23) and Terenas Stand LV (3) converted (2026-10-26, `runs/fullgame/demos-human-129-maps`): **none out of step**: 5,760 of 5,760 subgroup units, 1,795 of 1,834 buildings (98%); one playback timed out (`--retry-failed`). 14.1 hours of play, 202,756 side-steps, each game read through `bc.py`'s encoder in its own map's frame. 82 s a game per playback slot (two at a time beside the training, load ~90 on 32 threads): the other ~460 convertible 1.29 games take about 5 hours at two at a time (~10 slot-hours).

## Other versions

Playback in our 1.29.2 client (`replays.py --any-version`: the header set to 1.29 build 6060, the map found by its name; a test):

| version | what happened | why |
|---|---|---|
| 1.26, 1.28 (on map files identical to ours) | no order reached its unit; with the starting units' ids aligned (`version_shift`: 1.29 makes ~480 more objects before them, 24 players) the start is in step, then the first new building's ids miss within 3-6 s | the engines make different numbers of objects as the game runs: not one offset |
| 1.29 on NetEase's map versions (`--other-map-versions`, 2026-10-27) | Last Refuge, Terenas Stand: the start in step once the ids move by 3 (`version_shift`: the trees keep theirs; mines, creeps and starting units matched by position and hall), then the first new building's ids miss within ~7 s: sync 11/67 and 17/82 units, 5/24 and 10/27 buildings. Echo Isles, Turtle Rock, Twisted Meadows, Amazonia: the playback never starts (10 of 10: "did not reach the harness") | NetEase's map script makes 3 (4-player maps: 5) more objects before the preplaced units, and objects as the game runs (made and freed: the slots stay in step, the counts drift): 156 more within the first 4.5 s in every game, then more by what happens (+3 to +276 by 8-31 s): not one offset. Echo Isles, Turtle Rock and Twisted Meadows make ~96 more objects before the starting units, Amazonia's preplaced units come in another order: other map files. Ways in: resolve the replay's ids by slot alone in the replay build (the engine's lookup: not found yet), or NetEase's map files (not public) |
| 1.30 (Flood Plains 1v1 LV, identical map file) | the starting units in step; the first miss at ~10 s (an undead crypt) | balance: 1.30.0 made acolytes faster (220 to 250) and changed a few hundred values |

What the versions change (for playback, and for build orders as targets):

* **1.26-1.28**: no balance patch (1.27/1.28: launcher, Battle.net, fixes); the engine for 12 players. 1.29.0 (2018-04) changed heroes only: speed 270 to 290 for most, many hero skills, and four mechanics (Critical Strike no longer multiplies item damage, Carrion Swarm hits mechanical units, Healing Spray spares enemies, Cluster Rockets hit air). Object data could undo the hero values, not the engine's object counts. **Build orders only** (their unit timings are 1.29's; hero speeds and skills differ).
* **1.30.0-1.30.4** (2018-08 to 2019): a large balance patch (costs, HP, damage, build times: Necropolis 100 to 90 s, Mountain Giant 425 to 350 gold, Siege Engine 4 food, ...), tech-tree changes (Tauren Totem at Stronghold trains Spirit Walkers, Flak Cannons at Keep, Sentinel and Hippogryph Taming free then a research again in 1.30.2, Skeletal Longevity removed), engine fixes (Defend, dropped items), new stock map versions; the header's build (6061) does not tell 1.30.x apart (their balance differs: 1.30.1, 1.30.2). Most of it object data could express, but only with each sub-patch's exact data (we have none of their game files) and never the engine fixes. **Build orders only**: drop removed research, rescale per version.
* **1.31** (2019-05): new abilities and items (Incite Unholy Frenzy, Ritual Dagger, Sundering Blades), Critical Strike on all damage, Windwalk cooldown, Orb of Fire as anti-heal, changed drop tables. **Build orders only**.
* **1.32-2.0 (Reforged)**: another client and replay format, a balance patch every few months, a new map pool. **Build orders only**: keep the items of 1.29's vocabulary (`strategy.py --vocab`), and take timings relative to each version's own medians.

`strategy.py --replay-patches 1.28 1.29 ...` picks versions; strategies of other versions carry `"patch"`. Build times are 1.29's (`build_seconds`).

