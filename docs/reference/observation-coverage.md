# Observation coverage

What a person playing a 1.29 melee game can see or know that the agent's observation lacks: the counterpart of [order-coverage](order-coverage.md) for the inputs. Phase 1 (2026-11-06): the list, the gaps ranked, how each would be added. Phase 2 built the first eight behind `--obs-full 1` ([below](#built---obs-full-1-phase-2-2026-11-06)); the tables after it describe the inputs without the flag.

Checked two ways:

* **The game's tables** (`scripts/observation_coverage.py`): the buffs every melee unit, hero, the tavern's heroes, the mercenaries, the maps' creeps and the items can put on a unit; the items a hero can hold; the researches; the non-hero cooldowns and toggles; the unit types without an embedding. Its output is the generated part at the end. `--doc` rewrites that part.
* **The real game** (`scripts/obs_probe.py`): four games (human vs orc, orc vs night elf, undead vs human, undead vs night elf; echofast2, one at a time). The harness gets one more function that asks the natives about every unit each step. Spells are cast on placed units, the units fight, and the corpses are watched. Results are under [What the game exposes](#what-the-game-exposes).

The settings are fgself-194's (`runs/fgself-194/run.json`): `--inputs 2`, `--features 29`, `--neutral 2`, `--macros 3`, `--spells 1`, `--shop-items 1`, the tech, valid and economy masks, and `--memory 1` (the minGRU).

## Built: `--obs-full 1` (phase 2, 2026-11-06)

`selfplay --obs-full 1` (needs `--neutral`; `fullgame/obsfull.py`) adds what the ranked list below asks for, as a block of 70 channels at the very end of each entity's features (`features.A_OBS`) and three new entity kinds after the units. Off (the default), nothing runs: the harness's new code is behind `W3S_CFG_OBS_FULL`, the shim's behind the harness's call, and the features and the network are as before.

| gap (rank) | with `--obs-full 1` | from |
|---|---|---|
| buffs and debuffs (1) | per unit in sight: the 8 classifications (stunned, snared, poisoned, polymorphed, sleeping, ethereal, magic immune, plagued), its positive and negative magic buffs, armor and move speed now, move speed over its type's, and **every buff on it** as a sum of fixed 24-number codes, one code per buff id | the shim reads each unit's buff list from the game's memory (`shim/units.c` `o_buffs`: the list `0x6d6570` walks; the codes as each ability's `BuffID` spells them, so Unholy Frenzy is `BUhf`); `IsUnitType`, `UnitCountBuffsEx`, `BlzGetUnitArmor`, `GetUnitMoveSpeed`. Event 101 and its 102s, written only when they change. Defend sets no buff: its state shows in the speed (188 of 270). |
| corpses (3) | entities, kind "corpse": the dead unit's type and owner, its place, seconds since its death; the 8 nearest own units of those in sight | the harness keeps each dead unit (not a hero, building, summon or image) in a group of its own: event 106 at its death, 108 when its place's visibility changes, 107 when it is gone (decayed, raised, eaten, carried) |
| enemy upgrade levels (4) | per unit of either player in sight: its owner's weapon and armor upgrade level for its type, what its tooltip shows. An enemy's come from the research its buildings finished; an enemy out of sight shows nothing | the research-finished events already in each observation (`View` had dropped the enemy's); `UnitBalance`'s upgrades per type and `UpgradeData`'s effects (`ratd` weapons, `rarm` armor) |
| items (5) | a hero's items as a sum of fixed 16-number codes (all 56 apart, the 6 kinds before), each slot's charges; items on the ground as entities, kind "item" (their code, a powerup flag) | the hero's record's item types; event 105 (`GetItemCharges`); the `G` records |
| area spells in progress (6) | entities, kind "area": the spell's code, its point, radius, seconds left, the caster's side, for 14 lasting area spells (blizzard, flame strike, death and decay, earthquake, rain of fire, volcano, starfall, tranquility, stampede, locust swarm, cloud, monsoon) | events 104 (the spell's point, `GetSpellTargetX/Y`) and 109 (the ability, the seers that see the point); durations and radii from `AbilityData` |
| production and construction progress (7) | own buildings: the current item's progress and seconds left; any building under construction in sight: its progress | the start and end events already in the observation, `UnitBalance` / `UpgradeData` times; the map's pace (echofast2 builds faster than the tables) learned from the items that finish, the median of the last 15 |
| creep types (8) | the 7 Echo Isles creeps that read as unknown get types (and every creep of the five maps the runs play). Their embeddings start as unknown's: every token is as before | `obsfull.extend` / `grow` (`--full-vocab` adds the races', the tavern's and the mercenaries', not the creeps') |

The new entity kinds get an attention bias of -30 per kind (`FullGameNet.kind_bias`, a buffer in the state dict, and `config["obsfull"]`): the network plays exactly as it did. `--obs-fade N` raises the learner's to 0 over its first N updates, as `--memory-fade` did for the remembered enemies; the anchor and older networks keep -30. No order of the vocabulary takes a corpse, an item lying or an area (the pick-up order is a point's), so the target pointer never points at them (`target_logits(..., ent)`).

**Identity** (`tests/unit/test_obsfull.py`, fgself-182's checkpoint): with the flag, any values in the block and five new-kind entities, `model.evaluate` gives the same order logits, log-probabilities, value, point logits and pointer logits as the network without the flag, to the last bit (max |difference| 0.0). The actors' path (`FullGameNet(**config)` and the state dict) gives the same, bit for bit. With the kind bias raised to 0 the value moves (0.025): the bias is what hides them.

**Cost**, measured on the CPU (`scripts/obs_probe.py cost`, `scripts/obs_full_bench.py`):

* The game: insane against insane on echofast2, 700 steps each, two games each way. 8.07 ms of game thread a step off, 7.95 on: below the noise (±0.5 ms). ~2.2 events of kinds 101-109 a step: the shim writes a unit's only when they change.
* The observation path: `View.step` with the masks, per side-step, on a loaded machine (load ~30-46 on 32 cores): 400 → 634 µs at its quietest, 700 → 1,000 µs at its busiest, about +50%. `obsfull.apply` itself is 120-220 µs of that. The rest is the wider rows (176 channels instead of 106).
* The network: `ent` 106 → 176 inputs (+0.6% of a token's work); the new kinds add +2.7 entities a step (AI against AI, corpses mostly); a real game played by fgself-182's network with the flag saw 17.6 corpses in sight a step, so they are capped at the 8 nearest own units. The rollouts' entity rows are 66% wider: memory in the learner's batches (the GPU's) grows with them.

**Checked in the real game:** `scripts/obs_probe.py full` (`W3SIM_OBS_CHECK=1`: each buff in a unit's list against `GetUnitAbilityLevel`, and 24 common buffs the other way round). No mismatch. The lists held what was cast (`BUfa`, `BUhf`, `Bam2`, `Bfae`, `BEer`, `Bcrs`, `Bcri`, `Broa`, `Brej`, `Bcy2`, `BEah`, `Brai`). `scripts/obs_probe.py play` has fgself-182's network play insane on echofast2 through the actors' path with the flag.

**Not built here:**

* **The 26 zeroed channels** (hp change, the position's sines and cosines, the last order): fgself-182's weights for them are exactly zero, so `--features 55` (or no `--features`) turns them on with the network's outputs unchanged until it trains. They are a separate setting, not part of `--obs-full`: they did not help in fgself-21, and a run changing both could not tell which helped.
* **Fog memory** (`--inputs 3` hurt twice). What it would need instead:
  * a look at what the policy did with the remembered enemies first;
  * then the explored grid (`IsMaskedToPlayer` on 24 × 24 points every 8 steps, ~0.04 ms a step, as global numbers or a channel per remembered entity), behind the same kind of attention bias as here.
* **The map's terrain**: a spatial input (a coarse minimap image through a small CNN) or static map tokens. It matters for a map pool, not on one map.
* **Non-hero cooldowns and autocast state** (rank 9):
  * cooldowns: `BlzGetUnitAbilityCooldownRemaining` works on unit abilities (probe);
  * autocast: a bit of the ability object that a probe still has to find.

## What the agent reads today

* **The harness's records** (`warcraftsim/harness/w3sim.j`; the unit pass in C in `shim/units.c`; parsed by `protocol.py` / `native/w3obs.c`):
  * `U`: per unit its type, owner, x, y, facing, hp and max, mana and max, current order, flags (hero, building, worker, constructing, hidden, loaded, sleeping, paused, summoned, illusion, dead, flying), the players that see it, and a gold mine's gold. Heroes also send level, xp, skill points, six item types, and per ability slot its level and cooldown.
  * `P`: gold, lumber, food used and cap, upkeep, gathered, structures, result, start.
  * `E`: events (deaths, construction, training, research, upgrades, hero levels, spell effects, tree deaths, summons, item pick-ups) for every player.
  * `G`: items lying on the ground in sight.
  * `D`: trees, in the first observation only.
* **The network's inputs** (`fullgame/features.py`):
  * Per entity, 106 channels. `--features 29` zeroes 26 of them: the hp change, the position's sines and cosines, and the last order (they did not help in fgself-21). Each entity also gets a type embedding (162 types) and a current-order embedding (156 orders).
  * The entities: own units, then the enemy's and the neutrals' **in sight**, plus the shops always (`--neutral 2`). Dead units leave the table at their death, except heroes.
  * The global token: resources, supply, upkeep, the step, both races, 66 own research levels, the strategy block and the clock.
  * The value head alone gets a privileged block, the opponent's economy and army. That is right: a person doesn't see it either.

## Coverage, by group

Legend: **yes** is an input; **partly** says what is missing; **no** is not an input. The last column says how the game exposes it: a native (checked in the probe), memory (the shim would read the unit's object), derivable (from what the agent already gets plus the tables), or none.

### Own units

| what a person sees | agent | where / what is missing | game exposes |
|---|---|---|---|
| hp, max hp; mana, max mana | yes | channels 5-8 | `GetWidgetLife`, `GetUnitState` |
| a unit being hit (health bar dropping, "under attack") | partly | `A_HPD` (29) computed but zeroed by `--features 29`; the minGRU can infer it | derivable |
| position, facing | yes | 3-4, 23-24 (the Fourier position 30-49 is zeroed) | |
| current order | yes | order embedding; an id outside the vocabulary becomes 0 | `GetUnitCurrentOrder` |
| the order's target (point or unit) | partly | the last order the agent gave (`A_LAST`, 50-54) is computed but zeroed; the game's current target is not read | no 1.29 getter; the harness sees the orders it issues |
| attack: interval, damage, range, attack and armor type | via the type | the type embedding; the probe read base values only | `BlzGetUnitAttackCooldown(u, 1)`, `BlzGetUnitBaseDamage(u, 1)` (weapon index 1 is the first weapon in 1.29) |
| time to the next swing | no | | memory only (the attack ability at unit+0x3f4) |
| armor now (upgrades and buffs) | partly | own upgrade levels are in the global vector; buffs and auras are not | `BlzGetUnitArmor` (probe: footman 2 → 8 with Rhar 3 → 13 with inner fire) |
| move speed now (slow, ensnare, bloodlust, defend, auras) | no | | `GetUnitMoveSpeed` (probe: 270 → 0 ensnared, 150 slowed, 337 bloodlust, 188 defending, 125 sheep) |
| buffs and debuffs (bloodlust, inner fire, slow, ensnare, web, sleep, polymorph, hex, curse, frost, poison, faerie fire, cyclone, banish, divine shield, anti-magic shell, auras...) | **no** | 160 buffs, none encoded (table below) | `GetUnitAbilityLevel(u, buff)` per buff; the `IsUnitType` classifications (stunned, snared, poisoned, polymorphed, sleeping, ethereal, magic immune, plagued); `UnitCountBuffsEx`; memory (the unit's ability list) |
| an own unit is invisible (wind walk, invisibility, shadow meld) | no | | the buffs (`BOwk`, `Binv`, shadow meld) |
| polymorphed or hexed | no | the type stays (probe: a sheep reads `ogru` at full hp) | `UNIT_TYPE_POLYMORPHED`, `Bply`, `BOhx` |
| stunned, asleep, snared, ethereal, magic immune | partly | only a creep's sleep (15) and the paused flag | `IsUnitType` classifications |
| invulnerable (divine shield, potion) | no | | `BlzIsUnitInvulnerable` is broken in 1.29 (true for every unit); buffs `BHds`, `Bvul`; memory |
| hero ability level, cooldown, castable, skill point | yes | `--inputs 2`, 55-70 | |
| non-hero ability cooldowns (ensnare, purge, berserk, web, curse, raise dead...) | partly | the full vocabulary's masks track them from the agent's own orders | `BlzGetUnitAbilityCooldownRemaining` works for unit abilities (probe: ensnare 11.1 s after a cast) |
| autocast and toggle state (heal, slow, curse, web, raise dead, defend...) | partly | the mask tracks `fullvocab.DEFAULT_ON` and the agent's own orders | no native; memory (a bit of the ability object, still to be found) |
| hero level, skill points | yes | 20, 22 | |
| hero xp toward the next level | no | in the record (`C_HXP`), not a feature | `GetHeroXP` (sent) |
| hero attributes (str, agi, int) | no | | `GetHeroStr` etc. |
| inventory | partly | 5 numbers per slot (held, usable, heals, mana, town portal): 56 items, 6 kinds | `UnitItemInSlot` (sent) |
| item charges | no | | `GetItemCharges` (probe ok) |
| item cooldowns | no | | memory |
| a worker carrying gold or lumber | partly | "sent to lumber" (28); the load is not encoded | memory |
| construction progress | partly | the constructing flag (12); hp grows while it builds | derivable (`CONSTRUCT_START` time + build time) |
| production queue | partly | count queued (26), busy (27), and the kind it makes (101-105); not the unit type or the progress | derivable (`TRAIN/RESEARCH_START` + `UnitBalance`/`UpgradeData` times) |
| rally point | no | | `GetUnitRallyPoint` (probe ok) |
| transported units | partly | the loaded flag on the cargo; not which transport | `IsUnitInTransport` |
| summon's time left | partly | the summoned flag (17) | derivable (`SUMMON` event + duration) |
| own illusion | yes | 18 (the enemy's is zeroed on purpose) | |

### Enemy units in sight

| what a person sees | agent | where / what is missing | game exposes |
|---|---|---|---|
| hp, mana, position, facing, current order | yes | an enemy worker's build order is zeroed (the game doesn't show it) | |
| hero level, ability levels, items | yes | as the game shows them; no cooldowns (the game shows none) | |
| buffs and debuffs (what to dispel or purge, who is snared, banished, a sheep) | **no** | | as for own units |
| upgrade levels (the attack and armor icons' levels when you click a unit) | **no** | the enemy's `RESEARCH_FINISH` events are in the stream, and `View` drops them | the events; `GetPlayerTechCount(enemy, ...)` (probe ok); `BlzGetUnitArmor` |
| invisible enemies | yes (correctly unseen) | probe: an invisible footman 170 from a ghoul stays out of the enemy's visibility bits; detected ones are seen | |
| illusions | yes (correctly hidden) | the flag is zeroed for enemies | |
| units seen before, now in the fog | **no** in fgself-194 | `--inputs 3` (ghosts) exists, but fgself-107 and -117 were weaker with it | derivable |

### The map

| what a person sees | agent | where / what is missing | game exposes |
|---|---|---|---|
| what is in sight | yes | entities in sight; no visibility map | |
| explored (black mask) vs fog | **no** | | `IsMaskedToPlayer`, `IsFoggedToPlayer`, `IsVisibleToPlayer` per point (probe ok) |
| enemy buildings last seen (drawn in the fog) | **no** in fgself-194 | `--inputs 3` | derivable |
| trees, cut trees, lumber left | no | only as targets for harvest orders (the nearest tree to a point); 1,588 trees of 50 lumber each, then `TREE_DEATH` | `GetDestructableLife` (= lumber) |
| gold mines' gold | yes in sight | 21 | `GetResourceAmount` |
| gold mines and shops out of sight (on the minimap from the start) | partly | the shops are always entities (`--neutral 2`), the mines are not; the map is fixed | |
| creep camps in sight, their creeps and levels | partly | neutral entities; **7 of Echo Isles' 15 creep types have no type embedding** (murlocs, ogre magi, kobold tunneler, sea turtle, renegade wizard): their type is "unknown" | |
| creep camps out of sight (the minimap's creep display, TFT) | no | the camps are fixed on one map; not checked in the probe | |
| creep respawns | n/a | none in melee | |
| tavern heroes, shop and mercenary stock, restock timers | partly | masks only (`TechMask.stocked`, `fullvocab.lab_stock`) | no 1.29 native (order-coverage: the marketplace); memory |
| items on the ground (creep drops) | partly | the `G` records reach only `--macros`' pick-up mask | `G` records (sent) |
| corpses | **no** | dead units leave the agent's table at death | enumerations return dead units for ~92 s (probe) |
| blight | no | | `IsPointBlighted`; derivable from undead buildings |
| time of day | yes | `--inputs 2` global; computed from the step, 0.2-0.5 game hours off the game's in the probes | `GetFloatGameState(GAME_STATE_TIME_OF_DAY)` |
| pathing, cliffs, water, ramps | no | the map is fixed (learned implicitly); matters for a map pool | `IsTerrainPathable`, `GetTerrainCliffLevel` |

### Players

| what a person sees | agent | where / what is missing | game exposes |
|---|---|---|---|
| gold, lumber, food used and cap, upkeep | yes | global 0-4 | |
| own research levels | partly | 66 researches; 14 have no level (generated list below: barrage, flare, backpacks, ...) | |
| the enemy's research levels | **no** | (see enemy units) | |
| hero revival cost and time | partly | the cost in the mask; the altar's "making a hero" (101-105); no time left | derivable |
| hero count and limit, hero token | partly | masks | |
| enemy race | yes | global 10-13 | |
| enemy resources | no (correct) | value head only | |
| alliances, victory, score screen | n/a | 1v1 melee | |

### Transient

| what a person sees | agent | where / what is missing | game exposes |
|---|---|---|---|
| projectiles in flight | no | | no native in 1.29; memory |
| area spells in progress (blizzard, flame strike, death and decay, earthquake, rain of fire, volcano, starfall, tranquility...) | partly | a visible caster's channel order; **not where** | `GetSpellTargetX/Y` at the spell effect (probe: blizzard's point); durations in `AbilityData` |
| spells cast (enemy hero used storm bolt) | no | the `SPELL_EFFECT` events are in the stream, unused | the events |
| "your base is under attack" pings | partly | derivable from own hp (the hp-change channel is zeroed) | derivable |
| minimap pings, sound cues | n/a | 1v1: alerts only | |

## The gaps, ranked

By how much each plausibly matters for winning and for variety. The cost estimates use these figures:

* A native called from the shim costs ~0.5 µs. The JASS units pass took 9 ms of a step with 150 units, the same pass in C 3 ms, about 35 natives a unit.
* A step's game thread is ~7.5-8 ms. The harness is ~15% of that.
* The recorded Echo Isles games have 47 entities per player step (median), 76 at the 90th percentile, 107 at most. They have 9 deaths in the last 90 s (median), 23 at the 90th percentile.

1. **Buffs and debuffs, own and enemy**
   * Why: none of the 160 buffs is an input, and the states they cause are invisible: a sheep is a full-hp grunt, a snared footman reads as an idle one, an own wind-walking blademaster doesn't know it is invisible. It decides:
     * whom to dispel, purge, abolish or devour (summons are visible as such);
     * whether a buff is already on, so a cast isn't wasted (fgself-194's 23,235 episodes have 3.2 refused heals, 1.2 faerie fires, 0.6 curses and 0.2 purges a game, though not every refusal is a buff's);
     * whom to focus or leave (banished, cycloned, divine shield, a sheep);
     * when to pull a cursed, slowed or snared unit back.
   * Caster play, and so variety, depends on it.
   * Cost, tier A (natives only, no reverse engineering):
     * Per entity: the 8 classification bits, positive and negative magic buff counts (`UnitCountBuffsEx`), armor and move speed. That is ~12 channels.
     * Through natives that is ~12 × 0.5 µs × 100 units ≈ 0.6 ms a step. Through the shim's object reads (the classifications sit in the word at +0x454 that `o_class` already reads) it is <0.05 ms.
   * Cost, tier B (memory): one walk of each unit's ability list (below) gives every buff on it.
     * The feature: ≤4 buff indices per entity into the checker's buff vocabulary, as an `EmbeddingBag` (zero-initialised) summed into the token.
     * <0.1 ms a step in the shim. The `U` record grows by ≤4 tokens, and only when the buffs change.
2. **Fog memory and scouting**
   * Why: a person sees the black mask, and the enemy's buildings as last seen in the fog. fgself-194 sees neither: `--inputs 2`, and the minGRU alone.
   * Of 16 `base2` games, 13 tied at 40 minutes. In 8 of those 13 ties, the side ahead never saw the hall that survived (experiment log, 2026-11-01).
   * Cost:
     * `--inputs 3` exists (≤40 remembered-enemy tokens). Twice it was weaker (fgself-107, -117), so it needs a look at what goes wrong first.
     * An explored grid: `IsMaskedToPlayer` on 24 × 24 points, about 0.3 ms per agent player. Every 8 steps that is ~0.04 ms a step.
     * Its input could be 32 global numbers (the explored and visible share of 16 sectors), zero-initialised. Or a channel per remembered entity.
3. **Corpses**
   * Why: these need corpses:
     * undead: ghouls' cannibalize, necromancers' raise dead, the meat wagon, the crypt lord's carrion scarabs, animate dead;
     * orc: the spirit walker's ancestral spirit;
     * human: the paladin's resurrection.
   * The agent can't see one, and order-coverage leaves out get corpse and ancestral spirit for that reason. fgself-194 refuses 0.9 cannibalizes a game. Undead is 35% of the learner's games.
   * Cost:
     * Harness: keep dead non-hero units in the pass until their corpse is gone (one record at death, an `R` at decay). That costs ~0.
     * Features: a corpse entity kind: a side channel, seconds since death ÷ 90, and the dead unit's type embedding. At most 16 corpses, the nearest to own units.
     * Inference: +0-16 tokens on a median of 47 (76 at the 90th percentile). That is up to +20-35% of the transformer's work in a big fight, less with a lower cap.
4. **Enemy upgrade levels and armor**
   * Why: a person sees an enemy unit's attack and armor levels and its armor value. They decide whether to engage, and which research answers it.
   * Cost:
     * No harness change: the enemy's `RESEARCH_FINISH` events are already in each observation.
     * Two channels per entity: its owner's attack and armor upgrade level for its type ÷ 3 (`UpgradeData`'s effects by unit). Plus `BlzGetUnitArmor` ÷ 10: 1 native per visible unit, ~0.05 ms.
5. **Items**
   * Why: the hero block tells 6 kinds of 56 items apart, and the charges of 38 charged items are no input. Items on the ground (tomes, runes, potions from creeps) are no entities; only the pick-up macro's mask sees them.
   * Cost:
     * An item vocabulary (~60) with a zero-initialised embedding per slot, summed into the hero's token.
     * Charges: 6 channels; `GetItemCharges` is ≤6 natives a hero.
     * Ground items: an entity kind from the `G` records, +0-6 tokens.
6. **Area spells in progress**
   * Why: blizzard, flame strike, death and decay, earthquake, rain of fire and volcano are drawn on the ground; a person moves out of them. The agent sees at most a visible caster's channel order, not where. The demonstrations have ~16 area spells a game.
   * Cost:
     * Harness: x and y on the `SPELL_EFFECT` event (`GetSpellTargetX/Y`). That costs ~0.
     * Features: an area entity kind for the lasting ones: the spell's embedding, x, y, radius ÷ 1000, seconds left from `AbilityData`. +0-3 tokens.
     * Projectiles have no native.
7. **Production, construction and research progress**
   * Why: a person sees the progress bars and the queue's icons, and plans by them: when the hero, the tier or the upgrade lands.
   * Cost:
     * No harness change: `View` has the start events, and the build times are in `UnitBalance` / `UpgradeData` (`strategy.build_seconds` reads them).
     * Two channels per own building (the progress of the current item and the seconds left ÷ 60), and the hall's construction progress. ~0.
8. **Creep types without an embedding**
   * Why: 7 of Echo Isles' 15 creep types read as "unknown", so a murloc camp, an ogre magi camp and a renegade wizard look alike. That costs creeping choices.
   * Cost: vocabulary rows. Start each new embedding as a copy of index 0's ("unknown") so every token is unchanged, then let them train. Runtime 0.
9. **Non-hero cooldowns and autocast state**
   * Why: a person sees the cooldown sweep and the autocast glow on the command card. The mask tracks both from the agent's own orders, but the network doesn't see them, and the mask's guess can be wrong: fgself-194's games have 0.9 refused defends and undefends a game (0.51 and 0.39 in 23,235 episodes: the state was so already).
   * Cost:
     * Cooldowns: `BlzGetUnitAbilityCooldownRemaining`, 1-3 per caster, ~0.05 ms. As channels per ability slot of the unit's type, zero-initialised.
     * Autocast: a bit of the ability object. A probe still has to find it: toggle and diff the object.
10. **The map itself: trees, pathing, cliffs, blight, mines and camps out of sight**
    * Why: on one fixed map the network learns it. It matters for a map pool, and fgself-178's pool was not kept.
    * Cost: the largest. A spatial input (a coarse minimap image through a small CNN, as AlphaStar) or static map tokens. Blight by `IsPointBlighted`.

Not ranked, small:

* the zeroed channels (hp change, last order);
* hero xp and attributes;
* rally points (`GetUnitRallyPoint`);
* time to the next swing (memory);
* the exact clock (`GAME_STATE_TIME_OF_DAY`);
* a transport's cargo;
* a summon's time left.

## How the network grows without changing its outputs

* **Entity channels** go at the end of the entity vector. `model.load` already pads `ent.weight` with zero columns ("features added since (at the end): their weights start at 0"), so the output is identical. A new `--inputs` level, or a bit mask of blocks, keeps the order fixed.
* **Embedding sums** (buffs on a unit, item types per slot) are new zero-initialised modules added to the token. They add exactly 0 until trained.
* **Global numbers** become zero columns in `glob.0` after the clock (`grow_glob` places the strategy and clock blocks this way). The value-only privileged block stays last.
* **New entity kinds** (corpses, ground items, areas, remembered enemies) are **not** made identical by zero weights. A new token still carries its type embedding and the layer biases, and it changes every softmax over the entities. That is the suspected cause of fgself-107's loss. Two things keep the outputs identical up to e^-30 until the kind is trusted:
  * an attention key bias of -30 per kind (as `ghost_bias` and `--memory-fade` do for remembered enemies);
  * exclusion from the target pointer until an order that takes them is allowed (corpse orders, pick up).
* **Vocabulary rows** for new unit types: their embeddings start as copies of index 0 ("unknown"), so every token is unchanged.

## What the game exposes

From `scripts/obs_probe.py` (2026-11-06; 1.29 under Wine; four games):

* **Buffs are abilities on the unit.** `GetUnitAbilityLevel(unit, buff) > 0` showed every buff cast in the probes:
  * human: `Binf`, `Binv`, `Bply`, `Bslo`;
  * orc: `Bblo`, `BOwk`, `Beng`, `Bprg`;
  * undead: `BUfa`, `BUhf`, `Bam2`, `Bcri`, `Bcrs`;
  * night elf: `Bfae`, `BEer`, `Broa`, `Brej`, `Bspo`;
  * auras: `BHab`, `BEah`.

  The code is the ability's own spelling: Unholy Frenzy puts `BUhf` (its `BuffID`) on the unit, while `AbilityBuffData.slk` spells it `Buhf`, which reads 0. Defend puts no buff on (`Bdef` is the Scroll of Protection's); a defending footman shows only in its move speed (188).
* **The classifications follow the debuffs:**
  * `UNIT_TYPE_SNARED`: ensnare, entangling roots;
  * `UNIT_TYPE_POISONED`: slow poison;
  * `UNIT_TYPE_POLYMORPHED`: polymorph;
  * `UNIT_TYPE_MAGIC_IMMUNE`: the dryad;
  * faerie fire made the unit `IsUnitDetected` for the caster.
* **`UnitCountBuffsEx` ANDs its filters.** All flags true counts 0. (positive, magic, nothing else) counts the positive magic buffs without auras (bloodlust, inner fire: 1), and the negative variant the debuffs (slow: 1). Purge's `Bprg` counts in neither.
* **`BlzIsUnitInvulnerable` returned true for every unit**, so it is unusable in 1.29.
* **`IsUnitInvisible(u, p)` is true whenever p doesn't see u**, fogged and masked too. A truly invisible unit is "invisible and neither fogged nor masked". The harness's visibility bits already hide an invisible enemy: an invisible footman next to a ghoul stayed visible only to its owner, through the whole fight.
* **Weapons:** `BlzGetUnitAttackCooldown` and `BlzGetUnitBaseDamage` take weapon index 1 for the first weapon (0 returns 0).
  * The cooldown is the base interval: bloodlust didn't change it.
  * The base damage leaves out the upgrades: a footman reads 11 at Rhme 3.
* **`BlzGetUnitArmor` is the armor now**, with upgrades and buffs: 8 → 13 with inner fire, 7 → 3 under faerie fire, 6 → 11 with frost armor.
* **`GetUnitMoveSpeed` is the speed now**: bloodlust 337, slow 150, ensnared or entangled 0, sheep 100-125, defend 188, wind walk 447.
* **Polymorph keeps the type id and the hit points**: the agent's record shows a healthy `ogru`.
* **Spell targets:** `GetSpellTargetX/Y` at the spell effect gives the point of a point spell (blizzard), the target's place for a unit spell, and (0, 0) for an instant one.
* **Corpses:** `GroupEnumUnitsInRect` returns dead units, with `UNIT_TYPE_DEAD` and their place, for 184-186 steps (~92 s) after the death. A hero's corpse is gone after 1-6 s; dead heroes stay in the agent's table. The harness drops a dead unit from its group once it has reported the death, so no corpse reaches the agent.
  * Raise dead and cannibalize as immediate orders were refused next to fresh corpses (the ghoul was snared or a sheep at the time). Their targeted form needs a corpse entity, which is the AI's way.
* **Also working:**
  * `GetItemCharges`;
  * `GetUnitRallyPoint`;
  * `BlzGetUnitAbilityCooldownRemaining` on unit abilities;
  * `GetPlayerTechCount` on the enemy (its weapons and armor levels);
  * `IsVisibleToPlayer`, `IsFoggedToPlayer` and `IsMaskedToPlayer` at points;
  * the trees' life in the first observation (50: the lumber).
* **Not exposed by 1.29 natives:**
  * autocast state;
  * time to the next swing;
  * shop, tavern and mercenary stock and restock;
  * production progress;
  * projectiles;
  * a unit's current order target;
  * a worker's load.
* **Memory:** the shim's ability finder (`0x6d6570`; decompiled in `~/tools/w3analysis/decomp/batch2.c`) walks a unit's ability list:
  * the head is an agent handle pair at unit+0x3e8/0x3ec, resolved by `0x7fc7e0`;
  * each entry's next pair is at +0x24/+0x28, its flags at +0x20 (bit 0x20 picks the finder's last argument), its alias id at +0x34.

  Buffs are entries of that list, so one walk per unit gives all of them. The same walk reaches each ability object for the autocast and cooldown bits, once their offsets are found.

## Generated from the game's tables

<!-- observation_coverage.py: begin (generated; edit the script, not this part) -->

### What `fgself-194` reads

`--inputs 2`: 106 channels per entity, of which `--features 29` zeroes 26 (channels 29-54: hit points' change since the last step / max hp, sin(1πx), cos(1πx), ... last order: steps ago / 40).

### Buffs

160 buffs can be on a unit in these games; without `--obs-full` the observation encodes none of them (a few show indirectly: a sleeping creep's flag, a summon's flag, an own illusion's flag). With `--obs-full 1` every one reaches the network: the shim reads each unit's buff list (obsfull.K_BUFF, a code per buff id). `scripts/obs_probe.py buffs` asks the game for 81 of them (BUFFS); `GetUnitAbilityLevel(unit, buff)` answers for any.

| from | buffs (the units or items that cast them) |
|---|---|
| human | BHab AuraBrilliance (Hamg); BHad AuraDevotion (Hpal); BHbd Blizzard (Hamg); BHbn Banish (Hblm); BHbz BlizzardAoe (not probed) (Hamg); BHds DivineShield (Hpal); BHfs FlameStrike (Hblm); BHtc ThunderClap (Hmkg); BHwe WaterElemental (not probed) (Hamg); BPSE Pause (Hmkg); Bclf CloudOfFog (hdhw); Bcmg ControlMagic (hspt); Bdcb DrainCaster (not probed) (Hblm); Bdcl DrainCasterLife (not probed) (Hblm); Bdcm DrainCasterMana (not probed) (Hblm); Bdtb DrainTarget (not probed) (Hblm); Bdtl DrainTargetLife (not probed) (Hblm); Bdtm DrainTargetMana (not probed) (Hblm); Bhea Heal (not probed) (hmpr); Binf InnerFire (hmpr); Binv Invisibility (hsor); Bmlc MagicLeashCaster (not probed) (hdhw); Bmlt MagicLeashTarget (hdhw); Bply Polymorph (hsor); Bslo Slow (hsor) |
| orc | BOae AuraEndurance (Otch); BOea EarthquakeAoe (not probed) (Ofar); BOeq Earthquake (Ofar); BOhx Hex (Oshd); BOmi MirrorImage (Obla); BOsf SpiritWolf (not probed) (Ofar); BOsh Shockwave (not probed) (Otch); BOvc VoodooCaster (not probed) (Oshd); BOvd Voodoo (Oshd); BOwd Ward (not probed) (Oshd); BOwk WindWalk (Obla); BOww WhirlwindAoe (not probed) (Obla); BPSE Pause (Otch); Bakb AuraKotoBeast (not probed) (okod); Bblo Bloodlust (oshm); Bbof BallsOfFire (not probed) (ocat); Bbsk BerserkerRage (otbk); Bdig Digesting (okod); Bdvv DevourVision (not probed) (okod); Bena EnsnareAir (orai); Beng EnsnareGround (orai); Beye EvilEye (odoc); Bhwd HealingWard (odoc); Bliq LiquidFire (otbr); Blsa LightningShieldAoe (not probed) (oshm); Blsh LightningShield (oshm); Boar AuraRegenLife (not probed) (ohwd); Bpoi PoisonAttack (owyv); Bprg Purge (oshm); Bpsd PoisonAttackStackDoT (not probed) (owyv); Bspl SpiritLink (ospm, ospw) |
| undead | BNin Infernal (not probed) (Udre); BUan AnimateDead (Udea); BUau AuraUnholy (Udea); BUav AuraVampiric (Udre); BUcb CarrionScarab (not probed) (Ucrl); BUcs CarrionSwarm (not probed) (Udre); BUdd DeathAndDecayAoe (Ulic); BUfa FrostArmor (Ulic); BUhf UnholyFrenzy (unec); BUim Impale (Ucrl); BUsl Sleep (Udre); BUsp SleepPause (Udre); BUts ThornyShield (not probed) (Ucrl); Bam2 AntiMagicShell (Matrix) (uban); Bams AntiMagicShell (uban); Bapl AuraPlague (uabo, uplg); Bcri Cripple (unec); Bcrs Curse (uban); Bfro Frost (Ulic, ufro, unp1...); Bfrz FreezingBreath (ufro); Bplg PlagueWard (uplg); Bpoc Possession (not probed) (uban); Bpos Possession (uban); Brai RaiseDead (not probed) (unec); Brpl ReplenishLife (uobs); Brpm ReplenishMana (uobs); Bust SleepStun (not probed) (Udre); Bwea WebAir (ucry); Bweb Web (ucry) |
| nightelf | AEsd Starfall target (not probed) (Emoo); AEtr Tranquility target (not probed) (Ekee); BEah AuraThorns (Ekee); BEar AuraTrueshot (Emoo); BEer EntanglingRoots (Ekee); BEfn ForceOfNature (not probed) (Ekee); BEia ImmolationAoe (not probed) (Edem, Edmm); BEim Immolation (Edem, Edmm); BEme Metamorphosis (Edem, Edmm); BEsh ShadowStrike (not probed) (Ewar); BEst Scout (not probed) (Emoo); BEsv SpiritOfVengeance (not probed) (Ewar); Bcy2 CycloneTwo (not probed) (edot); Bcyc Cyclone (edot); Beat EatTree (not probed) (eaoe, eaom, eaow...); Bfae FaerieFire (edot, edtm); Bgra GrabTree (not probed) (emtg); Bmfa ManaFlareAoe (not probed) (efdr); Bmfl ManaFlare (not probed) (efdr); Bpsh PhaseShift (efdr); Brej Rejuvination (edoc); Broa Roar (edcm, edoc); Bspo SlowPoison (edry); Bssd SlowPoisonStackDoT (not probed) (edry); Bvng Vengeance (not probed) (espv) |
| tavern | BHca ColdArrow (Nngs); BNab AcidBomb (not probed) (Nal2, Nal3, Nalc...); BNba BlackArrow (Nbrn); BNbf BreathOfFire (not probed) (Npbm); BNcg ClockwerkGoblin (not probed) (Nrob, Ntin); BNcr ChemicalRage (not probed) (Nal2, Nal3, Nalc...); BNcs ClusterRockets (not probed) (Nrob, Ntin); BNdh DrunkenHaze (Npbm); BNdi DoomMinion (not probed) (Nplh); BNdm DarkMinion (not probed) (Nbrn); BNdo Doom (Nplh); BNef ElementalFury (not probed) (Npbm); BNeg EngineeringUpgrade (not probed) (Nrob, Ntin); BNfy SummonFactory (not probed) (Nrob, Ntin); BNhs HealingSpray (not probed) (Nal2, Nal3, Nalc...); BNht HowlOfTerror (Nplh); BNic Incinerate (not probed) (Nfir); BNlm LavaMonster (not probed) (Nfir); BNms ManaShield (Nngs); BNpi PermImmolation (not probed) (ninf); BNrd RainOfFire (not probed) (Nplh, nba2); BNrf RainOfFireAoe (not probed) (Nplh, nba2); BNsg SummonGrizzly (not probed) (Nbst); BNsi Silence (Nbrn); BNso SoulBurn (Nfir); BNsq SummonQuillbeast (not probed) (Nbst); BNst Stampede (not probed) (Nbst); BNsw SummonWarEagle (not probed) (Nbst); BNtm Transmute (not probed) (Nal2, Nal3, Nalc...); BNto Tornado (not probed) (Nngs); BNva VolcanoAOE (not probed) (Nfir); BNvc Volcano (not probed) (Nfir); BOwk WindWalk (npn2); BPSE Pause (nba2, ngz2, ngz3); Basl AuraSlow (not probed) (ntor); Bcri Cripple (nba2); Bcsd ColdArrowStackDoT (not probed) (Nngs); Bcsi ColdArrowStackInfo (not probed) (Nngs); Bcy2 CycloneTwo (not probed) (npn2); Bcyc Cyclone (npn2); Bdcb DrainCaster (not probed) (Nbrn); Bdcl DrainCasterLife (not probed) (Nbrn); Bdcm DrainCasterMana (not probed) (Nbrn); Bdtb DrainTarget (not probed) (Nbrn); Bdtl DrainTargetLife (not probed) (Nbrn); Bdtm DrainTargetMana (not probed) (Nbrn); Bfzy Frenzy (nqb2, nqb3); Bpig PandaImmolation (not probed) (npn1); Btdg TornadoDamageAoe (not probed) (ntor); Btsa TornadoSpinAoe (not probed) (ntor); Btsp TornadoSpin (not probed) (ntor) |
| mercenaries | Bhea Heal (not probed) (nfsp); Bslo Slow (ngrk) |
| creeps | BPSE Pause (nkot); BUts ThornyShield (not probed) (ntrg); Bblo Bloodlust (nomg); Bena EnsnareAir (nftt, nmrr); Beng EnsnareGround (nftt, nmrr); Blsa LightningShieldAoe (not probed) (nwzg); Blsh LightningShield (nwzg); Bpoi PoisonAttack (nmrm); Bprg Purge (nwzg); Bpsd PoisonAttackStackDoT (not probed) (nmrm); Bslo Slow (nkog) |
| items | BIcb Corruption (not probed) (ocor); BIil ItemIllusion (not probed) (will); BIrg Regeneration (hslv, plcl, rej3...); BIrl RegenLife (hslv, plcl, rej3...); BIrm RegenMana (not probed) (hslv, plcl, rej3...); BNsa Sanctuary (not probed) (ssan); Bam2 AntiMagicShell (Matrix) (pams); Bams AntiMagicShell (pams); Bdef Defense (spro); Bdet Detected (dust); Beye EvilEye (wswd); Binv Invisibility (pinv); Blsa LightningShieldAoe (not probed) (wlsd); Blsh LightningShield (wlsd); Bmec MechanicalCritter (not probed) (mcri); Brai RaiseDead (not probed) (rnec); Broa Roar (sror); Bspe SpeedBonus (shas); Bvul Invulnerable (pnvl, pnvu) |

### Items

56 item types a hero may hold in these games (the race shops, the goblin merchant, the maps' drop tables); the hero block's five numbers per slot (held, usable, heals, restores mana, town portal) tell 6 kinds apart. 38 of them have charges, which no input counts. (`--obs-full 1`: each item's own code and each slot's charges.) Items that look alike:

| held, usable, heals, mana, portal | items |
|---|---|
| 1, 1, 0, 0, 0 | 31: dust, manh, mcri, moon, pams, pinv, pnvl, pnvu, rnec, shas, skul, spre, spro, sror, ssan, stel, tdex, tdx2, texp, tgrh, tin2, tint, tpow, tret, tsct, tst2, tstr, will, wlsd, woms, wswd |
| 1, 0, 0, 0, 0 | 15: bspd, cnob, evtl, gcel, ocor, ofir, oli2, oven, penr, prvt, rat6, rat9, rde1, rde2, rlif |
| 1, 1, 1, 0, 0 | 5: hslv, pghe, phea, shea, sreg |
| 1, 1, 0, 1, 0 | 3: pgma, plcl, pman |

### Researches

The global vector has the player's own level of 66 researches (the vocabulary's upgrades); the enemy's levels are no input. Researches without a level there:

- **human**: Rhfl, Rhpm, Rhrt
- **orc**: Robf, Robk, Ropm, Rotr, Rowt
- **undead**: Ruex, Rupm
- **nightelf**: Reht, Renb, Repb, Repm

### Non-hero cooldowns and toggles

Only a hero's four ability slots report a cooldown; a unit's castable abilities with one, and the abilities with an autocast or on/off state (their state is no input; `fullvocab.Checks` tracks it from `DEFAULT_ON` and the agent's own orders for the mask):

| race | abilities with a cooldown | toggles |
|---|---|---|
| human | AHta (hatw), Aclf (hdhw), Acmg (hspt), Afla (hmtm), Ahea (hmpr), Ainf (hmpr), Amls (hdhw), Aply (hsor), Aslo (hsor), Asps (hspt) | Ahea (hmpr), Ahrp (hpea), Ainf (hmpr), Aslo (hsor), Asps (hspt) |
| orc | Aast (ospm), Ablo (oshm), Absk (otbk), Acpf (ospm), Aens (orai), Alsh (oshm), Apg2 (oshm) | Arep (opeo) |
| undead | Acri (unec), Acrs (uban), Advm (ubsp), Arai (unec), Arpl (uobs), Arpm (uobs), Astn (ugar), Auhf (unec), Aweb (ucry) | AUcb (Ucrl), AUfu (Ulic), Acrs (uban), Arai (unec), Arpl (uobs), Arpm (uobs), Arst (uaco), Aweb (ucry) |
| nightelf | Aco2 (earc), Aco3 (ehip), Acyc (edot), Adec (ehpr), Aesn (esen), Afa2 (edtm), Afae (edot), Agra (emtg), Amfl (efdr), Apsh (efdr), Arej (edoc), Atau (emtg), Avng (espv) | AHfa (Emoo), Aadm (edry), Afa2 (edtm), Afae (edot), Ambt (emow), Apsh (efdr), Aren (ewsp), Avng (espv) |
| tavern | ACcr (nba2), ACcy (npn2), ACrf (nba2), ANbl (ngz3), ANta (npn3), ANwk (npn2), Adsm (nba2), Afzy (nqb2), Awrs (nba2) | ANba (Nbrn), ANfa (Nngs), ANms (Nngs) |
| mercenaries | ACsw (ngrk), Anh1 (nfsp) | ACdm (nfsp), ACsw (ngrk), Anh1 (nfsp) |
| creeps | ACbb (nomg), ACen (nftt), ACls (nwzg), ACpu (nwzg), ACsw (nkog) | ACdm (nkog), ACsw (nkog) |

### Unit types

Types a player meets without an embedding of their own in `fgself-194`'s vocabulary (their entities get index 0, unknown). `--full-vocab 1` adds the races', the tavern's and the mercenaries' (`fullvocab.extend`), `--obs-full 1` the creeps' (`obsfull.extend`):

- **human**: hatw, hctw, hmtt, hrtt
- **orc**: oeye, ospm, ospw, otbk, owtw
- **undead**: ubsp, ushd, uzg2
- **nightelf**: edcm, edtm, ehip, ehpr, etrp
- **tavern**: Nal2, Nal3, Nalm, Nrob, nba2, ndr1, ndr2, ndr3, nfa1, nfa2, nfac, ngz1, ngz2, ngz3, nlv1, nlv2, nlv3, now2, now3, nowl, npn1, npn2, npn3, nqb1, nqb2, nqb3, ntor, nwe1, nwe2, nwe3
- **mercenaries**: ngir, ngsp, nzep
- **creeps**: nkot (Kobold Tunneler), nmrl (Murloc Tiderunner), nmrm (Murloc Nightcrawler), nmrr (Murloc Huntsman), nomg (Ogre Magi), ntrg (Gargantuan Sea Turtle), nwzg (Renegade Wizard)

<!-- observation_coverage.py: end -->
