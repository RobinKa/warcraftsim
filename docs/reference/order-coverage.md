# Order coverage

Which orders a player can give in a 1.29 melee game the policy's vocabulary has. The list of every order comes from the game's own tables (`warcraftsim/fullgame/fullvocab.py`: `Units\*UnitFunc.txt`, `UnitAbilities.slk`, `AbilityData.slk`, the ability order strings; hero abilities from `data/abilities.py`; the race shops' stock from `shopitems.py`; the neutral buildings of the maps below). Written by `scripts/order_coverage.py` for `/home/tora/warcraftsim/runs/fgself-194/vocab.json`; with `--full-vocab 1` a vocabulary gets every missing order (`fullvocab.extend`).

| race | orders before | orders with --full-vocab | unit types before | with --full-vocab |
|---|---|---|---|---|
| human | 219/273 | 273/273 | 50/85 | 85/85 |
| orc | 210/263 | 263/263 | 50/86 | 86/86 |
| undead | 215/278 | 278/278 | 51/85 | 85/85 |
| nightelf | 218/283 | 283/283 | 46/85 | 85/85 |

An order counts as covered in its own target form, or for a spell or unit ability aimed at a place also in the other one (the built-in AI casts blizzard, earthquake, stasis trap and dispel on a unit's place, raise dead and carrion scarabs on a corpse); a peasant's call to arms and a militia's back to work are the vocabulary's militiaconvert and militiaunconvert, which the agent turns into them.

## human

Missing before (each added by `--full-vocab 1`):

- **ability**: cyclone (unit), detectaoe (point), flare (point), frenzy (immediate), magicleash (unit), polymorph (unit), selfdestruct (unit), unload (point)
- **generic**: attackground (point), holdposition (immediate), patrol (point), setrally (point), setrally (unit)
- **item**: mcri (unit), drop item 1 (point), drop item 2 (point), drop item 3 (point), drop item 4 (point), drop item 5 (point), drop item 6 (point), give item 1 (unit), give item 2 (unit), give item 3 (unit), give item 4 (unit), give item 5 (unit), give item 6 (unit), pick up an item near a point (point), sell item 1 (unit), sell item 2 (unit), sell item 3 (unit), sell item 4 (unit), sell item 5 (unit), sell item 6 (unit)
- **neutral**: neutraldetectaoe (point)
- **research**: Rhfl (immediate), Rhrt (immediate)
- **toggle**: autodispeloff (immediate), coldarrows (immediate), healoff (immediate), innerfireoff (immediate), innerfireon (immediate), repairoff (immediate), repairon (immediate), slowoff (immediate), spellstealoff (immediate), uncoldarrows (immediate), undivineshield (immediate)
- **train**: hmtt (immediate), hrtt (immediate)
- **upgrade**: hatw (immediate), hctw (immediate)
- **ware**: ngir (unit), ngsp (unit), nzep (unit)
- **unit types**: Nal2, Nal3, Nalm, Nrob, hatw, hctw, hmtt, hrtt, nba2, ndr1, ndr2, ndr3, nfa1, nfa2, nfac, ngad, ngir, ngsp, ngz1, ngz2, ngz3, nlv1, nlv2, nlv3, npn1, npn2, npn3, nqb1, nqb2, nqb3, ntor, nwe1, nwe2, nwe3, nzep

## orc

Missing before (each added by `--full-vocab 1`):

- **ability**: ancestralspirittarget (unit), berserk (immediate), cyclone (unit), disenchant (point), evileye (point), frenzy (immediate), selfdestruct (unit), spiritlink (unit), unload (point)
- **build**: owtw (point)
- **generic**: attackground (point), holdposition (immediate), patrol (point), setrally (point), setrally (unit)
- **item**: tgrh (unit), drop item 1 (point), drop item 2 (point), drop item 3 (point), drop item 4 (point), drop item 5 (point), drop item 6 (point), give item 1 (unit), give item 2 (unit), give item 3 (unit), give item 4 (unit), give item 5 (unit), give item 6 (unit), pick up an item near a point (point), sell item 1 (unit), sell item 2 (unit), sell item 3 (unit), sell item 4 (unit), sell item 5 (unit), sell item 6 (unit)
- **morph**: corporealform (immediate)
- **neutral**: neutraldetectaoe (point)
- **research**: Robf (immediate), Robk (immediate), Rotr (immediate), Rowt (immediate)
- **toggle**: autodispeloff (immediate), coldarrows (immediate), healoff (immediate), repairoff (immediate), repairon (immediate), slowoff (immediate), uncoldarrows (immediate)
- **train**: ospm (immediate)
- **unmorph**: uncorporealform (immediate)
- **ware**: ngir (unit), ngsp (unit), nzep (unit)
- **unit types**: Nal2, Nal3, Nalm, Nrob, nba2, ndr1, ndr2, ndr3, nfa1, nfa2, nfac, ngad, ngir, ngsp, ngz1, ngz2, ngz3, nlv1, nlv2, nlv3, npn1, npn2, npn3, nqb1, nqb2, nqb3, ntor, nwe1, nwe2, nwe3, nzep, oeye, ospm, ospw, otbk, owtw

## undead

Missing before (each added by `--full-vocab 1`):

- **ability**: absorb (unit), cyclone (unit), devourmagic (point), freezingbreath (unit), frenzy (immediate), loadcorpse (unit), possession (unit), sacrifice (unit), selfdestruct (unit), unload (point), unloadallcorpses (immediate), unsummon (unit)
- **generic**: attackground (point), holdposition (immediate), patrol (point), setrally (point), setrally (unit)
- **item**: drop item 1 (point), drop item 2 (point), drop item 3 (point), drop item 4 (point), drop item 5 (point), drop item 6 (point), give item 1 (unit), give item 2 (unit), give item 3 (unit), give item 4 (unit), give item 5 (unit), give item 6 (unit), pick up an item near a point (point), sell item 1 (unit), sell item 2 (unit), sell item 3 (unit), sell item 4 (unit), sell item 5 (unit), sell item 6 (unit)
- **morph**: avengerform (immediate)
- **neutral**: neutraldetectaoe (point)
- **research**: Ruex (immediate)
- **toggle**: autodispeloff (immediate), carrionscarabsoff (immediate), carrionscarabson (immediate), coldarrows (immediate), curseoff (immediate), frostarmoroff (immediate), frostarmoron (immediate), healoff (immediate), raisedeadoff (immediate), raisedeadon (immediate), replenishlifeoff (immediate), replenishlifeon (immediate), replenishmanaoff (immediate), replenishmanaon (immediate), restorationoff (immediate), restorationon (immediate), slowoff (immediate), uncoldarrows (immediate), weboff (immediate)
- **unmorph**: unavengerform (immediate)
- **upgrade**: uzg2 (immediate)
- **ware**: ngir (unit), ngsp (unit), nzep (unit)
- **unit types**: Nal2, Nal3, Nalm, Nrob, nba2, ndr1, ndr2, ndr3, nfa1, nfa2, nfac, ngad, ngir, ngsp, ngz1, ngz2, ngz3, nlv1, nlv2, nlv3, npn1, npn2, npn3, nqb1, nqb2, nqb3, ntor, nwe1, nwe2, nwe3, nzep, ubsp, ushd, uzg2

## nightelf

Missing before (each added by `--full-vocab 1`):

- **ability**: ambush (immediate), corrosivebreath (unit), coupletarget (unit), cyclone (unit), decouple (immediate), eattree (tree), entangle (unit), frenzy (immediate), recharge (unit), root (point), selfdestruct (unit), unload (immediate), unload (point)
- **build**: etrp (point)
- **generic**: attackground (point), holdposition (immediate), patrol (point), setrally (point), setrally (unit)
- **item**: drop item 1 (point), drop item 2 (point), drop item 3 (point), drop item 4 (point), drop item 5 (point), drop item 6 (point), give item 1 (unit), give item 2 (unit), give item 3 (unit), give item 4 (unit), give item 5 (unit), give item 6 (unit), pick up an item near a point (point), sell item 1 (unit), sell item 2 (unit), sell item 3 (unit), sell item 4 (unit), sell item 5 (unit), sell item 6 (unit)
- **morph**: bearform (immediate), ravenform (immediate)
- **neutral**: neutraldetectaoe (point)
- **research**: Reht (immediate), Renb (immediate), Repb (immediate)
- **spell**: flamingarrowstarg (unit)
- **toggle**: autodispeloff (immediate), coldarrows (immediate), faeriefireoff (immediate), healoff (immediate), phaseshiftoff (immediate), rechargeoff (immediate), rechargeon (immediate), renewoff (immediate), renewon (immediate), slowoff (immediate), uncoldarrows (immediate), unflamingarrows (immediate), unroot (immediate), vengeanceoff (immediate)
- **train**: ehip (immediate)
- **unmorph**: unbearform (immediate), unravenform (immediate)
- **ware**: ngir (unit), ngsp (unit), nzep (unit)
- **unit types**: Nal2, Nal3, Nalm, Nrob, edcm, edtm, ehip, ehpr, etrp, nba2, ndr1, ndr2, ndr3, nfa1, nfa2, nfac, ngad, ngir, ngsp, ngz1, ngz2, ngz3, nlv1, nlv2, nlv3, now2, now3, nowl, npn1, npn2, npn3, nqb1, nqb2, nqb3, ntor, nwe1, nwe2, nwe3, nzep

## Left out on purpose

- **moveslot**: moving an item between inventory slots: no effect on play (an item acts from any slot; the policy reads and uses all six)
- **returnresources**: the engine's own (a worker returning its load): left out of every vocabulary on purpose (features.DROPPED_ORDERS: a policy that learned it interrupted working harvesters)
- **marketplace**: the marketplace's stock is random (blizzard.j PerformStockUpdates: an item of a level and class creeps have dropped, every 30 s) and 1.29's JASS can't read a shop's stock, so no mask could tell what it sells
- **backpack**: a unit's backpack (Rhpm, Ropm, Rupm, Repm research): the harness reports inventories of heroes only
- **awaken**: a tavern's instant revival of a dead hero: an altar revives it (the hero's train order), and the tavern's button picks no hero the policy can point at (dead heroes are no entities)

## Neutral buildings on the maps

Those that take an order: the tavern (`ntav`: its heroes), the mercenary camp (`nmer`: its mercenaries), the goblin laboratory (`ngad`: zeppelin, shredder, sapper, and its reveal), the goblin merchant (`ngme`: its items; it buys items too) and the marketplace (`nmrk`: it buys items; what it sells is left out, above). The others take none (fountains, creep huts).

| map | neutral buildings |
|---|---|
| (2)EchoIsles | ngme x1, ngol x4 (none), nmer x2, nmh0 x2 (none), nmh1 x8 (none), nmrk x1, ntav x1 |
| echofast2 | ngme x1, ngol x4 (none), nmer x2, nmh0 x2 (none), nmh1 x8 (none), nmrk x1, ntav x1 |
| lastrefugefast2 | ngad x2, ngme x2, ngnh x1 (none), ngol x6 (none), nmer x2, nmh0 x2 (none), nmh1 x3 (none), ntav x1, nten x1 (none), ntn2 x4 (none) |
| amazoniafast2 | nbse x1 (none), nfh0 x1 (none), nfh1 x1 (none), ngad x2, ngme x2, ngol x4 (none), nmer x2, nmg0 x1 (none), nmg1 x1 (none), ntav x1 |
| terenaslvfast2 | ngad x2, ngme x2, ngol x6 (none), nmer x2, ntav x1 |
