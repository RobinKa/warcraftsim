//===========================================================================
// w3obs: the observing harness, for playing back replays of games recorded WITHOUT a harness
// (people's games on a stock map; warcraftsim.fullgame.replays). Injected by
// warcraftsim.data.mapbuild.inject_observer.
//
// A replay is re-simulated from the map and the recorded actions, and the actions name units by
// the engine's object ids: a slot in its table of objects and the count of objects made before
// (data.w3g). Every object the harness makes (a trigger, a timer, a group, an event, the handler
// of a timer's code, the wrapper of a callback's code on its first use) takes the next id, so
// every unit made after it would have another id than in the recorded game, and the replay's
// orders would go to no unit. Hence:
//   * every object is made in W3S_Init, at the end of main (the map's own melee initialization,
//     starting units, AI and victory rules ran before as in the recorded game). Two groups made
//     first and last bracket them: the w3shim replay build reports their ids (MBOX - 40), and the
//     converter shifts every later id in the replay by their count (w3g.rewrite).
//   * a step makes nothing: no event triggers (an event that has a trigger makes objects each
//     time it fires: the harness reports no events, and the converter derives them), callbacks
//     used once at init, no GetWorldBounds() and no locations in a step.
//   * it changes nothing: no units, orders, GetRandom* or player state.
//
// Every step (a periodic game-time timer) it writes the observation of w3sim.j (the token stream
// of Preload() calls; the unit records come from the w3shim DLL's pass over the units, which the
// replay build extends with the engine's own construction, production and upgrade state) and syncs
// with the controller through the mailbox (GetPlayerTechMaxAllowed on the neutral passive player),
// which blocks the game until the controller has read it. Commands are read and ignored. Research
// levels are polled: an "E" record (kind 9, a = -1 - player, b = the upgrade, c = its level) when
// a level rises.
//
// @CONFIG@ lines are substituted at map build time (mapbuild.inject_observer).
//===========================================================================

globals
    // @CONFIG@ (replaced at map build time)
    constant real W3S_CFG_STEP_S = 0.5
    constant boolean W3S_CFG_SEND_DESTRUCTABLES = true
    constant integer W3S_VERSION = 4
    constant integer W3S_ABIL_SLOTS = 4  // warcraftsim.protocol.HERO_ABILITY_SLOTS
    constant integer W3S_MBOX = 1048576
    boolean w3s_full = true
    integer w3s_sum = 0
    hashtable w3s_ht = null  // the shim's per-unit table (its records' hashes)
    hashtable w3s_abil = null  // hero type -> ability code per slot (W3S_InitHeroAbilities)
    timer w3s_clock = null
    timer w3s_timer = null
    timer w3s_timer0 = null
    group w3s_group = null
    group w3s_all = null  // every unit seen (hidden ones stay: enumerations skip them)
    rect w3s_world = null
    group w3s_probe0 = null  // made first and last in W3S_Init: their ids bracket the harness's objects
    group w3s_probe1 = null
    integer w3s_seq = 0
    boolean w3s_first = true
    integer array w3s_ev
    integer w3s_nev = 0
    integer array w3s_halls  // (the shim's pass counts them; unused here)
    integer array w3s_mobile
    integer array w3s_alive
    integer w3s_cser = 0  // the shim makes the pass over the units (1)
    player w3s_np = null  // the neutral passive player (the mailbox's)
    integer w3s_nseer = 0  // the seers (every player who plays): their players and bits in the visibility mask
    player array w3s_seer_p
    integer array w3s_seer_bit
    boolean array w3s_seer
    integer array w3s_tree  // the trees at the start: id, type, x, y, life
    integer w3s_ntree = 0
    integer array w3s_upg  // the game's upgrades (W3S_InitUpgrades) and each seer's level of each
    integer w3s_nupg = 0
    integer array w3s_upg_level
endglobals

//===========================================================================
// tokens (as in w3sim.j: tag, integer fields, the fields' sum)

function W3S_Tok takes integer v returns nothing
    set w3s_sum = w3s_sum + v
    call Preload(I2S(v))
endfunction

function W3S_Rec takes string tag returns nothing
    set w3s_sum = 0
    call Preload(tag)
endfunction

function W3S_End takes nothing returns nothing
    call Preload(I2S(w3s_sum))
endfunction

function W3S_Ev takes integer kind, integer a, integer b, integer c returns nothing
    if w3s_nev < 8000 then
        set w3s_ev[w3s_nev] = kind
        set w3s_ev[w3s_nev + 1] = a
        set w3s_ev[w3s_nev + 2] = b
        set w3s_ev[w3s_nev + 3] = c
        set w3s_nev = w3s_nev + 4
    endif
endfunction

function W3S_Pow2 takes integer i returns integer
    local integer v = 1
    loop
        exitwhen i <= 0
        set v = v * 2
        set i = i - 1
    endloop
    return v
endfunction

// the players whose view the units' visibility bits report: everyone who plays
function W3S_Seers takes nothing returns nothing
    local integer i = 0
    set w3s_nseer = 0
    loop
        exitwhen i >= bj_MAX_PLAYERS
        set w3s_seer[i] = GetPlayerSlotState(Player(i)) == PLAYER_SLOT_STATE_PLAYING and not IsPlayerObserver(Player(i))
        if w3s_seer[i] then
            set w3s_seer_p[w3s_nseer] = Player(i)
            set w3s_seer_bit[w3s_nseer] = W3S_Pow2(i)
            set w3s_nseer = w3s_nseer + 1
        endif
        set i = i + 1
    endloop
endfunction

function W3S_AddUnit takes nothing returns nothing
    call GroupAddUnit(w3s_all, GetEnumUnit())
endfunction

//===========================================================================
// observation (the records of w3sim.j)

function W3S_RaceId takes race r returns integer
    if r == RACE_HUMAN then
        return 1
    elseif r == RACE_ORC then
        return 2
    elseif r == RACE_UNDEAD then
        return 3
    elseif r == RACE_NIGHTELF then
        return 4
    endif
    return 0
endfunction

// result: 0 (the replay's leave records tell how the game ended: warcraftsim.data.w3g)
function W3S_SerPlayers takes nothing returns nothing
    local integer i = 0
    local player p
    loop
        exitwhen i >= bj_MAX_PLAYERS
        set p = Player(i)
        if GetPlayerSlotState(p) != PLAYER_SLOT_STATE_EMPTY and not IsPlayerObserver(p) then
            call W3S_Rec("P")
            call W3S_Tok(i)
            call W3S_Tok(W3S_RaceId(GetPlayerRace(p)))
            call W3S_Tok(0)
            call W3S_Tok(GetPlayerState(p, PLAYER_STATE_RESOURCE_GOLD))
            call W3S_Tok(GetPlayerState(p, PLAYER_STATE_RESOURCE_LUMBER))
            call W3S_Tok(GetPlayerState(p, PLAYER_STATE_RESOURCE_FOOD_USED))
            call W3S_Tok(GetPlayerState(p, PLAYER_STATE_RESOURCE_FOOD_CAP))
            call W3S_Tok(GetPlayerState(p, PLAYER_STATE_GOLD_UPKEEP_RATE))
            call W3S_Tok(GetPlayerState(p, PLAYER_STATE_GOLD_GATHERED))
            call W3S_Tok(GetPlayerState(p, PLAYER_STATE_LUMBER_GATHERED))
            call W3S_Tok(GetPlayerStructureCount(p, true))
            call W3S_Tok(0)
            call W3S_Tok(R2I(GetStartLocationX(GetPlayerStartLocation(p))))
            call W3S_Tok(R2I(GetStartLocationY(GetPlayerStartLocation(p))))
            call W3S_End()
        endif
        set i = i + 1
    endloop
    set p = null
endfunction

// the trees, at init (into integers: the first observation writes them; their object ids to the shim)
function W3S_KeepTree takes nothing returns nothing
    local destructable d = GetEnumDestructable()
    local integer k = w3s_ntree * 5
    if k < 32000 then
        set w3s_tree[k] = GetHandleId(d)
        set w3s_tree[k + 1] = GetDestructableTypeId(d)
        set w3s_tree[k + 2] = R2I(GetDestructableX(d))
        set w3s_tree[k + 3] = R2I(GetDestructableY(d))
        set w3s_tree[k + 4] = R2I(GetDestructableLife(d) + 0.5)
        set w3s_ntree = w3s_ntree + 1
        call GetPlayerTechMaxAllowed(w3s_np, W3S_MBOX - 41)  // (its object ids: replays name trees by them)
        call GetPlayerTechMaxAllowed(w3s_np, 0 - GetHandleId(d))
    endif
    set d = null
endfunction

function W3S_SerTrees takes nothing returns nothing
    local integer k = 0
    loop
        exitwhen k >= w3s_ntree * 5
        call W3S_Rec("D")
        call W3S_Tok(w3s_tree[k])
        call W3S_Tok(w3s_tree[k + 1])
        call W3S_Tok(w3s_tree[k + 2])
        call W3S_Tok(w3s_tree[k + 3])
        call W3S_Tok(w3s_tree[k + 4])
        call W3S_End()
        set k = k + 5
    endloop
endfunction

function W3S_Order takes string name returns nothing
    call W3S_Rec("O")
    call W3S_Tok(OrderId(name))
    call W3S_End()
endfunction

// hero type -> ability code per slot (warcraftsim.data.abilities.hero_ability_table)
function W3S_InitHeroAbilities takes nothing returns nothing
    set w3s_abil = InitHashtable()
    // @HERO_ABILITIES@
endfunction

// must match warcraftsim.protocol.ORDER_NAMES
function W3S_SerOrders takes nothing returns nothing
    // @ORDERS@
endfunction

function W3S_Upgrade takes integer id returns nothing
    set w3s_upg[w3s_nupg] = id
    set w3s_nupg = w3s_nupg + 1
endfunction

// every upgrade of the game (UpgradeData.slk)
function W3S_InitUpgrades takes nothing returns nothing
    // @UPGRADES@
endfunction

// research levels that rose since the last step: "E" records (kind 9)
function W3S_PollResearch takes nothing returns nothing
    local integer s = 0
    local integer k
    local integer i
    local integer level
    loop
        exitwhen s >= w3s_nseer
        set i = GetPlayerId(w3s_seer_p[s])
        set k = 0
        loop
            exitwhen k >= w3s_nupg
            set level = GetPlayerTechCount(w3s_seer_p[s], w3s_upg[k], true)
            if level > w3s_upg_level[s * 128 + k] then
                set w3s_upg_level[s * 128 + k] = level
                if not w3s_first then  // (the levels a game starts with are no events)
                    call W3S_Ev(9, -1 - i, w3s_upg[k], level)
                endif
            endif
            set k = k + 1
        endloop
        set s = s + 1
    endloop
endfunction

// the shim's pass over the units (shim/units.c; w3sim.j's W3S_CSerInit): what it needs of the harness
function W3S_CSerInit takes nothing returns nothing
    local integer i = 0
    set w3s_cser = GetPlayerTechMaxAllowed(w3s_np, W3S_MBOX - 2)
    if w3s_cser != 1 then
        set w3s_cser = 0
        return
    endif
    call GetPlayerTechMaxAllowed(w3s_np, W3S_MBOX - 5)
    call GetPlayerTechMaxAllowed(w3s_np, 0 - GetHandleId(w3s_ht))
    call GetPlayerTechMaxAllowed(w3s_np, W3S_MBOX - 6)
    call GetPlayerTechMaxAllowed(w3s_np, 0 - GetHandleId(w3s_abil))
    call GetPlayerTechMaxAllowed(w3s_np, W3S_MBOX - 13)
    call GetPlayerTechMaxAllowed(w3s_np, 0 - GetHandleId(w3s_all))
    call GetPlayerTechMaxAllowed(w3s_np, W3S_MBOX - 14)
    call GetPlayerTechMaxAllowed(w3s_np, 0 - bj_MAX_PLAYERS)
    call GetPlayerTechMaxAllowed(w3s_np, W3S_MBOX - 7)
    loop
        exitwhen i >= bj_MAX_PLAYERS
        if w3s_seer[i] then
            call GetPlayerTechMaxAllowed(w3s_np, W3S_MBOX - 8)
            call GetPlayerTechMaxAllowed(w3s_np, 0 - (i + 1))
        endif
        set i = i + 1
    endloop
endfunction

function W3S_CPassEnum takes nothing returns nothing
    call GetPlayerTechMaxAllowed(w3s_np, 0 - GetHandleId(GetEnumUnit()))
endfunction

function W3S_CPass takes nothing returns nothing
    local integer i = 0
    if w3s_full or w3s_first then
        call GetPlayerTechMaxAllowed(w3s_np, W3S_MBOX - 10)
    else
        call GetPlayerTechMaxAllowed(w3s_np, W3S_MBOX - 9)
    endif
    call ForGroup(w3s_all, function W3S_CPassEnum)
    call GetPlayerTechMaxAllowed(w3s_np, W3S_MBOX - 15)
    loop
        exitwhen i >= bj_MAX_PLAYERS
        if GetPlayerSlotState(Player(i)) == PLAYER_SLOT_STATE_PLAYING then
            set w3s_halls[i] = GetPlayerTechMaxAllowed(w3s_np, W3S_MBOX - 100 - 3 * i)
            set w3s_mobile[i] = GetPlayerTechMaxAllowed(w3s_np, W3S_MBOX - 101 - 3 * i)
            set w3s_alive[i] = GetPlayerTechMaxAllowed(w3s_np, W3S_MBOX - 102 - 3 * i)
        endif
        set i = i + 1
    endloop
endfunction

function W3S_WriteObs takes nothing returns nothing
    local integer i
    set w3s_full = w3s_full or w3s_first
    call PreloadGenClear()
    call PreloadGenStart()
    call Preload("V")
    call W3S_Tok(W3S_VERSION)
    call W3S_Rec("K") // the local camera's target
    call W3S_Tok(R2I(GetCameraTargetPositionX()))
    call W3S_Tok(R2I(GetCameraTargetPositionY()))
    call W3S_End()
    call W3S_Rec("T")
    call W3S_Tok(w3s_seq)
    call W3S_Tok(R2I(TimerGetElapsed(w3s_clock) * 1000.0 + 0.5))
    call W3S_Tok(0)
    if w3s_full then
        call W3S_Tok(1)
    else
        call W3S_Tok(0)
    endif
    call W3S_End()
    call W3S_SerPlayers()
    if w3s_first then
        call W3S_SerOrders()
        call W3S_SerTrees()
    endif
    if w3s_cser == 1 then
        call GetPlayerTechMaxAllowed(w3s_np, W3S_MBOX - 11)  // (the records of W3S_CPass)
        call GetPlayerTechMaxAllowed(w3s_np, W3S_MBOX - 12)  // (and the units that left the game)
    endif
    set w3s_full = false
    set i = 0
    loop
        exitwhen i >= w3s_nev
        call W3S_Rec("E")
        call W3S_Tok(w3s_ev[i])
        call W3S_Tok(w3s_ev[i + 1])
        call W3S_Tok(w3s_ev[i + 2])
        call W3S_Tok(w3s_ev[i + 3])
        call W3S_End()
        set i = i + 4
    endloop
    set w3s_nev = 0
    call Preload("X")
    call PreloadGenEnd("w3sim\\obs.txt")
endfunction

function W3S_Step takes nothing returns nothing
    local integer n
    local integer i = 0
    // new units (entered the map since the last step; hidden ones are kept in w3s_all once seen)
    call GroupEnumUnitsInRect(w3s_group, w3s_world, null)
    call ForGroup(w3s_group, function W3S_AddUnit)
    call GroupClear(w3s_group)
    if w3s_cser == 1 then
        call W3S_CPass()
    endif
    call W3S_PollResearch()
    call W3S_WriteObs()
    set w3s_first = false
    set w3s_seq = w3s_seq + 1
    // the step sync (w3shim); the commands are read and ignored
    set n = GetPlayerTechMaxAllowed(w3s_np, W3S_MBOX - 1)
    loop
        exitwhen i >= n or i >= 8000
        call GetPlayerTechMaxAllowed(w3s_np, W3S_MBOX + 1 + i)
        set i = i + 1
    endloop
endfunction

//===========================================================================
// setup: called at the end of the map's main (after its melee initialization); every object the
// harness has is made here, between the two probe groups

function W3S_Init takes nothing returns nothing
    set w3s_probe0 = CreateGroup()
    set w3s_np = Player(PLAYER_NEUTRAL_PASSIVE)
    set w3s_ht = InitHashtable()
    set w3s_world = GetWorldBounds()
    call W3S_InitHeroAbilities()
    call W3S_InitUpgrades()
    set w3s_group = CreateGroup()
    set w3s_all = CreateGroup()
    call W3S_Seers()
    call GroupEnumUnitsInRect(w3s_group, w3s_world, null)
    call ForGroup(w3s_group, function W3S_AddUnit)
    call GroupClear(w3s_group)
    if W3S_CFG_SEND_DESTRUCTABLES then
        call EnumDestructablesInRect(bj_mapInitialPlayableArea, null, function W3S_KeepTree)
    endif
    call W3S_CSerInit()
    call ForGroup(w3s_group, function W3S_CPassEnum)  // (its code's wrapper: made now, not in a step)
    set w3s_clock = CreateTimer()
    call TimerStart(w3s_clock, 1000000.0, false, null)
    set w3s_timer0 = CreateTimer()
    call TimerStart(w3s_timer0, 0.0, false, function W3S_Step)  // the first observation at game time 0
    set w3s_timer = CreateTimer()
    call TimerStart(w3s_timer, W3S_CFG_STEP_S, true, function W3S_Step)
    set w3s_probe1 = CreateGroup()
    call GetPlayerTechMaxAllowed(w3s_np, W3S_MBOX - 40)
    call GetPlayerTechMaxAllowed(w3s_np, 0 - GetHandleId(w3s_probe0))
    call GetPlayerTechMaxAllowed(w3s_np, W3S_MBOX - 40)
    call GetPlayerTechMaxAllowed(w3s_np, 0 - GetHandleId(w3s_probe1))
endfunction
