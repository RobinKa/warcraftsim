//===========================================================================
// w3sim harness (injected into melee maps by warcraftsim.data.mapbuild)
//
// Every step (a periodic game-time timer) the harness
//   1. writes an observation as a token stream: one Preload() call per token, flushed with
//      PreloadGenEnd("w3sim\\obs.txt"). Tokens are short integers/tags only, because every
//      distinct JASS string is interned for the rest of the game.
//   2. reads the controller's commands from a mailbox: GetPlayerTechMaxAllowed on the neutral
//      passive player with key W3S_MBOX - 1 is answered by the w3shim DLL, which blocks until the
//      controller has read the observation and sent the commands (so the game is in lockstep
//      with Python), and returns their count; keys W3S_MBOX + 1 + i then return the integers.
//      (No file and no Preloader: Preloader compiled its file as JASS on every step, which cost
//      ~7 ms per step under Wine and leaked memory.)
//   3. applies the commands with the normal order natives.
//
// @CONFIG@ lines are substituted at map build time.
//===========================================================================

globals
    // @CONFIG@ (replaced at map build time)
    constant real W3S_CFG_STEP_S = 0.25
    constant real W3S_CFG_MAX_GAME_S = 0.0
    constant integer W3S_CFG_AGENT_MASK = 0
    constant boolean W3S_CFG_SCENARIO = false
    constant integer W3S_CFG_VICTORY = 0
    constant integer W3S_CFG_SCRIPTED_MASK = 0
    constant boolean W3S_CFG_SEND_DESTRUCTABLES = true
    constant real W3S_CFG_CLEAR_X = 0.0
    constant real W3S_CFG_CLEAR_Y = 0.0
    constant real W3S_CFG_CLEAR_R = 0.0
    constant integer W3S_VERSION = 4
    constant integer W3S_QS_SKILLS = 8
    constant integer W3S_ABIL_SLOTS = 4  // warcraftsim.protocol.HERO_ABILITY_SLOTS
    boolean w3s_full = true
    integer array w3s_removed
    integer w3s_nremoved = 0
    integer w3s_sum = 0
    constant integer W3S_MBOX = 1048576
    constant integer W3S_CHUNK = 40
    hashtable w3s_ht = null
    hashtable w3s_abil = null  // hero type -> ability code per slot (W3S_InitHeroAbilities)
    timer w3s_clock = null
    timer w3s_timer = null
    group w3s_group = null
    group w3s_all = null
    boolean w3s_configured = false
    boolean w3s_prune = false
    integer w3s_seq = 0
    unit array w3s_units
    integer w3s_nunits = 0
    integer w3s_cursor = 0
    trigger w3s_ser_trig = null
    trigger w3s_act_trig = null
    boolean array w3s_agent
    integer array w3s_result
    boolean w3s_over = false
    boolean w3s_first = true
    integer array w3s_ev
    integer w3s_nev = 0
    integer array w3s_cres
    integer w3s_ncres = 0
    integer array w3s_cmd
    integer w3s_ncmd = 0
    integer w3s_nplayers = 0
    boolean w3s_restart = false
    // spawns queued for the next scenario restart (op 92): random compositions chosen by Python
    integer w3s_nqs = 0
    integer array w3s_qs_player
    integer array w3s_qs_type
    real array w3s_qs_x
    real array w3s_qs_y
    real array w3s_qs_facing
    integer array w3s_qs_hp
    integer array w3s_qs_level
    integer array w3s_qs_nskill  // skills to learn (op 93), W3S_QS_SKILLS per queued spawn
    integer array w3s_qs_skill
    boolean w3s_end = false
    boolean array w3s_scripted
    integer array w3s_alive
    boolean array w3s_participant
endglobals

//===========================================================================
// tokens

// Every record is: tag, integer fields, checksum (sum of the fields, 32-bit wrap-around).
// The game's resource loader records its own preloads into the same buffer (from another
// thread), so a stray entry can land inside a record; the checksum lets the reader notice.
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

function W3S_Hid takes handle h returns integer
    if h == null then
        return 0
    endif
    return GetHandleId(h)
endfunction

//===========================================================================
// unit registry: handle id -> unit, so commands can refer to units by id

function W3S_Track takes unit u returns nothing
    call SaveUnitHandle(w3s_ht, GetHandleId(u), 0, u)
endfunction

function W3S_Unit takes integer hid returns unit
    return LoadUnitHandle(w3s_ht, hid, 0)
endfunction

// Units are collected from w3s_all (every unit that ever entered the map), not by an area
// enumeration: enumerations skip hidden units such as workers inside a gold mine.
function W3S_EnumAdd takes nothing returns nothing
    local unit u = GetEnumUnit()
    if GetUnitTypeId(u) == 0 then
        call GroupRemoveUnit(w3s_all, u) // removed from the game
        // its registry entries hold the handle: flushed, or every dead unit of every episode stays
        // referenced (a game that ran for an hour had grown by 150 MB and stepped 25% slower)
        call FlushChildHashtable(w3s_ht, GetHandleId(u))
        if w3s_prune and w3s_nremoved < 8000 then
            set w3s_removed[w3s_nremoved] = GetHandleId(u)
            set w3s_nremoved = w3s_nremoved + 1
        endif
    else
        set w3s_units[w3s_nunits] = u
        set w3s_nunits = w3s_nunits + 1
        if w3s_prune and IsUnitType(u, UNIT_TYPE_DEAD) and not IsUnitType(u, UNIT_TYPE_HERO) and LoadBoolean(w3s_ht, GetHandleId(u), 4) then
            // reported as dead already: drop it (heroes stay, they can be revived)
            call GroupRemoveUnit(w3s_all, u)
            call FlushChildHashtable(w3s_ht, GetHandleId(u))
            if w3s_nremoved < 8000 then
                set w3s_removed[w3s_nremoved] = GetHandleId(u)
                set w3s_nremoved = w3s_nremoved + 1
            endif
        endif
    endif
    set u = null
endfunction

function W3S_AddUnit takes nothing returns nothing
    call GroupAddUnit(w3s_all, GetEnumUnit())
    call W3S_Track(GetEnumUnit())
endfunction

function W3S_OnEnter takes nothing returns boolean
    call GroupAddUnit(w3s_all, GetTriggerUnit())
    call W3S_Track(GetTriggerUnit())
    return false
endfunction

//===========================================================================
// observation

function W3S_UnitFlags takes unit u returns integer
    local integer f = 0
    if IsUnitType(u, UNIT_TYPE_HERO) then
        set f = f + 1
    endif
    if IsUnitType(u, UNIT_TYPE_STRUCTURE) then
        set f = f + 2
    endif
    if IsUnitType(u, UNIT_TYPE_PEON) then
        set f = f + 4
    endif
    if LoadBoolean(w3s_ht, GetHandleId(u), 1) then
        set f = f + 8 // under construction
    endif
    if IsUnitHidden(u) then
        set f = f + 16
    endif
    if IsUnitLoaded(u) then
        set f = f + 32
    endif
    if UnitIsSleeping(u) then
        set f = f + 64
    endif
    if IsUnitPaused(u) then
        set f = f + 128
    endif
    if IsUnitType(u, UNIT_TYPE_SUMMONED) then
        set f = f + 256
    endif
    if IsUnitIllusion(u) then
        set f = f + 512
    endif
    if IsUnitType(u, UNIT_TYPE_DEAD) or GetWidgetLife(u) < 0.405 then
        set f = f + 1024
    endif
    if IsUnitType(u, UNIT_TYPE_FLYING) then
        set f = f + 2048
    endif
    return f
endfunction

function W3S_Visibility takes unit u returns integer
    local integer i = 0
    local integer bits = 0
    local integer bit = 1
    loop
        exitwhen i >= bj_MAX_PLAYERS
        if w3s_agent[i] and IsUnitVisible(u, Player(i)) then
            set bits = bits + bit
        endif
        set bit = bit * 2
        set i = i + 1
    endloop
    return bits
endfunction

// A unit record is only written when the unit changed since it was last written (a rolling
// hash over its fields is kept per unit), or on a full snapshot (first observation).
function W3S_SerUnit takes unit u returns nothing
    local integer hid = GetHandleId(u)
    local integer flags = W3S_UnitFlags(u)
    local integer typ = GetUnitTypeId(u)
    local integer owner = GetPlayerId(GetOwningPlayer(u))
    local integer x = R2I(GetUnitX(u))
    local integer y = R2I(GetUnitY(u))
    local integer facing = R2I(GetUnitFacing(u))
    local integer hp = R2I(GetWidgetLife(u) + 0.5)
    local integer maxhp = R2I(GetUnitState(u, UNIT_STATE_MAX_LIFE) + 0.5)
    local integer mana = R2I(GetUnitState(u, UNIT_STATE_MANA))
    local integer maxmana = R2I(GetUnitState(u, UNIT_STATE_MAX_MANA))
    local integer order = GetUnitCurrentOrder(u)
    local integer vis = W3S_Visibility(u)
    local integer res = GetResourceAmount(u)
    local integer h = ((((((((((typ * 31 + owner) * 31 + x) * 31 + y) * 31 + facing) * 31 + hp) * 31 + maxhp) * 31 + mana) * 31 + maxmana) * 31 + order) * 31 + flags) * 31 + vis
    local integer i
    local integer a
    local integer array alevel
    local integer array acool
    set h = h * 31 + res
    if IsUnitType(u, UNIT_TYPE_HERO) then
        set h = (h * 31 + GetHeroXP(u)) * 31 + GetHeroSkillPoints(u)
        set i = 0
        loop
            exitwhen i >= 6
            set h = h * 31 + GetItemTypeId(UnitItemInSlot(u, i))
            set i = i + 1
        endloop
        // each ability slot: learned level and cooldown left (0.1 s)
        set i = 0
        loop
            exitwhen i >= W3S_ABIL_SLOTS
            set a = LoadInteger(w3s_abil, typ, i)
            set alevel[i] = 0
            set acool[i] = 0
            if a != 0 then
                set alevel[i] = GetUnitAbilityLevel(u, a)
                if alevel[i] > 0 then
                    set acool[i] = R2I(BlzGetUnitAbilityCooldownRemaining(u, a) * 10.0 + 0.5)
                endif
            endif
            set h = (h * 31 + alevel[i]) * 31 + acool[i]
            set i = i + 1
        endloop
    endif
    if not w3s_full and HaveSavedInteger(w3s_ht, hid, 3) and LoadInteger(w3s_ht, hid, 3) == h then
        return
    endif
    call SaveInteger(w3s_ht, hid, 3, h)
    if IsUnitType(u, UNIT_TYPE_DEAD) then
        call SaveBoolean(w3s_ht, hid, 4, true)
    endif
    call W3S_Rec("U")
    call W3S_Tok(hid)
    call W3S_Tok(typ)
    call W3S_Tok(owner)
    call W3S_Tok(x)
    call W3S_Tok(y)
    call W3S_Tok(facing)
    call W3S_Tok(hp)
    call W3S_Tok(maxhp)
    call W3S_Tok(mana)
    call W3S_Tok(maxmana)
    call W3S_Tok(order)
    call W3S_Tok(flags)
    call W3S_Tok(vis)
    call W3S_Tok(res)
    if IsUnitType(u, UNIT_TYPE_HERO) then
        call W3S_Tok(GetHeroLevel(u))
        call W3S_Tok(GetHeroXP(u))
        call W3S_Tok(GetHeroSkillPoints(u))
        set i = 0
        loop
            exitwhen i >= 6
            call W3S_Tok(GetItemTypeId(UnitItemInSlot(u, i)))
            set i = i + 1
        endloop
        set i = 0
        loop
            exitwhen i >= W3S_ABIL_SLOTS
            call W3S_Tok(alevel[i])
            call W3S_Tok(acool[i])
            set i = i + 1
        endloop
    endif
    call W3S_End()
endfunction

function W3S_SerChunk takes nothing returns boolean
    local integer stop = w3s_cursor + W3S_CHUNK
    if stop > w3s_nunits then
        set stop = w3s_nunits
    endif
    loop
        exitwhen w3s_cursor >= stop
        call W3S_SerUnit(w3s_units[w3s_cursor])
        set w3s_units[w3s_cursor] = null
        set w3s_cursor = w3s_cursor + 1
    endloop
    return false
endfunction

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
            if w3s_agent[i] then
                call W3S_Tok(1)
            else
                call W3S_Tok(0)
            endif
            call W3S_Tok(GetPlayerState(p, PLAYER_STATE_RESOURCE_GOLD))
            call W3S_Tok(GetPlayerState(p, PLAYER_STATE_RESOURCE_LUMBER))
            call W3S_Tok(GetPlayerState(p, PLAYER_STATE_RESOURCE_FOOD_USED))
            call W3S_Tok(GetPlayerState(p, PLAYER_STATE_RESOURCE_FOOD_CAP))
            call W3S_Tok(GetPlayerState(p, PLAYER_STATE_GOLD_UPKEEP_RATE))
            call W3S_Tok(GetPlayerState(p, PLAYER_STATE_GOLD_GATHERED))
            call W3S_Tok(GetPlayerState(p, PLAYER_STATE_LUMBER_GATHERED))
            call W3S_Tok(GetPlayerStructureCount(p, true))
            call W3S_Tok(w3s_result[i])
            call W3S_Tok(R2I(GetStartLocationX(GetPlayerStartLocation(p))))
            call W3S_Tok(R2I(GetStartLocationY(GetPlayerStartLocation(p))))
            call W3S_End()
        endif
        set i = i + 1
    endloop
    set p = null
endfunction

function W3S_SerDestructable takes nothing returns nothing
    local destructable d = GetEnumDestructable()
    call SaveDestructableHandle(w3s_ht, GetHandleId(d), 0, d)
    call W3S_Rec("D")
    call W3S_Tok(GetHandleId(d))
    call W3S_Tok(GetDestructableTypeId(d))
    call W3S_Tok(R2I(GetDestructableX(d)))
    call W3S_Tok(R2I(GetDestructableY(d)))
    call W3S_Tok(R2I(GetDestructableLife(d) + 0.5))
    call W3S_End()
    set d = null
endfunction

function W3S_TreeDied takes nothing returns boolean
    call W3S_Ev(16, GetHandleId(GetTriggerDestructable()), 0, 0)
    return false
endfunction

function W3S_RegTree takes nothing returns nothing
    local trigger t = CreateTrigger()
    call TriggerRegisterDeathEvent(t, GetEnumDestructable())
    call TriggerAddCondition(t, Condition(function W3S_TreeDied))
    set t = null
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

function W3S_WriteObs takes nothing returns nothing
    local integer i
    set w3s_full = w3s_full or w3s_first
    call PreloadGenClear()
    call PreloadGenStart()
    call Preload("V")
    call W3S_Tok(W3S_VERSION)
    call W3S_Rec("T")
    call W3S_Tok(w3s_seq)
    call W3S_Tok(R2I(TimerGetElapsed(w3s_clock) * 1000.0 + 0.5))
    if w3s_over then
        call W3S_Tok(1)
    else
        call W3S_Tok(0)
    endif
    if w3s_full then
        call W3S_Tok(1)
    else
        call W3S_Tok(0)
    endif
    call W3S_End()
    call W3S_SerPlayers()
    if w3s_first then
        call W3S_SerOrders()
        if W3S_CFG_SEND_DESTRUCTABLES then
            call EnumDestructablesInRect(bj_mapInitialPlayableArea, null, function W3S_SerDestructable)
        endif
    endif
    // units
    set w3s_nunits = 0
    set w3s_cursor = 0
    set w3s_prune = true
    call ForGroup(w3s_all, function W3S_EnumAdd)
    set w3s_prune = false
    loop
        exitwhen w3s_cursor >= w3s_nunits
        set i = w3s_cursor
        call TriggerEvaluate(w3s_ser_trig)
        exitwhen w3s_cursor == i // a crashed chunk must not hang the game
    endloop
    set w3s_full = false
    // units that left the game
    set i = 0
    loop
        exitwhen i >= w3s_nremoved
        call W3S_Rec("R")
        call W3S_Tok(w3s_removed[i])
        call W3S_End()
        set i = i + 1
    endloop
    set w3s_nremoved = 0
    // events
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
    // results of the previous commands
    set i = 0
    loop
        exitwhen i >= w3s_ncres
        call W3S_Rec("C")
        call W3S_Tok(w3s_cres[i])
        call W3S_End()
        set i = i + 1
    endloop
    set w3s_ncres = 0
    call Preload("X")
    call PreloadGenEnd("w3sim\\obs.txt")
endfunction

//===========================================================================
// commands

function W3S_Mbox takes integer i returns integer
    return GetPlayerTechMaxAllowed(Player(PLAYER_NEUTRAL_PASSIVE), W3S_MBOX + i)
endfunction

// orders are carried out for agents' units and scripted opponents' (e.g. their casts, chosen in Python)
function W3S_Owned takes unit u returns boolean
    local integer p
    if u == null then
        return false
    endif
    set p = GetPlayerId(GetOwningPlayer(u))
    return w3s_agent[p] or w3s_scripted[p]
endfunction

function W3S_CountAliveEnum takes nothing returns nothing
    local unit u = GetEnumUnit()
    local integer p = GetPlayerId(GetOwningPlayer(u))
    if p < bj_MAX_PLAYERS and not IsUnitType(u, UNIT_TYPE_DEAD) and GetUnitTypeId(u) != 0 then
        set w3s_alive[p] = w3s_alive[p] + 1
    endif
    set u = null
endfunction

function W3S_CountAlive takes nothing returns nothing
    local integer i = 0
    loop
        exitwhen i >= bj_MAX_PLAYERS
        set w3s_alive[i] = 0
        set i = i + 1
    endloop
    call ForGroup(w3s_all, function W3S_CountAliveEnum)
endfunction

// Command layout (integers): op, then op-specific arguments. Coordinates are sent + 65536.
//  1 point order     hid order x y
//  2 target order    hid order target_hid
//  3 immediate order hid order            (also train / research / upgrade with order = type id)
//  4 build           hid unittype x y
//  5 learn skill     hid ability
//  6 target tree     hid order dest_hid
//  7 item order      hid order item_slot x y target_hid
// 90 set resources   player gold lumber   (debug)
// 91 spawn unit      player unittype x y  (debug)
// 99 restart game
function W3S_ApplyOne takes integer at returns integer
    local integer op = w3s_cmd[at]
    local unit u = W3S_Unit(w3s_cmd[at + 1])
    local boolean ok = false
    local integer n = 1
    if op == 1 then
        set n = 5
        if W3S_Owned(u) then
            set ok = IssuePointOrderById(u, w3s_cmd[at + 2], I2R(w3s_cmd[at + 3] - 65536), I2R(w3s_cmd[at + 4] - 65536))
        endif
    elseif op == 2 then
        set n = 4
        if W3S_Owned(u) then
            set ok = IssueTargetOrderById(u, w3s_cmd[at + 2], W3S_Unit(w3s_cmd[at + 3]))
        endif
    elseif op == 3 then
        set n = 3
        if W3S_Owned(u) then
            set ok = IssueImmediateOrderById(u, w3s_cmd[at + 2])
        endif
    elseif op == 4 then
        set n = 5
        if W3S_Owned(u) then
            set ok = IssueBuildOrderById(u, w3s_cmd[at + 2], I2R(w3s_cmd[at + 3] - 65536), I2R(w3s_cmd[at + 4] - 65536))
        endif
    elseif op == 5 then
        set n = 3
        if W3S_Owned(u) and GetHeroSkillPoints(u) > 0 then
            call SelectHeroSkill(u, w3s_cmd[at + 2])
            set ok = true
        endif
    elseif op == 6 then
        set n = 4
        if W3S_Owned(u) then
            set ok = IssueTargetOrderById(u, w3s_cmd[at + 2], LoadDestructableHandle(w3s_ht, w3s_cmd[at + 3], 0))
        endif
    elseif op == 7 then
        set n = 7
        if W3S_Owned(u) then
            if w3s_cmd[at + 6] != 0 then
                set ok = UnitUseItemTarget(u, UnitItemInSlot(u, w3s_cmd[at + 3]), W3S_Unit(w3s_cmd[at + 6]))
            elseif w3s_cmd[at + 4] != 0 then
                set ok = UnitUseItemPoint(u, UnitItemInSlot(u, w3s_cmd[at + 3]), I2R(w3s_cmd[at + 4] - 65536), I2R(w3s_cmd[at + 5] - 65536))
            else
                set ok = UnitUseItem(u, UnitItemInSlot(u, w3s_cmd[at + 3]))
            endif
        endif
    elseif op == 90 then
        set n = 4
        call SetPlayerState(Player(w3s_cmd[at + 1]), PLAYER_STATE_RESOURCE_GOLD, w3s_cmd[at + 2])
        call SetPlayerState(Player(w3s_cmd[at + 1]), PLAYER_STATE_RESOURCE_LUMBER, w3s_cmd[at + 3])
        set ok = true
    elseif op == 92 then
        // queue a spawn for the next restart: player, type, x, y, facing, hp per mille, hero level
        set n = 8
        if w3s_nqs < 64 then
            set w3s_qs_player[w3s_nqs] = w3s_cmd[at + 1]
            set w3s_qs_type[w3s_nqs] = w3s_cmd[at + 2]
            set w3s_qs_x[w3s_nqs] = I2R(w3s_cmd[at + 3] - 65536)
            set w3s_qs_y[w3s_nqs] = I2R(w3s_cmd[at + 4] - 65536)
            set w3s_qs_facing[w3s_nqs] = I2R(w3s_cmd[at + 5])
            set w3s_qs_hp[w3s_nqs] = w3s_cmd[at + 6]
            set w3s_qs_level[w3s_nqs] = w3s_cmd[at + 7]
            set w3s_qs_nskill[w3s_nqs] = 0
            set w3s_nqs = w3s_nqs + 1
            set ok = true
        endif
    elseif op == 93 then
        // the spawn queued last learns a skill when it spawns: ability code
        set n = 2
        if w3s_nqs > 0 and w3s_qs_nskill[w3s_nqs - 1] < W3S_QS_SKILLS then
            set w3s_qs_skill[(w3s_nqs - 1) * W3S_QS_SKILLS + w3s_qs_nskill[w3s_nqs - 1]] = w3s_cmd[at + 1]
            set w3s_qs_nskill[w3s_nqs - 1] = w3s_qs_nskill[w3s_nqs - 1] + 1
            set ok = true
        endif
    elseif op == 91 then
        set n = 5
        set u = CreateUnit(Player(w3s_cmd[at + 1]), w3s_cmd[at + 2], I2R(w3s_cmd[at + 3] - 65536), I2R(w3s_cmd[at + 4] - 65536), 270.0)
        set ok = u != null
    elseif op == 96 then
        set n = 3
        // local camera only: watching / videos, no effect on the simulation; a pan over one step
        // keeps frame-by-frame video smooth
        call PanCameraToTimed(I2R(w3s_cmd[at + 1] - 65536), I2R(w3s_cmd[at + 2] - 65536), W3S_CFG_STEP_S)
        set ok = true
    elseif op == 97 then
        set n = 1
        set w3s_end = true
        set ok = true
    elseif op == 98 then
        set n = 1
        set w3s_full = true
        set ok = true
    elseif op == 99 then
        set n = 1
        set w3s_restart = true
        set ok = true
    else
        set n = 1000000
    endif
    if ok then
        set w3s_cres[w3s_ncres] = 1
    else
        set w3s_cres[w3s_ncres] = 0
    endif
    set w3s_ncres = w3s_ncres + 1
    set u = null
    return n
endfunction

function W3S_ApplyChunk takes nothing returns boolean
    local integer done = 0
    loop
        exitwhen w3s_cursor >= w3s_ncmd or done >= 50
        set w3s_cursor = w3s_cursor + W3S_ApplyOne(w3s_cursor)
        set done = done + 1
    endloop
    return false
endfunction

function W3S_ReadCommands takes nothing returns nothing
    local integer i = 0
    set w3s_ncmd = W3S_Mbox(-1)  // the step sync (w3shim)
    if w3s_ncmd > 8000 then
        set w3s_ncmd = 8000
    endif
    loop
        exitwhen i >= w3s_ncmd
        set w3s_cmd[i] = W3S_Mbox(i + 1)
        set i = i + 1
    endloop
    set w3s_cursor = 0
    loop
        exitwhen w3s_cursor >= w3s_ncmd
        set i = w3s_cursor
        call TriggerEvaluate(w3s_act_trig)
        exitwhen w3s_cursor == i
    endloop
endfunction

//===========================================================================
// game result: melee rules (a player is defeated when its team has no structures left),
// but without removing players or showing dialogs, so the session stays alive.

function W3S_CheckResult takes nothing returns nothing
    local integer i = 0
    local integer j
    local integer alive = 0
    local boolean allied = true
    local player p
    if w3s_over then
        return
    endif
    if W3S_CFG_VICTORY == 1 then
        call W3S_CountAlive()
    endif
    loop
        exitwhen i >= bj_MAX_PLAYERS or W3S_CFG_VICTORY == 2
        set p = Player(i)
        if w3s_result[i] == 0 and GetPlayerSlotState(p) == PLAYER_SLOT_STATE_PLAYING and not IsPlayerObserver(p) then
            if W3S_CFG_VICTORY == 0 and MeleeGetAllyStructureCount(p) <= 0 then
                set w3s_result[i] = 2
            elseif W3S_CFG_VICTORY == 1 and w3s_participant[i] and w3s_alive[i] <= 0 then
                set w3s_result[i] = 2
            endif
        endif
        set i = i + 1
    endloop
    // victory: all undefeated players are mutually allied
    set i = 0
    loop
        exitwhen i >= bj_MAX_PLAYERS
        if w3s_result[i] == 0 and GetPlayerSlotState(Player(i)) == PLAYER_SLOT_STATE_PLAYING and not IsPlayerObserver(Player(i)) then
            set alive = alive + 1
            set j = 0
            loop
                exitwhen j >= bj_MAX_PLAYERS
                if j != i and w3s_result[j] == 0 and GetPlayerSlotState(Player(j)) == PLAYER_SLOT_STATE_PLAYING and not IsPlayerObserver(Player(j)) then
                    if not PlayersAreCoAllied(Player(i), Player(j)) then
                        set allied = false
                    endif
                endif
                set j = j + 1
            endloop
        endif
        set i = i + 1
    endloop
    if W3S_CFG_VICTORY != 2 and (alive == 0 or allied) then
        set w3s_over = true
        set i = 0
        loop
            exitwhen i >= bj_MAX_PLAYERS
            if w3s_result[i] == 0 and GetPlayerSlotState(Player(i)) == PLAYER_SLOT_STATE_PLAYING and not IsPlayerObserver(Player(i)) then
                set w3s_result[i] = 1
            endif
            set i = i + 1
        endloop
    endif
    if W3S_CFG_MAX_GAME_S > 0 and not w3s_over and TimerGetElapsed(w3s_clock) >= W3S_CFG_MAX_GAME_S then
        set w3s_over = true
        set i = 0
        loop
            exitwhen i >= bj_MAX_PLAYERS
            if w3s_result[i] == 0 and GetPlayerSlotState(Player(i)) == PLAYER_SLOT_STATE_PLAYING and not IsPlayerObserver(Player(i)) then
                set w3s_result[i] = 3
            endif
            set i = i + 1
        endloop
    endif
    set p = null
endfunction

//===========================================================================
// scenarios: remove everything, spawn the configured units; reset = do it again

function W3S_SpawnUnit takes integer p, integer unitType, real x, real y, real facing, integer hp returns nothing
    local unit u = CreateUnit(Player(p), unitType, x, y, facing)
    if hp > 0 then
        call BlzSetUnitMaxHP(u, hp)
        call SetUnitState(u, UNIT_STATE_LIFE, I2R(hp))
    endif
    call RemoveGuardPosition(u)
    call GroupAddUnit(w3s_all, u)
    call W3S_Track(u)
    set w3s_participant[p] = true
    set u = null
endfunction

// A queued spawn (op 92): hit points scaled to `permille` of the unit's own maximum (heroes
// included), heroes raised to `level`.
function W3S_SpawnQueued takes integer i returns nothing
    local unit u = CreateUnit(Player(w3s_qs_player[i]), w3s_qs_type[i], w3s_qs_x[i], w3s_qs_y[i], w3s_qs_facing[i])
    local integer hp
    local integer j
    if u == null then
        return
    endif
    if w3s_qs_level[i] > 1 and IsUnitType(u, UNIT_TYPE_HERO) then
        call SetHeroLevel(u, w3s_qs_level[i], false)
    endif
    set j = 0
    loop
        exitwhen j >= w3s_qs_nskill[i]
        call SelectHeroSkill(u, w3s_qs_skill[i * W3S_QS_SKILLS + j])
        set j = j + 1
    endloop
    if w3s_qs_hp[i] > 0 then
        set hp = R2I(GetUnitState(u, UNIT_STATE_MAX_LIFE) * I2R(w3s_qs_hp[i]) / 1000.0 + 0.5)
        if hp < 1 then
            set hp = 1
        endif
        call BlzSetUnitMaxHP(u, hp)
        call SetUnitState(u, UNIT_STATE_LIFE, I2R(hp))
    endif
    call RemoveGuardPosition(u)
    call GroupAddUnit(w3s_all, u)
    call W3S_Track(u)
    set w3s_participant[w3s_qs_player[i]] = true
    set u = null
endfunction

function W3S_ScenarioSpawn takes nothing returns nothing
    local integer i = 0
    // @SCENARIO_SPAWN@
    loop
        exitwhen i >= w3s_nqs
        call W3S_SpawnQueued(i)
        set i = i + 1
    endloop
    set w3s_nqs = 0
endfunction

function W3S_RemoveEnum takes nothing returns nothing
    local unit u = GetEnumUnit()
    call FlushChildHashtable(w3s_ht, GetHandleId(u))
    call RemoveUnit(u)
    set u = null
endfunction

function W3S_ClearTree takes nothing returns nothing
    local destructable d = GetEnumDestructable()
    local real dx = GetDestructableX(d) - W3S_CFG_CLEAR_X
    local real dy = GetDestructableY(d) - W3S_CFG_CLEAR_Y
    if dx * dx + dy * dy <= W3S_CFG_CLEAR_R * W3S_CFG_CLEAR_R then
        call RemoveDestructable(d)
    endif
    set d = null
endfunction

function W3S_ScenarioSetup takes nothing returns nothing
    local integer i = 0
    call ForGroup(w3s_all, function W3S_RemoveEnum)
    call GroupClear(w3s_all)
    loop
        exitwhen i >= bj_MAX_PLAYERS
        set w3s_participant[i] = false
        set w3s_result[i] = 0
        set i = i + 1
    endloop
    call W3S_ScenarioSpawn()
    set w3s_nev = 0
endfunction

function W3S_ScenarioReset takes nothing returns nothing
    call W3S_ScenarioSetup()
    set w3s_full = true
    set w3s_nremoved = 0
    set w3s_seq = 0
    set w3s_first = true
    set w3s_over = false
    set w3s_ncres = 0
    call TimerStart(w3s_clock, 1000000.0, false, null)
endfunction

// Scripted opponent: idle units attack-move to the nearest enemy.
function W3S_ScriptedUnit takes unit u returns nothing
    local unit best = null
    local unit v
    local real bd = 1000000000.0
    local real d
    local integer i = 0
    loop
        exitwhen i >= w3s_nunits
        set v = w3s_units[i]
        if v != null and IsUnitEnemy(v, GetOwningPlayer(u)) and not IsUnitType(v, UNIT_TYPE_DEAD) and GetPlayerId(GetOwningPlayer(v)) < bj_MAX_PLAYERS then
            set d = (GetUnitX(v) - GetUnitX(u)) * (GetUnitX(v) - GetUnitX(u)) + (GetUnitY(v) - GetUnitY(u)) * (GetUnitY(v) - GetUnitY(u))
            if d < bd then
                set bd = d
                set best = v
            endif
        endif
        set i = i + 1
    endloop
    if best != null then
        call IssuePointOrderById(u, 851983, GetUnitX(best), GetUnitY(best))
    endif
    set best = null
    set v = null
endfunction

function W3S_ScriptedChunk takes nothing returns boolean
    local unit u
    local integer stop = w3s_cursor + 10
    if stop > w3s_nunits then
        set stop = w3s_nunits
    endif
    loop
        exitwhen w3s_cursor >= stop
        set u = w3s_units[w3s_cursor]
        if w3s_scripted[GetPlayerId(GetOwningPlayer(u))] and not IsUnitType(u, UNIT_TYPE_DEAD) and GetUnitCurrentOrder(u) == 0 and not IsUnitType(u, UNIT_TYPE_STRUCTURE) then
            call W3S_ScriptedUnit(u)
        endif
        set w3s_cursor = w3s_cursor + 1
    endloop
    set u = null
    return false
endfunction

function W3S_RunScripted takes nothing returns nothing
    local integer before
    local trigger t
    if W3S_CFG_SCRIPTED_MASK == 0 then
        return
    endif
    set w3s_nunits = 0
    set w3s_cursor = 0
    call ForGroup(w3s_all, function W3S_EnumAdd)
    set t = CreateTrigger()
    call TriggerAddCondition(t, Condition(function W3S_ScriptedChunk))
    loop
        exitwhen w3s_cursor >= w3s_nunits
        set before = w3s_cursor
        call TriggerEvaluate(t)
        exitwhen w3s_cursor == before
    endloop
    call DestroyTrigger(t)
    set t = null
endfunction

//===========================================================================
// step

function W3S_Step takes nothing returns nothing
    call W3S_RunScripted()
    call W3S_CheckResult()
    call W3S_WriteObs()
    set w3s_first = false
    set w3s_seq = w3s_seq + 1
    call W3S_ReadCommands()
    if w3s_end then
        // end the game normally so the engine finalises the replay (LastReplay.w3g)
        set w3s_end = false
        call PauseTimer(w3s_timer)
        call EndGame(false)
        return
    endif
    if w3s_restart then
        set w3s_restart = false
        if W3S_CFG_SCENARIO then
            call W3S_ScenarioReset()
        else
            call PauseTimer(w3s_timer)
            call RestartGame(false)
        endif
    endif
endfunction

function W3S_FirstStep takes nothing returns nothing
    call W3S_Step()
    call TimerStart(w3s_timer, W3S_CFG_STEP_S, true, function W3S_Step)
endfunction

//===========================================================================
// events

function W3S_OnDeath takes nothing returns boolean
    call W3S_Ev(1, GetHandleId(GetTriggerUnit()), W3S_Hid(GetKillingUnit()), GetUnitTypeId(GetTriggerUnit()))
    return false
endfunction

function W3S_OnConstructStart takes nothing returns boolean
    local unit u = GetConstructingStructure()
    call SaveBoolean(w3s_ht, GetHandleId(u), 1, true)
    call W3S_Track(u)
    call W3S_Ev(2, GetHandleId(u), GetUnitTypeId(u), 0)
    set u = null
    return false
endfunction

function W3S_OnConstructFinish takes nothing returns boolean
    local unit u = GetConstructedStructure()
    call SaveBoolean(w3s_ht, GetHandleId(u), 1, false)
    call W3S_Ev(3, GetHandleId(u), GetUnitTypeId(u), 0)
    set u = null
    return false
endfunction

function W3S_OnConstructCancel takes nothing returns boolean
    call W3S_Ev(4, GetHandleId(GetCancelledStructure()), 0, 0)
    return false
endfunction

function W3S_OnTrainStart takes nothing returns boolean
    call W3S_Ev(5, GetHandleId(GetTriggerUnit()), GetTrainedUnitType(), 0)
    return false
endfunction

function W3S_OnTrainFinish takes nothing returns boolean
    call W3S_Track(GetTrainedUnit())
    call W3S_Ev(6, GetHandleId(GetTriggerUnit()), GetHandleId(GetTrainedUnit()), GetUnitTypeId(GetTrainedUnit()))
    return false
endfunction

function W3S_OnTrainCancel takes nothing returns boolean
    call W3S_Ev(7, GetHandleId(GetTriggerUnit()), GetTrainedUnitType(), 0)
    return false
endfunction

function W3S_OnResearchStart takes nothing returns boolean
    call W3S_Ev(8, GetHandleId(GetTriggerUnit()), GetResearched(), 0)
    return false
endfunction

function W3S_OnResearchFinish takes nothing returns boolean
    call W3S_Ev(9, GetHandleId(GetTriggerUnit()), GetResearched(), GetPlayerTechCount(GetOwningPlayer(GetTriggerUnit()), GetResearched(), true))
    return false
endfunction

function W3S_OnResearchCancel takes nothing returns boolean
    call W3S_Ev(10, GetHandleId(GetTriggerUnit()), GetResearched(), 0)
    return false
endfunction

function W3S_OnUpgradeStart takes nothing returns boolean
    call SaveBoolean(w3s_ht, GetHandleId(GetTriggerUnit()), 2, true)
    call W3S_Ev(11, GetHandleId(GetTriggerUnit()), GetUnitTypeId(GetTriggerUnit()), 0)
    return false
endfunction

function W3S_OnUpgradeFinish takes nothing returns boolean
    call SaveBoolean(w3s_ht, GetHandleId(GetTriggerUnit()), 2, false)
    call W3S_Ev(12, GetHandleId(GetTriggerUnit()), GetUnitTypeId(GetTriggerUnit()), 0)
    return false
endfunction

function W3S_OnUpgradeCancel takes nothing returns boolean
    call SaveBoolean(w3s_ht, GetHandleId(GetTriggerUnit()), 2, false)
    call W3S_Ev(13, GetHandleId(GetTriggerUnit()), GetUnitTypeId(GetTriggerUnit()), 0)
    return false
endfunction

function W3S_OnHeroLevel takes nothing returns boolean
    call W3S_Ev(14, GetHandleId(GetTriggerUnit()), GetHeroLevel(GetTriggerUnit()), 0)
    return false
endfunction

function W3S_OnSpell takes nothing returns boolean
    call W3S_Ev(15, GetHandleId(GetTriggerUnit()), GetSpellAbilityId(), W3S_Hid(GetSpellTargetUnit()))
    return false
endfunction

function W3S_OnSummon takes nothing returns boolean
    call W3S_Track(GetSummonedUnit())
    call W3S_Ev(17, GetHandleId(GetSummoningUnit()), GetHandleId(GetSummonedUnit()), GetUnitTypeId(GetSummonedUnit()))
    return false
endfunction

function W3S_OnItemPickup takes nothing returns boolean
    call W3S_Ev(18, GetHandleId(GetTriggerUnit()), GetItemTypeId(GetManipulatedItem()), 0)
    return false
endfunction

function W3S_Reg takes playerunitevent e, code c returns nothing
    local trigger t = CreateTrigger()
    local integer i = 0
    loop
        exitwhen i >= bj_MAX_PLAYER_SLOTS
        call TriggerRegisterPlayerUnitEvent(t, Player(i), e, null)
        set i = i + 1
    endloop
    call TriggerAddCondition(t, Condition(c))
    set t = null
endfunction

//===========================================================================
// setup

function W3S_Pow2 takes integer i returns integer
    local integer v = 1
    loop
        exitwhen i <= 0
        set v = v * 2
        set i = i - 1
    endloop
    return v
endfunction

function W3S_Configure takes nothing returns nothing
    local integer i = 0
    if w3s_configured then
        return
    endif
    set w3s_configured = true
    loop
        exitwhen i >= bj_MAX_PLAYERS
        set w3s_agent[i] = ModuloInteger(W3S_CFG_AGENT_MASK / W3S_Pow2(i), 2) == 1
        set w3s_scripted[i] = ModuloInteger(W3S_CFG_SCRIPTED_MASK / W3S_Pow2(i), 2) == 1
        set i = i + 1
    endloop
endfunction

// Melee AI for computer players that are not controlled by an agent. Agent and scripted
// computer players get no AI and no AI guard positions (their units would stay tethered to them).
function W3S_StartingAI takes nothing returns nothing
    local integer index = 0
    local player p
    local race r
    call W3S_Configure()
    loop
        set p = Player(index)
        if w3s_agent[index] or w3s_scripted[index] then
            call RemoveAllGuardPositions(p)
        endif
        set index = index + 1
        exitwhen index == bj_MAX_PLAYERS
    endloop
    set index = 0
    loop
        set p = Player(index)
        if GetPlayerSlotState(p) == PLAYER_SLOT_STATE_PLAYING and GetPlayerController(p) == MAP_CONTROL_COMPUTER and not w3s_agent[index] and not w3s_scripted[index] then
            set r = GetPlayerRace(p)
            if r == RACE_HUMAN then
                call PickMeleeAI(p, "human.ai", null, null)
            elseif r == RACE_ORC then
                call PickMeleeAI(p, "orc.ai", null, null)
            elseif r == RACE_UNDEAD then
                call PickMeleeAI(p, "undead.ai", null, null)
                call RecycleGuardPosition(bj_ghoul[index])
            elseif r == RACE_NIGHTELF then
                call PickMeleeAI(p, "elf.ai", null, null)
            endif
            call ShareEverythingWithTeamAI(p)
        endif
        set index = index + 1
        exitwhen index == bj_MAX_PLAYERS
    endloop
    set p = null
endfunction

// Replaces MeleeInitVictoryDefeat: results are tracked by W3S_CheckResult.
function W3S_InitVictoryDefeat takes nothing returns nothing
endfunction

function W3S_Init takes nothing returns nothing
    local trigger t
    local region r
    set w3s_ht = InitHashtable()
    call W3S_InitHeroAbilities()
    set w3s_group = CreateGroup()
    set w3s_all = CreateGroup()
    call W3S_Configure()
    call GroupEnumUnitsInRect(w3s_group, GetWorldBounds(), null)
    call ForGroup(w3s_group, function W3S_AddUnit)
    call GroupClear(w3s_group)
    set r = CreateRegion()
    call RegionAddRect(r, GetWorldBounds())
    set t = CreateTrigger()
    call TriggerRegisterEnterRegion(t, r, null)
    call TriggerAddCondition(t, Condition(function W3S_OnEnter))
    set w3s_clock = CreateTimer()
    call TimerStart(w3s_clock, 1000000.0, false, null)
    set w3s_timer = CreateTimer()
    set w3s_ser_trig = CreateTrigger()
    call TriggerAddCondition(w3s_ser_trig, Condition(function W3S_SerChunk))
    set w3s_act_trig = CreateTrigger()
    call TriggerAddCondition(w3s_act_trig, Condition(function W3S_ApplyChunk))
    call W3S_Reg(EVENT_PLAYER_UNIT_DEATH, function W3S_OnDeath)
    call W3S_Reg(EVENT_PLAYER_UNIT_CONSTRUCT_START, function W3S_OnConstructStart)
    call W3S_Reg(EVENT_PLAYER_UNIT_CONSTRUCT_FINISH, function W3S_OnConstructFinish)
    call W3S_Reg(EVENT_PLAYER_UNIT_CONSTRUCT_CANCEL, function W3S_OnConstructCancel)
    call W3S_Reg(EVENT_PLAYER_UNIT_TRAIN_START, function W3S_OnTrainStart)
    call W3S_Reg(EVENT_PLAYER_UNIT_TRAIN_FINISH, function W3S_OnTrainFinish)
    call W3S_Reg(EVENT_PLAYER_UNIT_TRAIN_CANCEL, function W3S_OnTrainCancel)
    call W3S_Reg(EVENT_PLAYER_UNIT_RESEARCH_START, function W3S_OnResearchStart)
    call W3S_Reg(EVENT_PLAYER_UNIT_RESEARCH_FINISH, function W3S_OnResearchFinish)
    call W3S_Reg(EVENT_PLAYER_UNIT_RESEARCH_CANCEL, function W3S_OnResearchCancel)
    call W3S_Reg(EVENT_PLAYER_UNIT_UPGRADE_START, function W3S_OnUpgradeStart)
    call W3S_Reg(EVENT_PLAYER_UNIT_UPGRADE_FINISH, function W3S_OnUpgradeFinish)
    call W3S_Reg(EVENT_PLAYER_UNIT_UPGRADE_CANCEL, function W3S_OnUpgradeCancel)
    call W3S_Reg(EVENT_PLAYER_HERO_LEVEL, function W3S_OnHeroLevel)
    call W3S_Reg(EVENT_PLAYER_UNIT_SPELL_EFFECT, function W3S_OnSpell)
    call W3S_Reg(EVENT_PLAYER_UNIT_SUMMON, function W3S_OnSummon)
    call W3S_Reg(EVENT_PLAYER_UNIT_PICKUP_ITEM, function W3S_OnItemPickup)
    if W3S_CFG_SCENARIO then
        if W3S_CFG_CLEAR_R > 0.0 then
            call EnumDestructablesInRect(bj_mapInitialPlayableArea, null, function W3S_ClearTree)
        endif
        call W3S_ScenarioSetup()
        call SetCameraPosition(W3S_CFG_CLEAR_X, W3S_CFG_CLEAR_Y) // local camera: for watching only
    endif
    call EnumDestructablesInRect(bj_mapInitialPlayableArea, null, function W3S_RegTree)
    set t = null
    set r = null
    // first observation at game time 0, then every step
    call TimerStart(w3s_timer, 0.0, false, function W3S_FirstStep)
endfunction
