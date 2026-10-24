/* The harness's pass over the units, made here instead of in JASS (w3sim.j: W3S_EnumAdd,
 * W3S_CountDecisive / W3S_CountAlive, W3S_SerUnit).
 *
 * A unit's record takes ~35 native calls and a few hundred JASS instructions, for every unit on
 * every step, and two more passes collect the units and count them for the result: with 150 units
 * that was 10 of a step's ~40 ms. The natives are plain cdecl functions in the executable (handles
 * and integers by value, a real returned as its bits in eax), so the same calls can be made from
 * here. The harness calls GetPlayerTechMaxAllowed(neutral passive, key) (the mailbox's native):
 *
 *   MBOX - 2                     1 if this is installed (2: only to check it, see below); a game's start
 *   MBOX - 5, then -(handle id)  the harness's hashtable (w3s_ht)
 *   MBOX - 6, then -(handle id)  its hero ability table (w3s_abil)
 *   MBOX - 7                     forget the seers
 *   MBOX - 8, then -(index + 1)  a seer (a player whose view the visibility bits report)
 *   MBOX - 13, then -(handle id) the group of all units (w3s_all)
 *   MBOX - 14, then -(n)         bj_MAX_PLAYERS
 *   MBOX - 16, then -(2 k + 1)   the game's players are 2k and 2k + 1: reported as players 0 and 1
 *   MBOX - 17, then -(n)         of n players in such pairs (the others are reported as 100 + their id)
 *   MBOX - 9 / MBOX - 10         a pass begins: the records are deltas / a full snapshot
 *   -(unit handle id)            the unit, once per pass: gone or reported dead (dropped from the
 *                                group), counted, its record kept if it changed
 *   MBOX - 15                    the pass is over
 *   MBOX - 100 - (3 p + k)       player p's count k: living town halls, living units that are not
 *                                structures, living units
 *   MBOX - 11                    the pass's unit records go into the observation
 *   MBOX - 12                    and the "R" records of the units dropped
 *
 * W3SIM_UNITS=0: not installed (the harness does it all). W3SIM_UNITS=2: the harness does it all
 * and also sends each unit here before it writes its record (MBOX - 3 / MBOX - 4: deltas / full);
 * the record made here is compared with the harness's tokens, and the mismatches are logged.
 *
 * The natives are found by name: the game registers each with `push signature; push name;
 * push function; call` (the first registration of a name is the real function).
 */
#include "w3shim.h"

#include "MinHook.h"

#include <stdlib.h>
#include <string.h>

typedef int(__cdecl *N1)(int);
typedef int(__cdecl *N2)(int, int);
typedef int(__cdecl *N3)(int, int, int);
typedef int(__cdecl *N4)(int, int, int, int);

enum {
    GetUnitTypeId, GetOwningPlayer, GetPlayerId, GetUnitX, GetUnitY, GetUnitFacing, GetWidgetLife, GetUnitState,
    GetUnitCurrentOrder, IsUnitVisible, GetResourceAmount, IsUnitType, IsUnitHidden, IsUnitLoaded, UnitIsSleeping,
    IsUnitPaused, IsUnitIllusion, GetHeroXP, GetHeroSkillPoints, GetHeroLevel, UnitItemInSlot, GetItemTypeId,
    GetUnitAbilityLevel, BlzGetUnitAbilityCooldownRemaining, LoadBoolean, LoadInteger, SaveInteger, SaveBoolean,
    HaveSavedInteger, Player, GroupRemoveUnit, FlushChildHashtable, ForGroup,
    GetPlayerSlotState, IsPlayerObserver, GetPlayerRace, GetPlayerState, GetPlayerStructureCount, GetPlayerStartLocation,
    GetStartLocationX, GetStartLocationY, GetPlayerAlliance, RemoveSavedBoolean, FlushParentHashtable, N_NATIVES
};
static const struct {
    const char *name, *sig;
} g_want[N_NATIVES] = {
    {"GetUnitTypeId", "(Hunit;)I"}, {"GetOwningPlayer", "(Hunit;)Hplayer;"}, {"GetPlayerId", "(Hplayer;)I"},
    {"GetUnitX", "(Hunit;)R"}, {"GetUnitY", "(Hunit;)R"}, {"GetUnitFacing", "(Hunit;)R"},
    {"GetWidgetLife", "(Hwidget;)R"}, {"GetUnitState", "(Hunit;Hunitstate;)R"}, {"GetUnitCurrentOrder", "(Hunit;)I"},
    {"IsUnitVisible", "(Hunit;Hplayer;)B"}, {"GetResourceAmount", "(Hunit;)I"}, {"IsUnitType", "(Hunit;Hunittype;)B"},
    {"IsUnitHidden", "(Hunit;)B"}, {"IsUnitLoaded", "(Hunit;)B"}, {"UnitIsSleeping", "(Hunit;)B"},
    {"IsUnitPaused", "(Hunit;)B"}, {"IsUnitIllusion", "(Hunit;)B"}, {"GetHeroXP", "(Hunit;)I"},
    {"GetHeroSkillPoints", "(Hunit;)I"}, {"GetHeroLevel", "(Hunit;)I"}, {"UnitItemInSlot", "(Hunit;I)Hitem;"},
    {"GetItemTypeId", "(Hitem;)I"}, {"GetUnitAbilityLevel", "(Hunit;I)I"},
    {"BlzGetUnitAbilityCooldownRemaining", "(Hunit;I)R"}, {"LoadBoolean", "(Hhashtable;II)B;"},
    {"LoadInteger", "(Hhashtable;II)I;"}, {"SaveInteger", "(Hhashtable;III)V"}, {"SaveBoolean", "(Hhashtable;IIB)V"},
    {"HaveSavedInteger", "(Hhashtable;II)B"}, {"Player", "(I)Hplayer;"}, {"GroupRemoveUnit", "(Hgroup;Hunit;)V"},
    {"FlushChildHashtable", "(Hhashtable;I)V"}, {"ForGroup", "(Hgroup;C)V"},
    {"GetPlayerSlotState", "(Hplayer;)Hplayerslotstate;"}, {"IsPlayerObserver", "(Hplayer;)B"},
    {"GetPlayerRace", "(Hplayer;)Hrace;"}, {"GetPlayerState", "(Hplayer;Hplayerstate;)I"},
    {"GetPlayerStructureCount", "(Hplayer;B)I"}, {"GetPlayerStartLocation", "(Hplayer;)I"},
    {"GetStartLocationX", "(I)R"}, {"GetStartLocationY", "(I)R"},
    {"GetPlayerAlliance", "(Hplayer;Hplayer;Halliancetype;)B"},
    {"RemoveSavedBoolean", "(Hhashtable;II)V"}, {"FlushParentHashtable", "(Hhashtable;)V"},
};
static void *g_fn[N_NATIVES];
static int g_ok; /* every native found: 1, or 2 to check the records against the harness's (W3SIM_UNITS=2) */

/* common.j: ConvertUnitState / ConvertUnitType values (these "handles" are the integers themselves) */
enum { STATE_MAX_LIFE = 1, STATE_MANA = 2, STATE_MAX_MANA = 3 };
enum { TYPE_HERO = 0, TYPE_DEAD = 1, TYPE_STRUCTURE = 2, TYPE_FLYING = 3, TYPE_SUMMONED = 10, TYPE_PEON = 16, TYPE_TOWNHALL = 18 };
#define ABIL_SLOTS 4 /* W3S_ABIL_SLOTS */
#define MAX_PLAYERS 28

static int g_ht, g_abil, g_all; /* the harness's hashtables and its group of all units (handle ids) */
static int g_max_players = 12;  /* bj_MAX_PLAYERS */
static int g_pair_first, g_paired; /* the current game's first player, of this many in pairs (0: no pairs) */
static int g_full;
static int g_pending; /* what the next negative argument is: 0 a unit, else the register (5, 6, 8, 13, 14, 16, 17) */
static int g_nseer, g_seer_player[MAX_PLAYERS], g_seer_bit[MAX_PLAYERS];
static int g_pass;                                                            /* in a pass */
static int g_halls[MAX_PLAYERS], g_mobile[MAX_PLAYERS], g_alive[MAX_PLAYERS]; /* the pass's counts */
static int *g_removed, g_nremoved, g_removed_cap;                             /* and the units it dropped */
static char *g_rec; /* the pass's unit records (tokens, a line each) */
static int g_rec_len, g_rec_cap;

static int n1(int k, int a) {
    return ((N1)g_fn[k])(a);
}
static int n2(int k, int a, int b) {
    return ((N2)g_fn[k])(a, b);
}
static int n3(int k, int a, int b, int c) {
    return ((N3)g_fn[k])(a, b, c);
}
/* a real: the native returns the float's bits */
static float r1(int k, int a) {
    int bits = n1(k, a);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}
static float r2(int k, int a, int b) {
    int bits = n2(k, a, b);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

/* W3SIM_CUTS & 32: the pure getters pass_unit and unit_record both call (GetUnitTypeId, IsUnitType(DEAD),
 * GetOwningPlayer + GetPlayerId, IsUnitType(HERO), GetWidgetLife) called once per unit; GetPlayerId by the
 * player's handle (a player's id and handle are fixed for a load of the map: forgotten at a game's first step).
 * The calls that make handles (UnitItemInSlot) are made as before. */
static int g_dedupe;
static int g_cobs_check_env; /* W3SIM_COBS_CHECK=1: the players' records and result checked against the harness's */
static struct {
    int u, typ, dead, owner, hero, have_hero;
} g_cur;
#define PID_CACHE 32
static int g_pid_h[PID_CACHE], g_pid_v[PID_CACHE], g_npid;

static int player_id(int h) {
    for (int i = 0; i < g_npid; i++)
        if (g_pid_h[i] == h)
            return g_pid_v[i];
    int v = ((N1)g_fn[GetPlayerId])(h);
    if (h && g_npid < PID_CACHE) {
        g_pid_h[g_npid] = h;
        g_pid_v[g_npid++] = v;
    }
    return v;
}

static int owner_id(int u) {
    if (!g_dedupe)
        return ((N1)g_fn[GetPlayerId])(((N1)g_fn[GetOwningPlayer])(u));
    return player_id(((N1)g_fn[GetOwningPlayer])(u));
}

/* a player's id as the controller sees it (the harness's W3S_Pid) */
static int pid(int i) {
    if (!g_paired || i >= g_paired)
        return i;
    if (i >= g_pair_first && i < g_pair_first + 2)
        return i - g_pair_first;
    return i + 100;
}

/* W3SIM_UNITS=2: the record made here is kept and compared with the tokens the harness then sends,
 * and its hash with the one the harness stored. */
#define V_MAX 40
static int g_v_unit, g_v_n, g_v_got, g_v_bad, g_v_open;
static int g_v_want[V_MAX];
static uint32_t g_v_hash;
static long g_v_units, g_v_records, g_v_mismatches;

static void verify_close(void) {
    if (!g_v_unit)
        return;
    g_v_units++;
    int saved = n3(HaveSavedInteger, g_ht, g_v_unit, 3) ? n3(LoadInteger, g_ht, g_v_unit, 3) : 0;
    int bad = g_v_bad || (uint32_t)saved != g_v_hash || (g_v_open && g_v_got != g_v_n);
    if (g_v_open)
        g_v_records++;
    if (bad && g_v_mismatches++ < 40)
        shim_log("unit records: MISMATCH unit %d: hash %u, the harness's %u; %d of %d tokens, first difference at %d",
                 g_v_unit, g_v_hash, (uint32_t)saved, g_v_got, g_v_n, g_v_bad - 1);
    if (g_v_units % 20000 == 0)
        shim_log("unit records: %ld units checked, %ld records, %ld mismatches", g_v_units, g_v_records, g_v_mismatches);
    g_v_unit = 0;
}

/* a token the harness sent (obs.c); only between a unit's call here and the next */
void units_verify_token(const char *s) {
    if (!g_v_unit || g_v_open == 2)
        return;
    if (!g_v_open) {
        if (s[0] == 'U' && !s[1])
            g_v_open = 1;
        return;
    }
    if ((s[0] < '0' || s[0] > '9') && s[0] != '-') { /* the next record's tag: this one is complete */
        g_v_open = 2;
        return;
    }
    if (g_v_got < g_v_n && atoi(s) != g_v_want[g_v_got] && !g_v_bad)
        g_v_bad = g_v_got + 1;
    g_v_got++;
}

/* one record: tag, integer fields, their sum (32-bit wrap-around), as the harness's W3S_Rec / W3S_Tok / W3S_End */
static uint32_t g_sum;

static void rec_bytes(const char *s, int n) {
    if (g_rec_len + n + 1 > g_rec_cap) {
        int cap = g_rec_cap ? g_rec_cap * 2 : 65536;
        while (g_rec_len + n + 1 > cap)
            cap *= 2;
        char *b = realloc(g_rec, cap);
        if (!b)
            return;
        g_rec = b;
        g_rec_cap = cap;
    }
    memcpy(g_rec + g_rec_len, s, n);
    g_rec_len += n;
    g_rec[g_rec_len++] = '\n';
}

static void tok(int v) {
    g_sum += (uint32_t)v;
    if (g_ok == 2) {
        if (g_v_n < V_MAX)
            g_v_want[g_v_n++] = v;
        return;
    }
    char buf[16], tmp[12];
    int n = 0, len = 0, neg = v < 0;
    uint32_t u = neg ? 0u - (uint32_t)v : (uint32_t)v;
    do {
        tmp[n++] = (char)('0' + u % 10);
        u /= 10;
    } while (u);
    if (neg)
        buf[len++] = '-';
    while (n)
        buf[len++] = tmp[--n];
    rec_bytes(buf, len);
}

/* W3SIM_CUTS & 8388608: the harness table's per-unit values the pass reads kept here, by handle id: key 3 (the last
 * record's hash) and key 4 (reported dead), which only this pass reads and writes, and key 1 (under construction),
 * which the harness writes on its construction events: its SaveBoolean / RemoveSavedBoolean / FlushChildHashtable /
 * FlushParentHashtable calls on the table are mirrored here (hooks on the natives). Each LoadBoolean / HaveSavedInteger /
 * LoadInteger / SaveInteger was a hashtable lookup per unit per step. */
#define KT_BASE 0x100000
static int g_kt;
static BYTE *g_k1, *g_k3h, *g_k4;
static uint32_t *g_k3;
static unsigned g_kcap;

static int kt_slot(int h) {
    unsigned i = (unsigned)h - KT_BASE;
    if (i < g_kcap)
        return (int)i;
    if (i >= (1u << 22))
        return -1;
    unsigned cap = g_kcap ? g_kcap : 65536;
    while (cap <= i)
        cap *= 2;
    BYTE *a = realloc(g_k1, cap), *b = realloc(g_k3h, cap), *d = realloc(g_k4, cap);
    uint32_t *c = realloc(g_k3, cap * sizeof *c);
    if (a)
        g_k1 = a;
    if (b)
        g_k3h = b;
    if (d)
        g_k4 = d;
    if (c)
        g_k3 = c;
    if (!a || !b || !c || !d)
        return -1;
    memset(g_k1 + g_kcap, 0, cap - g_kcap);
    memset(g_k3h + g_kcap, 0, cap - g_kcap);
    memset(g_k4 + g_kcap, 0, cap - g_kcap);
    memset(g_k3 + g_kcap, 0, (cap - g_kcap) * sizeof *g_k3);
    g_kcap = cap;
    return (int)i;
}

static void kt_clear(void) {
    if (!g_kcap)
        return;
    memset(g_k1, 0, g_kcap);
    memset(g_k3h, 0, g_kcap);
    memset(g_k4, 0, g_kcap);
}

typedef void(__cdecl *SaveBoolFn)(int, int, int, int);
typedef void(__cdecl *RemBoolFn)(int, int, int);
typedef void(__cdecl *FlushChildFn)(int, int);
typedef void(__cdecl *FlushParentFn)(int);
static SaveBoolFn SaveBool_orig;
static RemBoolFn RemBool_orig;
static FlushChildFn FlushChild_orig;
static FlushParentFn FlushParent_orig;

static void __cdecl SaveBool_hook(int ht, int parent, int child, int v) {
    if (g_kt && ht == g_ht && child == 1) {
        int i = kt_slot(parent);
        if (i >= 0)
            g_k1[i] = v != 0;
    }
    SaveBool_orig(ht, parent, child, v);
}

static void __cdecl RemBool_hook(int ht, int parent, int child) {
    if (g_kt && ht == g_ht && child == 1) {
        int i = kt_slot(parent);
        if (i >= 0)
            g_k1[i] = 0;
    }
    RemBool_orig(ht, parent, child);
}

static void __cdecl FlushChild_hook(int ht, int parent) {
    if (g_kt && ht == g_ht) {
        unsigned i = (unsigned)parent - KT_BASE;
        if (i < g_kcap)
            g_k1[i] = g_k3h[i] = g_k4[i] = 0;
    }
    FlushChild_orig(ht, parent);
}

static void __cdecl FlushParent_hook(int ht) {
    if (g_kt && ht == g_ht)
        kt_clear();
    FlushParent_orig(ht);
}

static void install_kt(void) {
    static const int which[4] = {SaveBoolean, RemoveSavedBoolean, FlushChildHashtable, FlushParentHashtable};
    void *hooks[4] = {(void *)SaveBool_hook, (void *)RemBool_hook, (void *)FlushChild_hook, (void *)FlushParent_hook};
    void **origs[4] = {(void **)&SaveBool_orig, (void **)&RemBool_orig, (void **)&FlushChild_orig, (void **)&FlushParent_orig};
    for (int k = 0; k < 4; k++) {
        BYTE *f = g_fn[which[k]];
        if (!(f[0] == 0x55 && f[1] == 0x8b && f[2] == 0xec)) {
            shim_log("unit records: table natives: unexpected code, not installed");
            return;
        }
    }
    MH_STATUS st = MH_Initialize();
    if (st != MH_OK && st != MH_ERROR_ALREADY_INITIALIZED)
        return;
    for (int k = 0; k < 4; k++)
        if (MH_CreateHook(g_fn[which[k]], hooks[k], origs[k]) != MH_OK || MH_EnableHook(g_fn[which[k]]) != MH_OK) {
            shim_log("unit records: table natives: hook failed");
            return;
        }
    g_kt = 1;
    shim_log("unit records: the table's per-unit values kept by the shim (W3SIM_CUTS & 8388608)");
}

/* W3S_SerUnit: the unit's record, if it changed since it was last written (or on a full snapshot) */
static void unit_record(int u) {
    /* (the order of the calls is W3S_SerUnit's: some of them make handles) */
    int flags = 0;
    int same = g_dedupe && g_cur.u == u; /* (pass_unit's values for this unit) */
    int hero = same && g_cur.have_hero ? g_cur.hero : n2(IsUnitType, u, TYPE_HERO) != 0;
    if (hero)
        flags += 1;
    if (n2(IsUnitType, u, TYPE_STRUCTURE))
        flags += 2;
    if (n2(IsUnitType, u, TYPE_PEON))
        flags += 4;
    int ki = g_kt && !g_paired ? kt_slot(u) : -1;
    if (ki >= 0 ? g_k1[ki] : n3(LoadBoolean, g_ht, u, 1))
        flags += 8; /* under construction */
    if (n1(IsUnitHidden, u))
        flags += 16;
    if (n1(IsUnitLoaded, u))
        flags += 32;
    if (n1(UnitIsSleeping, u))
        flags += 64;
    if (n1(IsUnitPaused, u))
        flags += 128;
    if (n2(IsUnitType, u, TYPE_SUMMONED))
        flags += 256;
    if (n1(IsUnitIllusion, u))
        flags += 512;
    int dead = same ? g_cur.dead : n2(IsUnitType, u, TYPE_DEAD) != 0;
    float life = 0;
    if (g_dedupe) {
        life = r1(GetWidgetLife, u);
        if (dead || life < 0.405f)
            flags += 1024;
    } else if (dead || r1(GetWidgetLife, u) < 0.405f)
        flags += 1024;
    if (n2(IsUnitType, u, TYPE_FLYING))
        flags += 2048;
#ifdef W3SIM_REPLAY
    flags |= replay_unit_flags(u, flags); /* (the replay build: buildings' state; the unit's object ids reported) */
#endif
    int typ = same ? g_cur.typ : n1(GetUnitTypeId, u);
    int owner = pid(same ? g_cur.owner : owner_id(u));
    int x = (int)r1(GetUnitX, u), y = (int)r1(GetUnitY, u), facing = (int)r1(GetUnitFacing, u);
    int hp = (int)((g_dedupe ? life : r1(GetWidgetLife, u)) + 0.5f);
    int maxhp = (int)(r2(GetUnitState, u, STATE_MAX_LIFE) + 0.5f);
    int mana = (int)r2(GetUnitState, u, STATE_MANA), maxmana = (int)r2(GetUnitState, u, STATE_MAX_MANA);
    int order = n1(GetUnitCurrentOrder, u);
    int vis = 0;
    for (int k = 0; k < g_nseer; k++)
        if (n2(IsUnitVisible, u, g_seer_player[k]))
            vis += g_seer_bit[k];
    int res = n1(GetResourceAmount, u);
    uint32_t h = (uint32_t)typ;
    h = h * 31 + (uint32_t)owner;
    h = h * 31 + (uint32_t)x;
    h = h * 31 + (uint32_t)y;
    h = h * 31 + (uint32_t)facing;
    h = h * 31 + (uint32_t)hp;
    h = h * 31 + (uint32_t)maxhp;
    h = h * 31 + (uint32_t)mana;
    h = h * 31 + (uint32_t)maxmana;
    h = h * 31 + (uint32_t)order;
    h = h * 31 + (uint32_t)flags;
    h = h * 31 + (uint32_t)vis;
    h = h * 31 + (uint32_t)res;
    int alevel[ABIL_SLOTS], acool[ABIL_SLOTS];
    if (hero) {
        h = (h * 31 + (uint32_t)n1(GetHeroXP, u)) * 31 + (uint32_t)n1(GetHeroSkillPoints, u);
        for (int i = 0; i < 6; i++)
            h = h * 31 + (uint32_t)n1(GetItemTypeId, n2(UnitItemInSlot, u, i));
        for (int i = 0; i < ABIL_SLOTS; i++) { /* each ability slot: learned level and cooldown left (0.1 s) */
            int a = n3(LoadInteger, g_abil, typ, i);
            alevel[i] = acool[i] = 0;
            if (a) {
                alevel[i] = n2(GetUnitAbilityLevel, u, a);
                if (alevel[i] > 0)
                    acool[i] = (int)(r2(BlzGetUnitAbilityCooldownRemaining, u, a) * 10.0f + 0.5f);
            }
            h = (h * 31 + (uint32_t)alevel[i]) * 31 + (uint32_t)acool[i];
        }
    }
    if (g_ok == 2) { /* the harness decides and stores; this record is only compared */
        g_v_unit = u;
        g_v_hash = h;
        g_v_n = g_v_got = g_v_bad = g_v_open = 0;
    } else {
        if (ki >= 0) {
            if (!g_full && g_k3h[ki] && g_k3[ki] == h)
                return;
            g_k3[ki] = h;
            g_k3h[ki] = 1;
            if (dead)
                g_k4[ki] = 1;
        } else {
            if (!g_full && n3(HaveSavedInteger, g_ht, u, 3) && (uint32_t)n3(LoadInteger, g_ht, u, 3) == h)
                return;
            ((N4)g_fn[SaveInteger])(g_ht, u, 3, (int)h);
            if (dead)
                ((N4)g_fn[SaveBoolean])(g_ht, u, 4, 1);
        }
        rec_bytes("U", 1);
    }
    g_sum = 0;
    tok(u);
    tok(typ);
    tok(owner);
    tok(x);
    tok(y);
    tok(facing);
    tok(hp);
    tok(maxhp);
    tok(mana);
    tok(maxmana);
    tok(order);
    tok(flags);
    tok(vis);
    tok(res);
    if (hero) {
        tok(n1(GetHeroLevel, u));
        tok(n1(GetHeroXP, u));
        tok(n1(GetHeroSkillPoints, u));
        for (int i = 0; i < 6; i++)
            tok(n1(GetItemTypeId, n2(UnitItemInSlot, u, i)));
        for (int i = 0; i < ABIL_SLOTS; i++) {
            tok(alevel[i]);
            tok(acool[i]);
        }
    }
    uint32_t sum = g_sum;
    tok((int)sum);
}

static void drop(int u) { /* out of the group of all units, its table entries with it; an "R" record */
    cuts_unit_begin(0);
    n2(GroupRemoveUnit, g_all, u);
    n2(FlushChildHashtable, g_ht, u);
    if (g_nremoved == g_removed_cap) {
        int cap = g_removed_cap ? g_removed_cap * 2 : 1024;
        int *b = realloc(g_removed, cap * sizeof *b);
        if (!b)
            return;
        g_removed = b;
        g_removed_cap = cap;
    }
    g_removed[g_nremoved++] = u;
}

/* a unit of the pass: W3S_EnumAdd (with pruning), W3S_CountDecisive / W3S_CountAlive, W3S_SerUnit */
static void pass_unit(int u) {
    g_cur.u = 0;
    cuts_unit_begin(1);
    int typ = n1(GetUnitTypeId, u);
    if (typ == 0) { /* removed from the game */
        drop(u);
        return;
    }
    int dead = n2(IsUnitType, u, TYPE_DEAD) != 0;
    int p = owner_id(u);
    if (g_dedupe) {
        g_cur.u = u;
        g_cur.typ = typ;
        g_cur.dead = dead;
        g_cur.owner = p;
        g_cur.have_hero = 0;
    }
    if (!dead && p >= 0 && p < g_max_players && p < MAX_PLAYERS) {
        g_alive[p]++;
        if (n2(IsUnitType, u, TYPE_TOWNHALL))
            g_halls[p]++;
        else if (!n2(IsUnitType, u, TYPE_STRUCTURE))
            g_mobile[p]++;
    }
    /* reported as dead already: dropped (heroes stay, they can be revived). As in the harness, its
     * record is still written this once (its stored hash went with its table entries). */
    if (dead) {
        int hero = n2(IsUnitType, u, TYPE_HERO) != 0;
        if (g_dedupe) {
            g_cur.hero = hero;
            g_cur.have_hero = 1;
        }
        int ki = g_kt && !g_paired ? kt_slot(u) : -1;
        if (!hero && (ki >= 0 ? g_k4[ki] : n3(LoadBoolean, g_ht, u, 4)))
            drop(u);
    }
    unit_record(u);
    cuts_unit_begin(0);
    g_cur.u = 0;
}

/* W3SIM_CUTS & 16: the pass's ForGroup(w3s_all, W3S_CPassEnum) without a JASS thread per unit. The native
 * runs as before (the group's snapshot, the code's object, the checks of each unit, the enum unit's handle
 * and its reference count: 0x5025c0, 0x502360); only its per-unit callback 0x5026a0 hands the enum unit's
 * handle to pass_unit here instead of running the JASS function (whose only effect was that call:
 * GetEnumUnit, GetHandleId and the mailbox native). */
#define VA(x) ((void *)(g_base + ((x) - 0x400000)))
#define VA_FORGROUP 0x4906f0    /* cdecl ForGroup(group, code) native */
#define VA_ENUM_CB 0x5026a0     /* cdecl int cb(CUnit *, ctx {jass state, group, code object}) */
#define VA_AGENT 0x4707f0       /* cdecl agent lookup(id1, id2) */
#define VA_ENUM_PUSH 0x4ebf00   /* thiscall(state, slot, agent, 0): the enum unit (its handle referenced) */
#define VA_ENUM_TOP 0x4ebc20    /* thiscall(state, slot): its handle (GetEnumUnit) */
#define VA_ENUM_POP 0x4ebce0    /* thiscall(state, slot, 1): popped, the reference released */
typedef void(__cdecl *ForGroupFn)(int, int);
typedef int(__cdecl *EnumCbFn)(BYTE *, DWORD *);
typedef BYTE *(__cdecl *AgentFn)(DWORD, DWORD);
typedef void(__fastcall *EnumPushFn)(void *, void *, int, BYTE *, int);
typedef int(__fastcall *EnumTopFn)(void *, void *, int);
typedef int(__fastcall *EnumPopFn)(void *, void *, int, int);
static ForGroupFn ForGroup_orig;
static EnumCbFn EnumCb_orig;
static int g_fg_pass;
static long g_fg_units;

static void __cdecl ForGroup_hook(int group, int code) {
    if (g_ok == 1 && g_pass && g_all && group == g_all && g_ht && g_abil && !g_pending && !g_fg_pass) {
        g_fg_pass = 1;
        ForGroup_orig(group, code);
        g_fg_pass = 0;
        return;
    }
    ForGroup_orig(group, code);
}

static int __cdecl EnumCb_hook(BYTE *unit, DWORD *ctx) {
    if (!g_fg_pass)
        return EnumCb_orig(unit, ctx);
    if (unit) { /* (0x5026a0's checks) */
        BYTE *a = ((AgentFn)VA(VA_AGENT))(*(DWORD *)(unit + 0xc), *(DWORD *)(unit + 0x10));
        if (a && *(DWORD *)(a + 0xc) == 0x2b61676c && *(DWORD *)(a + 0x20) == 0) {
            void *st = (void *)(size_t)ctx[0];
            ((EnumPushFn)VA(VA_ENUM_PUSH))(st, NULL, 10, unit, 0);
            if (ctx[2]) {
                int h = ((EnumTopFn)VA(VA_ENUM_TOP))(st, NULL, 10);
                if (h) {
                    g_fg_units++;
                    pass_unit(h);
                }
            }
            ((EnumPopFn)VA(VA_ENUM_POP))(st, NULL, 10, 1);
        }
    }
    return 1;
}

static void install_fg_pass(void) {
    BYTE *f = VA(VA_FORGROUP), *cb = VA(VA_ENUM_CB);
    if (!(f[0] == 0x55 && f[1] == 0x8b && f[2] == 0xec && cb[0] == 0x55 && cb[1] == 0x8b && cb[2] == 0xec) ||
        (BYTE *)g_fn[ForGroup] != f) {
        shim_log("unit records: ForGroup pass: unexpected code, not installed");
        return;
    }
    MH_STATUS st = MH_Initialize();
    if (st != MH_OK && st != MH_ERROR_ALREADY_INITIALIZED)
        return;
    if (MH_CreateHook(f, (void *)ForGroup_hook, (void **)&ForGroup_orig) != MH_OK || MH_EnableHook(f) != MH_OK ||
        MH_CreateHook(cb, (void *)EnumCb_hook, (void **)&EnumCb_orig) != MH_OK || MH_EnableHook(cb) != MH_OK) {
        shim_log("unit records: ForGroup pass: hook failed");
        return;
    }
    shim_log("unit records: the pass's ForGroup without JASS (W3SIM_CUTS & 16)");
}

/* W3SIM_CUTS & 262144: the harness's players' records (W3S_SerPlayers) and its check of the game's result
 * (W3S_CheckResult, victory 0 melee / 3 decisive, without pairs, scenarios or resets by script) made here with the same
 * natives: in JASS they cost ~0.5 of a step's ms (a JASS call, I2S and Preload per token, and the result's loops over
 * 24 x 24 players). The harness asks MBOX - 18 (1: on, 2: check: the harness also does both itself and the
 * records and results are compared here, mismatches logged), then MBOX - 28, -(victory + 1) to use it;
 *   MBOX - 19 / MBOX - 20, then -(player + 1)   an agent's player / not any more (W3S_StartAI)
 *   MBOX - 21                                   the result's check (returns 1: the game is over)
 *   MBOX - 22                                   the time limit reached: the players still in it tie (3)
 *   MBOX - 24                                   the players' records go into the observation (check: kept)
 *   MBOX - 25                                   (check) the harness's records just written, compared
 *   MBOX - 26 / MBOX - 27, then -(value + 1)    (check) the harness's result of the next player / its game over */
static int g_cobs, g_cobs_victory, g_cobs_over;
static int g_cobs_agent[MAX_PLAYERS], g_cobs_result[MAX_PLAYERS];
static int g_jres[MAX_PLAYERS], g_jres_n;
static char g_pexp[8192];
static int g_pexp_len, g_pobs_mark;
static long g_cobs_checks, g_cobs_bad;

static void pbuf_tok(char *b, int *len, int cap, int v, uint32_t *sum) {
    if (sum)
        *sum += (uint32_t)v;
    char tmp[12];
    int n = 0, neg = v < 0;
    uint32_t u = neg ? 0u - (uint32_t)v : (uint32_t)v;
    do {
        tmp[n++] = (char)('0' + u % 10);
        u /= 10;
    } while (u);
    if (*len + n + 2 > cap)
        return;
    if (neg)
        b[(*len)++] = '-';
    while (n)
        b[(*len)++] = tmp[--n];
    b[(*len)++] = '\n';
}

static int playing(int i, int *ph) { /* slot playing, not an observer (W3S_Active: no pairs here) */
    int p = n1(Player, i);
    if (ph)
        *ph = p;
    return n1(GetPlayerSlotState, p) == 1 && !n1(IsPlayerObserver, p);
}

static int coallied(int a, int b) { /* Blizzard.j PlayersAreCoAllied */
    return a == b || (n3(GetPlayerAlliance, a, b, 0) && n3(GetPlayerAlliance, b, a, 0));
}

static int race_id(int r) { /* W3S_RaceId: RACE_HUMAN .. RACE_NIGHTELF are ConvertRace(1 .. 4) */
    return r >= 1 && r <= 4 ? r : 0;
}

static int cobs_records(char *b, int cap) { /* W3S_SerPlayers */
    int len = 0;
    for (int i = 0; i < g_max_players && i < MAX_PLAYERS; i++) {
        int p = n1(Player, i);
        if (n1(GetPlayerSlotState, p) == 0 || n1(IsPlayerObserver, p))
            continue;
        uint32_t sum = 0;
        if (len + 2 < cap) {
            b[len++] = 'P';
            b[len++] = '\n';
        }
        pbuf_tok(b, &len, cap, pid(i), &sum);
        pbuf_tok(b, &len, cap, race_id(n1(GetPlayerRace, p)), &sum);
        pbuf_tok(b, &len, cap, g_cobs_agent[i] ? 1 : 0, &sum);
        pbuf_tok(b, &len, cap, n2(GetPlayerState, p, 1), &sum);  /* gold */
        pbuf_tok(b, &len, cap, n2(GetPlayerState, p, 2), &sum);  /* lumber */
        pbuf_tok(b, &len, cap, n2(GetPlayerState, p, 5), &sum);  /* food used */
        pbuf_tok(b, &len, cap, n2(GetPlayerState, p, 4), &sum);  /* food cap */
        pbuf_tok(b, &len, cap, n2(GetPlayerState, p, 13), &sum); /* gold upkeep rate */
        pbuf_tok(b, &len, cap, n2(GetPlayerState, p, 15), &sum); /* gold gathered */
        pbuf_tok(b, &len, cap, n2(GetPlayerState, p, 16), &sum); /* lumber gathered */
        pbuf_tok(b, &len, cap, n2(GetPlayerStructureCount, p, 1), &sum);
        pbuf_tok(b, &len, cap, g_cobs_result[i], &sum);
        int sl = n1(GetPlayerStartLocation, p);
        pbuf_tok(b, &len, cap, (int)r1(GetStartLocationX, sl), &sum);
        pbuf_tok(b, &len, cap, (int)r1(GetStartLocationY, sl), &sum);
        pbuf_tok(b, &len, cap, (int)sum, NULL);
    }
    return len;
}

static int cobs_check(void) { /* W3S_CheckResult (its time limit stays in JASS: MBOX - 22) */
    if (g_cobs_over)
        return 1;
    int maxp = g_max_players < MAX_PLAYERS ? g_max_players : MAX_PLAYERS, ph[MAX_PLAYERS], in[MAX_PLAYERS];
    for (int i = 0; i < maxp; i++)
        in[i] = playing(i, &ph[i]);
    for (int i = 0; i < maxp; i++) {
        if (g_cobs_result[i] != 0 || !in[i])
            continue;
        int ally = 0; /* MeleeGetAllyStructureCount */
        for (int j = 0; j < maxp; j++)
            if (coallied(ph[i], ph[j]))
                ally += n2(GetPlayerStructureCount, ph[j], 1);
        if (g_cobs_victory == 0 && ally <= 0)
            g_cobs_result[i] = 2;
        else if (g_cobs_victory == 3 && (ally <= 0 || (g_halls[i] <= 0 && g_mobile[i] <= 0)))
            g_cobs_result[i] = 2;
    }
    int alive = 0, allied = 1;
    for (int i = 0; i < maxp; i++) {
        if (g_cobs_result[i] != 0 || !in[i])
            continue;
        alive++;
        for (int j = 0; j < maxp; j++)
            if (j != i && g_cobs_result[j] == 0 && in[j] && !coallied(ph[i], ph[j]))
                allied = 0;
    }
    if (alive == 0 || allied) {
        g_cobs_over = 1;
        for (int i = 0; i < maxp; i++)
            if (g_cobs_result[i] == 0 && in[i])
                g_cobs_result[i] = 1;
    }
    return g_cobs_over;
}

static void cobs_tie(void) {
    g_cobs_over = 1;
    for (int i = 0; i < g_max_players && i < MAX_PLAYERS; i++)
        if (g_cobs_result[i] == 0 && playing(i, NULL))
            g_cobs_result[i] = 3;
}

static void cobs_mismatch(const char *what, int a, int b) {
    if (g_cobs_bad++ < 40)
        shim_log("players/result in C: MISMATCH %s: the shim's %d, the harness's %d", what, a, b);
}

/* W3SIM_CUTS & 2147483648 (round 4): the agents' commands applied here (the harness's W3S_ReadCommands asks MBOX - 35
 * after its W3S_BuyPending, when no purchase is pending and no agent's player is advised): W3S_ApplyOne's natives
 * that act or decide, in its order, for the ops 1 (point order), 2 (target order), 3 (immediate order), 5 (hero
 * skill), 6 (an order on a destructable), 87 (tax) and 90 (resources). A step with any other op (a build: its ring
 * of tries uses the VM's real arithmetic), or with a command cut short by the step's end, is the harness's, as
 * before; it then reads as many of the integers as W3S_ApplyOne reads (the mailbox returns, past this step's count,
 * what the harness's array would hold: the last value at that place since the map was loaded, MBOX - 37). The
 * getters W3S_ApplyOne calls only to look (W3S_Unit for the tax ops, the second owner lookup of the "advised"
 * test, I2R) are pure; LoadUnitHandle, GetOwningPlayer, GetPlayerId, GetHeroSkillPoints and GetPlayerState are
 * called as there, and every command's result is kept for the "C" records (MBOX - 36), which the harness wrote.
 *   MBOX - 32                     on? (1); forgets the agents' and scripted players
 *   MBOX - 33 / 34, -(player + 1) a player whose units take commands (W3S_Owned) / not any more (W3S_StartAI)
 *   MBOX - 35                     apply the step's commands: -1 done, else the number of integers to read
 *   MBOX - 36                     the "C" records of the last commands applied here
 *   MBOX - 37                     a load's first step: the harness's command array is all zeros
 * W3SIM_CUTS & 1073741824: MBOX - 29 (on? 1), then MBOX - 30, -(handle id): the player of W3S_Tok's calls (obs.c), and
 * MBOX - 31, -(W3S_VERSION + 1); MBOX - 39, then -(seq + 1), -(ms + 1), -(1 + over + 2 full): the "V", "K" and "T"
 * records written here (the camera's target from GetCameraTargetPositionX / Y and R2I, as the harness's). */
enum {
    LoadUnitHandle_, LoadDestructableHandle_, IssuePointOrderById_, IssueTargetOrderById_, IssueImmediateOrderById_,
    SelectHeroSkill_, SetPlayerState_, I2R_, N_CMD_NATIVES
};
static const struct {
    const char *name, *sig;
} g_want_cmd[N_CMD_NATIVES] = {
    {"LoadUnitHandle", "(Hhashtable;II)Hunit;"}, {"LoadDestructableHandle", "(Hhashtable;II)Hdestructable;"},
    {"IssuePointOrderById", "(Hunit;IRR)B"}, {"IssueTargetOrderById", "(Hunit;IHwidget;)B"},
    {"IssueImmediateOrderById", "(Hunit;I)B"}, {"SelectHeroSkill", "(Hunit;I)V"},
    {"SetPlayerState", "(Hplayer;Hplayerstate;I)V"}, {"I2R", "(I)R"},
};
static void *g_cfn[N_CMD_NATIVES];
enum { GetCameraTargetPositionX_, GetCameraTargetPositionY_, R2I_, N_TOK_NATIVES };
static const struct {
    const char *name, *sig;
} g_want_tok[N_TOK_NATIVES] = {
    {"GetCameraTargetPositionX", "()R"}, {"GetCameraTargetPositionY", "()R"}, {"R2I", "(R)I"},
};
static void *g_tfn[N_TOK_NATIVES];
enum { IsItemOwned_, IsItemVisible_, GetItemX_, GetItemY_, IsVisibleToPlayer_, GetItemType_, N_ITEM_NATIVES };
static const struct {
    const char *name, *sig;
} g_want_item[N_ITEM_NATIVES] = {
    {"IsItemOwned", "(Hitem;)B"}, {"IsItemVisible", "(Hitem;)B"}, {"GetItemX", "(Hitem;)R"}, {"GetItemY", "(Hitem;)R"},
    {"IsVisibleToPlayer", "(RRHplayer;)B"}, {"GetItemType", "(Hitem;)Hitemtype;"},
};
static void *g_ifn[N_ITEM_NATIVES];
static int g_citems_ok, g_citems, g_items_arm, g_in_items;
static long g_items_n;
static int g_ctok_ok, g_version, g_vkt[2];
static int g_capply_ok; /* every command native found */
static int g_capply;    /* the harness asked MBOX - 32 this game */
static int g_own[MAX_PLAYERS];
static int *g_cres, g_ncres, g_cres_cap;
static long g_ca_steps, g_ca_cmds, g_ca_declined;

static int c1(int k, int a) {
    return ((N1)g_cfn[k])(a);
}
static int c2(int k, int a, int b) {
    return ((N2)g_cfn[k])(a, b);
}
static int c3(int k, int a, int b, int c) {
    return ((N3)g_cfn[k])(a, b, c);
}

/* W3S_ApplyOne's n: how many integers an op takes (1000000: unknown, the step's last) */
static int jass_len(int op) {
    switch (op) {
    case 80: case 97: case 98: case 99: return 1;
    case 93: case 94: case 95: case 84: return 2;
    case 3: case 5: case 9: case 96: return 3;
    case 2: case 6: case 8: case 87: case 88: case 90: return 4;
    case 1: case 4: case 91: case 83: return 5;
    case 89: case 81: return 6;
    case 7: return 7;
    case 92: return 8;
    case 82: return 9;
    default: return 1000000;
    }
}

static int owned(int u) { /* W3S_Owned */
    if (!u)
        return 0;
    int p = owner_id(u);
    return p >= 0 && p < MAX_PLAYERS && g_own[p];
}

/* how many integers W3S_ApplyOne may read this step (it reads w3s_cmd[at + 1] for every op, its own n, and an
 * advised player's orders up to w3s_cmd[at + 4]); reading more than it does is harmless: past the step's count
 * the mailbox answers what the array holds there */
static int cmd_need(int *mine) {
    const int *m = sync_mbox(); /* w3s_cmd[i] is m[i + 1] */
    int n = m[0], need = n;
    *mine = 1;
    for (int at = 0; at < n;) {
        int op = m[at + 1], len = jass_len(op);
        if (at + (len > 5 ? len : 5) > need)
            need = at + (len > 5 ? len : 5);
        if (len == 1000000 || at + len > n || !(op == 1 || op == 2 || op == 3 || op == 5 || op == 6 || op == 87 || op == 90))
            *mine = 0;
        if (len == 1000000)
            break;
        at += len;
    }
    return need < 8000 ? need : 8000;
}

static int cmd_apply(void) {
    const int *m = sync_mbox();
    int n = m[0], mine, need = cmd_need(&mine);
    if (!mine) {
        g_ca_declined++;
        return need;
    }
    g_ca_steps++;
    for (int at = 0; at < n;) {
        const int *c = m + at + 1;
        int op = c[0], ok = 0, len = jass_len(op);
        if (op == 87 || op == 90) { /* (the player's index; W3S_Real is the identity without pairs) */
            int pl = n1(Player, c[1]);
            for (int k = 0; k < 2; k++) {
                int v = c[2 + k];
                if (op == 87) { /* what it has now less the tax (c - 65536), at least 0 (IMaxBJ) */
                    int d = (int)((unsigned)n2(GetPlayerState, pl, 1 + k) - ((unsigned)v - 65536u));
                    v = 0 > d ? 0 : d;
                }
                ((N3)g_cfn[SetPlayerState_])(pl, 1 + k, v);
            }
            ok = 1;
        } else {
            int u = c3(LoadUnitHandle_, g_ht, c[1], 0);
            if (owned(u)) {
                if (op == 1) {
                    int x = c1(I2R_, (int)((unsigned)c[3] - 65536u)), y = c1(I2R_, (int)((unsigned)c[4] - 65536u));
                    ok = ((N4)g_cfn[IssuePointOrderById_])(u, c[2], (int)&x, (int)&y) != 0;
                } else if (op == 2)
                    ok = c3(IssueTargetOrderById_, u, c[2], c3(LoadUnitHandle_, g_ht, c[3], 0)) != 0;
                else if (op == 3)
                    ok = c2(IssueImmediateOrderById_, u, c[2]) != 0;
                else if (op == 6)
                    ok = c3(IssueTargetOrderById_, u, c[2], c3(LoadDestructableHandle_, g_ht, c[3], 0)) != 0;
                else if (n1(GetHeroSkillPoints, u) > 0) { /* 5 */
                    c2(SelectHeroSkill_, u, c[2]);
                    ok = 1;
                }
            }
        }
        if (g_ncres == g_cres_cap) {
            int cap = g_cres_cap ? g_cres_cap * 2 : 256;
            int *b = realloc(g_cres, cap * sizeof *b);
            if (b) {
                g_cres = b;
                g_cres_cap = cap;
            }
        }
        if (g_ncres < g_cres_cap)
            g_cres[g_ncres++] = ok;
        g_ca_cmds++;
        at += len;
    }
    return -1;
}

static void vkt_records(int flags) { /* W3S_WriteObs's first records */
    typedef int(__cdecl * N0)(void);
    char b[160];
    int len = 0;
    uint32_t sum = 0;
    int x = ((N0)g_tfn[GetCameraTargetPositionX_])(), y = ((N0)g_tfn[GetCameraTargetPositionY_])();
    x = ((N1)g_tfn[R2I_])((int)&x); /* (a real argument: its address) */
    y = ((N1)g_tfn[R2I_])((int)&y);
    memcpy(b, "V\n", 2);
    len = 2;
    pbuf_tok(b, &len, sizeof b, g_version, NULL);
    memcpy(b + len, "K\n", 2);
    len += 2;
    pbuf_tok(b, &len, sizeof b, x, &sum);
    pbuf_tok(b, &len, sizeof b, y, &sum);
    pbuf_tok(b, &len, sizeof b, (int)sum, NULL);
    memcpy(b + len, "T\n", 2);
    len += 2;
    sum = 0;
    pbuf_tok(b, &len, sizeof b, g_vkt[0], &sum);
    pbuf_tok(b, &len, sizeof b, g_vkt[1], &sum);
    pbuf_tok(b, &len, sizeof b, flags & 1, &sum);
    pbuf_tok(b, &len, sizeof b, (flags >> 1) & 1, &sum);
    pbuf_tok(b, &len, sizeof b, (int)sum, NULL);
    obs_bytes(b, len);
}

static void cmd_records(void) { /* W3S_WriteObs's "C" records: the result, its sum */
    for (int i = 0; i < g_ncres; i++)
        obs_bytes(g_cres[i] ? "C\n1\n1\n" : "C\n0\n0\n", 6);
    g_ncres = 0;
}

/* W3SIM_CUTS & 8589934592 (round 4, with 1073741824): the items on the ground (W3S_WriteObs's EnumItemsInRect over the
 * whole map) without a JASS thread per item. The native and its walk run as before; its per-item callback (0x49b200:
 * the widget's type, its mask, the filter, the enum item's handle pushed and popped) runs here with the shim's
 * W3S_SerItem in place of the JASS function: the same natives in its order (IsItemOwned, IsItemVisible, GetWidgetLife,
 * GetItemX / Y, IsVisibleToPlayer per seer, SaveInteger of the step (from the "T" record's call) under key 9,
 * GetItemTypeId, R2I, GetItemType), the "G" record written here. The harness arms it (MBOX - 44, after MBOX - 43: on?)
 * for its next EnumItemsInRect only. */
#define VA_ENUMITEMS 0x48ff80 /* cdecl EnumItemsInRect(rect, filter, code) native */
#define VA_ITEM_CB 0x49b200   /* cdecl int cb(CWidget *, ctx {code, filter, jass state, code object}) */
typedef void(__cdecl *EnumItemsFn)(int, int, int);
typedef int(__cdecl *ItemCbFn)(BYTE *, DWORD *);
static EnumItemsFn EnumItems_orig;
static ItemCbFn ItemCb_orig;

static void ser_item(int h) { /* W3S_SerItem */
    if (((N1)g_ifn[IsItemOwned_])(h) || !((N1)g_ifn[IsItemVisible_])(h) || !(r1(GetWidgetLife, h) > 0.405f))
        return;
    int x = ((N1)g_ifn[GetItemX_])(h), y = ((N1)g_ifn[GetItemY_])(h), bits = 0;
    for (int k = 0; k < g_nseer; k++)
        if (((N3)g_ifn[IsVisibleToPlayer_])((int)&x, (int)&y, g_seer_player[k]))
            bits += g_seer_bit[k];
    ((N4)g_fn[SaveInteger])(g_ht, h, 9, g_vkt[0]);
    if (!bits)
        return;
    char b[96];
    int len = 2;
    uint32_t sum = 0;
    memcpy(b, "G\n", 2);
    pbuf_tok(b, &len, sizeof b, n1(GetItemTypeId, h), &sum);
    pbuf_tok(b, &len, sizeof b, ((N1)g_tfn[R2I_])((int)&x), &sum);
    pbuf_tok(b, &len, sizeof b, ((N1)g_tfn[R2I_])((int)&y), &sum);
    pbuf_tok(b, &len, sizeof b, bits, &sum);
    pbuf_tok(b, &len, sizeof b, ((N1)g_ifn[GetItemType_])(h) == 2 /* ITEM_TYPE_POWERUP */, &sum);
    pbuf_tok(b, &len, sizeof b, (int)sum, NULL);
    obs_bytes(b, len);
}

static void __cdecl EnumItems_hook(int rect, int filter, int code) {
    if (g_items_arm) {
        g_items_arm = 0;
        g_in_items = 1;
        EnumItems_orig(rect, filter, code);
        g_in_items = 0;
        return;
    }
    EnumItems_orig(rect, filter, code);
}

static int __cdecl ItemCb_hook(BYTE *w, DWORD *ctx) {
    if (!g_in_items || !w || !ctx)
        return ItemCb_orig(w, ctx);
    typedef int(__fastcall * VGet)(BYTE *, void *);
    typedef int(__cdecl * TypeFn)(void);
    typedef int(__fastcall * MaskFn)(BYTE *, void *);
    if (((VGet)(*(void ***)w)[0x1c / 4])(w, NULL) != ((TypeFn)VA(0x6c5350))())
        return 1;
    if ((*(DWORD *)(w + 0x198) & *(DWORD *)(w + 0x194)) != 0xffffffffu && ((MaskFn)VA(0x7fc7e0))(w + 0x194, NULL))
        return 1;
    void *st = (void *)(size_t)ctx[2];
    ((EnumPushFn)VA(VA_ENUM_PUSH))(st, NULL, 0xe, w, 0);
    BYTE *flt = (BYTE *)(size_t)ctx[1];
    if ((!flt || ((VGet)(*(void ***)flt)[0x78 / 4])(flt, NULL)) && ctx[3]) {
        ((EnumPushFn)VA(VA_ENUM_PUSH))(st, NULL, 0xd, w, 0);
        int h = ((EnumTopFn)VA(VA_ENUM_TOP))(st, NULL, 0xd);
        if (h) {
            g_items_n++;
            ser_item(h);
        }
        ((EnumPopFn)VA(VA_ENUM_POP))(st, NULL, 0xd, 1);
    }
    ((EnumPopFn)VA(VA_ENUM_POP))(st, NULL, 0xe, 1);
    return 1;
}

static void install_items(void) {
    BYTE *f = VA(VA_ENUMITEMS), *cb = VA(VA_ITEM_CB);
    /* the callback as read: its checks' calls, the pushes and pops, the code's run */
    static const struct {
        int at;
        DWORD to;
    } calls[] = {{0x26, 0x6c5350}, {0x47, 0x7fc7e0}, {0x58, 0x4ebf00}, {0x7b, 0x4ebf00}, {0x98, 0x4e4720}, {0xb4, 0x4ebce0}, {0xc0, 0x4ebce0}};
    int ok = f[0] == 0x55 && f[1] == 0x8b && f[2] == 0xec && cb[0] == 0x55 && cb[1] == 0x8b && cb[2] == 0xec &&
             cb[0x21] == 0xff && cb[0x22] == 0x50 && cb[0x23] == 0x1c && cb[0x66] == 0xff && cb[0x67] == 0x50 && cb[0x68] == 0x78;
    for (unsigned k = 0; ok && k < sizeof calls / sizeof calls[0]; k++)
        ok = cb[calls[k].at] == 0xe8 && cb + calls[k].at + 5 + *(int32_t *)(cb + calls[k].at + 1) == (BYTE *)VA(calls[k].to);
    if (!ok) {
        shim_log("items: unexpected code, not installed");
        return;
    }
    MH_STATUS s = MH_Initialize();
    if (s != MH_OK && s != MH_ERROR_ALREADY_INITIALIZED)
        return;
    if (MH_CreateHook(f, (void *)EnumItems_hook, (void **)&EnumItems_orig) != MH_OK || MH_EnableHook(f) != MH_OK ||
        MH_CreateHook(cb, (void *)ItemCb_hook, (void **)&ItemCb_orig) != MH_OK || MH_EnableHook(cb) != MH_OK) {
        shim_log("items: hook failed");
        return;
    }
    g_citems = 1;
}

/* W3SIM_PROFILE: where a step's time goes (mean ms per step, every 200 steps): GO -> the pass begins
 * (the game), the pass, -> the observation ends (the result, the players, the rest), -> the sync. */
static int g_prof;
static LONGLONG g_t_mark, g_ph_t[4];
static int g_ph_at = -1, g_ph_steps;

void units_mark(int phase) { /* 0: GO; 1: the pass begins; 2: it ends; 3: the observation ends; 4: the sync */
    if (!g_prof)
        return;
    LONGLONG now = real_qpc_ticks();
    if (g_ph_at >= 0 && g_ph_at < 4 && phase != 0)
        g_ph_t[g_ph_at] += now - g_t_mark;
    g_t_mark = now;
    g_ph_at = phase;
    if (phase == 4 && ++g_ph_steps == 200) {
        double f = 1000.0 / (double)real_qpc_freq() / g_ph_steps;
        shim_log("unit records: ms per step: the game %.2f, the pass over the units %.2f, the rest of the observation %.2f, "
                 "to the sync %.2f", g_ph_t[0] * f, g_ph_t[1] * f, g_ph_t[2] * f, g_ph_t[3] * f);
        memset(g_ph_t, 0, sizeof g_ph_t);
        g_ph_steps = 0;
    }
}

/* the harness's calls (see the top); `key`: the tech id it asked for. Returns 1 if the call was one of them. */
int units_call(int key, int mbox, int *result) {
    *result = 0;
#ifdef W3SIM_REPLAY
    if (replay_call(key, mbox, result))
        return 1;
#endif
    if (key == mbox - 2) { /* a game's first step (after a reload the last game's tables are gone) */
        g_v_unit = 0;
        g_ht = g_abil = g_all = g_nseer = g_pending = g_pass = g_nremoved = g_rec_len = g_pair_first = g_paired = 0;
        g_npid = 0;
        g_cur.u = 0;
        g_cobs = g_cobs_over = g_jres_n = 0;
        g_obs_raw_player = g_capply = g_ncres = 0;
        kt_clear();
        memset(g_cobs_agent, 0, sizeof g_cobs_agent);
        memset(g_cobs_result, 0, sizeof g_cobs_result);
        cuts_game_start();
        *result = obs_capture_on() ? g_ok : 0;
        return 1;
    }
    if (!g_ok)
        return 0;
    if (g_ok == 2)
        verify_close();
    if (key < 0) {
        int v = -key, reg = g_pending;
        g_pending = 0;
        if (reg == 5)
            g_ht = v;
        else if (reg == 6)
            g_abil = v;
        else if (reg == 13)
            g_all = v;
        else if (reg == 14)
            g_max_players = v;
        else if (reg == 16)
            g_pair_first = v - 1;
        else if (reg == 17)
            g_paired = v;
        else if (reg == 19 || reg == 20) {
            if (v >= 1 && v <= MAX_PLAYERS)
                g_cobs_agent[v - 1] = reg == 19;
        } else if (reg == 30) {
            g_obs_raw_player = v;
        } else if (reg == 31) {
            g_version = v - 1;
        } else if (reg == 39 || reg == 40) { /* the step, the time */
            g_vkt[reg - 39] = v - 1;
            g_pending = reg + 1;
        } else if (reg == 41) {
            vkt_records(v - 1);
        } else if (reg == 33 || reg == 34) {
            if (v >= 1 && v <= MAX_PLAYERS)
                g_own[v - 1] = reg == 33;
        } else if (reg == 28) {
            g_cobs_victory = v - 1;
            g_cobs = g_paired || (g_cobs_victory != 0 && g_cobs_victory != 3) ? 0 : (cuts_flags() & 262144) ? (g_cobs_check_env ? 2 : 1) : 0;
            shim_log("players/result in C: %s (victory %d)", g_cobs == 2 ? "checked against the harness's" : g_cobs ? "on" : "off", g_cobs_victory);
        } else if (reg == 26) {
            if (g_jres_n < MAX_PLAYERS && g_jres_n < g_max_players) {
                if (g_cobs_result[g_jres_n] != v - 1)
                    cobs_mismatch("a player's result", g_cobs_result[g_jres_n], v - 1);
                g_jres[g_jres_n++] = v - 1;
            }
        } else if (reg == 27) {
            g_cobs_checks++;
            if (g_cobs_over != v - 1)
                cobs_mismatch("game over", g_cobs_over, v - 1);
            if (g_cobs_checks % 2000 == 0)
                shim_log("players/result in C: %ld steps checked, %ld mismatches", g_cobs_checks, g_cobs_bad);
        }
        else if (reg == 8) {
            if (g_nseer < MAX_PLAYERS && v >= 1 && v <= MAX_PLAYERS) {
                g_seer_player[g_nseer] = n1(Player, v - 1);
                g_seer_bit[g_nseer++] = 1 << pid(v - 1);
            }
        } else if (g_ht && g_abil) {
            if (g_pass && g_all)
                pass_unit(v);
            else if (g_ok == 2)
                unit_record(v);
        }
        return 1;
    }
    if (key == mbox - 3 || key == mbox - 4) { /* (W3SIM_UNITS=2) */
        g_full = key == mbox - 4;
        return 1;
    }
    if (key == mbox - 9 || key == mbox - 10) {
        units_mark(1);
        g_full = key == mbox - 10;
        g_pass = 1;
        g_rec_len = 0;
        memset(g_halls, 0, sizeof g_halls);
        memset(g_mobile, 0, sizeof g_mobile);
        memset(g_alive, 0, sizeof g_alive);
        return 1;
    }
    if (key == mbox - 15) { /* the pass is over (its counts are asked for next) */
        g_pass = 0;
        units_mark(2);
        return 1;
    }
    if (key == mbox - 11) { /* the pass's records, where the harness wrote its own */
        if (g_rec_len)
            obs_bytes(g_rec, g_rec_len);
        g_rec_len = 0;
        return 1;
    }
    if (key == mbox - 12) {
        for (int i = 0; i < g_nremoved; i++) {
            g_sum = 0;
            g_rec_len = 0;
            rec_bytes("R", 1);
            tok(g_removed[i]);
            uint32_t sum = g_sum;
            tok((int)sum);
            obs_bytes(g_rec, g_rec_len);
        }
        g_rec_len = g_nremoved = 0;
        return 1;
    }
    if (key <= mbox - 100 && key > mbox - 100 - 3 * MAX_PLAYERS) {
        int i = mbox - 100 - key;
        *result = (i % 3 == 0 ? g_halls : i % 3 == 1 ? g_mobile : g_alive)[i / 3];
        return 1;
    }
    if (key == mbox - 5 || key == mbox - 6 || key == mbox - 8 || key == mbox - 13 || key == mbox - 14 || key == mbox - 16 ||
        key == mbox - 17 || key == mbox - 19 || key == mbox - 20 || key == mbox - 28 || key == mbox - 26 || key == mbox - 27) {
        g_pending = mbox - key;
        return 1;
    }
    if (key == mbox - 18) { /* (W3SIM_CUTS & 262144) the players' records and the result in C: 1, or 2 to check them */
        *result = (cuts_flags() & 262144) && g_ok == 1 ? (g_cobs_check_env ? 2 : 1) : 0;
        return 1;
    }
    if (g_cobs && key == mbox - 21) {
        g_jres_n = 0;
        *result = cobs_check();
        return 1;
    }
    if (g_cobs && key == mbox - 22) {
        cobs_tie();
        return 1;
    }
    if (g_cobs && key == mbox - 24) {
        if (g_cobs == 2) {
            const char *d;
            g_pobs_mark = obs_peek(&d);
            g_pexp_len = cobs_records(g_pexp, sizeof g_pexp);
        } else {
            char b[4096];
            int n = cobs_records(b, sizeof b);
            if (n)
                obs_bytes(b, n);
        }
        return 1;
    }
    if (g_cobs == 2 && key == mbox - 25) {
        const char *d;
        int n = obs_peek(&d) - g_pobs_mark;
        if (n != g_pexp_len || memcmp(d + g_pobs_mark, g_pexp, n) != 0) {
            if (g_cobs_bad++ < 40) {
                char a[160], b[160];
                int ka = n < 150 ? n : 150, kb = g_pexp_len < 150 ? g_pexp_len : 150;
                memcpy(a, d + g_pobs_mark, ka > 0 ? ka : 0);
                a[ka > 0 ? ka : 0] = 0;
                memcpy(b, g_pexp, kb);
                b[kb] = 0;
                for (char *c = a; *c; c++) if (*c == '\n') *c = ' ';
                for (char *c = b; *c; c++) if (*c == '\n') *c = ' ';
                shim_log("players/result in C: MISMATCH players' records: the harness's [%s], the shim's [%s]", a, b);
            }
        }
        return 1;
    }
    if (key == mbox - 7) {
        g_nseer = 0;
        return 1;
    }
    if (key == mbox - 29) { /* (W3SIM_CUTS & 1073741824) W3S_Tok's integers here */
        *result = (cuts_flags() & 1073741824) && g_ok == 1 && g_ctok_ok && obs_capture_on() ? 1 : 0;
        if (*result)
            shim_log("tokens: W3S_Tok's integers written by the shim (W3SIM_CUTS & 1073741824)");
        return 1;
    }
    if (key == mbox - 32) { /* (W3SIM_CUTS & 2147483648) the agents' commands applied here */
        g_capply = (cuts_flags() & 2147483648LL) && g_ok == 1 && g_capply_ok && !g_paired && obs_capture_on();
        memset(g_own, 0, sizeof g_own);
        g_ncres = 0;
        *result = g_capply;
        if (g_capply)
            shim_log("commands: applied by the shim (W3SIM_CUTS & 2147483648)");
        return 1;
    }
    if (key == mbox - 30 || key == mbox - 31 || key == mbox - 39 || key == mbox - 33 || key == mbox - 34) {
        g_pending = mbox - key;
        return 1;
    }
    if (g_capply && key == mbox - 35) {
        *result = cmd_apply();
        return 1;
    }
    if (g_capply && key == mbox - 38) { /* (the harness applies them: how many integers to read) */
        int mine;
        *result = cmd_need(&mine);
        g_ca_declined++;
        return 1;
    }
    if (key == mbox - 43) { /* (W3SIM_CUTS & 8589934592) the items on the ground here */
        *result = g_citems && g_ok == 1 && g_ctok_ok && obs_capture_on() ? 1 : 0;
        if (*result)
            shim_log("items: W3S_SerItem in the shim (W3SIM_CUTS & 8589934592)");
        return 1;
    }
    if (key == mbox - 44) {
        g_items_arm = g_citems;
        return 1;
    }
    if (g_capply && key == mbox - 36) {
        cmd_records();
        return 1;
    }
    if (g_capply && key == mbox - 37) { /* (the mailbox past a step's count: what the harness's array holds) */
        int *m = sync_mbox();
        memset(m + 1, 0, 8000 * sizeof *m);
        return 1;
    }
    return 0;
}

void units_cmd_report(void) {
    if (g_ca_steps || g_ca_declined)
        shim_log("commands/s: steps applied here %ld (%ld commands), the harness's %ld", g_ca_steps, g_ca_cmds, g_ca_declined);
    g_ca_steps = g_ca_cmds = g_ca_declined = 0;
}

void units_install(void) {
    IMAGE_DOS_HEADER *dos = (IMAGE_DOS_HEADER *)g_base;
    IMAGE_NT_HEADERS *nt = (IMAGE_NT_HEADERS *)(g_base + dos->e_lfanew);
    IMAGE_SECTION_HEADER *sec = IMAGE_FIRST_SECTION(nt);
    uintptr_t lo = (uintptr_t)g_base, hi = lo + nt->OptionalHeader.SizeOfImage;
    BYTE *text = g_base + sec->VirtualAddress;
    DWORD size = sec->Misc.VirtualSize;
    int found = 0;
    for (DWORD i = 0; i + 16 <= size; i++) {
        if (text[i] != 0x68 || text[i + 5] != 0x68 || text[i + 10] != 0x68 || text[i + 15] != 0xe8)
            continue;
        uintptr_t sig = *(uint32_t *)(text + i + 1), name = *(uint32_t *)(text + i + 6), fn = *(uint32_t *)(text + i + 11);
        if (sig < lo || sig + 64 > hi || name < lo || name + 64 > hi || fn < (uintptr_t)text || fn >= (uintptr_t)text + size)
            continue;
        for (int k = 0; k < N_NATIVES; k++)
            if (!g_fn[k] && strcmp((const char *)name, g_want[k].name) == 0 && strcmp((const char *)sig, g_want[k].sig) == 0) {
                g_fn[k] = (void *)fn;
                found++;
            }
        for (int k = 0; k < N_CMD_NATIVES; k++)
            if (!g_cfn[k] && strcmp((const char *)name, g_want_cmd[k].name) == 0 && strcmp((const char *)sig, g_want_cmd[k].sig) == 0) {
                g_cfn[k] = (void *)fn;
                g_capply_ok++;
            }
        for (int k = 0; k < N_TOK_NATIVES; k++)
            if (!g_tfn[k] && strcmp((const char *)name, g_want_tok[k].name) == 0 && strcmp((const char *)sig, g_want_tok[k].sig) == 0) {
                g_tfn[k] = (void *)fn;
                g_ctok_ok++;
            }
        for (int k = 0; k < N_ITEM_NATIVES; k++)
            if (!g_ifn[k] && strcmp((const char *)name, g_want_item[k].name) == 0 && strcmp((const char *)sig, g_want_item[k].sig) == 0) {
                g_ifn[k] = (void *)fn;
                g_citems_ok++;
            }
    }
    g_ctok_ok = g_ctok_ok == N_TOK_NATIVES;
    g_citems_ok = g_citems_ok == N_ITEM_NATIVES;
    g_capply_ok = g_capply_ok == N_CMD_NATIVES;
    char buf[16];
    int mode = GetEnvironmentVariableA("W3SIM_UNITS", buf, sizeof buf) ? atoi(buf) : 1;
    g_prof = GetEnvironmentVariableA("W3SIM_PROFILE", buf, sizeof buf) && buf[0] >= '1';
    g_ok = found == N_NATIVES ? mode : 0;
    g_cobs_check_env = GetEnvironmentVariableA("W3SIM_COBS_CHECK", buf, sizeof buf) && buf[0] == '1';
    if (g_ok == 1) {
        int cuts = cuts_flags();
        g_dedupe = (cuts & 32) != 0;
        if (g_dedupe)
            shim_log("unit records: getters once per unit, player ids by handle (W3SIM_CUTS & 32)");
        if (cuts & 16)
            install_fg_pass();
        if (cuts & 8388608)
            install_kt();
        if ((cuts_flags() & 8589934592LL) && g_citems_ok && g_ctok_ok)
            install_items();
    }
    if (g_ok)
        shim_log("unit records: %s (%d natives)", g_ok == 2 ? "checked against the harness's" : "written by the shim", found);
    else if (!mode)
        shim_log("unit records: written by the harness (W3SIM_UNITS=0)");
    else
        for (int k = 0; k < N_NATIVES; k++)
            if (!g_fn[k])
                shim_log("unit records: native %s %s not found, the harness writes them", g_want[k].name, g_want[k].sig);
}
