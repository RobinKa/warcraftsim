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
    HaveSavedInteger, Player, GroupRemoveUnit, FlushChildHashtable, N_NATIVES
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
    {"FlushChildHashtable", "(Hhashtable;I)V"},
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
static int g_full;
static int g_pending; /* what the next negative argument is: 0 a unit, else the register (5, 6, 8, 13, 14) */
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

/* W3S_SerUnit: the unit's record, if it changed since it was last written (or on a full snapshot) */
static void unit_record(int u) {
    /* (the order of the calls is W3S_SerUnit's: some of them make handles) */
    int flags = 0;
    int hero = n2(IsUnitType, u, TYPE_HERO) != 0;
    if (hero)
        flags += 1;
    if (n2(IsUnitType, u, TYPE_STRUCTURE))
        flags += 2;
    if (n2(IsUnitType, u, TYPE_PEON))
        flags += 4;
    if (n3(LoadBoolean, g_ht, u, 1))
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
    int dead = n2(IsUnitType, u, TYPE_DEAD) != 0;
    if (dead || r1(GetWidgetLife, u) < 0.405f)
        flags += 1024;
    if (n2(IsUnitType, u, TYPE_FLYING))
        flags += 2048;
    int typ = n1(GetUnitTypeId, u);
    int owner = n1(GetPlayerId, n1(GetOwningPlayer, u));
    int x = (int)r1(GetUnitX, u), y = (int)r1(GetUnitY, u), facing = (int)r1(GetUnitFacing, u);
    int hp = (int)(r1(GetWidgetLife, u) + 0.5f);
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
        if (!g_full && n3(HaveSavedInteger, g_ht, u, 3) && (uint32_t)n3(LoadInteger, g_ht, u, 3) == h)
            return;
        ((N4)g_fn[SaveInteger])(g_ht, u, 3, (int)h);
        if (dead)
            ((N4)g_fn[SaveBoolean])(g_ht, u, 4, 1);
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
    if (n1(GetUnitTypeId, u) == 0) { /* removed from the game */
        drop(u);
        return;
    }
    int dead = n2(IsUnitType, u, TYPE_DEAD) != 0;
    int p = n1(GetPlayerId, n1(GetOwningPlayer, u));
    if (!dead && p >= 0 && p < g_max_players && p < MAX_PLAYERS) {
        g_alive[p]++;
        if (n2(IsUnitType, u, TYPE_TOWNHALL))
            g_halls[p]++;
        else if (!n2(IsUnitType, u, TYPE_STRUCTURE))
            g_mobile[p]++;
    }
    /* reported as dead already: dropped (heroes stay, they can be revived). As in the harness, its
     * record is still written this once (its stored hash went with its table entries). */
    if (dead && !n2(IsUnitType, u, TYPE_HERO) && n3(LoadBoolean, g_ht, u, 4))
        drop(u);
    unit_record(u);
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
    if (key == mbox - 2) { /* a game's first step (after a reload the last game's tables are gone) */
        g_v_unit = 0;
        g_ht = g_abil = g_all = g_nseer = g_pending = g_pass = g_nremoved = g_rec_len = 0;
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
        else if (reg == 8) {
            if (g_nseer < MAX_PLAYERS && v >= 1 && v <= MAX_PLAYERS) {
                g_seer_player[g_nseer] = n1(Player, v - 1);
                g_seer_bit[g_nseer++] = 1 << (v - 1);
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
    if (key == mbox - 5 || key == mbox - 6 || key == mbox - 8 || key == mbox - 13 || key == mbox - 14) {
        g_pending = mbox - key;
        return 1;
    }
    if (key == mbox - 7) {
        g_nseer = 0;
        return 1;
    }
    return 0;
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
    }
    char buf[16];
    int mode = GetEnvironmentVariableA("W3SIM_UNITS", buf, sizeof buf) ? atoi(buf) : 1;
    g_prof = GetEnvironmentVariableA("W3SIM_PROFILE", buf, sizeof buf) && buf[0] >= '1';
    g_ok = found == N_NATIVES ? mode : 0;
    if (g_ok)
        shim_log("unit records: %s (%d natives)", g_ok == 2 ? "checked against the harness's" : "written by the shim", found);
    else if (!mode)
        shim_log("unit records: written by the harness (W3SIM_UNITS=0)");
    else
        for (int k = 0; k < N_NATIVES; k++)
            if (!g_fn[k])
                shim_log("unit records: native %s %s not found, the harness writes them", g_want[k].name, g_want[k].sig);
}
