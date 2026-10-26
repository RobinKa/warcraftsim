/* Approximations of the simulation (W3SIM_CUTS 2^44 and up). Each changes what the game computes: a game with one is
 * no longer the real game. For training games only, and only once approved: off by default and never in a run's
 * flags unless the user says so (evaluation games stay the exact game). Measured in docs/game-speedups.md
 * ("Approximations (disabled)").
 *
 * None of the three is recommended (2026-10-26: docs/game-speedups.md has the numbers):
 *
 *  17592186044416 (2^44)  REJECTED (the economy halves). Slow relation timers at half rate: a periodic relation
 *          timer (the 'sig^' wheel's, 0x7fae00 / 0x7f9660; its owner's vtable +0x48, for a signal source 0x8b8ae0
 *          -> 0x8b8d80 -> the sensors' 0x8bb120: their grid query, the enter/leave diff and its events) with a
 *          period (+8) of 1 s or more does its work on every other firing only: 0x8b8ae0 returns at once on the
 *          others (the wheel still pops and re-arms it). One-shot timers (+0x10 bit 0 clear: the cleanup at their
 *          end) and the source's next-tick timer (+0x20) always run. These timers are not polls: their owners set
 *          each next period from the situation (0.1 to 150 s, many odd values: when a unit could be in range at the
 *          soonest), so a skipped firing is a missed event (a worker at the mine waits a period longer)
 *  35184372088832 (2^45)  no gain measurable. Stuck units' local searches less often: after a local A* search
 *          (0x8a6cc0 -> 0x8bd560 -> the loop 1048576) that ran out of its node budget (+0x68, ~680), the unit's
 *          next local search waits 20 pathing ticks instead of the game's 10 (0x8a6cc0 answers "not now" as its
 *          own rate limit does)
 *  70368744177664 (2^46)  REJECTED, as 2^44: middle relation timers (periods of 0.25 s up to 1 s) at half rate
 *
 * (A first version of 2^44 and 2^46 doubled the period where a relation timer is made, 0x8b89c0: nothing changed,
 * neither the economy nor the speed, as the owners re-arm their timers with periods of their own.)
 *
 * W3SIM_APPROX_FROM_TICK=n: the approximations act from the pathing world's tick n on (32 a game second; the exact game
 * before it): a game measured in windows right after it, side by side with the exact one from the same state.
 *
 * W3SIM_APPROX_STATS=1: counters for choosing and judging them, in shim.log at each profile report (W3SIM_PROFILE):
 * the pathing ticks (32 a game second), the relation timers made (by caller and period) and fired (by period, with
 * their rdtsc cycles), the local and long-range searches (calls, cycles, repeats) and the local A* runs by expansions.
 *
 * Offsets are for the 1.29.2.9231 executable (VAs, 0x400000-based).
 */
#include "w3shim.h"

#include "MinHook.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VA_(x) (g_base + ((x) - 0x400000))

#define AX_SIG_SLOW 17592186044416LL
#define AX_ASTAR_WAIT 35184372088832LL
#define AX_SIG_MID 70368744177664LL

int g_ax_stats;
int g_ax_as_last_exh; /* (pathing.c: the last local A* run ran out of its node budget) */
long long g_ax_as_e, g_ax_as_lim, g_ax_as_exh, g_ax_as_exh_e, g_ax_as_empty, g_ax_as_empty_e;
long long g_ax_eb_n[8], g_ax_eb_c[8], g_ax_eb_e[8]; /* A* runs by expansions: <32, <64, <128, <256, <512, <=lim, more */
static long long g_ax;
static int g_ax_sigskip; /* (2^44 or 2^46 on) */
static DWORD g_ax_from; /* (W3SIM_APPROX_FROM_TICK) */
static long long g_sig_skips, g_lw_skips, g_lw_set;

static inline unsigned long long tsc(void) { return __builtin_ia32_rdtsc(); }
static inline float f_of(DWORD b) {
    float f;
    memcpy(&f, &b, 4);
    return f;
}

static DWORD path_ticks(void) { /* the pathing world's tick counter (0x8b5220 counts it) */
    BYTE *pw = *(BYTE **)VA_(0x1161d90);
    return pw ? *(DWORD *)(pw + 0x538) : 0;
}

static inline int ax_on(void) { return !g_ax_from || path_ticks() >= g_ax_from; }

static int ax_hook(DWORD va, void *fn, void **orig, const char *name) {
    BYTE *p = VA_(va);
    if (!(p[0] == 0x55 && p[1] == 0x8b && p[2] == 0xec)) {
        shim_log("approx: %s: unexpected prologue, not hooked", name);
        return 0;
    }
    if (MH_CreateHook(p, fn, orig) != MH_OK || MH_EnableHook(p) != MH_OK) {
        shim_log("approx: %s: hook failed", name);
        return 0;
    }
    return 1;
}

/* ---- 0x8b89c0: thiscall timer *add(wheel, owner, &key, float period, &due, x), ret 0x14 ----
 * a relation timer on a wheel (0x7fae00 / 0x7f9660 fire it: the owner's vtable +0x48, for a signal source 0x8b8ae0;
 * kept when its +0x10 bit 0 is set, then due again a period after it fired) */
typedef void *(__fastcall *TAddFn)(BYTE *, void *, void *, DWORD *, DWORD, DWORD *, DWORD);
static TAddFn TAdd_orig;
#define NTA 48
static struct { DWORD ret, period; long long n; } g_ta[NTA];
static int g_nta;
static long long g_ta_other;

static void *__fastcall TAdd_hook(BYTE *w, void *edx, void *owner, DWORD *key, DWORD period, DWORD *due, DWORD x) {
    if (g_ax_stats) {
        DWORD ret = (DWORD)(size_t)__builtin_return_address(0) - (DWORD)(size_t)g_base + 0x400000;
        int i = 0;
        while (i < g_nta && !(g_ta[i].ret == ret && g_ta[i].period == period))
            i++;
        if (i < g_nta)
            g_ta[i].n++;
        else if (g_nta < NTA)
            g_ta[g_nta].ret = ret, g_ta[g_nta].period = period, g_ta[g_nta++].n = 1;
        else
            g_ta_other++;
    }
    return TAdd_orig(w, edx, owner, key, period, due, x);
}

/* ---- 0x8b8ae0: thiscall void fire(source, timer), ret 4 (stats only): a signal source's relation timer fired ---- */
typedef void(__fastcall *SigFn)(BYTE *, void *, BYTE *);
static SigFn Sig_orig;
#define NPB 7 /* periods: 0, <=0.1, <=0.25, <0.5... (0.25, 0.5], (0.5, 1], (1, 4], more */
static long long g_sig_n[NPB], g_sig_c[NPB];
static int pbucket(float p) {
    return p <= 0 ? 0 : p <= 0.1001f ? 1 : p <= 0.25f ? 2 : p <= 0.5f ? 3 : p <= 1 ? 4 : p <= 4 ? 5 : 6;
}
/* per timer, direct-mapped by its address: its firings counted (a collision restarts the count: a firing more) */
#define NT 8192
static struct { BYTE *t; DWORD n; } g_tf[NT];
static void __fastcall Sig_hook(BYTE *src, void *edx, BYTE *t) {
    float p = f_of(*(DWORD *)(t + 8));
    if (g_ax_sigskip && (*(BYTE *)(t + 0x10) & 1) && t != *(BYTE **)(src + 0x20) &&
        (((g_ax & AX_SIG_SLOW) && p >= 1.0f) || ((g_ax & AX_SIG_MID) && p >= 0.25f && p < 1.0f)) && ax_on()) {
        unsigned h = ((unsigned)(size_t)t >> 3) * 2654435761u >> 19;
        if (g_tf[h].t != t)
            g_tf[h].t = t, g_tf[h].n = 0;
        if (g_tf[h].n++ & 1) {
            g_sig_skips++;
            return;
        }
    }
    if (!g_ax_stats) {
        Sig_orig(src, edx, t);
        return;
    }
    int k = pbucket(p);
    unsigned long long c0 = tsc();
    Sig_orig(src, edx, t);
    g_sig_n[k]++;
    g_sig_c[k] += tsc() - c0;
}

/* per follower (a unit's pathing agent), direct-mapped by its address: a collision only forgets an entry */
#define NF 4096
static inline unsigned fslot(const BYTE *f) { return ((unsigned)(size_t)f >> 5) * 2654435761u >> 20; }

/* ---- 0x8a6cc0: thiscall char local(follower, x, &goal), ret 8 ----
 * the local search to the next waypoint: at most once in 10 pathing ticks a follower (+0x7c: its last; the limit
 * dropped again when the movement class's budget had room), else AL 0 and nothing changed */
typedef DWORD(__fastcall *LocFn)(BYTE *, void *, DWORD, float *);
static LocFn Loc_orig;
static long long g_loc_n, g_loc_ok, g_loc_c, g_loc_rep10, g_loc_rep10_c;
static struct { BYTE *f; DWORD until, last; } g_lw[NF];
static DWORD __fastcall Loc_hook(BYTE *f, void *edx, DWORD x, float *goal) {
    DWORD now = path_ticks();
    unsigned h = fslot(f);
    if ((g_ax & AX_ASTAR_WAIT) && g_lw[h].f == f && (int)(g_lw[h].until - now) > 0 && ax_on()) {
        g_lw_skips++;
        return 0;
    }
    g_ax_as_last_exh = 0;
    unsigned long long c0 = g_ax_stats ? tsc() : 0;
    DWORD r = Loc_orig(f, edx, x, goal);
    if (g_ax_stats) {
        unsigned long long c = tsc() - c0;
        g_loc_c += c;
        g_loc_n++;
        if (r & 0xff) {
            g_loc_ok++;
            if (g_lw[h].f == f && now - g_lw[h].last < 10)
                g_loc_rep10++, g_loc_rep10_c += c;
        }
    }
    if (r & 0xff) {
        if (g_lw[h].f != f)
            g_lw[h].f = f, g_lw[h].until = now;
        g_lw[h].last = now;
        if ((g_ax & AX_ASTAR_WAIT) && g_ax_as_last_exh && ax_on()) {
            g_lw[h].until = now + 20;
            g_lw_set++;
        }
    }
    return r;
}

/* ---- 0x8a6a20: thiscall char longrange(follower, x, &goal, y), ret 0xc (stats only) ----
 * the long-range search, rate-limited as the local one (+0x80); repeats: the follower's previous search had the same
 * goal (within 4), or was less than 10 ticks before */
typedef DWORD(__fastcall *LongFn)(BYTE *, void *, DWORD, float *, DWORD);
static LongFn Long_orig;
static long long g_long_n, g_long_ok, g_long_c, g_long_rep, g_long_rep_c, g_long_rep10, g_long_rep10_c;
static struct { BYTE *f; float gx, gy; DWORD tick; } g_lr[NF];
static DWORD __fastcall Long_hook(BYTE *f, void *edx, DWORD x, float *goal, DWORD y) {
    float gx = goal[0], gy = goal[1];
    DWORD now = path_ticks();
    unsigned long long c0 = tsc();
    DWORD r = Long_orig(f, edx, x, goal, y);
    unsigned long long c = tsc() - c0;
    g_long_c += c;
    g_long_n++;
    if (r & 0xff) {
        g_long_ok++;
        unsigned h = fslot(f);
        if (g_lr[h].f == f) {
            float dx = g_lr[h].gx - gx, dy = g_lr[h].gy - gy;
            if (dx * dx + dy * dy < 16.0f)
                g_long_rep++, g_long_rep_c += c;
            if (now - g_lr[h].tick < 10)
                g_long_rep10++, g_long_rep10_c += c;
        }
        g_lr[h].f = f, g_lr[h].gx = gx, g_lr[h].gy = gy, g_lr[h].tick = now;
    }
    return r;
}

void approx_install(long long cuts) {
    char buf[8];
    g_ax_stats = GetEnvironmentVariableA("W3SIM_APPROX_STATS", buf, sizeof buf) && buf[0] == '1';
    g_ax = cuts & (AX_SIG_SLOW | AX_ASTAR_WAIT | AX_SIG_MID);
    if (!g_ax_stats && !g_ax)
        return;
    char fb[16];
    if (GetEnvironmentVariableA("W3SIM_APPROX_FROM_TICK", fb, sizeof fb))
        g_ax_from = (DWORD)strtoul(fb, NULL, 10);
    if (MH_Initialize() != MH_OK && MH_Initialize() != MH_ERROR_ALREADY_INITIALIZED)
        return;
    if (g_ax)
        shim_log("approx: APPROXIMATIONS ON (W3SIM_CUTS & %lld) from pathing tick %lu: this game is not the real game",
                 g_ax, (unsigned long)g_ax_from);
    if (g_ax_stats && VA_(0x8b8a47)[0] == 0xc2 && VA_(0x8b8a47)[1] == 0x14)
        ax_hook(0x8b89c0, (void *)TAdd_hook, (void **)&TAdd_orig, "relation timer add (stats)");
    /* 0x8b8ae0: the source's two timers at +0x1c and +0x20 compared, its ret 4 */
    if ((g_ax_stats || (g_ax & (AX_SIG_SLOW | AX_SIG_MID))) && VA_(0x8b8b6d)[0] == 0xc2 && VA_(0x8b8b6d)[1] == 0x04 &&
        ax_hook(0x8b8ae0, (void *)Sig_hook, (void **)&Sig_orig, "relation timer fire")) {
        g_ax_sigskip = (g_ax & (AX_SIG_SLOW | AX_SIG_MID)) != 0;
        if (g_ax_sigskip)
            shim_log("approx: periodic relation timers with periods of %s: their work on every other firing",
                     (g_ax & AX_SIG_SLOW) && (g_ax & AX_SIG_MID) ? "0.25 s and more" :
                     (g_ax & AX_SIG_SLOW) ? "1 s and more" : "0.25 s up to 1 s");
    }
    if ((g_ax & AX_ASTAR_WAIT) && !(cuts & 1048576))
        shim_log("approx: stuck units' local searches need 1048576 (the A* loop in C): not on");
    else if ((g_ax_stats || (g_ax & AX_ASTAR_WAIT)) &&
             ax_hook(0x8a6cc0, (void *)Loc_hook, (void **)&Loc_orig, "local search") && (g_ax & AX_ASTAR_WAIT))
        shim_log("approx: a local search out of nodes: the unit's next one 20 pathing ticks later");
    if (g_ax_stats) {
        ax_hook(0x8a6a20, (void *)Long_hook, (void **)&Long_orig, "long-range search (stats)");
        shim_log("approx: stats on");
    }
}

void approx_report(void) {
    if (g_ax)
        shim_log("approx: timer firings skipped=%lld; local searches put off=%lld, waits set=%lld", g_sig_skips,
                 g_lw_skips, g_lw_set);
    if (!g_ax_stats)
        return;
    shim_log("approx stats: ticks=%lu local=%lld ok=%lld (Mc %lld; <10 ticks after the last %lld, Mc %lld) long=%lld "
             "ok=%lld (Mc %lld; same goal %lld, Mc %lld; <10 ticks %lld, Mc %lld); A* e=%lld lim=%lld exhausted=%lld "
             "(e %lld) empty=%lld (e %lld)", (unsigned long)path_ticks(), g_loc_n, g_loc_ok, g_loc_c >> 20, g_loc_rep10,
             g_loc_rep10_c >> 20, g_long_n, g_long_ok, g_long_c >> 20, g_long_rep, g_long_rep_c >> 20, g_long_rep10,
             g_long_rep10_c >> 20, g_ax_as_e, g_ax_as_lim, g_ax_as_exh, g_ax_as_exh_e, g_ax_as_empty, g_ax_as_empty_e);
    char line[1024];
    int o = 0;
    for (int k = 0; k < NPB; k++)
        o += snprintf(line + o, sizeof line - o, " p%d=%lld (Mc %lld)", k, g_sig_n[k], g_sig_c[k] >> 20);
    o += snprintf(line + o, sizeof line - o, "; A* by e:");
    for (int k = 0; k < 7; k++)
        o += snprintf(line + o, sizeof line - o, " b%d=%lld (e %lld, Mc %lld)", k, g_ax_eb_n[k], g_ax_eb_e[k],
                      g_ax_eb_c[k] >> 20);
    shim_log("approx fires:%s", line);
    o = 0;
    for (int i = 0; i < g_nta && o < (int)sizeof line - 48; i++)
        o += snprintf(line + o, sizeof line - o, " %lx:%.3f=%lld", (unsigned long)g_ta[i].ret, f_of(g_ta[i].period),
                      g_ta[i].n);
    shim_log("approx timers:%s (other %lld)", line, g_ta_other);
}
