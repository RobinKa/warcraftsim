/* W3SIM_CUTS, round 3: the path searches' hot loops in C, the same results and the same stores in the same order.
 *
 *  524288  the long-range search's relaxation (0x8a3d00, thiscall relax(search, node, parent), ret 8): the
 *          neighbour's heuristic (an integer square root) computed after the early return instead of before it,
 *          both square roots as floor(sqrt(double)) inline (0x99b740 gives floor(sqrt) for all 2^32 inputs), the
 *          heap push inline (its growth through the original)
 * 1048576  the local A* search's loop (0x8bf8b0, fastcall run(search)) in C: the open list's pop and push inline
 *          (the same binary heap, the same sift steps: ties pop in the same order), a node's expansion (the ring of
 *          cells around it, 0x8c09c0 / 0x8c0ae0 / 0x8c0c10 / 0x8c0d40 by the unit's size class; the eight
 *          neighbours' nodes found or made first, then relaxed in order), the relaxation (0x8bf980, its heuristic
 *          0x8bda00 after the early return) and the lookup of a cell's node (0x8bcf30's fast path; a node is made
 *          by the original). A ring cell's query (0x8bdbe0) is remembered for the run as W3SIM_CUTS & 64 does
 *          (its answer and its side effects on the search's "met the target" and "blocked" flags replayed), in one
 *          array of (run, answer) words.
 * 4194304  the pathing map's update of a cell's four movement classes' "blocked" bits (0x8b78b0, thiscall
 *          update(map, cell, &mask, bit, x, y), ret 0x14: four sub-cell queries 0x8be7a0 -> 0x8be420) in C, the four
 *          queries inline with their stores in the same order (the query object's mask, the grid's stamp counter,
 *          each widget's stamp)
 *
 * Offsets are for the 1.29.2.9231 executable (VAs, 0x400000-based).
 */
#include "w3shim.h"

#include "MinHook.h"

#include <stdlib.h>
#include <string.h>

#define VA_(x) (g_base + ((x) - 0x400000))

#define VA_LRELAX 0x8a3d00 /* thiscall void relax(search, node, parent), ret 8 (the long-range search) */
#define VA_LUNLINK 0x8a3ff0
#define VA_ISQRT2 0x99b740
#define VA_PUSH 0x8a3090   /* thiscall void push(heap, key, node, gen), ret 0xc */
#define VA_POP 0x8a2d20    /* thiscall entry *pop(heap, entry *out), ret 4 */
#define VA_ASTAR 0x8bf8b0  /* fastcall int run(search) */
#define VA_NODE 0x8bcf30   /* thiscall int node(search, x, y), ret 8 */
#define VA_CELL 0x8bdbe0   /* thiscall char cell(search, x, y), ret 8 */
#define VA_HEUR 0x8bda00   /* cdecl unsigned h(int *xy, int *goal, int k) */
#define VA_AUNLINK 0x8bfbc0 /* thiscall void unlink(search, node), ret 4 */
#define VA_EXPAND 0x8c0890 /* thiscall void expand(search, node), ret 4 */

typedef void(__fastcall *PushFn)(BYTE *, void *, unsigned, int, unsigned);
typedef void(__fastcall *UnlinkFn)(BYTE *, void *, int);
typedef int(__fastcall *NodeFn)(BYTE *, void *, int, int);
typedef unsigned(__fastcall *CellFn)(BYTE *, void *, int, int);
typedef unsigned(__cdecl *HeurFn)(int *, int *, int);
typedef unsigned(__cdecl *IsqrtFn)(unsigned);
typedef void(__fastcall *LRelaxFn)(BYTE *, void *, int, int);
typedef int(__fastcall *AstarFn)(BYTE *);

static long g_lr_n, g_lr_early, g_as_runs, g_as_pops, g_as_hits, g_as_miss; /* (main thread; read approximately) */

static inline unsigned isqrt_f(unsigned x) {
    double d = (double)(int)(x >> 1) * 2.0 + (double)(x & 1), r; /* (exact: x < 2^32) */
    __asm__("sqrtsd %1, %0" : "=x"(r) : "x"(d));
    return (unsigned)(int)r; /* (r <= 65536: truncated) */
}

static inline unsigned uabs(int v) { /* cdq; xor; sub */
    unsigned s = (unsigned)(v >> 31);
    return ((unsigned)v ^ s) - s;
}

/* the heap's push (0x8a3090) when it has room; else the original (it grows the array) */
static inline void heap_push(BYTE *h, unsigned key, int node, unsigned gen) {
    DWORD cnt = *(DWORD *)(h + 0x1c);
    if (*(DWORD *)(h + 0x18) < cnt + 1) {
        ((PushFn)(void *)VA_(VA_PUSH))(h, NULL, key, node, gen);
        return;
    }
    cnt++;
    *(DWORD *)(h + 0x1c) = cnt;
    DWORD *arr = *(DWORD **)(h + 0xc);
    unsigned i = cnt - 1;
    while (i > 1) {
        DWORD *p = arr + (i >> 1) * 3;
        if (p[0] < key)
            break;
        DWORD *c = arr + i * 3;
        c[0] = p[0];
        c[1] = p[1];
        c[2] = p[2];
        i >>= 1;
    }
    DWORD *c = arr + i * 3;
    c[0] = key;
    c[1] = (DWORD)node;
    c[2] = gen;
}

/* the heap's pop (0x8a2d20); the caller has checked that it holds an entry (count >= 2: slot 0 is unused) */
static inline void heap_pop(BYTE *h, DWORD *out) {
    DWORD *arr = *(DWORD **)(h + 0xc);
    DWORD cnt = *(DWORD *)(h + 0x1c);
    out[0] = arr[3];
    out[1] = arr[4];
    out[2] = arr[5];
    DWORD *l = arr + (cnt - 1) * 3, l0 = l[0], l1 = l[1], l2 = l[2];
    if (cnt != 0) { /* (0x8a3f30(cnt - 1, 1): the last entry erased, nothing moved) */
        cnt--;
        *(DWORD *)(h + 0x1c) = cnt;
    }
    if (cnt < 2)
        return;
    unsigned m = cnt - 1, half = m >> 1, i = 1;
    if (half >= 1) {
        unsigned c;
        do {
            c = i * 2;
            __builtin_prefetch(arr + c * 6);  /* (the next level's two candidates) */
            __builtin_prefetch(arr + c * 6 + 6);
            if (c < m && !(arr[(c + 1) * 3] > arr[c * 3]))
                c++;
            if (l0 <= arr[c * 3])
                break;
            arr[i * 3] = arr[c * 3];
            arr[i * 3 + 1] = arr[c * 3 + 1];
            arr[i * 3 + 2] = arr[c * 3 + 2];
            i = c;
        } while (c <= half);
    }
    arr[i * 3] = l0;
    arr[i * 3 + 1] = l1;
    arr[i * 3 + 2] = l2;
}

/* ---- 524288: the long-range search's relaxation ---- */
static LRelaxFn LRelax_orig;

static void __fastcall LRelax_hook(BYTE *s, void *edx, int idx, int parent) {
    BYTE *nodes = *(BYTE **)(s + 0x5c);
    DWORD *n = (DWORD *)(nodes + idx * 0x24), *p = (DWORD *)(nodes + parent * 0x24);
    unsigned ax = uabs((int)p[0] - (int)n[0]) * 24u, ay = uabs((int)p[1] - (int)n[1]) * 24u;
    unsigned newg = isqrt_f(ay * ay + ax * ax) + p[5];
    g_lr_n++;
    if (n[3] == 0xffffffffu) {
        int dx = (int)n[0] - *(int *)(s + 0xbc), dy = (int)n[1] - *(int *)(s + 0xc0);
        unsigned d2 = (unsigned)dy * (unsigned)dy + (unsigned)dx * (unsigned)dx;
        if (d2 < *(DWORD *)(s + 0xcc)) {
            *(DWORD *)(s + 0xcc) = d2;
            *(int *)(s + 0xd0) = idx;
        }
    } else {
        if (newg >= n[5]) {
            g_lr_early++;
            return;
        }
        if (n[3] < 0xfffffffeu)
            ((UnlinkFn)(void *)VA_(VA_LUNLINK))(s, NULL, idx);
        else {
            n[2]++;
            n[3] = 0xffffffffu;
            n[4] = 0xffffffffu;
        }
    }
    unsigned hx = uabs((int)n[0] - *(int *)(s + 0xbc)) * 24u, hy = uabs((int)n[1] - *(int *)(s + 0xc0)) * 24u;
    unsigned h = isqrt_f(hy * hy + hx * hx);
    n[7] = (DWORD)parent;
    n[5] = newg;
    n[6] = h;
    ((BYTE *)n)[0x23] = 0;
    n[2]++;
    unsigned key = n[6] + n[5];
    n[3] = 0xfffffffeu;
    n[4] = 0xffffffffu;
    heap_push(s + 0x70, key, idx, n[2]);
}

/* ---- 1048576: the local A* search ---- */
static AstarFn Astar_orig;
static CellFn Cell_fn;
static NodeFn Node_fn;
static HeurFn Heur_fn;
static UnlinkFn Unlink_fn;
/* per cell, for the current run: [31:19] the run (13 bits), [18:3] its node + 1 (0: not known yet), [2] the ring query
 * remembered, [1] it met the target, [0] free */
static DWORD *g_m;
static unsigned g_mcap, g_mw, g_mh;
static DWORD g_mrun;
static int g_in;

/* the heuristic 0x8bda00 (k = 15), transcribed (checked against the code at install) */
static inline unsigned heur15(const DWORD *n, const BYTE *goal) {
    unsigned ax = uabs((int)n[0] - *(const int *)goal), ay = uabs((int)n[1] - *(const int *)(goal + 4));
    unsigned axk = ax * 15u, ayk = ay * 15u;
    unsigned mn = axk >= ayk ? ay * 15u : ax * 15u, mx = axk >= ayk ? ax * 15u : ay * 15u;
    if ((mx >> 2) >= mn)
        return mx;
    unsigned t = (mn >> 1) + mn;
    return (t >> 2) + (t >> 7) + mx + ((mx >> 6) - (mx >> 4));
}

/* a ring cell's query (0x8bdbe0), remembered for the run (W3SIM_CUTS & 64's argument: nothing in a run changes a
 * cell's answer; a node made in a cell links a type-2 entry the query skips) */
#define M_RUN(v) ((v) >> 19)
static inline int cellq(BYTE *s, int x, int y) {
    if ((unsigned)x < g_mw && (unsigned)y < g_mh) {
        unsigned i = (unsigned)y * g_mw + (unsigned)x;
        DWORD v = g_m[i];
        if (M_RUN(v) != g_mrun)
            v = g_mrun << 19;
        else if (v & 4) {
            if (v & 2)
                s[0xcc] = 1;
            if (!(v & 1))
                s[0xcd] = 1;
            g_as_hits++;
            return (int)(v & 1);
        }
        BYTE cc = s[0xcc];
        s[0xcc] = 0;
        int r = (BYTE)Cell_fn(s, NULL, x, y) != 0;
        BYTE met = s[0xcc];
        s[0xcc] = cc | met;
        g_m[i] = v | 4 | (r ? 1u : 0u) | (met ? 2u : 0u);
        g_as_miss++;
        return r;
    }
    return (BYTE)Cell_fn(s, NULL, x, y) != 0;
}

/* 0x8be080's walk: the top row left to right, the right column down, the bottom row right to left, the left column
 * up; bit k set if the k-th cell is blocked */
static inline unsigned ring(BYTE *s, int x, int y, int r, int n) {
    int x0 = x - r, y0 = y - r, i, j;
    unsigned mask = 0, bit = 1;
    for (i = x0; i < x0 + n; i++, bit += bit)
        if (!cellq(s, i, y0))
            mask |= bit;
    i--;
    for (j = y0 + 1; j < y0 + n; j++, bit += bit)
        if (!cellq(s, i, j))
            mask |= bit;
    i--;
    j--;
    for (; i >= x0; i--, bit += bit)
        if (!cellq(s, i, j))
            mask |= bit;
    j--;
    i++;
    for (; j >= y0 + 1; j--, bit += bit)
        if (!cellq(s, i, j))
            mask |= bit;
    return mask;
}

/* a cell's node (0x8bcf30): its type-2 entry of this run at the head of the cell's list; else the original
 * (it makes the node, or answers -1). Remembered per cell for the run: the head entry of a cell whose node this run
 * found or made stays that node's (only a node's making links or reuses a head entry, once per cell and run; the
 * search moves nothing) */
static inline int node_at(BYTE *s, BYTE *g, int x, int y) {
    if ((unsigned)x < g_mw && (unsigned)y < g_mh) {
        unsigned i = (unsigned)y * g_mw + (unsigned)x;
        DWORD v = g_m[i];
        if (M_RUN(v) != g_mrun)
            v = g_mrun << 19;
        else if (v & 0x7fff8)
            return (int)(((v >> 3) & 0xffff) - 1);
        int r = -1;
        DWORD c = (*(DWORD **)(g + 0x28))[i] & 0xffffff;
        if (c != 0xffffff) {
            DWORD *e = (DWORD *)(*(BYTE **)(g + 0x78) + c * 8);
            if ((e[0] & 0xff000000) == 0x2000000 && *(WORD *)(e + 1) == *(WORD *)(s + 0x20))
                r = (int)((WORD *)e)[3];
        }
        if (r < 0)
            r = Node_fn(s, NULL, x, y);
        if (r >= 0 && r < 0xffff)
            g_m[i] = v | ((DWORD)(r + 1) << 3);
        return r;
    }
    return Node_fn(s, NULL, x, y);
}

/* the relaxation 0x8bf980 (its heuristic after the early return) */
static inline void relax(BYTE *s, int idx, int parent, unsigned cost) {
    BYTE *nodes = *(BYTE **)(s + 0x30);
    DWORD *n = (DWORD *)(nodes + idx * 0x24);
    unsigned g = *(DWORD *)(nodes + parent * 0x24 + 0x14) + cost;
    if (n[3] == 0xffffffffu) {
        int dy = (int)n[1] - *(int *)(s + 0x8c), dx = (int)n[0] - *(int *)(s + 0x88);
        unsigned d2 = (unsigned)dy * (unsigned)dy + (unsigned)dx * (unsigned)dx;
        if (d2 < *(DWORD *)(s + 0x98)) {
            *(DWORD *)(s + 0x98) = d2;
            *(int *)(s + 0x9c) = idx;
        }
    } else {
        if (g >= n[5])
            return;
        if (n[3] < 0xfffffffeu)
            Unlink_fn(s, NULL, idx);
        else {
            n[2]++;
            n[3] = 0xffffffffu;
            n[4] = 0xffffffffu;
        }
    }
    unsigned h = heur15(n, s + 0x88);
    n[5] = g;
    n[7] = (DWORD)parent;
    n[6] = h;
    n[2]++;
    unsigned key = n[6] + n[5];
    n[3] = 0xfffffffeu;
    n[4] = 0xffffffffu;
    heap_push(s + 0x44, key, idx, n[2]);
}

/* the size classes' rings (radius, side) and the ring bits that block each of the eight neighbours, in 0x8c0890's
 * order: (-1,-1) (0,-1) (1,-1) (-1,0) (1,0) (-1,1) (0,1) (1,1); diagonal steps cost 0x15, straight ones 0xf */
static const struct {
    int r, n;
    unsigned m[8];
} g_sc[4] = {
    {1, 3, {0x83, 0x02, 0x0e, 0x80, 0x08, 0xe0, 0x20, 0x38}},
    {2, 4, {0xc07, 0x6, 0x3e, 0xc00, 0x30, 0xf80, 0x180, 0x1f0}},
    {2, 5, {0xe00f, 0xe, 0xfe, 0xe000, 0xe0, 0xfe00, 0xe00, 0xfe0}},
    {3, 6, {0xf001f, 0x1e, 0x3fe, 0xf0000, 0x3c0, 0xff800, 0x7800, 0x7fc0}},
};
static const int g_dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1}, g_dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
static const unsigned g_cost[8] = {0x15, 0xf, 0x15, 0xf, 0xf, 0x15, 0xf, 0x15};

static inline void expand(BYTE *s, BYTE *g, unsigned sc, int node) {
    BYTE *nodes = *(BYTE **)(s + 0x30);
    int x = *(int *)(nodes + node * 0x24), y = *(int *)(nodes + node * 0x24 + 4);
    unsigned mask = ring(s, x, y, g_sc[sc].r, g_sc[sc].n);
    if (sc == 0)
        mask &= 0xff; /* (0x8c09c0 tests the ring's byte) */
    int nb[8];
    for (int k = 0; k < 8; k++)
        nb[k] = (mask & g_sc[sc].m[k]) ? -1 : node_at(s, g, x + g_dx[k], y + g_dy[k]);
    if (s[0xcc])
        return;
    for (int k = 0; k < 8; k++)
        if (nb[k] != -1)
            relax(s, nb[k], node, g_cost[k]);
}

static int run(BYTE *s, BYTE *g, unsigned sc) {
    for (;;) {
        if (*(DWORD *)(s + 0x60) <= 1)
            return -1;
        DWORD e = *(DWORD *)(s + 0x6c);
        *(DWORD *)(s + 0x6c) = e + 1;
        if (e >= *(DWORD *)(s + 0x68))
            return -1;
        DWORD top[3];
        heap_pop(s + 0x44, top);
        g_as_pops++;
        int node = (int)top[1];
        DWORD *n = (DWORD *)(*(BYTE **)(s + 0x30) + node * 0x24);
        if (n[2] != top[2])
            continue;
        n[3] = 0xffffffffu;
        n[4] = 0xffffffffu;
        n[2]++;
        if (node == *(int *)(s + 0x94))
            return node;
        expand(s, g, sc, node);
        if (s[0xcc]) {
            s[0xcc] = 0;
            return node;
        }
        int last = *(int *)(s + 0x64);
        if (last != -3)
            *(DWORD *)(*(BYTE **)(s + 0x30) + last * 0x24 + 0x10) = (DWORD)node;
        n = (DWORD *)(*(BYTE **)(s + 0x30) + node * 0x24);
        n[2]++;
        n[3] = (DWORD)last;
        n[4] = 0xffffffffu;
        *(int *)(s + 0x64) = node;
    }
}

static int __fastcall Astar3_hook(BYTE *s) {
    unsigned sc = *(WORD *)(s + 0xa0);
    BYTE *g = *(BYTE **)(s + 0x1c);
    if (g_in || sc > 3 || !g)
        return Astar_orig(s);
    unsigned w = *(DWORD *)(g + 0x3c), h = *(DWORD *)(g + 0x40);
    if (!w || !h || w > 4096 || h > 4096)
        return Astar_orig(s);
    if (w * h > g_mcap) {
        free(g_m);
        g_m = calloc(w * h, sizeof *g_m);
        g_mcap = g_m ? w * h : 0;
        g_mrun = 0;
        if (!g_m)
            return Astar_orig(s);
    }
    g_mw = w;
    g_mh = h;
    if (++g_mrun >= 0x2000u) { /* (13 bits) */
        memset(g_m, 0, g_mcap * sizeof *g_m);
        g_mrun = 1;
    }
    g_as_runs++;
    g_in = 1;
    int r = run(s, g, sc);
    g_in = 0;
    return r;
}

/* ---- 4194304: the pathing map's cell update ---- */
typedef void(__fastcall *RCellFn)(BYTE *, void *, DWORD *, DWORD *, int, int, int);
static RCellFn RCell_orig;
static long g_rc_n;

/* 0x8be420: may a unit of the query's movement class (+0xa4) stand in sub-cell (x, y) */
static inline int region_q(BYTE *s, int x, int y) {
    BYTE *g = *(BYTE **)(s + 0x1c);
    unsigned w = *(DWORD *)(g + 0x3c);
    if ((unsigned)x >= w || (unsigned)y >= *(DWORD *)(g + 0x40))
        return 0;
    DWORD c = (*(DWORD **)(g + 0x28))[(unsigned)y * w + (unsigned)x];
    if ((c & *(DWORD *)(s + 0xa4)) & 0xff000000)
        return 0;
    DWORD e = c & 0xffffff;
    if (e == 0xffffff)
        return 1;
    DWORD st = ++*(DWORD *)(g + 0xb4);
    for (int cnt = 1; cnt < 0x32; cnt++) {
        DWORD *ent = (DWORD *)(*(BYTE **)(*(BYTE **)(s + 0x1c) + 0x78) + e * 8);
        DWORD wd = ent[0], kind = wd & 0xff000000;
        e = wd & 0xffffff;
        if (kind != 0x2000000) {
            BYTE *o = (BYTE *)(size_t)ent[1];
            if (*(DWORD *)(o + 0x38) != 0xffffffffu && ((*(DWORD *)(o + 0x34) >> 24) & 1) &&
                ((*(DWORD *)(o + 0x44) >> 28) & 1) && *(DWORD *)(o + 0x38) != st) {
                *(DWORD *)(o + 0x38) = st;
                if (kind == 0x1000000 && !(*(DWORD *)(o + 0x44) & 0x8fffffff) &&
                    ((*(DWORD *)(o + 0x34) & *(DWORD *)(s + 0xa4)) & 0xffffff))
                    return 0;
            }
        }
        if (e == 0xffffff)
            return 1;
    }
    return 1;
}

static void __fastcall RCell_hook(BYTE *self, void *edx, DWORD *cell, DWORD *maskp, int bit, int x, int y) {
    BYTE *s = *(BYTE **)(self + 0x24c);
    int blocked = 0;
    g_rc_n++;
    *(DWORD *)(s + 0xa4) = *maskp;
    blocked += !region_q(s, x, y);
    *(DWORD *)(s + 0xa4) = *maskp;
    blocked += !region_q(s, x + 1, y);
    *(DWORD *)(s + 0xa4) = *maskp;
    blocked += !region_q(s, x + 1, y + 1);
    *(DWORD *)(s + 0xa4) = *maskp;
    blocked += !region_q(s, x, y + 1);
    if (blocked) {
        DWORD v = blocked == 4 ? 0x40000000u : 0x80000000u;
        cell[1] = (cell[1] & ~(0xc0000000u >> (bit & 31))) | (v >> (bit & 31));
    }
}

static int hook_at(DWORD va, void *fn, void **orig, const char *name) {
    BYTE *p = VA_(va);
    if (!(p[0] == 0x55 && p[1] == 0x8b && p[2] == 0xec)) {
        shim_log("pathing: %s: unexpected prologue, not hooked", name);
        return 0;
    }
    if (MH_CreateHook(p, fn, orig) != MH_OK || MH_EnableHook(p) != MH_OK) {
        shim_log("pathing: %s: hook failed", name);
        return 0;
    }
    return 1;
}

/* the code as read: the calls and constants the C relies on */
static int check_call(DWORD at, DWORD target) {
    BYTE *p = VA_(at);
    return p[0] == 0xe8 && p + 5 + *(int32_t *)(p + 1) == VA_(target);
}

void pathing_install(long long cuts) {
    if (!(cuts & (524288 | 1048576 | 4194304)))
        return;
    if (MH_Initialize() != MH_OK && MH_Initialize() != MH_ERROR_ALREADY_INITIALIZED)
        return;
    /* the pure helpers against the executable's own: the square root and the heuristic */
    int bad = 0;
    for (unsigned k = 0, v = 7919; k < (1u << 17) && !bad; k++, v = v * 1664525u + 1013904223u)
        bad = ((IsqrtFn)(void *)VA_(VA_ISQRT2))(v >> (k & 15)) != isqrt_f(v >> (k & 15));
    for (unsigned k = 0, v = 104729; k < (1u << 17) && !bad; k++) {
        int a[2], b[2];
        v = v * 1664525u + 1013904223u;
        a[0] = (int)(v >> (k & 7)) - (int)(v >> 9);
        v = v * 1664525u + 1013904223u;
        a[1] = (int)(v >> (k & 7)) - (int)(v >> 9);
        b[0] = (int)(v & 0xfff) - 2048;
        b[1] = (int)((v >> 12) & 0xfff) - 2048;
        if (k & 1)
            a[0] = b[0] + (int)(k % 9) - 4, a[1] = b[1] + (int)(k % 7) - 3;
        DWORD n[2] = {(DWORD)a[0], (DWORD)a[1]};
        bad = ((HeurFn)(void *)VA_(VA_HEUR))(a, b, 15) != heur15(n, (BYTE *)b);
    }
    if (bad) {
        shim_log("pathing: a helper differs from the executable's in the self-test: not installed");
        return;
    }
    if ((cuts & 524288) && check_call(0x8a3d53, VA_ISQRT2) && check_call(0x8a3d90, VA_ISQRT2) &&
        check_call(0x8a3dba, VA_LUNLINK) && check_call(0x8a3e54, VA_PUSH)) {
        if (hook_at(VA_LRELAX, (void *)LRelax_hook, (void **)&LRelax_orig, "long-range relax"))
            shim_log("pathing: the long-range search's relaxation in C (W3SIM_CUTS & 524288)");
    } else if (cuts & 524288)
        shim_log("pathing: long-range relax: unexpected code, not hooked");
    if ((cuts & 1048576) && check_call(0x8bf8de, VA_POP) && check_call(0x8bf916, VA_EXPAND) &&
        check_call(0x8c08cc, 0x8c09c0) && check_call(0x8c08db, 0x8c0ae0) && check_call(0x8c08ea, 0x8c0c10) &&
        check_call(0x8c08f9, 0x8c0d40) && check_call(0x8c0919, 0x8bf980) && check_call(0x8c0af8, 0x8be080) &&
        check_call(0x8c0b12, VA_NODE) && check_call(0x8bcfc0, 0x8c22c0) && check_call(0x8bcfeb, 0x8a3480)) {
        Cell_fn = (CellFn)(void *)VA_(VA_CELL);
        Node_fn = (NodeFn)(void *)VA_(VA_NODE);
        Heur_fn = (HeurFn)(void *)VA_(VA_HEUR);
        Unlink_fn = (UnlinkFn)(void *)VA_(VA_AUNLINK);
        if (hook_at(VA_ASTAR, (void *)Astar3_hook, (void **)&Astar_orig, "A* loop"))
            shim_log("pathing: the local A* search's loop in C (W3SIM_CUTS & 1048576)");
    } else if (cuts & 1048576)
        shim_log("pathing: A* loop: unexpected code, not hooked");
    if ((cuts & 4194304) && check_call(0x8b78d7, 0x8be7a0) && check_call(0x8b7901, 0x8be7a0) &&
        check_call(0x8b7923, 0x8be7a0) && check_call(0x8b7945, 0x8be7a0) && check_call(0x8be7b6, 0x8be420) &&
        VA_(0x8b7981)[0] == 0xc2 && VA_(0x8b7981)[1] == 0x14) {
        if (hook_at(0x8b78b0, (void *)RCell_hook, (void **)&RCell_orig, "map cell update"))
            shim_log("pathing: the pathing map's cell update in C (W3SIM_CUTS & 4194304)");
    } else if (cuts & 4194304)
        shim_log("pathing: map cell update: unexpected code, not hooked");
}

void pathing_report(void) {
    shim_log("pathing/s: long relax=%ld early=%ld; A* runs=%ld pops=%ld cells hit=%ld miss=%ld; map cells=%ld", g_lr_n,
             g_lr_early, g_as_runs, g_as_pops, g_as_hits, g_as_miss, g_rc_n);
    g_lr_n = g_lr_early = g_as_runs = g_as_pops = g_as_hits = g_as_miss = g_rc_n = 0;
}
