/* W3SIM_CUTS (experiment, bits): game work that training does not need.
 *
 *   1  memoize the map-version lookup behind the Player / GetPlayerId natives (0x704330): it copies the
 *      cached 14 KB map info of the map path into a stack buffer to read one field (the 1.29 24-player
 *      compatibility check, version < 6060), on every call
 *   2  skip the frame's paint handler (0x959330, registered with Storm's event dispatcher) once the game
 *      steps: the world, UI and minimap are built for a device that draws nothing (W3SIM_DRAW=0)
 *   4  pool the events of the loopback connection's messages (0x7c1210 creates a manual-reset event per
 *      message, 0x7c1250 closes it): CreateEvent and CloseHandle are wineserver requests even with fsync
 *   8  textures without pixels once the game steps: the BLP decoder (0x83cdf0, thiscall
 *      decode_mips(format, &table, first_mip)) allocates the same mip table and buffers (0xcbecc0) but skips
 *      the JPEG decode of each level (0x83cd60); dimensions and mip counts stay as they were
 *
 *  16  (units.c) the harness's pass over the units: ForGroup(w3s_all, W3S_CPassEnum) without a JASS thread per unit
 *  32  (units.c) the getters pass_unit and unit_record both call, once per unit; GetPlayerId by player handle
 *  64  path search: the occupancy checks of a node's ring of cells (0x8be080 -> 0x8bdbe0) remembered per cell for
 *      the length of one A* run (0x8bf8b0), the cell's answer and its side effects (the search's "touched the
 *      target" and "blocked" flags) replayed
 * 256  paint (with drawing off) at most once every W3SIM_PAINT_EVERY steps (default 50) once the game steps, instead of
 *      ~8 times a wall second: what it builds is never shown. Unlike skipping it (2), the world frame's unit list
 *      that the per-turn fog pass walks is still rebuilt by it now and then (0x6125d0), so it stays short.
 * 512  Storm's heap choice: SMemAlloc hashes its caller's source file name and line (0x43fb00, a one-entry memo
 *      inside) on every allocation; for names in the executable's read-only data (string literals: their
 *      bytes never change) the hash is remembered per (name, line), and the one-entry memo's globals are left as
 *      the original's computation leaves them
 * 1024 A*: a neighbour's heuristic (0x8bda00, pure: the node's cell, the goal) computed after the relaxation's
 *      early return instead of before it (0x8bf980 in C; the same stores, in the same order)
 * 2048 the agent type check IsA(type, base) (0x7f55f0; the game memoizes it in a table of its own, 0x7f5630)
 *      answered from a small direct-mapped cache on the main thread for pairs it has answered before; forgotten
 *      at every game's first step and when the type registry changes
 * 128  soft float: count leading zeros (0x9a0330) as LZCNT; the integer square root (0x9a2150, 16 rounds of
 *      digit-by-digit) as floor(sqrt(double))
 *
 * Offsets are for the 1.29.2.9231 executable (RVAs; see ~/tools/w3analysis/report.md).
 */
#include "w3shim.h"

#include "MinHook.h"

#include <stdlib.h>
#include <string.h>

#define RVA_MAPVER 0x304330  /* int __cdecl map_version(const char *map_path) */
#define RVA_PAINT 0x559330   /* int __cdecl paint(int, int): the frame's paint handler, returns 1 */
#define RVA_EVT_INIT 0x3c1210 /* void __cdecl sync_init(int a, int b, OBJ *o): o->event = CreateEventA(0,1,0,0) */
#define RVA_EVT_FREE 0x3c1250 /* void __cdecl sync_free(OBJ *o): CloseHandle(o->event) */
#define RVA_BLP_DECODE 0x43cdf0 /* int __thiscall decode_mips(BLP *, int format, int *table, unsigned first) */
#define RVA_MIP_ALLOC 0x8becc0  /* int __cdecl mip_alloc(int format, unsigned w, unsigned h, int fill) */

static int g_cuts = -1;

int cuts_flags(void) {
    if (g_cuts < 0) {
        char buf[16];
        g_cuts = GetEnvironmentVariableA("W3SIM_CUTS", buf, sizeof buf) ? atoi(buf) : 0;
    }
    return g_cuts;
}
static volatile LONG g_mv_hits, g_mv_miss, g_paint_skipped, g_evt_reused, g_evt_new, g_tex_skipped;

/* 1: map version */
typedef int(__cdecl *MapVerFn)(const char *path);
static MapVerFn MapVer_orig;
static CRITICAL_SECTION g_mv_lock;
static char g_mv_path[MAX_PATH];
static int g_mv_val, g_mv_ok;

static int __cdecl MapVer_hook(const char *path) {
    if (!path)
        return MapVer_orig(path); /* (the current map: read directly, cheap) */
    EnterCriticalSection(&g_mv_lock);
    if (g_mv_ok && strncmp(path, g_mv_path, sizeof g_mv_path) == 0) {
        int v = g_mv_val;
        LeaveCriticalSection(&g_mv_lock);
        InterlockedIncrement(&g_mv_hits);
        return v;
    }
    LeaveCriticalSection(&g_mv_lock);
    int v = MapVer_orig(path);
    InterlockedIncrement(&g_mv_miss);
    if (v != -1 && strlen(path) < sizeof g_mv_path) {
        EnterCriticalSection(&g_mv_lock);
        strcpy(g_mv_path, path);
        g_mv_val = v;
        g_mv_ok = 1;
        LeaveCriticalSection(&g_mv_lock);
    }
    return v;
}

/* 2: paint; 256: paint now and then */
typedef int(__cdecl *PaintFn)(int, int);
static PaintFn Paint_orig;
static long g_paint_every = 50, g_paint_last = -1000000;

static int __cdecl Paint_hook(int a, int b) {
    long n = sync_count();
    if (n > 0) { /* (the loading screen and the first frame are drawn as before) */
        if (!(g_cuts & 256) || n - g_paint_last < g_paint_every) {
            InterlockedIncrement(&g_paint_skipped);
            return 1;
        }
        g_paint_last = n;
    }
    return Paint_orig(a, b);
}

/* 4: events */
typedef void(__cdecl *EvtInitFn)(int, int, DWORD *);
typedef void(__cdecl *EvtFreeFn)(DWORD *);
static EvtInitFn EvtInit_orig;
static EvtFreeFn EvtFree_orig;
#define POOL 256
static HANDLE g_pool[POOL];
static int g_npool;
static CRITICAL_SECTION g_pool_lock;

static void __cdecl EvtInit_hook(int a, int b, DWORD *o) {
    HANDLE h = NULL;
    EnterCriticalSection(&g_pool_lock);
    if (g_npool > 0)
        h = g_pool[--g_npool];
    LeaveCriticalSection(&g_pool_lock);
    if (!h) {
        EvtInit_orig(a, b, o);
        InterlockedIncrement(&g_evt_new);
        return;
    }
    ResetEvent(h); /* as created: manual reset, not signalled */
    o[0] = o[1] = o[2] = o[3] = 0;
    o[2] = (DWORD)a;
    o[3] = (DWORD)b;
    o[4] = (DWORD)(size_t)h;
    InterlockedIncrement(&g_evt_reused);
}

static void __cdecl EvtFree_hook(DWORD *o) {
    HANDLE h = (HANDLE)(size_t)o[4];
    if (h) {
        EnterCriticalSection(&g_pool_lock);
        if (g_npool < POOL) {
            g_pool[g_npool++] = h;
            h = NULL;
        }
        LeaveCriticalSection(&g_pool_lock);
    }
    if (h) {
        EvtFree_orig(o);
        return;
    }
    o[4] = 0;
}

/* 8: textures */
typedef int(__fastcall *BlpDecodeFn)(BYTE *self, void *edx, int format, int *table, unsigned first);
typedef int(__cdecl *MipAllocFn)(int format, unsigned w, unsigned h, int fill);
static BlpDecodeFn BlpDecode_orig;

static int __fastcall BlpDecode_hook(BYTE *self, void *edx, int format, int *table, unsigned first) {
    if (sync_count() <= 0)
        return BlpDecode_orig(self, edx, format, table, first);
    unsigned nmips = *(unsigned *)(self + 0x4b8);
    if (!((*(int *)(self + 0x20) != 0 || first == 0) && first < nmips))
        return 0;
    unsigned w = *(unsigned *)(self + 0x14) >> (first & 31), h = *(unsigned *)(self + 0x18) >> (first & 31);
    int t = ((MipAllocFn)(g_base + RVA_MIP_ALLOC))(format, w > 1 ? w : 1, h > 1 ? h : 1, 0);
    *table = t;
    InterlockedIncrement(&g_tex_skipped);
    return t != 0;
}

static int hook(DWORD rva, void *fn, void **orig, const char *name);

/* 64: the ring checks of an A* run (see the top). Within one call of the search loop 0x8bf8b0 nothing changes
 * what a cell query answers: its callees pop and push the open list, relax nodes, and map cells to node indices
 * (0x8bcf30 -> 0x8c22c0 links a type-2 entry at the head of the cell's list, which the query skips; the cell's
 * terrain byte is kept). The query 0x8bdbe0 (thiscall(search, x, y), AL: 1 free, 0 blocked) reads the cell's
 * bits and its list of pathing objects with the search's mask (+0xa4), target (+0xa8) and flag (+0xce), sets
 * +0xcc when it meets the target and +0xcd when it answers blocked; its object stamps (+0x38, a generation
 * counter incremented by every such query before it is compared) only keep one query from counting an object
 * twice. */
#define VA_(x) (g_base + ((x) - 0x400000))
#define VA_RING 0x8be080  /* thiscall unsigned ring(search, x, y, r, n), ret 0x10: bit k = k-th cell of the ring blocked */
#define VA_CELL 0x8bdbe0  /* thiscall char cell(search, x, y), ret 8 */
#define VA_ASTAR 0x8bf8b0 /* fastcall int run(search) */
typedef unsigned(__fastcall *RingFn)(BYTE *, void *, int, int, int, int);
typedef unsigned(__fastcall *CellFn)(BYTE *, void *, int, int);
typedef int(__fastcall *AstarFn)(BYTE *);
static RingFn Ring_orig;
static AstarFn Astar_orig;
static CellFn Cell_fn;
static BYTE *g_srch, *g_smap;
static DWORD g_sgen, g_smask, g_starget;
static BYTE g_sce;
static DWORD *g_stamp;
static BYTE *g_sval;
static unsigned g_sw, g_sh, g_scap;
static volatile LONG g_ring_calls, g_ring_orig, g_cell_hits, g_cell_miss;

static int __fastcall Astar_hook(BYTE *s) {
    BYTE *map = *(BYTE **)(s + 0x1c);
    BYTE *prev = g_srch;
    if (prev || !map) /* (nested: not remembered) */
        return Astar_orig(s);
    unsigned w = *(DWORD *)(map + 0x3c), h = *(DWORD *)(map + 0x40);
    if (w && h && w <= 4096 && h <= 4096) {
        if (w * h > g_scap) {
            free(g_stamp);
            free(g_sval);
            g_scap = w * h;
            g_stamp = calloc(g_scap, sizeof *g_stamp);
            g_sval = malloc(g_scap);
            if (!g_stamp || !g_sval) {
                g_scap = 0;
                return Astar_orig(s);
            }
        }
        if (++g_sgen == 0) {
            memset(g_stamp, 0, g_scap * sizeof *g_stamp);
            g_sgen = 1;
        }
        g_sw = w;
        g_sh = h;
        g_smap = map;
        g_smask = *(DWORD *)(s + 0xa4);
        g_starget = *(DWORD *)(s + 0xa8);
        g_sce = s[0xce];
        g_srch = s;
    }
    int r = Astar_orig(s);
    g_srch = NULL;
    return r;
}

static inline int cell_q(BYTE *s, int x, int y) {
    if ((unsigned)x < g_sw && (unsigned)y < g_sh) {
        unsigned i = (unsigned)y * g_sw + (unsigned)x;
        if (g_stamp[i] == g_sgen) {
            BYTE v = g_sval[i];
            if (v & 2)
                s[0xcc] = 1;
            if (!(v & 1))
                s[0xcd] = 1;
            g_cell_hits++;
            return v & 1;
        }
        BYTE cc = s[0xcc];
        s[0xcc] = 0;
        int r = (BYTE)Cell_fn(s, NULL, x, y) != 0;
        BYTE met = s[0xcc];
        s[0xcc] = cc | met;
        g_stamp[i] = g_sgen;
        g_sval[i] = (BYTE)((r ? 1 : 0) | (met ? 2 : 0));
        g_cell_miss++;
        return r;
    }
    return (BYTE)Cell_fn(s, NULL, x, y) != 0;
}

static unsigned __fastcall Ring_hook(BYTE *s, void *edx, int x, int y, int r, int n) {
    g_ring_calls++;
    if (s != g_srch || *(BYTE **)(s + 0x1c) != g_smap || *(DWORD *)(s + 0xa4) != g_smask ||
        *(DWORD *)(s + 0xa8) != g_starget || s[0xce] != g_sce) {
        g_ring_orig++;
        return Ring_orig(s, edx, x, y, r, n);
    }
    /* 0x8be080's walk: the top row left to right, the right column down, the bottom row right to left, the left
     * column up; bit k for the k-th cell */
    int x0 = x - r, y0 = y - r, i, j;
    unsigned mask = 0, bit = 1;
    for (i = x0; i < x0 + n; i++, bit += bit)
        if (!cell_q(s, i, y0))
            mask |= bit;
    i--;
    for (j = y0 + 1; j < y0 + n; j++, bit += bit)
        if (!cell_q(s, i, j))
            mask |= bit;
    i--;
    j--;
    for (; i >= x0; i--, bit += bit)
        if (!cell_q(s, i, j))
            mask |= bit;
    j--;
    i++;
    for (; j >= y0 + 1; j--, bit += bit)
        if (!cell_q(s, i, j))
            mask |= bit;
    return mask;
}


/* 512: SMemAlloc's file-name hash */
#define VA_FHASH 0x43fb00 /* cdecl unsigned hash(const char *file, unsigned line) */
#define G_FH_ON 0x10db538
#define G_FH_FIRST 0x1131350
#define G_FH_LINE 0x1131354
#define G_FH_PTR 0x1131358
#define G_FH_VAL 0x113135c
typedef unsigned(__cdecl *FHashFn)(const char *, unsigned);
static FHashFn FHash_orig;
static DWORD g_ro_lo[16], g_ro_hi[16];
static int g_nro;
static CRITICAL_SECTION g_fh_lock;
#define FH_SLOTS 1024
static struct {
    const char *p;
    unsigned line, v;
} g_fh[FH_SLOTS];
static volatile LONG g_fh_hits, g_fh_miss;

static int read_only(const void *p) {
    DWORD a = (DWORD)(size_t)p;
    for (int i = 0; i < g_nro; i++)
        if (a >= g_ro_lo[i] && a + 8 <= g_ro_hi[i])
            return 1;
    return 0;
}

static unsigned __cdecl FHash_hook(const char *file, unsigned line) {
    if (!file || !read_only(file))
        return FHash_orig(file, line);
    unsigned k = (((DWORD)(size_t)file >> 2) ^ (line * 2654435761u)) & (FH_SLOTS - 1);
    EnterCriticalSection(&g_fh_lock);
    if (g_fh[k].p == file && g_fh[k].line == line) {
        unsigned v = g_fh[k].v;
        /* (what the original's computation leaves in its one-entry memo) */
        *(DWORD *)VA_(G_FH_VAL) = v;
        *(DWORD *)VA_(G_FH_LINE) = line;
        *(DWORD *)VA_(G_FH_PTR) = (DWORD)(size_t)file;
        *(DWORD *)VA_(G_FH_FIRST) = *(const DWORD *)file;
        LeaveCriticalSection(&g_fh_lock);
        g_fh_hits++;
        return v;
    }
    unsigned v = FHash_orig(file, line);
    g_fh[k].p = file;
    g_fh[k].line = line;
    g_fh[k].v = v;
    LeaveCriticalSection(&g_fh_lock);
    g_fh_miss++;
    return v;
}

static void install_fhash(void) {
    IMAGE_NT_HEADERS *nt = (IMAGE_NT_HEADERS *)(g_base + ((IMAGE_DOS_HEADER *)g_base)->e_lfanew);
    IMAGE_SECTION_HEADER *sec = IMAGE_FIRST_SECTION(nt);
    for (int i = 0; i < nt->FileHeader.NumberOfSections && g_nro < 16; i++)
        if (!(sec[i].Characteristics & (IMAGE_SCN_MEM_WRITE | IMAGE_SCN_MEM_EXECUTE))) {
            g_ro_lo[g_nro] = (DWORD)(size_t)g_base + sec[i].VirtualAddress;
            g_ro_hi[g_nro++] = (DWORD)(size_t)g_base + sec[i].VirtualAddress + sec[i].Misc.VirtualSize;
        }
    InitializeCriticalSection(&g_fh_lock);
    if (g_nro && hook(VA_FHASH - 0x400000, (void *)FHash_hook, (void **)&FHash_orig, "SMemAlloc file hash"))
        shim_log("cuts: SMemAlloc file hash remembered (%d read-only sections)", g_nro);
}

/* 1024: A* relaxation with the heuristic after the early return */
#define VA_RELAX 0x8bf980 /* thiscall void relax(search, node, parent, cost), ret 0xc */
#define VA_HEUR 0x8bda00  /* cdecl unsigned h(int *xy, int *goal, int k) */
#define VA_UNLINK 0x8bfbc0 /* thiscall void unlink(search, node), ret 4 */
#define VA_PUSH 0x8a3090  /* thiscall void push(heap, key, node, gen), ret 0xc */
typedef void(__fastcall *RelaxFn)(BYTE *, void *, int, int, int);
typedef unsigned(__cdecl *HeurFn)(int *, int *, int);
typedef void(__fastcall *UnlinkFn)(BYTE *, void *, int);
typedef void(__fastcall *PushFn)(BYTE *, void *, unsigned, int, unsigned);
static RelaxFn Relax_orig;
static volatile LONG g_relax_n, g_relax_early;

static void __fastcall Relax_hook(BYTE *s, void *edx, int idx, int parent, int cost) {
    DWORD *n = (DWORD *)(*(BYTE **)(s + 0x30) + idx * 0x24);
    unsigned g = *(DWORD *)(*(BYTE **)(s + 0x30) + parent * 0x24 + 0x14) + (unsigned)cost;
    g_relax_n++;
    if (n[3] == 0xffffffffu) {
        int dy = (int)n[1] - *(int *)(s + 0x8c), dx = (int)n[0] - *(int *)(s + 0x88);
        unsigned d2 = (unsigned)dy * (unsigned)dy + (unsigned)dx * (unsigned)dx;
        if (d2 < *(DWORD *)(s + 0x98)) {
            *(DWORD *)(s + 0x98) = d2;
            *(int *)(s + 0x9c) = idx;
        }
    } else {
        if (g >= n[5]) {
            g_relax_early++;
            return;
        }
        if (n[3] < 0xfffffffeu)
            ((UnlinkFn)(void *)VA_(VA_UNLINK))(s, NULL, idx);
        else {
            n[2]++;
            n[3] = 0xffffffffu;
            n[4] = 0xffffffffu;
        }
    }
    unsigned h = ((HeurFn)(void *)VA_(VA_HEUR))((int *)n, (int *)(s + 0x88), 0xf);
    n[5] = g;
    n[7] = (DWORD)parent;
    n[6] = h;
    DWORD *m = (DWORD *)(*(BYTE **)(s + 0x30) + idx * 0x24);
    unsigned key = m[6] + m[5];
    m[2]++;
    unsigned gen = m[2];
    m[3] = 0xfffffffeu;
    m[4] = 0xffffffffu;
    ((PushFn)(void *)VA_(VA_PUSH))(s + 0x44, NULL, key, idx, gen);
}

/* 2048: IsA */
#define VA_ISA 0x7f55f0  /* cdecl int isa(type, base) */
#define G_REGISTRY 0x115e9ec
typedef int(__cdecl *IsaFn)(int, int);
static IsaFn Isa_orig;
#define ISA_SLOTS 2048
static struct {
    int a, b, r;
} g_isa[ISA_SLOTS];
static DWORD g_isa_reg;
static volatile LONG g_isa_hits, g_isa_miss, g_isa_gen;
static LONG g_isa_seen_gen;

void cuts_game_start(void) {
    InterlockedIncrement(&g_isa_gen);
}

static int __cdecl Isa_hook(int a, int b) {
    DWORD reg = *(DWORD *)VA_(G_REGISTRY);
    if (b == 0x2b616761 || a == b || !reg || GetCurrentThreadId() != g_step_tid)
        return Isa_orig(a, b);
    if (reg != g_isa_reg || g_isa_seen_gen != g_isa_gen) {
        memset(g_isa, 0, sizeof g_isa);
        g_isa_reg = reg;
        g_isa_seen_gen = g_isa_gen;
    }
    unsigned k = ((unsigned)a * 2654435761u ^ (unsigned)b * 40503u) >> 21; /* (11 bits) */
    if (g_isa[k].a == a && g_isa[k].b == b && (a || b)) {
        g_isa_hits++;
        return g_isa[k].r;
    }
    int r = Isa_orig(a, b);
    g_isa[k].a = a;
    g_isa[k].b = b;
    g_isa[k].r = r;
    g_isa_miss++;
    return r;
}

/* 128: soft float helpers */
#define VA_CLZ 0x9a0330   /* fastcall int clz(unsigned): bsr-based, 32 for 0 */
#define VA_ISQRT 0x9a2150 /* cdecl unsigned isqrt(unsigned): floor(sqrt(x)), 16 rounds */
typedef unsigned(__cdecl *IsqrtFn)(unsigned);
static IsqrtFn Isqrt_orig;

static unsigned __cdecl isqrt_orig_c(unsigned x) { /* (the original, transcribed: for the self-test) */
    unsigned rem = 0, root = 0;
    for (int k = 0; k < 16; k++) {
        rem = (rem << 2) | (x >> 30);
        x <<= 2;
        root += root;
        if (root < rem) {
            rem += ~root;
            root += 2;
        }
    }
    return root >> 1;
}

static unsigned __cdecl Isqrt_hook(unsigned x) {
    return (unsigned)__builtin_sqrt((double)x);
}

static int has_lzcnt(void) {
    unsigned a, b, c, d;
    __asm__ volatile("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(0x80000001), "c"(0));
    return (c >> 5) & 1;
}

static void install_softfloat(void) {
    BYTE *clz = VA_(VA_CLZ), *sq = VA_(VA_ISQRT);
    static const BYTE clz_code[] = {0x55, 0x8b, 0xec, 0x51, 0x89, 0x4d, 0xfc, 0x0f, 0xbd, 0x55, 0xfc};
    if (memcmp(clz, clz_code, sizeof clz_code) == 0 && has_lzcnt()) {
        /* in-process check against the original on edge cases and a sweep */
        typedef int(__fastcall *ClzFn)(unsigned);
        static const BYTE lz[] = {0xf3, 0x0f, 0xbd, 0xc1, 0xc3}; /* lzcnt eax, ecx; ret */
        BYTE *buf = VirtualAlloc(NULL, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
        int bad = 0;
        if (buf) {
            memcpy(buf, lz, sizeof lz);
            for (unsigned k = 0, v = 0; k < 1u << 20 && !bad; k++, v = v * 1664525u + 1013904223u) {
                unsigned t = k < 64 ? (k < 33 ? (k == 32 ? 0 : 1u << k) : ~0u >> (k - 33)) : v >> (k & 31);
                bad = ((ClzFn)(void *)clz)(t) != ((ClzFn)(void *)buf)(t);
            }
            VirtualFree(buf, 0, MEM_RELEASE);
        }
        DWORD old;
        if (!bad && VirtualProtect(clz, sizeof lz, PAGE_EXECUTE_READWRITE, &old)) {
            memcpy(clz, lz, sizeof lz);
            VirtualProtect(clz, sizeof lz, old, &old);
            FlushInstructionCache(GetCurrentProcess(), clz, sizeof lz);
            shim_log("cuts: clz as lzcnt");
        } else
            shim_log("cuts: clz NOT patched (bad=%d)", bad);
    } else
        shim_log("cuts: clz: unexpected code or no lzcnt, not patched");
    if (sq[0] == 0x55 && sq[1] == 0x8b && sq[2] == 0xec && sq[3] == 0x53) {
        /* the transcription against the code, then the replacement against the transcription */
        int bad = 0;
        for (unsigned k = 0, v = 12345; k < 1u << 18 && !bad; k++, v = v * 1664525u + 1013904223u) {
            unsigned t = k < 4096 ? k * k + (k & 1 ? 0u : ~0u) : v;
            unsigned o = ((IsqrtFn)(void *)sq)(t);
            bad = o != isqrt_orig_c(t) || o != Isqrt_hook(t);
        }
        if (!bad && hook(VA_ISQRT - 0x400000, (void *)Isqrt_hook, (void **)&Isqrt_orig, "isqrt"))
            shim_log("cuts: isqrt as floor(sqrt(double))");
        else if (bad)
            shim_log("cuts: isqrt: MISMATCH in the self-test, not hooked");
    }
}

static int hook(DWORD rva, void *fn, void **orig, const char *name) {
    BYTE *p = g_base + rva;
    if (!(p[0] == 0x55 && p[1] == 0x8b && p[2] == 0xec)) {
        shim_log("cuts: %s: unexpected prologue, not hooked", name);
        return 0;
    }
    if (MH_CreateHook(p, fn, orig) != MH_OK || MH_EnableHook(p) != MH_OK) {
        shim_log("cuts: %s: hook failed", name);
        return 0;
    }
    return 1;
}

void cuts_install(int draw) {
    if (!cuts_flags())
        return;
    if (MH_Initialize() != MH_OK && MH_Initialize() != MH_ERROR_ALREADY_INITIALIZED)
        return;
    InitializeCriticalSection(&g_mv_lock);
    InitializeCriticalSection(&g_pool_lock);
    if (g_cuts & 1)
        hook(RVA_MAPVER, (void *)MapVer_hook, (void **)&MapVer_orig, "map version");
    if ((g_cuts & (2 | 256)) && !draw) {
        char buf[16];
        if (GetEnvironmentVariableA("W3SIM_PAINT_EVERY", buf, sizeof buf))
            g_paint_every = atoi(buf);
        hook(RVA_PAINT, (void *)Paint_hook, (void **)&Paint_orig, "paint");
    }
    if (g_cuts & 4) {
        hook(RVA_EVT_INIT, (void *)EvtInit_hook, (void **)&EvtInit_orig, "event init");
        hook(RVA_EVT_FREE, (void *)EvtFree_hook, (void **)&EvtFree_orig, "event free");
    }
    if ((g_cuts & 8) && !draw)
        hook(RVA_BLP_DECODE, (void *)BlpDecode_hook, (void **)&BlpDecode_orig, "texture decode");
    if (g_cuts & 64) {
        Cell_fn = (CellFn)(void *)VA_(VA_CELL);
        if (hook(VA_ASTAR - 0x400000, (void *)Astar_hook, (void **)&Astar_orig, "A* run"))
            hook(VA_RING - 0x400000, (void *)Ring_hook, (void **)&Ring_orig, "A* ring");
    }
    if (g_cuts & 128)
        install_softfloat();
    if (g_cuts & 512)
        install_fhash();
    if (g_cuts & 1024)
        hook(VA_RELAX - 0x400000, (void *)Relax_hook, (void **)&Relax_orig, "A* relax");
    if (g_cuts & 2048)
        hook(VA_ISA - 0x400000, (void *)Isa_hook, (void **)&Isa_orig, "IsA");
    shim_log("cuts installed: %d", g_cuts);
}

void cuts_report(void) {
    if (g_cuts)
        shim_log("cuts/s: mapver hits=%ld miss=%ld paint skipped=%ld events reused=%ld new=%ld textures=%ld "
                 "rings=%ld (orig %ld) cells hit=%ld miss=%ld fhash hit=%ld miss=%ld relax=%ld early=%ld isa hit=%ld miss=%ld",
                 InterlockedExchange(&g_mv_hits, 0), InterlockedExchange(&g_mv_miss, 0),
                 InterlockedExchange(&g_paint_skipped, 0), InterlockedExchange(&g_evt_reused, 0),
                 InterlockedExchange(&g_evt_new, 0), InterlockedExchange(&g_tex_skipped, 0),
                 InterlockedExchange(&g_ring_calls, 0), InterlockedExchange(&g_ring_orig, 0),
                 InterlockedExchange(&g_cell_hits, 0), InterlockedExchange(&g_cell_miss, 0),
                 InterlockedExchange(&g_fh_hits, 0), InterlockedExchange(&g_fh_miss, 0),
                 InterlockedExchange(&g_relax_n, 0), InterlockedExchange(&g_relax_early, 0),
                 InterlockedExchange(&g_isa_hits, 0), InterlockedExchange(&g_isa_miss, 0));
}
