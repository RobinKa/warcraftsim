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
 * 4096 the path search's other integer square root (0x99b740: Newton's method with two divisions a round, called
 *      twice per node relaxed by the long-range search 0x8a3d00) as floor(sqrt(double)): the executable's own
 *      code gives floor(sqrt(x)) for every one of the 2^32 inputs (run natively, ~/tools/wt-cuts2-out/isqrt2),
 *      and a self-test against it in the process before hooking
 * 8192 a unit's ability by id (0x6d6570, thiscall find(unit, id, isa, by_base, a, b); UnitIsSleeping asks it for
 *      'Asla' for every unit every step): the same walk of the unit's ability list in C, the agent lookups of the
 *      list's links inlined (0x7fc7e0 -> 0x4707f0: reads), the ability's type still asked of its virtual getter, and
 *      on the main thread IsA(type, id) remembered per (type, id) for the game (the engine memoizes it too, in its
 *      type registry: the remembered answer is the one its first call gave; forgotten at every game's first step
 *      and when the world changes)
 * 16384 the widget grid's walk over a query's cells (0x8092a0 -> 0x809170 per cell -> 0x809030 per entry; every
 *      area enumeration, e.g. the harness's EnumItemsInRect over the whole map every step) in C: the same stores
 *      in the same order (the grid's stamp counter, each entry's stamp, the query's results), without a call per
 *      cell and per entry
 * 32768 (units.c) a unit's handle turned into its object once per unit of the harness's pass: the natives' conversions
 *      (0x483940 to a unit, 0x483ba0 to a widget: the JASS state, the handle table, a type check, the agent's
 *      liveness, a reference taken and given back, two exception frames) remembered from the first native of the
 *      unit to its last, the same pointer the conversion gives (nothing between them changes the handle or its
 *      object; forgotten when the unit is dropped from the group)
 * 65536 Wine-level: an observation and its header in one send (sync.c; each send is a few wineserver requests), and the
 *      calling thread's own priority remembered (GetThreadPriority(GetCurrentThread()) around the game's timing code,
 *      a wineserver request each time; the remembered value is the one the last SetThreadPriority through the
 *      executable's imports set, or the first query's)
 * 131072 Wine-level (sync.c): the controller's TCP connection made and used with Linux system calls instead of Winsock (no
 *      wineserver requests in the step sync)
 * 524288, 1048576, 4194304: pathing.c (the long-range search's relaxation, the local A* search's loop, the pathing map's
 *      cell update)
 * 16777216 (with 8192) a unit without an ability of the sleep's type remembered while its ability list's head is the same
 *      (CUnit::AddAbility 0x6d4560 puts a new ability at the head)
 * 2097152 (with 128) the soft float's integer square root (0x9a2150) as the inline sqrtsd of 4096 instead of the libm call
 *      (the same floor(sqrt(x)) for every input)
 * 33554432 (sync.c, with 131072) the step sync over the controller's Unix socket (W3SIM_SOCK) instead of TCP loopback
 * 262144 (units.c, with the harness's W3S_CSerInit handshake) the harness's players' records and its check of the game's
 *      result made by the shim with the same natives (W3SIM_COBS_CHECK=1: the harness does both as well and the shim
 *      compares); needs the harness from this change (older maps never ask)
 *
 * Round 4 (W3SIM_CUTS is parsed as 64 bits from here on):
 * 67108864, 134217728, 268435456, 536870912 (softfloat.c) the soft float's multiply, float-to-int truncation, floor and
 *      int-to-float as SSE instructions where each is provably the IEEE operation (the rest by the original)
 * 1073741824 (units.c, obs.c, sync.c; with the harness's W3S_CSerInit handshake) W3S_Tok's integers written by the shim
 *      (GetPlayerTechMaxAllowed(the token player, v): no I2S string, no Preload a token), and the "V", "K" and "T"
 *      records made by it (the camera's target with the same natives; the step, the time and the flags from the harness)
 * 2147483648 (units.c, with the handshake) the agents' commands applied by the shim, W3S_ApplyOne's natives in its order
 *      (ops 1, 2, 3, 5, 6, 87, 90; a step with any other op, or with a pending purchase or an advised player, the
 *      harness's as before), the "C" records of their results written by it
 * 4294967296 the name hash 0x433890 (every JASS identifier the VM resolves; file names too) in one pass, the same value
 *      (checked against the executable's own at install)
 * 8589934592 (units.c, with 1073741824 and the handshake) the items on the ground: EnumItemsInRect's per-item callback
 *      runs the shim's W3S_SerItem (the same natives in its order) instead of a JASS thread per item
 *
 * Round 5 (units.c, the units pass; each with 16, none needs a harness change):
 * 17179869184 the enum unit's handle remembered: ForGroup's callback skips its push and pop of a unit whose handle is
 *      live (the handle table's entry still holds the object), pushing only before a unit the pass drops
 * 34359738368 the unit's record from its object: the natives' reads and engine calls made directly on the callback's
 *      object (no handle conversions, no wrappers; x and y from one position call)
 * 68719476736 a hero's experience, skill points and items read once a record (they were read for the hash and again for
 *      the tokens)
 * 137438953472 the pass prefetches the units ahead of it in ForGroup's snapshot (0x4f5b20): their objects, agent table
 *      entries, agents, handle entries, sleep cache slots (hints only)
 * 274877906944 (with 34359738368) the owner's handle remembered per player object (GetOwningPlayer's handle map lookup)
 * 549755813888 (with 34359738368) the visibility bits of all seers at once: the game's option and the unit's cell asked
 *      once a unit, not once a seer
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

static long long g_cuts = -1;

long long cuts_flags(void) { /* (64 bits: bits from 2^31 on are round 4's) */
    if (g_cuts < 0) {
        char buf[32];
        g_cuts = GetEnvironmentVariableA("W3SIM_CUTS", buf, sizeof buf) ? _strtoi64(buf, NULL, 10) : 0;
        if (g_cuts < 0)
            g_cuts = 0;
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

void cuts_sleep_reset(void);
void cuts_game_start(void) {
    InterlockedIncrement(&g_isa_gen);
    cuts_sleep_reset();
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

static unsigned __cdecl Isqrt_fast(unsigned x) { /* (2097152: 0x9a2150 too as the inline sqrtsd; the same values) */
    double d = (double)(int)(x >> 1) * 2.0 + (double)(x & 1), r;
    __asm__("sqrtsd %1, %0" : "=x"(r) : "x"(d));
    return (unsigned)(int)r;
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
            bad = o != isqrt_orig_c(t) || o != Isqrt_hook(t) || o != Isqrt_fast(t);
        }
        if (!bad && hook(VA_ISQRT - 0x400000, (void *)((g_cuts & 2097152) ? Isqrt_fast : Isqrt_hook), (void **)&Isqrt_orig, "isqrt"))
            shim_log("cuts: isqrt as floor(sqrt(double))");
        else if (bad)
            shim_log("cuts: isqrt: MISMATCH in the self-test, not hooked");
    }
}

/* 4096: the Newton square root */
#define VA_ISQRT2 0x99b740 /* cdecl unsigned isqrt(unsigned) */
static IsqrtFn Isqrt2_orig;

static unsigned __cdecl Isqrt2_hook(unsigned x) {
    double d = (double)(int)(x >> 1) * 2.0 + (double)(x & 1), r; /* (exact: x < 2^32) */
    __asm__("sqrtsd %1, %0" : "=x"(r) : "x"(d));
    return (unsigned)(int)r; /* (r <= 65536: truncated) */
}

static void install_isqrt2(void) {
    BYTE *sq = VA_(VA_ISQRT2);
    static const BYTE code[] = {0x55, 0x8b, 0xec, 0x53, 0x8b, 0x5d, 0x08, 0x56, 0x57, 0x81, 0xfb, 0xff, 0x00, 0x00, 0x00};
    if (memcmp(sq, code, sizeof code) != 0) {
        shim_log("cuts: isqrt2: unexpected code, not hooked");
        return;
    }
    /* the replacement against the code: edge cases, every k*k - 1, k*k and k*k + 1, a sweep */
    int bad = 0;
    for (unsigned k = 0, v = 99991; k < (1u << 18) + 65536 * 3 && !bad; k++, v = v * 1664525u + 1013904223u) {
        unsigned t;
        if (k < 65536 * 3)
            t = (k / 3) * (k / 3) + (k % 3) - 1; /* (k = 0: 0xffffffff) */
        else if (k < 65536 * 3 + 4096)
            t = k - 65536 * 3;
        else
            t = v;
        bad = ((IsqrtFn)(void *)sq)(t) != Isqrt2_hook(t);
    }
    if (!bad && hook(VA_ISQRT2 - 0x400000, (void *)Isqrt2_hook, (void **)&Isqrt2_orig, "isqrt2"))
        shim_log("cuts: isqrt2 (0x99b740) as floor(sqrt(double))");
    else if (bad)
        shim_log("cuts: isqrt2: MISMATCH in the self-test, not hooked");
}

/* 8192: a unit's ability by id */
#define VA_FINDABIL 0x6d6570  /* thiscall BYTE *find(unit, id, isa, by_base, a, b), ret 0x14 */
#define G_AGENTS 0x1178f10    /* the agent table: +0xc/+0x1c the low ids' (entries of 8 bytes, count), +0x2c/+0x3c the high */
#define G_WORLD 0x115e9ec
typedef BYTE *(__fastcall *FindAbilFn)(BYTE *, void *, int, int, int, int, int);
typedef int(__fastcall *VGetFn)(BYTE *);
static FindAbilFn FindAbil_orig;
static IsaFn Isa_fn;
#define AISA_SLOTS 4096
static struct {
    int a, b, r;
} g_aisa[AISA_SLOTS];
static DWORD g_aisa_world;
static LONG g_aisa_seen_gen = -1;
static volatile LONG g_fa_calls, g_fa_isa_hits, g_fa_isa_miss;

static inline BYTE *agent_lookup(DWORD id, DWORD gen) { /* 0x4707f0 */
    BYTE *t = *(BYTE **)VA_(G_AGENTS);
    DWORD *e;
    if (!(id & 0x80000000u)) {
        if (id >= *(DWORD *)(t + 0x1c))
            return NULL;
        e = (DWORD *)(*(BYTE **)(t + 0xc) + id * 8);
    } else {
        id &= 0x7fffffffu;
        if (id >= *(DWORD *)(t + 0x3c))
            return NULL;
        e = (DWORD *)(*(BYTE **)(t + 0x2c) + id * 8);
    }
    if (e[0] != 0xfffffffeu)
        return NULL;
    BYTE *o = (BYTE *)(size_t)e[1];
    return *(DWORD *)(o + 0x18) == gen ? o : NULL;
}

static inline BYTE *link_obj(const DWORD *pair) { /* 0x7fc7e0 */
    BYTE *o = agent_lookup(pair[0], pair[1]);
    return o && *(DWORD *)(o + 0x20) == 0 ? *(BYTE **)(o + 0x54) : NULL;
}

static int isa_memo(int a, int b) {
    if (b == 0x2b616761 || a == b)
        return 1;
    DWORD world = *(DWORD *)VA_(G_WORLD);
    if (!world || GetCurrentThreadId() != g_step_tid)
        return Isa_fn(a, b);
    if (world != g_aisa_world || g_aisa_seen_gen != g_isa_gen) {
        memset(g_aisa, 0, sizeof g_aisa);
        g_aisa_world = world;
        g_aisa_seen_gen = g_isa_gen;
    }
    unsigned k = ((unsigned)a * 2654435761u ^ (unsigned)b * 40503u) >> 20; /* (12 bits) */
    if (g_aisa[k].a == a && g_aisa[k].b == b && (a || b)) {
        g_fa_isa_hits++;
        return g_aisa[k].r;
    }
    int r = Isa_fn(a, b);
    g_aisa[k].a = a;
    g_aisa[k].b = b;
    g_aisa[k].r = r;
    g_fa_isa_miss++;
    return r;
}

static inline BYTE *findabil_walk(BYTE *u, int id, int isa, int by_base, int fa, int fb) {
    switch (id) { /* (the ids kept in fields of the unit) */
    case 0x414e6275: case 0x41486275: case 0x41456275: case 0x41476275: case 0x414f6275: case 0x41556275:
        return *(BYTE **)(u + 0x400);
    case 0x41486572:
        return *(BYTE **)(u + 0x3fc);
    case 0x41496e76:
        return *(BYTE **)(u + 0x404);
    case 0x4161746b:
        return *(BYTE **)(u + 0x3f4);
    case 0x416d6f76:
        return *(BYTE **)(u + 0x3f8);
    }
    g_fa_calls++;
    if ((*(DWORD *)(u + 0x3ec) & *(DWORD *)(u + 0x3e8)) == 0xffffffffu)
        return NULL;
    BYTE *p = link_obj((DWORD *)(u + 0x3e8));
    while (p) {
        BYTE *cur = p;
        p = (*(DWORD *)(cur + 0x28) & *(DWORD *)(cur + 0x24)) == 0xffffffffu ? NULL : link_obj((DWORD *)(cur + 0x24));
        if ((cur[0x20] & 0x20) ? fb : fa) {
            int base = *(int *)(cur + 0x34);
            int v = ((VGetFn)(*(void ***)cur)[0x1c / 4])(cur);
            if ((by_base ? base : v) == id)
                return cur;
            if (isa && isa_memo(v, id))
                return cur;
        }
    }
    return NULL;
}

/* 16777216: a unit with no ability of the sleep's type ('Asla', what UnitIsSleeping asks for every unit every step)
 * remembered with its ability list's head: CUnit::AddAbility (0x6d4560) puts every new ability at the head of the
 * list (the head's agent ids change), a removal adds nothing and an ability's type is its class's, so while the head is
 * the same the unit still has none. W3SIM_SLEEPCHECK=1: the walk is made anyway and compared (mismatches logged). */
#define SLEEP_SLOTS 4096
static struct {
    BYTE *u;
    DWORD h0, h1;
} g_sl[SLEEP_SLOTS];
static int g_sl_on, g_sl_check;
static long g_sl_hits, g_sl_miss, g_sl_bad;

static BYTE *__fastcall FindAbil_hook(BYTE *u, void *edx, int id, int isa, int by_base, int fa, int fb) {
    if (!(g_sl_on && id == 0x41736c61 && isa && !by_base && fa && fb && GetCurrentThreadId() == g_step_tid))
        return findabil_walk(u, id, isa, by_base, fa, fb);
    unsigned k = ((unsigned)(size_t)u >> 4) & (SLEEP_SLOTS - 1);
    DWORD h0 = *(DWORD *)(u + 0x3e8), h1 = *(DWORD *)(u + 0x3ec);
    if (g_sl[k].u == u && g_sl[k].h0 == h0 && g_sl[k].h1 == h1) {
        g_sl_hits++;
        if (g_sl_check && findabil_walk(u, id, isa, by_base, fa, fb) && g_sl_bad++ < 40)
            shim_log("cuts: sleep cache: MISMATCH for unit %p (head %08lx %08lx)", u, h0, h1);
        return NULL;
    }
    BYTE *r = findabil_walk(u, id, isa, by_base, fa, fb);
    g_sl_miss++;
    if (!r) {
        g_sl[k].u = u;
        g_sl[k].h0 = h0;
        g_sl[k].h1 = h1;
    } else if (g_sl[k].u == u)
        g_sl[k].u = NULL;
    return r;
}

void cuts_sleep_prefetch(const BYTE *u) { /* (units.c, 137438953472: a unit ahead in the pass) its slot here */
    __builtin_prefetch(&g_sl[((unsigned)(size_t)u >> 4) & (SLEEP_SLOTS - 1)], 0, 3);
}

void cuts_sleep_reset(void) { /* (a game's first step) */
    memset(g_sl, 0, sizeof g_sl);
}

static void install_findabil(void) {
    BYTE *f = VA_(VA_FINDABIL), *lk = VA_(0x7fc7e0), *ag = VA_(0x4707f0);
    /* the code this replaces, as read (the walk at 0x6d65ff, the link lookup, the agent table's address) */
    static const BYTE walk[] = {0x8b, 0x81, 0xec, 0x03, 0x00, 0x00, 0x81, 0xc1, 0xe8, 0x03, 0x00, 0x00};
    static const BYTE link[] = {0xff, 0x71, 0x04, 0xff, 0x31, 0xe8};
    if (memcmp(f + 0x8f, walk, sizeof walk) != 0 || memcmp(lk, link, sizeof link) != 0 ||
        *(DWORD *)(ag + 8) != (DWORD)(size_t)VA_(G_AGENTS) || f[0xe6] != 0xff || f[0xe7] != 0x50 || f[0xe8] != 0x1c) {
        shim_log("cuts: find ability: unexpected code, not hooked");
        return;
    }
    Isa_fn = (IsaFn)(void *)VA_(VA_ISA);
    if (hook(VA_FINDABIL - 0x400000, (void *)FindAbil_hook, (void **)&FindAbil_orig, "find ability"))
        shim_log("cuts: a unit's ability by id in C (0x6d6570), IsA remembered on the main thread");
}

/* 16384: the grid walk */
#define VA_GRIDCELLS 0x8092a0 /* thiscall void cells(query, grid, int rect[4]), ret 8 */
#define VA_GRIDCELL 0x809170  /* thiscall void cell(query, grid, index, stamp), ret 0xc */
#define VA_WMASK 0x809030     /* cdecl uint64 mask(widget) */
#define VA_QGROW 0x9919c0     /* thiscall char grow(query, bytes, 1), ret 8 */
typedef void(__fastcall *GridCellsFn)(BYTE *, void *, BYTE *, int *);
typedef char(__fastcall *QGrowFn)(BYTE *, void *, unsigned, int);
static GridCellsFn GridCells_orig;
static volatile LONG g_gw_calls, g_gw_cells, g_gw_big;

static inline void grid_entry(BYTE *q, DWORD kind, BYTE *o, DWORD stamp, DWORD cs) {
    if (kind == 0x2000000)
        return;
    DWORD st = *(DWORD *)(o + 0x38);
    if (st == stamp || st == cs || st == 0xffffffffu)
        return;
    if (kind != 0x1000000) {
        *(DWORD *)(o + 0x38) = cs;
        return;
    }
    /* 0x809030's mask (the low and high words), against the query's */
    DWORD f40 = *(DWORD *)(o + 0x40), s34 = *(DWORD *)(o + 0x34);
    DWORD low = 1u << ((f40 >> 5) & 7);
    DWORD mlo = ((low & 2) ? 1u : 0u) << (f40 & 31);
    DWORD mhi = ((((((2 - ((s34 >> 26) & 1)) << 2) | (2 - ((s34 >> 25) & 1))) << 2) | (2 - ((f40 >> 8) & 1))) << 8) | low;
    if (((mlo & *(DWORD *)(q + 0x48)) | ((*(DWORD *)(o + 0x44) >> 28) & 1) | (mhi & *(DWORD *)(q + 0x4c))) == 0) {
        DWORD cnt = *(DWORD *)(q + 0x1c), cap = *(DWORD *)(q + 0x18);
        int ok = 1;
        if (cnt + 1 > cap) {
            DWORD grow = *(DWORD *)(q + 0x14);
            if (!grow)
                ok = 0;
            else {
                DWORD need = cnt - cap + 1, add = grow > need ? grow : need;
                if (((QGrowFn)(void *)VA_(VA_QGROW))(q, NULL, (cap + add) * 8, 1))
                    *(DWORD *)(q + 0x18) += add;
                else
                    ok = 0;
            }
        }
        if (ok) { /* (0x809400 for one entry) */
            DWORD c = *(DWORD *)(q + 0x1c);
            *(DWORD *)(q + 0x1c) = c + 1;
            DWORD *arr = *(DWORD **)(q + 0xc);
            arr[c * 2] = (DWORD)(size_t)o;
            arr[c * 2 + 1] = 0;
        }
    }
    *(DWORD *)(o + 0x38) = stamp;
}

/* a cell's list: its entries read first (the links only; processing never changes them) and their widgets
 * prefetched, then processed in the list's order as 0x809170 does (the walk is bound by the widgets' cache
 * misses: ~5,000 entries a step for the whole map) */
#define GW_BATCH 64
static inline void grid_cell(BYTE *q, BYTE *g, unsigned idx, DWORD stamp) {
    DWORD e = (*(DWORD **)(g + 0x28))[idx] & 0xffffff;
    if (e == 0xffffff)
        return;
    DWORD cs = ++*(DWORD *)(g + 0xb4);
    DWORD kinds[GW_BATCH];
    BYTE *objs[GW_BATCH];
    do {
        int n = 0;
        BYTE *ents = *(BYTE **)(g + 0x78);
        while (e != 0xffffff && n < GW_BATCH) {
            DWORD *ent = (DWORD *)(ents + e * 8);
            DWORD w = ent[0];
            kinds[n] = w & 0xff000000;
            objs[n] = (BYTE *)(size_t)ent[1];
            if (kinds[n] != 0x2000000) {
                __builtin_prefetch(objs[n] + 0x34, 1);
                __builtin_prefetch(objs[n] + 0x44, 0);
            }
            n++;
            e = w & 0xffffff;
        }
        for (int k = 0; k < n; k++)
            grid_entry(q, kinds[k], objs[k], stamp, cs);
    } while (e != 0xffffff);
}

static void __fastcall GridCells_hook(BYTE *q, void *edx, BYTE *g, int *r) {
    int y1 = *(int *)(g + 0x60), x1 = *(int *)(g + 0x5c), y0 = *(int *)(g + 0x58), x0 = *(int *)(g + 0x54);
    if (r[3] < y1)
        y1 = r[3];
    if (r[2] < x1)
        x1 = r[2];
    if (r[1] > y0)
        y0 = r[1];
    if (r[0] > x0)
        x0 = r[0];
    if (x0 >= x1 || y0 >= y1)
        return;
    g_gw_calls++;
    g_gw_cells += (x1 - x0) * (y1 - y0);
    if ((x1 - x0) * (y1 - y0) > 1000)
        g_gw_big++;
    DWORD stamp = ++*(DWORD *)(g + 0xb4);
    if (*(BYTE **)(q + 0x20))
        *(DWORD *)(*(BYTE **)(q + 0x20) + 0x38) = stamp;
    for (unsigned i = (unsigned)x0; i < (unsigned)x1; i++)
        for (unsigned j = (unsigned)y0; j < (unsigned)y1; j++)
            grid_cell(q, g, *(DWORD *)(g + 0x3c) * i + j, stamp);
}

static void install_gridwalk(void) {
    BYTE *f = VA_(VA_GRIDCELLS), *c = VA_(VA_GRIDCELL), *m = VA_(VA_WMASK);
    /* the code as read: the call per cell, the per-entry mask call, the push's helpers */
    if (f[0x8d] != 0xe8 || f + 0x92 + *(int32_t *)(f + 0x8e) != c || c[0x87] != 0xe8 ||
        c + 0x8c + *(int32_t *)(c + 0x88) != m || c[0xdb] != 0xe8 || c + 0xe0 + *(int32_t *)(c + 0xdc) != VA_(VA_QGROW) ||
        c[0xf8] != 0xe8 || c + 0xfd + *(int32_t *)(c + 0xf9) != VA_(0x809400) || memcmp(m + 0x40 - 0x30, "\x8b\x43\x40\xc1\xe8\x05", 6) != 0) {
        shim_log("cuts: grid walk: unexpected code, not hooked");
        return;
    }
    if (hook(VA_GRIDCELLS - 0x400000, (void *)GridCells_hook, (void **)&GridCells_orig, "grid walk"))
        shim_log("cuts: the widget grid's walk in C (0x8092a0)");
}

/* 4294967296 (round 4): the name hash (0x433890, cdecl unsigned hash(const char *name)): the name upper-cased ('a'-'z'
 * less 0x20, '/' as '\\', at most 1023 bytes) into a 1 KB buffer, then Jenkins' lookup2 (0x4347f0, init 0) over it.
 * Every JASS identifier the VM resolves goes through it (its symbol tables, 0x8cdca0: 1-3 % of the main thread with
 * the AI's scripts). The same function in one pass, the bytes mapped by a table as they are read, no buffer. */
static BYTE g_nh_up[256];
#define NH_MIX(a, b, c)                                                                                                \
    do {                                                                                                               \
        a -= b; a -= c; a ^= (c >> 13);                                                                                \
        b -= c; b -= a; b ^= (a << 8);                                                                                 \
        c -= a; c -= b; c ^= (b >> 13);                                                                                \
        a -= b; a -= c; a ^= (c >> 12);                                                                                \
        b -= c; b -= a; b ^= (a << 16);                                                                                \
        c -= a; c -= b; c ^= (b >> 5);                                                                                 \
        a -= b; a -= c; a ^= (c >> 3);                                                                                 \
        b -= c; b -= a; b ^= (a << 10);                                                                                \
        c -= a; c -= b; c ^= (b >> 15);                                                                                \
    } while (0)

static inline unsigned nh_word(const BYTE *k) {
    return (unsigned)g_nh_up[k[0]] | ((unsigned)g_nh_up[k[1]] << 8) | ((unsigned)g_nh_up[k[2]] << 16) |
           ((unsigned)g_nh_up[k[3]] << 24);
}

static unsigned nh_compute(const BYTE *s, unsigned len);

static unsigned __cdecl NameHash_hook(const BYTE *s) {
    unsigned len = 0;
    while (len < 1023 && s[len])
        len++;
    return nh_compute(s, len);
}

static unsigned nh_compute(const BYTE *s, unsigned len) {
    unsigned a = 0x9e3779b9u, b = 0x9e3779b9u, c = 0, n = len;
    const BYTE *k = s;
    for (; n >= 12; n -= 12, k += 12) {
        a += nh_word(k);
        b += nh_word(k + 4);
        c += nh_word(k + 8);
        NH_MIX(a, b, c);
    }
    c += len;
    switch (n) {
    case 11: c += (unsigned)g_nh_up[k[10]] << 24; /* fall through */
    case 10: c += (unsigned)g_nh_up[k[9]] << 16;  /* fall through */
    case 9: c += (unsigned)g_nh_up[k[8]] << 8;    /* fall through */
    case 8: b += (unsigned)g_nh_up[k[7]] << 24;   /* fall through */
    case 7: b += (unsigned)g_nh_up[k[6]] << 16;   /* fall through */
    case 6: b += (unsigned)g_nh_up[k[5]] << 8;    /* fall through */
    case 5: b += g_nh_up[k[4]];                   /* fall through */
    case 4: a += (unsigned)g_nh_up[k[3]] << 24;   /* fall through */
    case 3: a += (unsigned)g_nh_up[k[2]] << 16;   /* fall through */
    case 2: a += (unsigned)g_nh_up[k[1]] << 8;    /* fall through */
    case 1: a += g_nh_up[k[0]];                   /* fall through */
    default: break;
    }
    NH_MIX(a, b, c);
    return c;
}

static void install_namehash(void) {
    BYTE *f = VA_(0x433890);
    static const BYTE code[] = {0x55, 0x8b, 0xec, 0x81, 0xec, 0x04, 0x04, 0x00, 0x00};
    if (memcmp(f, code, sizeof code) != 0 || f[0x67] != 0xe8 || f + 0x6c + *(int32_t *)(f + 0x68) != VA_(0x4347f0)) {
        shim_log("cuts: name hash: unexpected code, not hooked");
        return;
    }
    for (int i = 0; i < 256; i++)
        g_nh_up[i] = (BYTE)(i >= 'a' && i <= 'z' ? i - 0x20 : i == '/' ? '\\' : i);
    /* against the executable's own: every length to 1100 (past its 1023-byte cut), random bytes, then words */
    typedef unsigned(__cdecl * HashFn)(const BYTE *);
    static BYTE buf[1200];
    unsigned r = 12345;
    for (int k = 0; k < 60000; k++) {
        int len = k < 1101 ? k : (int)(r % 64);
        for (int i = 0; i < len; i++) {
            r = r * 1664525u + 1013904223u;
            BYTE v = (BYTE)(r >> 24);
            buf[i] = (k & 1) ? (v ? v : 1) : (BYTE)("aZ/\\_09zA.:"[v % 11]);
        }
        buf[len] = 0;
        unsigned n = 0;
        while (n < 1023 && buf[n])
            n++;
        if (((HashFn)(void *)f)(buf) != nh_compute(buf, n)) {
            shim_log("cuts: name hash: MISMATCH in the self-test (length %d), not hooked", len);
            return;
        }
    }
    if (hook(0x433890 - 0x400000, (void *)NameHash_hook, NULL, "name hash"))
        shim_log("cuts: the name hash in one pass (W3SIM_CUTS & 4294967296)");
}

/* 32768: a unit's handle converted once per unit of the pass */
#define VA_H2UNIT 0x483940   /* cdecl CUnit *h2unit(handle) */
#define VA_H2WIDGET 0x483ba0 /* cdecl CWidget *h2widget(handle) */
typedef BYTE *(__cdecl *H2Fn)(int);
static H2Fn H2Unit_orig, H2Widget_orig;
static int g_h2_on, g_h2u_h, g_h2w_h;
static BYTE *g_h2u_p, *g_h2w_p;
static volatile LONG g_h2_hits, g_h2_miss;

void cuts_unit_begin(int on) { /* units.c: a unit's natives begin (on) or end (0); forgotten either way */
    g_h2u_h = g_h2w_h = 0;
    g_h2_on = on && (g_cuts & 32768) && H2Unit_orig;
}

static BYTE *__cdecl H2Unit_hook(int h) {
    if (g_h2_on && h && h == g_h2u_h) {
        g_h2_hits++;
        return g_h2u_p;
    }
    BYTE *r = H2Unit_orig(h);
    if (g_h2_on) {
        g_h2u_h = h;
        g_h2u_p = r;
        g_h2_miss++;
    }
    return r;
}

static BYTE *__cdecl H2Widget_hook(int h) {
    if (g_h2_on && h && h == g_h2w_h) {
        g_h2_hits++;
        return g_h2w_p;
    }
    BYTE *r = H2Widget_orig(h);
    if (g_h2_on) {
        g_h2w_h = h;
        g_h2w_p = r;
        g_h2_miss++;
    }
    return r;
}

static void install_h2(void) {
    static const BYTE code[] = {0x55, 0x8b, 0xec, 0x6a, 0xff, 0x68};
    BYTE *u = VA_(VA_H2UNIT), *w = VA_(VA_H2WIDGET);
    if (memcmp(u, code, sizeof code) != 0 || memcmp(w, code, sizeof code) != 0) {
        shim_log("cuts: handle conversions: unexpected code, not hooked");
        return;
    }
    if (hook(VA_H2UNIT - 0x400000, (void *)H2Unit_hook, (void **)&H2Unit_orig, "handle to unit") &&
        hook(VA_H2WIDGET - 0x400000, (void *)H2Widget_hook, (void **)&H2Widget_orig, "handle to widget"))
        shim_log("cuts: a unit's handle converted once per unit of the pass");
}

/* 65536: the calling thread's priority remembered */
static int(WINAPI *GetThreadPriority_orig)(HANDLE);
static BOOL(WINAPI *SetThreadPriority_orig)(HANDLE, int);
static __thread int t_prio, t_prio_known; /* (per thread) */
static volatile LONG g_prio_hits, g_prio_miss;

static int WINAPI GetThreadPriority_hook(HANDLE h) {
    if (h == GetCurrentThread() && t_prio_known) {
        g_prio_hits++;
        return t_prio;
    }
    int p = GetThreadPriority_orig(h);
    if (h == GetCurrentThread() && p != THREAD_PRIORITY_ERROR_RETURN) {
        t_prio = p;
        t_prio_known = 1;
        g_prio_miss++;
    }
    return p;
}

static BOOL WINAPI SetThreadPriority_hook(HANDLE h, int p) {
    BOOL ok = SetThreadPriority_orig(h, p);
    if (h == GetCurrentThread())
        t_prio_known = 0; /* (the next query asks again: what Wine made of it) */
    else
        t_prio_known = 0;
    return ok;
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
    pathing_install(g_cuts); /* (1048576 replaces 64's A* run: the loop in C remembers the ring's cells itself) */
    if ((g_cuts & 64) && !(g_cuts & 1048576)) {
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
    if (g_cuts & 4096)
        install_isqrt2();
    if (g_cuts & 8192) {
        char sc[8];
        g_sl_on = (g_cuts & 16777216) != 0;
        g_sl_check = g_sl_on && GetEnvironmentVariableA("W3SIM_SLEEPCHECK", sc, sizeof sc) && sc[0] == '1';
        install_findabil();
        if (g_sl_on)
            shim_log("cuts: units without a sleep ability remembered with their ability list's head%s", g_sl_check ? " (checked)" : "");
    }
    if (g_cuts & 16384)
        install_gridwalk();
    if (g_cuts & 32768)
        install_h2();
    if (g_cuts & 4294967296LL)
        install_namehash();
    if (g_cuts & 65536) {
        if (iat_hook((HMODULE)g_base, "KERNEL32.dll", "GetThreadPriority", GetThreadPriority_hook, (void **)&GetThreadPriority_orig) &&
            iat_hook((HMODULE)g_base, "KERNEL32.dll", "SetThreadPriority", SetThreadPriority_hook, (void **)&SetThreadPriority_orig))
            shim_log("cuts: the calling thread's priority remembered; observations in one send");
        else
            shim_log("cuts: thread priority imports not found");
    }
    softfloat_install(g_cuts);
    shim_log("cuts installed: %lld", g_cuts);
}

void cuts_report(void) {
    if (g_cuts & (524288 | 1048576 | 4194304))
        pathing_report();
    if (g_cuts & 2147483648LL)
        units_cmd_report();
    if (g_cuts)
        shim_log("cuts/s: mapver hits=%ld miss=%ld paint skipped=%ld events reused=%ld new=%ld textures=%ld "
                 "rings=%ld (orig %ld) cells hit=%ld miss=%ld fhash hit=%ld miss=%ld relax=%ld early=%ld isa hit=%ld miss=%ld "
                 "findabil=%ld (isa hit %ld miss %ld) gridwalk=%ld (big %ld) cells=%ld h2 hit=%ld miss=%ld prio hit=%ld miss=%ld",
                 InterlockedExchange(&g_mv_hits, 0), InterlockedExchange(&g_mv_miss, 0),
                 InterlockedExchange(&g_paint_skipped, 0), InterlockedExchange(&g_evt_reused, 0),
                 InterlockedExchange(&g_evt_new, 0), InterlockedExchange(&g_tex_skipped, 0),
                 InterlockedExchange(&g_ring_calls, 0), InterlockedExchange(&g_ring_orig, 0),
                 InterlockedExchange(&g_cell_hits, 0), InterlockedExchange(&g_cell_miss, 0),
                 InterlockedExchange(&g_fh_hits, 0), InterlockedExchange(&g_fh_miss, 0),
                 InterlockedExchange(&g_relax_n, 0), InterlockedExchange(&g_relax_early, 0),
                 InterlockedExchange(&g_isa_hits, 0), InterlockedExchange(&g_isa_miss, 0),
                 InterlockedExchange(&g_fa_calls, 0), InterlockedExchange(&g_fa_isa_hits, 0),
                 InterlockedExchange(&g_fa_isa_miss, 0), InterlockedExchange(&g_gw_calls, 0),
                 InterlockedExchange(&g_gw_big, 0), InterlockedExchange(&g_gw_cells, 0),
                 InterlockedExchange(&g_h2_hits, 0), InterlockedExchange(&g_h2_miss, 0),
                 InterlockedExchange(&g_prio_hits, 0), InterlockedExchange(&g_prio_miss, 0));
}
