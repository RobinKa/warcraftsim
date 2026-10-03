/* Turbo: run as many game turns per frame as the CPU allows.
 *
 * In an offline game the host thread emits "time slot" packets that grant game time, and
 * GameUpdate (once per frame) turns elapsed wall time -- scaled by a lag controller and clamped --
 * into pending time, consuming 25 ms turns while both pending time and granted time remain.
 * That pacing caps a game at roughly 25x realtime even with a fast virtual clock. With turbo on:
 *
 *   - every time slot the host emits grants `turbo_ms` of game time, and
 *   - every GameUpdate starts with `turbo_ms` of pending time and sees no elapsed wall time,
 *
 * so each frame simulates up to `turbo_ms` of game time and rendering is amortised over many
 * turns. The simulation itself is unchanged: turns are still 25 ms and in order. Offsets are for
 * the 1.29.2.9231 executable (the layout was documented by the MIT-licensed wc3env project).
 */
#include "w3shim.h"

#include "MinHook.h"

#include <stdio.h>
#include <string.h>

#define RVA_GAMEUPDATE 0x1aefd0      /* thiscall GameUpdate(game, now_ms), once per frame */
#define RVA_TURN_TIME_WRITE 0x54c960 /* cdecl put_u16(store, const u16 *ms): slot time */
#define RVA_TURN_TIME_RET 0x557db2   /* return address inside the offline host's slot builder */
#define GAME_PREV_NOW 0x2614         /* game+: previous now_ms */
#define GAME_PENDING_MS 0x2618       /* game+: pending game ms */
#define MAX_TURBO_MS 4000

static volatile LONG g_turbo_ms;

/* profiling (W3SIM_PROFILE=1): per-second totals in the log */
#define RVA_GXPRESENT 0x3d1070 /* cdecl GxPresent(flags): end of the frame */
static int g_profile;
static LONGLONG g_freq;
static volatile LONGLONG g_t_update, g_t_present, g_n_update, g_n_present, g_last_present_end;
static volatile LONGLONG g_t_frame_other;
typedef int(__cdecl *GxPresentFn)(int flags);
static GxPresentFn GxPresent_orig;

static LONGLONG now_qpc(void) {
    return real_qpc_ticks();
}

/* End of every frame: frame capture and the frame-stepped clock (see sync.c, clock.c). */
static int __cdecl GxPresent_hook(int flags) {
    LONGLONG t0 = g_profile ? now_qpc() : 0;
    int r = GxPresent_orig(flags);
    if (g_profile) {
        g_t_present += now_qpc() - t0;
        g_n_present++;
    }
    if (frame_capture_get())
        sync_frame();
    clock_frame();
    return r;
}

static volatile DWORD g_game_tid;

/* Sampling profiler: the main thread's EIP every ~1 ms, attributed to modules / exe pages. */
#define PAGE_BUCKETS 4096
static unsigned g_exe_pages[PAGE_BUCKETS];
#define NT_SLOTS 64
static BYTE *g_nt_addr[NT_SLOTS];
static unsigned g_nt_count[NT_SLOTS];

static const char *nearest_export(HMODULE mod, BYTE *ip) {
    BYTE *base = (BYTE *)mod;
    IMAGE_NT_HEADERS *nt = (IMAGE_NT_HEADERS *)(base + ((IMAGE_DOS_HEADER *)base)->e_lfanew);
    IMAGE_DATA_DIRECTORY d = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
    if (!d.VirtualAddress)
        return "?";
    IMAGE_EXPORT_DIRECTORY *ex = (IMAGE_EXPORT_DIRECTORY *)(base + d.VirtualAddress);
    DWORD *names = (DWORD *)(base + ex->AddressOfNames);
    WORD *ords = (WORD *)(base + ex->AddressOfNameOrdinals);
    DWORD *funcs = (DWORD *)(base + ex->AddressOfFunctions);
    const char *best = "?";
    BYTE *best_addr = NULL;
    for (DWORD i = 0; i < ex->NumberOfNames; i++) {
        BYTE *f = base + funcs[ords[i]];
        if (f <= ip && f > best_addr) {
            best_addr = f;
            best = (const char *)(base + names[i]);
        }
    }
    return best;
}
static unsigned g_mod_samples[16];
static char g_mod_names[16][32];
static int g_n_mods;
static unsigned g_other_samples, g_total_samples;

#define SYS_SLOTS 32
static DWORD g_sys_key[SYS_SLOTS];
static unsigned g_sys_count[SYS_SLOTS];
static DWORD g_caller_key[SYS_SLOTS];
static unsigned g_caller_count[SYS_SLOTS];

static void bump(DWORD *keys, unsigned *counts, DWORD key) {
    int k;
    for (k = 0; k < SYS_SLOTS && keys[k] && keys[k] != key; k++)
        ;
    if (k < SYS_SLOTS) {
        keys[k] = key;
        counts[k]++;
    }
}

static void top(const char *what, DWORD *keys, unsigned *counts) {
    for (int k = 0; k < 6; k++) {
        unsigned best = 0;
        int bi = -1;
        for (int j = 0; j < SYS_SLOTS; j++)
            if (counts[j] > best) {
                best = counts[j];
                bi = j;
            }
        if (bi < 0)
            break;
        shim_log("  %s %#x: %u%%", what, keys[bi], best * 100 / (g_total_samples ? g_total_samples : 1));
        counts[bi] = 0;
    }
    memset(keys, 0, SYS_SLOTS * sizeof(DWORD));
}

static void sample_main(HANDLE th) {
    CONTEXT ctx;
    ctx.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
    if (SuspendThread(th) == (DWORD)-1)
        return;
    BOOL ok = GetThreadContext(th, &ctx);
    ResumeThread(th);
    if (!ok)
        return;
    g_total_samples++;
    BYTE *ip = (BYTE *)ctx.Eip;
    HMODULE mod = NULL;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCSTR)ip, &mod)) {
        g_other_samples++;
        return;
    }
    if ((BYTE *)mod == g_base) {
        size_t page = (size_t)(ip - g_base) >> 12;
        if (page < PAGE_BUCKETS)
            g_exe_pages[page]++;
    } else {
        /* syscall number (eax) and the first return address into the exe on the stack */
        bump(g_sys_key, g_sys_count, ctx.Eax);
        DWORD *sp = (DWORD *)ctx.Esp;
        IMAGE_NT_HEADERS *nt = (IMAGE_NT_HEADERS *)(g_base + ((IMAGE_DOS_HEADER *)g_base)->e_lfanew);
        DWORD size = nt->OptionalHeader.SizeOfImage;
        for (int d = 0; d < 64; d++) {
            DWORD v;
            if (!ReadProcessMemory(GetCurrentProcess(), sp + d, &v, sizeof v, NULL))
                break;
            if (v > (DWORD)(size_t)g_base && v < (DWORD)(size_t)g_base + size) {
                bump(g_caller_key, g_caller_count, v - (DWORD)(size_t)g_base);
                break;
            }
        }
        /* bucket by 256-byte granule to resolve symbols later */
        BYTE *key = (BYTE *)((size_t)ip & ~(size_t)0xff);
        int k;
        for (k = 0; k < NT_SLOTS && g_nt_addr[k] && g_nt_addr[k] != key; k++)
            ;
        if (k < NT_SLOTS) {
            g_nt_addr[k] = key;
            g_nt_count[k]++;
        }
    }
    char name[MAX_PATH];
    GetModuleFileNameA(mod, name, sizeof name);
    char *base = strrchr(name, '\\');
    base = base ? base + 1 : name;
    int i;
    for (i = 0; i < g_n_mods; i++)
        if (!_stricmp(g_mod_names[i], base))
            break;
    if (i == g_n_mods && g_n_mods < 16) {
        lstrcpynA(g_mod_names[g_n_mods], base, 32);
        g_n_mods++;
    }
    if (i < 16)
        g_mod_samples[i]++;
}

static void report_samples(void) {
    char line[1024];
    int len = 0;
    for (int i = 0; i < g_n_mods; i++)
        if (g_mod_samples[i] * 100 >= g_total_samples)
            len += _snprintf(line + len, sizeof line - len, "%s=%u%% ", g_mod_names[i],
                             g_mod_samples[i] * 100 / (g_total_samples ? g_total_samples : 1));
    shim_log("samples=%u modules: %s other=%u", g_total_samples, line, g_other_samples);
    for (int k = 0; k < 8; k++) {
        unsigned best = 0;
        int bi = -1;
        for (int p = 0; p < PAGE_BUCKETS; p++)
            if (g_exe_pages[p] > best) {
                best = g_exe_pages[p];
                bi = p;
            }
        if (bi < 0)
            break;
        shim_log("  exe page %#07x: %u%%", bi << 12, best * 100 / (g_total_samples ? g_total_samples : 1));
        g_exe_pages[bi] = 0;
    }
    for (int k = 0; k < 10; k++) {
        unsigned best = 0;
        int bi = -1;
        for (int j = 0; j < NT_SLOTS; j++)
            if (g_nt_count[j] > best) {
                best = g_nt_count[j];
                bi = j;
            }
        if (bi < 0)
            break;
        HMODULE m = NULL;
        char mn[MAX_PATH] = "?";
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               (LPCSTR)g_nt_addr[bi], &m))
            GetModuleFileNameA(m, mn, sizeof mn);
        char *b = strrchr(mn, '\\');
        shim_log("  %s!%s (+%#x): %u%%", b ? b + 1 : mn, m ? nearest_export(m, g_nt_addr[bi]) : "?",
                 (unsigned)((BYTE *)g_nt_addr[bi] - (BYTE *)m), best * 100 / (g_total_samples ? g_total_samples : 1));
        g_nt_count[bi] = 0;
    }
    top("syscall eax", g_sys_key, g_sys_count);
    top("exe caller rva", g_caller_key, g_caller_count);
    memset(g_nt_addr, 0, sizeof g_nt_addr);
    memset(g_nt_count, 0, sizeof g_nt_count);
    memset(g_exe_pages, 0, sizeof g_exe_pages);
    memset(g_mod_samples, 0, sizeof g_mod_samples);
    g_total_samples = g_other_samples = 0;
}

static DWORD WINAPI profile_thread(LPVOID arg) {
    /* W3SIM_PROFILE=1: time totals; 2: also sample the main thread's EIP every ~1 ms (costly:
     * every sample suspends the thread, a wineserver round trip). Rates are per real second. */
    int sampling = (int)(INT_PTR)arg == 2;
    g_wait_stats = (int)(INT_PTR)arg >= 3;
    HANDLE th = NULL;
    LONGLONG t_last = now_qpc();
    for (int tick = 0;; tick++) {
        if (sampling) {
            for (int k = 0; k < 1000; k++) {
                Sleep_real(1);
                if (!th && g_game_tid)
                    th = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT, FALSE, g_game_tid);
                if (th)
                    sample_main(th);
            }
            if (tick % 2 == 1)
                report_samples();
        } else {
            Sleep_real(1000);
        }
        LONGLONG now = now_qpc();
        double secs = (double)(now - t_last) / (double)g_freq;
        t_last = now;
        LONGLONG u = g_t_update, p = g_t_present, nu = g_n_update, np = g_n_present, w = sync_wait_ticks();
        g_t_update = g_t_present = g_n_update = g_n_present = 0;
        sync_reset_wait_ticks();
        shim_log("profile/s: updates=%.1f (%.1f ms) presents=%.1f (%.1f ms) sync_wait=%.1f ms", nu / secs,
                 u * 1000.0 / g_freq / secs, np / secs, p * 1000.0 / g_freq / secs, w * 1000.0 / g_freq / secs);
        char phases[256];
        sync_phases(secs, phases, sizeof phases);
        shim_log("phases/s: %s", phases);
        if (g_wait_stats)
            clock_report_waits(secs);
        syncstat_report(secs);
    }
    return 0;
}

typedef int(__fastcall *GameUpdateFn)(void *self, void *edx, DWORD now);
static GameUpdateFn GameUpdate_orig;
typedef BYTE *(__cdecl *TurnTimeFn)(BYTE *store, const WORD *ms);
static TurnTimeFn TurnTime_orig;

static int __fastcall GameUpdate_hook(void *self, void *edx, DWORD now) {
    LONG t = g_turbo_ms;
    if (t > 0 && self) {
        BYTE *game = (BYTE *)self;
        *(DWORD *)(game + GAME_PENDING_MS) = (DWORD)t;
        now = *(DWORD *)(game + GAME_PREV_NOW); /* no elapsed time: the lag controller stays out */
    } else if (self) {
        /* Frame-stepped clock (video): a frame never advances the game by more than one frame
         * step. Otherwise time that passed before capture started (a replay counts its loading
         * time) is simulated in the first frame and the video skips the first seconds. */
        double step = clock_frame_seconds();
        DWORD prev = *(DWORD *)((BYTE *)self + GAME_PREV_NOW);
        DWORD max_ms = (DWORD)(step * 1000.0 + 0.5);
        if (step > 0 && prev && now - prev > max_ms && now - prev < 0x80000000u)
            now = prev + max_ms;
    }
    if (!g_profile)
        return GameUpdate_orig(self, edx, now);
    g_game_tid = GetCurrentThreadId();
    LONGLONG t0 = now_qpc();
    int r = GameUpdate_orig(self, edx, now);
    g_t_update += now_qpc() - t0;
    g_n_update++;
    return r;
}

static BYTE *__cdecl TurnTime_hook(BYTE *store, const WORD *ms) {
    LONG t = g_turbo_ms;
    if (t > 0 && ms && *ms && (BYTE *)__builtin_return_address(0) == g_base + RVA_TURN_TIME_RET) {
        WORD grant = (WORD)t;
        return TurnTime_orig(store, &grant);
    }
    return TurnTime_orig(store, ms);
}

void turbo_set(int ms) {
    if (ms < 0)
        ms = 0;
    if (ms > MAX_TURBO_MS)
        ms = MAX_TURBO_MS;
    ms -= ms % 25;
    InterlockedExchange(&g_turbo_ms, ms);
}

int turbo_get(void) {
    return (int)g_turbo_ms;
}

static int prologue_ok(BYTE *p) {
    return p[0] == 0x55 && p[1] == 0x8b && p[2] == 0xec; /* push ebp; mov ebp, esp */
}

void turbo_install(int ms) {
    BYTE *update = g_base + RVA_GAMEUPDATE, *writer = g_base + RVA_TURN_TIME_WRITE;
    BYTE *call = g_base + RVA_TURN_TIME_RET - 5;
    if (!prologue_ok(update) || !prologue_ok(writer) || call[0] != 0xe8 ||
        call + 5 + *(int32_t *)(call + 1) != writer) {
        shim_log("turbo: unsupported executable, not installed");
        return;
    }
    if (MH_Initialize() != MH_OK && MH_Initialize() != MH_ERROR_ALREADY_INITIALIZED) {
        shim_log("turbo: MinHook init failed");
        return;
    }
    MH_STATUS a = MH_CreateHook(update, (void *)GameUpdate_hook, (void **)&GameUpdate_orig);
    MH_STATUS b = MH_CreateHook(writer, (void *)TurnTime_hook, (void **)&TurnTime_orig);
    if (a != MH_OK || b != MH_OK || MH_EnableHook(update) != MH_OK || MH_EnableHook(writer) != MH_OK) {
        shim_log("turbo: hooks failed (%s, %s)", MH_StatusToString(a), MH_StatusToString(b));
        return;
    }
    turbo_set(ms);
    shim_log("turbo installed: %d ms per frame", turbo_get());
    BYTE *present = g_base + RVA_GXPRESENT;
    if (!prologue_ok(present) || MH_CreateHook(present, (void *)GxPresent_hook, (void **)&GxPresent_orig) != MH_OK ||
        MH_EnableHook(present) != MH_OK) {
        shim_log("GxPresent hook failed: no frame capture");
        return;
    }
    char buf[8];
    if (GetEnvironmentVariableA("W3SIM_PROFILE", buf, sizeof buf) && buf[0] >= '1' && buf[0] <= '9') {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        g_freq = f.QuadPart;
        g_profile = 1;
        CreateThread(NULL, 0, profile_thread, (LPVOID)(INT_PTR)(buf[0] - '0'), 0, NULL);
        shim_log("profiling on");
    }
}
