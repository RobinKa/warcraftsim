/* Virtual clock.
 *
 * virtual = virt0 + (real - real0) * speed, or a constant while frozen. The anchor is published
 * through a seqlock so timer calls from any thread never block. Real time is the CPU's time-stamp
 * counter: under Wine on WSL2 QueryPerformanceCounter is a system call, and the game reads its
 * clock (an rdtsc helper) thousands of times per game turn. QPC, GetTickCount and FILETIME are
 * derived from the same virtual TSC (scaled by the TSC/QPC ratio measured at startup).
 */
#include "w3shim.h"

#include <x86intrin.h>

typedef struct {
    int64_t real0, virt0, frozen_at;
    double speed;
    LONG frozen;
} Anchor;

static Anchor g_anchor;
static volatile LONG g_seq;
static CRITICAL_SECTION g_write_lock;
static int64_t g_freq;
static double g_tsc_per_qpc = 1.0;
static int64_t g_base_filetime, g_base_virt;
static DWORD g_wait_floor;

static BOOL(WINAPI *Qpc_orig)(LARGE_INTEGER *);
static DWORD(WINAPI *Tick_orig)(void);
static void(WINAPI *SysTime_orig)(LPFILETIME);
static VOID(WINAPI *Sleep_orig)(DWORD);
static DWORD(WINAPI *SleepEx_orig)(DWORD, BOOL);
static DWORD(WINAPI *Wfso_orig)(HANDLE, DWORD);
static DWORD(WINAPI *WfsoEx_orig)(HANDLE, DWORD, BOOL);
static DWORD(WINAPI *Wfmo_orig)(DWORD, const HANDLE *, BOOL, DWORD);
static DWORD(WINAPI *MsgWait_orig)(DWORD, const HANDLE *, BOOL, DWORD, DWORD);

static int64_t g_qpc0, g_tsc0; /* QPC and TSC at calibration: maps TSC time to the QPC time base */

static int64_t sys_qpc(void) {
    LARGE_INTEGER li;
    (Qpc_orig ? Qpc_orig : QueryPerformanceCounter)(&li);
    return li.QuadPart;
}

/* real time in QPC units, without a system call */
static int64_t real_qpc(void) {
    if (!g_tsc0)
        return sys_qpc();
    return g_qpc0 + (int64_t)((double)(int64_t)(__rdtsc() - g_tsc0) / g_tsc_per_qpc);
}

static Anchor anchor_read(int64_t *now) {
    Anchor a;
    LONG s1, s2;
    do {
        s1 = g_seq;
        MemoryBarrier();
        a = g_anchor;
        if (now)
            *now = real_qpc();
        MemoryBarrier();
        s2 = g_seq;
    } while ((s1 & 1) || s1 != s2);
    return a;
}

static void anchor_write(const Anchor *a) {
    InterlockedIncrement(&g_seq);
    MemoryBarrier();
    g_anchor = *a;
    MemoryBarrier();
    InterlockedIncrement(&g_seq);
}

static int64_t virt_at(const Anchor *a, int64_t now) {
    if (a->frozen)
        now = a->frozen_at;
    return a->virt0 + (int64_t)((double)(now - a->real0) * a->speed);
}

static int64_t virt_qpc(void) {
    int64_t now;
    Anchor a = anchor_read(&now);
    return virt_at(&a, now);
}

/* ticks -> units without overflow: divide first, then scale the remainder */
static int64_t to_units(int64_t ticks, int64_t units_per_sec) {
    return (ticks / g_freq) * units_per_sec + (ticks % g_freq) * units_per_sec / g_freq;
}

int64_t real_qpc_ticks(void) {
    return real_qpc();
}

void Sleep_real(DWORD ms) {
    (Sleep_orig ? Sleep_orig : Sleep)(ms);
}

double clock_speed(void) {
    return anchor_read(NULL).speed;
}

void clock_set_speed(double speed) {
    if (speed <= 0)
        speed = 1;
    EnterCriticalSection(&g_write_lock);
    Anchor a = g_anchor;
    int64_t now = a.frozen ? a.frozen_at : real_qpc();
    a.virt0 = virt_at(&a, now);
    a.real0 = now;
    a.speed = speed;
    anchor_write(&a);
    LeaveCriticalSection(&g_write_lock);
}

void clock_freeze(int frozen) {
    EnterCriticalSection(&g_write_lock);
    Anchor a = g_anchor;
    if (frozen && !a.frozen) {
        a.frozen_at = real_qpc();
        a.frozen = 1;
        anchor_write(&a);
    } else if (!frozen && a.frozen) {
        a.virt0 = virt_at(&a, a.frozen_at);
        a.real0 = real_qpc();
        a.frozen = 0;
        anchor_write(&a);
    }
    LeaveCriticalSection(&g_write_lock);
}

static BOOL WINAPI Qpc_hook(LARGE_INTEGER *out) {
    out->QuadPart = virt_qpc();
    return TRUE;
}

static DWORD WINAPI Tick_hook(void) {
    return (DWORD)to_units(virt_qpc(), 1000);
}

static void WINAPI SysTime_hook(LPFILETIME ft) {
    int64_t t = g_base_filetime + to_units(virt_qpc() - g_base_virt, 10000000);
    ft->dwLowDateTime = (DWORD)t;
    ft->dwHighDateTime = (DWORD)(t >> 32);
}

__attribute__((used)) uint64_t __cdecl virt_tsc(void);
__attribute__((used)) uint64_t __cdecl virt_tsc(void) {
    return (uint64_t)((double)virt_qpc() * g_tsc_per_qpc);
}

/* Replacement for the game's `rdtsc; ret` helper: result in edx:eax, every other register
 * preserved exactly like the original instruction sequence. */
__attribute__((naked)) static void rdtsc_stub(void) {
    __asm__ volatile("push %ecx\n\t"
                     "call _virt_tsc\n\t"
                     "pop %ecx\n\t"
                     "ret\n\t");
}

static void calibrate_tsc(void) {
    LARGE_INTEGER q0, q1;
    uint64_t t0, t1;
    QueryPerformanceCounter(&q0);
    t0 = __rdtsc();
    Sleep(50);
    QueryPerformanceCounter(&q1);
    t1 = __rdtsc();
    g_tsc_per_qpc = (double)(t1 - t0) / (double)(q1.QuadPart - q0.QuadPart);
    g_qpc0 = q1.QuadPart;
    g_tsc0 = (int64_t)t1;
}

static void patch_rdtsc(void) {
    BYTE *p = g_base + RVA_RDTSC_HELPER;
    if (p[0] != 0x0f || p[1] != 0x31 || p[2] != 0xc3) {
        shim_log("rdtsc helper not found (unsupported executable); precise timer not virtualised");
        return;
    }
    DWORD old;
    VirtualProtect(p, 8, PAGE_EXECUTE_READWRITE, &old);
    p[0] = 0xe9; /* jmp rel32 */
    *(int32_t *)(p + 1) = (int32_t)((BYTE *)rdtsc_stub - (p + 5));
    VirtualProtect(p, 8, old, &old);
    FlushInstructionCache(GetCurrentProcess(), p, 8);
    shim_log("rdtsc helper patched, tsc/qpc=%.3f", g_tsc_per_qpc);
}

/* Timeouts pace threads against wall time: divide finite ones by the speed factor. */
static DWORD scale_timeout(DWORD ms) {
    if (ms == 0 || ms == INFINITE)
        return ms;
    Anchor a = anchor_read(NULL);
    if (a.speed <= 1.0)
        return ms;
    DWORD s = (DWORD)(ms / a.speed);
    if (a.frozen && s == 0)
        return 1; /* do not spin on a clock that does not move */
    return s < g_wait_floor ? g_wait_floor : s;
}

static VOID WINAPI Sleep_hook(DWORD ms) {
    Sleep_orig(scale_timeout(ms));
}
static DWORD WINAPI SleepEx_hook(DWORD ms, BOOL alertable) {
    return SleepEx_orig(scale_timeout(ms), alertable);
}
static DWORD WINAPI Wfso_hook(HANDLE h, DWORD ms) {
    return Wfso_orig(h, scale_timeout(ms));
}
static DWORD WINAPI WfsoEx_hook(HANDLE h, DWORD ms, BOOL alertable) {
    return WfsoEx_orig(h, scale_timeout(ms), alertable);
}
static DWORD WINAPI Wfmo_hook(DWORD n, const HANDLE *h, BOOL all, DWORD ms) {
    return Wfmo_orig(n, h, all, scale_timeout(ms));
}
static DWORD WINAPI MsgWait_hook(DWORD n, const HANDLE *h, BOOL all, DWORD ms, DWORD mask) {
    return MsgWait_orig(n, h, all, scale_timeout(ms), mask);
}

void clock_install(double speed, DWORD wait_floor) {
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    g_freq = f.QuadPart;
    g_wait_floor = wait_floor;
    InitializeCriticalSection(&g_write_lock);
    calibrate_tsc();

    HMODULE exe = (HMODULE)g_base;
    int ok = 1;
    ok &= iat_hook(exe, "KERNEL32.dll", "QueryPerformanceCounter", Qpc_hook, (void **)&Qpc_orig);
    ok &= iat_hook(exe, "KERNEL32.dll", "GetTickCount", Tick_hook, (void **)&Tick_orig);
    ok &= iat_hook(exe, "KERNEL32.dll", "GetSystemTimeAsFileTime", SysTime_hook, (void **)&SysTime_orig);
    ok &= iat_hook(exe, "KERNEL32.dll", "Sleep", Sleep_hook, (void **)&Sleep_orig);
    ok &= iat_hook(exe, "KERNEL32.dll", "SleepEx", SleepEx_hook, (void **)&SleepEx_orig);
    ok &= iat_hook(exe, "KERNEL32.dll", "WaitForSingleObject", Wfso_hook, (void **)&Wfso_orig);
    ok &= iat_hook(exe, "KERNEL32.dll", "WaitForSingleObjectEx", WfsoEx_hook, (void **)&WfsoEx_orig);
    ok &= iat_hook(exe, "KERNEL32.dll", "WaitForMultipleObjects", Wfmo_hook, (void **)&Wfmo_orig);
    if (!iat_hook(exe, "USER32.dll", "MsgWaitForMultipleObjects", MsgWait_hook, (void **)&MsgWait_orig))
        shim_log("MsgWaitForMultipleObjects not imported (ok)");
    if (!ok)
        shim_log("warning: some clock imports were not found");

    Anchor a = {0};
    a.real0 = a.virt0 = real_qpc();
    a.speed = speed > 0 ? speed : 1.0;
    anchor_write(&a);

    FILETIME ft;
    SysTime_orig(&ft);
    g_base_filetime = ((int64_t)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
    g_base_virt = virt_qpc();

    patch_rdtsc();
    shim_log("clock installed: speed=%.1f wait_floor=%lu", a.speed, (unsigned long)wait_floor);
}
