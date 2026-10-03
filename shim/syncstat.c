/* W3SIM_PROFILE=4: the game's own calls to Win32 synchronization (events, mutexes, semaphores,
 * handles), counted per call site in the executable and reported each second. Under Wine every one of
 * them is a request to wineserver: ~300 a step in a whole game, a third of a game's CPU with the
 * server's. */
#include "w3shim.h"

#include <string.h>

enum { S_SET, S_RESET, S_CREATE_A, S_CREATE_W, S_CLOSE, S_RELMUTEX, S_RELSEM, S_APIS };
static const char *g_names[S_APIS] = {"SetEvent", "ResetEvent", "CreateEventA", "CreateEventW", "CloseHandle",
                                      "ReleaseMutex", "ReleaseSemaphore"};

static BOOL(WINAPI *SetEvent_orig)(HANDLE);
static BOOL(WINAPI *ResetEvent_orig)(HANDLE);
static HANDLE(WINAPI *CreateEventA_orig)(LPSECURITY_ATTRIBUTES, BOOL, BOOL, LPCSTR);
static HANDLE(WINAPI *CreateEventW_orig)(LPSECURITY_ATTRIBUTES, BOOL, BOOL, LPCWSTR);
static BOOL(WINAPI *CloseHandle_orig)(HANDLE);
static BOOL(WINAPI *ReleaseMutex_orig)(HANDLE);
static BOOL(WINAPI *ReleaseSemaphore_orig)(HANDLE, LONG, LPLONG);

#define SITES 128
static struct {
    volatile LONG key; /* rva * 8 + api + 1; 0: free */
    volatile LONG n;
} g_sites[SITES];

/* the manual-reset events the game made (handle -> state; only the game's own Set/Reset change them) */
#define EVENTS 256
static struct {
    volatile HANDLE h;
    volatile LONG state;
} g_events[EVENTS];
static volatile LONG g_redundant[S_APIS], g_tracked[S_APIS];

static int event_slot(HANDLE h, int add) {
    for (int i = 0; i < EVENTS; i++) {
        int k = (int)(((DWORD)(UINT_PTR)h / 4 * 2654435761u) % EVENTS + i) % EVENTS;
        if (g_events[k].h == h)
            return k;
        if (g_events[k].h == 0)
            return add && InterlockedCompareExchangePointer((PVOID *)&g_events[k].h, h, 0) == 0 ? k : -1;
    }
    return -1;
}

static void track(int api, HANDLE h) {
    int k = event_slot(h, 0);
    if (k < 0)
        return;
    InterlockedIncrement(&g_tracked[api]);
    LONG want = api == S_SET ? 1 : 0, was = InterlockedExchange(&g_events[k].state, want);
    if (was == want)
        InterlockedIncrement(&g_redundant[api]);
}

static void count(int api, void *ra) {
    LONG key = (LONG)(((DWORD)((BYTE *)ra - g_base)) * 8 + api + 1);
    for (int i = 0; i < SITES; i++) {
        int k = (int)(((DWORD)key * 2654435761u) % SITES + i) % SITES;
        if (g_sites[k].key == key || (g_sites[k].key == 0 && InterlockedCompareExchange(&g_sites[k].key, key, 0) == 0)) {
            InterlockedIncrement(&g_sites[k].n);
            return;
        }
    }
}

static BOOL WINAPI SetEvent_hook(HANDLE h) {
    count(S_SET, __builtin_return_address(0));
    track(S_SET, h);
    return SetEvent_orig(h);
}
static BOOL WINAPI ResetEvent_hook(HANDLE h) {
    count(S_RESET, __builtin_return_address(0));
    track(S_RESET, h);
    return ResetEvent_orig(h);
}
static HANDLE WINAPI CreateEventA_hook(LPSECURITY_ATTRIBUTES sa, BOOL manual, BOOL initial, LPCSTR name) {
    count(S_CREATE_A, __builtin_return_address(0));
    HANDLE h = CreateEventA_orig(sa, manual, initial, name);
    int k = h && manual && !name ? event_slot(h, 1) : -1;
    if (k >= 0)
        g_events[k].state = initial ? 1 : 0;
    return h;
}
static HANDLE WINAPI CreateEventW_hook(LPSECURITY_ATTRIBUTES sa, BOOL manual, BOOL initial, LPCWSTR name) {
    count(S_CREATE_W, __builtin_return_address(0));
    return CreateEventW_orig(sa, manual, initial, name);
}
static BOOL WINAPI CloseHandle_hook(HANDLE h) {
    count(S_CLOSE, __builtin_return_address(0));
    int k = event_slot(h, 0);
    if (k >= 0)
        g_events[k].h = (HANDLE)-1; /* (a tombstone: the slot stays taken, the handle is no event of ours) */
    return CloseHandle_orig(h);
}
static BOOL WINAPI ReleaseMutex_hook(HANDLE h) {
    count(S_RELMUTEX, __builtin_return_address(0));
    return ReleaseMutex_orig(h);
}
static BOOL WINAPI ReleaseSemaphore_hook(HANDLE h, LONG n, LPLONG prev) {
    count(S_RELSEM, __builtin_return_address(0));
    return ReleaseSemaphore_orig(h, n, prev);
}

static int g_on;

void syncstat_install(void) {
    char buf[8];
    if (!GetEnvironmentVariableA("W3SIM_PROFILE", buf, sizeof buf) || buf[0] < '4' || buf[0] > '9')
        return;
    HMODULE exe = (HMODULE)g_base;
    iat_hook(exe, "KERNEL32.dll", "SetEvent", SetEvent_hook, (void **)&SetEvent_orig);
    iat_hook(exe, "KERNEL32.dll", "ResetEvent", ResetEvent_hook, (void **)&ResetEvent_orig);
    iat_hook(exe, "KERNEL32.dll", "CreateEventA", CreateEventA_hook, (void **)&CreateEventA_orig);
    iat_hook(exe, "KERNEL32.dll", "CreateEventW", CreateEventW_hook, (void **)&CreateEventW_orig);
    iat_hook(exe, "KERNEL32.dll", "CloseHandle", CloseHandle_hook, (void **)&CloseHandle_orig);
    iat_hook(exe, "KERNEL32.dll", "ReleaseMutex", ReleaseMutex_hook, (void **)&ReleaseMutex_orig);
    iat_hook(exe, "KERNEL32.dll", "ReleaseSemaphore", ReleaseSemaphore_hook, (void **)&ReleaseSemaphore_orig);
    g_on = 1;
    shim_log("sync statistics on (W3SIM_PROFILE=4)");
}

void syncstat_report(double secs) {
    if (!g_on)
        return;
    struct { LONG key, n; } top[8];
    memset(top, 0, sizeof top);
    LONG total = 0;
    for (int i = 0; i < SITES; i++) {
        LONG n = InterlockedExchange(&g_sites[i].n, 0), key = g_sites[i].key;
        total += n;
        for (int j = 0; j < 8; j++)
            if (n > top[j].n) {
                memmove(&top[j + 1], &top[j], (7 - j) * sizeof top[0]);
                top[j].key = key;
                top[j].n = n;
                break;
            }
    }
    shim_log("sync calls: %.0f/s; on tracked manual-reset events: SetEvent %.0f/s (%.0f/s redundant), ResetEvent %.0f/s (%.0f/s redundant)",
             total / secs, InterlockedExchange(&g_tracked[S_SET], 0) / secs, InterlockedExchange(&g_redundant[S_SET], 0) / secs,
             InterlockedExchange(&g_tracked[S_RESET], 0) / secs, InterlockedExchange(&g_redundant[S_RESET], 0) / secs);
    for (int j = 0; j < 8 && top[j].n; j++)
        shim_log("  %s from exe+%#lx: %.0f/s", g_names[(top[j].key - 1) % 8], (unsigned long)((top[j].key - 1) / 8),
                 top[j].n / secs);
}
