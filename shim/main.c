/* w3shim.dll entry point: configuration, logging and hook installation.
 *
 * The launcher loads this DLL into the suspended game process; every hook is installed inside
 * DllMain (plain memory writes, no loader-lock hazards) so the game's first instruction already
 * runs hooked. Anything that needs other libraries (sockets) happens lazily later.
 */
#include "w3shim.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

BYTE *g_base;
static HANDLE g_log = INVALID_HANDLE_VALUE;
static CRITICAL_SECTION g_log_lock;

void shim_log(const char *fmt, ...) {
    if (g_log == INVALID_HANDLE_VALUE)
        return;
    char buf[1024];
    SYSTEMTIME t;
    GetLocalTime(&t);
    int len = _snprintf(buf, sizeof buf, "%02d:%02d:%02d.%03d ", t.wHour, t.wMinute, t.wSecond, t.wMilliseconds);
    va_list ap;
    va_start(ap, fmt);
    int n = _vsnprintf(buf + len, sizeof buf - len - 2, fmt, ap);
    va_end(ap);
    len = n < 0 ? (int)sizeof buf - 2 : len + n;
    buf[len++] = '\n';
    DWORD written;
    EnterCriticalSection(&g_log_lock);
    WriteFile(g_log, buf, len, &written, NULL);
    LeaveCriticalSection(&g_log_lock);
}

static double env_double(const char *name, double fallback) {
    char buf[64];
    return GetEnvironmentVariableA(name, buf, sizeof buf) ? atof(buf) : fallback;
}

BOOL WINAPI DllMain(HINSTANCE inst, DWORD reason, LPVOID reserved) {
    (void)reserved;
    if (reason != DLL_PROCESS_ATTACH)
        return TRUE;
    DisableThreadLibraryCalls(inst);
    InitializeCriticalSection(&g_log_lock);
    g_base = (BYTE *)GetModuleHandleA(NULL);

    char path[MAX_PATH];
    if (GetEnvironmentVariableA("W3SIM_LOG", path, sizeof path))
        g_log = CreateFileA(path, GENERIC_WRITE, FILE_SHARE_READ, NULL, CREATE_ALWAYS, 0, NULL);
    shim_log("w3shim %s attached: pid=%lu base=%p cmdline=%s", W3SHIM_VERSION, GetCurrentProcessId(), g_base,
             GetCommandLineA());

    clock_install(env_double("W3SIM_SPEED", 1.0), (DWORD)env_double("W3SIM_WAIT_FLOOR", 0));
    sync_install();
    turbo_install((int)env_double("W3SIM_TURBO_MS", 0));
    render_install((int)env_double("W3SIM_DRAW", 1));
    registry_install();
    audio_install();
    return TRUE;
}
