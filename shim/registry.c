/* Registry access of the game.
 *
 * W3SIM_TRACE_REG=1 logs the first registry calls (to find per-step registry traffic).
 */
#include "w3shim.h"

#include "MinHook.h"

#include <stdio.h>
#include <string.h>
#include <winternl.h>

static LSTATUS(WINAPI *RegOpenKeyExA_orig)(HKEY, LPCSTR, DWORD, REGSAM, PHKEY);
static LSTATUS(WINAPI *RegOpenKeyExW_orig)(HKEY, LPCWSTR, DWORD, REGSAM, PHKEY);
static LSTATUS(WINAPI *RegCreateKeyExA_orig)(HKEY, LPCSTR, DWORD, LPSTR, DWORD, REGSAM, const LPSECURITY_ATTRIBUTES,
                                              PHKEY, LPDWORD);
static LSTATUS(WINAPI *RegCreateKeyExW_orig)(HKEY, LPCWSTR, DWORD, LPWSTR, DWORD, REGSAM, const LPSECURITY_ATTRIBUTES,
                                              PHKEY, LPDWORD);
static LSTATUS(WINAPI *RegQueryValueExA_orig)(HKEY, LPCSTR, LPDWORD, LPDWORD, LPBYTE, LPDWORD);
static LSTATUS(WINAPI *RegQueryValueExW_orig)(HKEY, LPCWSTR, LPDWORD, LPDWORD, LPBYTE, LPDWORD);

static volatile LONG g_trace_left;

static void trace(const char *fmt, ...) {
    if (sync_count() < 20 || InterlockedDecrement(&g_trace_left) < 0)
        return; /* only while the game is being stepped */
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    _vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    buf[sizeof buf - 1] = 0;
    shim_log("reg: %s", buf);
}

static LSTATUS WINAPI RegOpenKeyExA_hook(HKEY k, LPCSTR sub, DWORD o, REGSAM s, PHKEY out) {
    LSTATUS r = RegOpenKeyExA_orig(k, sub, o, s, out);
    trace("OpenA %p %s -> %ld %p", k, sub ? sub : "(null)", r, out ? *out : 0);
    return r;
}
static LSTATUS WINAPI RegOpenKeyExW_hook(HKEY k, LPCWSTR sub, DWORD o, REGSAM s, PHKEY out) {
    LSTATUS r = RegOpenKeyExW_orig(k, sub, o, s, out);
    trace("OpenW %p %ls -> %ld %p", k, sub ? sub : L"(null)", r, out ? *out : 0);
    return r;
}
static LSTATUS WINAPI RegCreateKeyExA_hook(HKEY k, LPCSTR sub, DWORD r0, LPSTR c, DWORD o, REGSAM s,
                                           const LPSECURITY_ATTRIBUTES sa, PHKEY out, LPDWORD d) {
    LSTATUS r = RegCreateKeyExA_orig(k, sub, r0, c, o, s, sa, out, d);
    trace("CreateA %p %s -> %ld %p", k, sub ? sub : "(null)", r, out ? *out : 0);
    return r;
}
static LSTATUS WINAPI RegCreateKeyExW_hook(HKEY k, LPCWSTR sub, DWORD r0, LPWSTR c, DWORD o, REGSAM s,
                                           const LPSECURITY_ATTRIBUTES sa, PHKEY out, LPDWORD d) {
    LSTATUS r = RegCreateKeyExW_orig(k, sub, r0, c, o, s, sa, out, d);
    trace("CreateW %p %ls -> %ld %p", k, sub ? sub : L"(null)", r, out ? *out : 0);
    return r;
}
static LSTATUS WINAPI RegQueryValueExA_hook(HKEY k, LPCSTR name, LPDWORD res, LPDWORD type, LPBYTE data, LPDWORD n) {
    LSTATUS r = RegQueryValueExA_orig(k, name, res, type, data, n);
    trace("QueryA %p %s -> %ld", k, name ? name : "(default)", r);
    return r;
}
static LSTATUS WINAPI RegQueryValueExW_hook(HKEY k, LPCWSTR name, LPDWORD res, LPDWORD type, LPBYTE data, LPDWORD n) {
    LSTATUS r = RegQueryValueExW_orig(k, name, res, type, data, n);
    trace("QueryW %p %ls -> %ld", k, name ? name : L"(default)", r);
    return r;
}

typedef NTSTATUS(NTAPI *NtQueryValueKeyFn)(HANDLE, PUNICODE_STRING, int, PVOID, ULONG, PULONG);
static NtQueryValueKeyFn NtQueryValueKey_orig;

static NTSTATUS NTAPI NtQueryValueKey_hook(HANDLE key, PUNICODE_STRING name, int cls, PVOID info, ULONG len,
                                           PULONG out) {
    if (sync_count() >= 20 && g_trace_left > 0) {
        void *ret = __builtin_return_address(0);
        HMODULE m = NULL;
        char mn[MAX_PATH] = "?";
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               (LPCSTR)ret, &m))
            GetModuleFileNameA(m, mn, sizeof mn);
        char *b = strrchr(mn, '\\');
        trace("NtQueryValueKey from %s+%#lx thread %lu: %.*ls", b ? b + 1 : mn, (unsigned long)((BYTE *)ret - (BYTE *)m),
              GetCurrentThreadId(), name ? name->Length / 2 : 0, name ? name->Buffer : L"");
    }
    return NtQueryValueKey_orig(key, name, cls, info, len, out);
}

void registry_install(void) {
    char buf[16];
    if (!GetEnvironmentVariableA("W3SIM_TRACE_REG", buf, sizeof buf) || buf[0] != '1')
        return;
    g_trace_left = 400;
    HMODULE exe = (HMODULE)g_base;
    iat_hook(exe, "ADVAPI32.dll", "RegOpenKeyExA", RegOpenKeyExA_hook, (void **)&RegOpenKeyExA_orig);
    iat_hook(exe, "ADVAPI32.dll", "RegOpenKeyExW", RegOpenKeyExW_hook, (void **)&RegOpenKeyExW_orig);
    iat_hook(exe, "ADVAPI32.dll", "RegCreateKeyExA", RegCreateKeyExA_hook, (void **)&RegCreateKeyExA_orig);
    iat_hook(exe, "ADVAPI32.dll", "RegCreateKeyExW", RegCreateKeyExW_hook, (void **)&RegCreateKeyExW_orig);
    iat_hook(exe, "ADVAPI32.dll", "RegQueryValueExA", RegQueryValueExA_hook, (void **)&RegQueryValueExA_orig);
    iat_hook(exe, "ADVAPI32.dll", "RegQueryValueExW", RegQueryValueExW_hook, (void **)&RegQueryValueExW_orig);
    void *nt = (void *)GetProcAddress(GetModuleHandleA("ntdll.dll"), "NtQueryValueKey");
    MH_Initialize();
    if (!nt || MH_CreateHook(nt, (void *)NtQueryValueKey_hook, (void **)&NtQueryValueKey_orig) != MH_OK ||
        MH_EnableHook(nt) != MH_OK)
        shim_log("NtQueryValueKey hook failed");
    shim_log("registry tracing on");
}
