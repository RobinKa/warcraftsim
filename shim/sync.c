/* Step sync with the Python controller.
 *
 * The map harness writes its observation (PreloadGenEnd) and then reads its actions
 * (Preloader). When the game opens the action file we freeze the virtual clock, tell the
 * controller that an observation is ready and block until it answers; by then the action file
 * has been written. Protocol (one line each way, over TCP 127.0.0.1:W3SIM_PORT):
 *
 *   shim -> controller:  "OBS <n>\n"          n = sync counter
 *   controller -> shim:  "GO [speed=<x>] [turbo=<ms>] [nosync]\n"
 */
#include "w3shim.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

static HANDLE(WINAPI *CreateFileW_orig)(LPCWSTR, DWORD, DWORD, LPSECURITY_ATTRIBUTES, DWORD, DWORD, HANDLE);
static wchar_t g_sync_suffix[MAX_PATH] = L"w3sim\\act.txt";
static size_t g_sync_suffix_len;
static int g_port;
static SOCKET g_sock = INVALID_SOCKET;
static volatile LONG g_enabled;
static LONG g_counter;
static CRITICAL_SECTION g_sync_lock;
static volatile LONGLONG g_wait_ticks;

LONGLONG sync_wait_ticks(void) {
    return g_wait_ticks;
}

void sync_reset_wait_ticks(void) {
    g_wait_ticks = 0;
}

static int ends_with_ci(const wchar_t *s, const wchar_t *suffix, size_t suffix_len) {
    size_t n = wcslen(s);
    if (n < suffix_len)
        return 0;
    const wchar_t *tail = s + n - suffix_len;
    for (size_t i = 0; i < suffix_len; i++) {
        wchar_t a = towlower(tail[i]), b = towlower(suffix[i]);
        if (a == L'/')
            a = L'\\';
        if (a != b)
            return 0;
    }
    return 1;
}

static int connect_controller(void) {
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
        return 0;
    for (int attempt = 0; attempt < 200; attempt++) {
        SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        struct sockaddr_in addr = {0};
        addr.sin_family = AF_INET;
        addr.sin_port = htons((u_short)g_port);
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (connect(s, (struct sockaddr *)&addr, sizeof addr) == 0) {
            BOOL one = TRUE;
            setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char *)&one, sizeof one);
            g_sock = s;
            shim_log("connected to controller on port %d", g_port);
            return 1;
        }
        closesocket(s);
        Sleep(50);
    }
    shim_log("could not connect to controller on port %d; step sync disabled", g_port);
    return 0;
}

static int send_all(const char *buf, int len) {
    while (len > 0) {
        int n = send(g_sock, buf, len, 0);
        if (n <= 0)
            return 0;
        buf += n;
        len -= n;
    }
    return 1;
}

static int recv_line(char *buf, int cap) {
    int len = 0;
    while (len < cap - 1) {
        int n = recv(g_sock, buf + len, 1, 0);
        if (n <= 0)
            return 0;
        if (buf[len] == '\n')
            break;
        len++;
    }
    buf[len] = 0;
    return 1;
}

static void handle_go(const char *line) {
    const char *p = strstr(line, "speed=");
    if (p)
        clock_set_speed(atof(p + 6));
    p = strstr(line, "turbo=");
    if (p)
        turbo_set(atoi(p + 6));
    if (strstr(line, "nosync")) {
        InterlockedExchange(&g_enabled, 0);
        shim_log("controller disabled step sync");
    }
}

static void sync_point(void) {
    EnterCriticalSection(&g_sync_lock);
    if (g_sock == INVALID_SOCKET && !connect_controller()) {
        InterlockedExchange(&g_enabled, 0);
        LeaveCriticalSection(&g_sync_lock);
        return;
    }
    clock_freeze(1);
    LONGLONG t0 = real_qpc_ticks();
    char msg[64], reply[256];
    int n = _snprintf(msg, sizeof msg, "OBS %ld\n", ++g_counter);
    if (!send_all(msg, n) || !recv_line(reply, sizeof reply)) {
        shim_log("controller connection lost; exiting");
        ExitProcess(3);
    }
    if (strncmp(reply, "QUIT", 4) == 0) {
        shim_log("controller requested exit");
        ExitProcess(0);
    }
    handle_go(reply);
    g_wait_ticks += real_qpc_ticks() - t0;
    clock_freeze(0);
    LeaveCriticalSection(&g_sync_lock);
}

/* Preload(name) checks the disk for a local file of that name (Allow Local Files is on). The
 * harness preloads thousands of observation tokens per step, which are never files, so answer
 * "not found" for bare numbers and single letters without a system call. */
static DWORD(WINAPI *GetFileAttributesW_orig)(LPCWSTR);

static int is_token_name(LPCWSTR name) {
    const wchar_t *base = name;
    for (const wchar_t *c = name; *c; c++)
        if (*c == L'\\' || *c == L'/')
            base = c + 1;
    if (!*base)
        return 0;
    if (base[1] == 0 && base[0] >= L'A' && base[0] <= L'Z')
        return 1;
    const wchar_t *d = base[0] == L'-' ? base + 1 : base;
    if (!*d)
        return 0;
    for (; *d; d++)
        if (*d < L'0' || *d > L'9')
            return 0;
    return 1;
}

static DWORD WINAPI GetFileAttributesW_hook(LPCWSTR name) {
    if (name && is_token_name(name)) {
        SetLastError(ERROR_FILE_NOT_FOUND);
        return INVALID_FILE_ATTRIBUTES;
    }
    return GetFileAttributesW_orig(name);
}

static HANDLE WINAPI CreateFileW_hook(LPCWSTR name, DWORD access, DWORD share, LPSECURITY_ATTRIBUTES sa,
                                      DWORD disposition, DWORD flags, HANDLE templ) {
    if (g_enabled && name && (access & GENERIC_READ) && !(access & GENERIC_WRITE) &&
        ends_with_ci(name, g_sync_suffix, g_sync_suffix_len))
        sync_point();
    return CreateFileW_orig(name, access, share, sa, disposition, flags, templ);
}

void sync_install(void) {
    char buf[MAX_PATH];
    InitializeCriticalSection(&g_sync_lock);
    if (!iat_hook((HMODULE)g_base, "KERNEL32.dll", "GetFileAttributesW", GetFileAttributesW_hook,
                  (void **)&GetFileAttributesW_orig))
        shim_log("GetFileAttributesW import not found");
    if (GetEnvironmentVariableA("W3SIM_SYNC_FILE", buf, sizeof buf))
        MultiByteToWideChar(CP_UTF8, 0, buf, -1, g_sync_suffix, MAX_PATH);
    for (wchar_t *c = g_sync_suffix; *c; c++)
        *c = (*c == L'/') ? L'\\' : towlower(*c);
    g_sync_suffix_len = wcslen(g_sync_suffix);
    if (!GetEnvironmentVariableA("W3SIM_PORT", buf, sizeof buf)) {
        shim_log("W3SIM_PORT unset: free running, no step sync");
        return;
    }
    g_port = atoi(buf);
    if (!iat_hook((HMODULE)g_base, "KERNEL32.dll", "CreateFileW", CreateFileW_hook, (void **)&CreateFileW_orig)) {
        shim_log("CreateFileW import not found; step sync unavailable");
        return;
    }
    InterlockedExchange(&g_enabled, 1);
    shim_log("step sync on '%ls' via port %d", g_sync_suffix, g_port);
}
