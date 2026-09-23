/* Step sync with the Python controller.
 *
 * The map harness writes its observation (PreloadGenEnd) and then reads its actions
 * (Preloader). When the game opens the action file we freeze the virtual clock, tell the
 * controller that an observation is ready and block until it answers; by then the action file
 * has been written. Protocol (one line each way, over TCP 127.0.0.1:W3SIM_PORT):
 *
 *   shim -> controller:  "OBS <n>\n"          n = sync counter
 *   controller -> shim:  "GO [speed=<x>] [turbo=<ms>] [frame=<ms>] [capture=<0|1>] [nosync] [A <n> <int>...]\n"
 *
 * The sync point is the harness's call GetPlayerTechMaxAllowed(neutral passive, MBOX - 1), which we
 * hook: it blocks until "GO" and returns n; keys MBOX + 1 + i then return the i-th command
 * integer from the "A" list. (The harness used to read an action file with Preloader, whose open
 * of w3sim\act.txt is still a sync point, but that compiled the file as JASS on every step.)
 *
 * frame=<ms> switches the clock to frame-stepped mode (0: back to real time); with capture=1
 * every presented frame is reported ("FRAME <n>\n", answered by any line once the controller has
 * grabbed the window), so a video gets exactly one image per frame.
 */
#include "w3shim.h"

#include "MinHook.h"

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
static volatile LONG g_capture;
static LONG g_frames;

/* Step phases (profiling): after GO the game runs the actions and simulates until the harness's
 * next step, which serializes the observation (Preload tokens), writes it (PreloadGenEnd) and
 * reads the next actions (Preloader -> the sync point). */
static LONGLONG g_t_go, g_t_tokens, g_t_write;
static int g_phase;
static LONGLONG g_ph[3], g_ph_tokens, g_ph_steps, g_t_files, g_n_files;

void sync_phases(double secs, char *out, int cap) {
    double f = 1000.0 / (double)real_qpc_freq() / secs;
    _snprintf(out, cap, "steps=%.1f go->obs.txt=%.1f ms obs.txt->sync=%.1f ms file calls=%.1f/step (%.1f ms)",
              g_ph_steps / secs, (g_ph[0] + g_ph[1]) * f, g_ph[2] * f,
              g_ph_steps ? (double)g_n_files / g_ph_steps : 0.0, g_t_files * f);
    g_ph[0] = g_ph[1] = g_ph[2] = g_ph_tokens = g_ph_steps = g_t_files = g_n_files = 0;
}

void frame_capture_set(int on) {
    InterlockedExchange(&g_capture, on ? 1 : 0);
}

int frame_capture_get(void) {
    return (int)g_capture;
}

long sync_frame_count(void) {
    return g_frames;
}

long sync_count(void) {
    return g_counter;
}

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

/* One line from the controller. Buffered: under Wine every recv() is a wineserver round trip, so
 * reading byte by byte cost milliseconds per step. The controller sends one line per request. */
static char g_rbuf[160 * 1024]; /* a GO line carries up to MBOX_MAX command integers */
static int g_rlen;

static int recv_line(char *buf, int cap) {
    for (;;) {
        char *nl = memchr(g_rbuf, '\n', g_rlen);
        if (nl) {
            int n = (int)(nl - g_rbuf);
            int k = n < cap - 1 ? n : cap - 1;
            memcpy(buf, g_rbuf, k);
            buf[k] = 0;
            g_rlen -= n + 1;
            memmove(g_rbuf, nl + 1, g_rlen);
            return 1;
        }
        if (g_rlen == (int)sizeof g_rbuf)
            g_rlen = 0; /* an overlong line: drop it */
        int got = recv(g_sock, g_rbuf + g_rlen, (int)sizeof g_rbuf - g_rlen, 0);
        if (got <= 0)
            return 0;
        g_rlen += got;
    }
}

#define RVA_GET_TECH_MAX 0x94be0 /* cdecl GetPlayerTechMaxAllowed(HPLAYER, int techid) native */
#define MBOX 1048576
#define MBOX_MAX 8000
static int g_mbox[MBOX_MAX + 1]; /* [0] = count, then the command integers */

static void parse_actions(const char *p) {
    char *end;
    long n = strtol(p, &end, 10);
    if (n < 0)
        n = 0;
    if (n > MBOX_MAX)
        n = MBOX_MAX;
    for (long i = 0; i < n; i++)
        g_mbox[i + 1] = (int)strtol(end, &end, 10);
    g_mbox[0] = (int)n;
}

static void handle_go(const char *line) {
    const char *a = strstr(line, " A ");
    g_mbox[0] = 0;
    if (a)
        parse_actions(a + 3);
    const char *p = strstr(line, "speed=");
    if (p)
        clock_set_speed(atof(p + 6));
    p = strstr(line, "turbo=");
    if (p)
        turbo_set(atoi(p + 6));
    p = strstr(line, "frame=");
    if (p)
        clock_set_frame_step(atof(p + 6) / 1000.0);
    p = strstr(line, "capture=");
    if (p) {
        frame_capture_set(atoi(p + 8));
        shim_log("frame capture %s", g_capture ? "on" : "off");
    }
    if (strstr(line, "nosync")) {
        InterlockedExchange(&g_enabled, 0);
        shim_log("controller disabled step sync");
    }
}

static void sync_point(void) {
    EnterCriticalSection(&g_sync_lock);
    if (g_t_go) {
        LONGLONG now = real_qpc_ticks();
        LONGLONG t1 = g_phase >= 1 ? g_t_tokens : now, t2 = g_phase >= 2 ? g_t_write : now;
        g_ph[0] += t1 - g_t_go;
        g_ph[1] += t2 - t1;
        g_ph[2] += now - t2;
        g_ph_steps++;
    }
    if (g_sock == INVALID_SOCKET && !connect_controller()) {
        InterlockedExchange(&g_enabled, 0);
        LeaveCriticalSection(&g_sync_lock);
        return;
    }
    clock_freeze(1);
    LONGLONG t0 = real_qpc_ticks();
    char msg[64];
    static char reply[sizeof g_rbuf];
    const char *obs;
    int obs_len = obs_take(&obs);
    int n = obs_len >= 0 ? _snprintf(msg, sizeof msg, "OBS %ld %d\n", ++g_counter, obs_len)
                         : _snprintf(msg, sizeof msg, "OBS %ld\n", ++g_counter);
    int sent = send_all(msg, n) && (obs_len <= 0 || send_all(obs, obs_len));
    obs_reset();
    if (!sent || !recv_line(reply, sizeof reply)) {
        shim_log("controller connection lost; exiting");
        ExitProcess(3);
    }
    if (strncmp(reply, "QUIT", 4) == 0) {
        shim_log("controller requested exit");
        ExitProcess(0);
    }
    handle_go(reply);
    g_t_go = real_qpc_ticks();
    g_phase = 0;
    g_wait_ticks += g_t_go - t0;
    clock_freeze(0);
    LeaveCriticalSection(&g_sync_lock);
}

void sync_frame(void) {
    if (!g_capture || g_sock == INVALID_SOCKET)
        return;
    EnterCriticalSection(&g_sync_lock);
    /* the frame covers one frame step of game time (or the virtual time since the last frame) */
    static int64_t last_virt;
    int64_t now = clock_virtual_ticks();
    double secs = clock_frame_seconds();
    if (secs <= 0)
        secs = last_virt ? (double)(now - last_virt) / (double)real_qpc_freq() : 0.0;
    last_virt = now;
    char msg[96], reply[64], fmt[48];
    char *pcm;
    int pcm_len, n;
    if (audio_frame(secs, &pcm, &pcm_len, fmt, sizeof fmt))
        n = _snprintf(msg, sizeof msg, "FRAME %ld %d %s\n", ++g_frames, pcm_len, fmt);
    else
        n = _snprintf(msg, sizeof msg, "FRAME %ld\n", ++g_frames), pcm_len = 0;
    if (!send_all(msg, n) || (pcm_len > 0 && !send_all(pcm, pcm_len)) || !recv_line(reply, sizeof reply)) {
        shim_log("controller connection lost; exiting");
        ExitProcess(3);
    }
    LeaveCriticalSection(&g_sync_lock);
}

/* Preload(name) checks the disk for a local file of that name (Allow Local Files is on). The
 * harness preloads thousands of observation tokens per step, which are never files, so answer
 * "not found" for bare numbers and single letters without a system call. */
static DWORD(WINAPI *GetFileAttributesW_orig)(LPCWSTR);
static LONG g_file_trace = 60; /* W3SIM_TRACE_FILES: log the first file calls while stepping */

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
    if (g_counter > 30 && g_file_trace > 0 && getenv("W3SIM_TRACE_FILES")) {
        g_file_trace--;
        shim_log("file: GetFileAttributesW %ls", name ? name : L"(null)");
    }
    if (name && is_token_name(name)) {
        if (g_phase == 0) {
            g_t_tokens = real_qpc_ticks();
            g_phase = 1;
        }
        g_ph_tokens++;
        SetLastError(ERROR_FILE_NOT_FOUND);
        return INVALID_FILE_ATTRIBUTES;
    }
    LONGLONG t0 = real_qpc_ticks();
    DWORD r = GetFileAttributesW_orig(name);
    g_t_files += real_qpc_ticks() - t0;
    g_n_files++;
    return r;
}

static HANDLE WINAPI CreateFileW_hook(LPCWSTR name, DWORD access, DWORD share, LPSECURITY_ATTRIBUTES sa,
                                      DWORD disposition, DWORD flags, HANDLE templ) {
    if (g_counter > 30 && g_file_trace > 0 && getenv("W3SIM_TRACE_FILES")) {
        g_file_trace--;
        shim_log("file: CreateFileW %ls access=%#lx", name ? name : L"(null)", access);
    }
    if (g_enabled && name && (access & GENERIC_READ) && !(access & GENERIC_WRITE) &&
        ends_with_ci(name, g_sync_suffix, g_sync_suffix_len))
        sync_point();
    else if (name && (access & GENERIC_WRITE) && g_phase < 2 && ends_with_ci(name, L"obs.txt", 7)) {
        g_t_tokens = g_t_write = real_qpc_ticks();
        g_phase = 2;
    }
    LONGLONG t0 = real_qpc_ticks();
    HANDLE h = CreateFileW_orig(name, access, share, sa, disposition, flags, templ);
    g_t_files += real_qpc_ticks() - t0;
    g_n_files++;
    return h;
}

typedef int(__cdecl *GetTechMaxFn)(int player, int techid);
static GetTechMaxFn GetTechMax_orig;

static int __cdecl GetTechMax_hook(int player, int techid) {
    if (techid >= MBOX - 1 && techid <= MBOX + MBOX_MAX) {
        if (techid == MBOX - 1) {
            if (g_enabled)
                sync_point();
            return g_mbox[0];
        }
        return g_mbox[techid - MBOX];
    }
    return GetTechMax_orig(player, techid);
}

static void install_mailbox(void) {
    BYTE *f = g_base + RVA_GET_TECH_MAX;
    if (!(f[0] == 0x55 && f[1] == 0x8b && f[2] == 0xec)) {
        shim_log("mailbox: unsupported executable, not installed");
        return;
    }
    MH_STATUS st = MH_Initialize();
    if (st != MH_OK && st != MH_ERROR_ALREADY_INITIALIZED) {
        shim_log("mailbox: MinHook init failed");
        return;
    }
    if (MH_CreateHook(f, (void *)GetTechMax_hook, (void **)&GetTechMax_orig) != MH_OK || MH_EnableHook(f) != MH_OK)
        shim_log("mailbox: hook failed");
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
    install_mailbox();
    obs_install();
    InterlockedExchange(&g_enabled, 1);
    shim_log("step sync on the mailbox and '%ls' via port %d", g_sync_suffix, g_port);
}
