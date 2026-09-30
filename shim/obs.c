/* Observation capture: the harness's Preload tokens go to memory instead of a file.
 *
 * The harness serializes an observation as Preload(token) calls followed by
 * PreloadGenEnd("w3sim\obs.txt"). Natively every Preload checks the disk for a file of that name
 * and PreloadGenEnd writes all of them out as a script, which the controller then reads back:
 * dozens of file system calls per step, each a wineserver round trip under Wine (plus a registry
 * lookup of the Documents folder). With these hooks the tokens are collected here and sent to
 * the controller with the step sync (sync.c). Preloads outside observations are collected too and
 * then dropped (they only warm the file cache). Unsupported executable: not installed, the
 * harness falls back to the file.
 */
#include "w3shim.h"

#include "MinHook.h"

#include <stdlib.h>
#include <string.h>

#define RVA_JSTRING 0x69010         /* thiscall const char *(HSTRING): a JASS string argument */
#define RVA_PRELOAD 0xa4410         /* cdecl Preload(string) native */
#define RVA_PRELOAD_GEN_END 0xa4510 /* cdecl PreloadGenEnd(string) native */

typedef const char *(__attribute__((thiscall)) *JStringFn)(int handle);
typedef void(__cdecl *NativeStrFn)(int handle);

static JStringFn g_jstring;
static NativeStrFn Preload_orig, PreloadGenEnd_orig;
static int g_installed;

static char *g_buf;
static int g_len, g_cap;
static int g_ready; /* a complete observation is in the buffer */

static void append(const char *s, int n) {
    if (g_len + n + 1 > g_cap) {
        int cap = g_cap ? g_cap : 65536;
        while (g_len + n + 1 > cap)
            cap *= 2;
        char *b = realloc(g_buf, cap);
        if (!b)
            return;
        g_buf = b;
        g_cap = cap;
    }
    memcpy(g_buf + g_len, s, n);
    g_len += n;
    g_buf[g_len++] = '\n';
}

static void __cdecl Preload_hook(int h) {
    const char *s = g_jstring(h);
    if (s && *s && !g_ready) {
        units_verify_token(s);
        append(s, (int)strlen(s));
    }
}

/* tokens written by the shim itself (units.c: lines, each ending in a newline), as if the harness had sent them */
void obs_bytes(const char *s, int n) {
    if (g_installed && !g_ready && n > 0) {
        append(s, n - 1); /* (append adds the last newline) */
    }
}

static int is_obs_file(const char *s) {
    size_t n = strlen(s);
    return n >= 13 && _stricmp(s + n - 13, "w3sim\\obs.txt") == 0;
}

static void __cdecl PreloadGenEnd_hook(int h) {
    const char *s = h ? g_jstring(h) : NULL;
    if (s && is_obs_file(s)) {
        g_ready = 1; /* sent with the next step sync */
        units_mark(3);
        return;
    }
    PreloadGenEnd_orig(h);
}

int obs_capture_on(void) {
    return g_installed;
}

int obs_take(const char **data) {
    if (!g_ready)
        return -1;
    *data = g_buf;
    return g_len;
}

void obs_reset(void) {
    g_len = 0;
    g_ready = 0;
}

void obs_install(void) {
    BYTE *pre = g_base + RVA_PRELOAD, *end = g_base + RVA_PRELOAD_GEN_END;
    /* both natives start with `push ebp; mov ebp, esp; mov ecx, [ebp+8]; call jstring` */
    if (memcmp(pre, "\x55\x8b\xec\x8b\x4d\x08\xe8", 7) != 0 || memcmp(end, "\x55\x8b\xec\x8b\x4d\x08", 6) != 0 ||
        pre + 11 + *(int32_t *)(pre + 7) != g_base + RVA_JSTRING) {
        shim_log("obs capture: unsupported executable, observations go through the file");
        return;
    }
    g_jstring = (JStringFn)(g_base + RVA_JSTRING);
    MH_STATUS st = MH_Initialize();
    if ((st != MH_OK && st != MH_ERROR_ALREADY_INITIALIZED) ||
        MH_CreateHook(pre, (void *)Preload_hook, (void **)&Preload_orig) != MH_OK ||
        MH_CreateHook(end, (void *)PreloadGenEnd_hook, (void **)&PreloadGenEnd_orig) != MH_OK ||
        MH_EnableHook(pre) != MH_OK || MH_EnableHook(end) != MH_OK) {
        shim_log("obs capture: hooks failed, observations go through the file");
        return;
    }
    g_installed = 1;
    shim_log("obs capture: in memory");
}
