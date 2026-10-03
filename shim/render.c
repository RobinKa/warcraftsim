/* W3SIM_DRAW=0: nothing is drawn. The game still builds every frame (the simulation, its animations
 * and geometry run as before), but the Direct3D 9 device's draw calls, clears and presents return at
 * once: under Wine they become OpenGL that Mesa rasterizes in software, ~18% of a training game's main
 * thread, for pixels nobody looks at. (Videos are rendered from replays in a process of their own, which
 * draws.) Mesa's own no-op driver (GALLIUM_NOOP=1) left the game without a first frame.
 *
 * The game loads d3d9.dll itself: Direct3DCreate9 is hooked, then IDirect3D9::CreateDevice, then the
 * device's methods (MinHook on the functions the vtables point to).
 */
#include "w3shim.h"

#include "MinHook.h"

typedef HRESULT(WINAPI *CreateDeviceFn)(void *, UINT, DWORD, HWND, DWORD, void *, void ***);
typedef void *(WINAPI *Create9Fn)(UINT);

static Create9Fn Create9_orig;
static CreateDeviceFn CreateDevice_orig;
static volatile LONG g_skipped;

/* IDirect3DDevice9: 17 Present, 43 Clear, 81 DrawPrimitive, 82 DrawIndexedPrimitive, 83 DrawPrimitiveUP,
 * 84 DrawIndexedPrimitiveUP (stdcall: each stub pops its own arguments) */
static HRESULT WINAPI Present_stub(void *d, const void *a, const void *b, HWND w, const void *r) {
    return 0;
}
static HRESULT WINAPI Clear_stub(void *d, DWORD n, const void *rects, DWORD flags, DWORD color, float z, DWORD stencil) {
    return 0;
}
static HRESULT WINAPI Draw_stub(void *d, DWORD type, UINT start, UINT count) {
    InterlockedIncrement(&g_skipped);
    return 0;
}
static HRESULT WINAPI DrawIndexed_stub(void *d, DWORD type, INT base, UINT min, UINT n, UINT start, UINT count) {
    InterlockedIncrement(&g_skipped);
    return 0;
}
static HRESULT WINAPI DrawUP_stub(void *d, DWORD type, UINT count, const void *data, UINT stride) {
    InterlockedIncrement(&g_skipped);
    return 0;
}
static HRESULT WINAPI DrawIndexedUP_stub(void *d, DWORD type, UINT min, UINT n, UINT count, const void *idx, DWORD fmt,
                                         const void *data, UINT stride) {
    InterlockedIncrement(&g_skipped);
    return 0;
}

long render_skipped(void) {
    return InterlockedExchange(&g_skipped, 0);
}

static void hook_slot(void **vtbl, int i, void *stub, const char *name) {
    void *orig = NULL;
    MH_STATUS s = MH_CreateHook(vtbl[i], stub, &orig);
    if (s == MH_OK)
        s = MH_EnableHook(vtbl[i]);
    if (s != MH_OK && s != MH_ERROR_ALREADY_CREATED)
        shim_log("render: %s not hooked (%s)", name, MH_StatusToString(s));
}

static HRESULT WINAPI CreateDevice_hook(void *d3d, UINT adapter, DWORD type, HWND w, DWORD flags, void *params,
                                        void ***device) {
    HRESULT r = CreateDevice_orig(d3d, adapter, type, w, flags, params, device);
    static int done;
    if (r == 0 && device && *device && !done) {
        done = 1;
        void **vtbl = **device;
        hook_slot(vtbl, 17, (void *)Present_stub, "Present");
        hook_slot(vtbl, 43, (void *)Clear_stub, "Clear");
        hook_slot(vtbl, 81, (void *)Draw_stub, "DrawPrimitive");
        hook_slot(vtbl, 82, (void *)DrawIndexed_stub, "DrawIndexedPrimitive");
        hook_slot(vtbl, 83, (void *)DrawUP_stub, "DrawPrimitiveUP");
        hook_slot(vtbl, 84, (void *)DrawIndexedUP_stub, "DrawIndexedPrimitiveUP");
        shim_log("render: the device draws nothing (W3SIM_DRAW=0)");
    }
    return r;
}

static void *WINAPI Create9_hook(UINT version) {
    void **d3d = Create9_orig(version);
    static int done;
    if (d3d && !done) {
        done = 1;
        void **vtbl = *(void ***)d3d;
        if (MH_CreateHook(vtbl[16], (void *)CreateDevice_hook, (void **)&CreateDevice_orig) != MH_OK ||
            MH_EnableHook(vtbl[16]) != MH_OK)
            shim_log("render: CreateDevice not hooked");
    }
    return d3d;
}

void render_install(int draw) {
    if (draw)
        return;
    if (MH_Initialize() != MH_OK && MH_Initialize() != MH_ERROR_ALREADY_INITIALIZED) {
        shim_log("render: MinHook init failed");
        return;
    }
    HMODULE d3d9 = LoadLibraryA("d3d9.dll");
    void *create = d3d9 ? (void *)GetProcAddress(d3d9, "Direct3DCreate9") : NULL;
    if (!create || MH_CreateHook(create, (void *)Create9_hook, (void **)&Create9_orig) != MH_OK ||
        MH_EnableHook(create) != MH_OK) {
        shim_log("render: Direct3DCreate9 not hooked: drawing as usual");
        return;
    }
    shim_log("render: drawing off (W3SIM_DRAW=0)");
}
