/* Virtual sound card (W3SIM_AUDIO=1): the game's audio, in step with the virtual clock.
 *
 * The game plays sound through Miles (Mss32.dll), which outputs through DirectSound or waveOut.
 * We hook Miles's imports: DirectSoundCreate fails (so Miles uses waveOut) and the waveOut
 * functions are implemented here. Miles queues buffers of mixed PCM (waveOutWrite); a buffer
 * counts as played once the virtual clock has passed it, and is then handed back to Miles
 * (WOM_DONE) so it mixes the next one. During frame capture (video) every frame consumes exactly
 * one frame's worth of audio and sends it to the controller with the frame (sync.c), so sound
 * and picture stay in step however fast or slow the render runs. Otherwise played audio is
 * dropped. Without W3SIM_AUDIO the game has no audio device and stays silent (training).
 *
 * W3SIM_AUDIO_FREE=1 (videos): the sound card no longer holds the render to real time. Miles runs on its own
 * clock (QueryPerformanceCounter and timeGetTime, real time) and its own threads' timed waits: a timer thread
 * every 5 ms that mixes every 50 ms of that clock, and a thread every 15 ms that services streams. So its
 * imports are hooked as well, once capture starts: its clock becomes the audio played so far (game time), and
 * each piece of audio played wakes those two threads at once and waits for the timer thread's pass to finish.
 * Miles then sees what it sees at real time, in step with the game whatever the render's speed.
 */
#include "w3shim.h"

#include <mmsystem.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FAKE_HWO ((HWAVEOUT)(UINT_PTR)0x57335357)
#define MAX_QUEUE 64

static struct {
    CRITICAL_SECTION lock;
    int enabled, open, trace;
    WAVEFORMATEX fmt;
    DWORD cb_type;
    DWORD_PTR callback, instance;
    WAVEHDR *queue[MAX_QUEUE];
    int head, count;
    DWORD offset;        /* bytes of queue[head] already played */
    double carry;        /* fractional bytes owed to the next advance */
    int64_t last_virt;   /* virtual time of the last idle advance (QPC ticks) */
    LONGLONG underflow;  /* bytes padded with silence (Miles was late) */
    LONGLONG write_latency_sum, write_latency_n; /* queue ahead of each new buffer (while capturing) */
    LONG writes;         /* buffers written by Miles (timeline tracing) */
    DWORD debt;          /* silence padded for late audio, to drop when it arrives (stays in sync) */
    WAVEHDR *done[MAX_QUEUE]; /* finished buffers, handed back to Miles by the device thread */
    int ndone;
    HANDLE wake;
    /* W3SIM_AUDIO_FREE */
    int free_run;
    volatile LONG clock_on;       /* capture has started: Miles's clock is the audio played */
    volatile LONGLONG clock_ticks; /* the audio played so far (QPC ticks) */
    LONGLONG qpc_base;
    DWORD tgt_base;
    HANDLE timer_wake, stream_wake, pass_event;
    volatile LONG pass_req, pass_sent, pass_seen, pass_done; /* each piece's pass of Miles's timer thread */
    int passes_lost;              /* in a row without an answer; many: Miles's threads are not hooked, stop waiting */
    LONG pass_timeouts;
} A;

static FARPROC(WINAPI *GetProcAddress_orig)(HMODULE, LPCSTR);

static void trace(const char *what, unsigned a, unsigned b) {
    if (A.trace > 0) {
        A.trace--;
        shim_log("audio: %s %#x %#x", what, a, b);
    }
}

static void notify(UINT msg, DWORD_PTR p1) {
    switch (A.cb_type) {
    case CALLBACK_FUNCTION:
        ((void(CALLBACK *)(HWAVEOUT, UINT, DWORD_PTR, DWORD_PTR, DWORD_PTR))A.callback)(FAKE_HWO, msg, A.instance,
                                                                                         p1, 0);
        break;
    case CALLBACK_EVENT:
        SetEvent((HANDLE)A.callback);
        break;
    case CALLBACK_THREAD:
        PostThreadMessageA((DWORD)A.callback, msg == WOM_DONE ? MM_WOM_DONE : msg == WOM_OPEN ? MM_WOM_OPEN
                                                                                               : MM_WOM_CLOSE,
                           (WPARAM)FAKE_HWO, (LPARAM)p1);
        break;
    case CALLBACK_WINDOW:
        PostMessageA((HWND)A.callback, msg == WOM_DONE ? MM_WOM_DONE : msg == WOM_OPEN ? MM_WOM_OPEN : MM_WOM_CLOSE,
                     (WPARAM)FAKE_HWO, (LPARAM)p1);
        break;
    }
}

/* Play `bytes` from the queue into `out` (NULL: drop). Finished buffers go to A.done; the device
 * thread hands them back to Miles, as winmm does from its own thread (Miles's callback takes
 * locks that must not be taken on the game's main thread). Caller holds the lock. */
static void consume(char *out, DWORD bytes) {
    while (bytes > 0 && A.count > 0) {
        WAVEHDR *h = A.queue[A.head];
        DWORD n = h->dwBufferLength - A.offset;
        if (n > bytes)
            n = bytes;
        if (out) {
            memcpy(out, h->lpData + A.offset, n);
            out += n;
        }
        bytes -= n;
        A.offset += n;
        if (A.offset >= h->dwBufferLength) {
            h->dwFlags = (h->dwFlags & ~WHDR_INQUEUE) | WHDR_DONE;
            A.done[A.ndone++] = h;
            A.head = (A.head + 1) % MAX_QUEUE;
            A.count--;
            A.offset = 0;
        }
    }
    if (out && bytes > 0) {
        memset(out, A.fmt.wBitsPerSample == 8 ? 0x80 : 0, bytes);
        A.underflow += bytes;
    }
}

static DWORD queued_bytes(void) {
    DWORD n = 0;
    for (int i = 0; i < A.count; i++)
        n += A.queue[(A.head + i) % MAX_QUEUE]->dwBufferLength;
    return n - A.offset;
}

static DWORD owed_bytes(double seconds) {
    double want = seconds * A.fmt.nAvgBytesPerSec + A.carry;
    DWORD align = A.fmt.nBlockAlign ? A.fmt.nBlockAlign : 1;
    DWORD n = (DWORD)(want / align) * align;
    A.carry = want - n;
    return n;
}

/* W3SIM_AUDIO_FREE: Miles's clock and its threads' waits. Its threads as of Mss32.dll 6.1d (the game's copy), by
 * the return addresses of their timed waits: the timer thread's and the stream service thread's. */
#define RVA_TIMER_WAIT1 0x15ba
#define RVA_TIMER_WAIT2 0x1634
#define RVA_STREAM_WAIT1 0x1c6a1
#define RVA_STREAM_WAIT2 0x1c705
#define FREE_IDLE_MS 50 /* (nothing to do while no piece comes: these threads' timed waits at their longest) */
#define FREE_PIECE 0.025 /* seconds: Miles's output pieces (4416 bytes): what its timers see at a time */
#define FREE_NUDGE 0.005 /* seconds of its clock when it has nothing queued, at most FREE_STUCK times a frame */
#define FREE_STUCK 40
static BYTE *g_mss;
static BOOL(WINAPI *MssQpc_orig)(LARGE_INTEGER *);
static DWORD(WINAPI *MssTgt_orig)(void);
static DWORD(WINAPI *MssWfso_orig)(HANDLE, DWORD);

static LONGLONG clock_ticks_read(void) {
    return __sync_val_compare_and_swap(&A.clock_ticks, 0, 0);
}

static BOOL WINAPI mss_Qpc(LARGE_INTEGER *out) {
    if (!A.clock_on)
        return MssQpc_orig(out);
    out->QuadPart = A.qpc_base + clock_ticks_read();
    return TRUE;
}

static DWORD WINAPI mss_Tgt(void) {
    if (!A.clock_on)
        return MssTgt_orig();
    return A.tgt_base + (DWORD)(clock_ticks_read() * 1000 / real_qpc_freq());
}

static DWORD WINAPI mss_Wfso(HANDLE h, DWORD ms) {
    DWORD site = (DWORD)((BYTE *)__builtin_return_address(0) - g_mss);
    int timer = site == RVA_TIMER_WAIT1 || site == RVA_TIMER_WAIT2;
    if (!A.clock_on || !(timer || site == RVA_STREAM_WAIT1 || site == RVA_STREAM_WAIT2))
        return MssWfso_orig(h, ms);
    if (timer) { /* the pass that began at the last wake has ended */
        A.pass_done = A.pass_seen;
        SetEvent(A.pass_event);
    }
    HANDLE both[2] = {h, timer ? A.timer_wake : A.stream_wake};
    DWORD r = WaitForMultipleObjects(2, both, FALSE, FREE_IDLE_MS);
    if (timer)
        A.pass_seen = A.pass_sent;
    return r == WAIT_OBJECT_0 + 1 ? WAIT_TIMEOUT : r;
}

/* Hand back what was played and let Miles's clock run on by `seconds`; back when its timer thread has seen it. */
static void miles_pass(double seconds) {
    __sync_add_and_fetch(&A.clock_ticks, (LONGLONG)(seconds * real_qpc_freq() + 0.5));
    LONG want = InterlockedIncrement(&A.pass_req);
    SetEvent(A.wake);
    if (A.passes_lost >= 20)
        return;
    for (int tries = 0; A.pass_done - want < 0; tries++) {
        if (tries == 4) { /* (200 ms, loaded machines) */
            A.pass_timeouts++;
            if (++A.passes_lost == 20)
                shim_log("audio: Miles's timer thread does not answer: no longer waiting for it");
            return;
        }
        WaitForSingleObject(A.pass_event, FREE_IDLE_MS);
    }
    A.passes_lost = 0;
}

/* One frame's audio, a piece at a time: no real-time pacing, and as many passes of Miles as a real sound card's
 * time would have brought. */
static void frame_free(DWORD bytes, char *buf, DWORD align) {
    if (!A.clock_on) {
        LARGE_INTEGER q;
        MssQpc_orig(&q);
        A.qpc_base = q.QuadPart;
        A.tgt_base = MssTgt_orig();
        MemoryBarrier();
        A.clock_on = 1;
        shim_log("audio: free-running: Miles's clock is the audio played");
    }
    DWORD piece = (DWORD)(FREE_PIECE * A.fmt.nAvgBytesPerSec) / align * align, got = 0;
    int stuck = 0;
    if (!piece)
        piece = align;
    while (got < bytes) {
        EnterCriticalSection(&A.lock);
        DWORD q = queued_bytes() / align * align;
        if (A.debt && q > bytes - got) { /* (silence stood in for it: drop it, or everything after would lag) */
            DWORD drop = q - (bytes - got) < A.debt ? q - (bytes - got) : A.debt;
            drop = drop / align * align;
            consume(NULL, drop);
            A.debt -= drop;
            q -= drop;
        }
        DWORD n = bytes - got < piece ? bytes - got : piece;
        if (n > q)
            n = q;
        if (n)
            consume(buf + got, n);
        A.last_virt = 0;
        LeaveCriticalSection(&A.lock);
        got += n;
        if (n) {
            stuck = 0;
            miles_pass((double)n / A.fmt.nAvgBytesPerSec);
        } else if (++stuck <= FREE_STUCK) { /* Miles mixes on its clock's time: nothing queued, let that run on */
            miles_pass(FREE_NUDGE);
        } else { /* (its device closing, say) */
            DWORD rest = bytes - got;
            memset(buf + got, A.fmt.wBitsPerSample == 8 ? 0x80 : 0, rest);
            EnterCriticalSection(&A.lock);
            A.underflow += rest;
            A.debt += rest;
            LeaveCriticalSection(&A.lock);
            break;
        }
    }
}

int audio_frame(double seconds, char **data, int *len, char *fmt, int fmt_cap) {
    if (!A.enabled || !A.open)
        return 0;
    if (A.free_run) {
        static char *fbuf;
        static DWORD fcap;
        EnterCriticalSection(&A.lock);
        DWORD fbytes = owed_bytes(seconds), falign = A.fmt.nBlockAlign ? A.fmt.nBlockAlign : 1;
        DWORD flatency = A.write_latency_n ? (DWORD)(A.write_latency_sum / A.write_latency_n) / falign * falign : 0;
        LeaveCriticalSection(&A.lock);
        if (fbytes > fcap) {
            fcap = fbytes * 2;
            fbuf = realloc(fbuf, fcap);
        }
        frame_free(fbytes, fbuf, falign);
        static int freported;
        static LONGLONG flast;
        if (A.underflow != flast && freported < 20) {
            flast = A.underflow;
            freported++;
            shim_log("audio: %lld bytes padded with silence so far (Miles was late), %ld passes lost", A.underflow,
                     A.pass_timeouts);
        }
        *data = fbuf;
        *len = (int)fbytes;
        _snprintf(fmt, fmt_cap, "%lu %u %u %lu", A.fmt.nSamplesPerSec, A.fmt.nChannels, A.fmt.wBitsPerSample, flatency);
        return 1;
    }
    /* Real-time pacing: Miles mixes on its own thread, woken by real time. Played faster than
     * real time, each wake would have to cover more game time and new sounds would wait longer
     * before they are mixed. At real-time pace Miles behaves as with a real sound card and the
     * only latency is its queue, which we report (see below). */
    static int64_t pace_t0;
    static double pace_audio;
    int64_t now = real_qpc_ticks(), freq = real_qpc_freq();
    if (!pace_t0 || (double)(now - pace_t0) / freq > pace_audio + 1.0) { /* start, or after a pause */
        pace_t0 = now;
        pace_audio = 0;
    }
    double ahead = pace_audio - (double)(now - pace_t0) / freq;
    if (ahead > 0.002)
        Sleep_real((DWORD)(ahead * 1000));
    pace_audio += seconds;
    EnterCriticalSection(&A.lock);
    DWORD bytes = owed_bytes(seconds);
    DWORD q0 = queued_bytes();
    LONG w0 = A.writes;
    LeaveCriticalSection(&A.lock);
    int waited = 0;
    /* Miles mixes on its own real-time thread: give it a moment to queue what this frame needs */
    for (; waited < 50; waited++) {
        EnterCriticalSection(&A.lock);
        DWORD q = queued_bytes();
        LeaveCriticalSection(&A.lock);
        if (q >= bytes)
            break;
        Sleep_real(1);
    }
    static char *buf;
    static DWORD cap;
    if (bytes > cap) {
        cap = bytes * 2;
        buf = realloc(buf, cap);
    }
    EnterCriticalSection(&A.lock);
    /* Miles mixes a buffer when one comes back and queues it behind the others: a sound starting
     * now is heard that much later (the mean queue ahead of newly written buffers) */
    DWORD align = A.fmt.nBlockAlign ? A.fmt.nBlockAlign : 1;
    DWORD latency = A.write_latency_n ? (DWORD)(A.write_latency_sum / A.write_latency_n) / align * align : 0;
    /* audio that arrived late was replaced by silence: skip it now, or everything after would lag */
    DWORD q = queued_bytes();
    if (A.debt && q > bytes) {
        DWORD drop = q - bytes < A.debt ? q - bytes : A.debt;
        drop = drop / align * align;
        consume(NULL, drop);
        A.debt -= drop;
    }
    LONGLONG before = A.underflow;
    consume(buf, bytes);
    A.last_virt = 0;
    LONGLONG short_by = A.underflow - before;
    A.debt += (DWORD)short_by;
    LeaveCriticalSection(&A.lock);
    SetEvent(A.wake);
    static int timeline;
    if (A.trace && timeline < 400) {
        timeline++;
        shim_log("audio: frame %ld queued %lu need %lu waited %d ms writes %ld short %lld", sync_frame_count(), q0,
                 bytes, waited, A.writes - w0, short_by);
    }
    static int reported;
    if (short_by > 0 && reported < 20) {
        reported++;
        shim_log("audio: underrun, %lld of %lu bytes padded with silence (Miles was late)", short_by, bytes);
    }
    *data = buf;
    *len = (int)bytes;
    _snprintf(fmt, fmt_cap, "%lu %u %u %lu", A.fmt.nSamplesPerSec, A.fmt.nChannels, A.fmt.wBitsPerSample, latency);
    return 1;
}

/* The device thread: hands finished buffers back to Miles and, while no video is captured,
 * plays (drops) audio as virtual time passes (also while the game is loading, when no frames are
 * presented). */
static DWORD WINAPI device_thread(LPVOID arg) {
    int64_t freq = real_qpc_freq();
    for (;;) {
        WaitForSingleObject(A.wake, 5);
        WAVEHDR *done[MAX_QUEUE];
        EnterCriticalSection(&A.lock);
        if (A.open && !frame_capture_get()) {
            int64_t now = clock_virtual_ticks();
            if (A.last_virt && now > A.last_virt)
                consume(NULL, owed_bytes((double)(now - A.last_virt) / (double)freq));
            A.last_virt = now;
        }
        int n = A.ndone;
        memcpy(done, A.done, n * sizeof *done);
        A.ndone = 0;
        LeaveCriticalSection(&A.lock);
        for (int i = 0; i < n; i++)
            notify(WOM_DONE, (DWORD_PTR)done[i]);
        if (A.free_run && A.pass_sent != A.pass_req) { /* (the buffers are back: now Miles's turn) */
            A.pass_sent = A.pass_req;
            SetEvent(A.timer_wake);
            SetEvent(A.stream_wake);
        }
    }
    return 0;
}

LONGLONG audio_underflow_bytes(void) {
    return A.underflow;
}

/* ---- waveOut ----------------------------------------------------------------------------- */

static MMRESULT WINAPI my_waveOutGetDevCapsA(UINT_PTR id, LPWAVEOUTCAPSA caps, UINT size) {
    trace("GetDevCaps", (unsigned)id, size);
    if (!caps || size < sizeof(WAVEOUTCAPSA))
        return MMSYSERR_INVALPARAM;
    memset(caps, 0, size);
    caps->wMid = 1;
    caps->wPid = 1;
    caps->vDriverVersion = 0x100;
    strcpy(caps->szPname, "w3sim virtual audio");
    caps->dwFormats = WAVE_FORMAT_4S16 | WAVE_FORMAT_4M16 | WAVE_FORMAT_2S16 | WAVE_FORMAT_1S16 | WAVE_FORMAT_4S08;
    caps->wChannels = 2;
    return MMSYSERR_NOERROR;
}

static MMRESULT WINAPI my_waveOutGetID(HWAVEOUT h, LPUINT id) {
    if (id)
        *id = 0;
    return MMSYSERR_NOERROR;
}

static MMRESULT WINAPI my_waveOutOpen(LPHWAVEOUT phwo, UINT id, LPCWAVEFORMATEX fmt, DWORD_PTR cb, DWORD_PTR inst,
                                      DWORD flags) {
    trace("Open", fmt ? fmt->nSamplesPerSec : 0, flags);
    if (!fmt || fmt->wFormatTag != WAVE_FORMAT_PCM || (fmt->wBitsPerSample != 8 && fmt->wBitsPerSample != 16) ||
        fmt->nChannels < 1 || fmt->nChannels > 2)
        return WAVERR_BADFORMAT;
    if (flags & WAVE_FORMAT_QUERY)
        return MMSYSERR_NOERROR;
    EnterCriticalSection(&A.lock);
    if (A.open) {
        LeaveCriticalSection(&A.lock);
        return MMSYSERR_ALLOCATED;
    }
    A.fmt = *fmt;
    A.fmt.cbSize = 0;
    A.cb_type = flags & CALLBACK_TYPEMASK;
    A.callback = cb;
    A.instance = inst;
    A.head = A.count = 0;
    A.offset = 0;
    A.carry = 0;
    A.last_virt = 0;
    A.open = 1;
    LeaveCriticalSection(&A.lock);
    if (phwo)
        *phwo = FAKE_HWO;
    shim_log("audio: device opened, %lu Hz, %u channels, %u bits", fmt->nSamplesPerSec, fmt->nChannels,
             fmt->wBitsPerSample);
    notify(WOM_OPEN, 0);
    return MMSYSERR_NOERROR;
}

static MMRESULT WINAPI my_waveOutPrepareHeader(HWAVEOUT h, LPWAVEHDR hdr, UINT size) {
    if (!hdr)
        return MMSYSERR_INVALPARAM;
    hdr->dwFlags |= WHDR_PREPARED;
    return MMSYSERR_NOERROR;
}

static MMRESULT WINAPI my_waveOutUnprepareHeader(HWAVEOUT h, LPWAVEHDR hdr, UINT size) {
    if (!hdr)
        return MMSYSERR_INVALPARAM;
    if (hdr->dwFlags & WHDR_INQUEUE)
        return WAVERR_STILLPLAYING;
    hdr->dwFlags &= ~WHDR_PREPARED;
    return MMSYSERR_NOERROR;
}

static MMRESULT WINAPI my_waveOutWrite(HWAVEOUT h, LPWAVEHDR hdr, UINT size) {
    if (!hdr || !(hdr->dwFlags & WHDR_PREPARED))
        return WAVERR_UNPREPARED;
    EnterCriticalSection(&A.lock);
    if (A.count >= MAX_QUEUE) {
        LeaveCriticalSection(&A.lock);
        return MMSYSERR_NOMEM;
    }
    hdr->dwFlags = (hdr->dwFlags & ~WHDR_DONE) | WHDR_INQUEUE;
    if (frame_capture_get()) {
        A.write_latency_sum += queued_bytes();
        A.write_latency_n++;
    }
    A.queue[(A.head + A.count) % MAX_QUEUE] = hdr;
    A.count++;
    A.writes++;
    int n = A.count;
    LeaveCriticalSection(&A.lock);
    static int traced;
    if (A.trace > 0 && traced++ < 40)
        shim_log("audio: write %lu bytes, %d queued", hdr->dwBufferLength, n);
    return MMSYSERR_NOERROR;
}

static MMRESULT WINAPI my_waveOutReset(HWAVEOUT h) {
    trace("Reset", 0, 0);
    EnterCriticalSection(&A.lock);
    while (A.count > 0) {
        WAVEHDR *hd = A.queue[A.head];
        hd->dwFlags = (hd->dwFlags & ~WHDR_INQUEUE) | WHDR_DONE;
        A.done[A.ndone++] = hd;
        A.head = (A.head + 1) % MAX_QUEUE;
        A.count--;
    }
    A.offset = 0;
    LeaveCriticalSection(&A.lock);
    SetEvent(A.wake);
    return MMSYSERR_NOERROR;
}

static MMRESULT WINAPI my_waveOutClose(HWAVEOUT h) {
    trace("Close", 0, 0);
    EnterCriticalSection(&A.lock);
    if (A.count > 0) {
        LeaveCriticalSection(&A.lock);
        return WAVERR_STILLPLAYING;
    }
    A.open = 0;
    LeaveCriticalSection(&A.lock);
    notify(WOM_CLOSE, 0);
    return MMSYSERR_NOERROR;
}

/* ---- DirectSound: unavailable, so Miles falls back to waveOut ------------------------------ */

static HRESULT WINAPI no_DirectSoundCreate(const GUID *g, void **out, void *outer) {
    trace("DirectSoundCreate (refused)", 0, 0);
    if (out)
        *out = NULL;
    return 0x88780078; /* DSERR_NODRIVER */
}

static FARPROC WINAPI GetProcAddress_hook(HMODULE m, LPCSTR name) {
    if ((UINT_PTR)name > 0xffff) {
        if (A.trace > 0)
            trace(name, 0, 0);
        if (strncmp(name, "DirectSoundCreate", 17) == 0)
            return (FARPROC)no_DirectSoundCreate;
    }
    return GetProcAddress_orig(m, name);
}

/* Miles buffers ~280 ms of mixed audio ahead by default (preference 11: the waveOut output
 * buffer, 49152 bytes): a sound started now would be heard that much later. The game sets no
 * preferences; AIL_startup resets them, so set ours when the game opens the digital driver. */
typedef int(WINAPI *SetPrefFn)(unsigned, int);
static SetPrefFn g_set_pref;
/* 11: output buffer bytes (default 49152, ~280 ms), 45: buffer piece duration (default 100 -> 50 ms
 * pieces). Smaller: ~55 ms latency instead of ~180 ms, and less jitter. */
static char g_prefs[128] = "11=8820,45=20"; /* W3SIM_AUDIO_PREFS: "id=value,..." */
static void *(WINAPI *open_digital_driver_orig)(unsigned, int, int, unsigned);

static void *WINAPI open_digital_driver_hook(unsigned rate, int bits, int channels, unsigned flags) {
    for (char *p = g_prefs; g_set_pref && *p;) {
        char *end;
        unsigned id = strtoul(p, &end, 10);
        if (*end != '=')
            break;
        int v = strtol(end + 1, &end, 10);
        shim_log("audio: Miles preference %u = %d (was %d)", id, v, g_set_pref(id, v));
        p = *end == ',' ? end + 1 : end;
        if (!*end || *end != ',')
            break;
    }
    return open_digital_driver_orig(rate, bits, channels, flags);
}

/* W3SIM_AUDIO_TRACE: log the frame at which the game starts each sample (latency measurement) */
static int(WINAPI *start_sample_orig)(void *);

static int WINAPI start_sample_hook(void *sample) {
    if (frame_capture_get())
        shim_log("audio: start_sample at frame %ld", sync_frame_count());
    return start_sample_orig(sample);
}

void audio_install(void) {
    char buf[16];
    if (!GetEnvironmentVariableA("W3SIM_AUDIO", buf, sizeof buf) || buf[0] != '1')
        return;
    InitializeCriticalSection(&A.lock);
    A.trace = GetEnvironmentVariableA("W3SIM_AUDIO_TRACE", buf, sizeof buf) ? 200 : 0;
    HMODULE mss = GetModuleHandleA("mss32.dll");
    if (!mss) {
        shim_log("audio: Mss32.dll not loaded; no audio");
        return;
    }
    int ok = 1;
    ok &= iat_hook(mss, "KERNEL32.dll", "GetProcAddress", GetProcAddress_hook, (void **)&GetProcAddress_orig);
    ok &= iat_hook(mss, "WINMM.dll", "waveOutGetDevCapsA", my_waveOutGetDevCapsA, NULL);
    ok &= iat_hook(mss, "WINMM.dll", "waveOutGetID", my_waveOutGetID, NULL);
    ok &= iat_hook(mss, "WINMM.dll", "waveOutOpen", my_waveOutOpen, NULL);
    ok &= iat_hook(mss, "WINMM.dll", "waveOutPrepareHeader", my_waveOutPrepareHeader, NULL);
    ok &= iat_hook(mss, "WINMM.dll", "waveOutUnprepareHeader", my_waveOutUnprepareHeader, NULL);
    ok &= iat_hook(mss, "WINMM.dll", "waveOutWrite", my_waveOutWrite, NULL);
    ok &= iat_hook(mss, "WINMM.dll", "waveOutReset", my_waveOutReset, NULL);
    ok &= iat_hook(mss, "WINMM.dll", "waveOutClose", my_waveOutClose, NULL);
    if (!ok) {
        shim_log("audio: some Miles imports were not found; no audio");
        return;
    }
    if (A.trace) { /* W3SIM_AUDIO_TRACE: Miles's preferences (to find its buffering settings) */
        typedef int(WINAPI * GetPrefFn)(unsigned);
        GetPrefFn get = (GetPrefFn)GetProcAddress(mss, "_AIL_get_preference@4");
        for (unsigned i = 0; get && i < 64; i++)
            shim_log("audio: preference %u = %d", i, get(i));
    }
    /* Miles buffers ~280 ms of mixed audio ahead (preference 11, the waveOut output buffer size,
     * default 49152 bytes in ~50 ms pieces): a sound started now is heard that much later. The
     * game sets no preferences, so set a small buffer before it opens the digital driver. */
    g_set_pref = (SetPrefFn)GetProcAddress(mss, "_AIL_set_preference@8");
    GetEnvironmentVariableA("W3SIM_AUDIO_PREFS", g_prefs, sizeof g_prefs);
    if (!iat_hook((HMODULE)g_base, "mss32.dll", "_AIL_open_digital_driver@16", open_digital_driver_hook,
                  (void **)&open_digital_driver_orig))
        shim_log("audio: AIL_open_digital_driver import not found; Miles keeps its buffering");
    if (A.trace)
        iat_hook((HMODULE)g_base, "mss32.dll", "_AIL_start_sample@4", start_sample_hook, (void **)&start_sample_orig);
    if (GetEnvironmentVariableA("W3SIM_AUDIO_FREE", buf, sizeof buf) && buf[0] == '1') {
        g_mss = (BYTE *)mss;
        A.timer_wake = CreateEventA(NULL, FALSE, FALSE, NULL);
        A.stream_wake = CreateEventA(NULL, FALSE, FALSE, NULL);
        A.pass_event = CreateEventA(NULL, FALSE, FALSE, NULL);
        if (iat_hook(mss, "KERNEL32.dll", "QueryPerformanceCounter", mss_Qpc, (void **)&MssQpc_orig) &&
            iat_hook(mss, "WINMM.dll", "timeGetTime", mss_Tgt, (void **)&MssTgt_orig) &&
            iat_hook(mss, "KERNEL32.dll", "WaitForSingleObject", mss_Wfso, (void **)&MssWfso_orig))
            A.free_run = 1;
        else
            shim_log("audio: Miles's clock not hooked: real-time pacing");
    }
    A.wake = CreateEventA(NULL, FALSE, FALSE, NULL);
    CreateThread(NULL, 0, device_thread, NULL, 0, NULL);
    A.enabled = 1;
    shim_log("audio: virtual sound card installed");
}
