/* w3shim: in-process helper for Warcraft III 1.29.2 (x86) under Wine.
 *
 * - virtual clock: the game's timers (QPC, GetTickCount, FILETIME and its rdtsc helper) run
 *   `speed` times faster than wall time and stand still while the game waits for the
 *   controller; timed waits/sleeps are scaled down accordingly.
 * - step sync: when the map harness opens its action file (Preloader), the call blocks until
 *   the Python controller has read the observation and written the actions.
 * - frame capture (video): the clock advances a fixed game time per rendered frame and every
 *   presented frame is reported to the controller, which grabs the window before it continues.
 *
 * Configuration comes from environment variables (Wine passes the Linux environment through):
 *   W3SIM_SPEED       initial clock speed (default 1)
 *   W3SIM_PORT        TCP port of the controller on 127.0.0.1 (unset: no step sync)
 *   W3SIM_SYNC_FILE   file-name suffix that triggers step sync (default "w3sim\act.txt")
 *   W3SIM_WAIT_FLOOR  minimum scaled wait in ms (default 0)
 *   W3SIM_TURBO_MS    game time simulated per frame, bypassing turn pacing (default 0 = off)
 *   W3SIM_LOG         log file path (Windows path; default: none)
 */
#pragma once

#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <windows.h>
#include <stdint.h>

#define W3SHIM_VERSION "0.1"

/* Addresses in the 1.29.2.9231 executable (sha256 3f2ed012...0eed), as RVAs. */
#define RVA_RDTSC_HELPER 0x396c80 /* rdtsc; ret */

extern BYTE *g_base;

void shim_log(const char *fmt, ...);

/* iat.c */
int iat_hook(HMODULE module, const char *dll, const char *func, void *replacement, void **original);

/* clock.c */
void clock_install(double speed, DWORD wait_floor);
void clock_set_speed(double speed);
void clock_freeze(int frozen);
void clock_set_frame_step(double seconds); /* > 0: advance only per rendered frame; 0: real time */
void clock_frame(void);                    /* a frame was presented */
void clock_main_thread(void);              /* the calling thread steps the game (W3SIM_BG_SPEED) */
int64_t clock_virtual_ticks(void);         /* virtual time (QPC ticks) */
double clock_frame_seconds(void);          /* frame-stepped: game time per frame, else 0 */
void clock_report_waits(double secs);      /* W3SIM_PROFILE=3: per-thread wait statistics */
extern int g_wait_stats;
double clock_speed(void);
int64_t real_qpc_ticks(void);
int64_t real_qpc_freq(void);
void Sleep_real(DWORD ms);

/* sync.c */
void sync_install(void);
LONGLONG sync_wait_ticks(void);
long sync_count(void); /* step syncs so far */
long sync_frame_count(void); /* frames reported so far (frame capture) */
void sync_phases(double secs, char *out, int cap); /* per-second phase totals since the last call */
void sync_reset_wait_ticks(void);
extern volatile LONG g_sync_waiting;
void sync_frame(void); /* frame capture: report the presented frame, wait for the controller */
void frame_capture_set(int on);
int frame_capture_get(void);

/* obs.c */
void obs_install(void);
int obs_take(const char **data); /* the captured observation's length, or -1 if none */
int obs_peek(const char **data); /* the observation captured so far (its length) */
void obs_reset(void);

void obs_bytes(const char *s, int n); /* tokens from the shim itself (lines) */
extern int g_obs_raw_player;          /* (W3SIM_CUTS & 1073741824) the handle id of W3S_Tok's player */
void obs_raw_tok(int v);              /* one of W3S_Tok's integers, as its token */
int *sync_mbox(void);                 /* the mailbox: [0] the count, then the command integers */
int obs_capture_on(void);

/* units.c: unit records written by the shim (W3SIM_UNITS=0: by the harness, 2: both, compared) */
void units_install(void);
void units_cmd_report(void); /* (W3SIM_CUTS & 2147483648) the commands applied, per second */
int units_call(int key, int mbox, int *result);
void units_verify_token(const char *s);
void units_mark(int phase); /* W3SIM_PROFILE: a step's phases (0 GO, 3 the observation ends, 4 the sync) */

#ifdef W3SIM_REPLAY
/* replay.c (the replay build, make replay): object ids of units and of the harness's probes, buildings' state */
int replay_unit_flags(int u, int flags);
int replay_call(int key, int mbox, int *result);
#endif

/* audio.c (W3SIM_AUDIO=1) */
void audio_install(void);
/* the audio played during a captured frame of `seconds`; 0 if there is no audio device */
int audio_frame(double seconds, char **data, int *len, char *fmt, int fmt_cap);
LONGLONG audio_underflow_bytes(void);

/* registry.c */
void registry_install(void);

/* turbo.c */
void turbo_install(int ms);
void render_install(int draw); /* W3SIM_DRAW=0: the Direct3D device draws nothing (render.c) */
long render_skipped(void);     /* draw calls skipped since the last call */
void syncstat_install(void);   /* W3SIM_PROFILE=4: the game's synchronization calls by call site (syncstat.c) */
void syncstat_report(double secs);
void turbo_set(int ms);
void cuts_install(int draw); /* W3SIM_CUTS (cuts.c) */
long long cuts_flags(void);
void softfloat_install(long long cuts); /* W3SIM_CUTS 67108864.. (softfloat.c) */
void cuts_game_start(void); /* a game's first step (units.c) */
void cuts_unit_begin(int on); /* units.c: a unit's natives begin (1) / end (0) (W3SIM_CUTS & 32768) */
void cuts_sleep_prefetch(const BYTE *u); /* units.c: the sleep cache's slot of a unit ahead (W3SIM_CUTS & 137438953472) */
extern volatile DWORD g_step_tid; /* sync.c: the thread that steps the game */
void cuts_report(void);
void pathing_install(long long cuts); /* W3SIM_CUTS 524288, 1048576, 4194304 (pathing.c) */
void pathing_report(void);
int turbo_get(void);
