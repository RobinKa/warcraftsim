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
void sync_phases(double secs, char *out, int cap); /* per-second phase totals since the last call */
void sync_reset_wait_ticks(void);
void sync_frame(void); /* frame capture: report the presented frame, wait for the controller */
void frame_capture_set(int on);
int frame_capture_get(void);

/* obs.c */
void obs_install(void);
int obs_take(const char **data); /* the captured observation's length, or -1 if none */
void obs_reset(void);

/* registry.c */
void registry_install(void);

/* turbo.c */
void turbo_install(int ms);
void turbo_set(int ms);
int turbo_get(void);
