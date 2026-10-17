/* The w3shim build for playing back people's replays (`make replay`: W3SIM_REPLAY; only
 * warcraftsim.fullgame.replays loads it). Training and collection use the default build, which
 * has none of this.
 *
 * A replay names units by the engine's object ids: a pair of dwords at +0xc (a slot in the
 * engine's table of objects) and +0x10 (the count of objects made before) of every object.
 * Python needs them to know which unit a recorded selection or order means, and the observing
 * harness (w3obs.j) needs to know which ids its own objects took (two probe groups made first and
 * last in its initialization, reported with MBOX - 40), to shift the replay's later ids by them.
 * Both go to the file W3SIM_TAGS (a Windows path; default: tags.txt next to the log), one line each:
 *
 *   P <handle> <slot> <count>         a probe (in the order the harness reports them)
 *   D <handle> <slot> <count>         a tree (the harness reports them at its initialization)
 *   T <obs> <handle> <slot> <count>   a unit's ids, when first seen or changed (its handle reused),
 *                                     with the number of the observation its record goes into
 *
 * The harness cannot get the game's events (an event that has a trigger makes objects each time
 * it fires, so every unit made later would have other ids than in the recorded game): the unit
 * records carry the engine's own state of buildings instead, as flags (read from the unit's
 * object; its layout found by comparing an altar's, a barracks', a ziggurat's and a great hall's
 * bytes before and after construction, training and an upgrade):
 *
 *   8     under construction   (+0x164 & 0x2000000)
 *   4096  producing            (+0x3c4: training or researching, the queue not empty)
 *   8192  being upgraded       (+0x164 & 0x4000000: a town hall to a keep, ...)
 */
#include "w3shim.h"

#include <stdio.h>
#include <string.h>

#define VA(x) ((BYTE *)(g_base + ((x) - 0x400000)))
#define VA_JASS 0x113b6f4 /* the JASS state: +0x1c its handle manager, whose +0x198 is the handle table */
#define HANDLE_BASE 0x100000
#define OFF_FLAGS2 0x164
#define OFF_PRODUCING 0x3c4
#define BIT_CONSTRUCTING 0x2000000
#define BIT_UPGRADING 0x4000000
#define MAX_SEEN (1 << 17)

static HANDLE g_file = INVALID_HANDLE_VALUE;
static int g_opened;
static uint32_t g_seen[MAX_SEEN]; /* the count id last reported per handle slot (0: none) */
static char g_next; /* what the next negative key is: 'P' a probe, 'D' a tree (0: nothing of ours) */

static void out(const char *line, int n) {
    if (!g_opened) {
        char path[MAX_PATH];
        g_opened = 1;
        if (GetEnvironmentVariableA("W3SIM_TAGS", path, sizeof path) ||
            (GetEnvironmentVariableA("W3SIM_LOG", path, sizeof path) && strrchr(path, '\\') &&
             strcpy(strrchr(path, '\\') + 1, "tags.txt")))
            g_file = CreateFileA(path, GENERIC_WRITE, FILE_SHARE_READ, NULL, CREATE_ALWAYS, 0, NULL);
    }
    if (g_file != INVALID_HANDLE_VALUE) {
        DWORD written;
        WriteFile(g_file, line, (DWORD)n, &written, NULL);
    }
}

/* the object behind a handle (the handle table's entry: 12 bytes, the object at +4), or NULL */
static uint32_t *object_of(int h) {
    BYTE *jass = *(BYTE **)VA(VA_JASS);
    if (!jass || h < HANDLE_BASE || h >= HANDLE_BASE + 0x1000000)
        return NULL;
    BYTE *mgr = *(BYTE **)(jass + 0x1c);
    if (!mgr)
        return NULL;
    BYTE *table = *(BYTE **)(mgr + 0x198);
    return table ? *(uint32_t **)(table + (h - HANDLE_BASE) * 12 + 4) : NULL;
}

int replay_unit_flags(int u, int flags) {
    uint32_t *o = object_of(u);
    if (!o)
        return 0;
    int slot = (u - HANDLE_BASE) & (MAX_SEEN - 1);
    if (g_seen[slot] != o[4]) {
        char line[96];
        g_seen[slot] = o[4];
        out(line, _snprintf(line, sizeof line, "T %ld %d %u %u\n", sync_count() + 1, u, o[3], o[4]));
    }
    if (!(flags & 2)) /* (a structure's state only) */
        return 0;
    uint32_t f2 = o[OFF_FLAGS2 / 4];
    return (f2 & BIT_CONSTRUCTING ? 8 : 0) | (o[OFF_PRODUCING / 4] ? 4096 : 0) | (f2 & BIT_UPGRADING ? 8192 : 0);
}

int replay_call(int key, int mbox, int *result) {
    *result = 0;
    if (key == mbox - 40 || key == mbox - 41) { /* the next negative key is a probe's or a tree's handle */
        g_next = key == mbox - 40 ? 'P' : 'D';
        return 1;
    }
    if (key < 0 && g_next) {
        char line[96];
        uint32_t *o = object_of(-key);
        out(line, _snprintf(line, sizeof line, "%c %d %u %u\n", g_next, -key, o ? o[3] : 0, o ? o[4] : 0));
        g_next = 0;
        return 1;
    }
    return 0;
}
