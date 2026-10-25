/* The rules' per-step facts in C (loaded with ctypes by fullgame/rulesnative.py): what the heuristic teacher
 * (teacher.Teacher.commands and its expansion lesson) and the strategy modes' tracker (modes.Tracker.step) read
 * from a step's rows, in a pass or two over them instead of ~10-30 numpy calls (each ~3-50k instructions on a
 * step's ~100-400 rows: the two had cost ~700k instructions a side-step of an actor's ~2.5M).
 *
 * Rows: int64 [n][ncols], the unit columns of features.py. Distances in double as the numpy code computes them:
 * hypot of the differences (numpy's np.hypot is libm's), squared distances as dx * dx + dy * dy (gcc -O2 for
 * x86-64 contracts no FMA); the integer differences numpy takes in int64 taken in int64 here too. Ties where the
 * numpy code's order is argsort's own are reported, and the caller's numpy decides. Type classes: a table of type
 * codes (sorted) with a bit mask each (K_* below, rulesnative.py's).
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { C_STEP, C_ID, C_TYPE, C_OWNER, C_X, C_Y, C_FACING, C_HP, C_MAXHP, C_MANA, C_MAXMANA, C_ORDER, C_FLAGS, C_VIS,
       C_RESOURCE, C_HLEVEL };
#define HERO 1
#define STRUCTURE 2
#define WORKER 4
#define CONSTRUCTING 8
#define HIDDEN 16
#define DEAD 1024
#define CREEPS 12 /* the neutral hostile player */

#define K_ALTAR 1
#define K_TIER2 2
#define K_TIER3 4
#define K_WORKER 8
#define K_MINE 16
#define K_HOLD 32
#define K_OWN_MINE 64

static int64_t cls(const int64_t *keys, const int64_t *vals, int n, int64_t v) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (keys[mid] < v) lo = mid + 1; else hi = mid;
    }
    return lo < n && keys[lo] == v ? vals[lo] : 0;
}

static int in_sorted(const int64_t *keys, int n, int64_t v) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (keys[mid] < v) lo = mid + 1; else hi = mid;
    }
    return lo < n && keys[lo] == v;
}

/* Teacher.commands' facts of `side`: out[0..TH) the counts (rulesnative.teach's names), then the segments, packed:
 * its heroes' types (dead ones too), its living heroes' rows, its finished buildings' (type, id), its army's types
 * (no building, worker or hero). `out` holds TH + 5 * nrows. Returns the length used. */
#define TH 12
int w3r_teach(const int64_t *rows, int nrows, int ncols, int side, int64_t tavern, int64_t merchant,
              const int64_t *ck, const int64_t *cv, int nc, int64_t *out) {
    int64_t *seg = out + TH, *heroes = seg, *live = seg + nrows, *done = seg + 2 * (size_t)nrows,
            *army = seg + 4 * (size_t)nrows;
    int nh = 0, nl = 0, nd = 0, na = 0, n_altars = 0, n_alive = 0;
    int64_t tav = -1, mer = -1, done_or = 0, altar = -1, near_w = -1;
    for (int i = 0; i < nrows; i++) {
        const int64_t *r = rows + (size_t)i * ncols;
        int64_t fl = r[C_FLAGS], t = r[C_TYPE];
        int dead = (fl & DEAD) != 0;
        if (!dead && tav < 0 && t == tavern) tav = i;
        if (!dead && mer < 0 && t == merchant) mer = i;
        if (r[C_OWNER] != side) continue;
        if (fl & HERO) heroes[nh++] = t;
        if (dead) continue;
        n_alive++;
        int64_t k = cls(ck, cv, nc, t);
        if (fl & HERO) live[nl++] = i;
        if ((fl & STRUCTURE) && !(fl & CONSTRUCTING)) {
            done[2 * nd] = t; done[2 * nd + 1] = r[C_ID]; nd++; done_or |= k & (K_ALTAR | K_TIER2 | K_TIER3);
        }
        if (!(fl & (STRUCTURE | WORKER | HERO))) army[na++] = t;
        if (k & K_ALTAR) { n_altars++; if (altar < 0 && !(fl & CONSTRUCTING)) altar = i; }
    }
    if (tav >= 0) {  /* the worker nearest the tavern (np.argmin: the first of equals) */
        const int64_t *tv = rows + (size_t)tav * ncols;
        double bd = 0.0;
        for (int i = 0; i < nrows; i++) {
            const int64_t *r = rows + (size_t)i * ncols;
            if (r[C_OWNER] != side || (r[C_FLAGS] & DEAD) || !(cls(ck, cv, nc, r[C_TYPE]) & K_WORKER)) continue;
            double d = hypot((double)(r[C_X] - tv[C_X]), (double)(r[C_Y] - tv[C_Y]));
            if (near_w < 0 || d < bd) { near_w = i; bd = d; }
        }
    }
    out[0] = nh; out[1] = nl; out[2] = tav; out[3] = mer; out[4] = nd; out[5] = na; out[6] = done_or;
    out[7] = n_altars; out[8] = altar; out[9] = near_w; out[10] = n_alive; out[11] = 0;
    int at = nh;
    memmove(seg + at, live, sizeof(int64_t) * nl); at += nl;
    memmove(seg + at, done, sizeof(int64_t) * 2 * nd); at += 2 * nd;
    memmove(seg + at, army, sizeof(int64_t) * na); at += na;
    return TH + at;
}

/* Teacher._expansion's facts of `side` with its start (sx, sy): the natural (the living gold mine nearest the start
 * but farther than from_start), and around it (mx, my): out[0] 1, or -1 no natural, -2 two as near (the numpy code
 * decides); out[1..4) its id, x and y; out[4] the side's workers outside mines; out[5] its holds (halls, haunted or
 * entangled mines) within taken_near and out[6] whether those are all under construction; out[7] its workers (in
 * mines too) within worker_near; out[8] whether an own mine stands within taken_near, out[9] the first's id;
 * out[10] whether creeps stand within camp_near; out[11] its army (no building or worker), out[12] the heroes among
 * them; out[13] whether it has a worker outside mines, out[14] the id of the one nearest the natural (np.argmin:
 * the first of equals); with holds near, out[15] how many of the workers outside mines and outside worker_near go
 * there (3 at most, the nearest first: argsort's) and out[16..19) their ids (out[0] -2 when equal distances make
 * argsort's order matter); then the heroes' ids. Ids missing are 0. `out` holds EH + nrows. Returns the length. */
#define EH 19
int w3r_expansion(const int64_t *rows, int nrows, int ncols, int side, double sx, double sy, const int64_t *ck,
                  const int64_t *cv, int nc, double from_start, double taken_near, double worker_near,
                  double camp_near, int64_t *out) {
    int best = -1, tie = 0;
    double bd = 0.0;
    for (int q = 0; q < EH; q++) out[q] = 0;
    for (int i = 0; i < nrows; i++) {
        const int64_t *r = rows + (size_t)i * ncols;
        if ((r[C_FLAGS] & DEAD) || !(cls(ck, cv, nc, r[C_TYPE]) & K_MINE)) continue;
        double d = hypot((double)r[C_X] - sx, (double)r[C_Y] - sy);
        if (!(d > from_start)) continue;
        if (best < 0 || d < bd) { best = i; bd = d; tie = 0; }
        else if (d == bd) tie = 1;
    }
    if (best < 0 || tie) { out[0] = best < 0 ? -1 : -2; return EH; }
    const int64_t *m = rows + (size_t)best * ncols;
    double mx = (double)m[C_X], my = (double)m[C_Y];
    int n_workers = 0, n_taken = 0, all_cons = 1, n_mining = 0, n_army = 0, nh = 0, np_ = 0;
    int64_t target = -1, nearest = -1, pick[3] = {-1, -1, -1};
    double nd = 0.0, pd[3] = {0.0, 0.0, 0.0};
    int64_t *heroes = out + EH;
    for (int i = 0; i < nrows; i++) {
        const int64_t *r = rows + (size_t)i * ncols;
        int64_t fl = r[C_FLAGS];
        if (r[C_OWNER] != side || (fl & DEAD)) continue;
        int64_t k = cls(ck, cv, nc, r[C_TYPE]);
        if (!(fl & (STRUCTURE | WORKER))) { n_army++; if (fl & HERO) heroes[nh++] = r[C_ID]; }
        if (!(k & (K_WORKER | K_HOLD | K_OWN_MINE))) continue;
        double d = hypot((double)r[C_X] - mx, (double)r[C_Y] - my);
        if (k & K_WORKER) {
            if (d < worker_near) n_mining++;
            if (!(fl & HIDDEN)) {
                n_workers++;
                if (nearest < 0 || d < nd) { nearest = i; nd = d; }
                if (!(d < worker_near)) {  /* the three nearest (insertion; equal distances are checked below) */
                    int j = np_ < 3 ? np_ : 3;
                    while (j > 0 && d < pd[j - 1]) { if (j < 3) { pd[j] = pd[j - 1]; pick[j] = pick[j - 1]; } j--; }
                    if (j < 3) { pd[j] = d; pick[j] = i; if (np_ < 3) np_++; }
                }
            }
        }
        if ((k & K_HOLD) && d < taken_near) { n_taken++; if (!(fl & CONSTRUCTING)) all_cons = 0; }
        if ((k & K_OWN_MINE) && target < 0 && d < taken_near) target = i;
    }
    if (!n_taken) np_ = 0;  /* (the workers go only to a natural taken) */
    for (int i = 0; i < nrows && np_; i++) {  /* a picked distance twice: argsort's order of equals decides */
        const int64_t *r = rows + (size_t)i * ncols;
        int64_t fl = r[C_FLAGS];
        if (r[C_OWNER] != side || (fl & (DEAD | HIDDEN)) || !(cls(ck, cv, nc, r[C_TYPE]) & K_WORKER)) continue;
        double d = hypot((double)r[C_X] - mx, (double)r[C_Y] - my);
        if (d < worker_near) continue;
        for (int j = 0; j < np_; j++)
            if (d == pd[j] && i != pick[j]) { out[0] = -2; return EH; }
    }
    int creeps = 0;
    for (int i = 0; i < nrows && !creeps; i++) {
        const int64_t *r = rows + (size_t)i * ncols;
        if (r[C_OWNER] != CREEPS || (r[C_FLAGS] & DEAD)) continue;
        if (hypot((double)r[C_X] - mx, (double)r[C_Y] - my) < camp_near) creeps = 1;
    }
    out[0] = 1; out[1] = m[C_ID]; out[2] = m[C_X]; out[3] = m[C_Y]; out[4] = n_workers; out[5] = n_taken;
    out[6] = n_taken ? all_cons : 0; out[7] = n_mining; out[8] = target >= 0;
    out[9] = target >= 0 ? rows[(size_t)target * ncols + C_ID] : 0; out[10] = creeps; out[11] = n_army; out[12] = nh;
    out[13] = nearest >= 0; out[14] = nearest >= 0 ? rows[(size_t)nearest * ncols + C_ID] : 0; out[15] = np_;
    for (int j = 0; j < np_; j++) out[16 + j] = rows[(size_t)pick[j] * ncols + C_ID];
    return EH + nh;
}

/* modes.Tracker.step's measures of `side` (its living units) toward the enemy's start (ex, ey): out[0] its army (no
 * building or worker) within sqrt(near2), out[1] whether a unit of it is within sqrt(scout2), out[2] its workers
 * (types wk), out[3] its heroes' highest level (0), out[4] (need & 4) its army's food but the heroes' (food
 * table fk -> fv), out[5] (need & 1) its halls (types hk, buildings): with its own start (has_own) 1 + those
 * farther than sqrt(far2) from it and within sqrt(at2) of a gold mine (types gk, any row), else their count;
 * (need & 2) out[6] the enemy's workers (owner 1 - side, any) and out[7] its living units, then their ids.
 * `out` holds MH + 2 * nrows. Returns the length used. */
#define MH 8
int w3r_modes(const int64_t *rows, int nrows, int ncols, int side, double ex, double ey, double near2, double scout2,
              int has_own, double ox, double oy, double far2, int64_t at2, const int64_t *hk, int nhk,
              const int64_t *gk, int ngk, const int64_t *wk, int nwk, const int64_t *fk, const int64_t *fv, int nf,
              int need, int64_t *out) {
    int64_t near = 0, scout = 0, workers = 0, level = 0, food = 0, halls = 0, ne = 0, no = 0;
    int64_t *ids = out + MH;
    int n_mines = 0, *mines = NULL;
    if ((need & 1) && has_own) {
        mines = malloc(sizeof(int) * (nrows + 1));
        if (!mines) return -1;
        for (int i = 0; i < nrows; i++)
            if (in_sorted(gk, ngk, rows[(size_t)i * ncols + C_TYPE])) mines[n_mines++] = i;
    }
    if (need & 2)  /* the enemy's workers first (their ids), the side's own after them */
        for (int i = 0; i < nrows; i++) {
            const int64_t *r = rows + (size_t)i * ncols;
            if (r[C_OWNER] == 1 - side && (r[C_FLAGS] & WORKER)) ids[ne++] = r[C_ID];
        }
    for (int i = 0; i < nrows; i++) {
        const int64_t *r = rows + (size_t)i * ncols;
        int64_t fl = r[C_FLAGS];
        if (r[C_OWNER] != side || (fl & DEAD)) continue;
        if (need & 2) ids[ne + no++] = r[C_ID];
        double dx = (double)r[C_X] - ex, dy = (double)r[C_Y] - ey;
        double d2 = dx * dx + dy * dy;
        if (d2 < scout2) scout = 1;
        if (!(fl & (STRUCTURE | WORKER))) {
            if (d2 < near2) near++;
            if ((need & 4) && !(fl & HERO)) food += cls(fk, fv, nf, r[C_TYPE]);
        }
        if (in_sorted(wk, nwk, r[C_TYPE])) workers++;
        if ((fl & HERO) && r[C_HLEVEL] > level) level = r[C_HLEVEL];
        if ((need & 1) && (fl & STRUCTURE) && in_sorted(hk, nhk, r[C_TYPE])) {
            if (!has_own) { halls++; continue; }
            double fx_ = (double)r[C_X] - ox, fy_ = (double)r[C_Y] - oy;
            if (!(fx_ * fx_ + fy_ * fy_ > far2) || !n_mines) continue;
            int64_t best = 0;
            for (int j = 0; j < n_mines; j++) {
                const int64_t *g = rows + (size_t)mines[j] * ncols;
                int64_t gx = r[C_X] - g[C_X], gy = r[C_Y] - g[C_Y], d = gx * gx + gy * gy;
                if (j == 0 || d < best) best = d;
            }
            if (best < at2) halls++;
        }
    }
    if ((need & 1) && has_own) halls += 1;
    free(mines);
    out[0] = near; out[1] = scout; out[2] = workers; out[3] = level; out[4] = food; out[5] = halls; out[6] = ne;
    out[7] = no;
    return MH + ne + no;
}
