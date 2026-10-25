/* The rules' per-step facts in C (loaded with ctypes by fullgame/rulesnative.py): what the heuristic teacher
 * (teacher.Teacher.commands and its expansion lesson) and the strategy modes' tracker (modes.Tracker.step) read
 * from a step's rows, in a pass or two over them instead of ~10-30 numpy calls (each ~3-50k instructions on a
 * step's ~100-400 rows: the two had cost ~700k instructions a side-step of an actor's ~2.5M).
 *
 * Each function takes one int64 array (a ctypes argument had cost ~1,600 instructions): its head (rulesnative.py's
 * A_* slots) the rows' address and shape, the caller's constants and the addresses of its tables and doubles; the
 * output after the head (HEAD on).
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

/* the head's slots (rulesnative.py's A_*): A_ROWS the rows' descriptor (their address, rows, columns) */
enum { A_ROWS, A_SIDE = 3, A_KEYS, A_VALS, A_NKEYS, A_DOUBLES, A_X0, A_X1, A_X2, A_X3, A_X4, A_X5, A_X6, A_X7, A_X8,
       A_X9 };
#define HEAD 24
#define RD(a) ((const int64_t *)(a)[A_ROWS])

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

/* Teacher.commands' facts of the side (A_X0: the tavern's type, A_X1: the merchant's, A_X2/A_X3: the army mix's
 * unit types (address, how many); the type classes). out[0..TH): its heroes (dead ones too), its living heroes, the
 * tavern's row and the merchant's (the first living of each, -1), its finished buildings' types, its army (no
 * building, worker or hero), the finished buildings' classes (altar, tiers), its altars, the row of its first
 * finished altar (-1), the row of its worker nearest the tavern (np.argmin: the first of equals; -1), its living
 * units, how many mix types; then the heroes' types, each living hero's row, id, hit points, maximum and 6 items
 * (HF), each finished building type's first (type, id) in the rows' order, the army's count of each mix type. The
 * output holds TH + (HF + 3) * nrows + the mix types. Returns its length. */
#define TH 12
#define HF 10
#define C_ITEM0 18
int w3r_teach(int64_t *a) {
    const int64_t *rows = (const int64_t *)RD(a)[0], *ck = (const int64_t *)a[A_KEYS], *cv = (const int64_t *)a[A_VALS];
    const int64_t *mix = (const int64_t *)a[A_X2];
    int nrows = (int)RD(a)[1], ncols = (int)RD(a)[2], nc = (int)a[A_NKEYS], nmix = (int)a[A_X3];
    int64_t side = a[A_SIDE], tavern = a[A_X0], merchant = a[A_X1], *out = a + HEAD;
    int64_t *seg = out + TH, *heroes = seg, *live = seg + nrows, *done = seg + (1 + HF) * (size_t)nrows,
            *counts = seg + (3 + HF) * (size_t)nrows;
    int nh = 0, nl = 0, nd = 0, na = 0, n_altars = 0, n_alive = 0;
    int64_t tav = -1, mer = -1, done_or = 0, altar = -1, near_w = -1;
    for (int j = 0; j < nmix; j++) counts[j] = 0;
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
        if (fl & HERO) {
            int64_t *h = live + (size_t)HF * nl++;
            h[0] = i; h[1] = r[C_ID]; h[2] = r[C_HP]; h[3] = r[C_MAXHP];
            for (int q = 0; q < 6; q++) h[4 + q] = ncols > C_ITEM0 + q ? r[C_ITEM0 + q] : 0;
        }
        if ((fl & STRUCTURE) && !(fl & CONSTRUCTING)) {
            done_or |= k & (K_ALTAR | K_TIER2 | K_TIER3);
            int seen = 0;
            for (int q = 0; q < nd && !seen; q++) seen = done[2 * q] == t;
            if (!seen) { done[2 * nd] = t; done[2 * nd + 1] = r[C_ID]; nd++; }
        }
        if (!(fl & (STRUCTURE | WORKER | HERO))) {
            na++;
            for (int j = 0; j < nmix; j++) if (mix[j] == t) counts[j]++;
        }
        if (k & K_ALTAR) { n_altars++; if (altar < 0 && !(fl & CONSTRUCTING)) altar = i; }
    }
    if (tav >= 0) {
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
    out[7] = n_altars; out[8] = altar; out[9] = near_w; out[10] = n_alive; out[11] = nmix;
    int at = nh;
    memmove(seg + at, live, sizeof(int64_t) * HF * nl); at += HF * nl;
    memmove(seg + at, done, sizeof(int64_t) * 2 * nd); at += 2 * nd;
    memmove(seg + at, counts, sizeof(int64_t) * nmix); at += nmix;
    return TH + at;
}
/* Teacher._expansion's facts of the side (the type classes; doubles: its start sx, sy, from_start, taken_near,
 * worker_near, camp_near): the natural (the living gold mine nearest the start but farther than from_start), and
 * around it (mx, my): out[0] 1, or -1 no natural, -2 two as near (the numpy code decides); out[1..4) its id, x and y;
 * out[4] the side's workers outside mines; out[5] its holds (halls, haunted or entangled mines) within taken_near
 * and out[6] whether those are all under construction; out[7] its workers (in mines too) within worker_near; out[8]
 * whether an own mine stands within taken_near, out[9] the first's id; out[10] whether creeps stand within
 * camp_near; out[11] its army (no building or worker), out[12] the heroes among them; out[13] whether it has a
 * worker outside mines, out[14] the id of the one nearest the natural (np.argmin: the first of equals); with holds
 * near, out[15] how many of the workers outside mines and outside worker_near go there (3 at most, the nearest
 * first: argsort's) and out[16..19) their ids (out[0] -2 when equal distances make argsort's order matter); then
 * the heroes' ids. Ids missing are 0. The output holds EH + nrows. Returns its length. */
#define EH 19
int w3r_expansion(int64_t *a) {
    const int64_t *rows = (const int64_t *)RD(a)[0], *ck = (const int64_t *)a[A_KEYS], *cv = (const int64_t *)a[A_VALS];
    const double *dd = (const double *)a[A_DOUBLES];
    int nrows = (int)RD(a)[1], ncols = (int)RD(a)[2], nc = (int)a[A_NKEYS];
    int64_t side = a[A_SIDE], *out = a + HEAD;
    double sx = dd[0], sy = dd[1], from_start = dd[2], taken_near = dd[3], worker_near = dd[4], camp_near = dd[5];
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

/* modes.Tracker.step's measures of the side (its living units) toward the enemy's start; A_X0: need, A_X1: whether
 * its own start is known, A_X2: at2, then the tables' addresses and lengths: A_X3/A_X4 the halls' types, A_X5/A_X6
 * the gold mines', A_X7/A_X8 the workers', A_KEYS/A_VALS/A_NKEYS the food table; doubles: ex, ey, near2, scout2,
 * ox, oy, far2. out[0] its army (no building or worker) within sqrt(near2) of the enemy's start, out[1] whether a
 * unit of it is within sqrt(scout2), out[2] its workers, out[3] its heroes' highest level (0), out[4] (need & 4) its
 * army's food but the heroes', out[5] (need & 1) its halls (buildings): with its own start known 1 + those farther
 * than sqrt(far2) from it and within sqrt(at2) of a gold mine (any row), else their count; (need & 2) out[6] the
 * enemy's workers (owner 1 - side, any) and out[7] its living units, then their ids. The output holds MH + 2 *
 * nrows. Returns its length (-1: out of memory). */
#define MH 8
int w3r_modes(int64_t *a) {
    const int64_t *rows = (const int64_t *)RD(a)[0], *fk = (const int64_t *)a[A_KEYS], *fv = (const int64_t *)a[A_VALS];
    const int64_t *hk = (const int64_t *)a[A_X3], *gk = (const int64_t *)a[A_X5], *wk = (const int64_t *)a[A_X7];
    const double *dd = (const double *)a[A_DOUBLES];
    int nrows = (int)RD(a)[1], ncols = (int)RD(a)[2], nf = (int)a[A_NKEYS], nhk = (int)a[A_X4],
        ngk = (int)a[A_X6], nwk = (int)a[A_X8], need = (int)a[A_X0], has_own = (int)a[A_X1];
    int64_t side = a[A_SIDE], at2 = a[A_X2], *out = a + HEAD;
    double ex = dd[0], ey = dd[1], near2 = dd[2], scout2 = dd[3], ox = dd[4], oy = dd[5], far2 = dd[6];
    int64_t near = 0, scout = 0, workers = 0, level = 0, food = 0, halls = 0, ne = 0, no = 0;
    int64_t *ids = out + MH;
    int n_far = 0, *far = NULL;  /* (the halls far from its start: at a gold mine?, after the pass) */
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
            if (!(fx_ * fx_ + fy_ * fy_ > far2)) continue;
            if (!far && !(far = malloc(sizeof(int) * (nrows + 1)))) return -1;
            far[n_far++] = i;
        }
    }
    for (int q = 0; q < n_far; q++) {  /* the nearest gold mine (any row) within sqrt(at2), in int64 as numpy's */
        const int64_t *r = rows + (size_t)far[q] * ncols;
        int64_t best = 0;
        int any = 0;
        for (int j = 0; j < nrows; j++) {
            const int64_t *g = rows + (size_t)j * ncols;
            if (!in_sorted(gk, ngk, g[C_TYPE])) continue;
            int64_t gx = r[C_X] - g[C_X], gy = r[C_Y] - g[C_Y], d = gx * gx + gy * gy;
            if (!any || d < best) best = d;
            any = 1;
        }
        if (any && best < at2) halls++;
    }
    if ((need & 1) && has_own) halls += 1;
    free(far);
    out[0] = near; out[1] = scout; out[2] = workers; out[3] = level; out[4] = food; out[5] = halls; out[6] = ne;
    out[7] = no;
    return MH + ne + no;
}
