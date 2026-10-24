/* The actors' per-step entity features in C (features.Encoder.entities and View.priv, loaded with ctypes by
 * fullgame/featnative.py): the same selection and the same columns as the numpy code, in one pass over the rows
 * (numpy's calls on a step's ~100-200 rows had cost ~470 us a side-step; the actors' Python held games back).
 *
 * Rows: int64 [n][ncols], the unit columns of features.py (C_ID ... and the hero columns past them). Geometry: the
 * map's two start locations (sorted: left x, y, right x, y), DUEL_D, NEAR, SCALE, ANCHOR_X. Floats are computed in
 * double and stored as float, as numpy does when it assigns float64 results into the float32 feature array; the
 * sines and cosines of float32 inputs (facing, positions) in float as numpy's float32 ufuncs (their last bit may
 * differ: the actors store the features as float16).
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { C_STEP, C_ID, C_TYPE, C_OWNER, C_X, C_Y, C_FACING, C_HP, C_MAXHP, C_MANA, C_MAXMANA, C_ORDER, C_FLAGS, C_VIS,
       C_RESOURCE, C_HLEVEL, C_HXP, C_SKILLPTS };
#define HERO 1
#define STRUCTURE 2
#define WORKER 4
#define DEAD 1024
#define TYPE_CODE 0x1000000LL
static const int64_t FLAG_BITS[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 2048};
#define N_FLAGS 11
#define ILLUSION_COL (9 + 9)  /* FLAG_BITS' 512 */
static const int POS_FREQS[] = {1, 2, 4, 8, 16};
#define N_POS 5

static double squeeze(double d, const double *g) {
    double D = g[2] - g[0], duel = g[4], near = g[5];
    if (fabs(D - duel) < 1e-6) return d;
    double k = (duel - 2 * near) / (D - 2 * near);
    if (d <= near) return d;
    if (d >= D - near) return d - (D - duel);
    return near + (d - near) * k;
}

static void frame(double x, double y, double sign, const double *g, double *fx, double *fy) {
    double ox = sign > 0 ? g[0] : g[2], oy = sign > 0 ? g[1] : g[3];
    *fx = squeeze(sign * (x - ox), g) / g[6] + g[7];
    *fy = (y - oy) / g[6];
}

static int in_sorted(const int64_t *keys, int n, int64_t v, int64_t *val, const int64_t *vals) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (keys[mid] < v) lo = mid + 1; else hi = mid;
    }
    if (lo < n && keys[lo] == v) { if (val) *val = vals ? vals[lo] : 1; return 1; }
    return 0;
}

/* The selection (own, enemies in sight, neutral units known) and their features.
 * own_rank's PASSIVE ids (sorted); shops (sorted; with neutral_on the neutral buildings are known in the fog too).
 * queued / making (sorted ids with values), busy and lumber (sorted ids). counts out: n, n_own, n_enemy, n_neutral,
 * own_alive. f: [max_ent][F] zeroed by the caller; types, cur: [max_ent]. Returns n. */
int w3f_entities(const int64_t *rows, int nrows, int ncols, int player, double sign, const double *geo,
                 const int64_t *passive, int npassive, const int64_t *shops, int nshops, int neutral_on,
                 int max_own, int max_ent, int F, int a_pos,
                 const int64_t *queued_k, const int64_t *queued_v, int nqueued,
                 const int64_t *busy_k, int nbusy, const int64_t *lumber_k, int nlumber,
                 const int64_t *type_k, const int64_t *type_v, int ntype,
                 const int64_t *cur_k, const int64_t *cur_v, int ncur,
                 int64_t *idx, int64_t *counts, float *f, int64_t *types, int64_t *cur) {
    int64_t *own = malloc(sizeof(int64_t) * (nrows + 1)), *rank = malloc(sizeof(int64_t) * (nrows + 1));
    if (!own || !rank) { free(own); free(rank); return -1; }
    int n_own_all = 0, ranked = 0;
    for (int i = 0; i < nrows; i++) {
        const int64_t *r = rows + (size_t)i * ncols;
        if ((r[C_FLAGS] & DEAD) == 0 && r[C_OWNER] == player) own[n_own_all++] = i;
    }
    if (n_own_all > max_own) {  /* heroes, the army, buildings that make, workers, then farms and towers (stable) */
        ranked = 1;
        for (int k = 0; k < n_own_all; k++) {
            const int64_t *r = rows + (size_t)own[k] * ncols;
            int64_t fl = r[C_FLAGS];
            rank[k] = (fl & HERO) ? 0 : (fl & (STRUCTURE | WORKER)) == 0 ? 1
                      : (fl & STRUCTURE) ? (in_sorted(passive, npassive, r[C_TYPE], NULL, NULL) ? 4 : 2) : 3;
        }
        int64_t *sorted = malloc(sizeof(int64_t) * n_own_all);
        if (!sorted) { free(own); free(rank); return -1; }
        int at = 0;
        for (int64_t want = 0; want <= 4; want++)
            for (int k = 0; k < n_own_all; k++)
                if (rank[k] == want) sorted[at++] = own[k];
        memcpy(own, sorted, sizeof(int64_t) * n_own_all);
        free(sorted);
    }
    (void)ranked;
    int n_own = n_own_all < max_own ? n_own_all : max_own;
    int n = 0;
    for (int k = 0; k < n_own && n < max_ent; k++) idx[n++] = own[k];
    n_own = n;
    int n_enemy = 0, n_neutral = 0;
    for (int i = 0; i < nrows; i++) {  /* enemies in sight */
        const int64_t *r = rows + (size_t)i * ncols;
        if ((r[C_FLAGS] & DEAD) == 0 && r[C_OWNER] == 1 - player && ((r[C_VIS] >> player) & 1)) {
            if (n < max_ent) idx[n++] = i;
            n_enemy++;
        }
    }
    for (int i = 0; i < nrows; i++) {  /* neutral units in sight (and the shops, known from the start) */
        const int64_t *r = rows + (size_t)i * ncols;
        if ((r[C_FLAGS] & DEAD) || r[C_OWNER] == player || r[C_OWNER] == 1 - player) continue;
        int known = ((r[C_VIS] >> player) & 1) || (neutral_on && in_sorted(shops, nshops, r[C_TYPE], NULL, NULL));
        if (known) {
            if (n < max_ent) idx[n++] = i;
            n_neutral++;
        }
    }
    counts[0] = n; counts[1] = n_own; counts[2] = n_enemy; counts[3] = n_neutral; counts[4] = n_own_all;
    int a = 9 + N_FLAGS;
    for (int k = 0; k < n; k++) {
        const int64_t *r = rows + (size_t)idx[k] * ncols;
        float *o = f + (size_t)k * F;
        int is_own = k < n_own, is_enemy = k >= n_own && k < n_own + n_enemy;
        o[0] = is_own; o[1] = is_enemy; o[2] = !is_own && !is_enemy;
        double fx, fy;
        frame((double)r[C_X], (double)r[C_Y], sign, geo, &fx, &fy);
        o[3] = (float)fx; o[4] = (float)fy;
        double maxhp = r[C_MAXHP] > 1 ? (double)r[C_MAXHP] : 1.0;
        o[5] = (float)((double)r[C_HP] / maxhp);
        o[6] = (float)((double)r[C_MAXHP] / 1000.0);
        double maxmana = r[C_MAXMANA] > 1 ? (double)r[C_MAXMANA] : 1.0;
        o[7] = r[C_MAXMANA] > 0 ? (float)((double)r[C_MANA] / maxmana) : 0.0f;
        o[8] = (float)((double)r[C_MAXMANA] / 1000.0);
        for (int b = 0; b < N_FLAGS; b++) o[9 + b] = (r[C_FLAGS] & FLAG_BITS[b]) ? 1.0f : 0.0f;
        if (!is_own) o[ILLUSION_COL] = 0.0f;  /* (an enemy's illusion looks real) */
        o[a] = (float)((double)r[C_HLEVEL] / 10.0);
        o[a + 1] = (float)((double)r[C_RESOURCE] / 12500.0);
        if (is_own) o[a + 2] = (float)((double)r[C_SKILLPTS] / 3.0);
        float rad = (float)r[C_FACING] * (float)(M_PI / 180.0);
        o[a + 3] = sinf(rad);
        o[a + 4] = (float)sign * cosf(rad);
        o[a + 5] = r[C_ORDER] != 0;
        for (int p = 0; p < N_POS; p++) {
            float px = (float)(M_PI * POS_FREQS[p]) * o[3], py = (float)(M_PI * POS_FREQS[p]) * o[4];
            o[a_pos + 4 * p] = sinf(px); o[a_pos + 4 * p + 1] = cosf(px);
            o[a_pos + 4 * p + 2] = sinf(py); o[a_pos + 4 * p + 3] = cosf(py);
        }
        int64_t v;
        if (nqueued && in_sorted(queued_k, nqueued, r[C_ID], &v, queued_v))
            o[a + 6] = (float)((double)(v < 5 ? v : 5) / 5.0);
        if (is_own && nlumber && in_sorted(lumber_k, nlumber, r[C_ID], NULL, NULL)) o[a + 8] = 1.0f;
        if (nbusy && in_sorted(busy_k, nbusy, r[C_ID], NULL, NULL)) o[a + 7] = 1.0f;
        types[k] = in_sorted(type_k, ntype, r[C_TYPE], &v, type_v) ? v : 0;
        cur[k] = (!is_own && r[C_ORDER] >= TYPE_CODE) ? 0 : (in_sorted(cur_k, ncur, r[C_ORDER], &v, cur_v) ? v : 0);
    }
    free(own); free(rank);
    return n;
}

/* View.priv's per-side block: out[5 + 9 * side ...] for the player (side 0) and its opponent (side 1). */
void w3f_priv(const int64_t *rows, int nrows, int ncols, int player, double sign, const double *geo, float *out) {
    for (int side = 0; side < 2; side++) {
        int64_t who = side == 0 ? player : 1 - player;
        double workers = 0, army = 0, structs = 0, heroes = 0, hlevels = 0, hp = 0, maxhp = 0, sx = 0, sy = 0;
        for (int i = 0; i < nrows; i++) {
            const int64_t *r = rows + (size_t)i * ncols;
            if ((r[C_FLAGS] & DEAD) || r[C_OWNER] != who) continue;
            int64_t fl = r[C_FLAGS];
            if (fl & WORKER) workers++;
            if (fl & STRUCTURE) structs++;
            if (fl & HERO) { heroes++; hlevels += (double)r[C_HLEVEL]; }
            if (!(fl & WORKER) && !(fl & STRUCTURE)) {
                army++; hp += (double)r[C_HP]; maxhp += (double)r[C_MAXHP]; sx += (double)r[C_X]; sy += (double)r[C_Y];
            }
        }
        float *o = out + 5 + 9 * side;
        o[0] = (float)(workers / 20.0); o[1] = (float)(army / 30.0); o[2] = (float)(structs / 30.0);
        o[3] = (float)(heroes / 3.0); o[4] = (float)(hlevels / 20.0); o[5] = (float)(hp / 3000.0);
        o[6] = (float)(maxhp / 3000.0);
        if (army > 0) {
            double fx, fy;
            frame(sx / army, sy / army, sign, geo, &fx, &fy);
            o[7] = (float)fx; o[8] = (float)fy;
        }
    }
}
