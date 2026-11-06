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
#include <immintrin.h>
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
        o[a + 1] = (float)((double)r[C_RESOURCE] / geo[8]);  /* (a full mine of the game's map: geometry's last) */
        if (is_own) o[a + 2] = (float)((double)r[C_SKILLPTS] / 3.0);
        float rad = (float)r[C_FACING] * (float)(M_PI / 180.0);
        o[a + 3] = sinf(rad);
        o[a + 4] = (float)sign * cosf(rad);
        o[a + 5] = r[C_ORDER] != 0;
        for (int p = 0; p < N_POS && a_pos >= 0; p++) {  /* (a_pos < 0: none, a caller that zeroes them) */
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

/* float32 -> float16 with the CPU's F16C instruction (round to nearest even, as numpy's astype): numpy converts in
 * software, ~40 us for a step's ~100 x 106 features, ~7% of an actor's time. Returns 0 without F16C (the caller
 * converts with numpy then). */
__attribute__((target("avx,f16c"))) static void to_f16_vec(const float *src, uint16_t *dst, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm_storeu_si128((__m128i *)(dst + i), _mm256_cvtps_ph(_mm256_loadu_ps(src + i), _MM_FROUND_TO_NEAREST_INT));
    if (i < n) {
        float buf[8] = {0};
        uint16_t out[8];
        memcpy(buf, src + i, sizeof(float) * (n - i));
        _mm_storeu_si128((__m128i *)out, _mm256_cvtps_ph(_mm256_loadu_ps(buf), _MM_FROUND_TO_NEAREST_INT));
        memcpy(dst + i, out, sizeof(uint16_t) * (n - i));
    }
}

int w3f_to_f16(const float *src, uint16_t *dst, int n) {
    static int ok = -1;
    if (ok < 0)
        ok = __builtin_cpu_supports("f16c") && __builtin_cpu_supports("avx");
    if (!ok)
        return 0;
    to_f16_vec(src, dst, n);
    return 1;
}

/* View.step's per-step work for one player in one pass (featnative.view_step): the entities as w3f_entities (their
 * rows copied into `sel`), the hero block (A_HERO: features.Encoder._hero_block), what each own building makes
 * (A_PROD), the hit points' change since the view's last step (A_HPD; and this step's ids and hit points sorted by id,
 * for the next), the columns [active_f, f_base) zeroed (View.step's own zeroing), the privileged block's per-side
 * numbers (w3f_priv, when `priv`), and the own entities' workers on a building order or repairing and their frames
 * (play.BCAgent). Every number as the numpy code makes it.
 *
 * tabs (int64, made once per encoder): featnative.view_tables' T_ entries. keys: queued's sorted ids, their counts,
 * busy's sorted ids, lumber's sorted ids, making's sorted ids, their kinds + 1 (nq, nq, nb, nl, nm, nm long). prev,
 * prev_hp: the view's last sorted ids and their hit points (nprev each). out (int64): idx [0, E), types [E, 2E), cur [2E, 3E),
 * the sorted ids [3E, 4E), their hit points [4E, 5E), then from 5E a packed list: n, n_own, n_enemy, n_neutral,
 * own_all; 3 counts; the segments they count: the own entities' workers on a building order, those repairing, the
 * own entities' frames. Returns the packed list's length, or -1 (memory). */
enum { T_PASSIVE, T_NPASSIVE, T_SHOPS, T_NSHOPS, T_TYPEK, T_TYPEV, T_NTYPE, T_CURK, T_CURV, T_NCUR, T_GEO, T_MAXOWN,
       T_MAXENT, T_F, T_APOS, T_FBASE, T_AHPD, T_AHERO, T_APROD, T_HTYPES, T_NHERO, T_HSLOTS, T_ITEMK, T_NITEM,
       T_ITEMV, T_NEUTRAL, T_DOHERO, T_ROWCOLS, T_MAXL, T_N };
#define C_ITEM0 18
#define C_ABIL0 24
#define CONSTRUCTING 8
#define SUMMONED 256
#define REPAIR 852024

static int cmp_pair(const void *a, const void *b) {
    const int64_t *x = a, *y = b;
    return x[0] < y[0] ? -1 : x[0] > y[0] ? 1 : (x[1] < y[1] ? -1 : x[1] > y[1]);
}

static int lower(const int64_t *keys, int n, int64_t v) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (keys[mid] < v) lo = mid + 1; else hi = mid;
    }
    return lo;
}

int w3f_view(const int64_t *rows, int nrows, int ncols, int player, double sign, const int64_t *tabs,
             const int64_t *keys, int nq, int nb, int nl, int nm, const int64_t *prev, const int64_t *prev_hp, int nprev,
             int active_f, int64_t *out, float *f, int64_t *sel, float *priv) {
    const double *geo = (const double *)tabs[T_GEO];
    int E = (int)tabs[T_MAXENT], F = (int)tabs[T_F], fb = (int)tabs[T_FBASE];
    /* (the positions' sines and cosines not made when the zeroing below takes them all) */
    int a_pos_on = (active_f <= (int)tabs[T_APOS] && (int)tabs[T_APOS] + 4 * N_POS <= fb) ? -1 : (int)tabs[T_APOS];
    int64_t counts[5];
    int n = w3f_entities(rows, nrows, ncols, player, sign, geo, (const int64_t *)tabs[T_PASSIVE], (int)tabs[T_NPASSIVE],
                         (const int64_t *)tabs[T_SHOPS], (int)tabs[T_NSHOPS], (int)tabs[T_NEUTRAL], (int)tabs[T_MAXOWN],
                         E, F, a_pos_on, keys, keys + nq, nq, keys + 2 * nq, nb, keys + 2 * nq + nb, nl,
                         (const int64_t *)tabs[T_TYPEK], (const int64_t *)tabs[T_TYPEV], (int)tabs[T_NTYPE],
                         (const int64_t *)tabs[T_CURK], (const int64_t *)tabs[T_CURV], (int)tabs[T_NCUR],
                         out, counts, f, out + E, out + 2 * E);
    if (n < 0) return -1;
    int n_own = (int)counts[1];
    const int64_t *mk = keys + 2 * nq + nb + nl, *mv = mk + nm;
    int a_hpd = (int)tabs[T_AHPD], a_hero = (int)tabs[T_AHERO], a_prod = (int)tabs[T_APROD];
    int do_hero = (int)tabs[T_DOHERO] && ncols >= (int)tabs[T_ROWCOLS];
    const int64_t *htypes = (const int64_t *)tabs[T_HTYPES], *itemk = (const int64_t *)tabs[T_ITEMK];
    const double *hslots = (const double *)tabs[T_HSLOTS], *itemv = (const double *)tabs[T_ITEMV];
    int nhero = (int)tabs[T_NHERO], nitem = (int)tabs[T_NITEM], slot_w = 6 + (int)tabs[T_MAXL];
    /* (a slot: present, has a cast order, levels, the hero level its first needs, levels skipped, mana costs' count,
     * the mana costs) */
    int64_t *sid = out + 3 * E, *shp = out + 4 * E;
    int64_t *pairs = malloc(sizeof(int64_t) * 2 * (n + 1));
    if (!pairs) return -1;
    for (int k = 0; k < n; k++) {
        const int64_t *r = rows + (size_t)out[k] * ncols;
        memcpy(sel + (size_t)k * ncols, r, sizeof(int64_t) * ncols);
        float *o = f + (size_t)k * F;
        /* the hit points' change: the last step's of the same id (the first in its sorted ids), else its own */
        double last = (double)(float)r[C_HP];
        if (nprev) {
            int j = lower(prev, nprev, r[C_ID]);
            if (j > nprev - 1) j = nprev - 1;
            if (prev[j] == r[C_ID]) last = (double)(float)prev_hp[j];
        }
        double d = ((double)r[C_HP] - last) / (double)(r[C_MAXHP] > 1 ? r[C_MAXHP] : 1);
        d = d > -1.0 ? d : -1.0;
        d = d < 1.0 ? d : 1.0;
        o[a_hpd] = (float)d;
        pairs[2 * k] = r[C_ID];
        pairs[2 * k + 1] = k;
        if (k < n_own && nm) {  /* what an own building makes */
            int j = lower(mk, nm, r[C_ID]);
            if (j < nm && mk[j] == r[C_ID] && mv[j] > 0) o[a_prod + mv[j] - 1] = 1.0f;
        }
        if (do_hero && (r[C_FLAGS] & HERO)) {
            int h = lower(htypes, nhero, r[C_TYPE]);
            const double *slots = (h < nhero && htypes[h] == r[C_TYPE]) ? hslots + (size_t)h * 4 * slot_w : NULL;
            for (int s = 0; s < 4; s++) {
                int64_t lvl = r[C_ABIL0 + 2 * s], cd = r[C_ABIL0 + 2 * s + 1];
                int a = a_hero + 4 * s;
                o[a] = (float)((double)lvl / 3.0);
                const double *ab = slots ? slots + s * slot_w : NULL;
                if (k >= n_own || !ab || ab[0] == 0) continue;
                double c = (double)cd / 10.0 / 60.0;
                o[a + 2] = (float)(c < 1.0 ? c : 1.0);
                int64_t nmana = (int64_t)ab[5];
                double cost = (nmana && lvl > 0) ? ab[6 + ((lvl < nmana ? lvl : nmana) - 1)] : 0.0;
                o[a + 1] = (ab[1] != 0 && lvl > 0 && cd == 0 && (double)r[C_MANA] >= cost) ? 1.0f : 0.0f;
                int64_t skip = (int64_t)ab[4] ? (int64_t)ab[4] : 2;
                o[a + 3] = (r[C_SKILLPTS] > 0 && lvl < (int64_t)ab[2] && r[C_HLEVEL] >= (int64_t)ab[3] + lvl * skip)
                           ? 1.0f : 0.0f;
            }
            for (int s = 0; s < 6; s++) {
                int64_t it = r[C_ITEM0 + s];
                if (!it) continue;
                int b = a_hero + 4 * 4 + 5 * s;
                o[b] = 1.0f;
                int j = lower(itemk, nitem, it);
                if (j < nitem && itemk[j] == it)
                    for (int q = 0; q < 4; q++) o[b + 1 + q] = (float)itemv[4 * j + q];
            }
        }
        if (active_f < fb) memset(o + active_f, 0, sizeof(float) * (fb - active_f));
    }
    qsort(pairs, n, sizeof(int64_t) * 2, cmp_pair);  /* (by id, then position: numpy's stable argsort) */
    for (int k = 0; k < n; k++) {
        sid[k] = pairs[2 * k];
        shp[k] = sel[(size_t)pairs[2 * k + 1] * ncols + C_HP];
    }
    free(pairs);
    int64_t *pk = out + 5 * E, *seg = pk + 8;
    int at = 0;
    for (int q = 0; q < 5; q++) pk[q] = counts[q];
    pk[5] = pk[6] = pk[7] = 0;
    for (int k = 0; k < n_own; k++) {
        const int64_t *r = sel + (size_t)k * ncols;
        if ((r[C_FLAGS] & WORKER) && r[C_ORDER] >= TYPE_CODE) { seg[at++] = r[C_ID]; pk[5]++; }
    }
    for (int k = 0; k < n_own; k++) {
        const int64_t *r = sel + (size_t)k * ncols;
        if ((r[C_FLAGS] & WORKER) && r[C_ORDER] == REPAIR) { seg[at++] = r[C_ID]; pk[6]++; }
    }
    for (int k = 0; k < n_own; k++) {
        const int64_t *r = sel + (size_t)k * ncols;
        if (r[C_FLAGS] & CONSTRUCTING) { seg[at++] = r[C_ID]; pk[7]++; }
    }
    if (priv) w3f_priv(rows, nrows, ncols, player, sign, geo, priv);
    return 8 + at;
}

/* View.step's masks of one player's rows (featnative.side), into `out` (int64, at least 9 + 10 * nrows): 9 counts,
 * then the segments they count: its living buildings' ids, its living units' ids, its dead heroes (type, id, level:
 * 3 each), its heroes' types (the dead too), its finished buildings' types, (the 6th count: its army's size, no
 * segment), the types of its living units that are no building, worker, hero or summon, the rows of its living
 * heroes with a skill point (play.BCAgent._auto_skills), the rows of its living units under construction (the target
 * builder's frames). Returns the list's length. */
int w3f_side(const int64_t *rows, int nrows, int ncols, int player, int64_t *out) {
    /* one pass, each segment written at its own place (k * nrows after the counts; the dead heroes' 3 * nrows),
     * then moved together. */
    int64_t *cnt = out, *seg = out + 9;
    int64_t *built = seg, *alive_ids = seg + nrows, *dead = seg + 2 * (size_t)nrows, *heroes = seg + 5 * (size_t)nrows,
            *done = seg + 6 * (size_t)nrows, *mix = seg + 7 * (size_t)nrows, *skill = seg + 8 * (size_t)nrows,
            *frames = seg + 9 * (size_t)nrows;
    int64_t nb = 0, na = 0, nd = 0, nh = 0, nf = 0, nm = 0, ns = 0, nc = 0, army = 0;
    for (int i = 0; i < nrows; i++) {
        const int64_t *r = rows + (size_t)i * ncols;
        if (r[C_OWNER] != player) continue;
        int64_t fl = r[C_FLAGS];
        if (!(fl & DEAD)) {
            alive_ids[na++] = r[C_ID];
            if (fl & STRUCTURE) {
                built[nb++] = r[C_ID];
                if (!(fl & CONSTRUCTING)) done[nf++] = r[C_TYPE];
            } else if (!(fl & WORKER)) army++;
            if ((fl & HERO) && r[C_SKILLPTS] > 0) skill[ns++] = i;
            if (fl & CONSTRUCTING) frames[nc++] = i;
            if (!(fl & (STRUCTURE | WORKER | HERO | SUMMONED))) mix[nm++] = r[C_TYPE];
        } else if (fl & HERO) {
            dead[3 * nd] = r[C_TYPE]; dead[3 * nd + 1] = r[C_ID]; dead[3 * nd + 2] = r[C_HLEVEL];
            nd++;
        }
        if (fl & HERO) heroes[nh++] = r[C_TYPE];
    }
    cnt[0] = nb; cnt[1] = na; cnt[2] = nd; cnt[3] = nh; cnt[4] = nf; cnt[5] = army; cnt[6] = nm; cnt[7] = ns; cnt[8] = nc;
    int64_t at = nb;
    memmove(seg + at, alive_ids, sizeof(int64_t) * na); at += na;
    memmove(seg + at, dead, sizeof(int64_t) * 3 * nd); at += 3 * nd;
    memmove(seg + at, heroes, sizeof(int64_t) * nh); at += nh;
    memmove(seg + at, done, sizeof(int64_t) * nf); at += nf;
    memmove(seg + at, mix, sizeof(int64_t) * nm); at += nm;
    memmove(seg + at, skill, sizeof(int64_t) * ns); at += ns;
    memmove(seg + at, frames, sizeof(int64_t) * nc); at += nc;
    return 9 + (int)at;
}

/* w3f_side with its arguments in one int64 array: rows, nrows, ncols, player, out. */
int w3f_side_a(const int64_t *a) {
    return w3f_side((const int64_t *)a[0], (int)a[1], (int)a[2], (int)a[3], (int64_t *)a[4]);
}

/* play.BCAgent._entangle's distances: of the player's living finished buildings whose type is in `trees` (sorted), in
 * their order, the living gold mine of type `ngol` nearest each (the first of the nearest), unless it is farther than
 * `near` or a living mine of type `egol` stands within `held_near` of it: (tree id, mine id) pairs into out. Returns the
 * number of pairs. */
int w3f_entangle(const int64_t *rows, int nrows, int ncols, int player, const int64_t *trees, int ntrees, int64_t ngol,
                 int64_t egol, double near, double held_near, int64_t *out) {
    int n = 0;
    for (int i = 0; i < nrows; i++) {
        const int64_t *t = rows + (size_t)i * ncols;
        if (t[C_OWNER] != player || (t[C_FLAGS] & (CONSTRUCTING | DEAD)) || !in_sorted(trees, ntrees, t[C_TYPE], NULL, NULL))
            continue;
        int best = -1;
        double bd = 0.0;
        for (int j = 0; j < nrows; j++) {
            const int64_t *m = rows + (size_t)j * ncols;
            if (m[C_TYPE] != ngol || (m[C_FLAGS] & DEAD)) continue;
            double d = hypot((double)(m[C_X] - t[C_X]), (double)(m[C_Y] - t[C_Y]));
            if (best < 0 || d < bd) { best = j; bd = d; }
        }
        if (best < 0) return -1;  /* (no mine: the caller's) */
        if (bd > near) continue;
        const int64_t *m = rows + (size_t)best * ncols;
        int held = 0;
        for (int j = 0; j < nrows && !held; j++) {
            const int64_t *h = rows + (size_t)j * ncols;
            if (h[C_TYPE] != egol || (h[C_FLAGS] & DEAD)) continue;
            if (hypot((double)(h[C_X] - m[C_X]), (double)(h[C_Y] - m[C_Y])) < held_near) held = 1;
        }
        if (held) continue;
        out[2 * n] = t[C_ID];
        out[2 * n + 1] = m[C_ID];
        n++;
    }
    return n;
}

/* w3f_view with its arguments but `sign` in one int64 array (featnative.view_step: ctypes converts each argument it
 * passes, ~1,200 instructions each): rows, nrows, ncols, player, tabs, keys, nq, nb, nl, nm, prev, prev_hp, nprev,
 * active_f, out, f, sel, priv. */
int w3f_view_a(const int64_t *a, double sign) {
    return w3f_view((const int64_t *)a[0], (int)a[1], (int)a[2], (int)a[3], sign, (const int64_t *)a[4],
                    (const int64_t *)a[5], (int)a[6], (int)a[7], (int)a[8], (int)a[9], (const int64_t *)a[10],
                    (const int64_t *)a[11], (int)a[12], (int)a[13], (int64_t *)a[14], (float *)a[15], (int64_t *)a[16],
                    (float *)a[17]);
}

/* features.affordable: [n] 0/1 the order classes the player can pay for (costs [n][3]: gold, lumber, food); `nkept`
 * classes k1, k2 (their gold and lumber summed: rg, rl) kept from the others, each as it was before. */
void w3f_afford(const int64_t *restrict c, int n, int64_t gold, int64_t lumber, int64_t fu, int64_t fc, int nkept,
                int64_t k1, int64_t k2, int64_t rg, int64_t rl, uint8_t *restrict avail) {
    if (fc > 100) fc = 100;
    int64_t g2 = gold - rg > 0 ? gold - rg : 0, l2 = lumber - rl > 0 ? lumber - rl : 0;
    for (int i = 0; i < n; i++) {  /* (without branches: ~5x fewer instructions) */
        int64_t g = c[3 * i], l = c[3 * i + 1], f = c[3 * i + 2];
        avail[i] = (uint8_t)((g <= gold) & (l <= lumber) & ((f == 0) | (fu + f <= fc)));
    }
    if (nkept) {
        uint8_t v1 = avail[k1], v2 = nkept >= 2 ? avail[k2] : 0;
        for (int i = 0; i < n; i++) {
            int64_t g = c[3 * i], l = c[3 * i + 1];
            avail[i] &= (uint8_t)(((g <= g2) & (l <= l2)) | (g + l == 0));
        }
        avail[k1] = v1;
        if (nkept >= 2) avail[k2] = v2;
    }
    if (n) avail[0] = 1;
}

/* w3f_afford with its arguments in one int64 array: c, n, gold, lumber, fu, fc, nkept, k1, k2, rg, rl, avail. */
void w3f_afford_a(const int64_t *a) {
    w3f_afford((const int64_t *)a[0], (int)a[1], a[2], a[3], a[4], a[5], (int)a[6], a[7], a[8], a[9], a[10],
               (uint8_t *)a[11]);
}

/* macros.natural, site and camp in one pass: of the living rows whose type is in `mines` (sorted codes), the one
 * nearest (sx, sy) farther than from_start (numpy's hypot is libm's): out[0] its row (-1 none; -2 another one as near,
 * which numpy's argsort orders: the caller's numpy then), out[1..3] its id, x, y; out[4] whether a living unit of
 * player 0 or 1 whose type is in `holds` stands within taken_near of it, out[5] whether a living unit of player 12
 * stands within camp_near. */
void w3f_natural(const int64_t *rows, int nrows, int ncols, double sx, double sy, const int64_t *mines, int nmines,
                 const int64_t *holds, int nholds, double from_start, double taken_near, double camp_near, int64_t *out) {
    int best = -1, tie = 0;
    double bd = 0.0;
    for (int i = 0; i < nrows; i++) {
        const int64_t *r = rows + (size_t)i * ncols;
        if ((r[C_FLAGS] & DEAD) || !in_sorted(mines, nmines, r[C_TYPE], NULL, NULL)) continue;
        double d = hypot((double)r[C_X] - sx, (double)r[C_Y] - sy);
        if (!(d > from_start)) continue;
        if (best < 0 || d < bd) { best = i; bd = d; tie = 0; }
        else if (d == bd) tie = 1;
    }
    for (int q = 0; q < 6; q++) out[q] = 0;
    out[0] = best < 0 ? -1 : tie ? -2 : best;
    if (best < 0 || tie) return;
    const int64_t *m = rows + (size_t)best * ncols;
    out[1] = m[C_ID]; out[2] = m[C_X]; out[3] = m[C_Y];
    double mx = (double)m[C_X], my = (double)m[C_Y];
    for (int i = 0; i < nrows; i++) {
        const int64_t *r = rows + (size_t)i * ncols;
        if (r[C_FLAGS] & DEAD) continue;
        if (r[C_OWNER] != 12 && !(r[C_OWNER] <= 1 && in_sorted(holds, nholds, r[C_TYPE], NULL, NULL))) continue;
        double d = hypot((double)r[C_X] - mx, (double)r[C_Y] - my);
        if (r[C_OWNER] == 12 && d < camp_near) out[5] = 1;
        else if (r[C_OWNER] <= 1 && d < taken_near) out[4] = 1;
    }
}

/* features.spell_any (FUTILE_MASK) with its arguments in one int64 array: rows, nrows, ncols, player, the sorted
 * undead, mechanical, ancient and spell immune types (a pointer and a count each), n_sp, the spells (per spell: 1 if it
 * heals the undead, its caster's type), found (out: int64 [n_sp]). Per spell, whether a row is its target other than one
 * of the player's units of its caster's type: living and no building, neither mechanical nor ancient; one of the
 * player's of the kind it heals below full health, or another player's of the other kind that the player sees, not
 * spell immune (features.spell_targets' masks). */
void w3f_spell_any_a(const int64_t *a) {
    const int64_t *rows = (const int64_t *)a[0];
    int nrows = (int)a[1], ncols = (int)a[2];
    int64_t player = a[3];
    const int64_t *und = (const int64_t *)a[4], *mech = (const int64_t *)a[6], *anc = (const int64_t *)a[8],
                  *imm = (const int64_t *)a[10], *sp = (const int64_t *)a[13];
    int nu = (int)a[5], nm = (int)a[7], na = (int)a[9], ni = (int)a[11], n_sp = (int)a[12];
    int64_t *found = (int64_t *)a[14];
    int left = n_sp;
    for (int s = 0; s < n_sp; s++) found[s] = 0;
    for (int i = 0; i < nrows && left; i++) {
        const int64_t *r = rows + (size_t)i * ncols;
        int64_t fl = r[C_FLAGS], ty = r[C_TYPE];
        if ((fl & (DEAD | STRUCTURE)) || in_sorted(mech, nm, ty, NULL, NULL) || in_sorted(anc, na, ty, NULL, NULL))
            continue;
        int64_t u = in_sorted(und, nu, ty, NULL, NULL);
        if (r[C_OWNER] == player) {
            if (r[C_HP] < r[C_MAXHP])
                for (int s = 0; s < n_sp; s++)
                    if (!found[s] && u == sp[2 * s] && ty != sp[2 * s + 1]) { found[s] = 1; left--; }
        } else if (((r[C_VIS] >> player) & 1) == 1 && !in_sorted(imm, ni, ty, NULL, NULL)) {
            for (int s = 0; s < n_sp; s++)
                if (!found[s] && u != sp[2 * s]) { found[s] = 1; left--; }
        }
    }
}

/* View.futile's mines (features.mine_workers) with its arguments in one int64 array: rows, nrows, ncols, player, the
 * mine's type, the sorted mining orders (a pointer and a count), near (a double's bits), out. Of the player's living
 * finished units of the mine's type, in their order, each one's id and how many of the player's living workers work it:
 * within `near` of it (numpy's hypot is libm's, of the int64 differences) and on a mining order or inside it (hidden).
 * Returns the number of mines. */
#define HIDDEN 16
int w3f_mine_counts_a(const int64_t *a) {
    const int64_t *rows = (const int64_t *)a[0];
    int nrows = (int)a[1], ncols = (int)a[2];
    int64_t player = a[3], mtype = a[4];
    const int64_t *mining = (const int64_t *)a[5];
    int nmining = (int)a[6];
    double near;
    memcpy(&near, &a[7], sizeof near);
    int64_t *out = (int64_t *)a[8];
    int n = 0;
    for (int i = 0; i < nrows; i++) {
        const int64_t *m = rows + (size_t)i * ncols;
        if (m[C_OWNER] != player || (m[C_FLAGS] & (DEAD | CONSTRUCTING)) || m[C_TYPE] != mtype) continue;
        int64_t k = 0;
        for (int j = 0; j < nrows; j++) {
            const int64_t *w = rows + (size_t)j * ncols;
            int64_t fl = w[C_FLAGS];
            if (w[C_OWNER] != player || (fl & DEAD) || !(fl & WORKER)) continue;
            if (!(fl & HIDDEN) && !in_sorted(mining, nmining, w[C_ORDER], NULL, NULL)) continue;
            if (hypot((double)(w[C_X] - m[C_X]), (double)(w[C_Y] - m[C_Y])) < near) k++;
        }
        out[2 * n] = m[C_ID];
        out[2 * n + 1] = k;
        n++;
    }
    return n;
}

/* float16 -> float32 with the CPU's F16C instruction (exact, as numpy's astype; a NaN's bits as numpy keeps them: F16C
 * would set its quiet bit): the gateway's remote view steps into the shared rows (selfplay.SharedRows.write). Returns 0
 * without F16C (the caller converts with numpy then). */
__attribute__((target("avx,f16c"))) static void to_f32_vec(const uint16_t *src, float *dst, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(src + i))));
    if (i < n) {
        uint16_t h8[8] = {0};
        float f8[8];
        memcpy(h8, src + i, sizeof(uint16_t) * (n - i));
        _mm256_storeu_ps(f8, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)h8)));
        memcpy(dst + i, f8, sizeof(float) * (n - i));
    }
    for (i = 0; i < n; i++) {
        uint16_t h = src[i];
        if ((h & 0x7c00u) == 0x7c00u && (h & 0x03ffu)) {  /* (a NaN: numpy's npy_halfbits_to_floatbits) */
            uint32_t b = ((uint32_t)(h & 0x8000u) << 16) | 0x7f800000u | ((uint32_t)(h & 0x03ffu) << 13);
            memcpy(dst + i, &b, sizeof b);
        }
    }
}

int w3f_to_f32(const uint16_t *src, float *dst, int n) {
    static int ok = -1;
    if (ok < 0)
        ok = __builtin_cpu_supports("f16c") && __builtin_cpu_supports("avx");
    if (!ok)
        return 0;
    to_f32_vec(src, dst, n);
    return 1;
}

/* features.View.last_features with its arguments in one int64 array: sel (int64 [n][ncols]), n, ncols, n_own, f (float32
 * [n][F]), F, a_last, the view's last orders (records of View.last_packed: unit id, kind, x, y, target id, the order's
 * step t0; int64, int64, double, double, int64, int64), their count, t, sign, scale, age (the doubles' bits).
 * f[k][a_last + 4] = 1 for every own row k; an own row's unit with a last order: its age (t - t0) / age (at most 1), the
 * point it was sent to (kind 1) or its target's place (kind 2, the target among sel's rows: its last), relative to it (x
 * mirrored by sign) / scale clipped to [-2, 2], and its kind's flag: View.step's numbers in double, stored as float
 * (numpy's float64 into the float32 features). */
typedef struct { int64_t id, kind; double x, y; int64_t target, t0; } LastOrder;

void w3f_last_a(const int64_t *a) {
    const int64_t *sel = (const int64_t *)a[0];
    int n = (int)a[1], ncols = (int)a[2], n_own = (int)a[3];
    float *f = (float *)a[4];
    int F = (int)a[5], al = (int)a[6];
    const char *recs = (const char *)a[7];
    int m = (int)a[8];
    int64_t t = a[9];
    double sign, scale, age;
    memcpy(&sign, &a[10], sizeof sign);
    memcpy(&scale, &a[11], sizeof scale);
    memcpy(&age, &a[12], sizeof age);
    for (int k = 0; k < n_own; k++) {
        float *fk = f + (size_t)k * F;
        fk[al + 4] = 1.0f;
        int64_t id = sel[(size_t)k * ncols + C_ID];
        LastOrder v;
        int found = 0;
        for (int q = 0; q < m && !found; q++) {
            memcpy(&v, recs + q * sizeof(LastOrder), sizeof v);
            found = v.id == id;
        }
        if (!found) continue;
        int kind = (int)v.kind;
        double x = v.x, y = v.y;
        double g = (double)(t - v.t0) / age;
        fk[al + 4] = (float)(g < 1.0 ? g : 1.0);
        int j = -1;
        if (kind == 2)
            for (int i = n - 1; i >= 0; i--)
                if (sel[(size_t)i * ncols + C_ID] == v.target) { j = i; break; }
        if (j >= 0) {
            x = (double)sel[(size_t)j * ncols + C_X];
            y = (double)sel[(size_t)j * ncols + C_Y];
        }
        if (kind == 1 || j >= 0) {
            double vx = sign * (x - (double)sel[(size_t)k * ncols + C_X]) / scale;
            double vy = (y - (double)sel[(size_t)k * ncols + C_Y]) / scale;
            fk[al] = (float)(vx < -2.0 ? -2.0 : vx > 2.0 ? 2.0 : vx);
            fk[al + 1] = (float)(vy < -2.0 ? -2.0 : vy > 2.0 ? 2.0 : vy);
        }
        if (kind) fk[al + 1 + kind] = 1.0f;
    }
}

/* ---- --obs-full (fullgame/obsfull.py): the block of OBS_F channels at the features' end for a step's entities and the
 * new kinds' rows after them, the numbers obsfull's Python makes (its doubles stored as float, its float32 sums in its
 * order).
 *
 * A view's units' state (the harness's events 101, 102, 105: State.extras) is kept as each unit's block K_CLASS ..
 * K_PROD (State.row's numbers) in arrays its State owns: an open-addressing table hk (keys; OBS_EMPTY: free) and hs
 * (their slots) of H entries (a power of two), the blocks blk [cap][OBS_BLK], and meta {H, cap, count}. The caller
 * keeps room: count plus the step's 101 events at most cap and H / 2. A 101 makes the unit's block anew (no buffs, no
 * charges), a 102 adds its buff's code to it (State.row's sum, in the events' order), a 105 sets its charges. */
enum { K_KIND = 0, K_CLASS = 3, K_MAGIC = 11, K_ARMOR = 13, K_SPEED = 14, K_SPEED_R = 15, K_HAVE = 16, K_BUFF = 17,
       K_UPG = 41, K_ITEM = 43, K_CHARGE = 59, K_PROD = 65, K_AGE = 68, K_RADIUS = 69, OBS_F = 70, BUFF_D = 24,
       ITEM_D = 16, OBS_BLK = K_PROD - K_CLASS, MAX_CORPSES = 8, MAX_ITEMS = 6, MAX_AREAS = 4 };
#define EV_STATE 101
#define EV_BUFF 102
#define EV_CHARGES 105
#define OBS_EMPTY INT64_MIN
#define OBS_NOMEM (-(1 << 30))

/* The layout this file computes, into out (obsfull checks it against its own before it uses w3f_obs_a). */
int w3f_obs_layout(int64_t *out) {
    const int64_t v[] = {K_KIND, K_CLASS, K_MAGIC, K_ARMOR, K_SPEED, K_SPEED_R, K_HAVE, K_BUFF, K_UPG, K_ITEM, K_CHARGE,
                         K_PROD, K_AGE, K_RADIUS, OBS_F, BUFF_D, ITEM_D, MAX_CORPSES, MAX_ITEMS, MAX_AREAS,
                         EV_STATE, EV_BUFF, EV_CHARGES, C_ITEM0, OBS_NOMEM};
    memcpy(out, v, sizeof v);
    return (int)(sizeof v / sizeof v[0]);
}

static uint64_t obs_hash(int64_t k) {
    uint64_t x = (uint64_t)k * 0x9E3779B97F4A7C15ULL;
    return x ^ (x >> 31);
}

/* The entry of `key` in the table, or the free one where it would go. */
static int64_t obs_find(const int64_t *hk, int64_t H, int64_t key) {
    uint64_t m = (uint64_t)H - 1, i = obs_hash(key) & m;
    while (hk[i] != OBS_EMPTY && hk[i] != key) i = (i + 1) & m;
    return (int64_t)i;
}

static int64_t obs_slot(const int64_t *hk, const int64_t *hs, int64_t H, int64_t key) {
    int64_t i = obs_find(hk, H, key);
    return hk[i] == key ? hs[i] : -1;
}

/* The table grown: every key of (hk, hs, H) into (nk, ns, nH), nk all OBS_EMPTY. */
void w3f_obs_rehash(const int64_t *hk, const int64_t *hs, int64_t H, int64_t *nk, int64_t *ns, int64_t nH) {
    for (int64_t i = 0; i < H; i++) {
        if (hk[i] == OBS_EMPTY) continue;
        int64_t j = obs_find(nk, nH, hk[i]);
        nk[j] = hk[i];
        ns[j] = hs[i];
    }
}

static int has(const int64_t *ids, int n, int64_t v) {
    int i = lower(ids, n, v);
    return i < n && ids[i] == v;
}

static const float *code_row(const int64_t *ids, int n, const float *codes, int d, int64_t v) {
    return codes + (size_t)lower(ids, n, v) * d;  /* (the caller made sure it is there: has) */
}

/* A code (d, v) missing: into `missing` (pairs, `room` of them) once; the count so far. */
static int need(int64_t *missing, int room, int miss, int64_t d, int64_t v) {
    for (int q = 0; q < miss && q < room; q++)
        if (missing[2 * q] == d && missing[2 * q + 1] == v) return miss;
    if (miss < room) { missing[2 * miss] = d; missing[2 * miss + 1] = v; }
    return miss + 1;
}

static int64_t level_of(const int64_t *lv, int n, int64_t research) {  /* (ids, then their levels: a dict's get(c, 0)) */
    for (int i = 0; i < n; i++)
        if (lv[i] == research) return lv[n + i];
    return 0;
}

/* obsfull.apply's numbers after State.events, its arguments in one int64 array (the doubles' bits):
 *  0 sel (int64 [n + room][ncols]: the step's entities, then room for the new rows), 1 n, 2 ncols, 3 n_own, 4 seen (the
 *  entities in sight: the units' state, upgrades and progress only for them), 5 f (float32 [n + room][F]), 6 F, 7 A (the
 *  block's first channel), 8 types, 9 cur (int64 [n + room]), 10 room (new rows at most), 11 player, 12 t, 13 the
 *  step's seconds, 14 sign, 15 State.scale, 16-19 the store (hk, hs, blk, meta), 20 the step's events 101, 102, 105
 *  (int64 [ne][5]: step, kind, a, b, c), 21 ne, 22-24 the buffs' codes (sorted ids, their count, float32 [n][BUFF_D]),
 *  25-27 the items' codes (the same, ITEM_D), 28 missing (int64 pairs out: (d, id) of a code not in the tables), 29 its
 *  room in pairs, 30-32 the move speeds (sorted types, their speeds as doubles, count), 33-36 the types' upgrades
 *  (sorted types, count, int64 [count][3] (first, weapon count, armor count) into 36's research ids), 37-38 the player's
 *  levels (its research ids, then their levels; count), 39-40 the others' (State.enemy_levels), 41-43 State.since
 *  ((uid, t0) pairs, count, their seconds as doubles), 44-46 State.built (the same), 47-48 the corpses in sight ((type,
 *  owner, x, y, t0) each, in State.corpses' order; count), 49-50 the items on the ground (the first MAX_ITEMS in sight:
 *  (item, x, y, powerup)), 51-53 the area spells (the first MAX_AREAS in sight: (ability, x, y, owner); count; (radius,
 *  the step it ends) doubles), 54 the view's tables (featnative.view_tables: the geometry, the types' indices), 55 pos
 *  (float32 [room][5][2] out: a new row's x and y times each frequency's pi * k, the products numpy's sin and cos then
 *  take), 56 whether pos is wanted, 57 whether the rows have the heroes' item columns.
 * Returns the new rows written after the entities (their f rows, sel rows, types and cur whole), -k when k codes are
 * missing (nothing changed: the caller adds them and calls again), or OBS_NOMEM. */
int w3f_obs_a(const int64_t *a) {
    int64_t *sel = (int64_t *)a[0];
    int n = (int)a[1], ncols = (int)a[2], n_own = (int)a[3], seen = (int)a[4];
    float *f = (float *)a[5];
    int F = (int)a[6], A = (int)a[7];
    int64_t *types = (int64_t *)a[8], *cur = (int64_t *)a[9];
    int room = (int)a[10], p = (int)a[11];
    int64_t t = a[12];
    double step_s, sign, scale;
    memcpy(&step_s, &a[13], sizeof step_s);
    memcpy(&sign, &a[14], sizeof sign);
    memcpy(&scale, &a[15], sizeof scale);
    int64_t *hk = (int64_t *)a[16], *hs = (int64_t *)a[17], *meta = (int64_t *)a[19];
    float *blk = (float *)a[18];
    const int64_t *ev = (const int64_t *)a[20];
    int ne = (int)a[21];
    const int64_t *bid = (const int64_t *)a[22], *iid = (const int64_t *)a[25];
    int nbid = (int)a[23], niid = (int)a[26];
    const float *bcode = (const float *)a[24], *icode = (const float *)a[27];
    int64_t *missing = (int64_t *)a[28];
    int miss_room = (int)a[29];
    const int64_t *spk = (const int64_t *)a[30];
    const double *spv = (const double *)a[31];
    int nsp = (int)a[32];
    const int64_t *uk = (const int64_t *)a[33];
    int nuk = (int)a[34];
    const int64_t *uinfo = (const int64_t *)a[35], *ures = (const int64_t *)a[36];
    const int64_t *lv_own = (const int64_t *)a[37], *lv_oth = (const int64_t *)a[39];
    int nlo = (int)a[38], nle = (int)a[40];
    const int64_t *since = (const int64_t *)a[41], *built = (const int64_t *)a[44];
    int nsince = (int)a[42], nbuilt = (int)a[45];
    const double *since_s = (const double *)a[43], *built_s = (const double *)a[46];
    const int64_t *corpses = (const int64_t *)a[47], *ground = (const int64_t *)a[49], *areas = (const int64_t *)a[51];
    int ncorpse = (int)a[48], nground = (int)a[50], narea = (int)a[52];
    const double *area_d = (const double *)a[53];
    const int64_t *tabs = (const int64_t *)a[54];
    float *pos = (float *)a[55];
    int pos_on = (int)a[56], items_on = (int)a[57];
    int64_t H = meta[0];

    /* the codes it needs, before anything changes */
    int miss = 0;
    for (int e = 0; e < ne; e++)
        if (ev[5 * e + 1] == EV_BUFF && !has(bid, nbid, ev[5 * e + 3]))
            miss = need(missing, miss_room, miss, BUFF_D, ev[5 * e + 3]);
    if (items_on)
        for (int k = 0; k < n; k++) {
            const int64_t *r = sel + (size_t)k * ncols;
            if (!(r[C_FLAGS] & HERO)) continue;
            for (int s = 0; s < 6; s++)
                if (r[C_ITEM0 + s] && !has(iid, niid, r[C_ITEM0 + s]))
                    miss = need(missing, miss_room, miss, ITEM_D, r[C_ITEM0 + s]);
        }
    for (int g = 0; g < nground; g++)
        if (!has(iid, niid, ground[4 * g])) miss = need(missing, miss_room, miss, ITEM_D, ground[4 * g]);
    for (int r = 0; r < narea; r++)
        if (!has(bid, nbid, areas[4 * r])) miss = need(missing, miss_room, miss, BUFF_D, areas[4 * r]);
    if (miss) return -miss;

    /* the units' state: the step's events 101, 102, 105 */
    for (int e = 0; e < ne; e++) {
        const int64_t *x = ev + 5 * e;
        int64_t kind = x[1], u = x[2], b = x[3], c = x[4];
        if (kind == EV_STATE) {
            int64_t i = obs_find(hk, H, u), s;
            if (hk[i] == u) s = hs[i];
            else { s = meta[2]++; hk[i] = u; hs[i] = s; }
            float *r = blk + (size_t)s * OBS_BLK;
            memset(r, 0, OBS_BLK * sizeof *r);
            for (int q = 0; q < 8; q++)
                if (b >> q & 1) r[q] = 1.0f;
            r[K_MAGIC - K_CLASS] = (float)((double)(b >> 8 & 15) / 3.0);
            r[K_MAGIC - K_CLASS + 1] = (float)((double)(b >> 12 & 15) / 3.0);
            r[K_ARMOR - K_CLASS] = (float)((double)((c & 4095) - 2048) / 100.0);
            r[K_SPEED - K_CLASS] = (float)((double)(c >> 12) / 400.0);
            r[K_HAVE - K_CLASS] = 1.0f;
        } else if (kind == EV_BUFF || kind == EV_CHARGES) {
            int64_t s = obs_slot(hk, hs, H, u);
            if (s < 0) continue;  /* (no state for it: State.events ignores it) */
            float *r = blk + (size_t)s * OBS_BLK;
            if (kind == EV_BUFF) {
                const float *cd = code_row(bid, nbid, bcode, BUFF_D, b);
                for (int j = 0; j < BUFF_D; j++) r[K_BUFF - K_CLASS + j] += cd[j];
            } else
                for (int q = 0; q < 6; q++) {
                    int64_t ch = ((q < 3 ? b : c) >> (10 * (q % 3))) & 1023;
                    float v = (float)ch / 5.0f;  /* (numpy's float32 division, then np.minimum) */
                    r[K_CHARGE - K_CLASS + q] = v < 2.0f ? v : 2.0f;
                }
        }
    }

    /* the entities in sight: their state and its speed against their type's, then their owner's upgrade levels */
    for (int k = 0; k < seen && meta[2]; k++) {
        const int64_t *r = sel + (size_t)k * ncols;
        int64_t s = obs_slot(hk, hs, H, r[C_ID]);
        if (s < 0) continue;
        float *fk = f + (size_t)k * F + A;
        const float *b = blk + (size_t)s * OBS_BLK;
        memcpy(fk + K_CLASS, b, OBS_BLK * sizeof *b);
        int i = lower(spk, nsp, r[C_TYPE]);
        double base = i < nsp && spk[i] == r[C_TYPE] ? spv[i] : 0.0;
        fk[K_SPEED_R] = base > 0 ? (float)((double)b[K_SPEED - K_CLASS] * 400.0 / base) : 0.0f;
    }
    for (int k = 0; k < seen; k++) {
        const int64_t *r = sel + (size_t)k * ncols;
        int64_t o = r[C_OWNER];
        if (o != p && o != 1 - p) continue;
        int i = lower(uk, nuk, r[C_TYPE]);
        if (i >= nuk || uk[i] != r[C_TYPE]) continue;
        const int64_t *info = uinfo + 3 * (size_t)i, *lv = o == p ? lv_own : lv_oth;
        int nl = o == p ? nlo : nle;
        double v[2];
        int64_t at = info[0];
        for (int w = 0; w < 2; w++) {
            int64_t cnt = info[1 + w], best = 0;
            for (int64_t j = 0; j < cnt; j++) {
                int64_t l = level_of(lv, nl, ures[at + j]);
                if (j == 0 || l > best) best = l;
            }
            v[w] = cnt ? (double)best / 3.0 : 0.0;
            at += cnt;
        }
        if (v[0] != 0.0 || v[1] != 0.0) {
            float *fk = f + (size_t)k * F + A;
            fk[K_UPG] = (float)v[0];
            fk[K_UPG + 1] = (float)v[1];
        }
    }
    /* every hero's items */
    if (items_on)
        for (int k = 0; k < n; k++) {
            const int64_t *r = sel + (size_t)k * ncols;
            if (!(r[C_FLAGS] & HERO)) continue;
            float *fk = f + (size_t)k * F + A + K_ITEM;
            for (int s = 0; s < 6; s++) {
                if (!r[C_ITEM0 + s]) continue;
                const float *cd = code_row(iid, niid, icode, ITEM_D, r[C_ITEM0 + s]);
                for (int j = 0; j < ITEM_D; j++) fk[j] += cd[j];
            }
        }
    /* production's and construction's progress (at an id's last entity) */
    for (int w = 0; w < 2; w++) {
        const int64_t *d = w ? built : since;
        const double *secs = w ? built_s : since_s;
        int nd = w ? nbuilt : nsince;
        for (int e = 0; e < nd; e++) {
            int64_t uid = d[2 * e], t0 = d[2 * e + 1];
            int k = -1;
            for (int i = n - 1; i >= 0 && k < 0; i--)
                if (sel[(size_t)i * ncols + C_ID] == uid) k = i;
            if (k < 0 || secs[e] <= 0 || k >= seen || (!w && k >= n_own)) continue;
            if (w && !(sel[(size_t)k * ncols + C_FLAGS] & CONSTRUCTING)) continue;
            double total = secs[e] * scale, done = (double)(t - t0) * step_s, q = done / total;
            float *fk = f + (size_t)k * F + A;
            fk[K_PROD + 2 * w] = (float)(1.0 < q ? 1.0 : q);  /* (Python's min(q, 1.0)) */
            if (!w) {
                double left = total - done;
                fk[K_PROD + 1] = (float)((0.0 > left ? 0.0 : left) / 60.0);
            }
        }
    }

    /* the new kinds after the entities: corpses (the MAX_CORPSES nearest the player's own units when more are in
     * sight), items on the ground, area spells */
    if (room <= 0) return 0;
    int pick[MAX_CORPSES], npick = 0;
    if (ncorpse > MAX_CORPSES && n_own) {
        double *dist = malloc(sizeof(double) * (size_t)ncorpse);
        char *used = calloc((size_t)ncorpse, 1);
        if (!dist || !used) {
            free(dist);
            free(used);
            return OBS_NOMEM;
        }
        for (int i = 0; i < ncorpse; i++) {  /* (numpy's: the squares' sum, its least over the own units) */
            double cx = (double)corpses[5 * i + 2], cy = (double)corpses[5 * i + 3], best = 0.0;
            for (int j = 0; j < n_own; j++) {
                const int64_t *r = sel + (size_t)j * ncols;
                double dx = cx - (double)r[C_X], dy = cy - (double)r[C_Y], dd = dx * dx + dy * dy;
                if (j == 0 || dd < best) best = dd;
            }
            dist[i] = best;
        }
        for (; npick < MAX_CORPSES; npick++) {  /* (a stable argsort's first: the nearest, the first of equals) */
            int b = -1;
            for (int i = 0; i < ncorpse; i++)
                if (!used[i] && (b < 0 || dist[i] < dist[b])) b = i;
            used[b] = 1;
            pick[npick] = b;
        }
        free(dist);
        free(used);
    } else
        for (; npick < ncorpse && npick < MAX_CORPSES; npick++) pick[npick] = npick;
    const double *geo = (const double *)tabs[T_GEO];
    const int64_t *tk = (const int64_t *)tabs[T_TYPEK], *tv = (const int64_t *)tabs[T_TYPEV];
    int ntype = (int)tabs[T_NTYPE];
    int m = 0, total = npick + nground + narea;
    for (int j = 0; j < total && m < room; j++, m++) {
        int64_t *s = sel + (size_t)(n + m) * ncols;
        float *o = f + (size_t)(n + m) * F;
        memset(s, 0, sizeof *s * (size_t)ncols);
        memset(o, 0, sizeof *o * (size_t)F);
        int kind;
        int64_t id, typ = 0, owner = -1, x, y;
        if (j < npick) {
            const int64_t *c = corpses + 5 * pick[j];
            kind = 0, id = -(3000 + j), typ = c[0], owner = c[1], x = c[2], y = c[3];
            double age = (double)(t - c[4]) * step_s / 90.0;
            o[A + K_AGE] = (float)(2.0 < age ? 2.0 : age);
        } else if (j < npick + nground) {
            const int64_t *g = ground + 4 * (j - npick);
            kind = 1, id = -(1000 + (j - npick)), x = g[1], y = g[2];
            memcpy(o + A + K_ITEM, code_row(iid, niid, icode, ITEM_D, g[0]), ITEM_D * sizeof(float));
            o[A + K_AGE] = (float)(double)g[3];
        } else {
            int q = j - npick - nground;
            const int64_t *r = areas + 4 * q;
            kind = 2, id = -(2000 + q), x = r[1], y = r[2], owner = r[3];
            memcpy(o + A + K_BUFF, code_row(bid, nbid, bcode, BUFF_D, r[0]), BUFF_D * sizeof(float));
            double left = (area_d[2 * q + 1] - (double)t) * step_s / 30.0;
            o[A + K_AGE] = (float)(2.0 < left ? 2.0 : left);
            o[A + K_RADIUS] = (float)(area_d[2 * q] / 1000.0);
        }
        s[C_ID] = id, s[C_TYPE] = typ, s[C_OWNER] = owner, s[C_X] = x, s[C_Y] = y;
        o[owner == p ? 0 : owner == 1 - p ? 1 : 2] = 1.0f;
        o[A + K_KIND + kind] = 1.0f;
        double fx, fy;
        frame((double)x, (double)y, sign, geo, &fx, &fy);
        o[3] = (float)fx, o[4] = (float)fy;
        int64_t v;
        types[n + m] = in_sorted(tk, ntype, typ, &v, tv) ? v : 0;
        cur[n + m] = 0;
        if (pos_on)
            for (int q = 0; q < N_POS; q++) {  /* (numpy's float32 products: the Python number made float32 first) */
                float c = (float)(M_PI * POS_FREQS[q]);
                pos[10 * m + 2 * q] = c * o[3];
                pos[10 * m + 2 * q + 1] = c * o[4];
            }
    }
    return m;
}

/* obsfull.State.events' corpses: out[i] = 1 when a living row (not DEAD) has the id ids[i], else 0. Its arguments in
 * one int64 array: rows, nrows, ncols, ids, nids, out (int64). */
void w3f_ids_alive_a(const int64_t *a) {
    const int64_t *rows = (const int64_t *)a[0], *ids = (const int64_t *)a[3];
    int nrows = (int)a[1], ncols = (int)a[2], nids = (int)a[4];
    int64_t *out = (int64_t *)a[5];
    for (int i = 0; i < nids; i++) out[i] = 0;
    for (int r = 0; r < nrows; r++) {
        const int64_t *u = rows + (size_t)r * ncols;
        if (u[C_FLAGS] & DEAD) continue;
        for (int i = 0; i < nids; i++)
            if (u[C_ID] == ids[i]) out[i] = 1;
    }
}
