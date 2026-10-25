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
    int64_t counts[5];
    int n = w3f_entities(rows, nrows, ncols, player, sign, geo, (const int64_t *)tabs[T_PASSIVE], (int)tabs[T_NPASSIVE],
                         (const int64_t *)tabs[T_SHOPS], (int)tabs[T_NSHOPS], (int)tabs[T_NEUTRAL], (int)tabs[T_MAXOWN],
                         E, F, (int)tabs[T_APOS], keys, keys + nq, nq, keys + 2 * nq, nb, keys + 2 * nq + nb, nl,
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

/* View.step's masks of one player's rows (featnative.side), into `out` (int64, at least 7 + 8 * nrows): 7 counts,
 * then the segments they count: its living buildings' ids, its living units' ids, its dead heroes (type, id, level:
 * 3 each), its heroes' types (the dead too), its finished buildings' types, (the 6th count: its army's size, no
 * segment), the types of its living units that are no building, worker, hero or summon. Returns the list's length. */
int w3f_side(const int64_t *rows, int nrows, int ncols, int player, int64_t *out) {
    int64_t *cnt = out, *seg = out + 7;
    int at = 0;
    for (int q = 0; q < 7; q++) cnt[q] = 0;
    for (int pass = 0; pass <= 6; pass++) {
        if (pass == 5) continue;
        for (int i = 0; i < nrows; i++) {
            const int64_t *r = rows + (size_t)i * ncols;
            if (r[C_OWNER] != player) continue;
            int64_t fl = r[C_FLAGS];
            int alive = (fl & DEAD) == 0;
            switch (pass) {
            case 0:
                if (alive && (fl & STRUCTURE)) { seg[at++] = r[C_ID]; cnt[0]++; }
                if (alive && !(fl & (STRUCTURE | WORKER))) cnt[5]++;
                break;
            case 1: if (alive) { seg[at++] = r[C_ID]; cnt[1]++; } break;
            case 2:
                if ((fl & HERO) && !alive) { seg[at++] = r[C_TYPE]; seg[at++] = r[C_ID]; seg[at++] = r[C_HLEVEL]; cnt[2]++; }
                break;
            case 3: if (fl & HERO) { seg[at++] = r[C_TYPE]; cnt[3]++; } break;
            case 4: if (alive && (fl & STRUCTURE) && !(fl & CONSTRUCTING)) { seg[at++] = r[C_TYPE]; cnt[4]++; } break;
            case 6: if (!(fl & (DEAD | STRUCTURE | WORKER | HERO | SUMMONED))) { seg[at++] = r[C_TYPE]; cnt[6]++; } break;
            }
        }
    }
    return 7 + at;
}

/* features.affordable: [n] 0/1 the order classes the player can pay for (costs [n][3]: gold, lumber, food); `nkept`
 * classes k1, k2 (their gold and lumber summed: rg, rl) kept from the others, each as it was before. */
void w3f_afford(const int64_t *c, int n, int64_t gold, int64_t lumber, int64_t fu, int64_t fc, int nkept, int64_t k1,
                int64_t k2, int64_t rg, int64_t rl, uint8_t *avail) {
    if (fc > 100) fc = 100;
    int64_t g2 = gold - rg > 0 ? gold - rg : 0, l2 = lumber - rl > 0 ? lumber - rl : 0;
    uint8_t v1 = 0, v2 = 0;
    for (int i = 0; i < n; i++) {
        const int64_t *r = c + 3 * i;
        uint8_t a = r[0] <= gold && r[1] <= lumber && (r[2] == 0 || fu + r[2] <= fc);
        if (nkept >= 1 && i == k1) v1 = a;
        if (nkept >= 2 && i == k2) v2 = a;
        if (nkept && a) a = (r[0] <= g2 && r[1] <= l2) || r[0] + r[1] == 0;
        avail[i] = a;
    }
    if (nkept >= 1) avail[k1] = v1;
    if (nkept >= 2) avail[k2] = v2;
    if (n) avail[0] = 1;
}
