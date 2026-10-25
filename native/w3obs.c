/* Observation parsing in C (warcraftsim.native): the harness's token stream -> arrays, merged into
 * a unit table, without the GIL (ctypes releases it for the call).
 *
 * The same rules as protocol.parse_tokens and merge_observation:
 *   - lines are tokens; a line that is neither an integer nor one capital letter is dropped
 *     (the game's own preloads interleave);
 *   - every record but V and X ends with a checksum (int32 sum of its fields); a record with one
 *     stray token is repaired by dropping it, otherwise the record is skipped (damaged);
 *   - U records are deltas: an observation with full=1 clears the table first; R removes a unit;
 *     units keep the order they were first seen in (a Python dict's order).
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define UF 14            /* a unit's fields */
#define HERO_EXTRA 17    /* heroes: level, xp, skill points, 6 items, 4 x (ability level, cooldown) */
#define UCOLS (UF + HERO_EXTRA + 1) /* + whether the hero fields are there */
#define HERO_FLAG 1

enum { F_T = 4, F_P = 14, F_O = 1, F_D = 5, F_E = 4, F_C = 1, F_U = UF, F_R = 1, F_K = 2, F_I = 6, F_G = 5 };

typedef struct { int32_t *v; int n, cap, width; } Rows;

/* a new row at the end (its fields left to the caller), NULL when out of memory */
static int32_t *rows_add(Rows *r) {
    if (r->n == r->cap) {
        int cap = r->cap ? 2 * r->cap : 64;
        int32_t *v = realloc(r->v, (size_t)cap * r->width * sizeof(int32_t));
        if (!v) return NULL;
        r->v = v;
        r->cap = cap;
    }
    return r->v + (size_t)(r->n++) * r->width;
}

static int rows_push(Rows *r, const int32_t *vals, int k) {
    int32_t *dst = rows_add(r);
    if (!dst) return -1;
    if (k > r->width) k = r->width;
    memcpy(dst, vals, (size_t)k * sizeof(int32_t));
    if (k < r->width) memset(dst + k, 0, (size_t)(r->width - k) * sizeof(int32_t));
    return 0;
}

/* the unit table: rows in first-seen order (dead slots skipped), and id -> slot */
typedef struct {
    int32_t *rows;  /* [cap][UCOLS] */
    uint8_t *live;
    int n, cap, dead;
    int32_t *keys;  /* open addressing: id -> slot + 1 (0: empty) */
    int32_t *vals;
    int hcap;
} Table;

typedef struct {
    Table t;
    Rows players, events, results, issued, orders, dests, removed, units, ground; /* this observation */
    int32_t header[16];
    char *line;  /* scratch */
    int64_t *toks;
    uint8_t *is_tag;
    int tcap;
} State;

enum { H_SEQ, H_MS, H_OVER, H_FULL, H_VERSION, H_DAMAGED, H_ENDED, H_CAMERA, H_CAM_X, H_CAM_Y, H_NUNITS };

static uint32_t hash32(uint32_t x) { x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16; return x; }

static int table_find(Table *t, int32_t id) {
    uint32_t m = t->hcap - 1, i = hash32((uint32_t)id) & m;
    while (t->vals[i]) {
        if (t->keys[i] == id) return (int)i;
        i = (i + 1) & m;
    }
    return -(int)i - 1;  /* where it would go */
}

static void table_del_at(Table *t, int i) { /* backward-shift deletion */
    uint32_t m = t->hcap - 1, j = (uint32_t)i;
    t->vals[j] = 0;
    for (;;) {
        j = (j + 1) & m;
        if (!t->vals[j]) return;
        uint32_t home = hash32((uint32_t)t->keys[j]) & m;
        /* can the entry at j move to the hole at i? (its home is not between i+1 and j, cyclically) */
        if ((j > (uint32_t)i && (home <= (uint32_t)i || home > j)) || (j < (uint32_t)i && (home <= (uint32_t)i && home > j))) {
            t->keys[i] = t->keys[j];
            t->vals[i] = t->vals[j];
            t->vals[j] = 0;
            i = (int)j;
        }
    }
}

static int table_rehash(Table *t, int hcap) {
    int32_t *ok = t->keys, *ov = t->vals;
    int oc = t->hcap;
    t->keys = calloc(hcap, sizeof(int32_t));
    t->vals = calloc(hcap, sizeof(int32_t));
    if (!t->keys || !t->vals) return -1;
    t->hcap = hcap;
    for (int i = 0; i < oc; i++)
        if (ov[i]) {
            int p = -table_find(t, ok[i]) - 1;
            t->keys[p] = ok[i];
            t->vals[p] = ov[i];
        }
    free(ok);
    free(ov);
    return 0;
}

static void table_clear(Table *t) {
    t->n = t->dead = 0;
    memset(t->vals, 0, t->hcap * sizeof(int32_t));
}

static int table_compact(Table *t) {  /* drop dead slots, keeping the order */
    int k = 0;
    for (int s = 0; s < t->n; s++)
        if (t->live[s]) {
            if (k != s) {
                memcpy(t->rows + (size_t)k * UCOLS, t->rows + (size_t)s * UCOLS, UCOLS * sizeof(int32_t));
                t->live[k] = 1;
            }
            k++;
        }
    t->n = k;
    t->dead = 0;
    memset(t->vals, 0, t->hcap * sizeof(int32_t));
    for (int s = 0; s < t->n; s++) {
        int p = -table_find(t, t->rows[(size_t)s * UCOLS]) - 1;
        t->keys[p] = t->rows[(size_t)s * UCOLS];
        t->vals[p] = s + 1;
    }
    return 0;
}

static int table_put(Table *t, const int32_t *row) {
    int f = table_find(t, row[0]);
    if (f >= 0) {  /* an update keeps its place */
        memcpy(t->rows + (size_t)(t->vals[f] - 1) * UCOLS, row, UCOLS * sizeof(int32_t));
        return 0;
    }
    if ((t->n + 1) * 2 > t->hcap && table_rehash(t, t->hcap * 2)) return -1;
    if (t->n == t->cap) {
        int cap = t->cap * 2;
        int32_t *r = realloc(t->rows, (size_t)cap * UCOLS * sizeof(int32_t));
        uint8_t *l = realloc(t->live, cap);
        if (!r || !l) return -1;
        t->rows = r;
        t->live = l;
        t->cap = cap;
    }
    memcpy(t->rows + (size_t)t->n * UCOLS, row, UCOLS * sizeof(int32_t));
    t->live[t->n] = 1;
    int p = -table_find(t, row[0]) - 1;
    t->keys[p] = row[0];
    t->vals[p] = ++t->n;
    return 0;
}

static void table_remove(Table *t, int32_t id) {
    int f = table_find(t, id);
    if (f < 0) return;
    t->live[t->vals[f] - 1] = 0;
    t->dead++;
    table_del_at(t, f);
}

void *w3obs_new(void) {
    State *s = calloc(1, sizeof(State));
    if (!s) return NULL;
    s->t.cap = 1024;
    s->t.rows = malloc((size_t)s->t.cap * UCOLS * sizeof(int32_t));
    s->t.live = malloc(s->t.cap);
    s->t.hcap = 4096;
    s->t.keys = calloc(s->t.hcap, sizeof(int32_t));
    s->t.vals = calloc(s->t.hcap, sizeof(int32_t));
    Rows *rs[] = {&s->players, &s->events, &s->results, &s->issued, &s->orders, &s->dests, &s->removed, &s->units, &s->ground};
    int w[] = {F_P, F_E, F_C, F_I, F_O, F_D, F_R, UCOLS, F_G};
    for (int i = 0; i < 9; i++) rs[i]->width = w[i];
    return s;
}

void w3obs_free(void *p) {
    State *s = p;
    if (!s) return;
    Rows *rs[] = {&s->players, &s->events, &s->results, &s->issued, &s->orders, &s->dests, &s->removed, &s->units, &s->ground};
    for (int i = 0; i < 9; i++) free(rs[i]->v);
    free(s->t.rows); free(s->t.live); free(s->t.keys); free(s->t.vals); free(s->toks); free(s->is_tag);
    free(s);
}

static int base_fields(char tag) {
    switch (tag) {
    case 'T': return F_T; case 'P': return F_P; case 'O': return F_O; case 'D': return F_D; case 'E': return F_E;
    case 'C': return F_C; case 'U': return F_U; case 'R': return F_R; case 'K': return F_K; case 'I': return F_I; case 'G': return F_G;
    default: return 0;
    }
}

static int record_len(char tag, const int64_t *f, int n) {
    if (tag == 'U' && n > 11 && (f[11] & HERO_FLAG)) return F_U + HERO_EXTRA;
    return base_fields(tag);
}

static int32_t i32(int64_t v) { return (int32_t)(uint32_t)(uint64_t)v; }

/* _take_record: the record's fields into out (up to 64), their count; -1: damaged. *next: after it */
static int take_record(State *s, char tag, int i, int ntok, int32_t *out, int *next) {
    int base = base_fields(tag);
    int limit = base + (tag == 'U' ? HERO_EXTRA : 0) + 2;
    const int64_t *vals = s->toks + i;  /* (the record's tokens: up to the next tag or the limit) */
    int nv = 0, most = ntok - i < limit ? ntok - i : limit;
    while (nv < most && !s->is_tag[i + nv]) nv++;
    if (nv > base) {
        int n = record_len(tag, vals, nv);
        if (nv > n) {
            int64_t sum = 0;
            for (int k = 0; k < n; k++) sum += vals[k];
            if (i32(sum) == (int32_t)vals[n]) {
                for (int k = 0; k < n; k++) out[k] = (int32_t)vals[k];
                *next = i + n + 1;
                return n;
            }
        }
    }
    int64_t cand[64];
    for (int drop = 0; drop < nv; drop++) {
        int nc = 0;
        for (int k = 0; k < nv; k++)
            if (k != drop) cand[nc++] = vals[k];
        if (nc <= base) continue;
        int n = record_len(tag, cand, nc);
        if (nc > n && drop <= n) {
            int64_t sum = 0;
            for (int k = 0; k < n; k++) sum += cand[k];
            if (i32(sum) == (int32_t)cand[n]) {
                for (int k = 0; k < n; k++) out[k] = (int32_t)cand[k];
                *next = i + n + 2;
                return n;
            }
        }
    }
    *next = i;
    return -1;
}

/* Tokenize: keep lines matching -?\d+ (at most 18 digits) or [A-Z]. One pass, each line read once (a scan for its
 * end and then the line again had been ~70k instructions an observation of ~2 KB). */
static int tokenize(State *s, const char *d, int len) {
    int most = len / 2 + 2;  /* (a kept line is at least one character and its newline) */
    if (most > s->tcap) {
        int cap = s->tcap ? s->tcap : 4096;
        while (cap < most) cap *= 2;
        int64_t *t = realloc(s->toks, (size_t)cap * sizeof(int64_t));
        if (!t) return -1;
        s->toks = t;
        uint8_t *g = realloc(s->is_tag, cap);
        if (!g) return -1;
        s->is_tag = g;
        s->tcap = cap;
    }
    int64_t *toks = s->toks;
    uint8_t *is_tag = s->is_tag;
    const char *p = d, *end = d + len;
    int n = 0;
    while (p < end) {
        char c = *p;
        if (c == '\n') { p++; continue; }  /* (an empty line) */
        if (c >= 'A' && c <= 'Z') {
            if (p + 1 == end || p[1] == '\n') {
                toks[n] = c;
                is_tag[n++] = 1;
                p += 2;
                continue;
            }
        } else {
            const char *q = p + (c == '-');
            const char *digits = q;
            uint64_t v = 0;
            while (q < end && (unsigned char)(*q - '0') <= 9) v = v * 10 + (uint64_t)(*q++ - '0');
            if (q > digits && q - digits <= 18 && (q == end || *q == '\n')) {
                toks[n] = c == '-' ? -(int64_t)v : (int64_t)v;
                is_tag[n++] = 0;
                p = q + 1;
                continue;
            }
            p = q;
        }
        while (p < end && *p != '\n') p++;  /* (a line dropped) */
        p++;
    }
    return n;
}

/* Parse one observation and merge it into the unit table. Returns 0, or -1 (no end marker),
 * -2 (out of memory). The results: w3obs_header / w3obs_rows. */
int w3obs_parse(void *p, const char *data, int len) {
    State *s = p;
    Rows *rs[] = {&s->players, &s->events, &s->results, &s->issued, &s->orders, &s->dests, &s->removed, &s->units, &s->ground};
    for (int i = 0; i < 9; i++) rs[i]->n = 0;
    memset(s->header, 0, sizeof(s->header));
    s->header[H_VERSION] = -1;
    s->header[H_FULL] = 1;
    int ntok = tokenize(s, data, len);
    if (ntok < 0) return -2;
    int i = 0, ended = 0, damaged = 0;
    int32_t f[64];
    while (i < ntok) {
        if (!s->is_tag[i]) { i++; continue; }  /* (a number where a tag should be: parse_tokens skips it too) */
        char tag = (char)s->toks[i++];
        if (tag == 'X') { ended = 1; break; }
        if (tag == 'V') {
            s->header[H_VERSION] = (i < ntok && !s->is_tag[i]) ? (int32_t)s->toks[i] : -1;
            i++;
            continue;
        }
        if (!base_fields(tag)) continue;
        int next;
        int n = take_record(s, tag, i, ntok, f, &next);
        if (n < 0) {
            damaged++;
            while (i < ntok && !s->is_tag[i]) i++;
            continue;
        }
        i = next;
        int err = 0;
        switch (tag) {
        case 'U': {
            int32_t *row = rows_add(&s->units);
            if (!row) { err = 1; break; }
            memcpy(row, f, n * sizeof(int32_t));
            memset(row + n, 0, (UCOLS - 1 - n) * sizeof(int32_t));
            row[UCOLS - 1] = n > F_U;
            break;
        }
        case 'D': err = rows_push(&s->dests, f, n); break;
        case 'E': err = rows_push(&s->events, f, n); break;
        case 'C': err = rows_push(&s->results, f, n); break;
        case 'I': err = rows_push(&s->issued, f, n); break;
        case 'R': err = rows_push(&s->removed, f, n); break;
        case 'G': err = rows_push(&s->ground, f, n); break;
        case 'P': err = rows_push(&s->players, f, n); break;
        case 'O': err = rows_push(&s->orders, f, n); break;
        case 'T': s->header[H_SEQ] = f[0]; s->header[H_MS] = f[1]; s->header[H_OVER] = f[2] != 0;
                  s->header[H_FULL] = f[3] != 0; break;
        case 'K': s->header[H_CAMERA] = 1; s->header[H_CAM_X] = f[0]; s->header[H_CAM_Y] = f[1]; break;
        }
        if (err) return -2;
    }
    s->header[H_DAMAGED] = damaged;
    s->header[H_ENDED] = ended;
    if (!ended) return -1;
    return 0;
}

/* Merge the parsed observation into the table (after checking its version and sequence in Python). */
int w3obs_merge(void *p) {
    State *s = p;
    Table *t = &s->t;
    if (s->header[H_FULL]) table_clear(t);
    for (int k = 0; k < s->units.n; k++)
        if (table_put(t, s->units.v + (size_t)k * UCOLS)) return -2;
    for (int k = 0; k < s->removed.n; k++) table_remove(t, s->removed.v[k]);
    /* a cancelled frame (event 4, a = the structure): the game removes it outright and the units pass never sees it
       again, so no removed record came; it stayed in the table for good, a ghost every repair at it refused (~900 a
       human game; audit 9). A stopgap until the passes report units gone without dying (a wisp an ancient took). */
    for (int k = 0; k < s->events.n; k++)
        if (s->events.v[(size_t)k * F_E] == 4) table_remove(t, s->events.v[(size_t)k * F_E + 1]);
    if (t->dead > 64 && t->dead * 2 > t->n) table_compact(t);
    int live = t->n - t->dead;
    s->header[H_NUNITS] = live;
    return live;
}

/* The unit table (merged) into out [n][UCOLS], in first-seen order. */
int w3obs_table(void *p, int32_t *out, int cap) {
    State *s = p;
    Table *t = &s->t;
    int k = 0;
    for (int slot = 0; slot < t->n && k < cap; slot++)
        if (t->live[slot]) memcpy(out + (size_t)(k++) * UCOLS, t->rows + (size_t)slot * UCOLS, UCOLS * sizeof(int32_t));
    return k;
}

const int32_t *w3obs_header(void *p) { return ((State *)p)->header; }

/* which: 0 players, 1 events, 2 results, 3 issued, 4 orders, 5 dests, 6 removed, 7 unit deltas, 8 ground items */
const int32_t *w3obs_rows(void *p, int which, int *n, int *width) {
    State *s = p;
    Rows *rs[] = {&s->players, &s->events, &s->results, &s->issued, &s->orders, &s->dests, &s->removed, &s->units, &s->ground};
    if (which < 0 || which > 8) { *n = 0; *width = 0; return NULL; }
    *n = rs[which]->n;
    *width = rs[which]->width;
    return rs[which]->v;
}

int w3obs_ucols(void) { return UCOLS; }

/* table `which`'s row width (w3obs_rows' order) */
int w3obs_width(void *p, int which) {
    int n, w;
    w3obs_rows(p, which, &n, &w);
    return w;
}

/* The parsed observation for Python in one copy (eight calls with an array each had been ~100k instructions a step):
 * out = [the header (16)][each table's rows (9, w3obs_rows' order)][the tables but the unit deltas (7) in that order,
 * their rows of their widths]. Returns the ints that takes; out holds them all only when cap is enough (the header and
 * the row counts when cap is at least 25). */
int w3obs_export(void *p, int32_t *out, int cap) {
    State *s = p;
    Rows *rs[] = {&s->players, &s->events, &s->results, &s->issued, &s->orders, &s->dests, &s->removed, &s->units, &s->ground};
    int total = 16 + 9;
    for (int i = 0; i < 9; i++)
        if (i != 7) total += rs[i]->n * rs[i]->width;
    if (cap < 25) return total;
    memcpy(out, s->header, 16 * sizeof(int32_t));
    for (int i = 0; i < 9; i++) out[16 + i] = rs[i]->n;
    if (total > cap) return total;
    int at = 25;
    for (int i = 0; i < 9; i++)
        if (i != 7 && rs[i]->n) {
            memcpy(out + at, rs[i]->v, (size_t)rs[i]->n * rs[i]->width * sizeof(int32_t));
            at += rs[i]->n * rs[i]->width;
        }
    return total;
}

/* w3obs_parse, then w3obs_export: its size, or -2 (out of memory). (No end marker: the header's H_ENDED is 0.) */
int w3obs_parse_export(void *p, const char *data, int len, int32_t *out, int cap) {
    if (w3obs_parse(p, data, len) == -2) return -2;
    return w3obs_export(p, out, cap);
}

/* ---- fullgame.trace and play.unit_rows on the unit table (rows [n][UCOLS], as w3obs_table gives them) ---- */

#define DEAD_FLAG 1024

/* numpy's sum of a contiguous float64 array (np.add.reduce: 0.0 plus the pairwise sum of numpy/_core/src/umath/
 * loops_utils.h.src, blocks of up to 128 summed with 8 accumulators): the same additions in the same order */
static double pairwise_sum(const double *a, long n) {
    if (n < 8) {
        double res = -0.0;
        for (long i = 0; i < n; i++) res += a[i];
        return res;
    }
    if (n <= 128) {
        double r[8];
        for (int k = 0; k < 8; k++) r[k] = a[k];
        long i;
        for (i = 8; i < n - (n % 8); i += 8)
            for (int k = 0; k < 8; k++) r[k] += a[i + k];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) res += a[i];
        return res;
    }
    long n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

double w3obs_sum(const double *a, int n) { return 0.0 + pairwise_sum(a, n); }

/* trace.material on the unit table: io[p] = player p's (0, 1) living units' value times their hit points left, as its
 * numpy code computes it (each unit's operations in its order, the sum numpy's in the table's order). desc: the
 * addresses of keys (sorted type ids, int64) and vals (their values, double), nk, the addresses of the worker types
 * (int64) and their count; io in: hero_level (per level above the first; 0: none), worker_cap (0: none). (A few
 * arguments: each one of a ctypes call costs ~1k instructions.) Returns 0, 1 when a type is not in keys (the caller
 * makes its table again), -1 when out of memory. */
int w3obs_material(const int32_t *rows, int n, const int64_t *desc, double *io) {
    const int64_t *keys = (const int64_t *)(intptr_t)desc[0], *workers = (const int64_t *)(intptr_t)desc[3];
    const double *vals = (const double *)(intptr_t)desc[1];
    int nk = (int)desc[2], nw = (int)desc[4];
    double hero_level = io[0], worker_cap = io[1], *out = io;
    double *v = malloc((size_t)(n > 0 ? n : 1) * 3 * sizeof(double));
    uint8_t *who = malloc((size_t)(n > 0 ? n : 1) * 2);
    if (!v || !who) { free(v); free(who); return -1; }
    double *hp = v + n, *vh = v + 2 * n;
    uint8_t *work = who + n;
    int m = 0, nwork[2] = {0, 0};
    for (int i = 0; i < n; i++) {
        const int32_t *r = rows + (size_t)i * UCOLS;
        int32_t o = r[2];
        if ((r[11] & DEAD_FLAG) || (o != 0 && o != 1)) continue;
        int64_t t = r[1];
        int lo = 0, hi = nk;  /* (np.searchsorted: the first key not below t) */
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (keys[mid] < t) lo = mid + 1; else hi = mid;
        }
        if (lo == nk || keys[lo] != t) { free(v); free(who); return 1; }
        double x = vals[lo];
        if (hero_level != 0.0) {  /* (column 14: the hero's level; int32 arithmetic, as numpy's) */
            int32_t lv = (int32_t)((uint32_t)r[14] - 1u);
            x = x + hero_level * (double)(lv > 0 ? lv : 0);
        }
        v[m] = x;
        hp[m] = r[7] > 0 ? (double)r[6] / (double)r[7] : 1.0;
        who[m] = (uint8_t)o;
        work[m] = 0;
        for (int k = 0; k < nw; k++)
            if (workers[k] == t) { work[m] = 1; nwork[o]++; break; }
        m++;
    }
    if (worker_cap != 0.0)
        for (int p = 0; p < 2; p++)
            if ((double)nwork[p] > worker_cap)
                for (int k = 0; k < m; k++)
                    if (work[k] && who[k] == p) v[k] = v[k] * worker_cap / (double)nwork[p];
    /* (numpy evaluates v * worker_cap / n for all units and keeps it for the player's workers: the same numbers) */
    for (int p = 0; p < 2; p++) {  /* (each player's products, in order) */
        int c = 0;
        for (int k = 0; k < m; k++)
            if (who[k] == p) vh[c++] = v[k] * hp[k];
        out[p] = 0.0 + pairwise_sum(vh, c);
    }
    free(v);
    free(who);
    return 0;
}

/* trace.Production's owners as last seen (unit id -> owner), mirrored here: a scan hands Python only what changed; and
 * the hired types (a set of their own) */
#define HIRED_SLOTS 128
typedef struct {
    int32_t *keys, *vals;
    uint8_t *used;
    int cap, n;
    int64_t hired[HIRED_SLOTS];
    uint8_t hired_used[HIRED_SLOTS];
} Owners;

void w3obs_owners_free(void *p) {
    Owners *o = p;
    if (!o) return;
    free(o->keys);
    free(o->vals);
    free(o->used);
    free(o);
}

static int owners_alloc(Owners *o, int cap) {
    o->keys = malloc((size_t)cap * sizeof(int32_t));
    o->vals = malloc((size_t)cap * sizeof(int32_t));
    o->used = calloc(cap, 1);
    o->cap = cap;
    o->n = 0;
    return o->keys && o->vals && o->used ? 0 : -1;
}

static int hired_slot(const Owners *o, int64_t t) {
    uint32_t i = hash32((uint32_t)t ^ (uint32_t)((uint64_t)t >> 32)) & (HIRED_SLOTS - 1);
    while (o->hired_used[i] && o->hired[i] != t) i = (i + 1) & (HIRED_SLOTS - 1);
    return (int)i;
}

/* the state for one game's Production: hired[nh] (at most HIRED_SLOTS / 2) the hired units' types */
void *w3obs_owners_new(const int64_t *hired, int nh) {
    if (nh > HIRED_SLOTS / 2) return NULL;
    Owners *o = calloc(1, sizeof(Owners));
    if (!o) return NULL;
    if (owners_alloc(o, 1024)) { w3obs_owners_free(o); return NULL; }
    for (int k = 0; k < nh; k++) {
        int s = hired_slot(o, hired[k]);
        o->hired_used[s] = 1;
        o->hired[s] = hired[k];
    }
    return o;
}

void w3obs_owners_clear(void *p) {
    Owners *o = p;
    memset(o->used, 0, o->cap);
    o->n = 0;
}

/* the slot of id: its own, or the empty one where it would go */
static int owners_slot(const Owners *o, int32_t id) {
    uint32_t m = o->cap - 1, i = hash32((uint32_t)id) & m;
    while (o->used[i] && o->keys[i] != id) i = (i + 1) & m;
    return (int)i;
}

static int owners_grow(Owners *o) {
    Owners old = *o;
    if (owners_alloc(o, 2 * old.cap)) { free(o->keys); free(o->vals); free(o->used); *o = old; return -1; }
    for (int i = 0; i < old.cap; i++)
        if (old.used[i]) {
            int s = owners_slot(o, old.keys[i]);
            o->used[s] = 1;
            o->keys[s] = old.keys[i];
            o->vals[s] = old.vals[i];
            o->n++;
        }
    free(old.keys);
    free(old.vals);
    free(old.used);
    return 0;
}

/* Production.step's look at the table: the (id, type, owner) of each unit of a hired type, in the table's order, then
 * (with `owners`) each (id, owner) new or changed since the last such scan (the mirror set to it). out: [the triples'
 * count, the pairs' count, the triples, the pairs] (at most 2 + 5 n ints). Returns 0, -1 out of memory. */
int w3obs_prod_scan(void *p, const int32_t *rows, int n, int owners, int32_t *out) {
    Owners *o = p;
    int h = 0, q = 0;
    int32_t *at = out + 2;
    for (int i = 0; i < n; i++) {
        const int32_t *r = rows + (size_t)i * UCOLS;
        if (o->hired_used[hired_slot(o, r[1])]) {
            at[0] = r[0]; at[1] = r[1]; at[2] = r[2];
            at += 3;
            h++;
        }
    }
    if (owners)
        for (int i = 0; i < n; i++) {
            const int32_t *r = rows + (size_t)i * UCOLS;
            if (2 * (o->n + 1) > o->cap && owners_grow(o)) return -1;
            int s = owners_slot(o, r[0]);
            if (o->used[s] && o->vals[s] == r[2]) continue;
            if (!o->used[s]) { o->used[s] = 1; o->keys[s] = r[0]; o->n++; }
            o->vals[s] = r[2];
            at[0] = r[0]; at[1] = r[2];
            at += 2;
            q++;
        }
    out[0] = h;
    out[1] = q;
    return 0;
}

/* play.unit_rows on the unit table: out [n][cols] (int64), the step t, then the table's first cols - 1 columns */
void w3obs_unit_rows(const int32_t *rows, int n, int64_t t, int64_t *out, int cols) {
    for (int i = 0; i < n; i++) {
        const int32_t *r = rows + (size_t)i * UCOLS;
        int64_t *d = out + (size_t)i * cols;
        d[0] = t;
        for (int k = 1; k < cols; k++) d[k] = r[k - 1];
    }
}
