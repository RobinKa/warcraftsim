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

enum { F_T = 4, F_P = 14, F_O = 1, F_D = 5, F_E = 4, F_C = 1, F_U = UF, F_R = 1, F_K = 2, F_I = 6 };

typedef struct { int32_t *v; int n, cap, width; } Rows;

static int rows_push(Rows *r, const int32_t *vals, int k) {
    if (r->n == r->cap) {
        int cap = r->cap ? 2 * r->cap : 64;
        int32_t *v = realloc(r->v, (size_t)cap * r->width * sizeof(int32_t));
        if (!v) return -1;
        r->v = v;
        r->cap = cap;
    }
    int32_t *dst = r->v + (size_t)r->n * r->width;
    memset(dst, 0, r->width * sizeof(int32_t));
    memcpy(dst, vals, (size_t)(k < r->width ? k : r->width) * sizeof(int32_t));
    r->n++;
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
    Rows players, events, results, issued, orders, dests, removed, units; /* this observation */
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
    Rows *rs[] = {&s->players, &s->events, &s->results, &s->issued, &s->orders, &s->dests, &s->removed, &s->units};
    int w[] = {F_P, F_E, F_C, F_I, F_O, F_D, F_R, UCOLS};
    for (int i = 0; i < 8; i++) rs[i]->width = w[i];
    return s;
}

void w3obs_free(void *p) {
    State *s = p;
    if (!s) return;
    Rows *rs[] = {&s->players, &s->events, &s->results, &s->issued, &s->orders, &s->dests, &s->removed, &s->units};
    for (int i = 0; i < 8; i++) free(rs[i]->v);
    free(s->t.rows); free(s->t.live); free(s->t.keys); free(s->t.vals); free(s->toks); free(s->is_tag);
    free(s);
}

static int base_fields(char tag) {
    switch (tag) {
    case 'T': return F_T; case 'P': return F_P; case 'O': return F_O; case 'D': return F_D; case 'E': return F_E;
    case 'C': return F_C; case 'U': return F_U; case 'R': return F_R; case 'K': return F_K; case 'I': return F_I;
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
    int64_t vals[64] = {0};
    int nv = 0;
    for (int k = i; k < ntok && k < i + limit; k++) {
        if (s->is_tag[k]) break;
        vals[nv++] = s->toks[k];
    }
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

/* Tokenize: keep lines matching -?\d+ or [A-Z]. */
static int tokenize(State *s, const char *d, int len) {
    int n = 0;
    int start = 0;
    for (int pos = 0; pos <= len; pos++) {
        if (pos < len && d[pos] != '\n') continue;
        int a = start, b = pos;  /* the line [a, b) */
        start = pos + 1;
        if (b <= a) continue;
        if (n >= s->tcap) {
            int cap = s->tcap ? 2 * s->tcap : 4096;
            int64_t *t = realloc(s->toks, cap * sizeof(int64_t));
            uint8_t *g = realloc(s->is_tag, cap);
            if (!t || !g) return -1;
            s->toks = t; s->is_tag = g; s->tcap = cap;
        }
        if (b - a == 1 && d[a] >= 'A' && d[a] <= 'Z') {
            s->toks[n] = d[a];
            s->is_tag[n++] = 1;
            continue;
        }
        int k = a, neg = 0;
        if (d[k] == '-') { neg = 1; k++; }
        if (k == b || b - k > 18) continue;
        int64_t v = 0, ok = 1;
        for (; k < b; k++) {
            if (d[k] < '0' || d[k] > '9') { ok = 0; break; }
            v = v * 10 + (d[k] - '0');
        }
        if (!ok) continue;
        s->toks[n] = neg ? -v : v;
        s->is_tag[n++] = 0;
    }
    return n;
}

/* Parse one observation and merge it into the unit table. Returns 0, or -1 (no end marker),
 * -2 (out of memory). The results: w3obs_header / w3obs_rows. */
int w3obs_parse(void *p, const char *data, int len) {
    State *s = p;
    Rows *rs[] = {&s->players, &s->events, &s->results, &s->issued, &s->orders, &s->dests, &s->removed, &s->units};
    for (int i = 0; i < 8; i++) rs[i]->n = 0;
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
            int32_t row[UCOLS] = {0};
            memcpy(row, f, n * sizeof(int32_t));
            row[UCOLS - 1] = n > F_U;
            err = rows_push(&s->units, row, UCOLS);
            break;
        }
        case 'D': err = rows_push(&s->dests, f, n); break;
        case 'E': err = rows_push(&s->events, f, n); break;
        case 'C': err = rows_push(&s->results, f, n); break;
        case 'I': err = rows_push(&s->issued, f, n); break;
        case 'R': err = rows_push(&s->removed, f, n); break;
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

/* which: 0 players, 1 events, 2 results, 3 issued, 4 orders, 5 dests, 6 removed, 7 unit deltas */
const int32_t *w3obs_rows(void *p, int which, int *n, int *width) {
    State *s = p;
    Rows *rs[] = {&s->players, &s->events, &s->results, &s->issued, &s->orders, &s->dests, &s->removed, &s->units};
    if (which < 0 || which > 7) { *n = 0; *width = 0; return NULL; }
    *n = rs[which]->n;
    *width = rs[which]->width;
    return rs[which]->v;
}

int w3obs_ucols(void) { return UCOLS; }
