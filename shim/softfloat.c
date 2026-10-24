/* W3SIM_CUTS, round 4: the game's soft float (its own integer float routines, 0x99f000..0x9a3000, called by
 * pointer: out = op(&a, &b)) where a routine's result is provably the IEEE operation's, with the SSE instruction
 * on that domain and the original everywhere else (sfasm.h has the code and the domains):
 *
 *  67108864  multiply 0x99ffe0: the IEEE single product rounded toward zero when both exponent fields are nonzero
 *            and the result's exponent is normal (the original truncates the exact 48-bit product); it differs from
 *            IEEE for zero and denormal operands (a denormal is taken as 1.m * 2^-127; any zero gives +0, never
 *            -0), on underflow (+0, the sign dropped, no denormals), on overflow (the exponent wraps into the sign:
 *            no infinity) and for infinities and NaNs (taken as finite numbers of exponent 128)
 * 134217728  float to int, truncating, 0x9a07b0 (R2I and the engine's own conversions): cvttss2si for |x| < 2^31;
 *            from 2^31 on the original shifts by (exponent - 150) mod 32 (no 0x80000000 "indefinite")
 * 268435456  floor 0x9a1520: roundss for 1 <= |x| < 2^23; below 1 the original returns two constants of its own
 *            (0x112d804 for negative x, 0x112d88c otherwise: -0.0 gives the latter), from 2^23 on x itself
 * 536870912  int to float 0x9a1620: cvtsi2ss for |v| <= 2^24; above, the original truncates (IEEE rounds to
 *            nearest)
 *
 * The add (0x9a01c0) and subtract (0x9a00b0, the add with b's sign flipped) are not IEEE: the smaller operand is
 * aligned with one guard bit by an arithmetic shift (rounding toward minus infinity) and the sum's magnitude then
 * truncated, an exponent difference of 23 or more returns the larger operand unchanged, an exponent field of 0
 * returns the other operand, an exact cancellation gives +0, and there is no overflow handling; nor are the
 * reciprocal (0x9a1d00, a table and interpolation), the division (0x9a02d0: that reciprocal times a multiply), the
 * square root (0x9a20a0, an integer square root of the mantissa) or the table functions (0x9a05f0, 0x9a1da0):
 * those stay as they are.
 *
 * Each replacement is checked in the process against the executable's routine before it is hooked (edge cases and a
 * sweep), and natively over all 2^32 inputs (f2i, floor, i2f) and 2^32 random and boundary pairs (mul):
 * ~/tools/wt-cuts2-out/sf/nat2.c. The callers keep nothing in XMM registers across these calls (the 32-bit ABI
 * makes them all caller-saved; a scan of the executable's ~5,000 call sites finds none read before written).
 */
#include "w3shim.h"

#include "MinHook.h"

#include "sfasm.h"

#define VA_(x) (g_base + ((x) - 0x400000))

__asm__(SF_ASM("_"));
void *sf_mul_orig, *sf_f2i_orig, *sf_floor_orig, *sf_i2f_orig;

static unsigned g_rs = 2463534242u;
static unsigned rnd(void) {
    g_rs ^= g_rs << 13;
    g_rs ^= g_rs >> 17;
    g_rs ^= g_rs << 5;
    return g_rs;
}

/* operands: edge values, then random bit patterns and patterns with exponents near the fast paths' bounds */
static unsigned operand(unsigned k) {
    static const unsigned edge[] = {0, 0x80000000u, 1, 0x80000001u, 0x007fffffu, 0x00800000u, 0x80800000u, 0x3f800000u,
                                    0xbf800000u, 0x3f7fffffu, 0x3fffffffu, 0x4b000000u, 0x4affffffu, 0xcb000000u,
                                    0xcaffffffu, 0x4effffffu, 0x4f000000u, 0xcf000000u, 0xceffffffu, 0x7f7fffffu,
                                    0x7f800000u, 0xff800000u, 0x7fc00000u, 0x7f800001u, 0x3effffffu, 0xbf000001u};
    if (k < sizeof edge / sizeof edge[0])
        return edge[k];
    unsigned r = rnd();
    if (k & 1)
        return r;
    unsigned e = (k & 2) ? 100 + rnd() % 60 : rnd() % 256;
    return (r & 0x807fffffu) | (e << 23);
}

static int selftest(int which) {
    for (unsigned k = 0; k < (1u << 18); k++) {
        unsigned a = operand(k), b = operand(k * 7 + 3), o = 0x5a5a5a5au, f = 0xa5a5a5a5u;
        switch (which) {
        case 0:
            if ((k & 3) == 0) /* products near the result's exponent bounds */
                b = (b & 0x807fffffu) | (((unsigned)(((k >> 2) & 1 ? 380 : 128) + (int)(rnd() % 7) - 3 -
                                                     (int)((a >> 23) & 0xff)) & 0xff) << 23);
            ((SfBinFn)(void *)VA_(0x99ffe0))(&o, &a, &b);
            if (sf_mul_asm(&f, &a, &b) != &f || f != o)
                return 0;
            break;
        case 1:
            if (((SfF2iFn)(void *)VA_(0x9a07b0))(&a) != sf_f2i_asm(&a))
                return 0;
            break;
        case 2:
            ((SfUnFn)(void *)VA_(0x9a1520))(&o, &a);
            if (sf_floor_asm(&f, &a) != &f || f != o)
                return 0;
            break;
        case 3: {
            int v = (k & 2) ? (int)(a >> (rnd() & 31)) * ((k & 4) ? -1 : 1) : (int)a;
            ((SfI2fFn)(void *)VA_(0x9a1620))(&o, v);
            if (sf_i2f_asm(&f, v) != &f || f != o)
                return 0;
        } break;
        }
    }
    return 1;
}

static int has_sse41(void) {
    unsigned a, b, c, d;
    __asm__ volatile("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(1), "c"(0));
    return (c >> 19) & 1;
}

void softfloat_install(long long cuts) {
    static const struct {
        long long bit;
        DWORD va;
        void *fn, **orig;
        const char *name;
        BYTE code[8];
        int n;
    } r[4] = {
        {67108864, 0x99ffe0, (void *)sf_mul_asm, &sf_mul_orig, "multiply", {0x55, 0x8b, 0xec, 0x8b, 0x45, 0x0c, 0x53}, 7},
        {134217728, 0x9a07b0, (void *)sf_f2i_asm, &sf_f2i_orig, "float to int", {0x55, 0x8b, 0xec, 0x8b, 0x45, 0x08, 0x56}, 7},
        {268435456, 0x9a1520, (void *)sf_floor_asm, &sf_floor_orig, "floor", {0x55, 0x8b, 0xec, 0x8b, 0x45, 0x0c, 0x56}, 7},
        {536870912, 0x9a1620, (void *)sf_i2f_asm, &sf_i2f_orig, "int to float", {0x55, 0x8b, 0xec, 0x8b, 0x45, 0x0c, 0x85}, 7},
    };
    if (!(cuts & (67108864 | 134217728 | 268435456 | 536870912)))
        return;
    if (MH_Initialize() != MH_OK && MH_Initialize() != MH_ERROR_ALREADY_INITIALIZED)
        return;
    for (int i = 0; i < 4; i++) {
        if (!(cuts & r[i].bit))
            continue;
        BYTE *p = VA_(r[i].va);
        if (memcmp(p, r[i].code, r[i].n) != 0) {
            shim_log("soft float: %s: unexpected code, not hooked", r[i].name);
            continue;
        }
        if (i == 2 && !has_sse41()) {
            shim_log("soft float: floor: no SSE4.1, not hooked");
            continue;
        }
        *r[i].orig = p; /* (the self-test's fallback: the unhooked original) */
        if (!selftest(i)) {
            shim_log("soft float: %s: MISMATCH in the self-test, not hooked", r[i].name);
            continue;
        }
        void *tramp = NULL;
        if (MH_CreateHook(p, r[i].fn, &tramp) != MH_OK) {
            shim_log("soft float: %s: hook failed", r[i].name);
            continue;
        }
        *r[i].orig = tramp;
        if (MH_EnableHook(p) != MH_OK) {
            shim_log("soft float: %s: hook failed", r[i].name);
            continue;
        }
        shim_log("soft float: %s with SSE on its IEEE domain (W3SIM_CUTS & %lld)", r[i].name, r[i].bit);
    }
}
