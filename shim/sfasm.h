/* W3SIM_CUTS 67108864 / 134217728 / 268435456 / 536870912: the soft float's multiply, float-to-int truncation,
 * floor and int-to-float with SSE where their results are provably the IEEE operation's (softfloat.c). One copy of
 * the code for the shim (P "_": the i686 Windows symbol prefix) and for the native exhaustive test (P ""), which
 * runs the executable's own routines against these.
 *
 * cdecl, like the originals; EAX, ECX, EDX and XMM0-1 only (no stack, no MXCSR change: every SSE operation on the
 * fast paths is exact, so the rounding mode, DAZ and FTZ do not matter; an operand outside them may set a status
 * flag, which nothing reads: the C library's SSE2 math checks only MXCSR's control bits). Outside a fast path: a jump to the
 * original (P"sf_*_orig": the MinHook trampoline), with the stack as the caller left it.
 *
 * mul(out, &a, &b): both exponent fields nonzero, the exact double product p with 2^-125 <= |p| < 2^128 (so the
 *   original's own result exponent is in its normal range: it truncates the exact 48-bit product to 24 bits, which
 *   for these operands is the IEEE single product rounded toward zero): p's low 29 mantissa bits cleared, then
 *   narrowed (exact). Infinities and NaNs give p out of range; zeros and denormals fail the exponent test.
 * f2i(&a): exponent field < 127 (|x| < 1, zeros, denormals): 0 as the original; < 158 (|x| < 2^31): cvttss2si.
 * floor(out, &a): exponent field 127..149 (1 <= |x| < 2^23): roundss (floor, no precision exception: SSE4.1).
 * i2f(out, v): |v| <= 2^24: cvtsi2ss (exact; 0 gives +0 as the original).
 */
#define SF_ASM(P)                                                                                                     \
    ".text\n"                                                                                                         \
    ".globl " P "sf_mul_asm\n" P "sf_mul_asm:\n"                                                                      \
    "  movl 8(%esp), %eax\n"                                                                                          \
    "  movl 12(%esp), %ecx\n"                                                                                         \
    "  cvtss2sd (%eax), %xmm0\n"  /* (from memory: the product's chain starts at the loads) */                       \
    "  cvtss2sd (%ecx), %xmm1\n"                                                                                      \
    "  movl (%eax), %eax\n"                                                                                           \
    "  movl (%ecx), %ecx\n"                                                                                           \
    "  mulsd %xmm1, %xmm0\n"                                                                                          \
    "  testl $0x7f800000, %eax\n"                                                                                     \
    "  jz 9f\n"                                                                                                       \
    "  testl $0x7f800000, %ecx\n"                                                                                     \
    "  jz 9f\n"                                                                                                       \
    "  pshufd $1, %xmm0, %xmm1\n"                                                                                     \
    "  movd %xmm1, %edx\n"                                                                                            \
    "  andl $0x7fffffff, %edx\n"                                                                                      \
    "  subl $0x38200000, %edx\n"  /* |p| >= 2^-125: biased double exponent >= 898 */                                 \
    "  cmpl $0x0fd00000, %edx\n"  /* and < 2^128: < 898 + 253 */                                                     \
    "  jae 9f\n"                                                                                                      \
    "  pcmpeqd %xmm1, %xmm1\n"                                                                                        \
    "  psllq $29, %xmm1\n"                                                                                            \
    "  andpd %xmm1, %xmm0\n"                                                                                          \
    "  cvtsd2ss %xmm0, %xmm0\n"                                                                                       \
    "  movl 4(%esp), %eax\n"                                                                                          \
    "  movss %xmm0, (%eax)\n"                                                                                         \
    "  ret\n"                                                                                                         \
    "9: jmp *" P "sf_mul_orig\n"                                                                                      \
    ".globl " P "sf_f2i_asm\n" P "sf_f2i_asm:\n"                                                                      \
    "  movl 4(%esp), %eax\n"                                                                                          \
    "  movl (%eax), %eax\n"                                                                                           \
    "  movl %eax, %ecx\n"                                                                                             \
    "  andl $0x7f800000, %ecx\n"                                                                                      \
    "  cmpl $0x3f800000, %ecx\n"                                                                                      \
    "  jb 8f\n"                                                                                                       \
    "  cmpl $0x4f000000, %ecx\n"                                                                                      \
    "  jae 9f\n"                                                                                                      \
    "  movd %eax, %xmm0\n"                                                                                            \
    "  cvttss2si %xmm0, %eax\n"                                                                                       \
    "  ret\n"                                                                                                         \
    "8: xorl %eax, %eax\n"                                                                                            \
    "  ret\n"                                                                                                         \
    "9: jmp *" P "sf_f2i_orig\n"                                                                                      \
    ".globl " P "sf_floor_asm\n" P "sf_floor_asm:\n"                                                                  \
    "  movl 8(%esp), %eax\n"                                                                                          \
    "  movl (%eax), %eax\n"                                                                                           \
    "  movl %eax, %ecx\n"                                                                                             \
    "  andl $0x7f800000, %ecx\n"                                                                                      \
    "  subl $0x3f800000, %ecx\n"                                                                                      \
    "  cmpl $0x0b800000, %ecx\n"                                                                                      \
    "  jae 9f\n"                                                                                                      \
    "  movd %eax, %xmm0\n"                                                                                            \
    "  roundss $9, %xmm0, %xmm0\n"                                                                                    \
    "  movl 4(%esp), %eax\n"                                                                                          \
    "  movss %xmm0, (%eax)\n"                                                                                         \
    "  ret\n"                                                                                                         \
    "9: jmp *" P "sf_floor_orig\n"                                                                                    \
    ".globl " P "sf_i2f_asm\n" P "sf_i2f_asm:\n"                                                                      \
    "  movl 8(%esp), %ecx\n"                                                                                          \
    "  leal 0x1000000(%ecx), %edx\n"                                                                                  \
    "  cmpl $0x2000000, %edx\n"                                                                                       \
    "  ja 9f\n"                                                                                                       \
    "  cvtsi2ss %ecx, %xmm0\n"                                                                                        \
    "  movl 4(%esp), %eax\n"                                                                                          \
    "  movss %xmm0, (%eax)\n"                                                                                         \
    "  ret\n"                                                                                                         \
    "9: jmp *" P "sf_i2f_orig\n"

typedef unsigned *(__attribute__((cdecl)) * SfBinFn)(unsigned *, const unsigned *, const unsigned *);
typedef int(__attribute__((cdecl)) * SfF2iFn)(const unsigned *);
typedef unsigned *(__attribute__((cdecl)) * SfUnFn)(unsigned *, const unsigned *);
typedef unsigned *(__attribute__((cdecl)) * SfI2fFn)(unsigned *, int);
unsigned *__attribute__((cdecl)) sf_mul_asm(unsigned *, const unsigned *, const unsigned *);
int __attribute__((cdecl)) sf_f2i_asm(const unsigned *);
unsigned *__attribute__((cdecl)) sf_floor_asm(unsigned *, const unsigned *);
unsigned *__attribute__((cdecl)) sf_i2f_asm(unsigned *, int);
extern void *sf_mul_orig, *sf_f2i_orig, *sf_floor_orig, *sf_i2f_orig;
