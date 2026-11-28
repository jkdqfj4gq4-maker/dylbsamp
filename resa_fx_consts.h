cd /home/claude/work && cat > resa_fx_consts.h <<'EOF'
/*
 * resa_fx_consts.h  --  GL constants, reshadefx enum values and the PURE mapping tables of the ReShade-FX runtime.
 *
 * No GL calls, no globals, no allocation: everything here is a constant or a small function, so tests/test_fx_consts.c checks it on any machine
 * and tests/check_consts.py checks every GL value against Mesa's own enum-name table (or the Khronos headers when they are installed).
 *
 * NEEDS the GL typedefs (GLenum, GLint ...) to be visible first: resa_ao.c has them, the tests include tests/gl_min.h.
 * The constants are named FXGL_<name> (resa_ao.c's own list is GLC_<name>); both are the plain GL values, there is no clash.
 *
 * reshadefx values below are the ones of reshadefx::effect_module.hpp (filter_mode, texture_address_mode, pass_blend_op, pass_blend_factor,
 * primitive_topology).  They come from the upstream header; tests/test_fx_consts.c PINS them, so a different value in the vendored fork shows up
 * as one failing test line, not as a silently wrong picture.  >>> VERIFY AGAINST dep/reshadefx/include/effect_module.hpp WHEN THE SOURCE IS AT HAND <<<
 */
#ifndef RESA_FX_CONSTS_H
#define RESA_FX_CONSTS_H
#include "resa_fx_api.h"

#define FXK_GL_CONSTS(X) \
    /* sized / pixel formats */ \
    X(R8, 0x8229) X(RG8, 0x822B) X(R16F, 0x822D) X(R32F, 0x822E) X(RG16F, 0x822F) X(RG32F, 0x8230) \
    X(R32I, 0x8235) X(R32UI, 0x8236) X(RGBA16F, 0x881A) X(RGBA32F, 0x8814) X(RGB10_A2, 0x8059) \
    X(R16_EXT, 0x822A) X(RG16_EXT, 0x822C) X(RGBA16_EXT, 0x805B) \
    X(RED, 0x1903) X(RG, 0x8227) X(RED_INTEGER, 0x8D94) X(RGBA_INTEGER, 0x8D99) \
    X(UNSIGNED_SHORT, 0x1403) X(INT, 0x1404) X(UNSIGNED_INT, 0x1405) X(FLOAT, 0x1406) X(HALF_FLOAT, 0x140B) \
    X(UNSIGNED_INT_2_10_10_10_REV, 0x8368) \
    /* sampler state */ \
    X(NEAREST_MIPMAP_NEAREST, 0x2700) X(LINEAR_MIPMAP_NEAREST, 0x2701) X(NEAREST_MIPMAP_LINEAR, 0x2702) X(LINEAR_MIPMAP_LINEAR, 0x2703) \
    X(REPEAT, 0x2901) X(MIRRORED_REPEAT, 0x8370) \
    X(TEXTURE_MIN_LOD, 0x813A) X(TEXTURE_MAX_LOD, 0x813B) X(TEXTURE_BASE_LEVEL, 0x813C) X(TEXTURE_MAX_LEVEL, 0x813D) \
    /* blending */ \
    X(ZERO, 0) X(ONE, 1) X(SRC_COLOR, 0x0300) X(ONE_MINUS_SRC_COLOR, 0x0301) X(SRC_ALPHA, 0x0302) X(ONE_MINUS_SRC_ALPHA, 0x0303) \
    X(DST_ALPHA, 0x0304) X(ONE_MINUS_DST_ALPHA, 0x0305) X(DST_COLOR, 0x0306) X(ONE_MINUS_DST_COLOR, 0x0307) \
    X(FUNC_ADD, 0x8006) X(MIN, 0x8007) X(MAX, 0x8008) X(FUNC_SUBTRACT, 0x800A) X(FUNC_REVERSE_SUBTRACT, 0x800B) \
    X(BLEND_EQUATION_RGB, 0x8009) X(BLEND_EQUATION_ALPHA, 0x883D) \
    X(BLEND_DST_RGB, 0x80C8) X(BLEND_SRC_RGB, 0x80C9) X(BLEND_DST_ALPHA, 0x80CA) X(BLEND_SRC_ALPHA, 0x80CB) \
    /* primitives, limits, queries */ \
    X(POINTS, 0x0000) X(LINES, 0x0001) X(LINE_STRIP, 0x0003) X(TRIANGLE_STRIP, 0x0005) \
    X(MAX_COLOR_ATTACHMENTS, 0x8CDF) X(MAX_DRAW_BUFFERS, 0x8824) X(MAX_TEXTURE_SIZE, 0x0D33) X(MAX_TEXTURE_IMAGE_UNITS, 0x8872) \
    X(EXTENSIONS, 0x1F03) X(COLOR_CLEAR_VALUE, 0x0C22) X(FRAMEBUFFER_UNSUPPORTED, 0x8CDD) \
    X(TEXTURE_2D_ARRAY_UNUSED_PLACEHOLDER_DO_NOT_USE, 0x8C1A)

enum {
#define X(n, v) FXGL_##n = v,
    FXK_GL_CONSTS(X)
#undef X
    FXGL__END
};

/* every constant by name, for the tests */
typedef struct { const char *name; unsigned value; } FxkGlConst;
static const FxkGlConst fxk_gl_table[] __attribute__((unused)) = {
#define X(n, v) { #n, (unsigned)(v) },
    FXK_GL_CONSTS(X)
#undef X
    { 0, 0 }
};

/* ------------------------------------------------------------------ reshadefx enum values */
enum { FXK_ADDR_WRAP = 1, FXK_ADDR_MIRROR = 2, FXK_ADDR_CLAMP = 3, FXK_ADDR_BORDER = 4 };
enum { FXK_FILT_MIP_LINEAR = 0x01, FXK_FILT_MAG_LINEAR = 0x04, FXK_FILT_MIN_LINEAR = 0x10 };
enum { FXK_BOP_ADD = 1, FXK_BOP_SUBTRACT = 2, FXK_BOP_REVERSE_SUBTRACT = 3, FXK_BOP_MIN = 4, FXK_BOP_MAX = 5 };
enum { FXK_BF_ZERO = 0, FXK_BF_ONE = 1, FXK_BF_SRC_COLOR = 2, FXK_BF_ONE_MINUS_SRC_COLOR = 3, FXK_BF_SRC_ALPHA = 4, FXK_BF_ONE_MINUS_SRC_ALPHA = 5,
       FXK_BF_DST_ALPHA = 6, FXK_BF_ONE_MINUS_DST_ALPHA = 7, FXK_BF_DST_COLOR = 8, FXK_BF_ONE_MINUS_DST_COLOR = 9 };
enum { FXK_TOPO_POINTS = 1, FXK_TOPO_LINES = 2, FXK_TOPO_LINE_STRIP = 3, FXK_TOPO_TRIANGLES = 4, FXK_TOPO_TRIANGLE_STRIP = 5 };

/* what the bridge writes into __RESHADE__ (major * 10000 + minor * 100 + revision).  Effects test it with  #if __RESHADE__ >= 50000 .
 * Set it to the version of the vendored reshadefx; 5.9.2 is what DuckStation's fork was synced to when this was written (unverified). */
#define FXK_RESHADE_VERSION 50902

/* ------------------------------------------------------------------ reshadefx -> GL mapping (pure) */
static inline unsigned fxk_blend_factor(int f) {
    switch (f) {
    case FXK_BF_ZERO:                 return FXGL_ZERO;
    case FXK_BF_ONE:                  return FXGL_ONE;
    case FXK_BF_SRC_COLOR:            return FXGL_SRC_COLOR;
    case FXK_BF_ONE_MINUS_SRC_COLOR:  return FXGL_ONE_MINUS_SRC_COLOR;
    case FXK_BF_SRC_ALPHA:            return FXGL_SRC_ALPHA;
    case FXK_BF_ONE_MINUS_SRC_ALPHA:  return FXGL_ONE_MINUS_SRC_ALPHA;
    case FXK_BF_DST_ALPHA:            return FXGL_DST_ALPHA;
    case FXK_BF_ONE_MINUS_DST_ALPHA:  return FXGL_ONE_MINUS_DST_ALPHA;
    case FXK_BF_DST_COLOR:            return FXGL_DST_COLOR;
    case FXK_BF_ONE_MINUS_DST_COLOR:  return FXGL_ONE_MINUS_DST_COLOR;
    default:                          return FXGL_ONE;
    }
}
static inline unsigned fxk_blend_op(int op) {
    switch (op) {
    case FXK_BOP_ADD:              return FXGL_FUNC_ADD;
    case FXK_BOP_SUBTRACT:         return FXGL_FUNC_SUBTRACT;
    case FXK_BOP_REVERSE_SUBTRACT: return FXGL_FUNC_REVERSE_SUBTRACT;
    case FXK_BOP_MIN:              return FXGL_MIN;
    case FXK_BOP_MAX:              return FXGL_MAX;
    default:                       return FXGL_FUNC_ADD;
    }
}
/* texture address mode -> GL wrap.  ES 3.0 has no CLAMP_TO_BORDER: border falls back to clamp-to-edge and *approx is set so the caller can log it. */
static inline unsigned fxk_wrap(int a, int *approx) {
    switch (a) {
    case FXK_ADDR_WRAP:   return FXGL_REPEAT;
    case FXK_ADDR_MIRROR: return FXGL_MIRRORED_REPEAT;
    case FXK_ADDR_BORDER: if (approx) *approx = 1; return 0x812F;      /* GL_CLAMP_TO_EDGE */
    case FXK_ADDR_CLAMP:
    default:              return 0x812F;
    }
}
static inline unsigned fxk_mag_filter(int filter) { return (filter & FXK_FILT_MAG_LINEAR) ? 0x2601u : 0x2600u; }      /* GL_LINEAR : GL_NEAREST */
static inline unsigned fxk_min_filter(int filter, int has_mips) {
    int lin = (filter & FXK_FILT_MIN_LINEAR) != 0, mip = (filter & FXK_FILT_MIP_LINEAR) != 0;
    if (!has_mips) return lin ? 0x2601u : 0x2600u;
    if (lin) return mip ? FXGL_LINEAR_MIPMAP_LINEAR : FXGL_LINEAR_MIPMAP_NEAREST;
    return mip ? FXGL_NEAREST_MIPMAP_LINEAR : FXGL_NEAREST_MIPMAP_NEAREST;
}
static inline unsigned fxk_primitive(int topo) {
    switch (topo) {
    case FXK_TOPO_POINTS:         return FXGL_POINTS;
    case FXK_TOPO_LINES:          return FXGL_LINES;
    case FXK_TOPO_LINE_STRIP:     return FXGL_LINE_STRIP;
    case FXK_TOPO_TRIANGLE_STRIP: return FXGL_TRIANGLE_STRIP;
    case FXK_TOPO_TRIANGLES:
    default:                      return 0x0004u;                      /* GL_TRIANGLES */
    }
}

/* ------------------------------------------------------------------ texture formats */
typedef struct {
    int      fxf;            /* FXF_* */
    unsigned ifmt, fmt, type;/* glTexStorage2D internal format; glTexSubImage2D format + type */
    int      bytes, comps;   /* bytes per texel, channels */
    int      is_int;         /* integer texture (never filterable, sampled with isampler2D) */
    int      float_rt;       /* needs EXT_color_buffer_float / half_float to be a render target */
    int      norm16;         /* needs EXT_texture_norm16 */
    int      float32;        /* 32-bit float: filterable only with OES_texture_float_linear */
    const char *name;
} FxkFmt;

static const FxkFmt fxk_fmt[] = {
    { FXF_UNKNOWN, 0, 0, 0, 0, 0, 0, 0, 0, 0, "unknown" },
    { FXF_R8,      FXGL_R8,        FXGL_RED,  0x1401u,                          1, 1, 0, 0, 0, 0, "R8" },
    { FXF_R16,     FXGL_R16_EXT,   FXGL_RED,  FXGL_UNSIGNED_SHORT,              2, 1, 0, 0, 1, 0, "R16" },
    { FXF_R16F,    FXGL_R16F,      FXGL_RED,  FXGL_HALF_FLOAT,                  2, 1, 0, 1, 0, 0, "R16F" },
    { FXF_R32I,    FXGL_R32I,      FXGL_RED_INTEGER, FXGL_INT,                  4, 1, 1, 0, 0, 0, "R32I" },
    { FXF_R32U,    FXGL_R32UI,     FXGL_RED_INTEGER, FXGL_UNSIGNED_INT,         4, 1, 1, 0, 0, 0, "R32U" },
    { FXF_R32F,    FXGL_R32F,      FXGL_RED,  FXGL_FLOAT,                       4, 1, 0, 1, 0, 1, "R32F" },
    { FXF_RG8,     FXGL_RG8,       FXGL_RG,   0x1401u,                          2, 2, 0, 0, 0, 0, "RG8" },
    { FXF_RG16,    FXGL_RG16_EXT,  FXGL_RG,   FXGL_UNSIGNED_SHORT,              4, 2, 0, 0, 1, 0, "RG16" },
    { FXF_RG16F,   FXGL_RG16F,     FXGL_RG,   FXGL_HALF_FLOAT,                  4, 2, 0, 1, 0, 0, "RG16F" },
    { FXF_RG32F,   FXGL_RG32F,     FXGL_RG,   FXGL_FLOAT,                       8, 2, 0, 1, 0, 1, "RG32F" },
    { FXF_RGBA8,   0x8058u,        0x1908u,   0x1401u,                          4, 4, 0, 0, 0, 0, "RGBA8" },
    { FXF_RGBA16,  FXGL_RGBA16_EXT,0x1908u,   FXGL_UNSIGNED_SHORT,              8, 4, 0, 0, 1, 0, "RGBA16" },
    { FXF_RGBA16F, FXGL_RGBA16F,   0x1908u,   FXGL_HALF_FLOAT,                  8, 4, 0, 1, 0, 0, "RGBA16F" },
    { FXF_RGBA32F, FXGL_RGBA32F,   0x1908u,   FXGL_FLOAT,                      16, 4, 0, 1, 0, 1, "RGBA32F" },
    { FXF_RGB10A2, FXGL_RGB10_A2,  0x1908u,   FXGL_UNSIGNED_INT_2_10_10_10_REV, 4, 4, 0, 0, 0, 0, "RGB10A2" },
};
#define FXK_NFMT ((int)(sizeof fxk_fmt / sizeof fxk_fmt[0]))

static inline const FxkFmt *fxk_fmt_of(int fxf) { return (fxf > 0 && fxf < FXK_NFMT) ? &fxk_fmt[fxf] : &fxk_fmt[0]; }

/* What to try when a format cannot be a texture / render target on this GPU: the format itself first, then these, in this order (0 ends the list).
 * Rules: keep the precision class as long as possible (float before 8-bit), prefer MORE channels over a lower precision, and every chain ends in
 * RGBA8 which every ES 3.0 GPU renders to.  Integer formats have no fallback (an RGBA8 stand-in would make the effect wrong, not just different). */
static const unsigned char fxk_chain[FXK_NFMT][8] = {
    { 0 },
    /* R8      */ { FXF_R8,      FXF_RG8,     FXF_RGBA8, 0 },
    /* R16     */ { FXF_R16,     FXF_R16F,    FXF_R32F,  FXF_RGBA16F, FXF_R8,      FXF_RGBA8, 0 },
    /* R16F    */ { FXF_R16F,    FXF_R32F,    FXF_RGBA16F, FXF_RGBA32F, FXF_R8,    FXF_RGBA8, 0 },
    /* R32I    */ { FXF_R32I, 0 },
    /* R32U    */ { FXF_R32U, 0 },
    /* R32F    */ { FXF_R32F,    FXF_RGBA32F, FXF_R16F,  FXF_RGBA16F, FXF_RGBA8, 0 },
    /* RG8     */ { FXF_RG8,     FXF_RGBA8, 0 },
    /* RG16    */ { FXF_RG16,    FXF_RG16F,   FXF_RG32F, FXF_RGBA16F, FXF_RG8,    FXF_RGBA8, 0 },
    /* RG16F   */ { FXF_RG16F,   FXF_RG32F,   FXF_RGBA16F, FXF_RGBA32F, FXF_RG8,  FXF_RGBA8, 0 },
    /* RG32F   */ { FXF_RG32F,   FXF_RGBA32F, FXF_RG16F, FXF_RGBA16F, FXF_RGBA8, 0 },
    /* RGBA8   */ { FXF_RGBA8, 0 },
    /* RGBA16  */ { FXF_RGBA16,  FXF_RGBA16F, FXF_RGBA32F, FXF_RGB10A2, FXF_RGBA8, 0 },
    /* RGBA16F */ { FXF_RGBA16F, FXF_RGBA32F, FXF_RGB10A2, FXF_RGBA8, 0 },
    /* RGBA32F */ { FXF_RGBA32F, FXF_RGBA16F, FXF_RGB10A2, FXF_RGBA8, 0 },
    /* RGB10A2 */ { FXF_RGB10A2, FXF_RGBA8, 0 },
};

/* ------------------------------------------------------------------ std140 layout of one uniform (pure; the fake bridge and the runtime share it)
 * rows x cols as in FxUniformDesc (a float3 is rows 3, cols 1; a float3x4 is rows 3, cols 4).  GLSL sees a matrix with `cols` COLUMNS of `rows`
 * components (mat<cols>x<rows>), each column padded to a vec4 in std140.  Returns the size in bytes of the whole member (array included) and fills
 * *align and *stride (stride = bytes from one array element to the next, 0 when it is not an array). */
static inline int fxk_std140(int rows, int cols, int array_len, int *align, int *stride) {
    int a, sz;
    if (rows < 1) rows = 1;
    if (cols < 1) cols = 1;
    if (cols == 1) { a = rows == 1 ? 4 : (rows == 2 ? 8 : 16); sz = rows * 4; }
    else           { a = 16; sz = cols * 16; }
    if (array_len > 0) {
        int st = (sz + 15) & ~15;
        if (align) *align = 16;
        if (stride) *stride = st;
        return st * array_len;
    }
    if (align) *align = a;
    if (stride) *stride = 0;
    return sz;
}

#endif /* RESA_FX_CONSTS_H */
EOF
sed -i '/TEXTURE_2D_ARRAY_UNUSED_PLACEHOLDER_DO_NOT_USE/d' resa_fx_consts.h
sed -i 's|X(EXTENSIONS, 0x1F03) X(COLOR_CLEAR_VALUE, 0x0C22) X(FRAMEBUFFER_UNSUPPORTED, 0x8CDD) \\|X(EXTENSIONS, 0x1F03) X(COLOR_CLEAR_VALUE, 0x0C22) X(FRAMEBUFFER_UNSUPPORTED, 0x8CDD)|' resa_fx_consts.h
sed -n '20,48p' resa_fx_consts.h
Output
    X(R8, 0x8229) X(RG8, 0x822B) X(R16F, 0x822D) X(R32F, 0x822E) X(RG16F, 0x822F) X(RG32F, 0x8230) \
    X(R32I, 0x8235) X(R32UI, 0x8236) X(RGBA16F, 0x881A) X(RGBA32F, 0x8814) X(RGB10_A2, 0x8059) \
    X(R16_EXT, 0x822A) X(RG16_EXT, 0x822C) X(RGBA16_EXT, 0x805B) \
    X(RED, 0x1903) X(RG, 0x8227) X(RED_INTEGER, 0x8D94) X(RGBA_INTEGER, 0x8D99) \
    X(UNSIGNED_SHORT, 0x1403) X(INT, 0x1404) X(UNSIGNED_INT, 0x1405) X(FLOAT, 0x1406) X(HALF_FLOAT, 0x140B) \
    X(UNSIGNED_INT_2_10_10_10_REV, 0x8368) \
    /* sampler state */ \
    X(NEAREST_MIPMAP_NEAREST, 0x2700) X(LINEAR_MIPMAP_NEAREST, 0x2701) X(NEAREST_MIPMAP_LINEAR, 0x2702) X(LINEAR_MIPMAP_LINEAR, 0x2703) \
    X(REPEAT, 0x2901) X(MIRRORED_REPEAT, 0x8370) \
    X(TEXTURE_MIN_LOD, 0x813A) X(TEXTURE_MAX_LOD, 0x813B) X(TEXTURE_BASE_LEVEL, 0x813C) X(TEXTURE_MAX_LEVEL, 0x813D) \
    /* blending */ \
    X(ZERO, 0) X(ONE, 1) X(SRC_COLOR, 0x0300) X(ONE_MINUS_SRC_COLOR, 0x0301) X(SRC_ALPHA, 0x0302) X(ONE_MINUS_SRC_ALPHA, 0x0303) \
    X(DST_ALPHA, 0x0304) X(ONE_MINUS_DST_ALPHA, 0x0305) X(DST_COLOR, 0x0306) X(ONE_MINUS_DST_COLOR, 0x0307) \
    X(FUNC_ADD, 0x8006) X(MIN, 0x8007) X(MAX, 0x8008) X(FUNC_SUBTRACT, 0x800A) X(FUNC_REVERSE_SUBTRACT, 0x800B) \
    X(BLEND_EQUATION_RGB, 0x8009) X(BLEND_EQUATION_ALPHA, 0x883D) \
    X(BLEND_DST_RGB, 0x80C8) X(BLEND_SRC_RGB, 0x80C9) X(BLEND_DST_ALPHA, 0x80CA) X(BLEND_SRC_ALPHA, 0x80CB) \
    /* primitives, limits, queries */ \
    X(POINTS, 0x0000) X(LINES, 0x0001) X(LINE_STRIP, 0x0003) X(TRIANGLE_STRIP, 0x0005) \
    X(MAX_COLOR_ATTACHMENTS, 0x8CDF) X(MAX_DRAW_BUFFERS, 0x8824) X(MAX_TEXTURE_SIZE, 0x0D33) X(MAX_TEXTURE_IMAGE_UNITS, 0x8872) \
    X(EXTENSIONS, 0x1F03) X(COLOR_CLEAR_VALUE, 0x0C22) X(FRAMEBUFFER_UNSUPPORTED, 0x8CDD)

enum {
#define X(n, v) FXGL_##n = v,
    FXK_GL_CONSTS(X)
#undef X
    FXGL__END
};

/* every constant by name, for the tests */