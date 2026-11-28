/*
 * resa_fx_api.h  --  the contract between the ReShade-FX runtime (resa_fx.inc), the compiler bridge and the Linux test double.
 *
 * Plain C on purpose: resa_ao.c stays pure C, and the one C++ file (resa_fx_bridge.cpp, which links reshadefx) only has to fill these structs.
 *
 *   resa_fx_bridge.cpp   the real thing: reshadefx preprocessor + parser + GLSL back-end (DuckStation's dep/reshadefx fork, gles = true)
 *   tests/fx_fake.c      test double: hand-written modules whose sources look exactly like that back-end's GLES output
 *
 * Both call fxc_convert() (resa_fx_conv.inc) on every entry point, so what reaches the GL driver is strict GLSL ES 3.00.
 */
#ifndef RESA_FX_API_H
#define RESA_FX_API_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define FXM_NAME    96
#define FXM_LABEL   128
#define FXM_MAX_RT  8
#define FXM_MAX_SMP 16

typedef enum { FXU_BOOL = 0, FXU_INT, FXU_UINT, FXU_FLOAT } FxUBase;

/* what a uniform's  < source = "..." >  annotation asks the host for */
typedef enum {
    FXS_NONE = 0,       /* an ordinary setting: value comes from resa_fx.ini (or the default in the .fx) */
    FXS_FRAMETIME,      /* float, ms the last frame took */
    FXS_FRAMECOUNT,     /* int / uint */
    FXS_TIMER,          /* float, ms since the effects were loaded */
    FXS_DATE,           /* float4: year, month, day, seconds of the day */
    FXS_PINGPONG,       /* float2: value between min..max, direction */
    FXS_RANDOM,         /* int / float in min..max, new every frame */
    FXS_BUFREADY_DEPTH, /* bool: 1 when a depth image is available */
    FXS_ZERO            /* key, mouse*, overlay_*, freepie, ...: no input device here, always 0 */
} FxSource;

typedef struct {
    char  name[FXM_NAME];       /* name as written in the .fx */
    char  glname[FXM_NAME];     /* name of the plain uniform in the converted GLSL (filled by the bridge from fxc_convert) */
    char  label[FXM_LABEL];     /* ui_label, or the name */
    char  category[FXM_LABEL];
    char  ui_type[16];          /* "", slider, drag, input, combo, radio, color */
    char  items[512];           /* combo / radio entries, each ended by '\0', the list ended by an empty entry */
    int   base, rows, cols;     /* FxUBase, HLSL rows x cols (a float3 is rows = 3, cols = 1) */
    int   array_len;            /* 0 = not an array */
    int   offset, size;         /* std140 byte offset and size inside the effect's value buffer */
    int   source;               /* FxSource */
    float ui_min, ui_max, ui_step;
    int   has_min, has_max;
    float src_min, src_max, src_step, src_smooth;   /* pingpong / random annotations */
    uint32_t init[16];          /* default value, raw 32-bit words, row-major (arrays: not kept) */
    int   has_init;
} FxUniformDesc;

/* texture formats: same order as reshadefx::texture_format */
enum { FXF_UNKNOWN = 0, FXF_R8, FXF_R16, FXF_R16F, FXF_R32I, FXF_R32U, FXF_R32F, FXF_RG8, FXF_RG16, FXF_RG16F, FXF_RG32F,
       FXF_RGBA8, FXF_RGBA16, FXF_RGBA16F, FXF_RGBA32F, FXF_RGB10A2 };
enum { FXSEM_NONE = 0, FXSEM_COLOR, FXSEM_DEPTH };

typedef struct {
    char name[FXM_NAME];
    int  w, h, levels, fmt;
    int  semantic;              /* FXSEM_* : COLOR = the image being processed, DEPTH = the scene depth */
    int  render_target;         /* some pass renders into it */
    char source[160];           /* source = "file.png" annotation, or "" */
} FxTexDesc;

typedef struct {                /* one texture unit of one pass */
    int   unit;                 /* texture unit == the binding the sampler had in the generated code */
    int   tex;                  /* index into FxModule.tex */
    int   filter;               /* reshadefx::filter_mode: bit 0x10 = min linear, 0x04 = mag linear, 0x01 = mip linear */
    int   addr_u, addr_v;       /* 1 wrap, 2 mirror, 3 clamp, 4 border */
    float min_lod, max_lod, lod_bias;
    int   srgb;
    char  name[FXM_NAME];       /* the sampler's name in the converted GLSL (filled from fxc_convert) */
} FxBind;

typedef struct {
    char  name[FXM_NAME];
    char *vs, *ps;              /* strict GLSL ES 3.00, malloc'd; fxm_free() releases them */
    int   rt[FXM_MAX_RT];       /* texture index per render target, -1 = none.  rt[0] == -1: the pass writes the image being processed */
    int   nbind; FxBind bind[FXM_MAX_SMP];
    int   clear, gen_mips;
    int   blend, src_c, dst_c, op_c, src_a, dst_a, op_a;   /* reshadefx::blend_factor / blend_op values, render target 0 */
    int   wmask;                /* render target 0 write mask, 4 bits (1 = r ... 8 = a) */
    int   srgb_write;
    int   stencil;              /* unsupported: the bridge rejects the effect, kept for the log */
    int   topo, nverts;         /* reshadefx::primitive_topology, vertex count */
    int   vp_w, vp_h;           /* 0 = size of the target */
} FxPassDesc;

typedef struct {
    char name[FXM_NAME];
    char label[FXM_LABEL];
    int  enabled;               /* default: 1 unless  < enabled = false >  */
    int  first_pass, npass;
} FxTechDesc;

typedef struct {
    int ntex;  FxTexDesc *tex;
    int nuni;  FxUniformDesc *uni;  int uni_bytes;     /* uni_bytes = size of the value buffer */
    int npass; FxPassDesc *pass;
    int ntech; FxTechDesc *tech;
    int need_depth;             /* some texture has the DEPTH semantic, or some uniform asks for bufready_depth */
} FxModule;

/* what the host wants compiled */
typedef struct {
    const char *path;           /* full path of the .fx file */
    const char *const *inc;     /* include directories, NULL-terminated */
    const char *const *defs;    /* "NAME=VALUE" strings, NULL-terminated (BUFFER_WIDTH & co. are in here) */
    int w, h;                   /* render size: also becomes BUFFER_WIDTH / BUFFER_HEIGHT */
} FxCompileReq;

/* Implemented by the bridge (real or fake).  Returns 1 and fills *out, or 0 and writes a one-paragraph reason into err (compiler messages included). */
int  fxb_compile(const FxCompileReq *req, FxModule *out, char *err, size_t errlen);
/* Frees everything inside *m (not m itself). */
void fxm_free(FxModule *m);

#ifdef __cplusplus
}
#endif
#endif
