/*
 * resa_fx_bridge.cpp  --  compiler bridge of the ReShade-FX runtime:  .fx file  ->  FxModule   (the contract is resa_fx_api.h)
 *
 *   read .fx -> reshadefx::preprocessor -> reshadefx::parser -> GLSL ES codegen -> write_result(module)
 *   per pass:  finalize_code_for_entry_point(vs | ps) -> fxc_convert() (resa_fx_conv.inc) -> strdup -> FxPassDesc.vs / .ps
 *
 * Contract: returns 1 and an FxModule that the caller releases with fxm_free(), or 0 with *out left zeroed and a one-paragraph reason in err.
 *           Every pointer inside FxModule is calloc / malloc / strdup (never new), so fxm_free() and the C test double can free() them.
 *
 * reshadefx API assumptions (no header was at hand; everything that touches the library is in FxbCodegen, fxb_build(), and the
 * templates over `mod`, so a signature that differs in the vendored fork is a one-line fix):
 *   - Codegen.  resa_fx_conv.inc documents the back-end as  create_codegen_glsl(version = 300, gles = true, vulkan_semantics = false,
 *     debug_info = false, uniforms_to_spec_constants = false, enable_16bit_types = false, flip_vert_y = true)  and the source as
 *     std::string codegen::finalize_code_for_entry_point(name).  fxc_convert() hard-fails on anything else (non-ES #version, 16-bit types).
 *     The hand-over note instead says  reshadefx::codegen_glsl codegen(true)  and  finalize_code_for_entry_point(name, is_vertex):
 *     build with -DFXB_TASK_NOTE_API=1 for those shapes.  Only FxbCodegen changes.
 *   - Module type is reshadefx::module; for the renamed one build with -DFXB_MODULE_T=reshadefx::effect_module.
 *   - finalize_code_for_entry_point() is called AFTER write_result() (which only moves the module out; the function bodies stay in the codegen).
 *   - Annotations are std::vector<reshadefx::annotation>; texture_info::semantic is a string ("COLOR" / "DEPTH") or an enum with
 *     ::color / ::depth; the blend / write-mask fields of a pass are scalars or per-render-target arrays (render target 0 is used).
 *
 * Build: C++17, include path of dep/reshadefx/include, resa_fx_api.h and resa_fx_conv.inc next to this file, link reshadefx.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "reshadefx/effect_module.hpp"
#include "reshadefx/effect_preprocessor.hpp"
#include "reshadefx/effect_parser.hpp"
#include "reshadefx/effect_codegen.hpp"

#include "resa_fx_api.h"
#include "resa_fx_conv.inc"

#ifndef FXB_MODULE_T
#define FXB_MODULE_T reshadefx::module
#endif

/* value of __RESHADE__ (major * 10000 + minor * 100 + revision); resa_fx_consts.h calls it FXK_RESHADE_VERSION */
#ifndef FXB_RESHADE_VERSION
#ifdef FXK_RESHADE_VERSION
#define FXB_RESHADE_VERSION FXK_RESHADE_VERSION
#else
#define FXB_RESHADE_VERSION 50902
#endif
#endif

#if defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND)
#define FXB_EXCEPTIONS 1
#else
#define FXB_EXCEPTIONS 0
#endif

namespace {

/* ================================================================== the reshadefx back-end  [reshadefx API] */

#if defined(FXB_TASK_NOTE_API) && FXB_TASK_NOTE_API
struct FxbCodegen {                                    /* the shapes named in the hand-over note */
    reshadefx::codegen_glsl impl{ true };              /* true = GLES */
    reshadefx::codegen *base() { return &impl; }
    std::string entry(const std::string &name, bool vertex) { return impl.finalize_code_for_entry_point(name, vertex); }
};
#else
struct FxbCodegen {                                    /* the shapes resa_fx_conv.inc documents */
    std::unique_ptr<reshadefx::codegen> impl{ reshadefx::create_codegen_glsl(300, true, false, false, false, false, true) };
    reshadefx::codegen *base() { return impl.get(); }
    std::string entry(const std::string &name, bool) { return impl->finalize_code_for_entry_point(name); }
};
#endif

/* ================================================================== small helpers */

/* copy with truncation; false when the string did not fit */
template <size_t N> bool fxb_cpy(char (&dst)[N], const std::string &s)
{
    const size_t n = s.size() < N - 1 ? s.size() : N - 1;
    memcpy(dst, s.data(), n);
    dst[n] = 0;
    return n == s.size();
}

std::string fxb_trim(const std::string &s)
{
    size_t e = s.size();
    while (e && (s[e - 1] == '\n' || s[e - 1] == '\r' || s[e - 1] == ' ' || s[e - 1] == '\t')) e--;
    return s.substr(0, e);
}

bool fxb_read_file(const char *path, std::string &out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    if (f.bad()) return false;
    if (out.size() >= 3 && (unsigned char)out[0] == 0xEF && (unsigned char)out[1] == 0xBB && (unsigned char)out[2] == 0xBF) out.erase(0, 3);   /* UTF-8 BOM */
    return true;
}

/* std140 size and base alignment of one uniform: the same rules as fxk_std140() in resa_fx_consts.h.  rows x cols as in FxUniformDesc
 * (float3 = 3 x 1); a matrix is `cols` column vectors of `rows` components, each padded to a vec4; an array rounds its element up to 16. */
int fxb_std140(int rows, int cols, int array_len, int &align)
{
    int a, sz;
    if (rows < 1) rows = 1;
    if (cols < 1) cols = 1;
    if (cols == 1) { a = rows == 1 ? 4 : (rows == 2 ? 8 : 16); sz = rows * 4; }
    else           { a = 16; sz = cols * 16; }
    if (array_len > 0) { align = 16; return ((sz + 15) & ~15) * array_len; }
    align = a;
    return sz;
}

/* number of scalar components of a GLSL type as the back-end writes it (float, ivec3, mat4, mat3x4 ...); 0 = not recognised */
int fxb_glsl_comps(const char *t)
{
    int c = 0, r = 0;
    if (!strcmp(t, "float") || !strcmp(t, "int") || !strcmp(t, "uint") || !strcmp(t, "bool")) return 1;
    if (sscanf(t, "vec%d", &c) == 1 || sscanf(t, "ivec%d", &c) == 1 || sscanf(t, "uvec%d", &c) == 1 || sscanf(t, "bvec%d", &c) == 1) return c;
    if (sscanf(t, "mat%dx%d", &c, &r) == 2) return c * r;
    if (sscanf(t, "mat%d", &c) == 1) return c * c;
    return 0;
}

/* blend factors / ops / write mask are per render target in newer reshadefx and plain scalars in older ones: take render target 0 either way */
template <class T> auto fxb_first(const T &v, int) -> decltype(v[0]) { return v[0]; }
template <class T> const T &fxb_first(const T &v, long) { return v; }
template <class T> auto fxb_rt0(const T &v) -> decltype(fxb_first(v, 0)) { return fxb_first(v, 0); }

/* texture semantic: a string in upstream reshadefx, an enum in some forks */
int fxb_semantic_of(const std::string &s)
{
    if (s == "DEPTH") return FXSEM_DEPTH;
    if (s == "COLOR") return FXSEM_COLOR;
    return FXSEM_NONE;
}
template <class E, class = typename std::enable_if<std::is_enum<E>::value>::type>
int fxb_semantic_of(E s) { return s == E::depth ? FXSEM_DEPTH : (s == E::color ? FXSEM_COLOR : FXSEM_NONE); }

/* ---- annotations ---- */
template <class L> const reshadefx::annotation *fxb_ann(const L &list, const char *name)
{
    for (const auto &a : list)
        if (a.name == name) return &a;
    return nullptr;
}

bool fxb_ann_str(const reshadefx::annotation *a, std::string &out)
{
    if (!a || a->type.base != reshadefx::type::t_string) return false;
    out = a->value.string_data;
    return true;
}

bool fxb_ann_num(const reshadefx::annotation *a, int idx, float &out)       /* int / uint / bool / float annotation as a float */
{
    if (!a || idx < 0 || idx > 15) return false;
    switch (a->type.base) {
    case reshadefx::type::t_float: out = a->value.as_float[idx]; return true;
    case reshadefx::type::t_int:   out = (float)a->value.as_int[idx]; return true;
    case reshadefx::type::t_bool:
    case reshadefx::type::t_uint:  out = (float)a->value.as_uint[idx]; return true;
    default:                       return false;
    }
}

int fxb_source(const std::string &s)
{
    if (s == "frametime")      return FXS_FRAMETIME;
    if (s == "framecount")     return FXS_FRAMECOUNT;
    if (s == "timer")          return FXS_TIMER;
    if (s == "date")           return FXS_DATE;
    if (s == "pingpong")       return FXS_PINGPONG;
    if (s == "random")         return FXS_RANDOM;
    if (s == "bufready_depth") return FXS_BUFREADY_DEPTH;
    return s.empty() ? FXS_NONE : FXS_ZERO;            /* key, mouse*, overlay_*, freepie, ...: no input device here */
}

/* ui_items: entries each ended by '\0' (the string may carry embedded NULs), the list ended by an empty entry.  The buffer is zero filled and
 * at most 510 bytes are copied, so there are always two closing NULs whether or not the effect wrote the last '\0'. */
void fxb_items(char (&dst)[512], const std::string &s)
{
    memset(dst, 0, sizeof dst);
    memcpy(dst, s.data(), s.size() < sizeof dst - 2 ? s.size() : sizeof dst - 2);
}

/* ---- lookups in the module ---- */
template <class M> int fxb_find_tex(const M &mod, const std::string &n)          /* samplers and passes name a texture by unique_name */
{
    if (n.empty()) return -1;
    for (size_t i = 0; i < mod.textures.size(); i++) if (mod.textures[i].unique_name == n) return (int)i;
    for (size_t i = 0; i < mod.textures.size(); i++) if (mod.textures[i].name == n) return (int)i;
    return -1;
}

template <class M> const reshadefx::sampler_info *fxb_find_sampler(const M &mod, int unit, const char *name)
{
    for (const auto &s : mod.samplers) if ((int)s.binding == unit) return &s;   /* the binding the converted source declared */
    for (const auto &s : mod.samplers) if (s.unique_name == name) return &s;
    return nullptr;
}

/* ---- RAII over the two C structs that own memory ---- */
struct FxbConv {                                       /* one converted entry point */
    FxcOut o;
    FxbConv() { memset(&o, 0, sizeof o); }
    ~FxbConv() { fxc_free(&o); }
    FxbConv(const FxbConv &) = delete;
    FxbConv &operator=(const FxbConv &) = delete;
};

struct FxbModuleGuard {                                /* frees a half built module on any early return or exception */
    FxModule m;
    FxbModuleGuard() { memset(&m, 0, sizeof m); }
    ~FxbModuleGuard() { fxm_free(&m); }
    void release(FxModule *dst) { *dst = m; memset(&m, 0, sizeof m); }
    FxbModuleGuard(const FxbModuleGuard &) = delete;
    FxbModuleGuard &operator=(const FxbModuleGuard &) = delete;
};

/* converted source -> malloc'd copy.  fxc_convert()'s warnings (ES 3.10+ features it cannot fix) have no field in FxPassDesc, so they travel as
 * a trailing GLSL comment: invisible to the driver, visible in the source the runtime dumps when the driver refuses it. */
char *fxb_dup_source(const FxcOut &o)
{
    if (!o.src) return nullptr;
    if (!o.warn[0]) return strdup(o.src);
    static const char pre[] = "// resa_fx: may not compile on ES 3.00: ";
    const size_t n = strlen(o.src), a = sizeof pre - 1, w = strlen(o.warn);
    const bool nl = n && o.src[n - 1] != '\n';
    char *s = (char *)malloc(n + (nl ? 1 : 0) + a + w + 2);
    if (!s) return nullptr;
    size_t p = 0;
    memcpy(s, o.src, n); p += n;
    if (nl) s[p++] = '\n';
    memcpy(s + p, pre, a); p += a;
    memcpy(s + p, o.warn, w); p += w;
    s[p++] = '\n';
    s[p] = 0;
    return s;
}

/* ================================================================== preprocessor input */

void fxb_defines(reshadefx::preprocessor &pp, const FxCompileReq *req)                  /* [reshadefx API] add_macro_definition(name, value) */
{
    std::vector<std::string> seen;
    for (const char *const *d = req->defs; d && *d; ++d) {                              /* the host's "NAME=VALUE" strings win */
        const char *eq = strchr(*d, '=');
        const std::string name = eq ? std::string(*d, (size_t)(eq - *d)) : std::string(*d);
        if (name.empty()) continue;
        pp.add_macro_definition(name, eq ? std::string(eq + 1) : std::string("1"));
        seen.push_back(name);
    }
    auto fallback = [&](const char *name, const std::string &value) {                   /* what ReShade itself defines, unless the host did */
        for (const std::string &s : seen) if (s == name) return;
        pp.add_macro_definition(name, value);
    };
    fallback("__RESHADE__", std::to_string(FXB_RESHADE_VERSION));
    fallback("__RESHADE_PERFORMANCE_MODE__", "0");
    fallback("__RENDERER__", "0x13000");                                                /* ReShade numbering: 0x10000 = OpenGL, here "3.0" */
    fallback("__VENDOR__", "0");
    fallback("__APPLICATION__", "0");
    fallback("BUFFER_WIDTH", std::to_string(req->w));
    fallback("BUFFER_HEIGHT", std::to_string(req->h));
    fallback("BUFFER_RCP_WIDTH", "(1.0 / BUFFER_WIDTH)");
    fallback("BUFFER_RCP_HEIGHT", "(1.0 / BUFFER_HEIGHT)");
    fallback("BUFFER_COLOR_BIT_DEPTH", "8");
}

/* ================================================================== module -> FxModule */

bool fxb_textures(const FXB_MODULE_T &mod, const FxCompileReq *req, FxModule &m, std::string &why)
{
    const size_t n = mod.textures.size();
    m.ntex = (int)n;
    if (!n) return true;
    m.tex = (FxTexDesc *)calloc(n, sizeof(FxTexDesc));
    if (!m.tex) { why = "out of memory"; return false; }
    for (size_t i = 0; i < n; i++) {
        const reshadefx::texture_info &t = mod.textures[i];
        FxTexDesc &d = m.tex[i];
        std::string s;
        if (!fxb_cpy(d.name, t.name)) { why = "texture name too long: " + t.name; return false; }
        d.w = (int)t.width;
        d.h = (int)t.height;
        d.levels = (int)t.levels;
        d.fmt = (int)t.format;                                         /* FXF_* is in the order of reshadefx::texture_format */
        if (d.fmt < FXF_UNKNOWN || d.fmt > FXF_RGB10A2) d.fmt = FXF_UNKNOWN;
        d.semantic = fxb_semantic_of(t.semantic);
        if (d.semantic != FXSEM_NONE) { d.w = req->w; d.h = req->h; }  /* like ReShade: the COLOR / DEPTH textures have the size of the buffer */
        if (fxb_ann_str(fxb_ann(t.annotations, "source"), s) && !fxb_cpy(d.source, s)) { why = "texture '" + t.name + "': source path too long"; return false; }
    }
    return true;
}

bool fxb_uniforms(const FXB_MODULE_T &mod, FxModule &m, std::string &why)
{
    const size_t n = mod.uniforms.size();
    m.nuni = (int)n;
    if (!n) return true;
    m.uni = (FxUniformDesc *)calloc(n, sizeof(FxUniformDesc));
    if (!m.uni) { why = "out of memory"; return false; }

    int cursor = 0;
    for (size_t i = 0; i < n; i++) {
        const reshadefx::uniform_info &ui = mod.uniforms[i];
        const auto &an = ui.annotations;
        FxUniformDesc &d = m.uni[i];
        std::string s;
        float f = 0.f;

        if (!fxb_cpy(d.name, ui.name)) { why = "uniform name too long: " + ui.name; return false; }

        switch (ui.type.base) {
        case reshadefx::type::t_bool: d.base = FXU_BOOL; break;
        case reshadefx::type::t_int:  d.base = FXU_INT; break;
        case reshadefx::type::t_uint: d.base = FXU_UINT; break;
        default:                      d.base = FXU_FLOAT; break;
        }
        d.rows = ui.type.rows ? (int)ui.type.rows : 1;
        d.cols = ui.type.cols ? (int)ui.type.cols : 1;
        d.array_len = ui.type.array_length > 0 ? ui.type.array_length : 0;

        {                                                              /* std140 position in the value buffer (see fxb_std140) */
            int align = 4;
            d.size = fxb_std140(d.rows, d.cols, d.array_len, align);
            cursor = (cursor + align - 1) & ~(align - 1);
            d.offset = cursor;
            cursor += d.size;
        }

        if (fxb_ann_str(fxb_ann(an, "ui_label"), s) && !s.empty()) fxb_cpy(d.label, s); else fxb_cpy(d.label, ui.name);
        if (fxb_ann_str(fxb_ann(an, "ui_category"), s)) fxb_cpy(d.category, s);
        if (fxb_ann_str(fxb_ann(an, "ui_type"), s)) fxb_cpy(d.ui_type, s);
        if (fxb_ann_str(fxb_ann(an, "ui_items"), s)) fxb_items(d.items, s);
        if (fxb_ann_num(fxb_ann(an, "ui_min"), 0, f)) { d.ui_min = f; d.has_min = 1; }
        if (fxb_ann_num(fxb_ann(an, "ui_max"), 0, f)) { d.ui_max = f; d.has_max = 1; }
        if (fxb_ann_num(fxb_ann(an, "ui_step"), 0, f)) d.ui_step = f;

        if (fxb_ann_str(fxb_ann(an, "source"), s)) d.source = fxb_source(s);
        if (d.source == FXS_PINGPONG || d.source == FXS_RANDOM) {
            if (fxb_ann_num(fxb_ann(an, "min"), 0, f)) d.src_min = f;
            if (fxb_ann_num(fxb_ann(an, "max"), 0, f)) d.src_max = f;
            if (fxb_ann_num(fxb_ann(an, "step"), 0, f)) d.src_step = f;
            if (fxb_ann_num(fxb_ann(an, "smoothing"), 0, f)) d.src_smooth = f;
        }

        if (ui.has_initializer_value && d.array_len == 0) {            /* raw words, row-major; arrays keep no default */
            const int words = d.rows * d.cols < 16 ? d.rows * d.cols : 16;
            memcpy(d.init, ui.initializer_value.as_uint, (size_t)words * sizeof(uint32_t));
            d.has_init = 1;
        }
    }
    m.uni_bytes = (cursor + 15) & ~15;
    return true;
}

/* the converted shader's _Globals members must be the module's uniforms, in order: the runtime sets them by glname.  The first converted stage
 * names every uniform (glname), every later one must agree. */
bool fxb_uniforms_match(const FxcOut &o, FxModule &m, bool &named, const std::string &at, std::string &why)
{
    if (o.nuni != m.nuni) {
        why = at + ": the converted shader declares " + std::to_string(o.nuni) + " uniforms, the module has " + std::to_string(m.nuni);
        return false;
    }
    for (int i = 0; i < m.nuni; i++) {
        const FxUniformDesc &u = m.uni[i];
        const int got = fxb_glsl_comps(o.uni[i].type);
        if (o.uni[i].array_len != u.array_len || (got && got != u.rows * u.cols) || (named && strcmp(u.glname, o.uni[i].name))) {
            why = at + ": uniform #" + std::to_string(i) + " is '" + u.name + "' in the module but '" + o.uni[i].type + " " + o.uni[i].name +
                  "' in the converted shader";
            return false;
        }
    }
    if (!named) {
        for (int i = 0; i < m.nuni; i++) memcpy(m.uni[i].glname, o.uni[i].name, FXM_NAME);
        named = true;
    }
    return true;
}

/* the samplers of one converted entry point -> texture units of the pass */
bool fxb_samplers(const FXB_MODULE_T &mod, const FxcOut &o, FxPassDesc &pd, const std::string &at, std::string &why)
{
    for (int i = 0; i < o.nsmp; i++) {
        const FxcSmp &s = o.smp[i];
        bool have = false;
        for (int k = 0; k < pd.nbind; k++) if (pd.bind[k].unit == s.unit) { have = true; break; }
        if (have) continue;                                            /* the other stage reads it too */
        if (pd.nbind >= FXM_MAX_SMP) { why = at + ": more than " + std::to_string(FXM_MAX_SMP) + " textures are sampled by this pass"; return false; }
        const reshadefx::sampler_info *si = fxb_find_sampler(mod, s.unit, s.name);
        if (!si) { why = at + ": sampler '" + s.name + "' (binding " + std::to_string(s.unit) + ") is not in the module"; return false; }
        const int tex = fxb_find_tex(mod, si->texture_name);
        if (tex < 0) { why = at + ": sampler '" + s.name + "' reads texture '" + si->texture_name + "', which is not in the module"; return false; }
        FxBind &b = pd.bind[pd.nbind++];
        b.unit = s.unit;
        b.tex = tex;
        b.filter = (int)si->filter;
        b.addr_u = (int)si->address_u;
        b.addr_v = (int)si->address_v;
        b.min_lod = si->min_lod;
        b.max_lod = si->max_lod;
        b.lod_bias = si->lod_bias;
        b.srgb = si->srgb ? 1 : 0;
        memcpy(b.name, s.name, FXM_NAME);
    }
    return true;
}

bool fxb_stage(FxbCodegen &cg, const std::string &entry, bool is_ps, FxbConv &cx, const std::string &at, std::string &why)
{
    const std::string raw = cg.entry(entry, !is_ps);                   /* [reshadefx API] */
    char e[512];
    e[0] = 0;
    if (!fxc_convert(raw.c_str(), is_ps ? 1 : 0, &cx.o, e, sizeof e)) { why = at + ": " + e; return false; }
    return true;
}

template <class T>                                                     /* T = reshadefx's technique type */
bool fxb_pass(const FXB_MODULE_T &mod, FxbCodegen &cg, const T &tech, size_t k, FxPassDesc &pd, FxModule &m, bool &named, std::string &why)
{
    const auto &pi = tech.passes[k];
    std::string at = "technique '" + tech.name + "', pass " + std::to_string(k + 1);
    if (!pi.name.empty()) at += " ('" + pi.name + "')";

    fxb_cpy(pd.name, pi.name);

    for (int r = 0; r < FXM_MAX_RT; r++) {                             /* rt[0] == -1: the pass writes the image being processed */
        pd.rt[r] = -1;
        const std::string &rn = pi.render_target_names[r];
        if (rn.empty()) continue;
        pd.rt[r] = fxb_find_tex(mod, rn);
        if (pd.rt[r] < 0) { why = at + ": render target '" + rn + "' is not a texture of this effect"; return false; }
        m.tex[pd.rt[r]].render_target = 1;
    }

    pd.clear      = pi.clear_render_targets ? 1 : 0;
    pd.gen_mips   = pi.generate_mipmaps ? 1 : 0;
    pd.srgb_write = pi.srgb_write_enable ? 1 : 0;
    pd.blend      = fxb_rt0(pi.blend_enable) ? 1 : 0;                  /* reshadefx::pass_blend_factor / pass_blend_op values, as they are */
    pd.src_c      = static_cast<int>(fxb_rt0(pi.src_blend));
    pd.dst_c      = static_cast<int>(fxb_rt0(pi.dest_blend));
    pd.op_c       = static_cast<int>(fxb_rt0(pi.blend_op));
    pd.src_a      = static_cast<int>(fxb_rt0(pi.src_blend_alpha));
    pd.dst_a      = static_cast<int>(fxb_rt0(pi.dest_blend_alpha));
    pd.op_a       = static_cast<int>(fxb_rt0(pi.blend_op_alpha));
    pd.wmask      = static_cast<int>(fxb_rt0(pi.color_write_mask)) & 0xF;
    pd.stencil    = pi.stencil_enable ? 1 : 0;
    pd.topo       = static_cast<int>(pi.topology);
    pd.nverts     = (int)pi.num_vertices;
    pd.vp_w       = (int)pi.viewport_width;
    pd.vp_h       = (int)pi.viewport_height;

    if (pd.stencil) { why = at + ": stencil state is not supported by the GLES runtime"; return false; }
    if (pi.vs_entry_point.empty() || pi.ps_entry_point.empty()) { why = at + ": the pass has no vertex or no pixel shader (compute passes are not supported)"; return false; }

    FxbConv vs, ps;
    if (!fxb_stage(cg, pi.vs_entry_point, false, vs, at + ": vertex shader", why)) return false;
    if (!fxb_stage(cg, pi.ps_entry_point, true, ps, at + ": pixel shader", why)) return false;
    if (!fxb_uniforms_match(vs.o, m, named, at + ": vertex shader", why)) return false;
    if (!fxb_uniforms_match(ps.o, m, named, at + ": pixel shader", why)) return false;

    pd.vs = fxb_dup_source(vs.o);
    pd.ps = fxb_dup_source(ps.o);
    if (!pd.vs || !pd.ps) { why = "out of memory"; return false; }

    if (!fxb_samplers(mod, vs.o, pd, at, why)) return false;
    if (!fxb_samplers(mod, ps.o, pd, at, why)) return false;

    /* the binding numbers are effect wide: when one lies beyond the 16 texture units, number this pass's textures 0.. instead (the converted GLSL has
     * no bindings any more, the runtime sets each sampler uniform to FxBind.unit) */
    bool wide = false;
    for (int b = 0; b < pd.nbind; b++) if (pd.bind[b].unit >= FXM_MAX_SMP) wide = true;
    if (wide) for (int b = 0; b < pd.nbind; b++) pd.bind[b].unit = b;
    return true;
}

bool fxb_techniques(const FXB_MODULE_T &mod, FxbCodegen &cg, FxModule &m, std::string &why)
{
    size_t total = 0;
    for (const auto &t : mod.techniques) total += t.passes.size();

    m.ntech = (int)mod.techniques.size();
    m.tech = (FxTechDesc *)calloc(mod.techniques.size(), sizeof(FxTechDesc));
    m.npass = (int)total;
    m.pass = total ? (FxPassDesc *)calloc(total, sizeof(FxPassDesc)) : nullptr;
    if (!m.tech || (total && !m.pass)) { why = "out of memory"; return false; }

    bool named = (m.nuni == 0);                                        /* glnames are taken from the first converted stage */
    size_t flat = 0;
    for (size_t ti = 0; ti < mod.techniques.size(); ti++) {
        const auto &tech = mod.techniques[ti];
        FxTechDesc &td = m.tech[ti];
        std::string s;
        float f = 1.f;
        if (!fxb_cpy(td.name, tech.name)) { why = "technique name too long: " + tech.name; return false; }
        if (fxb_ann_str(fxb_ann(tech.annotations, "ui_label"), s) && !s.empty()) fxb_cpy(td.label, s); else fxb_cpy(td.label, tech.name);
        td.enabled = fxb_ann_num(fxb_ann(tech.annotations, "enabled"), 0, f) ? (f != 0.f) : 1;
        td.first_pass = (int)flat;
        td.npass = (int)tech.passes.size();
        for (size_t k = 0; k < tech.passes.size(); k++, flat++)
            if (!fxb_pass(mod, cg, tech, k, m.pass[flat], m, named, why)) return false;
    }
    return true;
}

/* DEPTH is needed when a texture has the DEPTH semantic (which also covers every pass that samples it) or a uniform asks for bufready_depth */
void fxb_need_depth(FxModule &m)
{
    m.need_depth = 0;
    for (int i = 0; i < m.ntex; i++) if (m.tex[i].semantic == FXSEM_DEPTH) m.need_depth = 1;
    for (int i = 0; i < m.nuni; i++) if (m.uni[i].source == FXS_BUFREADY_DEPTH) m.need_depth = 1;
}

/* ================================================================== the whole compile */

int fxb_build(const FxCompileReq *req, FxModule *out, std::string &why)
{
    if (!req || !req->path || !*req->path) { why = "no .fx path given"; return 0; }

    /* 1. preprocess  [reshadefx API] */
    std::string text;
    if (!fxb_read_file(req->path, text)) { why = std::string("cannot read '") + req->path + "'"; return 0; }

    reshadefx::preprocessor pp;
    for (const char *const *d = req->inc; d && *d; ++d) pp.add_include_path(std::string(*d));
    fxb_defines(pp, req);
    if (!pp.append_string(std::move(text), std::string(req->path))) { why = "preprocessor: " + fxb_trim(pp.errors()); return 0; }

    /* 2. parse into the GLSL ES back-end  [reshadefx API] */
    FxbCodegen cg;
    {
        reshadefx::parser parser;
        if (!parser.parse(std::move(pp.output()), cg.base())) { why = "parser: " + fxb_trim(parser.errors()); return 0; }
    }

    /* 3. the module  [reshadefx API] */
    FXB_MODULE_T mod;
    cg.base()->write_result(mod);
    if (mod.techniques.empty()) { why = "the effect declares no technique"; return 0; }

    /* 4. textures, uniforms, then techniques / passes (the converted shaders name the uniforms and the samplers) */
    FxbModuleGuard g;
    if (!fxb_textures(mod, req, g.m, why)) return 0;
    if (!fxb_uniforms(mod, g.m, why)) return 0;
    if (!fxb_techniques(mod, cg, g.m, why)) return 0;
    fxb_need_depth(g.m);

    g.release(out);
    return 1;
}

} // namespace

/* ================================================================== the C interface of resa_fx_api.h */

extern "C" int fxb_compile(const FxCompileReq *req, FxModule *out, char *err, size_t errlen)
{
    std::string why;
    int ok = 0;
    if (!out) {
        why = "no output module";
    } else {
        memset(out, 0, sizeof *out);
#if FXB_EXCEPTIONS
        try { ok = fxb_build(req, out, why); }
        catch (const std::exception &e) { why = std::string("exception in the effect compiler: ") + e.what(); ok = 0; }
        catch (...) { why = "unknown exception in the effect compiler"; ok = 0; }
#else
        ok = fxb_build(req, out, why);
#endif
    }
    if (!ok && err && errlen) snprintf(err, errlen, "%s", why.c_str());
    return ok;
}

extern "C" void fxm_free(FxModule *m)
{
    if (!m) return;
    if (m->pass)
        for (int i = 0; i < m->npass; i++) { free(m->pass[i].vs); free(m->pass[i].ps); }
    free(m->pass);
    free(m->tex);
    free(m->uni);
    free(m->tech);
    memset(m, 0, sizeof *m);
}
