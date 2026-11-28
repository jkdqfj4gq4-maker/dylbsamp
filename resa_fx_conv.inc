/*
 * resa_fx_conv.inc  --  reshadefx GLES back-end output  ->  strict GLSL ES 3.00 (what Apple's "OpenGL ES 3.0 Metal" driver accepts).
 *
 * Plain C that also compiles as C++ (the bridge is C++, the test double and the tests are C).  No GL calls, no globals: it is a pure
 * text -> text function, so it is unit-tested on any machine (tests/test_fx_conv.c).
 *
 * INPUT  = the string returned by  reshadefx::codegen::finalize_code_for_entry_point(name)  of a codegen made with
 *              create_codegen_glsl(version = 300, gles = true, vulkan_semantics = false, debug_info = false,
 *                                  uniforms_to_spec_constants = false, enable_16bit_types = false, flip_vert_y = true)
 *          That text is exactly what DuckStation's dep/reshadefx/src/effect_codegen_glsl.cpp writes (read in full, see HANDOFF.md section 2):
 *              #version 300 es
 *              precision highp float; precision highp int; precision highp sampler2D;
 *              layout(std140, binding = 0) uniform _Globals {  <one member per line>  };          every uniform of the effect, used or not
 *              #define discard ... (vertex stage only), #define shared / atomicXxx / barrier()...     harmless
 *              layout(binding = N) uniform sampler2D name;                                           only the samplers this entry point reads
 *              ... structs, constants, functions ...
 *              layout(location = L) out|in TYPE _out_x|_in_x;                                        the varyings of the glue main()
 *              void main() { ... }
 *
 * WHAT IS NOT LEGAL IN  #version 300 es  AND WHAT THIS DOES ABOUT IT
 *   1. layout(std140, binding = 0) uniform _Globals {...}   ->  one plain  "uniform TYPE NAME;"  per member (names returned in order,
 *                                                                 so the runtime uses glUniform*; no UBO, no std140 on the GL side)
 *   2. layout(binding = N) uniform sampler2D name;          ->  "uniform sampler2D name;"  (unit N and the name are returned; the
 *                                                                 runtime does glUniform1i(loc(name), N))
 *   3. layout(location = L) on a varying (VS out / PS in)   ->  ES 3.00 allows location only on VS inputs and PS outputs and links varyings
 *                                                                 BY NAME, while the back-end names them _out_x in the VS and _in_x in the PS.
 *                                                                 So: strip the layout, rename every varying to  <prefix><L>  (fxv_0, fxv_1, ...)
 *                                                                 everywhere in the text.  Same location in both stages => same name => links.
 *   4. integer varyings must be "flat" in ES 3.00           ->  "flat" is added (the back-end only writes it for HLSL nointerpolation)
 *   5. "noperspective" does not exist in ES                 ->  dropped, a warning is recorded
 *   6. isampler2D / usampler2D have no default precision    ->  two precision statements are added after "precision highp sampler2D;"
 *   7. any other #version / #extension of the back-end      ->  "#version 300 es" is forced; GL_NV_gpu_shader5 (16-bit types) is an error
 * Things this does NOT fix (they make the driver compile fail, the runtime logs the driver message plus out->warn and skips that effect):
 *   textureGather, image load/store, sampler1D/3D, bitfield*, bitCount, findMSB/LSB, fma, precise  (ES 3.10 / 3.20 features)
 *
 * Kept as they are on purpose: layout(location = N) out ...   on PS outputs (legal in ES 3.00 and needed for MRT),
 *                              layout(location = N) in ...    on VS inputs  (legal; the back-end only writes gl_VertexID, so there are none)
 */
#ifndef RESA_FX_CONV_INC
#define RESA_FX_CONV_INC

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "resa_fx_api.h"

#define FXC_MAX_VARY 32

typedef struct { char name[FXM_NAME]; char type[40]; int array_len; } FxcUni;          /* one member of _Globals */
typedef struct { int unit; char name[FXM_NAME]; char type[40]; } FxcSmp;               /* one sampler: binding N, name, "sampler2D" */

typedef struct {
    char   *src;                 /* malloc'd, NUL-terminated converted source */
    size_t  len;
    int     nuni;  FxcUni *uni;  /* malloc'd; members of _Globals in declaration order == reshadefx module.uniforms order */
    int     nsmp;  FxcSmp smp[FXM_MAX_SMP];
    char    warn[320];           /* things that will probably not compile on ES 3.00 (see above); only worth logging if the driver says no */
} FxcOut;

typedef struct { char *p; size_t n, cap; } FxcBuf;

static int fxc_put(FxcBuf *b, const char *s, size_t n) {
    if (b->n + n + 1 > b->cap) {
        size_t nc = b->cap ? b->cap : 8192;
        while (nc < b->n + n + 1) nc *= 2;
        char *q = (char *)realloc(b->p, nc);
        if (!q) return 0;
        b->p = q; b->cap = nc;
    }
    if (n) memcpy(b->p + b->n, s, n);
    b->n += n;
    b->p[b->n] = 0;
    return 1;
}
static int fxc_puts(FxcBuf *b, const char *s) { return fxc_put(b, s, strlen(s)); }

static int fxc_fail(char *err, size_t errlen, const char *fmt, ...) {
    if (err && errlen) { va_list ap; va_start(ap, fmt); vsnprintf(err, errlen, fmt, ap); va_end(ap); }
    return 0;
}

static int fxc_isid(int c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_'; }

/* whole-word replace of from -> to in b; returns 0 on out of memory */
static int fxc_replace_word(FxcBuf *b, const char *from, const char *to) {
    size_t fl = strlen(from), tl = strlen(to), i = 0, span = 0;
    FxcBuf o = { NULL, 0, 0 };
    if (!fl) return 1;
    while (i < b->n) {
        if (b->p[i] == from[0] && i + fl <= b->n && !memcmp(b->p + i, from, fl) &&
            (i == 0 || !fxc_isid((unsigned char)b->p[i - 1])) && (i + fl == b->n || !fxc_isid((unsigned char)b->p[i + fl]))) {
            if (!fxc_put(&o, b->p + span, i - span) || !fxc_put(&o, to, tl)) { free(o.p); return 0; }
            i += fl;
            span = i;
        } else i++;
    }
    if (!fxc_put(&o, b->p + span, b->n - span)) { free(o.p); return 0; }
    free(b->p);
    *b = o;
    return 1;
}

static int fxc_word_in(const char *text, const char *w) {      /* is w present as a whole word? */
    size_t wl = strlen(w);
    for (const char *p = text; (p = strstr(p, w)) != NULL; p += wl)
        if ((p == text || !fxc_isid((unsigned char)p[-1])) && !fxc_isid((unsigned char)p[wl])) return 1;
    return 0;
}

/* "layout(...)" at s: copy what is inside the parentheses, return the pointer after ')' (NULL if it is not a layout()) */
static const char *fxc_layout(const char *s, const char *end, char *inside, size_t cap) {
    if (end - s < 8 || memcmp(s, "layout(", 7) != 0) return NULL;
    const char *q = s + 7;
    const char *c = (const char *)memchr(q, ')', (size_t)(end - q));
    size_t n;
    if (!c) return NULL;
    n = (size_t)(c - q);
    if (n >= cap) return NULL;
    memcpy(inside, q, n);
    inside[n] = 0;
    return c + 1;
}
static int fxc_layout_int(const char *inside, const char *key, int *val) {
    size_t kl = strlen(key);
    for (const char *p = inside; (p = strstr(p, key)) != NULL; p += kl) {
        const char *q = p + kl;
        if (p > inside && fxc_isid((unsigned char)p[-1])) continue;
        if (fxc_isid((unsigned char)*q)) continue;
        while (*q == ' ') q++;
        if (*q != '=') continue;
        q++;
        while (*q == ' ') q++;
        if (*q < '0' || *q > '9') continue;
        *val = atoi(q);
        return 1;
    }
    return 0;
}

typedef struct { const char *s; size_t n; } FxcTok;
static int fxc_tokens(const char *s, const char *end, FxcTok *t, int max) {
    int n = 0;
    while (s < end) {
        const char *a;
        while (s < end && (*s == ' ' || *s == '\t')) s++;
        if (s >= end) break;
        a = s;
        while (s < end && *s != ' ' && *s != '\t') s++;
        if (n < max) { t[n].s = a; t[n].n = (size_t)(s - a); }
        n++;
    }
    return n;
}
static int fxc_tok_is(const FxcTok *t, const char *w) { size_t l = strlen(w); return t->n == l && !memcmp(t->s, w, l); }
static int fxc_tok_starts(const FxcTok *t, const char *w) { size_t l = strlen(w); return t->n >= l && !memcmp(t->s, w, l); }
/* copy a token without a trailing ';' */
static int fxc_tok_copy(char *dst, size_t cap, const FxcTok *t) {
    size_t n = t->n;
    while (n > 0 && t->s[n - 1] == ';') n--;
    if (n >= cap) return 0;
    memcpy(dst, t->s, n);
    dst[n] = 0;
    return 1;
}

static void fxc_warn(FxcOut *o, const char *what) {
    size_t l = strlen(o->warn), w = strlen(what);
    if (l + w + 3 >= sizeof o->warn) return;
    if (l) { o->warn[l++] = ','; o->warn[l++] = ' '; }
    memcpy(o->warn + l, what, w + 1);
}

static void fxc_free(FxcOut *o) {
    free(o->src);
    free(o->uni);
    memset(o, 0, sizeof *o);
}

/* is_ps: 0 = vertex shader source, 1 = fragment shader source.  Returns 1 and fills *out, or 0 and a one-line reason in err. */
static int fxc_convert(const char *in, int is_ps, FxcOut *out, char *err, size_t errlen) {
    static const char *const k_prefix[] = { "fxv_", "fxw_", "fxx_", "fxy_", "fxz_", NULL };
    static const char *const k_suspect[] = { "textureGather", "textureGatherOffset", "imageLoad", "imageStore", "image2D", "sampler1D",
                                             "sampler3D", "bitfieldExtract", "bitfieldInsert", "bitCount", "findMSB", "findLSB", "fma",
                                             "precise", NULL };
    struct Vary { char from[FXM_NAME]; char to[FXM_NAME]; int loc; };
    struct Vary vary[FXC_MAX_VARY];
    FxcBuf o = { NULL, 0, 0 };
    int nvary = 0, in_ubo = 0, saw_version = 0, saw_prec = 0, cap_uni = 0, ok = 0, lineno = 0;
    const char *prefix = NULL, *p;
    char pre_loc[FXM_NAME];
    memset(out, 0, sizeof *out);
    if (!in) return fxc_fail(err, errlen, "null source");

    for (int i = 0; k_prefix[i]; i++)
        if (!strstr(in, k_prefix[i])) { prefix = k_prefix[i]; break; }
    if (!prefix) return fxc_fail(err, errlen, "source already uses every fxv_/fxw_/... name the converter could rename varyings to");

    p = in;
    while (*p) {
        const char *e = strchr(p, '\n');
        const char *lend = e ? e : p + strlen(p);
        const char *ce = lend, *ls = p;
        size_t ll;
        lineno++;
        if (ce > p && ce[-1] == '\r') ce--;
        while (ls < ce && (*ls == ' ' || *ls == '\t')) ls++;
        ll = (size_t)(ce - ls);
        p = e ? e + 1 : lend;

        if (ll >= 8 && !memcmp(ls, "#version", 8)) {
            if (saw_version) { if (!fxc_put(&o, ls, ll) || !fxc_puts(&o, "\n")) goto oom; continue; }
            if (!memchr(ls, 'e', ll) || !strstr(ls, " es")) { fxc_fail(err, errlen, "line %d: '#version' is not an ES version - the codegen must be made with gles = true", lineno); goto done; }
            saw_version = 1;
            if (!fxc_puts(&o, "#version 300 es\n")) goto oom;
            continue;
        }
        if (ll >= 10 && !memcmp(ls, "#extension", 10)) {
            if (strstr(ls, "GL_NV_gpu_shader5")) { fxc_fail(err, errlen, "line %d: 16-bit types were enabled (GL_NV_gpu_shader5) - make the codegen with enable_16bit_types = false", lineno); goto done; }
            if (!fxc_put(&o, ls, ll) || !fxc_puts(&o, "\n")) goto oom;
            continue;
        }

        if (in_ubo) {                                               /* members of _Globals: one per line, "\tTYPE NAME;" or "\tTYPE NAME[N];" */
            if (ll >= 2 && ls[0] == '}' && ls[1] == ';') { in_ubo = 0; continue; }
            if (ll == 0) continue;
            if (ls[0] == '#') { if (!fxc_put(&o, ls, ll) || !fxc_puts(&o, "\n")) goto oom; continue; }
            {
                char m[256];
                FxcTok t[8];
                int nt;
                char *br;
                if (ll >= sizeof m) { fxc_fail(err, errlen, "line %d: uniform member line too long", lineno); goto done; }
                memcpy(m, ls, ll); m[ll] = 0;
                while (ll > 0 && (m[ll - 1] == ';' || m[ll - 1] == ' ')) m[--ll] = 0;
                {
                    int al = 0;
                    br = strchr(m, '[');
                    if (br) { al = atoi(br + 1); *br = 0; }
                    nt = fxc_tokens(m, m + strlen(m), t, 8);
                    if (nt < 2 || nt > 8) { fxc_fail(err, errlen, "line %d: cannot read the uniform member '%s'", lineno, m); goto done; }
                    if (out->nuni == cap_uni) {
                        int nc = cap_uni ? cap_uni * 2 : 64;
                        FxcUni *q = (FxcUni *)realloc(out->uni, (size_t)nc * sizeof *q);
                        if (!q) goto oom;
                        out->uni = q; cap_uni = nc;
                    }
                    {
                        FxcUni *u = &out->uni[out->nuni];
                        memset(u, 0, sizeof *u);
                        if (!fxc_tok_copy(u->name, sizeof u->name, &t[nt - 1]) || !fxc_tok_copy(u->type, sizeof u->type, &t[nt - 2])) {
                            fxc_fail(err, errlen, "line %d: uniform name or type too long", lineno); goto done;
                        }
                        u->array_len = al;
                        out->nuni++;
                    }
                }
                if (!fxc_puts(&o, "uniform ") || !fxc_puts(&o, ls)) goto oom;
                if (ls[ll > 0 ? 0 : 0] == 0 || o.p[o.n - 1] != '\n') { if (!fxc_puts(&o, "\n")) goto oom; }
            }
            continue;
        }

        if (ll >= 7 && !memcmp(ls, "layout(", 7)) {
            char inside[128];
            const char *rest = fxc_layout(ls, ce, inside, sizeof inside);
            int v = 0;
            if (rest) {
                if (strstr(inside, "std140") && memchr(rest, '_', (size_t)(ce - rest)) && strstr(rest, "_Globals")) { in_ubo = 1; continue; }
                if (fxc_layout_int(inside, "binding", &v)) {
                    FxcTok t[8];
                    int nt = fxc_tokens(rest, ce, t, 8);
                    if (strchr(inside, ',')) { fxc_fail(err, errlen, "line %d: storage image / compute resource ('layout(%s)') - ES 3.00 has no image load/store", lineno, inside); goto done; }
                    if (nt != 3 || !fxc_tok_is(&t[0], "uniform")) { fxc_fail(err, errlen, "line %d: cannot read the sampler declaration", lineno); goto done; }
                    if (out->nsmp >= FXM_MAX_SMP) { fxc_fail(err, errlen, "line %d: more than %d samplers", lineno, FXM_MAX_SMP); goto done; }
                    {
                        FxcSmp *s = &out->smp[out->nsmp];
                        memset(s, 0, sizeof *s);
                        s->unit = v;
                        if (!fxc_tok_copy(s->type, sizeof s->type, &t[1]) || !fxc_tok_copy(s->name, sizeof s->name, &t[2])) { fxc_fail(err, errlen, "line %d: sampler name too long", lineno); goto done; }
                        out->nsmp++;
                        if (!fxc_puts(&o, "uniform ") || !fxc_puts(&o, s->type) || !fxc_puts(&o, " ") || !fxc_puts(&o, s->name) || !fxc_puts(&o, ";\n")) goto oom;
                    }
                    continue;
                }
                if (fxc_layout_int(inside, "location", &v)) {
                    FxcTok t[12];
                    int nt = fxc_tokens(rest, ce, t, 12), io = -1;
                    for (int i = 0; i < nt && i < 12; i++) if (fxc_tok_is(&t[i], "in") || fxc_tok_is(&t[i], "out")) { io = i; break; }
                    if (io < 0 || nt > 12 || nt - io < 3) { fxc_fail(err, errlen, "line %d: cannot read the interface declaration", lineno); goto done; }
                    {
                        int is_out = fxc_tok_is(&t[io], "out");
                        int varying = is_ps ? !is_out : is_out;         /* VS out / PS in */
                        if (!varying) {                                  /* VS input or PS output: legal in ES 3.00, keep the line as it is */
                            if (!fxc_put(&o, ls, ll) || !fxc_puts(&o, "\n")) goto oom;
                            continue;
                        }
                        {
                            struct Vary *vy;
                            int has_flat = 0, is_int;
                            char nm[FXM_NAME];
                            if (nvary >= FXC_MAX_VARY) { fxc_fail(err, errlen, "line %d: more than %d varyings", lineno, FXC_MAX_VARY); goto done; }
                            for (int i = 0; i < nvary; i++) if (vary[i].loc == v) { fxc_fail(err, errlen, "line %d: two varyings at location %d", lineno, v); goto done; }
                            if (!fxc_tok_copy(nm, sizeof nm, &t[nt - 1])) { fxc_fail(err, errlen, "line %d: varying name too long", lineno); goto done; }
                            vy = &vary[nvary++];
                            memset(vy, 0, sizeof *vy);
                            vy->loc = v;
                            snprintf(vy->from, sizeof vy->from, "%s", nm);
                            snprintf(pre_loc, sizeof pre_loc, "%s%d", prefix, v);
                            snprintf(vy->to, sizeof vy->to, "%s", pre_loc);
                            for (int i = 0; i < io; i++) if (fxc_tok_is(&t[i], "flat")) has_flat = 1;
                            is_int = fxc_tok_starts(&t[io + 1], "int") || fxc_tok_starts(&t[io + 1], "uint") || fxc_tok_starts(&t[io + 1], "ivec") || fxc_tok_starts(&t[io + 1], "uvec");
                            if (is_int && !has_flat) { if (!fxc_puts(&o, "flat ")) goto oom; }
                            for (int i = 0; i < io; i++) {
                                if (fxc_tok_is(&t[i], "noperspective")) { fxc_warn(out, "noperspective dropped"); continue; }
                                if (!fxc_put(&o, t[i].s, t[i].n) || !fxc_puts(&o, " ")) goto oom;
                            }
                            if (!fxc_put(&o, t[io].s, t[io].n) || !fxc_puts(&o, " ")) goto oom;
                            for (int i = io + 1; i < nt - 1; i++) if (!fxc_put(&o, t[i].s, t[i].n) || !fxc_puts(&o, " ")) goto oom;
                            if (!fxc_puts(&o, nm) || !fxc_puts(&o, ";\n")) goto oom;
                        }
                    }
                    continue;
                }
            }
        }

        if (ll == 26 && !memcmp(ls, "precision highp sampler2D;", 26)) {       /* ES 3.00 has no default precision for the integer samplers */
            saw_prec = 1;
            if (!fxc_put(&o, ls, ll) || !fxc_puts(&o, "\nprecision highp isampler2D;\nprecision highp usampler2D;\n")) goto oom;
            continue;
        }

        if (!fxc_put(&o, p == in ? "" : ls - 0, 0)) goto oom;                    /* (no-op; keeps the structure of the branches symmetric) */
        if (!fxc_put(&o, ls, ll) || !fxc_puts(&o, "\n")) goto oom;
    }

    if (!saw_version) { fxc_fail(err, errlen, "no #version line - this is not reshadefx output"); goto done; }
    if (!saw_prec)    { fxc_fail(err, errlen, "no 'precision highp sampler2D;' line - the codegen must be made with gles = true"); goto done; }
    for (int i = 0; i < nvary; i++)
        if (!fxc_replace_word(&o, vary[i].from, vary[i].to)) goto oom;
    for (int i = 0; k_suspect[i]; i++) if (fxc_word_in(o.p, k_suspect[i])) fxc_warn(out, k_suspect[i]);

    out->src = o.p;
    out->len = o.n;
    o.p = NULL;
    ok = 1;
    goto done;
oom:
    fxc_fail(err, errlen, "out of memory");
done:
    free(o.p);
    if (!ok) { free(out->uni); out->uni = NULL; out->nuni = 0; out->nsmp = 0; }
    return ok;
}

#endif /* RESA_FX_CONV_INC */
