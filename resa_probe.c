/*
 * resa_probe.c  --  ReSA (iOS) renderer recon probe, v2
 *
 * Pure C, no Foundation / OpenGLES link dependency (GL is resolved at runtime).
 * Build: see build.yml. Inject like any other dylib, play ~3 minutes in OFFLINE mode.
 *
 * What it tells us (all written to  <app>/Documents/resa_probe.txt):
 *   1. loaded?        phone buzzes once ~3s after launch, file is created at load.
 *   2. GL reachable?  buzzes twice when the first hooked GL call happens.
 *   3. resolve seen?  buzzes three times when glBlitFramebuffer / MSAA resolve is called.
 *   4. which images import which GL / CoreVideo / Metal symbols (so we know what fishhook can reach)
 *   5. frame structure: FBO binds, attachments, discards, blits/resolves, draws between them,
 *      plus READ/DRAW framebuffer depth attachment type/size/format at every blit/resolve/discard.
 *
 * It never draws and never changes GL state (queries only; renderbuffer binding is restored).
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <dlfcn.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef RESA_TEST
#  include "machostub.h"
#else
#  include <mach/mach.h>
#  include <mach-o/dyld.h>
#  include <mach-o/loader.h>
#  include <mach-o/nlist.h>
#  include <sys/sysctl.h>
#  include "fishhook.h"
#endif

typedef unsigned int GLenum;
typedef unsigned int GLuint;
typedef unsigned int GLbitfield;
typedef int GLint;
typedef int GLsizei;

#define PROBE_VERSION "v2"
#define SETUP_CAP 800
#define WIN_CAP   1500
#define SNAP_CAP  40
#define SNAP_LEN  640
#define FIRST_CAP 4
#define NWIN      4
#define NCALLER   6

#ifdef RESA_TEST
static const double WIN_MIN_START[NWIN] = {1.0, 2.5, 4.0, 5.5};
#  define WIN_GAP     0.4
#  define WIN_WAIT    1.5
#  define WIN_MAXLEN  0.5
#else
static const double WIN_MIN_START[NWIN] = {10.0, 35.0, 70.0, 120.0};
#  define WIN_GAP     15.0   /* min seconds between windows               */
#  define WIN_WAIT    30.0   /* if no resolve/blit is ever seen, open anyway after this long */
#  define WIN_MAXLEN  1.0    /* seconds a window stays open               */
#endif

/* ------------------------------------------------------------------ utils */
static uint64_t now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)(ts.tv_nsec / 1000);
}
static uint64_t g_t0;
static double since_s(void) { return (double)(now_us() - g_t0) / 1e6; }

static uint32_t cur_tid(void) {
#ifdef RESA_TEST
    return (uint32_t)(((uintptr_t)pthread_self()) >> 8) & 0xffff;
#else
    return (uint32_t)pthread_mach_thread_np(pthread_self());
#endif
}

static int  g_fd = -1;
static char g_dir[600];

static void write_all(const char *s, size_t n) {
    if (g_fd < 0) return;
    while (n > 0) {
        ssize_t w = write(g_fd, s, n);
        if (w < 0) { if (errno == EINTR) continue; break; }
        if (w == 0) break;
        s += w; n -= (size_t)w;
    }
}
static void logf_(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void logf_(const char *fmt, ...) {
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if ((size_t)n >= sizeof buf) n = (int)sizeof buf - 1;
    write_all(buf, (size_t)n);
}
static void app(char *dst, size_t cap, const char *fmt, ...) __attribute__((format(printf, 3, 4)));
static void app(char *dst, size_t cap, const char *fmt, ...) {
    size_t l = strlen(dst);
    if (l + 1 >= cap) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(dst + l, cap - l, fmt, ap);
    va_end(ap);
}

static const char *ename(unsigned e, char *b) {
    switch (e) {
    case 0x8D40: return "FRAMEBUFFER";   case 0x8CA8: return "READ_FB";      case 0x8CA9: return "DRAW_FB";
    case 0x8D41: return "RENDERBUFFER";  case 0x0DE1: return "TEXTURE_2D";
    case 0x8CE0: return "COLOR0";        case 0x8D00: return "DEPTH_ATT";    case 0x8D20: return "STENCIL_ATT";
    case 0x821A: return "DEPTH_STENCIL_ATT";
    case 0x1800: return "COLOR";         case 0x1801: return "DEPTH";        case 0x1802: return "STENCIL";
    case 0x81A5: return "DEPTH16";       case 0x81A6: return "DEPTH24";      case 0x81A7: return "DEPTH32_OES";
    case 0x8CAC: return "DEPTH32F";      case 0x88F0: return "DEPTH24_STENCIL8";
    case 0x8CAD: return "DEPTH32F_STENCIL8"; case 0x84F9: return "DEPTH_STENCIL"; case 0x8D48: return "STENCIL8";
    case 0x8058: return "RGBA8";         case 0x8D62: return "RGB565";       case 0x8057: return "RGB5_A1";
    case 0x8056: return "RGBA4";         case 0x93A1: return "BGRA8";
    case 0x1908: return "RGBA";          case 0x1907: return "RGB";          case 0x80E1: return "BGRA";
    case 0x1902: return "DEPTH_COMPONENT";
    case 0x1401: return "UBYTE";         case 0x1403: return "USHORT";       case 0x1405: return "UINT";
    case 0x84FA: return "UINT_24_8";     case 0x1406: return "FLOAT";
    case 0x2600: return "NEAREST";       case 0x2601: return "LINEAR";
    default: break;
    }
    snprintf(b, 16, "0x%x", e);
    return b;
}
static const char *maskname(unsigned m, char *b) {
    b[0] = 0;
    if (m & 0x4000) strcat(b, "C");
    if (m & 0x0100) strcat(b, "D");
    if (m & 0x0400) strcat(b, "S");
    if (!b[0]) snprintf(b, 16, "0x%x", m);
    return b;
}
static int is_depthish(unsigned f) {
    switch (f) {
    case 0x81A5: case 0x81A6: case 0x81A7: case 0x8CAC: case 0x88F0: case 0x8CAD: case 0x84F9: case 0x1902: return 1;
    default: return 0;
    }
}

/* ------------------------------------------------------ hooked-function list */
#define FN_LIST(X) \
    X(glBindFramebuffer) X(glFramebufferRenderbuffer) X(glFramebufferTexture2D) \
    X(glRenderbufferStorage) X(glRenderbufferStorageMultisample) X(glRenderbufferStorageMultisampleAPPLE) \
    X(glDiscardFramebufferEXT) X(glInvalidateFramebuffer) X(glResolveMultisampleFramebufferAPPLE) \
    X(glBlitFramebuffer) X(glClear) X(glFlush) X(glFinish) \
    X(glTexImage2D) X(glTexStorage2D) X(glCopyTexImage2D) X(glCopyTexSubImage2D) \
    X(glDrawArrays) X(glDrawElements) X(glDrawArraysInstanced) X(glDrawElementsInstanced) X(glDrawRangeElements)

enum {
#define X(n) F_##n,
    FN_LIST(X)
#undef X
    NF
};
static const char *const fnames[NF] = {
#define X(n) #n,
    FN_LIST(X)
#undef X
};
static atomic_ullong calls[NF];
static void *fh_orig[NF];   /* written by fishhook: original pointer (non-NULL => fishhook found an import) */
static void *fb_orig[NF];   /* dlsym fallback */
static int   g_rebind_rc = -999;

static void noop_fn(void) {}
static void *orig_of(int f, const char *name) {
    void *p = fh_orig[f];
    if (p) return p;
    p = fb_orig[f];
    if (p) return p;
    p = dlsym(RTLD_DEFAULT, name);
    if (!p) p = (void *)noop_fn;
    fb_orig[f] = p;
    return p;
}
#define O(n) ((fn_##n)orig_of(F_##n, #n))

/* ------------------------------------------------------ GL query functions (resolved at runtime) */
typedef void (*fn_getint)(GLenum, GLint *);
typedef const unsigned char *(*fn_getstr)(GLenum);
typedef void (*fn_getfbap)(GLenum, GLenum, GLenum, GLint *);
typedef void (*fn_getrbp)(GLenum, GLenum, GLint *);
typedef void (*fn_bindrb)(GLenum, GLuint);
typedef GLenum (*fn_geterr)(void);
static fn_getint  p_getint;
static fn_getstr  p_getstr;
static fn_getfbap p_getfbap;
static fn_getrbp  p_getrbp;
static fn_bindrb  p_bindrb;
static fn_geterr  p_geterr;
static atomic_int g_gl_resolved;

static void resolve_gl(void) {
    if (atomic_load(&g_gl_resolved)) return;
    p_getint  = (fn_getint) dlsym(RTLD_DEFAULT, "glGetIntegerv");
    p_getstr  = (fn_getstr) dlsym(RTLD_DEFAULT, "glGetString");
    p_getfbap = (fn_getfbap)dlsym(RTLD_DEFAULT, "glGetFramebufferAttachmentParameteriv");
    p_getrbp  = (fn_getrbp) dlsym(RTLD_DEFAULT, "glGetRenderbufferParameteriv");
    p_bindrb  = (fn_bindrb) dlsym(RTLD_DEFAULT, "glBindRenderbuffer");
    p_geterr  = (fn_geterr) dlsym(RTLD_DEFAULT, "glGetError");
    atomic_store(&g_gl_resolved, 1);
}

/* GL info, captured once from the first hook call that has a current context */
static char g_gl_version[96], g_gl_renderer[96], g_gl_vendor[64], g_gl_ext[600];
static atomic_int g_gl_info_done, g_gl_info_tries;
static void gl_info_try(void) {
    if (atomic_load(&g_gl_info_done)) return;
    if (atomic_fetch_add(&g_gl_info_tries, 1) > 30) return;
    resolve_gl();
    if (!p_getstr) return;
    const char *v = (const char *)p_getstr(0x1F02);
    if (!v) return;                                   /* no context current on this thread */
    if (atomic_exchange(&g_gl_info_done, 1)) return;
    snprintf(g_gl_version, sizeof g_gl_version, "%s", v);
    const char *r  = (const char *)p_getstr(0x1F01);
    const char *vd = (const char *)p_getstr(0x1F00);
    const char *x  = (const char *)p_getstr(0x1F03);
    snprintf(g_gl_renderer, sizeof g_gl_renderer, "%s", r ? r : "?");
    snprintf(g_gl_vendor, sizeof g_gl_vendor, "%s", vd ? vd : "?");
    static const char *const kw[] = {
        "GL_APPLE_framebuffer_multisample", "GL_EXT_discard_framebuffer", "GL_OES_depth_texture",
        "GL_OES_packed_depth_stencil", "GL_OES_depth24", "GL_OES_depth32", "GL_EXT_shadow_samplers",
        "GL_OES_standard_derivatives", "GL_EXT_color_buffer_half_float", "GL_OES_texture_half_float", NULL };
    g_gl_ext[0] = 0;
    for (int i = 0; kw[i]; i++)
        if (x && strstr(x, kw[i])) app(g_gl_ext, sizeof g_gl_ext, "%s ", kw[i] + 3);
}

/* ------------------------------------------------------ caller bookkeeping */
static void *c_base[NF][NCALLER];
static char  c_name[NF][NCALLER][40];
static int   c_n[NF];
static atomic_flag c_lock = ATOMIC_FLAG_INIT;
static atomic_int  g_gl_seen;
static atomic_uint g_blit_n;       /* blit + resolve calls seen */

static void note_caller(int f, void *ra) {
    unsigned long long c = atomic_fetch_add_explicit(&calls[f], 1ull, memory_order_relaxed);
    if (c == 0) atomic_store_explicit(&g_gl_seen, 1, memory_order_relaxed);
    if (c >= 32 && (c & 1023ull) != 0) return;       /* sample: first 32 calls, then every 1024th */
    if (!atomic_load_explicit(&g_gl_info_done, memory_order_relaxed)) gl_info_try();
    Dl_info di;
    if (!ra || !dladdr(ra, &di) || !di.dli_fbase) return;
    while (atomic_flag_test_and_set_explicit(&c_lock, memory_order_acquire)) { }
    int known = 0;
    for (int i = 0; i < c_n[f]; i++) if (c_base[f][i] == di.dli_fbase) { known = 1; break; }
    if (!known && c_n[f] < NCALLER) {
        const char *nm = di.dli_fname ? di.dli_fname : "?";
        const char *sl = strrchr(nm, '/');
        if (sl) nm = sl + 1;
        c_base[f][c_n[f]] = di.dli_fbase;
        snprintf(c_name[f][c_n[f]], sizeof c_name[f][0], "%s", nm);
        c_n[f]++;
    }
    atomic_flag_clear_explicit(&c_lock, memory_order_release);
}

/* ------------------------------------------------------ event logs */
enum { E_BIND = 1, E_FBRB, E_FBTEX, E_RBSTORE, E_DISCARD, E_INVAL, E_RESOLVE, E_BLIT,
       E_CLEAR, E_FLUSH, E_FINISH, E_TEXDEPTH, E_COPYTEX, E_NTYPES };

typedef struct { uint32_t t_us, tid, draws; uint16_t type; int16_t snap; int32_t v[10]; } Ev;

static Ev setup_log[SETUP_CAP];  static atomic_uint setup_n;
static Ev win_log[WIN_CAP];      static atomic_uint win_n;
static atomic_int  win_active;
static atomic_uint draws_since;

static char g_snaps[SNAP_CAP][SNAP_LEN];  static atomic_uint g_snap_n;
static char g_first[FIRST_CAP][SNAP_LEN]; static int g_first_f[FIRST_CAP]; static uint32_t g_first_t[FIRST_CAP];
static atomic_uint g_first_n; static volatile int g_first_ok[FIRST_CAP];

static int is_setup_type(int t) { return t == E_FBRB || t == E_FBTEX || t == E_RBSTORE || t == E_TEXDEPTH; }
static int logging_on(void) {
    return atomic_load_explicit(&win_active, memory_order_relaxed) ||
           atomic_load_explicit(&setup_n, memory_order_relaxed) < SETUP_CAP;
}
static void ev(int type, int snap, const int *v, int nv) {
    Ev x;
    memset(&x, 0, sizeof x);
    x.t_us  = (uint32_t)(now_us() - g_t0);
    x.tid   = cur_tid();
    x.draws = atomic_exchange_explicit(&draws_since, 0u, memory_order_relaxed);
    x.type  = (uint16_t)type;
    x.snap  = (int16_t)snap;
    for (int i = 0; i < nv && i < 10; i++) x.v[i] = v[i];
    if (is_setup_type(type) && atomic_load(&setup_n) < SETUP_CAP) {
        unsigned i = atomic_fetch_add(&setup_n, 1u);
        if (i < SETUP_CAP) setup_log[i] = x;
    }
    if (atomic_load(&win_active)) {
        unsigned i = atomic_fetch_add(&win_n, 1u);
        if (i < WIN_CAP) win_log[i] = x;
    }
}

/* depth textures we saw being created (ES has no size query for textures) */
static struct { int tex, w, h, fmt; } g_dtex[64];
static atomic_int g_ndtex;
static void note_dtex(int tex, int w, int h, int fmt) {
    int i = atomic_fetch_add(&g_ndtex, 1);
    if (i < 64) { g_dtex[i].tex = tex; g_dtex[i].w = w; g_dtex[i].h = h; g_dtex[i].fmt = fmt; }
}
static int find_dtex(int tex) {
    int n = atomic_load(&g_ndtex);
    if (n > 64) n = 64;
    for (int i = n - 1; i >= 0; i--) if (g_dtex[i].tex == tex) return i;
    return -1;
}

/* ------------------------------------------------------ framebuffer snapshot (queries only) */
static void rb_desc(char *o, size_t cap, int name) {
    GLint prev = 0, w = 0, h = 0, f = 0, s = 0;
    char b[16];
    if (!p_bindrb || !p_getrbp) { snprintf(o, cap, "RB%d", name); return; }
    p_getint(0x8CA7, &prev);
    p_bindrb(0x8D41, (GLuint)name);
    p_getrbp(0x8D41, 0x8D42, &w);
    p_getrbp(0x8D41, 0x8D43, &h);
    p_getrbp(0x8D41, 0x8D44, &f);
    p_getrbp(0x8D41, 0x8CAB, &s);
    p_bindrb(0x8D41, (GLuint)prev);
    snprintf(o, cap, "RB%d(%dx%d %s s=%d)", name, w, h, ename((unsigned)f, b), s);
}
static void att_desc(char *o, size_t cap, GLenum target, GLenum att) {
    GLint type = -1, name = 0;
    char b[16];
    p_getfbap(target, att, 0x8CD0, &type);
    if (type == 0)  { snprintf(o, cap, "none"); return; }
    if (type < 0)   { snprintf(o, cap, "?"); return; }
    p_getfbap(target, att, 0x8CD1, &name);
    if (type == 0x8D41) {
        rb_desc(o, cap, name);
    } else if (type == 0x1702) {
        int k = find_dtex(name);
        if (k >= 0) snprintf(o, cap, "TEX%d(%dx%d %s)", name, g_dtex[k].w, g_dtex[k].h, ename((unsigned)g_dtex[k].fmt, b));
        else        snprintf(o, cap, "TEX%d", name);
    } else {
        snprintf(o, cap, "type0x%x:%d", type, name);
    }
}
static void fb_desc(char *o, size_t cap, GLenum target, GLenum binding_enum) {
    GLint fbo = -1;
    p_getint(binding_enum, &fbo);
    if (fbo <= 0) { snprintf(o, cap, "fbo=%d", fbo); return; }
    char c0[120], d[120], s[120];
    att_desc(c0, sizeof c0, target, 0x8CE0);
    att_desc(d,  sizeof d,  target, 0x8D00);
    att_desc(s,  sizeof s,  target, 0x8D20);
    snprintf(o, cap, "fbo=%d{c0=%s depth=%s stencil=%s}", fbo, c0, d, s);
}
static void snapshot_state(char *out, size_t cap) {
    resolve_gl();
    if (!p_getint || !p_getfbap) { snprintf(out, cap, "(GL query functions not found)"); return; }
    GLenum pre = p_geterr ? p_geterr() : 0;
    char rd[300], dr[300];
    GLint vp[4] = {-1, -1, -1, -1};
    fb_desc(rd, sizeof rd, 0x8CA8, 0x8CAA);
    fb_desc(dr, sizeof dr, 0x8CA9, 0x8CA6);
    p_getint(0x0BA2, vp);
    snprintf(out, cap, "READ %s | DRAW %s | vp=%d,%d,%d,%d", rd, dr, vp[0], vp[1], vp[2], vp[3]);
    if (pre) app(out, cap, " | engineErrBefore=0x%x", pre);
    if (p_geterr) { int guard = 0; GLenum e; while ((e = p_geterr()) != 0 && guard++ < 8) app(out, cap, " | oursErr=0x%x", e); }
}
static int snap_if_wanted(int f) {
    int want_win   = atomic_load(&win_active) && atomic_load(&g_snap_n) < SNAP_CAP;
    int want_first = atomic_load(&g_first_n) < FIRST_CAP;
    if (!want_win && !want_first) return -1;
    char tmp[SNAP_LEN];
    snapshot_state(tmp, sizeof tmp);
    int idx = -1;
    if (want_win) {
        unsigned i = atomic_fetch_add(&g_snap_n, 1u);
        if (i < SNAP_CAP) { memcpy(g_snaps[i], tmp, sizeof tmp); idx = (int)i; }
    }
    if (want_first) {
        unsigned j = atomic_fetch_add(&g_first_n, 1u);
        if (j < FIRST_CAP) {
            memcpy(g_first[j], tmp, sizeof tmp);
            g_first_f[j] = f;
            g_first_t[j] = (uint32_t)(now_us() - g_t0);
            g_first_ok[j] = 1;
        }
    }
    return idx;
}
static int query_fbo(GLenum target) {
    GLint v = -1;
    resolve_gl();
    if (p_getint) p_getint(target == 0x8CA8 ? 0x8CAA : 0x8CA6, &v);
    return (int)v;
}
static int query_int(GLenum pname) {
    GLint v = -1;
    resolve_gl();
    if (p_getint) p_getint(pname, &v);
    return (int)v;
}

/* ------------------------------------------------------ hooks */
#define HOOK_ENTER(n) note_caller(F_##n, __builtin_return_address(0))
#define DRAW_ENTER(n) do { HOOK_ENTER(n); atomic_fetch_add_explicit(&draws_since, 1u, memory_order_relaxed); } while (0)

typedef void (*fn_glBindFramebuffer)(GLenum, GLuint);
static void h_glBindFramebuffer(GLenum t, GLuint fb) {
    HOOK_ENTER(glBindFramebuffer);
    if (atomic_load_explicit(&win_active, memory_order_relaxed)) ev(E_BIND, -1, (int[]){(int)t, (int)fb}, 2);
    O(glBindFramebuffer)(t, fb);
}
typedef void (*fn_glFramebufferRenderbuffer)(GLenum, GLenum, GLenum, GLuint);
static void h_glFramebufferRenderbuffer(GLenum t, GLenum a, GLenum rt, GLuint rb) {
    HOOK_ENTER(glFramebufferRenderbuffer);
    if (logging_on()) ev(E_FBRB, -1, (int[]){(int)t, (int)a, (int)rt, (int)rb, query_fbo(t)}, 5);
    O(glFramebufferRenderbuffer)(t, a, rt, rb);
}
typedef void (*fn_glFramebufferTexture2D)(GLenum, GLenum, GLenum, GLuint, GLint);
static void h_glFramebufferTexture2D(GLenum t, GLenum a, GLenum tt, GLuint tex, GLint lvl) {
    HOOK_ENTER(glFramebufferTexture2D);
    if (logging_on()) ev(E_FBTEX, -1, (int[]){(int)t, (int)a, (int)tt, (int)tex, (int)lvl, query_fbo(t)}, 6);
    O(glFramebufferTexture2D)(t, a, tt, tex, lvl);
}
typedef void (*fn_glRenderbufferStorage)(GLenum, GLenum, GLsizei, GLsizei);
static void h_glRenderbufferStorage(GLenum t, GLenum fmt, GLsizei w, GLsizei h) {
    HOOK_ENTER(glRenderbufferStorage);
    if (logging_on()) ev(E_RBSTORE, -1, (int[]){(int)t, -1, (int)fmt, w, h, query_int(0x8CA7)}, 6);
    O(glRenderbufferStorage)(t, fmt, w, h);
}
typedef void (*fn_glRenderbufferStorageMultisample)(GLenum, GLsizei, GLenum, GLsizei, GLsizei);
static void h_glRenderbufferStorageMultisample(GLenum t, GLsizei s, GLenum fmt, GLsizei w, GLsizei h) {
    HOOK_ENTER(glRenderbufferStorageMultisample);
    if (logging_on()) ev(E_RBSTORE, -1, (int[]){(int)t, s, (int)fmt, w, h, query_int(0x8CA7)}, 6);
    O(glRenderbufferStorageMultisample)(t, s, fmt, w, h);
}
typedef void (*fn_glRenderbufferStorageMultisampleAPPLE)(GLenum, GLsizei, GLenum, GLsizei, GLsizei);
static void h_glRenderbufferStorageMultisampleAPPLE(GLenum t, GLsizei s, GLenum fmt, GLsizei w, GLsizei h) {
    HOOK_ENTER(glRenderbufferStorageMultisampleAPPLE);
    if (logging_on()) ev(E_RBSTORE, -1, (int[]){(int)t, s, (int)fmt, w, h, query_int(0x8CA7)}, 6);
    O(glRenderbufferStorageMultisampleAPPLE)(t, s, fmt, w, h);
}
typedef void (*fn_glDiscardFramebufferEXT)(GLenum, GLsizei, const GLenum *);
static void h_glDiscardFramebufferEXT(GLenum t, GLsizei n, const GLenum *att) {
    HOOK_ENTER(glDiscardFramebufferEXT);
    if (atomic_load_explicit(&win_active, memory_order_relaxed) || atomic_load(&g_first_n) < FIRST_CAP) {
        int a[3] = {-1, -1, -1};
        for (int i = 0; att && i < n && i < 3; i++) a[i] = (int)att[i];
        int sn = snap_if_wanted(F_glDiscardFramebufferEXT);
        ev(E_DISCARD, sn, (int[]){(int)t, n, a[0], a[1], a[2]}, 5);
    }
    O(glDiscardFramebufferEXT)(t, n, att);
}
typedef void (*fn_glInvalidateFramebuffer)(GLenum, GLsizei, const GLenum *);
static void h_glInvalidateFramebuffer(GLenum t, GLsizei n, const GLenum *att) {
    HOOK_ENTER(glInvalidateFramebuffer);
    if (atomic_load_explicit(&win_active, memory_order_relaxed) || atomic_load(&g_first_n) < FIRST_CAP) {
        int a[3] = {-1, -1, -1};
        for (int i = 0; att && i < n && i < 3; i++) a[i] = (int)att[i];
        int sn = snap_if_wanted(F_glInvalidateFramebuffer);
        ev(E_INVAL, sn, (int[]){(int)t, n, a[0], a[1], a[2]}, 5);
    }
    O(glInvalidateFramebuffer)(t, n, att);
}
typedef void (*fn_glResolveMultisampleFramebufferAPPLE)(void);
static void h_glResolveMultisampleFramebufferAPPLE(void) {
    HOOK_ENTER(glResolveMultisampleFramebufferAPPLE);
    atomic_fetch_add(&g_blit_n, 1u);
    if (atomic_load_explicit(&win_active, memory_order_relaxed) || atomic_load(&g_first_n) < FIRST_CAP) {
        int sn = snap_if_wanted(F_glResolveMultisampleFramebufferAPPLE);
        ev(E_RESOLVE, sn, (int[]){0}, 0);
    }
    O(glResolveMultisampleFramebufferAPPLE)();
}
typedef void (*fn_glBlitFramebuffer)(GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLbitfield, GLenum);
static void h_glBlitFramebuffer(GLint sx0, GLint sy0, GLint sx1, GLint sy1, GLint dx0, GLint dy0, GLint dx1, GLint dy1,
                                GLbitfield mask, GLenum filter) {
    HOOK_ENTER(glBlitFramebuffer);
    atomic_fetch_add(&g_blit_n, 1u);
    if (atomic_load_explicit(&win_active, memory_order_relaxed) || atomic_load(&g_first_n) < FIRST_CAP) {
        int sn = snap_if_wanted(F_glBlitFramebuffer);
        ev(E_BLIT, sn, (int[]){sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1, (int)mask, (int)filter}, 10);
    }
    O(glBlitFramebuffer)(sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1, mask, filter);
}
typedef void (*fn_glClear)(GLbitfield);
static void h_glClear(GLbitfield m) {
    HOOK_ENTER(glClear);
    if (atomic_load_explicit(&win_active, memory_order_relaxed)) ev(E_CLEAR, -1, (int[]){(int)m}, 1);
    O(glClear)(m);
}
typedef void (*fn_glFlush)(void);
static void h_glFlush(void) {
    HOOK_ENTER(glFlush);
    if (atomic_load_explicit(&win_active, memory_order_relaxed)) ev(E_FLUSH, -1, (int[]){0}, 0);
    O(glFlush)();
}
typedef void (*fn_glFinish)(void);
static void h_glFinish(void) {
    HOOK_ENTER(glFinish);
    if (atomic_load_explicit(&win_active, memory_order_relaxed)) ev(E_FINISH, -1, (int[]){0}, 0);
    O(glFinish)();
}
typedef void (*fn_glTexImage2D)(GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const void *);
static void h_glTexImage2D(GLenum t, GLint lvl, GLint ifmt, GLsizei w, GLsizei h, GLint border, GLenum fmt, GLenum type, const void *px) {
    HOOK_ENTER(glTexImage2D);
    if ((is_depthish((unsigned)ifmt) || is_depthish(fmt)) && lvl == 0 && logging_on()) {
        int tex = query_int(0x8069);
        note_dtex(tex, w, h, ifmt);
        ev(E_TEXDEPTH, -1, (int[]){0, ifmt, w, h, (int)fmt, (int)type, tex, 1}, 8);
    }
    O(glTexImage2D)(t, lvl, ifmt, w, h, border, fmt, type, px);
}
typedef void (*fn_glTexStorage2D)(GLenum, GLsizei, GLenum, GLsizei, GLsizei);
static void h_glTexStorage2D(GLenum t, GLsizei levels, GLenum ifmt, GLsizei w, GLsizei h) {
    HOOK_ENTER(glTexStorage2D);
    if (is_depthish(ifmt) && logging_on()) {
        int tex = query_int(0x8069);
        note_dtex(tex, w, h, (int)ifmt);
        ev(E_TEXDEPTH, -1, (int[]){1, (int)ifmt, w, h, 0, 0, tex, levels}, 8);
    }
    O(glTexStorage2D)(t, levels, ifmt, w, h);
}
typedef void (*fn_glCopyTexImage2D)(GLenum, GLint, GLenum, GLint, GLint, GLsizei, GLsizei, GLint);
static void h_glCopyTexImage2D(GLenum t, GLint lvl, GLenum ifmt, GLint x, GLint y, GLsizei w, GLsizei h, GLint b) {
    HOOK_ENTER(glCopyTexImage2D);
    if (atomic_load_explicit(&win_active, memory_order_relaxed)) ev(E_COPYTEX, -1, (int[]){0, (int)ifmt, w, h, x, y}, 6);
    O(glCopyTexImage2D)(t, lvl, ifmt, x, y, w, h, b);
}
typedef void (*fn_glCopyTexSubImage2D)(GLenum, GLint, GLint, GLint, GLint, GLint, GLsizei, GLsizei);
static void h_glCopyTexSubImage2D(GLenum t, GLint lvl, GLint xo, GLint yo, GLint x, GLint y, GLsizei w, GLsizei h) {
    HOOK_ENTER(glCopyTexSubImage2D);
    if (atomic_load_explicit(&win_active, memory_order_relaxed)) ev(E_COPYTEX, -1, (int[]){1, xo, w, h, x, y, yo}, 7);
    O(glCopyTexSubImage2D)(t, lvl, xo, yo, x, y, w, h);
}
typedef void (*fn_glDrawArrays)(GLenum, GLint, GLsizei);
static void h_glDrawArrays(GLenum m, GLint f, GLsizei c) { DRAW_ENTER(glDrawArrays); O(glDrawArrays)(m, f, c); }
typedef void (*fn_glDrawElements)(GLenum, GLsizei, GLenum, const void *);
static void h_glDrawElements(GLenum m, GLsizei c, GLenum t, const void *i) { DRAW_ENTER(glDrawElements); O(glDrawElements)(m, c, t, i); }
typedef void (*fn_glDrawArraysInstanced)(GLenum, GLint, GLsizei, GLsizei);
static void h_glDrawArraysInstanced(GLenum m, GLint f, GLsizei c, GLsizei n) { DRAW_ENTER(glDrawArraysInstanced); O(glDrawArraysInstanced)(m, f, c, n); }
typedef void (*fn_glDrawElementsInstanced)(GLenum, GLsizei, GLenum, const void *, GLsizei);
static void h_glDrawElementsInstanced(GLenum m, GLsizei c, GLenum t, const void *i, GLsizei n) { DRAW_ENTER(glDrawElementsInstanced); O(glDrawElementsInstanced)(m, c, t, i, n); }
typedef void (*fn_glDrawRangeElements)(GLenum, GLuint, GLuint, GLsizei, GLenum, const void *);
static void h_glDrawRangeElements(GLenum m, GLuint s, GLuint e, GLsizei c, GLenum t, const void *i) { DRAW_ENTER(glDrawRangeElements); O(glDrawRangeElements)(m, s, e, c, t, i); }

static void *const hooks[NF] = {
#define X(n) (void *)h_##n,
    FN_LIST(X)
#undef X
};

/* ------------------------------------------------------ report formatting */
static void fmt_event(const Ev *e, char *o, size_t cap) {
    char b1[16], b2[16], b3[16], b4[16];
    const int *v = e->v;
    switch (e->type) {
    case E_BIND:   snprintf(o, cap, "bindFramebuffer(%s, %d)", ename((unsigned)v[0], b1), v[1]); break;
    case E_FBRB:   snprintf(o, cap, "framebufferRenderbuffer(%s, %s, %s, rb=%d)   [boundFBO=%d]",
                            ename((unsigned)v[0], b1), ename((unsigned)v[1], b2), ename((unsigned)v[2], b3), v[3], v[4]); break;
    case E_FBTEX:  snprintf(o, cap, "framebufferTexture2D(%s, %s, %s, tex=%d, lvl=%d)   [boundFBO=%d]",
                            ename((unsigned)v[0], b1), ename((unsigned)v[1], b2), ename((unsigned)v[2], b3), v[3], v[4], v[5]); break;
    case E_RBSTORE:snprintf(o, cap, "renderbufferStorage(rb=%d %dx%d fmt=%s samples=%d)",
                            v[5], v[3], v[4], ename((unsigned)v[2], b1), v[1]); break;
    case E_DISCARD:
    case E_INVAL: {
        char l[120]; l[0] = 0;
        for (int i = 0; i < 3 && i < v[1]; i++) app(l, sizeof l, "%s ", ename((unsigned)v[2 + i], b1));
        snprintf(o, cap, "%s(%s, n=%d: %s)", e->type == E_DISCARD ? "DISCARD" : "INVALIDATE", ename((unsigned)v[0], b2), v[1], l);
        break; }
    case E_RESOLVE:snprintf(o, cap, "RESOLVE_MSAA_APPLE()"); break;
    case E_BLIT:   snprintf(o, cap, "BLIT src(%d,%d,%d,%d) -> dst(%d,%d,%d,%d) mask=%s filter=%s",
                            v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], maskname((unsigned)v[8], b1), ename((unsigned)v[9], b2)); break;
    case E_CLEAR:  snprintf(o, cap, "clear(%s)", maskname((unsigned)v[0], b1)); break;
    case E_FLUSH:  snprintf(o, cap, "flush()"); break;
    case E_FINISH: snprintf(o, cap, "finish()"); break;
    case E_TEXDEPTH: snprintf(o, cap, "DEPTH TEXTURE %s tex=%d %dx%d ifmt=%s fmt=%s type=%s levels=%d",
                            v[0] ? "texStorage2D" : "texImage2D", v[6], v[2], v[3],
                            ename((unsigned)v[1], b1), ename((unsigned)v[4], b2), ename((unsigned)v[5], b3), v[7]); (void)b4; break;
    case E_COPYTEX:snprintf(o, cap, "%s %dx%d from(%d,%d)", v[0] ? "copyTexSubImage2D" : "copyTexImage2D", v[2], v[3], v[4], v[5]); break;
    default:       snprintf(o, cap, "event type %d", e->type); break;
    }
}
static void dump_events(const Ev *arr, unsigned n, int with_snaps) {
    char line[420];
    unsigned ns = atomic_load(&g_snap_n);
    if (ns > SNAP_CAP) ns = SNAP_CAP;
    for (unsigned i = 0; i < n; i++) {
        const Ev *e = &arr[i];
        fmt_event(e, line, sizeof line);
        logf_("%8.3f t%04x d%-5u %s\n", (double)e->t_us / 1e6, e->tid & 0xffff, e->draws, line);
        if (with_snaps && e->snap >= 0 && (unsigned)e->snap < ns) logf_("             -> %s\n", g_snaps[e->snap]);
    }
}

static unsigned g_setup_dumped, g_first_dumped;
static void dump_status(const char *why) {
    logf_("\n==== STATUS: %s  (t=%.1fs) ====\n", why, since_s());
    if (atomic_load(&g_gl_info_done))
        logf_("GL: version=\"%s\" renderer=\"%s\" vendor=\"%s\"\n    ext: %s\n", g_gl_version, g_gl_renderer, g_gl_vendor, g_gl_ext);
    else
        logf_("GL: no hooked GL call has run with a current context yet\n");
    logf_("rebind_symbols -> %d (0 = ok)\n", g_rebind_rc);
    logf_("hook table  (calls | bound = fishhook found an import to rewrite | which images called it)\n");
    char none[1400]; none[0] = 0;
    for (int f = 0; f < NF; f++) {
        unsigned long long c = atomic_load(&calls[f]);
        int bound = fh_orig[f] != NULL;
        if (!c && !bound) { app(none, sizeof none, "%s ", fnames[f]); continue; }
        char cb[260]; cb[0] = 0;
        while (atomic_flag_test_and_set_explicit(&c_lock, memory_order_acquire)) { }
        for (int i = 0; i < c_n[f]; i++) app(cb, sizeof cb, "%s%s", i ? ", " : "", c_name[f][i]);
        atomic_flag_clear_explicit(&c_lock, memory_order_release);
        logf_("  %-40s calls=%-10llu bound=%s  callers=[%s]\n", fnames[f], c, bound ? "Y" : "n", cb);
    }
    logf_("  never bound, never called: %s\n", none[0] ? none : "(none)");

    unsigned sn = atomic_load(&setup_n);
    if (sn > SETUP_CAP) sn = SETUP_CAP;
    if (sn > g_setup_dumped) {
        logf_("-- setup log (renderbuffer storage / FBO attachments / depth textures), new entries --\n");
        dump_events(setup_log + g_setup_dumped, sn - g_setup_dumped, 0);
        g_setup_dumped = sn;
    }
    unsigned fn_ = atomic_load(&g_first_n);
    if (fn_ > FIRST_CAP) fn_ = FIRST_CAP;
    for (; g_first_dumped < fn_; g_first_dumped++) {
        if (!g_first_ok[g_first_dumped]) break;
        logf_("-- first blit/resolve/discard #%u at %.3fs via %s --\n   %s\n", g_first_dumped + 1,
              (double)g_first_t[g_first_dumped] / 1e6, fnames[g_first_f[g_first_dumped]], g_first[g_first_dumped]);
    }
}
static void dump_window(int k, double t_open, double t_close) {
    unsigned n = atomic_load(&win_n);
    if (n > WIN_CAP) n = WIN_CAP;
    unsigned cnt[E_NTYPES]; memset(cnt, 0, sizeof cnt);
    unsigned long long dsum = 0;
    for (unsigned i = 0; i < n; i++) { if (win_log[i].type < E_NTYPES) cnt[win_log[i].type]++; dsum += win_log[i].draws; }
    logf_("\n==== TRACE WINDOW %d   t=%.2fs..%.2fs   events=%u%s ====\n", k + 1, t_open, t_close, n, n >= WIN_CAP ? " (cap hit)" : "");
    logf_("summary: draws=%llu bind=%u fbRB=%u fbTex=%u discard=%u invalidate=%u resolveAPPLE=%u blit=%u clear=%u flush=%u finish=%u copyTex=%u\n",
          dsum, cnt[E_BIND], cnt[E_FBRB], cnt[E_FBTEX], cnt[E_DISCARD], cnt[E_INVAL], cnt[E_RESOLVE], cnt[E_BLIT], cnt[E_CLEAR],
          cnt[E_FLUSH], cnt[E_FINISH], cnt[E_COPYTEX]);
    logf_("columns: time  tid  dN=draw calls since previous event  event      (mask: C=color D=depth S=stencil)\n");
    dump_events(win_log, n, 1);
}

/* ------------------------------------------------------ Mach-O import scanner */
static int interesting(const char *s) {
    static const char *const pre[] = { "_gl", "_egl", "_EGL", "_CVOpenGLES", "_CVMetal", "_CVPixelBuffer", "_dlsym", "_dlopen",
        "_MTL", "_OBJC_CLASS_$_EAGL", "_OBJC_CLASS_$_CAMetal", "_OBJC_CLASS_$_CAEAGL", "_OBJC_CLASS_$_MTL", "_OBJC_CLASS_$_MTK",
        "_SDL_GL", "_SDL_Metal", "_ANGLE", NULL };
    for (int i = 0; pre[i]; i++) if (!strncmp(s, pre[i], strlen(pre[i]))) return 1;
    return 0;
}
static void scan_core(const struct mach_header_64 *h, const uint8_t *linkedit_base, const char *name) {
    const struct symtab_command *st = NULL;
    const struct dysymtab_command *dy = NULL;
    char links[1200]; links[0] = 0;
    const uint8_t *p = (const uint8_t *)(h + 1);
    for (uint32_t i = 0; i < h->ncmds; i++) {
        const struct load_command *lc = (const struct load_command *)p;
        if (lc->cmdsize < 8) break;
        if (lc->cmd == LC_SYMTAB) st = (const struct symtab_command *)lc;
        else if (lc->cmd == LC_DYSYMTAB) dy = (const struct dysymtab_command *)lc;
        else if (lc->cmd == LC_LOAD_DYLIB || lc->cmd == LC_LOAD_WEAK_DYLIB || lc->cmd == LC_REEXPORT_DYLIB) {
            const struct dylib_command *dc = (const struct dylib_command *)lc;
            const char *nm = (const char *)lc + dc->dylib.name.offset;
            const char *sl = strrchr(nm, '/');
            app(links, sizeof links, "%s ", sl ? sl + 1 : nm);
        }
        p += lc->cmdsize;
    }
    logf_("  image: %s\n    links: %s\n", name, links);
    if (!st) { logf_("    (no symbol table)\n"); return; }
    const struct nlist_64 *syms = (const struct nlist_64 *)(linkedit_base + st->symoff);
    const char *str = (const char *)(linkedit_base + st->stroff);
    uint32_t first = 0, count = st->nsyms;
    if (dy) { first = dy->iundefsym; count = dy->nundefsym; }
    size_t cap = 96 * 1024;
    char *buf = (char *)malloc(cap);
    if (!buf) return;
    buf[0] = 0;
    unsigned total = 0, hit = 0, linelen = 0;
    for (uint32_t i = 0; i < count; i++) {
        const struct nlist_64 *s = &syms[first + i];
        if (s->n_type & N_STAB) continue;
        if ((s->n_type & N_TYPE) != N_UNDF || !(s->n_type & N_EXT)) continue;
        if (s->n_un.n_strx >= st->strsize) continue;
        const char *nm = str + s->n_un.n_strx;
        total++;
        if (!interesting(nm)) continue;
        hit++;
        size_t l = strlen(nm);
        if (linelen + l > 100) { app(buf, cap, "\n      "); linelen = 0; }
        app(buf, cap, "%s ", nm);
        linelen += (unsigned)l + 1;
    }
    logf_("    imports: total=%u, GL/CV/Metal/dl-related=%u\n      ", total, hit);
    write_all(buf, strlen(buf));
    write_all("\n", 1);
    free(buf);
}
static void scan_image_mem(const struct mach_header *mh, intptr_t slide, const char *name) {
    const struct mach_header_64 *h = (const struct mach_header_64 *)mh;
    if (h->magic != MH_MAGIC_64) return;
    const struct segment_command_64 *le = NULL;
    const uint8_t *p = (const uint8_t *)(h + 1);
    for (uint32_t i = 0; i < h->ncmds; i++) {
        const struct load_command *lc = (const struct load_command *)p;
        if (lc->cmdsize < 8) break;
        if (lc->cmd == LC_SEGMENT_64) {
            const struct segment_command_64 *sg = (const struct segment_command_64 *)lc;
            if (!strncmp(sg->segname, SEG_LINKEDIT, 16)) le = sg;
        }
        p += lc->cmdsize;
    }
    if (!le) return;
    const uint8_t *base = (const uint8_t *)((uintptr_t)slide + (uintptr_t)le->vmaddr - (uintptr_t)le->fileoff);
    scan_core(h, base, name);
}
static const void *g_scanned[128];
static int g_nscanned;
static void scan_new_images(void) {
    uint32_t n = _dyld_image_count();
    for (uint32_t i = 0; i < n; i++) {
        const char *nm = _dyld_get_image_name(i);
        if (!nm || !strstr(nm, ".app/")) continue;          /* only images inside the app bundle */
        const struct mach_header *mh = _dyld_get_image_header(i);
        int seen = 0;
        for (int k = 0; k < g_nscanned; k++) if (g_scanned[k] == mh) { seen = 1; break; }
        if (seen || g_nscanned >= 128) continue;
        g_scanned[g_nscanned++] = mh;
        scan_image_mem(mh, _dyld_get_image_vmaddr_slide(i), nm);
    }
}

/* ------------------------------------------------------ feedback + housekeeping */
static void (*p_play)(unsigned);
static void buzz(int n) {
    logf_("[t=%.1fs] buzz x%d\n", since_s(), n);
#ifndef RESA_TEST
    if (!p_play) {
        void *h = dlopen("/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox", RTLD_LAZY);
        if (h) p_play = (void (*)(unsigned))dlsym(h, "AudioServicesPlaySystemSound");
    }
    for (int i = 0; i < n; i++) { if (p_play) p_play(4095u); usleep(450 * 1000); }
#else
    (void)n;
#endif
}
static void open_log(void) {
    const char *home = getenv("HOME");
    if (!home || !*home) home = "/tmp";
    snprintf(g_dir, sizeof g_dir, "%s/Documents", home);
    mkdir(g_dir, 0755);
    char cur[700], prev[700];
    snprintf(cur, sizeof cur, "%s/resa_probe.txt", g_dir);
    snprintf(prev, sizeof prev, "%s/resa_probe_prev.txt", g_dir);
    rename(cur, prev);                                       /* keep the previous run */
    g_fd = open(cur, O_WRONLY | O_CREAT | O_TRUNC | O_APPEND, 0644);
}
#ifndef RESA_TEST
static void sysstr(const char *name, char *out, size_t cap) {
    size_t n = cap - 1;
    out[0] = 0;
    if (sysctlbyname(name, out, &n, NULL, 0) == 0) out[n < cap ? n : cap - 1] = 0; else snprintf(out, cap, "?");
}
#endif
static void log_env(void) {
    logf_("resa_probe %s   pid=%d\n", PROBE_VERSION, (int)getpid());
#ifndef RESA_TEST
    char a[96], b[96], c[96];
    sysstr("hw.machine", a, sizeof a);
    sysstr("kern.osversion", b, sizeof b);
    sysstr("kern.osproductversion", c, sizeof c);
    logf_("device=%s  osBuild=%s  osVersion=%s\n", a, b, c);
    logf_("main exe: %s\n", _dyld_get_image_name(0));
#endif
    Dl_info di;
    if (dladdr((void *)log_env, &di) && di.dli_fname) logf_("this dylib: %s\n", di.dli_fname);
    logf_("log dir: %s   (fd=%d)\n", g_dir, g_fd);
    logf_("images scanned = those inside the .app bundle; 'bound' = fishhook found an import of that symbol in ANY loaded image\n");
}
static void install_hooks(void) {
#ifdef RESA_TEST
    logf_("TEST MODE: no fishhook\n");
#else
    static struct rebinding rb[NF];
    for (int i = 0; i < NF; i++) { rb[i].name = fnames[i]; rb[i].replacement = hooks[i]; rb[i].replaced = &fh_orig[i]; }
    g_rebind_rc = rebind_symbols(rb, NF);
    logf_("rebind_symbols(%d names) -> %d\n", NF, g_rebind_rc);
#endif
}

static void *watchdog_main(void *arg) {
    (void)arg;
    int b_load = 0, b_gl = 0, b_blit = 0, st6 = 0, st20 = 0, k = 0, open = 0;
    double t_open = 0, earliest = WIN_MIN_START[0];
    uint32_t last_imgs = 0;
    for (;;) {
        usleep(100 * 1000);
        double t = since_s();
        uint32_t ni = _dyld_image_count();
        if (ni != last_imgs && t >= 1.0) { last_imgs = ni; scan_new_images(); }
        if (!b_load && t >= 3.0) { b_load = 1; buzz(1); }
        else if (b_load && !b_gl && atomic_load(&g_gl_seen)) { b_gl = 1; buzz(2); }
        else if (b_gl && !b_blit && atomic_load(&g_blit_n) > 0) { b_blit = 1; buzz(3); }
        if (!st6 && t >= 6.0)  { st6 = 1;  dump_status("early"); }
        if (!st20 && t >= 20.0) { st20 = 1; dump_status("20s"); }
        if (!open && k < NWIN && t >= earliest) {
            if (atomic_load(&g_blit_n) > 0 || t >= earliest + WIN_WAIT) {
                atomic_store(&win_n, 0u);
                atomic_store(&g_snap_n, 0u);
                atomic_store(&win_active, 1);
                open = 1; t_open = t;
            }
        } else if (open && (atomic_load(&win_n) >= WIN_CAP || t - t_open > WIN_MAXLEN)) {
            atomic_store(&win_active, 0);
            usleep(150 * 1000);
            dump_window(k, t_open, since_s());
            dump_status("after window");
            open = 0; k++;
            earliest = (k < NWIN) ? (WIN_MIN_START[k] > t + WIN_GAP ? WIN_MIN_START[k] : t + WIN_GAP) : 1e9;
        }
    }
    return NULL;
}

#ifndef RESA_TEST
__attribute__((constructor))
#endif
static void probe_init(void) {
    g_t0 = now_us();
    open_log();
    log_env();
    install_hooks();
    pthread_t th;
    if (pthread_create(&th, NULL, watchdog_main, NULL) == 0) pthread_detach(th);
    else logf_("pthread_create failed\n");
}
