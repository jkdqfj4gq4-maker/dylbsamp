/*
 * resa_probe.c  --  ReSA (iOS) renderer recon probe, v3
 *
 * Pure C. No link dependency beyond libSystem: GL, the ObjC runtime and CoreVideo are resolved at runtime.
 * Build: see build.yml.  Inject like v2, then play in the 3D world (offline is fine) for ~2 minutes.
 * Output: <app>/Documents/resa_probe.txt   (the previous run is kept as resa_probe_prev.txt)
 *
 * What v2 showed: the engine's per-frame GL calls are NOT plain imports (only 8 setup calls were seen).
 * So v3 tries every route an engine can use to reach GL, and then traces whole frames:
 *   A. fishhook on imports                      (as v2)
 *   B. dlsym() wrapper                          -> every gl*/CV* name the engine looks up is handed a counting wrapper
 *   C. data-segment pointer scan of alyn_resa   -> finds (and re-points) GL function pointers cached before B was in place
 *   D. ObjC swizzles (data-only, no code patching): -[EAGLContext presentRenderbuffer:], +setCurrentContext:,
 *      -renderbufferStorage:fromDrawable:, -[CAMetalLayer nextDrawable], the ReSA CAMetalDisplayLink delegate
 *   E. ObjC class / method list of alyn_resa
 * Then, in a few short windows while the 3D scene is busy, it records the order of FBO binds, clears, depth-test
 * toggles, draw batches (per program), blits/discards and frame markers, with the call-site address of each.
 *
 * It never draws and never changes GL state (queries only; the renderbuffer binding is restored).
 * Buzzes:  x1 loaded (3s) | x2 first GL call seen | x3 first window captured | x4 all done, you can quit.
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <dlfcn.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>

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

typedef unsigned int  GLenum;
typedef unsigned int  GLuint;
typedef unsigned int  GLbitfield;
typedef unsigned char GLboolean;
typedef int           GLint;
typedef int           GLsizei;
typedef float         GLfloat;

#define PROBE_VERSION "v3"
#define SETUP_CAP 500
#define WIN_CAP   1500
#define SNAP_CAP  40
#define SNAP_LEN  640
#define NWIN      5
#define DS_CAP    900
#define NCALLER   6
#define NIMG      48
#define LOG_CAP_BYTES (1100u * 1024u)

#ifdef RESA_TEST
static const double WIN_MIN_START[NWIN] = {1.5, 2.2, 2.9, 3.6, 4.3};
static const double SCAN_AT[3] = {0.8, 2.0, 3.0};
#  define WIN_GAP     0.3
#  define WIN_WAIT    1.5
#  define WIN_MAXLEN  0.12
#  define BUSY_DPS    300
#  define BUSY_NEED   1
#else
static const double WIN_MIN_START[NWIN] = {13.0, 30.0, 48.0, 66.0, 84.0};
static const double SCAN_AT[3] = {12.5, 25.0, 60.0};
#  define WIN_GAP     12.0   /* min seconds between windows                                  */
#  define WIN_WAIT    90.0   /* open a window anyway this long after it became due           */
#  define WIN_MAXLEN  0.09   /* seconds a window stays open                                  */
#  define BUSY_DPS    6000   /* draw calls per second that count as "3D scene is running"    */
#  define BUSY_NEED   2      /* consecutive busy seconds before a window may open            */
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
static const char *basename_(const char *p) { const char *s = strrchr(p, '/'); return s ? s + 1 : p; }

static int  g_fd = -1;
static char g_dir[600];
static atomic_ullong g_bytes;

static void write_all(const char *s, size_t n) {
    if (g_fd < 0) return;
    if (atomic_fetch_add(&g_bytes, (unsigned long long)n) > LOG_CAP_BYTES) return;
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
    case 0x8056: return "RGBA4";         case 0x93A1: return "BGRA8";        case 0x881A: return "RGBA16F";
    case 0x8814: return "RGBA32F";       case 0x8059: return "RGB10_A2";     case 0x8229: return "R8";
    case 0x822B: return "RG8";           case 0x822E: return "R32F";         case 0x8C43: return "SRGB8_A8";
    case 0x1908: return "RGBA";          case 0x1907: return "RGB";          case 0x80E1: return "BGRA";
    case 0x1902: return "DEPTH_COMPONENT";
    case 0x1401: return "UBYTE";         case 0x1403: return "USHORT";       case 0x1405: return "UINT";
    case 0x84FA: return "UINT_24_8";     case 0x1406: return "FLOAT";        case 0x140B: return "HALF_FLOAT";
    case 0x2600: return "NEAREST";       case 0x2601: return "LINEAR";
    case 0x0B71: return "DEPTH_TEST";    case 0x0B90: return "STENCIL_TEST"; case 0x0C11: return "SCISSOR_TEST";
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
static const char *fourcc(uint32_t v, char *b) {
    if (v >= 0x20202020u && v <= 0x7e7e7e7eu) { b[0] = (char)(v >> 24); b[1] = (char)(v >> 16); b[2] = (char)(v >> 8); b[3] = (char)v; b[4] = 0; }
    else snprintf(b, 16, "0x%x", v);
    return b;
}

/* ------------------------------------------------------ hooked-function list */
#define FN_LIST(X) \
    X(glBindFramebuffer) X(glFramebufferRenderbuffer) X(glFramebufferTexture2D) \
    X(glRenderbufferStorage) X(glRenderbufferStorageMultisample) X(glRenderbufferStorageMultisampleAPPLE) \
    X(glDiscardFramebufferEXT) X(glInvalidateFramebuffer) X(glResolveMultisampleFramebufferAPPLE) \
    X(glBlitFramebuffer) X(glDrawBuffers) X(glReadPixels) X(glClear) X(glFlush) X(glFinish) \
    X(glTexImage2D) X(glTexStorage2D) X(glCopyTexImage2D) X(glCopyTexSubImage2D) \
    X(glEnable) X(glDisable) X(glDepthMask) X(glDepthFunc) X(glViewport) X(glUseProgram) \
    X(glDepthRangef) X(glClearDepthf) X(glUniformMatrix4fv) \
    X(glDrawArrays) X(glDrawElements) X(glDrawArraysInstanced) X(glDrawElementsInstanced) X(glDrawRangeElements) \
    X(CVOpenGLESTextureCacheCreateTextureFromImage) X(CVOpenGLESTextureCacheFlush) \
    X(CVPixelBufferCreate) X(CVMetalTextureCacheCreateTextureFromImage) X(CVMetalTextureCacheFlush)

enum {
#define X(n) F_##n,
    FN_LIST(X)
#undef X
    NF
};
#define NF_GL F_CVOpenGLESTextureCacheCreateTextureFromImage   /* entries below this index are GL calls */
static const char *const fnames[NF] = {
#define X(n) #n,
    FN_LIST(X)
#undef X
};
static atomic_ullong calls[NF];
static atomic_uint   dl_hand[NF];   /* times the dlsym wrapper handed out our hook            */
static atomic_uint   scan_hit[NF];  /* pointer slots found / re-pointed by the data scan      */
static void *fh_orig[NF];           /* fishhook: original pointer (non-NULL => an image imports it) */
static void *dl_orig[NF];           /* original pointer seen by the dlsym wrapper             */
static void *fb_orig[NF];           /* our own dlsym fallback                                 */

static void *(*real_dlsym)(void *, const char *);
static void *rdlsym(void *h, const char *n) { return real_dlsym ? real_dlsym(h, n) : dlsym(h, n); }

static void noop_fn(void) {}
static void *orig_of(int f) {
    void *p = fh_orig[f];
    if (p) return p;
    p = dl_orig[f];
    if (p) return p;
    p = fb_orig[f];
    if (p) return p;
    p = rdlsym(RTLD_DEFAULT, fnames[f]);
    if (!p) p = (void *)noop_fn;
    fb_orig[f] = p;
    return p;
}
static int have_orig(int f) { return orig_of(f) != (void *)noop_fn; }
#define O(n) ((fn_##n)orig_of(F_##n))

/* ------------------------------------------------------ GL query functions (real ones, resolved at runtime) */
typedef void (*fn_getint)(GLenum, GLint *);
typedef void (*fn_getbool)(GLenum, GLboolean *);
typedef GLboolean (*fn_isen)(GLenum);
typedef const unsigned char *(*fn_getstr)(GLenum);
typedef void (*fn_getfbap)(GLenum, GLenum, GLenum, GLint *);
typedef void (*fn_getrbp)(GLenum, GLenum, GLint *);
typedef void (*fn_bindrb)(GLenum, GLuint);
typedef GLenum (*fn_geterr)(void);
typedef GLuint (*fn_cvname)(void *);
static fn_getint  p_getint;
static fn_getbool p_getbool;
static fn_isen    p_isen;
static fn_getstr  p_getstr;
static fn_getfbap p_getfbap;
static fn_getrbp  p_getrbp;
static fn_bindrb  p_bindrb;
static fn_geterr  p_geterr;
static fn_cvname  p_cvname;
static atomic_int g_gl_resolved;

static void resolve_gl(void) {
    if (atomic_load(&g_gl_resolved)) return;
    p_getint  = (fn_getint) rdlsym(RTLD_DEFAULT, "glGetIntegerv");
    p_getbool = (fn_getbool)rdlsym(RTLD_DEFAULT, "glGetBooleanv");
    p_isen    = (fn_isen)   rdlsym(RTLD_DEFAULT, "glIsEnabled");
    p_getstr  = (fn_getstr) rdlsym(RTLD_DEFAULT, "glGetString");
    p_getfbap = (fn_getfbap)rdlsym(RTLD_DEFAULT, "glGetFramebufferAttachmentParameteriv");
    p_getrbp  = (fn_getrbp) rdlsym(RTLD_DEFAULT, "glGetRenderbufferParameteriv");
    p_bindrb  = (fn_bindrb) rdlsym(RTLD_DEFAULT, "glBindRenderbuffer");
    p_geterr  = (fn_geterr) rdlsym(RTLD_DEFAULT, "glGetError");
    p_cvname  = (fn_cvname) rdlsym(RTLD_DEFAULT, "CVOpenGLESTextureGetName");
    atomic_store(&g_gl_resolved, 1);
}
static int ctx_current(void) { resolve_gl(); return p_getstr && p_getstr(0x1F02) != NULL; }

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

/* ------------------------------------------------------ app image table (call-site names, pointer scan) */
typedef struct {
    uintptr_t lo, hi, slide, hdr;
    char name[40];
    uintptr_t dlo[4], dhi[4];
    int nd;
} Img;
static Img g_img[NIMG];
static atomic_int g_nimg;

static void img_register(const struct mach_header_64 *h, intptr_t slide, const char *name) {
    int n = atomic_load(&g_nimg);
    if (n >= NIMG) return;
    Img *im = &g_img[n];
    memset(im, 0, sizeof *im);
    im->slide = (uintptr_t)slide;
    im->hdr = (uintptr_t)h;
    snprintf(im->name, sizeof im->name, "%s", basename_(name));
    uintptr_t lo = (uintptr_t)-1, hi = 0;
    const uint8_t *p = (const uint8_t *)(h + 1);
    for (uint32_t i = 0; i < h->ncmds; i++) {
        const struct load_command *lc = (const struct load_command *)p;
        if (lc->cmdsize < 8) break;
        if (lc->cmd == LC_SEGMENT_64) {
            const struct segment_command_64 *sg = (const struct segment_command_64 *)lc;
            if (strncmp(sg->segname, "__PAGEZERO", 16) != 0) {
                uintptr_t a = (uintptr_t)sg->vmaddr + (uintptr_t)slide, b = a + (uintptr_t)sg->vmsize;
                if (a < lo) lo = a;
                if (b > hi) hi = b;
                if ((!strncmp(sg->segname, "__DATA", 16) || !strncmp(sg->segname, "__DATA_DIRTY", 16)) && im->nd < 4) {
                    im->dlo[im->nd] = a; im->dhi[im->nd] = b; im->nd++;
                }
            }
        }
        p += lc->cmdsize;
    }
    im->lo = lo; im->hi = hi;
    atomic_store(&g_nimg, n + 1);
}
static const char *fmt_ra(const void *ra, char *o, size_t cap) {
    uintptr_t a = (uintptr_t)ra;
    int n = atomic_load(&g_nimg);
    if (!a) { o[0] = 0; return o; }
    for (int i = 0; i < n && i < NIMG; i++)
        if (a >= g_img[i].lo && a < g_img[i].hi) { snprintf(o, cap, "%s+0x%lx", g_img[i].name, (unsigned long)(a - g_img[i].slide)); return o; }
    snprintf(o, cap, "0x%lx", (unsigned long)a);
    return o;
}

/* ------------------------------------------------------ caller bookkeeping (which image calls what) */
static void *c_base[NF][NCALLER];
static char  c_name[NF][NCALLER][40];
static int   c_n[NF];
static atomic_flag c_lock = ATOMIC_FLAG_INIT;
static atomic_int  g_gl_seen;

static void note_caller(int f, void *ra) {
    unsigned long long c = atomic_fetch_add_explicit(&calls[f], 1ull, memory_order_relaxed);
    if (c == 0 && f < NF_GL) atomic_store_explicit(&g_gl_seen, 1, memory_order_relaxed);
    if (c >= 24 && (c & 4095ull) != 0) return;       /* sample: first 24 calls, then every 4096th */
    if (!atomic_load_explicit(&g_gl_info_done, memory_order_relaxed)) gl_info_try();
    Dl_info di;
    if (!ra || !dladdr(ra, &di) || !di.dli_fbase) return;
    while (atomic_flag_test_and_set_explicit(&c_lock, memory_order_acquire)) { }
    int known = 0;
    for (int i = 0; i < c_n[f]; i++) if (c_base[f][i] == di.dli_fbase) { known = 1; break; }
    if (!known && c_n[f] < NCALLER) {
        c_base[f][c_n[f]] = di.dli_fbase;
        snprintf(c_name[f][c_n[f]], sizeof c_name[f][0], "%s", di.dli_fname ? basename_(di.dli_fname) : "?");
        c_n[f]++;
    }
    atomic_flag_clear_explicit(&c_lock, memory_order_release);
}

/* ------------------------------------------------------ event logs */
enum { E_SEG = 1, E_BIND, E_FBRB, E_FBTEX, E_RBSTORE, E_DISCARD, E_INVAL, E_RESOLVE, E_BLIT, E_DRAWBUF, E_READPIX,
       E_CLEAR, E_TEXBIG, E_COPYTEX, E_CAP, E_DEPTHMASK, E_DEPTHFUNC, E_VIEWPORT, E_DEPTHRANGE, E_CLEARDEPTH,
       E_MARK, E_CVTEX, E_CVPIX, E_CVMTL, E_SETCTX, E_EAGLSTORE, E_NTYPES };
enum { M_PRESENT, M_GLFLUSH, M_GLFINISH, M_CVGLFLUSH, M_DRAWABLE, M_LINK, M_CVMTLTEX, M_CVMTLFLUSH, M_N };
static const char *const mk_name[M_N] = {
    "EAGL presentRenderbuffer", "glFlush", "glFinish", "CVOpenGLESTextureCacheFlush", "CAMetalLayer nextDrawable",
    "ReSA display-link callback", "CVMetalTextureCacheCreateTextureFromImage", "CVMetalTextureCacheFlush" };
enum { K_BIND, K_CLEAR, K_DISCARD, K_INVAL, K_BLIT, K_RESOLVE, K_PRESENT, K_FLUSH, K_FINISH, K_DRAWABLE, K_N };
static const char *const k_name[K_N] = { "bindFramebuffer(!=0)", "clear", "discard", "invalidate", "blit", "resolve",
                                         "present", "glFlush", "glFinish", "nextDrawable" };
static const int k_limit[K_N] = { 6, 6, 8, 8, 8, 8, 4, 2, 2, 2 };

typedef struct { uint32_t t_us; uint16_t tid, type; int32_t v[10]; uintptr_t ra; int16_t snap; } Ev;

static Ev setup_log[SETUP_CAP];  static atomic_uint setup_n;
static Ev win_log[WIN_CAP];      static atomic_uint win_n;
static atomic_int  win_active, win_arm;
static atomic_uint g_win_gen;
static uint64_t    g_win_t0_us;
static atomic_ullong mk_cnt[M_N];
static uint32_t    mk_tid[M_N];

static char g_snaps[SNAP_CAP][SNAP_LEN];  static atomic_uint g_snap_n;
static atomic_uint win_snap_cnt[K_N];
typedef struct { char text[SNAP_LEN]; uint32_t t_us, tid; uintptr_t ra; volatile int ok; } FirstSnap;
static FirstSnap g_first[K_N][2];
static atomic_uint first_cnt[K_N];
static int first_dumped[K_N];

static int is_setup_type(int t) {
    switch (t) {
    case E_FBRB: case E_FBTEX: case E_RBSTORE: case E_TEXBIG: case E_CVTEX: case E_CVPIX: case E_DRAWBUF:
    case E_SETCTX: case E_EAGLSTORE: case E_DEPTHRANGE: case E_CLEARDEPTH: return 1;
    default: return 0;
    }
}
static inline int win_on(void) { return atomic_load_explicit(&win_active, memory_order_relaxed); }
static int setup_room(void) { return atomic_load_explicit(&setup_n, memory_order_relaxed) < SETUP_CAP; }

static void ev(int type, const int *v, int nv, const void *ra, int snap) {
    Ev x;
    memset(&x, 0, sizeof x);
    x.t_us = (uint32_t)(now_us() - g_t0);
    x.tid  = (uint16_t)cur_tid();
    x.type = (uint16_t)type;
    x.ra   = (uintptr_t)ra;
    x.snap = (int16_t)snap;
    for (int i = 0; i < nv && i < 10; i++) x.v[i] = v[i];
    if (is_setup_type(type) && setup_room()) {
        unsigned i = atomic_fetch_add(&setup_n, 1u);
        if (i < SETUP_CAP) setup_log[i] = x;
    }
    if (win_on()) {
        unsigned i = atomic_fetch_add(&win_n, 1u);
        if (i < WIN_CAP) win_log[i] = x;
    }
}

/* ------------------------------------------------------ per-thread GL state shadow + draw segments */
#define SF_DEPTH   1u
#define SF_DMASK   2u
#define SF_BLEND   4u
#define SF_CULL    8u
#define SF_STENCIL 16u
#define SF_SCISSOR 32u
typedef struct {
    int init, gen;
    unsigned prog, flags;
    int fbo, vp[4];
    unsigned sdraws, sverts, sprogs, pother;
    unsigned pk[6], pc[6];
} TS;
static __thread TS ts;

static const char *fmt_flags(unsigned f, char *o, size_t cap) {
    o[0] = 0;
    if (f & SF_DEPTH)   app(o, cap, "DEPTH_TEST ");
    if (f & SF_DMASK)   app(o, cap, "depthWrite ");
    if (f & SF_BLEND)   app(o, cap, "BLEND ");
    if (f & SF_CULL)    app(o, cap, "CULL ");
    if (f & SF_STENCIL) app(o, cap, "STENCIL ");
    if (f & SF_SCISSOR) app(o, cap, "SCISSOR ");
    if (!o[0]) snprintf(o, cap, "-");
    return o;
}
static void note_thread(void) {
    char nm[64];
    nm[0] = 0;
#ifdef RESA_TEST
    snprintf(nm, sizeof nm, "test-thread");
#else
    pthread_getname_np(pthread_self(), nm, sizeof nm);
#endif
    logf_("[thread] t%04x name='%s' main=%d first hooked call at t=%.2fs\n", cur_tid() & 0xffff, nm,
          pthread_main_np() ? 1 : 0, since_s());
}
static void ts_touch(void) {
    if (ts.init) return;
    ts.init = 1;
    ts.gen = -1;
    ts.flags = SF_DMASK;
    if (ctx_current()) {
        if (p_isen) {
            if (p_isen(0x0B71)) ts.flags |= SF_DEPTH;
            if (p_isen(0x0BE2)) ts.flags |= SF_BLEND;
            if (p_isen(0x0B44)) ts.flags |= SF_CULL;
            if (p_isen(0x0B90)) ts.flags |= SF_STENCIL;
            if (p_isen(0x0C11)) ts.flags |= SF_SCISSOR;
        }
        GLboolean b = 1;
        if (p_getbool) { p_getbool(0x0B72, &b); if (!b) ts.flags &= ~SF_DMASK; }
        GLint v = 0;
        p_getint(0x8B8D, &v); ts.prog = (unsigned)v;
        v = 0; p_getint(0x8CA6, &v); ts.fbo = v;
        p_getint(0x0BA2, ts.vp);
    }
    note_thread();
}
static void seg_reset(void) { ts.sdraws = ts.sverts = ts.sprogs = ts.pother = 0; }
static void seg_add(unsigned verts) {
    unsigned g = atomic_load_explicit(&g_win_gen, memory_order_relaxed);
    if (ts.gen != (int)g) { seg_reset(); ts.gen = (int)g; }
    ts.sdraws++;
    ts.sverts += verts;
    for (unsigned i = 0; i < ts.sprogs; i++) if (ts.pk[i] == ts.prog) { ts.pc[i]++; return; }
    if (ts.sprogs < 6) { ts.pk[ts.sprogs] = ts.prog; ts.pc[ts.sprogs] = 1; ts.sprogs++; } else ts.pother++;
}
static void seg_flush(void) {
    unsigned g = atomic_load_explicit(&g_win_gen, memory_order_relaxed);
    if (ts.gen != (int)g) { seg_reset(); ts.gen = (int)g; return; }
    if (!ts.sdraws) return;
    int v[10];
    memset(v, 0, sizeof v);
    v[0] = (int)ts.sdraws;
    v[1] = (int)ts.sverts;
    for (int k = 0; k < 3; k++) {                    /* top-3 programs by draw count */
        int best = -1;
        for (unsigned i = 0; i < ts.sprogs; i++) {
            int used = 0;
            for (int j = 0; j < k; j++) if (v[2 + 2 * j] == (int)ts.pk[i] && v[3 + 2 * j] == (int)ts.pc[i]) used = 1;
            if (!used && (best < 0 || ts.pc[i] > ts.pc[best])) best = (int)i;
        }
        if (best >= 0) { v[2 + 2 * k] = (int)ts.pk[best]; v[3 + 2 * k] = (int)ts.pc[best]; }
    }
    v[8] = (int)ts.flags;
    v[9] = (int)ts.sprogs + (ts.pother ? 100 : 0);
    ev(E_SEG, v, 10, NULL, -1);
    seg_reset();
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
/* ES has no size query for textures, so remember the render-target-sized ones we saw being created */
static struct { int tex, w, h, fmt; } g_btex[96];
static atomic_int g_nbtex;
static void note_btex(int tex, int w, int h, int fmt) {
    int i = atomic_fetch_add(&g_nbtex, 1);
    if (i < 96) { g_btex[i].tex = tex; g_btex[i].w = w; g_btex[i].h = h; g_btex[i].fmt = fmt; }
}
static int find_btex(int tex) {
    int n = atomic_load(&g_nbtex);
    if (n > 96) n = 96;
    for (int i = n - 1; i >= 0; i--) if (g_btex[i].tex == tex) return i;
    return -1;
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
        int k = find_btex(name);
        if (k >= 0) snprintf(o, cap, "TEX%d(%dx%d %s)", name, g_btex[k].w, g_btex[k].h, ename((unsigned)g_btex[k].fmt, b));
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
    if (!p_getstr || !p_getstr(0x1F02)) { snprintf(out, cap, "(no GL context current on this thread)"); return; }
    GLenum pre = p_geterr ? p_geterr() : 0;
    char rd[300], dr[300], fl[80];
    GLint vp[4] = {-1, -1, -1, -1};
    fb_desc(rd, sizeof rd, 0x8CA8, 0x8CAA);
    fb_desc(dr, sizeof dr, 0x8CA9, 0x8CA6);
    p_getint(0x0BA2, vp);
    snprintf(out, cap, "READ %s | DRAW %s | vp=%d,%d,%d,%d | %s prog=%u", rd, dr, vp[0], vp[1], vp[2], vp[3],
             fmt_flags(ts.flags, fl, sizeof fl), ts.prog);
    if (pre) app(out, cap, " | engineErrBefore=0x%x", pre);
    if (p_geterr) { int guard = 0; GLenum e; while ((e = p_geterr()) != 0 && guard++ < 8) app(out, cap, " | oursErr=0x%x", e); }
}
static int want_snap(int kind) {
    if (atomic_load(&g_snap_n) >= SNAP_CAP) return 0;
    return atomic_fetch_add(&win_snap_cnt[kind], 1u) < (unsigned)k_limit[kind];
}
static int snap_take(void) {
    char tmp[SNAP_LEN];
    snapshot_state(tmp, sizeof tmp);
    unsigned i = atomic_fetch_add(&g_snap_n, 1u);
    if (i >= SNAP_CAP) return -1;
    memcpy(g_snaps[i], tmp, sizeof tmp);
    return (int)i;
}
static void first_snap(int kind, const void *ra) {
    if (atomic_load_explicit(&first_cnt[kind], memory_order_relaxed) >= 2) return;
    unsigned j = atomic_fetch_add(&first_cnt[kind], 1u);
    if (j >= 2) return;
    FirstSnap *fs = &g_first[kind][j];
    snapshot_state(fs->text, sizeof fs->text);
    fs->t_us = (uint32_t)(now_us() - g_t0);
    fs->tid = cur_tid();
    fs->ra = (uintptr_t)ra;
    fs->ok = 1;
}
/* structural event inside a window: close the running draw segment, optionally snapshot, then log */
static void evw(int type, const int *v, int nv, const void *ra, int kind) {
    int sn = -1;
    seg_flush();
    if (kind >= 0 && want_snap(kind)) sn = snap_take();
    ev(type, v, nv, ra, sn);
}
static int query_int(GLenum pname) {
    GLint v = -1;
    resolve_gl();
    if (p_getint) p_getint(pname, &v);
    return (int)v;
}

/* ------------------------------------------------------ frame markers + window control */
static void open_window(void) {
    atomic_store(&win_n, 0u);
    atomic_store(&g_snap_n, 0u);
    for (int k = 0; k < K_N; k++) atomic_store(&win_snap_cnt[k], 0u);
    atomic_fetch_add(&g_win_gen, 1u);
    g_win_t0_us = now_us();
    atomic_store(&win_active, 1);
}
static void marker(int m, const void *ra, int kind) {
    atomic_fetch_add_explicit(&mk_cnt[m], 1ull, memory_order_relaxed);
    mk_tid[m] = cur_tid();
    if (atomic_load_explicit(&win_arm, memory_order_relaxed) && !win_on())
        if (atomic_exchange(&win_arm, 0)) open_window();       /* windows start right after a frame marker */
    if (win_on()) { ts_touch(); evw(E_MARK, (int[]){m}, 1, ra, kind); }
    if (kind >= 0) first_snap(kind, ra);
}

/* ------------------------------------------------------ dlsym log */
typedef struct { char name[44]; uint32_t cnt, t_ms; void *ra; unsigned char nul, wrapped; } DsEnt;
static DsEnt g_ds[DS_CAP];
static atomic_uint g_ds_n;
static short g_ds_hash[2048];
static atomic_flag g_ds_lock = ATOMIC_FLAG_INIT;
static atomic_ullong g_ds_total;
static unsigned g_ds_dumped;

static uint32_t fnv(const char *s) { uint32_t h = 2166136261u; while (*s) { h ^= (unsigned char)*s++; h *= 16777619u; } return h; }
static void ds_note(const char *name, void *res, void *ra, int wrapped) {
    atomic_fetch_add(&g_ds_total, 1ull);
    uint32_t h = fnv(name) & 2047;
    while (atomic_flag_test_and_set_explicit(&g_ds_lock, memory_order_acquire)) { }
    for (int probe = 0; probe < 2048; probe++, h = (h + 1) & 2047) {
        short ix = g_ds_hash[h];
        if (!ix) {
            unsigned n = atomic_load(&g_ds_n);
            if (n < DS_CAP) {
                DsEnt *e = &g_ds[n];
                snprintf(e->name, sizeof e->name, "%s", name);
                e->cnt = 1;
                e->t_ms = (uint32_t)((now_us() - g_t0) / 1000);
                e->ra = ra;
                e->nul = res == NULL;
                e->wrapped = (unsigned char)wrapped;
                g_ds_hash[h] = (short)(n + 1);
                atomic_store(&g_ds_n, n + 1);
            }
            break;
        }
        if (!strncmp(g_ds[ix - 1].name, name, sizeof g_ds[0].name - 1)) { g_ds[ix - 1].cnt++; break; }
    }
    atomic_flag_clear_explicit(&g_ds_lock, memory_order_release);
}
static int wrap_index(const char *n) {
    if (!((n[0] == 'g' && n[1] == 'l') || (n[0] == 'C' && n[1] == 'V'))) return -1;
    for (int f = 0; f < NF; f++) if (!strcmp(n, fnames[f])) return f;
    return -1;
}
static void *const hooks[NF];   /* defined after the hook functions */
static void *h_dlsym(void *handle, const char *name) {
    void *ra = __builtin_return_address(0);
    uintptr_t hv = (uintptr_t)handle;
    if (hv == (uintptr_t)-1 || hv == (uintptr_t)-3 || !name || !real_dlsym)    /* RTLD_NEXT / RTLD_SELF depend on the caller's image */
        return real_dlsym ? real_dlsym(handle, name) : NULL;
    void *p = real_dlsym(handle, name);
    int f = p ? wrap_index(name) : -1;
    ds_note(name, p, ra, f >= 0);
    if (f >= 0) {
        dl_orig[f] = p;
        atomic_fetch_add(&dl_hand[f], 1u);
        return hooks[f];
    }
    return p;
}

/* ------------------------------------------------------ projection-like matrices (for near/far/FOV later) */
typedef struct { unsigned prog; int loc; int tr; char tag; float m[16]; unsigned cnt; } PM;
static PM g_pm[10];
static atomic_uint g_pm_n;
static atomic_flag g_pm_lock = ATOMIC_FLAG_INIT;
static int near_f(float a, float b) { return fabsf(a - b) < 1e-4f; }
static void pm_check(GLint loc, GLboolean tr, const GLfloat *v) {
    char tag = 0;
    if (near_f(v[3], 0) && near_f(v[7], 0) && near_f(fabsf(v[11]), 1) && near_f(v[15], 0)) tag = 'A';      /* last row (0,0,+-1,0) when column-major */
    else if (near_f(v[12], 0) && near_f(v[13], 0) && near_f(fabsf(v[14]), 1) && near_f(v[15], 0)) tag = 'B'; /* same, row-major memory */
    if (!tag) return;
    while (atomic_flag_test_and_set_explicit(&g_pm_lock, memory_order_acquire)) { }
    unsigned n = atomic_load(&g_pm_n);
    int found = 0;
    for (unsigned i = 0; i < n && i < 10; i++)
        if (g_pm[i].loc == loc && g_pm[i].prog == ts.prog && !memcmp(g_pm[i].m, v, sizeof g_pm[i].m)) { g_pm[i].cnt++; found = 1; break; }
    if (!found && n < 10) {
        g_pm[n].prog = ts.prog; g_pm[n].loc = loc; g_pm[n].tr = tr; g_pm[n].tag = tag; g_pm[n].cnt = 1;
        memcpy(g_pm[n].m, v, sizeof g_pm[n].m);
        atomic_store(&g_pm_n, n + 1);
    }
    atomic_flag_clear_explicit(&g_pm_lock, memory_order_release);
}

/* ------------------------------------------------------ GL hooks */
#define HOOK_ENTER(n) void *ra_ = __builtin_return_address(0); (void)ra_; note_caller(F_##n, ra_); ts_touch()
#define NOOP_RET(n, rv) do { if (!have_orig(F_##n)) return (rv); } while (0)

typedef void (*fn_glBindFramebuffer)(GLenum, GLuint);
static void h_glBindFramebuffer(GLenum t, GLuint fb) {
    HOOK_ENTER(glBindFramebuffer);
    O(glBindFramebuffer)(t, fb);
    if (t == 0x8D40 || t == 0x8CA9) ts.fbo = (int)fb;
    if (win_on()) evw(E_BIND, (int[]){(int)t, (int)fb}, 2, ra_, K_BIND);
    if (fb != 0) first_snap(K_BIND, ra_);
}
typedef void (*fn_glFramebufferRenderbuffer)(GLenum, GLenum, GLenum, GLuint);
static void h_glFramebufferRenderbuffer(GLenum t, GLenum a, GLenum rt, GLuint rb) {
    HOOK_ENTER(glFramebufferRenderbuffer);
    if (win_on() || setup_room()) evw(E_FBRB, (int[]){(int)t, (int)a, (int)rt, (int)rb, query_int(t == 0x8CA8 ? 0x8CAA : 0x8CA6)}, 5, ra_, -1);
    O(glFramebufferRenderbuffer)(t, a, rt, rb);
}
typedef void (*fn_glFramebufferTexture2D)(GLenum, GLenum, GLenum, GLuint, GLint);
static void h_glFramebufferTexture2D(GLenum t, GLenum a, GLenum tt, GLuint tex, GLint lvl) {
    HOOK_ENTER(glFramebufferTexture2D);
    if (win_on() || setup_room()) evw(E_FBTEX, (int[]){(int)t, (int)a, (int)tt, (int)tex, (int)lvl, query_int(t == 0x8CA8 ? 0x8CAA : 0x8CA6)}, 6, ra_, -1);
    O(glFramebufferTexture2D)(t, a, tt, tex, lvl);
}
typedef void (*fn_glRenderbufferStorage)(GLenum, GLenum, GLsizei, GLsizei);
static void h_glRenderbufferStorage(GLenum t, GLenum fmt, GLsizei w, GLsizei h) {
    HOOK_ENTER(glRenderbufferStorage);
    if (win_on() || setup_room()) evw(E_RBSTORE, (int[]){(int)t, -1, (int)fmt, w, h, query_int(0x8CA7)}, 6, ra_, -1);
    O(glRenderbufferStorage)(t, fmt, w, h);
}
typedef void (*fn_glRenderbufferStorageMultisample)(GLenum, GLsizei, GLenum, GLsizei, GLsizei);
static void h_glRenderbufferStorageMultisample(GLenum t, GLsizei s, GLenum fmt, GLsizei w, GLsizei h) {
    HOOK_ENTER(glRenderbufferStorageMultisample);
    if (win_on() || setup_room()) evw(E_RBSTORE, (int[]){(int)t, s, (int)fmt, w, h, query_int(0x8CA7)}, 6, ra_, -1);
    O(glRenderbufferStorageMultisample)(t, s, fmt, w, h);
}
typedef void (*fn_glRenderbufferStorageMultisampleAPPLE)(GLenum, GLsizei, GLenum, GLsizei, GLsizei);
static void h_glRenderbufferStorageMultisampleAPPLE(GLenum t, GLsizei s, GLenum fmt, GLsizei w, GLsizei h) {
    HOOK_ENTER(glRenderbufferStorageMultisampleAPPLE);
    if (win_on() || setup_room()) evw(E_RBSTORE, (int[]){(int)t, s, (int)fmt, w, h, query_int(0x8CA7)}, 6, ra_, -1);
    O(glRenderbufferStorageMultisampleAPPLE)(t, s, fmt, w, h);
}
typedef void (*fn_glDiscardFramebufferEXT)(GLenum, GLsizei, const GLenum *);
static void h_glDiscardFramebufferEXT(GLenum t, GLsizei n, const GLenum *att) {
    HOOK_ENTER(glDiscardFramebufferEXT);
    int a[3] = {-1, -1, -1};
    for (int i = 0; att && i < n && i < 3; i++) a[i] = (int)att[i];
    if (win_on()) evw(E_DISCARD, (int[]){(int)t, n, a[0], a[1], a[2]}, 5, ra_, K_DISCARD);
    first_snap(K_DISCARD, ra_);
    O(glDiscardFramebufferEXT)(t, n, att);
}
typedef void (*fn_glInvalidateFramebuffer)(GLenum, GLsizei, const GLenum *);
static void h_glInvalidateFramebuffer(GLenum t, GLsizei n, const GLenum *att) {
    HOOK_ENTER(glInvalidateFramebuffer);
    int a[3] = {-1, -1, -1};
    for (int i = 0; att && i < n && i < 3; i++) a[i] = (int)att[i];
    if (win_on()) evw(E_INVAL, (int[]){(int)t, n, a[0], a[1], a[2]}, 5, ra_, K_INVAL);
    first_snap(K_INVAL, ra_);
    O(glInvalidateFramebuffer)(t, n, att);
}
typedef void (*fn_glResolveMultisampleFramebufferAPPLE)(void);
static void h_glResolveMultisampleFramebufferAPPLE(void) {
    HOOK_ENTER(glResolveMultisampleFramebufferAPPLE);
    if (win_on()) evw(E_RESOLVE, (int[]){0}, 0, ra_, K_RESOLVE);
    first_snap(K_RESOLVE, ra_);
    O(glResolveMultisampleFramebufferAPPLE)();
}
typedef void (*fn_glBlitFramebuffer)(GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLbitfield, GLenum);
static void h_glBlitFramebuffer(GLint sx0, GLint sy0, GLint sx1, GLint sy1, GLint dx0, GLint dy0, GLint dx1, GLint dy1,
                                GLbitfield mask, GLenum filter) {
    HOOK_ENTER(glBlitFramebuffer);
    if (win_on()) evw(E_BLIT, (int[]){sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1, (int)mask, (int)filter}, 10, ra_, K_BLIT);
    first_snap(K_BLIT, ra_);
    O(glBlitFramebuffer)(sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1, mask, filter);
}
typedef void (*fn_glDrawBuffers)(GLsizei, const GLenum *);
static void h_glDrawBuffers(GLsizei n, const GLenum *b) {
    HOOK_ENTER(glDrawBuffers);
    if (win_on() || setup_room()) {
        int a[3] = {-1, -1, -1};
        for (int i = 0; b && i < n && i < 3; i++) a[i] = (int)b[i];
        evw(E_DRAWBUF, (int[]){n, a[0], a[1], a[2]}, 4, ra_, -1);
    }
    O(glDrawBuffers)(n, b);
}
typedef void (*fn_glReadPixels)(GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void *);
static void h_glReadPixels(GLint x, GLint y, GLsizei w, GLsizei h, GLenum f, GLenum t, void *px) {
    HOOK_ENTER(glReadPixels);
    if (win_on()) evw(E_READPIX, (int[]){x, y, w, h, (int)f, (int)t, ts.fbo}, 7, ra_, -1);
    O(glReadPixels)(x, y, w, h, f, t, px);
}
typedef void (*fn_glClear)(GLbitfield);
static void h_glClear(GLbitfield m) {
    HOOK_ENTER(glClear);
    if (win_on()) evw(E_CLEAR, (int[]){(int)m, ts.fbo}, 2, ra_, K_CLEAR);
    first_snap(K_CLEAR, ra_);
    O(glClear)(m);
}
typedef void (*fn_glFlush)(void);
static void h_glFlush(void) {
    HOOK_ENTER(glFlush);
    marker(M_GLFLUSH, ra_, K_FLUSH);
    O(glFlush)();
}
typedef void (*fn_glFinish)(void);
static void h_glFinish(void) {
    HOOK_ENTER(glFinish);
    marker(M_GLFINISH, ra_, K_FINISH);
    O(glFinish)();
}
typedef void (*fn_glTexImage2D)(GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const void *);
static void h_glTexImage2D(GLenum t, GLint lvl, GLint ifmt, GLsizei w, GLsizei h, GLint border, GLenum fmt, GLenum type, const void *px) {
    HOOK_ENTER(glTexImage2D);
    if (lvl == 0 && (is_depthish((unsigned)ifmt) || is_depthish(fmt) || (w >= 600 && h >= 300)) && (win_on() || setup_room())) {
        int tex = query_int(0x8069);
        note_btex(tex, w, h, ifmt);
        evw(E_TEXBIG, (int[]){0, ifmt, w, h, (int)fmt, (int)type, tex, 1}, 8, ra_, -1);
    }
    O(glTexImage2D)(t, lvl, ifmt, w, h, border, fmt, type, px);
}
typedef void (*fn_glTexStorage2D)(GLenum, GLsizei, GLenum, GLsizei, GLsizei);
static void h_glTexStorage2D(GLenum t, GLsizei levels, GLenum ifmt, GLsizei w, GLsizei h) {
    HOOK_ENTER(glTexStorage2D);
    if ((is_depthish(ifmt) || (w >= 600 && h >= 300)) && (win_on() || setup_room())) {
        int tex = query_int(0x8069);
        note_btex(tex, w, h, (int)ifmt);
        evw(E_TEXBIG, (int[]){1, (int)ifmt, w, h, 0, 0, tex, levels}, 8, ra_, -1);
    }
    O(glTexStorage2D)(t, levels, ifmt, w, h);
}
typedef void (*fn_glCopyTexImage2D)(GLenum, GLint, GLenum, GLint, GLint, GLsizei, GLsizei, GLint);
static void h_glCopyTexImage2D(GLenum t, GLint lvl, GLenum ifmt, GLint x, GLint y, GLsizei w, GLsizei h, GLint b) {
    HOOK_ENTER(glCopyTexImage2D);
    if (win_on()) evw(E_COPYTEX, (int[]){0, (int)ifmt, w, h, x, y, ts.fbo}, 7, ra_, -1);
    O(glCopyTexImage2D)(t, lvl, ifmt, x, y, w, h, b);
}
typedef void (*fn_glCopyTexSubImage2D)(GLenum, GLint, GLint, GLint, GLint, GLint, GLsizei, GLsizei);
static void h_glCopyTexSubImage2D(GLenum t, GLint lvl, GLint xo, GLint yo, GLint x, GLint y, GLsizei w, GLsizei h) {
    HOOK_ENTER(glCopyTexSubImage2D);
    if (win_on()) evw(E_COPYTEX, (int[]){1, xo, w, h, x, y, ts.fbo}, 7, ra_, -1);
    O(glCopyTexSubImage2D)(t, lvl, xo, yo, x, y, w, h);
}
static unsigned cap_flag(GLenum cap) {
    switch (cap) {
    case 0x0B71: return SF_DEPTH;  case 0x0BE2: return SF_BLEND;   case 0x0B44: return SF_CULL;
    case 0x0B90: return SF_STENCIL; case 0x0C11: return SF_SCISSOR; default: return 0;
    }
}
static void cap_change(GLenum cap, int on, const void *ra) {
    unsigned fl = cap_flag(cap);
    if (!fl) return;
    int was = (ts.flags & fl) != 0;
    if (on) ts.flags |= fl; else ts.flags &= ~fl;
    if (was != on && (fl & (SF_DEPTH | SF_STENCIL | SF_SCISSOR)) && win_on()) evw(E_CAP, (int[]){(int)cap, on}, 2, ra, -1);
}
typedef void (*fn_glEnable)(GLenum);
static void h_glEnable(GLenum cap) { HOOK_ENTER(glEnable); cap_change(cap, 1, ra_); O(glEnable)(cap); }
typedef void (*fn_glDisable)(GLenum);
static void h_glDisable(GLenum cap) { HOOK_ENTER(glDisable); cap_change(cap, 0, ra_); O(glDisable)(cap); }
typedef void (*fn_glDepthMask)(GLboolean);
static void h_glDepthMask(GLboolean m) {
    HOOK_ENTER(glDepthMask);
    int was = (ts.flags & SF_DMASK) != 0;
    if (m) ts.flags |= SF_DMASK; else ts.flags &= ~SF_DMASK;
    if (was != (m != 0) && win_on()) evw(E_DEPTHMASK, (int[]){m != 0}, 1, ra_, -1);
    O(glDepthMask)(m);
}
typedef void (*fn_glDepthFunc)(GLenum);
static void h_glDepthFunc(GLenum f) {
    HOOK_ENTER(glDepthFunc);
    if (win_on()) evw(E_DEPTHFUNC, (int[]){(int)f}, 1, ra_, -1);
    O(glDepthFunc)(f);
}
typedef void (*fn_glViewport)(GLint, GLint, GLsizei, GLsizei);
static void h_glViewport(GLint x, GLint y, GLsizei w, GLsizei h) {
    HOOK_ENTER(glViewport);
    int changed = ts.vp[0] != x || ts.vp[1] != y || ts.vp[2] != w || ts.vp[3] != h;
    ts.vp[0] = x; ts.vp[1] = y; ts.vp[2] = w; ts.vp[3] = h;
    if (changed && win_on()) evw(E_VIEWPORT, (int[]){x, y, w, h, ts.fbo}, 5, ra_, -1);
    O(glViewport)(x, y, w, h);
}
typedef void (*fn_glUseProgram)(GLuint);
static void h_glUseProgram(GLuint p) { HOOK_ENTER(glUseProgram); ts.prog = p; O(glUseProgram)(p); }
typedef void (*fn_glDepthRangef)(GLfloat, GLfloat);
static void h_glDepthRangef(GLfloat n, GLfloat f) {
    HOOK_ENTER(glDepthRangef);
    static int last_n = -1, last_f = -1;
    int a = (int)(n * 1000000.0f), b = (int)(f * 1000000.0f);
    if ((a != last_n || b != last_f) && (win_on() || setup_room())) { last_n = a; last_f = b; evw(E_DEPTHRANGE, (int[]){a, b}, 2, ra_, -1); }
    O(glDepthRangef)(n, f);
}
typedef void (*fn_glClearDepthf)(GLfloat);
static void h_glClearDepthf(GLfloat d) {
    HOOK_ENTER(glClearDepthf);
    static int last = -1;
    int a = (int)(d * 1000000.0f);
    if (a != last && (win_on() || setup_room())) { last = a; evw(E_CLEARDEPTH, (int[]){a}, 1, ra_, -1); }
    O(glClearDepthf)(d);
}
typedef void (*fn_glUniformMatrix4fv)(GLint, GLsizei, GLboolean, const GLfloat *);
static void h_glUniformMatrix4fv(GLint loc, GLsizei cnt, GLboolean tr, const GLfloat *v) {
    HOOK_ENTER(glUniformMatrix4fv);
    if (cnt >= 1 && v && atomic_load_explicit(&g_pm_n, memory_order_relaxed) < 10) pm_check(loc, tr, v);
    O(glUniformMatrix4fv)(loc, cnt, tr, v);
}
typedef void (*fn_glDrawArrays)(GLenum, GLint, GLsizei);
static void h_glDrawArrays(GLenum m, GLint f, GLsizei c) {
    HOOK_ENTER(glDrawArrays);
    if (win_on()) seg_add((unsigned)(c > 0 ? c : 0));
    O(glDrawArrays)(m, f, c);
}
typedef void (*fn_glDrawElements)(GLenum, GLsizei, GLenum, const void *);
static void h_glDrawElements(GLenum m, GLsizei c, GLenum t, const void *i) {
    HOOK_ENTER(glDrawElements);
    if (win_on()) seg_add((unsigned)(c > 0 ? c : 0));
    O(glDrawElements)(m, c, t, i);
}
typedef void (*fn_glDrawArraysInstanced)(GLenum, GLint, GLsizei, GLsizei);
static void h_glDrawArraysInstanced(GLenum m, GLint f, GLsizei c, GLsizei n) {
    HOOK_ENTER(glDrawArraysInstanced);
    if (win_on()) seg_add((unsigned)(c > 0 ? c : 0));
    O(glDrawArraysInstanced)(m, f, c, n);
}
typedef void (*fn_glDrawElementsInstanced)(GLenum, GLsizei, GLenum, const void *, GLsizei);
static void h_glDrawElementsInstanced(GLenum m, GLsizei c, GLenum t, const void *i, GLsizei n) {
    HOOK_ENTER(glDrawElementsInstanced);
    if (win_on()) seg_add((unsigned)(c > 0 ? c : 0));
    O(glDrawElementsInstanced)(m, c, t, i, n);
}
typedef void (*fn_glDrawRangeElements)(GLenum, GLuint, GLuint, GLsizei, GLenum, const void *);
static void h_glDrawRangeElements(GLenum m, GLuint s, GLuint e, GLsizei c, GLenum t, const void *i) {
    HOOK_ENTER(glDrawRangeElements);
    if (win_on()) seg_add((unsigned)(c > 0 ? c : 0));
    O(glDrawRangeElements)(m, s, e, c, t, i);
}

/* CoreVideo bridge (GLES <-> Metal) */
typedef int (*fn_CVOpenGLESTextureCacheCreateTextureFromImage)(void *, void *, void *, void *, unsigned, int, int, int, unsigned, unsigned, size_t, void **);
static int h_CVOpenGLESTextureCacheCreateTextureFromImage(void *al, void *cache, void *img, void *attrs, unsigned target, int ifmt,
                                                          int w, int h, unsigned fmt, unsigned type, size_t plane, void **out) {
    HOOK_ENTER(CVOpenGLESTextureCacheCreateTextureFromImage);
    NOOP_RET(CVOpenGLESTextureCacheCreateTextureFromImage, -1);
    int r = O(CVOpenGLESTextureCacheCreateTextureFromImage)(al, cache, img, attrs, target, ifmt, w, h, fmt, type, plane, out);
    if (win_on() || setup_room()) {
        resolve_gl();
        int name = (r == 0 && out && *out && p_cvname) ? (int)p_cvname(*out) : -1;
        if (name > 0) note_btex(name, w, h, ifmt);
        evw(E_CVTEX, (int[]){(int)target, ifmt, w, h, (int)fmt, (int)type, (int)plane, name, r}, 9, ra_, -1);
    }
    return r;
}
typedef void (*fn_CVOpenGLESTextureCacheFlush)(void *, uint64_t);
static void h_CVOpenGLESTextureCacheFlush(void *cache, uint64_t opt) {
    HOOK_ENTER(CVOpenGLESTextureCacheFlush);
    marker(M_CVGLFLUSH, ra_, -1);
    NOOP_RET(CVOpenGLESTextureCacheFlush, (void)0);
    O(CVOpenGLESTextureCacheFlush)(cache, opt);
}
typedef int (*fn_CVPixelBufferCreate)(void *, size_t, size_t, uint32_t, void *, void **);
static int h_CVPixelBufferCreate(void *al, size_t w, size_t h, uint32_t fmt, void *attrs, void **out) {
    HOOK_ENTER(CVPixelBufferCreate);
    NOOP_RET(CVPixelBufferCreate, -1);
    int r = O(CVPixelBufferCreate)(al, w, h, fmt, attrs, out);
    if (win_on() || setup_room()) evw(E_CVPIX, (int[]){(int)w, (int)h, (int)fmt, r}, 4, ra_, -1);
    return r;
}
typedef int (*fn_CVMetalTextureCacheCreateTextureFromImage)(void *, void *, void *, void *, unsigned long, size_t, size_t, size_t, void **);
static int h_CVMetalTextureCacheCreateTextureFromImage(void *al, void *cache, void *img, void *attrs, unsigned long pf, size_t w, size_t h,
                                                       size_t plane, void **out) {
    HOOK_ENTER(CVMetalTextureCacheCreateTextureFromImage);
    NOOP_RET(CVMetalTextureCacheCreateTextureFromImage, -1);
    int r = O(CVMetalTextureCacheCreateTextureFromImage)(al, cache, img, attrs, pf, w, h, plane, out);
    marker(M_CVMTLTEX, ra_, -1);
    if (win_on() || setup_room()) evw(E_CVMTL, (int[]){(int)pf, (int)w, (int)h, (int)plane, r}, 5, ra_, -1);
    return r;
}
typedef void (*fn_CVMetalTextureCacheFlush)(void *, uint64_t);
static void h_CVMetalTextureCacheFlush(void *cache, uint64_t opt) {
    HOOK_ENTER(CVMetalTextureCacheFlush);
    marker(M_CVMTLFLUSH, ra_, -1);
    NOOP_RET(CVMetalTextureCacheFlush, (void)0);
    O(CVMetalTextureCacheFlush)(cache, opt);
}

static void *const hooks[NF] = {
#define X(n) (void *)h_##n,
    FN_LIST(X)
#undef X
};

/* ------------------------------------------------------ ObjC (resolved at runtime, data-only swizzles) */
static struct {
    void *(*getClass)(const char *);
    void *(*getMetaClass)(const char *);
    void *(*getInstanceMethod)(void *, void *);
    void *(*getClassMethod)(void *, void *);
    void *(*setImp)(void *, void *);
    void *(*selReg)(const char *);
    const char *(*selName)(void *);
    void *(*methodName)(void *);
    void **(*copyMethodList)(void *, unsigned *);
    const char **(*classNamesForImage)(const char *, unsigned *);
    int ok;
} oc;
static void resolve_objc(void) {
    oc.getClass          = (void *(*)(const char *))rdlsym(RTLD_DEFAULT, "objc_getClass");
    oc.getMetaClass      = (void *(*)(const char *))rdlsym(RTLD_DEFAULT, "objc_getMetaClass");
    oc.getInstanceMethod = (void *(*)(void *, void *))rdlsym(RTLD_DEFAULT, "class_getInstanceMethod");
    oc.getClassMethod    = (void *(*)(void *, void *))rdlsym(RTLD_DEFAULT, "class_getClassMethod");
    oc.setImp            = (void *(*)(void *, void *))rdlsym(RTLD_DEFAULT, "method_setImplementation");
    oc.selReg            = (void *(*)(const char *))rdlsym(RTLD_DEFAULT, "sel_registerName");
    oc.selName           = (const char *(*)(void *))rdlsym(RTLD_DEFAULT, "sel_getName");
    oc.methodName        = (void *(*)(void *))rdlsym(RTLD_DEFAULT, "method_getName");
    oc.copyMethodList    = (void **(*)(void *, unsigned *))rdlsym(RTLD_DEFAULT, "class_copyMethodList");
    oc.classNamesForImage = (const char **(*)(const char *, unsigned *))rdlsym(RTLD_DEFAULT, "objc_copyClassNamesForImage");
    oc.ok = oc.getClass && oc.getInstanceMethod && oc.getClassMethod && oc.setImp && oc.selReg;
}
static int swizzle(const char *cname, int meta, const char *sname, void *newimp, void **old) {
    if (!oc.ok) return 0;
    void *cls = oc.getClass(cname);
    if (!cls) return 0;
    void *sel = oc.selReg(sname);
    void *m = meta ? oc.getClassMethod(cls, sel) : oc.getInstanceMethod(cls, sel);
    if (!m) return 0;
    *old = oc.setImp(m, newimp);
    return 1;
}
static unsigned char (*old_present)(void *, void *, unsigned long);
static unsigned char h_present(void *self, void *sel, unsigned long target) {
    marker(M_PRESENT, __builtin_return_address(0), K_PRESENT);
    return old_present ? old_present(self, sel, target) : 0;
}
static atomic_uint g_setctx_n, g_eaglstore_n;
static unsigned char (*old_setctx)(void *, void *, void *);
static unsigned char h_setctx(void *self, void *sel, void *ctx) {
    unsigned n = atomic_fetch_add(&g_setctx_n, 1u);
    if (n < 24) ev(E_SETCTX, (int[]){ctx != NULL, (int)((uintptr_t)ctx & 0xffffffffu), (int)n}, 3, __builtin_return_address(0), -1);
    return old_setctx ? old_setctx(self, sel, ctx) : 0;
}
static unsigned char (*old_rbstore)(void *, void *, unsigned long, void *);
static unsigned char h_rbstore(void *self, void *sel, unsigned long target, void *drawable) {
    void *ra = __builtin_return_address(0);
    unsigned char r = old_rbstore ? old_rbstore(self, sel, target, drawable) : 0;
    if (atomic_fetch_add(&g_eaglstore_n, 1u) < 12 && ctx_current()) {
        GLint rb = 0, w = 0, h = 0, f = 0;
        p_getint(0x8CA7, &rb);
        if (p_getrbp) { p_getrbp((GLenum)target, 0x8D42, &w); p_getrbp((GLenum)target, 0x8D43, &h); p_getrbp((GLenum)target, 0x8D44, &f); }
        ev(E_EAGLSTORE, (int[]){(int)target, rb, w, h, f, r}, 6, ra, -1);
    }
    return r;
}
static void *(*old_nextdrawable)(void *, void *);
static void *h_nextdrawable(void *self, void *sel) {
    marker(M_DRAWABLE, __builtin_return_address(0), K_DRAWABLE);
    return old_nextdrawable ? old_nextdrawable(self, sel) : NULL;
}
static void (*old_link)(void *, void *, void *, void *);
static void h_link(void *self, void *sel, void *link, void *upd) {
    marker(M_LINK, __builtin_return_address(0), -1);
    if (old_link) old_link(self, sel, link, upd);
}
static void install_objc(void) {
    resolve_objc();
    if (!oc.ok) { logf_("objc: runtime functions not resolved, no swizzles\n"); return; }
    int a = swizzle("EAGLContext", 0, "presentRenderbuffer:", (void *)h_present, (void **)&old_present);
    int b = swizzle("EAGLContext", 1, "setCurrentContext:", (void *)h_setctx, (void **)&old_setctx);
    int c = swizzle("EAGLContext", 0, "renderbufferStorage:fromDrawable:", (void *)h_rbstore, (void **)&old_rbstore);
    int d = swizzle("CAMetalLayer", 0, "nextDrawable", (void *)h_nextdrawable, (void **)&old_nextdrawable);
    int e = swizzle("ReSAIOSPresenterMetalLinkDelegate", 0, "metalDisplayLink:needsUpdate:", (void *)h_link, (void **)&old_link);
    logf_("objc swizzles: EAGLContext presentRenderbuffer:=%d  +setCurrentContext:=%d  renderbufferStorage:fromDrawable:=%d  "
          "CAMetalLayer nextDrawable=%d  ReSAIOSPresenterMetalLinkDelegate metalDisplayLink:needsUpdate:=%d\n", a, b, c, d, e);
}
static int find_image_path(const char *needle, char *out, size_t cap) {
    uint32_t n = _dyld_image_count();
    for (uint32_t i = 0; i < n; i++) {
        const char *nm = _dyld_get_image_name(i);
        if (nm && strstr(nm, needle) && !strstr(nm, "resa_probe")) { snprintf(out, cap, "%s", nm); return 1; }
    }
    return 0;
}
static int name_interesting(const char *s) {
    static const char *const kw[] = { "Present", "Metal", "Link", "GL", "Render", "View", "Layer", "Context", "Display", "Frame",
                                      "Engine", "Surface", "Shader", "Post", "Bridge", "Draw", "Graphics", NULL };
    for (int i = 0; kw[i]; i++) if (strstr(s, kw[i])) return 1;
    return 0;
}
static void dump_objc_landscape(void) {
    if (!oc.ok || !oc.classNamesForImage) { logf_("[objc] class list unavailable\n"); return; }
    char path[512];
    if (!find_image_path("alyn_resa", path, sizeof path)) { logf_("[objc] alyn_resa image not found\n"); return; }
    unsigned n = 0;
    const char **names = oc.classNamesForImage(path, &n);
    logf_("\n==== OBJC CLASSES in alyn_resa: %u ====\n", n);
    if (!names) return;
    char line[160]; line[0] = 0;
    for (unsigned i = 0; i < n && i < 400; i++) {
        if (strlen(line) + strlen(names[i]) + 2 > 110) { logf_("  %s\n", line); line[0] = 0; }
        app(line, sizeof line, "%s ", names[i]);
    }
    if (line[0]) logf_("  %s\n", line);
    int shown = 0;
    for (unsigned i = 0; i < n && shown < 24; i++) {
        if (!name_interesting(names[i]) && strncmp(names[i], "ReSA", 4) != 0) continue;
        void *cls = oc.getClass(names[i]);
        if (!cls || !oc.copyMethodList || !oc.methodName || !oc.selName) continue;
        for (int meta = 0; meta < 2; meta++) {
            void *kc = meta ? (oc.getMetaClass ? oc.getMetaClass(names[i]) : NULL) : cls;
            if (!kc) continue;
            unsigned mn = 0;
            void **ml = oc.copyMethodList(kc, &mn);
            if (!ml || !mn) { free(ml); continue; }
            char ms[900]; ms[0] = 0;
            for (unsigned k = 0; k < mn && k < 36; k++) app(ms, sizeof ms, "%s ", oc.selName(oc.methodName(ml[k])));
            logf_("  %c[%s] (%u): %s%s\n", meta ? '+' : '-', names[i], mn, ms, mn > 36 ? "..." : "");
            free(ml);
        }
        shown++;
    }
    free((void *)names);
}

/* ------------------------------------------------------ data-segment scan: GL function pointers cached by the engine */
typedef struct { uintptr_t addr; int f; } GA;
static int ga_cmp(const void *a, const void *b) {
    uintptr_t x = ((const GA *)a)->addr, y = ((const GA *)b)->addr;
    return x < y ? -1 : (x > y ? 1 : 0);
}
static void scan_gl_ptrs(const char *why) {
    GA tab[NF_GL];
    int n = 0;
    for (int f = 0; f < NF_GL; f++) {
        void *p = rdlsym(RTLD_DEFAULT, fnames[f]);
        if (!p) continue;
        int dup = 0;
        for (int i = 0; i < n; i++) if (tab[i].addr == (uintptr_t)p) dup = 1;
        if (dup) continue;
        tab[n].addr = (uintptr_t)p; tab[n].f = f; n++;
    }
    if (!n) { logf_("[scan %s] no GL symbols resolved\n", why); return; }
    qsort(tab, (size_t)n, sizeof tab[0], ga_cmp);
    uintptr_t tmin = tab[0].addr, tmax = tab[n - 1].addr;
    unsigned hits = 0, patched = 0;
    unsigned long long scanned = 0, skipped = 0;
    int ni = atomic_load(&g_nimg), any = 0;
    long pg = sysconf(_SC_PAGESIZE);
    if (pg < 4096) pg = 4096;
    for (int ii = 0; ii < ni && ii < NIMG; ii++) {
        Img *im = &g_img[ii];
        if (!strstr(im->name, "alyn_resa")) continue;
        any = 1;
        for (int d = 0; d < im->nd; d++) {
            uintptr_t lo = im->dlo[d] & ~(uintptr_t)(pg - 1), hi = (im->dhi[d] + (uintptr_t)pg - 1) & ~(uintptr_t)(pg - 1);
            size_t npages = (size_t)((hi - lo) / (uintptr_t)pg);
            if (!npages || npages > 65536) continue;
            unsigned char *vec = (unsigned char *)calloc(npages, 1);
            if (!vec) continue;
#ifdef RESA_TEST
            int mc = mincore((void *)lo, hi - lo, vec);
#else
            int mc = mincore((void *)lo, hi - lo, (char *)vec);
#endif
            for (size_t pgi = 0; pgi < npages; pgi++) {
                if (mc == 0 && !(vec[pgi] & 1)) { skipped += (unsigned long long)pg; continue; }   /* never touched: do not fault it in */
                uintptr_t *p = (uintptr_t *)(lo + pgi * (size_t)pg), *e = (uintptr_t *)(lo + (pgi + 1) * (size_t)pg);
                scanned += (unsigned long long)pg;
                for (; p < e; p++) {
                    uintptr_t v = *p;
                    if (v < tmin || v > tmax) continue;
                    int a = 0, b = n - 1, hit = -1;
                    while (a <= b) { int mid = (a + b) / 2; if (tab[mid].addr == v) { hit = mid; break; } if (tab[mid].addr < v) a = mid + 1; else b = mid - 1; }
                    if (hit < 0) continue;
                    int f = tab[hit].f;
                    hits++;
                    atomic_fetch_add(&scan_hit[f], 1u);
                    char loc[64];
                    fmt_ra((const void *)p, loc, sizeof loc);
                    if (hits <= 60) logf_("  [scan %s] %-34s pointer stored at %s  -> re-pointed to hook\n", why, fnames[f], loc);
                    *p = (uintptr_t)hooks[f];
                    patched++;
                }
            }
            free(vec);
        }
    }
    logf_("[scan %s] alyn_resa found=%d  scanned=%llu KB (untouched pages skipped=%llu KB)  GL pointers found=%u  re-pointed=%u\n",
          why, any, scanned / 1024ull, skipped / 1024ull, hits, patched);
}

/* ------------------------------------------------------ report formatting */
static void fmt_event(const Ev *e, char *o, size_t cap) {
    char b1[16], b2[16], b3[16], b4[16], fl[80];
    const int *v = e->v;
    switch (e->type) {
    case E_SEG:    snprintf(o, cap, "   SEG draws=%d verts=%d progs=[p%d:%d p%d:%d p%d:%d] distinct=%d%s  state: %s", v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
                            v[9] % 100, v[9] >= 100 ? "+" : "", fmt_flags((unsigned)v[8], fl, sizeof fl)); break;
    case E_BIND:   snprintf(o, cap, "bindFramebuffer(%s, %d)", ename((unsigned)v[0], b1), v[1]); break;
    case E_FBRB:   snprintf(o, cap, "framebufferRenderbuffer(%s, %s, %s, rb=%d)   [boundFBO=%d]",
                            ename((unsigned)v[0], b1), ename((unsigned)v[1], b2), ename((unsigned)v[2], b3), v[3], v[4]); break;
    case E_FBTEX:  snprintf(o, cap, "framebufferTexture2D(%s, %s, %s, tex=%d, lvl=%d)   [boundFBO=%d]",
                            ename((unsigned)v[0], b1), ename((unsigned)v[1], b2), ename((unsigned)v[2], b3), v[3], v[4], v[5]); break;
    case E_RBSTORE:snprintf(o, cap, "renderbufferStorage(rb=%d %dx%d fmt=%s samples=%d)", v[5], v[3], v[4], ename((unsigned)v[2], b1), v[1]); break;
    case E_DISCARD:
    case E_INVAL: {
        char l[120]; l[0] = 0;
        for (int i = 0; i < 3 && i < v[1]; i++) app(l, sizeof l, "%s ", ename((unsigned)v[2 + i], b1));
        snprintf(o, cap, "%s(%s, n=%d: %s)", e->type == E_DISCARD ? "DISCARD" : "INVALIDATE", ename((unsigned)v[0], b2), v[1], l);
        break; }
    case E_RESOLVE:snprintf(o, cap, "RESOLVE_MSAA_APPLE()"); break;
    case E_BLIT:   snprintf(o, cap, "BLIT src(%d,%d,%d,%d) -> dst(%d,%d,%d,%d) mask=%s filter=%s",
                            v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], maskname((unsigned)v[8], b1), ename((unsigned)v[9], b2)); break;
    case E_DRAWBUF:snprintf(o, cap, "drawBuffers(n=%d: 0x%x 0x%x 0x%x)", v[0], (unsigned)v[1], (unsigned)v[2], (unsigned)v[3]); break;
    case E_READPIX:snprintf(o, cap, "readPixels(%d,%d %dx%d fmt=%s type=%s) [fbo=%d]", v[0], v[1], v[2], v[3], ename((unsigned)v[4], b1), ename((unsigned)v[5], b2), v[6]); break;
    case E_CLEAR:  snprintf(o, cap, "clear(%s) [fbo=%d]", maskname((unsigned)v[0], b1), v[1]); break;
    case E_TEXBIG: snprintf(o, cap, "TEXTURE %s tex=%d %dx%d ifmt=%s fmt=%s type=%s levels=%d", v[0] ? "texStorage2D" : "texImage2D", v[6], v[2], v[3],
                            ename((unsigned)v[1], b1), ename((unsigned)v[4], b2), ename((unsigned)v[5], b3), v[7]); (void)b4; break;
    case E_COPYTEX:snprintf(o, cap, "%s %dx%d from(%d,%d) [fbo=%d]", v[0] ? "copyTexSubImage2D" : "copyTexImage2D", v[2], v[3], v[4], v[5], v[6]); break;
    case E_CAP:    snprintf(o, cap, "%s(%s)", v[1] ? "enable" : "disable", ename((unsigned)v[0], b1)); break;
    case E_DEPTHMASK: snprintf(o, cap, "depthMask(%d)", v[0]); break;
    case E_DEPTHFUNC: snprintf(o, cap, "depthFunc(0x%x)", (unsigned)v[0]); break;
    case E_VIEWPORT:  snprintf(o, cap, "viewport(%d,%d,%d,%d) [fbo=%d]", v[0], v[1], v[2], v[3], v[4]); break;
    case E_DEPTHRANGE:snprintf(o, cap, "depthRangef(%.4f, %.4f)", v[0] / 1e6, v[1] / 1e6); break;
    case E_CLEARDEPTH:snprintf(o, cap, "clearDepthf(%.4f)", v[0] / 1e6); break;
    case E_MARK:   snprintf(o, cap, "=== MARK: %s", (v[0] >= 0 && v[0] < M_N) ? mk_name[v[0]] : "?"); break;
    case E_CVTEX:  snprintf(o, cap, "CVOpenGLESTextureCache...CreateTexture target=%s ifmt=%s %dx%d fmt=%s type=%s plane=%d -> tex=%d rc=%d",
                            ename((unsigned)v[0], b1), ename((unsigned)v[1], b2), v[2], v[3], ename((unsigned)v[4], b3), ename((unsigned)v[5], b4), v[6], v[7], v[8]); break;
    case E_CVPIX:  snprintf(o, cap, "CVPixelBufferCreate %dx%d format=%s rc=%d", v[0], v[1], fourcc((uint32_t)v[2], b1), v[3]); break;
    case E_CVMTL:  snprintf(o, cap, "CVMetalTextureCacheCreateTexture mtlPixelFormat=%d %dx%d plane=%d rc=%d", v[0], v[1], v[2], v[3], v[4]); break;
    case E_SETCTX: snprintf(o, cap, "+EAGLContext setCurrentContext:(%s ctx=0x%x) #%d", v[0] ? "set" : "nil", (unsigned)v[1], v[2]); break;
    case E_EAGLSTORE: snprintf(o, cap, "EAGL renderbufferStorage:fromDrawable: (%s) boundRB=%d -> %dx%d %s rc=%d", ename((unsigned)v[0], b1), v[1], v[2], v[3], ename((unsigned)v[4], b2), v[5]); break;
    default:       snprintf(o, cap, "event type %d", e->type); break;
    }
}
static void dump_events(const Ev *arr, unsigned n, int with_snaps) {
    char line[520], ra[80];
    unsigned ns = atomic_load(&g_snap_n);
    if (ns > SNAP_CAP) ns = SNAP_CAP;
    for (unsigned i = 0; i < n; i++) {
        const Ev *e = &arr[i];
        fmt_event(e, line, sizeof line);
        if (e->ra) { fmt_ra((const void *)e->ra, ra, sizeof ra); logf_("%8.3f t%04x %s    @%s\n", (double)e->t_us / 1e6, e->tid & 0xffff, line, ra); }
        else       logf_("%8.3f t%04x %s\n", (double)e->t_us / 1e6, e->tid & 0xffff, line);
        if (with_snaps && e->snap >= 0 && (unsigned)e->snap < ns) logf_("             -> %s\n", g_snaps[e->snap]);
    }
}

static unsigned g_setup_dumped;
static double g_prev_t;
static unsigned long long g_prev_calls[NF], g_prev_mk[M_N];
static void dump_projection(void) {
    unsigned n = atomic_load(&g_pm_n);
    if (n > 10) n = 10;
    if (!n) { logf_("projection-like matrices: none seen yet\n"); return; }
    logf_("projection-like matrices seen via glUniformMatrix4fv (tag A = last row (0,0,+-1,0) in column-major memory, B = same in row-major):\n");
    for (unsigned i = 0; i < n; i++) {
        const float *m = g_pm[i].m;
        logf_("  prog=%u loc=%d transpose=%d tag=%c seen=%u  [%.6g %.6g %.6g %.6g | %.6g %.6g %.6g %.6g | %.6g %.6g %.6g %.6g | %.6g %.6g %.6g %.6g]\n",
              g_pm[i].prog, g_pm[i].loc, g_pm[i].tr, g_pm[i].tag, g_pm[i].cnt,
              m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15]);
    }
}
static void dump_dlsym_new(void) {
    unsigned n = atomic_load(&g_ds_n);
    if (n > DS_CAP) n = DS_CAP;
    if (n <= g_ds_dumped) { logf_("dlsym: no new names (total calls so far=%llu)\n", (unsigned long long)atomic_load(&g_ds_total)); return; }
    logf_("dlsym wrapper: total calls=%llu, new distinct names=%u\n", (unsigned long long)atomic_load(&g_ds_total), n - g_ds_dumped);
    char other[1500]; other[0] = 0;
    char ra[80];
    for (unsigned i = g_ds_dumped; i < n; i++) {
        const DsEnt *e = &g_ds[i];
        int gl = (e->name[0] == 'g' && e->name[1] == 'l') || !strncmp(e->name, "CV", 2) || !strncmp(e->name, "egl", 3) || !strncmp(e->name, "EAGL", 4) ||
                 !strncmp(e->name, "MTL", 3) || !strncmp(e->name, "objc", 4) || !strncmp(e->name, "CA", 2) || !strncmp(e->name, "SDL", 3);
        if (gl) {
            fmt_ra(e->ra, ra, sizeof ra);
            logf_("  [%7.3fs] %-40s %s  x%u  from %s\n", e->t_ms / 1000.0, e->name, e->wrapped ? "WRAPPED" : (e->nul ? "(null)" : "plain  "), e->cnt, ra);
        } else if (strlen(other) < 1300) {
            app(other, sizeof other, "%s ", e->name);
        }
    }
    if (other[0]) logf_("  other names looked up: %s\n", other);
    g_ds_dumped = n;
}
static void dump_images(void) {
    int n = atomic_load(&g_nimg);
    logf_("app images (offsets in call-sites below are runtime address minus slide):\n");
    for (int i = 0; i < n && i < NIMG; i++) {
        logf_("  %-28s slide=0x%lx  hdr=0x%lx  data-ranges=%d\n", g_img[i].name, (unsigned long)g_img[i].slide, (unsigned long)g_img[i].hdr, g_img[i].nd);
    }
}
static void dump_status(const char *why) {
    double t = since_s(), dt = t - g_prev_t;
    if (dt < 0.05) dt = 0.05;
    logf_("\n==== STATUS: %s  (t=%.1fs, %.1fs since previous status) ====\n", why, t, dt);
    if (atomic_load(&g_gl_info_done))
        logf_("GL: version=\"%s\" renderer=\"%s\" vendor=\"%s\"\n    ext: %s\n", g_gl_version, g_gl_renderer, g_gl_vendor, g_gl_ext);
    else
        logf_("GL: no hooked GL call has run with a current context yet\n");
    logf_("hook table  (calls | rate/s since last status | imp = an image imports it (fishhook) | dl = times the dlsym wrapper handed out our hook | scan = pointer slots re-pointed)\n");
    char none[1500]; none[0] = 0;
    for (int f = 0; f < NF; f++) {
        unsigned long long c = atomic_load(&calls[f]);
        unsigned dh = atomic_load(&dl_hand[f]), sh = atomic_load(&scan_hit[f]);
        int bound = fh_orig[f] != NULL;
        if (!c && !bound && !dh && !sh) { app(none, sizeof none, "%s ", fnames[f]); continue; }
        char cb[260]; cb[0] = 0;
        while (atomic_flag_test_and_set_explicit(&c_lock, memory_order_acquire)) { }
        for (int i = 0; i < c_n[f]; i++) app(cb, sizeof cb, "%s%s", i ? ", " : "", c_name[f][i]);
        atomic_flag_clear_explicit(&c_lock, memory_order_release);
        logf_("  %-44s calls=%-9llu %8.0f/s  imp=%s dl=%u scan=%u  callers=[%s]\n", fnames[f], c, (double)(c - g_prev_calls[f]) / dt, bound ? "Y" : "n", dh, sh, cb);
        g_prev_calls[f] = c;
    }
    logf_("  never imported / looked up / called: %s\n", none[0] ? none : "(none)");
    logf_("frame markers (the one running at ~display rate is the frame boundary):\n");
    for (int m = 0; m < M_N; m++) {
        unsigned long long c = atomic_load(&mk_cnt[m]);
        if (c) logf_("  %-44s total=%-9llu %8.1f/s  last-thread=t%04x\n", mk_name[m], c, (double)(c - g_prev_mk[m]) / dt, mk_tid[m] & 0xffff);
        g_prev_mk[m] = c;
    }
    g_prev_t = t;
    dump_dlsym_new();
    dump_projection();
    unsigned sn = atomic_load(&setup_n);
    if (sn > SETUP_CAP) sn = SETUP_CAP;
    if (sn > g_setup_dumped) {
        logf_("-- setup log (render targets, EAGL, CoreVideo bridge, big/depth textures), new entries --\n");
        dump_events(setup_log + g_setup_dumped, sn - g_setup_dumped, 0);
        g_setup_dumped = sn;
    }
    for (int k = 0; k < K_N; k++)
        for (; first_dumped[k] < 2 && first_dumped[k] < (int)atomic_load(&first_cnt[k]); first_dumped[k]++) {
            const FirstSnap *fs = &g_first[k][first_dumped[k]];
            if (!fs->ok) break;
            char ra[80];
            fmt_ra((const void *)fs->ra, ra, sizeof ra);
            logf_("-- first %s #%d at %.3fs t%04x @%s --\n   %s\n", k_name[k], first_dumped[k] + 1, (double)fs->t_us / 1e6, fs->tid & 0xffff, ra, fs->text);
        }
}
static void dump_window(int k, double t_open, double t_close) {
    unsigned n = atomic_load(&win_n);
    if (n > WIN_CAP) n = WIN_CAP;
    unsigned cnt[E_NTYPES], mk[M_N];
    memset(cnt, 0, sizeof cnt);
    memset(mk, 0, sizeof mk);
    unsigned long long dsum = 0;
    for (unsigned i = 0; i < n; i++) {
        if (win_log[i].type < E_NTYPES) cnt[win_log[i].type]++;
        if (win_log[i].type == E_MARK && win_log[i].v[0] >= 0 && win_log[i].v[0] < M_N) mk[win_log[i].v[0]]++;
        if (win_log[i].type == E_SEG) dsum += (unsigned)win_log[i].v[0];
    }
    logf_("\n==== TRACE WINDOW %d   t=%.3fs..%.3fs   events=%u%s ====\n", k + 1, t_open, t_close, n, n >= WIN_CAP ? " (cap hit)" : "");
    logf_("summary: drawsInSegments=%llu segments=%u bind=%u fbRB=%u fbTex=%u clear=%u discard=%u invalidate=%u resolve=%u blit=%u copyTex=%u readPixels=%u "
          "depthTestToggles=%u depthMaskToggles=%u viewport=%u\n", dsum, cnt[E_SEG], cnt[E_BIND], cnt[E_FBRB], cnt[E_FBTEX], cnt[E_CLEAR], cnt[E_DISCARD], cnt[E_INVAL],
          cnt[E_RESOLVE], cnt[E_BLIT], cnt[E_COPYTEX], cnt[E_READPIX], cnt[E_CAP], cnt[E_DEPTHMASK], cnt[E_VIEWPORT]);
    logf_("markers in window:");
    for (int m = 0; m < M_N; m++) if (mk[m]) logf_("  [%s]=%u", mk_name[m], mk[m]);
    logf_("\ncolumns: time  tid  event  @call-site (image+unslid offset).  SEG = a run of draw calls between two other events.\n");
    dump_events(win_log, n, 1);
}

/* ------------------------------------------------------ Mach-O import scanner (as v2, plus image table) */
static int interesting(const char *s) {
    static const char *const pre[] = { "_gl", "_egl", "_EGL", "_CVOpenGLES", "_CVMetal", "_CVPixelBuffer", "_dlsym", "_dlopen", "_CFBundleGetFunctionPointer",
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
            app(links, sizeof links, "%s ", basename_((const char *)lc + dc->dylib.name.offset));
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
    img_register(h, slide, name);
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
}

/* ------------------------------------------------------ install hooks (fishhook on app images only) */
static const void *g_self_hdr;
static struct rebinding g_rb[NF + 1];
static int g_nrb;
static int image_is_app(const struct mach_header_64 *h) {
    if (h->magic != MH_MAGIC_64) return 0;
    if (h->filetype == MH_EXECUTE) return 1;
    const uint8_t *p = (const uint8_t *)(h + 1);
    for (uint32_t i = 0; i < h->ncmds; i++) {
        const struct load_command *lc = (const struct load_command *)p;
        if (lc->cmdsize < 8) break;
        if (lc->cmd == LC_ID_DYLIB) {
            const struct dylib_command *dc = (const struct dylib_command *)lc;
            return ((const char *)lc + dc->dylib.name.offset)[0] == '@';     /* @rpath/... = bundled framework, not a system library */
        }
        p += lc->cmdsize;
    }
    return 0;
}
static atomic_int g_rebind_images;
static void on_add_image(const struct mach_header *mh, intptr_t slide) {
    if ((const void *)mh == g_self_hdr) return;
    if (!image_is_app((const struct mach_header_64 *)mh)) return;
    rebind_symbols_image((void *)mh, slide, g_rb, (size_t)g_nrb);
    atomic_fetch_add(&g_rebind_images, 1);
}
static void install_hooks(void) {
    for (int i = 0; i < NF; i++) { g_rb[i].name = fnames[i]; g_rb[i].replacement = hooks[i]; g_rb[i].replaced = &fh_orig[i]; }
    g_rb[NF].name = "dlsym"; g_rb[NF].replacement = (void *)h_dlsym; g_rb[NF].replaced = (void **)&real_dlsym;
    g_nrb = NF + 1;
    Dl_info di;
    g_self_hdr = (dladdr((void *)install_hooks, &di)) ? di.dli_fbase : NULL;
    _dyld_register_func_for_add_image(on_add_image);             /* runs for images already loaded, then for each new one */
    logf_("hooks: fishhook rebinding of %d names + dlsym applied to %d app-bundle images so far (later images are patched as they load)\n",
          NF, atomic_load(&g_rebind_images));
    logf_("hooks: dlsym wrapper %s\n", real_dlsym ? "active (original dlsym captured)" : "NOT captured (no app image imports dlsym?)");
}

/* ------------------------------------------------------ watchdog thread */
static void *watchdog_main(void *arg) {
    (void)arg;
    int b_load = 0, b_gl = 0, b_first = 0, st6 = 0, st20 = 0, objc_done = 0, k = 0, open = 0, armed = 0, scans = 0, done = 0;
    double t_open = 0, t_arm = 0, earliest = WIN_MIN_START[0], t_due = -1, last_busy_check = 0;
    int busy_secs = 0;
    uint32_t last_imgs = 0;
    unsigned long long last_draws = 0;
    for (;;) {
        usleep(100 * 1000);
        double t = since_s();
        uint32_t ni = _dyld_image_count();
        if (ni != last_imgs && t >= 1.0) { last_imgs = ni; scan_new_images(); }
        if (!b_load && t >= 3.0) { b_load = 1; buzz(1); }
        else if (b_load && !b_gl && atomic_load(&g_gl_seen)) { b_gl = 1; buzz(2); }
        if (!objc_done && t >= 3.5) { objc_done = 1; dump_objc_landscape(); }
        if (!st6 && t >= 6.0)  { st6 = 1;  dump_images(); dump_status("early"); }
        if (!st20 && t >= 20.0) { st20 = 1; dump_status("20s"); }
        if (scans < 3 && t >= SCAN_AT[scans]) { scan_gl_ptrs(scans == 0 ? "1" : (scans == 1 ? "2" : "3")); scans++; }
        if (t - last_busy_check >= 1.0) {                         /* "is the 3D scene running?" = draw calls per second */
            unsigned long long d = 0;
            for (int f = F_glDrawArrays; f <= F_glDrawRangeElements; f++) d += atomic_load(&calls[f]);
            double dps = (double)(d - last_draws) / (t - last_busy_check);
            last_draws = d; last_busy_check = t;
            busy_secs = dps >= BUSY_DPS ? busy_secs + 1 : 0;
        }
        if (done) continue;
        if (!open && !armed && k < NWIN && t >= earliest) {
            if (t_due < 0) t_due = t;
            if (busy_secs >= BUSY_NEED || t >= t_due + WIN_WAIT) { atomic_store(&win_arm, 1); armed = 1; t_arm = t; }
        }
        if (armed && !open) {
            if (win_on()) { open = 1; armed = 0; t_open = since_s(); }
            else if (t - t_arm > 2.5) { atomic_store(&win_arm, 0); open_window(); open = 1; armed = 0; t_open = since_s(); }   /* no marker ever fired */
        } else if (open && (atomic_load(&win_n) >= WIN_CAP || (double)(now_us() - g_win_t0_us) / 1e6 > WIN_MAXLEN)) {
            atomic_store(&win_active, 0);
            usleep(150 * 1000);
            dump_window(k, t_open, since_s());
            dump_status("after window");
            if (k == 0) buzz(3);
            open = 0; k++; t_due = -1; busy_secs = 0;
            earliest = (k < NWIN) ? (WIN_MIN_START[k] > t + WIN_GAP ? WIN_MIN_START[k] : t + WIN_GAP) : 1e9;
            if (k >= NWIN) { done = 1; scan_gl_ptrs("final"); dump_status("final"); buzz(4); }
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
    install_objc();
    pthread_t th;
    if (pthread_create(&th, NULL, watchdog_main, NULL) == 0) pthread_detach(th);
    else logf_("pthread_create failed\n");
}
