/*
 * resa_ao.c  --  screen-space ambient occlusion (SSAO) for ReSA on iOS.   v1.0
 *
 * Injected dylib.  Pure C, no link dependency beyond libSystem/libobjc (GL + UIKit/EAGL are resolved at runtime).
 *
 * WHAT THE v3 PROBE LOG (resa_probe.txt, iPhone17,1) SHOWED, AND WHAT THIS RELIES ON
 *   - The engine (alyn_resa) reaches most GL entry points through dlsym()-resolved pointers (glUniformMatrix4fv is
 *     never imported, only looked up) and a few directly (glInvalidateFramebuffer is BOTH imported and looked up).
 *     So we hook two ways: a dlsym() wrapper (fishhook rebinding of "dlsym") and fishhook rebinding of the GL names.
 *   - Everything GL runs on the 'ReSA.Render' thread, with a current EAGLContext (OpenGL ES 3.0 over Metal).
 *   - Every frame the world is drawn into ONE offscreen scene FBO (fbo 47 in the log):
 *         colour = RGBA8 texture, depth+stencil = one DEPTH24_STENCIL8 renderbuffer, no MSAA, 1136x524 (render scale)
 *     and the last thing done to that FBO is
 *         glInvalidateFramebuffer(FRAMEBUFFER, 2, {DEPTH_ATTACHMENT, STENCIL_ATTACHMENT})
 *     Then the engine binds the full-resolution target (fbo 2/3), invalidates {COLOR0, DEPTH, STENCIL} on it (we must
 *     NOT run on that one: it contains colour), up-scales the scene texture into it and draws the HUD.
 *   => That first invalidate is the seam: the world is complete, depth is still valid, the HUD does not exist yet.
 *      We run AO right before it:  copy depth+colour -> AO -> 2 depth-aware blur passes -> write the result back into
 *      the scene colour texture.  Then the engine's own invalidate runs as usual, and its up-scale picks up our result.
 *   - Projection: the engine uploads a plain perspective matrix with glUniformMatrix4fv (transpose=0, column-major,
 *     w = +z, GL-style depth: ndc = A + B/z with A=m[10], B=m[14]; e.g. 0.8758/1.9042/1.00237/-1.8998 in the log).
 *     near/far drift continuously (0.9..0.99 / 800..1500) so we read the matrix every frame instead of guessing.
 *
 * SAFETY RULES
 *   - Every GL object we use is our own; all GL state we touch is saved first and restored exactly afterwards.
 *   - Any failure (shader compile, incomplete FBO, GL error twice in a row) switches AO off for the rest of the run
 *     and says why in Documents/resa_ao_log.txt.  The game keeps running either way.
 *
 * USER CONTROLS
 *   Documents/resa_ao.ini   (created on first run, re-read about every 1.5 s, so you can edit it while playing)
 *   Touch gestures (all five fingers on the screen at once, then lift):
 *       quick tap  = AO on/off
 *       hold >1.2s = cycle debug view  (AO -> AO only -> depth -> normals -> AO)
 *   (touch_toggle=0 in the ini disables the gestures)
 *
 * Build: see .github/workflows/build.yml.   Test build (Linux, Mesa llvmpipe): -DRESA_TEST, see tests/.
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

#ifndef RESA_TEST
#  include <mach-o/dyld.h>
#  include <sys/sysctl.h>
#  include <objc/runtime.h>
#  include <objc/message.h>
#  include "fishhook.h"
#endif

#define AO_VERSION "v1.1"

/* ------------------------------------------------------------------ GL types + constants (checked against the Khronos headers by tests/check_consts.py) */
typedef unsigned int  GLenum;
typedef unsigned int  GLuint;
typedef unsigned int  GLbitfield;
typedef unsigned char GLboolean;
typedef unsigned char GLubyte;
typedef int           GLint;
typedef int           GLsizei;
typedef float         GLfloat;
typedef char          GLchar;

#define GLC_NONE                               0
#define GLC_TEXTURE_2D                         0x0DE1
#define GLC_FRAMEBUFFER                        0x8D40
#define GLC_READ_FRAMEBUFFER                   0x8CA8
#define GLC_DRAW_FRAMEBUFFER                   0x8CA9
#define GLC_RENDERBUFFER                       0x8D41
#define GLC_COLOR_ATTACHMENT0                  0x8CE0
#define GLC_DEPTH_ATTACHMENT                   0x8D00
#define GLC_STENCIL_ATTACHMENT                 0x8D20
#define GLC_DEPTH_STENCIL_ATTACHMENT           0x821A
#define GLC_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE 0x8CD0
#define GLC_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME 0x8CD1
#define GLC_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE  0x8216
#define GLC_FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE 0x8217
#define GLC_FRAMEBUFFER_COMPLETE               0x8CD5
#define GLC_RENDERBUFFER_WIDTH                 0x8D42
#define GLC_RENDERBUFFER_HEIGHT                0x8D43
#define GLC_RENDERBUFFER_SAMPLES               0x8CAB
#define GLC_RENDERBUFFER_BINDING               0x8CA7
#define GLC_DRAW_FRAMEBUFFER_BINDING           0x8CA6
#define GLC_READ_FRAMEBUFFER_BINDING           0x8CAA
#define GLC_TEXTURE_BINDING_2D                 0x8069
#define GLC_ACTIVE_TEXTURE                     0x84E0
#define GLC_TEXTURE0                           0x84C0
#define GLC_SAMPLER_BINDING                    0x8919
#define GLC_CURRENT_PROGRAM                    0x8B8D
#define GLC_VERTEX_ARRAY_BINDING               0x85B5
#define GLC_VIEWPORT                           0x0BA2
#define GLC_DEPTH_TEST                         0x0B71
#define GLC_STENCIL_TEST                       0x0B90
#define GLC_BLEND                              0x0BE2
#define GLC_CULL_FACE                          0x0B44
#define GLC_SCISSOR_TEST                       0x0C11
#define GLC_DITHER                             0x0BD0
#define GLC_POLYGON_OFFSET_FILL                0x8037
#define GLC_RASTERIZER_DISCARD                 0x8C89
#define GLC_DEPTH_WRITEMASK                    0x0B72
#define GLC_COLOR_WRITEMASK                    0x0C23
#define GLC_COLOR_BUFFER_BIT                   0x4000
#define GLC_DEPTH_BUFFER_BIT                   0x0100
#define GLC_STENCIL_BUFFER_BIT                 0x0400
#define GLC_NEAREST                            0x2600
#define GLC_TEXTURE_MIN_FILTER                 0x2801
#define GLC_TEXTURE_MAG_FILTER                 0x2800
#define GLC_TEXTURE_WRAP_S                     0x2802
#define GLC_TEXTURE_WRAP_T                     0x2803
#define GLC_CLAMP_TO_EDGE                      0x812F
#define GLC_RGBA                               0x1908
#define GLC_UNSIGNED_BYTE                      0x1401
#define GLC_RGBA8                              0x8058
#define GLC_DEPTH24_STENCIL8                   0x88F0
#define GLC_DEPTH_COMPONENT24                  0x81A6
#define GLC_DEPTH_COMPONENT16                  0x81A5
#define GLC_DEPTH_COMPONENT32F                 0x8CAC
#define GLC_DEPTH32F_STENCIL8                  0x8CAD
#define GLC_TEXTURE                            0x1702
#define GLC_VERTEX_SHADER                      0x8B31
#define GLC_FRAGMENT_SHADER                    0x8B30
#define GLC_COMPILE_STATUS                     0x8B81
#define GLC_LINK_STATUS                        0x8B82
#define GLC_TRIANGLES                          0x0004
#define GLC_VERSION                            0x1F02
#define GLC_RENDERER                           0x1F01
#define GLC_PACK_ALIGNMENT                     0x0D05
#define GLC_PACK_ROW_LENGTH                    0x0D02
#define GLC_PACK_SKIP_ROWS                     0x0D03
#define GLC_PACK_SKIP_PIXELS                   0x0D04
#define GLC_PIXEL_PACK_BUFFER_BINDING          0x88ED

/* the GL entry points we call ourselves.  Resolved with the REAL dlsym, so they never go through our own hooks. */
#define GLFN(X) \
    X(void,     BindFramebuffer,        (GLenum, GLuint)) \
    X(void,     GenFramebuffers,        (GLsizei, GLuint *)) \
    X(void,     DeleteFramebuffers,     (GLsizei, const GLuint *)) \
    X(void,     FramebufferTexture2D,   (GLenum, GLenum, GLenum, GLuint, GLint)) \
    X(GLenum,   CheckFramebufferStatus, (GLenum)) \
    X(void,     GenTextures,            (GLsizei, GLuint *)) \
    X(void,     DeleteTextures,         (GLsizei, const GLuint *)) \
    X(void,     BindTexture,            (GLenum, GLuint)) \
    X(void,     TexStorage2D,           (GLenum, GLsizei, GLenum, GLsizei, GLsizei)) \
    X(void,     TexParameteri,          (GLenum, GLenum, GLint)) \
    X(void,     ActiveTexture,          (GLenum)) \
    X(void,     BindSampler,            (GLuint, GLuint)) \
    X(void,     BlitFramebuffer,        (GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLbitfield, GLenum)) \
    X(void,     GenVertexArrays,        (GLsizei, GLuint *)) \
    X(void,     DeleteVertexArrays,     (GLsizei, const GLuint *)) \
    X(void,     BindVertexArray,        (GLuint)) \
    X(GLuint,   CreateShader,           (GLenum)) \
    X(void,     ShaderSource,           (GLuint, GLsizei, const GLchar *const *, const GLint *)) \
    X(void,     CompileShader,          (GLuint)) \
    X(void,     GetShaderiv,            (GLuint, GLenum, GLint *)) \
    X(void,     GetShaderInfoLog,       (GLuint, GLsizei, GLsizei *, GLchar *)) \
    X(GLuint,   CreateProgram,          (void)) \
    X(void,     AttachShader,           (GLuint, GLuint)) \
    X(void,     LinkProgram,            (GLuint)) \
    X(void,     GetProgramiv,           (GLuint, GLenum, GLint *)) \
    X(void,     GetProgramInfoLog,      (GLuint, GLsizei, GLsizei *, GLchar *)) \
    X(void,     DeleteShader,           (GLuint)) \
    X(void,     DeleteProgram,          (GLuint)) \
    X(void,     UseProgram,             (GLuint)) \
    X(GLint,    GetUniformLocation,     (GLuint, const GLchar *)) \
    X(void,     Uniform1i,              (GLint, GLint)) \
    X(void,     Uniform2i,              (GLint, GLint, GLint)) \
    X(void,     Uniform3f,              (GLint, GLfloat, GLfloat, GLfloat)) \
    X(void,     Uniform4f,              (GLint, GLfloat, GLfloat, GLfloat, GLfloat)) \
    X(void,     Viewport,               (GLint, GLint, GLsizei, GLsizei)) \
    X(void,     DepthMask,              (GLboolean)) \
    X(void,     ColorMask,              (GLboolean, GLboolean, GLboolean, GLboolean)) \
    X(void,     DrawArrays,             (GLenum, GLint, GLsizei)) \
    X(void,     Enable,                 (GLenum)) \
    X(void,     Disable,                (GLenum)) \
    X(GLboolean, IsEnabled,             (GLenum)) \
    X(void,     GetIntegerv,            (GLenum, GLint *)) \
    X(void,     GetBooleanv,            (GLenum, GLboolean *)) \
    X(GLenum,   GetError,               (void)) \
    X(const GLubyte *, GetString,       (GLenum)) \
    X(void,     GetFramebufferAttachmentParameteriv, (GLenum, GLenum, GLenum, GLint *)) \
    X(void,     BindRenderbuffer,       (GLenum, GLuint)) \
    X(void,     GetRenderbufferParameteriv, (GLenum, GLenum, GLint *)) \
    X(void,     ReadPixels,             (GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void *)) \
    X(void,     PixelStorei,            (GLenum, GLint)) \
    X(GLboolean, IsFramebuffer,         (GLuint)) \
    X(GLboolean, IsTexture,             (GLuint))

typedef struct {
#define X(r, n, a) r (*n) a;
    GLFN(X)
#undef X
} GLF;
static GLF gl;

/* ------------------------------------------------------------------ utils + log */
static uint64_t now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)(ts.tv_nsec / 1000);
}
static uint64_t g_t0;
static double since_s(void) { return (double)(now_us() - g_t0) / 1e6; }

static int  g_fd = -1;
static char g_dir[600];
static atomic_ullong g_logbytes;
#define LOG_CAP_BYTES (256u * 1024u)

static void logf_(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void logf_(const char *fmt, ...) {
    char buf[1536];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (n < 0 || g_fd < 0) return;
    if ((size_t)n >= sizeof buf) n = (int)sizeof buf - 1;
    if (atomic_fetch_add(&g_logbytes, (unsigned long long)n) > LOG_CAP_BYTES) return;
    const char *s = buf;
    size_t left = (size_t)n;
    while (left > 0) {
        ssize_t w = write(g_fd, s, left);
        if (w < 0) { if (errno == EINTR) continue; break; }
        if (w == 0) break;
        s += w; left -= (size_t)w;
    }
}

static void open_log(void) {
#ifdef RESA_TEST
    g_fd = 2;
    snprintf(g_dir, sizeof g_dir, "%s", getenv("RESA_AO_DIR") ? getenv("RESA_AO_DIR") : "/tmp");
#else
    const char *home = getenv("HOME");
    if (!home || !*home) home = "/tmp";
    snprintf(g_dir, sizeof g_dir, "%s/Documents", home);
    mkdir(g_dir, 0755);
    char cur[700], prev[700];
    snprintf(cur, sizeof cur, "%s/resa_ao_log.txt", g_dir);
    snprintf(prev, sizeof prev, "%s/resa_ao_log_prev.txt", g_dir);
    rename(cur, prev);                                        /* keep the previous run */
    g_fd = open(cur, O_WRONLY | O_CREAT | O_TRUNC | O_APPEND, 0644);
#endif
}

/* ------------------------------------------------------------------ configuration (Documents/resa_ao.ini) */
typedef struct {
    int   view;            /* 0 off | 1 AO on | 2 AO only | 3 depth | 4 normals | 5 AO after 1st blur | 6 SSR buffer only */
    float radius;          /* world units (metres) */
    float intensity;
    float power;
    float bias;
    int   samples;         /* 4..32 */
    int   max_px;          /* cap on the screen-space radius in pixels */
    float min_px;          /* AO fades out when its screen radius gets below this */
    int   blur_radius;     /* 0..6 taps each side */
    float blur_sharp;      /* bigger = blur respects depth edges more */
    float fade_start, fade_end;   /* distance fade (metres) */
    float lum_protect;     /* 0..1: protect bright pixels (sun-lit walls, coronas) from darkening */
    int   compare;         /* 1 = left half of the screen without AO, right half with AO */
    float ssr_intensity;   /* screen-space reflections: strength of the added reflection, 0 = SSR off (0..2) */
    float ssr_max_dist;    /* SSR: how far (metres) a reflection ray may travel */
    int   ssr_steps;       /* SSR: ray-march steps per pixel (4..64) */
    int   touch_toggle;
    int   fingers;
    int   verbose;
} Cfg;

static Cfg g_cfg;
static atomic_int g_user_off;        /* toggled by the touch gesture */
static atomic_int g_view_override;   /* -1 = use the ini's view; set by the long-press gesture */
static time_t g_cfg_mtime;
static uint64_t g_cfg_next_us;

static void cfg_defaults(Cfg *c) {
    c->view = 1;
    c->radius = 1.5f;
    c->intensity = 1.0f;
    c->power = 1.5f;
    c->bias = 0.03f;
    c->samples = 16;
    c->max_px = 48;
    c->min_px = 3.0f;
    c->blur_radius = 4;
    c->blur_sharp = 2500.0f;
    c->fade_start = 80.0f;
    c->fade_end = 250.0f;
    c->lum_protect = 0.35f;
    c->compare = 0;
    c->ssr_intensity = 0.5f;
    c->ssr_max_dist = 15.0f;
    c->ssr_steps = 24;
    c->touch_toggle = 1;
    c->fingers = 5;
    c->verbose = 0;
}
static float clampf_(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
static int   clampi_(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

static void cfg_set(Cfg *c, const char *k, const char *v) {
    float f = (float)atof(v);
    int   i = atoi(v);
    if      (!strcmp(k, "view"))         c->view = clampi_(i, 0, 6);
    else if (!strcmp(k, "enabled"))      { if (!i) c->view = 0; }
    else if (!strcmp(k, "radius"))       c->radius = clampf_(f, 0.1f, 10.0f);
    else if (!strcmp(k, "intensity"))    c->intensity = clampf_(f, 0.0f, 4.0f);
    else if (!strcmp(k, "power"))        c->power = clampf_(f, 0.2f, 4.0f);
    else if (!strcmp(k, "bias"))         c->bias = clampf_(f, 0.0f, 0.5f);
    else if (!strcmp(k, "samples"))      c->samples = clampi_(i, 4, 32);
    else if (!strcmp(k, "max_px"))       c->max_px = clampi_(i, 8, 128);
    else if (!strcmp(k, "min_px"))       c->min_px = clampf_(f, 1.0f, 20.0f);
    else if (!strcmp(k, "blur_radius"))  c->blur_radius = clampi_(i, 0, 6);
    else if (!strcmp(k, "blur_sharp"))   c->blur_sharp = clampf_(f, 10.0f, 100000.0f);
    else if (!strcmp(k, "fade_start"))   c->fade_start = clampf_(f, 1.0f, 5000.0f);
    else if (!strcmp(k, "fade_end"))     c->fade_end = clampf_(f, 2.0f, 6000.0f);
    else if (!strcmp(k, "lum_protect"))  c->lum_protect = clampf_(f, 0.0f, 1.0f);
    else if (!strcmp(k, "compare"))      c->compare = i ? 1 : 0;
    else if (!strcmp(k, "touch_toggle")) c->touch_toggle = i ? 1 : 0;
    else if (!strcmp(k, "fingers"))      c->fingers = clampi_(i, 2, 10);
    else if (!strcmp(k, "verbose"))      c->verbose = i ? 1 : 0;
    else if (!strcmp(k, "ssr_intensity")) c->ssr_intensity = clampf_(f, 0.0f, 2.0f);
    else if (!strcmp(k, "ssr_max_dist"))  c->ssr_max_dist = clampf_(f, 1.0f, 100.0f);
    else if (!strcmp(k, "ssr_steps"))     c->ssr_steps = clampi_(i, 4, 64);
}

static const char CFG_DEFAULT_TEXT[] =
    "; resa_ao.ini  -  ReSA SSAO settings.  Saved changes are picked up about every 1.5 seconds while the game runs.\n"
    "; Lines starting with ; or # are comments.\n"
    ";\n"
    "; view:  1 = AO on (normal)   0 = off\n"
    ";        2 = AO only (white world, so you can see exactly what AO does)\n"
    ";        3 = depth (bands every 5 m)   4 = reconstructed normals   5 = AO after the first blur pass only\n"
    ";        6 = reflections only (the SSR buffer, x2): shows exactly where reflections are drawn\n"
    "; Tip: if something looks wrong, view=3 and view=4 tell us instantly whether depth/normals are being read correctly.\n"
    "view=1\n"
    ";\n"
    "; radius in metres (GTA units).  1.0 = tight contact shadows, 2.5 = broad darkening.\n"
    "radius=1.5\n"
    "; strength of the darkening (0..4) and the contrast curve applied to it (higher power = darker creases, lighter flats)\n"
    "intensity=1.0\n"
    "power=1.5\n"
    "; depth tolerance against self-shadowing on flat ground (metres).  Raise it if roads/walls look dirty or noisy.\n"
    "bias=0.03\n"
    "; quality: samples per pixel (4..32).  16 is fine, 24-32 is smoother.  blur_radius = taps each side of the 2 blur passes.\n"
    "samples=16\n"
    "blur_radius=4\n"
    "blur_sharp=2500\n"
    "max_px=48\n"
    "min_px=3\n"
    "; AO fades out between these two distances (metres)\n"
    "fade_start=80\n"
    "fade_end=250\n"
    "; 0..1: how much bright pixels (sunlit walls, lamps, coronas) are spared from darkening\n"
    "lum_protect=0.35\n"
    "; 1 = split screen: left half without AO, right half with AO (handy for before/after screenshots)\n"
    "compare=0\n"
    "; touch gestures: this many fingers on the screen at once, then lift.  quick tap = AO on/off, hold >1.2 s = cycle debug view\n"
    "touch_toggle=1\n"
    "fingers=5\n"
    "; SSR = screen-space reflections of the world on flat ground (roads, floors).  ssr_intensity: 0 = off, 0.5 = default, max 2.\n"
    "ssr_intensity=0.5\n"
    "; how far (metres) a reflection ray may travel, and how many steps it takes (4..64).  More steps = fewer missed thin objects, slower.\n"
    "ssr_max_dist=15\n"
    "ssr_steps=24\n"
    "; 1 = write a short status line to resa_ao_log.txt every ~10 s\n"
    "verbose=0\n";

static void cfg_load(int announce) {
    char path[700];
    snprintf(path, sizeof path, "%s/resa_ao.ini", g_dir);
    Cfg c;
    cfg_defaults(&c);
    struct stat st;
    if (stat(path, &st) != 0) {                         /* first run: write a documented default file */
        int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd >= 0) { ssize_t w = write(fd, CFG_DEFAULT_TEXT, sizeof CFG_DEFAULT_TEXT - 1); (void)w; close(fd); }
        g_cfg = c;
        if (stat(path, &st) == 0) g_cfg_mtime = st.st_mtime;
        if (announce) logf_("config: no resa_ao.ini yet, wrote defaults to %s\n", path);
        return;
    }
    g_cfg_mtime = st.st_mtime;
    FILE *fp = fopen(path, "r");
    if (fp) {
        char line[256];
        while (fgets(line, sizeof line, fp)) {
            char *p = line;
            while (*p == ' ' || *p == '\t') p++;
            if (*p == ';' || *p == '#' || *p == '\n' || *p == '\r' || !*p) continue;
            char *eq = strchr(p, '=');
            if (!eq) continue;
            *eq = 0;
            char *k = p, *v = eq + 1;
            for (char *e = k + strlen(k); e > k && (e[-1] == ' ' || e[-1] == '\t'); ) *--e = 0;
            for (char *q = k; *q; q++) if (*q >= 'A' && *q <= 'Z') *q = (char)(*q + 32);
            while (*v == ' ' || *v == '\t') v++;
            cfg_set(&c, k, v);
        }
        fclose(fp);
    }
    if (c.fade_end <= c.fade_start + 1.0f) c.fade_end = c.fade_start + 1.0f;
    g_cfg = c;
    atomic_store(&g_view_override, -1);                 /* an edited ini wins over the gesture */
    if (announce)
        logf_("config: view=%d radius=%.2f intensity=%.2f power=%.2f bias=%.3f samples=%d blur=%d/%.0f max_px=%d min_px=%.1f fade=%.0f..%.0f "
              "lum_protect=%.2f compare=%d touch=%d(%d fingers) ssr=%.2f/%.1fm/%d steps\n",
              c.view, c.radius, c.intensity, c.power, c.bias, c.samples, c.blur_radius, c.blur_sharp, c.max_px, c.min_px,
              c.fade_start, c.fade_end, c.lum_protect, c.compare, c.touch_toggle, c.fingers,
              c.ssr_intensity, c.ssr_max_dist, c.ssr_steps);
}
static void cfg_poll(void) {
    uint64_t t = now_us();
    if (t < g_cfg_next_us) return;
    g_cfg_next_us = t + 1500000ull;
    char path[700];
    snprintf(path, sizeof path, "%s/resa_ao.ini", g_dir);
    struct stat st;
    if (stat(path, &st) == 0 && st.st_mtime != g_cfg_mtime) cfg_load(1);
}

/* ------------------------------------------------------------------ hook plumbing (same mechanism as the probe) */
#define FN_LIST(X) X(glInvalidateFramebuffer) X(glUniformMatrix4fv)
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
static void *fh_orig[NF];     /* fishhook: original pointer (an image imports the symbol)            */
static void *dl_orig[NF];     /* original handed back by the real dlsym when the engine looked it up  */
static void *fb_orig[NF];     /* our own dlsym fallback                                              */

static void *(*real_dlsym)(void *, const char *);
static void *rdlsym(void *h, const char *n) { return real_dlsym ? real_dlsym(h, n) : dlsym(h, n); }

static void *orig_of(int f) {
    void *p = fh_orig[f];
    if (p) return p;
    p = dl_orig[f];
    if (p) return p;
    p = fb_orig[f];
    if (p) return p;
    p = rdlsym(RTLD_DEFAULT, fnames[f]);
    fb_orig[f] = p;
    return p;
}

#ifdef RESA_TEST
static void *(*g_test_getproc)(const char *);
void resa_test_set_getproc(void *(*fn)(const char *)) { g_test_getproc = fn; }
static void *gl_getproc(const char *name) { return g_test_getproc ? g_test_getproc(name) : NULL; }
#else
static void *gl_getproc(const char *name) { return rdlsym(RTLD_DEFAULT, name); }
#endif

static int gl_load(void) {
    int missing = 0;
#define X(r, n, a) gl.n = (r (*) a)gl_getproc("gl" #n); if (!gl.n) { missing++; logf_("GL: missing gl" #n "\n"); }
    GLFN(X)
#undef X
    return missing == 0;
}

/* tracked EAGLContext of the calling thread (set by the +setCurrentContext: swizzle; NULL = unknown) */
static __thread void *tl_ctx;

/* ------------------------------------------------------------------ projection capture */
typedef struct { float p00, p11, A, B, near_, far_; unsigned cnt; unsigned seq; } Proj;
#define NPROJ 6
static Proj g_pc[NPROJ];                 /* candidates seen since the last AO run */
static int  g_pc_n;
static unsigned g_pc_seq;
static Proj g_proj_last;                 /* sticky: last projection chosen */
static int  g_proj_valid;
static atomic_flag g_pc_lock = ATOMIC_FLAG_INIT;

static void proj_check(GLboolean tr, const float *m) {
    float a[16];
    if (tr) { for (int c = 0; c < 4; c++) for (int r = 0; r < 4; r++) a[c * 4 + r] = m[r * 4 + c]; }
    else memcpy(a, m, sizeof a);                       /* a[col*4+row] */
    for (int i = 0; i < 16; i++) if (!isfinite(a[i])) return;
    const float e = 1e-5f;
    if (fabsf(a[1]) > e || fabsf(a[2]) > e || fabsf(a[3]) > e || fabsf(a[4]) > e || fabsf(a[6]) > e || fabsf(a[7]) > e) return;
    if (fabsf(a[12]) > e || fabsf(a[13]) > e || fabsf(a[15]) > e) return;
    if (fabsf(a[8]) > e || fabsf(a[9]) > e) return;    /* off-centre projections are not what the scene pass uses */
    float s = a[11];
    if (fabsf(fabsf(s) - 1.0f) > 1e-4f) return;        /* w = +-z : a real perspective matrix */
    float p00 = fabsf(a[0]), p11 = fabsf(a[5]);
    if (p00 < 0.05f || p11 < 0.05f || p00 > 50.0f || p11 > 50.0f) return;
    float A = s * a[10], B = a[14];
    if (!(A > 1.0f) || fabsf(B) < 1e-6f) return;
    float nr = B / (-1.0f - A), fr = B / (1.0f - A);   /* z = B / (ndc - A) at ndc=-1 and +1 */
    if (!(nr > 0.01f && fr > nr && fr < 1.0e6f)) return;
    while (atomic_flag_test_and_set_explicit(&g_pc_lock, memory_order_acquire)) { }
    int found = -1;
    for (int i = 0; i < g_pc_n; i++)
        if (fabsf(g_pc[i].A - A) < 1e-6f && fabsf(g_pc[i].B - B) < 1e-5f && fabsf(g_pc[i].p11 - p11) < 1e-5f) { found = i; break; }
    if (found >= 0) { g_pc[found].cnt++; g_pc[found].seq = ++g_pc_seq; }
    else {
        int slot = g_pc_n < NPROJ ? g_pc_n++ : 0;
        if (g_pc_n == NPROJ) {                         /* full: replace the weakest */
            slot = 0;
            for (int i = 1; i < NPROJ; i++) if (g_pc[i].cnt < g_pc[slot].cnt) slot = i;
        }
        Proj p = { p00, p11, A, B, nr, fr, 1, ++g_pc_seq };
        g_pc[slot] = p;
    }
    atomic_flag_clear_explicit(&g_pc_lock, memory_order_release);
}
/* choose the projection of this frame: the most frequently uploaded one since the last run (ties: the latest) */
static int proj_take(Proj *out) {
    while (atomic_flag_test_and_set_explicit(&g_pc_lock, memory_order_acquire)) { }
    if (g_pc_n > 0) {
        int best = 0;
        for (int i = 1; i < g_pc_n; i++)
            if (g_pc[i].cnt > g_pc[best].cnt || (g_pc[i].cnt == g_pc[best].cnt && g_pc[i].seq > g_pc[best].seq)) best = i;
        g_proj_last = g_pc[best];
        g_proj_valid = 1;
        g_pc_n = 0;
    }
    int ok = g_proj_valid;
    if (ok) *out = g_proj_last;
    atomic_flag_clear_explicit(&g_pc_lock, memory_order_release);
    return ok;
}

/* ------------------------------------------------------------------ shaders (GLSL ES 3.00) */
#define GLSL_HDR "#version 300 es\nprecision highp float;\nprecision highp int;\nprecision highp sampler2D;\n"

static const char VS_SRC[] =
    GLSL_HDR
    "void main(){\n"
    "  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));\n"
    "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
    "}\n";

/* shared by all three fragment shaders: depth -> view-space position */
#define GLSL_COMMON \
    "uniform sampler2D uDepth;\n" \
    "uniform vec4 uProj;\n"    /* x = 1/P00, y = 1/P11, z = A, w = B  (view z = B / (ndc - A)) */ \
    "uniform vec4 uSize;\n"    /* w, h, 1/w, 1/h */ \
    "out vec4 oColor;\n" \
    "float linZ(float d){ float den = min(d * 2.0 - 1.0 - uProj.z, -1e-7); return uProj.w / den; }\n" \
    "ivec2 clampP(ivec2 p){ return clamp(p, ivec2(0), ivec2(uSize.xy) - ivec2(1)); }\n" \
    "float zAt(ivec2 p){ return linZ(texelFetch(uDepth, clampP(p), 0).r); }\n" \
    "vec3 posAt(ivec2 p){\n" \
    "  ivec2 q = clampP(p);\n" \
    "  float z = linZ(texelFetch(uDepth, q, 0).r);\n" \
    "  vec2 uv = (vec2(q) + 0.5) * uSize.zw;\n" \
    "  return vec3((uv * 2.0 - 1.0) * uProj.xy * z, z);\n" \
    "}\n"

/* pass 1: ambient obscurance (Alchemy / SAO style) from depth only.  Output: r = visibility, gba = packed normal. */
static const char FS_AO_SRC[] =
    GLSL_HDR GLSL_COMMON
    "uniform vec4 uAo;\n"      /* x = radius (m), y = intensity, z = bias (m), w = focal length in pixels */
    "uniform vec4 uAo2;\n"     /* x = max radius px, y = min radius px (fade), z = normalisation K */
    "uniform int uSamples;\n"
    "float ign(vec2 p){ return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }\n"
    "void main(){\n"
    "  ivec2 ip = ivec2(gl_FragCoord.xy);\n"
    "  float d0 = texelFetch(uDepth, ip, 0).r;\n"
    "  if (d0 >= 0.999999) { oColor = vec4(1.0, 0.5, 0.5, 1.0); return; }\n"
    "  ivec2 sz = ivec2(uSize.xy);\n"
    "  vec3 P  = posAt(ip);\n"
    "  vec3 Pl = posAt(ip + ivec2(-1, 0)); vec3 Pr = posAt(ip + ivec2(1, 0));\n"
    "  vec3 Pd = posAt(ip + ivec2(0, -1)); vec3 Pu = posAt(ip + ivec2(0, 1));\n"
    "  bool useR = abs(Pr.z - P.z) < abs(P.z - Pl.z);\n"
    "  bool useU = abs(Pu.z - P.z) < abs(P.z - Pd.z);\n"
    "  if (ip.x == 0) useR = true;  if (ip.x == sz.x - 1) useR = false;\n"
    "  if (ip.y == 0) useU = true;  if (ip.y == sz.y - 1) useU = false;\n"
    "  vec3 dx = useR ? (Pr - P) : (P - Pl);\n"
    "  vec3 dy = useU ? (Pu - P) : (P - Pd);\n"
    "  vec3 cr = cross(dx, dy);\n"
    "  float cl = length(cr);\n"
    "  vec3 N = cl > 1e-12 ? cr / cl : vec3(0.0, 0.0, -1.0);\n"
    "  if (dot(N, P) > 0.0) N = -N;\n"                       /* face the camera, whatever the handedness */
    "  float rPx  = uAo.x * uAo.w / P.z;\n"
    "  float rC   = min(rPx, uAo2.x);\n"
    "  float Reff = rC * P.z / uAo.w;\n"
    "  float invR2 = 1.0 / (Reff * Reff);\n"
    "  float bias = uAo.z + 0.0015 * P.z;\n"
    "  float n1 = ign(gl_FragCoord.xy);\n"
    "  float n2 = ign(gl_FragCoord.xy + vec2(47.0, 17.0));\n"
    "  float invN = 1.0 / float(uSamples);\n"
    "  float occ = 0.0;\n"
    "  for (int i = 0; i < uSamples; i++) {\n"
    "    float t  = (float(i) + n2) * invN;\n"
    "    float a  = float(i) * 2.3999632 + n1 * 6.2831853;\n"
    "    float rr = max(t * rC, 1.0);\n"
    "    ivec2 sp = ip + ivec2(round(vec2(cos(a), sin(a)) * rr));\n"
    "    vec3 v = posAt(sp) - P;\n"
    "    float vv = dot(v, v);\n"
    "    float vn = dot(v, N) - bias;\n"
    "    float f  = max(1.0 - vv * invR2, 0.0);\n"
    "    occ += f * f * max(vn, 0.0) * inversesqrt(vv + 1e-6);\n"
    "  }\n"
    "  float ao = 1.0 - clamp(occ * invN * uAo.y * uAo2.z, 0.0, 1.0);\n"
    "  ao = mix(1.0, ao, smoothstep(uAo2.y * 0.5, uAo2.y, rPx));\n"
    "  oColor = vec4(ao, N * 0.5 + 0.5);\n"
    "}\n";

/* pass 2: depth-aware blur along uDir (slope-aware, so tilted ground is smoothed but silhouettes are not) */
static const char FS_BLUR_SRC[] =
    GLSL_HDR GLSL_COMMON
    "uniform sampler2D uAoTex;\n"
    "uniform ivec2 uDir;\n"
    "uniform vec4 uBlur;\n"    /* x = sharpness, y = radius (taps), z = 1/(2 sigma^2) */
    "void main(){\n"
    "  ivec2 ip = ivec2(gl_FragCoord.xy);\n"
    "  vec4 c0 = texelFetch(uAoTex, ip, 0);\n"
    "  float d0 = texelFetch(uDepth, ip, 0).r;\n"
    "  if (d0 >= 0.999999) { oColor = c0; return; }\n"
    "  float z0 = zAt(ip);\n"
    "  float zA = zAt(ip - uDir);\n"
    "  float zB = zAt(ip + uDir);\n"
    "  float sl = (abs(zB - z0) < abs(z0 - zA)) ? (zB - z0) : (z0 - zA);\n"
    "  float sum = c0.r;\n"
    "  float ws = 1.0;\n"
    "  int R = int(uBlur.y);\n"
    "  for (int k = 1; k <= R; k++) {\n"
    "    float g = exp(-float(k * k) * uBlur.z);\n"
    "    for (int s = -1; s <= 1; s += 2) {\n"
    "      ivec2 q = clampP(ip + uDir * (k * s));\n"
    "      float zq = zAt(q);\n"
    "      float zp = z0 + sl * float(k * s);\n"
    "      float dz = (zq - zp) / z0;\n"
    "      float w = g * exp(-dz * dz * uBlur.x);\n"
    "      sum += w * texelFetch(uAoTex, q, 0).r;\n"
    "      ws += w;\n"
    "    }\n"
    "  }\n"
    "  oColor = vec4(sum / ws, c0.gba);\n"
    "}\n";

/* pass 2b: screen-space reflections (SSR) of the world on flat, upward-facing surfaces (roads, pavements, floors, car roofs).
 *
 * Inputs : unit 0 = depth copy (uDepth), unit 2 = scene colour copy (uScene).   Uniforms: uProj/uSize via set_common(),
 *          uSsr = (intensity, max ray length in metres, ray-march steps).
 * Output : rgb = the light to ADD to the pixel (0 = no reflection).  The composite pass adds it AFTER the AO darkening.
 *          It is written as a screen blend  refl * (1 - scene),  so it can never push a pixel past white and bright
 *          surfaces (concrete, road markings) get less than dark asphalt, like a real reflection does.
 *
 * How it works, per pixel:
 *   1. normal from depth (same method as the AO pass); only surfaces whose normal points "up" in view space are used
 *      (view +Y = up, camera roll ~0: true for the engine's cameras; pitch up to ~35 deg is full strength, then it fades)
 *   2. mirror the view ray about that normal, march the reflected ray through the depth buffer in screen space
 *      (perspective-correct 1/z interpolation, samples dense near the pixel and sparse far away, per-pixel jitter),
 *      accept a hit only if the ray goes from in front of a surface to just behind it, refine it by bisection, and
 *      reject hits on surfaces facing away from the ray (that is what stops the road reflecting itself)
 *   3. colour at the hit (small cross blur that grows with ray length = glossy, not a mirror), faded near the screen edges
 *   4. no hit within the ray length: fall back to the far environment = the screen pixel in the direction of the ray
 *      (for a level camera that is the mirror image of the sky/horizon; above the top of the screen: the top row),
 *      blended in smoothly as the hit gets far
 *   5. strength = ssr_intensity * Fresnel (Schlick, F0 0.15: more reflection at grazing angles) * flatness * material,
 *      where "material" removes most of the reflection from green (grass, bushes) surfaces
 */
static const char FS_SSR_SRC[] =
    GLSL_HDR GLSL_COMMON
    "uniform sampler2D uScene;\n"
    "uniform vec3 uSsr;\n"                                  /* x = intensity, y = max ray length (m), z = steps */
    "float ign(vec2 p){ return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }\n"
    /* view-space point -> continuous pixel coordinates (pixel centres are at +0.5) */
    "vec2 toPx(vec3 q){ return (q.xy / (uProj.xy * q.z) * 0.5 + 0.5) * uSize.xy; }\n"
    /* surface normal from depth, facing the camera (same neighbour choice as the AO pass: never differences across an edge) */
    "vec3 nrmFrom(ivec2 ip, vec3 P){\n"
    "  ivec2 sz = ivec2(uSize.xy);\n"
    "  vec3 Pl = posAt(ip + ivec2(-1, 0)); vec3 Pr = posAt(ip + ivec2(1, 0));\n"
    "  vec3 Pd = posAt(ip + ivec2(0, -1)); vec3 Pu = posAt(ip + ivec2(0, 1));\n"
    "  bool useR = abs(Pr.z - P.z) < abs(P.z - Pl.z);\n"
    "  bool useU = abs(Pu.z - P.z) < abs(P.z - Pd.z);\n"
    "  if (ip.x == 0) useR = true;  if (ip.x == sz.x - 1) useR = false;\n"
    "  if (ip.y == 0) useU = true;  if (ip.y == sz.y - 1) useU = false;\n"
    "  vec3 dx = useR ? (Pr - P) : (P - Pl);\n"
    "  vec3 dy = useU ? (Pu - P) : (P - Pd);\n"
    "  vec3 cr = cross(dx, dy);\n"
    "  float cl = length(cr);\n"
    "  vec3 N = cl > 1e-12 ? cr / cl : vec3(0.0, 0.0, -1.0);\n"
    "  return dot(N, P) > 0.0 ? -N : N;\n"
    "}\n"
    "float edgeFade(vec2 uv){ return smoothstep(0.0, 0.08, min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y))); }\n"
    "void main(){\n"
    "  oColor = vec4(0.0, 0.0, 0.0, 1.0);\n"                 /* default: nothing to add */
    "  ivec2 ip = ivec2(gl_FragCoord.xy);\n"
    "  if (texelFetch(uDepth, ip, 0).r >= 0.999999) return;\n"     /* sky */
    "  vec3 P = posAt(ip);\n"
    "  if (P.z > 300.0) return;\n"
    /* ---- is this a floor? */
    "  vec3 N = nrmFrom(ip, P);\n"
    "  float flo = smoothstep(0.62, 0.82, N.y);\n"
    "  if (flo <= 0.0) return;\n"
    /* ---- how strong: material (not grass), Fresnel, user intensity */
    "  vec3 base = texelFetch(uScene, ip, 0).rgb;\n"
    "  float veg = smoothstep(0.02, 0.12, base.g - max(base.r, base.b));\n"
    "  float mx  = max(base.r, max(base.g, base.b));\n"
    "  float sat = (mx - min(base.r, min(base.g, base.b))) / max(mx, 0.05);\n"
    "  float mat = (1.0 - 0.9 * veg) * (1.0 - 0.5 * smoothstep(0.35, 0.7, sat));\n"
    "  vec3 V = normalize(P);\n"
    "  float NoV = clamp(-dot(N, V), 0.0, 1.0);\n"
    "  float fr = pow(1.0 - NoV, 5.0);\n"
    "  float wgt = uSsr.x * (0.15 + 0.85 * fr) * flo * mat;\n"
    "  if (wgt < 0.004) return;\n"
    /* ---- the reflected ray, clipped to the near plane and to the screen */
    "  vec3 R = reflect(V, N);\n"
    "  vec3 q0 = P + N * (0.03 + 0.004 * P.z);\n"            /* start a little above the surface */
    "  float tmax = uSsr.y;\n"
    "  if (R.z < 0.0) tmax = min(tmax, (0.3 - q0.z) / R.z);\n"
    "  if (tmax < 0.1) return;\n"
    "  vec3 q1 = q0 + R * tmax;\n"
    "  vec2 p0 = toPx(q0);\n"
    "  vec2 dd = toPx(q1) - p0;\n"
    "  float k0 = 1.0 / q0.z;\n"
    "  float k1 = 1.0 / q1.z;\n"                             /* 1/z is linear along a screen-space line */
    "  vec2 tb = vec2(1e9);\n"
    "  if (dd.x >  1e-3) tb.x = (uSize.x - 1.0 - p0.x) / dd.x;\n"
    "  if (dd.x < -1e-3) tb.x = (1.0 - p0.x) / dd.x;\n"
    "  if (dd.y >  1e-3) tb.y = (uSize.y - 1.0 - p0.y) / dd.y;\n"
    "  if (dd.y < -1e-3) tb.y = (1.0 - p0.y) / dd.y;\n"
    "  float smax = clamp(min(tb.x, tb.y), 0.0, 1.0);\n"
    "  float lpx = length(dd) * smax;\n"
    /* ---- march */
    "  bool hit = false;\n"
    "  float sHit = 0.0;\n"
    "  if (lpx >= 2.0) {\n"
    "    int n = int(clamp(uSsr.z + 0.5, 4.0, 64.0));\n"
    "    n = min(n, int(lpx * 0.5) + 2);\n"                   /* never more than one sample per 2 px */
    "    float inv = 1.0 / float(n);\n"
    "    float jit = ign(gl_FragCoord.xy);\n"
    "    float sPrev = 0.0;\n"
    "    float rzPrev = q0.z;\n"
    "    float dPrev = -1.0;\n"                               /* the start point is in front of the surface by construction */
    "    for (int i = 0; i < n; i++) {\n"
    "      float u = (float(i) + 0.5 + 0.5 * jit) * inv;\n"
    "      float s = smax * (0.35 * u + 0.65 * u * u);\n"     /* dense near the pixel, sparse far away */
    "      float rz = 1.0 / (k0 + (k1 - k0) * s);\n"
    "      float df = rz - zAt(ivec2(p0 + dd * s));\n"        /* > 0: the ray is behind the visible surface */
    "      float th = abs(rz - rzPrev) + 0.12 + 0.04 * rz;\n" /* thickness: what one step can cross, plus a little */
    "      if (df > 0.0 && df < th && dPrev <= 0.0) {\n"
    "        float sa = sPrev;\n"
    "        float sb = s;\n"
    "        for (int b = 0; b < 6; b++) {\n"                 /* bisection between 'in front' and 'behind' */
    "          float sm = 0.5 * (sa + sb);\n"
    "          float dm = 1.0 / (k0 + (k1 - k0) * sm) - zAt(ivec2(p0 + dd * sm));\n"
    "          if (dm > 0.0) sb = sm; else sa = sm;\n"
    "        }\n"
    "        ivec2 hp = ivec2(p0 + dd * sb);\n"
    "        float db = 1.0 / (k0 + (k1 - k0) * sb) - zAt(hp);\n"
    "        if (db < th) {\n"                                /* a real surface, not the edge of a nearer object */
    "          vec3 Nh = nrmFrom(hp, posAt(hp));\n"
    "          if (dot(Nh, R) < 0.05) { hit = true; sHit = sb; break; }\n"   /* the surface must face the ray */
    "        }\n"
    "      }\n"
    "      dPrev = df;\n"
    "      sPrev = s;\n"
    "      rzPrev = rz;\n"
    "    }\n"
    "  }\n"
    /* ---- colour at the hit, glossy and edge-faded; m = how much to fade over to the far environment (1 = no hit) */
    "  vec3 hitAcc = vec3(0.0);\n"
    "  float m = 1.0;\n"
    "  if (hit) {\n"
    "    vec2 hpx = p0 + dd * sHit;\n"
    "    ivec2 hp = ivec2(hpx);\n"
    "    int rb = int(clamp(0.5 + 0.04 * length(dd) * sHit, 0.0, 5.0));\n"
    "    vec3 col = texelFetch(uScene, clampP(hp), 0).rgb * 0.4;\n"
    "    col += (texelFetch(uScene, clampP(hp + ivec2(rb, 0)), 0).rgb + texelFetch(uScene, clampP(hp - ivec2(rb, 0)), 0).rgb\n"
    "          + texelFetch(uScene, clampP(hp + ivec2(0, rb)), 0).rgb + texelFetch(uScene, clampP(hp - ivec2(0, rb)), 0).rgb) * 0.15;\n"
    "    float rzh = 1.0 / (k0 + (k1 - k0) * sHit);\n"
    "    vec3 qh = vec3((hpx / uSize.xy * 2.0 - 1.0) * uProj.xy * rzh, rzh);\n"
    "    hitAcc = col * edgeFade(hpx / uSize.xy);\n"
    "    m = smoothstep(0.55, 1.0, length(qh - q0) / uSsr.y);\n"
    "  }\n"
    /* ---- far environment: the pixel the ray direction points at, if that is on screen and clearly farther than the ray reached */
    "  vec3 envAcc = vec3(0.0);\n"
    "  if (m > 0.0 && R.z > 0.08) {\n"
    "    vec2 uv = toPx(R) / uSize.xy;\n"
    "    uv.y = min(uv.y, 0.96);\n"                          /* mirror direction above the top of the screen: take the top row's sky */
    "    if (uv.x > 0.0 && uv.x < 1.0 && uv.y > 0.0) {\n"
    "      ivec2 ep = ivec2(uv * uSize.xy);\n"
    "      float ez = zAt(ep);\n"
    "      float rzEnd = q1.z;\n"
    "      float ok = smoothstep(rzEnd * 1.1 + 2.0, rzEnd * 1.4 + 8.0, ez);\n"
    "      float ef = smoothstep(0.0, 0.08, min(uv.x, 1.0 - uv.x)) * smoothstep(0.0, 0.08, uv.y);\n"
    "      envAcc = texelFetch(uScene, ep, 0).rgb * ok * ef;\n"
    "    }\n"
    "  }\n"
    "  vec3 acc = mix(hitAcc, envAcc, m);\n"
    "  float wf = 1.0 - exp(-wgt);\n"                         /* soft saturation: intensity 2 is stronger than 1, never clips */
    "  oColor = vec4(acc * (1.0 - base) * wf, 1.0);\n"        /* screen blend: base + refl never exceeds 1 */
    "}\n";

/* pass 3: second blur direction + apply to the scene colour (or show a debug view) */
static const char FS_COMP_SRC[] =
    GLSL_HDR GLSL_COMMON
    "uniform sampler2D uAoTex;\n"
    "uniform sampler2D uScene;\n"
    "uniform sampler2D uSsrTex;\n"   /* unit 3: reflections to add (only read when uSsr.x > 0) */
    "uniform vec3 uSsr;\n"           /* x = SSR intensity this frame, 0 = SSR was not run: do not touch uSsrTex */
    "uniform ivec2 uDir;\n"
    "uniform vec4 uBlur;\n"
    "uniform vec4 uFx;\n"      /* x = fade start, y = fade end, z = luminance protect, w = power */
    "uniform ivec2 uMode;\n"   /* x = view, y = compare */
    "void main(){\n"
    "  ivec2 ip = ivec2(gl_FragCoord.xy);\n"
    "  vec4 scn = texelFetch(uScene, ip, 0);\n"
    "  if (uMode.y > 0 && ip.x < int(uSize.x * 0.5)) { oColor = scn; return; }\n"
    "  float d0 = texelFetch(uDepth, ip, 0).r;\n"
    "  bool sky = d0 >= 0.999999;\n"
    "  vec4 c0 = texelFetch(uAoTex, ip, 0);\n"
    "  float z0 = sky ? 10000.0 : zAt(ip);\n"
    "  float vis = 1.0;\n"
    "  if (!sky) {\n"
    "    float zA = zAt(ip - uDir);\n"
    "    float zB = zAt(ip + uDir);\n"
    "    float sl = (abs(zB - z0) < abs(z0 - zA)) ? (zB - z0) : (z0 - zA);\n"
    "    float sum = c0.r;\n"
    "    float ws = 1.0;\n"
    "    int R = int(uBlur.y);\n"
    "    for (int k = 1; k <= R; k++) {\n"
    "      float g = exp(-float(k * k) * uBlur.z);\n"
    "      for (int s = -1; s <= 1; s += 2) {\n"
    "        ivec2 q = clampP(ip + uDir * (k * s));\n"
    "        float zq = zAt(q);\n"
    "        float zp = z0 + sl * float(k * s);\n"
    "        float dz = (zq - zp) / z0;\n"
    "        float w = g * exp(-dz * dz * uBlur.x);\n"
    "        sum += w * texelFetch(uAoTex, q, 0).r;\n"
    "        ws += w;\n"
    "      }\n"
    "    }\n"
    "    vis = sum / ws;\n"
    "  }\n"
    "  int view = uMode.x;\n"
    "  if (view == 2) { oColor = vec4(vec3(vis), 1.0); return; }\n"
    "  if (view == 3) {\n"
    "    float near01 = 1.0 - clamp(z0 / 150.0, 0.0, 1.0);\n"
    "    float band = fract(z0 * 0.2);\n"
    "    oColor = sky ? vec4(0.0, 0.0, 0.25, 1.0) : vec4(vec3(0.15 + 0.6 * near01 + 0.25 * band), 1.0);\n"
    "    return;\n"
    "  }\n"
    "  if (view == 4) { oColor = vec4(c0.gba, 1.0); return; }\n"
    "  if (view == 5) { oColor = vec4(vec3(c0.r), 1.0); return; }\n"
    "  vec3 refl = uSsr.x > 0.0 ? texelFetch(uSsrTex, ip, 0).rgb : vec3(0.0);\n"
    "  if (view == 6) { oColor = vec4(refl * 2.0, 1.0); return; }\n"
    "  float fade = 1.0 - smoothstep(uFx.x, uFx.y, z0);\n"
    "  float vp = pow(max(vis, 0.0), uFx.w);\n"
    "  float lum = dot(scn.rgb, vec3(0.299, 0.587, 0.114));\n"
    "  float prot = 1.0 - uFx.z * smoothstep(0.55, 1.0, lum);\n"
    "  oColor = vec4(scn.rgb * mix(1.0, vp, fade * prot) + refl * fade, scn.a);\n"   /* reflection is added AFTER the AO darkening */
    "}\n";

/* normalisation of the occlusion sum, calibrated with tests/ (flat ground = 1.0, 90-degree crease ~ 0.5 at intensity 1) */
#define AO_NORM_K 2.6f

/* ------------------------------------------------------------------ GL resources */
typedef struct {
    GLint uDepth, uProj, uSize, uAo, uAo2, uSamples, uAoTex, uScene, uDir, uBlur, uFx, uMode, uSsr, uSsrTex;
    GLuint prog;
} Prog;

typedef struct {
    GLenum ifmt;          /* internal format of our depth copy */
    GLenum attach;        /* DEPTH_ATTACHMENT or DEPTH_STENCIL_ATTACHMENT */
    GLbitfield mask;      /* blit mask for depth(+stencil) */
    const char *name;
} DFmt;
static const DFmt DF_D24S8  = { GLC_DEPTH24_STENCIL8,  GLC_DEPTH_STENCIL_ATTACHMENT, GLC_DEPTH_BUFFER_BIT | GLC_STENCIL_BUFFER_BIT, "D24S8" };
static const DFmt DF_D24    = { GLC_DEPTH_COMPONENT24, GLC_DEPTH_ATTACHMENT,         GLC_DEPTH_BUFFER_BIT, "D24" };
static const DFmt DF_D16    = { GLC_DEPTH_COMPONENT16, GLC_DEPTH_ATTACHMENT,         GLC_DEPTH_BUFFER_BIT, "D16" };
static const DFmt DF_D32F   = { GLC_DEPTH_COMPONENT32F, GLC_DEPTH_ATTACHMENT,        GLC_DEPTH_BUFFER_BIT, "D32F" };
static const DFmt DF_D32FS8 = { GLC_DEPTH32F_STENCIL8, GLC_DEPTH_STENCIL_ATTACHMENT, GLC_DEPTH_BUFFER_BIT | GLC_STENCIL_BUFFER_BIT, "D32FS8" };

typedef struct {
    int w, h;
    const DFmt *df;
    void *ctx;
    GLuint tex_depth, tex_color, tex_ao[2], tex_ssr;
    GLuint fbo_copy, fbo_ao[2], fbo_ssr;
    GLuint vao;
    Prog ao, blur, comp, ssr;     /* ssr.prog == 0: reflections are off for this run, AO keeps running */
    int ok;
} Res;
static Res R;

static int gl_err_drain(const char *where, int logit) {
    int n = 0;
    GLenum e;
    while ((e = gl.GetError()) != 0 && n < 16) {
        if (logit) logf_("GL error 0x%x after %s\n", e, where);
        n++;
    }
    return n;
}

static GLuint compile_shader(GLenum type, const char *src, const char *name) {
    GLuint s = gl.CreateShader(type);
    if (!s) { logf_("shader %s: glCreateShader failed\n", name); return 0; }
    const GLchar *srcs[1] = { src };
    gl.ShaderSource(s, 1, srcs, NULL);
    gl.CompileShader(s);
    GLint ok = 0;
    gl.GetShaderiv(s, GLC_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024] = {0};
        GLsizei n = 0;
        gl.GetShaderInfoLog(s, sizeof log - 1, &n, log);
        logf_("shader %s FAILED to compile:\n%s\n", name, log);
        gl.DeleteShader(s);
        return 0;
    }
    return s;
}
static int build_prog(Prog *p, GLuint vs, const char *fs_src, const char *name) {
    memset(p, 0, sizeof *p);
    GLuint fs = compile_shader(GLC_FRAGMENT_SHADER, fs_src, name);
    if (!fs) return 0;
    GLuint pr = gl.CreateProgram();
    gl.AttachShader(pr, vs);
    gl.AttachShader(pr, fs);
    gl.LinkProgram(pr);
    gl.DeleteShader(fs);
    GLint ok = 0;
    gl.GetProgramiv(pr, GLC_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024] = {0};
        GLsizei n = 0;
        gl.GetProgramInfoLog(pr, sizeof log - 1, &n, log);
        logf_("program %s FAILED to link:\n%s\n", name, log);
        gl.DeleteProgram(pr);
        return 0;
    }
    p->prog = pr;
#define U(n) p->n = gl.GetUniformLocation(pr, #n);
    U(uDepth) U(uProj) U(uSize) U(uAo) U(uAo2) U(uSamples) U(uAoTex) U(uScene) U(uDir) U(uBlur) U(uFx) U(uMode) U(uSsr) U(uSsrTex)
#undef U
    return 1;
}

static void tex_alloc(GLuint tex, GLenum ifmt, int w, int h) {
    gl.BindTexture(GLC_TEXTURE_2D, tex);
    gl.TexStorage2D(GLC_TEXTURE_2D, 1, ifmt, w, h);
    gl.TexParameteri(GLC_TEXTURE_2D, GLC_TEXTURE_MIN_FILTER, GLC_NEAREST);
    gl.TexParameteri(GLC_TEXTURE_2D, GLC_TEXTURE_MAG_FILTER, GLC_NEAREST);
    gl.TexParameteri(GLC_TEXTURE_2D, GLC_TEXTURE_WRAP_S, GLC_CLAMP_TO_EDGE);
    gl.TexParameteri(GLC_TEXTURE_2D, GLC_TEXTURE_WRAP_T, GLC_CLAMP_TO_EDGE);
}/* ====================================================================================================================
 * PART C  --  GL resources, GL-state save/restore, and the AO frame itself.      (append after part_b.c)
 *
 * INTERFACE FOR PART D (this is everything D needs from C):
 *   ao_init()                                  once, in the constructor, after open_log() and cfg_load(1).
 *   ao_on_scene_invalidate(target, n, attach)  call it from the glInvalidateFramebuffer hook with the hook's own
 *                                              arguments, BEFORE calling the original.  It recognises the scene FBO by
 *                                              itself (n==2 {DEPTH,STENCIL} on a renderbuffer-depth FBO), so D can just
 *                                              forward every call.  It never throws; on any problem it logs and returns.
 *   ao_toggle_user() / ao_cycle_view()         from the 5-finger tap / long-press gesture.
 *   proj_check(transpose, m)    [part B]       D's glUniformMatrix4fv hook calls it once per matrix (count can be > 1).
 *   tl_ctx                      [part B]       D's +[EAGLContext setCurrentContext:] swizzle keeps it up to date.  C only
 *                                              uses it to notice a context switch (then it forgets, never deletes, its
 *                                              GL names, because the same numbers can belong to the engine there).
 *
 * WHAT HAPPENS EACH FRAME (all on the render thread, inside the engine's glInvalidateFramebuffer call):
 *   probe scene FBO -> take projection -> save GL state -> blit {colour, depth(+stencil)} of the scene FBO into our
 *   copy FBO -> AO pass -> blur pass (horizontal) -> composite pass (vertical blur + apply to the colour copy, written
 *   straight into the engine's scene FBO) -> restore GL state.  The engine's own invalidate then runs as usual.
 *
 * SAFETY: all objects are ours; all state we touch is saved and restored; any shader/FBO build failure, or a GL error on
 *   two frames in a row, switches AO off for the run and says why in resa_ao_log.txt.  A single GL error rebuilds our
 *   size-dependent objects once (covers a render-scale change) and aborts that frame BEFORE it writes into the scene.
 * ==================================================================================================================== */

#ifndef GLC_RENDERBUFFER_INTERNAL_FORMAT
#define GLC_RENDERBUFFER_INTERNAL_FORMAT 0x8D44
#endif

#define AO_UNITS 4            /* texture units we use: 0 = depth copy, 1 = AO texture, 2 = scene colour copy, 3 = SSR texture */

static struct { unsigned inval, ran, noproj, skipped, errors; } g_st;
static int        g_dead;          /* AO is permanently off for this run (the reason is in the log) */
static int        g_gl_ready;      /* gl_load() succeeded */
static int        g_err_streak;    /* consecutive AO frames that ended in a GL error */
static int        g_skip_logs;     /* rate limit for "ignoring this invalidate" lines */
static unsigned   g_since_build;   /* successful AO frames since our sized objects were (re)built */
static uint64_t   g_status_next_us;
static atomic_int g_ao_inited;

/* ------------------------------------------------------------------ run-time switches */
static void ao_die(const char *why) {
    if (g_dead) return;
    g_dead = 1;
    logf_("AO DISABLED for the rest of this run: %s\n", why);
}

static int ao_effective_view(void) {
    if (atomic_load(&g_user_off)) return 0;
    int ov = atomic_load(&g_view_override);
    return ov >= 0 ? ov : g_cfg.view;
}

static void ao_toggle_user(void) {
    int now_off = (atomic_fetch_xor(&g_user_off, 1) ^ 1) & 1;
    logf_("gesture: AO %s\n", now_off ? "OFF" : "ON");
}

static void ao_cycle_view(void) {                       /* AO -> AO only -> depth -> normals -> AO */
    int ov = atomic_load(&g_view_override);
    int cur = ov >= 0 ? ov : g_cfg.view;
    int next = (cur >= 1 && cur < 4) ? cur + 1 : 1;
    atomic_store(&g_view_override, next);
    logf_("gesture: debug view -> %d\n", next);
}

/* NOTE: cfg_load() returns early on the very first run (no ini yet) WITHOUT resetting g_view_override, and a zero-initialised
 * override means "view 0 = off".  So the reset to -1 must happen here too. */
static void ao_init(void) {
    if (atomic_exchange(&g_ao_inited, 1)) return;
    atomic_store(&g_view_override, -1);
    atomic_store(&g_user_off, 0);
    memset(&R, 0, sizeof R);
    logf_("AO %s: init done, waiting for the first scene invalidate\n", AO_VERSION);
}

/* ------------------------------------------------------------------ GL state save / restore */
typedef struct {
    GLint     fbo_draw, fbo_read, rbo, prog, vao, active, vp[4];
    GLint     tex[AO_UNITS], smp[AO_UNITS];
    GLboolean cap[8];
    GLboolean dmask, cmask[4];
} GLState;

static const GLenum k_caps[8] = {
    GLC_DEPTH_TEST, GLC_STENCIL_TEST, GLC_BLEND, GLC_CULL_FACE,
    GLC_SCISSOR_TEST, GLC_DITHER, GLC_POLYGON_OFFSET_FILL, GLC_RASTERIZER_DISCARD
};

static void state_save(GLState *s) {
    memset(s, 0, sizeof *s);
    gl.GetIntegerv(GLC_DRAW_FRAMEBUFFER_BINDING, &s->fbo_draw);
    gl.GetIntegerv(GLC_READ_FRAMEBUFFER_BINDING, &s->fbo_read);
    gl.GetIntegerv(GLC_RENDERBUFFER_BINDING, &s->rbo);
    gl.GetIntegerv(GLC_CURRENT_PROGRAM, &s->prog);
    gl.GetIntegerv(GLC_VERTEX_ARRAY_BINDING, &s->vao);
    gl.GetIntegerv(GLC_VIEWPORT, s->vp);
    gl.GetIntegerv(GLC_ACTIVE_TEXTURE, &s->active);
    for (int u = 0; u < AO_UNITS; u++) {
        gl.ActiveTexture(GLC_TEXTURE0 + (GLenum)u);
        gl.GetIntegerv(GLC_TEXTURE_BINDING_2D, &s->tex[u]);
        gl.GetIntegerv(GLC_SAMPLER_BINDING, &s->smp[u]);
    }
    gl.ActiveTexture((GLenum)s->active);                  /* the loop above moved the active unit; state_save must not leave a trace */
    for (int i = 0; i < 8; i++) s->cap[i] = gl.IsEnabled(k_caps[i]);
    gl.GetBooleanv(GLC_DEPTH_WRITEMASK, &s->dmask);
    gl.GetBooleanv(GLC_COLOR_WRITEMASK, s->cmask);
}

static void state_restore(const GLState *s) {
    for (int u = 0; u < AO_UNITS; u++) {
        gl.ActiveTexture(GLC_TEXTURE0 + (GLenum)u);
        gl.BindTexture(GLC_TEXTURE_2D, (GLuint)s->tex[u]);
        gl.BindSampler((GLuint)u, (GLuint)s->smp[u]);
    }
    gl.ActiveTexture((GLenum)s->active);
    gl.BindRenderbuffer(GLC_RENDERBUFFER, (GLuint)s->rbo);
    gl.BindFramebuffer(GLC_READ_FRAMEBUFFER, (GLuint)s->fbo_read);
    gl.BindFramebuffer(GLC_DRAW_FRAMEBUFFER, (GLuint)s->fbo_draw);
    gl.BindVertexArray((GLuint)s->vao);
    gl.UseProgram((GLuint)s->prog);
    gl.Viewport(s->vp[0], s->vp[1], s->vp[2], s->vp[3]);
    for (int i = 0; i < 8; i++) { if (s->cap[i]) gl.Enable(k_caps[i]); else gl.Disable(k_caps[i]); }
    gl.DepthMask(s->dmask);
    gl.ColorMask(s->cmask[0], s->cmask[1], s->cmask[2], s->cmask[3]);
}

/* ------------------------------------------------------------------ our GL objects (all calls run inside the save/restore bracket) */
static void res_free_sized(Res *r) {                     /* only valid in the context that created them */
    gl.DeleteFramebuffers(1, &r->fbo_copy);
    gl.DeleteFramebuffers(2, r->fbo_ao);
    gl.DeleteFramebuffers(1, &r->fbo_ssr);
    gl.DeleteTextures(1, &r->tex_depth);
    gl.DeleteTextures(1, &r->tex_color);
    gl.DeleteTextures(2, r->tex_ao);
    gl.DeleteTextures(1, &r->tex_ssr);
    r->fbo_copy = r->fbo_ao[0] = r->fbo_ao[1] = r->fbo_ssr = 0;
    r->tex_depth = r->tex_color = r->tex_ao[0] = r->tex_ao[1] = r->tex_ssr = 0;
    r->ok = 0;
}

static void res_free_all(Res *r) {
    res_free_sized(r);
    if (r->ao.prog)   gl.DeleteProgram(r->ao.prog);
    if (r->blur.prog) gl.DeleteProgram(r->blur.prog);
    if (r->comp.prog) gl.DeleteProgram(r->comp.prog);
    if (r->ssr.prog)  gl.DeleteProgram(r->ssr.prog);
    if (r->vao)       gl.DeleteVertexArrays(1, &r->vao);
    void *ctx = r->ctx;
    memset(r, 0, sizeof *r);
    r->ctx = ctx;
}

static int res_build_programs(Res *r) {
    GLuint vs = compile_shader(GLC_VERTEX_SHADER, VS_SRC, "vs");
    if (!vs) return 0;
    int ok = build_prog(&r->ao,   vs, FS_AO_SRC,   "ao")   &&
             build_prog(&r->blur, vs, FS_BLUR_SRC, "blur") &&
             build_prog(&r->comp, vs, FS_COMP_SRC, "comp");
    int ssr_ok = ok && build_prog(&r->ssr, vs, FS_SSR_SRC, "ssr");      /* optional: if it fails, AO still runs, just no reflections */
    gl.DeleteShader(vs);
    if (!ok) return 0;
    gl.GenVertexArrays(1, &r->vao);
    const GLubyte *ver = gl.GetString(GLC_VERSION), *rnd = gl.GetString(GLC_RENDERER);
    logf_("res: AO programs built OK, SSR program %s.  GL_VERSION=\"%s\"  GL_RENDERER=\"%s\"\n",
          ssr_ok ? "OK" : "FAILED (reflections off, AO unaffected)", ver ? (const char *)ver : "?", rnd ? (const char *)rnd : "?");
    return 1;
}

static int attach_fbo(GLuint fbo, GLuint color, const DFmt *depth_fmt, GLuint depth_tex, const char *what) {
    gl.BindFramebuffer(GLC_FRAMEBUFFER, fbo);
    gl.FramebufferTexture2D(GLC_FRAMEBUFFER, GLC_COLOR_ATTACHMENT0, GLC_TEXTURE_2D, color, 0);
    if (depth_fmt) gl.FramebufferTexture2D(GLC_FRAMEBUFFER, depth_fmt->attach, GLC_TEXTURE_2D, depth_tex, 0);
    GLenum st = gl.CheckFramebufferStatus(GLC_FRAMEBUFFER);
    if (st != GLC_FRAMEBUFFER_COMPLETE) { logf_("res: %s FBO incomplete (status 0x%x)\n", what, st); return 0; }
    return 1;
}

static int res_build_sized(Res *r, int w, int h, const DFmt *df) {
    GLuint t[5], f[4];
    gl.GenTextures(5, t);
    r->tex_depth = t[0]; r->tex_color = t[1]; r->tex_ao[0] = t[2]; r->tex_ao[1] = t[3]; r->tex_ssr = t[4];
    gl.ActiveTexture(GLC_TEXTURE0);
    tex_alloc(r->tex_depth, df->ifmt, w, h);
    tex_alloc(r->tex_color, GLC_RGBA8, w, h);
    tex_alloc(r->tex_ao[0], GLC_RGBA8, w, h);
    tex_alloc(r->tex_ao[1], GLC_RGBA8, w, h);
    tex_alloc(r->tex_ssr, GLC_RGBA8, w, h);
    gl.GenFramebuffers(4, f);
    r->fbo_copy = f[0]; r->fbo_ao[0] = f[1]; r->fbo_ao[1] = f[2]; r->fbo_ssr = f[3];
    if (!attach_fbo(r->fbo_copy, r->tex_color, df, r->tex_depth, "copy (colour+depth)")) return 0;
    if (!attach_fbo(r->fbo_ao[0], r->tex_ao[0], NULL, 0, "ao[0]")) return 0;
    if (!attach_fbo(r->fbo_ao[1], r->tex_ao[1], NULL, 0, "ao[1]")) return 0;
    if (!attach_fbo(r->fbo_ssr, r->tex_ssr, NULL, 0, "ssr")) return 0;
    if (gl_err_drain("res_build_sized", 1)) return 0;
    r->w = w; r->h = h; r->df = df; r->ok = 1;
    return 1;
}

/* make our objects match (context, size, depth format).  Returns 0 if that is impossible (AO is then switched off). */
static int res_ensure(void *ctx, int w, int h, const DFmt *df) {
    if (R.ctx && ctx && R.ctx != ctx) {
        logf_("res: EAGLContext changed %p -> %p; forgetting our old GL names (NOT deleting them: the numbers may be the engine's there)\n", R.ctx, ctx);
        memset(&R, 0, sizeof R);
    }
    if (ctx) R.ctx = ctx;
    if (!R.ao.prog || !R.blur.prog || !R.comp.prog) {
        if (!res_build_programs(&R)) { res_free_all(&R); ao_die("shader build failed (see the shader log above)"); return 0; }
    }
    if (!R.ok || R.w != w || R.h != h || R.df != df) {
        res_free_sized(&R);
        if (!res_build_sized(&R, w, h, df)) { res_free_all(&R); ao_die("could not build the AO framebuffers/textures"); return 0; }
        g_since_build = 0;
        logf_("res: sized objects ready: %dx%d, depth copy %s (blit mask 0x%x)\n", w, h, df->name, (unsigned)(GLC_COLOR_BUFFER_BIT | df->mask));
    }
    return 1;
}

/* ------------------------------------------------------------------ is the draw framebuffer the world-scene FBO? */
typedef struct { GLuint fbo; int w, h; const DFmt *df; } Scene;

static int scene_skip(const char *why, int a, int b) {
    g_st.skipped++;
    if (g_skip_logs < 6) { g_skip_logs++; logf_("scene: ignoring this invalidate (%s: %d, %d)\n", why, a, b); }
    return 0;
}

static const DFmt *dfmt_from_ifmt(GLint f) {
    switch ((GLenum)f) {
        case GLC_DEPTH24_STENCIL8:  return &DF_D24S8;
        case GLC_DEPTH_COMPONENT24: return &DF_D24;
        case GLC_DEPTH_COMPONENT16: return &DF_D16;
        case GLC_DEPTH_COMPONENT32F: return &DF_D32F;
        case GLC_DEPTH32F_STENCIL8: return &DF_D32FS8;
        default: return NULL;
    }
}
static const DFmt *dfmt_from_sizes(GLint d, GLint s) {
    if (d == 24 && s == 8) return &DF_D24S8;
    if (d == 24 && s == 0) return &DF_D24;
    if (d == 16 && s == 0) return &DF_D16;
    if (d == 32 && s == 0) return &DF_D32F;
    if (d == 32 && s == 8) return &DF_D32FS8;
    return NULL;
}

static int scene_probe(Scene *sc) {
    GLint fbo = 0, cty = 0, dty = 0, dnm = 0, rb0 = 0, w = 0, h = 0, ifmt = 0;
    gl.GetIntegerv(GLC_DRAW_FRAMEBUFFER_BINDING, &fbo);
    if (!fbo) return scene_skip("draw framebuffer is 0", 0, 0);
    gl.GetFramebufferAttachmentParameteriv(GLC_DRAW_FRAMEBUFFER, GLC_COLOR_ATTACHMENT0, GLC_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &cty);
    gl.GetFramebufferAttachmentParameteriv(GLC_DRAW_FRAMEBUFFER, GLC_DEPTH_ATTACHMENT, GLC_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &dty);
    gl.GetFramebufferAttachmentParameteriv(GLC_DRAW_FRAMEBUFFER, GLC_DEPTH_ATTACHMENT, GLC_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &dnm);
    if (cty == (GLint)GLC_NONE) return scene_skip("no colour attachment", fbo, 0);
    if (dty != (GLint)GLC_RENDERBUFFER) return scene_skip("depth attachment is not a renderbuffer", fbo, dty);
    gl.GetIntegerv(GLC_RENDERBUFFER_BINDING, &rb0);
    gl.BindRenderbuffer(GLC_RENDERBUFFER, (GLuint)dnm);
    gl.GetRenderbufferParameteriv(GLC_RENDERBUFFER, GLC_RENDERBUFFER_WIDTH, &w);
    gl.GetRenderbufferParameteriv(GLC_RENDERBUFFER, GLC_RENDERBUFFER_HEIGHT, &h);
    gl.GetRenderbufferParameteriv(GLC_RENDERBUFFER, GLC_RENDERBUFFER_INTERNAL_FORMAT, &ifmt);
    gl.BindRenderbuffer(GLC_RENDERBUFFER, (GLuint)rb0);
    if (gl_err_drain("scene probe", g_skip_logs < 6)) return scene_skip("GL error while probing", fbo, 0);
    if (w < 128 || h < 128 || w > 8192 || h > 8192) return scene_skip("odd size", w, h);
    const DFmt *df = dfmt_from_ifmt(ifmt);
    if (!df) {
        GLint ds = 0, ss = 0;
        gl.GetFramebufferAttachmentParameteriv(GLC_DRAW_FRAMEBUFFER, GLC_DEPTH_ATTACHMENT, GLC_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE, &ds);
        gl.GetFramebufferAttachmentParameteriv(GLC_DRAW_FRAMEBUFFER, GLC_DEPTH_ATTACHMENT, GLC_FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE, &ss);
        gl_err_drain("scene probe sizes", 0);
        df = dfmt_from_sizes(ds, ss);
        if (!df) return scene_skip("unsupported depth format (ifmt, depth bits)", ifmt, ds);
    }
    sc->fbo = (GLuint)fbo; sc->w = w; sc->h = h; sc->df = df;
    return 1;
}

/* ------------------------------------------------------------------ one-off diagnostic: read back the AO texture and log what is in it
 * (GL ES 3.0 cannot read depth back, but r = visibility and a = packed normal-z, and sky is written as (1, .5, .5, 1),
 *  so these numbers tell us from the log alone whether depth + projection were understood.) */
static void ao_diag(const char *tag) {
    GLint pbo = 0, pa = 4, prl = 0, psr = 0, psp = 0;
    gl.GetIntegerv(GLC_PIXEL_PACK_BUFFER_BINDING, &pbo);
    if (pbo) { logf_("diag(%s): skipped, a PIXEL_PACK_BUFFER (%d) is bound\n", tag, pbo); return; }
    size_t npx = (size_t)R.w * (size_t)R.h;
    unsigned char *buf = (unsigned char *)malloc(npx * 4);
    if (!buf) return;
    gl.GetIntegerv(GLC_PACK_ALIGNMENT, &pa);
    gl.GetIntegerv(GLC_PACK_ROW_LENGTH, &prl);
    gl.GetIntegerv(GLC_PACK_SKIP_ROWS, &psr);
    gl.GetIntegerv(GLC_PACK_SKIP_PIXELS, &psp);
    gl.PixelStorei(GLC_PACK_ALIGNMENT, 4);
    gl.PixelStorei(GLC_PACK_ROW_LENGTH, 0);
    gl.PixelStorei(GLC_PACK_SKIP_ROWS, 0);
    gl.PixelStorei(GLC_PACK_SKIP_PIXELS, 0);
    gl.BindFramebuffer(GLC_READ_FRAMEBUFFER, R.fbo_ao[1]);
    gl.ReadPixels(0, 0, R.w, R.h, GLC_RGBA, GLC_UNSIGNED_BYTE, buf);
    gl.PixelStorei(GLC_PACK_ALIGNMENT, pa);
    gl.PixelStorei(GLC_PACK_ROW_LENGTH, prl);
    gl.PixelStorei(GLC_PACK_SKIP_ROWS, psr);
    gl.PixelStorei(GLC_PACK_SKIP_PIXELS, psp);
    size_t sky = 0, geo = 0, d90 = 0, d50 = 0;
    double sum = 0.0;
    int mn = 255;
    for (size_t i = 0; i < npx; i++) {
        const unsigned char *p = buf + i * 4;
        if (p[3] == 255) { sky++; continue; }
        geo++;
        sum += p[0];
        if (p[0] < 230) d90++;
        if (p[0] < 128) d50++;
        if (p[0] < mn) mn = p[0];
    }
    logf_("diag(%s): %dx%d  sky=%.1f%%  geometry: AO mean=%.3f min=%.2f  darker than 0.9: %.1f%%  darker than 0.5: %.1f%%\n",
          tag, R.w, R.h, 100.0 * (double)sky / (double)npx, geo ? sum / (255.0 * (double)geo) : 1.0, mn / 255.0,
          geo ? 100.0 * (double)d90 / (double)geo : 0.0, geo ? 100.0 * (double)d50 / (double)geo : 0.0);
    free(buf);
}

/* ------------------------------------------------------------------ the AO frame */
static void bind_tex(int unit, GLuint tex) {
    gl.ActiveTexture(GLC_TEXTURE0 + (GLenum)unit);
    gl.BindSampler((GLuint)unit, 0);                      /* the engine's sampler objects must not override our parameters */
    gl.BindTexture(GLC_TEXTURE_2D, tex);
}
static void set_common(const Prog *p, const Proj *pj, int w, int h) {
    gl.Uniform1i(p->uDepth, 0);
    gl.Uniform4f(p->uProj, 1.0f / pj->p00, 1.0f / pj->p11, pj->A, pj->B);
    gl.Uniform4f(p->uSize, (float)w, (float)h, 1.0f / (float)w, 1.0f / (float)h);
}
static void draw_into(GLuint fbo, int w, int h) {
    gl.BindFramebuffer(GLC_FRAMEBUFFER, fbo);
    gl.Viewport(0, 0, w, h);
    gl.DrawArrays(GLC_TRIANGLES, 0, 3);                   /* one full-screen triangle from gl_VertexID, no buffers */
}

/* returns 0 = done, 1 = GL error (the scene FBO is only written by the very last pass) */
static int ao_frame(const Scene *sc, const Proj *pj, int view) {
    const Cfg c = g_cfg;
    const int w = sc->w, h = sc->h;
    const float invsig = c.blur_radius > 0 ? 2.0f / (float)(c.blur_radius * c.blur_radius) : 1.0f;   /* sigma = radius/2 */
    int err = 0;
    const int use_ssr = R.ssr.prog && c.ssr_intensity > 0.0f && (view == 1 || view == 6);   /* SSR only where it can be seen */
#define STAGE(name) do { if (gl_err_drain(name, 1)) { err = 1; goto out; } } while (0)

    /* nothing of the engine's state may leak into our passes */
    for (int i = 0; i < 8; i++) gl.Disable(k_caps[i]);
    gl.DepthMask(0);
    gl.ColorMask(1, 1, 1, 1);

    /* 1. copy colour + depth(+stencil) out of the scene FBO (same size, NEAREST: a plain copy) */
    gl.BindFramebuffer(GLC_READ_FRAMEBUFFER, sc->fbo);
    gl.BindFramebuffer(GLC_DRAW_FRAMEBUFFER, R.fbo_copy);
    gl.BlitFramebuffer(0, 0, w, h, 0, 0, w, h, GLC_COLOR_BUFFER_BIT | sc->df->mask, GLC_NEAREST);
    STAGE("blit scene->copy");

    gl.BindVertexArray(R.vao);

    /* 2. ambient obscurance -> ao[0]  (r = visibility, gba = packed normal) */
    gl.UseProgram(R.ao.prog);
    set_common(&R.ao, pj, w, h);
    gl.Uniform4f(R.ao.uAo, c.radius, c.intensity, c.bias, 0.5f * (float)h * pj->p11);   /* focal length in pixels */
    gl.Uniform4f(R.ao.uAo2, (float)c.max_px, c.min_px, AO_NORM_K, 0.0f);
    gl.Uniform1i(R.ao.uSamples, c.samples);
    bind_tex(0, R.tex_depth);
    draw_into(R.fbo_ao[0], w, h);
    STAGE("ao pass");

    /* 3. horizontal depth-aware blur: ao[0] -> ao[1] */
    gl.UseProgram(R.blur.prog);
    set_common(&R.blur, pj, w, h);
    gl.Uniform1i(R.blur.uAoTex, 1);
    gl.Uniform2i(R.blur.uDir, 1, 0);
    gl.Uniform4f(R.blur.uBlur, c.blur_sharp, (float)c.blur_radius, invsig, 0.0f);
    bind_tex(0, R.tex_depth);
    bind_tex(1, R.tex_ao[0]);
    draw_into(R.fbo_ao[1], w, h);
    STAGE("blur pass");

    /* 3b. screen-space reflections: depth + scene-colour copy -> tex_ssr (rgb = reflection to ADD, 0 = none) */
    if (use_ssr) {
        gl.UseProgram(R.ssr.prog);
        set_common(&R.ssr, pj, w, h);
        gl.Uniform1i(R.ssr.uScene, 2);
        gl.Uniform3f(R.ssr.uSsr, c.ssr_intensity, c.ssr_max_dist, (float)c.ssr_steps);
        bind_tex(0, R.tex_depth);
        bind_tex(2, R.tex_color);
        draw_into(R.fbo_ssr, w, h);
        STAGE("ssr pass");
    }

    /* 4. vertical blur + apply to the colour copy, written into the ENGINE'S scene FBO (it samples only our copies) */
    gl.UseProgram(R.comp.prog);
    set_common(&R.comp, pj, w, h);
    gl.Uniform1i(R.comp.uAoTex, 1);
    gl.Uniform1i(R.comp.uScene, 2);
    gl.Uniform2i(R.comp.uDir, 0, 1);
    gl.Uniform4f(R.comp.uBlur, c.blur_sharp, (float)c.blur_radius, invsig, 0.0f);
    gl.Uniform4f(R.comp.uFx, c.fade_start, c.fade_end, c.lum_protect, c.power);
    gl.Uniform2i(R.comp.uMode, view, c.compare);
    gl.Uniform1i(R.comp.uSsrTex, 3);
    gl.Uniform3f(R.comp.uSsr, use_ssr ? c.ssr_intensity : 0.0f, 0.0f, 0.0f);   /* 0 = shader never reads unit 3 */
    bind_tex(0, R.tex_depth);
    bind_tex(1, R.tex_ao[1]);
    bind_tex(2, R.tex_color);
    bind_tex(3, R.tex_ssr);
    draw_into(sc->fbo, w, h);
    STAGE("composite pass");

    {
        unsigned fno = ++g_since_build;
        if (fno == 1)
            logf_("AO: first frame OK  view=%d  scene fbo %u %dx%d  proj p00=%.4f p11=%.4f near=%.3f far=%.1f  samples=%d radius=%.2f  ssr=%s\n",
                  view, sc->fbo, w, h, pj->p00, pj->p11, pj->near_, pj->far_, c.samples, c.radius, use_ssr ? "on" : "off");
        if (fno == 30 || (c.verbose && fno % 900 == 0)) ao_diag("blur1");
    }
out:
#undef STAGE
    return err;
}

/* ------------------------------------------------------------------ entry point, called by part D's glInvalidateFramebuffer hook */
static void ao_status_line(int view) {
    uint64_t t = now_us();
    if (!g_cfg.verbose || t < g_status_next_us) return;
    g_status_next_us = t + 10000000ull;
    logf_("status t=%.0fs  invalidates=%u ao_frames=%u no_projection=%u ignored=%u gl_errors=%u  view=%d  fbo=%dx%d  near=%.2f far=%.1f\n",
          since_s(), g_st.inval, g_st.ran, g_st.noproj, g_st.skipped, g_st.errors, view, R.w, R.h, g_proj_last.near_, g_proj_last.far_);
}

static void ao_on_scene_invalidate(GLenum target, GLsizei n, const GLenum *att) {
    if (g_dead || n != 2 || !att) return;
    if (target != GLC_FRAMEBUFFER && target != GLC_DRAW_FRAMEBUFFER) return;
    int hasD = 0, hasS = 0;
    for (int i = 0; i < 2; i++) { hasD |= (att[i] == GLC_DEPTH_ATTACHMENT); hasS |= (att[i] == GLC_STENCIL_ATTACHMENT); }
    if (!hasD || !hasS) return;                           /* the other invalidate (COLOR0+DEPTH+STENCIL) is the full-res target */

    if (!atomic_load(&g_ao_inited)) ao_init();
    g_st.inval++;
    cfg_poll();
    int view = ao_effective_view();
    ao_status_line(view);
    if (view <= 0) return;
    if (!g_gl_ready) {
        if (!gl_load()) { ao_die("a required GL function could not be resolved (see the 'GL: missing' lines)"); return; }
        g_gl_ready = 1;
    }

    gl_err_drain("engine", 0);                            /* errors the engine left pending are not ours */
    Scene sc = { 0, 0, 0, NULL };
    if (!scene_probe(&sc)) return;
    Proj pj;
    if (!proj_take(&pj)) { g_st.noproj++; return; }
    float asp = (pj.p11 / pj.p00) / ((float)sc.w / (float)sc.h);       /* sanity: projection aspect ~ this FBO's aspect */
    if (asp < 0.7f || asp > 1.4f) { scene_skip("projection aspect does not match this FBO", (int)(pj.p11 / pj.p00 * 100.0f), sc.w * 100 / sc.h); return; }

    GLState st;
    state_save(&st);
    int rc = res_ensure(tl_ctx, sc.w, sc.h, sc.df) ? ao_frame(&sc, &pj, view) : 2;
    state_restore(&st);
    if (gl_err_drain("state restore", 1)) rc = 1;

    if (rc == 0) { g_st.ran++; g_err_streak = 0; }
    else if (rc == 1) {
        g_st.errors++;
        if (++g_err_streak >= 2) ao_die("GL error on two AO frames in a row (the lines above name the stage)");
        else { R.ok = 0; logf_("AO: GL error, rebuilding our objects once and trying again\n"); }
    }
}
/* ================================================================= end of part C *//* ====================================================================================================================
 * PART D  --  hooks, touch gesture, constructor.      (append after part_c.c; this is the LAST part)
 *
 *   1. hk_glInvalidateFramebuffer  every call -> ao_on_scene_invalidate() (part C decides if it is the scene FBO),
 *                                  THEN the engine's original function.
 *   2. hk_glUniformMatrix4fv       every uploaded matrix -> proj_check() (part B), then the original.
 *   3. hk_dlsym                    alyn_resa looks its GL functions up with dlsym(); for our two names we hand out the hooks
 *                                  and remember the real pointer in dl_orig[].  glInvalidateFramebuffer is ALSO imported by
 *                                  alyn_resa, so both names are rebound with fishhook too (fh_orig[]).  Only images inside
 *                                  the app bundle are patched (as in the probe), later ones as they load (alloader at ~6 s).
 *   4. ObjC swizzles (data-only, fine on a stock iPhone):
 *        +[EAGLContext setCurrentContext:]   keeps tl_ctx.  The probe saw the MAIN thread and 'ReSA.Render' both call it,
 *                                            including with nil, so tl_ctx follows exactly what each thread was given.
 *        -[UIApplication sendEvent:]         feeds the 5-finger gesture.  The original ALWAYS runs: no event is swallowed.
 *   5. constructor: banner, ini, ao_init(), install everything.
 *   6. one summary line at 20 s and 60 s in resa_ao_log.txt (hook calls, AO frames, why not), so a "nothing happened"
 *      run can be diagnosed from the log alone.
 *
 * Known limit: the dlsym wrapper forwards RTLD_NEXT / RTLD_SELF relative to THIS dylib.  The probe ran 145 s with an
 * identical wrapper and the app was fine, so nothing in the bundle depends on those two handles.
 * ==================================================================================================================== */

/* ------------------------------------------------------------------ helpers */
static atomic_uint g_calls_inval, g_calls_matrix;
static atomic_int  g_hook_first[NF];
static atomic_int  g_sum_stage;
static atomic_int  g_patched;                 /* app-bundle images patched by fishhook so far */

static void thread_name(char *out, size_t n) {
    out[0] = 0;
    pthread_getname_np(pthread_self(), out, n);
    if (!out[0]) snprintf(out, n, "(unnamed)");
}

static void hook_first(int f) {
    if (atomic_load_explicit(&g_hook_first[f], memory_order_relaxed)) return;
    if (atomic_exchange(&g_hook_first[f], 1)) return;
    char tn[64];
    thread_name(tn, sizeof tn);
    logf_("hook: first %s call  thread='%s'  t=%.2fs\n", fnames[f], tn, since_s());
}

static void maybe_summary(void) {
    int st = atomic_load(&g_sum_stage);
    double t = since_s();
    if (!((st == 0 && t >= 20.0) || (st == 1 && t >= 60.0))) return;
    if (!atomic_compare_exchange_strong(&g_sum_stage, &st, st + 1)) return;
    logf_("summary t=%.0fs: hook calls invalidate=%u matrix=%u | scene invalidates=%u ao_frames=%u no_projection=%u ignored=%u "
          "gl_errors=%u | dead=%d view=%d | images patched=%d | proj near=%.2f far=%.1f\n",
          t, atomic_load(&g_calls_inval), atomic_load(&g_calls_matrix), g_st.inval, g_st.ran, g_st.noproj, g_st.skipped,
          g_st.errors, g_dead, ao_effective_view(), atomic_load(&g_patched), g_proj_last.near_, g_proj_last.far_);
}

/* ------------------------------------------------------------------ ObjC glue (iOS only) */
#ifndef RESA_TEST
#define MSG(R, ...) ((R (*)(__VA_ARGS__))objc_msgSend)           /* Apple's documented objc_msgSend cast (GCC warns about it, Clang does not) */

static Class g_cls_eagl;
static SEL   S_currentContext, S_type, S_allTouches, S_allObjects, S_count, S_objectAtIndex, S_phase;
static IMP   orig_setCurrentContext, orig_sendEvent;
static int   g_objc_tries;

static BOOL hk_setCurrentContext(id self, SEL cmd, id ctx) {
    BOOL ok = ((BOOL (*)(id, SEL, id))orig_setCurrentContext)(self, cmd, ctx);
    if (ok) tl_ctx = (void *)ctx;                         /* nil included: that thread has no context any more */
    return ok;
}

/* number of fingers still on the screen in this touch event (began / moved / stationary); -1 = could not tell */
static int touches_active(id ev) {
    id set = MSG(id, id, SEL)(ev, S_allTouches);
    if (!set) return -1;
    id arr = MSG(id, id, SEL)(set, S_allObjects);
    if (!arr) return -1;
    unsigned long n = MSG(unsigned long, id, SEL)(arr, S_count);
    int active = 0;
    for (unsigned long i = 0; i < n && i < 16; i++) {
        id t = MSG(id, id, SEL, unsigned long)(arr, S_objectAtIndex, i);
        if (!t) continue;
        long ph = MSG(long, id, SEL)(t, S_phase);
        if (ph >= 0 && ph <= 2) active++;                 /* UITouchPhaseBegan / Moved / Stationary */
    }
    return active;
}
#endif /* !RESA_TEST */

/* ------------------------------------------------------------------ the gesture state machine (pure, unit-tested on Linux)
 * all `need` fingers down at once -> armed.  The FIRST drop below `need` decides: held < 1.2 s = tap (AO on/off),
 * >= 1.2 s = hold (cycle debug view).  After that nothing fires until every finger is up again, so one finger that stays
 * down for gameplay afterwards can never turn into a "hold".  A gesture older than 20 s is discarded.
 * returns 0 = nothing, 1 = tap, 2 = hold */
typedef struct { int st; uint64_t t0; } Gest;
#define GEST_HOLD_US 1200000ull
#define GEST_MAX_US  20000000ull

static int gest_feed(Gest *g, int active, int need, uint64_t t) {
    switch (g->st) {
    case 0:
        if (active >= need) { g->st = 1; g->t0 = t; }
        return 0;
    case 1: {
        if (active >= need) return 0;
        g->st = active > 0 ? 2 : 0;
        uint64_t dt = t - g->t0;
        if (dt > GEST_MAX_US) return 0;
        return dt >= GEST_HOLD_US ? 2 : 1;
    }
    default:
        if (active == 0) g->st = 0;
        return 0;
    }
}

#ifndef RESA_TEST
static Gest g_gest;                                       /* main thread only */

static void gesture_apply(int act) {
    if (act == 1) ao_toggle_user();
    else if (act == 2) ao_cycle_view();
}

static void hk_sendEvent(id self, SEL cmd, id ev) {
    if (ev && g_cfg.touch_toggle && MSG(long, id, SEL)(ev, S_type) == 0) {          /* UIEventTypeTouches */
        int a = touches_active(ev);
        if (a >= 0) gesture_apply(gest_feed(&g_gest, a, g_cfg.fingers, now_us()));
    }
    ((void (*)(id, SEL, id))orig_sendEvent)(self, cmd, ev);
}

/* safe to call repeatedly: installs whatever is still missing (UIKit/EAGL classes might not exist yet at load time) */
static void install_objc(void) {
    g_objc_tries++;
    if (!S_currentContext) {
        S_currentContext = sel_registerName("currentContext");
        S_type           = sel_registerName("type");
        S_allTouches     = sel_registerName("allTouches");
        S_allObjects     = sel_registerName("allObjects");
        S_count          = sel_registerName("count");
        S_objectAtIndex  = sel_registerName("objectAtIndex:");
        S_phase          = sel_registerName("phase");
    }
    if (!orig_setCurrentContext) {
        Class c = objc_getClass("EAGLContext");
        Method m = c ? class_getClassMethod(c, sel_registerName("setCurrentContext:")) : NULL;
        if (m) {
            g_cls_eagl = c;
            orig_setCurrentContext = method_getImplementation(m);          /* saved BEFORE the swap */
            method_setImplementation(m, (IMP)hk_setCurrentContext);
        }
    }
    if (!orig_sendEvent) {
        Class c = objc_getClass("UIApplication");
        Method m = c ? class_getInstanceMethod(c, sel_registerName("sendEvent:")) : NULL;
        if (m) {
            orig_sendEvent = method_getImplementation(m);
            method_setImplementation(m, (IMP)hk_sendEvent);
        }
    }
}

/* the swizzle only sees setCurrentContext: calls made after it was installed; ask EAGL directly if we still do not know */
static void note_context(void) {
    if (!tl_ctx && g_cls_eagl) tl_ctx = (void *)MSG(id, id, SEL)((id)g_cls_eagl, S_currentContext);
    if ((!orig_setCurrentContext || !orig_sendEvent) && g_objc_tries < 12 && (g_calls_inval % 60u) == 0) install_objc();
}
#else
static void note_context(void) { }
#endif

/* ------------------------------------------------------------------ the GL hooks */
typedef void (*fn_InvalidateFramebuffer)(GLenum, GLsizei, const GLenum *);
typedef void (*fn_UniformMatrix4fv)(GLint, GLsizei, GLboolean, const GLfloat *);

static void hk_glInvalidateFramebuffer(GLenum target, GLsizei n, const GLenum *att) {
    hook_first(F_glInvalidateFramebuffer);
    atomic_fetch_add_explicit(&g_calls_inval, 1, memory_order_relaxed);
    note_context();
    ao_on_scene_invalidate(target, n, att);               /* BEFORE the engine's call: the depth is still valid here */
    maybe_summary();
    fn_InvalidateFramebuffer o = (fn_InvalidateFramebuffer)orig_of(F_glInvalidateFramebuffer);
    if (o) o(target, n, att);
}

static void hk_glUniformMatrix4fv(GLint loc, GLsizei count, GLboolean transpose, const GLfloat *v) {
    hook_first(F_glUniformMatrix4fv);
    atomic_fetch_add_explicit(&g_calls_matrix, 1, memory_order_relaxed);
    if (v && count > 0 && !g_dead) {
        int n = count < 4 ? count : 4;                    /* bone palettes (count 30+) never hold the projection */
        for (int i = 0; i < n; i++) proj_check(transpose, v + 16 * i);
    }
    fn_UniformMatrix4fv o = (fn_UniformMatrix4fv)orig_of(F_glUniformMatrix4fv);
    if (o) o(loc, count, transpose, v);
}

static void *hook_for(int f) {
    switch (f) {
        case F_glInvalidateFramebuffer: return (void *)hk_glInvalidateFramebuffer;
        case F_glUniformMatrix4fv:      return (void *)hk_glUniformMatrix4fv;
        default:                        return NULL;
    }
}

/* never calls dlsym() itself: in this image that name is rebound to US (infinite recursion) */
static void *hk_dlsym(void *handle, const char *name) {
    void *p = real_dlsym ? real_dlsym(handle, name) : NULL;
    if (!p || !name || name[0] != 'g' || name[1] != 'l') return p;
    for (int f = 0; f < NF; f++) {
        if (strcmp(name, fnames[f]) != 0) continue;
        void *h = hook_for(f);
        if (!h) break;
        dl_orig[f] = p;
        static atomic_int n_logged;
        if (atomic_fetch_add(&n_logged, 1) < 8) logf_("dlsym: handing out our %s hook (real=%p)\n", name, p);
        return h;
    }
    return p;
}

/* ------------------------------------------------------------------ fishhook + constructor (iOS only) */
#ifndef RESA_TEST
static struct rebinding g_rebs[3];
static char g_match[1100];                                /* "/ReSALauncher.app/" (or the exe's folder if there is no .app) */
static int  g_match_prefix;

static void on_add_image(const struct mach_header *mh, intptr_t slide) {
    Dl_info di;
    if (!dladdr(mh, &di) || !di.dli_fname) return;
    int hit = g_match_prefix ? strncmp(di.dli_fname, g_match, strlen(g_match)) == 0 : strstr(di.dli_fname, g_match) != NULL;
    if (!hit) return;
    rebind_symbols_image((void *)mh, slide, g_rebs, sizeof g_rebs / sizeof g_rebs[0]);
    atomic_fetch_add(&g_patched, 1);
}

static void install_fishhook(const char *exe) {
    g_rebs[0].name = "dlsym";                   g_rebs[0].replacement = (void *)hk_dlsym;
    g_rebs[0].replaced = (void **)&real_dlsym;
    g_rebs[1].name = "glInvalidateFramebuffer"; g_rebs[1].replacement = (void *)hk_glInvalidateFramebuffer;
    g_rebs[1].replaced = &fh_orig[F_glInvalidateFramebuffer];
    g_rebs[2].name = "glUniformMatrix4fv";      g_rebs[2].replacement = (void *)hk_glUniformMatrix4fv;
    g_rebs[2].replaced = &fh_orig[F_glUniformMatrix4fv];

    const char *app = NULL;
    for (const char *p = strstr(exe, ".app/"); p; p = strstr(p + 1, ".app/")) app = p;     /* the last ".app/" */
    if (app) {
        const char *s = app;
        while (s > exe && s[-1] != '/') s--;
        snprintf(g_match, sizeof g_match, "/%.*s/", (int)(app + 4 - s), s);
        g_match_prefix = 0;
    } else {
        snprintf(g_match, sizeof g_match, "%s", exe);
        char *slash = strrchr(g_match, '/');
        if (slash) slash[1] = 0;
        g_match_prefix = 1;
    }
    _dyld_register_func_for_add_image(on_add_image);      /* runs for every loaded image right now, then for each new one */
    logf_("hooks: fishhook (dlsym + 2 GL names) on images matching '%s' (%s), %d patched so far\n",
          g_match, g_match_prefix ? "prefix" : "bundle folder", atomic_load(&g_patched));
}

__attribute__((constructor))
static void resa_ao_ctor(void) {
    g_t0 = now_us();
    open_log();
    char exe[1024] = "", model[64] = "?", osb[64] = "?";
    uint32_t xs = sizeof exe;
    if (_NSGetExecutablePath(exe, &xs) != 0) exe[0] = 0;
    size_t ms = sizeof model, os = sizeof osb;
    sysctlbyname("hw.machine", model, &ms, NULL, 0);
    sysctlbyname("kern.osversion", osb, &os, NULL, 0);
    Dl_info me;
    logf_("resa_ao %s  pid=%d  device=%s  osBuild=%s\nexe: %s\ndylib: %s\nlog dir: %s\n", AO_VERSION, (int)getpid(), model, osb,
          exe, dladdr((void *)resa_ao_ctor, &me) && me.dli_fname ? me.dli_fname : "?", g_dir);
    cfg_load(1);
    ao_init();
    install_objc();
    logf_("objc: +[EAGLContext setCurrentContext:] %s, -[UIApplication sendEvent:] %s%s\n",
          orig_setCurrentContext ? "swizzled" : "NOT FOUND (will retry)", orig_sendEvent ? "swizzled" : "NOT FOUND (will retry)",
          g_cfg.touch_toggle ? "" : "  (gesture disabled in the ini)");
    install_fishhook(exe);
}
#endif /* !RESA_TEST */

/* ------------------------------------------------------------------ Linux test entry points (tests/smoke_gl.c) */
#ifdef RESA_TEST
void resa_test_init(void)                                  { g_t0 = now_us(); open_log(); cfg_load(1); ao_init(); }
void resa_test_invalidate(GLenum t, GLsizei n, const GLenum *a) { hk_glInvalidateFramebuffer(t, n, a); }
void resa_test_matrix(GLint l, GLsizei c, GLboolean tr, const GLfloat *v) { hk_glUniformMatrix4fv(l, c, tr, v); }
void *resa_test_dlsym(void *h, const char *n, void *(*real)(void *, const char *)) { real_dlsym = real; return hk_dlsym(h, n); }
int  resa_test_gesture(Gest *g, int active, int need, uint64_t t_us) { return gest_feed(g, active, need, t_us); }
void resa_test_apply(int act)                         { if (act == 1) ao_toggle_user(); else if (act == 2) ao_cycle_view(); }
#endif
