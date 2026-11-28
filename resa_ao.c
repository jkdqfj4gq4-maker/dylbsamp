/*
 * resa_ao.c  --  screen-space ambient occlusion (SSAO) for ReSA on iOS.   v1.4
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
 *   Documents/custom.frag   (optional: use_custom_shader=1 in the ini, file name(s) = custom_shader= in the ini)
 *                           your own post-processing.  It runs as the very last pass(es), ON TOP of AO + reflections, and is
 *                           re-read about once a second, so saving a file while playing is enough.  Write only the GLSL body
 *                           (void main() plus helpers): the loader adds #version, precision, uScene, uInput, uDepth, uProj, uSize,
 *                           uTime, oColor and the helpers linZ()/zAt()/posAt().  A broken file never hurts the game: the log says
 *                           why (file, pass, line) and the normal look stays.  Runs only while AO is on (view=1); the debug views
 *                           are never post-processed.
 *                           MULTI-PASS (v1.3):  custom_shader=dof.frag, grain.frag   = one pass per file, in that order
 *                                               //@pass name   (a line of its own inside a file) starts a new pass; whatever stands
 *                                               before the first marker is shared by all the passes of that file
 *                           uScene = what the previous pass drew (first pass: AO + reflections), uInput = the image that entered the
 *                           chain (the same for every pass), uDepth = depth.  At most 8 files and 8 passes in all.
 *                           SHADERTOY (v1.4):   a Shadertoy "Image" shader pastes in as it is: void mainImage(out vec4, in vec2) plus
 *                                               iChannel0 (= uScene), iChannel1 (= uInput), iChannel2 (= uDepth), iChannel3 (= uLut),
 *                                               iResolution, iTime, iFrame, iTimeDelta, iMouse (always 0), iDate, texture2D() ...  The loader
 *                                               writes main() for you and ignores #version / #extension lines and uniform declarations of those
 *                                               names.  The output alpha is replaced by the scene's own alpha.
 *                           LUT (v1.4):         uLut = Documents/<custom_lut> (ini; default lut.png), any PNG (8 / 16 bit, interlaced or not).
 *                                               The loader decodes it itself (no ImageIO), uploads it as an RGBA8 texture on unit 4 and reloads
 *                                               it when the file changes.  No file = a neutral LUT, so a LUT shader then changes nothing.  It is
 *                                               only loaded while a pass mentions uLut / iChannel3.  The sample lut.frag reads the usual layouts.
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

#define AO_VERSION "v1.4"

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
#define GLC_LINEAR                             0x2601
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
#define GLC_PIXEL_UNPACK_BUFFER                0x88EC
#define GLC_PIXEL_UNPACK_BUFFER_BINDING        0x88EF
#define GLC_UNPACK_ROW_LENGTH                  0x0CF2
#define GLC_UNPACK_SKIP_ROWS                   0x0CF3
#define GLC_UNPACK_SKIP_PIXELS                 0x0CF4
#define GLC_UNPACK_ALIGNMENT                   0x0CF5

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
    X(void,     Uniform1f,              (GLint, GLfloat)) \
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
    X(GLboolean, IsTexture,             (GLuint)) \
    X(void,     TexImage2D,             (GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const void *)) \
    X(void,     BindBuffer,             (GLenum, GLuint))

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
    char  custom_shader_file[1024];  /* file name(s) inside Documents: "custom.frag" or "dof.frag, grain.frag" (comma separated, in pass order) */
    int   use_custom_shader;         /* 1 = run that file / those files as the very last pass(es), 0 = off */
    char  custom_lut[128];           /* PNG inside Documents that the custom shaders sample as uLut / iChannel3 (default lut.png) */
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
    c->use_custom_shader = 0;
    snprintf(c->custom_shader_file, sizeof c->custom_shader_file, "custom.frag");
    snprintf(c->custom_lut, sizeof c->custom_lut, "lut.png");
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
    else if (!strcmp(k, "use_custom_shader")) c->use_custom_shader = i ? 1 : 0;
    else if (!strcmp(k, "custom_lut")) {                   /* one PNG inside Documents (uLut / iChannel3 in the custom shaders) */
        size_t n = strlen(v);
        while (n > 0 && (v[n - 1] == '\n' || v[n - 1] == '\r' || v[n - 1] == ' ' || v[n - 1] == '\t')) n--;
        if (n > 0 && n < sizeof c->custom_lut && !memchr(v, '/', n)) { memcpy(c->custom_lut, v, n); c->custom_lut[n] = 0; }
        else if (n > 0) logf_("config: custom_lut ignored (%s) - use one plain file name inside Documents\n", n >= sizeof c->custom_lut ? "too long" : "it contains a '/'");
    }
    else if (!strcmp(k, "custom_shader")) {                /* plain file name(s) inside Documents, comma separated; trailing blanks / CR / LF are not part of it */
        size_t n = strlen(v);
        while (n > 0 && (v[n - 1] == '\n' || v[n - 1] == '\r' || v[n - 1] == ' ' || v[n - 1] == '\t')) n--;
        if (n > 0 && n < sizeof c->custom_shader_file && !memchr(v, '/', n)) { memcpy(c->custom_shader_file, v, n); c->custom_shader_file[n] = 0; }
        else if (n > 0) logf_("config: custom_shader ignored (%s) - use plain file names inside Documents, at most %d characters in all\n",
                              n >= sizeof c->custom_shader_file ? "too long" : "it contains a '/'", (int)sizeof c->custom_shader_file - 1);
    }
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
    "; your own post-processing shader(s): GLSL files in the Documents folder (see the sample custom.frag), run as the very last passes\n"
    "; on top of AO + reflections.  1 = on.  They reload by themselves about once a second when you save a file.  If one has an error\n"
    "; the normal look stays and resa_ao_log.txt shows file, pass and line.  One file = one pass.  Several files (comma separated) run\n"
    "; one after the other, e.g.  custom_shader=dof.frag, grain.frag  and a file can hold several passes itself: start each one with a\n"
    "; line  //@pass name  (at most 8 passes in all).  Write the file name(s) alone, nothing after them on the line.\n"
    "; Shadertoy shaders paste in unchanged (void mainImage + iChannel0 / iResolution / iTime ...): the loader writes main() itself.\n"
    "use_custom_shader=0\n"
    "custom_shader=custom.frag\n"
    "; a colour-grade PNG for the custom shaders (uLut, or iChannel3 in a Shadertoy shader): any LUT picture from ReShade / ENB / photo apps,\n"
    "; copied into Documents under this name.  It reloads by itself when the file changes; without the file a neutral LUT is used (no grade).\n"
    "custom_lut=lut.png\n"
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
        char line[1280];                                  /* long enough for a custom_shader= list of 8 names */
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
    if (announce)
        logf_("config: custom shader %s  file(s)='%s'  lut='%s'\n", c.use_custom_shader ? "ON" : "off", c.custom_shader_file, c.custom_lut);
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

static const char FS_SSR_SRC[] =
    GLSL_HDR GLSL_COMMON
    "uniform sampler2D uScene;\n"
    "uniform vec3 uSsr;\n"                                  /* x = intensity, y = max ray length (m), z = steps */
    "float ign(vec2 p){ return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }\n"
    "vec2 toPx(vec3 q){ return (q.xy / (uProj.xy * q.z) * 0.5 + 0.5) * uSize.xy; }\n"
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
    "float edgeFade(vec2 uv){ return smoothstep(0.0, 0.15, min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y))); }\n"
    "void main(){\n"
    "  oColor = vec4(0.0, 0.0, 0.0, 1.0);\n"
    "  ivec2 ip = ivec2(gl_FragCoord.xy);\n"
    "  if (texelFetch(uDepth, ip, 0).r >= 0.999999) return;\n"     /* sky */
    "  vec3 P = posAt(ip);\n"
    "  if (P.z > 300.0) return;\n"
    "  vec3 N = nrmFrom(ip, P);\n"
    "  float flo = smoothstep(0.62, 0.82, N.y);\n"                 /* only flat ground */
    "  if (flo <= 0.0) return;\n"
    "  vec3 base = texelFetch(uScene, ip, 0).rgb;\n"
    "  float veg = smoothstep(0.02, 0.12, base.g - max(base.r, base.b));\n"
    "  float mat = (1.0 - 0.9 * veg);\n"                           /* less reflection on grass */
    "  vec3 V = normalize(P);\n"
    "  float NoV = clamp(-dot(N, V), 0.0, 1.0);\n"
    "  float fr = pow(1.0 - NoV, 5.0);\n"                          /* Fresnel: grazing angles reflect more */
    "  float wgt = uSsr.x * (0.15 + 0.85 * fr) * flo * mat;\n"
    "  if (wgt < 0.004) return;\n"
    "  vec3 R = reflect(V, N);\n"
    "  vec3 q0 = P + N * (0.03 + 0.004 * P.z);\n"
    "  float tmax = uSsr.y;\n"
    "  if (R.z < 0.0) tmax = min(tmax, (0.3 - q0.z) / R.z);\n"
    "  if (tmax < 0.1) return;\n"
    "  vec3 q1 = q0 + R * tmax;\n"
    "  vec2 p0 = toPx(q0);\n"
    "  vec2 dd = toPx(q1) - p0;\n"
    "  float k0 = 1.0 / q0.z;\n"
    "  float k1 = 1.0 / q1.z;\n"
    "  vec2 tb = vec2(1e9);\n"
    "  if (dd.x >  1e-3) tb.x = (uSize.x - 1.0 - p0.x) / dd.x;\n"
    "  if (dd.x < -1e-3) tb.x = (1.0 - p0.x) / dd.x;\n"
    "  if (dd.y >  1e-3) tb.y = (uSize.y - 1.0 - p0.y) / dd.y;\n"
    "  if (dd.y < -1e-3) tb.y = (1.0 - p0.y) / dd.y;\n"
    "  float smax = clamp(min(tb.x, tb.y), 0.0, 1.0);\n"
    "  float lpx = length(dd) * smax;\n"
    "  bool hit = false;\n"
    "  float sHit = 0.0;\n"
    "  if (lpx >= 2.0) {\n"
    "    int n = int(clamp(uSsr.z + 0.5, 4.0, 64.0));\n"
    "    n = min(n, int(lpx * 0.5) + 2);\n"
    "    float inv = 1.0 / float(n);\n"
    "    float jit = ign(gl_FragCoord.xy);\n"
    "    float sPrev = 0.0;\n"
    "    float rzPrev = q0.z;\n"
    "    float dPrev = -1.0;\n"
    "    for (int i = 0; i < n; i++) {\n"
    "      float u = (float(i) + 0.5 + 0.5 * jit) * inv;\n"
    "      float s = smax * (0.35 * u + 0.65 * u * u);\n"
    "      float rz = 1.0 / (k0 + (k1 - k0) * s);\n"
    "      float df = rz - zAt(ivec2(p0 + dd * s));\n"
    "      float th = abs(rz - rzPrev) + 0.12 + 0.04 * rz;\n"
    "      if (df > 0.0 && df < th && dPrev <= 0.0) {\n"
    "        float sa = sPrev;\n"
    "        float sb = s;\n"
    "        for (int b = 0; b < 6; b++) {\n"
    "          float sm = 0.5 * (sa + sb);\n"
    "          float dm = 1.0 / (k0 + (k1 - k0) * sm) - zAt(ivec2(p0 + dd * sm));\n"
    "          if (dm > 0.0) sb = sm; else sa = sm;\n"
    "        }\n"
    "        ivec2 hp = ivec2(p0 + dd * sb);\n"
    "        float db = 1.0 / (k0 + (k1 - k0) * sb) - zAt(hp);\n"
    "        if (db < th) {\n"
    "          vec3 Nh = nrmFrom(hp, posAt(hp));\n"
    "          if (dot(Nh, R) < 0.05) { hit = true; sHit = sb; break; }\n"
    "        }\n"
    "      }\n"
    "      dPrev = df;\n"
    "      sPrev = s;\n"
    "      rzPrev = rz;\n"
    "    }\n"
    "  }\n"
    "  if (hit) {\n"
    "    vec2 hpx = p0 + dd * sHit;\n"
    "    vec2 huv = hpx / uSize.xy;\n"
    "    float edge = edgeFade(huv);\n"                           /* fade out if the hit is near the screen edge */
    "    if (edge > 0.0) {\n"
    "      ivec2 hp = ivec2(hpx);\n"
    "      int rb = int(clamp(0.5 + 0.04 * length(dd) * sHit, 0.0, 5.0));\n"
    "      vec3 col = texelFetch(uScene, clampP(hp), 0).rgb * 0.4;\n"
    "      col += (texelFetch(uScene, clampP(hp + ivec2(rb, 0)), 0).rgb + texelFetch(uScene, clampP(hp - ivec2(rb, 0)), 0).rgb\n"
    "            + texelFetch(uScene, clampP(hp + ivec2(0, rb)), 0).rgb + texelFetch(uScene, clampP(hp - ivec2(0, rb)), 0).rgb) * 0.15;\n"
    "      float wf = 1.0 - exp(-wgt);\n"
    "      oColor = vec4(col * (1.0 - base) * wf * edge, 1.0);\n"
    "    }\n"
    "  }\n"
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
    GLuint tex_pp[2], fbo_pp[2];  /* custom chain: ping-pong targets between its passes, made on first use (res_ensure_pp), 0 = not made */
    GLuint vao;
    GLuint tex_lut;               /* uLut: the user's PNG (or the neutral LUT) on unit LUT_UNIT; not size dependent, made by lut_sync() while a custom pass uses it */
    GLuint vs;                    /* the shared vertex shader, kept alive so load_custom_shader() can link every chain pass against it */
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
}
static void tex_filter(GLuint tex, GLint filter) {         /* bilinear for the images a custom shader may sample with texture(); texelFetch() ignores it */
    gl.BindTexture(GLC_TEXTURE_2D, tex);
    gl.TexParameteri(GLC_TEXTURE_2D, GLC_TEXTURE_MIN_FILTER, filter);
    gl.TexParameteri(GLC_TEXTURE_2D, GLC_TEXTURE_MAG_FILTER, filter);
}
/* ====================================================================================================================
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
 *   straight into the engine's scene FBO) -> [optional custom shader chain, ao_frame step 5] -> restore GL state.
 *   The engine's own invalidate then runs as usual.
 *
 * SAFETY: all objects are ours; all state we touch is saved and restored; any shader/FBO build failure, or a GL error on
 *   two frames in a row, switches AO off for the run and says why in resa_ao_log.txt.  A single GL error rebuilds our
 *   size-dependent objects once (covers a render-scale change) and aborts that frame BEFORE it writes into the scene.
 * ==================================================================================================================== */

#ifndef GLC_RENDERBUFFER_INTERNAL_FORMAT
#define GLC_RENDERBUFFER_INTERNAL_FORMAT 0x8D44
#endif

#define AO_UNITS 5            /* texture units we use: 0 = depth copy, 1 = AO texture (custom chain: uInput), 2 = scene colour copy (chain: uScene), 3 = SSR texture, 4 = LUT (chain: uLut) */

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

/* ------------------------------------------------------------------ PNG decoder (for the LUT / user texture: Documents/lut.png)
 * Pure C: its own inflate (no libz) and no ImageIO / CoreGraphics, so the dylib keeps "no link dependency beyond libSystem/libobjc", and the
 * pixel values are exactly the bytes in the file (no colour management: a LUT is data, not a picture).
 * Understands every PNG a LUT can come in: grey / grey+alpha / RGB / RGBA / palette (+tRNS), 1..16 bits per sample, interlaced or not.
 * 16-bit samples are rounded to 8 bits; gAMA / sRGB / iCCP and the CRCs are ignored.  Output is always RGBA8, row 0 = TOP row of the file. */
#define PNG_MAX_FILE    (32u * 1024u * 1024u)     /* file size cap */
#define PNG_MAX_DIM     4096                      /* per side */
#define PNG_MAX_PIXELS  (4u * 1024u * 1024u)      /* in total (2048 x 2048): a 16-bit RGBA one is 32 MB of raw data while decoding */

typedef struct {
    const unsigned char *in; size_t inl, ip;
    unsigned long bitbuf; int bitcnt;
    unsigned char *out; size_t outl, op;
    int err;
} Inf;
typedef struct { short count[16]; short symbol[288]; } InfHuff;

static int inf_bits(Inf *s, int need) {
    unsigned long v = s->bitbuf;
    while (s->bitcnt < need) {
        if (s->ip >= s->inl) { s->err = 1; return 0; }
        v |= (unsigned long)s->in[s->ip++] << s->bitcnt;
        s->bitcnt += 8;
    }
    s->bitbuf = v >> need;
    s->bitcnt -= need;
    return (int)(v & ((1UL << need) - 1UL));
}
/* canonical Huffman decoding, one bit at a time (slow but tiny, and a LUT is a few hundred KB at most) */
static int inf_decode(Inf *s, const InfHuff *h) {
    int code = 0, first = 0, index = 0;
    for (int len = 1; len <= 15; len++) {
        code |= inf_bits(s, 1);
        if (s->err) return -1;
        int count = h->count[len];
        if (code - count < first) return h->symbol[index + (code - first)];
        index += count; first += count; first <<= 1; code <<= 1;
    }
    return -1;
}
/* returns 0 = complete code, >0 = incomplete, <0 = over-subscribed */
static int inf_build(InfHuff *h, const short *length, int n) {
    short offs[16];
    for (int len = 0; len < 16; len++) h->count[len] = 0;
    for (int sym = 0; sym < n; sym++) h->count[length[sym]]++;
    if (h->count[0] == n) return 0;
    int left = 1;
    for (int len = 1; len < 16; len++) { left <<= 1; left -= h->count[len]; if (left < 0) return left; }
    offs[1] = 0;
    for (int len = 1; len < 15; len++) offs[len + 1] = (short)(offs[len] + h->count[len]);
    for (int sym = 0; sym < n; sym++) if (length[sym] != 0) h->symbol[offs[length[sym]]++] = (short)sym;
    return left;
}
static int inf_codes(Inf *s, const InfHuff *lc, const InfHuff *dc) {
    static const short lens[29]  = { 3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258 };
    static const short lext[29]  = { 0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0 };
    static const short dists[30] = { 1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577 };
    static const short dext[30]  = { 0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13 };
    for (;;) {
        int sym = inf_decode(s, lc);
        if (sym < 0) return -1;
        if (sym < 256) {
            if (s->op >= s->outl) return -1;
            s->out[s->op++] = (unsigned char)sym;
        } else if (sym == 256) {
            return 0;
        } else {
            sym -= 257;
            if (sym >= 29) return -1;
            int len = lens[sym] + inf_bits(s, lext[sym]);
            int ds = inf_decode(s, dc);
            if (ds < 0 || ds >= 30) return -1;
            size_t dist = (size_t)dists[ds] + (size_t)inf_bits(s, dext[ds]);
            if (s->err || dist > s->op || s->op + (size_t)len > s->outl) return -1;
            for (int i = 0; i < len; i++) { s->out[s->op] = s->out[s->op - dist]; s->op++; }
        }
    }
}
static int inf_stored(Inf *s) {
    s->bitbuf = 0; s->bitcnt = 0;                     /* the bit buffer never holds more than the rest of one byte: dropping it = byte alignment */
    if (s->ip + 4 > s->inl) return -1;
    unsigned len  = (unsigned)s->in[s->ip]     | ((unsigned)s->in[s->ip + 1] << 8);
    unsigned nlen = (unsigned)s->in[s->ip + 2] | ((unsigned)s->in[s->ip + 3] << 8);
    s->ip += 4;
    if (len != (~nlen & 0xFFFFu)) return -1;
    if (s->ip + len > s->inl || s->op + len > s->outl) return -1;
    memcpy(s->out + s->op, s->in + s->ip, len);
    s->ip += len; s->op += len;
    return 0;
}
static int inf_fixed(Inf *s) {
    short lengths[288];
    InfHuff lc, dc;
    int sym;
    for (sym = 0; sym < 144; sym++) lengths[sym] = 8;
    for (; sym < 256; sym++) lengths[sym] = 9;
    for (; sym < 280; sym++) lengths[sym] = 7;
    for (; sym < 288; sym++) lengths[sym] = 8;
    inf_build(&lc, lengths, 288);
    for (sym = 0; sym < 30; sym++) lengths[sym] = 5;
    inf_build(&dc, lengths, 30);
    return inf_codes(s, &lc, &dc);
}
static int inf_dynamic(Inf *s) {
    static const short order[19] = { 16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15 };
    short lengths[320];
    InfHuff lc, dc;
    int nlen = inf_bits(s, 5) + 257, ndist = inf_bits(s, 5) + 1, ncode = inf_bits(s, 4) + 4;
    if (s->err || nlen > 286 || ndist > 30) return -1;
    int idx;
    for (idx = 0; idx < ncode; idx++) lengths[order[idx]] = (short)inf_bits(s, 3);
    for (; idx < 19; idx++) lengths[order[idx]] = 0;
    if (s->err || inf_build(&lc, lengths, 19) != 0) return -1;
    idx = 0;
    while (idx < nlen + ndist) {
        int sym = inf_decode(s, &lc);
        if (sym < 0) return -1;
        if (sym < 16) { lengths[idx++] = (short)sym; continue; }
        int len = 0, rep;
        if (sym == 16) { if (idx == 0) return -1; len = lengths[idx - 1]; rep = 3 + inf_bits(s, 2); }
        else if (sym == 17) rep = 3 + inf_bits(s, 3);
        else rep = 11 + inf_bits(s, 7);
        if (s->err || idx + rep > nlen + ndist) return -1;
        while (rep--) lengths[idx++] = (short)len;
    }
    if (lengths[256] == 0) return -1;                 /* no end-of-block code */
    int e = inf_build(&lc, lengths, nlen);
    if (e < 0 || (e > 0 && nlen - lc.count[0] != 1)) return -1;
    e = inf_build(&dc, lengths + nlen, ndist);
    if (e < 0 || (e > 0 && ndist - dc.count[0] != 1)) return -1;
    return inf_codes(s, &lc, &dc);
}
/* a zlib stream that must inflate to exactly outl bytes */
static int inf_zlib(const unsigned char *in, size_t inl, unsigned char *out, size_t outl) {
    if (inl < 2 || (in[0] & 0x0F) != 8 || ((((unsigned)in[0] << 8) | in[1]) % 31u) != 0 || (in[1] & 0x20)) return -1;
    Inf s;
    memset(&s, 0, sizeof s);
    s.in = in + 2; s.inl = inl - 2; s.out = out; s.outl = outl;
    int last;
    do {
        last = inf_bits(&s, 1);
        int type = inf_bits(&s, 2);
        if (s.err) return -1;
        int r = type == 0 ? inf_stored(&s) : type == 1 ? inf_fixed(&s) : type == 2 ? inf_dynamic(&s) : -1;
        if (r != 0) return -1;
    } while (!last);
    return s.op == outl ? 0 : -1;
}

static uint32_t png_be32(const unsigned char *p) { return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3]; }

/* undo the per-row filters of `rows` scanlines (each: 1 filter byte + rb data bytes) in place */
static int png_unfilter(unsigned char *d, unsigned rows, size_t rb, size_t bpp) {
    const unsigned char *prev = NULL;
    for (unsigned y = 0; y < rows; y++) {
        unsigned char *row = d + (size_t)y * (rb + 1);
        int ft = row[0];
        unsigned char *x = row + 1;
        if (ft > 4) return -1;
        for (size_t i = 0; i < rb; i++) {
            int a = i >= bpp ? x[i - bpp] : 0;
            int b = prev ? prev[i] : 0;
            int c = (prev && i >= bpp) ? prev[i - bpp] : 0;
            int v = x[i];
            switch (ft) {
                case 1: v += a; break;
                case 2: v += b; break;
                case 3: v += (a + b) >> 1; break;
                case 4: {
                    int p = a + b - c, pa = abs(p - a), pb = abs(p - b), pc = abs(p - c);
                    v += (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);
                    break;
                }
                default: break;
            }
            x[i] = (unsigned char)v;
        }
        prev = x;
    }
    return 0;
}

typedef struct { int ct, depth, spp; unsigned char plte[768]; int nplte; unsigned char trns[256]; int ntrns; } PngFmt;

static unsigned png_s16(unsigned v) { return (v * 255u + 32767u) / 65535u; }

/* pixel x of an unfiltered scanline -> RGBA8 */
static void png_px(const PngFmt *f, const unsigned char *row, unsigned x, unsigned char *o) {
    unsigned v;
    switch (f->ct) {
        case 0: case 3: {                                      /* one sample per pixel: grey or palette index, 1..16 bits */
            if (f->depth == 16) { v = png_s16(((unsigned)row[x * 2] << 8) | row[x * 2 + 1]); }
            else if (f->depth == 8) v = row[x];
            else {
                unsigned per = 8u / (unsigned)f->depth, sh = 8u - (unsigned)f->depth * (x % per + 1u);
                v = (row[x / per] >> sh) & ((1u << f->depth) - 1u);
            }
            if (f->ct == 3) {
                if ((int)v < f->nplte) { o[0] = f->plte[v * 3]; o[1] = f->plte[v * 3 + 1]; o[2] = f->plte[v * 3 + 2]; }
                else { o[0] = o[1] = o[2] = 0; }
                o[3] = (int)v < f->ntrns ? f->trns[v] : 255;
            } else {
                if (f->depth < 8) v = v * 255u / ((1u << f->depth) - 1u);
                o[0] = o[1] = o[2] = (unsigned char)v; o[3] = 255;
            }
            return;
        }
        default: {                                             /* 2 = RGB, 4 = grey+alpha, 6 = RGBA; 8 or 16 bits */
            const unsigned char *p = row + (size_t)x * (size_t)f->spp * (size_t)(f->depth / 8);
            unsigned s[4];
            for (int i = 0; i < f->spp; i++) s[i] = f->depth == 16 ? png_s16(((unsigned)p[i * 2] << 8) | p[i * 2 + 1]) : p[i];
            if (f->ct == 4) { o[0] = o[1] = o[2] = (unsigned char)s[0]; o[3] = (unsigned char)s[1]; }
            else { o[0] = (unsigned char)s[0]; o[1] = (unsigned char)s[1]; o[2] = (unsigned char)s[2]; o[3] = f->ct == 6 ? (unsigned char)s[3] : 255; }
            return;
        }
    }
}

/* Decodes a PNG file held in memory into a fresh RGBA8 buffer (caller frees).  NULL + *why on failure. */
static unsigned char *png_decode(const unsigned char *f, size_t n, int *ow, int *oh, const char **why) {
    static const unsigned char sig[8] = { 0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A };
    static const unsigned char ax[7] = { 0, 4, 0, 2, 0, 1, 0 }, ay[7] = { 0, 0, 4, 0, 2, 0, 1 };
    static const unsigned char dx[7] = { 8, 8, 4, 4, 2, 2, 1 }, dy[7] = { 8, 8, 8, 4, 4, 2, 2 };
    unsigned char *idat = NULL, *raw = NULL, *rgba = NULL;
    size_t idat_n = 0, pos = 8;
    unsigned w = 0, h = 0;
    int have_ihdr = 0, interlace = 0, seen_end = 0;
    PngFmt fmt;
    memset(&fmt, 0, sizeof fmt);
    *why = "not a PNG file";
    if (n < 8 + 12 || memcmp(f, sig, 8) != 0) return NULL;
    while (pos + 12 <= n && !seen_end) {
        uint32_t len = png_be32(f + pos);
        const unsigned char *type = f + pos + 4, *data = f + pos + 8;
        if (len > n - pos - 12) { *why = "file is cut off (a chunk runs past the end)"; goto fail; }
        if (!memcmp(type, "IHDR", 4)) {
            if (len != 13) { *why = "bad IHDR"; goto fail; }
            w = png_be32(data); h = png_be32(data + 4);
            fmt.depth = data[8]; fmt.ct = data[9];
            interlace = data[12];
            if (data[10] != 0 || data[11] != 0 || interlace > 1) { *why = "unsupported PNG variant"; goto fail; }
            have_ihdr = 1;
        } else if (!memcmp(type, "PLTE", 4)) {
            if (len % 3 != 0 || len > 768) { *why = "bad palette"; goto fail; }
            memcpy(fmt.plte, data, len); fmt.nplte = (int)(len / 3);
        } else if (!memcmp(type, "tRNS", 4)) {
            fmt.ntrns = (int)(len < 256 ? len : 256);
            memcpy(fmt.trns, data, (size_t)fmt.ntrns);
        } else if (!memcmp(type, "IDAT", 4)) {
            if (idat_n + len > PNG_MAX_FILE) { *why = "too much image data"; goto fail; }
            unsigned char *g = (unsigned char *)realloc(idat, idat_n + len + 1);
            if (!g) { *why = "out of memory"; goto fail; }
            idat = g;
            memcpy(idat + idat_n, data, len);
            idat_n += len;
        } else if (!memcmp(type, "IEND", 4)) {
            seen_end = 1;
        }
        pos += 12 + (size_t)len;
    }
    if (!have_ihdr || idat_n == 0) { *why = "no image data"; goto fail; }
    switch (fmt.ct) {
        case 0: fmt.spp = 1; if (fmt.depth != 1 && fmt.depth != 2 && fmt.depth != 4 && fmt.depth != 8 && fmt.depth != 16) fmt.spp = 0; break;
        case 2: fmt.spp = 3; if (fmt.depth != 8 && fmt.depth != 16) fmt.spp = 0; break;
        case 3: fmt.spp = 1; if (fmt.depth != 1 && fmt.depth != 2 && fmt.depth != 4 && fmt.depth != 8) fmt.spp = 0; break;
        case 4: fmt.spp = 2; if (fmt.depth != 8 && fmt.depth != 16) fmt.spp = 0; break;
        case 6: fmt.spp = 4; if (fmt.depth != 8 && fmt.depth != 16) fmt.spp = 0; break;
        default: fmt.spp = 0; break;
    }
    if (!fmt.spp) { *why = "unsupported colour type / bit depth"; goto fail; }
    if (fmt.ct == 3 && fmt.nplte == 0) { *why = "palette image without a palette"; goto fail; }
    if (w == 0 || h == 0 || w > PNG_MAX_DIM || h > PNG_MAX_DIM || (uint64_t)w * h > PNG_MAX_PIXELS) { *why = "image too big (at most 4096 per side, 4 million pixels)"; goto fail; }

    {
        const unsigned bits = (unsigned)(fmt.depth * fmt.spp);
        const size_t bpp = bits >= 8 ? bits / 8 : 1;
        size_t raw_n = 0;
        if (!interlace) raw_n = (((size_t)w * bits + 7) / 8 + 1) * h;
        else for (int p = 0; p < 7; p++) {
            unsigned pw = w > ax[p] ? (w - ax[p] + dx[p] - 1) / dx[p] : 0, ph = h > ay[p] ? (h - ay[p] + dy[p] - 1) / dy[p] : 0;
            if (pw && ph) raw_n += (((size_t)pw * bits + 7) / 8 + 1) * ph;
        }
        raw = (unsigned char *)malloc(raw_n);
        rgba = (unsigned char *)malloc((size_t)w * h * 4);
        if (!raw || !rgba) { *why = "out of memory"; goto fail; }
        if (inf_zlib(idat, idat_n, raw, raw_n) != 0) { *why = "corrupt or cut-off image data"; goto fail; }
        free(idat); idat = NULL;

        if (!interlace) {
            const size_t rb = ((size_t)w * bits + 7) / 8;
            if (png_unfilter(raw, h, rb, bpp) != 0) { *why = "bad scanline filter"; goto fail; }
            for (unsigned y = 0; y < h; y++) {
                const unsigned char *row = raw + (size_t)y * (rb + 1) + 1;
                for (unsigned x = 0; x < w; x++) png_px(&fmt, row, x, rgba + ((size_t)y * w + x) * 4);
            }
        } else {
            size_t off = 0;
            for (int p = 0; p < 7; p++) {
                unsigned pw = w > ax[p] ? (w - ax[p] + dx[p] - 1) / dx[p] : 0, ph = h > ay[p] ? (h - ay[p] + dy[p] - 1) / dy[p] : 0;
                if (!pw || !ph) continue;
                const size_t rb = ((size_t)pw * bits + 7) / 8;
                if (png_unfilter(raw + off, ph, rb, bpp) != 0) { *why = "bad scanline filter"; goto fail; }
                for (unsigned py = 0; py < ph; py++) {
                    const unsigned char *row = raw + off + (size_t)py * (rb + 1) + 1;
                    for (unsigned px = 0; px < pw; px++)
                        png_px(&fmt, row, px, rgba + (((size_t)ay[p] + (size_t)py * dy[p]) * w + ax[p] + (size_t)px * dx[p]) * 4);
                }
                off += (rb + 1) * ph;
            }
        }
    }
    free(raw);
    *ow = (int)w; *oh = (int)h; *why = NULL;
    return rgba;
fail:
    free(idat); free(raw); free(rgba);
    return NULL;
}

/* ------------------------------------------------------------------ the LUT / user texture: Documents/<custom_lut> -> R.tex_lut (texture unit LUT_UNIT, uLut / iChannel3) */
#define LUT_UNIT 4
typedef struct {
    char name[128]; int known, present; time_t mtime; long long size;     /* what we saw of the file last time (known = 0: nothing looked at yet, or forgotten after a context change) */
    int from_file, w, h;                                                  /* what R.tex_lut is made of: from_file = the PNG, 0 = the neutral LUT */
} LutSig;
static LutSig g_lut;
static void lut_forget(void) { memset(&g_lut, 0, sizeof g_lut); }          /* the names in R.tex_lut are forgotten by the memset of R itself */

/* ------------------------------------------------------------------ custom shader chain (Documents/<custom_shader>, hot-reloaded)
 * The user's files are only the BODY of fragment shaders.  We put CUSTOM_HDR in front of each pass, compile it, link it with the shared
 * vertex shader and run the passes in order as the very last stage (ao_frame, step 5), reading the finished AO + reflections image.
 *
 *   custom_shader=custom.frag                one file, one pass (exactly what v1.2 did)
 *   custom_shader=dof.frag, grain.frag       two files = two passes, in this order (commas; at most CUSTOM_MAX_FILES files)
 *   a file with  //@pass [name]  lines       the file is cut at those lines and every section is its own pass.  Whatever stands BEFORE the
 *                                            first marker is shared: it is put in front of every section of that file (constants, helpers).
 *
 * Pass k sees   uScene = the output of pass k-1  (pass 1: the AO + reflections image)
 *               uInput = the image that entered the chain (AO + reflections), the same in every pass
 *               uDepth = the depth copy of this frame, with uProj / uSize / linZ() / zAt() / posAt() from GLSL_COMMON, and uTime (seconds)
 * Every pass but the last renders into one of two ping-pong buffers; only the last pass writes the engine's scene FBO.
 * The images are RGBA8 and bilinear-filtered (texture() and texelFetch() both work).  Alpha is carried through the chain: a pass may use
 * it as scratch space (the DOF sample keeps the circle of confusion there), the LAST pass should put the scene's alpha back (uInput.a).
 * ALL OR NOTHING: the chain is used only if every pass of every listed file built.  Any problem (missing file, compile / link error,
 * GL error) is logged and the built-in look stays: it can never take AO down. */
#define CUSTOM_PATH_MAX    760
#define CUSTOM_MAX_BYTES   (256 * 1024)          /* per file */
#define CUSTOM_MAX_FILES   8
#define CUSTOM_MAX_PASSES  8                     /* in total, over all files */
#define CUSTOM_NAME_MAX    128

typedef struct {
    GLuint prog;
    GLint  uScene, uInput, uDepth, uProj, uSize, uTime, uLut;   /* -1 = this pass does not use it */
    char   label[CUSTOM_NAME_MAX + 64];                     /* for the log: "dof.frag" or "dof.frag pass 2/3 'gather'" */
} CPass;
static CPass g_cp[CUSTOM_MAX_PASSES];
static int   g_cp_n;                              /* passes in the chain; 0 = no custom shader active, ao_frame uses the built-in composite */

/* what we looked at last time: the list and, per file, existence + mtime + size.  A change in any of it rebuilds the whole chain. */
typedef struct { char name[CUSTOM_NAME_MAX]; int present; time_t mtime; long long size; } CFileSig;
static CFileSig g_csig[CUSTOM_MAX_FILES];
static int      g_csig_n = -1;                    /* -1 = nothing looked at yet (or forgotten after a context change) */
static char     g_clist[sizeof ((Cfg *)0)->custom_shader_file];
static uint64_t g_custom_next_us;                 /* hot-reload poll: about once a second, not every frame */

/* What the user's files get for free.  GLSL_COMMON ALREADY declares uDepth, uProj, uSize, oColor, linZ(), clampP(), zAt() and posAt():
 * declaring uDepth a second time is a compile error, so only uScene, uInput and uTime are added here.  Each pass then gets its own
 * "#line N" (see custom_make_src) so the compiler's error lines count in lines of the USER'S file, not of this header. */
static const char CUSTOM_HDR[] =
    GLSL_HDR GLSL_COMMON
    "uniform sampler2D uScene;\n"
    "uniform sampler2D uInput;\n"
    "uniform float uTime;\n"
    "uniform sampler2D uLut;\n"                                  /* Documents/<custom_lut>; a neutral LUT while there is no such file */
    /* Shadertoy names.  #defines cost nothing: a name nobody writes is never expanded, so plain shaders are not affected. */
    "#define iChannel0 uScene\n"
    "#define iChannel1 uInput\n"
    "#define iChannel2 uDepth\n"
    "#define iChannel3 uLut\n"
    "#define iResolution vec3(uSize.xy, 1.0)\n"
    "#define iChannelResolution vec3[4](vec3(uSize.xy, 1.0), vec3(uSize.xy, 1.0), vec3(uSize.xy, 1.0), vec3(vec2(textureSize(uLut, 0)), 1.0))\n"
    "#define iTime uTime\n"
    "#define iTimeDelta (1.0 / 60.0)\n"
    "#define iFrame int(uTime * 60.0)\n"
    "#define iMouse vec4(0.0)\n"
    "#define iDate vec4(2026.0, 1.0, 1.0, mod(uTime, 86400.0))\n"
    "#define texture2D texture\n"
    "#define texture2DLod textureLod\n"
    "#define texture2DProj textureProj\n";

/* what makes a Shadertoy shader run: main() calls mainImage().  The alpha goes back to the scene's own (a Shadertoy image has no meaningful alpha). */
static const char CUSTOM_MAIN_BRIDGE[] =
    "\nvoid main(){ mainImage(oColor, gl_FragCoord.xy); oColor.a = texelFetch(uInput, ivec2(gl_FragCoord.xy), 0).a; }\n";

static void custom_drop(void) {                   /* delete our programs (current context only) and go back to the built-in composite */
    for (int i = 0; i < g_cp_n; i++) if (g_cp[i].prog) gl.DeleteProgram(g_cp[i].prog);
    memset(g_cp, 0, sizeof g_cp);
    g_cp_n = 0;
}
static void custom_forget(void) {                 /* the EAGLContext changed: the old names are not ours here, never delete them */
    memset(g_cp, 0, sizeof g_cp);
    lut_forget();
    g_cp_n = 0;
    g_csig_n = -1;
    g_clist[0] = 0;
    g_custom_next_us = 0;
}

/* "a.frag, b.frag" -> names[]; returns how many names there are (more than max is possible: only the first max are stored) */
static int custom_names(const char *list, char names[][CUSTOM_NAME_MAX], int max) {
    int n = 0;
    const char *p = list;
    while (*p) {
        const char *e = strchr(p, ',');
        const char *a = p, *b = e ? e : p + strlen(p);
        while (a < b && (*a == ' ' || *a == '\t')) a++;
        while (b > a && (b[-1] == ' ' || b[-1] == '\t')) b--;
        if (b > a) {
            if (n < max) {
                size_t l = (size_t)(b - a);
                if (l >= CUSTOM_NAME_MAX) l = CUSTOM_NAME_MAX - 1;
                memcpy(names[n], a, l);
                names[n][l] = 0;
            }
            n++;
        }
        if (!e) break;
        p = e + 1;
    }
    return n;
}

/* ------------------------------------------------------------------ Shadertoy compatibility helpers (custom_read_file / custom_make_src) */
static int st_is_id(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_'; }

/* is w present as a whole word in [a, b)? */
static int st_has_word(const char *a, const char *b, const char *w) {
    const size_t n = strlen(w);
    for (const char *p = a; p + n <= b; p++)
        if (!memcmp(p, w, n) && (p == a || !st_is_id(p[-1])) && (p + n == b || !st_is_id(p[n]))) return 1;
    return 0;
}

/* Blanks, IN PLACE and without changing any length (so every #line number stays right), what would break a pasted Shadertoy / WebGL shader:
 *   #version ...  and  #extension ...                     the loader writes its own header, and #extension may not follow a declaration in GLSL ES 3.00
 *   uniform ... iResolution|iTime|iTimeDelta|iFrame|iMouse|iDate|iChannel0..3|iChannelResolution ... ;     the loader #defines those names itself
 * Only whole lines that START with those words are touched; their first two characters become "//". */
static void shadertoy_sanitize(char *buf, size_t len) {
    static const char *const names[] = { "iResolution", "iTime", "iTimeDelta", "iFrame", "iMouse", "iDate", "iChannel0", "iChannel1", "iChannel2",
                                         "iChannel3", "iChannelResolution", NULL };
    char *end = buf + len, *p = buf;
    while (p < end) {
        char *nl = (char *)memchr(p, '\n', (size_t)(end - p));
        char *le = nl ? nl : end;
        char *s = p;
        while (s < le && (*s == ' ' || *s == '\t')) s++;
        int kill = 0;
        if (s < le && *s == '#') {
            char *q = s + 1;
            while (q < le && (*q == ' ' || *q == '\t')) q++;
            kill = (le - q >= 7 && !memcmp(q, "version", 7)) || (le - q >= 9 && !memcmp(q, "extension", 9));
        } else if (le - s >= 8 && !memcmp(s, "uniform", 7) && (s[7] == ' ' || s[7] == '\t')) {
            for (int i = 0; names[i] && !kill; i++) kill = st_has_word(s, le, names[i]);
        }
        if (kill && le - s >= 2) { s[0] = '/'; s[1] = '/'; }
        p = nl ? nl + 1 : end;
    }
}

/* does the code in [p, p+n) define  void <name>( ... )  ?  Comments are skipped.  (mainImage = a Shadertoy shader; main = it has its own entry point) */
static int has_func_def(const char *p, size_t n, const char *name) {
    const size_t nl = strlen(name);
    size_t i = 0;
    while (i < n) {
        if (p[i] == '/' && i + 1 < n && p[i + 1] == '/') { while (i < n && p[i] != '\n') i++; continue; }
        if (p[i] == '/' && i + 1 < n && p[i + 1] == '*') { i += 2; while (i + 1 < n && !(p[i] == '*' && p[i + 1] == '/')) i++; i += 2; continue; }
        if (st_is_id(p[i])) {
            size_t j = i;
            while (j < n && st_is_id(p[j])) j++;
            if (!(p[i] >= '0' && p[i] <= '9') && j - i == nl && !memcmp(p + i, name, nl)) {
                size_t k = j, b = i;
                while (k < n && (p[k] == ' ' || p[k] == '\t' || p[k] == '\r' || p[k] == '\n')) k++;
                while (b > 0 && (p[b - 1] == ' ' || p[b - 1] == '\t' || p[b - 1] == '\r' || p[b - 1] == '\n')) b--;
                if (k < n && p[k] == '(' && b >= 4 && !memcmp(p + b - 4, "void", 4) && (b == 4 || !st_is_id(p[b - 5]))) return 1;
            }
            i = j;
            continue;
        }
        i++;
    }
    return 0;
}

/* reads a whole file into a fresh NUL-terminated buffer (UTF-8 byte-order mark removed, some editors add one); NULL + log on failure */
static char *custom_read_file(const char *path, const char *name, long long size, size_t *len) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { logf_("custom shader: cannot open '%s' (%s) - the built-in look stays\n", name, strerror(errno)); return NULL; }
    char *buf = (char *)malloc((size_t)size + 2);
    if (!buf) { fclose(fp); logf_("custom shader: out of memory\n"); return NULL; }
    size_t n = fread(buf, 1, (size_t)size, fp);
    fclose(fp);
    if (n >= 3 && (unsigned char)buf[0] == 0xEF && (unsigned char)buf[1] == 0xBB && (unsigned char)buf[2] == 0xBF) {
        memmove(buf, buf + 3, n - 3);
        n -= 3;
    }
    buf[n] = 0;
    *len = n;
    shadertoy_sanitize(buf, n);                         /* #version / #extension / uniform iTime... lines of pasted shaders: blanked, same length */
    return buf;
}

/* one section of a file = the text of one pass.  line = the 1-based line of the FILE where the text starts. */
typedef struct { const char *text; size_t len; int line; char name[40]; } Section;

/* is [p, end) a pass marker line:  optional blanks, "//", optional blanks, "@pass", then end of line or a blank and an optional name?  */
static int pass_marker(const char *p, const char *end, char *name, size_t name_max) {
    while (p < end && (*p == ' ' || *p == '\t')) p++;
    if (end - p < 2 || p[0] != '/' || p[1] != '/') return 0;
    p += 2;
    while (p < end && (*p == ' ' || *p == '\t')) p++;
    if (end - p < 5 || memcmp(p, "@pass", 5) != 0) return 0;
    p += 5;
    if (p < end && *p != ' ' && *p != '\t' && *p != '\r') return 0;          /* "@passes" is not a marker */
    while (p < end && (*p == ' ' || *p == '\t')) p++;
    size_t n = (size_t)(end - p);
    while (n > 0 && (p[n - 1] == ' ' || p[n - 1] == '\t' || p[n - 1] == '\r')) n--;
    if (n >= name_max) n = name_max - 1;
    memcpy(name, p, n);
    name[n] = 0;
    return 1;
}

/* Cuts a file at its //@pass lines.  No marker: one section = the whole file, no shared prelude (= v1.2 behaviour).
 * Otherwise *pre / *pre_len = everything before the first marker, shared by all sections, and section i = the lines after marker i up
 * to the next marker.  Returns the number of sections, or -1 if there are more than max. */
static int custom_split(const char *buf, size_t len, const char **pre, size_t *pre_len, Section *sec, int max) {
    const char *end = buf + len, *p = buf;
    int n = 0, line = 1;
    *pre = buf;
    *pre_len = 0;
    while (p < end) {
        const char *nl = (const char *)memchr(p, '\n', (size_t)(end - p));
        const char *le = nl ? nl : end;                    /* end of this line, without its '\n' */
        const char *next = nl ? nl + 1 : end;
        char nm[40];
        if (pass_marker(p, le, nm, sizeof nm)) {
            if (n >= max) return -1;
            if (n == 0) *pre_len = (size_t)(p - buf);
            else sec[n - 1].len = (size_t)(p - sec[n - 1].text);
            sec[n].text = next;
            sec[n].len = 0;
            sec[n].line = line + 1;
            snprintf(sec[n].name, sizeof sec[n].name, "%s", nm);
            n++;
        }
        p = next;
        line++;
    }
    if (n == 0) {
        sec[0].text = buf; sec[0].len = len; sec[0].line = 1; sec[0].name[0] = 0;
        return 1;
    }
    sec[n - 1].len = (size_t)(end - sec[n - 1].text);
    return n;
}

/* header + [#line 1, shared prelude] + #line <first line of the section> + section.  Caller frees. */
static char *custom_make_src(const char *pre, size_t pre_len, const Section *s) {
    const size_t hlen = sizeof CUSTOM_HDR - 1;
    /* Shadertoy style (void mainImage, and no main() of its own anywhere in this pass): append the main() that calls it.  Only THIS section's text counts: it is not NUL-terminated at s->len. */
    const int bridge = has_func_def(s->text, s->len, "mainImage") && !has_func_def(s->text, s->len, "main") && !has_func_def(pre, pre_len, "main");
    char *src = (char *)malloc(hlen + 2 * 24 + pre_len + s->len + 4 + sizeof CUSTOM_MAIN_BRIDGE);
    if (!src) return NULL;
    size_t o = 0;
    memcpy(src, CUSTOM_HDR, hlen);
    o += hlen;
    if (pre_len) {
        o += (size_t)snprintf(src + o, 24, "#line 1\n");
        memcpy(src + o, pre, pre_len);
        o += pre_len;
        if (src[o - 1] != '\n') src[o++] = '\n';
    }
    o += (size_t)snprintf(src + o, 24, "#line %d\n", s->line);
    memcpy(src + o, s->text, s->len);
    o += s->len;
    src[o++] = '\n';
    if (bridge) { memcpy(src + o, CUSTOM_MAIN_BRIDGE, sizeof CUSTOM_MAIN_BRIDGE - 1); o += sizeof CUSTOM_MAIN_BRIDGE - 1; }
    src[o] = 0;
    return src;
}

/* compile + link one pass against the shared vertex shader.  Returns 0 = ok, 1 = compile failed, 2 = link failed (both logged). */
static int custom_build_pass(GLuint vs, const char *src, const char *label, CPass *out) {
    GLuint fs = compile_shader(GLC_FRAGMENT_SHADER, src, label);      /* on failure it logs the compiler's message */
    if (!fs) return 1;
    GLuint pr = gl.CreateProgram();
    if (!pr) { logf_("custom shader: glCreateProgram failed\n"); gl.DeleteShader(fs); return 2; }
    gl.AttachShader(pr, vs);
    gl.AttachShader(pr, fs);
    gl.LinkProgram(pr);
    gl.DeleteShader(fs);                                              /* stays alive inside the program, freed with it */
    GLint lk = 0;
    gl.GetProgramiv(pr, GLC_LINK_STATUS, &lk);
    if (!lk) {
        char log[1024] = {0};
        GLsizei ln = 0;
        gl.GetProgramInfoLog(pr, sizeof log - 1, &ln, log);
        logf_("custom shader: %s FAILED to link:\n%s\n", label, log);
        gl.DeleteProgram(pr);
        return 2;
    }
    memset(out, 0, sizeof *out);
    out->prog   = pr;
    out->uScene = gl.GetUniformLocation(pr, "uScene");
    out->uInput = gl.GetUniformLocation(pr, "uInput");
    out->uDepth = gl.GetUniformLocation(pr, "uDepth");
    out->uProj  = gl.GetUniformLocation(pr, "uProj");
    out->uSize  = gl.GetUniformLocation(pr, "uSize");
    out->uTime  = gl.GetUniformLocation(pr, "uTime");
    out->uLut   = gl.GetUniformLocation(pr, "uLut");
    snprintf(out->label, sizeof out->label, "%s", label);
    return 0;
}

/* builds every pass of one file and appends it to g_cp.  Returns 1 on success; on failure it has logged why (the caller drops the chain). */
static int custom_load_file(GLuint vs, const CFileSig *sg) {
    char path[CUSTOM_PATH_MAX];
    snprintf(path, sizeof path, "%s/%s", g_dir, sg->name);
    if (sg->size <= 0 || sg->size > CUSTOM_MAX_BYTES) {
        logf_("custom shader: '%s' is %lld bytes (allowed 1..%d) - the built-in look stays\n", sg->name, sg->size, CUSTOM_MAX_BYTES);
        return 0;
    }
    size_t len = 0;
    char *buf = custom_read_file(path, sg->name, sg->size, &len);
    if (!buf) return 0;
    if (len == 0) { logf_("custom shader: '%s' is empty - the built-in look stays\n", sg->name); free(buf); return 0; }

    Section sec[CUSTOM_MAX_PASSES];
    const char *pre = NULL;
    size_t pre_len = 0;
    int ns = custom_split(buf, len, &pre, &pre_len, sec, CUSTOM_MAX_PASSES);
    if (ns < 0) {
        logf_("custom shader: '%s' has more than %d //@pass sections - the built-in look stays\n", sg->name, CUSTOM_MAX_PASSES);
        free(buf);
        return 0;
    }
    int ok = 1;
    for (int i = 0; ok && i < ns; i++) {
        if (g_cp_n >= CUSTOM_MAX_PASSES) {
            logf_("custom shader: more than %d passes in total - the built-in look stays\n", CUSTOM_MAX_PASSES);
            ok = 0;
            break;
        }
        char label[CUSTOM_NAME_MAX + 64];
        if (ns == 1 && !sec[0].name[0]) snprintf(label, sizeof label, "%s", sg->name);
        else snprintf(label, sizeof label, "%s pass %d/%d%s%s%s", sg->name, i + 1, ns, sec[i].name[0] ? " '" : "", sec[i].name, sec[i].name[0] ? "'" : "");
        char *src = custom_make_src(pre, pre_len, &sec[i]);
        if (!src) { logf_("custom shader: out of memory\n"); ok = 0; break; }
        int rc = custom_build_pass(vs, src, label, &g_cp[g_cp_n]);
        free(src);
        if (rc) {
            if (rc == 1)
                logf_("custom shader: hint - the loader adds #version, precision, uScene, uInput, uDepth, uProj, uSize, uTime, uLut and oColor (do not declare them); for Shadertoy "
                      "shaders (void mainImage) also iChannel0-3, iResolution, iTime, iFrame, iMouse and main().  Line numbers above are lines of YOUR file\n");
            ok = 0;
            break;
        }
        CPass *cp = &g_cp[g_cp_n++];
        logf_("custom shader: %s compiled and linked OK%s  uniforms: scene=%d input=%d depth=%d proj=%d size=%d time=%d lut=%d (-1 = your code does not use it)\n",
              cp->label, has_func_def(sec[i].text, sec[i].len, "mainImage") ? " (Shadertoy mainImage)" : "", cp->uScene, cp->uInput, cp->uDepth, cp->uProj, cp->uSize, cp->uTime, cp->uLut);
    }
    free(buf);
    return ok;
}

/* ------------------------------------------------------------------ the LUT texture (all calls run inside the GL save/restore bracket, on the render thread) */

/* does any pass of the chain mention uLut / iChannel3?  Only then is the PNG decoded and uploaded at all. */
static int custom_uses_lut(void) {
    for (int i = 0; i < g_cp_n; i++) if (g_cp[i].uLut >= 0) return 1;
    return 0;
}

/* RGBA8 -> a NEW texture (bilinear, clamped).  What the upload could trip over in the ENGINE'S state (a pixel-unpack buffer, row length, skips) is
 * set to plain values for the call and put back right after.  Returns 0 and logs on a GL error; nothing leaks. */
static GLuint lut_upload(const unsigned char *rgba, int w, int h) {
    GLint ub = 0, rl = 0, sr = 0, sp = 0, al = 4;
    gl_err_drain("lut upload (before)", 0);
    gl.GetIntegerv(GLC_PIXEL_UNPACK_BUFFER_BINDING, &ub);
    gl.GetIntegerv(GLC_UNPACK_ROW_LENGTH, &rl);
    gl.GetIntegerv(GLC_UNPACK_SKIP_ROWS, &sr);
    gl.GetIntegerv(GLC_UNPACK_SKIP_PIXELS, &sp);
    gl.GetIntegerv(GLC_UNPACK_ALIGNMENT, &al);
    if (ub) gl.BindBuffer(GLC_PIXEL_UNPACK_BUFFER, 0);
    gl.PixelStorei(GLC_UNPACK_ROW_LENGTH, 0);
    gl.PixelStorei(GLC_UNPACK_SKIP_ROWS, 0);
    gl.PixelStorei(GLC_UNPACK_SKIP_PIXELS, 0);
    gl.PixelStorei(GLC_UNPACK_ALIGNMENT, 4);
    GLuint t = 0;
    gl.GenTextures(1, &t);
    gl.ActiveTexture(GLC_TEXTURE0 + (GLenum)LUT_UNIT);        /* unit LUT_UNIT is saved and restored by the bracket */
    gl.BindTexture(GLC_TEXTURE_2D, t);
    gl.TexParameteri(GLC_TEXTURE_2D, GLC_TEXTURE_MIN_FILTER, GLC_LINEAR);
    gl.TexParameteri(GLC_TEXTURE_2D, GLC_TEXTURE_MAG_FILTER, GLC_LINEAR);
    gl.TexParameteri(GLC_TEXTURE_2D, GLC_TEXTURE_WRAP_S, GLC_CLAMP_TO_EDGE);
    gl.TexParameteri(GLC_TEXTURE_2D, GLC_TEXTURE_WRAP_T, GLC_CLAMP_TO_EDGE);
    gl.TexImage2D(GLC_TEXTURE_2D, 0, (GLint)GLC_RGBA8, w, h, 0, GLC_RGBA, GLC_UNSIGNED_BYTE, rgba);
    gl.PixelStorei(GLC_UNPACK_ROW_LENGTH, rl);
    gl.PixelStorei(GLC_UNPACK_SKIP_ROWS, sr);
    gl.PixelStorei(GLC_UNPACK_SKIP_PIXELS, sp);
    gl.PixelStorei(GLC_UNPACK_ALIGNMENT, al);
    if (ub) gl.BindBuffer(GLC_PIXEL_UNPACK_BUFFER, (GLuint)ub);
    if (gl_err_drain("lut upload", 1)) { gl.DeleteTextures(1, &t); return 0; }
    return t;
}

/* the neutral LUT used while there is no (valid) file: 256x16, sixteen 16x16 slices, red along x, green down, blue per slice */
static unsigned char *lut_identity(int *w, int *h) {
    enum { N = 16 };
    unsigned char *p = (unsigned char *)malloc((size_t)N * N * N * 4);
    if (!p) return NULL;
    for (int b = 0; b < N; b++) for (int g = 0; g < N; g++) for (int r = 0; r < N; r++) {
        unsigned char *o = p + ((size_t)g * (N * N) + (size_t)b * N + (size_t)r) * 4;
        o[0] = (unsigned char)(r * 255 / (N - 1)); o[1] = (unsigned char)(g * 255 / (N - 1)); o[2] = (unsigned char)(b * 255 / (N - 1)); o[3] = 255;
    }
    *w = N * N; *h = N;
    return p;
}

/* reads + decodes a PNG file; NULL + log on any problem */
static unsigned char *lut_load_png(const char *path, const char *name, long long size, int *w, int *h) {
    if (size <= 0 || size > (long long)PNG_MAX_FILE) { logf_("lut: '%s' is %lld bytes (allowed 1..%u) - ignored\n", name, size, PNG_MAX_FILE); return NULL; }
    FILE *fp = fopen(path, "rb");
    if (!fp) { logf_("lut: cannot open '%s' (%s)\n", name, strerror(errno)); return NULL; }
    unsigned char *buf = (unsigned char *)malloc((size_t)size);
    if (!buf) { fclose(fp); logf_("lut: out of memory\n"); return NULL; }
    size_t n = fread(buf, 1, (size_t)size, fp);
    fclose(fp);
    const char *why = NULL;
    unsigned char *px = png_decode(buf, n, w, h, &why);
    free(buf);
    if (!px) logf_("lut: '%s' rejected: %s\n", name, why ? why : "?");
    return px;
}

/* Keeps R.tex_lut in step with Documents/<custom_lut>.  Cheap when nothing changed (one stat()).  Only does anything while a pass of the chain uses
 * uLut.  No file = the neutral LUT, so a LUT shader changes nothing.  A broken new file never replaces a working LUT (it is reported once). */
static void lut_sync(void) {
    if (!custom_uses_lut()) return;
    LutSig now;
    memset(&now, 0, sizeof now);
    snprintf(now.name, sizeof now.name, "%s", g_cfg.custom_lut);
    char path[CUSTOM_PATH_MAX];
    struct stat st;
    snprintf(path, sizeof path, "%s/%s", g_dir, now.name);
    if (stat(path, &st) == 0 && S_ISREG(st.st_mode)) { now.present = 1; now.mtime = st.st_mtime; now.size = (long long)st.st_size; }
    if (R.tex_lut && g_lut.known && !strcmp(g_lut.name, now.name) && g_lut.present == now.present && g_lut.mtime == now.mtime && g_lut.size == now.size) return;
    snprintf(g_lut.name, sizeof g_lut.name, "%s", now.name);          /* remembered even if loading fails: reported once, not every second */
    g_lut.known = 1; g_lut.present = now.present; g_lut.mtime = now.mtime; g_lut.size = now.size;

    unsigned char *px = NULL;
    int w = 0, h = 0, from_file = 0;
    if (now.present) { px = lut_load_png(path, now.name, now.size, &w, &h); from_file = px != NULL; }
    else logf_("lut: '%s' not found in Documents - the neutral LUT (no colour change) is used until it appears\n", now.name);
    if (!px) {
        if (now.present) { if (R.tex_lut) { logf_("lut: the previous LUT stays\n"); return; } }   /* a broken file never replaces a working LUT */
        else if (R.tex_lut && !g_lut.from_file) return;                                          /* already neutral */
        px = lut_identity(&w, &h);
        if (!px) return;
    }
    GLuint t = lut_upload(px, w, h);
    free(px);
    if (!t) { logf_("lut: could not make the texture (%dx%d)\n", w, h); return; }
    if (R.tex_lut) gl.DeleteTextures(1, &R.tex_lut);
    R.tex_lut = t;
    g_lut.from_file = from_file; g_lut.w = w; g_lut.h = h;
    if (from_file) logf_("lut: '%s' loaded, %dx%d (uLut / iChannel3, texture unit %d)\n", now.name, w, h, LUT_UNIT);
    else logf_("lut: neutral %dx%d LUT in use\n", w, h);
}

/* Returns 1 if a custom chain is active afterwards, 0 if not (the built-in composite is used).  Cheap when nothing changed: one stat() per file. */
#include "resa_fx_api.h"

static int load_fx_file(const CFileSig *sg) {
    FxCompileReq req = {0};
    char path[CUSTOM_PATH_MAX];
    snprintf(path, sizeof path, "%s/%s", g_dir, sg->name);
    req.path = path;
    req.w = R.w;
    req.h = R.h;

    const char *incs[] = { g_dir, NULL };
    req.inc = incs;

    FxModule mod = {0};
    char err[512] = {0};

    logf_("ReShade FX: Compiling %s...\n", sg->name);
    if (!fxb_compile(&req, &mod, err, sizeof(err))) {
        logf_("ReShade FX: '%s' FAILED to compile:\n%s\n", sg->name, err);
        return 0;
    }

    logf_("ReShade FX: '%s' compiled OK. %d passes.\n", sg->name, mod.npass);

    for (int i = 0; i < mod.npass; i++) {
        if (g_cp_n >= CUSTOM_MAX_PASSES) break;
        FxPassDesc *p = &mod.pass[i];

        GLuint vs = compile_shader(GLC_VERTEX_SHADER, p->vs, p->name);
        GLuint fs = compile_shader(GLC_FRAGMENT_SHADER, p->ps, p->name);
        if (!vs || !fs) {
            if(vs) gl.DeleteShader(vs);
            if(fs) gl.DeleteShader(fs);
            continue;
        }

        GLuint pr = gl.CreateProgram();
        gl.AttachShader(pr, vs);
        gl.AttachShader(pr, fs);
        gl.LinkProgram(pr);
        gl.DeleteShader(vs);
        gl.DeleteShader(fs);

        CPass *cp = &g_cp[g_cp_n++];
        memset(cp, 0, sizeof *cp);
        cp->prog = pr;
        cp->uScene = gl.GetUniformLocation(pr, "uScene");
        cp->uInput = gl.GetUniformLocation(pr, "uInput");
        cp->uDepth = gl.GetUniformLocation(pr, "uDepth");
        cp->uProj  = gl.GetUniformLocation(pr, "uProj");
        cp->uSize  = gl.GetUniformLocation(pr, "uSize");
        cp->uTime  = gl.GetUniformLocation(pr, "uTime");
        cp->uLut   = gl.GetUniformLocation(pr, "uLut");
        snprintf(cp->label, sizeof cp->label, "%s pass %d", sg->name, i);
    }

    fxm_free(&mod);
    return 1;
}

static int load_custom_shader(GLuint vs) {
    if (!vs) return 0;
    char names[CUSTOM_MAX_FILES][CUSTOM_NAME_MAX];
    const int total = custom_names(g_cfg.custom_shader_file, names, CUSTOM_MAX_FILES);
    const int nf = total < CUSTOM_MAX_FILES ? total : CUSTOM_MAX_FILES;

    CFileSig sig[CUSTOM_MAX_FILES];
    memset(sig, 0, sizeof sig);
    for (int i = 0; i < nf; i++) {
        char path[CUSTOM_PATH_MAX];
        struct stat st;
        snprintf(sig[i].name, sizeof sig[i].name, "%s", names[i]);
        snprintf(path, sizeof path, "%s/%s", g_dir, names[i]);
        if (stat(path, &st) == 0 && S_ISREG(st.st_mode)) {
            sig[i].present = 1;
            sig[i].mtime = st.st_mtime;
            sig[i].size = (long long)st.st_size;
        }
    }
    int same = g_csig_n == nf && !strcmp(g_clist, g_cfg.custom_shader_file);
    for (int i = 0; same && i < nf; i++)
        same = sig[i].present == g_csig[i].present && sig[i].mtime == g_csig[i].mtime && sig[i].size == g_csig[i].size;
    if (same) return g_cp_n > 0;                      /* same files as last time (a broken one is reported once, not every second) */
    memcpy(g_csig, sig, sizeof sig);
    g_csig_n = nf;
    snprintf(g_clist, sizeof g_clist, "%s", g_cfg.custom_shader_file);

    gl_err_drain("custom shader (before)", 0);
    custom_drop();                                    /* a changed file replaces the old chain; if the new one fails there is none */

    if (nf == 0) {                                    /* "custom_shader=," : nothing to load */
        logf_("custom shader: no file name in custom_shader='%s' - the built-in look stays\n", g_cfg.custom_shader_file);
        return 0;
    }
    if (total > CUSTOM_MAX_FILES) {
        logf_("custom shader: %d files listed, at most %d are allowed - the built-in look stays\n", total, CUSTOM_MAX_FILES);
        return 0;
    }
    int missing = 0;
    for (int i = 0; i < nf; i++)
        if (!sig[i].present) {
            logf_("custom shader: '%s' not found in Documents - the built-in look stays until it appears\n", sig[i].name);
            missing++;
        }
    if (missing) return 0;                            /* it loads the moment the file shows up: its signature changes then */

    int ok = 1;
    for (int i = 0; ok && i < nf; i++) {
        if (strstr(sig[i].name, ".fx")) {
            ok = load_fx_file(&sig[i]);
        } else {
            ok = custom_load_file(vs, &sig[i]);
        }
    }
    if (ok && gl_err_drain("custom shader load", 1)) {
        logf_("custom shader: GL error while setting it up - the built-in look stays\n");
        ok = 0;
    }
    if (ok && custom_uses_lut()) {                    /* a pass samples uLut / iChannel3: the texture must exist before the chain may run */
        lut_sync();
        if (!R.tex_lut) { logf_("custom shader: could not make the LUT texture - the built-in look stays\n"); ok = 0; }
    }
    if (!ok) {
        custom_drop();                                /* all or nothing: no half chains */
        logf_("custom shader: '%s' rejected - the built-in look stays until the file(s) are fixed\n", g_cfg.custom_shader_file);
        return 0;
    }
    logf_("custom shader: '%s' OK - %d pass%s\n", g_cfg.custom_shader_file, g_cp_n, g_cp_n == 1 ? "" : "es");
    return 1;
}

/* hot reload: called once per AO frame inside the GL save/restore bracket, but only touches the disk about once a second */
static void custom_poll(void) {
    if (!g_cfg.use_custom_shader || !R.vs) return;
    uint64_t t = now_us();
    if (t < g_custom_next_us) return;
    g_custom_next_us = t + 1000000ull;
    load_custom_shader(R.vs);
    lut_sync();                                       /* hot reload of the LUT picture (does nothing unless a pass uses uLut) */
}

/* ------------------------------------------------------------------ our GL objects (all calls run inside the save/restore bracket) */
static void res_free_sized(Res *r) {                     /* only valid in the context that created them */
    gl.DeleteFramebuffers(1, &r->fbo_copy);
    gl.DeleteFramebuffers(2, r->fbo_ao);
    gl.DeleteFramebuffers(1, &r->fbo_ssr);
    gl.DeleteFramebuffers(2, r->fbo_pp);
    gl.DeleteTextures(1, &r->tex_depth);
    gl.DeleteTextures(1, &r->tex_color);
    gl.DeleteTextures(2, r->tex_ao);
    gl.DeleteTextures(1, &r->tex_ssr);
    gl.DeleteTextures(2, r->tex_pp);
    r->fbo_copy = r->fbo_ao[0] = r->fbo_ao[1] = r->fbo_ssr = r->fbo_pp[0] = r->fbo_pp[1] = 0;
    r->tex_depth = r->tex_color = r->tex_ao[0] = r->tex_ao[1] = r->tex_ssr = r->tex_pp[0] = r->tex_pp[1] = 0;
    r->ok = 0;
}

static void res_free_all(Res *r) {
    res_free_sized(r);
    if (r->ao.prog)   gl.DeleteProgram(r->ao.prog);
    if (r->blur.prog) gl.DeleteProgram(r->blur.prog);
    if (r->comp.prog) gl.DeleteProgram(r->comp.prog);
    if (r->ssr.prog)  gl.DeleteProgram(r->ssr.prog);
    if (r->vao)       gl.DeleteVertexArrays(1, &r->vao);
    if (r->vs)        gl.DeleteShader(r->vs);
    if (r->tex_lut) gl.DeleteTextures(1, &r->tex_lut);
    lut_forget();
    custom_drop();                                       /* the custom chain's programs live outside Res but in the same context */
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
    if (!ok) { gl.DeleteShader(vs); return 0; }
    r->vs = vs;                                          /* NOT deleted here: the custom shader chain links against it (freed in res_free_all) */
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
    tex_filter(r->tex_ao[0], GLC_LINEAR);                 /* the custom chain reads this one as uInput / its first uScene */
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

/* The ping-pong targets of the custom chain (ao_frame step 5): made the first time a chain with 2+ passes runs, and again after a size
 * change (res_free_sized drops them with everything else that depends on the size).  need = how many of the two are used (1 or 2).
 * Returns 0 on any failure, after cleaning up after itself: the caller then gives up the chain, never AO. */
static int res_ensure_pp(Res *r, int need) {
    gl.ActiveTexture(GLC_TEXTURE0);                       /* tex_alloc binds on the active unit; unit 0 is saved and restored by the bracket */
    for (int i = 0; i < need; i++) {
        if (r->tex_pp[i] && r->fbo_pp[i]) continue;
        gl.GenTextures(1, &r->tex_pp[i]);
        gl.GenFramebuffers(1, &r->fbo_pp[i]);
        tex_alloc(r->tex_pp[i], GLC_RGBA8, r->w, r->h);
        tex_filter(r->tex_pp[i], GLC_LINEAR);
        if (!attach_fbo(r->fbo_pp[i], r->tex_pp[i], NULL, 0, "custom ping-pong")) goto fail;
    }
    if (gl_err_drain("custom ping-pong buffers", 1)) goto fail;
    return 1;
fail:
    gl.DeleteFramebuffers(2, r->fbo_pp);
    gl.DeleteTextures(2, r->tex_pp);
    r->fbo_pp[0] = r->fbo_pp[1] = r->tex_pp[0] = r->tex_pp[1] = 0;
    return 0;
}

/* make our objects match (context, size, depth format).  Returns 0 if that is impossible (AO is then switched off). */
static int res_ensure(void *ctx, int w, int h, const DFmt *df) {
    if (R.ctx && ctx && R.ctx != ctx) {
        logf_("res: EAGLContext changed %p -> %p; forgetting our old GL names (NOT deleting them: the numbers may be the engine's there)\n", R.ctx, ctx);
        memset(&R, 0, sizeof R);
        custom_forget();                                 /* same rule for the custom chain's programs: forget, never delete */
    }
    if (ctx) R.ctx = ctx;
    if (!R.ao.prog || !R.blur.prog || !R.comp.prog) {
        if (!res_build_programs(&R)) { res_free_all(&R); ao_die("shader build failed (see the shader log above)"); return 0; }
        if (g_cfg.use_custom_shader) {                   /* programs are ready: build the custom chain against the same vertex shader */
            g_custom_next_us = now_us() + 1000000ull;
            load_custom_shader(R.vs);
        }
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
    int use_custom = c.use_custom_shader && g_cp_n > 0 && view == 1 && (R.tex_lut || !custom_uses_lut());   /* debug views (2..6) are never post-processed */
    int n_custom = 0;                                                                        /* passes of the chain that run this frame */
#define STAGE(name) do { if (gl_err_drain(name, 1)) { err = 1; goto out; } } while (0)

    /* nothing of the engine's state may leak into our passes */
    for (int i = 0; i < 8; i++) gl.Disable(k_caps[i]);
    gl.DepthMask(0);
    gl.ColorMask(1, 1, 1, 1);

    /* a chain of 2+ passes needs ping-pong targets between its passes; if they cannot be made, only the chain is given up, never AO */
    if (use_custom && g_cp_n > 1 && !res_ensure_pp(&R, g_cp_n > 2 ? 2 : 1)) {
        logf_("custom shader: could not make the ping-pong buffers - the chain is switched off until a file changes (AO keeps running)\n");
        custom_drop();
        use_custom = 0;
    }
    if (use_custom) n_custom = g_cp_n;

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

    /* 4. vertical blur + apply to the colour copy.  Normally this is written straight into the ENGINE'S scene FBO (it samples only
     *    our copies).  With a custom chain it goes into tex_ao[0] instead (free again since the blur pass): that is the image that
     *    enters the chain (uInput), and step 5 finishes the job. */
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
    draw_into(use_custom ? R.fbo_ao[0] : sc->fbo, w, h);
    STAGE("composite pass");

    /* 5. the user's shader chain (Documents/custom_shader=...).  Pass k reads the image the previous pass drew (uScene, unit 2), the image
     *    that entered the chain (uInput, unit 1: always tex_ao[0], never written) and the depth copy (uDepth, unit 0).  Every pass but the
     *    last renders into a ping-pong buffer (pass 0 -> pp[0], pass 1 -> pp[1], pass 2 -> pp[0], ...), so a pass never reads the texture
     *    it is writing; the LAST pass writes the engine's scene FBO, so nothing of ours reaches the scene before the very end.  A GL error
     *    in any pass switches only the chain off and finishes THIS frame with the plain AO image, so a bad shader can never take AO down. */
    if (use_custom) {
        const float t_now = (float)since_s();
        if (R.tex_lut) bind_tex(LUT_UNIT, R.tex_lut);     /* uLut: stays bound for every pass of the chain */
        GLuint prev = R.tex_ao[0];                        /* uScene of the next pass */
        int bad = -1;                                     /* index of the pass that raised a GL error */
        for (int i = 0; i < n_custom; i++) {
            const CPass *cp = &g_cp[i];
            gl.UseProgram(cp->prog);
            gl.Uniform1i(cp->uDepth, 0);
            gl.Uniform1i(cp->uInput, 1);
            gl.Uniform1i(cp->uScene, 2);
            gl.Uniform4f(cp->uProj, 1.0f / pj->p00, 1.0f / pj->p11, pj->A, pj->B);
            gl.Uniform4f(cp->uSize, (float)w, (float)h, 1.0f / (float)w, 1.0f / (float)h);
            gl.Uniform1f(cp->uTime, t_now);
            gl.Uniform1i(cp->uLut, LUT_UNIT);
            bind_tex(0, R.tex_depth);
            bind_tex(1, R.tex_ao[0]);
            bind_tex(2, prev);
            draw_into(i == n_custom - 1 ? sc->fbo : R.fbo_pp[i & 1], w, h);
            if (gl_err_drain(cp->label, 1)) { bad = i; break; }
            prev = R.tex_pp[i & 1];
        }
        if (bad >= 0) {
            logf_("custom shader: GL error while drawing %s - the whole chain is switched off until a file changes (AO keeps running)\n",
                  g_cp[bad].label);
            custom_drop();
            gl.BindFramebuffer(GLC_READ_FRAMEBUFFER, R.fbo_ao[0]);
            gl.BindFramebuffer(GLC_DRAW_FRAMEBUFFER, sc->fbo);
            gl.BlitFramebuffer(0, 0, w, h, 0, 0, w, h, GLC_COLOR_BUFFER_BIT, GLC_NEAREST);
        }
        STAGE("custom shader chain");
    }

    {
        unsigned fno = ++g_since_build;
        if (fno == 1)
            logf_("AO: first frame OK  view=%d  scene fbo %u %dx%d  proj p00=%.4f p11=%.4f near=%.3f far=%.1f  samples=%d radius=%.2f  ssr=%s  custom=%d pass%s\n",
                  view, sc->fbo, w, h, pj->p00, pj->p11, pj->near_, pj->far_, c.samples, c.radius, use_ssr ? "on" : "off",
                  n_custom, n_custom == 1 ? "" : "es");
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
    int rc = 2;
    if (res_ensure(tl_ctx, sc.w, sc.h, sc.df)) {
        custom_poll();                                    /* hot reload of the custom shader file (about once a second) */
        rc = ao_frame(&sc, &pj, view);
    }
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
