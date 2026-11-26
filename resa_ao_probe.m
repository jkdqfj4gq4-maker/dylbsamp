//
// resa_ao_probe.m
//
// Purely diagnostic. Hooks glBlitFramebuffer (the call alyn_resa itself makes once per
// frame, inside its MSAA-resolve function, right before handing the frame to the Metal
// presenter). Because alyn_resa is calling real GL there, a context is guaranteed current
// on whatever thread this fires on -- that's the whole point of hooking at this specific
// call instead of anything event-based.
//
// What it does, once, on the first call: logs the thread, confirms EAGLContext
// currentContext, and reads the currently-bound framebuffer's DEPTH attachment type
// (GL_TEXTURE vs GL_RENDERBUFFER) plus its size/format. Then it calls straight through to
// the real glBlitFramebuffer with the original arguments, unchanged. No shader, no drawing,
// no state left modified. It writes its answer to a file and stops hooking.
//
// Needs: fishhook (github.com/facebook/fishhook -- vendor fishhook.c/.h next to this file).
// Build as a dylib (OpenGLES + Foundation), inject it however you already inject into
// ReSALauncher. Run one frame of gameplay, then pull:
//   <app Documents>/resa_ao_probe.txt
// and send it back.
//
#import <Foundation/Foundation.h>
#import <OpenGLES/EAGL.h>
#import <OpenGLES/ES2/gl.h>
#import <OpenGLES/ES2/glext.h>
#include <pthread.h>
#include <stdatomic.h>
#include "fishhook.h"

static void (*orig_glBlitFramebuffer)(GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLbitfield, GLenum) = NULL;
static atomic_int already_logged = 0;

static NSString *AttachmentTypeName(GLenum type) {
    if (type == GL_NONE) return @"NONE";
    if (type == GL_TEXTURE) return @"TEXTURE";
    if (type == GL_RENDERBUFFER) return @"RENDERBUFFER";
    return [NSString stringWithFormat:@"0x%x", type];
}

static void WriteLog(NSString *text) {
    NSLog(@"[resa_ao_probe] %@", text);
    NSString *dir = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject;
    NSString *path = [dir stringByAppendingPathComponent:@"resa_ao_probe.txt"];
    NSString *stamped = [NSString stringWithFormat:@"%@\n%@\n", [NSDate date], text];
    NSData *data = [stamped dataUsingEncoding:NSUTF8StringEncoding];
    NSFileHandle *fh = [NSFileHandle fileHandleForWritingAtPath:path];
    if (!fh) {
        [[NSFileManager defaultManager] createFileAtPath:path contents:nil attributes:nil];
        fh = [NSFileHandle fileHandleForWritingAtPath:path];
    }
    [fh seekToEndOfFile];
    [fh writeData:data];
    [fh closeFile];
}

static void hooked_glBlitFramebuffer(GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1,
                                      GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1,
                                      GLbitfield mask, GLenum filter) {
    int expected = 0;
    if (atomic_compare_exchange_strong(&already_logged, &expected, 1)) {
        NSMutableString *out = [NSMutableString string];

        BOOL isMain = [NSThread isMainThread];
        pthread_t self_t = pthread_self();
        [out appendFormat:@"thread: isMain=%d pthread=%p\n", isMain, (void *)self_t];

        EAGLContext *ctx = [EAGLContext currentContext];
        [out appendFormat:@"EAGLContext current: %@ (api=%ld)\n", ctx ? @"YES" : @"NO (!)",
            ctx ? (long)ctx.API : -1];

        GLint fbo = 0;
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &fbo);
        [out appendFormat:@"bound FBO: %d\n", fbo];

        GLint vp[4] = {0};
        glGetIntegerv(GL_VIEWPORT, vp);
        [out appendFormat:@"viewport: %d,%d %dx%d\n", vp[0], vp[1], vp[2], vp[3]];

        for (int i = 0; i < 2; i++) {
            GLenum attachment = (i == 0) ? GL_DEPTH_ATTACHMENT : GL_STENCIL_ATTACHMENT;
            GLint type = 0, name = 0;
            glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, attachment,
                GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &type);
            glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, attachment,
                GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &name);
            [out appendFormat:@"%s attachment: %@ object=%d\n",
                (i == 0 ? "DEPTH" : "STENCIL"), AttachmentTypeName((GLenum)type), name];

            if (type == GL_RENDERBUFFER && name != 0) {
                GLint prevRB = 0;
                glGetIntegerv(GL_RENDERBUFFER_BINDING, &prevRB);
                glBindRenderbuffer(GL_RENDERBUFFER, name);
                GLint w = 0, h = 0, fmt = 0, depthBits = 0, samples = 0;
                glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_WIDTH, &w);
                glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_HEIGHT, &h);
                glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_INTERNAL_FORMAT, &fmt);
                glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_DEPTH_SIZE, &depthBits);
                glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_SAMPLES_APPLE, &samples);
                glBindRenderbuffer(GL_RENDERBUFFER, prevRB);
                [out appendFormat:@"   renderbuffer: %dx%d internalformat=0x%x depthBits=%d samples=%d\n",
                    w, h, fmt, depthBits, samples];
            }
            if (type == GL_TEXTURE && name != 0) {
                [out appendFormat:@"   texture name=%d -- directly sampleable, no copy needed\n", name];
            }
        }

        GLenum err = glGetError();
        [out appendFormat:@"glGetError after probe: 0x%x\n", err];
        [out appendFormat:@"\nreading guide: TEXTURE depth attachment -> sample it directly in an AO shader.\n"
                            "RENDERBUFFER -> needs one extra step (attach a same-size depth TEXTURE and either\n"
                            "render to that instead, or blit/copy into it) before an AO pass can read depth.\n"];

        WriteLog(out);
    }

    orig_glBlitFramebuffer(srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter);
}

__attribute__((constructor))
static void resa_ao_probe_init(void) {
    struct rebinding r = {
        .name = "glBlitFramebuffer",
        .replacement = (void *)hooked_glBlitFramebuffer,
        .replaced = (void **)&orig_glBlitFramebuffer,
    };
    int res = rebind_symbols(&r, 1);
    NSLog(@"[resa_ao_probe] loaded, rebind_symbols -> %d (0 = ok)", res);
}