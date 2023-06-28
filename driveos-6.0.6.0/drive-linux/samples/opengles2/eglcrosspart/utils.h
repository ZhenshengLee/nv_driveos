/*
 * utils.h
 *
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

// Description: Common includes, definitions, and extensions

#ifndef _UTILS_H
#define _UTILS_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <unistd.h>
#include <sys/un.h>
#include <signal.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <time.h>
#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <EGL/eglext.h>
#include <GLES2/gl2ext.h>
#include "nvgldemo.h"

typedef unsigned long long timev_t;

#define CHECK_GL_ERROR() {\
    GLenum error = glGetError();\
    if(error != GL_NO_ERROR) {\
        printf("OpenGL error %s(%d): %d\n", __FILE__, __LINE__, error);\
        exit(-1);\
    } }

#define CHECK_EGL_ERROR() {\
    GLenum error = eglGetError();\
    if(error != EGL_SUCCESS) {\
        printf("EGL error %s(%d): %d\n", __FILE__, __LINE__, error);\
        exit(-1);\
    } }

// Strict C compiler settings throw unused variables as error.
// This makes sure the errors don't occur.
// Mainly happens variables are used only in asserts. This gives errors in release builds.
#define UNUSED_VAR(x) (void)(x)

/*
 * Error codes
 */
enum ECODE {
    E_OK = 0,              // Success
    E_NOMEM = 1,           // Ran out of memory
    E_FAIL = 2,            // Generic error
    E_NOCONFIG = 4,        // For GrUtil, no surface configuration matched given hints
    E_TIMEOUT = 8,         // An operation timed out
    E_NOTIMPLEMENTED = 16  // Not implemented
};


/*
 * List of extensions to load with setupExtensions()
 */
#if defined(EGL_EXT_stream_consumer_qnxscreen_window) && !defined(ANDROID)
#define NVEGL_QNX_CAR2_SCREEN_EXTENSION(T) \
        T( PFNEGLSTREAMCONSUMERQNXSCREENWINDOWEXTPROC, \
                            eglStreamConsumerQNXScreenWindowEXT ) \
        T( PFNEGLSTREAMCONSUMERACQUIREATTRIBEXTPROC, eglStreamConsumerAcquireAttribEXT )
#else
#define NVEGL_QNX_CAR2_SCREEN_EXTENSION(T)
#endif //EGL_EXT_stream_consumer_qnxscreen_window

#ifndef ANDROID
#define EXTENSION_LIST(T) \
    NVEGL_QNX_CAR2_SCREEN_EXTENSION(T) \
    T( PFNEGLCREATESTREAMKHRPROC,          eglCreateStreamKHR ) \
    T( PFNEGLDESTROYSTREAMKHRPROC,         eglDestroyStreamKHR ) \
    T( PFNEGLQUERYSTREAMKHRPROC,           eglQueryStreamKHR ) \
    T( PFNEGLQUERYSTREAMU64KHRPROC,        eglQueryStreamu64KHR ) \
    T( PFNEGLQUERYSTREAMTIMEKHRPROC,       eglQueryStreamTimeKHR ) \
    T( PFNEGLSTREAMATTRIBKHRPROC,          eglStreamAttribKHR ) \
    T( PFNEGLSTREAMCONSUMERACQUIREKHRPROC, eglStreamConsumerAcquireKHR ) \
    T( PFNEGLSTREAMCONSUMERRELEASEKHRPROC, eglStreamConsumerReleaseKHR ) \
    T( PFNEGLSTREAMCONSUMERGLTEXTUREEXTERNALKHRPROC, \
                        eglStreamConsumerGLTextureExternalKHR ) \
    T( PFNEGLGETSTREAMFILEDESCRIPTORKHRPROC, eglGetStreamFileDescriptorKHR) \
    T( PFNEGLCREATESTREAMFROMFILEDESCRIPTORKHRPROC, eglCreateStreamFromFileDescriptorKHR) \
    T( PFNEGLCREATESTREAMPRODUCERSURFACEKHRPROC, eglCreateStreamProducerSurfaceKHR)
#else
#define EXTENSION_LIST(T) \
    NVEGL_QNX_CAR2_SCREEN_EXTENSION(T)
#endif

#define EXTLST_DECL(tx, x)  tx x = NULL;
#define EXTLST_EXTERN(tx, x) extern tx x;
#define EXTLST_ENTRY(tx, x) { (extlst_fnptr_t *)&x, #x },

/*
 * Load all extensions defined in EXTENSION_LIST. If an extension cannot be loaded, it is
 * assumed that the tests will fail and so the test harness aborts.
 */
void setupExtensions(void);

/*
 * Get the current time in microseconds
 */
void getTime(timev_t *time);

unsigned int
utilLoadShaderSrcStrings(const char* vertSrc, int vertSrcSize,
                         const char* fragSrc, int fragSrcSize,
                         unsigned char link,
                         unsigned char debugging);

#endif // _UTILS_H
