/*
 * nvgldemo.h
 *
 * Copyright (c) 2007-2019,2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef __NVGLDEMO_H
#define __NVGLDEMO_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NVGLDEMO_MAX_NAME 256

#define NVGLDEMO_DEFAULT_WIDTH 800
#define NVGLDEMO_DEFAULT_HEIGHT 480
#define NVGLDEMO_DEFAULT_SURFACE_ID 9000

#define MAX_EGL_STREAM_ATTR 16

//
// OS-dependent functionality
//

// For standard functions, we use macros to allow overrides.
// For now, just use posix.
#define EXIT    exit
#define MALLOC  malloc
#define REALLOC realloc
#define FREE    free
#define MEMSET  memset
#define MEMCPY  memcpy
#define STRLEN  strlen
#define STRCMP  strcmp
#define STRNCMP strncmp
#define STRTOL  strtol
#define STRTOF  strtof
#define STRNCPY strncpy
#define STRSTR  strstr
#define SNPRINTF snprintf
#define POW     (float)pow
#define SQRT    (float)sqrt
#define ISQRT(val) ((float)1.0/(float)sqrt(val))
#define COS     (float)cos
#define SIN     (float)sin
#define ATAN2   (float)atan2
#define PI      (float)M_PI
#define ASSERT  assert

#define NVGLDEMO_EGL_GET_PROC_ADDR(name, fail, type)          \
    do {                                          \
        p##name = (type)eglGetProcAddress(#name);\
        if (!p##name) {                           \
            NvGlDemoLog("%s load fail.\n",#name); \
            goto fail;                         \
        }                                         \
    } while (0)

#define NVGLDEMO_EGL_GET_DISPLAY(nativeDisplay) \
    eglGetDisplay(nativeDisplay)

#define NVGLDEMO_EGL_INITIALIZE(eglDisplay, major, minor) \
        eglInitialize(eglDisplay, major, minor)

#define NVGLDEMO_EGL_QUERY_STRING(eglDisplay, name) \
        eglQueryString(eglDisplay, name)


long long NvGlDemoSysTime(void);
#define SYSTIME (NvGlDemoSysTime)

// More complex functions have their own OS-specific implementation
void
NvGlDemoLog(
    const char* message, ...);

char*
NvGlDemoLoadFile(
    const char *file,
    unsigned int *size);

// window system interface type
typedef enum NvGlDemoInterfaceEnum
{
    NvGlDemoInterface_Unknown       = 0x00000000,
    NvGlDemoInterface_Android,
    NvGlDemoInterface_KD,
    NvGlDemoInterface_Null,
    NvGlDemoInterface_Win32,
    NvGlDemoInterface_X11,
    NvGlDemoInterface_SimpleDisp,
    NvGlDemoInterface_Wayland,
    NvGlDemoInterface_Device,
    NvGlDemoInterface_DRM,
    NvGlDemoInterface_WF,
    NvGlDemoInterface_QnxScreen,

    NvGlDemoInterface_Force32       = 0x7fffffff
} NvGlDemoInterface;

//
// Window/graphics state
//

// Window System and EGL objects
typedef struct NvGlDemoPlatformState NvGlDemoPlatformState;
typedef struct {
    NativeDisplayType       nativeDisplay;
    NativeWindowType        nativeWindow;
    EGLDisplay              display;
    EGLStreamKHR            stream;
    EGLSurface              surface;
    EGLConfig               config;
    EGLContext              context;
    EGLint                  width;
    EGLint                  height;
    NvGlDemoInterface       platformType;
    NvGlDemoPlatformState*  platform;
} NvGlDemoState;

extern NvGlDemoState demoState;
extern int           g_ServerID;
extern int           g_ClientID;

// Top level initialization/termination functions
int
NvGlDemoInitialize(
    int* argc, char** argv,
    const char *appName,
    int glversion, int depthbits, int stencilbits);

int
NvGlDemoInitializeParsed(
    int* argc, char** argv,
    const char *appName,
    int glversion, int depthbits, int stencilbits);

void
NvGlDemoShutdown(void);

// Window system specific functions

// Open the drm FD and populate the drm device
// Returns: 0 - successful
//         -1 - failure
#ifdef NVGLDEMO_HAS_DEVICE
int
NvGlDemoCreateDrmDevice(void);
#if defined(__QNX__)
// Open the WFDDevice
// Returns: 0 - successful
//         -1 - failure
int
NvGlDemoCreateWfdDevice(void);
#endif //__QNX__
#endif //NVGLDEMO_HAS_DEVICE

// Get platform specific display attributes needed for
// eglGetPlatformDisplay().
// Returns : 0 when successful
//         :-1 when error
int
NvGlDemoGetDisplayAttribs(EGLint**);

int
NvGlDemoDisplayInit(void);

void
NvGlDemoDisplayTerm(void);

int
NvGlDemoWindowInit(
    int* argc, char** argv,
    const char* appName);

void
NvGlDemoWindowTerm(void);

EGLNativePixmapType
NvGlDemoPixmapCreate(
    unsigned int width,
    unsigned int height,
    unsigned int depth);

void
NvGlDemoPixmapDelete(
    EGLNativePixmapType pixmap);

EGLBoolean
NvGlDemoCreateCrossPartitionEGLStream(void);

EGLBoolean
NvGlDemoPrepareStreamToAttachProducer(void);

// Returns 1: On success
//         0: On failure
int
NvGlDemoInitConsumerProcess(void);

// Returns 1: On success
//         0: On failure
int
NvGlDemoInitProducerProcess(void);

void
NvGlDemoEglTerminate(void);

//
// Event handling
//

// Window system event callback types
typedef void (*NvGlDemoCloseCB)(void);
typedef void (*NvGlDemoResizeCB)(int w, int h);
typedef void (*NvGlDemoKeyCB)(char key, int state);
typedef void (*NvGlDemoPointerCB)(int x, int y);
typedef void (*NvGlDemoButtonCB)(int button, int state);

// Functions to set and trigger callbacks
void
NvGlDemoSetCloseCB(NvGlDemoCloseCB cb);

void
NvGlDemoSetResizeCB(NvGlDemoResizeCB cb);

void
NvGlDemoSetKeyCB(NvGlDemoKeyCB cb);

void
NvGlDemoSetPointerCB(NvGlDemoPointerCB cb);

void
NvGlDemoSetButtonCB(NvGlDemoButtonCB cb);

void
NvGlDemoCheckEvents(void);

void
NvGlDemoWaitEvents(void);

#ifdef NVGLDEMO_HAS_DEVICE
void
NvGlDemoSetDisplayAlpha(float alpha);
#endif

EGLBoolean
NvGlDemoSwapInterval(EGLDisplay dpy, EGLint interval);

//
// Command line parsing
//
typedef enum {
    NVGL_DEMO_OPTION_TIMEOUT = 1 << 0
} NvGlDemoOption;

// Parsed options
typedef struct {
    unsigned int flags;                     // For tracking what all options
                                            // are set explictly.
    int windowSize[2];                      // Window size
    int windowOffset[2];                    // Window offset
    int windowOffsetValid;                  // Window offset was requested
    int displaySize[2];                     // Display size
    int useCurrentMode;                     // Keeps the current display mode
    int displayRate;                        // Display refresh rate
    char displayName[NVGLDEMO_MAX_NAME];    // Display selection
    int displayLayer;                       // Display layer
    int displayBlend;                       // Display layer blending
    float displayAlpha;                     // Display constant blending alpha
    float displayColorKey[8];               // Display color key range
                                            // [0-3] RGBA LOW, [4-7] RGBA HIGH
    int msaa;                               // Multi-sampling
    int csaa;                               // Coverage sampling
    int buffering;                          // N-buffered swaps
    char eglstreamsock[NVGLDEMO_MAX_NAME];  // Source for EGLStream fd
    int useProgramBin;                      // Program binary enable option
    int displayNumber;                      // Display output number
    int eglQnxScreenTest;                   // To enable the eglQnxScreenTest
    int enablePostSubBuffer;                // Set before initializing to enable
    int enableMutableRenderBuffer;          // Set before initializing to enable
    int fullScreen;                         // To start the demo in fullscreen mode
    int nFifo;                              // FIFO mode for eglstreams. 0 -> mailbox
    int latency;                            // Egl Stream Consumer latency
    int timeout;                            // Egl Stream Consumer acquire timeout
    char proctype[NVGLDEMO_MAX_NAME];       // Specify producer or consumer
    char ipAddr[NVGLDEMO_MAX_NAME];         // IP address for cross-partition EGLStreams.
    int frames;                             // Number of frames to run.
    int postWindowFlags;                    // QNX CAR2 Screen post window flags
    int surfaceType;                        // Surface Type - Normal/Bottom_Origin
    int port;                               // Port number for cross-partition EGLStreams
    int surface_id;                         // Surface ID for weston ivi-shell
    int renderahead;                        // Max number of in-flight GPU
                                            // frames in mailbox mode
    float duration;                         // Demo duration in seconds
    float inactivityTime;                   // Interval for inactivity testing
    int isSmart;                            // can detect termination of cross-partition stream
    int isDrmNvdcPermissive;                // Set drm-nvdc permissive
    int isProtected;                        // Set protected content
    int allocator;                          // Allocator to use for initial dumb buffer allocation
} NvGlDemoOptions;

// Values for displayBlend option
typedef enum {
    NvGlDemoDisplayBlend_None = 0,   // No blending
    NvGlDemoDisplayBlend_ConstAlpha, // Constant value alpha blending value
    NvGlDemoDisplayBlend_PixelAlpha, // Per pixel alpha blending
    NvGlDemoDisplayBlend_ColorKey    // Color keyed blending
} NvGlDemoDisplayBlend;

// Values for allocator option
typedef enum {
    NvGlDemoAllocator_DRM = 0, // Use DRM IOCTLS for dumb buffer allocation
    NvGlDemoAllocator_GBM,     // Use GBM (minigbm) for dumb buffer allocations
} NvGlDemoAllocator;

extern NvGlDemoOptions demoOptions;

// Parsing utilities
char**
NvGlDemoArgDuplicate(
    int argc, const char *const *argv);

const char*
NvGlDemoArgExtract(
    int* argc, char** argv, int index);

void
NvGlDemoArgAdvance(
    int* argc, char** argv, int index, int count);

int
NvGlDemoArgMatch(
    int* argc, char** argv, int index, const char* test);

int
NvGlDemoArgMatchStr(
    int* argc, char** argv, int index, const char* test,
    const char* usage,
    unsigned int maxlen, char* value);

int
NvGlDemoArgMatchInt(
    int* argc, char** argv, int index, const char* test,
    const char* usage, int minval, int maxval,
    unsigned int count, int* value);

int
NvGlDemoArgMatchFlt(
    int* argc, char** argv, int index, const char* test,
    const char* usage, float minval, float maxval,
    unsigned int count, float* value);

const char*
NvGlDemoArgUsageString(void);

int
NvGlDemoArgParse(
    int* argc, char** argv);

int
NvGlDemoArgFailed(void);


//
// Math/matrix operations
//

#define eps 1e-4

int
eq(float a, float b);

void
NvGlDemoMatrixIdentity(
    float m[16]);

int
NvGlDemoMatrixEquals(
    float a[16], float b[16]);

void
NvGlDemoMatrixTranspose(
    float m[16]);

void
NvGlDemoMatrixMultiply(
    float m0[16], float m1[16]);

void
NvGlDemoMatrixMultiply_4x4_3x3(
    float m0[16], float m1[9]);

void
NvGlDemoMatrixMultiply_3x3(
    float m0[9], float m1[9]);

void
NvGlDemoMatrixFrustum(
    float m[16],
    float l, float r, float b, float t, float n, float f);

void
NvGlDemoMatrixOrtho(
    float m[16],
    float l, float r, float b, float t, float n, float f);

void
NvGlDemoMatrixScale(
    float m[16], float x, float y, float z);

void
NvGlDemoMatrixTranslate(
    float m[16], float x, float y, float z);

void
NvGlDemoMatrixRotate_create3x3(
    float m[9],
    float theta, float x, float y, float z);

void
NvGlDemoMatrixRotate(
    float m[16], float theta, float x, float y, float z);

void
NvGlDemoMatrixRotate_3x3(
    float m[9], float theta, float x, float y, float z);

float
NvGlDemoMatrixDeterminant(
    float m[16]);

void
NvGlDemoMatrixInverse(
    float m[16]);

void
NvGlDemoMatrixCopy(
    float dest[16], float src[16]);

void
NvGlDemoMatrixVectorMultiply(
    float m[16], float v[4]);

void
NvGlDemoMatrixPrint(
    float a[16]);

//
// Texture utilities
//

unsigned int
NvGlDemoLoadTga(
    unsigned int target,
    unsigned int count,
    const char** names);

unsigned int
NvGlDemoLoadTgaFromBuffer(
    unsigned int target,
    unsigned int count,
    unsigned char** buffer);

//
// Shader setup
//

unsigned int
NvGlDemoLoadShaderBinStrings(
    const char* vertBin, unsigned int vertBinSize,
    const char* fragBin, unsigned int fragBinSize,
    unsigned char link,
    unsigned char debugging);

unsigned int
NvGlDemoLoadShaderBinFiles(
    const char* vertFile,
    const char* fragFile,
    unsigned char link,
    unsigned char debugging);

unsigned int
NvGlDemoLoadShaderSrcStrings(
    const char* vertSrc, int vertSrcSize,
    const char* fragSrc, int fragSrcSize,
    unsigned char link,
    unsigned char debugging);

unsigned int
NvGlDemoLoadShaderSrcFiles(
    const char* vertFile,
    const char* fragFile,
    unsigned char link,
    unsigned char debugging);

// Tries to load a previously saved
// binary program from a file. Returns the program object if successful,
// 0 otherwise. If not successful, it may be due to the binary
// being produced with an outdated compiler. In that case, try
// to load the source shaders, recompile, and save the binary
// again.
unsigned int
NvGlDemoLoadBinaryProgram(
    const char* fileName,
    unsigned char debugging);

unsigned int NvGlDemoLinkProgram(
    unsigned int prog,
    unsigned char debugging);
// Save file, returns bytes written.
int
NvGlDemoSaveFile(
    const char *file,
    unsigned char *data,
    unsigned int size);

// Thread Create
void *
NvGlDemoThreadCreate(
    void *(*start_routine) (void *),
    void *arg);

// Thread Destroy
int
NvGlDemoThreadJoin(
    void *threadId,
    void **retval);

// Semaphore Create
void *
NvGlDemoSemaphoreCreate(
    int pshared,
    unsigned int value);

// Semaphore Destroy
int
NvGlDemoSemaphoreDestroy(
    void *semId);

// Semaphore Wait
int
NvGlDemoSemaphoreWait(
    void *semId);

// Semaphore Post
int
NvGlDemoSemaphorePost(
    void *semId);

// Thread Yield
int
NvGlDemoThreadYield(void);

int
NvGlDemoCreateSocket(void);

void
NvGlDemoServerBind(int sockID);

void
NvGlDemoServerListen(int sockID);

int
NvGlDemoServerAccept(int sockID);

void
NvGlDemoServerReceive(int clientSockID);

int
NvGlDemoClientConnect(const char * ip, int sockID);

void
NvGlDemoClientSend(const char* data, int sockID);

// Save the binary program to a file.
//
unsigned int
NvGlDemoSaveBinaryProgram(
        unsigned int progName,
        const char* fileName);

unsigned int NvGlDemoLoadPreCombinedShader(
    const char* vertSrc, int vertSrcSize,
    const char* fragSrc, int fragSrcSize,
    unsigned char link,
    unsigned char debugging,
    const char* prgFile );

unsigned int NvGlDemoLoadExternShader(
    const char* vertFile,
    const char* fragFile,
    unsigned char link,
    unsigned char debugging,
    const char* prgFile );

// For maximum flexibility, we support several options for loading shaders.
// When USE_BINARY_SHADERS is defined, the demos expect precompiled shader
//   binaries. When it is not defined, shader source is compiled at runtime.
//   The former can speed load time and eliminate the need to install the
//   shader compiler on the target platform, while the latter allows for
//   greater flexibility, including dynamically generated shaders.
// When USE_EXTERN_SHADERS is defined, the demos search for separate files
//   containing the binaries or source, and load them at runtime. This
//   allows shaders to be modified without having to recompile the application.
//   When it is not defined, the shaders binaries or source are built int
//   the application at compile time, eliminating the need for extra files.
//   In general, the former is often desirable when developing and debugging
//   an application, while the latter is desirable for release.
// The following macros reduce the amount of code needed in individual
//   demos to support all four combinations.
#define STRINGIFY(x) #x
#ifdef USE_EXTERN_SHADERS
#  ifdef USE_BINARY_SHADERS
// TODO: When binary shaders are used, this code path is redundant.
// This needs a cleanup. Bug - http://nvbugs/200399721
#    define VERTFILE(f) ""
#    define FRAGFILE(f) ""
#  else  // USE_BINARY_SHADERS
#    define VERTFILE(f) STRINGIFY(f.glslv)
#    define FRAGFILE(f) STRINGIFY(f.glslf)
#  endif // USE_BINARY_SHADERS
#  define LOADPROGSHADER(v,f,l,d,pb) \
          NvGlDemoLoadExternShader(v, f, l, d, pb)
#  define LOADSHADER(v,f,l,d) \
          NvGlDemoLoadExternShader(v, f, l, d, NULL)
#else  // USE_EXTERN_SHADERS
#  ifdef USE_BINARY_SHADERS
// We don't support embedding shader binary within the final executable.
#  error "This combination is not supported"
#  else  // USE_BINARY_SHADERS
#    define VERTFILE(f) STRINGIFY(f.glslvh)
#    define FRAGFILE(f) STRINGIFY(f.glslfh)
#  endif // USE_BINARY_SHADERS
#  define LOADPROGSHADER(v,f,l,d,pb) \
          NvGlDemoLoadPreCombinedShader(v, sizeof(v), f, sizeof(f), l, d, pb)
#  define LOADSHADER(v,f,l,d) \
          NvGlDemoLoadPreCombinedShader(v, sizeof(v), f, sizeof(f), l, d, NULL)
#endif // USE_EXTERN_SHADERS

#define PROGFILE(f) STRINGIFY(f.bin)

//
// Fetch a file descriptor from a named socket
//

int
NvGlDemoFdFromSocket(char const *name);

//
// Close the file descriptor we got from NvGlDemoFdFromSocket()
//

int
NvGlDemoCloseFd(int fd);

//
// Circular Queue implementation
//

void
NvGlDemoCqInitIndex(int);

int
NvGlDemoCqFull(void);

int
NvGlDemoCqEmpty(void);

int
NvGlDemoCqInsertIndex(void);

int
NvGlDemoCqDeleteIndex(void);

//
// Per-frame utility functions
//

// A non-zero return indicates success.
int
NvGlDemoPreSwapInit(void);

// This should be called before a swap.
// A non-zero return indicates success.
int
NvGlDemoPreSwapExec(void);

void
NvGlDemoPreSwapShutdown(void);


// This function must be called after every swap buffers call
// Returns 1: when successful
//         0: when failed
int
NvGlDemoPostSwap(void);

//
// Inactivity interval functions
//

void
NvGlDemoInactivityInit(void);

void
NvGlDemoInactivitySleep(void);

//
// Renderahead implementation
//

int
NvGlDemoThrottleRendering(void);

int
NvGlDemoThrottleInit(void);

void
NvGlDemoThrottleShutdown(void);

#ifdef __cplusplus
}
#endif

#endif // __NVGLDEMO_H
