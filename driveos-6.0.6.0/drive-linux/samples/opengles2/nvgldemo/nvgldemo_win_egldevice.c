/* Copyright (c) 2014 - 2022 NVIDIA Corporation.  All rights reserved.
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

#ifdef NVGLDEMO_HAS_DEVICE

#include <ctype.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <signal.h>
#include <sys/mman.h>
#include "nvgldemo.h"
#include "nvgldemo_win_egldevice.h"

#define ARRAY_LEN(_arr) (sizeof(_arr) / sizeof(_arr[0]))

/* XXX khronos eglext.h does not yet have EGL_OPENWF_DEVICE_EXT */
#ifndef EGL_OPENWF_DEVICE_EXT
#define EGL_OPENWF_DEVICE_EXT 0x333D
#endif

//======================================================================
// Backend initialization
//======================================================================

// Library access
static bool          isOutputInitDone = false;
// This check is true if the platform support EGLOutput DRM atomic extension
static bool          isEGLOutputDrmAtomicEXTSupported = false;

// EGL Device specific variable
static EGLint        devCount = 0;
static EGLDeviceEXT* devList = NULL;
static struct NvGlOutputDevice *nvGlOutDevLst = NULL;

static PFNEGLQUERYDEVICESEXTPROC         peglQueryDevicesEXT = NULL;
static PFNEGLQUERYDEVICESTRINGEXTPROC    peglQueryDeviceStringEXT = NULL;
static PFNEGLQUERYDEVICEATTRIBEXTPROC    peglQueryDeviceAttribEXT = NULL;
static PFNEGLGETPLATFORMDISPLAYEXTPROC   peglGetPlatformDisplayEXT = NULL;
static PFNEGLGETOUTPUTLAYERSEXTPROC      peglGetOutputLayersEXT = NULL;
static PFNEGLQUERYOUTPUTLAYERSTRINGEXTPROC
peglQueryOutputLayerStringEXT = NULL;
static PFNEGLQUERYOUTPUTLAYERATTRIBEXTPROC
peglQueryOutputLayerAttribEXT = NULL;
static PFNEGLCREATESTREAMKHRPROC         peglCreateStreamKHR = NULL;
static PFNEGLDESTROYSTREAMKHRPROC        peglDestroyStreamKHR = NULL;
static PFNEGLSTREAMCONSUMEROUTPUTEXTPROC peglStreamConsumerOutputEXT = NULL;
static PFNEGLCREATESTREAMPRODUCERSURFACEKHRPROC
peglCreateStreamProducerSurfaceKHR = NULL;
static PFNEGLOUTPUTLAYERATTRIBEXTPROC      peglOutputLayerAttribEXT = NULL;
static PFNEGLSTREAMATTRIBKHRPROC           peglStreamAttribKHR = NULL;
static PFNEGLSTREAMCONSUMERACQUIREATTRIBEXTPROC
peglStreamConsumerAcquireAttribEXT = NULL;

#if !defined(__QNX__)

// GBM specific variable
static void*         libminiGBM = NULL;

// DRM Device specific variable
static void*         libDRM   = NULL;
static struct NvGlDemoDRMDevice *nvGlDrmDev = NULL;

typedef int (*PFNDRMOPEN)(const char*, const char*);
typedef int (*PFNDRMCLOSE)(int);
typedef drmModeResPtr (*PFNDRMMODEGETRESOURCES)(int);
typedef void (*PFNDRMMODEFREERESOURCES)(drmModeResPtr);
typedef drmModePlaneResPtr (*PFNDRMMODEGETPLANERESOURCES)(int);
typedef void (*PFNDRMMODEFREEPLANERESOURCES)(drmModePlaneResPtr);
typedef drmModeConnectorPtr (*PFNDRMMODEGETCONNECTOR)(int, uint32_t);
typedef void (*PFNDRMMODEFREECONNECTOR)(drmModeConnectorPtr);
typedef drmModeEncoderPtr (*PFNDRMMODEGETENCODER)(int, uint32_t);
typedef void (*PFNDRMMODEFREEENCODER)(drmModeEncoderPtr);
typedef drmModePlanePtr (*PFNDRMMODEGETPLANE)(int, uint32_t);
typedef void (*PFNDRMMODEFREEPLANE)(drmModePlanePtr);
typedef int (*PFNDRMMODESETCRTC)(int, uint32_t, uint32_t, uint32_t, uint32_t,
        uint32_t*, int, drmModeModeInfoPtr);
typedef drmModeCrtcPtr (*PFNDRMMODEGETCRTC)(int, uint32_t);
typedef int (*PFNDRMMODESETPLANE)(int, uint32_t, uint32_t, uint32_t, uint32_t,
        uint32_t, uint32_t, uint32_t, uint32_t,
        uint32_t, uint32_t, uint32_t, uint32_t);
typedef void (*PFNDRMMODEFREECRTC)(drmModeCrtcPtr);
typedef drmModeAtomicReqPtr (*PFNDRMMODEATOMICALLOC)(void);
typedef int (*PFNDRMMODEATOMICADDPROPERTY)(drmModeAtomicReqPtr, uint32_t, uint32_t, uint64_t);
typedef int (*PFNDRMMODEATOMICCOMMIT)(int, drmModeAtomicReqPtr, uint32_t, void*);
typedef void (*PFNDRMMODEATOMICFREE)(drmModeAtomicReqPtr);
typedef drmModeObjectPropertiesPtr (*PFNDRMMODEOBJECTGETPROPERTIES)(int, uint32_t, uint32_t);
typedef drmModePropertyPtr (*PFNDRMMODEGETPROPERTY)(int, uint32_t);
typedef void (*PFNDRMMODEFREEPROPERTY)(drmModePropertyPtr);
typedef void (*PFNDRMMODEFREEOBJECTPROPERTIES)(drmModeObjectPropertiesPtr);
typedef int (*PFNDRMSETCLIENTCAP)(int, uint64_t, uint64_t);
typedef int (*PFNDRMIOCTL)(int, unsigned long, void*);
typedef int (*PFNDRMMODEADDFB2)(int, uint32_t, uint32_t, uint32_t, uint32_t*, uint32_t*,
        uint32_t*, uint32_t*, uint32_t);
typedef int (*PFNDRMMODECREATEPROPERTYBLOB) (int , const void *, size_t, uint32_t *);
typedef int (*PFNDRMMODEDESTROYPROPERTYBLOB) (int , uint32_t );
typedef drmVersionPtr (*PFNDRMGETVERSION)(int);
typedef void (*PFNDRMFREEVERSION)(drmVersionPtr);

static PFNDRMOPEN                   pdrmOpen = NULL;
static PFNDRMCLOSE                  pdrmClose = NULL;
static PFNDRMMODEGETRESOURCES       pdrmModeGetResources = NULL;
static PFNDRMMODEFREERESOURCES      pdrmModeFreeResources = NULL;
static PFNDRMMODEGETPLANERESOURCES  pdrmModeGetPlaneResources = NULL;
static PFNDRMMODEFREEPLANERESOURCES pdrmModeFreePlaneResources = NULL;
static PFNDRMMODEGETCONNECTOR       pdrmModeGetConnector = NULL;
static PFNDRMMODEFREECONNECTOR      pdrmModeFreeConnector = NULL;
static PFNDRMMODEGETENCODER         pdrmModeGetEncoder = NULL;
static PFNDRMMODEFREEENCODER        pdrmModeFreeEncoder = NULL;
static PFNDRMMODEGETPLANE           pdrmModeGetPlane = NULL;
static PFNDRMMODEFREEPLANE          pdrmModeFreePlane = NULL;
static PFNDRMMODESETCRTC            pdrmModeSetCrtc = NULL;
static PFNDRMMODEGETCRTC            pdrmModeGetCrtc = NULL;
static PFNDRMMODESETPLANE           pdrmModeSetPlane = NULL;
static PFNDRMMODEFREECRTC           pdrmModeFreeCrtc = NULL;
static PFNDRMMODEATOMICALLOC        pdrmModeAtomicAlloc = NULL;
static PFNDRMMODEATOMICADDPROPERTY  pdrmModeAtomicAddProperty = NULL;
static PFNDRMMODEATOMICCOMMIT       pdrmModeAtomicCommit = NULL;
static PFNDRMMODEATOMICFREE         pdrmModeAtomicFree = NULL;
static PFNDRMMODEOBJECTGETPROPERTIES pdrmModeObjectGetProperties = NULL;
static PFNDRMMODEGETPROPERTY        pdrmModeGetProperty = NULL;
static PFNDRMMODEFREEPROPERTY       pdrmModeFreeProperty = NULL;
static PFNDRMMODEFREEOBJECTPROPERTIES pdrmModeFreeObjectProperties = NULL;
static PFNDRMSETCLIENTCAP           pdrmSetClientCap = NULL;
static PFNDRMIOCTL                  pdrmIoctl = NULL;
static PFNDRMMODEADDFB2             pdrmModeAddFB2 = NULL;
static PFNDRMMODECREATEPROPERTYBLOB pdrmModeCreatePropertyBlob = NULL;
static PFNDRMMODEDESTROYPROPERTYBLOB pdrmModeDestroyPropertyBlob = NULL;
static PFNDRMGETVERSION             pdrmGetVersion = NULL;
static PFNDRMFREEVERSION            pdrmFreeVersion = NULL;

// GBM function pointers

typedef struct gbm_device* (*PFNGBMCREATEDEVICE)(int);
typedef struct gbm_bo* (*PFNGBMBOCREATE)(struct gbm_device *, uint32_t, uint32_t,
                                         uint32_t, uint32_t);
typedef uint32_t (*PFNGBMBOGETSTRIDEFORPLANE)(struct gbm_bo *, int);
typedef union gbm_bo_handle (*PFNGBMBOGETHANDLEFORPLANE)(struct gbm_bo *, int);
typedef void (*PFNGBMBODESTROY)(struct gbm_bo *);
typedef void (*PFNGBMDEVICEDESTROY)(struct gbm_device *);

static PFNGBMCREATEDEVICE pgbm_create_device = NULL;
static PFNGBMBOCREATE pgbm_bo_create = NULL;
static PFNGBMBOGETSTRIDEFORPLANE pgbm_bo_get_stride_for_plane = NULL;
static PFNGBMBOGETHANDLEFORPLANE pgbm_bo_get_handle_for_plane = NULL;
static PFNGBMBODESTROY pgbm_bo_destroy = NULL;
static PFNGBMDEVICEDESTROY pgbm_device_destroy = NULL;

// Macro to load DRM function pointers
#if !defined(__INTEGRITY)
#define NVGLDEMO_LOAD_DRM_PTR(name, type)               \
    do {                                      \
        p##name = (type)dlsym(libDRM, #name); \
        if (!p##name) {                       \
            NvGlDemoLog("%s load fail.\n",#name); \
            goto NvGlDemoInitDrmDevice_fail;                     \
        }                                     \
    } while (0)
#else
#define NVGLDEMO_LOAD_DRM_PTR(name, type)               \
    p##name = name
#endif

// Macro to load GBM function pointers
#if !defined(__INTEGRITY)
#define NVGLDEMO_LOAD_GBM_PTR(name, type)         \
    do {                                          \
        p##name = (type)dlsym(libminiGBM, #name);     \
        if (!p##name) {                           \
            NvGlDemoLog("%s load fail.\n",#name); \
            goto NvGlDemoInitDrmDevice_fail;      \
        }                                         \
    } while (0)
#else
#define NVGLDEMO_LOAD_GBM_PTR(name, type)         \
    p##name = name
#endif

#else
// WFD Device specific variable
static void* libWFD  = NULL;
static struct NvGlDemoWFDDevice *nvGlWfdDev = NULL;

typedef WFDint (*PFNWFDENUMERATEDEVICES)(WFDint *const deviceIds, const WFDint deviceIdsCount,
                                 const WFDint *const filterList);
typedef WFDDevice (*PFNWFDCREATEDEVICE)(WFDint deviceId, const WFDint *attribList);
typedef WFDErrorCode (*PFNWFDDESTROYDEVICE)(WFDDevice device);
typedef void (*PFNWFDDEVICECOMMIT)(WFDDevice device, WFDCommitType type, WFDHandle handle);
typedef WFDint (*PFNWFDENUMERATEPIPELINES)(WFDDevice device, WFDint *pipelineIds,
                WFDint pipelineIdsCount, const WFDint *filterList);
typedef WFDint (*PFNWFDENUMERATEPORTS)(WFDDevice device, WFDint *portIds,
                WFDint portIdsCount, const WFDint *filterList);
typedef WFDErrorCode (*PFNWFDGETERROR)(WFDDevice device);
typedef WFDint (*PFNWFDGETPORTMODES)(WFDDevice device, WFDPort port,
                WFDPortMode* modes, WFDint modesCount);
typedef WFDPort (*PFNWFDCREATEPORT)(WFDDevice device, WFDint portId,
                const WFDint* attribList);
typedef WFDPipeline (*PFNWFDCREATEPIPELINE)(WFDDevice device, WFDint pipelineId,
                const WFDint* attribList);
typedef WFDint (*PFNWFDGETPORTMODEATTRIBI)(WFDDevice device, WFDPort port,
                WFDPortMode mode, WFDPortModeAttrib attrib);
typedef void (*PFNWFDSETPIPELINEATTRIBF)(WFDDevice device, WFDPipeline pipeline,
            WFDPipelineConfigAttrib attrib, WFDfloat value);
typedef void (*PFNWFDBINDPIPELINETOPORT)(WFDDevice device, WFDPort port, WFDPipeline pipeline);
typedef void (*PFNWFDSETPORTMODE)(WFDDevice device, WFDPort port, WFDPortMode mode);
typedef void (*PFNWFDDESTROYPORT)(WFDDevice device, WFDPort port);
typedef void (*PFNWFDDESTROYPIPELINE)(WFDDevice device, WFDPipeline pipeline);
typedef WFDPortMode (*PFNWFDGETCURRENTPORTMODE)(WFDDevice device,WFDPort port);
typedef void (*PFNWFDSETPIPELINEATTRIBIV)(WFDDevice device, WFDPipeline pipeline,
            WFDPipelineConfigAttrib attrib, WFDint count, const WFDint* value);
typedef void (*PFNWFDBINDSOURCETOPIPELINE)(WFDDevice device, WFDPipeline pipeline,
            WFDSource source, WFDTransition transition, const WFDRect *region);
typedef WFDint (*PFNWFDGETPORTATTRIBI)(WFDDevice device, WFDPort port,
            WFDPortConfigAttrib attrib );
typedef void (*PFNWFDGETPORTATTRIBIV)(WFDDevice device, WFDPort port,
            WFDPortConfigAttrib attrib, WFDint count, WFDint *value);

static PFNWFDENUMERATEDEVICES pwfdEnumerateDevices = NULL;
static PFNWFDCREATEDEVICE pwfdCreateDevice = NULL;
static PFNWFDDESTROYDEVICE pwfdDestroyDevice = NULL;
static PFNWFDDEVICECOMMIT pwfdDeviceCommit = NULL;
static PFNWFDENUMERATEPIPELINES pwfdEnumeratePipelines = NULL;
static PFNWFDENUMERATEPORTS pwfdEnumeratePorts = NULL;
static PFNWFDGETPORTMODES pwfdGetPortModes = NULL;
static PFNWFDGETERROR pwfdGetError = NULL;
static PFNWFDCREATEPORT pwfdCreatePort = NULL;
static PFNWFDCREATEPIPELINE pwfdCreatePipeline = NULL;
static PFNWFDGETPORTMODEATTRIBI pwfdGetPortModeAttribi = NULL;
static PFNWFDSETPIPELINEATTRIBF pwfdSetPipelineAttribf = NULL;
static PFNWFDBINDPIPELINETOPORT pwfdBindPipelineToPort = NULL;
static PFNWFDSETPORTMODE pwfdSetPortMode = NULL;
static PFNWFDDESTROYPORT pwfdDestroyPort = NULL;
static PFNWFDDESTROYPIPELINE pwfdDestroyPipeline = NULL;
static PFNWFDGETCURRENTPORTMODE pwfdGetCurrentPortMode = NULL;
static PFNWFDSETPIPELINEATTRIBIV pwfdSetPipelineAttribiv = NULL;
static PFNWFDBINDSOURCETOPIPELINE pwfdBindSourceToPipeline = NULL;
static PFNWFDGETPORTATTRIBI pwfdGetPortAttribi = NULL;
static PFNWFDGETPORTATTRIBIV pwfdGetPortAttribiv = NULL;

// Macro to load function pointers
#if !defined(__INTEGRITY)
#define NVGLDEMO_LOAD_WFD_PTR(name, type)               \
    do {                                      \
        p##name = (type)dlsym(libWFD, #name); \
        if (!p##name) {                       \
            NvGlDemoLog("%s load fail.\n",#name); \
            goto NvGlDemoInitWfdDevice_fail;                     \
        }                                     \
    } while (0)
#else
#define NVGLDEMO_LOAD_WFD_PTR(name, type)               \
    p##name = name
#endif
#endif //!__QNX__

static struct sigaction sigint;
static void signal_int(int signum);

// Extension checking utility
static bool CheckExtension(const char *exts, const char *ext)
{
    int extLen = (int)strlen(ext);
    const char *end = exts + strlen(exts);

    while (exts < end) {
        while (*exts == ' ') {
            exts++;
        }
        int n = strcspn(exts, " ");
        if ((extLen == n) && (strncmp(ext, exts, n) == 0)) {
            return true;
        }
        exts += n;
    }
    return EGL_FALSE;
}

//======================================================================
// EGL Desktop functions
//======================================================================

static bool NvGlDemoInitEglDevice(void)
{
    const char* exts = NULL;
    EGLint n = 0;

    // Get extension string
    exts = eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS);
    if (!exts) {
        NvGlDemoLog("eglQueryString fail.\n");
        goto NvGlDemoInitEglDevice_fail;
    }

    // Check extensions and load functions needed for using outputs
    if (!CheckExtension(exts, "EGL_EXT_device_base") ||
            !CheckExtension(exts, "EGL_EXT_platform_base") ||
            !CheckExtension(exts, "EGL_EXT_platform_device")) {
        NvGlDemoLog("egldevice platform ext is not there.\n");
        goto NvGlDemoInitEglDevice_fail;
    }

    NVGLDEMO_EGL_GET_PROC_ADDR(eglQueryDevicesEXT, NvGlDemoInitEglDevice_fail, PFNEGLQUERYDEVICESEXTPROC);
    NVGLDEMO_EGL_GET_PROC_ADDR(eglQueryDeviceStringEXT, NvGlDemoInitEglDevice_fail, PFNEGLQUERYDEVICESTRINGEXTPROC);
    NVGLDEMO_EGL_GET_PROC_ADDR(eglQueryDeviceAttribEXT, NvGlDemoInitEglDevice_fail, PFNEGLQUERYDEVICEATTRIBEXTPROC);
    NVGLDEMO_EGL_GET_PROC_ADDR(eglGetPlatformDisplayEXT, NvGlDemoInitEglDevice_fail, PFNEGLGETPLATFORMDISPLAYEXTPROC);
    NVGLDEMO_EGL_GET_PROC_ADDR(eglGetOutputLayersEXT, NvGlDemoInitEglDevice_fail, PFNEGLGETOUTPUTLAYERSEXTPROC);
    NVGLDEMO_EGL_GET_PROC_ADDR(eglQueryOutputLayerStringEXT, NvGlDemoInitEglDevice_fail, PFNEGLQUERYOUTPUTLAYERSTRINGEXTPROC);
    NVGLDEMO_EGL_GET_PROC_ADDR(eglQueryOutputLayerAttribEXT, NvGlDemoInitEglDevice_fail, PFNEGLQUERYOUTPUTLAYERATTRIBEXTPROC);
    NVGLDEMO_EGL_GET_PROC_ADDR(eglCreateStreamKHR, NvGlDemoInitEglDevice_fail, PFNEGLCREATESTREAMKHRPROC);
    NVGLDEMO_EGL_GET_PROC_ADDR(eglDestroyStreamKHR, NvGlDemoInitEglDevice_fail, PFNEGLDESTROYSTREAMKHRPROC);
    NVGLDEMO_EGL_GET_PROC_ADDR(eglStreamConsumerOutputEXT, NvGlDemoInitEglDevice_fail, PFNEGLSTREAMCONSUMEROUTPUTEXTPROC);
    NVGLDEMO_EGL_GET_PROC_ADDR(eglCreateStreamProducerSurfaceKHR, NvGlDemoInitEglDevice_fail,  PFNEGLCREATESTREAMPRODUCERSURFACEKHRPROC);
    NVGLDEMO_EGL_GET_PROC_ADDR(eglOutputLayerAttribEXT, NvGlDemoInitEglDevice_fail,  PFNEGLOUTPUTLAYERATTRIBEXTPROC);
    NVGLDEMO_EGL_GET_PROC_ADDR(eglStreamAttribKHR, NvGlDemoInitEglDevice_fail,  PFNEGLSTREAMATTRIBKHRPROC);
    NVGLDEMO_EGL_GET_PROC_ADDR(eglStreamConsumerAcquireAttribEXT, NvGlDemoInitEglDevice_fail,  PFNEGLSTREAMCONSUMERACQUIREATTRIBEXTPROC);

    // Load device list
    if (!peglQueryDevicesEXT(0, NULL, &n) || !n) {
        NvGlDemoLog("peglQueryDevicesEXT fail.\n");
        goto NvGlDemoInitEglDevice_fail;
    }

    nvGlOutDevLst = (struct NvGlOutputDevice *)MALLOC(n*sizeof(struct NvGlOutputDevice));
    if(!nvGlOutDevLst){
        goto NvGlDemoInitEglDevice_fail;
    }

    devList = (EGLDeviceEXT*)MALLOC(n * sizeof(EGLDeviceEXT));
    if (!devList || !peglQueryDevicesEXT(n, devList, &devCount) || !devCount) {
        NvGlDemoLog("peglQueryDevicesEXT fail.\n");
        goto NvGlDemoInitEglDevice_fail;
    }

    // Intial Setup
    NvGlDemoResetEglDevice();

    if (demoState.platform->curDevIndx < devCount) {
        demoState.nativeDisplay = (NativeDisplayType)devList[demoState.platform->curDevIndx];
        // Success
        return true;
    }

NvGlDemoInitEglDevice_fail:

    NvGlDemoResetModule();

    return false;
}

// Create EGLDevice desktop
static bool NvGlDemoCreateEglDevice(EGLint devIndx)
{
    struct NvGlOutputDevice *devOut = NULL;
    EGLint n = 0;

    if ((!nvGlOutDevLst) || (devIndx >= devCount)) {
        goto NvGlDemoCreateEglDevice_fail;
    }

    // Set device
    devOut = &nvGlOutDevLst[devIndx];
    devOut->index = devIndx;
    devOut->device = devList[devIndx];
    devOut->eglDpy = demoState.display;

    if ((devOut->eglDpy==EGL_NO_DISPLAY)) {
        NvGlDemoLog("peglGetPlatformDisplayEXT-fail.\n");
        goto NvGlDemoCreateEglDevice_fail;
    }

    // Check for output extension
    const char* exts = eglQueryString(devOut->eglDpy, EGL_EXTENSIONS);
    if (!exts ||
            !CheckExtension(exts, "EGL_EXT_output_base") ||
            !CheckExtension(exts, "EGL_KHR_stream") ||
            !CheckExtension(exts, "EGL_KHR_stream_producer_eglsurface") ||
            !CheckExtension(exts, "EGL_EXT_stream_consumer_egloutput")) {
        NvGlDemoLog("eglstream ext is not there..\n");
        goto NvGlDemoCreateEglDevice_fail;
    }

    if (CheckExtension(exts, "EGL_NV_output_drm_atomic")) {
        // When EGL_NV_output_drm_atomic is enabled and atomic commits
        // are performed, GLSI schedules a modeset request along with the surfaces
        // with pre-flip syncpoints attached. Sending pre-flip syncpoints with
        // modeset reqs isn't supported by NvKms yet, so disable the atomic paths
        // on Embedded-Linux untill this restriction in NvKms is removed.
        // syncpoints
#ifdef NVGLDEMO_IS_EMBEDDED_LINUX
        isEGLOutputDrmAtomicEXTSupported = false;
#else
        isEGLOutputDrmAtomicEXTSupported = true;
#endif
    }

    // Obtain the total number of available layers and allocate an array of
    // window pointers for them
    if (!peglGetOutputLayersEXT(devOut->eglDpy, NULL, NULL, 0, &n) || !n) {
        NvGlDemoLog("peglGetOutputLayersEXT_fail[%u]\n",n);
        goto NvGlDemoCreateEglDevice_fail;
    }
    devOut->layerList  = (EGLOutputLayerEXT*)MALLOC(n * sizeof(EGLOutputLayerEXT));
    devOut->windowList = (struct NvGlDemoWindowDevice*)MALLOC(n * sizeof(struct NvGlDemoWindowDevice));
    if (devOut->layerList && devOut->windowList) {
        NvGlDemoResetEglDeviceLyrLst(devOut);
        memset(devOut->windowList, 0, (n*sizeof(struct NvGlDemoWindowDevice)));
        memset(devOut->layerList, 0, (n*sizeof(EGLOutputLayerEXT)));
    } else {
        NvGlDemoLog("Failed to allocate list of layers and windows");
        goto NvGlDemoCreateEglDevice_fail;
    }

    devOut->enflag = true;
    return true;

NvGlDemoCreateEglDevice_fail:

    NvGlDemoTermEglDevice();
    return false;
}

// Create the EGL Device surface
static bool NvGlDemoCreateSurfaceBuffer(void)
{
    EGLint layerIndex;
    struct NvGlOutputDevice *outDev = NULL;
    struct NvGlDemoWindowDevice *winDev = NULL;
    EGLint swapInterval = 1;
    EGLint attr[MAX_EGL_STREAM_ATTR * 2 + 1];
    int attrIdx        = 0;

    if (demoOptions.nFifo > 0)
    {
        attr[attrIdx++] = EGL_STREAM_FIFO_LENGTH_KHR;
        attr[attrIdx++] = demoOptions.nFifo;
    }


    if (isEGLOutputDrmAtomicEXTSupported) {

        // Switch auto acquire off initially as we do the configuration on the
        // first frame post the swap buffers call. We enable auto-acquire after
        // the first frame is manually acquired using the atomic request.
        attr[attrIdx++] = EGL_CONSUMER_AUTO_ACQUIRE_EXT;
        attr[attrIdx++] = EGL_FALSE;
    }

    attr[attrIdx++] = EGL_NONE;

    if ((!nvGlOutDevLst) || (!demoState.platform) || \
            (demoState.platform->curDevIndx >= devCount)  || \
            (nvGlOutDevLst[demoState.platform->curDevIndx].enflag == false)){
        return false;
    }
    outDev = &nvGlOutDevLst[demoState.platform->curDevIndx];

    // Fail if no layers available
    if ((!outDev) || (outDev->layerUsed >= outDev->layerCount) || (!outDev->windowList) || \
            (!outDev->layerList)){
        return false;
    }

    // Try the default
    if ((outDev->layerDefault < outDev->layerCount) && (outDev->windowList[outDev->layerDefault].enflag == false)) {
        outDev->layerIndex = outDev->layerDefault;
    }

    // If that's not available either, find the first unused layer
    else {
        for (layerIndex=0; layerIndex < outDev->layerCount; ++layerIndex) {
            if (outDev->windowList[layerIndex].enflag == false) {
                break;
            }
        }
        assert(layerIndex < outDev->layerCount);
        outDev->layerIndex = layerIndex;
    }

    outDev->layerUsed++;
    winDev = &outDev->windowList[outDev->layerIndex];

    //Create a stream
    if (demoState.stream == EGL_NO_STREAM_KHR) {
        winDev->stream = peglCreateStreamKHR(outDev->eglDpy, attr);
        demoState.stream = winDev->stream;
    }

    if (demoState.stream == EGL_NO_STREAM_KHR) {
        return false;
    }

    // Connect the output layer to the stream
    if (!peglStreamConsumerOutputEXT(outDev->eglDpy, demoState.stream,
                outDev->layerList[outDev->layerIndex])) {
        return false;
    }

    if (!NvGlDemoSwapInterval(outDev->eglDpy, swapInterval)) {
        return false;
    }

    winDev->index = outDev->layerIndex;
    winDev->enflag = true;

    return true;
}

//Reset EGL Device Layer List
static void NvGlDemoResetEglDeviceLyrLst(struct NvGlOutputDevice *devOut)
{
    int indx;
    for (indx=0;((devOut && devOut->windowList)&&(indx<devOut->layerCount));indx++) {
        devOut->windowList[indx].enflag = false;
        devOut->windowList[indx].index = 0;
        devOut->windowList[indx].stream = EGL_NO_STREAM_KHR;
        devOut->windowList[indx].surface = EGL_NO_SURFACE;
    }
    return;
}

// Destroy all EGL Output Devices
static void NvGlDemoResetEglDevice(void)
{
    int indx;
    for (indx=0;indx<devCount;indx++) {
        nvGlOutDevLst[indx].enflag = false;
        nvGlOutDevLst[indx].index = 0;
        nvGlOutDevLst[indx].device = 0;
        nvGlOutDevLst[indx].eglDpy = 0;
        nvGlOutDevLst[indx].layerCount = 0;
        nvGlOutDevLst[indx].layerDefault = 0;
        nvGlOutDevLst[indx].layerIndex = 0;
        nvGlOutDevLst[indx].layerUsed = 0;
        NvGlDemoResetEglDeviceLyrLst(&nvGlOutDevLst[indx]);
        nvGlOutDevLst[indx].layerList = NULL;
        nvGlOutDevLst[indx].windowList = NULL;
    }
}

// Free EGL Device stream buffer
static void NvGlDemoTermWinSurface(void)
{
    struct NvGlOutputDevice *outDev = NULL;
    struct NvGlDemoWindowDevice *winDev = NULL;

    if ((!nvGlOutDevLst) || (!demoState.platform) || \
            (demoState.platform->curDevIndx >= devCount)  || \
            (nvGlOutDevLst[demoState.platform->curDevIndx].enflag == false)) {
        return;
    }
    outDev = &nvGlOutDevLst[demoState.platform->curDevIndx];
    // Fail if no layers available
    if ((!outDev) || (outDev->layerUsed > outDev->layerCount) || (!outDev->windowList) || \
            (!outDev->layerList) || (outDev->layerIndex >= outDev->layerCount)) {
        return;
    }
    winDev = &outDev->windowList[outDev->layerIndex];
    if ((!winDev) || (winDev->enflag == false)) {
        NvGlDemoLog("NvGlDemoTermWinSurface-fail\n");
        return;
    }

    if (winDev->stream != EGL_NO_STREAM_KHR && demoState.stream != EGL_NO_STREAM_KHR) {
        (void)peglDestroyStreamKHR(outDev->eglDpy, winDev->stream);
        demoState.stream = EGL_NO_STREAM_KHR;
    }

    outDev->windowList[winDev->index].enflag = false;
    outDev->windowList[winDev->index].index = 0;
    outDev->windowList[winDev->index].stream = EGL_NO_STREAM_KHR;
    outDev->windowList[winDev->index].surface = EGL_NO_SURFACE;
    outDev->layerUsed--;
    outDev->eglDpy = EGL_NO_DISPLAY;

    demoState.platform->curDevIndx = 0;
    return;
}

// Terminate Egl Device
static void NvGlDemoTermEglDevice(void)
{
    if (nvGlOutDevLst)
    {
        int indx;
        for (indx=0;indx<devCount;indx++) {
            if (nvGlOutDevLst[indx].layerList){
                FREE(nvGlOutDevLst[indx].layerList);
            }
            if (nvGlOutDevLst[indx].windowList) {
                FREE(nvGlOutDevLst[indx].windowList);
            }
            demoState.nativeDisplay = EGL_NO_DISPLAY;
            nvGlOutDevLst[indx].eglDpy = EGL_NO_DISPLAY;
        }
        if (devList) {
            FREE(devList);
        }
        FREE(nvGlOutDevLst);
        devList = NULL;
        nvGlOutDevLst = NULL;
        devCount = 0;
        NvGlDemoResetEglDeviceFnPtr();
    }
    return;
}

// Reset all EGL Device Function ptr
static void NvGlDemoResetEglDeviceFnPtr(void)
{
    peglQueryDevicesEXT = NULL;
    peglQueryDeviceStringEXT = NULL;
    peglQueryDeviceAttribEXT = NULL;
    peglGetPlatformDisplayEXT = NULL;
    peglGetOutputLayersEXT = NULL;
    peglQueryOutputLayerStringEXT = NULL;
    peglQueryOutputLayerAttribEXT = NULL;
    peglCreateStreamKHR = NULL;
    peglDestroyStreamKHR = NULL;
    peglStreamConsumerOutputEXT = NULL;
    peglCreateStreamProducerSurfaceKHR = NULL;
    peglOutputLayerAttribEXT = NULL;
}

#if defined(__QNX__)

//======================================================================
// WFD Desktop functions
//======================================================================

static void NvGlDemoResetWfdDevice(void)
{
    if (nvGlWfdDev) {
        nvGlWfdDev->wfdDeviceHandle = 0;
        nvGlWfdDev->numPorts = 0;
        nvGlWfdDev->numPipes = 0;
        nvGlWfdDev->portInfo = NULL;
        nvGlWfdDev->pipeInfo = NULL;
        nvGlWfdDev->curPortIndex = 0;
        nvGlWfdDev->curPipeIndex = 0;
    }
}

static void NvGlDemoTermWfdDevice(void)
{
    if (nvGlWfdDev) {
        struct NvGlDemoWFDPipe *pipe = nvGlWfdDev->pipeInfo;
        if (nvGlWfdDev->wfdDeviceHandle && pipe->pipeHandle) {
            pwfdBindSourceToPipeline(nvGlWfdDev->wfdDeviceHandle, pipe->pipeHandle,
                    (WFDSource)0, WFD_TRANSITION_AT_VSYNC, NULL);
            pwfdDeviceCommit(nvGlWfdDev->wfdDeviceHandle, WFD_COMMIT_PIPELINE,
                    pipe->pipeHandle);
        }

        pwfdDestroyPipeline(nvGlWfdDev->wfdDeviceHandle, nvGlWfdDev->pipeInfo->pipeHandle);
        pwfdDestroyPort(nvGlWfdDev->wfdDeviceHandle, nvGlWfdDev->portInfo->portHandle);
        pwfdDestroyDevice(nvGlWfdDev->wfdDeviceHandle);

        if (nvGlWfdDev->portInfo) { 
            free(nvGlWfdDev->portInfo);
        }
        if (nvGlWfdDev->pipeInfo) {
            free(nvGlWfdDev->pipeInfo);
        }

#if !defined(__INTEGRITY)
        if (libWFD) {
            dlclose(libWFD);
            libWFD = NULL;
        }
#endif
        free(nvGlWfdDev);
        NvGlDemoResetWfdDeviceFnPtr();
        nvGlWfdDev = NULL;
    }
}

static bool NvGlDemoInitializeWfdOutputMode(void)
{
    struct NvGlOutputDevice *outDev = NULL;
    int offsetX = 0;
    int offsetY = 0;
    int modeIndex = -1;
    unsigned int modeSize = 0;
    unsigned int modeX = 0;
    unsigned int modeY = 0;
    int foundMatchingDisplayRate = 1;
    struct NvGlDemoWFDPort *port;
    struct NvGlDemoWFDPipe *pipe;
    WFDint srcRect[4];
    WFDint dstRect[4];
    int i;

    // If not specified, use default window size
    if (!demoOptions.windowSize[0])
        demoOptions.windowSize[0] = NVGLDEMO_DEFAULT_WIDTH;
    if (!demoOptions.windowSize[1])
        demoOptions.windowSize[1] = NVGLDEMO_DEFAULT_HEIGHT;

    // Parse global plane alpha
    if (demoOptions.displayAlpha < 0.0 || demoOptions.displayAlpha > 1.0) {
        //If unspecified or out of range, default to 1.0
        NvGlDemoLog("Alpha value specified for constant blending is not in range [0, 1]. Using alpha 1.0.\n");
        demoOptions.displayAlpha = 1.0;
    }

    offsetX = demoOptions.windowOffset[0];
    offsetY = demoOptions.windowOffset[1];

    port = nvGlWfdDev->portInfo;

    if (demoOptions.displaySize[0]) {
        for (i = 0; i < port->numPortModes; i++) {
            WFDint width;
            WFDint height;
            WFDint refreshRate;

            width = pwfdGetPortModeAttribi(nvGlWfdDev->wfdDeviceHandle, port->portHandle,
                        port->portModes[i], WFD_PORT_MODE_WIDTH);
            height = pwfdGetPortModeAttribi(nvGlWfdDev->wfdDeviceHandle, port->portHandle,
                        port->portModes[i], WFD_PORT_MODE_HEIGHT);
            refreshRate = pwfdGetPortModeAttribi(nvGlWfdDev->wfdDeviceHandle, port->portHandle,
                        port->portModes[i], WFD_PORT_MODE_REFRESH_RATE);
            if ((demoOptions.displaySize[0] == width) &&
                (demoOptions.displaySize[1] == height)) {
                modeIndex = i;
                modeX = (unsigned int)width;
                modeY = (unsigned int)height;
                if (demoOptions.displayRate) {
                    if (refreshRate == (int)demoOptions.displayRate) {
                        foundMatchingDisplayRate = 1;
                        break;
                    }
                    else {
                        foundMatchingDisplayRate = 0;
                        break;
                    }
                }
            }
        }

        if (!modeX || !modeY) {
            NvGlDemoLog("Unsupported Displaysize.\n");
            goto NvGlDemoInitializeWfdOutputMode_fail;
        }
        if (!foundMatchingDisplayRate) {
            NvGlDemoLog("Specified Refresh rate is not Supported with Specified Display size.\n");
            goto NvGlDemoInitializeWfdOutputMode_fail;
        }
    } else if (demoOptions.useCurrentMode) {
        if (demoOptions.displayRate) {
            NvGlDemoLog("Refresh Rate should not be specified with Current Mode Parameter.\n");
            goto NvGlDemoInitializeWfdOutputMode_fail;
        }

        // Check to see if there is already a mode set
        if (WFD_INVALID_HANDLE != pwfdGetCurrentPortMode(nvGlWfdDev->wfdDeviceHandle,
                port->portHandle)) {
            modeIndex = -1;
        }
    } else {
        if (demoOptions.displayRate) {
            NvGlDemoLog("Refresh Rate should not be specified alone.\n");
            goto NvGlDemoInitializeWfdOutputMode_fail;
        }

        //Choose the largest supported mode
        for (i = 0; i < port->numPortModes; i++) {
            WFDint width;
            WFDint height;

            width = pwfdGetPortModeAttribi(nvGlWfdDev->wfdDeviceHandle, port->portHandle,
                        port->portModes[i], WFD_PORT_MODE_WIDTH);
            height = pwfdGetPortModeAttribi(nvGlWfdDev->wfdDeviceHandle, port->portHandle,
                        port->portModes[i], WFD_PORT_MODE_HEIGHT);

            unsigned int size = width * height;
            if (size > modeSize) {
                modeIndex = i;
                modeSize = size;
                modeX = width;
                modeY = height;
            }
        }
    }

    if (modeIndex >= 0) {
        pwfdSetPortMode(nvGlWfdDev->wfdDeviceHandle, port->portHandle, port->portModes[modeIndex]);
        if (WFD_ERROR_NONE != pwfdGetError(nvGlWfdDev->wfdDeviceHandle)) {
            NvGlDemoLog("pwfdSetPortMode fail\n");
            goto NvGlDemoInitializeWfdOutputMode_fail;
        }
    }

    pipe = nvGlWfdDev->pipeInfo;

    //Set the alpha property of the pipeline
    pwfdSetPipelineAttribf(nvGlWfdDev->wfdDeviceHandle, pipe->pipeHandle,
            WFD_PIPELINE_GLOBAL_ALPHA, demoOptions.displayAlpha);
    if (WFD_ERROR_NONE != pwfdGetError(nvGlWfdDev->wfdDeviceHandle)) {
        NvGlDemoLog("Set global alpha fail\n");
        goto NvGlDemoInitializeWfdOutputMode_fail;
    }

    srcRect[0] = 0;
    srcRect[1] = 0;
    srcRect[2] = demoOptions.windowSize[0];
    srcRect[3] = demoOptions.windowSize[1];

    dstRect[0] = offsetX;
    dstRect[1] = offsetY;
    dstRect[2] = demoOptions.windowSize[0];
    dstRect[3] = demoOptions.windowSize[1];

    pwfdSetPipelineAttribiv(nvGlWfdDev->wfdDeviceHandle, pipe->pipeHandle,
                    WFD_PIPELINE_SOURCE_RECTANGLE, 4, srcRect);
    if (WFD_ERROR_NONE != pwfdGetError(nvGlWfdDev->wfdDeviceHandle)) {
        NvGlDemoLog("Setting source rect fail\n");
        goto NvGlDemoInitializeWfdOutputMode_fail;
    }

    pwfdSetPipelineAttribiv(nvGlWfdDev->wfdDeviceHandle, pipe->pipeHandle,
                    WFD_PIPELINE_DESTINATION_RECTANGLE, 4, dstRect);
    if (WFD_ERROR_NONE != pwfdGetError(nvGlWfdDev->wfdDeviceHandle)) {
        NvGlDemoLog("Setting dest rect fail\n");
        goto NvGlDemoInitializeWfdOutputMode_fail;
    }

    //Bind the chosen port and pipeline
    pwfdBindPipelineToPort(nvGlWfdDev->wfdDeviceHandle, port->portHandle, pipe->pipeHandle);
    if (WFD_ERROR_NONE != pwfdGetError(nvGlWfdDev->wfdDeviceHandle)) {
        NvGlDemoLog("pwfdBindPipelineToPort fail\n");
        goto NvGlDemoInitializeWfdOutputMode_fail;
    }

    pwfdDeviceCommit(nvGlWfdDev->wfdDeviceHandle, WFD_COMMIT_ENTIRE_DEVICE, WFD_INVALID_HANDLE);
    if (WFD_ERROR_NONE != pwfdGetError(nvGlWfdDev->wfdDeviceHandle)) {
        NvGlDemoLog("pwfdDeviceCommit fail\n");
        goto NvGlDemoInitializeWfdOutputMode_fail;
    }

    outDev = &nvGlOutDevLst[demoState.platform->curDevIndx];
    outDev->layerDefault = nvGlWfdDev->pipeInfo->layer;

    return true;

NvGlDemoInitializeWfdOutputMode_fail:
    return false;
}

int NvGlDemoCreateWfdDevice()
{
    nvGlWfdDev->wfdDeviceHandle = pwfdCreateDevice(WFD_DEFAULT_DEVICE_ID, NULL);
    if (!nvGlWfdDev->wfdDeviceHandle) {
        NvGlDemoLog("pwfdCreateDevice fail\n");
        return -1;
    }

    return 0;
}

// Gets DRM/EGLDevice desktop resources and populates the DRM device
static bool NvGlDemoGetWfdDevice(EGLint devIndx)
{
    struct NvGlOutputDevice *devOut = NULL;
    WFDint numPorts, numPipes, numBindablePipes;
    WFDint *portIDs;
    WFDint *pipeIDs;
    WFDint *bindablePipeIDs;
    bool isPipeBindable = false;
    bool ret = true;
    EGLAttrib layerAttrib[3] = {
        0,
        0,
        EGL_NONE
    };
    EGLOutputLayerEXT tempLayer;
    int i, n, numPortModes;

    devOut = &nvGlOutDevLst[devIndx];

    nvGlWfdDev->numPorts = pwfdEnumeratePorts(nvGlWfdDev->wfdDeviceHandle, NULL, 0, NULL);
    if (!nvGlWfdDev->numPorts) {
        NvGlDemoLog("pwfdEnumeratePorts fail\n");
        return false;
    }

    // Use user specified screen or default. If user specified screen
    // does not exist then return error with usage hint.
    nvGlWfdDev->curPortIndex = demoState.platform->curConnIndx;
    if (nvGlWfdDev->curPortIndex >= nvGlWfdDev->numPorts) {
        NvGlDemoLog("Display output %d is not available, try using another display using option <-dispno>.\n", nvGlWfdDev->curPortIndex);
        return false;
    }

    nvGlWfdDev->numPipes = pwfdEnumeratePipelines(nvGlWfdDev->wfdDeviceHandle, NULL, 0, NULL);
    if (!nvGlWfdDev->numPipes) {
        NvGlDemoLog("pwfdEnumeratePipelines fail\n");
        return false;
    }

    // Allocate space for a single port and a single pipeline that corresponds to user supplied
    // display and layer or default. However allocate enough portIDs and pipeIDs to hold all
    // since these IDs are not consecutive from first ID.
    nvGlWfdDev->portInfo = (struct NvGlDemoWFDPort*)malloc(sizeof(struct NvGlDemoWFDPort));
    nvGlWfdDev->pipeInfo = (struct NvGlDemoWFDPipe*)malloc(sizeof(struct NvGlDemoWFDPipe));
    portIDs = (WFDint*)malloc(nvGlWfdDev->numPorts * sizeof(WFDint));
    pipeIDs = (WFDint*)malloc(nvGlWfdDev->numPipes * sizeof(WFDint));
    if (!nvGlWfdDev->portInfo || !nvGlWfdDev->pipeInfo ||
            !portIDs || !pipeIDs) {
        NvGlDemoLog("Wfd Res Alloc fail\n");
        return false;
    }

    memset(nvGlWfdDev->portInfo, 0, sizeof(struct NvGlDemoWFDPort));
    memset(nvGlWfdDev->pipeInfo, 0, sizeof(struct NvGlDemoWFDPipe));
    memset(portIDs, 0, nvGlWfdDev->numPorts * sizeof(WFDint));
    memset(pipeIDs, 0, nvGlWfdDev->numPipes * sizeof(WFDint));

    numPorts = pwfdEnumeratePorts(nvGlWfdDev->wfdDeviceHandle, portIDs, nvGlWfdDev->numPorts, NULL);
    if (numPorts != nvGlWfdDev->numPorts) {
         NvGlDemoLog("pwfdEnumeratePorts fail\n");
         ret = false;
         goto NvGlDemoGetWfdDevice_fail;
    }

    nvGlWfdDev->portInfo->portId = portIDs[nvGlWfdDev->curPortIndex];
    nvGlWfdDev->portInfo->portHandle =
        pwfdCreatePort(nvGlWfdDev->wfdDeviceHandle, portIDs[nvGlWfdDev->curPortIndex], NULL);
    if (!nvGlWfdDev->portInfo->portHandle) {
        NvGlDemoLog("pwfdCreatePort fail\n");
        ret = false;
        goto NvGlDemoGetWfdDevice_fail;
    }

    nvGlWfdDev->portInfo->numPortModes = pwfdGetPortModes(nvGlWfdDev->wfdDeviceHandle,
        nvGlWfdDev->portInfo->portHandle, NULL, 0);
    if (!nvGlWfdDev->portInfo->numPortModes) {
        NvGlDemoLog("pwfdGetPortModes fail\n");
        ret = false;
        goto NvGlDemoGetWfdDevice_fail;
    }

    nvGlWfdDev->portInfo->portModes = malloc(nvGlWfdDev->portInfo->numPortModes * sizeof(WFDPortMode));
    if (!nvGlWfdDev->portInfo->portModes) {
        NvGlDemoLog("Wfd Res Alloc fail\n");
        ret = false;
        goto NvGlDemoGetWfdDevice_fail;
    }
    memset(nvGlWfdDev->portInfo->portModes, 0, nvGlWfdDev->portInfo->numPortModes * sizeof(WFDPortMode));

    numPortModes = pwfdGetPortModes(nvGlWfdDev->wfdDeviceHandle, nvGlWfdDev->portInfo->portHandle,
        nvGlWfdDev->portInfo->portModes, nvGlWfdDev->portInfo->numPortModes);
    if (numPortModes != nvGlWfdDev->portInfo->numPortModes) {
        NvGlDemoLog("pwfdGetPortModes fail\n");
        ret = false;
        goto NvGlDemoGetWfdDevice_fail;
    }

    numPipes = pwfdEnumeratePipelines(nvGlWfdDev->wfdDeviceHandle, pipeIDs, nvGlWfdDev->numPipes, NULL);
    if (numPipes != nvGlWfdDev->numPipes) {
        NvGlDemoLog("pwfdEnumeratePorts fail\n");
        ret = false;
        goto NvGlDemoGetWfdDevice_fail;
    }

    if (demoOptions.displayLayer >= 0) {
        nvGlWfdDev->curPipeIndex = demoOptions.displayLayer;
    }

    if (nvGlWfdDev->curPipeIndex >= nvGlWfdDev->numPipes) {
        NvGlDemoLog("Display pipeline %d is not available.\n",nvGlWfdDev->curPipeIndex);
        NvGlDemoLog("Range of available pipeline IDs: [0:%d].\n",nvGlWfdDev->numPipes - 1);
        ret = false;
        goto NvGlDemoGetWfdDevice_fail;
    }

    // Check if the user selected pipe is bindable on this port
    numBindablePipes = pwfdGetPortAttribi(nvGlWfdDev->wfdDeviceHandle, nvGlWfdDev->portInfo->portHandle,
        WFD_PORT_PIPELINE_ID_COUNT);
    if (pwfdGetError(nvGlWfdDev->wfdDeviceHandle) != WFD_ERROR_NONE) {
        NvGlDemoLog("pwfdGetPortAttribi fail\n");
        ret = false;
        goto NvGlDemoGetWfdDevice_fail;
    }
    if (!numBindablePipes) {
        NvGlDemoLog("No layers exist on display %d.\n", nvGlWfdDev->curPortIndex);
        ret = false;
        goto NvGlDemoGetWfdDevice_fail;
    }

    bindablePipeIDs = (WFDint*)malloc(numBindablePipes * sizeof(WFDint));
    if (!bindablePipeIDs) {
        NvGlDemoLog("Wfd Res Alloc fail\n");
        ret = false;
        goto NvGlDemoGetWfdDevice_fail;
    }

    memset(bindablePipeIDs, 0, numBindablePipes * sizeof(WFDint));
    pwfdGetPortAttribiv(nvGlWfdDev->wfdDeviceHandle, nvGlWfdDev->portInfo->portHandle,
        WFD_PORT_BINDABLE_PIPELINE_IDS, numBindablePipes, bindablePipeIDs);
    if (pwfdGetError(nvGlWfdDev->wfdDeviceHandle) != WFD_ERROR_NONE) {
        NvGlDemoLog("pwfdGetPortAttribiv fail\n");
        ret = false;
        goto NvGlDemoGetWfdDevice_fail;
    }

    for (i = 0; i < numBindablePipes; ++i) {
        if (pipeIDs[nvGlWfdDev->curPipeIndex] == bindablePipeIDs[i]){
            isPipeBindable = true;
            break;
        }
    }
    if (!isPipeBindable) {
        NvGlDemoLog("Layer %d is not bindable on display %d.\n", nvGlWfdDev->curPipeIndex, nvGlWfdDev->curPortIndex);
        ret = false;
        goto NvGlDemoGetWfdDevice_fail;
    }

    // Create the user selected pipe
    nvGlWfdDev->pipeInfo->pipeId = pipeIDs[nvGlWfdDev->curPipeIndex];
    nvGlWfdDev->pipeInfo->pipeHandle =
        pwfdCreatePipeline(nvGlWfdDev->wfdDeviceHandle, pipeIDs[nvGlWfdDev->curPipeIndex], NULL);
    if (!nvGlWfdDev->pipeInfo->pipeHandle) {
        NvGlDemoLog("pwfdCreatePipeline fail\n");
        if (pwfdGetError(nvGlWfdDev->wfdDeviceHandle) == WFD_ERROR_IN_USE) {
            NvGlDemoLog("Layer %d is already bound.\n");
        }
        ret = false;
        goto NvGlDemoGetWfdDevice_fail;
    }

    // Get the EGL layer corresponding to this WFD pipe
    layerAttrib[0] = EGL_OPENWF_PIPELINE_ID_EXT;
    layerAttrib[1] = nvGlWfdDev->pipeInfo->pipeId;
    if (peglGetOutputLayersEXT(devOut->eglDpy, layerAttrib, &tempLayer, 1, &n) && n > 0) {
        devOut->layerList[0] = tempLayer;
        devOut->layerCount = 1;
        nvGlWfdDev->pipeInfo->layer = 0;
    }

    if (!devOut->layerCount) {
        NvGlDemoLog("Suitable layer count is 0.\n");
        ret = false;
        goto NvGlDemoGetWfdDevice_fail;
    }

    goto NvGlDemoGetWfdDevice_success;

NvGlDemoGetWfdDevice_fail:
    NvGlDemoTermWfdDevice();

NvGlDemoGetWfdDevice_success:
    if (bindablePipeIDs) {
        free(bindablePipeIDs);
    }
    if (portIDs) {
        free(portIDs);
    }
    if (pipeIDs) {
        free(pipeIDs);
    }

    return ret;
}

static void NvGlDemoResetWfdDeviceFnPtr(void)
{
    pwfdEnumerateDevices = NULL;
    pwfdCreateDevice = NULL;
    pwfdDestroyDevice = NULL;
    pwfdDeviceCommit = NULL;
    pwfdEnumeratePipelines = NULL;
    pwfdEnumeratePorts = NULL;
    pwfdGetPortModes = NULL;
    pwfdGetError = NULL;
    pwfdCreatePort = NULL;
    pwfdCreatePipeline = NULL;
    pwfdGetPortModeAttribi = NULL;
    pwfdSetPipelineAttribf = NULL;
    pwfdBindPipelineToPort = NULL;
    pwfdSetPortMode = NULL;
    pwfdDestroyPort = NULL;
    pwfdDestroyPipeline = NULL;
    pwfdGetCurrentPortMode = NULL;
    pwfdSetPipelineAttribiv = NULL;
    pwfdBindSourceToPipeline = NULL;
    pwfdGetPortAttribi = NULL;
    pwfdGetPortAttribiv = NULL;
}

static bool NvGlDemoInitWfdDevice(void)
{
#if !defined(__INTEGRITY)
    // Open WFD library
    libWFD = dlopen("libtegrawfd.so", RTLD_LAZY);
    if (!libWFD) {
        NvGlDemoLog("dlopen-libtegrawfd.so fail.\n");
        return false;
    }
#endif

    NVGLDEMO_LOAD_WFD_PTR(wfdEnumerateDevices, PFNWFDENUMERATEDEVICES);
    NVGLDEMO_LOAD_WFD_PTR(wfdCreateDevice, PFNWFDCREATEDEVICE);
    NVGLDEMO_LOAD_WFD_PTR(wfdDestroyDevice, PFNWFDDESTROYDEVICE);
    NVGLDEMO_LOAD_WFD_PTR(wfdDeviceCommit, PFNWFDDEVICECOMMIT);
    NVGLDEMO_LOAD_WFD_PTR(wfdEnumeratePipelines, PFNWFDENUMERATEPIPELINES);
    NVGLDEMO_LOAD_WFD_PTR(wfdEnumeratePorts, PFNWFDENUMERATEPORTS);
    NVGLDEMO_LOAD_WFD_PTR(wfdGetPortModes, PFNWFDGETPORTMODES);
    NVGLDEMO_LOAD_WFD_PTR(wfdGetError, PFNWFDGETERROR);
    NVGLDEMO_LOAD_WFD_PTR(wfdCreatePort, PFNWFDCREATEPORT);
    NVGLDEMO_LOAD_WFD_PTR(wfdCreatePipeline, PFNWFDCREATEPIPELINE);
    NVGLDEMO_LOAD_WFD_PTR(wfdGetPortModeAttribi, PFNWFDGETPORTMODEATTRIBI);
    NVGLDEMO_LOAD_WFD_PTR(wfdSetPipelineAttribf, PFNWFDSETPIPELINEATTRIBF);
    NVGLDEMO_LOAD_WFD_PTR(wfdBindPipelineToPort, PFNWFDBINDPIPELINETOPORT);
    NVGLDEMO_LOAD_WFD_PTR(wfdSetPortMode, PFNWFDSETPORTMODE);
    NVGLDEMO_LOAD_WFD_PTR(wfdDestroyPort, PFNWFDDESTROYPORT);
    NVGLDEMO_LOAD_WFD_PTR(wfdDestroyPipeline, PFNWFDDESTROYPIPELINE);
    NVGLDEMO_LOAD_WFD_PTR(wfdGetCurrentPortMode, PFNWFDGETCURRENTPORTMODE);
    NVGLDEMO_LOAD_WFD_PTR(wfdSetPipelineAttribiv, PFNWFDSETPIPELINEATTRIBIV);
    NVGLDEMO_LOAD_WFD_PTR(wfdBindSourceToPipeline, PFNWFDBINDSOURCETOPIPELINE);
    NVGLDEMO_LOAD_WFD_PTR(wfdGetPortAttribi, PFNWFDGETPORTATTRIBI);
    NVGLDEMO_LOAD_WFD_PTR(wfdGetPortAttribiv, PFNWFDGETPORTATTRIBIV);

    nvGlWfdDev = (struct NvGlDemoWFDDevice*)MALLOC(sizeof(struct NvGlDemoWFDDevice));
    if (!nvGlWfdDev) {
        NvGlDemoLog("Could not allocate Wfd Device specific storage memory.\n");
        goto NvGlDemoInitWfdDevice_fail;
    }
    NvGlDemoResetWfdDevice();

    // Success
    return true;

NvGlDemoInitWfdDevice_fail:
    NvGlDemoTermWfdDevice();
    NvGlDemoLog("NvGlDemoInitDrmDevice fail.\n");
    return false;
}

// This function does the necessary post swap buffers processing
// In case of EGLOutput+drm backend we pass the atomic mode setting structure
// Ret: 1 for success
//      0 for failure
int NvGlDemoPostSwap(void)
{
    return 1;
}

#else

//======================================================================
// DRM Desktop functions
//======================================================================

// Load the EGL and DRM libraries if available
static bool NvGlDemoInitDrmDevice(void)
{
#if !defined(__INTEGRITY)
    // Open DRM library
    libDRM = dlopen("libdrm.so.2", RTLD_LAZY);
    if (!libDRM) {
        NvGlDemoLog("dlopen-libdrm.so.2 fail.\n");
        return false;
    }

    if (demoOptions.allocator == NvGlDemoAllocator_GBM) {
        libminiGBM = dlopen("libgbm.so.1", RTLD_LAZY);
        if (!libminiGBM) {
            NvGlDemoLog("dlopen-libgbm.so.1 fail.\n");
            return false;
        } else {
            // Get GBM functions
            NVGLDEMO_LOAD_GBM_PTR(gbm_create_device, PFNGBMCREATEDEVICE);
            NVGLDEMO_LOAD_GBM_PTR(gbm_bo_create, PFNGBMBOCREATE);
            NVGLDEMO_LOAD_GBM_PTR(gbm_bo_get_stride_for_plane, PFNGBMBOGETSTRIDEFORPLANE);
            NVGLDEMO_LOAD_GBM_PTR(gbm_bo_get_handle_for_plane, PFNGBMBOGETHANDLEFORPLANE);
            NVGLDEMO_LOAD_GBM_PTR(gbm_bo_destroy, PFNGBMBODESTROY);
            NVGLDEMO_LOAD_GBM_PTR(gbm_device_destroy, PFNGBMDEVICEDESTROY);
        }
    }
#endif

    // Get DRM functions
    NVGLDEMO_LOAD_DRM_PTR(drmOpen, PFNDRMOPEN);
    NVGLDEMO_LOAD_DRM_PTR(drmClose, PFNDRMCLOSE);
    NVGLDEMO_LOAD_DRM_PTR(drmModeGetResources, PFNDRMMODEGETRESOURCES);
    NVGLDEMO_LOAD_DRM_PTR(drmModeFreeResources, PFNDRMMODEFREERESOURCES);
    NVGLDEMO_LOAD_DRM_PTR(drmModeGetPlaneResources, PFNDRMMODEGETPLANERESOURCES);
    NVGLDEMO_LOAD_DRM_PTR(drmModeFreePlaneResources, PFNDRMMODEFREEPLANERESOURCES);
    NVGLDEMO_LOAD_DRM_PTR(drmModeGetConnector, PFNDRMMODEGETCONNECTOR);
    NVGLDEMO_LOAD_DRM_PTR(drmModeFreeConnector, PFNDRMMODEFREECONNECTOR);
    NVGLDEMO_LOAD_DRM_PTR(drmModeGetEncoder, PFNDRMMODEGETENCODER);
    NVGLDEMO_LOAD_DRM_PTR(drmModeFreeEncoder, PFNDRMMODEFREEENCODER);
    NVGLDEMO_LOAD_DRM_PTR(drmModeGetPlane, PFNDRMMODEGETPLANE);
    NVGLDEMO_LOAD_DRM_PTR(drmModeFreePlane, PFNDRMMODEFREEPLANE);
    NVGLDEMO_LOAD_DRM_PTR(drmModeSetCrtc, PFNDRMMODESETCRTC);
    NVGLDEMO_LOAD_DRM_PTR(drmModeGetCrtc, PFNDRMMODEGETCRTC);
    NVGLDEMO_LOAD_DRM_PTR(drmModeSetPlane, PFNDRMMODESETPLANE);
    NVGLDEMO_LOAD_DRM_PTR(drmModeFreeCrtc, PFNDRMMODEFREECRTC);
    NVGLDEMO_LOAD_DRM_PTR(drmModeAtomicAlloc, PFNDRMMODEATOMICALLOC);
    NVGLDEMO_LOAD_DRM_PTR(drmModeAtomicAddProperty, PFNDRMMODEATOMICADDPROPERTY);
    NVGLDEMO_LOAD_DRM_PTR(drmModeAtomicCommit, PFNDRMMODEATOMICCOMMIT);
    NVGLDEMO_LOAD_DRM_PTR(drmModeAtomicFree, PFNDRMMODEATOMICFREE);
    NVGLDEMO_LOAD_DRM_PTR(drmModeObjectGetProperties, PFNDRMMODEOBJECTGETPROPERTIES);
    NVGLDEMO_LOAD_DRM_PTR(drmModeGetProperty, PFNDRMMODEGETPROPERTY);
    NVGLDEMO_LOAD_DRM_PTR(drmModeFreeProperty, PFNDRMMODEFREEPROPERTY);
    NVGLDEMO_LOAD_DRM_PTR(drmModeFreeObjectProperties, PFNDRMMODEFREEOBJECTPROPERTIES);
    NVGLDEMO_LOAD_DRM_PTR(drmSetClientCap, PFNDRMSETCLIENTCAP);
    NVGLDEMO_LOAD_DRM_PTR(drmIoctl, PFNDRMIOCTL);
    NVGLDEMO_LOAD_DRM_PTR(drmModeAddFB2, PFNDRMMODEADDFB2);
    NVGLDEMO_LOAD_DRM_PTR(drmModeCreatePropertyBlob, PFNDRMMODECREATEPROPERTYBLOB);
    NVGLDEMO_LOAD_DRM_PTR(drmModeDestroyPropertyBlob, PFNDRMMODEDESTROYPROPERTYBLOB);
    NVGLDEMO_LOAD_DRM_PTR(drmGetVersion, PFNDRMGETVERSION);
    NVGLDEMO_LOAD_DRM_PTR(drmFreeVersion, PFNDRMFREEVERSION);

    nvGlDrmDev =
        (struct NvGlDemoDRMDevice*)MALLOC(sizeof(struct NvGlDemoDRMDevice));
    if (!nvGlDrmDev) {
        NvGlDemoLog("Could not allocate Drm Device specific storage memory.\n");
        goto NvGlDemoInitDrmDevice_fail;
    }
    NvGlDemoResetDrmDevice();

    // Success
    return true;

NvGlDemoInitDrmDevice_fail:

    NvGlDemoTermDrmDevice();
    NvGlDemoLog("NvGlDemoInitDrmDevice fail.\n");
    return false;
}

// Return the plane type for the specified objectID
static int GetDrmPlaneType(int drmFd, uint32_t objectID)
{
    uint32_t i;
    int j;
    int found = 0;
    uint64_t value = 0;
    int planeType = DRM_PLANE_TYPE_OVERLAY;

    drmModeObjectPropertiesPtr pModeObjectProperties =
        pdrmModeObjectGetProperties(drmFd, objectID, DRM_MODE_OBJECT_PLANE);

    for (i = 0; i < pModeObjectProperties->count_props; i++) {
       drmModePropertyPtr pProperty =
           pdrmModeGetProperty(drmFd, pModeObjectProperties->props[i]);

       if (pProperty == NULL) {
           NvGlDemoLog("Unable to query property.\n");
       }

       if(STRCMP("type", pProperty->name) == 0) {
           value = pModeObjectProperties->prop_values[i];
           found = 1;
           for (j = 0; j < pProperty->count_enums; j++) {
               if (value == (pProperty->enums[j]).value) {
                   if (STRCMP( "Primary", (pProperty->enums[j]).name) == 0) {
                       planeType = DRM_PLANE_TYPE_PRIMARY;
                   } else if (STRCMP( "Overlay", (pProperty->enums[j]).name) == 0) {
                       planeType = DRM_PLANE_TYPE_OVERLAY;
                   } else {
                       planeType = DRM_PLANE_TYPE_CURSOR;
                   }
               }
           }
       }

       pdrmModeFreeProperty(pProperty);

       if (found)
           break;
    }

    pdrmModeFreeObjectProperties(pModeObjectProperties);

    if (!found) {
       NvGlDemoLog("Unable to find value for property \'type.\'\n");
    }

    return planeType;
}

// Opens the DRM FD  if not already opened for the DRM Device. It also sets all
// the capabilities that the platform needs to be supported and initializes GBM
// on required platforms.
// Returns : 0 - successful
//           -1 - Failure
int NvGlDemoCreateDrmDevice()
{
    int devIndx = demoState.platform->curDevIndx;
    const char * devStr = NULL;

    // Get DRM device string
    if ((devStr = peglQueryDeviceStringEXT(devList[devIndx],
                                           EGL_DRM_DEVICE_FILE_EXT)) == NULL) {
        NvGlDemoLog("EGL_DRM_DEVICE_FILE_EXT fail\n");
        return -1;
    }

    // Check if the backend is drm-nvdc
    if ((nvGlDrmDev->isDrmNvdc = (strcmp(devStr, "drm-nvdc") == 0))) {
        if ((nvGlDrmDev->fd = pdrmOpen(devStr, NULL)) == -1) {
            NvGlDemoLog("%s open fail\n", devStr);
            return -1;
        }

        // Set DRM permissive only when using drm-nvdc backend in the driver
        if (demoOptions.isDrmNvdcPermissive == -1) {
            // Gather the DRM client caps
            struct drm_tegra_get_client_cap drmClientCapArgs;
            // Get the value from DRM
            drmClientCapArgs.cap = DRM_CLIENT_CAP_DRM_NVDC_PERMISSIVE;
            if (pdrmIoctl(nvGlDrmDev->fd, DRM_IOCTL_TEGRA_GET_CLIENT_CAP,
                          &drmClientCapArgs) < 0) {
                NvGlDemoLog("Failed to get DRM cap: "
                            "DRM_CLIENT_CAP_DRM_NVDC_PERMISSIVE \n");
                return -1;
            }
            nvGlDrmDev->isDrmNvdcPermissive = drmClientCapArgs.val;
        } else {
            // Set the CAPS value depending on user input
            if (!(pdrmSetClientCap(nvGlDrmDev->fd, DRM_CLIENT_CAP_DRM_NVDC_PERMISSIVE,
                                   demoOptions.isDrmNvdcPermissive) == 0)) {
                NvGlDemoLog("DRM_CLIENT_CAP_DRM_NVDC_PERMISSIVE not available.\n");
                return -1;
            }
            nvGlDrmDev->isDrmNvdcPermissive = !!demoOptions.isDrmNvdcPermissive;
        }
    } else {
        if ((nvGlDrmDev->fd = open(devStr, O_RDWR|O_CLOEXEC)) == -1) {
            NvGlDemoLog("%s open fail\n", devStr);
            return -1;
        }
        nvGlDrmDev->isDrmNvdcPermissive = demoOptions.isDrmNvdcPermissive = -1;
    }

    // Set isDrmNvdc based on drm version string
    drmVersionPtr version;
    version = pdrmGetVersion(nvGlDrmDev->fd);
    if (!strcmp(version->name, "drm-nvdc") ||
        !strcmp(version->name, "tegra-udrm") ||
        !strcmp(version->name, "tegra")) {
        nvGlDrmDev->isDrmNvdc = 1;
    }
    pdrmFreeVersion(version);

    // Set Atomic Modeset and Universal Plane capabilities
    if (!(pdrmSetClientCap(nvGlDrmDev->fd, DRM_CLIENT_CAP_ATOMIC, 1) == 0)) {
        NvGlDemoLog("DRM_CLIENT_CAP_ATOMIC not available.\n");
        return -1;
    }

    if (!(pdrmSetClientCap(nvGlDrmDev->fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1) == 0)) {
        NvGlDemoLog("DRM_CLIENT_CAP_UNIVERSAL_PLANES not available.\n");
        return -1;
    }

    if (demoOptions.allocator == NvGlDemoAllocator_GBM) {
        nvGlDrmDev->gbmDev = pgbm_create_device(nvGlDrmDev->fd);
        if (!nvGlDrmDev->gbmDev) {
            NvGlDemoLog("Failed to create GBM device\n");
            return -1;
        }
    }

    return 0;
}

// Gets DRM/EGLDevice desktop resources and populates the DRM device
static bool NvGlDemoGetDrmDevice(EGLint devIndx)
{
    struct NvGlOutputDevice *devOut = NULL;
    EGLOutputLayerEXT tempLayer;
    int i = 0, j = 0, n = 0, layerIndex = 0;

    EGLAttrib layerAttrib[3] = {
        0,
        0,
        EGL_NONE
    };

    devOut = &nvGlOutDevLst[devIndx];

    if (!nvGlDrmDev->fd) {
        NvGlDemoLog("No drm device fd opened. Please initialize drm first\n");
        return false;
    }

    if ((nvGlDrmDev->res = pdrmModeGetResources(nvGlDrmDev->fd)) == NULL) {
        NvGlDemoLog("pdrmModeGetResources fail\n");
        return false;
    }
    if ((nvGlDrmDev->planes = pdrmModeGetPlaneResources(nvGlDrmDev->fd)) == NULL) {
        NvGlDemoLog("pdrmModeGetPlaneResources fail\n");
        return false;
    }
    // Validate connector, if requested
    if (nvGlDrmDev->connDefault >= nvGlDrmDev->res->count_connectors) {
        NvGlDemoLog("con def != max con\n");
        return false;
    }

    // Allocate info arrays for DRM state
    nvGlDrmDev->connInfo = (NvGlDemoDRMConn*)malloc(nvGlDrmDev->res->count_connectors * sizeof(NvGlDemoDRMConn));
    nvGlDrmDev->crtcInfo = (NvGlDemoDRMCrtc*)malloc(nvGlDrmDev->res->count_crtcs * sizeof(NvGlDemoDRMCrtc));
    nvGlDrmDev->planeInfo = (NvGlDemoDRMPlane*)malloc(nvGlDrmDev->planes->count_planes * sizeof(NvGlDemoDRMPlane));
    if (!nvGlDrmDev->connInfo || !nvGlDrmDev->crtcInfo || !nvGlDrmDev->planeInfo) {
        NvGlDemoLog("Drm Res Alloc fail\n");
        return false;
    }
    memset(nvGlDrmDev->connInfo, 0, nvGlDrmDev->res->count_connectors * sizeof(NvGlDemoDRMConn));
    memset(nvGlDrmDev->crtcInfo, 0, nvGlDrmDev->res->count_crtcs * sizeof(NvGlDemoDRMCrtc));
    memset(nvGlDrmDev->planeInfo, 0, nvGlDrmDev->planes->count_planes * sizeof(NvGlDemoDRMPlane));

    // Parse connector info
    for (i=0; i<nvGlDrmDev->res->count_connectors; ++i) {

        if (demoOptions.displayNumber != -1 && demoOptions.displayNumber != i) {
            // If user provides a dispno, connect only to that dispno
            continue;
        }
        // Start with no crtc assigned
        nvGlDrmDev->connInfo[i].crtcMapping = -1;

        // Skip if not connector
        drmModeConnector* conn = pdrmModeGetConnector(nvGlDrmDev->fd, nvGlDrmDev->res->connectors[i]);
        if (!conn || (conn->connection != DRM_MODE_CONNECTED)) {
            if (conn) {
                // Free the connector info
                pdrmModeFreeConnector(conn);
            }
            continue;
        }

        // If the connector has no modes available, try to use the current
        // mode.  Show a warning if the user didn't specifically request that.
        if (conn->count_modes <= 0 && (!demoOptions.useCurrentMode)) {
            NvGlDemoLog("Warning: No valid modes found for connector, "
                        "using -currentmode\n");
            demoOptions.useCurrentMode = 1;
        }

        // If we don't already have a default, use this one
        if (nvGlDrmDev->connDefault < 0) {
            nvGlDrmDev->connDefault = i;
        }

        // Mark as valid
        nvGlDrmDev->connInfo[i].valid = true;
        demoState.platform->curConnIndx = i;

        // Find the possible crtcs
        for (j=0; j<conn->count_encoders; ++j) {
            drmModeEncoder* enc = pdrmModeGetEncoder(nvGlDrmDev->fd, conn->encoders[j]);
            nvGlDrmDev->connInfo[i].crtcMask |= enc->possible_crtcs;
            pdrmModeFreeEncoder(enc);
        }
        // Free the connector info
        pdrmModeFreeConnector(conn);
    }

    if (demoOptions.displayNumber != -1 && !nvGlDrmDev->connInfo[demoOptions.displayNumber].valid) {
        NvGlDemoLog("Error: Not a valid/connected dispno %d \n", demoOptions.displayNumber);
        return false;
    }
    // Parse plane info
    for (i=0; i<(int)nvGlDrmDev->planes->count_planes; ++i) {
        drmModePlane* plane = pdrmModeGetPlane(nvGlDrmDev->fd, nvGlDrmDev->planes->planes[i]);
        nvGlDrmDev->planeInfo[i].layer = -1;
        nvGlDrmDev->planeInfo[i].crtcMask = plane->possible_crtcs;
        pdrmModeFreePlane(plane);
        nvGlDrmDev->planeInfo[i].planeType = GetDrmPlaneType(nvGlDrmDev->fd, nvGlDrmDev->planes->planes[i]);
    }

    // Map layers to planes
    layerAttrib[0] = EGL_DRM_PLANE_EXT;
    for (i=0; i<(int)nvGlDrmDev->planes->count_planes; ++i) {
        layerAttrib[1] = nvGlDrmDev->planes->planes[i];
        if (peglGetOutputLayersEXT(devOut->eglDpy, layerAttrib, &tempLayer, 1, &n) && n > 0) {
            devOut->layerList[layerIndex] = tempLayer;
            devOut->layerCount++;
            nvGlDrmDev->planeInfo[i].layer = layerIndex++;
        }
    }

    if (!devOut->layerCount) {
        NvGlDemoLog("Layer count is 0.\n");
        return false;
    }

    return true;
}

// Returns the plane index of the plane chosen to be rendered by the user (or
// the default layer to be drawn onto) for the current CRTC. It verifies that
// the plane is free and ready to be used
// Returns: a positive plane index - successful
//          -1 - for failure
static int NvGlDemoChooseRenderingPlaneIndex(int crtcIndex)
{
    unsigned int availableLayers = 0, planeIndex = 0;
    for (planeIndex = 0; planeIndex < nvGlDrmDev->planes->count_planes;
         ++planeIndex) {
        if (!nvGlDrmDev->planeInfo[planeIndex].used &&
            (nvGlDrmDev->planeInfo[planeIndex].crtcMask & (1 << crtcIndex)) &&
            (availableLayers++ == (unsigned int)demoOptions.displayLayer)) {

            // Check if the rendering plane is valid
            if (planeIndex == nvGlDrmDev->planes->count_planes) {
                NvGlDemoLog("ERROR: Layer ID %d is not valid on display %d.\n",
                            demoOptions.displayLayer,
                            demoOptions.displayNumber);
                NvGlDemoLog("Range of available Layer IDs: [0, %d]",
                            availableLayers - 1);
                return -1;
            }
            return planeIndex;
        }
    }
    return -1;
}

// Returns the plane index of the primary plane for the current CRTC
// Returns: a positive plane index - successful
//          -1 - for failure
static int NvGlDemoGetPrimaryPlaneIndex(int crtcIndex)
{
    unsigned int planeIndex = 0;
    for (planeIndex = 0; planeIndex < nvGlDrmDev->planes->count_planes;
         ++planeIndex) {
        if (nvGlDrmDev->planeInfo[planeIndex].planeType == DRM_PLANE_TYPE_PRIMARY
            && (nvGlDrmDev->planeInfo[planeIndex].crtcMask & (1 << crtcIndex))) {
            return planeIndex;
        }
    }
    return -1;
}

// Look through drm's list of properties for this object and return the
// ID of the one matching the provided propName string.
//
// Return non-zero property ID for success
//        0 for failure
static uint32_t getPropertyId(int fd, drmModeObjectPropertiesPtr properties,
                              const char *propName)
{
    uint32_t propertyId = 0;
    uint32_t i;

    for (i = 0; i < properties->count_props; i++) {
        drmModePropertyPtr p = pdrmModeGetProperty(fd, properties->props[i]);

        if (!p) {
            NvGlDemoLog("Unable to query property.\n");
            return 0;
        }

        if (strcmp(propName, p->name) == 0) {
            propertyId = p->prop_id;
        }

        pdrmModeFreeProperty(p);

        if (propertyId) {
            break;
        }
    }
    return propertyId;
}


// This function fills the atomic request structure with an active mode.
// A modeId is retrieved by creating a property blob and then the active mode
// property is set with the new modeId.
// Ret 0 for success
//     -1 or an error code for failure
static int NvGLDemoAtomicAddMode(int fd, drmModeAtomicReqPtr req,
                                 uint32_t modeId, uint32_t connId,
                                 uint32_t crtcId)
{
    int err = 0;
    uint32_t propId;
    drmModeObjectPropertiesPtr crtcProps = NULL;
    crtcProps = pdrmModeObjectGetProperties(fd, crtcId, DRM_MODE_OBJECT_CRTC);
    if (!crtcProps) {
        return -1;
    }
    drmModeObjectPropertiesPtr connProps = NULL;
    connProps = pdrmModeObjectGetProperties(fd, connId,
                                            DRM_MODE_OBJECT_CONNECTOR);
    if (!connProps) {
        err = -1;
        goto end;
    }

#define SET_PROP(propName, properties, setProp, propValue) \
    propId = getPropertyId(fd, properties, propName); \
    if (propId == 0) { \
        err = -1; \
        goto end; \
    } \
    err = pdrmModeAtomicAddProperty(req, setProp, propId, propValue) <= 0; \
    if (err) { \
        goto end; \
    }

    SET_PROP("CRTC_ID", connProps, connId, crtcId);
    SET_PROP("MODE_ID", crtcProps, crtcId, modeId);
    SET_PROP("ACTIVE", crtcProps, crtcId, 1);

#undef SET_PROP

end:
    pdrmModeFreeObjectProperties(crtcProps);
    pdrmModeFreeObjectProperties(connProps);

    return err;
}

// This function is used to set input atomic mode request structure with the
// input and output plane information. This function also sets the alpha and
// thus helps in configuring the whole plane with the given planeId.
// Ret 0 for success
//     -1 or error codes for failure
static int NvGLDemoAtomicPlaneResize(int fd, drmModeAtomicReqPtr req,
                                     uint32_t planeId, uint32_t crtcId,
                                     uint32_t crtcX, uint32_t crtcY,
                                     uint32_t crtcW, uint32_t crtcH,
                                     uint32_t srcX, uint32_t srcY,
                                     uint32_t srcW, uint32_t srcH)
{
    int err = 0;
    uint32_t propId;
    drmModeObjectPropertiesPtr planeProps;
    planeProps = pdrmModeObjectGetProperties(fd, planeId, DRM_MODE_OBJECT_PLANE);
    if (!planeProps) {
        return -1;
    }

#define SET_PROP(propName, properties, setProp, propValue) \
    propId = getPropertyId(fd, properties, propName); \
    if (propId == 0) { \
        err = -1; \
        goto end; \
    } \
    err = pdrmModeAtomicAddProperty(req, setProp, propId, propValue) <= 0; \
    if (err) { \
        goto end; \
    }

    SET_PROP("CRTC_X", planeProps, planeId, crtcX);
    SET_PROP("CRTC_Y", planeProps, planeId, crtcY);
    SET_PROP("CRTC_W", planeProps, planeId, crtcW);
    SET_PROP("CRTC_H", planeProps, planeId, crtcH);
    SET_PROP("SRC_X", planeProps, planeId, srcX);
    SET_PROP("SRC_Y", planeProps, planeId, srcY);
    SET_PROP("SRC_W", planeProps, planeId, srcW);
    SET_PROP("SRC_H", planeProps, planeId, srcH);
    SET_PROP("CRTC_ID", planeProps, planeId, crtcId);

#undef SET_PROP

end:
    pdrmModeFreeObjectProperties(planeProps);
    return err;
}

// This function is used to set alpha property to the plane pointed by the
// plane ID and add this information to the DRM atomic request structure
// Ret 0 for success
//     -1 or error codes for failure
static int NvGLDemoAtomicSetAlpha(int fd, drmModeAtomicReqPtr req,
                                  uint32_t planeId, uint32_t alpha)
{
    int err = 0;
    uint32_t propId;
    drmModeObjectPropertiesPtr planeProps;
    planeProps = pdrmModeObjectGetProperties(fd, planeId, DRM_MODE_OBJECT_PLANE);
    if (!planeProps) {
        return -1;
    }

#define SET_PROP(propName, properties, setProp, propValue) \
    propId = getPropertyId(fd, properties, propName); \
    if (propId == 0) { \
        err = -1; \
        goto end; \
    } \
    err = pdrmModeAtomicAddProperty(req, setProp, propId, propValue) <= 0; \
    if (err) { \
        goto end; \
    }

    SET_PROP("alpha", planeProps, planeId, alpha);

#undef SET_PROP

end:
    pdrmModeFreeObjectProperties(planeProps);
    return err;
}

// This function sets the atomic mode request structure with the CRTC id and the
// frame buffer ID of the frame which we want to set on that particular plane
// Ret 0 for success
//     -1 or err for failure
static int NvGLDemoAtomicAddFrameBuffer(int fd, drmModeAtomicReqPtr req,
                                        uint32_t connId, uint32_t crtcId,
                                        uint32_t planeId, uint32_t fbId)
{
    int err = 0;;
    uint32_t propId;
    drmModeObjectPropertiesPtr connProps = NULL, planeProps = NULL;
    connProps = pdrmModeObjectGetProperties(fd, connId,
                                            DRM_MODE_OBJECT_CONNECTOR);
    if (!connProps) {
        return -1;
    }
    planeProps = pdrmModeObjectGetProperties(fd, planeId, DRM_MODE_OBJECT_PLANE);
    if (!planeProps) {
        err = -1;
        goto end;
    }

#define SET_PROP(propName, properties, setProp, propValue) \
    propId = getPropertyId(fd, properties, propName); \
    if (propId == 0) { \
        err = -1; \
        goto end; \
    } \
    err = pdrmModeAtomicAddProperty(req, setProp, propId, propValue) <= 0; \
    if (err) { \
        goto end; \
    }

    SET_PROP("CRTC_ID",  connProps, connId, crtcId);
    SET_PROP("FB_ID",  planeProps, planeId, fbId);

#undef SET_PROP

end:
    if (planeProps) {
        pdrmModeFreeObjectProperties(planeProps);
    }
    if (connProps) {
        pdrmModeFreeObjectProperties(connProps);
    }

    return err;
}

static int NvGlDemoCreateDRMDumbBuffer(uint32_t width, int height,
                                       uint32_t bitDepth, bool plane_bo)
{
    // Set a dumb frame buffer to the primary plane (CRTC)
    struct drm_mode_create_dumb creq;
    struct drm_mode_map_dumb mreq;
    struct drm_mode_destroy_dumb dreq;
    NvGLDemoBO *dumbBO = malloc(sizeof(*dumbBO));
    if (!dumbBO) {
        return -1;
    }
    memset(dumbBO, 0, sizeof(*dumbBO));
    memset(&creq, 0, sizeof(creq));
    creq.width = width;
    creq.height = height;
    creq.bpp = bitDepth;
    uint8_t* map = NULL;

    // Create the dumb buffer
    if (pdrmIoctl(nvGlDrmDev->fd, DRM_IOCTL_MODE_CREATE_DUMB, &creq) < 0) {
        NvGlDemoLog("Unable to create dumb buffer\n");
        free(dumbBO);
        return -1;
    }

    // Map the frame buffer
    memset(&mreq, 0, sizeof(mreq));
    mreq.handle = creq.handle;
    if (pdrmIoctl(nvGlDrmDev->fd, DRM_IOCTL_MODE_MAP_DUMB, &mreq)) {
        NvGlDemoLog("Unable to map dumb buffer\n");
        goto fail;
    }

    // Map and clear the FB
    if (nvGlDrmDev->isDrmNvdc) {
        map = (uint8_t*)(mreq.offset);
    } else {
        map = (uint8_t*)mmap(0, creq.size, PROT_READ | PROT_WRITE, MAP_SHARED,
                             nvGlDrmDev->fd, mreq.offset);
        if (map == MAP_FAILED) {
            NvGlDemoLog("cannot mmap dumb buffer\n");
            goto fail;
        }
    }

    // Clear the frame buffer
    memset(map, 0x00, creq.size);
    dumbBO->bo_handle = creq.handle;
    dumbBO->width = width;
    dumbBO->height = height;
    dumbBO->pitch = creq.pitch;
    dumbBO->data = map;
    dumbBO->size = creq.size;

    nvGlDrmDev->crtcInfo[nvGlDrmDev->currCrtcIndx].dumbBufferCreated = true;
    // Store the buffer address in the crtcInfo structure
    if (plane_bo) {
        nvGlDrmDev->crtcInfo[nvGlDrmDev->currCrtcIndx].dumbBO[1] = dumbBO;
    } else {
        nvGlDrmDev->crtcInfo[nvGlDrmDev->currCrtcIndx].dumbBO[0] = dumbBO;
    }

    return 0;

fail:
    memset (&dreq, 0, sizeof (dreq));
    dreq.handle = creq.handle;
    pdrmIoctl(nvGlDrmDev->fd, DRM_IOCTL_MODE_DESTROY_DUMB, &dreq);
    free(dumbBO);
    return -1;
}

static void NvGlDemoDestroyDRMDumbBuffer(void)
{
    if (!nvGlDrmDev->crtcInfo ||
        !nvGlDrmDev->crtcInfo[nvGlDrmDev->currCrtcIndx].dumbBufferCreated) {
        return;
    }
    struct drm_mode_destroy_dumb dreq;

    for(int i = 0; i < 2; i++) {
        NvGLDemoBO *dumbBO = nvGlDrmDev->crtcInfo[nvGlDrmDev->currCrtcIndx].dumbBO[i];
        if (dumbBO) {
            munmap (dumbBO->data, dumbBO->size);
            memset(&dreq, 0, sizeof(dreq));
            dreq.handle = dumbBO->bo_handle;
            pdrmIoctl(nvGlDrmDev->fd, DRM_IOCTL_MODE_DESTROY_DUMB, &dreq);
            free(dumbBO);
        }
    }
}

static int NvGlDemoCreateGBMDumbBuffer(uint32_t width, uint32_t height, bool plane_bo)
{
    NvGLDemoBO *dumbBO = malloc(sizeof(*dumbBO));
    if (!dumbBO) {
        return -1;
    }

    memset(dumbBO, 0, sizeof(*dumbBO));
    dumbBO->width = width;
    dumbBO->height = height;
    dumbBO->gbmBo = pgbm_bo_create(nvGlDrmDev->gbmDev, width, height, DRM_FORMAT_ARGB8888,
                                  GBM_BO_USE_LINEAR); // We default to using ARGB8888 for the dumb bo.
    if (!dumbBO->gbmBo) {
        NvGlDemoLog("Unable to create dumb buffer via GBM\n");
        free(dumbBO);
        return -1;
    }

    dumbBO->pitch = pgbm_bo_get_stride_for_plane(dumbBO->gbmBo, 0);
    dumbBO->bo_handle = pgbm_bo_get_handle_for_plane(dumbBO->gbmBo, 0).u32;

    nvGlDrmDev->crtcInfo[nvGlDrmDev->currCrtcIndx].dumbBufferCreated = true;
    // Store the buffer address in the crtcInfo structure
    if (plane_bo) {
        nvGlDrmDev->crtcInfo[nvGlDrmDev->currCrtcIndx].dumbBO[1] = dumbBO;
    } else {
        nvGlDrmDev->crtcInfo[nvGlDrmDev->currCrtcIndx].dumbBO[0] = dumbBO;
    }

    return 0;
}

static void NvGlDemoDestroyGBMDumbBuffer(void)
{
    if (!nvGlDrmDev->crtcInfo ||
        !nvGlDrmDev->crtcInfo[nvGlDrmDev->currCrtcIndx].dumbBufferCreated) {
        return;
    }

    for(int i = 0; i < 2; i++) {
        NvGLDemoBO *dumbBO = nvGlDrmDev->crtcInfo[nvGlDrmDev->currCrtcIndx].dumbBO[i];
        if (dumbBO) {
            if (dumbBO->gbmBo) {
                pgbm_bo_destroy(dumbBO->gbmBo);
            }
        }
        free(dumbBO);
    }
}

// Creates a frame buffer and adds it to the crtcInfo structure for the current
// crtcIndex
//
// On L4T, the allocation is done via DRM IOCTLS.
// On Embedded-Linux, the allocation is done via GBM (minigbm). This is because the DRM KMD on Embedded-Linux
// does not support allocating large contigious buffers. minigbm does not have this issue as it uses the NvMap UMD
// for allocation specifically on Embedded-Linux.
//
// Ret:  0 if successful
//      :-1 for a failure
static int NvGlDemoCreateDumbBuffer(uint32_t width, int height,
                                    uint32_t bitDepth, bool plane_bo)
{
    if (demoOptions.allocator == NvGlDemoAllocator_GBM) {
        return NvGlDemoCreateGBMDumbBuffer(width, height, plane_bo);
    } else {
        return NvGlDemoCreateDRMDumbBuffer(width, height, bitDepth, plane_bo);
    }
}

// Destroys the dumb buffer for the current CRTC
static void NvGlDemoDestroyDumbBuffer(void)
{
    if (demoOptions.allocator == NvGlDemoAllocator_GBM) {
        return NvGlDemoDestroyGBMDumbBuffer();
    } else {
        return NvGlDemoDestroyDRMDumbBuffer();
    }
}

// This function does the necessary post swap buffers processing
// In case of EGLOutput+drm backend we pass the atomic mode setting structure
// Ret: 1 for success
//      0 for failure
int NvGlDemoPostSwap(void)
{
    EGLAttrib acquireAttr[3];
    int attrCount = 0;
    int error = 1;

    // Return if the EGLOutput DRM atomic extension is not supported as we
    // must have done the modeset and the plane config in the application and
    // we do not need to send in the atomic request to EGL.
    if (!isEGLOutputDrmAtomicEXTSupported) {
        return error;
    }

    // This check makes sure that we do the plane configuration only for the
    // first swap buffers call.
    if (!demoState.platform || !demoState.platform->pAtomicReq) {
        return error;
    }

    acquireAttr[attrCount++] = EGL_DRM_ATOMIC_REQUEST_NV;
    acquireAttr[attrCount++] = (EGLAttrib)demoState.platform->pAtomicReq;
    acquireAttr[attrCount] = EGL_NONE;

    // Consume the first frame to be displayed
    if (!peglStreamConsumerAcquireAttribEXT(demoState.display,
                                            demoState.stream,
                                            acquireAttr)) {
        error = 0;
        NvGlDemoLog("Unable to acquire the first frame (error 0x%x)\n",
                    eglGetError());
        goto end;
    }

    // Set the auto acquire attribute back to TRUE to resume normal operations
    if (!peglStreamAttribKHR(demoState.display,
                             demoState.stream,
                             EGL_CONSUMER_AUTO_ACQUIRE_EXT,
                             EGL_TRUE)) {
        error = 0;
        NvGlDemoLog("Unable to set consumer auto acquire (error 0x%x)\n",
                    eglGetError());
        goto end;
    }

end:

    // Destroy the atomic request structures
    pdrmModeDestroyPropertyBlob(nvGlDrmDev->fd, demoState.platform->modeId);
    demoState.platform->modeId = 0;
    pdrmModeAtomicFree(demoState.platform->pAtomicReq);
    // Set the pointer to NULL to avoid calling the manual frame consume
    // for every frame
    demoState.platform->pAtomicReq = NULL;
    return error;
}

// Set output mode
static bool NvGlDemoInitializeDrmOutputMode(void)
{
    int offsetX = 0;
    int offsetY = 0;
    unsigned int sizeX = 0;
    unsigned int sizeY = 0;
    unsigned int alpha = 255;

    int crtcIndex = -1;
    int i = 0;
    int renderingPlaneIndex = ~0;
    int primaryPlaneIndex = ~0;

    drmModeConnector* conn = NULL;
    drmModeEncoder* enc = NULL;
    struct NvGlOutputDevice *outDev = NULL;
    drmModeCrtcPtr currCrtc = NULL;
    drmModeModeInfoPtr newMode = NULL;

    unsigned int modeSize = 0;
    int modeIndex = -1;
    unsigned int modeX = 0;
    unsigned int modeY = 0;
    int foundMatchingDisplayRate = 1;

    // Input plane dimensions
    uint32_t srcX = 0, srcY = 0, srcW = 0, srcH = 0;
    // Output plane dimensions
    uint32_t crtcX = 0, crtcY = 0, crtcW = 0, crtcH = 0;

    // Get currently active connector Id
    uint32_t connId = 0;
    // Get the currently used CRTC Id
    uint32_t crtcId = 0;
    // Plane Id to set atomic properties on
    uint32_t renderingPlaneId = 0;
    uint32_t initialFbId = 0;
    uint32_t planeFbId = 0;

    // If not specified, use default window size
    if (!demoOptions.windowSize[0])
        demoOptions.windowSize[0] = NVGLDEMO_DEFAULT_WIDTH;
    if (!demoOptions.windowSize[1])
        demoOptions.windowSize[1] = NVGLDEMO_DEFAULT_HEIGHT;

    // Parse global plane alpha
    if(demoOptions.displayAlpha < 0.0 || demoOptions.displayAlpha > 1.0) {
        //If unspecified or out of range, default to 1.0
        NvGlDemoLog("Alpha value specified for constant blending is not in range [0, 1]. Using alpha 1.0.\n");
        demoOptions.displayAlpha = 1.0;
    }
    alpha = (unsigned int)(demoOptions.displayAlpha * (nvGlDrmDev->isDrmNvdc ? 0xff : 0xffff));

    offsetX = demoOptions.windowOffset[0];
    offsetY = demoOptions.windowOffset[1];
    sizeX = demoOptions.windowSize[0];
    sizeY = demoOptions.windowSize[1];

    nvGlDrmDev->curConnIndx = demoState.platform->curConnIndx;
    // If a specific screen was requested, use it
    if ((nvGlDrmDev->curConnIndx >= nvGlDrmDev->res->count_connectors) ||
            !nvGlDrmDev->connInfo[nvGlDrmDev->curConnIndx].valid) {
        NvGlDemoLog("Display output %d is not available, try using another display using option <-dispno>.\n",nvGlDrmDev->curConnIndx);
        goto NvGlDemoInitializeDrmOutputMode_fail;
    }

    // Get the current state of the connector
    conn = pdrmModeGetConnector(nvGlDrmDev->fd, nvGlDrmDev->res->connectors[nvGlDrmDev->curConnIndx]);
    if (!conn) {
        NvGlDemoLog("pdrmModeGetConnector-fail\n");
        goto NvGlDemoInitializeDrmOutputMode_fail;
    }
    enc = pdrmModeGetEncoder(nvGlDrmDev->fd, conn->encoder_id);
    if (enc) {
        for (i=0; i<nvGlDrmDev->res->count_crtcs; ++i) {
            if (nvGlDrmDev->res->crtcs[i] == enc->crtc_id) {
                nvGlDrmDev->currCrtcIndx = i;
            }
        }
        pdrmModeFreeEncoder(enc);
    } else {
        // If connector does not have an encoder attached, use the first one.
        enc = pdrmModeGetEncoder(nvGlDrmDev->fd, conn->encoders[0]);
        if (enc) {
            nvGlDrmDev->currCrtcIndx = enc->crtc_id;
            pdrmModeFreeEncoder(enc);
        } else {
            NvGlDemoLog("Failed to get an encoder\n");
            goto NvGlDemoInitializeDrmOutputMode_fail;
        }
    }

    if (nvGlDrmDev->connInfo[nvGlDrmDev->curConnIndx].crtcMapping >= 0) {
        crtcIndex = nvGlDrmDev->connInfo[nvGlDrmDev->curConnIndx].crtcMapping;
        assert(crtcIndex == nvGlDrmDev->currCrtcIndx);
    } else if (nvGlDrmDev->currCrtcIndx >= 0) {
        crtcIndex = nvGlDrmDev->currCrtcIndx;
        assert(!nvGlDrmDev->crtcInfo[crtcIndex].mapped);
    } else {
        for (crtcIndex=0; crtcIndex<nvGlDrmDev->res->count_crtcs; ++crtcIndex) {
            if (!nvGlDrmDev->crtcInfo[crtcIndex].mapped &&
                    (nvGlDrmDev->connInfo[nvGlDrmDev->curConnIndx].crtcMask & (1 << crtcIndex))) {
                break;
            }
        }
        if (crtcIndex == nvGlDrmDev->res->count_crtcs) {
            goto NvGlDemoInitializeDrmOutputMode_fail;
        }
        nvGlDrmDev->currCrtcIndx = crtcIndex;
    }

    renderingPlaneIndex = NvGlDemoChooseRenderingPlaneIndex(crtcIndex);
    if (renderingPlaneIndex < 0) {
        NvGlDemoLog("Failed to get the desired plane index for crtc: \n",
                    nvGlDrmDev->res->crtcs[crtcIndex]);
        goto NvGlDemoInitializeDrmOutputMode_fail;
    }
    primaryPlaneIndex = NvGlDemoGetPrimaryPlaneIndex(crtcIndex);
    if (primaryPlaneIndex < 0) {
        NvGlDemoLog("Failed to get the primary plane index for crtc: %d\n",
                    nvGlDrmDev->res->crtcs[crtcIndex]);
        goto NvGlDemoInitializeDrmOutputMode_fail;
    }

    // Update the mode set info
    if (!nvGlDrmDev->crtcInfo[crtcIndex].mapped) {
        if (demoOptions.displaySize[0]) {

            // Check whether the chosen mode is supported or not
            for (i=0; i<conn->count_modes; ++i) {
                drmModeModeInfoPtr mode = conn->modes + i;
                if (mode->hdisplay == demoOptions.displaySize[0]
                    && mode->vdisplay == demoOptions.displaySize[1]) {
                    modeIndex = i;
                    modeX = (unsigned int)mode->hdisplay;
                    modeY = (unsigned int)mode->vdisplay;
                    if (demoOptions.displayRate) {
                        if (mode->vrefresh == (unsigned int)demoOptions.displayRate) {
                            foundMatchingDisplayRate = 1;
                            break;
                        } else {
                            foundMatchingDisplayRate = 0;
                            continue;
                        }
                    }
                    break;
                }
            }
            if (!modeX || !modeY) {
                NvGlDemoLog("Unsupported Displaysize.\n");
                goto NvGlDemoInitializeDrmOutputMode_fail;
            }
            if (!foundMatchingDisplayRate) {
                NvGlDemoLog("Specified Refresh rate is not Supported with Specified Display size.\n");
                goto NvGlDemoInitializeDrmOutputMode_fail;
            }
        } else if (demoOptions.useCurrentMode) {
            if (demoOptions.displayRate) {
                NvGlDemoLog("Refresh Rate should not be specified with Current Mode Parameter.\n");
                goto NvGlDemoInitializeDrmOutputMode_fail;
            }
            // Check to see if there is already a mode set
            if ((currCrtc = (drmModeCrtcPtr)pdrmModeGetCrtc(nvGlDrmDev->fd,
                       nvGlDrmDev->res->crtcs[crtcIndex])) != NULL) {
                modeIndex = -1;
            }
        } else {
            if (demoOptions.displayRate) {
                NvGlDemoLog("Refresh Rate should not be specified alone.\n");
                goto NvGlDemoInitializeDrmOutputMode_fail;
            }
            // Choose the preferred mode if it's set,
            // Or else choose the largest supported mode
            for (i=0; i<conn->count_modes; ++i) {
                drmModeModeInfoPtr mode = conn->modes + i;
                if (mode->type & DRM_MODE_TYPE_PREFERRED) {
                    modeIndex = i;
                    modeX = (unsigned int)mode->hdisplay;
                    modeY = (unsigned int)mode->vdisplay;
                    break;
                }
                unsigned int size = (unsigned int)mode->hdisplay
                    * (unsigned int)mode->vdisplay;
                if (size > modeSize) {
                    modeIndex = i;
                    modeSize = size;
                    modeX = (unsigned int)mode->hdisplay;
                    modeY = (unsigned int)mode->vdisplay;
                }
            }
        }

        // Get the new mode if a mode is required to be set
        if (modeIndex >= 0) {
            newMode = conn->modes + modeIndex;
            // Mark CRTC as initialized and mapped to the connector
            nvGlDrmDev->connInfo[nvGlDrmDev->curConnIndx].crtcMapping = crtcIndex;
            nvGlDrmDev->crtcInfo[crtcIndex].modeX = modeX;
            nvGlDrmDev->crtcInfo[crtcIndex].modeY = modeY;
            nvGlDrmDev->crtcInfo[crtcIndex].mapped = true;

            // If a size wasn't specified, use the whole screen
            if (!sizeX || !sizeY) {
                assert(!sizeX && !sizeY && !offsetX && !offsetY);
                sizeX = nvGlDrmDev->crtcInfo[crtcIndex].modeX;
                sizeY = nvGlDrmDev->crtcInfo[crtcIndex].modeY;
                demoOptions.windowSize[0] = sizeX;
                demoOptions.windowSize[1] = sizeY;
            }

            // Set the mode for the CRTC if not sending atomic info to EGL
            if (!isEGLOutputDrmAtomicEXTSupported) {
                if (!nvGlDrmDev->isDrmNvdc) {
                    // upstream drm driver requires initial fb to be passed to setCrtc.
                    if (NvGlDemoCreateDumbBuffer(modeX, modeY, 32, false)) {
                        NvGlDemoLog("Could not create the initial dumb buffer\n");
                        goto NvGlDemoInitializeDrmOutputMode_fail;
                    }
                    // Assigned a supported color format
                    uint32_t colorFormat = DRM_FORMAT_ARGB8888;
                    uint32_t offset = 0;
                    // Frame buffer for CRTC
                    NvGLDemoBO *dumbBO = nvGlDrmDev->crtcInfo[nvGlDrmDev->currCrtcIndx].dumbBO[0];
                    // Create a frame buffer object for the dumb-buffer
                    if (pdrmModeAddFB2(nvGlDrmDev->fd, modeX, modeY, colorFormat,
                                       &(dumbBO->bo_handle), &(dumbBO->pitch), &offset,
                                       &initialFbId, 0) < 0) {
                        NvGlDemoLog("Unable to create dumb FB\n");
                        goto NvGlDemoInitializeDrmOutputMode_fail;
                    }
                } else {
                    initialFbId = -1;
                }
                if (pdrmModeSetCrtc(nvGlDrmDev->fd, nvGlDrmDev->res->crtcs[crtcIndex],
                                    initialFbId, 0, 0,
                                    &nvGlDrmDev->res->connectors[nvGlDrmDev->curConnIndx],
                                     1, conn->modes + modeIndex)) {
                    NvGlDemoLog("pdrmModeSetCrtc-fail setting crtc mode\n");
                    goto NvGlDemoInitializeDrmOutputMode_fail;
                }

                if ((currCrtc = (drmModeCrtcPtr) pdrmModeGetCrtc(nvGlDrmDev->fd,
                                nvGlDrmDev->res->crtcs[crtcIndex])) != NULL) {
                    NvGlDemoLog("Demo Mode: %d x %d @ %d\n",
                    currCrtc->mode.hdisplay, currCrtc->mode.vdisplay,
                    currCrtc->mode.vrefresh);
                } else {
                    NvGlDemoLog("Failed to get current mode.\n");
                }
            }
        }

        // Set the alpha and the plane configuration here if not sending atomic
        // info to EGL
        planeFbId = initialFbId;
        if (!isEGLOutputDrmAtomicEXTSupported) {
            drmModeAtomicReqPtr pAtomic;
            int ret = 0;

            if (renderingPlaneIndex != primaryPlaneIndex && !nvGlDrmDev->isDrmNvdc) {
                if (NvGlDemoCreateDumbBuffer(sizeX, sizeY, 32, true)) {
                     NvGlDemoLog("Could not create the initial dumb buffer\n");
                    goto NvGlDemoInitializeDrmOutputMode_fail;
                 }
                 // Assigned a supported color format
                 uint32_t colorFormat = DRM_FORMAT_ARGB8888;
                 uint32_t offset = 0;
                 // Frame buffer for CRTC
                 NvGLDemoBO *dumbBO = nvGlDrmDev->crtcInfo[nvGlDrmDev->currCrtcIndx].dumbBO[1];
                 // Create a frame buffer object for the dumb-buffer
                 if (pdrmModeAddFB2(nvGlDrmDev->fd, sizeX, sizeY, colorFormat,
                                    &(dumbBO->bo_handle), &(dumbBO->pitch), &offset,
                                    &planeFbId, 0) < 0) {
                     NvGlDemoLog("Unable to create dumb FB\n");
                     goto NvGlDemoInitializeDrmOutputMode_fail;
                 }
            }

            pAtomic = pdrmModeAtomicAlloc();
            if (pAtomic == NULL) {
                NvGlDemoLog("Failed to allocate the property set\n");
                goto NvGlDemoInitializeDrmOutputMode_fail;
            }

            // Set the alpha property of the plane
            if (NvGLDemoAtomicSetAlpha(nvGlDrmDev->fd, pAtomic,
                                       nvGlDrmDev->planes->planes[renderingPlaneIndex],
                                       alpha) != 0) {
                NvGlDemoLog("Could not set the alpha property\n");
            } else {

                ret = pdrmModeAtomicCommit(nvGlDrmDev->fd, pAtomic, 0, NULL /* user_data */);
            }
            pdrmModeAtomicFree(pAtomic);

            if (ret != 0) {
                NvGlDemoLog("Failed to commit properties. Error code: %d\n", ret);
                goto NvGlDemoInitializeDrmOutputMode_fail;
            }

            if (pdrmModeSetPlane(nvGlDrmDev->fd,
                                 nvGlDrmDev->planes->planes[renderingPlaneIndex],
                                 nvGlDrmDev->res->crtcs[crtcIndex], planeFbId, 0,
                                 offsetX, offsetY, sizeX, sizeY,
                                 0, 0, sizeX << 16, sizeY << 16)) {
                NvGlDemoLog("pdrmModeSetPlane-fail\n");
                goto NvGlDemoInitializeDrmOutputMode_fail;
            }
            nvGlDrmDev->currPlaneIndx = renderingPlaneIndex;
        } else {

            // Allocate a DRM atomics request structure
            drmModeAtomicReqPtr req = pdrmModeAtomicAlloc();

            if (req == NULL) {
                NvGlDemoLog("Unable to allocate drm atomics req structure\n");
                goto NvGlDemoInitializeDrmOutputMode_fail;
            }

            // Non-permissive (normal) DRM drivers require a FB to be set to
            // the primary plane for a mode set, and because an FB won't be
            // provided by EGLStream in the case of rendering != primary,
            // we have to create a dumb buffer for the primary plane.
            // A 1x1 buffer suffices for this.

            srcX = 0;
            crtcX = offsetX;
            srcY = 0;
            crtcY = offsetY;
            connId = nvGlDrmDev->res->connectors[nvGlDrmDev->curConnIndx];
            crtcId = nvGlDrmDev->res->crtcs[nvGlDrmDev->currCrtcIndx];

            if (renderingPlaneIndex != primaryPlaneIndex &&
                !nvGlDrmDev->isDrmNvdcPermissive) {

                // Configure the primary plane
                uint32_t primaryPlaneId = nvGlDrmDev->planes->planes[primaryPlaneIndex];
                // Create a dumb buffer to be set on the primary plane(CRTC)
                // of size modeX x modeY
                srcW = crtcW = modeX;
                srcH = crtcH = modeY;
                // The lowest bit depth that we support for the given color format
                // is 8 bits per pixel. We provide the smallest bit depth as we
                // want the size of the dumb buffer to be as small as possible
                uint32_t bitDepth = 8;
                if (NvGlDemoCreateDumbBuffer(srcW, srcH, bitDepth, false)) {
                    NvGlDemoLog("Could not create the initial dumb buffer\n");
                    goto NvGlDemoInitializeDrmOutputMode_fail;
                }
                // Assigned a supported color format
                uint32_t colorFormat = DRM_FORMAT_ARGB8888;
                uint32_t offset = 0;
                // Frame buffer for CRTC
                uint32_t fbId = 0;
                NvGLDemoBO *dumbBO = nvGlDrmDev->crtcInfo[nvGlDrmDev->currCrtcIndx].dumbBO[0];
                // Create a frame buffer object for the dumb-buffer
                if (pdrmModeAddFB2(nvGlDrmDev->fd, srcW, srcH, colorFormat,
                                   &(dumbBO->bo_handle), &(dumbBO->pitch), &offset,
                                   &fbId, 0) < 0) {
                    NvGlDemoLog("Unable to create dumb FB\n");
                    goto NvGlDemoInitializeDrmOutputMode_fail;
                }
                // Add the dumb buffer to the atomics request structure
                if (NvGLDemoAtomicAddFrameBuffer(nvGlDrmDev->fd, req, connId, crtcId,
                                                 primaryPlaneId, fbId) != 0 ) {
                    NvGlDemoLog("Could not set the atomic frame buffer property\n");
                    NvGlDemoDestroyDumbBuffer();
                    goto NvGlDemoInitializeDrmOutputMode_fail;
                }

                // Set plane geometry for the primary plane
                if (NvGLDemoAtomicPlaneResize(nvGlDrmDev->fd, req, primaryPlaneId,
                                              crtcId, crtcX, crtcY, crtcW, crtcH,
                                              srcX << 16, srcY << 16,
                                              srcW << 16, srcH << 16) != 0) {
                    NvGlDemoLog("Could not resize the atomic plane properties\n");
                    goto NvGlDemoInitializeDrmOutputMode_fail;
                }
            }

            // Set the mode information if there is a need of a mode set
            if (newMode) {
                if (pdrmModeCreatePropertyBlob(nvGlDrmDev->fd, newMode,
                                               sizeof(*newMode),
                                               &demoState.platform->modeId) != 0) {
                    NvGlDemoLog("Unable to atomic create mode property blob\n");
                    goto NvGlDemoInitializeDrmOutputMode_fail;
                }
                if (demoState.platform->modeId == 0) {
                    goto NvGlDemoInitializeDrmOutputMode_fail;
                }
                if (NvGLDemoAtomicAddMode(nvGlDrmDev->fd, req,
                                          demoState.platform->modeId, connId,
                                          crtcId) != 0) {
                    NvGlDemoLog("Could not set the atomic mode properties for"
                                " the rendering plane\n");
                    goto NvGlDemoInitializeDrmOutputMode_fail;
                }
            }

            // Configure the rendering plane
            srcW = crtcW = sizeX;
            srcH = crtcH = sizeY;
            renderingPlaneId = nvGlDrmDev->planes->planes[renderingPlaneIndex];

            if (NvGLDemoAtomicPlaneResize(nvGlDrmDev->fd, req, renderingPlaneId,
                                          crtcId, crtcX, crtcY, crtcW, crtcH,
                                          srcX << 16, srcY << 16, srcW << 16,
                                          srcH << 16) != 0) {
                NvGlDemoLog("Could not resize the atomic plane properties\n");
                goto NvGlDemoInitializeDrmOutputMode_fail;
            }

            // Set the alpha property of the plane
            if (NvGLDemoAtomicSetAlpha(nvGlDrmDev->fd, req, renderingPlaneId,
                                       alpha) != 0) {
                // Setting the alpha property of the main plane fails on NvKms. Log the error
                // and continue.
                NvGlDemoLog("Could not set the alpha property\n");
            }

            // Set the common atomic request structure in the NvGLDemo DRM device
            // to send to EGL
            demoState.platform->pAtomicReq = req;
            // Set the rendering plane index as the current plane index
            nvGlDrmDev->currPlaneIndx = renderingPlaneIndex;
        }
    }

    outDev = &nvGlOutDevLst[demoState.platform->curDevIndx];

    outDev->layerDefault = nvGlDrmDev->planeInfo[renderingPlaneIndex].layer;
    nvGlDrmDev->planeInfo[renderingPlaneIndex].used = true;

    if (conn) {
        pdrmModeFreeConnector(conn);
    }

    if (currCrtc) {
        pdrmModeFreeCrtc(currCrtc);
    }

    return true;

NvGlDemoInitializeDrmOutputMode_fail:

    if (conn != NULL) {
        NvGlDemoLog("List of available display modes\n");
        for (i=0; i<conn->count_modes; ++i) {
            drmModeModeInfoPtr mode = conn->modes + i;
            NvGlDemoLog("%d x %d @ %d\n", mode->hdisplay, mode->vdisplay, mode->vrefresh);
        }
    }

    // Clean up and return
    if (conn) {
        pdrmModeFreeConnector(conn);
    }

    if (currCrtc) {
        pdrmModeFreeCrtc(currCrtc);
    }

    return false;

}

// Reset DRM Device
static void NvGlDemoResetDrmDevice(void)
{
    if(nvGlDrmDev)
    {
        nvGlDrmDev->fd = 0;
        nvGlDrmDev->res = NULL;
        nvGlDrmDev->planes = NULL;
        nvGlDrmDev->connDefault = -1;
        nvGlDrmDev->curConnIndx = -1;
        nvGlDrmDev->currCrtcIndx = -1;
        nvGlDrmDev->currPlaneIndx = -1;
        nvGlDrmDev->connInfo = NULL;
        nvGlDrmDev->crtcInfo = NULL;
        nvGlDrmDev->planeInfo = NULL;
    }
    return;
}

// Reset DRM sub driver connection status
static void NvGlDemoResetDrmConnection(void)
{
    int offsetX = 0;
    int offsetY = 0;
    unsigned int sizeX = 0;
    unsigned int sizeY = 0;

    offsetX = demoOptions.windowOffset[0];
    offsetY = demoOptions.windowOffset[1];
    sizeX = demoOptions.windowSize[0];
    sizeY = demoOptions.windowSize[1];


    // Currently as universal planes is enabled by default for every
    // primary plane rendering that we do we execute a SetPlane(0) so that
    // we do not miss any CRTC de-init calls and do not divert from the old behavior

    if(nvGlDrmDev && demoState.platform ) {
        if((nvGlDrmDev->connInfo) && (nvGlDrmDev->curConnIndx != -1)) {
            nvGlDrmDev->connInfo[nvGlDrmDev->curConnIndx].crtcMapping = -1;
        }
        // Mark plane as unused
        if((nvGlDrmDev->planeInfo) && (nvGlDrmDev->currPlaneIndx != -1) &&
           (nvGlDrmDev->currCrtcIndx != -1)) {
            if (pdrmModeSetPlane(nvGlDrmDev->fd,
                                 nvGlDrmDev->planes->planes[nvGlDrmDev->currPlaneIndx],
                                 nvGlDrmDev->res->crtcs[nvGlDrmDev->currCrtcIndx],
                                 0, 0, offsetX, offsetY, sizeX, sizeY, 0, 0,
                                 sizeX << 16, sizeY << 16)) {
                NvGlDemoLog("pdrmModeSetPlane-fail\n");
            }
            nvGlDrmDev->planeInfo[nvGlDrmDev->currPlaneIndx].used = false;
        }
        demoState.platform->curConnIndx = 0;
    }
    return;
}

// Terminate Drm Device
static void NvGlDemoTermDrmDevice(void)
{
    if(nvGlDrmDev) {

        if(nvGlDrmDev->connInfo) {
            FREE(nvGlDrmDev->connInfo);
            nvGlDrmDev->connInfo = NULL;
        }

        // Free the atomic req if still allocated
        if (demoState.platform->pAtomicReq) {
            pdrmModeAtomicFree(demoState.platform->pAtomicReq);
            demoState.platform->pAtomicReq = NULL;
        }

        if (demoState.platform->modeId) {
            pdrmModeDestroyPropertyBlob(nvGlDrmDev->fd,
                                        demoState.platform->modeId);
            demoState.platform->modeId = 0;
        }

        // Destroy the initial dumb buffer
        NvGlDemoDestroyDumbBuffer();

        if(nvGlDrmDev->crtcInfo) {
            FREE(nvGlDrmDev->crtcInfo);
            nvGlDrmDev->crtcInfo = NULL;
        }

        if(nvGlDrmDev->planeInfo) {
            FREE(nvGlDrmDev->planeInfo);
            nvGlDrmDev->planeInfo = NULL;
        }
        if (nvGlDrmDev->planes) {
            pdrmModeFreePlaneResources(nvGlDrmDev->planes);
        }
        if (nvGlDrmDev->res) {
            pdrmModeFreeResources(nvGlDrmDev->res);
        }

        if ((demoOptions.allocator == NvGlDemoAllocator_GBM) && (nvGlDrmDev->gbmDev)) {
            pgbm_device_destroy(nvGlDrmDev->gbmDev);
        }

        if (pdrmClose(nvGlDrmDev->fd)) {
            NvGlDemoLog("drmClose failed\n");
        }
#if !defined(__INTEGRITY)
        if(libDRM) {
            dlclose(libDRM);
            libDRM = NULL;
        }
#endif
        FREE(nvGlDrmDev);
        NvGlDemoResetDrmDeviceFnPtr();
        nvGlDrmDev = NULL;
    }
    return;
}

// Reset all Drm Device Function ptr
static void NvGlDemoResetDrmDeviceFnPtr(void)
{
    pdrmOpen = NULL;
    pdrmClose = NULL;
    pdrmModeGetResources = NULL;
    pdrmModeFreeResources = NULL;
    pdrmModeGetPlaneResources = NULL;
    pdrmModeFreePlaneResources = NULL;
    pdrmModeGetConnector = NULL;
    pdrmModeFreeConnector = NULL;
    pdrmModeGetEncoder = NULL;
    pdrmModeFreeEncoder = NULL;
    pdrmModeGetPlane = NULL;
    pdrmModeFreePlane = NULL;
    pdrmModeSetCrtc = NULL;
    pdrmModeGetCrtc = NULL;
    pdrmModeSetPlane = NULL;
    pdrmModeFreeCrtc = NULL;
    pdrmModeAtomicAlloc = NULL;
    pdrmModeAtomicAddProperty = NULL;
    pdrmModeAtomicCommit = NULL;
    pdrmModeAtomicFree = NULL;
    pdrmModeObjectGetProperties = NULL;
    pdrmModeGetProperty = NULL;
    pdrmModeFreeProperty = NULL;
    pdrmModeFreeObjectProperties = NULL;
    pdrmSetClientCap = NULL;
    pdrmIoctl = NULL;
    pdrmModeAddFB2 = NULL;
    pdrmModeCreatePropertyBlob = NULL;
    pdrmModeDestroyPropertyBlob = NULL;
    pdrmGetVersion = NULL;
    pdrmFreeVersion = NULL;
}
#endif //__QNX__

//======================================================================
// Nvgldemo Display functions
//======================================================================

// Initialize access to the display system
int NvGlDemoDisplayInit(void)
{
    // Only try once
    if (isOutputInitDone) {
        return 1;
    }

// If display option is specified, but isn't supported, then exit.

    if (demoOptions.displayName[0]) {
        NvGlDemoLog("Setting display output is not supported. Exiting.\n");
        goto NvGlDemoDisplayInit_fail;
    }
    if ((demoOptions.displayBlend >= NvGlDemoDisplayBlend_None) ||
        (demoOptions.displayColorKey[0] >= 0.0)) {
        NvGlDemoLog("blending are not supported. Exiting.\n");
        goto NvGlDemoDisplayInit_fail;
    }

    // Allocate a structure for the platform-specific state
    demoState.platform =
        (NvGlDemoPlatformState*)MALLOC(sizeof(NvGlDemoPlatformState));
    if (!demoState.platform) {
        NvGlDemoLog("Could not allocate platform specific storage memory.\n");
        goto NvGlDemoDisplayInit_fail;
    }

    demoState.platform->curDevIndx = 0;
    demoState.platform->curConnIndx = demoOptions.displayNumber;

    sigint.sa_handler = signal_int;
    sigemptyset(&sigint.sa_mask);
    sigint.sa_flags = SA_RESETHAND;
    sigaction(SIGINT, &sigint, NULL);

    if (NvGlDemoInitEglDevice()) {
#if !defined(__QNX__)
        if (NvGlDemoInitDrmDevice())
#else
        if (NvGlDemoInitWfdDevice())
#endif//!__QNX__
        {
            // DRM/WFD Output functions are available
            isOutputInitDone = true;
            demoState.platformType = NvGlDemoInterface_Device;
            // Success
            return 1;
        }
    }

NvGlDemoDisplayInit_fail:

    NvGlDemoResetModule();

    return 0;
}

// Terminate access to the display system
void NvGlDemoDisplayTerm(void)
{
    if (!isOutputInitDone) {
        NvGlDemoLog("Display_init not yet done[%d].\n",isOutputInitDone);
        return;
    }
    // End Device Setup
#if !defined (__QNX__)
    NvGlDemoTermDrmDevice();
#else
    NvGlDemoTermWfdDevice();
#endif//!__QNX__
    NvGlDemoTermEglDevice();
    // Reset Module
    NvGlDemoResetModule();
}

static void NvGlDemoResetModule(void)
{
    if (demoState.platform) {
        FREE(demoState.platform);
    }

    if (nvGlOutDevLst) {
        FREE(nvGlOutDevLst);
    }

    if (devList) {
        FREE(devList);
    }

#if !defined(__QNX__)
    if(nvGlDrmDev) {
        FREE(nvGlDrmDev);
        nvGlDrmDev = NULL;
    }

#if !defined(__INTEGRITY)
    if(libDRM) {
        dlclose(libDRM);
        libDRM = NULL;
    }
#endif //!__INTEGRITY
    NvGlDemoResetDrmDeviceFnPtr();
#else
    if (nvGlWfdDev) {
        FREE(nvGlWfdDev);
        nvGlWfdDev = NULL;
    }

#if !defined(__INTEGRITY)
    if (libWFD) {
        dlclose(libWFD);
        libWFD = NULL;
    }
#endif//!__INTEGRITY
    NvGlDemoResetWfdDeviceFnPtr();
#endif//!__QNX__

    demoState.platform = NULL;
    demoState.platformType = NvGlDemoInterface_Unknown;
    demoState.nativeDisplay = EGL_NO_DISPLAY;
    nvGlOutDevLst = NULL;
    devList = NULL;

    // Reset module global variables
    isOutputInitDone = false;
    devCount = 0;

    NvGlDemoResetEglDeviceFnPtr();
    return;
}

//======================================================================
// Nvgldemo Window functions
//======================================================================

// Window creation
int NvGlDemoWindowInit(
        int* argc, char** argv,
        const char* appName)
{
    if (!isOutputInitDone) {
        NvGlDemoLog("Display_init not yet done[%d].\n",isOutputInitDone);
        return 0;
    }
    // Create the EGL Device and DRM Device
    if (NvGlDemoCreateEglDevice(demoState.platform->curDevIndx)){
#if !defined(__QNX__)
        if(!NvGlDemoGetDrmDevice(demoState.platform->curDevIndx)){
            goto NvGlDemoWindowInit_fail;
        }
#else
        if (!NvGlDemoGetWfdDevice(demoState.platform->curDevIndx)){
            goto NvGlDemoWindowInit_fail;
        }
#endif//!__QNX__
    }

    // Make the Output requirement for Devices
#if !defined(__QNX__)
    if (NvGlDemoInitializeDrmOutputMode()){
        if(!NvGlDemoCreateSurfaceBuffer()){
            goto NvGlDemoWindowInit_fail;
        }
    } else {
        goto NvGlDemoWindowInit_fail;
    }
#else
    if (NvGlDemoInitializeWfdOutputMode()) {
        if (!NvGlDemoCreateSurfaceBuffer()) {
            goto NvGlDemoWindowInit_fail;
        }
    } else {
        goto NvGlDemoWindowInit_fail;
    }
#endif//!__QNX__
    return 1;

NvGlDemoWindowInit_fail:

    // Clean up and return
    NvGlDemoWindowTerm();
    NvGlDemoResetModule();
    return 0;
}

// Close the window
void NvGlDemoWindowTerm(void)
{
    if (!isOutputInitDone) {
        NvGlDemoLog("Display_init not yet done[%d].\n",isOutputInitDone);
        return;
    }

#if !defined(__QNX__)
    NvGlDemoResetDrmConnection();
#endif

    NvGlDemoTermWinSurface();
    return;
}

// Allocates and populates the display attribs that the egldevice platform needs
// to use to provide to eglGetPlatformDisplayEXT to get the Native display handle
// Returns 0: success
//         -1: failure
int NvGlDemoGetDisplayAttribs(EGLint** displayAttribs)
{
    EGLint attrCount = 3;
    EGLint* pDisplayAttrib = NULL;

    if (displayAttribs == NULL) {
        return -1;
    }

    *displayAttribs = (EGLint*) MALLOC(sizeof(EGLint) * attrCount);
    if (*displayAttribs == NULL) {
        NvGlDemoLog("Could not allocate display attributes");
        return -1;
    }
    pDisplayAttrib = *displayAttribs;
    attrCount = 0;

#if !defined(__QNX__)
    pDisplayAttrib[attrCount++] = EGL_DRM_MASTER_FD_EXT;
    pDisplayAttrib[attrCount++] = nvGlDrmDev->fd;
#else
    pDisplayAttrib[attrCount++] = EGL_OPENWF_DEVICE_EXT;
    pDisplayAttrib[attrCount++] = nvGlWfdDev->wfdDeviceHandle;
#endif//!__QNX___
    pDisplayAttrib[attrCount] = EGL_NONE;

    return 0;
}

//
// TODO: Pixmap support
//

EGLNativePixmapType NvGlDemoPixmapCreate(
        unsigned int width,
        unsigned int height,
        unsigned int depth)
{
    NvGlDemoLog("EGLDevice pixmap functions not supported\n");
    return (EGLNativePixmapType)0;
}

void NvGlDemoPixmapDelete(
        EGLNativePixmapType pixmap)
{
    NvGlDemoLog("EGLDevice pixmap functions not supported\n");
}

//======================================================================
// Nvgldemo Event Function
//======================================================================

//
// TODO: Callback handling
//
static NvGlDemoCloseCB   closeCB   = NULL;
static NvGlDemoResizeCB  resizeCB  = NULL;
static NvGlDemoKeyCB     keyCB     = NULL;
static NvGlDemoPointerCB pointerCB = NULL;
static NvGlDemoButtonCB  buttonCB  = NULL;

void NvGlDemoSetCloseCB(NvGlDemoCloseCB cb)     { closeCB   = cb; }
void NvGlDemoSetResizeCB(NvGlDemoResizeCB cb)   { resizeCB  = cb; }
void NvGlDemoSetKeyCB(NvGlDemoKeyCB cb)         { keyCB     = cb; }
void NvGlDemoSetPointerCB(NvGlDemoPointerCB cb) { pointerCB = cb; }
void NvGlDemoSetButtonCB(NvGlDemoButtonCB cb)   { buttonCB  = cb; }

void NvGlDemoCheckEvents(void)
{
    //TODO: implementation
    //
}

void NvGlDemoWaitEvents(void)
{
}

static void signal_int(int signum)
{
    if(closeCB)
        closeCB();
}

EGLBoolean NvGlDemoSwapInterval(EGLDisplay dpy, EGLint interval)
{
    struct NvGlOutputDevice *outDev = NULL;
    EGLAttrib swapInterval = 1;
    char *configStr = NULL;

    if ((!nvGlOutDevLst) || (!demoState.platform) ||
        (demoState.platform->curDevIndx >= devCount) ||
        (nvGlOutDevLst[demoState.platform->curDevIndx].enflag == false) ||
        (!peglOutputLayerAttribEXT)) {
        return EGL_FALSE;
    }
    outDev = &nvGlOutDevLst[demoState.platform->curDevIndx];
    // Fail if no layers available
    if ((!outDev) || (outDev->layerUsed > outDev->layerCount) || (!outDev->windowList) ||
        (!outDev->layerList)) {
        return EGL_FALSE;
    }

    if ((outDev->layerIndex >= outDev->layerCount)) {
        NvGlDemoLog("NvGlDemoSwapInterval_fail[Layer -%d]\n",outDev->layerIndex);
        return EGL_FALSE;
    }

    // To allow the interval to be overridden by an environment variable exactly the same way like a normal window system.
    configStr = getenv("__GL_SYNC_TO_VBLANK");

    if (!configStr)
        configStr = getenv("NV_SWAPINTERVAL");

    // Environment variable is higher priority than runtime setting
    if (configStr) {
        swapInterval = (EGLAttrib)strtol(configStr, NULL, 10);
    } else {
        swapInterval = (EGLint)interval;
    }

    if (!peglOutputLayerAttribEXT(outDev->eglDpy,
            outDev->layerList[outDev->layerIndex],
            EGL_SWAP_INTERVAL_EXT,
            swapInterval)) {
        NvGlDemoLog("peglOutputLayerAttribEXT_fail[%d %d]\n",outDev->layerList[outDev->layerIndex],swapInterval);
        return EGL_FALSE;
    }

    return EGL_TRUE;
}

#if defined(__QNX__)
void NvGlDemoSetDisplayAlpha(float alpha)
{
    return;
}
#else
void NvGlDemoSetDisplayAlpha(float alpha)
{
    unsigned int newAlpha = (unsigned int)(alpha * 255);
    if (newAlpha > 255) {
        NvGlDemoLog("Alpha value specified for constant blending in not in specified range [0,1]. Using alpha 1.0\n");
        newAlpha = 255;
    }

    drmModeAtomicReqPtr pAtomic;
    const uint32_t flags = DRM_MODE_ATOMIC_NONBLOCK;

    pAtomic = pdrmModeAtomicAlloc();
    if (pAtomic == NULL) {
        NvGlDemoLog("Failed to allocate the property set\n");
        return;
    }

    pdrmModeAtomicAddProperty(pAtomic, nvGlDrmDev->planes->planes[nvGlDrmDev->currPlaneIndx],
                              nvGlDrmDev->currPlaneAlphaPropID, newAlpha);

    int ret = pdrmModeAtomicCommit(nvGlDrmDev->fd, pAtomic, flags, NULL /* user_data */);

    pdrmModeAtomicFree(pAtomic);

    if (ret != 0) {
        NvGlDemoLog("Failed to commit properties. Error code: %d\n", ret);
    }
}
#endif //__QNX__

#endif // NVGLDEMO_HAS_DEVICE
