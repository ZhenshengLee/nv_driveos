/* Copyright (c) 2014-2022 NVIDIA Corporation.  All rights reserved.
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

#ifndef __NVGLDEMO_WIN_EGLDEVICE_H
#define __NVGLDEMO_WIN_EGLDEVICE_H

#ifdef NVGLDEMO_HAS_DEVICE

#include <EGL/egl.h>
#include <EGL/eglext.h>

#if !defined(__QNX__)
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <drm_fourcc.h>
#include "tegra_drm.h"
#include <gbm.h>

#else
#include "WF/wfd.h"
#include "WF/wfdext.h"
#endif//!__QNX__

#if defined(__INTEGRITY)
#include <stdbool.h>
#else
typedef enum {
    false=0,
    true=1
} bool;
#endif

#if !defined(__QNX__)
#ifndef DRM_CLIENT_CAP_DRM_NVDC_PERMISSIVE
#define DRM_CLIENT_CAP_DRM_NVDC_PERMISSIVE 6
#endif

#ifndef DRM_PLANE_TYPE_OVERLAY
#define DRM_PLANE_TYPE_OVERLAY 0
#endif

#ifndef DRM_PLANE_TYPE_PRIMARY
#define DRM_PLANE_TYPE_PRIMARY 1
#endif

#ifndef DRM_PLANE_TYPE_CURSOR
#define DRM_PLANE_TYPE_CURSOR  2
#endif
#endif //!__QNX__

// Platform-specific state info
struct NvGlDemoPlatformState
{
    // Input - Device Instance index
    int                 curDevIndx;
    // Input - Connector Index
    int                 curConnIndx;
#if !defined(__QNX__)
    // drm Mode Atomic Request Struct
    drmModeAtomicReqPtr pAtomicReq;
    // drm Mode Atomic Property Blob
    uint32_t            modeId;
#endif //!__QNX__
};

// EGLOutputLayer window List
struct NvGlDemoWindowDevice
{
    bool enflag;
    EGLint                  index;
    EGLStreamKHR            stream;
    EGLSurface              surface;
};

// EGLOutputDevice
struct NvGlOutputDevice
{
    bool                             enflag;
    EGLint                           index;
    EGLDeviceEXT                     device;
    EGLDisplay                       eglDpy;
    EGLint                           layerCount;
    EGLint                           layerDefault;
    EGLint                           layerIndex;
    EGLint                           layerUsed;
    EGLOutputLayerEXT*               layerList;
    struct NvGlDemoWindowDevice*     windowList;
};

#if !defined(__QNX__)
// Parsed DRM info structures
typedef struct {
    bool             valid;
    unsigned int     crtcMask;
    int              crtcMapping;
} NvGlDemoDRMConn;

// A structure to hold the dumb buffer
typedef struct {
    uint32_t    bo_handle;
    uint32_t    width;
    uint32_t    height;
    uint32_t    pitch;
    uint8_t*    data;
    size_t      size;
    struct gbm_bo *gbmBo;
} NvGLDemoBO;

typedef struct {
    unsigned int modeX;
    unsigned int modeY;
    bool         mapped;
    NvGLDemoBO   *dumbBO[2];
    bool         dumbBufferCreated;
} NvGlDemoDRMCrtc;

typedef struct {
    EGLint           layer;
    unsigned int     crtcMask;
    bool             used;
    int              planeType;
} NvGlDemoDRMPlane;

// DRM+EGLDesktop desktop class
struct NvGlDemoDRMDevice
{
    int                 fd;
    drmModeRes*         res;
    drmModePlaneRes*    planes;

    int                 connDefault;
    int                 curConnIndx;
    int                 currCrtcIndx;
    int                 currPlaneIndx;
    bool                isDrmNvdc;
    bool                isDrmNvdcPermissive;
    unsigned int        currPlaneAlphaPropID;

    NvGlDemoDRMConn*    connInfo;
    NvGlDemoDRMCrtc*    crtcInfo;
    NvGlDemoDRMPlane*   planeInfo;
    struct gbm_device  *gbmDev;
};

struct PropertyIDAddress {
    const char*  name;
    uint32_t*    ptr;
};

#else

struct NvGlDemoWFDPort
{
    WFDint portId;
    WFDint portHandle;
    WFDint numPortModes;
    WFDPortMode *portModes;
};

struct NvGlDemoWFDPipe
{
    WFDint pipeId;
    WFDint pipeHandle;
    WFDint layer;
};

// WFD+EGLDesktop desktop class
struct NvGlDemoWFDDevice
{
    WFDDevice wfdDeviceHandle;
    WFDint numPorts;
    WFDint numPipes;
    struct NvGlDemoWFDPort *portInfo;
    struct NvGlDemoWFDPipe *pipeInfo;
    WFDint curPortIndex;
    WFDint curPipeIndex;
};
#endif //!__QNX__

// EGL Device internal api
static bool NvGlDemoInitEglDevice(void);
static bool NvGlDemoCreateEglDevice(EGLint devIndx);
static bool NvGlDemoCreateSurfaceBuffer(void);
static void NvGlDemoResetEglDeviceLyrLst(struct NvGlOutputDevice *devOut);
static void NvGlDemoResetEglDevice(void);
static void NvGlDemoTermWinSurface(void);
static void NvGlDemoTermEglDevice(void);
static void NvGlDemoResetEglDeviceFnPtr(void);

#if !defined(__QNX__)
// DRM Device internal api
static bool NvGlDemoInitDrmDevice(void);
static bool NvGlDemoGetDrmDevice( EGLint devIndx );
static bool NvGlDemoInitializeDrmOutputMode( void );
static void NvGlDemoResetDrmDevice(void);
static void NvGlDemoResetDrmConnection(void);
static void NvGlDemoTermDrmDevice(void);
static void NvGlDemoResetDrmDeviceFnPtr(void);
#else
//WFD device internal api
static void NvGlDemoResetWfdDevice(void);
static void NvGlDemoTermWfdDevice(void);
static bool NvGlDemoInitializeWfdOutputMode(void);
static bool NvGlDemoGetWfdDevice(EGLint devIndx);
static void NvGlDemoResetWfdDeviceFnPtr(void);
static bool NvGlDemoInitWfdDevice(void);
#endif //!__QNX__

// Module internal api
static void NvGlDemoResetModule(void);

#endif // NVGLDEMO_HAS_DEVICE

#endif // __NVGLDEMO_WIN_EGLDEVICE_H


