/*
 * nvgldemo_main.c
 *
 * Copyright (c) 2007-2019, NVIDIA CORPORATION. All rights reserved.
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

//
// This file illustrates how to set up the main rendering context and
//   surface for a GL application.
//

#include "nvgldemo.h"
#include <GLES2/gl2.h>
#include <unistd.h>

// Global demo state
NvGlDemoState demoState = {
    (NativeDisplayType)0,  // nativeDisplay
    (NativeWindowType)0,   // nativeWindow
    EGL_NO_DISPLAY,        // display
    EGL_NO_STREAM_KHR,     // stream
    EGL_NO_SURFACE,        // surface
    (EGLConfig)0,          // config
    EGL_NO_CONTEXT,        // context
    0,                     // width
    0,                     // height
    NvGlDemoInterface_Unknown,  // platform Type
    NULL                   // platform
};

// Used with cross-p mode.
// The user has to set these. Not sure if there is a better place to put this.
int g_ServerID = -1;
int g_ClientID = -1;
int g_ServerAllocatedID = -1;

char const *producerMsg = "Hi, This is a sample producer";

// EGL Device specific variable
static EGLint        devCount = 0;
static EGLDeviceEXT* devList = NULL;
static PFNEGLQUERYDEVICESEXTPROC    peglQueryDevicesEXT = NULL;

// Maximum number of attributes for EGL calls
#define MAX_ATTRIB 31

// Start up, parsing nvgldemo options and initializing window system and EGL
int
NvGlDemoInitialize(
    int* argc, char** argv,
    const char *appName,
    int glversion, int depthbits, int stencilbits)
{
    // Parse the nvgldemo command line options
    if (!NvGlDemoArgParse(argc, argv)) return 0;

    // Do the startup using the parsed options
    return NvGlDemoInitializeParsed(argc, argv, appName,
                                    glversion, depthbits, stencilbits);
}

EGLBoolean NvGlDemoPrepareStreamToAttachProducer(void)
{
    EGLBoolean eglStatus = EGL_FALSE;
    EGLint streamState = EGL_STREAM_STATE_EMPTY_KHR;

#ifndef ANDROID
    PFNEGLQUERYSTREAMKHRPROC peglQueryStreamKHR = NULL;
    NVGLDEMO_EGL_GET_PROC_ADDR(eglQueryStreamKHR, fail, PFNEGLQUERYSTREAMKHRPROC);
#endif

    // Wait for the consumer to connect to the stream or for failure
    do {
#ifndef ANDROID
        eglStatus = peglQueryStreamKHR(demoState.display, demoState.stream,
                                       EGL_STREAM_STATE_KHR, &streamState);
#else
        eglStatus = eglQueryStreamKHR(demoState.display, demoState.stream,
                                      EGL_STREAM_STATE_KHR, &streamState);
#endif
        if (!eglStatus) {
            NvGlDemoLog("Producer : Could not query EGL stream state\n");
            goto fail;
        }
    } while ((streamState == EGL_STREAM_STATE_INITIALIZING_NV) ||
              (streamState == EGL_STREAM_STATE_CREATED_KHR));

   // Should now be in CONNECTING state
    if (streamState != EGL_STREAM_STATE_CONNECTING_KHR) {
        NvGlDemoLog("Producer: Stream in bad state\n");
        goto fail;
    }

    return EGL_TRUE;

fail:
    return EGL_FALSE;
}

EGLBoolean NvGlDemoCreateCrossPartitionEGLStream(void)
{
    EGLint attr[MAX_ATTRIB * 2 + 1];
    int attrIdx        = 0;
    EGLint streamState = EGL_STREAM_STATE_EMPTY_KHR;
    EGLBoolean ret     = EGL_FALSE;

    int producer = strncmp(demoOptions.proctype, "consumer", NVGLDEMO_MAX_NAME);

#ifndef ANDROID
    PFNEGLCREATESTREAMKHRPROC peglCreateStreamKHR = NULL;
    NVGLDEMO_EGL_GET_PROC_ADDR(eglCreateStreamKHR, fail, PFNEGLCREATESTREAMKHRPROC);

    PFNEGLQUERYSTREAMKHRPROC peglQueryStreamKHR = NULL;
    NVGLDEMO_EGL_GET_PROC_ADDR(eglQueryStreamKHR, fail, PFNEGLQUERYSTREAMKHRPROC);
#endif

    if (demoOptions.nFifo > 0)
    {
        attr[attrIdx++] = EGL_STREAM_FIFO_LENGTH_KHR;
        attr[attrIdx++] = demoOptions.nFifo;
    }

    // Create stream with the right attributes.
    {
        attr[attrIdx++] = EGL_CONSUMER_LATENCY_USEC_KHR;
        attr[attrIdx++] = demoOptions.latency;

        if (demoOptions.flags & NVGL_DEMO_OPTION_TIMEOUT) {
            attr[attrIdx++] = EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR;
            attr[attrIdx++] = demoOptions.timeout;
        }

        attr[attrIdx++] = EGL_STREAM_TYPE_NV;
        if (!strncmp(demoOptions.ipAddr, "127.0.0.1", NVGLDEMO_MAX_NAME)) {
            attr[attrIdx++] = EGL_STREAM_CROSS_PROCESS_NV;
        } else {
            attr[attrIdx++] = EGL_STREAM_CROSS_PARTITION_NV;
        }

        attr[attrIdx++] = EGL_STREAM_ENDPOINT_NV;
        attr[attrIdx++] = producer ? EGL_STREAM_PRODUCER_NV : EGL_STREAM_CONSUMER_NV;

        attr[attrIdx++] = EGL_STREAM_PROTOCOL_NV;
        attr[attrIdx++] = EGL_STREAM_PROTOCOL_SOCKET_NV;

        attr[attrIdx++] = EGL_SOCKET_HANDLE_NV;
        attr[attrIdx++] = producer ? g_ClientID : g_ServerID;

        attr[attrIdx++] = EGL_SOCKET_TYPE_NV;
        attr[attrIdx++] = EGL_SOCKET_TYPE_INET_NV;
        attr[attrIdx++] = EGL_NONE;
    }

#ifndef ANDROID
    demoState.stream = peglCreateStreamKHR(demoState.display, attr);
#else
    demoState.stream = eglCreateStreamKHR(demoState.display, attr);
#endif

    do {
        //pthread_yield();
#ifndef ANDROID
        ret = peglQueryStreamKHR(demoState.display, demoState.stream, EGL_STREAM_STATE_KHR, &streamState);
#else
        ret = eglQueryStreamKHR(demoState.display, demoState.stream, EGL_STREAM_STATE_KHR, &streamState);
#endif
        if (!ret) {
            NvGlDemoLog("Could not query EGL stream state\n");
            goto fail;
        }
    } while (streamState == EGL_STREAM_STATE_INITIALIZING_NV);

    if (!(streamState == EGL_STREAM_STATE_CREATED_KHR) &&
        !((streamState == EGL_STREAM_STATE_CONNECTING_KHR) && producer)) {
        NvGlDemoLog("EGL stream is not in valid starting state\n");
        goto fail;
    }

    return ret;

fail:
    NvGlDemoLog("\nError while creating EGL stream!\n");
    NvGlDemoLog("\nEGL error code: 0x%x\n", eglGetError());
    return ret;
}

int NvGlDemoInitConsumerProcess(void)
{
    int serverID;
    int err = 0;

    if (demoOptions.isSmart != 1) {
        serverID = NvGlDemoCreateSocket();
        if (serverID != -1) {
            NvGlDemoServerBind(serverID);
            NvGlDemoServerListen(serverID);
        }
    } else {
        serverID = g_ServerAllocatedID;
    }

    if (serverID != -1) {
        int clientsockfd = NvGlDemoServerAccept(serverID);
        if (clientsockfd != -1) {
            g_ServerID = clientsockfd;
            NvGlDemoServerReceive(clientsockfd);
            err = 1;
        }
        close(serverID);
        g_ServerAllocatedID = -1;
    }

    return err;
}

// Extension checking utility
static int NvGlDemoCheckExtension(const char *exts, const char *ext)
{
    int extLen = (int)strlen(ext);
    const char *end = exts + strlen(exts);

    while (exts < end) {
        while (*exts == ' ') {
            exts++;
        }
        int n = strcspn(exts, " ");
        if ((extLen == n) && (strncmp(ext, exts, n) == 0)) {
            return 1;
        }
        exts += n;
    }
    return 0;
}

int NvGlDemoInitProducerProcess(void)
{
    int err = 0;

    NvGlDemoLog("Producer connecting to %s", demoOptions.ipAddr);

    int clientID = NvGlDemoCreateSocket();

    if (clientID != -1) {
        g_ClientID = clientID;

        NvGlDemoClientConnect(demoOptions.ipAddr, g_ClientID);
        NvGlDemoClientSend(producerMsg, g_ClientID);
        err = 1;
    }

    return err;
}

static void NvGlDemoTermEglDeviceExt(void)
{
    if (devList) {
        FREE(devList);
    }

    devList = NULL;
    devCount = 0;
    peglQueryDevicesEXT = NULL;

    demoState.nativeDisplay = EGL_NO_DISPLAY;
    demoState.platformType  = NvGlDemoInterface_Unknown;
}

static int NvGlDemoInitEglDeviceExt(void)
{
    const char* exts = NULL;
    EGLint n = 0;

    // Get extension string
    exts = NVGLDEMO_EGL_QUERY_STRING(EGL_NO_DISPLAY, EGL_EXTENSIONS);
    if (!exts) {
        NvGlDemoLog("eglQueryString fail.\n");
        goto NvGlDemoInitEglDeviceExt_fail;
    }

    // Check extensions and load functions needed for using outputs
    if (!NvGlDemoCheckExtension(exts, "EGL_EXT_device_base") ||
            !NvGlDemoCheckExtension(exts, "EGL_EXT_platform_base") ||
            !NvGlDemoCheckExtension(exts, "EGL_EXT_platform_device")) {
        NvGlDemoLog("egldevice platform ext is not there.\n");
        goto NvGlDemoInitEglDeviceExt_fail;
    }

    NVGLDEMO_EGL_GET_PROC_ADDR(eglQueryDevicesEXT, NvGlDemoInitEglDeviceExt_fail, PFNEGLQUERYDEVICESEXTPROC);

    // Load device list
    if (!peglQueryDevicesEXT(0, NULL, &n) || !n) {
        NvGlDemoLog("peglQueryDevicesEXT fail.\n");
        goto NvGlDemoInitEglDeviceExt_fail;
    }

    devList = (EGLDeviceEXT*)MALLOC(n * sizeof(EGLDeviceEXT));
    if (!devList || !peglQueryDevicesEXT(n, devList, &devCount) || !devCount) {
        NvGlDemoLog("peglQueryDevicesEXT fail.\n");
        goto NvGlDemoInitEglDeviceExt_fail;
    }

    if(devCount > 0) {
        demoState.nativeDisplay = (NativeDisplayType)devList[0];
        demoState.platformType  = NvGlDemoInterface_Device;
        // Success
        return 1;
    }

NvGlDemoInitEglDeviceExt_fail:

    NvGlDemoLog("NvGlDemoInitEglDeviceExt-fail.\n");

    NvGlDemoTermEglDeviceExt();

    return 0;
}

// Start up, initializing native window system and EGL after nvgldemo
//   options have been parsed. (Still need argc/argv for window system
//   options.)
int
NvGlDemoInitializeParsed(
    int* argc, char** argv,
    const char *appName,
    int glversion, int depthbits, int stencilbits)
{
    EGLint cfgAttrs[2*MAX_ATTRIB+1], cfgAttrIndex=0;
    EGLint ctxAttrs[2*MAX_ATTRIB+1], ctxAttrIndex=0;
    EGLint srfAttrs[2*MAX_ATTRIB+1], srfAttrIndex=0;
    const char* extensions;
    EGLConfig* configList = NULL;
    EGLint     configCount;
    EGLBoolean eglStatus;
    GLint max_VP_dims[] = {-1, -1};
    EGLenum   eglExtType = 0;
    EGLint* displayAttribs = NULL;
    PFNEGLQUERYSTREAMKHRPROC peglQueryStreamKHR = NULL;
    EGLBoolean isCrossProcessProducer = !strncmp(demoOptions.proctype,
                                                 "producer", NVGLDEMO_MAX_NAME);
    EGLBoolean isCrossProcessConsumer = !strncmp(demoOptions.proctype,
                                                 "consumer", NVGLDEMO_MAX_NAME);

    if (isCrossProcessProducer) {
        if (!NvGlDemoInitProducerProcess()) goto fail;
#ifndef ANDROID
        if (!NvGlDemoInitEglDeviceExt()) return 0;
#endif
    } else if (isCrossProcessConsumer) {
        if (!NvGlDemoInitConsumerProcess()) goto fail;
    }

    if (!isCrossProcessProducer) {
        // Initialize display access
        if (!NvGlDemoDisplayInit()) return 0;
    }

    extensions = NVGLDEMO_EGL_QUERY_STRING(EGL_NO_DISPLAY, EGL_EXTENSIONS);
    if (extensions && STRSTR(extensions, "EGL_EXT_platform_base")) {
        switch (demoState.platformType) {
            case NvGlDemoInterface_X11:
                eglExtType = EGL_PLATFORM_X11_EXT;
                break;
            case NvGlDemoInterface_Wayland:
                eglExtType = EGL_PLATFORM_WAYLAND_EXT;
                break;
            case NvGlDemoInterface_Device:
            case NvGlDemoInterface_DRM:
            case NvGlDemoInterface_WF:
                eglExtType = EGL_PLATFORM_DEVICE_EXT;
                break;
            default:
                eglExtType = 0;
                break;
       }
    }
    else {
      eglExtType = 0;
    }

#ifdef NVGLDEMO_HAS_DEVICE
    if (demoState.platformType == NvGlDemoInterface_Device &&
        !isCrossProcessProducer) {
#if !defined(__QNX__)
        if (NvGlDemoCreateDrmDevice()) {
            NvGlDemoLog("Could not create the DRM device\n");
            goto fail;
        }
#else
        if (NvGlDemoCreateWfdDevice()) {
            NvGlDemoLog("Could not create the WFD device\n");
            goto fail;
        }
#endif//!__QNX__
    }
#endif//NVGLDEMO_HAS_DEVICE

    if (!isCrossProcessProducer && NvGlDemoGetDisplayAttribs(&displayAttribs)) {
        NvGlDemoLog("Could not get display attributes for the platform\n");
        goto fail;
    }

    // Obtain the EGL display
    demoState.display = EGL_NO_DISPLAY;
    if (eglExtType) {
        PFNEGLGETPLATFORMDISPLAYEXTPROC  peglGetPlatformDisplayEXT = NULL;
        NVGLDEMO_EGL_GET_PROC_ADDR(eglGetPlatformDisplayEXT, fail, PFNEGLGETPLATFORMDISPLAYEXTPROC);
        demoState.display = peglGetPlatformDisplayEXT(eglExtType,
                                                      demoState.nativeDisplay,
                                                      displayAttribs);
    }
    if (displayAttribs) {
        FREE(displayAttribs);
    }
    if ((demoState.display == EGL_NO_DISPLAY) && (eglExtType != EGL_PLATFORM_DEVICE_EXT)) {
        eglExtType = 0;
        demoState.display = NVGLDEMO_EGL_GET_DISPLAY(demoState.nativeDisplay);
    }
    if (demoState.display == EGL_NO_DISPLAY) {
        NvGlDemoLog("EGL failed to obtain display.\n");
        goto fail;
    }

    // Initialize EGL
    eglStatus = NVGLDEMO_EGL_INITIALIZE(demoState.display, 0, 0);
    if (!eglStatus) {
        NvGlDemoLog("EGL failed to initialize.\n");
        goto fail;
    }

    // Query EGL extensions
    extensions = NVGLDEMO_EGL_QUERY_STRING(demoState.display, EGL_EXTENSIONS);

    // Bind GL API
    eglBindAPI(EGL_OPENGL_ES_API);

    // This is kinda ugly, but this function does so much and is not modularized.
    if ((isCrossProcessConsumer ||  isCrossProcessProducer) &&
        !NvGlDemoCreateCrossPartitionEGLStream()) {
        goto fail;
    }

    NVGLDEMO_EGL_GET_PROC_ADDR(eglQueryStreamKHR, fail, PFNEGLQUERYSTREAMKHRPROC);

    // Request GL version
    cfgAttrs[cfgAttrIndex++] = EGL_RENDERABLE_TYPE;
    cfgAttrs[cfgAttrIndex++] = (glversion == 2) ? EGL_OPENGL_ES2_BIT
                                                : EGL_OPENGL_ES_BIT;
    ctxAttrs[ctxAttrIndex++] = EGL_CONTEXT_CLIENT_VERSION;
    ctxAttrs[ctxAttrIndex++] = glversion;

    // Request a minimum of 1 bit each for red, green, blue, and alpha
    // Setting these to anything other than DONT_CARE causes the returned
    //   configs to be sorted with the largest bit counts first.
    cfgAttrs[cfgAttrIndex++] = EGL_RED_SIZE;
    cfgAttrs[cfgAttrIndex++] = 1;
    cfgAttrs[cfgAttrIndex++] = EGL_GREEN_SIZE;
    cfgAttrs[cfgAttrIndex++] = 1;
    cfgAttrs[cfgAttrIndex++] = EGL_BLUE_SIZE;
    cfgAttrs[cfgAttrIndex++] = 1;
    cfgAttrs[cfgAttrIndex++] = EGL_ALPHA_SIZE;
    cfgAttrs[cfgAttrIndex++] = 1;

    int surfaceTypeMask = 0;

#ifndef ANDROID
    if ((demoOptions.eglstreamsock[0] != '\0')
        || (eglExtType == EGL_PLATFORM_DEVICE_EXT)
        || ((demoState.platformType == NvGlDemoInterface_QnxScreen)
        && (demoOptions.eglQnxScreenTest == 1))) {
#else
    if (demoOptions.proctype[0] != '\0') {
#endif

        surfaceTypeMask |= EGL_STREAM_BIT_KHR;
    }

    // If application requires depth or stencil, request them
    if (depthbits) {
        cfgAttrs[cfgAttrIndex++] = EGL_DEPTH_SIZE;
        cfgAttrs[cfgAttrIndex++] = depthbits;
    }
    if (stencilbits) {
        cfgAttrs[cfgAttrIndex++] = EGL_STENCIL_SIZE;
        cfgAttrs[cfgAttrIndex++] = stencilbits;
    }

    // Request antialiasing
    cfgAttrs[cfgAttrIndex++] = EGL_SAMPLES;
    cfgAttrs[cfgAttrIndex++] = demoOptions.msaa;
#ifdef EGL_NV_coverage_sample
    if (STRSTR(extensions, "EGL_NV_coverage_sample")) {
        cfgAttrs[cfgAttrIndex++] = EGL_COVERAGE_SAMPLES_NV;
        cfgAttrs[cfgAttrIndex++] = demoOptions.csaa;
        cfgAttrs[cfgAttrIndex++] = EGL_COVERAGE_BUFFERS_NV;
        cfgAttrs[cfgAttrIndex++] = demoOptions.csaa ? 1 : 0;
    } else
#endif // EGL_NV_coverage_sample
    if (demoOptions.csaa) {
        NvGlDemoLog("Coverage sampling not supported.\n");
        goto fail;
    }
    if (demoOptions.isProtected && !STRSTR(extensions, "EGL_EXT_protected_content")) {
        NvGlDemoLog("VPR memory is not supported");
        goto fail;
    }

    // NvGlDemoInterface_QnxScreen will be set as the platform for QNX while creating window surface
    if (demoState.platformType != NvGlDemoInterface_QnxScreen
        || demoOptions.eglstreamsock[0] != '\0') {
        // Request buffering
        if (demoOptions.buffering) {

            srfAttrs[srfAttrIndex++] = EGL_RENDER_BUFFER;

            switch (demoOptions.buffering) {
                case 1:
                    srfAttrs[srfAttrIndex++] = EGL_SINGLE_BUFFER;
                    break;

                case 2:
                    srfAttrs[srfAttrIndex++] = EGL_BACK_BUFFER;
                    break;

                #ifdef EGL_TRIPLE_BUFFER_NV
                case 3:
                    if(!STRSTR(extensions, "EGL_NV_triple_buffer")) {
                        NvGlDemoLog("TRIPLE_BUFFER not supported.\n");
                        goto fail;
                    }
                    srfAttrs[srfAttrIndex++] = EGL_TRIPLE_BUFFER_NV;
                    break;
                #endif

                #ifdef EGL_QUADRUPLE_BUFFER_NV
                case 4:
                    if(!STRSTR(extensions, "EGL_NV_quadruple_buffer")) {
                        NvGlDemoLog("QUADRUPLE_BUFFER not supported.\n");
                        goto fail;
                    }
                    srfAttrs[srfAttrIndex++] = EGL_QUADRUPLE_BUFFER_NV;
                    break;
                #endif

                default:
                    NvGlDemoLog("Buffering (%d) not supported.\n",
                                demoOptions.buffering);
                    goto fail;
            }
        }
    }

    if (demoOptions.enablePostSubBuffer) {
        srfAttrs[srfAttrIndex++] = EGL_POST_SUB_BUFFER_SUPPORTED_NV;
        srfAttrs[srfAttrIndex++] = EGL_TRUE;
    }

    if (demoOptions.enableMutableRenderBuffer) {
        surfaceTypeMask |= EGL_MUTABLE_RENDER_BUFFER_BIT_KHR;
    }

    if (surfaceTypeMask) {
        cfgAttrs[cfgAttrIndex++] = EGL_SURFACE_TYPE;
        cfgAttrs[cfgAttrIndex++] = surfaceTypeMask;
    }

    if (demoOptions.isProtected) {
        ctxAttrs[ctxAttrIndex++] = EGL_PROTECTED_CONTENT_EXT;
        ctxAttrs[ctxAttrIndex++] = EGL_TRUE;
        srfAttrs[srfAttrIndex++] = EGL_PROTECTED_CONTENT_EXT;
        srfAttrs[srfAttrIndex++] = EGL_TRUE;
    }

    // Terminate attribute lists
    cfgAttrs[cfgAttrIndex++] = EGL_NONE;
    ctxAttrs[ctxAttrIndex++] = EGL_NONE;

    // Find out how many configurations suit our needs
    eglStatus = eglChooseConfig(demoState.display, cfgAttrs,
                                NULL, 0, &configCount);
    if (!eglStatus || !configCount) {
        NvGlDemoLog("EGL failed to return any matching configurations.\n");
        goto fail;
    }

    // Allocate room for the list of matching configurations
    configList = (EGLConfig*)MALLOC(configCount * sizeof(EGLConfig));
    if (!configList) {
        NvGlDemoLog("Allocation failure obtaining configuration list.\n");
        goto fail;
    }

    // Obtain the configuration list from EGL
    eglStatus = eglChooseConfig(demoState.display, cfgAttrs,
                                configList, configCount, &configCount);
    if (!eglStatus || !configCount) {
        NvGlDemoLog("EGL failed to populate configuration list.\n");
        goto fail;
    }

    // Select an EGL configuration that matches the native window
    // Currently we just choose the first one, but we could search
    //   the list based on other criteria.
    demoState.config = configList[0];
    FREE(configList);
    configList = 0;

    if (demoOptions.eglstreamsock[0] != '\0') {
#if defined(EGL_KHR_stream_producer_eglsurface) \
        && defined(EGL_KHR_stream_cross_process_fd) \
        && !defined(__INTEGRITY)
    if (STRSTR(extensions, "EGL_KHR_stream_producer_eglsurface")
        && STRSTR(extensions, "EGL_KHR_stream_cross_process_fd")) {
            PFNEGLCREATESTREAMFROMFILEDESCRIPTORKHRPROC
                peglCreateStreamFromFileDescriptorKHR = NULL;
            EGLNativeFileDescriptorKHR fd;

            NVGLDEMO_EGL_GET_PROC_ADDR(eglCreateStreamFromFileDescriptorKHR, fail, PFNEGLCREATESTREAMFROMFILEDESCRIPTORKHRPROC);
            // Get the file descriptor associated with the consumer's EGL Stream
            fd = NvGlDemoFdFromSocket(demoOptions.eglstreamsock);
            if (fd == -1) {
                NvGlDemoLog("Problem reading fd from socket.\n");
                goto fail;
            }

            // Create our own stream handle from file descriptor
            demoState.stream = peglCreateStreamFromFileDescriptorKHR(
                                        demoState.display, fd);
            NvGlDemoCloseFd(fd);
            if (demoState.stream == EGL_NO_STREAM_KHR) {
                NvGlDemoLog("Couldn't create EGL Stream from fd.\n");
                goto fail;
            }
        }
        else
#endif
        {
            NvGlDemoLog("EGL stream not supported.\n");
            goto fail;
        }
    }
    else {
        // This is not needed in case of cross partition egldevice producer.
        if (!isCrossProcessProducer) {
            // Create the window
            // (This may make use of the config, which is why it is chosen first.)
            if (!NvGlDemoWindowInit(argc, argv, appName)) goto fail;
        }
    }

#ifndef ANDROID
    if ((demoOptions.eglstreamsock[0] != '\0')
        || (eglExtType == EGL_PLATFORM_DEVICE_EXT)
        || ((demoState.platformType == NvGlDemoInterface_QnxScreen)
        && (demoOptions.eglQnxScreenTest == 1))) {
#else
    if (demoOptions.proctype[0] != '\0') {
#endif

#ifdef ANDROID
        if (isCrossProcessProducer) {
#endif
            srfAttrs[srfAttrIndex++] = EGL_WIDTH;
            srfAttrs[srfAttrIndex++] = demoOptions.windowSize[0]
                                        ? demoOptions.windowSize[0]
                                        : NVGLDEMO_DEFAULT_WIDTH;
            srfAttrs[srfAttrIndex++] = EGL_HEIGHT;
            srfAttrs[srfAttrIndex++] = demoOptions.windowSize[1]
                                        ? demoOptions.windowSize[1]
                                        : NVGLDEMO_DEFAULT_HEIGHT;
#ifdef ANDROID
        }
#endif
    }
    srfAttrs[srfAttrIndex++] = EGL_NONE;

    // For cross-p mode consumer, demoState.stream = EGL_NO_STREAM_KHR. But we don't need the
    // below code path.
    if(demoState.stream != EGL_NO_STREAM_KHR && !isCrossProcessConsumer) {
        if (!NvGlDemoPrepareStreamToAttachProducer()) {
            goto fail;
        }
#ifndef ANDROID
       PFNEGLCREATESTREAMPRODUCERSURFACEKHRPROC
          peglCreateStreamProducerSurfaceKHR = NULL;
       NVGLDEMO_EGL_GET_PROC_ADDR(eglCreateStreamProducerSurfaceKHR, fail, PFNEGLCREATESTREAMPRODUCERSURFACEKHRPROC);
#endif
         demoState.surface =
#ifndef ANDROID
                peglCreateStreamProducerSurfaceKHR(
#else
                eglCreateStreamProducerSurfaceKHR(
#endif
                    demoState.display,
                    demoState.config,
                    demoState.stream,
                    srfAttrs);
    }
    else {
        PFNEGLCREATEPLATFORMWINDOWSURFACEEXTPROC peglCreatePlatformWindowSurfaceEXT = NULL;
        if(eglExtType != 0) {
           NVGLDEMO_EGL_GET_PROC_ADDR(eglCreatePlatformWindowSurfaceEXT, fail, PFNEGLCREATEPLATFORMWINDOWSURFACEEXTPROC);
        }
        switch (eglExtType) {
                case 0:
                    demoState.surface =
                        eglCreateWindowSurface(demoState.display,
                                               demoState.config,
                                               demoState.nativeWindow,
                                               srfAttrs);
                    break;
                case EGL_PLATFORM_X11_EXT:
                    demoState.surface =
                        peglCreatePlatformWindowSurfaceEXT(demoState.display,
                                                          demoState.config,
                                                          &demoState.nativeWindow,
                                                          srfAttrs);
                    break;
                case EGL_PLATFORM_DEVICE_EXT:
                    demoState.surface = (EGLSurface)demoState.nativeWindow;
                    break;
                default:
                    demoState.surface =
                        peglCreatePlatformWindowSurfaceEXT(demoState.display,
                                                          demoState.config,
                                                          demoState.nativeWindow,
                                                          srfAttrs);
                    break;
       }
    }

    if (eglExtType == EGL_PLATFORM_DEVICE_EXT && isCrossProcessConsumer) {
        return 1;
    } else {
        if (demoState.surface == EGL_NO_SURFACE) {
            NvGlDemoLog("EGL couldn't create window surface.\n");
            goto fail;
        }

        // Create an EGL context
        demoState.context =
            eglCreateContext(demoState.display,
                             demoState.config,
                             NULL,
                             ctxAttrs);
        if (!demoState.context) {
            NvGlDemoLog("EGL couldn't create context.\n");
            goto fail;
        }

        // Make the context and surface current for rendering
        eglStatus = eglMakeCurrent(demoState.display,
                                   demoState.surface, demoState.surface,
                                   demoState.context);
        if (!eglStatus) {
            NvGlDemoLog("EGL couldn't make context/surface current.\n");
            goto fail;
        }

        // Query the EGL surface width and height
        eglStatus =  eglQuerySurface(demoState.display, demoState.surface,
                                     EGL_WIDTH,  &demoState.width)
                  && eglQuerySurface(demoState.display, demoState.surface,
                                 EGL_HEIGHT, &demoState.height);
        if (!eglStatus) {
            NvGlDemoLog("EGL couldn't get window width/height.\n");
            goto fail;
        }

        // Query the Maximum Viewport width and height
        glGetIntegerv(GL_MAX_VIEWPORT_DIMS, max_VP_dims);
        if (max_VP_dims[0] == -1 ||  max_VP_dims[1] == -1) {
            NvGlDemoLog("Couldn't query maximum viewport dimensions.\n");
            goto fail;
        }

        // Check for the Maximum Viewport width and height
        if (demoOptions.windowSize[0] > max_VP_dims[0] ||
            demoOptions.windowSize[1] > max_VP_dims[1]) {
            NvGlDemoLog("Window size exceeds maximum limit of %d x %d.\n",
                        max_VP_dims[0], max_VP_dims[1]);
            goto fail;
        }

        return 1;
    }

    // On failure, clean up partial initialization
    fail:
    if (configList) FREE(configList);
    NvGlDemoShutdown();
    return 0;
}

// Shut down, freeing all EGL and native window system resources.
void
NvGlDemoShutdown(void)
{
    EGLBoolean eglStatus;

    // Clear rendering context
    // Note that we need to bind the API to unbind... yick
    if (demoState.display != EGL_NO_DISPLAY) {
        eglBindAPI(EGL_OPENGL_ES_API);
        eglStatus = eglMakeCurrent(demoState.display,
                                   EGL_NO_SURFACE, EGL_NO_SURFACE,
                                   EGL_NO_CONTEXT);
        if (!eglStatus)
            NvGlDemoLog("Error clearing current surfaces/context.\n");
    }

    // Destroy the EGL context
    if (demoState.context != EGL_NO_CONTEXT) {
        eglStatus = eglDestroyContext(demoState.display, demoState.context);
        if (!eglStatus)
            NvGlDemoLog("Error destroying EGL context.\n");
        demoState.context = EGL_NO_CONTEXT;
    }

    // Destroy the EGL surface
    if (demoState.surface != EGL_NO_SURFACE) {
        eglStatus = eglDestroySurface(demoState.display, demoState.surface);
        if (!eglStatus)
            NvGlDemoLog("Error destroying EGL surface.\n");
        demoState.surface = EGL_NO_SURFACE;
    }

#if defined(EGL_KHR_stream_producer_eglsurface)
    // Destroy the EGL stream
    if ((demoState.stream != EGL_NO_STREAM_KHR) &&
        (demoState.platformType != NvGlDemoInterface_QnxScreen)) {
        PFNEGLDESTROYSTREAMKHRPROC
            pEglDestroyStreamKHR;

        pEglDestroyStreamKHR =
            (PFNEGLDESTROYSTREAMKHRPROC)
                eglGetProcAddress("eglDestroyStreamKHR");
        if (pEglDestroyStreamKHR != NULL)
            pEglDestroyStreamKHR(demoState.display, demoState.stream);
        demoState.stream = EGL_NO_STREAM_KHR;
    }
#endif

    // Close the window
    NvGlDemoWindowTerm();

    NvGlDemoEglTerminate();

    // Terminate display access
    NvGlDemoDisplayTerm();

    NvGlDemoTermEglDeviceExt();
}

void
NvGlDemoEglTerminate(void)
{
    EGLBoolean eglStatus;

#if !defined(__INTEGRITY)
    if (g_ServerID != -1) {
        close(g_ServerID);
        g_ServerID = -1;
    }

    if (g_ClientID != -1) {
        close(g_ClientID);
        g_ClientID = -1;
    }
#endif

    // Terminate EGL
    if (demoState.display != EGL_NO_DISPLAY) {
        eglStatus = eglTerminate(demoState.display);
        if (!eglStatus)
            NvGlDemoLog("Error terminating EGL.\n");
        demoState.display = EGL_NO_DISPLAY;
    }

    // Release EGL thread
    eglStatus = eglReleaseThread();
    if (!eglStatus)
        NvGlDemoLog("Error releasing EGL thread.\n");
}

#ifdef EGL_NV_system_time
// Gets the system time in nanoseconds
long long
NvGlDemoSysTime(void)
{
    static PFNEGLGETSYSTEMTIMENVPROC eglGetSystemTimeNV = NULL;
    static int inited = 0;
    static long long nano = 1;
    if(!inited)
    {
        PFNEGLGETSYSTEMTIMEFREQUENCYNVPROC eglGetSystemTimeFrequencyNV =
            (PFNEGLGETSYSTEMTIMEFREQUENCYNVPROC)eglGetProcAddress("eglGetSystemTimeFrequencyNV");
        eglGetSystemTimeNV = (PFNEGLGETSYSTEMTIMENVPROC)eglGetProcAddress("eglGetSystemTimeNV");

        ASSERT(eglGetSystemTimeFrequencyNV && eglGetSystemTimeNV);

        // Compute factor for converting eglGetSystemTimeNV() to nanoseconds
        nano = 1000000000/eglGetSystemTimeFrequencyNV();
        inited = 1;
    }

    return nano*eglGetSystemTimeNV();
}
#endif // EGL_NV_system_time
