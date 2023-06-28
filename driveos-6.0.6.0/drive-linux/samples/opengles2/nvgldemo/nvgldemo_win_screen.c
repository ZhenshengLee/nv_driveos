/*
 * nvgldemo_win_screen.c
 *
 * Copyright (c) 2014-2019, NVIDIA CORPORATION. All rights reserved.
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
// This file illustrates how to access the display and create a context
// & window for GL application using QNX CAR 2.1 Platform.
//

//TODO add conditional for DEMO_USE_CAR2

#include "nvgldemo.h"
#include "nvgldemo_win_screen.h"

#include <EGL/egl.h>

#ifdef EGL_EXT_stream_consumer_qnxscreen_window
#include <GLES2/gl2.h>

static PFNEGLCREATESTREAMKHRPROC                   peglCreateStreamKHR = NULL;
static PFNEGLQUERYSTREAMKHRPROC                    peglQueryStreamKHR = NULL;
static PFNEGLDESTROYSTREAMKHRPROC                  peglDestroyStreamKHR = NULL;
static PFNEGLSTREAMCONSUMERQNXSCREENWINDOWEXTPROC  peglStreamConsumerQNXScreenWindowEXT = NULL;
static PFNEGLSTREAMCONSUMERACQUIREATTRIBEXTPROC    peglStreamConsumerAcquireAttribEXT = NULL;

static NvGlDemoEglQnxScreenInfo *peglQnxScreenInfo = NULL;

static int eglQnxScreenConsumerInit(void);
static void eglQnxScreenConsumerTerm(void);
static int eglQnxScreenConsumerStart(void);
static void eglQnxScreenConsumerStop(void);
static int eglQnxScreenConsumerConnect(void);
static inline int eglQnxScreenConsumerAcquireFrame(void);
static inline void eglQnxScreenConsumerDisconnect(void);
static void *ThreadEntryPoint(void *arg);
#endif //EGL_EXT_stream_consumer_qnxscreen_window

int choose_format(EGLDisplay, EGLConfig);

int choose_format(void *disp, void *egl_conf)
{
    int ret;
    EGLint bufSize;

    EGLDisplay dpy = (EGLDisplay)(disp);
    EGLConfig cfg = (EGLConfig) (egl_conf);

    eglGetConfigAttrib(dpy, cfg, EGL_BUFFER_SIZE, &bufSize);

    switch (bufSize) {
        case 16:
            ret = SCREEN_FORMAT_RGB565;
        break;
        case 24:
            ret = SCREEN_FORMAT_RGBX8888;
        break;
        case 32:
        default:
            ret = SCREEN_FORMAT_RGBA8888;
        break;
    }

    return ret;
}

//
// Window system startup/shutdown
//

int
NvGlDemoGetDisplayAttribs(EGLint** displayAttribs)
{
    // Nothing to do
    return 0;
}

// Initialize access to the display system
int NvGlDemoDisplayInit(void)
{
    int displayCount, attached = 0;
    screen_display_t displayHandle[16];
    int displayIDs[16];
    int activeDisplayIdx = -1;
    demoState.platformType = NvGlDemoInterface_QnxScreen;

    // Call the eglQnxScreenConsumer module if option is enabled
    NVGLDEMO_EGL_EXT_EXEC_QNX_SCREEN_CONSUMER_RET_API(demoOptions.eglQnxScreenTest,
        eglQnxScreenConsumerInit, 0, int, NULL);

    // Allocate a structure for the platform-specific state
    demoState.platform = (NvGlDemoPlatformState*) MALLOC(
            sizeof(NvGlDemoPlatformState));
    if (!demoState.platform) {
        NvGlDemoLog("Could not allocate platform specific storage memory.\n");
        goto fail;
    }
    MEMSET(demoState.platform, 0, sizeof(NvGlDemoPlatformState));

    // Create QNX CAR 2.1  context
    int rc = screen_create_context(&demoState.platform->screen_ctx, 0);
    if (rc) {
        NvGlDemoLog("Failed to create context.\n");
        goto fail;
    }

    // Set the requested display
    screen_get_context_property_iv(demoState.platform->screen_ctx,
                                   SCREEN_PROPERTY_DISPLAY_COUNT, &displayCount);
    screen_get_context_property_pv(demoState.platform->screen_ctx,
                                   SCREEN_PROPERTY_DISPLAYS,
                                   (void **) displayHandle);

    for (int idx = 0; idx < displayCount; idx++) {

        rc = screen_get_display_property_iv(displayHandle[idx], SCREEN_PROPERTY_ID,
                                            &displayIDs[idx]);
        if (rc) {
            NvGlDemoLog("Failed to queryDisplay IDs.\n");
            goto fail;
        } else if (displayIDs[idx] == (demoOptions.displayNumber + 1)) {
            activeDisplayIdx = idx;
        }
    }

    if (activeDisplayIdx < 0) {
        NvGlDemoLog("Requested display %d not present.\n",
                    demoOptions.displayNumber);
        goto fail;
    }

    screen_get_display_property_iv(displayHandle[activeDisplayIdx],
                                   SCREEN_PROPERTY_ATTACHED,
                                   &attached);
    if (attached) {
        demoState.platform->screen_display =
            displayHandle[activeDisplayIdx];
    } else {
        NvGlDemoLog("Requested display %d not attached.\n",
                    demoOptions.displayNumber);
        goto fail;
    }

    // Setting demoState.nativeDisplay = demoState.platform->screen_display
    // causes eglGetDisplay() to fail. Just use EGL_DEFAULT_DISPLAY.
    demoState.nativeDisplay = (EGLNativeDisplayType) EGL_DEFAULT_DISPLAY;

    if (demoOptions.displayName[0]) {
        NvGlDemoLog("Setting display output is not supported. Exiting.\n");
        goto fail;
    }

    if (demoOptions.displayRate) {
        NvGlDemoLog("Setting display refresh is not supported. Exiting.\n");
        goto fail;
    }

    if (demoOptions.displayBlend == NvGlDemoDisplayBlend_ColorKey) {
        NvGlDemoLog("Display Color Key is not supported. Exiting.\n");
        goto fail;
    }

    // But do not fail on the unsupported "displaysize"
    // option.
    // We will ignore it on QNX CAR 2.1.
    if (demoOptions.displaySize[0]) {
        NvGlDemoLog("Setting display size is not supported. Ignoring.\n");
    }

    return 1;

fail:
    NvGlDemoDisplayTerm();
    return 0;
}

// Terminate access to the display system
void NvGlDemoDisplayTerm(void)
{
    demoState.platformType = NvGlDemoInterface_Unknown;

    // Call the eglQnxScreenConsumer module if option is enabled
    NVGLDEMO_EGL_EXT_EXEC_QNX_SCREEN_CONSUMER_NORET_API(demoOptions.eglQnxScreenTest,
        eglQnxScreenConsumerTerm, NULL);

    if (!demoState.platform) {
        return;
    }

    NvGlDemoWindowTerm();

    if (demoState.platform->screen_ctx) {
        screen_destroy_context(demoState.platform->screen_ctx);
    }

    FREE(demoState.platform);
    demoState.platform = NULL;
}

// Create the window
int NvGlDemoWindowInit(int *argc, char **argv, const char *appName)
{
    int usage = SCREEN_USAGE_OPENGL_ES2;
    int nbuffers = 2;
    EGLint interval = 1;
    int rc, format;
    int pipelineId, defaultpipelineId = 0;
    int zorder;
    const int NUM_PIPELINES = 4;

    // Call the eglQnxScreenConsumer module if option is enabled
    NVGLDEMO_EGL_EXT_EXEC_QNX_SCREEN_CONSUMER_RET_API(demoOptions.eglQnxScreenTest,
        eglQnxScreenConsumerStart, 0, int, NULL);

    if (!demoState.platform) {
        return 0;
    }

    if (!demoState.platform->screen_ctx) {
        return 0;
    }

    if (demoOptions.buffering && demoOptions.buffering > 2) {
        nbuffers = demoOptions.buffering;
    }

    // If not specified, use default window size
    if (!demoOptions.windowSize[0])
        demoOptions.windowSize[0] = NVGLDEMO_DEFAULT_WIDTH;
    if (!demoOptions.windowSize[1])
        demoOptions.windowSize[1] = NVGLDEMO_DEFAULT_HEIGHT;

    // Create a native window
    rc = screen_create_window(&demoState.platform->screen_win, demoState.platform->screen_ctx);
    if (rc) {
        NvGlDemoLog("Failed to create native window.\n");
        goto fail;
    }
    demoState.nativeWindow = (NativeWindowType)demoState.platform->screen_win;

    // Set the requested display
    screen_set_window_property_pv(demoState.platform->screen_win,
                                  SCREEN_PROPERTY_DISPLAY,
                                  (void **) &demoState.platform->screen_display);

    // Configure the requested layer.
    // Screen API does not have much support for querying the display pipeline information.
    // Hard coding the pipeline ID's and the number of pipelines supported per display for now.
    if (demoOptions.displayLayer >= NUM_PIPELINES) {
        NvGlDemoLog("Failed to set the requested layer %d.\n", demoOptions.displayLayer);
        goto fail;
    }

    // Query defualt pipelineId from Screen compositor
    rc = screen_get_window_property_iv(demoState.platform->screen_win, SCREEN_PROPERTY_PIPELINE,
                                                                            &defaultpipelineId);
    if (rc) {
        NvGlDemoLog("Query the default layer info is failed Req[%d].\n", demoOptions.displayLayer);
        goto fail;
    }
    // Each display can have maximum 10 pipeline ids
    demoOptions.displayNumber = defaultpipelineId / 10;

    // We use the default composited pipeline when the
    // pipeline to choose is not explicitly specified
    pipelineId = demoOptions.displayLayer >= 0 ? ((10 * demoOptions.displayNumber) +
                 demoOptions.displayLayer + 1) : defaultpipelineId;

    rc = screen_set_window_property_iv(demoState.platform->screen_win, SCREEN_PROPERTY_PIPELINE, &pipelineId);
    if (rc) {
        NvGlDemoLog("Failed to set the requested layer %d.\n", demoOptions.displayLayer);
        goto fail;
    }

    // Select and set window property SCREEN_PROPERTY_FORMAT
    format = choose_format(demoState.display, demoState.config);
    rc = screen_set_window_property_iv(demoState.platform->screen_win,
            SCREEN_PROPERTY_FORMAT, &format);
    if (rc) {
        NvGlDemoLog("Failed to set window property SCREEN_PROPERTY_FORMAT.\n");
        goto fail;
    }

    // defaultpipelineId used by QNX Screen compositor
    if (pipelineId != defaultpipelineId) {
        usage |= SCREEN_USAGE_OVERLAY;
    }

    // Set window property SCREEN_PROPERTY_USAGE
    rc = screen_set_window_property_iv(demoState.platform->screen_win,
            SCREEN_PROPERTY_USAGE, &usage);
    if (rc) {
        NvGlDemoLog("Failed to set window property SCREEN_PROPERTY_USAGE.\n");
        goto fail;
    }

    // Set window property SCREEN_PROPERTY_SWAP_INTERVAL
    rc = screen_set_window_property_iv(demoState.platform->screen_win,
            SCREEN_PROPERTY_SWAP_INTERVAL, &interval);
    if (rc) {
        NvGlDemoLog("Failed to set window property SCREEN_PROPERTY_SWAP_INTERVAL.\n");
        goto fail;
    }

    if (demoOptions.windowSize[0] > 0 && demoOptions.windowSize[1] > 0) {
        // Set window property SCREEN_PROPERTY_SIZE
        rc = screen_set_window_property_iv(demoState.platform->screen_win,
                SCREEN_PROPERTY_SIZE, demoOptions.windowSize);
        if (rc) {
            NvGlDemoLog("Failed to set window property SCREEN_PROPERTY_SIZE.\n");
            goto fail;
        }
    }

    if (demoOptions.windowOffset[0] != 0 || demoOptions.windowOffset[1] != 0) {
        // Set window property SCREEN_PROPERTY_POSTION
        rc = screen_set_window_property_iv(demoState.platform->screen_win,
                SCREEN_PROPERTY_POSITION, demoOptions.windowOffset);
        if (rc) {
            NvGlDemoLog("Failed to set window property SCREEN_PROPERTY_POSITION.\n");
            goto fail;
        }
    }

    int alpha;
    int transperency = SCREEN_TRANSPARENCY_SOURCE;
    switch (demoOptions.displayBlend) {
        case NvGlDemoDisplayBlend_PixelAlpha:
            transperency = SCREEN_TRANSPARENCY_SOURCE_OVER;
            break;
        case NvGlDemoDisplayBlend_ConstAlpha:
            transperency = SCREEN_TRANSPARENCY_NONE;
            alpha = (int)(demoOptions.displayAlpha * 255.0f);
            if (alpha < 0 || alpha > 255) {
                NvGlDemoLog("Alpha value specified for constant blending is not in range [0, 1]. Using alpha 1.0.\n");
                alpha = 255;
            }
            rc = screen_set_window_property_iv(demoState.platform->screen_win,
                    SCREEN_PROPERTY_GLOBAL_ALPHA, &alpha);
            if (rc) {
                NvGlDemoLog("Failed to set window property SCREEN_PROPERTY_GLOBAL_ALPHA.\n");
                goto fail;
            }
            break;
        case NvGlDemoDisplayBlend_None:
            transperency = SCREEN_TRANSPARENCY_SOURCE;
            break;
        case NvGlDemoDisplayBlend_ColorKey:
        // Not supported. Should not hit this code path.
            assert(0);
        default:
            break;
    }

    rc = screen_set_window_property_iv(demoState.platform->screen_win,
            SCREEN_PROPERTY_TRANSPARENCY, &transperency);
    if (rc) {
        NvGlDemoLog("Failed to set window property SCREEN_TRANSPARENCY_SOURCE_OVER.\n");
        goto fail;
    }

    // zorder is proportional to displayLayer
    zorder = demoOptions.displayLayer < 0 ? (defaultpipelineId - 1) % 10 : demoOptions.displayLayer;
    rc = screen_set_window_property_iv(demoState.platform->screen_win,
                                       SCREEN_PROPERTY_ZORDER, &zorder);
    if (rc) {
        NvGlDemoLog("Failed to set zorder on layer %d.\n", demoOptions.displayLayer);
        goto fail;
    }



    // Create QNX CAR 2.1 window buffers
    rc = screen_create_window_buffers(demoState.platform->screen_win, nbuffers);
    if (rc) {
        NvGlDemoLog("Failed to create window buffers.\n");
        goto fail;
    }

    return 1;

fail:
    NvGlDemoWindowTerm();
    return 0;
}

int NvGlDemoPostSwap(void)
{
    // Nothing to do
    return 1;
}

// Close the window
void NvGlDemoWindowTerm(void)
{
    // Call the eglQnxScreenConsumer module if option is enabled
    NVGLDEMO_EGL_EXT_EXEC_QNX_SCREEN_CONSUMER_NORET_API(demoOptions.eglQnxScreenTest,
        eglQnxScreenConsumerStop, NULL);

    if (!demoState.platform) {
        return;
    }

    if (!demoState.platform->screen_ctx) {
        return;
    }

    if (!demoState.platform->screen_win) {
        return;
    }

    // Destroy native window & event
    screen_destroy_window(demoState.platform->screen_win);
    screen_destroy_event(demoState.platform->screen_ev);
    demoState.platform->screen_win = NULL;
    demoState.platform->screen_ev = NULL;
    demoState.nativeWindow = NULL;
}

//
// Pixmap support
//

EGLNativePixmapType NvGlDemoPixmapCreate(unsigned int width,
        unsigned int height, unsigned int depth)
{
    screen_pixmap_t screen_pix;
    int usage, format, size[2];
    int rc;

    // Call the eglQnxScreenConsumer module if option is enabled
    NVGLDEMO_EGL_EXT_EXEC_QNX_SCREEN_CONSUMER_RET_API(demoOptions.eglQnxScreenTest,
        NULL, 0, EGLNativePixmapType, NULL);

    // Create QNX CAR 2.1 pixmap
    rc = screen_create_pixmap(&screen_pix, demoState.platform->screen_ctx);
    if (rc) {
        NvGlDemoLog("Failed to create pixmap.\n");
        return 0;
    }

    // Set pixmap property SCREEN_PROPERTY_USAGE
    usage = SCREEN_USAGE_READ | SCREEN_USAGE_NATIVE | SCREEN_USAGE_OPENGL_ES2;
    rc = screen_set_pixmap_property_iv(screen_pix, SCREEN_PROPERTY_USAGE, &usage);
    if (rc) {
        NvGlDemoLog("Failed to set pixmap property SCREEN_PROPERTY_USAGE.\n");
        goto fail;
    }

    // Set pixmap property SCREEN_PROPERTY_FORMAT
    format = SCREEN_FORMAT_RGBA8888;
    rc = screen_set_pixmap_property_iv(screen_pix, SCREEN_PROPERTY_FORMAT, &format);
    if (rc) {
        NvGlDemoLog("Failed to set pixmap property SCREEN_PROPERTY_FORMAT.\n");
        goto fail;
    }

    // Set pixmap property SCREEN_PROPERTY_SIZE
    size[0] = width;
    size[1] = height;
    rc = screen_set_pixmap_property_iv(screen_pix, SCREEN_PROPERTY_BUFFER_SIZE, size);
    if (rc) {
        NvGlDemoLog("Failed to set pixmap property SCREEN_PROPERTY_BUFFER_SIZE.\n");
        goto fail;
    }
    rc = screen_create_pixmap_buffer(screen_pix);
    if (rc) {
        NvGlDemoLog("Failed to create pixmap buffer.\n");
        goto fail;
    }


    return (EGLNativePixmapType)screen_pix;

fail:
    NvGlDemoPixmapDelete(screen_pix);
    return 0;
}


void NvGlDemoPixmapDelete(EGLNativePixmapType screen_pix)
{
    // Call the eglQnxScreenConsumer module if option is enabled
    NVGLDEMO_EGL_EXT_EXEC_QNX_SCREEN_CONSUMER_NORET_API(demoOptions.eglQnxScreenTest,
        NULL, NULL);

    if (screen_pix) {
        screen_destroy_pixmap(screen_pix);
    }
}

//
// Callback handling
//
static NvGlDemoCloseCB closeCB = NULL;
static NvGlDemoResizeCB resizeCB = NULL;
static NvGlDemoKeyCB keyCB = NULL;
static NvGlDemoPointerCB pointerCB = NULL;
static NvGlDemoButtonCB buttonCB = NULL;


static void NvGlDemoUpdateEventMask(void)
{
    static int rc = 1;

    if(rc) {
       rc = screen_create_event(&demoState.platform->screen_ev);
    }
}

void NvGlDemoSetCloseCB(NvGlDemoCloseCB cb)
{
    // Call the eglQnxScreenConsumer module if option is enabled
    NVGLDEMO_EGL_EXT_EXEC_QNX_SCREEN_CONSUMER_NORET_API(demoOptions.eglQnxScreenTest,
        NULL, NULL);

    closeCB = cb;
    NvGlDemoUpdateEventMask();
}

void NvGlDemoSetResizeCB(NvGlDemoResizeCB cb)
{
    // Call the eglQnxScreenConsumer module if option is enabled
    NVGLDEMO_EGL_EXT_EXEC_QNX_SCREEN_CONSUMER_NORET_API(demoOptions.eglQnxScreenTest,
        NULL, NULL);

    resizeCB = cb;
    NvGlDemoUpdateEventMask();
}

void NvGlDemoSetKeyCB(NvGlDemoKeyCB cb)
{
    // Call the eglQnxScreenConsumer module if option is enabled
    NVGLDEMO_EGL_EXT_EXEC_QNX_SCREEN_CONSUMER_NORET_API(demoOptions.eglQnxScreenTest,
        NULL, NULL);

    keyCB = cb;
    NvGlDemoUpdateEventMask();
}

void NvGlDemoSetPointerCB(NvGlDemoPointerCB cb)
{
    // Call the eglQnxScreenConsumer module if option is enabled
    NVGLDEMO_EGL_EXT_EXEC_QNX_SCREEN_CONSUMER_NORET_API(demoOptions.eglQnxScreenTest,
        NULL, NULL);

    pointerCB = cb;
    NvGlDemoUpdateEventMask();
}

void NvGlDemoSetButtonCB(NvGlDemoButtonCB cb)
{
    // Call the eglQnxScreenConsumer module if option is enabled
    NVGLDEMO_EGL_EXT_EXEC_QNX_SCREEN_CONSUMER_NORET_API(demoOptions.eglQnxScreenTest,
        NULL, NULL);

    buttonCB = cb;
    NvGlDemoUpdateEventMask();
}

// Add keys here, that are used in demo apps.
static unsigned char GetKeyPress(int *screenKey)
{
    unsigned char key = '\0';

    switch (*screenKey) {

        case KEYCODE_F1:
            key = NvGlDemoKeyCode_F1;
        break;
        case KEYCODE_LEFT:
            key = NvGlDemoKeyCode_LeftArrow;
        break;
        case KEYCODE_RIGHT:
            key = NvGlDemoKeyCode_RightArrow;
        break;
        case KEYCODE_UP:
            key = NvGlDemoKeyCode_UpArrow;
        break;
        case KEYCODE_DOWN:
            key = NvGlDemoKeyCode_DownArrow;
        break;
        case KEYCODE_HOME:
            key = NvGlDemoKeyCode_Home;
        break;
        case KEYCODE_END:
            key = NvGlDemoKeyCode_End;
        break;
        case KEYCODE_PG_UP:
            key = NvGlDemoKeyCode_PageUp;
        break;
        case KEYCODE_PG_DOWN:
            key = NvGlDemoKeyCode_PageDown;
        break;
        case KEYCODE_ESCAPE:
            key = NvGlDemoKeyCode_Escape;
        break;
        case KEYCODE_DELETE:
            key = NvGlDemoKeyCode_Delete;
        break;
        default:
            /* For "normal" keys, Screen KEYCODE is just ASCII. */
            if (*screenKey <= 127) {
                key = *screenKey;
            }
        break;
    }

    return key;
}

void NvGlDemoCheckEvents(void)
{

    static int vis = 1, val = 1;
    int rc;
    int pos[2];

    // Call the eglQnxScreenConsumer module if option is enabled
    NVGLDEMO_EGL_EXT_EXEC_QNX_SCREEN_CONSUMER_NORET_API(demoOptions.eglQnxScreenTest,
        NULL, NULL);

    /**
     ** We start the loop by processing any events that might be in our
     ** queue. The only event that is of interest to us are the resize
     ** and close events. The timeout variable is set to 0 (no wait) or
     ** forever depending if the window is visible or invisible.
     **/

    while (!screen_get_event(demoState.platform->screen_ctx,
                demoState.platform->screen_ev, vis ? 0ull : ~0ull)) {

           // Get QNX CAR 2.1 event property
           rc = screen_get_event_property_iv(demoState.platform->screen_ev,
                   SCREEN_PROPERTY_TYPE, &val);
           if (rc || val == SCREEN_EVENT_NONE) {
               break;
           }

           switch (val) {

               case SCREEN_EVENT_CLOSE:
                    /**
                     ** All we have to do when we receive the close event is
                     ** exit the application loop.
                     **/
                    if (closeCB) {
                        closeCB();
                    }
                    break;

               case SCREEN_EVENT_PROPERTY:
                    /**
                     ** We are interested in visibility changes so we can pause
                     ** or unpause the rendering.
                     **/
                    rc = screen_get_event_property_iv(demoState.platform->screen_ev, SCREEN_PROPERTY_NAME, &val);
                    if (rc || val == SCREEN_EVENT_NONE) {
                        break;
                    }
                    switch (val) {

                        case SCREEN_PROPERTY_VISIBLE:
                            /**
                             ** The new visibility status is not included in the
                             ** event, so we must get it ourselves.
                             **/
                            rc = screen_get_window_property_iv(demoState.platform->screen_win, SCREEN_PROPERTY_VISIBLE, &vis);
                            if (rc) {
                                NvGlDemoLog("Failed to get window property SCREEN_PROPERTY_VISIBLE.\n");
                            }
                            break;

                        default:
                            break;
                    }
                    break;

               case SCREEN_EVENT_POINTER:
                    /**
                     ** The event will come as a screen pointer event, with an (x,y)
                     ** coordinate relative to the window's upper left corner and a
                     ** select value.
                     **/
                    rc = screen_get_event_property_iv(demoState.platform->screen_ev, SCREEN_PROPERTY_POSITION, pos);
                    if (!rc) {
                        if (pointerCB) {
                            pointerCB(pos[0], pos[1]);
                        }
                    }
                    rc = screen_get_event_property_iv(demoState.platform->screen_ev, SCREEN_PROPERTY_BUTTONS, &val);
                    if (!rc) {
                        if (buttonCB) {
                            buttonCB(val, 1);
                        }
                    }
                    break;

               case SCREEN_EVENT_KEYBOARD:
#if _SCREEN_VERSION < 20000
                    rc = screen_get_event_property_iv(demoState.platform->screen_ev, SCREEN_PROPERTY_KEY_FLAGS, &val);
#else
                    rc = screen_get_event_property_iv(demoState.platform->screen_ev, SCREEN_PROPERTY_FLAGS, &val);
#endif
                    if (rc || val == SCREEN_EVENT_NONE) {
                        break;
                    }
                    if (val & KEY_DOWN) {
#if _SCREEN_VERSION < 20000
                        rc = screen_get_event_property_iv(demoState.platform->screen_ev, SCREEN_PROPERTY_KEY_SYM, &val);
#else
                        rc = screen_get_event_property_iv(demoState.platform->screen_ev, SCREEN_PROPERTY_SYM, &val);
#endif
                        if (rc || val == SCREEN_EVENT_NONE) {
                            break;
                        }
                        unsigned char key;
                        key = GetKeyPress(&val);
                        if (key != '\0') {
                            keyCB(key, 1);
                        }
                    }
                    break;

               default:
                    break;
           }
    }
}

void NvGlDemoWaitEvents(void)
{
    // TODO. should call screen_get_event(... ,-1)
    NvGlDemoCheckEvents();
}

EGLBoolean NvGlDemoSwapInterval(EGLDisplay dpy, EGLint interval)
{
    // Call the eglQnxScreenConsumer module if option is enabled
    NVGLDEMO_EGL_EXT_EXEC_QNX_SCREEN_CONSUMER_RET_API(demoOptions.eglQnxScreenTest,
        NULL, EGL_TRUE, EGLBoolean,"eglqnxscreen - Please use -postflags <interval> option to configure swapinterval\n");

    return eglSwapInterval(dpy, interval);
}

#ifdef EGL_EXT_stream_consumer_qnxscreen_window
//
//eglQnxScreenConsumer startup/shutdown
//

// Initialize access to the eglQnxScreenConsumer
static int eglQnxScreenConsumerInit(void)
{

    if (demoOptions.displayName[0]) {
        NvGlDemoLog("Setting display output is not supported. Exiting.\n");
        goto fail;
    }

    if (demoOptions.displayRate) {
        NvGlDemoLog("Setting display refresh is not supported. Exiting.\n");
        goto fail;
    }

    if (demoOptions.displayBlend >= NvGlDemoDisplayBlend_None) {
        NvGlDemoLog("Display Blend is not supported. Exiting.\n");
        goto fail;
    }

    if (demoOptions.displayAlpha >= 0.0) {
        NvGlDemoLog("Display Alpha is not supported. Exiting.\n");
        goto fail;
    }

    if (demoOptions.displayColorKey[0] >= 0.0) {
        NvGlDemoLog("Display Color Key is not supported. Exiting.\n");
        goto fail;
    }

    // Allocate a structure for the platform-specific state
    peglQnxScreenInfo = (NvGlDemoEglQnxScreenInfo*) MALLOC(
            sizeof(NvGlDemoEglQnxScreenInfo));
    if (!peglQnxScreenInfo) {
        NvGlDemoLog("Could not allocate platform specific storage memory.\n");
        goto fail;
    }
    MEMSET(peglQnxScreenInfo, 0, sizeof(NvGlDemoEglQnxScreenInfo));

    return 1;

fail:
    NvGlDemoDisplayTerm();
    return 0;
}

// Terminate access to the eglQnxScreenConsumer
static void eglQnxScreenConsumerTerm(void)
{
    if (!peglQnxScreenInfo) {
        return;
    }

    NvGlDemoWindowTerm();

    peglCreateStreamKHR = NULL;
    peglQueryStreamKHR = NULL;
    peglDestroyStreamKHR = NULL;
    peglStreamConsumerQNXScreenWindowEXT = NULL;
    peglStreamConsumerAcquireAttribEXT = NULL;

    FREE(peglQnxScreenInfo);
    peglQnxScreenInfo = NULL;
}

// Start the eglQnxScreenConsumer
static int eglQnxScreenConsumerStart(void)
{
    const char* extensions = NULL;

    if (!peglQnxScreenInfo) {
        return 0;
    }

    // Get extension string
    extensions = eglQueryString(demoState.display, EGL_EXTENSIONS);
    if (!extensions) {
        NvGlDemoLog("eglQueryString fail.\n");
        goto fail;
    }

    if (!strstr(extensions, "EGL_EXT_stream_consumer_qnxscreen_window")) {
        NvGlDemoLog("EGL does not support EGL_EXT_stream_consumer_qnxscreen_window extension.\n");
        goto fail;
    }

    NVGLDEMO_EGL_GET_PROC_ADDR(eglCreateStreamKHR,                  fail, PFNEGLCREATESTREAMKHRPROC);
    NVGLDEMO_EGL_GET_PROC_ADDR(eglQueryStreamKHR,                   fail, PFNEGLQUERYSTREAMKHRPROC);
    NVGLDEMO_EGL_GET_PROC_ADDR(eglDestroyStreamKHR,                 fail, PFNEGLDESTROYSTREAMKHRPROC);
    NVGLDEMO_EGL_GET_PROC_ADDR(eglStreamConsumerQNXScreenWindowEXT, fail, PFNEGLSTREAMCONSUMERQNXSCREENWINDOWEXTPROC);
    NVGLDEMO_EGL_GET_PROC_ADDR(eglStreamConsumerAcquireAttribEXT,   fail, PFNEGLSTREAMCONSUMERACQUIREATTRIBEXTPROC);

    // If not specified, use default window size
    if (!demoOptions.windowSize[0])
        demoOptions.windowSize[0] = NVGLDEMO_DEFAULT_WIDTH;
    if (!demoOptions.windowSize[1])
        demoOptions.windowSize[1] = NVGLDEMO_DEFAULT_HEIGHT;

    // Create Synchonization semaphore
    if ((peglQnxScreenInfo->sem = NvGlDemoSemaphoreCreate(0,0)) == NULL) {
        NvGlDemoLog("Failed to create synchonization semaphore.\n");
        goto fail;
    }

    // Create Consumer Thread
    if ((peglQnxScreenInfo->thread = NvGlDemoThreadCreate(ThreadEntryPoint, NULL)) == NULL) {
        NvGlDemoLog("Failed to create consumer thread.\n");
        goto fail;
     }

    // Ensure eglStream and consumer connection established
    if (NvGlDemoSemaphoreWait(peglQnxScreenInfo->sem) != 0) {
        NvGlDemoLog("NvGlDemoSemaphoreWait-failed.\n");
        goto fail;
     }

    if ((demoState.stream == EGL_NO_STREAM_KHR) || (peglQnxScreenInfo->state != THREAD_RUN)) {
        NvGlDemoLog("Failed to create consumer connection.\n");
        goto fail;
     }

    return 1;

fail:
    NvGlDemoWindowTerm();
    return 0;
}

// Stop the eglQnxScreenConsumer
static void eglQnxScreenConsumerStop(void)
{
    if ((!peglQnxScreenInfo) || (peglQnxScreenInfo->state != THREAD_RUN)) {
        return;
    }

    peglQnxScreenInfo->state = THREAD_DESTROY;

    if (NvGlDemoThreadJoin(peglQnxScreenInfo->thread, NULL) != 0) {
        NvGlDemoLog("Thread Join fail - Consumer thread not yet exit.\n");
    }

    if (NvGlDemoSemaphoreWait(peglQnxScreenInfo->sem) != 0) {
        NvGlDemoLog("NvGlDemoSemaphoreWait-failed.\n");
    }

    if (peglQnxScreenInfo->state != THREAD_EXIT) {
        NvGlDemoLog("Resource Leak - Consumer thread not yet exit[%d].\n",peglQnxScreenInfo->state);
    }

    if (NvGlDemoSemaphoreDestroy(peglQnxScreenInfo->sem) != 0) {
        NvGlDemoLog("Resource Leak - sem destroy failed.\n");
    }

    peglQnxScreenInfo->state = THREAD_NONE;
}

static int eglQnxScreenConsumerConnect(void)
{
    EGLint attr[MAX_EGL_STREAM_ATTR * 2 + 1];
    int attrIdx = 0;
    EGLBoolean eglStatus;
    EGLint streamState = EGL_STREAM_STATE_EMPTY_KHR;
    EGLBoolean ret     = EGL_FALSE;
    EGLint fifo_length = 0, latency = 0, timeout = 0;
    EGLAttrib attrib_list[MAX_EGL_STREAM_ATTR * 2 + 1] = {EGL_NONE};

    if (demoState.display != EGL_NO_DISPLAY) {

        attr[attrIdx++] = EGL_STREAM_FIFO_LENGTH_KHR;
        attr[attrIdx++] = demoOptions.nFifo;

        attr[attrIdx++] = EGL_CONSUMER_LATENCY_USEC_KHR;
        attr[attrIdx++] = demoOptions.latency;
        if (demoOptions.flags & NVGL_DEMO_OPTION_TIMEOUT) {
            attr[attrIdx++] = EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR;
            attr[attrIdx++] = demoOptions.timeout;
        }
        attr[attrIdx++] = EGL_NONE;

        demoState.stream = peglCreateStreamKHR(demoState.display, attr);

        if (demoState.stream == EGL_NO_STREAM_KHR) {
            NvGlDemoLog("eglCreateStreamKHR failed\n");
            goto fail;
         }

        ret = peglQueryStreamKHR(demoState.display, demoState.stream, EGL_STREAM_STATE_KHR, &streamState);
        if (!ret) {
            NvGlDemoLog("Could not query EGL stream state\n");
            goto fail;
        }

        if (!(streamState == EGL_STREAM_STATE_CREATED_KHR)) {
            NvGlDemoLog("EGL stream is not in valid starting state\n");
            goto fail;
        }

        // Get stream attributes
        if(!peglQueryStreamKHR(demoState.display, demoState.stream, EGL_STREAM_FIFO_LENGTH_KHR, &fifo_length)) {
            NvGlDemoLog("Consumer: eglQueryStreamKHR EGL_STREAM_FIFO_LENGTH_KHR failed\n");
        }
        if(!peglQueryStreamKHR(demoState.display, demoState.stream, EGL_CONSUMER_LATENCY_USEC_KHR, &latency)) {
            NvGlDemoLog("Consumer: eglQueryStreamKHR EGL_CONSUMER_LATENCY_USEC_KHR failed\n");
        }
        if(!peglQueryStreamKHR(demoState.display, demoState.stream, EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR, &timeout)) {
            NvGlDemoLog("Consumer: eglQueryStreamKHR EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR failed\n");
        }

        NvGlDemoLog("EGL Stream Info: fifo_length - %d latency - %d timeout - %d postWindowFlags - %d dispno - %d layerno - %d surfaceType - %d display_pos - [%d %d] display_sz[%d %d]\n",
            fifo_length, latency, timeout, demoOptions.postWindowFlags, demoOptions.displayNumber, demoOptions.displayLayer, demoOptions.surfaceType,demoOptions.windowOffset[0],demoOptions.windowOffset[1], demoOptions.displaySize[0],demoOptions.displaySize[1]);

        attrIdx = 0;

        attrib_list[attrIdx++] = EGL_CONSUMER_ACQUIRE_QNX_FLUSHING_EXT;
        attrib_list[attrIdx++] = demoOptions.postWindowFlags;

        attrib_list[attrIdx++] = EGL_CONSUMER_ACQUIRE_QNX_DISPNO_EXT;
        attrib_list[attrIdx++] = demoOptions.displayNumber;

        attrib_list[attrIdx++] = EGL_CONSUMER_ACQUIRE_QNX_LAYERNO_EXT;
        attrib_list[attrIdx++] = demoOptions.displayLayer;

        attrib_list[attrIdx++] = EGL_CONSUMER_ACQUIRE_QNX_SURFACE_TYPE_EXT;
        attrib_list[attrIdx++] = demoOptions.surfaceType;

        attrib_list[attrIdx++] = EGL_CONSUMER_ACQUIRE_QNX_DISPLAY_POS_X_EXT;
        attrib_list[attrIdx++] = demoOptions.windowOffset[0];

        attrib_list[attrIdx++] = EGL_CONSUMER_ACQUIRE_QNX_DISPLAY_POS_Y_EXT;
        attrib_list[attrIdx++] = demoOptions.windowOffset[1];

        attrib_list[attrIdx++] = EGL_CONSUMER_ACQUIRE_QNX_DISPLAY_WIDTH_EXT;
        attrib_list[attrIdx++] = demoOptions.displaySize[0];

        attrib_list[attrIdx++] = EGL_CONSUMER_ACQUIRE_QNX_DISPLAY_HEIGHT_EXT;
        attrib_list[attrIdx++] = demoOptions.displaySize[1];

        attrib_list[attrIdx++] = EGL_NONE;

        // Attach ourselves to eglStream as a QNX screen window consumer.
        eglStatus = peglStreamConsumerQNXScreenWindowEXT(demoState.display,
                                                         demoState.stream,
                                                         attrib_list);

        if (eglStatus != EGL_TRUE) {
            NvGlDemoLog("eglqnxscreen window connection is establishment failed[%d]\n",eglGetError());
            goto fail;
         }

        return 1;
    }

fail:
   return 0;
}

static inline int eglQnxScreenConsumerAcquireFrame(void)
{
    EGLint streamState = 0;
    EGLBoolean eglStatus;

    if ((demoState.display != EGL_NO_DISPLAY) && (demoState.stream != EGL_NO_STREAM_KHR)) {

        eglStatus = peglQueryStreamKHR(demoState.display, demoState.stream,
                                      EGL_STREAM_STATE_KHR, &streamState);

        if (streamState == EGL_STREAM_STATE_DISCONNECTED_KHR) {
            //NvGlDemoLog("EGL qnx screen Consumer: EGL_STREAM_STATE_DISCONNECTED_KHR received\n");
            goto fail;
        }

        // Producer hasn't generated any frames yet
        if ((streamState != EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR) &&
            (streamState != EGL_STREAM_STATE_OLD_FRAME_AVAILABLE_KHR)) {
            if (NvGlDemoThreadYield() != 0) {
                NvGlDemoLog("NvGlDemoThreadYield-failed\n");
            }
            goto success;
        }

        eglStatus = peglStreamConsumerAcquireAttribEXT(demoState.display,
                                                       demoState.stream, NULL);
        if (!eglStatus) {
            //NvGlDemoLog("Missed frame\n");
            if (NvGlDemoThreadYield() != 0) {
                NvGlDemoLog("NvGlDemoThreadYield-failed\n");
            }
        }

success:
        return 1;
    }

fail:
   return 0;
}

static inline void eglQnxScreenConsumerDisconnect(void)
{
    if ((demoState.display != EGL_NO_DISPLAY) && (demoState.stream != EGL_NO_STREAM_KHR)) {
        peglDestroyStreamKHR(demoState.display, demoState.stream);
        demoState.stream = EGL_NO_STREAM_KHR;
     }

   return;
}

static void *ThreadEntryPoint(void *arg) {

    peglQnxScreenInfo->state = THREAD_CREATED;

    if (eglQnxScreenConsumerConnect() == 1) {

        peglQnxScreenInfo->state = THREAD_RUN;

        if (NvGlDemoSemaphorePost(peglQnxScreenInfo->sem) != 0) {
            NvGlDemoLog("NvGlDemoSemaphorePost-failed\n");
        }

        while (peglQnxScreenInfo->state == THREAD_RUN) {
            if (eglQnxScreenConsumerAcquireFrame() != 1) {
                break;
            }
        }
        eglQnxScreenConsumerDisconnect();
    }

    peglQnxScreenInfo->state = THREAD_EXIT;

    if (NvGlDemoSemaphorePost(peglQnxScreenInfo->sem) != 0) {
        NvGlDemoLog("NvGlDemoSemaphorePost-failed\n");
    }
    return NULL;
}

#endif //EGL_EXT_stream_consumer_qnxscreen_window

