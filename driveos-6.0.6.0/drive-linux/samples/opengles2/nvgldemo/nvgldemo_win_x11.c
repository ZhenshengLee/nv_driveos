/*
 * nvgldemo_win_x11.c
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
// This file illustrates how to access the display and create a window for
//   a GL application using X11.
//

#include "nvgldemo.h"
#include "nvgldemo_win_x11.h"
#include <X11/Xos.h>

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
int
NvGlDemoDisplayInit(void)
{
    // Allocate a structure for the platform-specific state
    demoState.platform =
        (NvGlDemoPlatformState*)MALLOC(sizeof(NvGlDemoPlatformState));
    if (!demoState.platform) {
        NvGlDemoLog("Could not allocate platform specific storage memory.\n");
        goto fail;
    }
    demoState.platform->XDisplay = NULL;
    demoState.platform->XScreen  = 0;
    demoState.platform->XWindow  = (Window)0;

    // Need this for cross-process eglStream capabilities
    if (!XInitThreads()) {
        NvGlDemoLog("XInitThreads Failed. \n");
        goto fail;
    }

    // Open X display
    demoState.platform->XDisplay = XOpenDisplay(NULL);
    if (demoState.platform->XDisplay == NULL) {
        NvGlDemoLog("X failed to open display.\n");
        goto fail;
    }
    demoState.platform->XScreen =
        DefaultScreen(demoState.platform->XDisplay);
    demoState.nativeDisplay =
        (EGLNativeDisplayType)demoState.platform->XDisplay;
    demoState.platformType = NvGlDemoInterface_X11;

    // If display option is specified, but isn't supported, then exit.

    if (demoOptions.displayName[0]) {
        NvGlDemoLog("Setting display output is not supported. Exiting.\n");
        goto fail;
    }

    if (demoOptions.displayRate) {
        NvGlDemoLog("Setting display refresh is not supported. Exiting.\n");
        goto fail;
    }

    if ((demoOptions.displayBlend >= NvGlDemoDisplayBlend_None) ||
        (demoOptions.displayAlpha >= 0.0) ||
        (demoOptions.displayColorKey[0] >= 0.0) ||
        (demoOptions.displayLayer > 0)) {
        NvGlDemoLog("Display layers are not supported. Exiting.\n");
        goto fail;
    }

    // But do not fail on the unsupported "displaysize"
    // option.
    // We will ignore it on X11.

    if (demoOptions.displaySize[0]) {
        NvGlDemoLog("Setting display size is not supported. Ignoring.\n");
    }

    return 1;

    fail:
    FREE(demoState.platform);
    demoState.platform = NULL;

    return 0;
}

// Terminate access to the display system
void
NvGlDemoDisplayTerm(void)
{
    if (demoState.platform) {
        if (demoState.platform->XDisplay) {
            XCloseDisplay(demoState.platform->XDisplay);
            demoState.platform->XDisplay = NULL;
        }

        FREE(demoState.platform);
        demoState.platform = NULL;
        demoState.platformType = NvGlDemoInterface_Unknown;
    }
}

// Create the window
int
NvGlDemoWindowInit(
    int* argc, char** argv,
    const char* appName)
{
    XSizeHints size_hints;
    Atom       wm_destroy_window;

    // If not specified, use default window size
    if (!demoOptions.windowSize[0])
        demoOptions.windowSize[0] = NVGLDEMO_DEFAULT_WIDTH;
    if (!demoOptions.windowSize[1])
        demoOptions.windowSize[1] = NVGLDEMO_DEFAULT_HEIGHT;

    // Create a native window
    demoState.platform->XWindow =
        XCreateSimpleWindow(demoState.platform->XDisplay,
                            RootWindow(demoState.platform->XDisplay,
                                       demoState.platform->XScreen),
                            demoOptions.windowOffset[0],
                            demoOptions.windowOffset[1],
                            demoOptions.windowSize[0],
                            demoOptions.windowSize[1],
                            0,
                            BlackPixel(demoState.platform->XDisplay,
                                       demoState.platform->XScreen),
                            WhitePixel(demoState.platform->XDisplay,
                                       demoState.platform->XScreen));
    if (!demoState.platform->XWindow) {
        NvGlDemoLog("Error creating native window.\n");
        goto fail;
    }
    demoState.nativeWindow = (NativeWindowType)demoState.platform->XWindow;

    // Set up window properties
    size_hints.flags = PPosition | PSize | PMinSize;
    size_hints.x = demoOptions.windowOffset[0];
    size_hints.y = demoOptions.windowOffset[1];
    size_hints.min_width  = (demoOptions.windowSize[0] < 320)
                          ?  demoOptions.windowSize[0] : 320;
    size_hints.min_height = (demoOptions.windowSize[1] < 240)
                          ?  demoOptions.windowSize[1] : 240;
    size_hints.width  = demoOptions.windowSize[0];
    size_hints.height = demoOptions.windowSize[1];

    XSetStandardProperties(demoState.platform->XDisplay,
                           demoState.platform->XWindow,
                           appName, appName, None,
                           argv, *argc, &size_hints);

    XSetWindowBackgroundPixmap(demoState.platform->XDisplay,
                               demoState.platform->XWindow,
                               None);

    wm_destroy_window = XInternAtom(demoState.platform->XDisplay,
                                    "WM_DELETE_WINDOW", False);
    XSetWMProtocols(demoState.platform->XDisplay,
                    demoState.platform->XWindow,
                    &wm_destroy_window,
                    True);

    // Make sure the MapNotify event goes into the queue
    XSelectInput(demoState.platform->XDisplay,
                 demoState.platform->XWindow,
                 StructureNotifyMask);

    XMapWindow(demoState.platform->XDisplay,
               demoState.platform->XWindow);

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
void
NvGlDemoWindowTerm(void)
{
    // Close the native window
    if (demoState.platform && demoState.platform->XWindow) {
        if (demoState.platform->XDisplay) {
        XDestroyWindow(demoState.platform->XDisplay,
                       demoState.platform->XWindow);
        }

        demoState.platform->XWindow = (Window)0;
        demoState.nativeWindow = (NativeWindowType)0;
    }
}

//
// Pixmap support
//

EGLNativePixmapType
NvGlDemoPixmapCreate(
    unsigned int width,
    unsigned int height,
    unsigned int depth)
{
    Pixmap pixmap;
    pixmap = XCreatePixmap(demoState.platform->XDisplay,
                           (Window)demoState.nativeWindow,
                           width, height, depth);
    if (!pixmap)
        NvGlDemoLog("Error creating X11 pixmap.\n");
    return (EGLNativePixmapType)pixmap;
}

void
NvGlDemoPixmapDelete(
    EGLNativePixmapType pixmap)
{
    XFreePixmap(demoState.platform->XDisplay, (Pixmap)pixmap);
}

//
// Callback handling
//
static NvGlDemoCloseCB   closeCB   = NULL;
static NvGlDemoResizeCB  resizeCB  = NULL;
static NvGlDemoKeyCB     keyCB     = NULL;
static NvGlDemoPointerCB pointerCB = NULL;
static NvGlDemoButtonCB  buttonCB  = NULL;
static long              eventMask = 0;

static void NvGlDemoUpdateEventMask(void)
{
    eventMask = StructureNotifyMask;
    if (keyCB) eventMask |= (KeyPressMask | KeyReleaseMask);
    if (pointerCB) eventMask |= (PointerMotionMask | PointerMotionHintMask);
    if (buttonCB) eventMask |= (ButtonPressMask | ButtonReleaseMask);
    if (demoState.platform->XWindow) {
        XSelectInput(demoState.platform->XDisplay,
                     demoState.platform->XWindow,
                     eventMask);
    }
}

void NvGlDemoSetCloseCB(NvGlDemoCloseCB cb)
{
    closeCB  = cb;
    NvGlDemoUpdateEventMask();
}

void NvGlDemoSetResizeCB(NvGlDemoResizeCB cb)
{
    resizeCB = cb;
    NvGlDemoUpdateEventMask();
}

void NvGlDemoSetKeyCB(NvGlDemoKeyCB cb)
{
    keyCB = cb;
    NvGlDemoUpdateEventMask();
}

void NvGlDemoSetPointerCB(NvGlDemoPointerCB cb)
{
    pointerCB = cb;
    NvGlDemoUpdateEventMask();
}

void NvGlDemoSetButtonCB(NvGlDemoButtonCB cb)
{
    buttonCB = cb;
    NvGlDemoUpdateEventMask();
}

void NvGlDemoCheckEvents(void)
{
    XEvent event;
    KeySym key;
    char   str[2] = {0,0};
    Window root, child;
    int    rootX, rootY;
    int    winX, winY;
    unsigned int mask;

    while (XPending(demoState.platform->XDisplay)) {
        XNextEvent(demoState.platform->XDisplay, &event);
        switch (event.type) {

            case ConfigureNotify:
                if (!resizeCB) break;
                if (  (event.xconfigure.width  != demoState.width)
                   || (event.xconfigure.height != demoState.height) ) {
                    resizeCB(event.xconfigure.width, event.xconfigure.height);
                    demoState.width = event.xconfigure.width;
                    demoState.height = event.xconfigure.height;
                }
                break;

            case ClientMessage:
                if (!closeCB) break;
                if (event.xclient.message_type ==
                        XInternAtom(demoState.platform->XDisplay,
                                    "WM_PROTOCOLS", True) &&
                    (Atom)event.xclient.data.l[0] ==
                        XInternAtom(demoState.platform->XDisplay,
                                    "WM_DELETE_WINDOW", True))
                    closeCB();
                break;

            case KeyPress:
            case KeyRelease:
                if (!keyCB) break;
                XLookupString(&event.xkey, str, 2, &key, NULL);
                if (!str[0] || str[1]) break; // Only care about basic keys
                keyCB(str[0], (event.type == KeyPress) ? 1 : 0);
                break;

            case MotionNotify:
                if (!pointerCB) break;
                XQueryPointer(demoState.platform->XDisplay,
                              event.xmotion.window,
                              &root, &child,
                              &rootX, &rootY, &winX, &winY, &mask);
                pointerCB(winX, winY);
                break;

            case ButtonPress:
            case ButtonRelease:
                if (!buttonCB) break;
                buttonCB(event.xbutton.button,
                         (event.type == ButtonPress) ? 1 : 0);
                break;

            case MapNotify:
                // Once the window is mapped, can set the focus
                XSetInputFocus(demoState.platform->XDisplay,
                               event.xmap.window,
                               RevertToNone,
                               CurrentTime);
                break;

           default:
               break;
        }
    }
}

void NvGlDemoWaitEvents(void)
{
    //TODO. should call XNextEvent directly.
    NvGlDemoCheckEvents();
}

EGLBoolean NvGlDemoSwapInterval(EGLDisplay dpy, EGLint interval)
{
    return eglSwapInterval(dpy, interval);
}

