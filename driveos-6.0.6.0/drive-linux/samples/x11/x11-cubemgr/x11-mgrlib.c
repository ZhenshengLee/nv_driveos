/*
 * x11-mgrlib.c
 *
 * Copyright (c) 2012-2012, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

// Utility functions for some basic window management.

//============================================================================

// Demo utilities
#include "nvgldemo.h"
#include "nvgldemo_win_x11.h"

// EGL/GLES2 and extensions
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>

// X11 headers
#include <X11/Xlibint.h>

// X11 extensions
#include <X11/extensions/Xcomposite.h>
#include <X11/extensions/Xdamage.h>

// Window manager utilities
#include "x11-mgrlib.h"

// NVIDIA logo texture data
static const char *logo =
#   include "nvidia.h"
;

static Atom         cmAtom = None;

//Event & Error base values for X11 extensions
static int          compositeBaseEvent = 0;
static int          compositeBaseError = 0;
int                 damageBaseEvent = 0;
static int          damageBaseError = 0;

// Extension function pointers
static PFNEGLCREATEIMAGEKHRPROC            pEglCreateImageKHR = NULL;
static PFNEGLDESTROYIMAGEKHRPROC           pEglDestroyImageKHR = NULL;
static PFNGLEGLIMAGETARGETTEXTURE2DOESPROC pGlEGLImageTargetTexture2DOES = NULL;

// Texture for nvidia logo
GLuint              logoTex = 0;

// Buffer queue - buffers used by the compositor for rendering
// Fixed size - size is specified by the compositor
MgrObject          *bufferList = NULL;
static int          bufferCount = 0;

// Window queue - windows waiting to be composited
// resizable linked list
static WindowObj   *nextWindow = NULL;
static WindowObj   *lastWindow = NULL;

//============================================================================

// Print X11 errors
static int PrintError(Display *dpy, XErrorEvent *event)
{
    char buffer[256];
    char msg[256];
    _XExtension *ext = NULL;

    XGetErrorText(dpy, event->error_code, buffer, sizeof(buffer));
    NvGlDemoLog("\nX Error: %d %s\n", event->error_code, buffer);

    if (event->request_code < 128) {
        snprintf(msg, sizeof(msg), "%d", event->request_code);
        XGetErrorDatabaseText(dpy, "XRequest", msg, "unknown", buffer,
                              sizeof(buffer));
    } else {
        ext = dpy->ext_procs;
        while (ext) {
            if (ext->codes.major_opcode == event->request_code) {
                snprintf(buffer, sizeof(buffer), "%s", ext->name);
                break;
            }
            ext = ext->next;
        }

    }
    NvGlDemoLog("Major opcode: %d (%s)\n", event->request_code, buffer);

    if (ext) {
        snprintf(msg, sizeof(msg), "%s.%d", ext->name, event->minor_code);
        XGetErrorDatabaseText(dpy, "XRequest", msg, "unknown", buffer,
                              sizeof(buffer));
        NvGlDemoLog("Minor opcode: %d (%s)\n", event->minor_code, buffer);
    }

    NvGlDemoLog("Resource id: 0x%0x, Serial: %d\n\n", event->resourceid,
                event->serial);

    // Exit if XCompositeRedirectSubwindows or XCompositeUnredirectWindow failed
    // This is a fatal situation, most likely to happen if another manager
    // (like metacity) is running
    if (ext && !strcmp(ext->name, "Composite") &&
        (event->minor_code == 2 || event->minor_code == 3)) {
        NvGlDemoLog("Fatal Error! Maybe another composite manager is already "
                    "running. Exiting!\n");
        exit(1);
    }

    return 0;
}

// Initialize resources used by this library
int
MgrLibInit(void)
{
    const char     *extensions;
    Window          rootWindow = None;

    XSetErrorHandler(PrintError);

    // TODO: This does not seem to be working with metacity, unable to detect
    // TODO: This could possibly use the XScreen number instead of 0
    cmAtom = XInternAtom(demoState.platform->XDisplay, "_NET_WM_CM_S0", 0);
    if (XGetSelectionOwner(demoState.platform->XDisplay, cmAtom) != None) {
        NvGlDemoLog("Another composite manager is already running\n");
        return 0;
    } else {
        XSetSelectionOwner(demoState.platform->XDisplay, cmAtom,
                           demoState.platform->XWindow, CurrentTime);
    }

    if (!XCompositeQueryExtension(demoState.platform->XDisplay,
                                  &compositeBaseEvent, &compositeBaseError)) {
        NvGlDemoLog("Composite extension not supported\n");
        return 0;
    }

    if (!XDamageQueryExtension(demoState.platform->XDisplay,
                               &damageBaseEvent, &damageBaseError)) {
        NvGlDemoLog("Damage extension not supported\n");
        return 0;
    }

    rootWindow = RootWindow(demoState.platform->XDisplay,
                            demoState.platform->XScreen);
    XCompositeRedirectSubwindows(demoState.platform->XDisplay, rootWindow,
                                 CompositeRedirectManual);
    XCompositeUnredirectWindow(demoState.platform->XDisplay,
                               demoState.platform->XWindow,
                               CompositeRedirectManual);
    XSelectInput(demoState.platform->XDisplay, rootWindow,
                 SubstructureNotifyMask);

    // Get EGL extension list
    extensions = eglQueryString(demoState.display, EGL_EXTENSIONS);

    // Check for top level EGLImage extension
    if (!STRSTR(extensions, "EGL_KHR_image")) {
        NvGlDemoLog("EGL_KHR_image extension not supported\n");
        return 0;
    }

    // Get GL extension list
    extensions = (const char*)glGetString(GL_EXTENSIONS);

    // Check for extension that allows creation of 2D texture from EGLImage
    if (!STRSTR(extensions, "GL_OES_EGL_image")) {
        NvGlDemoLog("GL_OES_EGL_image extension not supported\n");
        return 0;
    }

    // Obtain image extension function pointers
    pEglCreateImageKHR =
        (PFNEGLCREATEIMAGEKHRPROC)
            eglGetProcAddress("eglCreateImageKHR");

    pEglDestroyImageKHR =
        (PFNEGLDESTROYIMAGEKHRPROC)
            eglGetProcAddress("eglDestroyImageKHR");

    pGlEGLImageTargetTexture2DOES =
        (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)
            eglGetProcAddress("glEGLImageTargetTexture2DOES");

    // Make sure all functions were found
    if (!pEglCreateImageKHR
        || !pEglDestroyImageKHR
        || !pGlEGLImageTargetTexture2DOES) {
        NvGlDemoLog("Failed to get all EGLImage function pointers\n");
        return 0;
    }

    // Create a texture with the NVIDIA logo
    glGenTextures(1, &logoTex);
    glBindTexture(GL_TEXTURE_2D, logoTex);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 420, 420, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, logo);

    return 1;
}

// Release resources used by this library
void
MgrLibTerm(void)
{
    WindowObj      *temp = nextWindow;
    WindowObj      *remove = NULL;

    while (temp) {
        remove = temp;
        temp = temp->next;
        FREE(remove);
    }

    nextWindow = lastWindow = NULL;

    if (demoState.platform && demoState.platform->XDisplay) {
        if (XGetSelectionOwner(demoState.platform->XDisplay, cmAtom) ==
                               demoState.platform->XWindow) {
            XSetSelectionOwner(demoState.platform->XDisplay, cmAtom,
                               None, CurrentTime);
        }
    }

    if (logoTex) {
        glDeleteTextures(1, &logoTex);
        logoTex = 0;
    }
}

// Initialize object list
int
MgrObjectListInit(
    int             alloc)
{
    ASSERT(alloc > 0);
    bufferList = (MgrObject*)MALLOC(alloc * sizeof(MgrObject));
    if (!bufferList) return 0;
    MEMSET(bufferList, 0, alloc * sizeof(MgrObject));
    bufferCount = alloc;
    return 1;
}

// Terminate object list, destroying all objects
void
MgrObjectListTerm(void)
{
    if (bufferList) FREE(bufferList);
    bufferCount = 0;
}

// Create egl image from a X11 window pixmap
void CreateEglImage(MgrObject *obj)
{
    Pixmap              pixmap = None;
    XWindowAttributes   winAtt;
    Status              rvint;

    rvint = XGetWindowAttributes(demoState.platform->XDisplay,
                                 obj->window, &winAtt);
    if (!rvint) {
        NvGlDemoLog("Failed to retrieve window size\n");
        return;
    }

    pixmap = XCompositeNameWindowPixmap(demoState.platform->XDisplay,
                                        obj->window);
    if (pixmap == None) {
        NvGlDemoLog("Could not create pixmap\n");
        return;
    }

    obj->eglImage = pEglCreateImageKHR(demoState.display,
                                       EGL_NO_CONTEXT,
                                       EGL_NATIVE_PIXMAP_KHR,
                                       (EGLClientBuffer)pixmap, NULL);

    XFreePixmap(demoState.platform->XDisplay, pixmap);

    if (obj->eglImage == NULL) {
        NvGlDemoLog("Could not create egl image\n");
        return;
    }

    glGenTextures(1, &(obj->texID));
    glBindTexture(GL_TEXTURE_2D, obj->texID);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,    GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,    GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    pGlEGLImageTargetTexture2DOES(GL_TEXTURE_2D, obj->eglImage);

    obj->winSize[0] = winAtt.width;
    obj->winSize[1] = winAtt.height;
    obj->bufSize[0] = winAtt.width;
    obj->bufSize[1] = winAtt.height;
}

// Destroy egl image
static void DestroyEglImage(MgrObject *obj)
{
    if (obj->eglImage) {
        pEglDestroyImageKHR(demoState.display, obj->eglImage);
        obj->eglImage = NULL;
    }

    glDeleteTextures(1, &(obj->texID));
    obj->texID = 0;
    obj->winSize[0] = obj->winSize[1] = 0;
    obj->bufSize[0] = obj->bufSize[1] = 0;
}

// Check if window is not the compositor window
static int IsValidWindow(Window window)
{
    return (window != None && window != demoState.platform->XWindow);
}

// Add window to list of window queue and create damage so that
// we can detect the first eglSwapBuffer from the client
// indicating this window to be ready for compositing
static void CreateWindow(Window window)
{
    WindowObj           *temp = NULL;
    XWindowAttributes   winAtt;
    Status              rvint;

    rvint = XGetWindowAttributes(demoState.platform->XDisplay,
                                 window, &winAtt);
    if (!rvint) {
        NvGlDemoLog("Failed to retrieve window attributes\n");
        return;
    }

    if (winAtt.class == InputOnly) {
        NvGlDemoLog("Ignoring client window, as it is 'InputOnly'\n");
        rvint = False;
    } else if (winAtt.override_redirect == True) {
        NvGlDemoLog("Ignoring client window, as requested by the"
                    " client via 'override_redirect'\n");
        rvint = False;
    }

    if (!rvint) {
        XCompositeUnredirectWindow(demoState.platform->XDisplay, window,
                                   CompositeRedirectManual);
        return;
    }

    if (!XDamageCreate(demoState.platform->XDisplay, window,
                       XDamageReportNonEmpty)) {
        NvGlDemoLog("Could not create damage for window 0x%08x, "
                    "ignoring it\n", window);
        return;
    }

    temp = (WindowObj*)MALLOC(sizeof(WindowObj));
    temp->window = window;
    temp->ready = 0;
    temp->next = NULL;

    if (!nextWindow)
        nextWindow = temp;

    if (lastWindow)
        lastWindow->next = temp;

    lastWindow = temp;
}

// Remove window from the window queue
static void DestroyWindow(Window window)
{
    WindowObj      *temp = nextWindow;
    WindowObj      *remove = NULL;

    if (temp) {
        if (temp->window == window) {
            remove = temp;
            nextWindow = temp->next;
            if (nextWindow == NULL)
                lastWindow = NULL;
        }
        while (temp->next) {
            if (temp->next->window == window) {
                remove = temp->next;
                temp->next = temp->next->next;
                if (lastWindow == remove)
                    lastWindow = temp;
                break;
            }
            temp = temp->next;
        }
    }

    if (remove)
        FREE(remove);
}

// Mark a window ready for compositing
void MarkReady(Window window, Damage damage)
{
    int             i;
    WindowObj      *temp = nextWindow;

    if (!IsValidWindow(window))
        return;

    // Destroy the damage so that we don't get any further damage events
    XDamageDestroy(demoState.platform->XDisplay, damage);

    // Search for this window in the window queue
    // in case this the very first eglSwapBuffer from the client
    while (temp) {
        if (temp->window == window) {
            temp->ready = 1;
            for (i = 0; i < bufferCount; i++) {
                if (bufferList[i].window == None) {
                    bufferList[i].window = window;
                    DestroyWindow(window);
                    return;
                }
            }
            return;
        }
        temp = temp->next;
    }

    // Search for the window in the buffer queue
    // in case this is the first eglSwapBuffer after a window resize
    for (i = 0; i < bufferCount; i++) {
        if (bufferList[i].window == window) {
            DestroyEglImage(&bufferList[i]);
            CreateEglImage(&bufferList[i]);
            return;
        }
    }
}

// Return the next ready window and remove it from the window queue
static Window FindNextReadyWindow(void)
{
    Window          found = None;
    WindowObj      *temp = nextWindow;

    while (temp) {
        if (temp->ready) {
            found = temp->window;
            DestroyWindow(found);
            break;
        }
        temp = temp->next;
    }

    return found;
}

// Add window to the window queue
void MapWindow(Window window)
{
    if (!IsValidWindow(window))
        return;

    CreateWindow(window);
}

// Remove window from the buffer queue & window queue
// Replace with a window that is waiting to be composited in the window queue
void UnmapWindow(Window window)
{
    int             i;

    if (!IsValidWindow(window))
        return;

    for (i = 0; i < bufferCount; i++) {
        if (bufferList[i].window == window) {
            DestroyEglImage(&bufferList[i]);
            bufferList[i].window = FindNextReadyWindow();
            return;
        }
    }

    DestroyWindow(window);
}

// On resize a new X11 pixmap will be allocated so create damage to
// detect the next eglSwapBuffer, till then don't destroy the egl image
// so that we can display the last frame from the client
// If damage creation fails, remove this window
// and find a new window which is ready
void ConfigureWindow(Window window)
{
    int             i;

    if (!IsValidWindow(window))
        return;

    for (i = 0; i < bufferCount; i++) {
        if (bufferList[i].window == window) {
            if (!XDamageCreate(demoState.platform->XDisplay, window,
                            XDamageReportNonEmpty)) {
                NvGlDemoLog("Could not create damage for window 0x%08x, "
                            "removing it\n", window);
                DestroyEglImage(&bufferList[i]);
                bufferList[i].window = FindNextReadyWindow();
                return;
            }
            return;
        }
    }
}
