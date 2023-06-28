/*
 * main.c
 *
 * Copyright (c) 2003-2019, NVIDIA CORPORATION. All rights reserved.
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
// Main loop and event handling
//

#include "nvgldemo.h"
#include "screen.h"

// Flag indicating it is time to shut down
static GLboolean shutdown = GL_FALSE;

// Callback to close window
static void
closeCB(void)
{
    shutdown = GL_TRUE;
}

// Callback to resize window
static void
resizeCB(
    int width,
    int height)
{
    Screen_resize(width, height);
}

// Callback to handle key presses
static void
keyCB(
    char key,
    int  state)
{
    // Ignoring releases
    if (!state) return;

    Screen_callback(key, 0, 0);
}

static void
usage(void)
{
    NvGlDemoLog("Usage: ctree [options]\n"
                "  Put into demo mode:\n"
                "    [-demo]\n"
                "  Use low resolution textures:\n"
                "    [-smalltex]\n"
                "  Disable the menu interface:\n"
                "    [-nomenu]\n"
                "  Disable rendering of the sky:\n"
                "    [-nosky]\n"
                "  Turn on framerate logging:\n"
                "    [-fps]\n");
    NvGlDemoLog(NvGlDemoArgUsageString());
}

// Entry point of this demo program.
int main(int argc, char **argv)
{
    int         failure = 1;
    GLboolean   fpsFlag  = GL_FALSE;
    GLboolean   demoMode = GL_FALSE;
    GLboolean   startup  = GL_FALSE;

    // Initialize window system and EGL
    if (!NvGlDemoInitialize(&argc, argv, "ctree", 2, 8, 0)) {
        goto done;
    }

    // Parse non-generic command line options
    while (argc > 1) {
        // Demo mode
        if (NvGlDemoArgMatch(&argc, argv, 1, "-demo")) {
            demoMode = GL_TRUE;
        }

        // Use small textures
        else if (NvGlDemoArgMatch(&argc, argv, 1, "-smalltex")) {
            Screen_setSmallTex();
        }

        // Disable menu
        else if (NvGlDemoArgMatch(&argc, argv, 1, "-nomenu")) {
            Screen_setNoMenu();
        }

        // Disable sky rendering
        else if (NvGlDemoArgMatch(&argc, argv, 1, "-nosky")) {
            Screen_setNoSky();
        }

        // FPS output
        else if (NvGlDemoArgMatch(&argc, argv, 1, "-fps")) {
            fpsFlag = GL_TRUE;
        }

        // Unknown or failure
        else {
            if (!NvGlDemoArgFailed())
                NvGlDemoLog("Unknown command line option (%s)\n", argv[1]);
            goto done;
        }
    }

    // Parsing succeeded
    startup = GL_TRUE;

    // Set up callbacks
    NvGlDemoSetCloseCB(closeCB);
    NvGlDemoSetResizeCB(resizeCB);
    NvGlDemoSetKeyCB(keyCB);

    // Initialize the rendering
    if (!Screen_initialize(demoOptions.duration, demoState.width,
                           demoState.height, fpsFlag)) {
        Screen_deinitialize();
        goto done;
    }

    if (demoMode) { Screen_setDemoParams(); }

    // Initialize PreSwap functions
    if (!NvGlDemoPreSwapInit()) {
        goto done;
    }

    // Render until time limit is reached or application is terminated
    do {
        // Render the next frame
        Screen_draw();

        // Execute PreSwap functions
        if (!NvGlDemoPreSwapExec()) {
            goto done;
        }

        // Swap the next frame
        if (eglSwapBuffers(demoState.display, demoState.surface) != EGL_TRUE) {
            if (demoState.stream) {
                NvGlDemoLog("Consumer has disconnected, exiting.");
            }
            goto done;
        }

        if (!NvGlDemoPostSwap()) {
            NvGlDemoLog("Failed post swap buffers processing");
            goto done;
        }

        // Process any window system events
        NvGlDemoCheckEvents();
    } while (!Screen_isFinished() && !shutdown);

    // Success
    failure = 0;

    done:

    // Clean up the rendering resources
    Screen_deinitialize();

    // If basic startup failed, print usage message in case it was due
    //   to bad command line arguments.
    if (!startup) {
        usage();
    }

    // Free PreSwap function resources
    NvGlDemoPreSwapShutdown();

    // Clean up EGL and window system
    NvGlDemoShutdown();

    return failure;
}
