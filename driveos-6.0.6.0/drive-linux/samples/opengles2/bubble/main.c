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
// Main bubble demo function and event callbacks
//

#include "nvgldemo.h"
#include "bubble.h"

// Bubble state info
static BubbleState bubbleState;

// Flag indicating it is time to shut down
static GLboolean   shutdown = GL_FALSE;

// Callback to close window
static void
closeCB(void)
{
    shutdown = GL_TRUE;
}

// Callback to resize window
static void
resizeCB(int width, int height)
{
    BubbleState_reshapeViewport(&bubbleState, width, height);
}

// Callback to handle key presses
static void
keyCB(char key, int state)
{
    // Ignoring releases
    if (!state) return;

    BubbleState_callback(&bubbleState, key);
}

#ifndef USE_FAKE_MOUSE
// Callback to move mouse
static void
pointerCB(int x, int y)
{
    BubbleState_mouse(&bubbleState, x, y);
}

// Callback to handle button presses
static void
buttonCB(int button, int state)
{
    if (button != 1) return; // Only use left mouse
    BubbleState_leftMouse(&bubbleState, (bool)state);
}
#endif // USE_FAKE_MOUSE

static void
usage(void)
{
    NvGlDemoLog("Usage: bubble [options]\n"
                "  Frequency with which to automatically poke bubble:\n"
                "    [-autopoke <frames>]\n"
                "  Turn on framerate logging:\n"
                "    [-fps]\n"
                "  To run in Fixed-Frame Increment mode"
                " (Fixed increment of frames):\n"
                "    [-ff]\n"
                "  Delta used in Fixed-Frame Increment which defines"
                " amount of change in scene:\n"
                "    [-delta <value>]\n");
    NvGlDemoLog(NvGlDemoArgUsageString());
}

// Main entry point
int main(int argc, char **argv)
{
    int         failure = 1;
    int         autopoke = 0;
    int         fpsflag  = 0;
    int         startup  = 0;
    bool        fframeIncrement   = false;
    float       delta             = -1.0f;

    // Initialize window system and EGL
    if (!NvGlDemoInitialize(&argc, argv, "bubble", 2, 8, 0)) {
        goto done;
    }

    // Parse non-generic command line options
    while (argc > 1) {
        // Auto-poking
        if (NvGlDemoArgMatchInt(&argc, argv, 1, "-autopoke",
                                     "<frames>", 1, 65535,
                                     1, &autopoke)) {
            // No additional action needed
        }

        // FPS output
        else if (NvGlDemoArgMatch(&argc, argv, 1, "-fps")) {
            fpsflag = 1;
        }

        // Fixed frame
        else if (NvGlDemoArgMatch(&argc, argv, 1, "-ff")) {
            fframeIncrement = true;
        }

        // Delta used in Fixed-frame
        else if (NvGlDemoArgMatchFlt(&argc, argv, 1, "-delta",
                                     "<value>", 0.0f, 31536000.0f,
                                     1, &delta)) {
            // No additional action needed
        }

        // Unknown or failure
        else {
            if (!NvGlDemoArgFailed())
                NvGlDemoLog("Unknown command line option (%s)\n", argv[1]);
            goto done;
        }
    }

    // Parsing succeeded
    startup = 1;

    // Set up callbacks
    NvGlDemoSetCloseCB(closeCB);
    NvGlDemoSetResizeCB(resizeCB);
    NvGlDemoSetKeyCB(keyCB);
#ifndef USE_FAKE_MOUSE
    NvGlDemoSetPointerCB(pointerCB);
    NvGlDemoSetButtonCB(buttonCB);
#endif

    // Initialize bubble state
    if (!BubbleState_init(&bubbleState,
                      autopoke, demoOptions.duration, fpsflag, fframeIncrement,
                      delta, demoState.width, demoState.height)) {
        goto done;
    }

    // Initialize PreSwap functions
    if (!NvGlDemoPreSwapInit()) {
        goto done;
    }

    // Render until time limit is reached or application is terminated
    do {
        // Render the next frame
        BubbleState_tick(&bubbleState);

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

    } while (!bubbleState.drawFinal && !shutdown);

    // If time limit reached, indicate that's why we're exiting
    if ((demoOptions.duration > 0.0f) && bubbleState.drawFinal) {
        NvGlDemoLog("Exiting after %d seconds\n", (int)demoOptions.duration);

    }

    // Success
    failure = 0;

    done:

    // Clean up the bubble resources
    BubbleState_term(&bubbleState);

    // If basic startup failed, print usage message in case it was due
    //   to bad command line arguments.
    if (!startup) {
        usage();
    }

    // Clean up PreSwap resources
    NvGlDemoPreSwapShutdown();

    // Clean up EGL and window system
    NvGlDemoShutdown();

    return failure;
}
