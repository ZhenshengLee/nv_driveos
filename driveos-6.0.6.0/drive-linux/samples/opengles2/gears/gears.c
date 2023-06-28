/*
 * gears.c
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
// This file illustrates a simple rendering program. It uses nvgldemo to
//   initialize the window and graphics context, and gearslib to perform
//   rendering. This is wrapped with a loop that handles window system
//   callbacks and updates the gear rotation angle every frame.
//

#include "nvgldemo.h"
#include "gearslib.h"

// The default number of seconds after which the test will end.
#define TIME_LIMIT 5.0

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
resizeCB(int width, int height)
{
    glViewport(0, 0, width, height);
    gearsResize(width, height);
}

// Callback to handle key presses
static void
keyCB(char key, int state)
{
    // Ignoring releases
    if (!state) return;

    if ((key == 'q') || (key == 'Q')) shutdown = GL_TRUE;
}

static void
usage(void)
{
    NvGlDemoLog("Usage: gears [options] [runtime]\n"
                    "  (negative runtime means \"forever\")\n" );
    NvGlDemoLog("\n  Clear color option:\n"
                    "    [-clearColor <r> <g> <b> <a>] (background color)\n\n");
    NvGlDemoLog(NvGlDemoArgUsageString());
}

// Entry point of this demo program.
int main(int argc, char **argv)
{
    int         failure = 1;
    long long   startTime, currTime, endTime;
    int         runforever = 0;
    int         frames     = 0;
    float       angle      = 0.0;
    float       color[4];
    int         i;

    // Initialize window system and EGL
    if (!NvGlDemoInitialize(&argc, argv, "gears", 2, 8, 0)) {
        goto done;
    }

    // Initialize the gears rendering
    if (!gearsInit(demoState.width, demoState.height)) {
        goto done;
    }

    // Set up callbacks
    NvGlDemoSetCloseCB(closeCB);
    NvGlDemoSetResizeCB(resizeCB);
    NvGlDemoSetKeyCB(keyCB);

    // Querying -clearColor arguments
    if (argc > 2) {
        for (i = 1; i < argc; i++) {
            if (NvGlDemoArgMatchFlt(&argc, argv, i, "-clearColor",
                        "<R G B A> (float)", 0.0f, 1.0f, 4, color)) {
                glClearColor(color[0], color[1], color[2], color[3]);
                break;
            }
        }
    }

    // If -1 wasn't specified, and a duration <= 0.0 was given, it means the
    // duration is the NVGLDemoDefault. For other applications this means "run
    // forever", but to maintain Gears' legacy behavior this is caught and
    // turned into a five second runtime.
    if (demoOptions.duration > -1.0f && demoOptions.duration <= 0.0f) {
        demoOptions.duration = TIME_LIMIT;
    }

    // Allow legacy runtime selection
    if (argc == 2) {
        demoOptions.duration = STRTOF(argv[1], NULL);
        if (!demoOptions.duration) goto done;
    } else if (argc > 2) {
       goto done;
    }

    // Print runtime
    if (demoOptions.duration < 0.0f) {
        runforever = 1;
        NvGlDemoLog(" running forever...\n");
    } else {
        NvGlDemoLog(" running for %f seconds...\n", demoOptions.duration);
    }

    // Initialize PreSwap functions
    if (!NvGlDemoPreSwapInit()) {
        goto done;
    }

    // Allocate the resource if renderahead is specified
    // Get start time and compute end time
    startTime = endTime = currTime = SYSTIME();
    endTime += (long long)(1000000000.0 * demoOptions.duration);

    // Main loop.
    do {
        // Draw a frame
        gearsRender(angle);

        // Execute PreSwap functions
        if (!NvGlDemoPreSwapExec()) {
            goto done;
        }

        // Swap a frame
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

        // Increment frame count, get time and update angle
        ++frames;
        currTime = SYSTIME();
        angle = (float)((120ull * (currTime-startTime) / 1000000000ull) % 360);

        // Check whether time limit has been exceeded
        if (!runforever && !shutdown) shutdown = (endTime <= currTime);
    } while (!shutdown);

    // Success
    failure = 0;

    done:

    // If any frames were generated, print the framerate
    if (frames) {
        NvGlDemoLog("Total FPS: %f\n",
                    (float)frames / ((currTime - startTime) / 1000000000ull));
    }

    // Otherwise something went wrong. Print usage message in case it
    //   was due to bad command line arguments.
    else {
        usage();
    }

    // Clean up gears rendering resources
    gearsTerm();

    // Free PreSwap function resources
    NvGlDemoPreSwapShutdown();

    // Clean up EGL and window system
    NvGlDemoShutdown();

    return failure;
}
