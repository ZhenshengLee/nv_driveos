/*
 * drawfont.c
 *
 * Copyright (c) 2003-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

//
// This file provides a simple example of how to use the nvtexfont
//   library by drawing a short message to the screen.
//

#include <GLES2/gl2.h>
#include "nvgldemo.h"
#include "nvtexfont.h"

// Flag indicating it is time to shut down
static GLboolean shutdown = GL_FALSE;

// Font info
static NVTexfontRasterFont *nvtxf = NULL;

// Message settings with defaults
static float foreground[3] = { 1.0f, 1.0f, 1.0f }; // Foreground color
static float background[3] = { 0.0f, 0.0f, 0.0f }; // Background color
static float msgWidth      = 0.9f;                 // Fraction of screen width
static int   animate       = 0;                    // Animate the text
static char  *message      = NULL;

// Position/scaling of message
static float msgHeight  = 0.0f;
static float msgPos[2]  = { 0.0f, 0.0f };
static float msgScale   = 1.0f;
static float animPos[2] = { 0.0f, 0.0f };
static float animVel[2] = { 0.0f, 0.0f };

// Initialize message rendering
static int
messageInit(void)
{
    int   pixWidth, pixAscent, pixDescent;

    // Initialize font rendering
    nvtxf = nvtexfontInitRasterFont(NV_TEXFONT_DEFAULT,
                                    0, GL_TRUE,
                                    GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR);
    if (!nvtxf) return 0;

    // Get dimensions for this message
    nvtexfontGetStringMetrics(nvtxf,
                              message, STRLEN(message),
                              &pixWidth, &pixAscent, &pixDescent);

    // The current nvtexfont implementation (which isn't great) does
    //   some magic such that a scale factor of 1.0 produces text whose
    //   ascent is roughly 1/10 the height of the viewport. Note that
    //   it is a common scale factor for X and Y, so aspect ratios
    //   of the text are not properly preserved if the viewport isn't
    //   square. This whole thing needs an overhaul. For now, we have
    //   to do some magic of our own to get the message width to come
    //   out as desired.
    msgScale  = 10.0f * msgWidth * pixAscent / pixWidth;
    msgHeight = 0.1f * msgScale * (pixAscent + pixDescent) / pixAscent;
    msgPos[0] = -msgWidth;
    msgPos[1] = 0.5f * (pixDescent - pixAscent) / pixWidth;

    // If animated, start it moving
    if (animate)
        animVel[0] = animVel[1] = 0.2f;

    // Set clear color to background
    glClearColor(background[0], background[1], background[2], 1.0f);

    return 1;
}

// Render a frame
static void
messageRender(
    float delta)
{
    // Clear the screen
    glClear(GL_COLOR_BUFFER_BIT);

    // If animated, move the message
    if (animate) {

        // Adjust the position
        animPos[0] += delta * animVel[0];
        animPos[1] += delta * animVel[1];

        // Bounce at edges
        if ((animPos[0] > +(1.0f-msgWidth)) && (animVel[0] > 0.0f))
            animVel[0] = -animVel[0];
        if ((animPos[0] < -(1.0f-msgWidth)) && (animVel[0] < 0.0f))
            animVel[0] = -animVel[0];
        if ((animPos[1] > +(1.0f-msgHeight)) && (animVel[1] > 0.0f))
            animVel[1] = -animVel[1];
        if ((animPos[1] < -(1.0f-msgHeight)) && (animVel[1] < 0.0f))
            animVel[1] = -animVel[1];
    }

    // Render the string
    nvtexfontRenderString_All(nvtxf, message,
                              animPos[0] + msgPos[0],
                              animPos[1] + msgPos[1],
                              msgScale, msgScale,
                              foreground[0], foreground[1], foreground[2]);
}

// Clean up message rendering resources
static void
messageTerm(void)
{
    // Clean up font resources
    if (nvtxf) {
        nvtexfontUnloadRasterFont(nvtxf);
        nvtxf = NULL;
    }
}

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
    NvGlDemoLog("Usage: drawfont [options] <message>\n"
                "  Foreground/background colors (range [0,1]):\n"
                "    [-fg <R G B>]\n"
                "    [-bg <R G B>]\n"
                "  Width of text as a fraction of screen width:\n"
                "    [-width <fraction>]\n"
                "  Animate the message:\n"
                "    [-animate]\n");
    NvGlDemoLog(NvGlDemoArgUsageString());
}

// Entry point of this demo program.
int main(int argc, char **argv)
{
    int         failure = 1;
    long long   currTime, endTime;
    float       deltaTime = 0.0f;
    int         startup = 0;

    // Initialize window system and EGL
    if (!NvGlDemoInitialize(&argc, argv, "drawfont", 2, 0, 0)) {
        goto done;
    }

    // Initialize PreSwap functions
    NvGlDemoPreSwapInit();

    // Parse non-generic command line options
    while (argc > 2) {

        // Foreground color
        if (NvGlDemoArgMatchFlt(&argc, argv, 1, "-fg",
                                "<R G B> (float)", 0.0f, 1.0f,
                                3, foreground)) {
            // No additional action needed
        }

        // Background color
        else if (NvGlDemoArgMatchFlt(&argc, argv, 1, "-bg",
                                     "<R G B> (float)", 0.0f, 1.0f,
                                     3, background)) {
            // No additional action needed
        }

        // Message width
        else if (NvGlDemoArgMatchFlt(&argc, argv, 1, "-width",
                                     "<fraction>", 0.0f, 1.0f,
                                     1, &msgWidth)) {
            // No additional action needed
        }

        // Randomly animate
        else if (NvGlDemoArgMatch(&argc, argv, 1, "-animate")) {
            animate = 1;
        }

        // Unknown or failure
        else {
            if (!NvGlDemoArgFailed())
                NvGlDemoLog("Unknown command line option (%s)\n", argv[1]);
            goto done;
        }
    }

    // Should be one argument left containing the message
    if (argc != 2) goto done;
    message = argv[1];

    // Initialize message rendering
    if (!messageInit()) goto done;

    // Set up callbacks
    NvGlDemoSetCloseCB(closeCB);
    NvGlDemoSetResizeCB(resizeCB);
    NvGlDemoSetKeyCB(keyCB);

    // Get start time and compute end time
    if (demoOptions.duration <= 0.0) {
        NvGlDemoLog(" running forever...\n");
    } else {
        NvGlDemoLog(" running for %f seconds...\n", demoOptions.duration);
    }
    endTime = currTime = SYSTIME();
    endTime += (long long)(1000000000.0 * demoOptions.duration);

    startup = 1;

    // Main loop
    while (!shutdown) {
        long long prevTime;

        // Execute PreSwap functions
        NvGlDemoPreSwapExec();

        // Draw and swap a frame
        messageRender(deltaTime);
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

        // Compute delta from last time and check limit
        prevTime = currTime;
        currTime = SYSTIME();
        deltaTime = (float)(currTime-prevTime) / 1000000000ull;
        if (demoOptions.duration > 0.0f) shutdown = shutdown || (endTime <= currTime);

    }

    // Success
    failure = 0;

    done:

    // If basic startup failed, print usage message in case it was due
    //   to bad command line arguments.
    if (!startup) {
        usage();
    }

    // Clean up message rendering
    messageTerm();

    // Clean up PreSwap function resources
    NvGlDemoPreSwapShutdown();

    // Clean up EGL and window system
    NvGlDemoShutdown();

    return failure;
}
