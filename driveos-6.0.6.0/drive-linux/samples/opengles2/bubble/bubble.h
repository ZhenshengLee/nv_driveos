/*
 * bubble.h
 *
 * Copyright (c) 2003-2015, NVIDIA CORPORATION. All rights reserved.
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
// Bubble demo state and top level rendering functions.
//

#ifndef __BUBBLE_H
#define __BUBBLE_H

#include <GLES2/gl2.h>
#include <limits.h>
#include "algebra.h"
#include "envcube.h"
#include "shape.h"
#include "nvtexfont.h"

// Simple boolean type
typedef enum {false = 0, true} bool;

// Bubble demo state
typedef struct {
    double          timeStart;      // Time of initialization
    double          timeCurrent;    // Current time
    float           timeDuration;   // Total requested run time (0 = forever)
    bool            drawFinal;      // If true, draw final frame and exit
    int             swapInterval;   // Current swap interval

    int             fpsFlag;        // FPS reporting enabled
    int             fpsCount;       // Frames since last fps computation
    double          fpsTime;        // Time of last fps computation
    float           fpsValue;       // Last value computed for fps
    float           fpsInterval;    // Time between fps computations
    float           delta;          // delta time in-between frames
    bool            fframeIncrement;    // Run in Fixed-Frame Increment mode (Fixed increment of frames)

    unsigned long   currentFrameNumber; // current frame number since start of app
    int             autopokeInterval;   // Autopoking interval (0 = disabled)
    int             autopokeCount;      // Frames since last autopoke

    int             mouseX, mouseY;     // Absolute mouse position (wrt corner)
    float           screenX, screenY;   // Relative mouse position (wrt center)
    bool            mouseDown;          // Mouse currently pressed

    int             windowWidth;    // Window dimensions
    int             windowHeight;

    float           heading;        // Camera control
    float           pitch;
    Quat            orient;
    float           hv, pv;

    float           aspect;         // Projection parameters
    float           nearHeight;
    float           nearDistance;
    float           farDistance;
    float           eyeDistance;

    enum Mode       {POINT_MODE, WIRE_MODE, CUBE_MODE} mode;    // Render mode
    EnvCube         *envCube;       // Cubemap state
    unsigned int    cubeTexture;    // Cubemap texture ID
    Shape           *bubble;        // Bubble shape state
    int             mouseFlag;      // Enable mouse crosshair cursor

    NVTexfontRasterFont *font;      // Display font info
} BubbleState;

int
BubbleState_init(
    BubbleState  *b,
    int          autopoke,
    float        duration,
    int          fpsFlag,
    bool         fframeIncrement,
    float        delta,
    GLsizei      width,
    GLsizei      height);

void
BubbleState_reshapeViewport(
    BubbleState  *b,
    GLsizei      w,
    GLsizei      h);

void
BubbleState_mouse(
    BubbleState  *b,
    GLsizei      x,
    GLsizei      y);

void
BubbleState_leftMouse(
    BubbleState  *b,
    bool         click);

void
BubbleState_callback(
    BubbleState  *b,
    int          key);

void
BubbleState_tick(
    BubbleState  *b);

void
BubbleState_term(
    BubbleState  *b);

#endif // __BUBBLE_H
