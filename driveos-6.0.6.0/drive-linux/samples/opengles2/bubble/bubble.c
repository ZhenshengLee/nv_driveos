/*
 * bubble.c
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
// Bubble demo state control and top level rendering functions
//

#include "nvgldemo.h"
#include "bubble.h"
#include "shaders.h"

/** Macro for determining the size of an array */
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

///////////////////////////////////////////////////////////////////////////////
//
// Setup/shutdown
//

// Set up bubble geometry
static void
BubbleState_buildBubble(
    BubbleState  *b,
    int          depth)
{
    if (b->bubble) {
        Bubble_destroy(b->bubble);
    }
    switch (depth) {
        case 2: b->bubble = Bubble_create( 6); break;
        case 3: b->bubble = Bubble_create(13); break;
        case 4: b->bubble = Bubble_create(25); break;
        default: return;
    }
    Bubble_calcNormals(b->bubble);
}

// Set up the bubble window state
int
BubbleState_init(
    BubbleState  *b,
    int          autopoke,
    float        duration,
    int          fpsFlag,
    bool         fframeIncrement,
    float        delta,
    GLsizei      width,
    GLsizei      height)
{
    // Initialize timer
    b->timeStart          = 0.0;
    b->timeCurrent        = 0.0;
    b->timeDuration       = duration;
    b->drawFinal          = false;
    b->swapInterval       = 1;

    // Initialize fps computation
    b->fpsFlag            = fpsFlag;
    b->fpsCount           = 0;
    b->fpsTime            = 0.0;
    b->fpsValue           = 0.0f;
    b->fpsInterval        = 5.0f;
    b->delta              = delta;
    b->fframeIncrement    = fframeIncrement;
    b->currentFrameNumber = 1;

    // Initialize autopoke
    b->autopokeInterval = autopoke;
    b->autopokeCount    = 0;

    // Initialize mouse state
    b->mouseX           = 0;
    b->mouseY           = 0;
    b->screenX          = 0.0f;
    b->screenY          = 0.0f;
    b->mouseDown        = false;

    // Initialize window size
    b->windowWidth      = 0;
    b->windowHeight     = 0;

    // Initialize camera position
    b->heading          = 0.0f;
    b->pitch            = 0.0f;
    quat_identity(&b->orient);
    b->hv               = 0.0f;
    b->pv               = 0.0f;

    // Initialize projection parameters
    b->aspect           = 1.0f;
    b->nearHeight       = 1.0f;
    b->nearDistance     = 2.0f;
    b->farDistance      = 40.0f;
    b->eyeDistance      = 5.0f;

    // Initialize objects
    b->mode             = CUBE_MODE;
    b->envCube          = NULL;
    b->cubeTexture      = 0;
    b->bubble           = NULL;
    b->mouseFlag        = true;
    b->font             = NULL;

    // Construct the bubble geometry with maximum subdivision
    BubbleState_buildBubble(b, 4);

    // Initialize GL settings
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClearDepthf(1.0f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);

    // Load the shaders, if not successful quit.
    if (!LoadShaders()) {
        return 0;
    }

    // Initialize the viewport
    BubbleState_reshapeViewport(b, width, height);

    // Set up the environment map
    b->envCube     = EnvCube_build();
    b->cubeTexture = EnvCube_getCube(b->envCube);

    // Start mouse out to left of window center
    //   If there is a real mouse in use, this will be overridden by an event.
    //   If a fake mouse is in use, this starts us rotating by default.
    BubbleState_mouse(b, (b->windowWidth)/3, (b->windowHeight)/2);

    // Set up font library
    b->font = nvtexfontInitRasterFont(NV_TEXFONT_DEFAULT, 0, GL_TRUE,
                                      GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR);

    // Set rendering start time
    b->timeStart = b->timeCurrent = (double)SYSTIME() / 1000000000.0;
    b->fpsTime   = b->timeStart;

    return 1;
}

// Clean up the bubble window resources
void
BubbleState_term(
    BubbleState  *b)
{
    if (b != NULL) {
        if (b->envCube != NULL) EnvCube_destroy(b->envCube);
        if (b->bubble != NULL)  Bubble_destroy(b->bubble);
        if (b->font != NULL)    nvtexfontUnloadRasterFont(b->font);
    }
    FreeShaders();
}

///////////////////////////////////////////////////////////////////////////////
//
// Mouse/keyboard event handling
//

// Pick a point on the bubble to poke when mouse/space is pressed
static void
BubbleState_pick(
    BubbleState  *b)
{
    float4x4 m;
    float3 e;
    float3 n;
    float y = b->nearHeight * b->screenY;
    float x = b->nearHeight * b->aspect * b->screenX;

    e[0] = 0.0f;
    e[1] = 0.0f;
    e[2] = b->eyeDistance;
    n[0] = x;
    n[1] = y;
    n[2] = -b->nearDistance;
    quat_mat(m, &(b->orient));
    mat_invert(m);
    pnt_transform(e, m);
    vec_transform(n, m);
    Bubble_pick(b->bubble, e, n);
}

// Handle left mouse button press
void
BubbleState_leftMouse(
    BubbleState  *b,
    bool         click)
{
    b->mouseDown = click;
    if (click == true) {
        BubbleState_pick(b);
    }
}

// Handle mouse movement
void
BubbleState_mouse(
    BubbleState  *b,
    int          x,
    int          y)
{
    int windowX = 0, windowY = 0;

    // Update absolute position if within window boundaries
    if ((x > 0) && (x < b->windowWidth))
        b->mouseX = x;
    if ((y > 0) && (y < b->windowHeight))
        b->mouseY = y;

    // Map to [-1,+1]. Invert Y, because GL origin is lower left.
    windowY = b->windowHeight/2 - b->mouseY;
    b->screenY = ((float)windowY) / (((float)b->windowHeight)/2.0f);
    windowX = b->mouseX - b->windowWidth/2;
    b->screenX = ((float)windowX) / (((float)b->windowWidth) /2.0f);
    b->mouseFlag = true;
}

// Handle key press
void
BubbleState_callback(
    BubbleState  *b,
     int         key)
{
    int triCount = 0;
    int i = 0;

    switch (key) {
        case 'q':
        case 'Q':
            b->drawFinal = true;
            break;
        case 'h':
        case 'H':
            NvGlDemoLog("\n"
                "h: help (this output)\n"
#ifdef USE_FAKE_MOUSE
                "i: move cursor up\n"
                "j: move cursor left\n"
                "k: move cursor down\n"
                "l: move cursor right\n"
#else
                "mouse: steer the bubble\n"
#endif
                "f: toggle fps display\n"
                "v: toggle wait for vsync (swap interval)\n"
                "m: toggle cursor display\n"
                "t: print triangles/frame\n"
                "p: render points\n"
                "w: render wireframe\n"
                "c: render full bubble\n"
#ifdef YOU_REALLY_WANT_UNSTABLE_GEOMETRY
                "2: low res bubble\n"
#endif
                "3: medium res bubble\n"
                "4: high res bubble\n"
#ifdef USE_FAKE_MOUSE
                "SPACE: poke the bubble\n"
#else
                "mouse button: poke the bubble\n"
#endif
                "q: quit\n"
                "\n");
            break;
#ifdef USE_FAKE_MOUSE
#define MOUSE_DELTA (10)
        case 'i':
        case 'I':
            BubbleState_mouse(b, b->mouseX, (b->mouseY)-MOUSE_DELTA);
            break;
        case 'k':
        case 'K':
            BubbleState_mouse(b, b->mouseX, (b->mouseY)+MOUSE_DELTA);
            break;
        case 'j':
        case 'J':
            BubbleState_mouse(b, (b->mouseX)-MOUSE_DELTA, b->mouseY);
            break;
        case 'l':
        case 'L':
            BubbleState_mouse(b, (b->mouseX)+MOUSE_DELTA, b->mouseY);
            break;
#endif
        case 'f':
        case 'F':
            b->fpsFlag = !(b->fpsFlag);
            break;
        // EGL 1.1
        case 'v':
        case 'V':
            {
                b->swapInterval = !(b->swapInterval);
                NvGlDemoSwapInterval(demoState.display, b->swapInterval);
                NvGlDemoLog("swap interval set to: %d\n", b->swapInterval);
            }
            break;
        case 'm':
        case 'M':
            b->mouseFlag = !(b->mouseFlag);
            break;
        case 't':
        case 'T':
            triCount = 0;
            // count up the triangles in the tristrips
            for ( i=0 ; i<(b->bubble)->numTristrips ; i++ ) {
                triCount += (b->bubble)->tristrips[i].numVerts - 2;
            }
            // add in the 6 walls of the cube that we're in
            triCount += 6*2;
            {
                NvGlDemoLog("%d triangles/frame\n", triCount);
            }
            break;
#ifdef YOU_REALLY_WANT_UNSTABLE_GEOMETRY
        case '2':
            BubbleState_buildBubble(b, 2);
            break;
#endif
        case '3':
            BubbleState_buildBubble(b, 3);
            break;
        case '4':
            BubbleState_buildBubble(b, 4);
            break;
        case 'p':
        case 'P':
            b->mode = POINT_MODE;
            break;
        case 'w':
        case 'W':
            b->mode = WIRE_MODE;
            break;
        case 'c':
        case 'C':
            b->mode = CUBE_MODE;
            break;
#ifdef USE_FAKE_MOUSE
        case ' ':
            BubbleState_leftMouse(b, true);
            BubbleState_leftMouse(b, false);
            break;
#endif
    }
}

///////////////////////////////////////////////////////////////////////////////
//
// Rendering
//

// Draw the mouse pointer
static void
BubbleState_drawmouse(
    BubbleState  *b)
{
    const GLfloat crosshair[8] = { -3.5f,  0.0f,
                                    3.5f,  0.0f,
                                    0.0f, -3.5f,
                                    0.0f,  3.5f};

    // Draw crosshairs
    glUseProgram(prog_mouse);
    glUniform2f(uloc_mouseCenter, (GLfloat)b->mouseX,
                                  (GLfloat)(b->windowHeight-1-b->mouseY));
    glLineWidth(1.0f);
    glEnableVertexAttribArray(aloc_mouseVertex);
    glVertexAttribPointer(aloc_mouseVertex,
                          2, GL_FLOAT, GL_FALSE, 0, (void *)crosshair);
    glDrawArrays(GL_LINES, 0, 4);
    glDisableVertexAttribArray(aloc_mouseVertex);
}

// Draw the frame
static void
BubbleState_draw(
    BubbleState  *b,
    float        delta)
{
    const float3 pitchAxis   = {1.0f, 0.0f, 0.0f};
    const float3 headingAxis = {0.0f, 1.0f, 0.0f};
    Quat     q1, q2;
    float4x4 modelview;
    float4x4 projection;
    float4x4 normal;
    float4x4 m;
    float3   v;

    // Clear the buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Figure out orientation and set up modelview to center bubble
    quat_setfv(&q1, b->pitch, pitchAxis);
    quat_setfv(&q2, b->heading, headingAxis);
    quat_multiply(&q1, &q2);
    quat_prescribe(&(b->orient), &q1);
    quat_mat(m, &(b->orient));
    mat_invert(m);
    v[0] =  0.0f;
    v[1] =  0.0f;
    v[2] = -1.0f;
    vec_transform(v, m);
    vec_scale(v, b->eyeDistance);
    mat_identity(modelview);
    mat_translate(modelview, v[0], v[1], v[2]);

    // Compute the projection matrix for current orientation
    mat_identity(projection);
    mat_frustum(projection,
                -b->nearHeight * b->aspect,
                 b->nearHeight * b->aspect,
                -b->nearHeight,
                 b->nearHeight,
                 b->nearDistance,
                 b->farDistance);
    quat_mat(m, &b->orient);
    mat_multiply(projection, m);

    // Load the environment cube program, set the matrix, and draw the cube
    glUseProgram(prog_cube);
    glUniformMatrix4fv(uloc_cubeProjMat, 1, GL_FALSE, (GLfloat*)&projection);
    EnvCube_draw(b->envCube);

    // Perform bubble dynamics
    // TODO: These calculations should take the frame rate into account.
    // TODO:   Currently the bubble dynamics just apply a fixed delta
    // TODO:   per frame. So when running at a higher frame rate, the
    // TODO:   bubble will appear more animated.
    if(delta != 0.0) {
        Bubble_calcVelocity(b->bubble);
        Bubble_filterVelocity(b->bubble);
        Bubble_calcNormals(b->bubble);
    }

    // Draw the bubble. For full bubble we also need the normal matrix
    if (b->mode == CUBE_MODE) {
        glUseProgram(prog_bubble);
        MEMCPY(normal, modelview, 16*sizeof(float));
        mat_transpose(normal);
        mat_invert_part(normal);
        glUniformMatrix4fv(uloc_bubbleProjMat, 1, GL_FALSE,
                           (GLfloat*)&projection);
        glUniformMatrix4fv(uloc_bubbleViewMat, 1, GL_FALSE,
                           (GLfloat*)&modelview);
        glUniformMatrix4fv(uloc_bubbleNormMat, 1, GL_FALSE,
                           (GLfloat*)&normal);
        Bubble_draw(b->bubble, b->envCube->cubeTexture);
    } else {
        glUseProgram(prog_mesh);
        glUniformMatrix4fv(uloc_meshProjMat, 1, GL_FALSE,
                           (GLfloat*)&projection);
        glUniformMatrix4fv(uloc_meshViewMat, 1, GL_FALSE,
                           (GLfloat*)&modelview);
        if (b->mode == WIRE_MODE)
            Bubble_drawEdges(b->bubble);
        else
            Bubble_drawVertices(b->bubble);
    }

    // Update fps
    if (b->fpsFlag && b->fpsValue != 0.0f) {
        char   buf[30];
        SNPRINTF(buf, ARRAY_SIZE(buf), "FPS: %5.1f", b->fpsValue);
        nvtexfontRenderString_All(b->font, buf,
                                  0.5f, -0.8f, 0.5f, 0.5f, 1.0f, 1.0f, 1.0f);
    }
    // Update frame number
    if (b->fframeIncrement) {
        char   buf[30];
        SNPRINTF(buf, ARRAY_SIZE(buf), "Frame# %lu", b->currentFrameNumber);
        nvtexfontRenderString_All(b->font, buf,
                                  -0.8f, -0.85f, 0.4f, 0.4f, 1.0f, 1.0f, 1.0f);
    }

    // Draw mouse crosshairs
    if (b->mouseFlag) {
        glDisable(GL_DEPTH_TEST);
        BubbleState_drawmouse(b);
        glEnable(GL_DEPTH_TEST);
    }
}

// Update bubble orientation
static void
BubbleState_rotate(
    BubbleState  *b,
    float        delta)
{
    // Maximum rotation is one revolution every 8 seconds.
    // Maximum acceleration is 0 to full speed in 1 second.
    const float maxrot = 2.0f * PI / 8.0f;
    const float maxacc = maxrot / 1.0f;

    // If the framerate is very low or we hit a lag spike, subdivide the
    //   calculation to keep things stable.
    if (delta > 0.25f) {
        BubbleState_rotate(b, 0.5f*delta);
        BubbleState_rotate(b, 0.5f*delta);
    }

    // Otherwise compute the change in rotation/velocity for this time delta
    else {
        // Accelerate based on mouse position
        b->hv += delta * maxacc * b->screenX;
        b->pv -= delta * maxacc * b->screenY;

        // Apply spring force to move pitch towards center
        // (Spring coefficient is just strong enough to counterbalance
        //    maximum mouse acceleration as we approach the poles.)
        b->pv -= delta * maxacc * (b->pitch / (0.45f * PI));

        // Apply damping force to neutralize rotation
        // (Damping coefficient is just strong enough to keep speed
        //  below maximum.)
        b->hv *= maxrot / (maxrot + delta * maxacc);
        b->pv *= maxrot / (maxrot + delta * maxacc);

        // Apply current rotation speed
        b->heading += b->hv * delta;
        b->pitch   += b->pv * delta;
        if (b->heading >  PI) b->heading -= (float)(2.0f*PI);
        if (b->heading < -PI) b->heading += (float)(2.0f*PI);
    }
}

// Process the next frame
void
BubbleState_tick(
    BubbleState  *b)
{
    double time;
    float  delta;
    float  fpsDelta;

    // Get current time and calculate delta since last frame
    time = (double)SYSTIME() / 1000000000.0;
    delta = (float)(time - b->timeCurrent);
    b->timeCurrent = time;

    // Update fps calculation
    fpsDelta = (float)(time - b->fpsTime);
    if (fpsDelta >= b->fpsInterval) {
        b->fpsValue = (float)b->fpsCount / fpsDelta;
        b->fpsCount = 0;
        b->fpsTime  = time;
        if (b->fpsFlag)
            NvGlDemoLog("fps: %f\n", b->fpsValue);
    }
    b->fpsCount++;

    // In Fixed Frame Increment mode, if delta is not set by user, save time
    // delta in bubble state in first frame, and use that in later frames
    // irrespective of current delta of time.
    if (b->fframeIncrement) {
        if (b->delta < 0.0f) {
            if (delta > 0.0001f) {
                b->delta = delta;
            }
        } else {
            delta = b->delta;
        }
    }

    // Update the orientation
    BubbleState_rotate(b, delta);

    // Draw the frame
    BubbleState_draw(b, delta);

    // Update frame number
    if (b->currentFrameNumber == LONG_MAX) {
        b->currentFrameNumber = 0;
        if (b->fframeIncrement) {
            NvGlDemoLog("Frame# reached MAX[%lu]. Resetting...\n", LONG_MAX);
        }
    }
    b->currentFrameNumber++;

    // Check for exit
    if ((b->timeDuration > 0.0) &&
        (b->timeCurrent-b->timeStart >= b->timeDuration)) {
        b->drawFinal = true;
    }

    // Handle auto-poking
    (b->autopokeCount)++;
    if (b->autopokeInterval && b->autopokeCount == b->autopokeInterval) {
        BubbleState_pick(b);
        b->autopokeCount = 0;
    }
}

// Initialize/change the viewport
void
BubbleState_reshapeViewport(
    BubbleState  *b,
    GLsizei      w,
    GLsizei      h)
{
    if (w==0 || h==0) return;
#ifdef USE_FAKE_MOUSE
    if (b->windowWidth && b->windowHeight) {
        b->mouseX = (b->mouseX * w) / b->windowWidth;
        b->mouseY = (b->mouseY * h) / b->windowHeight;
    }
#endif
    b->windowWidth = w;
    b->windowHeight = h;
    glViewport(0, 0, w, h);
    b->aspect = ((float)(b->windowWidth))/((float)(b->windowHeight));
    glUseProgram(prog_mouse);
    glUniform2f(uloc_mouseWindow, 1.0f/w, 1.0f/h);
}
