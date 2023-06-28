/*
 * x11-cubemgr.c
 *
 * Copyright (c) 2012-2012, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

// Example X11 window manager application which maps windows to the
//   faces of a rotating cube.

#ifdef DEMO_USE_KD
#error "Requires non-KD build of demos (set DEMO_USE_KD=0)"
#endif


/*=========================================================================*/
/*                              Header Files                               */

// Demo utilities
#include "nvgldemo.h"
#include "nvgldemo_win_x11.h"

// EGL/GLES2 and extensions
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>

// X11 extensions
#include <X11/extensions/Xdamage.h>

// Window manager utilities
#include "x11-mgrlib.h"

#include "unistd.h"

#define MIN_FRAME_TIME 8333333

/*=========================================================================*/
/*                             Rendering state                             */

// Shader source/binary
#ifdef USE_EXTERN_SHADERS
static const char shad_cubemgrVert[] = { VERTFILE(cubemgr_vert) };
static const char shad_cubemgrFrag[] = { FRAGFILE(cubemgr_frag) };
#else
static const char shad_cubemgrVert[] = {
#   include VERTFILE(cubemgr_vert)
};
static const char shad_cubemgrFrag[] = {
#   include FRAGFILE(cubemgr_frag)
};
#endif

// Shader program
static GLuint prog_cubemgr = 0;
static GLint  uloc_cubemgrCamera;
static GLint  uloc_cubemgrObject;
static GLint  uloc_cubemgrTexscale;
static GLint  uloc_cubemgrTexrange;

// Scene transformations
static const GLfloat depthnear =  5.0f;
static const GLfloat depthfar  = 60.0f;


// Vertex position/texture data for cube
static GLfloat cube[24][5] =  {
    { -1.0f, -1.0f, +1.0f,   -1.0f, -1.0f },
    { +1.0f, -1.0f, +1.0f,   +1.0f, -1.0f },
    { -1.0f, +1.0f, +1.0f,   -1.0f, +1.0f },
    { +1.0f, +1.0f, +1.0f,   +1.0f, +1.0f },

    { +1.0f, +1.0f, -1.0f,   +1.0f, -1.0f },
    { -1.0f, +1.0f, -1.0f,   -1.0f, -1.0f },
    { +1.0f, -1.0f, -1.0f,   +1.0f, +1.0f },
    { -1.0f, -1.0f, -1.0f,   -1.0f, +1.0f },

    { -1.0f, +1.0f, -1.0f,   -1.0f, -1.0f },
    { -1.0f, -1.0f, -1.0f,   +1.0f, -1.0f },
    { -1.0f, +1.0f, +1.0f,   -1.0f, +1.0f },
    { -1.0f, -1.0f, +1.0f,   +1.0f, +1.0f },

    { +1.0f, -1.0f, +1.0f,   +1.0f, -1.0f },
    { +1.0f, +1.0f, +1.0f,   -1.0f, -1.0f },
    { +1.0f, -1.0f, -1.0f,   +1.0f, +1.0f },
    { +1.0f, +1.0f, -1.0f,   -1.0f, +1.0f },

    { +1.0f, -1.0f, -1.0f,   -1.0f, -1.0f },
    { +1.0f, -1.0f, +1.0f,   +1.0f, -1.0f },
    { -1.0f, -1.0f, -1.0f,   -1.0f, +1.0f },
    { -1.0f, -1.0f, +1.0f,   +1.0f, +1.0f },

    { -1.0f, +1.0f, +1.0f,   +1.0f, -1.0f },
    { -1.0f, +1.0f, -1.0f,   -1.0f, -1.0f },
    { +1.0f, +1.0f, +1.0f,   +1.0f, +1.0f },
    { +1.0f, +1.0f, -1.0f,   -1.0f, +1.0f }
};

/*=========================================================================*/
/*                                Rendering                                */

// Adjust rendering parameters for window size
static int
CubeMgrRenderResize(void)
{
    XWindowAttributes winAtt;
    GLint   isize[2];
    GLfloat fsize[2];
    GLfloat matrix[16];
    Status   rvint;

    // Load current window size
    rvint = XGetWindowAttributes(demoState.platform->XDisplay,
                                 demoState.platform->XWindow, &winAtt);
    if (!rvint) {
        NvGlDemoLog("Failed to retrieve window size\n");
        return 0;
    }
    isize[0] = winAtt.width;
    isize[1] = winAtt.height;

    demoState.width  = isize[0];
    demoState.height = isize[1];

    // Frustrum size preserves window aspect ratio
    if (isize[0] > isize[1]) {
        fsize[0] = (GLfloat)isize[0] / (GLfloat)isize[1];
        fsize[1] = 1.0f;
    } else {
        fsize[1] = (GLfloat)isize[1] / (GLfloat)isize[0];
        fsize[0] = 1.0f;
    }

    // Compute perspective matrix
    NvGlDemoMatrixIdentity(matrix);
    NvGlDemoMatrixFrustum(matrix,
                          -fsize[0], +fsize[0],
                          -fsize[1], +fsize[1],
                          depthnear, depthfar);
    glUniformMatrix4fv(uloc_cubemgrCamera, 1, GL_FALSE, matrix);

    // Set viewport
    glViewport(0, 0, isize[0], isize[1]);

    return 1;
}

// Initialize GL state and obtain rendering resources
static int
CubeMgrRenderInit(void)
{
    GLuint  loc;

    // Render pipeline settings
    glEnable(GL_DEPTH_TEST);

    // Load and use the shader program
    prog_cubemgr = LOADSHADER(shad_cubemgrVert, shad_cubemgrFrag,
                              GL_FALSE, GL_FALSE);
    if (!prog_cubemgr) {
        NvGlDemoLog("Failed to create and link shader program");
        return 0;
    }
    glLinkProgram(prog_cubemgr);
    glUseProgram(prog_cubemgr);

    // Get the uniform locations
    uloc_cubemgrCamera   = glGetUniformLocation(prog_cubemgr, "cameramat");
    uloc_cubemgrObject   = glGetUniformLocation(prog_cubemgr, "objectmat");
    uloc_cubemgrTexscale = glGetUniformLocation(prog_cubemgr, "texscale");
    uloc_cubemgrTexrange = glGetUniformLocation(prog_cubemgr, "texrange");

    // Initialize viewport
    if (!CubeMgrRenderResize())
        return 0;

    // Vertex coordinate setup
    loc = glGetAttribLocation(prog_cubemgr, "vtxpos");
    glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE,
                          5*sizeof(GLfloat), &cube[0][0]);
    glEnableVertexAttribArray(loc);
    loc = glGetAttribLocation(prog_cubemgr, "vtxtex");
    glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE,
                          5*sizeof(GLfloat), &cube[0][3]);
    glEnableVertexAttribArray(loc);

    // Set program's texture unit to 0
    loc = glGetUniformLocation(prog_cubemgr, "texunit");
    glUniform1i(loc, 0);

    return (glGetError() == GL_NO_ERROR) ? 1 : 0;
}

// Clean up rendering resources
static void
CubeMgrRenderTerm(void)
{
    // Delete the program
    if (prog_cubemgr) {
        glDeleteProgram(prog_cubemgr);
        prog_cubemgr = 0;
    }
}

// Draw the cube
static void
CubeMgrRenderFrame(
    float   t)
{
    GLfloat     angA = 60.0f * t * 0.175f;
    GLfloat     angB = 60.0f * t * 0.050f;
    GLfloat     matrix[16];
    MgrObject  *window;
    int         i;

    // Clear buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Set up object transformation matrix
    NvGlDemoMatrixIdentity(matrix);
    NvGlDemoMatrixTranslate(matrix, 0.0f, 0.0f, -30.0f);
    NvGlDemoMatrixRotate(matrix, angA, 0.6f, 0.8f, 0.0f);
    NvGlDemoMatrixRotate(matrix, angB, 0.0f, 1.0f, 1.0f);
    NvGlDemoMatrixScale(matrix, 3.0f, 3.0f, 3.0f);
    glUniformMatrix4fv(uloc_cubemgrObject, 1, GL_FALSE, matrix);

    // Draw each face of the cube
    for (i=0; i<6; ++i) {

        // Default texture to the nvidia image
        GLuint  texID       = logoTex;
        GLfloat texscale[2] = { 1.1f, 1.1f };
        GLfloat texrange[2] = { 1.0f, 1.0f };

        // If there's a window image available, use it.
        window = &bufferList[i];
        if (window->window != None) {
            if (!window->eglImage)
                CreateEglImage(window);

            if (window->eglImage) {
                texID = window->texID;

                if (window->winSize[0] > window->winSize[1]) {
                    texscale[0] = +1.1f;
                    texscale[1] = +1.1f * (GLfloat)window->winSize[0]
                                        / (GLfloat)window->winSize[1];
                } else {
                    texscale[1] = +1.1f;
                    texscale[0] = +1.1f * (GLfloat)window->winSize[1]
                                        / (GLfloat)window->winSize[0];
                }

                // X11 stores buffers top-to-bottom so they need to be inverted
                // along the Y-axis. Our texture coordinates range from -1 to
                // +1 so scaling them with -1 inverts the texture
                texscale[1] *= -1.0f;

                // Window may not occupy the full buffer, so select portion of
                //   texture to use
                // This is a NOOP in X11 and is not removed only to mimic
                // kdgui-cubemgr
                if (window->winSize[0] < window->bufSize[0])
                    texrange[0] = (GLfloat)window->winSize[0]
                                / (GLfloat)window->bufSize[0];
                if (window->winSize[1] < window->bufSize[1])
                    texrange[1] = (GLfloat)window->winSize[1]
                                / (GLfloat)window->bufSize[1];
            }
        }

        // Bind texture and set scaling and section
        glBindTexture(GL_TEXTURE_2D, texID);
        glUniform2f(uloc_cubemgrTexscale, texscale[0], texscale[1]);
        glUniform2f(uloc_cubemgrTexrange, texrange[0], texrange[1]);

        // Draw the face
        glDrawArrays(GL_TRIANGLE_STRIP, 4*i, 4);
    }

    // Post buffers
    eglSwapBuffers(demoState.display, demoState.surface);
}

/*=========================================================================*/
/*                            Window Management                            */

// Initialize window manager
static int
CubeMgrManagerInit(void)
{
    // Initialize buffer list (fixed size 6)
    if (!MgrObjectListInit(6)) {
        NvGlDemoLog("Failed to create window list\n");
        return 0;
    }

    return 1;
}

// Release manager resources
static void
CubeMgrManagerTerm(void)
{
    // Release any remaining windows/streams
    MgrObjectListTerm();
}

/*=========================================================================*/
/*                                Top Level                                */

// Entry point of this demo program.
int main(int argc, char **argv)
{
    int         initialized = 0;
    int         failure = 1;
    long long   initTime;
    long long   currTime;
    long long   lastTime;
    XEvent      event;

    // Initialize window system and EGL
    if (!NvGlDemoInitialize(&argc, argv, "x11-cubemgr", 2, 16, 0))
        goto done;

    if (!MgrLibInit())
        goto done;

    // Initialize rendering state/resources
    if (!CubeMgrRenderInit())
        goto done;

    // Initialize window/stream management
    if (!CubeMgrManagerInit())
        goto done;

    initialized = 1;
    initTime = SYSTIME();
    lastTime = initTime;
    while(1)
    {
        while (XPending(demoState.platform->XDisplay)) {
            XNextEvent(demoState.platform->XDisplay, &event);
            switch (event.type) {
                case MapNotify:
                    MapWindow(event.xmap.window);
                    break;
                case UnmapNotify:
                    UnmapWindow(event.xunmap.window);
                    break;
                case ConfigureNotify:
                    if (event.xconfigure.window == demoState.platform->XWindow)
                        CubeMgrRenderResize();
                    else
                        ConfigureWindow(event.xconfigure.window);
                    break;
                default:
                    if (event.type == damageBaseEvent + XDamageNotify)
                        MarkReady(((XDamageNotifyEvent*)(&event))->drawable,
                                  ((XDamageNotifyEvent*)(&event))->damage);
                    break;
            }
        }

        currTime = SYSTIME();
        if ((currTime - lastTime) < MIN_FRAME_TIME) {
            usleep((MIN_FRAME_TIME - (currTime - lastTime)) / 1000);
            currTime = SYSTIME();
        }
        lastTime = currTime;
        CubeMgrRenderFrame((currTime - initTime) *  0.000000001f);
    }

    // Success
    failure = 0;

    done:

    // If initialization failed, print usage message in case it was due
    //   to bad command line arguments.
    if (!initialized) {
        NvGlDemoLog("Usage: x11-cubemgr [options]\n");
        NvGlDemoLog(NvGlDemoArgUsageString());
    }

    // Clean up management resources
    CubeMgrManagerTerm();

    // Clean up rendering resources
    CubeMgrRenderTerm();

    // Clean up manager library resources
    MgrLibTerm();

    // Clean up EGL and window system
    NvGlDemoShutdown();

    return failure;
}
