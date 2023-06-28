/*
 * shaders.c
 *
 * Copyright (c) 2003-2014, NVIDIA CORPORATION. All rights reserved.
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
// Bubble demo shader setup
//

#include "nvgldemo.h"
#include "shaders.h"

// Depending on compile options, we either build in the shader sources or
//   binaries or load them from external data files at runtime.
//   (Variables are initialized to the file contents or name, respectively).

#ifdef USE_EXTERN_SHADERS
static const char shad_bubbleVert[] = { BUBBLE_PREFIX VERTFILE(bubble_vert) };
static const char shad_bubbleFrag[] = { BUBBLE_PREFIX FRAGFILE(bubble_frag) };
static const char shad_meshVert[]   = { BUBBLE_PREFIX VERTFILE(mesh_vert) };
static const char shad_meshFrag[]   = { BUBBLE_PREFIX FRAGFILE(mesh_frag) };
static const char shad_cubeVert[]   = { BUBBLE_PREFIX VERTFILE(envcube_vert) };
static const char shad_cubeFrag[]   = { BUBBLE_PREFIX FRAGFILE(envcube_frag) };
static const char shad_mouseVert[]  = { BUBBLE_PREFIX VERTFILE(mouse_vert) };
static const char shad_mouseFrag[]  = { BUBBLE_PREFIX FRAGFILE(mouse_frag) };
#else
static const char shad_bubbleVert[] = {
#   include VERTFILE(bubble_vert)
};
static const char shad_bubbleFrag[] = {
#   include FRAGFILE(bubble_frag)
};
static const char shad_meshVert[]   = {
#   include VERTFILE(mesh_vert)
};
static const char shad_meshFrag[]   = {
#   include FRAGFILE(mesh_frag)
};
static const char shad_cubeVert[]   = {
#   include VERTFILE(envcube_vert)
};
static const char shad_cubeFrag[]   = {
#   include FRAGFILE(envcube_frag)
};
static const char shad_mouseVert[]  = {
#   include VERTFILE(mouse_vert)
};
static const char shad_mouseFrag[]  = {
#   include FRAGFILE(mouse_frag)
};
#endif

static const char bubblePrgBin[] = { PROGFILE(bubble_prog) };
static const char meshPrgBin[] = { PROGFILE(mesh_prog) };
static const char cubePrgBin[] = { PROGFILE(envcube_prog) };
static const char mousePrgBin[] = { PROGFILE(mouse_prog) };


// Reflective bubble shader
GLint prog_bubble = 0;
GLint uloc_bubbleViewMat;
GLint uloc_bubbleNormMat;
GLint uloc_bubbleProjMat;
GLint uloc_bubbleTexUnit;
GLint aloc_bubbleVertex;
GLint aloc_bubbleNormal;

// Wireframe/point mesh shader
GLint prog_mesh = 0;
GLint uloc_meshViewMat;
GLint uloc_meshProjMat;
GLint aloc_meshVertex;

// Environment cube shader
GLint prog_cube = 0;
GLint uloc_cubeProjMat;
GLint uloc_cubeTexUnit;
GLint aloc_cubeVertex;

// Crosshair shader
GLint prog_mouse = 0;
GLint uloc_mouseWindow;
GLint uloc_mouseCenter;
GLint aloc_mouseVertex;

int
LoadShaders(void)
{
    GLboolean success;

    // Load the shaders (The macro handles the details of binary vs.
    //   source and external vs. internal)
    prog_bubble = LOADPROGSHADER(shad_bubbleVert, shad_bubbleFrag,
                             GL_TRUE, GL_FALSE,
                             bubblePrgBin);
    prog_mesh   = LOADPROGSHADER(shad_meshVert, shad_meshFrag,
                             GL_TRUE, GL_FALSE,
                             meshPrgBin);
    prog_cube   = LOADPROGSHADER(shad_cubeVert, shad_cubeFrag,
                             GL_TRUE, GL_FALSE,
                             cubePrgBin);
    prog_mouse  = LOADPROGSHADER(shad_mouseVert, shad_mouseFrag,
                             GL_TRUE, GL_FALSE,
                             mousePrgBin);
    success = prog_bubble && prog_mesh && prog_cube && prog_mouse;
    if (!success) {
        NvGlDemoLog("Error occured loading shaders");
        return 0;
    }

    // Load locations for bubble shader
    uloc_bubbleViewMat = glGetUniformLocation(prog_bubble, "modelview");
    uloc_bubbleNormMat = glGetUniformLocation(prog_bubble, "imodelview");
    uloc_bubbleProjMat = glGetUniformLocation(prog_bubble, "projection");
    uloc_bubbleTexUnit = glGetUniformLocation(prog_bubble, "texunit");
    aloc_bubbleVertex  = glGetAttribLocation(prog_bubble, "vertex");
    aloc_bubbleNormal  = glGetAttribLocation(prog_bubble, "normal");
    success =  (uloc_bubbleViewMat >= 0)
            && (uloc_bubbleNormMat >= 0)
            && (uloc_bubbleProjMat >= 0)
            && (uloc_bubbleTexUnit >= 0)
            && (aloc_bubbleVertex >= 0)
            && (aloc_bubbleNormal >= 0);
    if (!success) {
        NvGlDemoLog("Error occured retrieving bubble shader locations");
        return 0;
    }
    glUseProgram(prog_bubble);
    glUniform1i(uloc_bubbleTexUnit, 0);

    // Load locations for environment cube shader
    uloc_meshViewMat = glGetUniformLocation(prog_mesh, "modelview");
    uloc_meshProjMat = glGetUniformLocation(prog_mesh, "projection");
    aloc_meshVertex =  glGetAttribLocation(prog_mesh, "vertex");
    success =  (uloc_meshViewMat >= 0)
            && (uloc_meshProjMat >= 0)
            && (aloc_meshVertex >= 0);
    if (!success) {
        NvGlDemoLog("Error occured retrieving mesh shader locations");
        return 0;
    }

    // Load locations for environement cube shader
    uloc_cubeProjMat = glGetUniformLocation(prog_cube, "projection");
    uloc_cubeTexUnit = glGetUniformLocation(prog_cube, "texunit");
    aloc_cubeVertex =  glGetAttribLocation(prog_cube, "vertex");
    success =  (uloc_cubeProjMat >= 0)
            && (uloc_cubeTexUnit >= 0)
            && (aloc_cubeVertex >= 0);
    if (!success) {
        NvGlDemoLog("Error occured retrieving cube shader locations");
        return 0;
    }
    glUseProgram(prog_cube);
    glUniform1i(uloc_cubeTexUnit, 0);

    // Load locations for environement mouse shader
    uloc_mouseWindow = glGetUniformLocation(prog_mouse, "window");
    uloc_mouseCenter = glGetUniformLocation(prog_mouse, "center");
    aloc_mouseVertex = glGetAttribLocation(prog_mouse, "vertex");
    success =  (uloc_mouseWindow >= 0)
            && (uloc_mouseCenter >= 0)
            && (aloc_mouseVertex >= 0);
    if (!success) {
        NvGlDemoLog("Error occured retrieving mouse shader locations");
        return 0;
    }

    return 1;
}

void
FreeShaders(void)
{
    if (prog_mouse)  glDeleteProgram(prog_mouse);
    if (prog_cube)   glDeleteProgram(prog_cube);
    if (prog_mesh)   glDeleteProgram(prog_mesh);
    if (prog_bubble) glDeleteProgram(prog_bubble);
}
