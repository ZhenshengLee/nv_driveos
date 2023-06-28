/*
 * shaders.c
 *
 * Copyright (c) 2007-2014, NVIDIA CORPORATION. All rights reserved.
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
// Shader setup
//

#include <GLES2/gl2.h>

#include "nvgldemo.h"
#include "shaders.h"

// Depending on compile options, we either build in the shader sources or
//   binaries or load them from external data files at runtime.
//   (Variables are initialized to the file contents or names, respectively.)
#ifdef USE_EXTERN_SHADERS
static const char shad_lightingVert[]   = { CTREE_PREFIX VERTFILE(lighting_vert) };
static const char shad_solidsFrag[]     = { CTREE_PREFIX FRAGFILE(solids_frag) };
static const char shad_leavesFrag[]     = { CTREE_PREFIX FRAGFILE(leaves_frag) };
static const char shad_simplecolVert[]  = { CTREE_PREFIX VERTFILE(simplecol_vert) };
static const char shad_simplecolFrag[]  = { CTREE_PREFIX FRAGFILE(simplecol_frag) };
static const char shad_simpletexVert[]  = { CTREE_PREFIX VERTFILE(simpletex_vert) };
static const char shad_simpletexFrag[]  = { CTREE_PREFIX FRAGFILE(simpletex_frag) };
static const char shad_overlaycolVert[] = { CTREE_PREFIX VERTFILE(overlaycol_vert) };
static const char shad_overlaycolFrag[] = { CTREE_PREFIX FRAGFILE(overlaycol_frag) };
static const char shad_overlaytexVert[] = { CTREE_PREFIX VERTFILE(overlaytex_vert) };
static const char shad_overlaytexFrag[] = { CTREE_PREFIX FRAGFILE(overlaytex_frag) };
#else
static const char shad_lightingVert[]   = {
#   include VERTFILE(lighting_vert)
};
static const char shad_solidsFrag[]     = {
#   include FRAGFILE(solids_frag)
};
static const char shad_leavesFrag[]     = {
#   include FRAGFILE(leaves_frag)
};
static const char shad_simplecolVert[]  = {
#   include VERTFILE(simplecol_vert)
};
static const char shad_simplecolFrag[]  = {
#   include FRAGFILE(simplecol_frag)
};
static const char shad_simpletexVert[]  = {
#   include VERTFILE(simpletex_vert)
};
static const char shad_simpletexFrag[]  = {
#   include FRAGFILE(simpletex_frag)
};
static const char shad_overlaycolVert[] = {
#   include VERTFILE(overlaycol_vert)
};
static const char shad_overlaycolFrag[] = {
#   include FRAGFILE(overlaycol_frag)
};
static const char shad_overlaytexVert[] = {
#   include VERTFILE(overlaytex_vert)
};
static const char shad_overlaytexFrag[] = {
#   include FRAGFILE(overlaytex_frag)
};
#endif

static const char solidsPrgBin[] = { PROGFILE(solids_prog) };
static const char leavesPrgBin[] = { PROGFILE(leaves_prog) };
static const char simplecolPrgBin[] = { PROGFILE(simplecol_prog) };
static const char simpletexPrgBin[] = { PROGFILE(simpletex_prog) };
static const char overlaycolPrgBin[] = { PROGFILE(overlaycol_prog) };
static const char overlaytexPrgBin[] = { PROGFILE(overlaytex_prog) };

// Ground and branch shader (lit objects with full opacity)
GLint prog_solids = 0;
GLint uloc_solidsLights;
GLint uloc_solidsLightPos;
GLint uloc_solidsLightCol;
GLint uloc_solidsMvpMat;
GLint uloc_solidsTexUnit;
GLint aloc_solidsVertex;
GLint aloc_solidsNormal;
GLint aloc_solidsColor;
GLint aloc_solidsTexcoord;

// Leaves shader (lit objects with alphatest)
GLint prog_leaves = 0;
GLint uloc_leavesLights;
GLint uloc_leavesLightPos;
GLint uloc_leavesLightCol;
GLint uloc_leavesMvpMat;
GLint uloc_leavesTexUnit;
GLint aloc_leavesVertex;
GLint aloc_leavesNormal;
GLint aloc_leavesColor;
GLint aloc_leavesTexcoord;

// Simple colored object shader
GLint prog_simplecol = 0;
GLint uloc_simplecolMvpMat;
GLint aloc_simplecolVertex;
GLint aloc_simplecolColor;

// Simple textured object shader
GLint prog_simpletex = 0;
GLint uloc_simpletexColor;
GLint uloc_simpletexMvpMat;
GLint uloc_simpletexTexUnit;
GLint aloc_simpletexVertex;
GLint aloc_simpletexTexcoord;

// Colored overlay shader
GLint prog_overlaycol = 0;
GLint aloc_overlaycolVertex;
GLint aloc_overlaycolColor;

// Textured overlay shader
GLint prog_overlaytex = 0;
GLint uloc_overlaytexTexUnit;
GLint aloc_overlaytexVertex;
GLint aloc_overlaytexTexcoord;

// Load all the shaders and extract uniform/attribute locations
int
LoadShaders(void)
{
    GLboolean success;

    // Load the shaders (The macro handles the details of binary vs.
    //   source and external vs. internal)
    prog_solids     = LOADPROGSHADER(shad_lightingVert, shad_solidsFrag,
                                 GL_TRUE, GL_FALSE,
                                 solidsPrgBin);
    prog_leaves     = LOADPROGSHADER(shad_lightingVert, shad_leavesFrag,
                                 GL_TRUE, GL_FALSE,
                                 leavesPrgBin);
    prog_simplecol  = LOADPROGSHADER(shad_simplecolVert, shad_simplecolFrag,
                                 GL_TRUE, GL_FALSE,
                                 simplecolPrgBin);
    prog_simpletex  = LOADPROGSHADER(shad_simpletexVert, shad_simpletexFrag,
                                 GL_TRUE, GL_FALSE,
                                 simpletexPrgBin);
    prog_overlaycol = LOADPROGSHADER(shad_overlaycolVert, shad_overlaycolFrag,
                                 GL_TRUE, GL_FALSE,
                                 overlaycolPrgBin);
    prog_overlaytex = LOADPROGSHADER(shad_overlaytexVert, shad_overlaytexFrag,
                                 GL_TRUE, GL_FALSE,
                                 overlaytexPrgBin);
    success =  prog_solids && prog_leaves
            && prog_simplecol  && prog_simpletex
            && prog_overlaycol && prog_overlaytex;
    if (!success) {
        NvGlDemoLog("Error occured loading shaders\n");
        return 0;
    }

    // Load locations for branch/ground shader
    uloc_solidsLights   = glGetUniformLocation(prog_solids, "lights");
    uloc_solidsLightPos = glGetUniformLocation(prog_solids, "lightpos");
    uloc_solidsLightCol = glGetUniformLocation(prog_solids, "lightcol");
    uloc_solidsMvpMat   = glGetUniformLocation(prog_solids, "mvpmatrix");
    uloc_solidsTexUnit  = glGetUniformLocation(prog_solids, "texunit");
    aloc_solidsVertex   = glGetAttribLocation(prog_solids,  "vertex");
    aloc_solidsNormal   = glGetAttribLocation(prog_solids,  "normal");
    aloc_solidsColor    = glGetAttribLocation(prog_solids,  "color");
    aloc_solidsTexcoord = glGetAttribLocation(prog_solids,  "texcoord");
    success =  (uloc_solidsLights   >= 0)
            && (uloc_solidsLightPos >= 0)
            && (uloc_solidsLightCol >= 0)
            && (uloc_solidsMvpMat   >= 0)
            && (uloc_solidsTexUnit  >= 0)
            && (aloc_solidsVertex   >= 0)
            && (aloc_solidsNormal   >= 0)
            && (aloc_solidsColor    >= 0)
            && (aloc_solidsTexcoord >= 0);
    if (!success) {
        NvGlDemoLog(
            "Error occured retrieving branch/ground shader locations\n");
        return 0;
    }

    // Load locations for leaves shader
    uloc_leavesLights   = glGetUniformLocation(prog_leaves, "lights");
    uloc_leavesLightPos = glGetUniformLocation(prog_leaves, "lightpos");
    uloc_leavesLightCol = glGetUniformLocation(prog_leaves, "lightcol");
    uloc_leavesMvpMat   = glGetUniformLocation(prog_leaves, "mvpmatrix");
    uloc_leavesTexUnit  = glGetUniformLocation(prog_leaves, "texunit");
    aloc_leavesVertex   = glGetAttribLocation(prog_leaves,  "vertex");
    aloc_leavesNormal   = glGetAttribLocation(prog_leaves,  "normal");
    aloc_leavesColor    = glGetAttribLocation(prog_leaves,  "color");
    aloc_leavesTexcoord = glGetAttribLocation(prog_leaves,  "texcoord");
    success =  (uloc_leavesLights   >= 0)
            && (uloc_leavesLightPos >= 0)
            && (uloc_leavesLightCol >= 0)
            && (uloc_leavesMvpMat   >= 0)
            && (uloc_leavesTexUnit  >= 0)
            && (aloc_leavesVertex   >= 0)
            && (aloc_leavesNormal   >= 0)
            && (aloc_leavesColor    >= 0)
            && (aloc_leavesTexcoord >= 0);
    if (!success) {
        NvGlDemoLog("Error occured retrieving leaves shader locations\n");
        return 0;
    }

    // Load locations for simple color shader
    uloc_simplecolMvpMat = glGetUniformLocation(prog_simplecol, "mvpmatrix");
    aloc_simplecolVertex = glGetAttribLocation(prog_simplecol, "vertex");
    aloc_simplecolColor  = glGetAttribLocation(prog_simplecol, "color");
    success =  (aloc_simplecolColor  >= 0)
            && (uloc_simplecolMvpMat >= 0)
            && (aloc_simplecolVertex >= 0);
    if (!success) {
        NvGlDemoLog(
            "Error occured retrieving simple color shader locations\n");
        return 0;
    }

    // Load locations for simple texture shader
    uloc_simpletexColor    = glGetUniformLocation(prog_simpletex, "color");
    uloc_simpletexMvpMat   = glGetUniformLocation(prog_simpletex, "mvpmatrix");
    uloc_simpletexTexUnit  = glGetUniformLocation(prog_simpletex, "texunit");
    aloc_simpletexVertex   = glGetAttribLocation(prog_simpletex,  "vertex");
    aloc_simpletexTexcoord = glGetAttribLocation(prog_simpletex,  "texcoord");
    success =  (uloc_simpletexTexUnit  >= 0)
            && (uloc_simpletexMvpMat   >= 0)
            && (uloc_simpletexColor    >= 0)
            && (aloc_simpletexVertex   >= 0)
            && (aloc_simpletexTexcoord >= 0);
    if (!success) {
        NvGlDemoLog(
            "Error occured retrieving simple texture shader locations\n");
        return 0;
    }

    // Load locations for colored overlay shader
    aloc_overlaycolVertex = glGetAttribLocation(prog_overlaycol, "vertex");
    aloc_overlaycolColor  = glGetAttribLocation(prog_overlaycol, "color");
    success =  (aloc_overlaycolColor  >= 0)
            && (aloc_overlaycolVertex >= 0);
    if (!success) {
        NvGlDemoLog(
            "Error occured retrieving colored overlay shader locations\n");
        return 0;
    }

    // Load locations for textured overlay shader
    uloc_overlaytexTexUnit  = glGetUniformLocation(prog_overlaytex, "texunit");
    aloc_overlaytexVertex   = glGetAttribLocation(prog_overlaytex, "vertex");
    aloc_overlaytexTexcoord = glGetAttribLocation(prog_overlaytex, "texcoord");
    success =  (uloc_overlaytexTexUnit  >= 0)
            && (aloc_overlaytexVertex   >= 0)
            && (aloc_overlaytexTexcoord >= 0);
    if (!success) {
        NvGlDemoLog(
            "Error occured retrieving textured overlay shader locations\n");
        return 0;
    }

    return 1;
}

void
FreeShaders(void)
{
    if (prog_overlaytex) glDeleteProgram(prog_overlaytex);
    if (prog_overlaycol) glDeleteProgram(prog_overlaycol);
    if (prog_simpletex)  glDeleteProgram(prog_simpletex);
    if (prog_simplecol)  glDeleteProgram(prog_simplecol);
    if (prog_leaves)     glDeleteProgram(prog_leaves);
    if (prog_solids)     glDeleteProgram(prog_solids);
}
