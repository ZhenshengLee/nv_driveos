/*
 * shaders.h
 *
 * Copyright (c) 2003-2012, NVIDIA CORPORATION. All rights reserved.
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
// Shader programs and attribute/uniform locations
//

#ifndef __SHADERS_H
#define __SHADERS_H

#include <GLES2/gl2.h>

#ifdef ANDROID
#define CTREE_PREFIX "ctree/"
#else
#define CTREE_PREFIX ""
#endif

// Ground and branch shader (lit objects with full opacity)
extern GLint prog_solids;
extern GLint uloc_solidsLights;
extern GLint uloc_solidsLightPos;
extern GLint uloc_solidsLightCol;
extern GLint uloc_solidsMvpMat;
extern GLint uloc_solidsTexUnit;
extern GLint aloc_solidsVertex;
extern GLint aloc_solidsNormal;
extern GLint aloc_solidsColor;
extern GLint aloc_solidsTexcoord;

// Leaves shader (lit objects with alphatest)
extern GLint prog_leaves;
extern GLint uloc_leavesLights;
extern GLint uloc_leavesLightPos;
extern GLint uloc_leavesLightCol;
extern GLint uloc_leavesMvpMat;
extern GLint uloc_leavesTexUnit;
extern GLint aloc_leavesVertex;
extern GLint aloc_leavesNormal;
extern GLint aloc_leavesColor;
extern GLint aloc_leavesTexcoord;

// Simple colored object shader
extern GLint prog_simplecol;
extern GLint uloc_simplecolMvpMat;
extern GLint aloc_simplecolVertex;
extern GLint aloc_simplecolColor;

// Simple textured object shader
extern GLint prog_simpletex;
extern GLint uloc_simpletexColor;
extern GLint uloc_simpletexMvpMat;
extern GLint uloc_simpletexTexUnit;
extern GLint aloc_simpletexVertex;
extern GLint aloc_simpletexTexcoord;

// Colored overlay shader
extern GLint prog_overlaycol;
extern GLint aloc_overlaycolVertex;
extern GLint aloc_overlaycolColor;

// Textured overlay shader
extern GLint prog_overlaytex;
extern GLint uloc_overlaytexTexUnit;
extern GLint aloc_overlaytexVertex;
extern GLint aloc_overlaytexTexcoord;

// Load/free shaders
extern int  LoadShaders(void);
extern void FreeShaders(void);

#endif // __SHADERS_H
