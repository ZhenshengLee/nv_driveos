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
// Bubble demo shader access
//

#ifndef __SHADERS_H
#define __SHADERS_H

#include <GLES2/gl2.h>

#ifdef ANDROID
#define BUBBLE_PREFIX "bubble/"
#else
#define BUBBLE_PREFIX ""
#endif

// Reflective bubble shader
extern GLint prog_bubble;
extern GLint uloc_bubbleViewMat;
extern GLint uloc_bubbleNormMat;
extern GLint uloc_bubbleProjMat;
extern GLint uloc_bubbleTexUnit;
extern GLint aloc_bubbleVertex;
extern GLint aloc_bubbleNormal;

// Wireframe/point mesh shader
extern GLint prog_mesh;
extern GLint uloc_meshViewMat;
extern GLint uloc_meshProjMat;
extern GLint aloc_meshVertex;

// Env cube shader
extern GLint prog_cube;
extern GLint uloc_cubeProjMat;
extern GLint uloc_cubeTexUnit;
extern GLint aloc_cubeVertex;

// Crosshair shader
extern GLint prog_mouse;
extern GLint uloc_mouseWindow;
extern GLint uloc_mouseCenter;
extern GLint aloc_mouseVertex;

// Function to load all the shaders
extern int
LoadShaders(void);

extern void
FreeShaders(void);

#endif // __SHADERS_H
