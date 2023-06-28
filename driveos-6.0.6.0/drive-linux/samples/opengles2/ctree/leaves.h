/*
 * leaves.h
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
// Leaf functions
//

#ifndef __LEAVES_H
#define __LEAVES_H

#include <GLES2/gl2.h>
#include "vector.h"

// Initialization and clean-up
void Leaves_initialize(GLuint front, GLuint back, float radius);
void Leaves_deinitialize(void);
void Leaves_clear(void);
void Leaves_setRadius(float r);
void Leaves_add(float4x4 m);

// Query
int Leaves_polyCount(void);
int Leaves_leafCount(void);
int Leaves_sizeVBO(void);

// Rendering
void Leaves_buildVBO(void);
void Leaves_draw(int use_VBO);

#endif // __LEAVES_H
