/*
 * branches.h
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
// Branch polygon setup/rendering
//

#ifndef __BRANCHES_H
#define __BRANCHES_H

#include <GLES2/gl2.h>
#include "vector.h"

// Number of faces for cylinders representing each branch
#define BRANCHES_FACETS 5

// Initialization and clean-up
void Branches_initialize(GLuint t);
void Branches_deinitialize(void);
void Branches_clear(void);

// Creation
int  Branches_add(float n[3], float tc[2], float v[3]);
void Branches_addIndex(unsigned int);
void Branches_generateStump(int *lower);
void Branches_buildCylinder(
        int *idx, float4x4 mat, float taper, float texcoordY, GLboolean low);

// Query
int  Branches_polyCount(void);
int  Branches_branchCount(void);
int  Branches_sizeVBO(void);
int  Branches_numVertices(void);

// Rendering
void Branches_draw(int useVBO);
void Branches_buildVBO(void);

#endif // __BRANCHES_H
