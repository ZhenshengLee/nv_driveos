/*
 * vbo.h
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
// Optional functions for rendering with VBOs
//

#ifndef __VBO_H
#define __VBO_H

#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>

// Uncomment to enable tracking counts of vertices rendered
//#define DUMP_VERTS_PER_SEC
#ifdef DUMP_VERTS_PER_SEC
extern int   vertCount;
#define glDrawArrays(mode, start, count)                     \
                    vertCount += (count);                    \
                    glDrawArrays(mode, start, count)
#define glDrawElements(mode, count, type, indices)           \
                    vertCount += (count);                    \
                    glDrawElements(mode, count, type, indices)
#endif

// All objects share a single VBO
#define VBO_NAME 1

// Macro to align elements properly when packed into the VBO
#define VBO_ALIGNMENT 4
#define VBO_align(size) ((size + VBO_ALIGNMENT - 1) & ~(VBO_ALIGNMENT - 1))

// Flags indicating VBO is requested and properly initialized
extern int   vboInitialized;
extern int   useVBO;

// Initialization and clean-up
GLboolean    VBO_init  (void);
void         VBO_deinit(void);
unsigned long VBO_alloc (int size);
GLboolean    VBO_setup (int size);

#endif // __VBO_H
