/*
 * vbo.c
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
// VBO functions
//

#include "nvgldemo.h"
#include "vbo.h"

static long vboptr = 0;
static unsigned int vbosize = 0;
int vboInitialized = 0;
int useVBO = 0;

GLboolean
VBO_init(void)
{
    vboInitialized = 1;
    useVBO = 1;
    return GL_TRUE;
}

void
VBO_deinit(void)
{
    GLuint bufs[] = { VBO_NAME };

    if (vboInitialized) {
        glDeleteBuffers(sizeof bufs / sizeof *bufs, bufs);
    }

    vbosize = vboptr = 0;
}

GLboolean
VBO_setup(
    int size)
{
    int res;

    // clear out prior errors
    while (glGetError() != GL_NO_ERROR);

    VBO_deinit();
    glBindBuffer(GL_ARRAY_BUFFER, VBO_NAME);
    glBufferData(GL_ARRAY_BUFFER, size, NULL, GL_DYNAMIC_DRAW);
    if ((res = glGetError()) == GL_NO_ERROR) {
        vbosize = size;
        return GL_TRUE;
    }
    else {
        do {
            // Print out the error
            NvGlDemoLog("Error: %x\n", res);
        } while ((res = glGetError()) != GL_NO_ERROR);
        return GL_FALSE;
    }
}

unsigned long
VBO_alloc(
    int size)
{
    unsigned long ret = 0;
    size = VBO_align(size);
    if (((unsigned int) size) > vbosize) {
        NvGlDemoLog("VBO out of space\n");
    } else {
        ret = vboptr;
        vboptr += size;
        vbosize -= size;
    }
    return ret;
}
