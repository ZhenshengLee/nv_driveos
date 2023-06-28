/*
 * pictures.c
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
// Texture images
//

#include "nvgldemo.h"
#include "picture.h"
#include "shaders.h"

Picture*
Picture_new(
    GLuint t)
{
    Picture *o = (Picture*)MALLOC(sizeof(Picture));
    o->left = 0.0f;
    o->right = 1.0f;
    o->bottom = 0.0f;
    o->top = 1.0f;
    o->texture = t;
    return o;
}

void
Picture_delete(
    Picture *o)
{
    FREE(o);
}


void
Picture_setPos(
    Picture *o,
    float   l,
    float   r,
    float   b,
    float   t)
{
    o->left = l;
    o->right = r;
    o->bottom = b;
    o->top = t;
}

void
Picture_draw(
    Picture *o)
{
    const float texcoords[] = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        1.0f, 1.0f,
        0.0f, 1.0f
    };
    float vertices[4*2];

    // Load program and bind texture
    glUseProgram(prog_overlaytex);
    glBindTexture(GL_TEXTURE_2D, o->texture);

    // Set up vertex coordinates
    vertices[0] = vertices[6] = o->left;
    vertices[1] = vertices[3] = o->bottom;
    vertices[2] = vertices[4] = o->right;
    vertices[5] = vertices[7] = o->top;

    // Render
    glEnableVertexAttribArray(aloc_overlaytexVertex);
    glEnableVertexAttribArray(aloc_overlaytexTexcoord);
    glVertexAttribPointer(aloc_overlaytexVertex,
                          2, GL_FLOAT, GL_FALSE, 0 , vertices);
    glVertexAttribPointer(aloc_overlaytexTexcoord,
                          2, GL_FLOAT, GL_FALSE, 0, texcoords);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    glDisableVertexAttribArray(aloc_overlaytexVertex);
    glDisableVertexAttribArray(aloc_overlaytexTexcoord);
}
