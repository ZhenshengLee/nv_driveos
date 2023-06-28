/*
 * envcube.c
 *
 * Copyright (c) 2003-2016, NVIDIA CORPORATION. All rights reserved.
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
// Bubble environment cube setup and rendering
//

#include "nvgldemo.h"
#include "envcube.h"
#include "shaders.h"

#include "back_img.h"
#include "bottom_img.h"
#include "front_img.h"
#include "left_img.h"
#include "right_img.h"
#include "top_img.h"

#define MAXSIZE 512

//Texture files for each face
static unsigned char *textures[] = {
    right_img,
    left_img,
    top_img,
    bottom_img,
    back_img,
    front_img
};

// Cube vertices
static float cube_vertices[3*8] = {
    -1.0f,  1.0f, -1.0f, // l t f
    -1.0f, -1.0f, -1.0f, // l b f
     1.0f, -1.0f, -1.0f, // r b f
     1.0f,  1.0f, -1.0f, // r t f
    -1.0f,  1.0f,  1.0f, // l t b
    -1.0f, -1.0f,  1.0f, // l b b
     1.0f, -1.0f,  1.0f, // r b b
     1.0f,  1.0f,  1.0f  // r t b
};

// Cube vertex index list
static unsigned char cube_tristrip[] =
    { 4, 7, 5, 6, 2, 7, 3, 4, 0, 5, 1, 2, 0, 3 };

// Create environment cube resources
EnvCube*
EnvCube_build(void)
{
    EnvCube* e = MALLOC(sizeof(EnvCube));
    if (e == NULL) return NULL;
    MEMSET(e, 0, sizeof(EnvCube));
    e->cubeTexture = NvGlDemoLoadTgaFromBuffer(GL_TEXTURE_CUBE_MAP, 6, textures);

    return e;
}

// Draw surrounding cube
void
EnvCube_draw(
    EnvCube* e)
{
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, e->cubeTexture);

    glEnableVertexAttribArray(aloc_cubeVertex);

    glVertexAttribPointer(aloc_cubeVertex,
                          3, GL_FLOAT, GL_FALSE, 0, cube_vertices);
    glDrawElements(GL_TRIANGLE_STRIP,
                   (sizeof(cube_tristrip) / sizeof(*cube_tristrip)),
                   GL_UNSIGNED_BYTE, cube_tristrip);

    glDisableVertexAttribArray(aloc_cubeVertex);
}

// Retrieve cube map texture
unsigned int
EnvCube_getCube(
    EnvCube* e)
{
    return e->cubeTexture;
}

// Free environment cube resources
void
EnvCube_destroy(
    EnvCube* e)
{
    FREE(e);
}
