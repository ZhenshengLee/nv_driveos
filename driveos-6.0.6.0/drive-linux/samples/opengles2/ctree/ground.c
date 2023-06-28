/*
 * ground.c
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
// Ground plane
//

#include "ground.h"
#include "random.h"
#include "vbo.h"
#include "vector.h"
#include "shaders.h"

// Mesh resolution
#define RESOLUTION 10
#define MAXSIZE (RESOLUTION + 1)
#define MAXAREA (MAXSIZE * MAXSIZE)

// Ground texture
static GLuint texture;

// Ground vertex info
static float3 vertices [MAXSIZE*MAXSIZE];
static float3 normals  [MAXSIZE*MAXSIZE];
static float2 texcoords[MAXSIZE*MAXSIZE];
static float3 colors   [MAXSIZE*MAXSIZE];
static GLubyte indicies[RESOLUTION*2*MAXSIZE];

// Ground VBO indices
static unsigned long VBOvertices, VBOnormals, VBOtexcoords, VBOcolors;

// Utility function to handling index wrapping
static int
wrap(
    int i)
{
    while (i > RESOLUTION) i -= RESOLUTION;
    while (i < 0)          i += RESOLUTION;
    return i;
}

// Build the ground plane vertices
static void
build(void)
{
    int i, j, step;
    GLubyte *indx;

    for (j = 0; j < MAXSIZE; ++j)
    {
        float y = ((float)j) / ((float)RESOLUTION) * 2.0f - 1.0f;
        float py = y * GROUND_SIZE / 2.0f;
        float t = (((float)j) / ((float)RESOLUTION));

        for (i = 0; i< MAXSIZE; ++i)
        {
            float pz = ((float)GetRandom() - 0.5f) * 0.04f * GROUND_SIZE;

            float x = ((float)i)/((float)RESOLUTION) * 2.0f - 1.0f;
            float px = x * GROUND_SIZE / 2.0f;
            float s = (((float)i) / ((float)RESOLUTION));
            float r2;

            set_3(vertices [j*MAXSIZE+i], px, py, pz);
            set_2(texcoords[j*MAXSIZE+i], s, t);

            r2 = (x*x+y*y);
            if (r2>1.0) r2 = 0.0;
            else r2 = 1.0f-r2;
            set_3(colors[j*MAXSIZE+i], r2, r2, r2);
        }
    }

    // wrap edge
    for (j=0; j<RESOLUTION; j++) {
        vertices[RESOLUTION*MAXSIZE+j][2] = vertices[j][2];
        vertices[j*MAXSIZE+RESOLUTION][2] = vertices[j*MAXSIZE][2];
    }

    vertices[RESOLUTION*MAXSIZE+RESOLUTION][2] = vertices[0][2];

    // Construct the normal vectors.
    step = (int)(GROUND_SIZE / ((float)RESOLUTION));
    for (j = 0; j < MAXSIZE; ++j) {

        for (i = 0; i < MAXSIZE; ++i) {

            float3 v0 = {(float) step, 0.0f,
                         vertices[j*MAXSIZE+wrap(i + 1)][2]
                         - vertices[j*MAXSIZE+wrap(i - 1)][2]};
            float3 v1 = {0.0f, (float) step,
                         vertices[wrap(j + 1)*MAXSIZE+i][2]
                         - vertices[wrap(j - 1)*MAXSIZE+i][2]};
            float3 v;
            cross_3(v, v0, v1);
            normalize_f3(normals[j*MAXSIZE+i], v);
        }
    }

    // Wrap the normal vectors.
    for (j = 0; j < MAXSIZE; j++) {
        copy_3(normals[j*MAXSIZE+RESOLUTION], normals[j*MAXSIZE]);
        copy_3(normals[RESOLUTION*MAXSIZE+j], normals[j]);
    }
    copy_3(normals[RESOLUTION*MAXSIZE+RESOLUTION], normals[0]);

    // Make the indicies for the triangle strips.
    indx = indicies;
    for (j=0; j<RESOLUTION; ++j) {
        for (i=0; i<MAXSIZE; ++i) {
            *indx++ = (j+1)*MAXSIZE+i;
            *indx++ = j*MAXSIZE+i;
        }
    }
}

// Intialize the ground
void
Ground_initialize(
    GLuint t)
{
    // Create the terrain mesh
    build();

    // Save the texture
    texture = t;
}

// Release ground resources
void
Ground_deinitialize(void)
{
    // No action required
}

// Draw the ground
void
Ground_draw(
    int useVBO)
{
    glUseProgram(prog_solids);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    glBindTexture(GL_TEXTURE_2D, texture);

    int j;

    glEnableVertexAttribArray(aloc_solidsVertex);
    glEnableVertexAttribArray(aloc_solidsNormal);
    glEnableVertexAttribArray(aloc_solidsColor);
    glEnableVertexAttribArray(aloc_solidsTexcoord);

    if (useVBO) {
        glBindBuffer(GL_ARRAY_BUFFER, VBO_NAME);
        glVertexAttribPointer(aloc_solidsVertex,
                              3, GL_FLOAT, GL_FALSE, 0, (void*)VBOvertices);
        glVertexAttribPointer(aloc_solidsNormal,
                              3, GL_FLOAT, GL_FALSE, 0, (void*)VBOnormals);
        glVertexAttribPointer(aloc_solidsColor,
                              3, GL_FLOAT, GL_FALSE, 0, (void*)VBOcolors);
        glVertexAttribPointer(aloc_solidsTexcoord,
                              2, GL_FLOAT, GL_FALSE, 0, (void*)VBOtexcoords);
    }  else {
        glVertexAttribPointer(aloc_solidsVertex,
                              3, GL_FLOAT, GL_FALSE, 0, vertices);
        glVertexAttribPointer(aloc_solidsNormal,
                              3, GL_FLOAT, GL_FALSE, 0, normals);
        glVertexAttribPointer(aloc_solidsColor,
                              3, GL_FLOAT, GL_FALSE, 0, colors);
        glVertexAttribPointer(aloc_solidsTexcoord,
                              2, GL_FLOAT, GL_FALSE, 0, texcoords);
    }

    for (j=0; j<RESOLUTION; ++j) {
        glDrawElements(GL_TRIANGLE_STRIP, 2*MAXSIZE, GL_UNSIGNED_BYTE,
                       (const GLvoid*) &indicies[j*2*MAXSIZE]);
    }

    if (useVBO) {
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    glDisableVertexAttribArray(aloc_solidsVertex);
    glDisableVertexAttribArray(aloc_solidsNormal);
    glDisableVertexAttribArray(aloc_solidsColor);
    glDisableVertexAttribArray(aloc_solidsTexcoord);
}

// Query number of ground polygons
int
Ground_polyCount(void)
{
    return RESOLUTION * RESOLUTION * 2;
}

// Query size of ground VBOs
int
Ground_sizeVBO(void)
{
    return VBO_align(sizeof(float3) * MAXSIZE * MAXSIZE) +     // vertices
           VBO_align(sizeof(float3) * MAXSIZE * MAXSIZE) +     // normals
           VBO_align(sizeof(float2) * MAXSIZE * MAXSIZE) +     // texcoords
           VBO_align(sizeof(float3) * MAXSIZE * MAXSIZE);      // colours
}

// Construct ground VBOs
void
Ground_buildVBO(void)
{
    glBindBuffer(GL_ARRAY_BUFFER, VBO_NAME);

    VBOvertices  = VBO_alloc(sizeof(float3) * MAXSIZE * MAXSIZE);
    VBOnormals   = VBO_alloc(sizeof(float3) * MAXSIZE * MAXSIZE);
    VBOtexcoords = VBO_alloc(sizeof(float2) * MAXSIZE * MAXSIZE);
    VBOcolors    = VBO_alloc(sizeof(float3) * MAXSIZE * MAXSIZE);

    glBufferSubData(GL_ARRAY_BUFFER, VBOvertices,
                    sizeof(float3) * MAXSIZE * MAXSIZE, vertices);
    glBufferSubData(GL_ARRAY_BUFFER, VBOnormals,
                    sizeof(float3) * MAXSIZE * MAXSIZE, normals);
    glBufferSubData(GL_ARRAY_BUFFER, VBOtexcoords,
                    sizeof(float2) * MAXSIZE * MAXSIZE, texcoords);
    glBufferSubData(GL_ARRAY_BUFFER, VBOcolors,
                    sizeof(float3) * MAXSIZE * MAXSIZE, colors);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}
