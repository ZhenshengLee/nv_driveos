/*
 * leaves.c
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
// Construct and draw leaf polygons
//

#include "vbo.h"
#include "leaves.h"
#include "random.h"
#include "array.h"
#include "vector.h"
#include "shaders.h"

// Leaf textures
static GLuint texture;
static GLuint backTexture;

// Size and number of leaves
static float radius;
static int count;

// Leaf vertex info
static Array vertices;
static Array normals;
static Array normalsBack;
static Array colors;
static Array texcoords;

// VBOs for leaf vertices
static unsigned long VBOvertices, VBOnormals, VBOnormalsBack, VBOcolors;
static unsigned long VBOtexcoords;

void
Leaves_initialize(
    GLuint fTex,
    GLuint bTex,
    float r)
{
    Array_init(&vertices, sizeof(float3));
    Array_init(&normals, sizeof(float3));
    Array_init(&normalsBack, sizeof(float3));
    Array_init(&colors, sizeof(float3));
    Array_init(&texcoords, sizeof(float2));

    texture = fTex;
    backTexture = bTex;
    radius = r;
    count = 0;

    if (useVBO) {
        VBOvertices = VBOnormals = VBOnormalsBack = VBOcolors =
            VBOtexcoords = 0;
    }
}

void
Leaves_deinitialize(void)
{
    Array_destroy(&vertices);
    Array_destroy(&normals);
    Array_destroy(&normalsBack);
    Array_destroy(&colors);
    Array_destroy(&texcoords);
}

void
Leaves_clear(void)
{
    Array_clear(&vertices);
    Array_clear(&normals);
    Array_clear(&normalsBack);
    Array_clear(&colors);
    Array_clear(&texcoords);
    count = 0;
}

void
Leaves_setRadius(
    float r)
{
    radius = r;
}

static void
add_a_set(
    float    *front,
    float    *back,
    float    *t,
    float    *v,
    float    *c,
    float4x4 mat)
{
    float3 tmp;

    Array_push(&normals, front);
    Array_push(&normalsBack, back);
    Array_push(&texcoords, t);
    Array_push(&colors, c);
    transform_f3(tmp, mat, v);
    Array_push(&vertices, tmp);
}

void
Leaves_add(
    float4x4 mat)
{
    int i;

    float3 vec = {1.0f, 1.0f, 1.0f};
    float3 front, back;
    float3 c0, c1, c2, c3;

    float2 t0 = {1.0f, 0.0f};
    float2 t1 = {0.0f, 0.0f};
    float2 t2 = {0.0f, 1.0f};
    float2 t3 = {1.0f, 1.0f};

    float3 v0 = {0.0f, -1.0f * radius, 0.0f};
    float3 v1 = {0.0f, 1.0f * radius, 0.0f};
    float3 v2 = {0.0f, 1.0f * radius, 2.0f * radius};
    float3 v3 = {0.0f, -1.0f * radius, 2.0f * radius};

    transformVec_f3(front, mat, vec);

    back[0] = -front[0];
    back[1] = -front[1];
    back[2] = -front[2];

    ++count;

    for (i=0; i<3; i++) {
        c0[i] = (float) GetRandom();
        c1[i] = (float) GetRandom();
        c2[i] = (float) GetRandom();
        c3[i] = (float) GetRandom();
    }

    add_a_set(front, back, t0, v0, c0, mat);
    add_a_set(front, back, t1, v1, c1, mat);
    add_a_set(front, back, t2, v2, c2, mat);
    add_a_set(front, back, t0, v0, c0, mat);
    add_a_set(front, back, t2, v2, c2, mat);
    add_a_set(front, back, t3, v3, c3, mat);
}

void
Leaves_draw(
    int useVBO)
{
    glUseProgram(prog_leaves);

    glEnableVertexAttribArray(aloc_leavesVertex);
    glEnableVertexAttribArray(aloc_leavesNormal);
    glEnableVertexAttribArray(aloc_leavesColor);
    glEnableVertexAttribArray(aloc_leavesTexcoord);

    if (useVBO) {
        glBindBuffer(GL_ARRAY_BUFFER, VBO_NAME);
        glVertexAttribPointer(aloc_leavesVertex,
                              3, GL_FLOAT, GL_FALSE, 0, (void*)VBOvertices);
        glVertexAttribPointer(aloc_leavesNormal,
                              3, GL_FLOAT, GL_FALSE, 0, (void*)VBOnormals);
        glVertexAttribPointer(aloc_leavesColor,
                              3, GL_FLOAT, GL_FALSE, 0, (void*)VBOcolors);
        glVertexAttribPointer(aloc_leavesTexcoord,
                              2, GL_FLOAT, GL_FALSE, 0, (void*)VBOtexcoords);
    } else {
        glVertexAttribPointer(aloc_leavesVertex,
                              3, GL_FLOAT, GL_FALSE, 0, vertices.buffer);
        glVertexAttribPointer(aloc_leavesNormal,
                              3, GL_FLOAT, GL_FALSE, 0, normals.buffer);
        glVertexAttribPointer(aloc_leavesColor,
                              3, GL_FLOAT, GL_FALSE, 0, colors.buffer);
        glVertexAttribPointer(aloc_leavesTexcoord,
                              2, GL_FLOAT, GL_FALSE, 0, texcoords.buffer);
    }

    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
                        GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glBindTexture(GL_TEXTURE_2D, texture);
    glDrawArrays(GL_TRIANGLES, 0, count*6);

    if (useVBO) {
        glVertexAttribPointer(aloc_leavesNormal,
                            3, GL_FLOAT, GL_FALSE, 0, (void*)VBOnormalsBack);
    } else {
        glVertexAttribPointer(aloc_leavesNormal,
                              3, GL_FLOAT, GL_FALSE, 0, normalsBack.buffer);
    }

    glCullFace(GL_FRONT);
    glBindTexture(GL_TEXTURE_2D, backTexture);
    glDrawArrays(GL_TRIANGLES, 0, count*6);

    if (useVBO) {
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    glDisableVertexAttribArray(aloc_leavesVertex);
    glDisableVertexAttribArray(aloc_leavesNormal);
    glDisableVertexAttribArray(aloc_leavesColor);
    glDisableVertexAttribArray(aloc_leavesTexcoord);
    glDisable(GL_BLEND);
}

int
Leaves_polyCount(void)
{
    // the last factor of two is because the front and back are being
    // drawn separately
    return count * 2 * 2;
}

int
Leaves_leafCount(void)
{
    return count;
}

// how much of space this would take in VBO.
int
Leaves_sizeVBO(void)
{
    int numverts = Leaves_leafCount() * 6;
    return
        VBO_align(numverts * 3 * sizeof(GLfloat)) +   // colors
        VBO_align(numverts * 2 * sizeof(GLfloat)) +   // texcoords
        VBO_align(numverts * 3 * sizeof(GLfloat)) +   // normals
        VBO_align(numverts * 3 * sizeof(GLfloat)) +   // normalsBack
        VBO_align(numverts * 3 * sizeof(GLfloat));    // verts
}

void
Leaves_buildVBO(void)
{
    int v = count * 6;
    glBindBuffer(GL_ARRAY_BUFFER, VBO_NAME);
    VBOvertices     = VBO_alloc(v * 3 * sizeof(float));
    VBOnormals      = VBO_alloc(v * 3 * sizeof(float));
    VBOnormalsBack = VBO_alloc(v * 3 * sizeof(float));
    VBOcolors       = VBO_alloc(v * 3 * sizeof(float));
    VBOtexcoords    = VBO_alloc(v * 2 * sizeof(float));

    glBufferSubData(GL_ARRAY_BUFFER, VBOvertices, v * 3 * sizeof(float),
                    vertices.buffer);
    glBufferSubData(GL_ARRAY_BUFFER, VBOnormals, v * 3 * sizeof(float),
                    normals.buffer);
    glBufferSubData(GL_ARRAY_BUFFER, VBOnormalsBack, v * 3 * sizeof(float),
                    normalsBack.buffer);
    glBufferSubData(GL_ARRAY_BUFFER, VBOcolors, v * 3 * sizeof(float),
                    colors.buffer);
    glBufferSubData(GL_ARRAY_BUFFER, VBOtexcoords, v * 2 * sizeof(float),
                    texcoords.buffer);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}
