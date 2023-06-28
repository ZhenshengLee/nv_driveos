/*
 * branches.c
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
// Construct/draw branch polygons
//

#include "nvgldemo.h"
#include "tree.h"
#include "branches.h"
#include "array.h"
#include "vbo.h"
#include "shaders.h"

// Precomputed cos/sin values for facets of branch cylinder
static float2 trig[BRANCHES_FACETS + 1];

// Branch vertex data
static Array vertices;
static Array normals;
static Array texcoords;
static Array indices;

// VBOs for branch vertex data
static unsigned long VBOvertices;
static unsigned long VBOnormals;
static unsigned long VBOtexcoords;

// Branch texture
static GLuint texture;

// Initialize branch data structures
void
Branches_initialize(
    GLuint t)
{
    int i;

    Array_init(&vertices, sizeof(float3));
    Array_init(&normals, sizeof(float3));
    Array_init(&texcoords, sizeof(float2));
    Array_init(&indices, sizeof(unsigned int));

    texture = t;

    if (useVBO) {
        VBOvertices = 0;
        VBOnormals = 0;
        VBOtexcoords = 0;
    }

    for (i=0; i< BRANCHES_FACETS + 1; ++i) {
        float u = (float)(2.0f * PI / (float)BRANCHES_FACETS) * (float)i;
        set_2(trig[i], COS(u), SIN(u));
    }
}

// Free branch data
void
Branches_deinitialize(void)
{
    Array_destroy(&vertices);
    Array_destroy(&normals);
    Array_destroy(&texcoords);
    Array_destroy(&indices);
}

// Reset branch data
void
Branches_clear(void)
{
    Array_clear(&vertices);
    Array_clear(&normals);
    Array_clear(&texcoords);
    Array_clear(&indices);
}

// Add branch vertex
int
Branches_add(
    float3 n,
    float2 tc,
    float3 v)
{
    int index = vertices.elemCount;
    ASSERT(normals.elemCount == index);
    ASSERT(texcoords.elemCount == index);

    Array_push(&normals, n);
    Array_push(&texcoords, tc);
    Array_push(&vertices, v);

    return index;
}

// Add vertex to index list
void
Branches_addIndex(
   unsigned int i)
{
   Array_push(&indices, &i);
}

// Draw all branches
void
Branches_draw(
    int useVBO)
{
    int i, size, stride;

    glUseProgram(prog_solids);

    glVertexAttrib3f(aloc_solidsColor, 1.0,1.0,1.0);
    glEnableVertexAttribArray(aloc_solidsVertex);
    glEnableVertexAttribArray(aloc_solidsNormal);
    glEnableVertexAttribArray(aloc_solidsTexcoord);

    if (useVBO) {
        glBindBuffer(GL_ARRAY_BUFFER, VBO_NAME);
        glVertexAttribPointer(aloc_solidsVertex,
                              3, GL_FLOAT, GL_FALSE, 0, (void*)VBOvertices);
        glVertexAttribPointer(aloc_solidsNormal,
                              3, GL_FLOAT, GL_FALSE, 0, (void*)VBOnormals);
        glVertexAttribPointer(aloc_solidsTexcoord,
                              2, GL_FLOAT, GL_FALSE, 0, (void*)VBOtexcoords);
    } else {
        glVertexAttribPointer(aloc_solidsVertex,
                              3, GL_FLOAT, GL_FALSE, 0, vertices.buffer);
        glVertexAttribPointer(aloc_solidsNormal,
                              3, GL_FLOAT, GL_FALSE, 0, normals.buffer);
        glVertexAttribPointer(aloc_solidsTexcoord,
                              2, GL_FLOAT, GL_FALSE, 0, texcoords.buffer);

    }

    size = indices.elemCount;
    stride = (BRANCHES_FACETS + 1) * 2;
    ASSERT(size % stride == 0);

    glBindTexture(GL_TEXTURE_2D, texture);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    for (i=0; i<size; i+=stride) {
        glDrawElements(GL_TRIANGLE_STRIP, stride, GL_UNSIGNED_INT,
                       Array_get(&indices, i));
    }

    if (useVBO) {
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    glDisableVertexAttribArray(aloc_solidsVertex);
    glDisableVertexAttribArray(aloc_solidsNormal);
    glDisableVertexAttribArray(aloc_solidsTexcoord);
}

// Query number of branch polygons
int
Branches_polyCount(void)
{
    int stride = (BRANCHES_FACETS+1)*2;
    int cylCount = indices.elemCount/stride;
    return cylCount * BRANCHES_FACETS * 2;
}

// Query number of branch segments
int
Branches_branchCount(void)
{
    int stride = (BRANCHES_FACETS+1)*2;
    int cylCount = indices.elemCount/stride;
    return (cylCount+1) / 2;
}

// Construct VBOs
void
Branches_buildVBO(void)
{
    int v;

    v = vertices.elemCount;

    glBindBuffer(GL_ARRAY_BUFFER, VBO_NAME);
    VBOvertices  = VBO_alloc(v * 3 * sizeof(float));
    VBOnormals   = VBO_alloc(v * 3 * sizeof(float));
    VBOtexcoords = VBO_alloc(v * 2 * sizeof(float));

    glBufferSubData(GL_ARRAY_BUFFER, VBOvertices, v * 3 * sizeof(float),
                    vertices.buffer);
    glBufferSubData(GL_ARRAY_BUFFER, VBOnormals, v * 3 * sizeof(float),
                    normals.buffer);
    glBufferSubData(GL_ARRAY_BUFFER, VBOtexcoords, v * 2 * sizeof(float),
                    texcoords.buffer);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// Query total size of VBOs
int
Branches_sizeVBO(void)
{
    int numverts = Branches_numVertices();
    return
        VBO_align(numverts*3*sizeof(GLfloat)) +   // normals
        VBO_align(numverts*2*sizeof(GLfloat)) +   // texcoords
        VBO_align(numverts*3*sizeof(GLfloat));    // verts.
}

// Query number of branch vertices
int
Branches_numVertices(void)
{
    return vertices.elemCount;
}

// Generate vertices for base of tree
void
Branches_generateStump(
    int *lower)
{
    int i;
    float branchRadius;

    for (i = 0; i < BRANCHES_FACETS+1; ++i)
    {
        float t = 1.0f / BRANCHES_FACETS * i;
        float g0 = trig[i][0];
        float g1 = trig[i][1];

        Branches_addIndex(lower[i]);
        branchRadius = treeParams[TREE_PARAM_BRANCH_SIZE];
        {
            float3 n = {g0, g1, 0.5f};
            float2 tc = {t, -branchRadius - 0.5f};
            float3 v = {g0 * branchRadius * 1.5f,
                        g1 * branchRadius * 1.5f,
                        -0.5f};
            Branches_addIndex(Branches_add(n, tc, v));
        }
    }
}

// Generate vertices for a branch segment
void
Branches_buildCylinder(
    int       *idx,
    float4x4  mat,
    float     taper,
    float     texcoordY,
    GLboolean low)
{
    int i;
    for (i=0; i<(BRANCHES_FACETS+1); ++i)
    {
        float t = 1.0f / BRANCHES_FACETS * i;
        float3 n, v;
        float *g = trig[i];
        float branchRadius;

        set_3(n, g[0], g[1], 0.0f);
        transformVeci_f3(n, mat);
        branchRadius = treeParams[TREE_PARAM_BRANCH_SIZE];
        if (low)
        {
            set_3(v, g[0] * branchRadius,
                     g[1] * branchRadius,
                     branchRadius);
        }
        else
        {
            set_3(v, g[0] * branchRadius * taper,
                     g[1] * branchRadius * taper,
                     1.0f - branchRadius);
        }
        transformi_f3(v, mat);

        {
            float2 tc = {t, texcoordY};
            idx[i] = Branches_add(n, tc, v);
        }
    }
}
