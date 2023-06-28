/*
 * shape.c
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
// Bubble sphere setup, animation, and rendering
//

#include "nvgldemo.h"
#include "shape.h"
#include "shaders.h"

// Used to hold 1.0f/n values
#define ONE_OVER_SIZE 257
static float OneOver[ONE_OVER_SIZE];

// Render the bubble as polygons
void
Bubble_draw(
    Shape        *b,
    unsigned int cube_texture)
{
    Tristrip *tstrip = b->tristrips;
    int      i;

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cube_texture);

    glEnableVertexAttribArray(aloc_bubbleVertex);
    glEnableVertexAttribArray(aloc_bubbleNormal);
    glVertexAttribPointer(aloc_bubbleVertex,
                          3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          &(b->vertices[0].p[0]));
    glVertexAttribPointer(aloc_bubbleNormal,
                          3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          &(b->vertices[0].n[0]));

    for (i = 0; i<b->numTristrips; i++, tstrip++) {
        glDrawElements(GL_TRIANGLE_STRIP,
                       tstrip->numVerts,
                       GL_UNSIGNED_SHORT,
                       tstrip->indices);
    }

    glDisableVertexAttribArray(aloc_bubbleVertex);
    glDisableVertexAttribArray(aloc_bubbleNormal);
}

// Render the bubble as a mesh
void
Bubble_drawEdges(
    Shape *b)
{
    Tristrip *tstrip = b->tristrips;
    unsigned short *edgeIndices;
    int i, j;

    glEnableVertexAttribArray(aloc_meshVertex);
    glVertexAttribPointer(aloc_meshVertex,
                          3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          &(b->vertices[0].p[0]));

    // TODO: Allocating and freeing every time == bad
    edgeIndices = (unsigned short*)
        MALLOC((((b->tristrips)->numVerts+1)>>1) * sizeof(unsigned short));
    for (i = 0; i<b->numTristrips; i++, tstrip++) {
        // draw the zig-zags
        glDrawElements(GL_LINE_STRIP,
                       tstrip->numVerts,
                       GL_UNSIGNED_SHORT,
                       tstrip->indices);
        // draw one "side" of the tristrip (don't need to draw the other
        // side because our neighboring tristrip will do it)
        for ( j=0 ; j<tstrip->numVerts ; j+=2 ) {
            edgeIndices[j>>1] = tstrip->indices[j];
        }
        glDrawElements(GL_LINE_STRIP,
                       (tstrip->numVerts+1)>>1,
                       GL_UNSIGNED_SHORT,
                       edgeIndices);
    }
    FREE(edgeIndices);
    glDisableVertexAttribArray(aloc_meshVertex);
}

// Render the bubbles as points
void
Bubble_drawVertices(
    Shape *b)
{
    Tristrip *tstrip = b->tristrips;
    unsigned short *edgeIndices;
    int i, j;

    glEnableVertexAttribArray(aloc_meshVertex);
    glVertexAttribPointer(aloc_meshVertex,
                          3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          &(b->vertices[0].p[0]));

    // TODO: Allocating and freeing every time == bad
    edgeIndices = (unsigned short*)
        MALLOC((((b->tristrips)->numVerts+1)>>1)*sizeof(unsigned short));
    for (i = 0; i < b->numTristrips; i++, tstrip++) {
        // draw one "side" of the tristrip (don't need to draw the other
        // side because our neighboring tristrip will do it)
        for (j = 0; j < tstrip->numVerts; j += 2) {
            edgeIndices[j>>1] = tstrip->indices[j];
        }
        glDrawElements(GL_POINTS,
                       (tstrip->numVerts+1)>>1,
                       GL_UNSIGNED_SHORT,
                       edgeIndices);
    }
    glDisableVertexAttribArray(aloc_meshVertex);
    FREE(edgeIndices);
}

// Calculate surface normal at each bubble vertex
void
Bubble_calcNormals(
    Shape *b)
{
    Vertex **verts;
    Tristrip *tstrip;
    int i, j;
    Vertex *v0, *v1, *v2;
    float ax, ay, az;
    float bx, by, bz;
    float nx, ny, nz;

    for (i=0, tstrip = b->tristrips; i<b->numTristrips; i++, tstrip++) {
        float sign = 1.0f;
        verts = tstrip->vertices;
        v1 = verts[0];
        v2 = verts[1];
        ax = v2->p[0] - v1->p[0];
        ay = v2->p[1] - v1->p[1];
        az = v2->p[2] - v1->p[2];
        for (j = 0; j < tstrip->numVerts-2; j++, verts++, sign *= -1.0f) {
            v0 = v1;
            v1 = v2;
            v2 = verts[2];
            // Copy over the previous vector.  We invert the direction
            // every other vertex to ping-pong the normal.
            bx = sign * ax;
            by = sign * ay;
            bz = sign * az;
            ax = v2->p[0] - v1->p[0];
            ay = v2->p[1] - v1->p[1];
            az = v2->p[2] - v1->p[2];
            nx = ay*bz - az*by;
            ny = az*bx - ax*bz;
            nz = ax*by - ay*bx;
            v0->n[0] += nx;
            v0->n[1] += ny;
            v0->n[2] += nz;
            v1->n[0] += nx;
            v1->n[1] += ny;
            v1->n[2] += nz;
            v2->n[0] += nx;
            v2->n[1] += ny;
            v2->n[2] += nz;
        }
    }
}

// Add a new vertex at a given latitude/longitude
static void
MakeNewVertex(
    Shape *shape,
    float latAngle,
    float longAngle)
{
    Vertex *vert;
    float  cosLat;

    vert = shape->vertices + shape->numVerts++;
    cosLat = (float)COS(latAngle),
    vert->p[0] = cosLat * (float)COS(longAngle),
    vert->p[1] = cosLat * (float)SIN(longAngle),
    vert->p[2] = (float)SIN(latAngle);
    // set home position
    vert->h[0] = vert->p[0];
    vert->h[1] = vert->p[1];
    vert->h[2] = vert->p[2];
    // zero initial velocity
    vert->v[0] =
    vert->v[1] =
    vert->v[2] = 0.0f;
    // zero average velocity
    vert->a[0] =
    vert->a[1] =
    vert->a[2] = 0.0f;
}

// Add a new edge between two vertices
static void
MakeNewEdge(
    Shape *shape,
    int   vertId0,
    int   vertId1)
{
    Vertex *v0 = shape->vertices + vertId0,
           *v1 = shape->vertices + vertId1;
    Edge   *edge;
    float  dx, dy, dz;

    edge = shape->edges + shape->numEdges++;
    edge->v0id = vertId0;
    edge->v1id = vertId1;

    dx = v1->p[0] - v0->p[0];
    dy = v1->p[1] - v0->p[1];
    dz = v1->p[2] - v0->p[2];
    edge->l = (float)SQRT(dx*dx + dy*dy + dz*dz);
}

// Create a new bubble with a given subdivision level
Shape*
Bubble_create(
    int depth)
{
#   define EXPECTED_VERTS      ( 4*depth*depth + 2)
#   define EXPECTED_EDGES      (12*depth*depth)
#   define NUM_TRISTRIPS       ( 4*depth)
#   define NUM_VERTS_PER_STRIP ( 2*depth + 2)
    Shape    *shape;
    int      i, j, quadrant, vertId;
    int      vertTab[MAX_DEPTH+1][MAX_DEPTH+1];
    float    oo_depth = 1.0f/(float)depth;
    int      numCuts, startCol, startRow;
    Tristrip *tstrip;
    // check that we're not overly ambitious
    if (depth > MAX_DEPTH) {
        NvGlDemoLog("Increase MAX_DEPTH");
        return NULL;
    }
    if (4*(depth + 1) > ONE_OVER_SIZE-1) {
        NvGlDemoLog("Increase ONE_OVER_SIZE");
        return NULL;
    }

    shape = MALLOC(sizeof(Shape));
    if (!shape) {
        NvGlDemoLog("out of memory.");
        return NULL;
    }
    shape->final_drag = 0.99f;
    shape->initial_drag = 0.7f;
    shape->drag = shape->initial_drag;

    // initialize OneOver table
    OneOver[0] = 0.0f;
    for (i = 1; i < ONE_OVER_SIZE; i++) {
        OneOver[i] = 1.0f/(float)i;
    }

    // allocate all the space we'll use up front
    shape->numVerts = 0;
    shape->numEdges = 0;
    shape->numTristrips = NUM_TRISTRIPS;
    shape->vertices  = (Vertex*)   MALLOC(EXPECTED_VERTS * sizeof(Vertex));
    MEMSET(shape->vertices, 0, EXPECTED_VERTS * sizeof(Vertex));
    shape->edges     = (Edge*)     MALLOC(EXPECTED_EDGES * sizeof(Edge));
    MEMSET(shape->edges, 0, EXPECTED_EDGES * sizeof(Edge));
    shape->tristrips = (Tristrip*) MALLOC(NUM_TRISTRIPS  * sizeof(Tristrip));
    MEMSET(shape->tristrips, 0, NUM_TRISTRIPS  * sizeof(Tristrip));
    for (i = 0; i < NUM_TRISTRIPS; i++) {
        shape->tristrips[i].numVerts = NUM_VERTS_PER_STRIP;
        shape->tristrips[i].vertices = (Vertex**)
            MALLOC(NUM_VERTS_PER_STRIP * sizeof(Vertex*));
        MEMSET(shape->tristrips[i].vertices, 0,
               NUM_VERTS_PER_STRIP * sizeof(Vertex*));
        shape->tristrips[i].indices  = (unsigned short*)
            MALLOC(NUM_VERTS_PER_STRIP * sizeof(unsigned short));
        MEMSET(shape->tristrips[i].indices, 0,
               NUM_VERTS_PER_STRIP * sizeof(unsigned short));
    }

    // generate all the vertices
    MakeNewVertex(shape, (float)(0.5f*PI), 0.0f);
    for (i = 1; i < 2*depth; i++) {
        int numCuts = (i <= depth) ? 4*i : 4*(2*depth - i);
        float latAngle = (float)(0.5f * PI * (1.0f - (float)i*oo_depth));
        for (j=0; j < numCuts; j++) {
            float longAngle = (float)(2.0f * PI * (float)j * OneOver[numCuts]);
            MakeNewVertex(shape, latAngle, longAngle);
        }
    }
    MakeNewVertex(shape, (float)(-0.5f*PI), 0.0f);
    if (shape->numVerts != EXPECTED_VERTS) {
        NvGlDemoLog("Whooops, wrong number of verts"
                    " (%d overserved, %d expected)\n",
                    shape->numVerts,
                    EXPECTED_VERTS);
    }
    for (quadrant = 0; quadrant < 4; quadrant++) {
        // generate vertTab, a table of vertex indices used to build
        // edges and tristrips
        vertTab[0][0] = 0;
        vertTab[depth][depth] = shape->numVerts-1;
        for (i=1, vertId=1; i<2*depth; i++) {
            if (i <= depth) {
                numCuts  = 4*i;
                startCol = i;
                startRow = 0;
            } else {
                numCuts  = 4*(2*depth-i);
                startCol = depth;
                startRow = i-depth;
            }
            for (j=0; j <= (numCuts>>2); j++) {
                // the 4th quadrant is special because it has to wrap
                // back around to the 1st quadrant
                if ((quadrant == 3) &&
                    ((startCol-j == 0) || (startRow+j == depth))) {
                    vertTab[startCol-j][startRow+j] =
                        vertId + j + quadrant*(numCuts>>2) - numCuts;
                } else {
                    vertTab[startCol-j][startRow+j] =
                        vertId + j + quadrant*(numCuts>>2);
                }
            }
            vertId += numCuts;
        }
        // generate edges
        // "forward slash" edges
        for (i=1; i <= depth; i++) {
            for (j=0; j < depth; j++) {
                MakeNewEdge(shape, vertTab[i][j], vertTab[i][j+1]);
            }
        }
        // "backward slash" edges
        for (j = 0; j < depth; j++) {
            for (i = 0; i < depth; i++) {
                MakeNewEdge(shape, vertTab[i][j], vertTab[i+1][j]);
            }
        }
        // "horizontal" edges
        for (i = 1; i < 2*depth; i++) {
            int rowEdges, startCol, startRow;
            if (i <= depth) {
                rowEdges = i;
                startCol = i;
                startRow = 0;
            } else {
                rowEdges = 2*depth-i;
                startCol = depth;
                startRow = i-depth;
            }
            for ( j=0 ; j<rowEdges ; j++ ) {
                MakeNewEdge(shape, vertTab[startCol-j]  [startRow+j],
                    vertTab[startCol-j-1][startRow+j+1]);
            }
        }
        // generate tristrips
        tstrip = shape->tristrips + quadrant * depth;
        for (j = 0; j < depth; j++, tstrip++) {
            for (i = 0; i <= depth; i++) {
                tstrip->indices[2*i+0] = (unsigned short)vertTab[i][j+1];
                tstrip->indices[2*i+1] = (unsigned short)vertTab[i][j];
                tstrip->vertices[2*i+0] = shape->vertices + vertTab[i][j];
                tstrip->vertices[2*i+1] = shape->vertices + vertTab[i][j+1];
            }
            for (i = 0; i < tstrip->numVerts - 2; i++) {
            // bump the connectivity count
                tstrip->vertices[i+0]->connectedness++;
                tstrip->vertices[i+1]->connectedness++;
                tstrip->vertices[i+2]->connectedness++;
            }
        }
    }
    if (shape->numEdges != EXPECTED_EDGES) {
        NvGlDemoLog("Whoops, wrong number of edges"
                    " (%d observed, %d expected)\n",
                    shape->numEdges,
                    EXPECTED_EDGES);
    }
    return shape;
}

// Free bubble descriiption and resources
void
Bubble_destroy(
    Shape *b)
{
    int i;
    // check to see if we need to free memory
    if (b->vertices) {
        FREE(b->vertices);
        b->vertices = NULL;
    }
    if (b->edges) {
        FREE(b->edges);
        b->edges = NULL;
    }
    if (b->tristrips) {
        for (i = 0; i < b->numTristrips; i++) {
            FREE(b->tristrips[i].vertices);
            FREE(b->tristrips[i].indices);
        }
        FREE(b->tristrips);
        b->tristrips = NULL;
    }
    FREE(b);
}

// Apply spring forces to update the velocity of each vertex
void
Bubble_calcVelocity(
    Shape *b)
{
    Vertex *vert;
    Edge *edge;
    int i;
    float k1 = 0.005f; // spring home
    float k2 = 0.6f;   // edge spring

    for (i=0, vert=b->vertices; i<b->numVerts; i++, vert++) {
        float half_oo_nc = 0.5f * OneOver[vert->connectedness];
        vert->v[0] += (vert->h[0] - vert->p[0]) * k1 + vert->n[0] * half_oo_nc;
        vert->v[1] += (vert->h[1] - vert->p[1]) * k1 + vert->n[1] * half_oo_nc;
        vert->v[2] += (vert->h[2] - vert->p[2]) * k1 + vert->n[2] * half_oo_nc;
        vert->a[0] =
        vert->a[1] =
        vert->a[2] =
        vert->n[0] =
        vert->n[1] =
        vert->n[2] = 0.0f;
    }

    for (i=0, edge = b->edges; i < b->numEdges; i++, edge++) {
        Vertex *vert0 = b->vertices + edge->v0id;
        Vertex *vert1 = b->vertices + edge->v1id;
        float fx,  fy,  fz;
        float fLen, scale;
        float v0scale, v1scale;
        fx = vert1->p[0] - vert0->p[0];
        fy = vert1->p[1] - vert0->p[1];
        fz = vert1->p[2] - vert0->p[2];
        fLen = (float)SQRT(fx*fx + fy*fy + fz*fz);
        scale = k2 * (fLen - edge->l) / fLen;
        // Check for vertices connected to 4 edges.  Since all but
        // 6 verts are connected to 6 edges, scale up the weight
        // of this edge to make if effectively the same as if the
        // vertex was connected to 6 edges.
        v1scale = (vert1->connectedness == 4) ? 1.5f * scale : scale;
        v0scale = (vert0->connectedness == 4) ? 1.5f * scale : scale;
        vert0->a[0] += (vert1->v[0] -= v1scale * fx);
        vert0->a[1] += (vert1->v[1] -= v1scale * fy);
        vert0->a[2] += (vert1->v[2] -= v1scale * fz);
        vert1->a[0] += (vert0->v[0] += v0scale * fx);
        vert1->a[1] += (vert0->v[1] += v0scale * fy);
        vert1->a[2] += (vert0->v[2] += v0scale * fz);
    }
}

// Apply drag coefficient to slow each vertex
void
Bubble_filterVelocity(
    Shape *b)
{
    Vertex *vert;
    int i;

    b->drag += 0.01f;
    if (b->drag > b->final_drag)
    b->drag = b->final_drag;
    for (i=0, vert = b->vertices; i < b->numVerts; i++, vert++) {
        float tenth_oo_c = 0.1f * OneOver[vert->connectedness];
        float v0, v1, v2;
        v0 = (vert->v[0] * 0.9f) + (vert->a[0] * tenth_oo_c);
        v1 = (vert->v[1] * 0.9f) + (vert->a[1] * tenth_oo_c);
        v2 = (vert->v[2] * 0.9f) + (vert->a[2] * tenth_oo_c);
        vert->p[0] += v0;
        vert->p[1] += v1;
        vert->p[2] += v2;
        vert->v[0] = v0 * b->drag;
        vert->v[1] = v1 * b->drag;
        vert->v[2] = v2 * b->drag;
    }
}

// Compute distance from selection point and vertex
static float
Bubble_pickDistance(
    const float3 e,
    const float3 n,
    const float3 t)
{
    float s;
    float3 t2;
    float3 p;
    vec_prescribe(t2, t);
    vec_subs(t2, e);
    s = vec_dot(n, t2)/vec_dot(n, n);
    vec_prescribe(p, n);
    vec_scale(p, s);
    vec_add(p, e);
    vec_prescribe(t2, t);
    vec_subs(t2, p);
    return vec_dot(t2, t2);
}


// Identify vertex closest to screen selection point
void
Bubble_pick(
    Shape       *b,
    const float3 e,
    const float3 n)
{
    float closest_distance = 1.0e+10;
    Vertex *vert, *closest = NULL;
    int i;
    float *p0;
    float3 t;

    for (i = 0, vert = b->vertices; i < b->numVerts; i++, vert++) {
        float3 vn;
        float distance;
        vec_prescribe(t, vert->p);
        vec_prescribe(vn, vert->n);
        distance = Bubble_pickDistance(e, n, t);
        if ((distance < closest_distance) && (vec_dot(vn, n) < 0.0f)) {
            closest_distance = distance;
            closest = vert;
        }
    }
    if (!closest) return;
    p0 = closest->p;
    for (i=0, vert = b->vertices; i < b->numVerts; i++, vert++) {
        float *p1 = vert->p;
        float s;
        t[0] = p1[0] - p0[0];
        t[1] = p1[1] - p0[1];
        t[2] = p1[2] - p0[2];
        s = (float)POW(10.0f, t[0]*t[0] + t[1]*t[1] + t[2]*t[2]);
        s = 1.0f/s;
        vert->v[0] -= vert->p[0] * s * 0.15f;
        vert->v[1] -= vert->p[1] * s * 0.15f;
        vert->v[2] -= vert->p[2] * s * 0.15f;
    }
}
