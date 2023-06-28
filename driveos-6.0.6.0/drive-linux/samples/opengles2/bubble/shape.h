/*
 * shape.h
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
// Bubble shape description and functions
//

#ifndef __SHAPE_H
#define __SHAPE_H

#include "algebra.h"

#define MAX_DEPTH 64

// Bubble vertex description
typedef struct {
    float p[3];  // point
    float n[3];  // normal
    float v[3];  // velocity
    float h[3];  // home
    float a[3];  // neighborhood velocity for averaging
    int connectedness; // indicates how many triangles this vertex is
                       // connected to, as well as how many edges
} Vertex;

// Bubble edge description
typedef struct {
    short v0id;
    short v1id;
    float l; // length
} Edge;

// Triangle strip forming part of the bubble
typedef struct {
    int            numVerts;  // numTris == numVerts-2
    Vertex         **vertices;
    unsigned short *indices;
} Tristrip;

// Complete bubble description
typedef struct {
    int      numTristrips;
    Tristrip *tristrips;
    int      numEdges;
    Edge     *edges;
    int      numVerts;
    Vertex   *vertices;
    float    final_drag;
    float    initial_drag;
    float    drag;
} Shape;

// Function declarations

Shape*
Bubble_create(
    int depth);

void
Bubble_destroy(
    Shape *b);

void
Bubble_draw(
    Shape        *b,
    unsigned int cube_texture);

void
Bubble_calcNormals(
    Shape *b);

void
Bubble_pick(
    Shape        *b,
    const float3 e,
    const float3 n);

void
Bubble_calcVelocity(
    Shape *b);

void
Bubble_filterVelocity(
    Shape *b);

void
Bubble_drawVertices(
    Shape *b);

void
Bubble_drawEdges(
    Shape *b);

#endif //__SHAPE_H
