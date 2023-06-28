/*
 * lighting_vert.glslv
 *
 * Copyright (c) 2007-2012, NVIDIA CORPORATION. All rights reserved.
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

/* Vertex shader for ground, branches, and leaves (all lit objects) */

//NOTE: any changes to NUM_LIGHTS must also be made to screen.c
#define NUM_LIGHTS 8

// Lighting parameters
uniform int  lights;                // Number of active lights
uniform vec3 lightpos[NUM_LIGHTS];  // Worldspace position of light
uniform vec3 lightcol[NUM_LIGHTS];  // Color of light
const float  atten1 = 1.0;          // Linear attenuation weight
const float  atten2 = 0.1;          // Quadratic attenuation weight

// Projection*modelview matrix
uniform mat4 mvpmatrix;

// Input vertex parameters
attribute vec3 vertex;
attribute vec3 normal;
attribute vec3 color;
attribute vec2 texcoord;

// Output parameters for fragment shader
varying vec3 colorVar;
varying vec2 texcoordVar;

void main() {

    vec3  totLight;
    vec3  normaldir;
    vec3  lightvec;
    float lightdist;
    vec3  lightdir;
    float ldotn;
    float attenuation;
    int   i;

    // Initialize lighting contribution
    totLight = vec3(0.0, 0.0, 0.0);

    // Normalize normal vector
    normaldir = normalize(normal);

    // Add contribution of each light
    for (i=0; i<lights; i++) {
        // Compute direction/distance to light
        lightvec  = lightpos[i] - vertex;
        lightdist = length(lightvec);
        lightdir  = lightvec / lightdist;

        // Compute dot product of light and normal vectors
        ldotn = clamp(dot(lightdir, normaldir), 0.0, 1.0);

        // Compute attenuation factor
        attenuation = (atten1 + atten2 * lightdist) * lightdist;

        // Add contribution of this light
        totLight += (ldotn / attenuation) * lightcol[i];
    }

    // Output material * total light
    colorVar = totLight * color;

    // Pass through the texture coordinate
    texcoordVar = texcoord;

    // Transform the vertex
    gl_Position = mvpmatrix * vec4(vertex,1.0);
}
