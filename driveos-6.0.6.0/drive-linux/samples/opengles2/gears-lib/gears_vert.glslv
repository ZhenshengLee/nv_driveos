/*
 * gears_vert.glslv
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

// Constant uniforms
uniform vec3 light_dir;  // Light 0 direction
uniform mat4 proj_mat;   // Projection matrix.

// Per-frame/per-object uniforms
uniform mat4 mview_mat;  // Model-view matrix
uniform vec3 material;   // Ambient and diffuse material

// Per-vertex attributes
attribute vec3 pos_attr;
attribute vec3 nrm_attr;

// Output vertex color
varying vec3 col_var;

void main()
{
    // Transformed position is projection * modelview * pos
    gl_Position = proj_mat * mview_mat * vec4(pos_attr, 1.0);

    // We know modelview matrix has no scaling, so don't need a separate
    //   inverse-transpose for transforming normals
    vec3 normal = vec3(normalize(mview_mat * vec4(nrm_attr, 0.0)));

    // Compute dot product of light and normal vectors
    float ldotn = max(dot(normal, light_dir), 0.0);

    // Compute affect of lights and clamp
    //   Global ambient is default GL value of 0.2
    //   Light0 ambient is default GL value of 0.0
    //   Light0 diffuse is default GL value of 1.0
    col_var = min((ldotn+0.2) * material, 1.0);
}
