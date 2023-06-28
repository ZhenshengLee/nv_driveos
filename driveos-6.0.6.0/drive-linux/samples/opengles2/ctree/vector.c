/*
 * vector.c
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
// Vector/matrix operations
//

#include "nvgldemo.h"
#include "vector.h"

double ident_matrix_d[4][4] = {
    {1.0, 0.0, 0.0, 0.0},
    {0.0, 1.0, 0.0, 0.0},
    {0.0, 0.0, 1.0, 0.0},
    {0.0, 0.0, 0.0, 1.0}
};

float ident_matrix_f[4][4] = {
    {1.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 1.0f, 0.0f},
    {0.0f, 0.0f, 0.0f, 1.0f}
};

static void
copy_f4x4(
    float4x4 dest,
    float4x4 src)
{
    int r, c;
    for (r = 0; r < 4; r++)
    {
        for (c = 0; c < 4; c++)
        {
            dest[r][c] = src[r][c];
        }
    }
}

void
add_f3(
    float3 dest,
    float3 src,
    float3 vec)
{
    dest[0] = src[0] + vec[0];
    dest[1] = src[1] + vec[1];
    dest[2] = src[2] + vec[2];
}

void
addi_f3(
    float3 dest,
    float3 vec)
{
    dest[0] += vec[0];
    dest[1] += vec[1];
    dest[2] += vec[2];
}

void
subtr_f2(
    float2 dest,
    float2 src,
    float2 vec)
{
    dest[0] = src[0] - vec[0];
    dest[1] = src[1] - vec[1];
}

void
mult_f3f(
    float3 dest,
    float3 src,
    float  s)
{
    dest[0] = src[0] * s;
    dest[1] = src[1] * s;
    dest[2] = src[2] * s;
}

void
multi_f3f(
    float3 dest,
    float  s)
{
    dest[0] *= s;
    dest[1] *= s;
    dest[2] *= s;
}

void
mult_f4x4(
    float4x4 dest,
    float4x4 src,
    float4x4 mat)
{
    int r, c, i;
    for (r = 0; r < 4; r++)
    {
        for (c = 0; c < 4; c++)
        {
            dest[r][c] = 0;
            for (i = 0; i < 4; i++)
            {
                dest[r][c] += src[r][i] * mat[i][c];
            }
        }
    }
}

void
multi_f4x4(
    float4x4 dest,
    float4x4 mat)
{
    float4x4 res;
    mult_f4x4(res, dest, mat);
    copy_f4x4(dest, res);
}

void
div_f3(
    float3 dest,
    float3 src,
    float  s)
{
    dest[0] = src[0] / s;
    dest[1] = src[1] / s;
    dest[2] = src[2] / s;
}

void
divi_f3(
    float3 dest,
    float  s)
{
    dest[0] /= s;
    dest[1] /= s;
    dest[2] /= s;
}

// pad the vector with 1 and multiply it by 4x4 matrix.
void
transform_f3(
    float3   dest,
    float4x4 mat,
    float3   vec)
{
    int i, j;
    for (i = 0; i < 3; i++)
    {
        dest[i] = mat[3][i];
        for (j = 0; j < 3; j++)
        {
            dest[i] += mat[j][i] * vec[j];
        }
    }
}

void
transformi_f3(
    float3   dest,
    float4x4 mat)
{
    float3 res;
    transform_f3(res, mat, dest);
    copy_3(dest, res);
}

// pad the vector with 0 and multiply it by 4x4 matrix.
void
transformVec_f3(
    float3   dest,
    float4x4 mat,
    float3   vec)
{
    int i, j;
    for (i = 0; i < 3; i++)
    {
        dest[i] = 0;
        for (j = 0; j < 3; j++)
        {
            dest[i] += mat[j][i] * vec[j];
        }
    }
}

void
transformVeci_f3(
    float3   dest,
    float4x4 mat)
{
    float3 res;
    transformVec_f3(res, mat, dest);
    copy_3(dest, res);
}

void
makeTranslate(
    float4x4 dest,
    float    v0,
    float    v1,
    float    v2)
{
    dest[0][0] = 1; dest[0][1] = 0; dest[0][2] = 0; dest[0][3] = 0;
    dest[1][0] = 0; dest[1][1] = 1; dest[1][2] = 0; dest[1][3] = 0;
    dest[2][0] = 0; dest[2][1] = 0; dest[2][2] = 1; dest[2][3] = 0;
    dest[3][0] = v0; dest[3][1] = v1; dest[3][2] = v2; dest[3][3] = 1;
}

void
makeScale(
    float4x4 dest,
    float    v0,
    float    v1,
    float    v2)
{
    dest[0][0] = v0; dest[0][1] = 0; dest[0][2] = 0; dest[0][3] = 0;
    dest[1][0] = 0; dest[1][1] = v1; dest[1][2] = 0; dest[1][3] = 0;
    dest[2][0] = 0; dest[2][1] = 0; dest[2][2] = v2; dest[2][3] = 0;
    dest[3][0] = 0; dest[3][1] = 0; dest[3][2] = 0; dest[3][3] = 1;
}

static void
quat(
    double4 dest,
    float3  axis,
    double  angle)
{
    double length = SIN(angle / 2.0);
    double r = dot_3(axis, axis);

    if (r == 0.0) {
        set_4(dest, 0.0, 0.0, 0.0, 1.0);
    } else {
        set_4(dest, axis[0] * length, axis[1] * length,
              axis[2] * length, COS(angle / 2.0));
    }
}

void
makeRotation(
    float4x4 dest,
    float    v0,
    float    v1,
    float    v2,
    float    d)
{
    double4 q;
    float3 axis = {v0, v1, v2};
    double x,y,z,w;

    quat(q, axis, d);

    x = q[0];
    y = q[1];
    z = q[2];
    w = q[3];

    set_4(dest[0],
          1.0f + 2.0f * (float) (-y*y - z*z),
          2.0f * (float) ( x*y + w*z),
          2.0f * (float) ( x*z - w*y),
          0.0f);
    set_4(dest[1],
          2.0f * (float) ( x*y - w*z),
          1.0f + 2.0f * (float) (-x*x - z*z),
          2.0f * (float) ( y*z + w*x),
          0.0f);
    set_4(dest[2],
          2.0f * (float) ( x*z + w*y),
          2.0f * (float) ( y*z - w*x),
          1.0f + 2.0f * (float) (-x*x - y*y),
          0.0f);
    set_4(dest[3], 0.0f, 0.0f, 0.0f, 1.0f);
}

void
normalize_f3(
    float3 dest,
    float3 src)
{
    float len = SQRT(dot_3(src, src));
    div_f3(dest, src, len);
}

void
normalizei_f3(
    float3 dest)
{
    float len = dot_3(dest, dest);
    divi_f3(dest, len);
}

float
clamp(
    float v,
    float min,
    float max)
{
    v = (v < min) ? min : v;
    v = (v > max) ? max : v;
    return v;
}

