/*
 * algebra.c
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
// Vector/matrix/quaternion operations used by the bubble calculations
//

// TODO: Some of this is redundant with nvgldemo. We should fold the
// TODO:   remaining functionality into it and then use those functions.

#include "nvgldemo.h"
#include "algebra.h"

// apply matrix to vector, result stored on this vector
void vec_transform(float3 v, float4x4 m)
{
    float x = v[0]*m[0][0] + v[1]*m[1][0] + v[2]*m[2][0];
    float y = v[0]*m[0][1] + v[1]*m[1][1] + v[2]*m[2][1];
    float z = v[0]*m[0][2] + v[1]*m[1][2] + v[2]*m[2][2];
    v[0] = x;
    v[1] = y;
    v[2] = z;
}

// apply matrix to point, result stored on this point
void pnt_transform(float3 v, float4x4 m)
{
    float x = v[0]*m[0][0] + v[1]*m[1][0] + v[2]*m[2][0] + m[3][0];
    float y = v[0]*m[0][1] + v[1]*m[1][1] + v[2]*m[2][1] + m[3][1];
    float z = v[0]*m[0][2] + v[1]*m[1][2] + v[2]*m[2][2] + m[3][2];
    v[0] = x;
    v[1] = y;
    v[2] = z;
}

static void mat2_multiply(float m1[4], float m2[4])
{
    float m3[4];
    m3[0] = m1[0]*m2[0] + m1[1]*m2[2];
    m3[1] = m1[0]*m2[1] + m1[1]*m2[3];
    m3[2] = m1[2]*m2[0] + m1[3]*m2[2];
    m3[3] = m1[2]*m2[1] + m1[3]*m2[3];
    m1[0] = m3[0];
    m1[1] = m3[1];
    m1[2] = m3[2];
    m1[3] = m3[3];
}

static void mat2_scale(float m[4], float a)
{
    m[0] *= a;
    m[1] *= a;
    m[2] *= a;
    m[3] *= a;
}

static void mat2_invert(float m[4])
{
    float q = m[0];
    m[0] = m[3];
    m[3] = q;
    q = 1.0f/(m[0]*m[3]-m[2]*m[1]);
    m[2] = -m[2];
    m[1] = -m[1];
    mat2_scale(m, q);
}

// inverses the matrix
void mat_invert (float4x4 m)
{
    float m1[4], m2[4], m3[4], m4[4];
    float4x4 res;
    m1[0] = m[2][0];
    m1[1] = m[3][0];
    m1[2] = m[2][1];
    m1[3] = m[3][1];
    m2[0] = m[3][3];
    m2[1] = -m[3][2];
    m2[2] = -m[2][3];
    m2[3] = m[2][2];
    mat2_multiply(m1, m2);
    m2[0] = m[0][2];
    m2[1] = m[1][2];
    m2[2] = m[0][3];
    m2[3] = m[1][3];
    mat2_multiply(m1, m2);
    mat2_scale(m1, 1.0f/(m[2][2]*m[3][3]-m[3][2]*m[2][3]));
    m1[0] = m[0][0] - m1[0];
    m1[1] = m[1][0] - m1[1];
    m1[2] = m[0][1] - m1[2];
    m1[3] = m[1][1] - m1[3];
    mat2_invert(m1);
    m2[0] = m[2][2];
    m2[1] = m[3][2];
    m2[2] = m[2][3];
    m2[3] = m[3][3];
    mat2_invert(m2);
    res[0][0] = m1[0];
    res[0][1] = m1[2];
    res[1][0] = m1[1];
    res[1][1] = m1[3];
    m3[0] = -m1[0];
    m3[1] = -m1[1];
    m3[2] = -m1[2];
    m3[3] = -m1[3];
    m4[0] = m[2][0];
    m4[1] = m[3][0];
    m4[2] = m[2][1];
    m4[3] = m[3][1];
    mat2_multiply(m3, m4);
    mat2_multiply(m3, m2);
    res[2][0] = m3[0];
    res[2][1] = m3[2];
    res[3][0] = m3[1];
    res[3][1] = m3[3];
    m3[0] = -m2[0];
    m3[1] = -m2[1];
    m3[2] = -m2[2];
    m3[3] = -m2[3];
    m4[0] = m[0][2];
    m4[1] = m[1][2];
    m4[2] = m[0][3];
    m4[3] = m[1][3];
    mat2_multiply(m3, m4);
    mat2_multiply(m3, m1);
    res[0][2] = m3[0];
    res[0][3] = m3[2];
    res[1][2] = m3[1];
    res[1][3] = m3[3];
    m3[0] = -m3[0];
    m3[1] = -m3[1];
    m3[2] = -m3[2];
    m3[3] = -m3[3];
    m4[0] = m[2][0];
    m4[1] = m[3][0];
    m4[2] = m[2][1];
    m4[3] = m[3][1];
    mat2_multiply(m3, m4);
    mat2_multiply(m3, m2);
    res[2][2] = m3[0]+m2[0];
    res[2][3] = m3[2]+m2[2];
    res[3][2] = m3[1]+m2[1];
    res[3][3] = m3[3]+m2[3];
    m[0][0] = res[0][0]; m[0][1] = res[0][1];
    m[0][2] = res[0][2]; m[0][3] = res[0][3];
    m[1][0] = res[1][0]; m[1][1] = res[1][1];
    m[1][2] = res[1][2]; m[1][3] = res[1][3];
    m[2][0] = res[2][0]; m[2][1] = res[2][1];
    m[2][2] = res[2][2]; m[2][3] = res[2][3];
    m[3][0] = res[3][0]; m[3][1] = res[3][1];
    m[3][2] = res[3][2]; m[3][3] = res[3][3];
}

// inverses the left-upper 3x3 part of matrix
void mat_invert_part(float4x4 m)
{
    float4x4 res;
    float det = m[0][0]*m[1][1]*m[2][2]+
        m[1][0]*m[2][1]*m[0][2]+
        m[2][0]*m[0][1]*m[1][2]-
        m[0][0]*m[2][1]*m[1][2]-
        m[1][0]*m[0][1]*m[2][2]-
        m[2][0]*m[1][1]*m[0][2];
    res[0][0] = (m[1][1]*m[2][2]-m[2][1]*m[1][2])/det;
    res[1][0] = (m[0][1]*m[2][2]-m[2][1]*m[0][2])/det;
    res[2][0] = (m[0][1]*m[1][2]-m[1][1]*m[0][2])/det;
    res[0][1] = (m[1][0]*m[2][2]-m[2][0]*m[1][2])/det;
    res[1][1] = (m[0][0]*m[2][2]-m[2][0]*m[0][2])/det;
    res[2][1] = (m[0][0]*m[1][2]-m[1][0]*m[0][2])/det;
    res[0][2] = (m[1][0]*m[2][1]-m[2][0]*m[1][1])/det;
    res[1][2] = (m[0][0]*m[2][1]-m[2][0]*m[0][1])/det;
    res[2][2] = (m[0][0]*m[1][1]-m[1][0]*m[0][1])/det;
    m[0][0] = res[0][0];
    m[0][1] = -res[1][0];
    m[0][2] = res[2][0];
    m[1][0] = -res[0][1];
    m[1][1] = res[1][1];
    m[1][2] = -res[2][1];
    m[2][0] = res[0][2];
    m[2][1] = -res[1][2];
    m[2][2] = res[2][2];
}

// transposes the matrix
void mat_transpose(float4x4 m)
{
    float s;
    s = m[0][1]; m[0][1] = m[1][0]; m[1][0] = s;
    s = m[0][2]; m[0][2] = m[2][0]; m[2][0] = s;
    s = m[0][3]; m[0][3] = m[3][0]; m[3][0] = s;
    s = m[1][2]; m[1][2] = m[2][1]; m[2][1] = s;
    s = m[1][3]; m[1][3] = m[3][1]; m[3][1] = s;
    s = m[2][3]; m[2][3] = m[3][2]; m[3][2] = s;
}

// an identity
void quat_identity(Quat *a)
{
    a->r = 1.0f;
    a->i = 0.0f;
    a->j = 0.0f;
    a->k = 0.0f;
}

// sets a quaternion
void quat_setf3(Quat *a, float nr, float ni, float nj, float nk)
{
    a->r=nr;
    a->i=ni;
    a->j=nj;
    a->k=nk;
}

// sets a quaternion
void quat_setfv(Quat *a, float radians, const float3 axis)
{
    float dst_l = (float)SIN(radians/2.0);
    float src_l = (float)SQRT(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
    if (src_l == 0.0f) {
        a->i = 0.0f;
        a->j = 0.0f;
        a->k = 0.0f;
        a->r = 1.0f;
    } else {
        float s = dst_l / src_l;
        a->i = axis[0] * s;
        a->j = axis[1] * s;
        a->k = axis[2] * s;
        a->r = (float)COS(radians/2.0);
    }
}

// multiplies quaternions, result on the first argument
void quat_multiply(Quat *a, const Quat *b)
{
    Quat p;
    p.r = a->r*b->r - a->i*b->i - a->j*b->j - a->k*b->k;
    p.i = a->j*b->k - b->j*a->k + a->r*b->i + b->r*a->i;
    p.j = a->k*b->i - b->k*a->i + a->r*b->j + b->r*a->j;
    p.k = a->i*b->j - b->i*a->j + a->r*b->k + b->r*a->k;
    a->r = p.r;
    a->i = p.i;
    a->j = p.j;
    a->k = p.k;
}

// makes the matrix (used in transformations) from quaternion
void quat_mat(float4x4 m, Quat *a)
{
    m[0][0] = 1.0f - 2.0f * (a->j*a->j + a->k*a->k);
    m[0][1] = 2.0f * (a->i*a->j + a->r*a->k);
    m[0][2] = 2.0f * (a->i*a->k - a->r*a->j);
    m[0][3] = 0.0f;
    m[1][0] = 2.0f * (a->i*a->j - a->r*a->k);
    m[1][1] = 1.0f - 2.0f * (a->i*a->i + a->k*a->k);
    m[1][2] = 2.0f * (a->j*a->k + a->r*a->i);
    m[1][3] = 0.0f;
    m[2][0] = 2.0f * (a->i*a->k + a->r*a->j);
    m[2][1] = 2.0f * (a->j*a->k - a->r*a->i);
    m[2][2] = 1.0f - 2.0f * (a->i*a->i + a->j*a->j);
    m[2][3] = 0.0f;
    m[3][0] = 0.0f;
    m[3][1] = 0.0f;
    m[3][2] = 0.0f;
    m[3][3] = 1.0f;
}

// a=b;
void quat_prescribe(Quat *a, Quat *b)
{
    a->r = b->r;
    a->i = b->i;
    a->j = b->j;
    a->k = b->k;
}

// set matrix to identity
void mat_identity(float4x4 m)
{
    m[0][1] =
    m[0][2] =
    m[0][3] =
    m[1][0] =
    m[1][2] =
    m[1][3] =
    m[2][0] =
    m[2][1] =
    m[2][3] =
    m[3][0] =
    m[3][1] =
    m[3][2] = 0.0f;
    m[0][0] =
    m[1][1] =
    m[2][2] =
    m[3][3] = 1.0f;
}

// muliplies matrices, the result on the first argument
void mat_multiply(float4x4 m0, float4x4 m1)
{
    int r, c, i;
    float m[4];
    for(r = 0; r < 4; r++) {
        m[0] = m[1] = m[2] = m[3] = 0.0f;
        for(c = 0; c < 4; c++) {
            for(i = 0; i < 4; i++) {
                m[c] += m0[i][r] * m1[c][i];
            }
        }
        for(c = 0; c < 4; c++) { m0[c][r] = m[c]; }
    }
}

// simmulates glTranslatef
void mat_translate(float4x4 m, float x, float y, float z)
{
    float4x4 m2;
    m2[0][0] =
    m2[1][1] =
    m2[2][2] =
    m2[3][3] = 1.0f;
    m2[0][1] =
    m2[0][2] =
    m2[0][3] =
    m2[1][0] =
    m2[1][2] =
    m2[1][3] =
    m2[2][0] =
    m2[2][1] =
    m2[2][3] = 0.0f;
    m2[3][0] = x;
    m2[3][1] = y;
    m2[3][2] = z;
    mat_multiply(m, m2);
}

// simmulates glOrthof
void mat_ortho(float4x4 m,
      float l, float r, float b, float t, float n, float f)
{
    float4x4 m1;
    m1[0][1] =
    m1[0][2] =
    m1[0][3] =
    m1[1][0] =
    m1[1][2] =
    m1[1][3] =
    m1[2][0] =
    m1[2][1] =
    m1[2][3] = 0.0f;
    m1[0][0] = 2.0f / (r - l);
    m1[1][1] = 2.0f / (t - b);
    m1[2][2] = 2.0f / (f - n);
    m1[3][0] = (r + l) / (l - r);
    m1[3][1] = (t + b) / (b - t);
    m1[3][2] = (f + n) / (n - f);
    m1[3][3] = 1.0f;
    mat_multiply(m, m1);
}

// simmulates glFrustumf
void mat_frustum(float4x4 m,
        float l, float r, float b, float t, float n, float f)
{
    float4x4 m1;
    m1[0][1] =
    m1[0][2] =
    m1[0][3] =
    m1[1][0] =
    m1[1][2] =
    m1[1][3] =
    m1[3][0] =
    m1[3][1] =
    m1[3][3] = 0.0f;
    m1[0][0] = 2.0f * n / (r - l);
    m1[1][1] = 2.0f * n / (t - b);
    m1[2][0] = (r + l) / (r - l);
    m1[2][1] = (t + b) / (t - b);
    m1[2][2] = (f + n) / (n - f);
    m1[2][3] = -1.0f;
    m1[3][2] = -2.0f * f * n / (f - n);
    mat_multiply(m, m1);
}

// simmulates glScalef
void mat_scale(float4x4 m, float x, float y, float z)
{
    float4x4 m1;
    m1[0][0] = x;
    m1[1][1] = y;
    m1[2][2] = z;
    m1[3][3] = 1.0f;
    m1[0][1] =
    m1[0][2] =
    m1[0][3] =
    m1[1][0] =
    m1[1][2] =
    m1[1][3] =
    m1[2][0] =
    m1[2][1] =
    m1[2][3] =
    m1[3][0] =
    m1[3][1] =
    m1[3][2] = 0.0f;
    mat_multiply(m, m1);
}

// multiply vector by scalar
void vec_scale(float3 v, const float s)
{
    v[0] *= s;
    v[1] *= s;
    v[2] *= s;
}

// add vectors, the result on the first argument (a=a+b)
void vec_add(float3 a, const float3 b)
{
    a[0] += b[0];
    a[1] += b[1];
    a[2] += b[2];
}

// substract vectors, the result on the first argument
void vec_subs(float3 a, const float3 b)
{
    a[0] -= b[0];
    a[1] -= b[1];
    a[2] -= b[2];
}

// dot product of vectors
float vec_dot(const float3 a, const float3 b)
{
    float res = 0.0f;
    res += a[0]*b[0];
    res += a[1]*b[1];
    res += a[2]*b[2];
    return res;
}

// a=b;
void vec_prescribe(float3 a, const float3 b)
{
    a[0] = b[0];
    a[1] = b[1];
    a[2] = b[2];
}

