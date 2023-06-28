/*
 * nvgldemo_math.c
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

//
// In GLES2, the fixed function pipeline, including matrix stacks,
//   are not available. Applications are responsible for performing
//   their own matrix manipulations, when needed, and uploading the
//   results to the shaders. This file provides examples of common
//   matrix operations, as well as a few other simple math utilities.
//   Matrices assume column-major order, as is standard for GL.
//

#include "nvgldemo.h"

// Check whether two floating point values differ by only a small amount
int
eq(float a, float b)
{
    float diff = a-b;
    if (diff < 0) {
        diff = -diff;
    }
    return diff <= eps;
}

// Initialize a 4x4 matrix to identity
//   m <- I
void
NvGlDemoMatrixIdentity(
    float m[16])
{
    MEMSET(m, 0, sizeof(float) * 16);
    m[4 * 0 + 0] = m[4 * 1 + 1] = m[4 * 2 + 2] = m[4 * 3 + 3] = 1.0;
}

// Check whether two 4x4 matrices are equal (within a small tolerance)
//   (a ~== b)
int
NvGlDemoMatrixEquals(
    float a[16], float b[16])
{
    int i;
    for (i = 0; i < 16; ++i) {
        if (!eq(a[i], b[i]))
            return 0;
    }

    return 1;
}

// Transpose a 4x4 matrix in place
//   m <- m-transpose
void
NvGlDemoMatrixTranspose(
    float m[16])
{
    int i, j;
    float t;
    for (i = 1; i < 4; ++i) {
        for (j = 0; j < i; ++j)  {
            t = m[4*i+j];
            m[4*i+j] = m[4*j+i];
            m[4*j+i] = t;
        }
    }
}

// Multiply the second 4x4 matrix into the first
//   m0 <- m0 * m1
void
NvGlDemoMatrixMultiply(
    float m0[16], float m1[16])
{
    int r, c, i;
    for (r = 0; r < 4; r++) {
        float m[4] = {0.0, 0.0, 0.0, 0.0};
        for (c = 0; c < 4; c++) {
            for (i = 0; i < 4; i++) {
                m[c] += m0[4 * i + r] * m1[4 * c + i];
            }
        }
        for (c = 0; c < 4; c++) {
            m0[4 * c + r] = m[c];
        }
    }
}

// Multiply the 3x3 matrix into the 4x4
//   m0 <- m0 * m1
void
NvGlDemoMatrixMultiply_4x4_3x3(
    float m0[16], float m1[9])
{
    int r, c, i;
    for (r = 0; r < 4; r++) {
        float m[3] = {0.0, 0.0, 0.0};
        for (c = 0; c < 3; c++) {
            for (i = 0; i < 3; i++) {
                m[c] += m0[4 * i + r] * m1[3 * c + i];
            }
        }
        for (c = 0; c < 3; c++) {
            m0[4 * c + r] = m[c];
        }
    }
}

// Multiply the second 3x3 matrix into the first
//   m0 <- m0 * m1
void
NvGlDemoMatrixMultiply_3x3(
    float m0[9], float m1[9])
{
    int r, c, i;
    for (r = 0; r < 3; r++) {
        float m[3] = {0.0, 0.0, 0.0};
        for (c = 0; c < 3; c++) {
            for(i = 0; i < 3; i++) {
                m[c] += m0[3 * i + r] * m1[3 * c + i];
            }
        }
        for (c = 0; c < 3; c++) {
            m0[3 * c + r] = m[c];
        }
    }
}

// Apply perspective projection to a 4x4 matrix
//   m <- m * perspective(l,r,b,t,n,f)
void
NvGlDemoMatrixFrustum(
    float m[16],
    float l, float r, float b, float t, float n, float f)
{
    float m1[16];
    float rightMinusLeftInv, topMinusBottomInv, farMinusNearInv, twoNear;

    rightMinusLeftInv = 1.0f / (r - l);
    topMinusBottomInv = 1.0f / (t - b);
    farMinusNearInv = 1.0f / (f - n);
    twoNear = 2.0f * n;

    m1[ 0] = twoNear * rightMinusLeftInv;
    m1[ 1] = 0.0f;
    m1[ 2] = 0.0f;
    m1[ 3] = 0.0f;

    m1[ 4] = 0.0f;
    m1[ 5] = twoNear * topMinusBottomInv;
    m1[ 6] = 0.0f;
    m1[ 7] = 0.0f;

    m1[ 8] = (r + l) * rightMinusLeftInv;
    m1[ 9] = (t + b) * topMinusBottomInv;
    m1[10] = -(f + n) * farMinusNearInv;
    m1[11] = -1.0f;

    m1[12] = 0.0f;
    m1[13] = 0.0f;
    m1[14] = -(twoNear * f) * farMinusNearInv;
    m1[15] = 0.0f;

    NvGlDemoMatrixMultiply(m, m1);
}

// Apply orthographic projection to a 4x4 matrix
//   m <- m * ortho(l,r,b,t,n,f) 
void
NvGlDemoMatrixOrtho(
    float m[16],
    float l, float r, float b, float t, float n, float f)
{
    float m1[16];
    float rightMinusLeftInv, topMinusBottomInv, farMinusNearInv;

    rightMinusLeftInv = 1.0f / (r - l);
    topMinusBottomInv = 1.0f / (t - b);
    farMinusNearInv = 1.0f / (f - n);

    m1[ 0] = 2.0f * rightMinusLeftInv;
    m1[ 1] = 0.0f;
    m1[ 2] = 0.0f;
    m1[ 3] = 0.0f;

    m1[ 4] = 0.0f;
    m1[ 5] = 2.0f * topMinusBottomInv;
    m1[ 6] = 0.0f;
    m1[ 7] = 0.0f;

    m1[ 8] = 0.0f;
    m1[ 9] = 0.0f;
    m1[10] = -2.0f * farMinusNearInv;
    m1[11] = 0.0f;

    m1[12] = -(r + l) * rightMinusLeftInv;
    m1[13] = -(t + b) * topMinusBottomInv;
    m1[14] = -(f + n) * farMinusNearInv;
    m1[15] = 1.0f;

    NvGlDemoMatrixMultiply(m, m1);
}

// Apply scaling to a 4x4 matrix
//   m <- m * scale(x,y,z)
void
NvGlDemoMatrixScale(
    float m[16], float x, float y, float z)
{
    float m1[16];
    NvGlDemoMatrixIdentity(m1);

    m1[4 * 0 + 0] = x;
    m1[4 * 1 + 1] = y;
    m1[4 * 2 + 2] = z;

    NvGlDemoMatrixMultiply(m, m1);
}

// Apply translation to a 4x4 matrix
//   m <- m + translate(x,y,z)
void
NvGlDemoMatrixTranslate(
    float m[16], float x, float y, float z)
{
    float m1[16];
    NvGlDemoMatrixIdentity(m1);

    m1[4 * 3 + 0] = x;
    m1[4 * 3 + 1] = y;
    m1[4 * 3 + 2] = z;

    NvGlDemoMatrixMultiply(m, m1);
}

// Initialize a 3x3 rotation matrix
//   m <- rotate(th,x,y,z)
void
NvGlDemoMatrixRotate_create3x3(
    float m[9],
    float theta, float x, float y, float z)
{
    float len = SQRT(x * x + y * y + z * z);
    float u0 = x / len;
    float u1 = y / len;
    float u2 = z / len;
    float rad = (float)(theta / 180 * PI);
    float c = COS(rad);
    float s = SIN(rad);
    m[3 * 0 + 0] = u0 * u0 + c * (1 - u0 * u0) + s * 0;
    m[3 * 0 + 1] = u0 * u1 + c * (0 - u0 * u1) + s * u2;
    m[3 * 0 + 2] = u0 * u2 + c * (0 - u0 * u2) - s * u1;

    m[3 * 1 + 0] = u1 * u0 + c * (0 - u1 * u0) - s * u2;
    m[3 * 1 + 1] = u1 * u1 + c * (1 - u1 * u1) + s * 0;
    m[3 * 1 + 2] = u1 * u2 + c * (0 - u1 * u2) + s * u0;

    m[3 * 2 + 0] = u2 * u0 + c * (0 - u2 * u0) + s * u1;
    m[3 * 2 + 1] = u2 * u1 + c * (0 - u2 * u1) - s * u0;
    m[3 * 2 + 2] = u2 * u2 + c * (1 - u2 * u2) + s * 0;
}

// Apply a rotation to a 4x4 matrix
//   m <- m * rotate(th,x,y,z)
void
NvGlDemoMatrixRotate(
    float m[16], float theta, float x, float y, float z)
{
    float r[9];
    NvGlDemoMatrixRotate_create3x3(r, theta, x, y, z);
    NvGlDemoMatrixMultiply_4x4_3x3(m, r);
}

// Apply a rotation to a 3x3 matrix
//   m <- m * rotate(th,x,y,z)
void
NvGlDemoMatrixRotate_3x3(
    float m[9], float theta, float x, float y, float z)
{
    float r[9];
    NvGlDemoMatrixRotate_create3x3(r, theta, x, y, z);
    NvGlDemoMatrixMultiply_3x3(m, r);
}

// Compute the determinant of a 4x4 matrix
//   det(m)
float
NvGlDemoMatrixDeterminant(
    float m[16])
{
    return m[4*0+3] * m[4*1+2] * m[4*2+1] * m[4*3+0]
         - m[4*0+2] * m[4*1+3] * m[4*2+1] * m[4*3+0]
         - m[4*0+3] * m[4*1+1] * m[4*2+2] * m[4*3+0]
         + m[4*0+1] * m[4*1+3] * m[4*2+2] * m[4*3+0]
         + m[4*0+2] * m[4*1+1] * m[4*2+3] * m[4*3+0]
         - m[4*0+1] * m[4*1+2] * m[4*2+3] * m[4*3+0]
         - m[4*0+3] * m[4*1+2] * m[4*2+0] * m[4*3+1]
         + m[4*0+2] * m[4*1+3] * m[4*2+0] * m[4*3+1]
         + m[4*0+3] * m[4*1+0] * m[4*2+2] * m[4*3+1]
         - m[4*0+0] * m[4*1+3] * m[4*2+2] * m[4*3+1]
         - m[4*0+2] * m[4*1+0] * m[4*2+3] * m[4*3+1]
         + m[4*0+0] * m[4*1+2] * m[4*2+3] * m[4*3+1]
         + m[4*0+3] * m[4*1+1] * m[4*2+0] * m[4*3+2]
         - m[4*0+1] * m[4*1+3] * m[4*2+0] * m[4*3+2]
         - m[4*0+3] * m[4*1+0] * m[4*2+1] * m[4*3+2]
         + m[4*0+0] * m[4*1+3] * m[4*2+1] * m[4*3+2]
         + m[4*0+1] * m[4*1+0] * m[4*2+3] * m[4*3+2]
         - m[4*0+0] * m[4*1+1] * m[4*2+3] * m[4*3+2]
         - m[4*0+2] * m[4*1+1] * m[4*2+0] * m[4*3+3]
         + m[4*0+1] * m[4*1+2] * m[4*2+0] * m[4*3+3]
         + m[4*0+2] * m[4*1+0] * m[4*2+1] * m[4*3+3]
         - m[4*0+0] * m[4*1+2] * m[4*2+1] * m[4*3+3]
         - m[4*0+1] * m[4*1+0] * m[4*2+2] * m[4*3+3]
         + m[4*0+0] * m[4*1+1] * m[4*2+2] * m[4*3+3];
}

// Invert a 4x4 matrix in place
//   m <- inv(m)
void
NvGlDemoMatrixInverse(
    float m[16])
{
    float a[16];
    float det;
    int i;
    float b[16], e[16];

    a[4*0+0] = m[4*1+2]*m[4*2+3]*m[4*3+1]
             - m[4*1+3]*m[4*2+2]*m[4*3+1]
             + m[4*1+3]*m[4*2+1]*m[4*3+2]
             - m[4*1+1]*m[4*2+3]*m[4*3+2]
             - m[4*1+2]*m[4*2+1]*m[4*3+3]
             + m[4*1+1]*m[4*2+2]*m[4*3+3];
    a[4*0+1] = m[4*0+3]*m[4*2+2]*m[4*3+1]
             - m[4*0+2]*m[4*2+3]*m[4*3+1]
             - m[4*0+3]*m[4*2+1]*m[4*3+2]
             + m[4*0+1]*m[4*2+3]*m[4*3+2]
             + m[4*0+2]*m[4*2+1]*m[4*3+3]
             - m[4*0+1]*m[4*2+2]*m[4*3+3];
    a[4*0+2] = m[4*0+2]*m[4*1+3]*m[4*3+1]
             - m[4*0+3]*m[4*1+2]*m[4*3+1]
             + m[4*0+3]*m[4*1+1]*m[4*3+2]
             - m[4*0+1]*m[4*1+3]*m[4*3+2]
             - m[4*0+2]*m[4*1+1]*m[4*3+3]
             + m[4*0+1]*m[4*1+2]*m[4*3+3];
    a[4*0+3] = m[4*0+3]*m[4*1+2]*m[4*2+1]
             - m[4*0+2]*m[4*1+3]*m[4*2+1]
             - m[4*0+3]*m[4*1+1]*m[4*2+2]
             + m[4*0+1]*m[4*1+3]*m[4*2+2]
             + m[4*0+2]*m[4*1+1]*m[4*2+3]
             - m[4*0+1]*m[4*1+2]*m[4*2+3];
    a[4*1+0] = m[4*1+3]*m[4*2+2]*m[4*3+0]
             - m[4*1+2]*m[4*2+3]*m[4*3+0]
             - m[4*1+3]*m[4*2+0]*m[4*3+2]
             + m[4*1+0]*m[4*2+3]*m[4*3+2]
             + m[4*1+2]*m[4*2+0]*m[4*3+3]
             - m[4*1+0]*m[4*2+2]*m[4*3+3];
    a[4*1+1] = m[4*0+2]*m[4*2+3]*m[4*3+0]
             - m[4*0+3]*m[4*2+2]*m[4*3+0]
             + m[4*0+3]*m[4*2+0]*m[4*3+2]
             - m[4*0+0]*m[4*2+3]*m[4*3+2]
             - m[4*0+2]*m[4*2+0]*m[4*3+3]
             + m[4*0+0]*m[4*2+2]*m[4*3+3];
    a[4*1+2] = m[4*0+3]*m[4*1+2]*m[4*3+0]
             - m[4*0+2]*m[4*1+3]*m[4*3+0]
             - m[4*0+3]*m[4*1+0]*m[4*3+2]
             + m[4*0+0]*m[4*1+3]*m[4*3+2]
             + m[4*0+2]*m[4*1+0]*m[4*3+3]
             - m[4*0+0]*m[4*1+2]*m[4*3+3];
    a[4*1+3] = m[4*0+2]*m[4*1+3]*m[4*2+0]
             - m[4*0+3]*m[4*1+2]*m[4*2+0]
             + m[4*0+3]*m[4*1+0]*m[4*2+2]
             - m[4*0+0]*m[4*1+3]*m[4*2+2]
             - m[4*0+2]*m[4*1+0]*m[4*2+3]
             + m[4*0+0]*m[4*1+2]*m[4*2+3];
    a[4*2+0] = m[4*1+1]*m[4*2+3]*m[4*3+0]
             - m[4*1+3]*m[4*2+1]*m[4*3+0]
             + m[4*1+3]*m[4*2+0]*m[4*3+1]
             - m[4*1+0]*m[4*2+3]*m[4*3+1]
             - m[4*1+1]*m[4*2+0]*m[4*3+3]
             + m[4*1+0]*m[4*2+1]*m[4*3+3];
    a[4*2+1] = m[4*0+3]*m[4*2+1]*m[4*3+0]
             - m[4*0+1]*m[4*2+3]*m[4*3+0]
             - m[4*0+3]*m[4*2+0]*m[4*3+1]
             + m[4*0+0]*m[4*2+3]*m[4*3+1]
             + m[4*0+1]*m[4*2+0]*m[4*3+3]
             - m[4*0+0]*m[4*2+1]*m[4*3+3];
    a[4*2+2] = m[4*0+1]*m[4*1+3]*m[4*3+0]
             - m[4*0+3]*m[4*1+1]*m[4*3+0]
             + m[4*0+3]*m[4*1+0]*m[4*3+1]
             - m[4*0+0]*m[4*1+3]*m[4*3+1]
             - m[4*0+1]*m[4*1+0]*m[4*3+3]
             + m[4*0+0]*m[4*1+1]*m[4*3+3];
    a[4*2+3] = m[4*0+3]*m[4*1+1]*m[4*2+0]
             - m[4*0+1]*m[4*1+3]*m[4*2+0]
             - m[4*0+3]*m[4*1+0]*m[4*2+1]
             + m[4*0+0]*m[4*1+3]*m[4*2+1]
             + m[4*0+1]*m[4*1+0]*m[4*2+3]
             - m[4*0+0]*m[4*1+1]*m[4*2+3];
    a[4*3+0] = m[4*1+2]*m[4*2+1]*m[4*3+0]
             - m[4*1+1]*m[4*2+2]*m[4*3+0]
             - m[4*1+2]*m[4*2+0]*m[4*3+1]
             + m[4*1+0]*m[4*2+2]*m[4*3+1]
             + m[4*1+1]*m[4*2+0]*m[4*3+2]
             - m[4*1+0]*m[4*2+1]*m[4*3+2];
    a[4*3+1] = m[4*0+1]*m[4*2+2]*m[4*3+0]
             - m[4*0+2]*m[4*2+1]*m[4*3+0]
             + m[4*0+2]*m[4*2+0]*m[4*3+1]
             - m[4*0+0]*m[4*2+2]*m[4*3+1]
             - m[4*0+1]*m[4*2+0]*m[4*3+2]
             + m[4*0+0]*m[4*2+1]*m[4*3+2];
    a[4*3+2] = m[4*0+2]*m[4*1+1]*m[4*3+0]
             - m[4*0+1]*m[4*1+2]*m[4*3+0]
             - m[4*0+2]*m[4*1+0]*m[4*3+1]
             + m[4*0+0]*m[4*1+2]*m[4*3+1]
             + m[4*0+1]*m[4*1+0]*m[4*3+2]
             - m[4*0+0]*m[4*1+1]*m[4*3+2];
    a[4*3+3] = m[4*0+1]*m[4*1+2]*m[4*2+0]
             - m[4*0+2]*m[4*1+1]*m[4*2+0]
             + m[4*0+2]*m[4*1+0]*m[4*2+1]
             - m[4*0+0]*m[4*1+2]*m[4*2+1]
             - m[4*0+1]*m[4*1+0]*m[4*2+2]
             + m[4*0+0]*m[4*1+1]*m[4*2+2];

    det = NvGlDemoMatrixDeterminant(m);

    for(i = 0; i < 16; ++i)
        a[i] /= det;

    NvGlDemoMatrixIdentity(e);

    NvGlDemoMatrixCopy(b, m);
    NvGlDemoMatrixMultiply(b, a);

    NvGlDemoMatrixCopy(m, a);
}

// Copy a 4x4 matrix
//   dest <- src
void
NvGlDemoMatrixCopy(
    float dest[16], float src[16])
{
    MEMCPY(dest, src, 16*sizeof(float));
}

// Multiply a 4x4 matrix into a 4-vector
//   v <- m * v
void
NvGlDemoMatrixVectorMultiply(
    float m[16],  float v[4])
{
    float res[4];

    res[0] = m[ 0] * v[0] + m[ 4] * v[1] + m[ 8] * v[2] + m[12] * v[3];
    res[1] = m[ 1] * v[0] + m[ 5] * v[1] + m[ 9] * v[2] + m[13] * v[3];
    res[2] = m[ 2] * v[0] + m[ 6] * v[1] + m[10] * v[2] + m[14] * v[3];
    res[3] = m[ 3] * v[0] + m[ 7] * v[1] + m[11] * v[2] + m[15] * v[3];

    MEMCPY(v, res, sizeof(res));
}

// Print a 4x4 matrix to the log
void
NvGlDemoMatrixPrint(
    float a[16])
{
    int i, j;

    for(i = 0; i < 4; ++i) {
        for(j = 0; j < 4; ++j) {
            NvGlDemoLog("%f%c", a[4*i + j], j == 3 ? '\n' : ' ');
        }
    }
}
