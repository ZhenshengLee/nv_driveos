/*
 * algebra.h
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

#ifndef __ALGEBRA_H
#define __ALGEBRA_H

/* algebraic functions */

/* a quaternion */
typedef struct {
    float r, i, j, k;
} Quat;
/* a vector */
typedef float float3[3];
/* a matrix */
typedef float float4x4[4][4];


/* apply matrix to vector, result stored on this vector */
void vec_transform(float3 v, float4x4 m);
/* apply matrix to point, result stored on this point */
void pnt_transform(float3 v, float4x4 m);
/* inverses the matrix */
void mat_invert(float4x4 m);
/* inverses the left-upper 3x3 part of matrix */
void mat_invert_part(float4x4 m);
/* transposes the matrix */
void mat_transpose(float4x4 m);
/* an identity */
void quat_identity(Quat *a);
/* sets a quaternion */
void quat_setf3(Quat *a, float nr, float ni, float nj, float nk);
/* sets a quaternion */
void quat_setfv(Quat *a, float radians, const float3 axis);
/* multiplies quaternions, result on the first argument */
void quat_multiply(Quat *a, const Quat *b);
/* makes the matrix (used in transformations) from quaternion */
void quat_mat(float4x4 m, Quat *a);
/* a=b; */
void quat_prescribe(Quat *a, Quat *b);
/* set matrix to identity */
void mat_identity(float4x4 m);
/* muliplies matrices, the result on the first argument */
void mat_multiply(float4x4 m0, float4x4 m1);
/* simmulates glTranslatef */
void mat_translate(float4x4 m, float x, float y, float z);
/* simmulates glOrthof */
void mat_ortho(float4x4 m,
    float l, float r, float b, float t, float n, float f);
/* simmulates glFrustumf */
void mat_frustum(float4x4 m,
    float l, float r, float b, float t, float n, float f);
/* simmulates glScalef */
void mat_scale(float4x4 m, float x, float y, float z);
/* multiply vector by scalar */
void vec_scale(float3 v, const float s);
/* add vectors, the result on the first argument (a=a+b) */
void vec_add(float3 a, const float3 b);
/* substract vectors, the result on the first argument (a=a-b) */
void vec_subs(float3 a, const float3 b);
/* dot product of vectors */
float vec_dot(const float3 a, const float3 b);
/* a=b; */
void vec_prescribe(float3 a, const float3 b);

#endif //__ALGEBRA_H
