/*
 * vector.h
 *
 * Copyright (c) 2003-2015, NVIDIA CORPORATION. All rights reserved.
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

// TODO: Portions of this are redundant with nvgldemo and other demo
// TODO:   applications. It should be rolled into nvgldemo.

#ifndef __VECTOR_H
#define __VECTOR_H


typedef float float2[2];
typedef float float3[3];
typedef float float4[4];
typedef float float4x4[4][4];

typedef double double2[2];
typedef double double3[3];
typedef double double4[4];
typedef double double4x4[4][4];


#define copy_2(target, f) \
  {(target)[0] = (f)[0]; (target)[1] = (f)[1];}

#define copy_3(target, f) \
  {(target)[0] = (f)[0]; (target)[1] = (f)[1]; (target)[2] = (f)[2];}

#define copy_4(target, f) \
  {(target)[0] = (f)[0]; (target)[1] = (f)[1]; (target)[2] = (f)[2]; (target)[3] = (f)[3];}

//void copy_4x4(f4x4 dest, f4x4 src);

#define set_2(target, f0, f1) \
  {(target)[0] = f0; (target)[1] = f1;}

#define set_3(target, f0, f1, f2) \
  {(target)[0] = f0; (target)[1] = f1; (target)[2] = f2;}

#define set_4(target, f0, f1, f2, f3) \
  {(target)[0] = f0; (target)[1] = f1; (target)[2] = f2; (target)[3] = f3;}

void add_f3(float3 dest, float3 src, float3 vec);
void addi_f3(float3 dest, float3 vec);

void subtr_f2(float2 dest, float2 src, float2 vec);

void mult_f3f(float3 dest, float3 src, float s);
void multi_f3f(float3 dest, float s);
void mult_f4x4(float4x4 dest, float4x4 src, float4x4 mat);
void multi_f4x4(float4x4 dest, float4x4 mat);

void div_f3(float3 dest, float3 src, float s);
void divi_f3(float3 dest, float s);

#define dot_2(v0, v1) \
   (v0[0] * v1[0] + v0[1] * v1[1])

#define dot_3(v0, v1) \
   (v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2])

#define dot_4(v0, v1) \
   (v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2] + v0[3] * v1[3])

#define cross_3(dest, v0, v1) {\
   dest[0] = v0[1] * v1[2] - v0[2] * v1[1];\
   dest[1] = v0[2] * v1[0] - v0[0] * v1[2];\
   dest[2] = v0[0] * v1[1] - v0[1] * v1[0];}

void transform_f3(float3 dest, float4x4 mat, float3 vec);
void transform_f4(float4 dest, float4x4 mat, float4 vec);
void transformi_f3(float3 dest, float4x4 mat);
void transformi_f4(float4 dest, float4x4 mat);
void transformVec_f3(float3 dest, float4x4 mat, float3 vec);
void transformVeci_f3(float3 dest, float4x4 mat);


extern double4x4 ident_matrix_d;
extern float4x4 ident_matrix_f;

void makeTranslate(float4x4 dest, float v0, float v1, float v2);
void makeScale(float4x4 dest, float v0, float v1, float v2);
void makeRotation(float4x4 dest, float v0, float v1, float v2, float d);

void normalize_f3(float3 dest, float3 src);
void normalizei_f3(float3 dest);

#define PI_F 3.14159274F
// utility functions / define's.
#define degToRadD(d) ((d) * PI_F / 180.0)
#define radToDegD(r) ((r) * 180.0 / PI_F)
#define degToRadF(d) ((d) * (float)PI_F / 180.0f)
#define radToDegF(r) ((r) * 180.0f / (float)PI_F)

float clamp(float v, float minval, float maxval);

#ifndef min // QNX defines these
#define min(v0, v1) ((v0) < (v1) ? (v0) : (v1))
#endif

#ifndef max
#define max(v0, v1) ((v0) > (v1) ? (v0) : (v1))
#endif

#endif // __VECTOR_H
