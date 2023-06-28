/*
 * firefly.h
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
// Firefly representation/animation
//

#ifndef __FIREFLY_H
#define __FIREFLY_H

#include "vector.h"

// Global arrays holding firefly data
extern float *fPos;
extern float *fWings;
extern float *fVel;
extern float *fColor;
extern float *fHsva;

// Firefly structure
//   Has pointers to above arrays which keeps transformation code simpler
//   (avoids allocating temp arrays)
typedef struct {
    float *pos;
    float *wings;
    float *vel;
    float *c;
    float range;
    float *hsva;
} Firefly;

// Initialization and clean-up
void Firefly_global_init(int num);
void Firefly_init(Firefly *o, int num);
void Firefly_destroy(Firefly *o);
void Firefly_global_destroy(void);

// Animation and rendering
void Firefly_move(Firefly *o);
void Firefly_draw(Firefly *o);

#endif // __FIREFLY_H
