/*
 * slider.h
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
// Control sliders
//

#ifndef __SLIDER_H
#define __SLIDER_H

#include <GLES2/gl2.h>
#include "vector.h"

// Slider description
typedef struct {
    GLboolean visible;

    GLboolean focus;
    float     left, right, bottom, top;
    float     minimum, maximum, value;
    float     knobPos;
    GLboolean mouseDown;

    float3    rodColor;
    float2    rodVertices[12];
    float3    rodColors[12];

    float3    knobColor;

    float3    knobColors[5];
    float2    knobVertices[8];
} Slider;

// Initialization and clean-up
Slider *Slider_new(float minimum, float maximum, float init);
void Slider_delete(Slider *o);
void Slider_setCB (Slider *o, void (*CB)(float));
void Slider_setPos(Slider *o,
                   float left, float right, float bottom, float top);

// Control
GLboolean Slider_setValue(Slider *o, float val);
float Slider_getValue(Slider *o);
void Slider_select(Slider *o, GLboolean sel);

// Rendering
void Slider_draw(Slider *o);

#endif // __SLIDER_H
