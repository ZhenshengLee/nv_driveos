/*
 * firefly.c
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
// Firefly animation
//

#include <GLES2/gl2.h>

#include "nvgldemo.h"
#include "random.h"
#include "shaders.h"
#include "firefly.h"

// Number of random "wing" vertices comprising a firefly
#define NUM_WINGS (10)

// arrays holding firefly data
float *fPos;
float *fWings;
float *fVel;
float *fColor;
float *fHsva;

// Convert HSVA color representation to RGBA
static void
hsva2rgba(
    float3 rgba,
    float4 hsva)
{
    float hue = hsva[0] - (float)((int)hsva[0]);
    float sat = clamp(hsva[1], 0.0f, 1.0f);
    float val = clamp(hsva[2], 0.0f, 1.0f);

    float hh = hue * 6.0f;
    float p,q,t;

    int i = (int)hh;

    hh -= i;

    p = val * (1 - sat);
    q = val * (1 - sat * hh);
    t = val * (1 - (sat * (1 - hh)));

    switch (i) {
    case 0: set_3(rgba, val, t, p); break;
    case 1: set_3(rgba, q, val, p); break;
    case 2: set_3(rgba, p, val, t); break;
    case 3: set_3(rgba, p, q, val); break;
    case 4: set_3(rgba, t, p, val); break;
    case 5: set_3(rgba, val, p, q); break;
    }
}

// Initialize firefly data structures
void
Firefly_global_init(
    int count)
{
    fPos   = (float*)MALLOC(sizeof(float) * count * 3);
    fWings = (float*)MALLOC(sizeof(float) * count * 3 * NUM_WINGS);
    fVel   = (float*)MALLOC(sizeof(float) * count * 3);
    fColor = (float*)MALLOC(sizeof(float) * count * 3);
    fHsva  = (float*)MALLOC(sizeof(float) * count * 4);
}

// Initialize single firefly
void
Firefly_init(
    Firefly *o,
    int     num)
{
    int i;

    o->pos = fPos + (num * 3);
    o->wings = fWings + (num * 3 * NUM_WINGS);
    o->vel = fVel + (num * 3);
    o->c = fColor + (num * 3);
    o->hsva = fHsva + (num * 4);

    set_3(o->pos, 0, 0, 3.0);

    for (i=0; i<NUM_WINGS; ++i) {
        set_3(o->wings + 3*i, 0, 0, 0);
    }

    set_3(o->vel, 0, 0, 0);
    set_3(o->c, 1, 1, 1);
    o->range = 8;
    set_4(o->hsva, (float)GetRandom(), 0.4f, 1.0f, 1.0f);
}

// Destroy single firefly
void
Firefly_destroy(
    Firefly *o)
{
    // No action required
}

// Destroy firefly data structures
void
Firefly_global_destroy(void)
{
    FREE(fPos);
    FREE(fWings);
    FREE(fVel);
    FREE(fColor);
    FREE(fHsva);
}

// Randomly move a firefly
void
Firefly_move(
    Firefly *o)
{
    int i;

    float r1 = ((float) GetRandom()) * 2.0f - 1.0f;
    float r2 = ((float) GetRandom()) * 2.0f - 1.0f;
    float r3 = ((float) GetRandom()) * 2.0f - 1.0f;

    float hs = 0.005f;
    float rs = 0.08f;
    float3 r = {rs * r1, rs * r2, rs * r3};

    float3 tmp;
    float3 home;

    o->hsva[0] += 0.01f;
    if (o->hsva[0] > 1.0) o->hsva[0] -= 1.0;
    hsva2rgba(o->c, o->hsva);

    home[0] = hs * -o->pos[0];
    home[1] = hs * -o->pos[1];
    home[2] = hs * (-o->pos[2]+o->range/2.0f);

    add_f3(tmp, o->vel, home);
    addi_f3(tmp, r);
    mult_f3f(o->vel, tmp, 0.97f);
    mult_f3f(tmp, o->vel, 0.5f);
    addi_f3(o->pos, tmp);

    if (o->pos[2] < 0.5) { o->pos[2] = 0.5; }

    for (i=0; i<NUM_WINGS; ++i) {
        r1 = (((float) GetRandom())-0.5f) * 0.2f;
        r2 = (((float) GetRandom())-0.5f) * 0.2f;
        r3 = (((float) GetRandom())-0.5f) * 0.2f;
        set_3(o->wings + 3*i, o->pos[0]+r1, o->pos[1]+r2, o->pos[2]+r3);
    }
}

// Draw a firefly
void
Firefly_draw(
    Firefly *o)
{
    int i;
    float vertices[(NUM_WINGS+1)*3];
    float colors  [(NUM_WINGS+1)*4];
    float *v = vertices, *c = colors;

    for (i=0; i<NUM_WINGS+1; i++) {
        *c++ = o->c[0];
        *c++ = o->c[1];
        *c++ = o->c[2];
        *c++ = 0.0;
    }
    colors[3] = 1.0;

    *v++ = o->pos[0];
    *v++ = o->pos[1];
    *v++ = o->pos[2];

    for (i=0; i<NUM_WINGS; ++i) {
        *v++ = o->wings[3*i+0];
        *v++ = o->wings[3*i+1];
        *v++ = o->wings[3*i+2];
    }

    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
                        GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);

    glEnableVertexAttribArray(aloc_simplecolVertex);
    glEnableVertexAttribArray(aloc_simplecolColor);
    glVertexAttribPointer(aloc_simplecolVertex,
                          3, GL_FLOAT, GL_FALSE, 0, vertices);
    glVertexAttribPointer(aloc_simplecolColor,
                          4, GL_FLOAT, GL_FALSE, 0, colors);
    glDrawArrays(GL_TRIANGLE_FAN, 0, NUM_WINGS+1);
    glDisableVertexAttribArray(aloc_simplecolVertex);
    glDisableVertexAttribArray(aloc_simplecolColor);

    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
}
