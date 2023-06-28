/*
 * slider.c
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
// Slider controls
//

#include "nvgldemo.h"
#include "vector.h"
#include "slider.h"
#include "shaders.h"

static int rodGeom[6][2] = {
    {4,  7},
    {4,  9},
    {2,  9},
    {4, 11},
    {2, 11},
    {4, 13}
};

static int knobPolys[5][4] = {
    {0, 1, 3, 2},
    {2, 3, 5, 4},
    {4, 5, 7, 6},
    {6, 7, 1, 0},
    {1, 3, 5, 7}
};

static int knobGeom[8][2] = {
    {0,   2},
    {0,   6},
    {-4, 10},
    {-2, 10},
    {0,  18},
    {0,  14},
    {4,  10},
    {2,  10}
};

static void
Slider_build_rod(
    Slider *o)
{
    float scale = (o->top-o->bottom)/20.0f;
    int i;
    for (i=0; i<6; ++i) {
        float x = rodGeom[i][0]*scale;
        float y = rodGeom[i][1]*scale + o->bottom;

        set_2(o->rodVertices[i], x+o->left, y);
        set_2(o->rodVertices[i+6], o->right-x, y);
    }
}

static void
Slider_build_colors(
    Slider *o)
{
    {
        float3 white = {1.0, 1.0, 1.0};
        float3 dark, medium;
        div_f3(dark, o->rodColor, 3.0f);
        copy_3(medium, o->rodColor);

        copy_3(o->rodColors[0], dark);
        copy_3(o->rodColors[1], medium);
        copy_3(o->rodColors[2], dark);
        copy_3(o->rodColors[3], white);
        copy_3(o->rodColors[4], dark);
        copy_3(o->rodColors[5], medium);

        copy_3(o->rodColors[6], dark);
        copy_3(o->rodColors[7], medium);
        copy_3(o->rodColors[8], medium);
        copy_3(o->rodColors[9], white);
        copy_3(o->rodColors[10], medium);
        copy_3(o->rodColors[11], medium);
    }

    {
        float3 white = {1.0, 1.0, 1.0};
        float3 medium_dark, dark, medium, light;
        div_f3(medium_dark, o->knobColor, 2.0f);
        div_f3(dark, o->knobColor, 3.0f);
        copy_3(medium, o->knobColor);
        add_f3(light, white, o->knobColor);
        divi_f3(light, 2.0f);


        copy_3(o->knobColors[0], dark);
        copy_3(o->knobColors[1], light);
        copy_3(o->knobColors[2], white);
        copy_3(o->knobColors[3], medium_dark);
        copy_3(o->knobColors[4], medium);
    }
}


static void
Slider_build_knob(
    Slider *o)
{
    float scale = (o->top-o->bottom)/20.0f;
    float length = (o->right-o->left)/scale;

    float n = (o->value - o->minimum)/(o->maximum-o->minimum);
    int  i;

    o->knobPos = 4 + (length-8) * n;
    for (i=0; i<8; ++i) {
        float x = (knobGeom[i][0]+o->knobPos)*scale+o->left;
        float y = knobGeom[i][1]*scale+o->bottom;
        set_2(o->knobVertices[i], x, y);
    }
}

Slider*
Slider_new(
    float mn,
    float mx,
    float init)
{
    Slider *o = (Slider*)MALLOC(sizeof(Slider));
    o->minimum = mn;
    o->maximum = mx;
    o->value = init;
    o->mouseDown = GL_FALSE;

    set_3(o->rodColor, 0.2f, 0.1f, 0.8f);
    set_3(o->knobColor, 1.0f, 0.2f, 0.4f);
    Slider_build_colors(o);

    return o;
}

void
Slider_delete(
    Slider *o)
{
    FREE(o);
}


void
Slider_setPos(
    Slider *o,
    float  l,
    float  r,
    float  b,
    float  t)
{
    o->left = l;
    o->right = r;
    o->bottom = b;
    o->top = t;

    Slider_build_knob(o);
    Slider_build_rod(o);
}

void
Slider_draw(
    Slider *o)
{
    unsigned short indices[8] = {0,6,1,7,3,9,5,11};
    int            k;

    // Load shader program and enable vertex array
    glUseProgram(prog_overlaycol);
    glEnableVertexAttribArray(aloc_overlaycolVertex);

    // Draw slider
    glEnableVertexAttribArray(aloc_overlaycolColor);
    glVertexAttribPointer(aloc_overlaycolVertex,
                          2, GL_FLOAT, GL_FALSE, 0, o->rodVertices);
    glVertexAttribPointer(aloc_overlaycolColor,
                          3, GL_FLOAT, GL_FALSE, 0, o->rodColors);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 6);
    glDrawArrays(GL_TRIANGLE_STRIP, 6, 6);
    glDrawElements(GL_TRIANGLE_STRIP, 8, GL_UNSIGNED_SHORT, indices);
    glDisableVertexAttribArray(aloc_overlaycolColor);

    // Draw knob
    glVertexAttribPointer(aloc_overlaycolVertex,
                          2, GL_FLOAT, GL_FALSE, 0, o->knobVertices);
    for (k=0; k<5; ++k) {
        glVertexAttrib3fv(aloc_overlaycolColor, o->knobColors[k]);
        glDrawElements(GL_TRIANGLE_FAN, 4, GL_UNSIGNED_INT, knobPolys[k]);
    }

    // Disable vertex array
    glDisableVertexAttribArray(aloc_overlaycolVertex);
}

GLboolean
Slider_setValue(
    Slider *o,
    float  val)
{
    float oldval = o->value;
    o->value = val;
    if (o->value>o->maximum) o->value = o->maximum;
    if (o->value<o->minimum) o->value = o->minimum;
    if (oldval == o->value)
        return GL_FALSE;
    Slider_build_knob(o);
    return GL_TRUE;
}

float
Slider_getValue(
    Slider *o)
{
    return o->value;
}

void
Slider_select(
    Slider    *o,
    GLboolean sel)
{
    if (sel) {
        set_3(o->rodColor, 0.0f, 0.9f, 0.2f);
        set_3(o->knobColor, 1.0f, 0.2f, 0.4f);
    } else {
        set_3(o->rodColor, 0.2f, 0.1f, 0.8f);
        set_3(o->knobColor, 1.0f, 0.2f, 0.4f);
    }
    Slider_build_colors(o);
}
