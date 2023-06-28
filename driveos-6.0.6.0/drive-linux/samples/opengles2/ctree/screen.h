/*
 * screen.h
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
// Top level scene control/rendering
//

#ifndef __SCREEN_H
#define __SCREEN_H

#include <GLES2/gl2.h>

int  Screen_initialize(float d, GLsizei w, GLsizei h, GLboolean fpsFlag);
void Screen_resize(GLsizei w, GLsizei h);
void Screen_deinitialize(void);

void Screen_setDemoParams(void);
void Screen_setSmallTex(void);
void Screen_setNoSky(void);
void Screen_setNoMenu(void);

void Screen_draw(void);
void Screen_callback(int key, int x, int y);

GLboolean Screen_isFinished(void);

#endif // __SCREEN_H
