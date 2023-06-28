/*
 * vtxraster.glslf
 *
 * Copyright (c) 2007 - 2012 NVIDIA Corporation.  All rights reserved.
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
 *
 * This software is based upon texfont, with consent from Mark J. Kilgard,
 * provided under the following terms:
 *
 * Copyright (c) Mark J. Kilgard, 1997.
 *
 * This program is freely distributable without licensing fees  and is
 * provided without guarantee or warrantee expressed or  implied. This
 * program is -not- in the public domain.
 */

//
// Vertex shader for rasterized fonts
//

attribute vec2 vertArray;
attribute vec2 texArray;

uniform float scaleX, scaleY;
uniform float screenX, screenY;
uniform int offset;

varying vec2 texCoord;

void main()  {
    //gl_Position = vec4( scale * (vertArray + vec2(offset, 0.0)) + vec2(screenX, screenY), 1.0, 1.0);
    gl_Position = vec4( screenX + scaleX * (vertArray.x + float(offset)),
                        screenY + scaleY * (vertArray.y),
                        1.0, 1.0);
    texCoord = texArray;
    //gl_Position = vec4(attrObjCoord, 0.0, 1.0);
}
