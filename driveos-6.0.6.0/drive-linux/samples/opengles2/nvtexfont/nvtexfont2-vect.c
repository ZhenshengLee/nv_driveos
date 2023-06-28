/*
 * nvtexfont2-vect.c
 *
 * Copyright (c) 2003 - 2012 NVIDIA Corporation.  All rights reserved.
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

#include "GLES2/gl2.h"

#include "nvgldemo.h"
#include "nvtexfont.h"
#include "nvtexfont-priv.h"

#include "inconsolata.h"

int
nvtexfontInVectorFont(
    NVTexfontVectorFont *vtf,
    int                 c)
{
    return (vtf->offsets[c] > -1);
}

NVTexfontVectorFont*
nvtexfontInitVectorFont(
    NVTexfontVectorFontName font,
    GLboolean               antialias,
    GLboolean               use_vbo)
{
    GLint savedArrayBufferBinding;
    NVTexfontVectorFont *vtf = MALLOC(sizeof(NVTexfontVectorFont));

    if (vectorProg == 0)
        initShaders();

    switch (font)
    {
    case NV_VECTOR_TEXFONT_INCONSOLATA:
        vtf->vertices = NVTEXFONT_inconsolata_VERTS;
        vtf->offsets = NVTEXFONT_inconsolata_OFFSETS;
        vtf->counts = NVTEXFONT_inconsolata_COUNTS;
        vtf->widths = NVTEXFONT_inconsolata_WIDTHS;
        vtf->vertNo = NVTEXFONT_inconsolata_VERTNO;

        break;
    default:
        goto initVectorError;
    }
    vtf->antialias = antialias;

    if (use_vbo == GL_TRUE)
    {

        glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &savedArrayBufferBinding);

        glGenBuffers(1, &vtf->vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vtf->vbo);
        glBufferData(GL_ARRAY_BUFFER, vtf->vertNo * 2,
                     vtf->vertices, GL_STATIC_DRAW);

        glBindBuffer(GL_ARRAY_BUFFER, savedArrayBufferBinding);
    }
    else
    {
        vtf->vbo = 0;
    }

    return vtf;
initVectorError:
    FREE(vtf);
    return 0;
}

void
nvtexfontUnloadVectorFont(
    NVTexfontVectorFont *vtf)
{
    deinitShaders();

    if (vtf->vbo)
    {
        glDeleteBuffers(1, &vtf->vbo);
        vtf->vbo = 0;
    }

    FREE(vtf);
}

static int
renderVecGlyph(
    NVTexfontVectorFont *vtf,
    int                 c)
{
    if (vtf->offsets[c] == -1)
        return vtf->widths[c];

    if (vtf->antialias)
    {
        /* CURRENTLY UNIMPLEMENTED */
    }

    glDrawArrays(GL_TRIANGLES, vtf->offsets[c], vtf->counts[c]);

    return vtf->widths[c];
}

void
nvtexfontRenderVecString_All(
    NVTexfontVectorFont *vtf,
    char                *string,
    float               x,
    float               y,
    float               scaleX,
    float               scaleY,
    float               r,
    float               g,
    float               b)
{
    GLboolean savedBlendEnabled, savedDepthTestEnabled, savedCullFaceEnabled;
    GLint savedBlendSrcRGB, savedBlendSrcAlpha;
    GLint savedBlendDstRGB, savedBlendDstAlpha;
    GLint savedVertArrayBinding, savedVertArrayEnabled, savedVertArraySize;
    GLint savedVertArrayStride, savedVertArrayType, savedVertArrayNormalized;
    void *savedVertArrayPointer;
    GLint savedCurrentProgram;
    GLint savedArrayBufferBinding;

    GLuint uniScale, uniColor, uniScreenPos, uniOffset;

    int offset = 0, i = 0;

    /*****************
     * SAVE GL STATE *
     *****************/

    savedBlendEnabled = glIsEnabled(GL_BLEND);
    savedDepthTestEnabled = glIsEnabled(GL_DEPTH_TEST);
    savedCullFaceEnabled = glIsEnabled(GL_CULL_FACE);

    glGetIntegerv(GL_BLEND_SRC_RGB, &savedBlendSrcRGB);
    glGetIntegerv(GL_BLEND_SRC_ALPHA, &savedBlendSrcAlpha);
    glGetIntegerv(GL_BLEND_DST_RGB, &savedBlendDstRGB);
    glGetIntegerv(GL_BLEND_DST_ALPHA, &savedBlendDstAlpha);

    glGetVertexAttribiv(VERT_ARRAY, GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING,
                        &savedVertArrayBinding);
    glGetVertexAttribiv(VERT_ARRAY, GL_VERTEX_ATTRIB_ARRAY_ENABLED,
                        &savedVertArrayEnabled);
    glGetVertexAttribiv(VERT_ARRAY, GL_VERTEX_ATTRIB_ARRAY_SIZE,
                        &savedVertArraySize);
    glGetVertexAttribiv(VERT_ARRAY, GL_VERTEX_ATTRIB_ARRAY_STRIDE,
                        &savedVertArrayStride);
    glGetVertexAttribiv(VERT_ARRAY, GL_VERTEX_ATTRIB_ARRAY_TYPE,
                        &savedVertArrayType);
    glGetVertexAttribiv(VERT_ARRAY, GL_VERTEX_ATTRIB_ARRAY_NORMALIZED,
                        &savedVertArrayNormalized);
    glGetVertexAttribPointerv(VERT_ARRAY, GL_VERTEX_ATTRIB_ARRAY_POINTER,
                              &savedVertArrayPointer);

    glGetIntegerv(GL_CURRENT_PROGRAM, &savedCurrentProgram);

    /************
     * SETUP GL *
     ************/

    glEnable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
                        GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(vectorProg);

    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &savedArrayBufferBinding);

    glEnableVertexAttribArray(VERT_ARRAY);

    if (vtf->vbo)
    {
        glBindBuffer(GL_ARRAY_BUFFER, vtf->vbo);
        glVertexAttribPointer(VERT_ARRAY, 2, GL_UNSIGNED_BYTE, GL_FALSE, 0, 0);
    }
    else
    {
        glVertexAttribPointer(VERT_ARRAY, 2, GL_UNSIGNED_BYTE, GL_FALSE,
                              0, vtf->vertices);
    }

    /*****************
     * SETUP SHADERS *
     *****************/

    uniScale = glGetUniformLocation(vectorProg, "scale");
    uniColor = glGetUniformLocation(vectorProg, "color");
    uniScreenPos = glGetUniformLocation(vectorProg, "screenPos");

    // Adjust scale so character size is approximately 1/10th screen height
    glUniform2f(uniScale, 0.002f * scaleX, 0.002f * scaleY);
    glUniform3f(uniColor, r, g, b);
    glUniform2f(uniScreenPos, x, y);

    uniOffset = glGetUniformLocation(vectorProg, "offset");

    /*************
     * DRAW TEXT *
     *************/

    while (string[i]) {
        glUniform1i(uniOffset, offset);
        offset += renderVecGlyph(vtf, string[i]);
        i++;
    }

    /********************
     * RESTORE GL STATE *
     ********************/

    glBindBuffer(GL_ARRAY_BUFFER, savedArrayBufferBinding);
    glUseProgram(savedCurrentProgram);


    if (!savedVertArrayEnabled)
        glDisableVertexAttribArray(VERT_ARRAY);

    glVertexAttribPointer(VERT_ARRAY, savedVertArraySize,
                          savedVertArrayType, savedVertArrayNormalized,
                          savedVertArrayStride, savedVertArrayPointer);

    if (!savedBlendEnabled)
        glDisable(GL_BLEND);
    glBlendFuncSeparate(savedBlendSrcRGB, savedBlendDstRGB,
                        savedBlendSrcAlpha, savedBlendDstAlpha);

    if (savedDepthTestEnabled)
        glEnable(GL_DEPTH_TEST);

    if (savedCullFaceEnabled)
        glEnable(GL_CULL_FACE);
}

void
nvtexfontRenderVecString_Pos(
    NVTexfontVectorFont *vtf,
    NVTexfontContext    tfc,
    char                *string,
    float               x,
    float               y)
{
    nvtexfontRenderVecString_All(vtf, string, x, y,
                                 ((nvtexfontState *)tfc)->scaleX,
                                 ((nvtexfontState *)tfc)->scaleY,
                                 ((nvtexfontState *)tfc)->r,
                                 ((nvtexfontState *)tfc)->g,
                                 ((nvtexfontState *)tfc)->b);
}

void
nvtexfontRenderVecString(
    NVTexfontVectorFont *vtf,
    NVTexfontContext    tfc,
    char                *string)
{
    nvtexfontRenderVecString_All(vtf, string,
                                 ((nvtexfontState *)tfc)->x,
                                 ((nvtexfontState *)tfc)->y,
                                 ((nvtexfontState *)tfc)->scaleX,
                                 ((nvtexfontState *)tfc)->scaleY,
                                 ((nvtexfontState *)tfc)->r,
                                 ((nvtexfontState *)tfc)->g,
                                 ((nvtexfontState *)tfc)->b);
}
