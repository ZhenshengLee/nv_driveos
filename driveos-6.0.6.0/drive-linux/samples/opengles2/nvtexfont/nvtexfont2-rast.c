/*
 * nvtexfont2-rast.c
 *
 * Copyright (c) 2003 - 2019 NVIDIA Corporation.  All rights reserved.
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
#include "GLES2/gl2ext.h"

#include "nvgldemo.h"
#include "nvtexfont.h"
#include "nvtexfont-priv.h"

#include "helvetica.h"

/* ctype.h introduces non-portability with C libraries.                 */
/* Specifically, the symbol __ctype_b could be undefined.               */
/* We use ctype.h for the upper and lower case functions.               */
/* Avoid the problem altogether with some simple macros.                */
/* Note toupper() and tolower() assume the input is lower and upper     */
/* respectively, which it is as used below in getGlyphIndex().          */
#define myislower(c) ((c) >= 'a' && (c) <= 'z')
#define myisupper(c) ((c) >= 'A' && (c) <= 'Z')
#define mytoupper(c) ('A' + ((c)-'a'))
#define mytolower(c) ('a' + ((c)-'A'))

static int
getGlyphIndex(
    NVTexfontRasterFont *txf,
    int                 c)
{
    int indx;

    /* Automatically substitute uppercase letters with lowercase if not
       uppercase available (and vice versa). */
    if ((c >= txf->min_glyph) && (c < txf->min_glyph + txf->range)) {
        indx = txf->lut[c - txf->min_glyph];
        if (indx != -1) {
            return indx;
        }
        if (myislower(c)) {
            c = mytoupper(c);
            if ((c >= txf->min_glyph) && (c < txf->min_glyph + txf->range)) {
                return txf->lut[c - txf->min_glyph];
            }
        }
        if (myisupper(c)) {
            c = mytolower(c);
            if ((c >= txf->min_glyph) && (c < txf->min_glyph + txf->range)) {
                return txf->lut[c - txf->min_glyph];
            }
        }
    }
    return -1;
}


static unsigned char*
getFontData(
    NVTexfontRasterFontName font)
{
    switch (font) {
    case NV_TEXFONT_HELVETICA:
        return helvetica_font_data;
    default:
        return 0;
    }
}

/*
 * TexFont originally would fread() from a binary TexFont file.
 * NVTexfontRasterFont has the TexFont data built into the library in the
 * form of a C header file.
 * So each "fread(..., file)" line in the original code has been replaced
 * by a "BUFREAD(..., buffer)" below, where BUFREAD is the following macro.
 */
#define BUFREAD(TO,SIZE,NUM,BUFFER)               \
        (MEMCPY(TO, BUFFER, SIZE*NUM), BUFFER += (SIZE*NUM), (NUM))

static NVTexfontRasterFont*
loadFontData(
    NVTexfontRasterFontName font)
{
    NVTexfontRasterFont *txf;
    GLfloat w, h, xstep, ystep;
    char fileid[4], tmp;
    unsigned char *texbitmap = NULL;
    int min_glyph, max_glyph;
    int endianness, swap, format, stride, width, height;
    int i, j, got;
    unsigned char *buffer = NULL;
    GLfloat *st;
    GLshort *vert;
    NVTexfontGlyphInfo *tgi;

    txf = NULL;

    buffer = getFontData(font);
    if (!buffer) {
        lastError = "Unsupported font";
        goto error;
    }

    txf = (NVTexfontRasterFont *) MALLOC(sizeof(NVTexfontRasterFont));
    if (txf == NULL) {
        lastError = "out of memory.";
        goto error;
    }
    /* For easy cleanup in error case. */
    MEMSET(txf, 0, sizeof(NVTexfontRasterFont));
    txf->tgi = NULL;
    txf->lut = NULL;
    txf->teximage = NULL;

    got = BUFREAD(fileid, 1, 4, buffer);
    if (got != 4 || STRNCMP(fileid, "\377txf", 4)) {
        lastError = "not a texture font file.";
        goto error;
    }
    ASSERT(sizeof(int) == 4);  /* Ensure external file format size. */
    got = BUFREAD(&endianness, sizeof(int), 1, buffer);
    if (got == 1 && endianness == 0x12345678) {
        swap = 0;
    } else if (got == 1 && endianness == 0x78563412) {
        swap = 1;
    } else {
        lastError = "not a texture font file.";
        goto error;
    }

#   define EXPECT(n) \
    if (got != n) { lastError = "premature end of file."; goto error; }

    got = BUFREAD(&format, sizeof(int), 1, buffer);
    EXPECT(1);
    got = BUFREAD(&txf->tex_width, sizeof(int), 1, buffer);
    EXPECT(1);
    got = BUFREAD(&txf->tex_height, sizeof(int), 1, buffer);
    EXPECT(1);
    got = BUFREAD(&txf->max_ascent, sizeof(int), 1, buffer);
    EXPECT(1);
    got = BUFREAD(&txf->max_descent, sizeof(int), 1, buffer);
    EXPECT(1);
    got = BUFREAD(&txf->num_glyphs, sizeof(int), 1, buffer);
    EXPECT(1);

    if (swap) {
        SWAPL(&format, tmp);
        SWAPL(&txf->tex_width, tmp);
        SWAPL(&txf->tex_height, tmp);
        SWAPL(&txf->max_ascent, tmp);
        SWAPL(&txf->max_descent, tmp);
        SWAPL(&txf->num_glyphs, tmp);
    }
    txf->tgi = (NVTexfontGlyphInfo *)
        MALLOC(txf->num_glyphs * sizeof(NVTexfontGlyphInfo));
    if (txf->tgi == NULL) {
        lastError = "out of memory.";
        goto error;
    }

    /* Ensure external file format size. */
    ASSERT(sizeof(NVTexfontGlyphInfo) == 12);

    got = BUFREAD(txf->tgi, sizeof(NVTexfontGlyphInfo),
                  txf->num_glyphs, buffer);
    EXPECT(txf->num_glyphs);

    if (swap) {
        for (i = 0; i < txf->num_glyphs; i++) {
            SWAPS(&txf->tgi[i].c, tmp);
            SWAPS(&txf->tgi[i].x, tmp);
            SWAPS(&txf->tgi[i].y, tmp);
        }
    }

    w = (GLfloat) txf->tex_width;
    h = (GLfloat) txf->tex_height;
    xstep = 0.5f / w;
    ystep = 0.5f / h;

    txf->st   = (GLfloat *) MALLOC(txf->num_glyphs * sizeof(GLfloat) * 8);
    txf->vert = (GLshort *) MALLOC(txf->num_glyphs * sizeof(GLshort) * 8);
    if(!txf->st || !txf->vert) {
        lastError = "out of memory.";
        goto error;
    }

    st   = txf->st;
    vert = txf->vert;
    tgi  = txf->tgi;
    for (i = 0; i < txf->num_glyphs; i++,tgi++,st+=8,vert+=8) {
        st  [0] = tgi->x / w - xstep;
        st  [1] = tgi->y / h - ystep;
        vert[0] = tgi->xoffset;
        vert[1] = tgi->yoffset;
        st  [2] = (tgi->x + tgi->width) / w + xstep;
        st  [3] = tgi->y / h - ystep;
        vert[2] = tgi->xoffset + tgi->width;
        vert[3] = tgi->yoffset;
        st  [4] = (tgi->x + tgi->width) / w + xstep;
        st  [5] = (tgi->y + tgi->height) / h + ystep;
        vert[4] = tgi->xoffset + tgi->width;
        vert[5] = tgi->yoffset + tgi->height;
        st  [6] = tgi->x / w - xstep;
        st  [7] = (tgi->y + tgi->height) / h + ystep;
        vert[6] = tgi->xoffset;
        vert[7] = tgi->yoffset + tgi->height;
    }

    min_glyph = txf->tgi[0].c;
    max_glyph = txf->tgi[0].c;
    for (i = 1; i < txf->num_glyphs; i++) {
        if (txf->tgi[i].c < min_glyph) {
            min_glyph = txf->tgi[i].c;
        }
        if (txf->tgi[i].c > max_glyph) {
            max_glyph = txf->tgi[i].c;
        }
    }
    txf->min_glyph = min_glyph;
    txf->range = max_glyph - min_glyph + 1;

    txf->lut = (int *) MALLOC(txf->range * sizeof(int));
    if (txf->lut == NULL) {
        lastError = "out of memory.";
        goto error;
    }
    for (i = 0; i < txf->range; i++) {
        txf->lut[i] = -1;
    }
    for (i = 0; i < txf->num_glyphs; i++) {
        txf->lut[txf->tgi[i].c - txf->min_glyph] = i;
    }

    switch (format) {
    case TXF_FORMAT_BYTE:
        txf->teximage = (unsigned char *)
            MALLOC(txf->tex_width * txf->tex_height);
        if (txf->teximage == NULL) {
            lastError = "out of memory.";
            goto error;
        }
        got = BUFREAD(txf->teximage, 1, txf->tex_width * txf->tex_height,
                      buffer);
        EXPECT(txf->tex_width * txf->tex_height);
        break;
    case TXF_FORMAT_BITMAP:
        width = txf->tex_width;
        height = txf->tex_height;
        stride = (width + 7) >> 3;
        texbitmap = (unsigned char *) MALLOC(stride * height);
        if (texbitmap == NULL) {
            lastError = "out of memory.";
            goto error;
        }
        got = BUFREAD(texbitmap, 1, stride * height, buffer);
        EXPECT(stride * height);
        txf->teximage = (unsigned char *) MALLOC(width * height);
        if (txf->teximage == NULL) {
            lastError = "out of memory.";
            goto error;
        }
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                if (texbitmap[i * stride + (j >> 3)] & (1 << (j & 7))) {
                    txf->teximage[i * width + j] = 255;
                } else {
                    txf->teximage[i * width + j] = 0;
                }
            }
        }
        FREE(texbitmap);
        break;
    }

    return txf;

error:

    if (texbitmap) {
        FREE(texbitmap);
    }

    if (txf) {
        if (txf->tgi)
            FREE(txf->tgi);
        if (txf->lut)
            FREE(txf->lut);
        if (txf->teximage)
            FREE(txf->teximage);
        if (txf->st)
            FREE(txf->st);
        if (txf->vert)
            FREE(txf->vert);
        FREE(txf);
    }

    return NULL;
}

static int
renderGlyph(
    NVTexfontRasterFont *txf,
    int                 c)
{
    int indx = getGlyphIndex(txf, c);
    if (indx == -1)
        return 0;

    glDrawArrays(GL_TRIANGLE_FAN, 4*indx, 4);
    return txf->tgi[indx].advance;
}

static GLuint
establishTexture(
    NVTexfontRasterFont *txf,
    GLuint              texobj,
    GLboolean           setupMipmaps,
    GLenum              minFilter,
    GLenum              magFilter)
{
    GLint savedTextureBinding;
    GLsizei levels;
    const char* glExtensions;

    if (txf->texobj == 0) {
        if (texobj == 0) {
            glGenTextures(1, &txf->texobj);
        } else {
            txf->texobj = texobj;
        }
    }

    glGetIntegerv(GL_TEXTURE_BINDING_2D, &savedTextureBinding);
    glBindTexture(GL_TEXTURE_2D, txf->texobj);

    if (demoOptions.isProtected) {
        glExtensions = (const char *) glGetString(GL_EXTENSIONS);
        if (!STRSTR(glExtensions, "GL_EXT_protected_textures")) {
            NvGlDemoLog("Protected Textures not supported\n");
            return 0;
        }
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_PROTECTED_EXT, GL_TRUE);
    }

    levels = setupMipmaps ? 1 + floor(log2(fmax(txf->tex_width, txf->tex_height))) : 1;
    // Earlier we would use the alpha channel here, now use the red channel to upload
    // This is necessary since there are no sized alpha formats for glTexStorage2D.
    // The shader has been modified accordingly as well.
    glTexStorage2D(GL_TEXTURE_2D, levels, GL_R8,
                   txf->tex_width, txf->tex_height);

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    txf->tex_width, txf->tex_height,
                    GL_RED, GL_UNSIGNED_BYTE, txf->teximage);
    if (setupMipmaps)
        glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameterf(GL_TEXTURE_2D,
                    GL_TEXTURE_MIN_FILTER, (GLfloat) minFilter);
    glTexParameterf(GL_TEXTURE_2D,
                    GL_TEXTURE_MAG_FILTER, (GLfloat) magFilter);

    glBindTexture(GL_TEXTURE_2D, savedTextureBinding);

    return txf->texobj;
}

int
nvtexfontInRasterFont(
    NVTexfontRasterFont *txf,
    int                 c)
{
    /* NOTE: No uppercase/lowercase substituion. */
    if ((c >= txf->min_glyph) && (c < txf->min_glyph + txf->range)) {
        if (txf->lut[c - txf->min_glyph]) {
            return 1;
        }
    }
    return 0;
}


NVTexfontRasterFont*
nvtexfontInitRasterFont(
    NVTexfontRasterFontName font,
    GLuint                  texobj,
    GLboolean               setupMipmaps,
    GLenum                  minFilter,
    GLenum                  magFilter)
{
    NVTexfontRasterFont *txf = loadFontData(font);

    if (rasterProg == 0)
        initShaders();

    if (txf) {
        establishTexture(txf, texobj, setupMipmaps, minFilter, magFilter);
    }
    return txf;
}

void
nvtexfontUnloadRasterFont(
    NVTexfontRasterFont *txf)
{

    deinitShaders();

    if (txf->texobj)
        glDeleteTextures(1, &txf->texobj);
    if (txf->teximage)
        FREE(txf->teximage);

    FREE(txf->tgi);
    FREE(txf->lut);
    FREE(txf->st);
    FREE(txf->vert);
    FREE(txf);
}

void
nvtexfontRenderString_All(
    NVTexfontRasterFont *txf,
    char                *string,
    float               x,
    float               y,
    float               scaleX,
    float               scaleY,
    float               r,
    float               g,
    float               b)
{

    GLint savedActiveTexture;
    GLint savedTextureBinding;
    GLboolean savedBlendEnable, savedDepthTestEnabled;
    GLint savedBlendSrcRGB, savedBlendSrcAlpha;
    GLint savedBlendDstRGB, savedBlendDstAlpha;
    GLint savedArrayBufferBinding;
    GLint savedVertArrayBinding, savedVertArrayEnabled, savedVertArraySize;
    GLint savedVertArrayStride, savedVertArrayType, savedVertArrayNormalized;
    void *savedVertArrayPointer;
    GLint savedTexArrayBinding, savedTexArrayEnabled, savedTexArraySize;
    GLint savedTexArrayStride, savedTexArrayType, savedTexArrayNormalized;
    void *savedTexArrayPointer;
    GLint savedProgram;

    GLuint uniScaleX, uniScaleY, uniX, uniY, uniTex, uniColor, uniOffset;

    int offset = 0, i = 0;

    /*****************
     * SAVE GL STATE *
     *****************/

    glGetIntegerv(GL_ACTIVE_TEXTURE, &savedActiveTexture);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &savedTextureBinding);
    savedBlendEnable = glIsEnabled(GL_BLEND);
    savedDepthTestEnabled = glIsEnabled(GL_DEPTH_TEST);

    glGetIntegerv(GL_BLEND_SRC_RGB, &savedBlendSrcRGB);
    glGetIntegerv(GL_BLEND_SRC_ALPHA, &savedBlendSrcAlpha);
    glGetIntegerv(GL_BLEND_DST_RGB, &savedBlendDstRGB);
    glGetIntegerv(GL_BLEND_DST_ALPHA, &savedBlendDstAlpha);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &savedArrayBufferBinding);

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

    glGetVertexAttribiv(TEX_ARRAY, GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING,
                        &savedTexArrayBinding);
    glGetVertexAttribiv(TEX_ARRAY, GL_VERTEX_ATTRIB_ARRAY_ENABLED,
                        &savedTexArrayEnabled);
    glGetVertexAttribiv(TEX_ARRAY, GL_VERTEX_ATTRIB_ARRAY_SIZE,
                        &savedTexArraySize);
    glGetVertexAttribiv(TEX_ARRAY, GL_VERTEX_ATTRIB_ARRAY_STRIDE,
                        &savedTexArrayStride);
    glGetVertexAttribiv(TEX_ARRAY, GL_VERTEX_ATTRIB_ARRAY_TYPE,
                        &savedTexArrayType);
    glGetVertexAttribiv(TEX_ARRAY, GL_VERTEX_ATTRIB_ARRAY_NORMALIZED,
                        &savedTexArrayNormalized);
    glGetVertexAttribPointerv(TEX_ARRAY, GL_VERTEX_ATTRIB_ARRAY_POINTER,
                              &savedTexArrayPointer);
    glGetIntegerv(GL_CURRENT_PROGRAM, &savedProgram);

    /************
     * SETUP GL *
     ************/

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, txf->texobj);
    glEnable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
                        GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glEnableVertexAttribArray(VERT_ARRAY);
    glVertexAttribPointer(VERT_ARRAY, 2, GL_SHORT, GL_FALSE, 0, txf->vert);
    glEnableVertexAttribArray(TEX_ARRAY);
    glVertexAttribPointer(TEX_ARRAY, 2, GL_FLOAT, GL_FALSE, 0, txf->st);

    /*****************
     * SETUP SHADERS *
     *****************/

    glUseProgram(rasterProg);
    uniScaleX = glGetUniformLocation(rasterProg, "scaleX");
    uniScaleY = glGetUniformLocation(rasterProg, "scaleY");
    uniX = glGetUniformLocation(rasterProg, "screenX");
    uniY = glGetUniformLocation(rasterProg, "screenY");
    uniTex = glGetUniformLocation(rasterProg, "fontTex");
    uniColor = glGetUniformLocation(rasterProg, "color");

    // Adjust scale so character size is approximately 1/10th screen height
    glUniform1f(uniScaleX, scaleX * 0.2f / txf->max_ascent);
    glUniform1f(uniScaleY, scaleY * 0.2f / txf->max_ascent);
    glUniform1f(uniX, x);
    glUniform1f(uniY, y);
    glUniform1i(uniTex, 0);
    glUniform3f(uniColor, r, g, b);

    uniOffset = glGetUniformLocation(rasterProg, "offset");

    /*************
     * DRAW TEXT *
     *************/

    while (string[i]) {
        glUniform1i(uniOffset, offset);
        offset += renderGlyph(txf, string[i++]);
    }

    /********************
     * RESTORE GL STATE *
     ********************/

    if (!savedVertArrayEnabled)
        glDisableVertexAttribArray(VERT_ARRAY);
    if (!savedTexArrayEnabled)
        glDisableVertexAttribArray(TEX_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, savedVertArrayBinding);
    glVertexAttribPointer(VERT_ARRAY, savedVertArraySize,
                          savedVertArrayType, savedVertArrayNormalized,
                          savedVertArrayStride, savedVertArrayPointer);
    glBindBuffer(GL_ARRAY_BUFFER, savedTexArrayBinding);
    glVertexAttribPointer(TEX_ARRAY, savedTexArraySize,
                          savedTexArrayType, savedTexArrayNormalized,
                          savedTexArrayStride, savedTexArrayPointer);
    glBindBuffer(GL_ARRAY_BUFFER, savedArrayBufferBinding);

    if (!savedBlendEnable)
        glDisable(GL_BLEND);
    glBlendFuncSeparate(savedBlendSrcRGB, savedBlendDstRGB,
                        savedBlendSrcAlpha, savedBlendDstAlpha);

    glUseProgram(savedProgram);

    glActiveTexture(savedActiveTexture);
    glBindTexture(GL_TEXTURE_2D, savedTextureBinding);

    if (savedDepthTestEnabled)
        glEnable(GL_DEPTH_TEST);
}

void
nvtexfontRenderString_Pos(
    NVTexfontRasterFont *txf,
    NVTexfontContext    tfc,
    char                *string,
    float               x,
    float               y)
{
    nvtexfontRenderString_All(txf, string, x, y,
                              ((nvtexfontState *)tfc)->scaleX,
                              ((nvtexfontState *)tfc)->scaleY,
                              ((nvtexfontState *)tfc)->r,
                              ((nvtexfontState *)tfc)->g,
                              ((nvtexfontState *)tfc)->b);
}

void
nvtexfontRenderString(
    NVTexfontRasterFont *txf,
    NVTexfontContext    tfc,
    char                *string)
{
    nvtexfontRenderString_All(txf, string,
                              ((nvtexfontState *)tfc)->x,
                              ((nvtexfontState *)tfc)->y,
                              ((nvtexfontState *)tfc)->scaleX,
                              ((nvtexfontState *)tfc)->scaleY,
                              ((nvtexfontState *)tfc)->r,
                              ((nvtexfontState *)tfc)->g,
                              ((nvtexfontState *)tfc)->b);
}

void
nvtexfontGetStringMetrics(
    NVTexfontRasterFont *txf,
    char                *string,
    int                 len,
    int                 *width,
    int                 *max_ascent,
    int                 *max_descent)
{
    int w, i, indx;

    w = 0;
    for (i = 0; i < len; i++) {
        if (string[i] == 27) {
            switch (string[i + 1]) {
            case 'M': i += 4;  break;
            case 'T': i += 7;  break;
            case 'L': i += 7;  break;
            case 'F': i += 13; break;
            }
        } else {
            indx = getGlyphIndex(txf, string[i]);
            if (indx != -1) {
                w += txf->tgi[indx].advance;
            }
        }
    }
    *width = w;
    *max_ascent = txf->max_ascent;
    *max_descent = txf->max_descent;
}
