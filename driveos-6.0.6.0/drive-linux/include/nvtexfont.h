/*
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

#ifndef __NV_TEXFONT_H__
#define __NV_TEXFONT_H__


/**
 * @file
 * <b>NVIDIA Graphics 3D Font Text API</b>
 *
 * @b Description: NVTexfont is an NVIDIA library that draws font text in
 *                a 3D scene.
 */


/**
 * @defgroup 3d_group 3D Font Text APIs
 *
 * Implements 3D APIs for font text used by applications.
 *
 * For details on the original implementation, see:
 * <a class="el"
 *  href="http://www.opengl.org/code/detail/texfont_glut_based_library_for_creating_fonts_out_of_textures"
 *  target="_blank">
 *  http://www.opengl.org/code/detail/texfont_glut_based_library_for_creating_fonts_out_of_textures</a>.
 *
 * The OpenGL ES 2.0 NVTexfont library has been expanded to include both raster
 * and vector fonts. The main functions are described below.
 *
 * @ingroup graph_group
 * @{
 */
#ifdef __cplusplus
extern "C" {
#endif

//
// Public data types
//

#define TXF_FORMAT_BYTE     0
#define TXF_FORMAT_BITMAP   1

typedef void * NVTexfontContext;

typedef enum {
    NV_TEXFONT_DEFAULT = 0,
    NV_TEXFONT_HELVETICA = 0,
} NVTexfontRasterFontName;

typedef enum {
    NV_VECTOR_TEXFONT_DEFAULT = 0,
    NV_VECTOR_TEXFONT_INCONSOLATA = 0
} NVTexfontVectorFontName;

typedef struct _NVTexfontRasterFont NVTexfontRasterFont;
typedef struct _NVTexfontVectorFont NVTexfontVectorFont;

//
// General functions
//

extern char*
nvtexfontErrorString(void);

/**
 * Allocates a handle for an NVTexfont context. Default settings are
 * held for each context.
 */
extern NVTexfontContext
nvtexfontAllocContext(void);

/**
 * Frees the specified NVTexfont context handle.
 */
extern void
nvtexfontFreeContext(
    NVTexfontContext tfc);

/**
 * Sets the x,y attributes for the NVTexfont context \a tfc.
 *
 * \param tfc Specifies the NVTexfont context for which to set attributes.
 * \param x Specifies the x attribute to set.
 * \param y Specifies the y attribute to set.
 */
extern void
nvtexfontSetContextPos(
    NVTexfontContext tfc,
    float            x,
    float            y);

/**
 * Sets the x,y attributes for the NVTexfont context \a tfc.
 *
 * \param tfc Specifies the NVTexfont context for which to set attributes.
 * \param scaleX Specifies the scale value for X.
 * \param scaleY Specifies the scale value for Y.
 */
extern void
nvtexfontSetContextScale(
    NVTexfontContext tfc,
    float            scaleX,
    float            scaleY);

extern void
nvtexfontSetContextColor(
    NVTexfontContext tfc,
    float            r,
    float            g,
    float            b);

// TODO: Raster and vector font structures/functions should be merged
// TODO:   so that we initialize a font object as one or the other and
// TODO:   then use the same set of top level functions for rendering.

/** @name Raster Font Functions
 *
 */
/*@{*/

extern int
nvtexfontInRasterFont(
    NVTexfontRasterFont     *txf,
    int                     c);
/**
 * Initializes an \c %NVTexfontRasterFont.
 *
 * By default, NVTexfont supports only the ::NV_TEXFONT_HELVETICA font.
 * ::NV_TEXFONT_DEFAULT is the same as \c NV_TEXFONT_HELVETICA. More may be
 * compiled into the library, at the cost of making the library larger.
 *
 * \param font Specifies the raster font to initialize.
 * \param texobj Specifies the texture object. If 0, a name is generated
 *     using the OpenGL 4 \c glGenTextures function.
 * \param setupMipmaps Specifies the MIP maps to accompany this texture.
 * \param minFilter Specifies the minimum filter for the MIP maps.
 * \param magFilter Specifies the maximum filter for the MIP maps.
 */
extern NVTexfontRasterFont*
nvtexfontInitRasterFont(
    NVTexfontRasterFontName font,
    GLuint                  texobj,
    GLboolean               setupMipmaps,
    GLenum                  minFilter,
    GLenum                  magFilter);

/**
 *  Frees textures used by the font \a txf.
 *
 * \param txf A pointer to the textures to free.
 */
extern void
nvtexfontUnloadRasterFont(
    NVTexfontRasterFont     *txf);

/**
 * Renders the specified null-terminated string by using the NVTexfont \a txf.
 *
 * The text gets drawn at position (x, y) on the screen with the relevant scale
 * and the color (r,g,b).
 *
 * The size of the text is one unit per font pixel. This is scaled so that
 * the height of the text is 1/10th the height of the screen when scale = 1.0.
 * \c txfGetStringMetrics may be used to get the width, maximum ascent from
 * baseline, and maximum descent from baseline.
 *
 * \param txf A pointer to the texture font.
 * \param string A pointer to the string to render.
 * \param x The x position at which to render.
 * \param y The y position at which to render.
 * \param scaleX The X scale.
 * \param scaleY The Y scale.
 * \param r The red value.
 * \param g The green value.
 * \param b The blue value.
 */
extern void
nvtexfontRenderString_All(
    NVTexfontRasterFont     *txf,
    char                    *string,
    float                   x,
    float                   y,
    float                   scaleX,
    float                   scaleY,
    float                   r,
    float                   g,
    float                   b);

/**
 * Renders the specified null-terminated string as above but uses the
 * position/scale/color from the \c NVTexfontContext where omitted.
 *
 * \param txf A pointer to the texture font.
 * \param tfc The texture font context.
 * \param string A pointer to the string to render.
 * \param x The x position at which to draw.
 * \param y The y position at which to draw.
 */
extern void
nvtexfontRenderString_Pos(
    NVTexfontRasterFont     *txf,
    NVTexfontContext        tfc,
    char                    *string,
    float                   x,
    float                   y);

extern void
nvtexfontRenderString(
    NVTexfontRasterFont     *txf,
    NVTexfontContext        tfc,
    char                    *string);

extern void
nvtexfontGetStringMetrics(
    NVTexfontRasterFont     *txf,
    char                    *string,
    int                     len,
    int                     *width,
    int                     *max_ascent,
    int                     *max_descent);

/*@}*/
/** @name Vector Font Functions
 *
*/
/*@{*/

extern int
nvtexfontInVectorFont(
    NVTexfontVectorFont     *vtf,
    int                     c);

/**
 * Initialize an NVTexfontVectorFont.
 *
 * By default, NVTexfont supports one vector font:
 * ::NV_VECTOR_TEXFONT_INCONSOLATA. ::NV_TEXFONT_DEFAULT is the same as
 * \c NV_VECTOR_TEXFONT_INCONSOLATA.
 *
 * \param font Specifies the vector font to initialize.
 * \param antialias Specifies whether the text should be antialiased when drawn.
 * \param use_vbo Specifies whether a vertex buffer object (VBO) should be
 *     used to speed up the rendering.
 */
extern NVTexfontVectorFont*
nvtexfontInitVectorFont(
    NVTexfontVectorFontName font,
    GLboolean               antialias,
    GLboolean               use_vbo);

/**
 * Frees resources used by the vector font \a vtf, including the VBO if used.
 *
 * \param vtf A pointer to the vector font resources to free.
 */
extern void
nvtexfontUnloadVectorFont(
    NVTexfontVectorFont     *vtf);

/**
 * Renders the specified null-terminated string by using the
 * \c NVTexfontVectorFont \a vtf.
 *
 * A scale of 1.0 is equivalent to 1/10 the height of the viewport. The text
 * gets drawn in the color specified by (r,g,b).
 *
 * \param vtf A pointer to the vector font to use when rendering.
 * \param string A pointer to the string to render.
 * \param scaleX The scale for X.
 * \param scaleY The scale for Y.
 * \param x The x position at which to draw, after being adjusted for \a scaleX.
 * \param y The y position at which to draw, after being adjusted for \a scaleY.
 * \param r The red value.
 * \param g The green value.
 * \param b The blue value.
 */
extern void
nvtexfontRenderVecString_All(
    NVTexfontVectorFont     *vtf,
    char                    *string,
    float                   x,
    float                   y,
    float                   scaleX,
    float                   scaleY,
    float                   r,
    float                   g,
    float                   b);

/**
 * Renders the specified null-terminated string as above but uses the
 * position/scale/color from the \c NVTexfontContext where omitted.
 *
 * \param vtf A pointer to the vector font to use when rendering.
 * \param tfc The texture font context.
 * \param string A pointer to the string to render.
 * \param x The x position at which to draw.
 * \param y The y position at which to draw.
 */
extern void
nvtexfontRenderVecString_Pos(
    NVTexfontVectorFont     *vtf,
    NVTexfontContext        tfc,
    char                    *string,
    float                   x,
    float                   y);

/**
 * Renders the specified null-terminated string as above but uses the
 * position/scale/color from the \c NVTexfontContext where omitted.
 *
 * \param vtf A pointer to the vector font to use when rendering.
 * \param tfc The texture font context.
 * \param string A pointer to the string to render.
 */
extern void
nvtexfontRenderVecString(
    NVTexfontVectorFont     *vtf,
    NVTexfontContext        tfc,
    char                    *string);

/** @} */

#ifdef __cplusplus
}
#endif

/** @} */
#endif /* __NV_TEXFONT_H__ */

