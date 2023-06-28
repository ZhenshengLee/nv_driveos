/*
 * nvgldemo_texture.c
 *
 * Copyright (c) 2010-2019, NVIDIA CORPORATION. All rights reserved.
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
// This file illustrates how to load an image file into a texture.
//   Only uncompressed RGB/RGBA TGA files are supported.
//

#include "nvgldemo.h"
#include "GLES2/gl2.h"
#include "GLES2/gl2ext.h"

static int failedImageIndex = -1;
// Load a set of TGA files as a texture.
//   Returns the ID, and leaves it bound to current texture unit.
unsigned int
NvGlDemoLoadTgaFromBuffer(
    unsigned int target,
    unsigned int count,
    unsigned char** buffer)
{
    unsigned char* filedata = NULL;
    unsigned int facetarget=0;
    unsigned int width, height, bpp, format, sformat, pot=0;
    unsigned int cubewidth=0, cubeheight=0, cubeformat=0;
    unsigned char* body;
    unsigned char* temp;
    unsigned int i, j, k;
    const char* glExtensions;
    GLuint id;

    // Validate inputs
    switch (target) {

        case GL_TEXTURE_2D:
            facetarget = GL_TEXTURE_2D;
            if (count != 1) {
                NvGlDemoLog("Unexpected file count (%d) for 2D texture\n",
                            count);
                return 0;
            }
            break;

        case GL_TEXTURE_CUBE_MAP:
            facetarget = GL_TEXTURE_CUBE_MAP_POSITIVE_X;
            if (count != 6) {
                NvGlDemoLog("Unexpected file count (%d) for cube map\n",
                            count);
                return 0;
            }
            break;

        default:
            NvGlDemoLog("Unsupported texture target 0x(%04x)\n", target);
            return 0;
    }

    // Create and bind texture
    glGenTextures(1, &id);
    glBindTexture(target, id);
    if (demoOptions.isProtected) {
        glExtensions = (const char *) glGetString(GL_EXTENSIONS);
        if (!STRSTR(glExtensions, "GL_EXT_protected_textures")) {
            NvGlDemoLog("Protected Textures not supported\n");
            return 0;
        }
        glTexParameteri(target, GL_TEXTURE_PROTECTED_EXT, GL_TRUE);
    }

    // Load each face
    for (k=0; k<count; k++) {

        // Load file
        filedata = buffer[k];

        // Parse header
        if ((filedata[1] != 0) || (filedata[2] != 2)) {
            NvGlDemoLog("Cannot parse image %d\n.", k);
            NvGlDemoLog("  Only uncompressed tga files are supported");
            failedImageIndex = k;
            goto fail;
        }
        width  = ((unsigned int)filedata[13] << 8)
               |  (unsigned int)filedata[12];
        height = ((unsigned int)filedata[15] << 8)
               |  (unsigned int)filedata[14];
        bpp    =  (unsigned int)filedata[16] >> 3;
        format = (bpp == 4) ? GL_RGBA : GL_RGB;
        sformat = (bpp == 4) ? GL_RGBA8 : GL_RGB8;
        pot    = ((width & (width-1))==0) && ((height & (height-1))==0);

        // For cubemaps, validate size/format
        if (target == GL_TEXTURE_CUBE_MAP) {

            if (width != height) {
                NvGlDemoLog("Texture %d is not square (%d x %d)\n",
                            k, width, height);
                failedImageIndex = k;
                goto fail;
            }

            if (k == 0) {
                cubewidth  = width;
                cubeheight = height;
                cubeformat = format;
            } else if ((width  != cubewidth)  ||
                       (height != cubeheight) ||
                       (format != cubeformat)) {
                NvGlDemoLog("Texture %d does not match texture 0 on this cube\n"
                            "  (%d,%d,0x%04x) vs (%d,%d,0x%04x)\n",
                            k,
                            width, height, format,
                            cubewidth, cubeheight, cubeformat);
                failedImageIndex = k;
                goto fail;
            }

        }

        // Get image data and convert BGRA to RGBA
        temp = body = filedata + 18;
        for (i = 0; i < height; i++) {
            for(j = 0; j < width; j++) {
                unsigned char c;

                c = temp[0];
                temp[0] = temp[2];
                temp[2] = c;

                temp += bpp;
            }
        }

        // Upload data
        if (k == 0) {
            glTexStorage2D(target, 1 + floor(log2(fmax(width, height))), sformat,
                           width, height);
        }
        glTexSubImage2D(facetarget+k, 0, 0, 0, width, height,
                        format, GL_UNSIGNED_BYTE, body);

    }

    // Set texture parameters and generate mipmaps if appropriate
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    if (pot) {
        glTexParameteri(target, GL_TEXTURE_MIN_FILTER,
                        GL_LINEAR_MIPMAP_LINEAR);
        glGenerateMipmap(target);
    } else {
        glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    }

    // Clean up and return
    return id;

    fail:
    if (id) glDeleteTextures(1, &id);
    return 0;
}

// Load a set of TGA files as a texture.
// Returns the ID, and leaves it bound to current texture unit.
unsigned int
NvGlDemoLoadTga(
    unsigned int target,
    unsigned int count,
    const char** names)
{
    unsigned int filesize;
    unsigned char **buffer = (unsigned char**) malloc(count * sizeof(unsigned char*));
    unsigned int k;
    GLuint id = 0;

    for (k=0; k<count; k++) {

        buffer[k] = (unsigned char*)NvGlDemoLoadFile(names[k], &filesize);
        if (filesize == 0 || !buffer[k]) {
            goto finish;
        }
    }

    id = NvGlDemoLoadTgaFromBuffer(target, count, buffer);

    if (id == 0) {
        NvGlDemoLog("File %s failed\n", names[failedImageIndex]);
    }

    // Clean up and return
    finish:
    for (k=0; k<count && buffer[k]; k++) {
        FREE(buffer[k]);
    }
    FREE(buffer);

    return id;

}
