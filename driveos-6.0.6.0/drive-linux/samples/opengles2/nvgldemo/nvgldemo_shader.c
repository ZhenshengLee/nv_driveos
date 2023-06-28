/*
 * nvgldemo_shader.c
 *
 * Copyright (c) 2007-2014, NVIDIA CORPORATION. All rights reserved.
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
// This file illustrates how to set up a GLES2+ shader using either a
//   precompiled binary or runtime compiled source.
//

#include "nvgldemo.h"
#include <GLES3/gl3.h>
#include <GLES2/gl2ext.h>

typedef void (*PFNGLPROGRAMBINARY)(GLuint, GLenum, const void *, GLsizei);
typedef void (*PFNGLGETPROGRAMBINARY)(GLuint, GLsizei, GLsizei *,GLenum *,void *);

// Function to print logs when shader compilation fails
static void
NvGlDemoShaderDebug(
    GLuint obj, GLenum status, const char* op)
{
    int success;
    int len;
    char *str = NULL;
    if (status == GL_COMPILE_STATUS) {
        glGetShaderiv(obj, GL_INFO_LOG_LENGTH, &len);
        if (len > 0) {
            str = (char*)MALLOC(len * sizeof(char));
            glGetShaderInfoLog(obj, len, NULL, str);
        }
    } else { // LINK or VALIDATE
        glGetProgramiv(obj, GL_INFO_LOG_LENGTH, &len);
        if (len > 0) {
            str = (char*)MALLOC(len * sizeof(char));
            glGetProgramInfoLog(obj, len, NULL, str);
        }
    }
    if (str != NULL && *str != '\0') {
        NvGlDemoLog("--- %s log ---\n", op);
        NvGlDemoLog(str);
    }
    if (str) { FREE(str); }

    // check the compile / link status.
    if (status == GL_COMPILE_STATUS) {
        glGetShaderiv(obj, status, &success);
        if (!success) {
            glGetShaderiv(obj, GL_SHADER_SOURCE_LENGTH, &len);
            if (len > 0) {
                str = (char*)MALLOC(len * sizeof(char));
                glGetShaderSource(obj, len, NULL, str);
                if (str != NULL && *str != '\0') {
                    NvGlDemoLog("--- %s code ---\n", op);
                    NvGlDemoLog(str);
                }
                FREE(str);
            }
        }
    } else { // LINK or VALIDATE
        glGetProgramiv(obj, status, &success);
    }

    if (!success)
    {
        NvGlDemoLog("--- %s failed ---\n", op);
        EXIT(-1);
    }
}

#ifdef USE_BINARY_SHADERS
// Take precompiled shader binaries and builds a shader program
unsigned int
NvGlDemoLoadShaderBinStrings(
    const char* vertBin, unsigned int vertBinSize,
    const char* fragBin, unsigned int fragBinSize,
    unsigned char link,
    unsigned char debugging)
{
    // Shaders not supportable for pre-ES2 OpenGL
    // Binary shaders not supported for non-ES OpenGL
#   ifndef GL_ES_VERSION_2_0
    return 0;
#   else  // GL_ES_VERSION_2_0

    GLuint prog       = 0;
    GLuint vertShader = 0;
    GLuint fragShader = 0;
    GLint  linkStatus = GL_FALSE;
    GLenum error      = GL_NO_ERROR;

    // Create the GL shader objects
    vertShader = glCreateShader(GL_VERTEX_SHADER);
    fragShader = glCreateShader(GL_FRAGMENT_SHADER);

    if (!vertShader || !fragShader)
        goto NvGlDemoLoadShaderBinStrings_fail;

    // Load the binary data into the shader objects
    error = glGetError();
    if (error != GL_NO_ERROR) {
        NvGlDemoLog("--- Error before loading shaders ---\n");
        goto NvGlDemoLoadShaderBinStrings_fail;
    }

    glShaderBinary(1, &vertShader,
                   GL_NVIDIA_PLATFORM_BINARY_NV, vertBin, vertBinSize);
    error = glGetError();
    if (error != GL_NO_ERROR) {
        NvGlDemoLog("--- Vertex glShaderBinary failed ---\n");
        goto NvGlDemoLoadShaderBinStrings_fail;
    }
    glShaderBinary(1, &fragShader,
                   GL_NVIDIA_PLATFORM_BINARY_NV, fragBin, fragBinSize);
    error = glGetError();
    if (error != GL_NO_ERROR) {
        NvGlDemoLog("--- Fragment glShaderBinary failed ---\n");
        goto NvGlDemoLoadShaderBinStrings_fail;
    }

    // Create the program
    prog = glCreateProgram();
    if (!prog)
        goto NvGlDemoLoadShaderBinStrings_fail;

    // Attach the shaders to the program
    glAttachShader(prog, vertShader);
    glAttachShader(prog, fragShader);

#if 0
    // Delete the shaders
    glDeleteShader(vertShader);
    glDeleteShader(fragShader);
#endif

    // Link (if requested) the shader program
    if (link) {
        glLinkProgram(prog);
        glGetProgramiv(prog, GL_LINK_STATUS, &linkStatus);
        if (debugging || ((GLboolean)linkStatus != GL_TRUE))
            NvGlDemoShaderDebug(prog, GL_LINK_STATUS, "Program Link");
        if ((GLboolean)linkStatus != GL_TRUE)
            goto NvGlDemoLoadShaderBinStrings_fail;
    }

    return prog;

NvGlDemoLoadShaderBinStrings_fail:

    if (vertShader)
        glDeleteShader(vertShader);

    if (fragShader)
        glDeleteShader(fragShader);

    if (prog)
        glDeleteProgram(prog);

    return 0;

#   endif // GL_ES_VERSION_2_0
}

// Takes shader binary files, loads them, and builds a shader program
unsigned int
NvGlDemoLoadShaderBinFiles(
    const char* vertFile,
    const char* fragFile,
    unsigned char link,
    unsigned char debugging)
{
    // Shaders not supportable for pre-ES2 OpenGL
    // Binary shaders not supported for non-ES OpenGL
#   ifndef GL_ES_VERSION_2_0
    return 0;
#   else  // GL_ES_VERSION_2_0

    GLuint prog = 0;
    char*  vertBinary = 0;
    char*  fragBinary = 0;
    GLint  vertBinarySize;
    GLint  fragBinarySize;

    // Load the shader files
    vertBinary    = NvGlDemoLoadFile(vertFile, (unsigned int*)&vertBinarySize);
    fragBinary    = NvGlDemoLoadFile(fragFile, (unsigned int*)&fragBinarySize);
    if (!vertBinary || !fragBinary) goto done;

    // Create a shader program
    prog = NvGlDemoLoadShaderBinStrings(vertBinary, vertBinarySize,
                                        fragBinary, fragBinarySize,
                                        link, debugging);

    done:

    FREE(fragBinary);
    FREE(vertBinary);
    return prog;

#   endif // GL_ES_VERSION_2_0
}
#endif

// Takes shader source strings, compiles them, and builds a shader program
unsigned int
NvGlDemoLoadShaderSrcStrings(
    const char* vertSrc, int vertSrcSize,
    const char* fragSrc, int fragSrcSize,
    unsigned char link,
    unsigned char debugging)
{
    GLuint prog       = 0;
    GLuint vertShader = 0;
    GLuint fragShader = 0;
    GLint  status     = GL_FALSE;

    // Create the GL shader objects
    vertShader = glCreateShader(GL_VERTEX_SHADER);
    fragShader = glCreateShader(GL_FRAGMENT_SHADER);

    if (!vertShader || !fragShader)
        goto NvGlDemoLoadShaderSrcStrings_fail;

    // Load shader sources into GL and compile
    glShaderSource(vertShader, 1, (const char**)&vertSrc, &vertSrcSize);
    glCompileShader(vertShader);
    glGetShaderiv(vertShader, GL_COMPILE_STATUS, &status);
    if (debugging || ((GLboolean)status != GL_TRUE))
        NvGlDemoShaderDebug(vertShader, GL_COMPILE_STATUS, "Vert Compile");
    if ((GLboolean)status != GL_TRUE)
        goto NvGlDemoLoadShaderSrcStrings_fail;

    glShaderSource(fragShader, 1, (const char**)&fragSrc, &fragSrcSize);
    glCompileShader(fragShader);
    glGetShaderiv(fragShader, GL_COMPILE_STATUS, &status);
    if (debugging || ((GLboolean)status != GL_TRUE))
        NvGlDemoShaderDebug(fragShader, GL_COMPILE_STATUS, "Frag Compile");
    if ((GLboolean)status != GL_TRUE)
        goto NvGlDemoLoadShaderSrcStrings_fail;

    // Create the program
    prog = glCreateProgram();
    if (!prog)
        goto NvGlDemoLoadShaderSrcStrings_fail;

    // Attach the shaders to the program
    glAttachShader(prog, vertShader);
    glAttachShader(prog, fragShader);

    // Delete the shaders
    glDeleteShader(vertShader);
    glDeleteShader(fragShader);

    // Link (if requested) and validate the shader program
    if (link) {
        glLinkProgram(prog);
        glGetProgramiv(prog, GL_LINK_STATUS, &status);
        if (debugging || ((GLboolean)status != GL_TRUE))
            NvGlDemoShaderDebug(prog, GL_LINK_STATUS, "Program Link");
        if ((GLboolean)status != GL_TRUE)
            goto NvGlDemoLoadShaderSrcStrings_fail;
    }

    return prog;

 NvGlDemoLoadShaderSrcStrings_fail:

    if (vertShader)
        glDeleteShader(vertShader);

    if (fragShader)
        glDeleteShader(fragShader);

    if (prog)
        glDeleteProgram(prog);

    return 0;
}

// Takes shader source files, loads them, and builds a shader program
unsigned int
NvGlDemoLoadShaderSrcFiles(
    const char* vertFile,
    const char* fragFile,
    unsigned char link,
    unsigned char debugging)
{
    // Shaders not supportable for pre-ES2 OpenGL
#   ifndef GL_ES_VERSION_2_0
    return 0;
#   else  // GL_ES_VERSION_2_0

    GLuint prog = 0;
    char*  vertSource;
    char*  fragSource;
    GLint  vertSourceSize;
    GLint  fragSourceSize;

    // Load the shader files
    vertSource    = NvGlDemoLoadFile(vertFile, (unsigned int*)&vertSourceSize);
    fragSource    = NvGlDemoLoadFile(fragFile, (unsigned int*)&fragSourceSize);
    if (!vertSource || !fragSource) goto done;

    // Create a shader program
    prog = NvGlDemoLoadShaderSrcStrings(vertSource, vertSourceSize,
                                        fragSource, fragSourceSize,
                                        link, debugging);

    done:

    FREE(fragSource);
    FREE(vertSource);
    return prog;

#   endif // GL_ES_VERSION_2_0
}

unsigned int
NvGlDemoLoadBinaryProgram(
    const char* fileName,
    unsigned char debugging)
{
#ifndef GL_ES_VERSION_3_0
    return 0;
#else
    GLuint prog       = 0;
    GLint  status     = GL_FALSE;
    GLenum binaryFormat;
    const void* binary;
    unsigned int length;
    unsigned int fileSize;
    unsigned int totAllocSz = 0;
    char* data;
    PFNGLPROGRAMBINARY pglProgramBinary = NULL;

    data = NvGlDemoLoadFile(fileName, &fileSize);

    // Length comes first, then binary format, then the entire program binary.
    // WARNING: file transfer across different architectures (endianness etc)
    // is therefore not supported.
    if (data) {
        length = *(unsigned int*)(data);
        binaryFormat = (GLenum)*(unsigned int*)(data + sizeof(unsigned int));
        binary = (data + sizeof(unsigned int) * 2);
        totAllocSz = (length + (sizeof(unsigned int) * 2));
        if (totAllocSz != fileSize) {
           NvGlDemoLog("Failed loading-size mismatch %d %d\n",fileSize,totAllocSz);
           goto NvGlDemoLoadBinaryProgram_fail;
        }
    } else {
        if (debugging) {
           NvGlDemoLog("Failed loading %s.\n", fileName);
        }
        goto NvGlDemoLoadBinaryProgram_fail;
    }

    // Create the program
    prog = glCreateProgram();
    if (!prog) {
        goto NvGlDemoLoadBinaryProgram_fail;
    }
    NVGLDEMO_EGL_GET_PROC_ADDR(glProgramBinary, NvGlDemoLoadBinaryProgram_fail, PFNGLPROGRAMBINARY);
    pglProgramBinary(prog, binaryFormat, binary, length);
    glGetProgramiv(prog, GL_LINK_STATUS, &status);
    if (debugging || ((GLboolean)status != GL_TRUE))
        NvGlDemoShaderDebug(prog, GL_LINK_STATUS, "Binary Program Link");
    if ((GLboolean)status != GL_TRUE)
        goto NvGlDemoLoadBinaryProgram_fail;

    FREE(data);

    return prog;

NvGlDemoLoadBinaryProgram_fail:
    FREE(data);

    if (prog)
        glDeleteProgram(prog);

    return 0;
#endif
}

// Save the binary program to memory.
unsigned int
NvGlDemoSaveBinaryProgram(
        unsigned int progName,
        const char* fileName)
{
#ifndef GL_ES_VERSION_3_0
    return 0;
#else
    GLsizei binaryLength = 0;
    void* binary = NULL;
    GLenum error = GL_NO_ERROR;
    unsigned int bytesWritten = 0;
    unsigned int allocSize = 0;
    PFNGLGETPROGRAMBINARY pglGetProgramBinary = NULL;

    glGetProgramiv(progName, GL_PROGRAM_BINARY_LENGTH, &binaryLength);

    error = glGetError();
    if (error != GL_NO_ERROR) {
        NvGlDemoLog("NvGlDemoSaveBinaryProgram: glGetProgramiv failed with %x\n", error);
        goto end;
    }

    allocSize = binaryLength + sizeof(unsigned int) * 2;
    binary = MALLOC(allocSize);
    NVGLDEMO_EGL_GET_PROC_ADDR(glGetProgramBinary, end, PFNGLGETPROGRAMBINARY);
    // Store length in integer, then format (as an integer), then followed
    // by the actual program binary.
    // WARNING: not portable across architectures
    pglGetProgramBinary(progName,
            binaryLength,
            (GLsizei*)binary,
            (GLenum*)((unsigned char*)binary + sizeof(GLsizei)),
            ((unsigned char*)binary + sizeof(GLsizei) + sizeof(GLenum))
            );

    error = glGetError();
    if (error != GL_NO_ERROR) {
        NvGlDemoLog("NvGlDemoSaveBinaryProgram: glGetProgramBinary failed with %x\n", error);
        goto end;
    }

    bytesWritten = NvGlDemoSaveFile(fileName, binary, allocSize);

end:
    if (binary) {
        FREE(binary);
    }

    return bytesWritten;

#endif
}

// Link the program.
unsigned int NvGlDemoLinkProgram(
        unsigned int prog,
        unsigned char debugging)
{
    GLint  status = GL_FALSE;

    if (prog) {
        glLinkProgram(prog);
        glGetProgramiv(prog, GL_LINK_STATUS, &status);
        if (debugging || ((GLboolean)status != GL_TRUE)) {
            NvGlDemoShaderDebug(prog, GL_LINK_STATUS, "Program Link");
        }
        if ((GLboolean)status != GL_TRUE) {
            goto NvGlDemoLinkProgram_fail;
        }
    }

    return prog;

NvGlDemoLinkProgram_fail:

    if (prog)
        glDeleteProgram(prog);

    return 0;
}

#ifdef USE_EXTERN_SHADERS
unsigned int NvGlDemoLoadExternShader(
        const char* vertFile,
        const char* fragFile,
        unsigned char link,
        unsigned char debugging,
        const char* prgFile)
{
    GLuint prog = 0;

    if (prgFile && demoOptions.useProgramBin) {
        // Try to load binary program, only load shaders if we fail.
        prog = NvGlDemoLoadBinaryProgram(prgFile, debugging);
        if (debugging && prog) {
           NvGlDemoLog("Success loading binary program.\n");
        } else if (!prog) {
           NvGlDemoLog("Binary program does not exist. Trying to create program from source now.\n");
        }
    }
    if (!prog) {
#ifdef USE_BINARY_SHADERS
        if (demoOptions.useProgramBin) {
           NvGlDemoLog("Can't use both binary shaders and binary programs.\n");
           return 0;
        }
        prog = NvGlDemoLoadShaderBinFiles( vertFile,
                fragFile,
                link,
                debugging );
#else
        prog = NvGlDemoLoadShaderSrcFiles( vertFile,
                fragFile,
                link,
                debugging );
#endif /* USE_BINARY_SHADERS */
        if (demoOptions.useProgramBin) {
           /* Before dumping program binary, we need to make sure that program is linked.
              Otherwise glGetProgramBinary returns GL_INVALID_OPERATION = 0x502 error.*/
           if ((link == GL_FALSE) && prog && prgFile) {
               prog = NvGlDemoLinkProgram(prog, debugging);
           }

           if (prog && prgFile) {
               if (!NvGlDemoSaveBinaryProgram(prog, prgFile)) {
                   NvGlDemoLog("Failed saving binary program.\n");
               }
           }
        }
    }
    return prog;
}
#endif // USE_EXTERN_SHADERS

unsigned int NvGlDemoLoadPreCombinedShader(
        const char* vertSrc, int vertSrcSize,
        const char* fragSrc, int fragSrcSize,
        unsigned char link,
        unsigned char debugging,
        const char* prgFile )
{
    GLuint prog = 0;

    if (prgFile && demoOptions.useProgramBin) {
        // Try to load binary program, only load shaders if we fail.
        prog = NvGlDemoLoadBinaryProgram(prgFile, debugging);
        if (debugging && prog) {
            NvGlDemoLog("Success loading binary program.\n");
        } else if (!prog) {
            NvGlDemoLog("Binary program does not exist. Trying to create program from source now.\n");
        }
    }
    if (!prog) {
#ifdef USE_BINARY_SHADERS
        if (demoOptions.useProgramBin) {
           NvGlDemoLog("Can't use both binary shaders and binary programs.\n");
           return 0;
        }
        prog = NvGlDemoLoadShaderBinStrings( vertSrc,
                vertSrcSize,
                fragSrc,
                fragSrcSize,
                link,
                debugging );
#else
        prog = NvGlDemoLoadShaderSrcStrings( vertSrc,
                vertSrcSize,
                fragSrc,
                fragSrcSize,
                link,
                debugging );
#endif /* USE_BINARY_SHADERS */
        if (demoOptions.useProgramBin) {
           /* Before dumping program binary, we need to make sure that program is linked.
              Otherwise glGetProgramBinary returns GL_INVALID_OPERATION = 0x502 error.*/
           if ((link == GL_FALSE) && prog && prgFile) {
               prog = NvGlDemoLinkProgram(prog, debugging);
           }

           if (prog && prgFile) {
               if (!NvGlDemoSaveBinaryProgram(prog, prgFile)) {
                   NvGlDemoLog("Failed saving binary program.\n");
               }
           }
        }
    }

    return prog;
}
