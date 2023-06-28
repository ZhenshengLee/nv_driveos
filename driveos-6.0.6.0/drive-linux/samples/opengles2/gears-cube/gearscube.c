/*
 * gearscube.c
 *
 * Copyright (c) 2003-2019, NVIDIA CORPORATION. All rights reserved.
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

// This file illustrates several different render-to-texture approaches,
//   drawing the spinning gears to an offscreen surface and then mapping
//   it to the faces of a spinning cube.

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include "nvgldemo.h"
#include "gearslib.h"

// Depending on compile options, we either build in the shader sources or
//   binaries or load them from external data files at runtime.
//   (Variables are initialized to the file contents or name, respectively).
#ifdef USE_EXTERN_SHADERS
static const char cubeVertShader[] = { VERTFILE(cube_vert) };
static const char cubeFragShader[] = { FRAGFILE(cube_frag) };
#else
static const char cubeVertShader[] = {
#   include VERTFILE(cube_vert)
};
static const char cubeFragShader[] = {
#   include FRAGFILE(cube_frag)
};
#endif

// Flag indicating it is time to shut down
static GLboolean shutdown = GL_FALSE;

// Compile-time defines for methods which we will support. Comment out
//   to disable.
// 1) Texture bound to framebuffer object:
//    In GLES2, this is the prefered method to do render-to-texture
//    within an application.
// 2) Render to pixmap, bind to EGLImage and from there to texture:
//    Although we do not implement the general case here, this serves
//    as a good example of how to handle render-to-texture when the
//    renderer resides in another process or uses a different API.
//    The pixmap could be shared with another application, or could
//    be rendered to using native window system functions. By binding
//    it in this way, we can make use of it directly in GLES2 without
//    any additional copy steps.
// 3) Render to pbuffer, bind or copy to texture:
//    These examples are provided mainly as legacy methods from GLES1,
//    which did not provide framebuffer objects without extension.
//    Binding is preferable to copying, but is not required to be
//    supported by all platforms.
// Note: Two other possible variants which we don't recommend and don't
//   provide examples for are creating a texture or renderbuffer, binding
//   that to an EGLImage, and then binding that to a renderbuffer or
//   texture. These paths add the extra overhead of involving EGL without
//   any benefit over the other methods.
#define METHOD_FRAMEBUFFER
#define METHOD_PBUFFER
#if defined(EGL_KHR_image_pixmap) && defined(GL_OES_EGL_image)
#define METHOD_PIXMAP
#endif

// Enumerants for the supported methods
typedef enum {
    #ifdef METHOD_FRAMEBUFFER
    Method_Framebuffer,
    #endif
    #ifdef METHOD_PIXMAP
    Method_Pixmap,
    #endif
    #ifdef METHOD_PBUFFER
    Method_PbufferCopy,
    #endif
    Method_Count
} Methods;
Methods method = (Methods)0;

// Pixmap/image support fields
#ifdef METHOD_PIXMAP
PFNEGLCREATEIMAGEKHRPROC            pImageCreate   = NULL;
PFNEGLDESTROYIMAGEKHRPROC           pImageDestroy  = NULL;
PFNGLEGLIMAGETARGETTEXTURE2DOESPROC pImageToTex    = NULL;
EGLNativePixmapType                 gearsPixmap    = (EGLNativePixmapType)0;
EGLImageKHR                         gearsImage     = EGL_NO_IMAGE_KHR;
#endif // METHOD_PIXMAP

// Framebuffer object support fields
#ifdef METHOD_FRAMEBUFFER
GLuint          gearsFBO = 0;
GLuint          gearsRBO = 0;
#endif // METHOD_FRAMEBUFFER

// Gears rendering context/surfaces
GLint           texSize = 256;
EGLConfig       gearsConfig;
EGLContext      gearsContext = EGL_NO_CONTEXT;
EGLSurface      gearsSurface = EGL_NO_SURFACE;
GLuint          gearsTexture = 0;

// Cube shader
GLint           prog_cube = 0;
GLint           uloc_cubeCameraMat;
GLint           uloc_cubeObjectMat;
GLint           uloc_cubeTexUnit;
const GLfloat   depthnear =  5.0f;
const GLfloat   depthfar  = 60.0f;

// Cube vertex positions and texture coordinates
//   Shader maps texture coordinates [-1,+1] to [0,1] and solid fills
//   anything outside the texture. We pass +/- 1.1 to leave a small border.
GLfloat         cubeVert[24][5] =  {
    { -1.0f, -1.0f, +1.0f,  -1.1f, -1.1f },
    { +1.0f, -1.0f, +1.0f,  +1.1f, -1.1f },
    { -1.0f, +1.0f, +1.0f,  -1.1f, +1.1f },
    { +1.0f, +1.0f, +1.0f,  +1.1f, +1.1f },

    { +1.0f, +1.0f, -1.0f,  +1.1f, -1.1f },
    { -1.0f, +1.0f, -1.0f,  -1.1f, -1.1f },
    { +1.0f, -1.0f, -1.0f,  +1.1f, +1.1f },
    { -1.0f, -1.0f, -1.0f,  -1.1f, +1.1f },

    { -1.0f, +1.0f, -1.0f,  -1.1f, -1.1f },
    { -1.0f, -1.0f, -1.0f,  +1.1f, -1.1f },
    { -1.0f, +1.0f, +1.0f,  -1.1f, +1.1f },
    { -1.0f, -1.0f, +1.0f,  +1.1f, +1.1f },

    { +1.0f, -1.0f, +1.0f,  +1.1f, -1.1f },
    { +1.0f, +1.0f, +1.0f,  -1.1f, -1.1f },
    { +1.0f, -1.0f, -1.0f,  +1.1f, +1.1f },
    { +1.0f, +1.0f, -1.0f,  -1.1f, +1.1f },

    { +1.0f, -1.0f, -1.0f,  -1.1f, -1.1f },
    { +1.0f, -1.0f, +1.0f,  +1.1f, -1.1f },
    { -1.0f, -1.0f, -1.0f,  -1.1f, +1.1f },
    { -1.0f, -1.0f, +1.0f,  +1.1f, +1.1f },

    { -1.0f, +1.0f, +1.0f,  +1.1f, -1.1f },
    { -1.0f, +1.0f, -1.0f,  -1.1f, -1.1f },
    { +1.0f, +1.0f, +1.0f,  +1.1f, +1.1f },
    { +1.0f, +1.0f, -1.0f,  -1.1f, +1.1f }
};

//===========================================================================

// Set up camera matrix and viewport
static void
cubeViewSet(
    int     width,
    int     height)
{
    GLfloat matrix[16];
    GLfloat aspect;

    // Make sure correct context is current
    eglMakeCurrent(demoState.display,
                   demoState.surface, demoState.surface,
                   demoState.context);

    // Set the perspective projection
    NvGlDemoMatrixIdentity(matrix);
    if (width >= height) {
        aspect = (GLfloat)width / (GLfloat)height;
        NvGlDemoMatrixFrustum(matrix,
                              -aspect, aspect,
                              -1.0f, 1.0f,
                              depthnear, depthfar);
    } else {
        aspect = (GLfloat)height / (GLfloat)width;
        NvGlDemoMatrixFrustum(matrix,
                              -1.0f, 1.0f,
                              -aspect, aspect,
                              depthnear, depthfar);
    }
    glUniformMatrix4fv(uloc_cubeCameraMat, 1, GL_FALSE, matrix);

    // Set viewport
    glViewport(0, 0, width, height);
}

// Initialize cube rendering context
static GLboolean
cubeSceneInit(
    int     width,
    int     height)
{
    GLuint  aloc;

    // Make main context current
    eglMakeCurrent(demoState.display,
                   demoState.surface, demoState.surface,
                   demoState.context);

    // Load the shaders (The macro handles the details of binary vs.
    //   source and external vs. internal)
    prog_cube = LOADSHADER(cubeVertShader, cubeFragShader, GL_TRUE, GL_FALSE);
    if (!prog_cube) return GL_FALSE;
    glUseProgram(prog_cube);

    // Extract uniform locations
    uloc_cubeCameraMat = glGetUniformLocation(prog_cube, "cameramat");
    uloc_cubeObjectMat = glGetUniformLocation(prog_cube, "objectmat");
    uloc_cubeTexUnit   = glGetUniformLocation(prog_cube, "texunit");

    // Set and enable cube coordinates
    aloc = glGetAttribLocation(prog_cube, "vtxpos");
    glVertexAttribPointer(aloc, 3, GL_FLOAT, GL_FALSE, 5*sizeof(GLfloat),
                          &cubeVert[0][0]);
    glEnableVertexAttribArray(aloc);
    aloc = glGetAttribLocation(prog_cube, "vtxtex");
    glVertexAttribPointer(aloc, 2, GL_FLOAT, GL_FALSE, 5*sizeof(GLfloat),
                          &cubeVert[0][3]);
    glEnableVertexAttribArray(aloc);

    // Set up texture to be used for the gears
    glUniform1i(uloc_cubeTexUnit, 0);
    glGenTextures(1, &gearsTexture);
    glBindTexture(GL_TEXTURE_2D, gearsTexture);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    // Initialize the camera and viewport
    cubeViewSet(width, height);

    // Rendering settings
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.2f, 0.1f, 0.2f, 1.0f);

    return (glGetError() == GL_NO_ERROR) ? GL_TRUE : GL_FALSE;
}

// Draw a frame of the cube
static GLboolean
cubeSceneRender(void)
{
    static GLfloat          angA = 0.0f;
    static GLfloat          angB = 0.0f;
    GLfloat                 matrix[16];
    GLint                   i;

    // Make main context current
    eglMakeCurrent(demoState.display,
                   demoState.surface, demoState.surface,
                   demoState.context);

    // Clear buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Make sure cube program is current
    glUseProgram(prog_cube);

    // Rotate the cube and set up object transformation matrix
    angA += 0.175f;
    angB += 0.050f;
    NvGlDemoMatrixIdentity(matrix);
    NvGlDemoMatrixTranslate(matrix, 0.0f, 0.0f, -30.0f);
    NvGlDemoMatrixRotate(matrix, angA, 0.6f, 0.8f, 0.0f);
    NvGlDemoMatrixRotate(matrix, angB, 0.0f, 1.0f, 1.0f);
    NvGlDemoMatrixScale(matrix, 3.0f, 3.0f, 3.0f);
    glUniformMatrix4fv(uloc_cubeObjectMat, 1, GL_FALSE, matrix);

    // Draw each face of the cube
    for (i=0; i<6; ++i)
        glDrawArrays(GL_TRIANGLE_STRIP, 4*i, 4);

    return (glGetError() == GL_NO_ERROR) ? GL_TRUE : GL_FALSE;
}

// Clean up GL resources for cube
static void
cubeSceneTerm(void)
{
    // Make main context current
    eglMakeCurrent(demoState.display,
                   demoState.surface, demoState.surface,
                   demoState.context);

    // Delete texture
    if (gearsTexture) {
        glDeleteTextures(1, &gearsTexture);
        gearsTexture = 0;
    }

    // Delete shader
    if (prog_cube) {
        glDeleteProgram(prog_cube);
        prog_cube = 0;
    }
}

//===========================================================================

// Initialize the gears rendering methods
static GLboolean
gearsMethodInit(void)
{
    EGLint       count;
    EGLBoolean   status;
    const char*  surfDesc = NULL;

    // Attributes for config, context, and pbuffer
    EGLint cfgAttrs[] = {
        EGL_SURFACE_TYPE,        EGL_DONT_CARE,
        EGL_BIND_TO_TEXTURE_RGB, EGL_DONT_CARE,
        EGL_RED_SIZE,            1,
        EGL_GREEN_SIZE,          1,
        EGL_BLUE_SIZE,           1,
        EGL_ALPHA_SIZE,          1,
        EGL_DEPTH_SIZE,          16,
        EGL_SAMPLE_BUFFERS,      0,
        EGL_SAMPLES,             0,
        EGL_RENDERABLE_TYPE,     EGL_OPENGL_ES2_BIT,
        EGL_NONE
    };
    EGLint ctxAttrs[] = {
        EGL_CONTEXT_CLIENT_VERSION, 2,
        EGL_NONE
    };
    #ifdef METHOD_PBUFFER
    EGLint bufAttrs[] = {
        EGL_WIDTH,               texSize,
        EGL_HEIGHT,              texSize,
        EGL_NONE
        };
    #endif // METHOD_PBUFFER

    // Adjust config settings based on method
    switch (method) {

        #ifdef METHOD_FRAMEBUFFER
        case Method_Framebuffer:
            // EGL_SURFACE_TYPE can remain EGL_DONT_CARE, does not affect FBOs
            surfDesc = "FBOs";
            break;
        #endif // METHOD_FRAMEBUFFER

        #ifdef METHOD_PIXMAP
        case Method_Pixmap:
            cfgAttrs[1] = EGL_PIXMAP_BIT;
            surfDesc = "pixmaps";
            break;
        #endif // METHOD_PIXMAP

        #ifdef METHOD_PBUFFER
        case Method_PbufferCopy:
            cfgAttrs[1] = EGL_PBUFFER_BIT;
            surfDesc = surfDesc ? surfDesc : "pbuffers";
            break;
        #endif // METHOD_PBUFFER

        default:
            return GL_FALSE;
            break;
    }

    // Obtain a matching config
    status = eglChooseConfig(demoState.display, cfgAttrs,
                             &gearsConfig, 1, &count);
    if (!status || !count) {
        NvGlDemoLog("Couldn't obtain config for gears rendering\n");
        NvGlDemoLog("EGL driver may not support %s\n", surfDesc);
        return GL_FALSE;
    }

    // Create EGL context for rendering gears
    gearsContext = eglCreateContext(demoState.display, gearsConfig,
                                    demoState.context, ctxAttrs);
    if (gearsContext == EGL_NO_CONTEXT) {
        NvGlDemoLog("Couldn't obtain context for gears rendering\n");
        return GL_FALSE;
    }

    // Create surfaces to support selected method
    switch (method) {

        #ifdef METHOD_FRAMEBUFFER
        case Method_Framebuffer:

            // Use main EGL surface
            gearsSurface = demoState.surface;
            eglMakeCurrent(demoState.display,
                           gearsSurface, gearsSurface,
                           gearsContext);

            // Create a fresh texture image
            glBindTexture(GL_TEXTURE_2D, gearsTexture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texSize, texSize, 0,
                         GL_RGB, GL_UNSIGNED_BYTE, NULL);
            glBindTexture(GL_TEXTURE_2D, 0);

            // Create a depth buffer
            glGenRenderbuffers(1, &gearsRBO);
            glBindRenderbuffer(GL_RENDERBUFFER, gearsRBO);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16,
                                  texSize, texSize);
            glBindRenderbuffer(GL_RENDERBUFFER, 0);

            // Allocate framebuffer and attach texture and depth buffer
            glGenFramebuffers(1, &gearsFBO);
            glBindFramebuffer(GL_FRAMEBUFFER, gearsFBO);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_2D, gearsTexture, 0);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                      GL_RENDERBUFFER, gearsRBO);

            break;
        #endif // METHOD_FRAMEBUFFER

        #ifdef METHOD_PIXMAP
        case Method_Pixmap:

            // Load the EGLImage functions and make sure they're available
            pImageCreate =
                (PFNEGLCREATEIMAGEKHRPROC)
                    eglGetProcAddress("eglCreateImageKHR");
            pImageDestroy =
                (PFNEGLDESTROYIMAGEKHRPROC)
                    eglGetProcAddress("eglDestroyImageKHR");
            pImageToTex =
                (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)
                    eglGetProcAddress("glEGLImageTargetTexture2DOES");
            if (!pImageCreate || !pImageDestroy || !pImageToTex) {
                NvGlDemoLog("Can't find EGLImageKHR extension functions\n");
                return GL_FALSE;
            }

            // Create the pixmap
            gearsPixmap = NvGlDemoPixmapCreate(texSize, texSize, 32);
            if (gearsPixmap == (EGLNativePixmapType)0) {
                NvGlDemoLog("Couldn't create pixmap\n");
                return GL_FALSE;
            }

            // Create EGL surface for the pixmap
            gearsSurface = eglCreatePixmapSurface(demoState.display,
                                                  gearsConfig,
                                                  gearsPixmap,
                                                  NULL);
            if (gearsSurface == EGL_NO_SURFACE) {
                NvGlDemoLog("Couldn't create EGLSurface for pixmap\n");
                return GL_FALSE;
            }

            // Create EGL image for the pixmap
            gearsImage = pImageCreate(demoState.display,
                                      EGL_NO_CONTEXT,
                                      EGL_NATIVE_PIXMAP_KHR,
                                      (EGLClientBuffer)gearsPixmap,
                                      NULL);
            if (gearsImage == EGL_NO_IMAGE_KHR) {
                NvGlDemoLog("Couldn't create EGLImage for pixmap\n");
                return GL_FALSE;
            }

            // Bind render context and pixmap
            eglMakeCurrent(demoState.display,
                           gearsSurface, gearsSurface,
                           gearsContext);

            // Bind pixmap image to texture
            glBindTexture(GL_TEXTURE_2D, gearsTexture);
            pImageToTex(GL_TEXTURE_2D, gearsImage);
            glBindTexture(GL_TEXTURE_2D, 0);

            break;
        #endif // METHOD_PIXMAP

        #ifdef METHOD_PBUFFER
        case Method_PbufferCopy:

            // Create pbuffer
            gearsSurface = eglCreatePbufferSurface(demoState.display,
                                                   gearsConfig,
                                                   bufAttrs);
            if (gearsSurface == EGL_NO_SURFACE) {
                NvGlDemoLog("Couldn't create pbuffer\n");
                return GL_FALSE;
            }

            // Bind render context and pbuffer
            eglMakeCurrent(demoState.display,
                           gearsSurface, gearsSurface,
                           gearsContext);

            // For copy, create a fresh texture image
            if (method == Method_PbufferCopy) {
                glBindTexture(GL_TEXTURE_2D, gearsTexture);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texSize, texSize, 0,
                             GL_RGB, GL_UNSIGNED_BYTE, NULL);
                glBindTexture(GL_TEXTURE_2D, 0);
            }

            break;
        #endif // METHOD_PBUFFER

        default:
            return GL_FALSE;
            break;
    }

    // Set the viewport to occupy the full render surface
    glViewport(0, 0, texSize, texSize);

    // Initialize the resources needed for gears rendering
    if (!gearsInit(texSize, texSize))
        return GL_FALSE;

    // Check for GL errors
    return (glGetError() == GL_NO_ERROR) ? GL_TRUE : GL_FALSE;
}

// Render a gears frame to the texture
static GLboolean
gearsMethodRender(
    int angle)
{
    // Make context and appropriate surface current
    eglMakeCurrent(demoState.display,
                   gearsSurface, gearsSurface,
                   gearsContext);

    // Render a gears frame
    gearsRender(angle);

    // For pbuffer copy, copy results to texture
    #ifdef METHOD_PBUFFER
    if (method == Method_PbufferCopy) {
        glBindTexture(GL_TEXTURE_2D, gearsTexture);
        glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, texSize, texSize);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    #endif // METHOD_PBUFFER

    return GL_TRUE;
}

// Clean up the gears rendering
static void
gearsMethodTerm(void)
{
    // Make the gears context current
    if (gearsContext != EGL_NO_CONTEXT)
        eglMakeCurrent(demoState.display,
                       demoState.surface, demoState.surface,
                       gearsContext);

    // Clean up the gears resources
    gearsTerm();

    #ifdef METHOD_FRAMEBUFFER
    // Delete the framebuffer and renderbuffer
    if (gearsFBO) {
        glDeleteFramebuffers(1, &gearsFBO);
        gearsFBO = 0;
    }
    if (gearsRBO) {
        glDeleteRenderbuffers(1, &gearsRBO);
        gearsRBO = 0;
    }
    #endif // METHOD_FRAMEBUFFER

    // Clear the context/surface
    eglMakeCurrent(demoState.display,
                   EGL_NO_SURFACE, EGL_NO_SURFACE,
                   EGL_NO_CONTEXT);

    // Delete the gears EGL context
    if (gearsContext != EGL_NO_CONTEXT) {
        eglDestroyContext(demoState.display, gearsContext);
        gearsContext = EGL_NO_CONTEXT;
    }

    // Delete the gears EGL surface if not the main surface
    if (gearsSurface != EGL_NO_SURFACE) {
        if (gearsSurface != demoState.surface)
            eglDestroySurface(demoState.display, gearsSurface);
        gearsSurface = EGL_NO_SURFACE;
    }

    #ifdef METHOD_PIXMAP
    // Delete the pixmap image
    if (gearsImage != EGL_NO_IMAGE_KHR) {
        pImageDestroy(demoState.display, gearsImage);
        gearsImage = EGL_NO_IMAGE_KHR;
    }
    // Delete the pixmap
    if (gearsPixmap != (EGLNativePixmapType)0) {
        NvGlDemoPixmapDelete(gearsPixmap);
        gearsPixmap = (EGLNativePixmapType)0;
    }
    #endif // METHOD_PIXMAP
}

//===========================================================================

// Callback to close window
static void
closeCB(void)
{
    shutdown = GL_TRUE;
}

// Callback to resize window
static void
resizeCB(int width, int height)
{
    cubeViewSet(width, height);
}

// Callback to handle key presses
static void
keyCB(char key, int state)
{
    // Ignoring releases
    if (!state) return;

    if ((key == 'q') || (key == 'Q'))
        shutdown = GL_TRUE;
}

static void
usage(void)
{
    NvGlDemoLog("Usage: gearscube [options]\n"
                "    (negative runTime means \"forever\")\n"
                "  Method to use for render-to-texture:\n"
                "    [-method {"
                        #ifdef METHOD_FRAMEBUFFER
                        " fbo "
                        #endif // METHOD_FRAMEBUFFER
                        #ifdef METHOD_PIXMAP
                        " pixmap "
                        #endif // METHOD_PIXMAP
                        #ifdef METHOD_PBUFFER
                        " pbuffercopy "
                        #endif // METHOD_PBUFFER
                        "}\n"
                "  Texture size:\n"
                "    [-texsize <size>]\n");
    NvGlDemoLog(NvGlDemoArgUsageString());
}

// Entry point of this demo program.
int main(int argc, char **argv)
{
    int         failure = 1;
    char        methodName[NVGLDEMO_MAX_NAME];
    long long   startTime, currTime, endTime;
    int         runforever = 0;
    int         frames     = 0;
    float       angle      = 0.0;

    // Initialize window system and EGL
    if (!NvGlDemoInitialize(&argc, argv, "gearscube", 2, 8, 0)) {
        goto done;
    }

    // Set up PreSwap functions
    NvGlDemoPreSwapInit();

    // Parse non-generic command line options
    while (argc > 1) {

        // Method
        if (NvGlDemoArgMatchStr(&argc, argv, 1, "-method",
                                "{"
                                #ifdef METHOD_FRAMEBUFFER
                                " fbo "
                                #endif // METHOD_FRAMEBUFFER
                                #ifdef METHOD_PIXMAP
                                " pixmap "
                                #endif // METHOD_PIXMAP
                                #ifdef METHOD_PBUFFER
                                " pbuffercopy "
                                #endif // METHOD_PBUFFER
                                "}",
                                NVGLDEMO_MAX_NAME,
                                methodName)) {

            #ifdef METHOD_FRAMEBUFFER
            if (!STRCMP(methodName, "fbo"))
                method = Method_Framebuffer;
            else
            #endif // METHOD_FRAMEBUFFER
            #ifdef METHOD_PIXMAP
            if (!STRCMP(methodName, "pixmap"))
                method = Method_Pixmap;
            else
            #endif // METHOD_PIXMAP
            #ifdef METHOD_PBUFFER
            if (!STRCMP(methodName, "pbuffercopy"))
                method = Method_PbufferCopy;
            else
            #endif // METHOD_PBUFFER
            {
                NvGlDemoLog("Method %s unrecognized/unsupported\n",
                            methodName);
                goto done;
            }
        }

        // Texture size
        else if (NvGlDemoArgMatchInt(&argc, argv, 1, "-texsize",
                                     "<size>", 16, 2048,
                                     1, &texSize)) {
            // No additional action needed
        }

        // Unknown or failure
        else {
            if (!NvGlDemoArgFailed())
                NvGlDemoLog("Unknown command line option (%s)\n", argv[1]);
            goto done;
        }
    }

    // Initialize the cube rendering
    if (!cubeSceneInit(demoState.width, demoState.height))
        goto done;

    // Intialize the gears rendering
    if (!gearsMethodInit())
        goto done;

    // Set up callbacks
    NvGlDemoSetCloseCB(closeCB);
    NvGlDemoSetResizeCB(resizeCB);
    NvGlDemoSetKeyCB(keyCB);

    // Draw a frame. It will cause libraries to load before counting for fps
    gearsMethodRender(angle);
    cubeSceneRender();
    glFinish();

    // Print runtime
    if (demoOptions.duration <= 0.0f) {
        runforever = 1;
        NvGlDemoLog(" running forever...\n");
    } else {
        NvGlDemoLog(" running for %.2f seconds...\n", demoOptions.duration);
    }

    // Get start time and compute end time
    startTime = endTime = currTime = SYSTIME();
    endTime += (long long)(1000000000.0 * demoOptions.duration);

    // Main loop.
    do {
        // Execute PreSwap functions
        NvGlDemoPreSwapExec();

        // Draw and swap a frame
        gearsMethodRender(angle);
        cubeSceneRender();
        if (eglSwapBuffers(demoState.display, demoState.surface) != EGL_TRUE) {
            if (demoState.stream) {
                NvGlDemoLog("Consumer has disconnected, exiting.");
            }
            goto done;
        }

        // Post swap buffers processing
        if (!NvGlDemoPostSwap()) {
            NvGlDemoLog("Failed post swap buffers processing");
            goto done;
        }

        // Process any window system events
        NvGlDemoCheckEvents();

        // Increment frame count, get time and update angle
        ++frames;
        currTime = SYSTIME();
        angle = (float)((120ull * (currTime-startTime) / 1000000000ull) % 360);

        // Check whether time limite has been exceeded
        if (!runforever && !shutdown) shutdown = (endTime <= currTime);
    } while (!shutdown);

    // Success
    failure = 0;

    done:

    // If any frames were generated, print the framerate
    if (frames) {
        NvGlDemoLog("Total FPS: %f\n",
                    (float)frames /(((currTime - startTime) / 1000000ull) / 1000.0));
    }

    // Otherwise something went wrong. Print usage message in case it
    //   was due to bad command line arguments.
    else {
        usage();
    }

    // Clean up rendering resources
    gearsMethodTerm();
    cubeSceneTerm();

    // Clean up PreSwap function resources
    NvGlDemoPreSwapShutdown();

    // Clean up EGL and window system
    NvGlDemoShutdown();

    return failure;
}
