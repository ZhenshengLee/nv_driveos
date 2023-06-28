/*
 * eglstreamcube.c
 *
 * Copyright (c) 2013-2019, NVIDIA CORPORATION. All rights reserved.
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

// This file illustrates reading textures from EGL streams and then mapping
//   them to the faces of a spinning cube.

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include "nvgldemo.h"
#include <unistd.h>

#if defined (__INTEGRITY)
#include <sys/uio.h>
#endif

#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <poll.h>
#include <errno.h>
#include <signal.h>
#include <sys/time.h>

// Depending on compile options, we either build in the shader sources or
//   binaries or load them from external data files at runtime.
//   (Variables are initialized to the file contents or name, respectively).
#ifdef USE_EXTERN_SHADERS
static const char cubeVertShader[] = { VERTFILE(cube_vert) };
static const char cubeFragShader[] = { FRAGFILE(cube_frag) };
static const char logoFragShader[] = { FRAGFILE(logo_frag) };
#else
static const char cubeVertShader[] = {
#   include VERTFILE(cube_vert)
};
static const char cubeFragShader[] = {
#   include FRAGFILE(cube_frag)
};
static const char logoFragShader[] = {
#   include FRAGFILE(logo_frag)
};
#endif

static const char eglstreamcubePrgBin[] = { PROGFILE(cube_prog) };
static const char eglstreamlogoPrgBin[] = { PROGFILE(logo_prog) };

static const char logoData[] = {
#include "nvidia.h"
};

// Flag indicating it is time to shut down
static GLboolean demoShutdown = GL_FALSE;
static GLboolean TexInit = GL_FALSE;

// Extensions used by this demo
#define EXTENSION_LIST(T) \
    T( PFNEGLQUERYSTREAMKHRPROC,           eglQueryStreamKHR ) \
    T( PFNEGLQUERYSTREAMU64KHRPROC,        eglQueryStreamu64KHR ) \
    T( PFNEGLSTREAMCONSUMERACQUIREKHRPROC, eglStreamConsumerAcquireKHR ) \
    T( PFNEGLSTREAMCONSUMERRELEASEKHRPROC, eglStreamConsumerReleaseKHR ) \
    T( PFNEGLCREATESTREAMKHRPROC,          eglCreateStreamKHR ) \
    T( PFNEGLDESTROYSTREAMKHRPROC,         eglDestroyStreamKHR ) \
    T( PFNEGLSTREAMCONSUMERGLTEXTUREEXTERNALKHRPROC, \
                        eglStreamConsumerGLTextureExternalKHR ) \
    T( PFNEGLGETSTREAMFILEDESCRIPTORKHRPROC, \
                        eglGetStreamFileDescriptorKHR )

#define EXTLST_DECL(tx, x) static tx x = NULL;
#define EXTLST_ENTRY(tx, x) { (extlst_fnptr_t *)&x, #x },

EXTENSION_LIST(EXTLST_DECL)
typedef void (*extlst_fnptr_t)(void);
static const struct {
    extlst_fnptr_t *fnptr;
    char const *name;
} extensionList[] = { EXTENSION_LIST(EXTLST_ENTRY) };

// Socket information for listening for client connections
static char socketName[NVGLDEMO_MAX_NAME] = "/tmp/nvidia.eglstreamdemo";
static int socketFD = -1;

// Gears rendering context/surfaces
#define CUBE_MAX_CLIENTS 6
typedef struct ClientState
{
    EGLStreamKHR    stream;
    EGLNativeFileDescriptorKHR fd;
    GLuint          texture;
} ClientState;

ClientState clientList[CUBE_MAX_CLIENTS];

#define CLIENT_NO_TEXTURE (GLuint)0
GLuint clientTexturePool[CUBE_MAX_CLIENTS];

// Cube shader
GLint           prog_cube       = 0;
GLint           uloc_cubeCameraMat;
GLint           uloc_cubeObjectMat;
GLint           uloc_cubeTexUnit;

// Fallback shader for unattached client faces
GLint           prog_logo       = 0;
GLint           uloc_logoCameraMat;
GLint           uloc_logoObjectMat;
GLint           uloc_logoTexUnit;
GLuint          logoTexture     = 0;

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

    { -1.0f, +1.0f, -1.0f,  -1.1f, -1.1f },
    { -1.0f, -1.0f, -1.0f,  +1.1f, -1.1f },
    { -1.0f, +1.0f, +1.0f,  -1.1f, +1.1f },
    { -1.0f, -1.0f, +1.0f,  +1.1f, +1.1f },

    { +1.0f, +1.0f, -1.0f,  +1.1f, -1.1f },
    { -1.0f, +1.0f, -1.0f,  -1.1f, -1.1f },
    { +1.0f, -1.0f, -1.0f,  +1.1f, +1.1f },
    { -1.0f, -1.0f, -1.0f,  -1.1f, +1.1f },

    { +1.0f, -1.0f, +1.0f,  +1.1f, -1.1f },
    { +1.0f, +1.0f, +1.0f,  -1.1f, -1.1f },
    { +1.0f, -1.0f, -1.0f,  +1.1f, +1.1f },
    { +1.0f, +1.0f, -1.0f,  -1.1f, +1.1f },

    { -1.0f, +1.0f, +1.0f,  +1.1f, -1.1f },
    { -1.0f, +1.0f, -1.0f,  -1.1f, -1.1f },
    { +1.0f, +1.0f, +1.0f,  +1.1f, +1.1f },
    { +1.0f, +1.0f, -1.0f,  -1.1f, +1.1f },

    { +1.0f, -1.0f, -1.0f,  -1.1f, -1.1f },
    { +1.0f, -1.0f, +1.0f,  +1.1f, -1.1f },
    { -1.0f, -1.0f, -1.0f,  -1.1f, +1.1f },
    { -1.0f, -1.0f, +1.0f,  +1.1f, +1.1f }

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
    glUseProgram(prog_cube);
    glUniformMatrix4fv(uloc_cubeCameraMat, 1, GL_FALSE, matrix);
    glUseProgram(prog_logo);
    glUniformMatrix4fv(uloc_logoCameraMat, 1, GL_FALSE, matrix);

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
    prog_cube = LOADPROGSHADER(cubeVertShader, cubeFragShader, GL_TRUE, GL_FALSE, eglstreamcubePrgBin);
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

    // Set up texture to be used for the clients
    glUniform1i(uloc_cubeTexUnit, 0);

    /* TODO: should we bind something first? */
    glTexParameterf(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);


    /* Once again, for the logo texture when nothing is attached. */

    // Load the shaders (The macro handles the details of binary vs.
    //   source and external vs. internal)
    prog_logo = LOADPROGSHADER(cubeVertShader, logoFragShader, GL_TRUE, GL_FALSE, eglstreamlogoPrgBin);
    if (!prog_logo) return GL_FALSE;
    glUseProgram(prog_logo);

    // Extract uniform locations
    uloc_logoCameraMat = glGetUniformLocation(prog_logo, "cameramat");
    uloc_logoObjectMat = glGetUniformLocation(prog_logo, "objectmat");
    uloc_logoTexUnit   = glGetUniformLocation(prog_logo, "texunit");

    // Set and enable default coordinates
    aloc = glGetAttribLocation(prog_logo, "vtxpos");
    glVertexAttribPointer(aloc, 3, GL_FLOAT, GL_FALSE, 5*sizeof(GLfloat),
                          &cubeVert[0][0]);
    glEnableVertexAttribArray(aloc);
    aloc = glGetAttribLocation(prog_logo, "vtxtex");
    glVertexAttribPointer(aloc, 2, GL_FLOAT, GL_FALSE, 5*sizeof(GLfloat),
                          &cubeVert[0][3]);
    glEnableVertexAttribArray(aloc);

    // Set up texture to be used for the logo
    glUniform1i(uloc_logoTexUnit, 0);
    glGenTextures(1, &logoTexture);
    glBindTexture(GL_TEXTURE_2D, logoTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 420, 420, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, logoData);
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

    // Rotate the cube and set up object transformation matrix
    angA += 0.175f;
    angB += 0.050f;
    angA = fmod(angA, 360.0);
    angB = fmod(angB, 360.0);
    NvGlDemoMatrixIdentity(matrix);
    NvGlDemoMatrixTranslate(matrix, 0.0f, 0.0f, -30.0f);
    NvGlDemoMatrixRotate(matrix, angA, 0.6f, 0.8f, 0.0f);
    NvGlDemoMatrixRotate(matrix, angB, 0.0f, 1.0f, 1.0f);
    NvGlDemoMatrixScale(matrix, 3.0f, 3.0f, 3.0f);
    glUseProgram(prog_cube);
    glUniformMatrix4fv(uloc_cubeObjectMat, 1, GL_FALSE, matrix);
    glUseProgram(prog_logo);
    glUniformMatrix4fv(uloc_logoObjectMat, 1, GL_FALSE, matrix);

    // Draw each face of the cube
    for (i=0; i<CUBE_MAX_CLIENTS; ++i) {
        if (clientList[i].texture != CLIENT_NO_TEXTURE) {
                glUseProgram(prog_cube);
                glBindTexture(GL_TEXTURE_EXTERNAL_OES, clientList[i].texture);
        }
        else {
            glUseProgram(prog_logo);
            glBindTexture(GL_TEXTURE_2D, logoTexture);
        }
        glDrawArrays(GL_TRIANGLE_STRIP, 4*i, 4);
    }

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

    if (logoTexture) {
        glDeleteTextures(1, &logoTexture);
        logoTexture = 0;
    }

    // Delete shaders
    if (prog_logo) {
        glDeleteProgram(prog_logo);
        prog_logo = 0;
    }
    if (prog_cube) {
        glDeleteProgram(prog_cube);
        prog_cube = 0;
    }
}

//===========================================================================

static GLboolean
deadClient(ClientState *client)
{
    client->texture = CLIENT_NO_TEXTURE;
    if (client->stream != EGL_NO_STREAM_KHR) {
        EGLint state = 0;
        eglQueryStreamKHR(demoState.display, client->stream,
                          EGL_STREAM_STATE_KHR, &state);
        // Release the acquired frame if the stream hasn't disconnected yet.
        if (state != EGL_STREAM_STATE_DISCONNECTED_KHR) {
            if (!eglStreamConsumerReleaseKHR(demoState.display,
                                              client->stream))
                NvGlDemoLog("Release frame failed.\n");
        }
        if (!eglDestroyStreamKHR(demoState.display, client->stream))
            NvGlDemoLog("Couldn't destroy EGL stream\n");
        client->stream = EGL_NO_STREAM_KHR;

        /* client->fd can only be valid if stream was valid */
        if (client->fd != EGL_NO_FILE_DESCRIPTOR_KHR) {
            close(client->fd);
            client->fd = EGL_NO_FILE_DESCRIPTOR_KHR;
        }
    }

    return GL_TRUE;
}


static GLboolean
newClient(int socketFD, ClientState *client, GLuint texture)
{
    EGLint streamAttr[7] = { EGL_STREAM_FIFO_LENGTH_KHR, 4, EGL_NONE };
    int numAttrs= 0;
    struct sockaddr_un conn_addr = { 0 };
    struct msghdr msg = { 0 };
    struct cmsghdr *cmsg;
    struct iovec iov;
    union { char buf[CMSG_SPACE(sizeof(int))]; long align; } ctl;
    socklen_t conn_addr_len = sizeof(conn_addr);
    int fd = -1;

    fd = accept(socketFD, (struct sockaddr *)&conn_addr, &conn_addr_len);
    if (fd == -1) {
        NvGlDemoLog("Couldn't accept connection\n");
        goto fail;
    }

    if (demoOptions.nFifo > 0) {
        streamAttr[numAttrs++] = EGL_STREAM_FIFO_LENGTH_KHR;
        streamAttr[numAttrs++] = demoOptions.nFifo;
    }

    if (demoOptions.flags & NVGL_DEMO_OPTION_TIMEOUT) {
        streamAttr[numAttrs++] = EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR;
        streamAttr[numAttrs++] = demoOptions.timeout;
    }

    if (demoOptions.latency > 0) {
        streamAttr[numAttrs++] = EGL_CONSUMER_LATENCY_USEC_KHR;
        streamAttr[numAttrs++] = demoOptions.latency;
    }
    streamAttr[numAttrs++] = EGL_NONE;

    client->stream = eglCreateStreamKHR(demoState.display, streamAttr);
    if (client->stream == EGL_NO_STREAM_KHR) {
        NvGlDemoLog("Couldn't create EGL stream.\n");
        goto fail;
    }

    client->fd = eglGetStreamFileDescriptorKHR(demoState.display, client->stream);
    if (client->fd == EGL_NO_FILE_DESCRIPTOR_KHR) {
        NvGlDemoLog("Couldn't get stream file descriptor.\n");
        goto fail;
    }

    glBindTexture(GL_TEXTURE_EXTERNAL_OES, texture);
    if (!eglStreamConsumerGLTextureExternalKHR(demoState.display, client->stream)) {
        NvGlDemoLog("Couldn't bind texture.\n");
        goto fail;
    }

    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    iov.iov_base = "x";
    iov.iov_len = 1;
    msg.msg_control = ctl.buf;
    msg.msg_controllen = sizeof(ctl.buf);
    cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    MEMCPY(CMSG_DATA(cmsg), &client->fd, sizeof(int));
    msg.msg_controllen = cmsg->cmsg_len;

    if (sendmsg(fd, &msg, 0) <= 0) {
        NvGlDemoLog("Couldn't send fd to client.\n");
        goto fail;
    }

    (void)close(fd);

    NvGlDemoLog("New connection on fd %d.\n", client->fd);

    return GL_TRUE;

fail:
    if (fd != -1)
        close(fd);
    deadClient(client);
    return GL_FALSE;
}


static GLboolean
clientMethodPoll(void)
{
    unsigned char i;

    for (i = 0; i < CUBE_MAX_CLIENTS; i++) {
        clientList[i].texture = CLIENT_NO_TEXTURE;

        /* for any free faces, service pending connections */
        if (clientList[i].stream == EGL_NO_STREAM_KHR) {
            struct pollfd fds = { socketFD, POLLIN };

            if (poll(&fds, 1, 0) > 0) {
                if (!newClient(socketFD, &clientList[i], clientTexturePool[i]))
                    return GL_FALSE;
            }
        }

        if (clientList[i].stream != EGL_NO_STREAM_KHR) {
            if (!eglStreamConsumerAcquireKHR(demoState.display,
                                             clientList[i].stream)) {
                EGLint state = 0;
                eglQueryStreamKHR(demoState.display, clientList[i].stream,
                                  EGL_STREAM_STATE_KHR, &state);

                /* Check why acquire failed */
                switch (state)
                {
                case EGL_STREAM_STATE_DISCONNECTED_KHR:
                    NvGlDemoLog("Lost connection with fd %d.\n", clientList[i].fd);
                    deadClient(&clientList[i]);
                    break;
                case EGL_STREAM_STATE_EMPTY_KHR:
                    break;
                case EGL_STREAM_STATE_CONNECTING_KHR:
                    break;
                case EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR:
                    /* New frame became available now */
                    break;
                default:
                    NvGlDemoLog("Unexpected stream state: %04x.\n", state);
                    return GL_FALSE;
                }
            }
            else {
                /* We have a valid texture this time around. */
                clientList[i].texture = clientTexturePool[i];

            }
        }
    }

    return GL_TRUE;
}

// Initialize the client rendering methods
static GLboolean
clientMethodInit(void)
{
    GLuint i;

    memset(clientTexturePool, 0, sizeof(clientTexturePool));

    for (i = 0; i < (sizeof(extensionList) / sizeof(*extensionList)); i++) {
        *extensionList[i].fnptr = eglGetProcAddress(extensionList[i].name);
        if (*extensionList[i].fnptr == NULL) {
            NvGlDemoLog("Couldn't get address of %s().\n", extensionList[i].name);
            goto fail;
        }
    }

    socketFD = socket(PF_UNIX, SOCK_STREAM, 0);
    if (socketFD == -1) {
        NvGlDemoLog("Couldn't create socket.\n");
        goto fail;
    }
    unlink(socketName);

    {
        struct sockaddr_un sock_addr = { 0 };
        sock_addr.sun_family = AF_UNIX;
        size_t strLen = strlen(socketName);
        if (strLen >= sizeof(sock_addr.sun_path) - 1 )
                strLen = sizeof(sock_addr.sun_path) - 1;
        memcpy(sock_addr.sun_path, (const char*)socketName, strLen);
        if (bind(socketFD, (const struct sockaddr *)&sock_addr, sizeof(sock_addr))) {
            NvGlDemoLog("Couldn't bind socket name \"%s\".", socketName);
            goto fail;
        }
    }

    if (listen(socketFD, CUBE_MAX_CLIENTS)) {
        NvGlDemoLog("Couldn't listen on socket.\n");
        goto fail;
    }

    glGenTextures(CUBE_MAX_CLIENTS, clientTexturePool);
    TexInit = (glGetError() == GL_NO_ERROR);

    // Check for GL errors
    return (TexInit) ? GL_TRUE : GL_FALSE;

fail:
    if (socketFD != -1)
        (void)close(socketFD);
    socketFD = -1;
    return GL_FALSE;
}

// Clean up the client state
static void
clientMethodTerm(void)
{
    int i;
    if (socketFD != -1) {
        close(socketFD);
        socketFD = -1;
        unlink(socketName);
    }

    for (i = 0; i < CUBE_MAX_CLIENTS; i++)
        deadClient(&clientList[i]);

    if(TexInit)
        glDeleteTextures(CUBE_MAX_CLIENTS, clientTexturePool);
}

//===========================================================================

// Callback to close window
static void
closeCB(void)
{
    demoShutdown = GL_TRUE;
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
        demoShutdown = GL_TRUE;
}

static void (*oldhandler)(int) = SIG_ERR;
static void
inthandler(int sig)
{
    if (sig == SIGINT) {
        demoShutdown = GL_TRUE;
        if (oldhandler != SIG_ERR)
            signal(sig, oldhandler);
    }
}

static void
usage(void)
{
    NvGlDemoLog("Usage: eglstreamcube [options] [command] [command options]\n"
                "  Listen on <socket name>:\n"
                "    [-socket <socket name>]\n");
    NvGlDemoLog(NvGlDemoArgUsageString());
}

// Entry point of this demo program.
int main(int argc, char **argv)
{
    int         failure = 1;
    long long   startTime, currTime, endTime;
    int         runforever = 0;
    int         frames     = 0;

    if (!NvGlDemoArgParse(&argc, argv)) {
        goto done;
    }

    while (argc > 1) {
        // name of socket on which to listen for connections
        if (NvGlDemoArgMatchStr(&argc, argv, 1, "-socket",
                                "<socket name>",
                                sizeof(socketName),
                                socketName)) {
            // No additional action needed
        }

        // Unknown or failure
        else {
            if (!NvGlDemoArgFailed())
                NvGlDemoLog("Unknown command line option (%s)\n", argv[1]);
            goto done;
        }
    }

    // Initialize window system and EGL
    if (!NvGlDemoInitializeParsed(&argc, argv, "eglstreamcube", 2, 8, 0)) {
        goto done;
    }

    // Initialize the cube rendering
    if (!cubeSceneInit(demoState.width, demoState.height))
        goto done;

    // Intialize the client rendering
    if (!clientMethodInit())
        goto done;

    // Set up PreSwap functions
    NvGlDemoPreSwapInit();

    // Set up callbacks
    NvGlDemoSetCloseCB(closeCB);
    NvGlDemoSetResizeCB(resizeCB);
    NvGlDemoSetKeyCB(keyCB);

    // Print runtime
    if (demoOptions.duration <= 0.0) {
        runforever = 1;
        NvGlDemoLog(" running forever...\n");
    } else {
        NvGlDemoLog(" running for %f seconds...\n", demoOptions.duration);
    }

    // Get start time and compute end time
    startTime = endTime = currTime = SYSTIME();
    endTime += (long long)(1000000000.0 * demoOptions.duration);

    // Trap ^C so we can try to do a tidy shutdown.
    // (Use ^\ if you want an untidy one.)
    oldhandler = signal(SIGINT, inthandler);

    // Main loop.
    do {
        // Client updates
        if (!clientMethodPoll()) {
            NvGlDemoLog("Failure within clientMethodPoll()\n");
            demoShutdown = GL_TRUE;
        }

        // Execute PreSwap functions
        NvGlDemoPreSwapExec();

        // Draw and swap a frame
        if (!cubeSceneRender()) {
            NvGlDemoLog("Failure within cubeSceneRender()\n");
            demoShutdown = GL_TRUE;
        }
        if (eglSwapBuffers(demoState.display, demoState.surface) != EGL_TRUE) {
            if (demoState.stream) {
                NvGlDemoLog("Consumer has disconnected, exiting.");
            }
            goto done;
        }

        if (!NvGlDemoPostSwap()) {
            NvGlDemoLog("Failed post swap buffers processing");
            goto done;
        }

        // Process any window system events
        NvGlDemoCheckEvents();

        // Increment frame count and get time
        ++frames;
        currTime = SYSTIME();

        // Check whether time limite has been exceeded
        if (!runforever && !demoShutdown) demoShutdown = (endTime <= currTime);
    } while (!demoShutdown);

    // Restore original ^C handler
    signal(SIGINT, oldhandler);

    // Success
    failure = 0;

    done:

    // If any frames were generated, print the framerate
    if (frames) {
        NvGlDemoLog("Total FPS: %f\n",
                    (float)frames / ((currTime - startTime) / 1000000000ull));
    }

    // Otherwise something went wrong. Print usage message in case it
    //   was due to bad command line arguments.
    else {
        usage();
    }

    // Clean up rendering resources
    clientMethodTerm();
    cubeSceneTerm();

    // Clean up PreSwap functions
    NvGlDemoPreSwapShutdown();

    // Clean up EGL and window system
    NvGlDemoShutdown();

    return failure;
}
