/*
 * Copyright (c) 2018, NVIDIA Corporation.  All Rights Reserved.
 *
 * BY INSTALLING THE SOFTWARE THE USER AGREES TO THE TERMS BELOW.

 *
 * User agrees to use the software under carefully controlled conditions
 * and to inform all employees and contractors who have access to the software
 * that the source code of the software is confidential and proprietary
 * information of NVIDIA and is licensed to user as such.  User acknowledges
 * and agrees that protection of the source code is essential and user shall
 * retain the source code in strict confidence.  User shall restrict access to
 * the source code of the software to those employees and contractors of user
 * who have agreed to be bound by a confidentiality obligation which
 * incorporates the protections and restrictions substantially set forth
 * herein, and who have a need to access the source code in order to carry out
 * the business purpose between NVIDIA and user.  The software provided
 * herewith to user may only be used so long as the software is used solely
 * with NVIDIA products and no other third party products (hardware or
 * software).   The software must carry the NVIDIA copyright notice shown
 * above.  User must not disclose, copy, duplicate, reproduce, modify,
 * publicly display, create derivative works of the software other than as
 * expressly authorized herein.  User must not under any circumstances,
 * distribute or in any way disseminate the information contained in the
 * source code and/or the source code itself to third parties except as
 * expressly agreed to by NVIDIA.  In the event that user discovers any bugs
 * in the software, such bugs must be reported to NVIDIA and any fixes may be
 * inserted into the source code of the software by NVIDIA only.  User shall
 * not modify the source code of the software in any way.  User shall be fully
 * responsible for the conduct of all of its employees, contractors and
 * representatives who may in any way violate these restrictions.
 *
 * NO WARRANTY
 * THE ACCOMPANYING SOFTWARE (INCLUDING OBJECT AND SOURCE CODE) PROVIDED BY
 * NVIDIA TO USER IS PROVIDED "AS IS."  NVIDIA DISCLAIMS ALL WARRANTIES,
 * EXPRESS, IMPLIED OR STATUTORY, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
 * WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.

 * LIMITATION OF LIABILITY
 * NVIDIA SHALL NOT BE LIABLE TO USER, USERS CUSTOMERS, OR ANY OTHER PERSON
 * OR ENTITY CLAIMING THROUGH OR UNDER USER FOR ANY LOSS OF PROFITS, INCOME,
 * SAVINGS, OR ANY OTHER CONSEQUENTIAL, INCIDENTAL, SPECIAL, PUNITIVE, DIRECT
 * OR INDIRECT DAMAGES (WHETHER IN AN ACTION IN CONTRACT, TORT OR BASED ON A
 * WARRANTY), EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGES.  THESE LIMITATIONS SHALL APPLY NOTWITHSTANDING ANY FAILURE OF THE
 * ESSENTIAL PURPOSE OF ANY LIMITED REMEDY.  IN NO EVENT SHALL NVIDIAS
 * AGGREGATE LIABILITY TO USER OR ANY OTHER PERSON OR ENTITY CLAIMING THROUGH
 * OR UNDER USER EXCEED THE AMOUNT OF MONEY ACTUALLY PAID BY USER TO NVIDIA
 * FOR THE SOFTWARE PROVIDED HEREWITH.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include "nvgldemo.h"
#include <time.h>

#if defined (GL_EXT_memory_object)
PFNGLCREATEMEMORYOBJECTSEXTPROC pglCreateMemoryObjectsEXT;
PFNGLTEXSTORAGEMEM2DEXTPROC     pglTexStorageMem2DEXT;
#endif
#if defined (GL_EXT_memory_object_fd)
PFNGLIMPORTMEMORYFDEXTPROC      pglImportMemoryFdEXT ;
#endif

static const char vertexShaderSource[] =
    "attribute vec4 POSITION;\n"
    "attribute vec2 vtxtex;"
    "varying vec2 texCoord;"
    "void main()\n"
    "{\n"
    "    gl_Position = vec4((POSITION.x * 2.0) - 1.0,"
    "                       (POSITION.y * 2.0) - 1.0,"
    "                       0.0, 1.0);\n"
    "    texCoord = vtxtex;"
    "}\n";

static const char fragmentShaderSource[] =
    "precision mediump float;"
    "uniform sampler2D tex;"
    "varying vec2 texCoord;"
    "void main()  {"
    "    gl_FragColor = texture2D(tex, texCoord);"
    "}";

static const GLfloat vert[] = {
    0.2, 0.2,
    0.8, 0.2,
    0.8, 0.8,
    0.2, 0.8};
static const GLfloat tex[] = {
    0, 0,
    1, 0,
    1, 1,
    0, 1};

#define NUM_FRAMES 1000

static char socket_name[64];

static int Connect(void)
{
    int sock;
     struct sockaddr_un client_addr;
    memset(&client_addr, 0, sizeof(client_addr));
    client_addr.sun_family = AF_UNIX;
    strcpy(client_addr.sun_path, socket_name);

    sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) {
        printf("create socket failed\n");
        return sock;
    }

    while (!connect(sock, (struct sockaddr*)&client_addr,
           sizeof(client_addr))) {
        printf("Waiting for connection with vulkan process \n");
        sleep(1);
    }

    return sock;
}

static int receiveFD(int socket)  // receive fd from socket
{
    struct msghdr msg = {0};

    char m_buffer[256];
    struct iovec io = { .iov_base = m_buffer, .iov_len = sizeof(m_buffer) };
    msg.msg_iov = &io;
    msg.msg_iovlen = 1;

    char c_buffer[256];
    msg.msg_control = c_buffer;
    msg.msg_controllen = sizeof(c_buffer);

    if (recvmsg(socket, &msg, 0) < 0) {
        printf("Failed to receive message over socket\n");
    }

    struct cmsghdr * cmsg = CMSG_FIRSTHDR(&msg);

    unsigned char * data = CMSG_DATA(cmsg);
    int fd = *((int*) data);

    return fd;
}

static bool InitGraphicsState(GLuint mem_fd, GLuint mem_size, GLuint tex_w, GLuint tex_h, GLboolean writeTex)
{
    GLuint memory, m_tex;
    glClearColor(0.2, 0.5, 0.9, 1.0);
    GLuint programObj;
    GLuint sampler;
    glGenSamplers(1, &sampler);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, vert);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, tex);
    glEnableVertexAttribArray(1);
    programObj = NvGlDemoLoadShaderSrcStrings(vertexShaderSource, sizeof(vertexShaderSource),
                                              fragmentShaderSource, sizeof(fragmentShaderSource),
                                              0, 0);
    glBindAttribLocation(programObj, 0, "POSITION");
    glBindAttribLocation(programObj, 1, "vtxtex");
    glLinkProgram(programObj);
    glUseProgram(programObj);

    glGenTextures(1, &m_tex);
    glBindTexture(GL_TEXTURE_2D, m_tex);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
#if defined (GL_EXT_memory_object) && defined (GL_EXT_memory_object_fd)
    // Fetch GL extension functions.
    pglImportMemoryFdEXT = (PFNGLIMPORTMEMORYFDEXTPROC)eglGetProcAddress("glImportMemoryFdEXT");
    pglCreateMemoryObjectsEXT = (PFNGLCREATEMEMORYOBJECTSEXTPROC)eglGetProcAddress("glCreateMemoryObjectsEXT");

    pglTexStorageMem2DEXT = (PFNGLTEXSTORAGEMEM2DEXTPROC)eglGetProcAddress("glTexStorageMem2DEXT");
    if(!pglImportMemoryFdEXT  || !pglCreateMemoryObjectsEXT || !pglTexStorageMem2DEXT) {
        printf("Failed to fetch required GL extension functions. Exiting\n");
        return false;
    }
    // Create GL texture using the imported tex memory handle.
    pglCreateMemoryObjectsEXT(1, &memory);
    pglImportMemoryFdEXT(memory, mem_size, GL_HANDLE_TYPE_OPAQUE_FD_EXT, mem_fd);
    pglTexStorageMem2DEXT(GL_TEXTURE_2D, 1, GL_RGBA8,
                          tex_w, tex_h,
                          memory,
                          0);
#else
    printf("Required GL extensions are not supported. Exiting\n");
    return false;
#endif
    if (writeTex) {
        GLubyte texColorData[tex_w*tex_h*4]; //w *h*num_components
        GLuint x, y, index = 0, rnd = 16;
        // Generate random seed value for generating random texture tint.
        srand(time(0));
        rnd = rnd + (int)((220.0*rand())/RAND_MAX);
        // Prepare random color pattern to be uploaded as GL texture data.
        for (y = 0; y < tex_h; y++) {
            for (x = 0; x < tex_w; x++) {
                texColorData[index++] = x * rnd / tex_w;
                texColorData[index++] = y * rnd / tex_h;
                texColorData[index++] = 0xFF - (x * y * 0xFF / tex_w / tex_h);
                texColorData[index++] = rnd;
            }
        }
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, tex_w, tex_h, GL_RGBA, GL_UNSIGNED_BYTE, texColorData);
        glFinish();
    }
    glBindSampler(0, sampler);
    return true;
}

static void Display(void)
{
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    glFinish();
}

static void
usage(void)
{
    NvGlDemoLog("Usage: exttex_import [options]\n"
                "  client-id as an identifier to socket name for connection with Vulkan process:\n"
                "    [--client-id <number 1-16> (Must be provided in case of multiple clients, defaults to 1) \n"
                "  Enable Random Texture data upload in this process:\n"
                "    [--write-tex]\n");
    NvGlDemoLog(NvGlDemoArgUsageString());
}

int main(int argc, char **argv)
{
    GLint imp_fd=-1, mem_size=-1, tex_dim[2] = {-1, -1};
    GLboolean writeTex = GL_FALSE;
    int frames = 0;
    int sock = -1;
    int client_id;
    int startup  = 0;

    // Initialize window system and EGL
    if (!NvGlDemoInitialize(&argc, argv, "exttex_import", 2, 8, 0)) {
        goto done;
    }

    // Parse non-generic command line options
    while (argc > 1) {
        // client-id
        if (NvGlDemoArgMatchInt(&argc, argv, 1, "--client-id",
                                "<number>", 1, 16,
                                1, &client_id)) {
            char buf[4];
            sprintf(buf, "%d", client_id);
            strcpy(socket_name, "/tmp/vksock");
            strcat(socket_name, buf);
            printf("Using socket name %s to connect with Vulkan process\n", socket_name);
        }

        // upload texture data or not
        else if (NvGlDemoArgMatch(&argc, argv, 1, "--write-tex")) {
            writeTex = GL_TRUE;
        }

        // Unknown or failure
        else {
            if (!NvGlDemoArgFailed())
                NvGlDemoLog("Unknown command line option (%s)\n", argv[1]);
            goto done;
        }
    }
    startup = 1;

    sock = Connect();
    if (sock > 0) {
        printf("Successfully connected with a Vulkan process\n");
    } else {
        printf("Failed to connect with Vulkan process. Exiting\n");
        goto done;
    }
    // Import texture memory handle from Vulkan process.
    imp_fd = receiveFD(sock);
    if (imp_fd < 0) {
        printf("Failed to import memory handle. Exiting.\n");
        goto done;
    }
    printf("Imported Memory handle FD = %d\n", imp_fd);

    // Receive texture's dimensions.
    while (!recv(sock, tex_dim, 2*sizeof(GLint), 0));
    if (tex_dim[0] == -1 || tex_dim[1] == -1) {
        printf("Failed to get texture dimensions. Exiting.\n");
        goto done;
    }
    // Receive Texture's backing memory size.
    while (!recv(sock, &mem_size, sizeof(GLint), 0));
    if (mem_size < 0) {
        printf("Failed to get memory size from Vulkan process. Exiting.\n");
        goto done;
    }
    printf("Texture Memory_size = %d, width = %d, height = %d\n", mem_size, tex_dim[0], tex_dim[1]);

    if(!InitGraphicsState(imp_fd, mem_size, tex_dim[0], tex_dim[1], writeTex)) {
        goto done;
    }

    printf("Rendering %d frames using the imported texture...\n", NUM_FRAMES);
    while (frames++ < NUM_FRAMES) {
        glClear(GL_COLOR_BUFFER_BIT);
        Display();
        // Add the fence object in queue and wait accordingly
        if (!NvGlDemoThrottleRendering()) {
            goto done;
        }
        eglSwapBuffers(demoState.display, demoState.surface);
    }

    // Deallocate the resources used by renderahead
    NvGlDemoThrottleShutdown();

    // Clean up EGL and window system
    NvGlDemoShutdown();

done:
    // If basic startup failed, print usage message in case it was due
    //   to bad command line arguments.
    if (!startup) {
        usage();
    }

    close(sock);
    unlink(socket_name);
    return 0;
}
