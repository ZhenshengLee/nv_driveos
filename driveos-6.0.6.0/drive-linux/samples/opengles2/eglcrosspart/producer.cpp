/*
 * Producer.cpp
 *
 * Copyright (c) 2015-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "producer.h"

EXTENSION_LIST(EXTLST_EXTERN)

extern volatile bool sigCaught;

static const char vshader_prod[] = {
    "attribute vec4 Vertex;\n"
    "uniform highp float Theta;\n"

    "void main (void) {\n"
    "    mat4 r = mat4(1.0);   \n"
    "    mat4 s = mat4(1.0);   \n"
    "    s[0][0] = sin(Theta);  \n"
    "    s[1][1] = sin(Theta);  \n"
    "    r[0][0] = cos(Theta);  \n"
    "    r[0][2] = sin(Theta);  \n"
    "    r[2][0] = -sin(Theta); \n"
    "    r[2][2] = cos(Theta);  \n"
    "    gl_Position = r * s * vec4(Vertex.xy, 0, 1);\n"
    "}\n"
};

static const char fshader_prod[] = {
    "uniform highp float Theta;\n"

    "void main (void) {\n"
    "    gl_FragColor = vec4(sin(Theta),\n"
    "                        cos(Theta),\n"
    "                        sin(Theta),\n"
    "                        1) / 2.0 + 0.5;\n"
    "}\n"
};

static GLProducerArgs args;

static void producerCleanUp()
{
    glDeleteProgram(args.shaderID);
    glDeleteBuffers(1, &args.quadVboID);
    NvGlDemoThrottleShutdown();
}

static bool producerInit()
{
    NvGlDemoLog("\nInitializing producer!");

    // pos_x, pos_y, uv_u, uv_v
    float v[12] = {
        -1.0f, -1.0f, 0.0f, 0.0f,
         0.0f,  1.0f, 0.0f, 1.0f,
         1.0f, -1.0f, 1.0f, 1.0f,
    };

    glGenBuffers(1, &args.quadVboID);
    glBindBuffer(GL_ARRAY_BUFFER, args.quadVboID);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)sizeof(v), v, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    CHECK_GL_ERROR();

    args.shaderID = utilLoadShaderSrcStrings(vshader_prod, sizeof(vshader_prod),
                                            fshader_prod, sizeof(fshader_prod),
                                            GL_TRUE, GL_TRUE);

    assert(args.shaderID);
    CHECK_GL_ERROR();

    GLuint vertexLoc = glGetAttribLocation(args.shaderID, "Vertex");
    args.thetaLoc = glGetUniformLocation(args.shaderID, "Theta");

    glUseProgram(args.shaderID);

    glBindBuffer(GL_ARRAY_BUFFER, args.quadVboID);
    glVertexAttribPointer(vertexLoc, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(vertexLoc);

    CHECK_GL_ERROR();

    // Allocate the resource if renderahead is specified
    if (!NvGlDemoThrottleInit()) {
        return false;
    }

    return true;
}

static void fpsCount()
{
    static unsigned frames = 0;
    static long long frame_time = 0;
    long long t = 0;
    static unsigned long usec = 0;

    t = SYSTIME();
    usec += frames++ ? (t - frame_time)/1000 : 0;
    frame_time = t;
    if (usec >= 5000000) {
        unsigned fps = ((unsigned long long)frames * 10000000 + usec - 1) / usec;
        printf("FPS: %3u.%1u\n", fps / 10, fps % 10);
        usec = 0;
        frames = 0;
    }
}

static bool producerDraw()
{
    static float theta = 0.0f;
    EGLint streamState = 0;
    EGLBoolean eglStatus;
    ECODE status;

    eglStatus = eglQueryStreamKHR(args.display, args.eglStream,
                                  EGL_STREAM_STATE_KHR, &streamState);
    assert(eglStatus);

    if (streamState == EGL_STREAM_STATE_DISCONNECTED_KHR) {
        NvGlDemoLog("GL Producer: EGL_STREAM_STATE_DISCONNECTED_KHR received\n");
        return false;
    }

    glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUniform1f(args.thetaLoc, theta);
    theta += 0.1f;
    if (theta >= 360.0f) {
        theta = 0.0f;
    }

    CHECK_GL_ERROR();

    glDrawArrays(GL_TRIANGLES, 0, 3);

    // Add the fence object in queue and wait accordingly
    if (!NvGlDemoThrottleRendering()) {
         return false;
    }

    if (!eglSwapBuffers(demoState.display, demoState.surface)) {
        NvGlDemoLog("Producer: EGL swap buffers failed with code: 0x%x\n",
                     eglGetError());
        return false;
    }

    // For silencing release builds
    UNUSED_VAR(status);
    UNUSED_VAR(eglStatus);

    fpsCount();
    return true;
}

static bool createAndRunProducer()
{
    bool status;
    bool timeOver = false;
    // Determine end time in nanoseconds
    const float endTime = SYSTIME() + (demoOptions.duration * 1000000000);

    NvGlDemoPreSwapInit();

    // Initialize the producer.
    status = producerInit();

    while (status && !timeOver && !sigCaught &&
           ((demoOptions.isSmart != 1) || isConsumerAlive())) {
        NvGlDemoPreSwapExec();
        status = producerDraw();

        // If a duration is specified allow an exit after it is finished
        if (demoOptions.duration > 0.0f && SYSTIME() >= endTime) {
            timeOver = true;
            NvGlDemoLog("Producer duration has elapsed.\n");
        }
    }

    producerCleanUp();

    NvGlDemoPreSwapShutdown();

    return !(timeOver || sigCaught);
}

bool runProducerProcess(EGLStreamKHR eglStream, EGLDisplay display)
{
    bool ret = true;
    NvGlDemoLog("\nStarting producer process!\n");

    args.eglStream = eglStream;
    args.display   = display;

    // Create a producer. It will connect to the EGL stream.
    ret = createAndRunProducer();

    // delete stream
    eglDestroyStreamKHR(display, eglStream);
    return ret;
}

