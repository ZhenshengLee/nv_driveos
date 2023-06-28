/*
 * Consumer.cpp
 *
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "consumer.h"

#define DEMO_EGL_STATUS(eglStatus,failureMasg)      \
    do {                                    \
        if (eglStatus != EGL_TRUE) {        \
            NvGlDemoLog(failureMasg);       \
            return eglStatus;               \
        }                                   \
    } while (0)

EXTENSION_LIST(EXTLST_EXTERN)

extern volatile bool sigCaught;

static GLConsumerArgs args;

static const char vshader_con[] = {
    "attribute vec4 Vertex;\n"

    "varying vec2 TexCoord;\n"

    "void main (void) {\n"
    "    gl_Position = vec4(Vertex.xy, 0, 1);\n"
    "    TexCoord = Vertex.zw;\n"
    "}\n"
};

static const char fshader_con[] ={
    "#extension GL_NV_EGL_stream_consumer_external: enable\n"
    "#extension GL_OES_EGL_image_external : enable\n"

    "uniform samplerExternalOES Texture0;\n"

    "varying lowp vec2 TexCoord;\n"

    "void main (void) {\n"
    "    gl_FragColor = texture2D(Texture0, TexCoord);\n"
    "}\n"
};

static void consumerCleanUp()
{
#ifdef EGL_EXT_stream_consumer_qnxscreen_window
    if (demoOptions.eglQnxScreenTest != 1)
#endif /*EGL_EXT_stream_consumer_qnxscreen_window */
    {
        glDeleteProgram(args.shaderID);
        glDeleteTextures(1, &args.videoTexID);
        glDeleteBuffers(1, &args.quadVboID);
    }
}

static EGLBoolean consumerInit()
{
    EGLBoolean eglStatus = EGL_TRUE;

    // pos_x, pos_y, uv_u, uv_v
    float v[24] = {
        -1.0f, -1.0f, 0.0f, 0.0f,
        -1.0f,  1.0f, 0.0f, 1.0f,
         1.0f,  1.0f, 1.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f,  1.0f, 1.0f, 1.0f,
         1.0f, -1.0f, 1.0f, 0.0f,
    };

#if defined(EGL_EXT_stream_consumer_qnxscreen_window) && !defined(ANDROID)
    if (demoOptions.eglQnxScreenTest == 1) {
        EGLAttrib attrib_list[MAX_EGL_STREAM_ATTR * 2 + 1] = {EGL_NONE};
        int attrIdx        = 0;

        attrib_list[attrIdx++] = EGL_CONSUMER_ACQUIRE_QNX_FLUSHING_EXT;
        attrib_list[attrIdx++] = demoOptions.postWindowFlags;

        attrib_list[attrIdx++] = EGL_CONSUMER_ACQUIRE_QNX_DISPNO_EXT;
        attrib_list[attrIdx++] = demoOptions.displayNumber;

        attrib_list[attrIdx++] = EGL_CONSUMER_ACQUIRE_QNX_LAYERNO_EXT;
        attrib_list[attrIdx++] = demoOptions.displayLayer;

        attrib_list[attrIdx++] = EGL_CONSUMER_ACQUIRE_QNX_SURFACE_TYPE_EXT;
        attrib_list[attrIdx++] = demoOptions.surfaceType;

        attrib_list[attrIdx++] = EGL_CONSUMER_ACQUIRE_QNX_DISPLAY_POS_X_EXT;
        attrib_list[attrIdx++] = demoOptions.windowOffset[0];

        attrib_list[attrIdx++] = EGL_CONSUMER_ACQUIRE_QNX_DISPLAY_POS_Y_EXT;
        attrib_list[attrIdx++] = demoOptions.windowOffset[1];

        attrib_list[attrIdx++] = EGL_CONSUMER_ACQUIRE_QNX_DISPLAY_WIDTH_EXT;
        attrib_list[attrIdx++] = demoOptions.displaySize[0];

        attrib_list[attrIdx++] = EGL_CONSUMER_ACQUIRE_QNX_DISPLAY_HEIGHT_EXT;
        attrib_list[attrIdx++] = demoOptions.displaySize[1];

        attrib_list[attrIdx++] = EGL_NONE;

        // Attach ourselves to eglStream as a QNX screen window consumer.
        eglStatus = eglStreamConsumerQNXScreenWindowEXT(args.display,
                                                        args.eglStream,
                                                        attrib_list);

        DEMO_EGL_STATUS(eglStatus,"eglStreamConsumerQNXScreenWindowEXT-failed");
    } else
#endif /*EGL_EXT_stream_consumer_qnxscreen_window && !ANDROID*/
    {
        glGenBuffers(1, &args.quadVboID);
        glBindBuffer(GL_ARRAY_BUFFER, args.quadVboID);
        glBufferData(GL_ARRAY_BUFFER, sizeof(v), v, GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        CHECK_GL_ERROR();

        args.shaderID = utilLoadShaderSrcStrings(vshader_con, sizeof(vshader_con),
                                                 fshader_con, sizeof(fshader_con),
                                                 GL_TRUE, GL_TRUE);

        assert(args.shaderID);
        CHECK_GL_ERROR();

        GLuint vertexLoc  = glGetAttribLocation(args.shaderID, "Vertex");
        GLuint textureLoc = glGetUniformLocation(args.shaderID, "Texture0");

        // Create a GL texture ID
        glGenTextures(1, &args.videoTexID);

        // Bind the GL texture ID to the GL_TEXTURE_EXTERNAL_OES target
        glBindTexture(GL_TEXTURE_EXTERNAL_OES, args.videoTexID);

        // Attach ourselves to eglStream as a texture consumer, using the context
        // bound to the texture unit above.
        eglStatus = eglStreamConsumerGLTextureExternalKHR(args.display,
                                                          args.eglStream);

        DEMO_EGL_STATUS(eglStatus,"eglStreamConsumerGLTextureExternalKHR-failed");

        glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

        glTexParameterf(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameterf(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glUseProgram(args.shaderID);
        glUniform1i(textureLoc, 0);

        glBindBuffer(GL_ARRAY_BUFFER, args.quadVboID);
        glVertexAttribPointer(vertexLoc, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);

        glEnableVertexAttribArray(vertexLoc);

        CHECK_GL_ERROR();

        eglSwapInterval(args.display, 1);
    }

    return eglStatus;
}

static EGLBoolean consumerDraw()
{
    EGLint streamState = 0;
    EGLBoolean eglStatus = EGL_TRUE;

    eglStatus = eglQueryStreamKHR(args.display, args.eglStream,
                                  EGL_STREAM_STATE_KHR, &streamState);

    DEMO_EGL_STATUS(eglStatus,"eglQueryStreamKHR-failed");

    if (streamState == EGL_STREAM_STATE_DISCONNECTED_KHR) {
        NvGlDemoLog("GL Consumer: EGL_STREAM_STATE_DISCONNECTED_KHR received.\n");
        consumerCleanUp();
        return EGL_FALSE;
    }

    // Producer hasn't generated any frames yet
    if ((streamState != EGL_STREAM_STATE_NEW_FRAME_AVAILABLE_KHR) &&
        (streamState != EGL_STREAM_STATE_OLD_FRAME_AVAILABLE_KHR)) {
        return eglStatus;
    }
#if defined(EGL_EXT_stream_consumer_qnxscreen_window) && !defined(ANDROID)
    if (demoOptions.eglQnxScreenTest == 1) {
    // Attach ourselves to eglStream as a QNX screen window consumer.
    eglStatus = eglStreamConsumerAcquireAttribEXT(args.display,
                                                  args.eglStream,
                                                  NULL);

    DEMO_EGL_STATUS(eglStatus,"eglStreamConsumerAcquireEXT-failed");

    EGLuint64KHR frameNum;
    eglStatus = eglQueryStreamu64KHR(args.display, args.eglStream,
                                     EGL_CONSUMER_FRAME_KHR, &frameNum);

    DEMO_EGL_STATUS(eglStatus,"eglQueryStreamu64KHR-failed");
    } else
#endif /*EGL_EXT_stream_consumer_qnxscreen_window && !ANDROID*/
    {
        glClearColor(1.0f, 1.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        eglStatus = eglStreamConsumerAcquireKHR(args.display,
                                                args.eglStream);

        DEMO_EGL_STATUS(eglStatus,"eglStreamConsumerAcquireKHR-failed");

        EGLuint64KHR frameNum;
        eglStatus = eglQueryStreamu64KHR(args.display, args.eglStream,
                                        EGL_CONSUMER_FRAME_KHR, &frameNum);

        DEMO_EGL_STATUS(eglStatus,"eglQueryStreamu64KHR-failed");

        glDrawArrays(GL_TRIANGLES, 0, 6);

        eglStatus = eglStreamConsumerReleaseKHR(args.display,
                                                args.eglStream);

        DEMO_EGL_STATUS(eglStatus,"eglStreamConsumerReleaseKHR-failed");

        if (!eglSwapBuffers(demoState.display, demoState.surface)) {
            NvGlDemoLog("\eglSwapBuffers-failed!\n");
            return EGL_FALSE;
        }
    }

    return eglStatus;
}

static bool createAndRunConsumer()
{
    bool timeOver = false;
    // Determine end time in nanoseconds
    const float endTime = SYSTIME() + (demoOptions.duration * 1000000000);
    // Set up PreSwap
    NvGlDemoPreSwapInit();

    // Initialize the consumer
    if (consumerInit() != EGL_TRUE) {
        NvGlDemoLog("\nconsumerInit-failed!\n");
        exit(0);
    }

    EGLBoolean eglStatus = EGL_TRUE;

    while (eglStatus == EGL_TRUE && !timeOver && !sigCaught &&
           ((demoOptions.isSmart != 1) || isProducerAlive())) {
        NvGlDemoPreSwapExec();
        eglStatus = consumerDraw();

        // Check to verify if the flip was successful before calling post swap
        if (eglStatus == EGL_TRUE) {
            if (!NvGlDemoPostSwap()) {
                NvGlDemoLog("Failed post swap buffers processing");
                exit(0);
            }
        }

        // If a duration is specified allow an exit after it is finished
        if (demoOptions.duration > 0.0f && SYSTIME() >= endTime) {
            timeOver = true;
            NvGlDemoLog("Consumer duration has elapsed.\n");
        }
    }
    printf("Drawing finished\n");
    consumerCleanUp();
    NvGlDemoPreSwapShutdown();

    return !(timeOver || sigCaught);
}

static bool runGLTextureConsumer(EGLStreamKHR eglStream, EGLDisplay display)
{
    bool ret = true;
    NvGlDemoLog("\nStarting GL Texture Consumer process !\n");

    args.eglStream = eglStream;
    args.display   = display;

    // Create and run consumer
    ret = createAndRunConsumer();

    // delete stream.
    eglDestroyStreamKHR(display, eglStream);

    return ret;
}

static bool runEGLOutputConsumer(EGLStreamKHR eglStream, EGLDisplay display)
{
    bool timeOver = false;
    // Determine end time in nanoseconds
    const float endTime = SYSTIME() + (demoOptions.duration * 1000000000);
    EGLint streamState;
    EGLBoolean eglStatus = EGL_TRUE;

    // (Don't) handle inactivity for EGLStream consumption.
    if (demoOptions.inactivityTime > 0.0f) {
        NvGlDemoLog("Inactivity testing is not available for EGLStream consumers.");
    }

    NvGlDemoLog("\nStarting EGLOutput Consumer process !\n");

    args.eglStream = eglStream;
    args.display   = display;

    // Run consumer
    while (eglStatus == EGL_TRUE && !timeOver && !sigCaught &&
           ((demoOptions.isSmart != 1) || isProducerAlive())) {
        streamState = 0;
        eglStatus = eglQueryStreamKHR(args.display, args.eglStream,
                                      EGL_STREAM_STATE_KHR, &streamState);

        DEMO_EGL_STATUS(eglStatus,"eglQueryStreamKHR-failed");

        if (streamState == EGL_STREAM_STATE_DISCONNECTED_KHR) {
            NvGlDemoLog("EGLOutput Consumer: EGL_STREAM_STATE_DISCONNECTED_KHR received.\n");
            eglStatus = EGL_FALSE;
        }

        // If a duration is specified allow an exit after it is finished
        if (demoOptions.duration > 0.0f && SYSTIME() >= endTime) {
            timeOver = true;
            NvGlDemoLog("Consumer duration has elapsed.\n");
        }
    }

    // delete stream.
    eglDestroyStreamKHR(display, eglStream);
    return !(timeOver || sigCaught);
}

bool runConsumerProcess(EGLStreamKHR eglStream, EGLDisplay display)
{
    if (demoState.platformType == NvGlDemoInterface_Device ||
        demoState.platformType == NvGlDemoInterface_DRM ||
        demoState.platformType == NvGlDemoInterface_WF) {
        return runEGLOutputConsumer(demoState.stream, demoState.display);
    }
    else {
        return runGLTextureConsumer(demoState.stream, demoState.display);
    }
}
