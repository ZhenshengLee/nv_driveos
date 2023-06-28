/*
 * Consumer.h
 *
 * Copyright (c) 2015-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef __CONSUMER__
#define __CONSUMER__

#include "utils.h"

/*
 * OpenGL consumer arguments
 */
typedef struct {
    int              width;       // Output width
    int              height;      // Output height
    GLuint           shaderID;    // Shader
    GLuint           quadVboID;   // Quad used to render output of EGL stream
    GLuint           videoTexID;  // EGL stream texture
    EGLStreamKHR     eglStream;   // EGL stream to read from
    EGLDisplay       display;     // EGL display
    int              latency;     // Stream consumer latency
} GLConsumerArgs;

typedef struct {
    int              width;       // Output width
    int              height;      // Output height
    EGLStreamKHR     eglStream;   // EGL stream to read from
    EGLDisplay       display;     // EGL display
} EGLOutputConsumerArgs;

bool runConsumerProcess(EGLStreamKHR eglstream, EGLDisplay display);
int createConsumerPingPongThread();
bool isProducerAlive();
void resetConsumerPingPong();

#endif
