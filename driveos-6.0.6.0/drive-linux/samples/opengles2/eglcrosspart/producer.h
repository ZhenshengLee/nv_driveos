/*
 * Producer.h
 *
 * Copyright (c) 2015-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef __PRODUCER__
#define __PRODUCER__

#include "utils.h"

/*
 * OpenGL producer arguments
 */
typedef struct {
    EGLStreamKHR    eglStream;   // EGL stream to write to
    int             width;       // Output width
    int             height;      // Output height
    int             latency;     // Stream consumer latency
    GLuint          shaderID;    // Shader
    GLuint          quadVboID;   // Quad used to render output of EGL stream
    GLint           thetaLoc;    // Location of "Theta" parameter in shader
    EGLDisplay      display;     // EGL display
} GLProducerArgs;

bool runProducerProcess(EGLStreamKHR eglstream, EGLDisplay display);
int createProducerPingPongThread();
bool isConsumerAlive();
void resetProducerPingPong();
#endif
