/*
 * nvgldemo_parse.c
 *
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

#include "nvgldemo.h"
#include <unistd.h>

static long long sleepTime;
static long long sleepInterval;
static GLsync *syncobjarr;

//
// This file contains utility functions which a producer is expected to run
// during every frame of image production.

//
// Wrapper per-frame function
//

// Set up the various functions that are expected to run per-frame
// A non-zero return indicates success.
int
NvGlDemoPreSwapInit(void) {
    // Allocate the resource if renderahead is specified
    if (!NvGlDemoThrottleInit()) {
        return 0;
    }

    NvGlDemoInactivityInit();
    return 1;
}

// This should be called before every swap.
// A non-zero return indicates success.
int
NvGlDemoPreSwapExec(void) {
    // Add the fence object in queue and wait accordingly
    if (!NvGlDemoThrottleRendering()) {
        return 0;
    }

    NvGlDemoInactivitySleep();

    return 1;
}

// Deallocate resources from renderahead.
void
NvGlDemoPreSwapShutdown(void) {
    // Deallocate the resources used by renderahead
    NvGlDemoThrottleShutdown();
}

//
// Inactivity implementation
//

// Calculate times for halting periodically at the start of the program.
void
NvGlDemoInactivityInit(void) {
    sleepTime = SYSTIME();
    sleepInterval = (long long)(1000000000.0 * demoOptions.inactivityTime);

    if (demoOptions.inactivityTime > 0.0f) {
        NvGlDemoLog(" sleeping at %f second intervals...\n",
                    demoOptions.inactivityTime);
    }
}

void
NvGlDemoInactivitySleep(void) {
    // Handle inactivity if appropriate
    if (sleepInterval > 0) {
        if (SYSTIME() > sleepTime + sleepInterval) {
            usleep(sleepInterval / 1000);
            sleepTime = SYSTIME();
        }
    }
}

//
// Required logic for renderahead
//

int
NvGlDemoThrottleInit()
{

    if (!demoOptions.nFifo)
    {
        if (demoOptions.renderahead == -1) {
            return 1;
        } else {
            NvGlDemoCqInitIndex(demoOptions.renderahead);
            syncobjarr = malloc(demoOptions.renderahead * sizeof(GLsync));

            if (syncobjarr == NULL) {
                return 0;
            }
            return 1;
        }

        NvGlDemoLog(" Renderahead allowed is %d \n", demoOptions.renderahead);
    }
    return 1;
}

int
NvGlDemoThrottleRendering()
{
    int index;

    if (!demoOptions.nFifo) {

        switch(demoOptions.renderahead) {
        case -1:
            break;
        case 0:
            glFinish();
            break;
        default:
            {
                if (NvGlDemoCqFull()) {
                    index = NvGlDemoCqDeleteIndex();

                    if (index == -1) {
                        break;
                    }

                    GLsync  tempobj = syncobjarr[index];
                    long long wait_time = 10000000; //10 ms
                    long long total_time = 0;
                    long long time_out = 10000000000; //10 sec
                    GLenum ret;

                    do {
                        ret = glClientWaitSync(tempobj,
                                               GL_SYNC_FLUSH_COMMANDS_BIT,
                                               wait_time);
                        total_time += wait_time;
                    } while(ret != GL_ALREADY_SIGNALED &&
                            ret != GL_CONDITION_SATISFIED &&
                            total_time < time_out);

                    // Timeout of 10 sec
                    if (total_time >= time_out) {
                        return 0;
                    }
                    glDeleteSync(tempobj);
                }
                GLsync syncobj = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
                index = NvGlDemoCqInsertIndex();

                if (index == -1) {
                    break;
                }

                syncobjarr[index] = syncobj;
            }
        }
    }
    return 1;
}

void
NvGlDemoThrottleShutdown()
{
    if (demoOptions.renderahead > 0) {
        free(syncobjarr);
    }
}
