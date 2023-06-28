/*
 * demo.cpp
 *
 * Copyright (c) 2015-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <signal.h>
#include "utils.h"
#include "producer.h"
#include "consumer.h"
#include "nvgldemo.h"

EXTENSION_LIST(EXTLST_EXTERN)

volatile bool sigCaught = false;

static bool isProducer = false;
static bool isConsumer = false;

static bool run()
{
    if (isProducer) {
        return runProducerProcess(demoState.stream, demoState.display);
    } else if (isConsumer) {
        return runConsumerProcess(demoState.stream, demoState.display);
    }

    return true;
}

static void closeCB(void)
{
    sigCaught = true;
}

static void printHelp()
{
    NvGlDemoLog("Usage: eglcrosspart [options] [command] [command options]\n"
                "  smart or not (use heartbeat socket)\n"
                "    [-smart <1 or 0>]\n");
    NvGlDemoLog(NvGlDemoArgUsageString());
}

// Usage: o Consumer: ./eglcrosspart -proctype consumer
//        o Producer:. /eglcrosspart -ip 12.0.0.11 -proctype producer
int main(int argc, char *argv[])
{
    bool shouldExit = false;

    if ((argc < 2) || !NvGlDemoArgParse(&argc, argv)) {
        printHelp();
        exit(0);
    }

    // There are no app-specific command line options defined for this app.
    // Exit if any options are left over after parsing.
    // TODO: "-smart" is parsed as a generic argument, though it is only
    //       used by eglcrosspart.
    if (argc > 1) {
        NvGlDemoLog("Unknown command line option (%s)\n", argv[1]);
        printHelp();
        exit(0);
    }


    if (!strncmp(demoOptions.proctype, "consumer", NVGLDEMO_MAX_NAME)){
        isConsumer = true;
    } else if (!strncmp(demoOptions.proctype, "producer", NVGLDEMO_MAX_NAME)) {
        isProducer = true;
    }

    while (!shouldExit) {
        // Setting up ping-pong conditionlly
        if (demoOptions.isSmart == 1) {
            NvGlDemoLog("Can detect abnormal termination of other end..\n");
            if ((isConsumer && !createConsumerPingPongThread()) ||
                (isProducer && !createProducerPingPongThread())) {
                 NvGlDemoLog("%s side of ping-pong has been initiated \n", isProducer ? "Producer" : "Consumer");
            }
            else {
                NvGlDemoLog("fail to initiate %s side of ping-pong \n", isProducer ? "Producer" : "Consumer");
                exit(0);
            }
        } else {
            NvGlDemoLog("Can not detect abnormal termination of other end..\n");
        }

        if (!(isConsumer && (demoOptions.eglQnxScreenTest == 1))) {
            if (!NvGlDemoInitializeParsed(&argc, argv, "eglcrosspart", 2, 8, 0)) {
                if (sigCaught) {
                    NvGlDemoLog("Consumer finished running\n");
                    exit(0);
                }
                NvGlDemoLog("Error during initialization. Please check help\n");
                exit(0);
           }
        } else {
            EGLBoolean eglStatus;
            const char* extensions = NULL;

            if (!NvGlDemoInitConsumerProcess()) {
                NvGlDemoLog("EGL failed to create consumer socket.\n");
                exit(0);
            }

            demoState.display = NVGLDEMO_EGL_GET_DISPLAY(demoState.nativeDisplay);
            if (demoState.display == EGL_NO_DISPLAY) {
                NvGlDemoLog("EGL failed to obtain display.\n");
                exit(0);
            }

            // Initialize EGL
            eglStatus = NVGLDEMO_EGL_INITIALIZE(demoState.display, 0, 0);
            if (!eglStatus) {
                NvGlDemoLog("EGL failed to initialize.\n");
                exit(0);
            }

            // Get extension string
            extensions = NVGLDEMO_EGL_QUERY_STRING(demoState.display, EGL_EXTENSIONS);
            if (!extensions) {
                NvGlDemoLog("eglQueryString fail.\n");
                exit(0);
            }

            if (!strstr(extensions, "EGL_EXT_stream_consumer_qnxscreen_window")) {
                NvGlDemoLog("EGL does not support EGL_EXT_stream_consumer_qnxscreen_window extension.\n");
                exit(0);
            }


            if(!NvGlDemoCreateCrossPartitionEGLStream()) {
                NvGlDemoLog("EGL failed to create CrossPartitionEGLStream.\n");
                exit(0);
            }
        }

        // Setup the EGL extensions.
        setupExtensions();

        NvGlDemoSetCloseCB(closeCB);

        if (!run() || sigCaught) {
            shouldExit = true;
        }

        NvGlDemoLog("Finished running\n");

        if (demoOptions.isSmart == 1) {
            if (isConsumer) {
                resetConsumerPingPong();
            } else if (isProducer) {
                resetProducerPingPong();
            }
        }

        demoState.stream = EGL_NO_STREAM_KHR;

        if (!(isConsumer && (demoOptions.eglQnxScreenTest == 1))) {
            NvGlDemoShutdown();
        } else {
            NvGlDemoEglTerminate();
        }

        if (isProducer && !shouldExit) {
            // waiting for 10 seconds. Consumer should be started in 10 seconds
            // otherwise producer will exit which defeats the purpose of while loop.
            NvGlDemoLog("Waiting for 10 secs..\n");
            sleep(10);
        }
    }
    return 0;
}
