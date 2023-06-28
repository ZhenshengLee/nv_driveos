/*
 * pingPongProducer.cpp
 *
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <errno.h>
#include "producer.h"
#include "pingPong.h"

static int ppProducerSocket = -1;
static bool consumerAliveStatus = false;
static pthread_t producerPingPongThread = 0;
static sem_t consumerAliveSem;
static bool shouldProducerPingPongThreadExit = false;

extern int g_ClientID;

static void *producerPingPong (void *data)
{
    char aliveMsg[ALIVE_MSG_LEN] = {ALIVE_MSG_CHAR};
    struct timeval tv;

    tv.tv_sec = ABNORMAL_TERMINATION_TIMEOUT;
    tv.tv_usec = 0;

    setsockopt(ppProducerSocket, SOL_SOCKET, SO_RCVTIMEO, (char *)&tv, sizeof tv);
    consumerAliveStatus = true;
    if (sem_post(&consumerAliveSem) == -1) {
        NvGlDemoLog("ERROR Semphore post error");
        exit(1);
    }

    while(!shouldProducerPingPongThreadExit) {
       if (write(ppProducerSocket, aliveMsg, 1) != 1) {
            NvGlDemoLog("ERROR ping-pong failed ");
            break;
        }
        usleep(SLEEP_TIME_IN_MICROSECONDS);
        if(read(ppProducerSocket, aliveMsg, 1) <= 0) {
            NvGlDemoLog("Consumer termination detected");
            break;
        }
    }
    consumerAliveStatus = false;
    if (shutdown(g_ClientID, SHUT_RDWR)) {
        NvGlDemoLog("Perhaps, eglstream's socket got already closed in driver. Please ignore.");
    }
    close(ppProducerSocket);
    ppProducerSocket = -1;
    NvGlDemoLog("producerPingPong exiting");
    return NULL;
}

void resetProducerPingPong() {
    if (ppProducerSocket != -1) {
        close(ppProducerSocket);
        ppProducerSocket = -1;
    }
    if (producerPingPongThread) {
        shouldProducerPingPongThreadExit = true;
        pthread_join(producerPingPongThread, NULL);
        producerPingPongThread = 0;
    }

    if (sem_destroy(&consumerAliveSem) == -1) {
        NvGlDemoLog("Semphore destroy error");
    }

    consumerAliveStatus = false;
    return;
}

int createProducerPingPongThread()
{
    // Create client socket.
    ppProducerSocket = NvGlDemoCreateSocket();
    if (ppProducerSocket < 0) {
        NvGlDemoLog("ERROR opening socket");
        goto fail;
    }

    NvGlDemoLog("Connecting to pingPong server");
    if (NvGlDemoClientConnect(demoOptions.ipAddr, ppProducerSocket)) {
        NvGlDemoLog("Could not connect to pingPong server");
        goto fail;
    }
    NvGlDemoLog("Connected to pingPong server");

    if (sem_init(&consumerAliveSem, 0, 0) == -1) {
        NvGlDemoLog("ERROR Semphore init error");
        goto fail;
    }

    shouldProducerPingPongThreadExit = false;

    if (pthread_create(&producerPingPongThread, NULL, producerPingPong, (void *) &ppProducerSocket) != 0) {
        NvGlDemoLog("ERROR in thread creation");
        goto fail;
    }

    if (sem_wait(&consumerAliveSem) == -1) {
        NvGlDemoLog("ERROR Semphore wait error");
        goto fail;
    }
    return 0;

fail:
    resetProducerPingPong();
    return 1;
}

bool isConsumerAlive()
{
    return consumerAliveStatus;
}
