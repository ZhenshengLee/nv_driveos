/*
 * pingPongConsumer.cpp
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
#include "consumer.h"
#include "pingPong.h"

static bool producerAliveStatus = false;
static pthread_t consumerPingPongThread = 0;
static sem_t producerAliveSem;
static bool shouldConsumerPingPongThreadExit = false;
static int ppConsumerSocket = -1;

extern int g_ServerID;
extern int g_ServerAllocatedID;


static void *consumerPingPong(void *data) {
    char aliveMsg[ALIVE_MSG_LEN] = {ALIVE_MSG_CHAR};
    struct timeval tv;

    tv.tv_sec = ABNORMAL_TERMINATION_TIMEOUT;
    tv.tv_usec = 0;

    setsockopt(ppConsumerSocket, SOL_SOCKET, SO_RCVTIMEO, (char *)&tv, sizeof tv);
    producerAliveStatus = true;
    if (sem_post(&producerAliveSem) == -1) {
        NvGlDemoLog("ERROR Semphore post error");
        exit(1);
    }

    while (!shouldConsumerPingPongThreadExit) {
        if (read(ppConsumerSocket, aliveMsg, 1) <= 0) {
            NvGlDemoLog("Producer termination detected");
            break;
        }

        usleep(SLEEP_TIME_IN_MICROSECONDS);
        if (write(ppConsumerSocket, aliveMsg, 1) != 1) {
            NvGlDemoLog("ERROR ping-pong failed");
            break;
        }
    }
    producerAliveStatus = false;
    if (shutdown(g_ServerID, SHUT_RDWR)) {
        NvGlDemoLog("Perhaps, eglstream's socket got already closed in driver. Please ignore.");
    }
    g_ServerID = -1;
    close(ppConsumerSocket);
    ppConsumerSocket = -1;
    NvGlDemoLog("consumerPingPong exiting");
    return NULL;
}

void resetConsumerPingPong() {

    if (ppConsumerSocket != -1) {
        close(ppConsumerSocket);
        ppConsumerSocket = -1;
    }
    if (consumerPingPongThread) {
        shouldConsumerPingPongThreadExit = true;
        pthread_join(consumerPingPongThread, NULL);
        consumerPingPongThread = 0;
    }

    if (sem_destroy(&producerAliveSem) == -1) {
        NvGlDemoLog("Semphore destroy error");
    }

    producerAliveStatus = false;
    return;
}

int createConsumerPingPongThread()
{
    int serverID = NvGlDemoCreateSocket();

    if (serverID != -1) {
        NvGlDemoServerBind(serverID);
        NvGlDemoServerListen(serverID);
        g_ServerAllocatedID = serverID;
    }

    // Will be returned after client makes connect call.
    NvGlDemoLog("Waiting for pingPong client to connect");
    ppConsumerSocket = NvGlDemoServerAccept(g_ServerAllocatedID);
    if (ppConsumerSocket < 0) {
        NvGlDemoLog("ERROR on accept");
        goto fail;
    }
    NvGlDemoLog("PingPong client connected");

    if (sem_init(&producerAliveSem, 0, 0) == -1) {
        NvGlDemoLog("ERROR Semphore init error");
        goto fail;
    }

    shouldConsumerPingPongThreadExit = false;

    if (pthread_create(&consumerPingPongThread, NULL, consumerPingPong, NULL) != 0) {
        NvGlDemoLog("ERROR in thread creation..Can't detect abnormal client "
                    "termination");
        goto fail;
    }

    if (sem_wait(&producerAliveSem) == -1) {
        NvGlDemoLog("ERROR Semphore wait error");
        goto fail;
    }
    return 0;

fail :
    resetConsumerPingPong();
    return 1;
}

bool isProducerAlive()
{
    return producerAliveStatus;
}
