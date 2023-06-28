/*
 * nvgldemo_os_posix.c
 *
 * Copyright (c) 2007-2017, NVIDIA CORPORATION. All rights reserved.
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

//
// This file implements a few basic OS-specific utilities in POSIX.
//


#include "nvgldemo.h"
#include <sys/stat.h>
#include <stdio.h>
#include <stdarg.h>
#include <time.h>

#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <string.h>
#include <stdio.h>

extern char const *producerMsg;

#ifndef EGL_NV_system_time
// Gets the system time in nanoseconds
long long
NvGlDemoSysTime(void)
{
    struct timespec tp;
    clock_gettime(CLOCK_MONOTONIC, &tp);
    return (long long)tp.tv_sec*(long long)1000000000 + (long long)tp.tv_nsec;
}
#endif // EGL_NV_system_time

// Prints a message to standard out
void
NvGlDemoLog(
    const char* message, ...)
{
    va_list ap;
    int length;

    length = strlen(message);
    if (length > 0) {
        va_start(ap, message);
        vprintf(message,ap);
        va_end(ap);

        // if not newline terminated, add a newline
        if (message[length-1] != '\n') {
            printf("\n");
        }
    }
}

// Loads a data file into memory
char*
NvGlDemoLoadFile(
    const char *file,
    unsigned int *size)
{
    const char *filename;
    char path[1024];
    size_t filesize;
    char *data;
    struct stat st;
    FILE *f = 0;

#ifdef ANDROID
    // Look in data first
    snprintf(path, sizeof(path), "/data/graphics/demo/%s", file);
    if (!(f = fopen(filename = path, "rb"))) {
        printf("Unable to read file from %s\n", filename);
        return 0;
    }
#else
    // Look in both the current and parent dir
    snprintf(path, sizeof(path), "../%s", file);
    if (!(f = fopen(filename = path + 1, "rb")) &&
        !(f = fopen(filename = path, "rb"))) {
        return 0;
    }
#endif

    stat(filename, &st);
    if (st.st_size < 0) {
        printf("Unable to get file size: %s\n", filename);
        fclose(f);
        return 0;
    }
    filesize = st.st_size;

    data = (char *)malloc(filesize + 1);
    if (!data) {
        printf("Unable to alloc memory for file: %s\n", filename);
        fclose(f);
        return 0;
    }
    if (fread(data, 1, filesize, f) != filesize) {
        printf("Unable to read file: %s\n", filename);
        free(data);
        fclose(f);
        return 0;
    }
    data[filesize] = 0;
    fclose(f);

    if (size) *size = filesize;
    return data;
}

int
NvGlDemoSaveFile(
    const char *file,
    unsigned char *data,
    unsigned int size)
{
    const char *filename;
    char path[1024];
    struct stat st;
    FILE *f;
    size_t written;

    // Write in current directory
    snprintf(path, sizeof(path), "./%s", file);
    if (!(f = fopen(filename = path, "wb"))) {
        printf("Unable to open file: %s\n", file);
        return 0;
    }

    stat(filename, &st);
    if (st.st_size < 0) {
        printf("Unable to get file size: %s\n", filename);
        fclose(f);
        return 0;
    }

    if ((written = fwrite(data, 1, size, f)) != size) {
        printf("Unable to write file: %s, wrote %zu of %u\n", filename, written, size);
        fclose(f);
        return 0;
    }
    fclose(f);

    return size;
}

void *
NvGlDemoThreadCreate(
    void *(*start_routine) (void *),
    void *arg)
{
    pthread_t *thread = NULL;

    thread = (pthread_t *)malloc(sizeof(pthread_t));

    if (thread) {
        if (pthread_create(thread, NULL, start_routine, arg) != 0) {
            free(thread);
            thread = NULL;
        }
    }

    return (void *)thread;
}

int
NvGlDemoThreadJoin(
    void *threadId,
    void **retval)
{
    int ret = -1;
    ret = pthread_join(*((pthread_t *)threadId), retval);
    free(threadId);
    return ret;
}

void *
NvGlDemoSemaphoreCreate(
    int pshared,
    unsigned int value)
{
    sem_t *sem = NULL;

    sem = (sem_t *)malloc(sizeof(sem_t));

    if (sem) {
        if (sem_init(sem, pshared, value) != 0) {
            free(sem);
            sem = NULL;
        }
    }

    return (void *)sem;
}

int
NvGlDemoSemaphoreDestroy(
    void *semId)
{
    int ret = -1;
    ret = sem_destroy((sem_t *)semId);
    free(semId);
    return ret;
}

int
NvGlDemoSemaphoreWait(
    void *semId)
{
    return sem_wait((sem_t *)semId);
}

int
NvGlDemoSemaphorePost(
    void *semId)
{
    return sem_post((sem_t *)semId);
}

int
NvGlDemoThreadYield(void)
{
    return sched_yield();
}

int NvGlDemoCreateSocket(void)
{
    int sockID = socket(AF_INET , SOCK_STREAM , 0);

    if (-1 == sockID) {
        NvGlDemoLog("Socket creation failed!\n");
    }

    return sockID;
}

void NvGlDemoServerBind(int sockID)
{
    struct sockaddr_in server;
    int yes = 1;

    if (setsockopt(sockID, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(int)) == -1) {
        perror("setsockopt SO_REUSEADDR-failed:");
        return;
    }
    server.sin_family      = AF_INET;
    server.sin_addr.s_addr = INADDR_ANY;
    server.sin_port        = htons(demoOptions.port);

    while (bind(sockID, (struct sockaddr *)&server, sizeof(server)) < 0) {
        perror("Waiting for Server socket bind complete:");
    }
    NvGlDemoLog("Server bind done. Waiting for connections.\n");

    return;
}

void NvGlDemoServerListen(int sockID)
{
    listen(sockID , 3);
}

int NvGlDemoServerAccept(int sockID)
{
    struct sockaddr_in client;
    int c = sizeof(struct sockaddr_in);

    int clientSockID = accept(sockID, (struct sockaddr *)&client, (socklen_t*)&c);
    if (clientSockID < 0) {
        NvGlDemoLog("Server failed to accept connection to %d\n", sockID);
    } else {
        NvGlDemoLog("Connection accepted\n");
    }

    return clientSockID;
}

void NvGlDemoServerReceive(int clientSockID)
{
    NvGlDemoLog("Server listening...\n");
    char msg[100];

    int readSize = recv(clientSockID , msg , strlen(producerMsg) , 0);
    if (readSize > 0) {
        msg[readSize] = 0;
        NvGlDemoLog("Message received: %s\n", msg);
    } else {
        NvGlDemoLog("\nServer didn't receive message!");
    }
}

// Below funtion will return 0 on success, return 1 on failure
int NvGlDemoClientConnect(const char * ip, int sockID)
{
    struct sockaddr_in server;
    int ret = 1;
    server.sin_addr.s_addr = inet_addr(ip);
    server.sin_family      = AF_INET;
    server.sin_port        = htons(demoOptions.port);

    if (connect(sockID , (struct sockaddr *)&server , sizeof(server)) < 0) {
        NvGlDemoLog("Client failed to connect to %s\n", ip);
        ret = 1;
    } else {
        NvGlDemoLog("Client now connected to server\n");
        ret = 0;
    }

    return ret;
}

void NvGlDemoClientSend(const char* data, int sockID)
{
    if ( send(sockID , data , strlen(data) , 0) < 0 ) {
        NvGlDemoLog("Client failed to send data.\n");
    } else {
        NvGlDemoLog("Send sucess!\n");
    }

    return;
}

