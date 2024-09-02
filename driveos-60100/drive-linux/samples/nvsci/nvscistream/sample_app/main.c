/* NvSciStream Safety Sample App - main application
 *
 * Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <unistd.h>
#include <stdio.h>
#include "nvscisync.h"
#include "nvscibuf.h"
#include "nvsciipc.h"
#include "nvscistream.h"
#include "block_info.h"

/* NvSci modules */
NvSciSyncModule sciSyncModule;
NvSciBufModule  sciBufModule;

/* NvSciIpc Endpoint */
NvSciIpcEndpoint ipcsrcEndpoint = 0U;
NvSciIpcEndpoint ipcdstEndpoint = 0U;

/* Default test arguments
* Number of packets : 3
* Number of stream conusumers: 2
* Limiter block limit : 3 packet
* Timeout set to -1 default
*/
TestArgs testArgs = {3U, 2U, 3U, -1};

/* Total stream blocks that are
* created for this usecase */
NvSciStreamBlock producer;
NvSciStreamBlock staticPool;
NvSciStreamBlock queue[2];
NvSciStreamBlock ipcSrc[2];
NvSciStreamBlock c2cSrc;
NvSciStreamBlock c2cDst;
NvSciStreamBlock multicast;
NvSciStreamBlock consumer;
NvSciStreamBlock ipcDst[2];
NvSciStreamBlock returnSync;
NvSciStreamBlock limiter;

/* pthread variables for thread synchronization */
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

/* Temporary buffer for copying the data
* from consumer packet to producer packet */
uint8_t* tempBuffer = NULL;

/* Flag indicating the status of avilable packet from
* producer
*/
bool producerPktAvailable = false;

/* Flag indicating the status of buffer copy from
consumer packet to producer packet */
bool bufferCopyDone = false;

/* Flag indicatng the termination of the QM
consumer */
bool streamEnd = false;

/* Print command line options */
static void print_usage(const char *str)
{
    printf("%s [options]\n", str);
    printf("\n [-a]            Launches ASIL process.");
    printf("\n [-q]            Launches QM proxy process.");
    printf("\n [-p]            Launches QM2 process.");
}

/*
 * Main application function.
 *   As per standards, return of 0 indicates success and anything
 *   else is failure.
 */
int main(int argc, char *argv[])
{
    /*
     * Initialize NvSci libraries
     */
    if (NvSciError_Success != NvSciSyncModuleOpen(&sciSyncModule)) {
        printf("Unable to open NvSciSync module\n");
    }
    if (NvSciError_Success != NvSciBufModuleOpen(&sciBufModule)) {
        printf("Unable to open NvSciBuf module\n");
    }
    if (NvSciError_Success != NvSciIpcInit()) {
        printf("Unable to initialize NvSciIpc\n");
    }

    memset(testArgs.c2csrcChannel, 0U, sizeof(testArgs.c2csrcChannel));
    memset(testArgs.c2cdstChannel, 0U, sizeof(testArgs.c2cdstChannel));

    /* Parse command line */
    int32_t opt;
    int32_t rv = 0U;
    while ((opt = getopt(argc, argv, "aqp")) != EOF) {
        switch (opt) {
        case 'a': /* Launch ASIL process */
            rv = handleASILProcess();
            break;
        case 'q': /* Launch QM proxy process */
            /* Ipc channel used to communicate with C2C consumer */
            if (argv[optind]) {
                strcpy(testArgs.c2csrcChannel, argv[optind]);
            } else {
                print_usage(argv[0]);
                break;
            }
            rv = handleQMProxyProcess();
            break;
        case 'p': /* Launch QM process */
            /* Ipc channel used to communicate with C2C producer */
            if (argv[optind]) {
                strcpy(testArgs.c2cdstChannel, argv[optind]);
            } else {
                print_usage(argv[0]);
                break;
            }
            rv = handleQMProcess();
            break;
        default:
            print_usage(argv[0]);
            break;
        }
    }

    /* Release buffer/sync resources */
    if (sciBufModule != NULL) {
        NvSciBufModuleClose(sciBufModule);
        sciBufModule = NULL;
    }

    if (sciSyncModule != NULL) {
        NvSciSyncModuleClose(sciSyncModule);
        sciSyncModule = NULL;
    }

    /* Close the NvSciIpc endpoint */
    if (ipcsrcEndpoint) {
        if (NvSciError_Success !=
            NvSciIpcCloseEndpointSafe(ipcsrcEndpoint, false)) {
            printf("Failed to close ipcsrc endpoint\n");
        }
        ipcsrcEndpoint = 0U;
    }

    if (ipcdstEndpoint) {
        if (NvSciError_Success !=
            NvSciIpcCloseEndpointSafe(ipcdstEndpoint, false)) {
            printf("Failed to close ipcdst endpoint\n");
        }
        ipcdstEndpoint = 0U;
    }

    /* freeing the resources */
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond);
    if (tempBuffer != NULL) {
        free(tempBuffer);
    }

    NvSciIpcDeinit();

    return rv;
}
