/*
 * Copyright (c) 2020 NVIDIA Corporation.  All Rights Reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation
 * is strictly prohibited.
 */

#include "rawstream.h"


NvSciSyncModule   syncModule          = NULL;
NvSciBufModule    bufModule           = NULL;

NvSciSyncAttrList producerSignalAttrs = NULL;
NvSciSyncAttrList consumerSignalAttrs = NULL;
NvSciSyncAttrList producerWaitAttrs   = NULL;
NvSciSyncAttrList consumerWaitAttrs   = NULL;
NvSciSyncAttrList prodToConsAttrs     = NULL;
NvSciSyncAttrList consToProdAttrs     = NULL;
NvSciSyncObj      consumerSignalObj   = NULL;
NvSciSyncObj      producerSignalObj   = NULL;
NvSciSyncObj      consumerWaitObj     = NULL;
NvSciSyncObj      producerWaitObj     = NULL;

NvSciBufAttrList  producerWriteAttrs  = NULL;
NvSciBufAttrList  consumerReadAttrs   = NULL;
NvSciBufAttrList  combinedBufAttrs    = NULL;
Buffer            buffers[totalBuffers];
IpcWrapper        ipcWrapper;

int main(int argc, char *argv[])
{
    NvSciError err;
    int producer;
    const char* endpoint;
    int ret = 0;

    if ((argc == 2) && (strcmp(argv[1], "-p") == 0)){
        producer = 1;
        endpoint = "Producer";
    } else if ((argc == 2) && (strcmp(argv[1], "-c") == 0)) {
        producer = 0;
        endpoint = "Consumer";
    } else {
        fprintf(stderr,
                "The usage of the app is ./rawstream followed by -p or -c\n");
        fprintf(stderr,
                "  -p denotes producer and -c denotes consumer\n");
        return 1;
    }

    fprintf(stderr, "%p application starting\n", endpoint);

    // Open sync module (shared by both all threads)
    err = NvSciSyncModuleOpen(&syncModule);
    if (NvSciError_Success != err) {
        fprintf(stderr, "%s unable to open sync module (%x)\n",
                endpoint, err);
        return 1;
    }

    // Open buf module (shared by both all threads)
    err = NvSciBufModuleOpen(&bufModule);
    if (NvSciError_Success != err) {
        fprintf(stderr, "%s unable to open buf module (%x)\n",
                endpoint, err);
        ret = 1;
        goto close_sync_module;
    }

    // Initialize IPC library
    err = NvSciIpcInit();
    if (NvSciError_Success != err) {
        fprintf(stderr, "%s unable to init ipc library (%x)\n",
                endpoint, err);
        ret = 1;
        goto close_buf_module;
    }

    // Establish IPC communications based on endpoint
    // TODO: Settle on final IPC channel names
    if (producer == 1) {
        err = ipcInit("nvscisync_a_0", &ipcWrapper);
    } else {
        err = ipcInit("nvscisync_a_1", &ipcWrapper);
    }
    if (NvSciError_Success != err) {
        fprintf(stderr, "%s unable to initialize communication (%x)\n",
                endpoint, err);
        ret = 1;
        goto deinit_IPC;
    }

    // Test communication by exchanging a simple handshake message
    const int send_handshake = 12345;
    err = ipcSend(&ipcWrapper, &send_handshake, sizeof(send_handshake));
    if (NvSciError_Success != err) {
        fprintf(stderr, "%s failed to send handshake (%x)\n",
                endpoint, err);
        ret = 1;
        goto deinit_IPC;
    }

    int recv_handshake = 0;
    ipcRecvFill(&ipcWrapper, &recv_handshake, sizeof(recv_handshake));
    if (NvSciError_Success != err) {
        fprintf(stderr, "%s failed to receive handshake (%x)\n",
                endpoint, err);
        ret = 1;
        goto deinit_IPC;
    }

    if (send_handshake != recv_handshake) {
        fprintf(stderr, "%s handshake did not match (%x)\n",
                endpoint, err);
        ret = 1;
        goto deinit_IPC;
    }

    // Initialize buffer list
    for (uint32_t i=0; i<totalBuffers; ++i) {
        buffers[i].owner = 0;
        buffers[i].fence = NvSciSyncFenceInitializer;
    }

    // Launch appropriate thread
    if (producer == 1) {
        // Launch producer threads
        pthread_t producerThread;

        if (0 != pthread_create(&producerThread, NULL, producerFunc, &ret)) {
            fprintf(stderr, "Failed to launch producer\n");
            ret = 1;
            goto deinit_IPC;
        }

        // Wait for thread to finish
        (void)pthread_join(producerThread, NULL);

    } else {
        // Launch consumer threads
        pthread_t consumerThread;

        if (0 != pthread_create(&consumerThread, NULL, consumerFunc, &ret)) {
            fprintf(stderr, "Failed to launch consumer\n");
            ret = 1;
            goto deinit_IPC;
        }

        // Wait for thread to finish
        (void)pthread_join(consumerThread, NULL);
    }

deinit_IPC:
    ipcDeinit(&ipcWrapper);
    (void)NvSciIpcDeinit();
close_buf_module:
    (void)NvSciBufModuleClose(bufModule);
close_sync_module:
    (void)NvSciSyncModuleClose(syncModule);

    fprintf(stderr, "Sample completed\n");

    return ret;
}

// Checksum calculation
#define CRC32_POLYNOMIAL 0xEDB88320L

uint32_t GenerateCRC(uint8_t* data_ptr,
                     uint32_t height,
                     uint32_t width,
                     uint32_t pitch)
{
    uint32_t y = 0U, x = 0U;
    uint32_t crc = 0U, tmp;
    static uint32_t crcTable[256];
    static int initialized = 0;

    //Initilaize CRC table, which is an one time operation
    if (!initialized) {
        for (int i = 0; i <= 255; i++) {
            tmp = i;
            for (int j = 8; j > 0; j--) {
                if (tmp & 1) {
                  tmp = (tmp >> 1) ^ CRC32_POLYNOMIAL;
                } else {
                  tmp >>= 1;
                }
            }
            crcTable[i] = tmp;
        }
        initialized = 1;
    }

    //Calculate CRC for the data
    for (y = 0U; y < height; y++) {
        for (x = 0U; x < width; x++) {
            tmp = (crc >> 8) & 0x00FFFFFFL;
            crc = tmp ^ crcTable[((uint32_t) crc ^ *(data_ptr + x)) & 0xFF];
        }
        data_ptr += pitch;
    }

    return crc;
}
