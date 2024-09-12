/*
 * Copyright (c) 2020-2021 NVIDIA Corporation.  All Rights Reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation
 * is strictly prohibited.
 */

#include "rawstream.h"

void* producerFunc(void* arg)
{
    CudaClientInfo cudaInfo;
    NvSciError     sciErr;
    int            cudaErr;

    *(int*)arg = 1;
    fprintf(stderr, "Producer starting\n");

    // Do common cuda initialization
    if (!setupCuda(&cudaInfo)) {
        goto done;
    }

    // Create an empty sync attribute list for signaling permissions.
    sciErr = NvSciSyncAttrListCreate(syncModule, &producerSignalAttrs);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Unable to create producer signal attrs (%x)\n",
                sciErr);
        goto done;
    }

    // Query CUDA for attributes needed to signal syncs
    cudaErr = cudaDeviceGetNvSciSyncAttributes(producerSignalAttrs,
                                               cudaInfo.deviceId,
                                               cudaNvSciSyncAttrSignal);
    if (cudaSuccess != cudaErr) {
        fprintf(stderr,
                "Could not query signal attributes from CUDA (%d)\n",
                cudaErr);
        goto done;
    }

    fprintf(stderr, "Producer signal attributes established\n");

    // Create an empty sync attribute list for waiting permissions.
    sciErr = NvSciSyncAttrListCreate(syncModule, &producerWaitAttrs);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Unable to create producer wait attrs (%x)\n",
                sciErr);
        goto done;
    }

    // Query CUDA for attributes needed to wait for syncs
    cudaErr = cudaDeviceGetNvSciSyncAttributes(producerWaitAttrs,
                                               cudaInfo.deviceId,
                                               cudaNvSciSyncAttrWait);
    if (cudaSuccess != cudaErr) {
        fprintf(stderr,
                "Could not query wait attributes from CUDA (%d)\n",
                cudaErr);
        goto done;
    }

    fprintf(stderr, "Producer wait attributes established\n");

    // Export producer's wait attributes to a form suitable for IPC
    size_t sendWaitAttrListSize = 0U;
    void* sendWaitListDesc = NULL;
    sciErr = NvSciSyncAttrListIpcExportUnreconciled(&producerWaitAttrs,
                                                    1,
                                                    ipcWrapper.endpoint,
                                                    &sendWaitListDesc,
                                                    &sendWaitAttrListSize);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Unable to export producer wait attrs (%x)\n",
                sciErr);
        goto done;
    }

    // Send the size of the producer's wait attributes to the consumer,
    //   so it knows how much data to expect
    sciErr = ipcSend(&ipcWrapper,
                     &sendWaitAttrListSize,
                     sizeof(sendWaitAttrListSize));
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Unable to send producer wait attrs size (%x)\n",
                sciErr);
        goto done;
    }

    // Send the exported form of the producer's wait attributes
    sciErr = ipcSend(&ipcWrapper,
                     sendWaitListDesc,
                     sendWaitAttrListSize);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Unable to send producer wait attrs (%x)\n",
                sciErr);
        goto done;
    }

    // Wait to receive the size of the consumer's wait attributes
    size_t recvWaitAttrListSize = 0U;
    sciErr = ipcRecvFill(&ipcWrapper,
                         &recvWaitAttrListSize,
                         sizeof(recvWaitAttrListSize));
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Unable to recv consumer wait attr size (%x)\n",
                sciErr);
        goto done;
    }

    // Allocate a buffer big enough to receive the consumer's wait attributes
    void* recvWaitListDesc = malloc(recvWaitAttrListSize);
    if (NULL == recvWaitListDesc) {
        sciErr = NvSciError_InsufficientMemory;
        fprintf(stderr,
                "Sync attr allocation failed (%x)\n",
                sciErr);
        goto done;
    }

    // Wait to receive consumer's wait attributes
    sciErr = ipcRecvFill(&ipcWrapper,
                         recvWaitListDesc,
                         recvWaitAttrListSize);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Unable to recv consumer wait attrs (%x)\n",
                sciErr);
        goto done;
    }

    // Convert the received consumer wait attributes to an attribute list
    sciErr = NvSciSyncAttrListIpcImportUnreconciled(syncModule,
                                                    ipcWrapper.endpoint,
                                                    recvWaitListDesc,
                                                    recvWaitAttrListSize,
                                                    &consumerWaitAttrs);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Unable to import consumer wait attrs (%x)\n",
                sciErr);
        goto done;
    }

    // Get combined attributes for producer to consumer signaling
    NvSciSyncAttrList syncAllAttrs[2], syncConflictAttrs;
    syncAllAttrs[0] = producerSignalAttrs;
    syncAllAttrs[1] = consumerWaitAttrs;
    sciErr = NvSciSyncAttrListReconcile(syncAllAttrs,
                                        2,
                                        &prodToConsAttrs,
                                        &syncConflictAttrs);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Can't merge producer->consumer attrs (%x)\n",
                sciErr);
        goto done;
    }

    // Allocate producer to consumer sync object
    sciErr = NvSciSyncObjAlloc(prodToConsAttrs, &producerSignalObj);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Can't allocate producer->consumer sync (%x)\n",
                sciErr);
        goto done;
    }

    // Export sync attributes and object to a form suitable for IPC
    void* sendObjAndListDesc = NULL;
    size_t sendObjAndListSize = 0U;
    sciErr = NvSciSyncIpcExportAttrListAndObj(producerSignalObj,
                                              NvSciSyncAccessPerm_WaitOnly,
                                              ipcWrapper.endpoint,
                                              &sendObjAndListDesc,
                                              &sendObjAndListSize);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Can't export producer->consumer sync description (%x)\n",
                sciErr);
        goto done;
    }

    // Send the size of the sync description to the consumer,
    //   so it knows how much data to expect
    sciErr = ipcSend(&ipcWrapper, &sendObjAndListSize, sizeof(size_t));
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Can't send producer->consumer sync description size(%x)\n",
                sciErr);
        goto done;
    }

    // Send the sync description to the consumer
    sciErr = ipcSend(&ipcWrapper, sendObjAndListDesc, sendObjAndListSize);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Can't send producer->consumer sync description (%x)\n",
                sciErr);
        goto done;
    }

    // Wait to receive the size of the consumer->producer sync desription
    size_t recvObjAndListSize = 0U;
    sciErr = ipcRecvFill(&ipcWrapper,
                         &recvObjAndListSize,
                         sizeof(size_t));
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Can't recv consumer->produce sync description size (%x)\n",
                sciErr);
        goto done;
    }

    // Allocate a buffer big enough to receive the description
    void* recvObjAndListDesc = malloc(recvObjAndListSize);
    if (NULL == recvObjAndListDesc) {
        sciErr = NvSciError_InsufficientMemory;
        fprintf(stderr,
                "Sync description allocation failed (%x)\n",
                sciErr);
        goto done;
    }

    // Wait to receive consumer->producer sync description
    sciErr = ipcRecvFill(&ipcWrapper,
                         recvObjAndListDesc,
                         recvObjAndListSize);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Can't receive consumer->producer sync description (%x)\n",
                sciErr);
        goto done;
    }

    // Convert the received consumer->producer sync description to a
    //   sync attribute list and object
    sciErr = NvSciSyncIpcImportAttrListAndObj(syncModule,
                                              ipcWrapper.endpoint,
                                              recvObjAndListDesc,
                                              recvObjAndListSize,
                                              &producerWaitAttrs,
                                              1,
                                              NvSciSyncAccessPerm_WaitOnly,
                                              ipcWrapper.endpoint,
                                              &producerWaitObj);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr, "Can't import consumer->producer sync (%x)\n", sciErr);
        goto done;
    }

    // Set up CUDA sync objects, importing NvSciSync objects
    if (!setupCudaSync(&cudaInfo, producerSignalObj, producerWaitObj)) {
        goto done;
    }

    fprintf(stderr, "Producer exchanged sync objects with consumer\n");

    // Create an empty buffer attribute list for producer buffers
    sciErr = NvSciBufAttrListCreate(bufModule, &producerWriteAttrs);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Unable to create producer buffer attrs (%x)\n",
                sciErr);
        goto done;
    }

    // Fill producer buffer attribute list with values
    NvSciBufAttrKeyValuePair bufKeyValue[6];
    NvSciRmGpuId gpuId;
    memcpy(&gpuId.bytes, &cudaInfo.uuid.bytes, sizeof(cudaInfo.uuid.bytes));
    bufKeyValue[0].key   = NvSciBufGeneralAttrKey_GpuId;
    bufKeyValue[0].value = &gpuId;
    bufKeyValue[0].len   = sizeof(gpuId);
    NvSciBufType bufType = NvSciBufType_RawBuffer;
    bufKeyValue[1].key   = NvSciBufGeneralAttrKey_Types;
    bufKeyValue[1].value = &bufType;
    bufKeyValue[1].len   = sizeof(bufType);
    NvSciBufAttrValAccessPerm bufPerm = NvSciBufAccessPerm_ReadWrite;
    bufKeyValue[2].key   = NvSciBufGeneralAttrKey_RequiredPerm;
    bufKeyValue[2].value = &bufPerm;
    bufKeyValue[2].len   = sizeof(bufPerm);
    bool   bufAccessFlag = true;
    bufKeyValue[3].key   = NvSciBufGeneralAttrKey_NeedCpuAccess;
    bufKeyValue[3].value = &bufAccessFlag;
    bufKeyValue[3].len   = sizeof(bufAccessFlag);
    uint64_t rawsize     = (128 * 1024);
    bufKeyValue[4].key   = NvSciBufRawBufferAttrKey_Size;
    bufKeyValue[4].value = &rawsize;
    bufKeyValue[4].len   = sizeof(rawsize);
    uint64_t align       = (4 * 1024);
    bufKeyValue[5].key   = NvSciBufRawBufferAttrKey_Align;
    bufKeyValue[5].value = &align;
    bufKeyValue[5].len   = sizeof(align);

    sciErr = NvSciBufAttrListSetAttrs(producerWriteAttrs, bufKeyValue, 6);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr, "Unable to fill producer buffer attrs (%x)\n", sciErr);
        goto done;
    }

    fprintf(stderr, "Producer buffer attributes established\n");

    // Wait to receive the size of the consumer's buffer attributes
    size_t consumerReadAttrsSize = 0U;
    sciErr = ipcRecvFill(&ipcWrapper,
                         &consumerReadAttrsSize,
                         sizeof(consumerReadAttrsSize));
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Unable to recv consumer buffer attr size (%x)\n",
                sciErr);
        goto done;
    }

    // Allocate a buffer big enough to receive the consumer's buffer attributes
    void* consumerReadAttrsDesc = malloc(consumerReadAttrsSize);
    if (NULL == recvWaitListDesc) {
        sciErr = NvSciError_InsufficientMemory;
        fprintf(stderr, "Buffer attr allocation failed(%x)\n", sciErr);
        goto done;
    }

    // Wait to receive the consumer's buffer attributes
    sciErr = ipcRecvFill(&ipcWrapper,
                         consumerReadAttrsDesc,
                         consumerReadAttrsSize);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr, "Unable to recv consumer buffer attrs (%x)\n", sciErr);
        goto done;
    }

    // Convert the received consumer buffer attributes to an attribute list
    sciErr = NvSciBufAttrListIpcImportUnreconciled(bufModule,
                                                   ipcWrapper.endpoint,
                                                   consumerReadAttrsDesc,
                                                   consumerReadAttrsSize,
                                                   &consumerReadAttrs);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Unable to import consumer buffer attrs (%x)\n",
                sciErr);
        goto done;
    }

    // Get combined attributes for buffers
    NvSciBufAttrList bufAllAttrs[2], bufConflictAttrs;
    bufAllAttrs[0] = producerWriteAttrs;
    bufAllAttrs[1] = consumerReadAttrs;
    sciErr = NvSciBufAttrListReconcile(bufAllAttrs, 2,
                                       &combinedBufAttrs, &bufConflictAttrs);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr, "Can't merge buffer attrs (%x)\n", sciErr);
        goto done;
    }

    // Export combined buffer attributes to a form suitable for IPC
    void* sendBufListDesc = NULL;
    size_t sendBufListSize = 0U;
    sciErr = NvSciBufAttrListIpcExportReconciled(combinedBufAttrs,
                                                 ipcWrapper.endpoint,
                                                 &sendBufListDesc,
                                                 &sendBufListSize);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Can't export reconciled buffer attrs to consumer (%x)\n",
                sciErr);
        goto done;
    }

    // Send the size of the combined buffer attributes to the consumer,
    //   so it knows how much data to expect
    sciErr = ipcSend(&ipcWrapper,
                     &sendBufListSize,
                     sizeof(sendBufListSize));
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Unable to send combined buffer attrs size (%x)\n",
                sciErr);
        goto done;
    }

    // Send the exported form of the combined buffer attributes
    sciErr = ipcSend(&ipcWrapper,
                     sendBufListDesc,
                     sendBufListSize);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr, "Unable to send combined buffer attrs (%x)\n", sciErr);
        goto done;
    }

    // Extract attributes needed by CUDA
    if (!setupCudaBufAttr(&cudaInfo, combinedBufAttrs)) {
        goto done;
    }

    // Allocate all buffers
    for (uint32_t i=0U; i<totalBuffers; ++i) {

        Buffer* buf = &buffers[i];

        // Allocate the buffer
        sciErr = NvSciBufObjAlloc(combinedBufAttrs, &buf->obj);
        if (NvSciError_Success != sciErr) {
            fprintf(stderr, "Can't allocate buffer %d (%x)\n", i, sciErr);
            goto done;
        }

        // Export buffer object to a form suitable for IPC
        // Note: Unlike attribute lists, the exported form of objects has
        //       a fixed size.
        NvSciBufObjIpcExportDescriptor objDesc;
        sciErr = NvSciBufObjIpcExport(buf->obj,
                                      NvSciBufAccessPerm_ReadWrite,
                                      ipcWrapper.endpoint,
                                      &objDesc);
        if (NvSciError_Success != sciErr) {
            fprintf(stderr,
                    "Unable to export buffer %d object to consumer (%x)\n",
                    i, sciErr);
            goto done;
        }

        // Send the buffer description to the consumer
        sciErr = ipcSend(&ipcWrapper,
                         &objDesc,
                         sizeof(NvSciBufObjIpcExportDescriptor));
        if (NvSciError_Success != sciErr) {
            fprintf(stderr, "Unable to send buffer %d (%x)\n", i, sciErr);
            goto done;
        }

        // Import the buffer into CUDA
        if (!setupCudaBuffer(&cudaInfo, buf)) {
            goto done;
        }
    }

    fprintf(stderr, "Producer buffers established and transmitted\n");

    // Send all frames
    uint32_t currFrame  = 0;
    uint32_t currBuffer = 0;
    Packet packet;
    while (currFrame < totalFrames) {
        fprintf(stderr, "Producer starting frame %d in buffer %d\n",
                currFrame, currBuffer);
        Buffer* buf = &buffers[currBuffer];

        // Wait for buffer to be available
        // Note: On first frame for each buffer, the producer already owns
        //       it, so this is skipped. On subsequent frames it must wait
        //       for the buffer's return.
        while (buf->owner != 0U) {

            // Wait for next returned buffer
            sciErr = ipcRecvFill(&ipcWrapper, &packet, sizeof(packet));
            if (NvSciError_Success != sciErr) {
                fprintf(stderr,
                        "Failure to recv buffer from consumer (%x)\n",
                        sciErr);
                goto done;
            }

            // Import transmitted fence description to a fence
            sciErr = NvSciSyncIpcImportFence(producerWaitObj,
                                             &packet.fenceDesc,
                                             &buffers[packet.bufferId].fence);
            if (NvSciError_Success != sciErr) {
                fprintf(stderr,
                        "Failure to import fence from consumer (%x)\n",
                        sciErr);
                goto done;
            }

            // Extract checksum from packet
            buffers[packet.bufferId].crc = packet.crc;

            // Mark producer as owner of this buffer
            buffers[packet.bufferId].owner = 0U;
        }

        // Wait for fence returned by consumer before rendering
        if (!waitCudaFence(&cudaInfo, buf)) {
            goto done;
        }

        // CUDA rendering to buffer
        (void)memset(cudaInfo.bufCopy, (currFrame & 0xFF), cudaInfo.bufSize);

        cudaErr = cudaMemcpy2DAsync(buf->ptr,
                                    cudaInfo.bufSize,
                                    cudaInfo.bufCopy,
                                    cudaInfo.bufSize,
                                    cudaInfo.bufSize,
                                    1,
                                    cudaMemcpyHostToDevice,
                                    cudaInfo.stream);
        if (cudaSuccess != cudaErr) {
            fprintf(stderr, "Unable to initiate CUDA copy (%d)\n", cudaErr);
            goto done;
        }

        // Generate new fence for the sync object
        if (!signalCudaFence(&cudaInfo, buf)) {
            goto done;
        }

        // Wait for operation to finish and compute checksum
        // IMPORTANT NOTE:
        //   A normal stream application would not perform these steps.
        //   A checksum is not required for streaming, and waiting for
        //     operations to finish (which we only need because the
        //     checksum is calculated by the CPU) introduces bubbles
        //     in the hardware pipeline. A real application can rely on
        //     the generated NvSciSync fences for synchronization.
        //   These steps are only taken in this sample application
        //     because the consumer has no output visible to the user,
        //     so the checksum allows us to verify that the application
        //     is behaving properly.
        cudaDeviceSynchronize();
        buf->crc = GenerateCRC(cudaInfo.bufCopy,
                               1,
                               cudaInfo.bufSize,
                               cudaInfo.bufSize);

        fprintf(stderr, "Producer wrote frame %d in buffer %d\n",
                currFrame, currBuffer);

        // Mark buffer as owned by consumer now
        buf->owner = 1U;

        // Export buffer index, checksum, and fence for transmission over IPC
        packet.bufferId = currBuffer;
        packet.crc      = buf->crc;
        sciErr = NvSciSyncIpcExportFence(&buf->fence,
                                         ipcWrapper.endpoint,
                                         &packet.fenceDesc);
        if (NvSciError_Success != sciErr) {
            fprintf(stderr, "Unable to export producer fence (%x)\n", sciErr);
            goto done;
        }

        // Send buffer index and fence to consumer
        sciErr = ipcSend(&ipcWrapper, &packet, sizeof(packet));
        if (NvSciError_Success != sciErr) {
            fprintf(stderr,
                    "Failure to send buffer to consumer (%x)\n",
                    sciErr);
            goto done;
        }

        fprintf(stderr, "Producer finished frame %d in buffer %d\n",
                currFrame, currBuffer);

        // Advance buffer and frame
        currBuffer = (currBuffer + 1U) % totalBuffers;
        currFrame++;
    }


    // Success
    *(int*)arg = 0;
done:
    // Free CUDA resources
    deinitCudaBuffer(buffers, totalBuffers);
    deinitCuda(&cudaInfo);

    // Free NvSci objects
    if (NULL != producerSignalAttrs)
        NvSciSyncAttrListFree(producerSignalAttrs);
    if (NULL != consumerWaitAttrs)
        NvSciSyncAttrListFree(consumerWaitAttrs);
    if (NULL != sendWaitListDesc)
        NvSciSyncAttrListFreeDesc(sendWaitListDesc);
    if (NULL != producerWaitAttrs)
        NvSciSyncAttrListFree(producerWaitAttrs);
    if (NULL != prodToConsAttrs)
        NvSciSyncAttrListFree(prodToConsAttrs);
    if (NULL != syncConflictAttrs)
        NvSciSyncAttrListFree(syncConflictAttrs);
    if (NULL != producerSignalObj)
        NvSciSyncObjFree(producerSignalObj);
    if (NULL != sendObjAndListDesc)
        NvSciSyncAttrListAndObjFreeDesc(sendObjAndListDesc);
    if (NULL != producerWaitObj)
        NvSciSyncObjFree(producerWaitObj);
    if (NULL != producerWriteAttrs)
        NvSciBufAttrListFree(producerWriteAttrs);
    if (NULL != consumerReadAttrs)
        NvSciBufAttrListFree(consumerReadAttrs);
    if (NULL != combinedBufAttrs)
        NvSciBufAttrListFree(combinedBufAttrs);
    if (NULL != sendBufListDesc)
        NvSciBufAttrListFreeDesc(sendBufListDesc);

    // Free malloc'd resources
    if (NULL != recvWaitListDesc)
        free(recvWaitListDesc);
    if (NULL != recvObjAndListDesc)
        free(recvObjAndListDesc);
    if (NULL != consumerReadAttrsDesc)
        free(consumerReadAttrsDesc);

    fprintf(stderr, "Producer exiting\n");
    return NULL;
}
