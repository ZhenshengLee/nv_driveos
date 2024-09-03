/*
 * Copyright (c) 2020-2023 NVIDIA Corporation.  All Rights Reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation
 * is strictly prohibited.
 */

#include "rawstream.h"

void* consumerFunc(void* arg)
{
    CudaClientInfo cudaInfo;
    NvSciError     sciErr;
    int            cudaErr;

    *(int*)arg = 1;
    fprintf(stderr, "Consumer starting\n");

    // Do common cuda initialization
    if (!setupCuda(&cudaInfo)) {
        goto done;
    }

    // Create an empty sync attribute list for signaling permissions.
    sciErr = NvSciSyncAttrListCreate(syncModule, &consumerSignalAttrs);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Unable to create consumer signal attrs (%x)\n",
                sciErr);
        goto done;
    }

    // Query CUDA for attributes needed to signal syncs
    cudaErr = cudaDeviceGetNvSciSyncAttributes(consumerSignalAttrs,
                                               cudaInfo.deviceId,
                                               cudaNvSciSyncAttrSignal);
    if (cudaSuccess != cudaErr) {
        fprintf(stderr,
                "Could not query signal attributes from CUDA (%d)\n",
                cudaErr);
        goto done;
    }

    fprintf(stderr, "Consumer signal attributes established\n");

    // Create an empty sync attribute list for waiting permissions.
    sciErr = NvSciSyncAttrListCreate(syncModule, &consumerWaitAttrs);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Unable to create consumer wait attrs (%x)\n",
                sciErr);
        goto done;
    }

    // Query CUDA for attributes needed to wait for syncs
    cudaErr = cudaDeviceGetNvSciSyncAttributes(consumerWaitAttrs,
                                               cudaInfo.deviceId,
                                               cudaNvSciSyncAttrWait);
    if (cudaSuccess != cudaErr) {
        fprintf(stderr,
                "Could not query wait attributes from CUDA (%d)\n",
                cudaErr);
        goto done;
    }

    fprintf(stderr, "Consumer wait attributes established\n");

    // Export consumer's wait attributes to a form suitable for IPC
    size_t sendWaitAttrListSize = 0U;
    void* sendWaitListDesc = NULL;
    sciErr = NvSciSyncAttrListIpcExportUnreconciled(&consumerWaitAttrs,
                                                    1,
                                                    ipcWrapper.endpoint,
                                                    &sendWaitListDesc,
                                                    &sendWaitAttrListSize);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Unable to export consumer wait attrs (%x)\n",
                sciErr);
        goto done;
    }

    // Send the size of the consumer's wait attributes to the producer,
    //   so it knows how much data to expect
    sciErr = ipcSend(&ipcWrapper,
                     &sendWaitAttrListSize,
                     sizeof(sendWaitAttrListSize));
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Unable to send consumer wait attrs size (%x)\n",
                sciErr);
        goto done;
    }

    // Send the exported form of the consumer's wait attributes
    sciErr = ipcSend(&ipcWrapper,
                     sendWaitListDesc,
                     sendWaitAttrListSize);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr, "Unable to send consumer wait attrs (%x)\n", sciErr);
        goto done;
    }

    // Wait to receive the size of the producer's wait attributes
    size_t recvWaitAttrListSize = 0U;
    sciErr = ipcRecvFill(&ipcWrapper,
                         &recvWaitAttrListSize,
                         sizeof(recvWaitAttrListSize));
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Unable to recv producer wait attr size (%x)\n",
                sciErr);
        goto done;
    }

    // Allocate a buffer big enough to receive the producer's wait attributes
    void* recvWaitListDesc = malloc(recvWaitAttrListSize);
    if (recvWaitListDesc == NULL) {
        sciErr = NvSciError_InsufficientMemory;
        fprintf(stderr,
                "Sync attr allocation failed (%x)\n",
                sciErr);
        goto done;
    }

    // Wait to receive producer's wait attributes
    sciErr = ipcRecvFill(&ipcWrapper,
                         recvWaitListDesc,
                         recvWaitAttrListSize);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Unable to recv producer wait attrs (%x)\n",
                sciErr);
        goto done;
    }

    // Convert the received producer wait attributes to an attribute list
    sciErr = NvSciSyncAttrListIpcImportUnreconciled(syncModule,
                                                    ipcWrapper.endpoint,
                                                    recvWaitListDesc,
                                                    recvWaitAttrListSize,
                                                    &producerWaitAttrs);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Unable to import producer wait attrs (%x)\n",
                sciErr);
        goto done;
    }

    // Get combined attributes for consumer to producer signaling
    NvSciSyncAttrList syncAllAttrs[2], syncConflictAttrs;
    syncAllAttrs[0] = consumerSignalAttrs;
    syncAllAttrs[1] = producerWaitAttrs;
    sciErr = NvSciSyncAttrListReconcile(syncAllAttrs, 2,
                                        &consToProdAttrs, &syncConflictAttrs);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Can't merge consumer->producer attrs (%x)\n",
                sciErr);
        goto done;
    }

    // Allocate consumer to producer sync object
    sciErr = NvSciSyncObjAlloc(consToProdAttrs, &consumerSignalObj);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Can't allocate consumer->producer sync (%x)\n",
                sciErr);
        goto done;
    }

    // Export sync attributes and object to a form suitable for IPC
    void* sendObjAndListDesc = NULL;
    size_t sendObjAndListSize = 0U;
    sciErr = NvSciSyncIpcExportAttrListAndObj(consumerSignalObj,
                                              NvSciSyncAccessPerm_WaitOnly,
                                              ipcWrapper.endpoint,
                                              &sendObjAndListDesc,
                                              &sendObjAndListSize);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Can't export consumer->producer sync description (%x)\n",
                sciErr);
        goto done;
    }

    // Send the size of the sync description to the producer,
    //   so it knows how much data to expect
    sciErr = ipcSend(&ipcWrapper, &sendObjAndListSize, sizeof(size_t));
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Can't send consumer->producer sync description size(%x)\n",
                sciErr);
        goto done;
    }

    // Send the sync description to the producer
    sciErr = ipcSend(&ipcWrapper, sendObjAndListDesc, sendObjAndListSize);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Can't send consumer->producer sync description (%x)\n",
                sciErr);
        goto done;
    }

    // Wait to receive the size of the producer->consumer sync desription
    size_t recvObjAndListSize = 0U;
    sciErr = ipcRecvFill(&ipcWrapper,
                         &recvObjAndListSize,
                         sizeof(size_t));
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Can't recv producer->consumer sync description size (%x)\n",
                sciErr);
        goto done;
    }

    // Allocate a buffer big enough to receive the description
    void* recvObjAndListDesc = malloc(recvObjAndListSize);
    if (NULL == recvObjAndListDesc) {
        sciErr = NvSciError_InsufficientMemory;
        fprintf(stderr, "Sync description allocation failed (%x)\n", sciErr);
        goto done;
    }

    // Wait to receive producer->consumer sync description
    sciErr = ipcRecvFill(&ipcWrapper,
                         recvObjAndListDesc,
                         recvObjAndListSize);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Can't receive producer->consumer sync description (%x)\n",
                sciErr);
        goto done;
    }

    // Convert the received producer->consumer sync description to a
    //   sync attribute list and object
    sciErr = NvSciSyncIpcImportAttrListAndObj(syncModule,
                                              ipcWrapper.endpoint,
                                              recvObjAndListDesc,
                                              recvObjAndListSize,
                                              &consumerWaitAttrs,
                                              1,
                                              NvSciSyncAccessPerm_WaitOnly,
                                              ipcWrapper.endpoint,
                                              &consumerWaitObj);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Can't import producer->consumer sync (%x)\n",
                sciErr);
        goto done;
    }

    // Validate imported reconciled attribute list and object
    {
        NvSciSyncAttrList consumerWaitList;

        sciErr = NvSciSyncAttrListValidateReconciledAgainstAttrs(
            consToProdAttrs,
            NULL,
            0,
            NvSciSyncAccessPerm_SignalOnly);
        if (NvSciError_Success != sciErr) {
            fprintf(
                stderr,
                "Validation of consToProd list failed: %x\n", sciErr);
            goto done;
        }

        sciErr = NvSciSyncObjGetAttrList(consumerWaitObj,
                                         &consumerWaitList);
        if (NvSciError_Success != sciErr) {
            fprintf(stderr,
                    "Can't get the reconciled list from consumer wait object (%x)\n",
                    sciErr);
            goto done;
        }

        sciErr = NvSciSyncAttrListValidateReconciledAgainstAttrs(
            consumerWaitList,
            NULL,
            0,
            NvSciSyncAccessPerm_WaitOnly);
        if (NvSciError_Success != sciErr) {
            fprintf(
                stderr,
                "Validation of imported reconciled consumer wait list failed: %x\n",
                sciErr);
            goto done;
        }

        sciErr = NvSciSyncObjValidate(consumerWaitObj);
        if (NvSciError_Success != sciErr) {
            fprintf(
                stderr,
                "Validation of imported consumer wait object failed: %x\n",
                sciErr);
            goto done;
        }
    }

    // Set up CUDA sync objects, importing NvSciSync objects
    if (!setupCudaSync(&cudaInfo, consumerSignalObj, consumerWaitObj)) {
        goto done;
    }

    fprintf(stderr, "Consumer exchanged sync objects with producer\n");

    // Create an empty buffer attribute list for consumer buffers
    sciErr = NvSciBufAttrListCreate(bufModule, &consumerReadAttrs);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Unable to create consumer buffer attrs (%x)\n",
                sciErr);
        goto done;
    }

    // Fill consumer buffer attribute list with values
    NvSciBufAttrKeyValuePair bufKeyValue[4];
    NvSciRmGpuId gpuId;
    memcpy(&gpuId.bytes, &cudaInfo.uuid.bytes, sizeof(cudaInfo.uuid.bytes));
    bufKeyValue[0].key   = NvSciBufGeneralAttrKey_GpuId;
    bufKeyValue[0].value = &gpuId;
    bufKeyValue[0].len   = sizeof(gpuId);
    NvSciBufType bufType = NvSciBufType_RawBuffer;
    bufKeyValue[1].key   = NvSciBufGeneralAttrKey_Types;
    bufKeyValue[1].value = &bufType;
    bufKeyValue[1].len   = sizeof(bufType);
    NvSciBufAttrValAccessPerm bufPerm = NvSciBufAccessPerm_Readonly;
    bufKeyValue[2].key   = NvSciBufGeneralAttrKey_RequiredPerm;
    bufKeyValue[2].value = &bufPerm;
    bufKeyValue[2].len   = sizeof(bufPerm);
    bool   bufAccessFlag = true;
    bufKeyValue[3].key   = NvSciBufGeneralAttrKey_NeedCpuAccess;
    bufKeyValue[3].value = &bufAccessFlag;
    bufKeyValue[3].len   = sizeof(bufAccessFlag);

    sciErr = NvSciBufAttrListSetAttrs(consumerReadAttrs, bufKeyValue, 4);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr, "Unable to fill consumer buffer attrs (%x)\n", sciErr);
        goto done;
    }

    fprintf(stderr, "Consumer buffer attributes established\n");

    // Export consumer buffer attributes in a form suitable for IPC
    size_t consumerReadAttrsSize = 0U;
    void* consumerReadAttrsDesc = NULL;
    sciErr = NvSciBufAttrListIpcExportUnreconciled(&consumerReadAttrs,
                                                   1,
                                                   ipcWrapper.endpoint,
                                                   &consumerReadAttrsDesc,
                                                   &consumerReadAttrsSize);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Unable to export consumer buffer attrs (%x)\n",
                sciErr);
        goto done;
    }

    // Send size of consumer buffer attributes
    sciErr = ipcSend(&ipcWrapper,
                     &consumerReadAttrsSize,
                     sizeof(consumerReadAttrsSize));
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Unable to send consumer buffer attrs size (%x)\n",
                sciErr);
        goto done;
    }

    // Send consumer buffer attributes
    sciErr = ipcSend(&ipcWrapper,
                     consumerReadAttrsDesc,
                     consumerReadAttrsSize);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr, "Unable to send consumer buffer attrs (%x)\n", sciErr);
        goto done;
    }

    // Wait to receive the size of the combined buffer attributes
    size_t recvBufListSize = 0U;
    sciErr = ipcRecvFill(&ipcWrapper,
                         &recvBufListSize,
                         sizeof(recvBufListSize));
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Unable to receive combinedbuffer attr size (%x)\n",
                sciErr);
        goto done;
    }

    // Allocate a buffer big enough to receive the combined buffer attributes
    void* recvBufListDesc = malloc(recvBufListSize);
    if (NULL == recvBufListDesc) {
        sciErr = NvSciError_InsufficientMemory;
        fprintf(stderr, "Buffer attr allocation failed(%x)\n", sciErr);
        goto done;
    }

    // Receive the combined buffer attributes
    sciErr = ipcRecvFill(&ipcWrapper,
                         recvBufListDesc,
                         recvBufListSize);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Unable to recv combined buffer attr desc (%x)\n",
                sciErr);
        goto done;
    }

    // Convert the combined buffer attributes to an attribute list
    sciErr = NvSciBufAttrListIpcImportReconciled(bufModule,
                                                 ipcWrapper.endpoint,
                                                 recvBufListDesc,
                                                 recvBufListSize,
                                                 NULL,
                                                 0,
                                                 &combinedBufAttrs);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr,
                "Unable to import combined buffer attr (%x)\n",
                sciErr);
        goto done;
    }

    // Extract attributes needed by CUDA
    if (!setupCudaBufAttr(&cudaInfo, combinedBufAttrs)) {
        goto done;
    }

    // Receive all buffers
    for (uint32_t i=0U; i<totalBuffers; ++i) {

        Buffer* buf = &buffers[i];

        // Receive the next buffer description
        NvSciBufObjIpcExportDescriptor objDesc;
        sciErr = ipcRecvFill(&ipcWrapper,
                             &objDesc,
                             sizeof(NvSciBufObjIpcExportDescriptor));
        if (NvSciError_Success != sciErr) {
            fprintf(stderr,
                    "Unable to recv buffer %d from producer (%x)\n",
                    i, sciErr);
            goto done;
        }

        // Convert buffer description to a buffer object
        sciErr = NvSciBufObjIpcImport(ipcWrapper.endpoint,
                                      &objDesc,
                                      combinedBufAttrs,
                                      NvSciBufAccessPerm_Readonly,
                                      1000U,
                                      &buf->obj);
        if (NvSciError_Success != sciErr) {
            fprintf(stderr,
                    "Unable to import buffer %d from producer (%x)\n",
                    i, sciErr);
            goto done;
        }

        // Import the buffer into CUDA
        if (!setupCudaBuffer(&cudaInfo, buf)) {
            goto done;
        }

        // Validate handles before entering runtime phase
        {
            NvSciBufAttrList reconciledList;

            sciErr = NvSciBufObjGetAttrList(buf->obj, &reconciledList);
            if (NvSciError_Success != sciErr) {
                fprintf(stderr,
                        "Can't get the reconciled list from NvSciBufObj %d (%x)\n",
                        i, sciErr);
                goto done;
            }

            NvSciBufAttrListValidateReconciledAgainstAttrs(reconciledList, bufKeyValue, 4);
            if (NvSciError_Success != sciErr) {
                fprintf(stderr,
                        "Validation of combinedBufAttrs list failed: %x\n", sciErr);
                goto done;
            }

            sciErr = NvSciBufObjValidate(buf->obj);
            if (NvSciError_Success != sciErr) {
                fprintf(stderr,
                        "Validation of imported buffer %d failed (%x)\n",
                        i, sciErr);
                goto done;
            }
        }
    }

    fprintf(stderr, "Consumer buffers received and established\n");

    // Receive all frames
    uint32_t currFrame  = 0;
    uint32_t currBuffer = 0;
    Packet packet;
    while (currFrame < totalFrames) {
        fprintf(stderr, "Consumer starting frame %d in buffer %d\n",
                currFrame, currBuffer);
        Buffer* buf = &buffers[currBuffer];

        // Wait for buffer to be available
        while (buf->owner != 1U) {

            // Wait for next presented buffer
            sciErr = ipcRecvFill(&ipcWrapper, &packet, sizeof(packet));
            if (NvSciError_Success != sciErr) {
                fprintf(stderr,
                        "Failure to recv buffer from producer (%x)\n",
                        sciErr);
                goto done;
            }

            // Import transmitted fence description to a fence
            sciErr = NvSciSyncIpcImportFence(consumerWaitObj,
                                             &packet.fenceDesc,
                                             &buffers[packet.bufferId].fence);
            if (NvSciError_Success != sciErr) {
                fprintf(stderr,
                        "Failure to import fence from producer (%x)\n",
                        sciErr);
                goto done;
            }

            // copy CRC data from packet
            buffers[packet.bufferId].crc = packet.crc;

            // Mark consumer as owner of this buffer
            buffers[packet.bufferId].owner = 1U;
        }

        // Wait for fence generated by producer before reading
        if (!waitCudaFence(&cudaInfo, buf)) {
            goto done;
        }

        // Read the buffer to the local copy
        cudaErr = cudaMemcpy2DAsync(cudaInfo.bufCopy,
                                    cudaInfo.bufSize,
                                    buf->ptr,
                                    cudaInfo.bufSize,
                                    cudaInfo.bufSize,
                                    1,
                                    cudaMemcpyDeviceToHost,
                                    cudaInfo.stream);
        if (cudaSuccess != cudaErr) {
            fprintf(stderr, "Unable to initiate CUDA copy (%d)\n", cudaErr);
            goto done;
        }

        // Wait for operation to finish, then compute and compare checksum
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
        uint32_t crc = GenerateCRC(cudaInfo.bufCopy,
                                   1,
                                   cudaInfo.bufSize,
                                   cudaInfo.bufSize);
        if (buf->crc != crc) {
            fprintf(stderr, "Checksums don't match (%x vs %x)\n",
                    crc, buf->crc);
            goto done;
        }

        fprintf(stderr, "Consumer read frame %d in buffer %d\n",
                currFrame, currBuffer);

        // Generate new fence indicating when reading has finished
        if (!signalCudaFence(&cudaInfo, buf)) {
            goto done;
        }

        // Mark buffer as owned by producer now
        buf->owner = 0U;

        // Export buffer index and fence for transmission over IPC
        // There is no checksum for the return trip.
        packet.bufferId = currBuffer;
        packet.crc      = 0U;
        sciErr = NvSciSyncIpcExportFence(&buf->fence,
                                         ipcWrapper.endpoint,
                                         &packet.fenceDesc);
        if (NvSciError_Success != sciErr) {
            fprintf(stderr, "Unable to export consumer fence (%x)\n", sciErr);
            goto done;
        }

        // Send buffer index and fence to producer
        sciErr = ipcSend(&ipcWrapper, &packet, sizeof(packet));
        if (NvSciError_Success != sciErr) {
            fprintf(stderr,
                    "Failure to send buffer to producer (%x)\n",
                    sciErr);
            goto done;
        }

        fprintf(stderr, "Consumer finished frame %d in buffer %d\n",
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
    if (NULL != consumerSignalAttrs)
        NvSciSyncAttrListFree(consumerSignalAttrs);
    if (NULL != consumerWaitAttrs)
        NvSciSyncAttrListFree(consumerWaitAttrs);
    if (NULL != sendWaitListDesc)
        NvSciSyncAttrListFreeDesc(sendWaitListDesc);
    if (NULL != producerWaitAttrs)
        NvSciSyncAttrListFree(producerWaitAttrs);
    if (NULL != consToProdAttrs)
        NvSciSyncAttrListFree(consToProdAttrs);
    if (NULL != syncConflictAttrs)
        NvSciSyncAttrListFree(syncConflictAttrs);
    if (NULL != consumerSignalObj)
        NvSciSyncObjFree(consumerSignalObj);
    if (NULL != sendObjAndListDesc)
        NvSciSyncAttrListAndObjFreeDesc(sendObjAndListDesc);
    if (NULL != consumerWaitObj)
        NvSciSyncObjFree(consumerWaitObj);
    if (NULL != consumerReadAttrs)
        NvSciBufAttrListFree(consumerReadAttrs);
    if (NULL != consumerReadAttrsDesc)
        NvSciBufAttrListFreeDesc(consumerReadAttrsDesc);
    if (NULL != combinedBufAttrs)
        NvSciBufAttrListFree(combinedBufAttrs);

    // Free malloc'd resources
    if (NULL != recvWaitListDesc)
        free(recvWaitListDesc);
    if (NULL != recvObjAndListDesc)
        free(recvObjAndListDesc);
    if (NULL != recvBufListDesc)
        free(recvBufListDesc);

    fprintf(stderr, "Consumer exiting\n");
    return NULL;
}
