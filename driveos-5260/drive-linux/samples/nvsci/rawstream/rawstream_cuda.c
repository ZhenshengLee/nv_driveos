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

// Initialize CUDA info
bool setupCuda(CudaClientInfo* info)
{
    int cudaErr;

    info->deviceId = 0;
    info->stream = NULL;
    info->signalerSem = NULL;
    info->waiterSem = NULL;
    info->bufCopy = NULL;

    int numOfGPUs = 0;
    cudaErr = cudaGetDeviceCount(&numOfGPUs);
    if (cudaSuccess != cudaErr) {
        fprintf(stderr, "Failed to get compute-capable devices (%d)\n", cudaErr);
        return false;
    }

    cudaErr = cudaSetDevice(info->deviceId);
    if (cudaSuccess != cudaErr) {
        fprintf(stderr, "Failed to set CUDA device (%d)\n", cudaErr);
        return false;
    }

    cudaErr = cuDeviceGetUuid(&info->uuid, info->deviceId);
    if (CUDA_SUCCESS != cudaErr) {
        fprintf(stderr, "Failed to query CUDA UUID (%d)\n", cudaErr);
        return false;
    }

    return true;
}

// Create CUDA sync objects and map to imported NvSciSync
bool setupCudaSync(CudaClientInfo* info,
                   NvSciSyncObj sciSignalObj,
                   NvSciSyncObj sciWaitObj)
{
    cudaExternalSemaphoreHandleDesc extSemDesc;
    int cudaErr;

    // Create CUDA stream for signaling and waiting
    cudaErr = cudaStreamCreateWithFlags(&info->stream,
                                        cudaStreamNonBlocking);
    if (cudaSuccess != cudaErr) {
        fprintf(stderr,
                "Unable to create CUDA stream (%d)\n",
                cudaErr);
        return false;
    }

    // Import signaler sync object to CUDA semaphore
    memset(&extSemDesc, 0, sizeof(extSemDesc));
    extSemDesc.type = cudaExternalSemaphoreHandleTypeNvSciSync;
    extSemDesc.handle.nvSciSyncObj = sciSignalObj;
    cudaErr = cudaImportExternalSemaphore(&info->signalerSem, &extSemDesc);
    if (cudaSuccess != cudaErr) {
        fprintf(stderr,
                "Unable to import signal sync object to CUDA (%d)\n",
                cudaErr);
        return false;
    }

    // Import waiter sync object to CUDA semaphore
    memset(&extSemDesc, 0, sizeof(extSemDesc));
    extSemDesc.type = cudaExternalSemaphoreHandleTypeNvSciSync;
    extSemDesc.handle.nvSciSyncObj = sciWaitObj;
    cudaErr = cudaImportExternalSemaphore(&info->waiterSem, &extSemDesc);
    if (cudaSuccess != cudaErr) {
        fprintf(stderr,
                "Unable to import wait sync object to CUDA (%d)\n",
                cudaErr);
        return false;
    }

    return true;
}

// Extract info from buffer attributes needed by CUDA
bool setupCudaBufAttr(CudaClientInfo* info,
                      NvSciBufAttrList attrs)
{
    NvSciBufAttrKeyValuePair queryKeyValue[] = {
        { NvSciBufGeneralAttrKey_Types,  NULL, 0 },
        { NvSciBufRawBufferAttrKey_Size, NULL, 0 },
    };

    NvSciError sciErr = NvSciBufAttrListGetAttrs(attrs, queryKeyValue, 2);
    if (NvSciError_Success != sciErr) {
        fprintf(stderr, "Unable to query buffer type/size (%x)\n", sciErr);
        return false;
    }

    // TODO: Original sample queries BufType but doesn't seem to do anything
    //       with it. Might not be needed.
    info->bufType = *((NvSciBufType*)(queryKeyValue[0].value));
    info->bufSize = *((uint64_t*)(queryKeyValue[1].value));

    // Allocate storage for a copy of the buffer contents
    info->bufCopy = (uint8_t*)malloc(info->bufSize);
    if (NULL == info->bufCopy) {
        fprintf(stderr, "Unable to allocate buffer copy\n");
        return false;
    }
    (void)memset(info->bufCopy, 0, info->bufSize);

    return true;
}

// Import NvSciBuf into CUDA
bool setupCudaBuffer(CudaClientInfo* info,
                     Buffer* buf)
{
    int cudaErr;

    // Import buffer to cuda as external memory
    cudaExternalMemoryHandleDesc memHandleDesc;
    memset(&memHandleDesc, 0, sizeof(memHandleDesc));
    memHandleDesc.type = cudaExternalMemoryHandleTypeNvSciBuf;
    memHandleDesc.handle.nvSciBufObject = buf->obj;
    memHandleDesc.size = info->bufSize;

    cudaErr = cudaImportExternalMemory(&buf->extMem, &memHandleDesc);
    if (cudaSuccess != cudaErr) {
        fprintf(stderr,
                "Unable to import buffer to CUDA (%d)\n",
                cudaErr);
        return false;
    }

    // Map to cuda memory buffer
    cudaExternalMemoryBufferDesc bufferDesc;
    memset(&bufferDesc, 0, sizeof(bufferDesc));
    bufferDesc.size = info->bufSize;
    bufferDesc.offset = 0;
    cudaErr = cudaExternalMemoryGetMappedBuffer((void *)&buf->ptr,
                                                buf->extMem,
                                                &bufferDesc);
    if (cudaSuccess != cudaErr) {
        fprintf(stderr, "Unable to map CUDA buffer (%d)\n", cudaErr);
        return false;
    }

    return true;
}

// Tell CUDA to wait for the fence associated with a buffer
bool waitCudaFence(CudaClientInfo* info,
                   Buffer* buf)
{
    cudaExternalSemaphoreWaitParams waitParams;
    memset(&waitParams, 0, sizeof(waitParams));
    waitParams.params.nvSciSync.fence = &buf->fence;
    waitParams.flags = 0;
    int cudaErr = cudaWaitExternalSemaphoresAsync(&info->waiterSem,
                                                  &waitParams,
                                                  1,
                                                  info->stream);
    if (cudaSuccess != cudaErr) {
        fprintf(stderr, "Unable to wait for fence (%d)\n", cudaErr);
        return false;
    }

    NvSciSyncFenceClear(&buf->fence);

    return true;
}

// Tell CUDA to generate a fence for a buffer
bool signalCudaFence(CudaClientInfo* info,
                     Buffer* buf)
{
    cudaExternalSemaphoreSignalParams signalParams;
    memset(&signalParams, 0, sizeof(signalParams));
    signalParams.params.nvSciSync.fence = &buf->fence;
    signalParams.flags = 0;

    int cudaErr = cudaSignalExternalSemaphoresAsync(&info->signalerSem,
                                                    &signalParams,
                                                    1,
                                                    info->stream);
    if (cudaSuccess != cudaErr) {
        fprintf(stderr, "Unable to signal fence (%d)\n", cudaErr);
        return false;
    }

    return true;
}

void deinitCuda(CudaClientInfo* info)
{
    if (NULL != info->bufCopy) {
        free(info->bufCopy);
        info->bufCopy = NULL;
    }
    if (NULL != info->signalerSem) {
        (void)cudaDestroyExternalSemaphore(info->signalerSem);
        info->signalerSem = NULL;
    }
    if (NULL != info->waiterSem) {
        (void)cudaDestroyExternalSemaphore(info->waiterSem);
        info->waiterSem = NULL;
    }
    if (NULL != info->stream) {
        (void)cudaStreamDestroy(info->stream);
        info->stream = NULL;
    }
}

void deinitCudaBuffer(Buffer* buf, int num)
{
    int i;
    for (i = 0; i < num; ++i) {
        if (NULL != buf[i].ptr)
            cudaFree(buf[i].ptr);
        if (NULL != buf[i].extMem)
            (void)cudaDestroyExternalMemory(buf[i].extMem);
        if (NULL != buf[i].obj)
            NvSciBufObjFree(buf[i].obj);
    }
}

