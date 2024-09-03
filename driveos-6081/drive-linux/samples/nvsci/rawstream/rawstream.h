/*
 * Copyright (c) 2020-2021 NVIDIA Corporation.  All Rights Reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation
 * is strictly prohibited.
 */

#ifndef _rawstream_h
#define _rawstream_h

#include <unistd.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <nvscisync.h>
#include <nvscibuf.h>
#include "cuda.h"
#include "cuda_runtime_api.h"

// Constants controlling configuration
#define totalFrames   32U
#define totalBuffers   4U

// Sync/Buf modules
extern NvSciSyncModule   syncModule;
extern NvSciBufModule    bufModule;

// Exchanged sync info
extern NvSciSyncAttrList producerSignalAttrs;
extern NvSciSyncAttrList consumerSignalAttrs;
extern NvSciSyncAttrList producerWaitAttrs;
extern NvSciSyncAttrList consumerWaitAttrs;
extern NvSciSyncAttrList prodToConsAttrs;
extern NvSciSyncAttrList consToProdAttrs;
extern NvSciSyncObj      consumerSignalObj;
extern NvSciSyncObj      producerSignalObj;
extern NvSciSyncObj      consumerWaitObj;
extern NvSciSyncObj      producerWaitObj;

// Exchanged buf info
extern NvSciBufAttrList  producerWriteAttrs;
extern NvSciBufAttrList  consumerReadAttrs;
extern NvSciBufAttrList  combinedBufAttrs;

// CUDA info common to producer and consumer
typedef struct {
    int                     deviceId;
    CUuuid                  uuid;
    cudaStream_t            stream;
    cudaExternalSemaphore_t signalerSem;
    cudaExternalSemaphore_t waiterSem;
    NvSciBufType            bufType;
    uint64_t                bufSize;
    uint8_t*                bufCopy;
} CudaClientInfo;

// List of buffers with status
typedef struct {
    // Buffer handle
    NvSciBufObj          obj;
    // CUDA external memory object
    cudaExternalMemory_t extMem;
    // Mapping into virtual memory
    uint8_t*             ptr;
    // Current owner (0 = producer, 1 = consumer)
    uint32_t             owner;
    // Fence to wait for
    NvSciSyncFence       fence;
    // Checksum for error checking
    uint32_t             crc;
} Buffer;
extern Buffer buffers[totalBuffers];

// packet data
// Note: The checksum is not, in general, needed in a real streaming
//       application. All that is required is something to identify
//       the buffer and provide the fences. See comments in the producer
//       and consumer for the reason for the checksum.
typedef struct {
    // buffer identifier
    uint32_t                          bufferId;
    // buffer checksum
    uint32_t                          crc;
    // Fence to wait for
    NvSciSyncFenceIpcExportDescriptor fenceDesc;
} Packet;

// IPC related info
typedef struct {
    // NvSciIpc handle
    NvSciIpcEndpoint            endpoint;
    // IPC channel info
    struct NvSciIpcEndpointInfo info;

    // QNX: Channel id to get event
    int32_t chId;
    // QNX: Connection id to send event in library
    int32_t connId;
    // Linux: IPC event fd
    int32_t ipcEventFd;
} IpcWrapper;
extern IpcWrapper ipcWrapper;

// CUDA data types
typedef struct cudaExternalSemaphoreHandleDesc cudaExternalSemaphoreHandleDesc;
typedef struct cudaExternalMemoryHandleDesc cudaExternalMemoryHandleDesc;
typedef struct cudaExternalMemoryBufferDesc cudaExternalMemoryBufferDesc;
typedef struct cudaExternalSemaphoreWaitParams cudaExternalSemaphoreWaitParams;
typedef struct cudaExternalSemaphoreSignalParams cudaExternalSemaphoreSignalParams;

// Utility functions
extern uint32_t GenerateCRC(uint8_t* data, uint32_t width, uint32_t height, uint32_t pitch);

// Thread functions
extern void* producerFunc(void*);
extern void* consumerFunc(void*);

// IPC functions
extern NvSciError ipcInit(const char* endpointName, IpcWrapper* ipcWrapper);
extern NvSciError ipcSend(IpcWrapper* ipcWrapper, const void* buf, const size_t size);
extern NvSciError ipcRecvFill(IpcWrapper* ipcWrapper, void* buf, const size_t size);
extern void ipcDeinit(IpcWrapper* ipcWrapper);

// CUDA-specific operations
extern bool setupCuda(CudaClientInfo* info);
extern bool setupCudaSync(CudaClientInfo* info,
                          NvSciSyncObj sciSignalObj,
                          NvSciSyncObj sciWaitObj);
extern bool setupCudaBufAttr(CudaClientInfo* info,
                             NvSciBufAttrList attrs);
extern bool setupCudaBuffer(CudaClientInfo* info,
                            Buffer* buf);
extern bool waitCudaFence(CudaClientInfo* info,
                          Buffer* buf);
extern bool signalCudaFence(CudaClientInfo* info,
                            Buffer* buf);
extern void deinitCuda(CudaClientInfo* info);
extern void deinitCudaBuffer(Buffer* buf, int num);

#endif // _rawstream_h
