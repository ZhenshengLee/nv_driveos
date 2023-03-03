/* NvSciStream Event Loop Driven Sample App - consumer block for use case 2
 *
 * Copyright (c) 2021-2022 NVIDIA Corporation. All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software, related documentation and any
 * modifications thereto. Any use, reproduction, disclosure or distribution
 * of this software and related documentation without an express license
 * agreement from NVIDIA Corporation is strictly prohibited.
 */

/*
 * This implements the consumer for use case 2: nvmedia to cuda streaming
 */

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include "nvscibuf.h"
#include "nvscisync.h"
#include "nvscistream.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "block_info.h"
#include "event_loop.h"
#include "usecase2.h"
#include "util.h"

/* Internal data structure used to track packets */
typedef struct {
    /* The packet handle use for NvSciStream functions */
    NvSciStreamPacket       handle;
    /* NvSci buffer object for the packet's data buffer */
    NvSciBufObj             dataObj;
    /* CUDA external memory handle for the data buffer */
    cudaExternalMemory_t    dataExtMem;
    /* CUDA device memory pointer for the data buffer */
    void*                   dataDevPtr;
    /* Local system memory buffer used as the target for CUDA operations */
    uint8_t*                dataDstPtr;
} ConsPacket;

/* Internal data used by the consumer block */
typedef struct {
    /* Common block info */
    BlockData               common;
    /* CUDA device ID and UUID */
    int32_t                 cudaDeviceId;
    CUuuid                  cudaUuid;
    /* CUDA consumer stream */
    cudaStream_t            cudaStream;

    /* CUDA sync attributes required for signaling */
    NvSciSyncAttrList       signalAttr;
    /* CUDA sync attributes required for waiting */
    NvSciSyncAttrList       waiterAttr;
    /* Sync object for CUDA to signal after processing data */
    NvSciSyncObj            signalObj;
    /* CUDA semaphore mapped to sync object */
    cudaExternalSemaphore_t signalSem;

    /* Sync object to wait for before processing data */
    NvSciSyncObj            waiterObj;
    /* CUDA semaphore mapped to sync object */
    cudaExternalSemaphore_t waiterSem;

    /* Size for data buffer after reconciling all requirements */
    uint64_t                dataSize;
    /* Number of packets provided by pool */
    uint32_t                numPacket;
    /* Information about each packet */
    ConsPacket              packets[MAX_PACKETS];

    /* Number of payloads processed so far */
    uint32_t                counter;
} ConsData;

/* Free up consumer block resources */
static void deleteConsumer(
    ConsData* consData)
{
    /* Destroy block */
    if (consData->common.block != 0) {
        (void)NvSciStreamBlockDelete(consData->common.block);
        consData->common.block = 0;
    }

    ConsPacket *packet = NULL;
    for (uint32_t i=0;i<consData->numPacket; i++)
    {
        packet = &consData->packets[i];
        if (packet->handle != NvSciStreamPacket_Invalid)
        {
            (void)cudaFree(packet->dataDevPtr);
            if (packet->dataExtMem) {
                (void)cudaDestroyExternalMemory(packet->dataExtMem);
                packet->dataExtMem = 0;
            }

            if (packet->dataDstPtr) {
                free(packet->dataDstPtr);
                packet->dataDstPtr = NULL;
            }

            /* Free buffer objects */
            if (packet->dataObj) {
                NvSciBufObjFree(packet->dataObj);
                packet->dataObj = NULL;
            }
        }
    }

    if (consData->waiterObj != NULL) {
        (void)cudaDestroyExternalSemaphore(consData->waiterSem);
        consData->waiterSem = 0;
        NvSciSyncObjFree(consData->waiterObj);
        consData->waiterObj = NULL;
    }

    if (consData->signalObj != NULL) {
        (void)cudaDestroyExternalSemaphore(consData->signalSem);
        consData->signalSem = 0;
        NvSciSyncObjFree(consData->signalObj);
        consData->signalObj = NULL;
    }

    /* Free data */
    free(consData);
}

/* Handle initialization of CUDA resources for consumer */
static int32_t handleConsumerInit(
    ConsData* consData)
{
    if (opts.endInfo) {
        /* Query endpoint info from producer */
        uint32_t size = INFO_SIZE;
        char info[INFO_SIZE];
        NvSciError err = NvSciStreamBlockUserInfoGet(
                            consData->common.block,
                            NvSciStreamBlockType_Producer, 0U,
                            ENDINFO_NAME_PROC,
                            &size, &info);
        if (NvSciError_Success == err) {
            printf("Producer info: %s\n", info);
        } else if (NvSciError_StreamInfoNotProvided == err) {
            printf("Info not provided by the producer\n");
        } else {
            printf("Failed (%x) to query the producer info\n", err);
            return 0;
        }
    }

    /*
    * Init CUDA
    */
    int32_t     cudaRtErr;
    CUresult    cudaErr;
    /* Get stack limit */
    size_t      unused;

    cudaRtErr = cudaDeviceGetLimit(&unused, cudaLimitStackSize);
    if (cudaSuccess != cudaRtErr) {
        printf("Failed (%x) to get CUDA device limit\n", cudaRtErr);
        return 0;
    }

    /* Set CUDA device */
    consData->cudaDeviceId = 0;
    cudaRtErr = cudaSetDevice(consData->cudaDeviceId);
    if (cudaSuccess != cudaRtErr) {
        printf("Failed (%x) to set CUDA device\n", cudaRtErr);
        return 0;
    }

    /* Get UUID for CUDA device */
    cudaErr = cuDeviceGetUuid(&consData->cudaUuid, consData->cudaDeviceId);
    if (CUDA_SUCCESS != cudaErr) {
        printf("Failed (%x) to get CUDA UUID\n", cudaErr);
        return 0;
    }

    /* Get CUDA streams to be used for asynchronous operation */
    cudaRtErr = cudaStreamCreateWithFlags(&consData->cudaStream,
                                          cudaStreamNonBlocking);
    if (cudaSuccess != cudaRtErr) {
        printf("Failed (%x) to create CUDA stream\n", cudaRtErr);
        return 0;
    }

    return 1;
}

/* Handle setup of supported buffer attributes */
static int32_t handleConsumerElemSupport(
    ConsData* consData)
{
    NvSciError             sciErr;
    uint32_t               bufName = ELEMENT_NAME_IMAGE;
    NvSciBufAttrList       bufAttr = NULL;

    /*
     * Data buffer requires read access by CPU and the GPU of the cuda
     *   device, and uses an image buffer. (Size is specified by producer.)
     */
    NvSciBufAttrValAccessPerm dataPerm = NvSciBufAccessPerm_Readonly;
    uint8_t dataCpu                    = 1U;
    NvSciRmGpuId dataGpu               = { 0 };
    NvSciBufType dataBufType           = NvSciBufType_Image;
    memcpy(&dataGpu.bytes, &consData->cudaUuid.bytes, sizeof(dataGpu.bytes));
    NvSciBufAttrKeyValuePair dataKeyVals[] = {
        { NvSciBufGeneralAttrKey_GpuId, &dataGpu, sizeof(dataGpu) },
        { NvSciBufGeneralAttrKey_Types, &dataBufType, sizeof(dataBufType) },
        { NvSciBufGeneralAttrKey_RequiredPerm, &dataPerm, sizeof(dataPerm) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &dataCpu, sizeof(dataCpu) }
    };

    /* Create and fill attribute list for data buffer */
    sciErr = NvSciBufAttrListCreate(sciBufModule, &bufAttr);
    if (NvSciError_Success != sciErr) {
        printf("Consumer failed (%x) to create data attribute list\n", sciErr);
        return 0;
    }
    sciErr = NvSciBufAttrListSetAttrs(bufAttr,
                                      dataKeyVals,
                                      sizeof(dataKeyVals) /
                                          sizeof(NvSciBufAttrKeyValuePair));
    if (NvSciError_Success != sciErr) {
        printf("Consumer failed (%x) to fill data attribute list\n", sciErr);
        return 0;
    }

    /*
     * Inform stream of the attributes
     *   Once sent, the attribute list is no longer needed
     */
    sciErr = NvSciStreamBlockElementAttrSet(consData->common.block,
                                            bufName, bufAttr);
    if (NvSciError_Success != sciErr) {
        printf("Consumer failed (%x) to send element attribute\n", sciErr);
        return 0;
    }
    NvSciBufAttrListFree(bufAttr);

    /* Indicate that all element information has been exported */
    sciErr = NvSciStreamBlockSetupStatusSet(consData->common.block,
                                            NvSciStreamSetup_ElementExport,
                                            true);
    if (NvSciError_Success != sciErr) {
        printf("Consumer failed (%x) to complete element export\n", sciErr);
        return 0;
    }

    return 1;
}

/* Handle receipt of chosen element attributes */
static int32_t handleConsumerElemSetting(
    ConsData*              consData)
{
    NvSciError err;

    /*
     * This application does not need to query the element count, because we
     *   know it is always 1. But we do so anyways to show how it is done.
     */
    uint32_t count;
    err = NvSciStreamBlockElementCountGet(consData->common.block,
                                          NvSciStreamBlockType_Pool,
                                          &count);
    if (NvSciError_Success != err) {
        printf("Consumer failed (%x) to query element count\n", err);
        return 0;
    }
    if (1U != count) {
        printf("Consumer received unexpected element count (%d)\n", count);
        return 0;
    }

    /*
     * Query element type and attributes.
     *   For this simple use case, there is only one type, so we could pass
     *   NULL for that parameter. We query it only to show how its done.
     */
    uint32_t type;
    NvSciBufAttrList bufAttr;
    err = NvSciStreamBlockElementAttrGet(consData->common.block,
                                         NvSciStreamBlockType_Pool, 0U,
                                         &type, &bufAttr);
    if (NvSciError_Success != err) {
        printf("Consumer failed (%x) to query element attr\n", err);
        return 0;
    }

    /* Validate type */
    if (ELEMENT_NAME_IMAGE != type) {
        printf("Consumer received unknown element type (%x)\n", type);
        return 0;
    }

    /* Extract data size from attributes */
    NvSciBufAttrKeyValuePair keyVals[] = {
        { NvSciBufImageAttrKey_Size, NULL, 0 }
    };
    err = NvSciBufAttrListGetAttrs(bufAttr, keyVals, 1);
    if (NvSciError_Success != err) {
        printf("Consumer failed (%x) to obtain buffer size\n", err);
        return 0;
    }
    consData->dataSize = *((const uint64_t*)(keyVals[0].value));

    /* Don't need to keep attribute list */
    NvSciBufAttrListFree(bufAttr);

    /*
     * Indicate element will be used.
     * This is the default, and we can omit this call in most applications,
     *   but we illustrate its use for applications that only use some
     *   of the buffers.
     */
    err = NvSciStreamBlockElementUsageSet(consData->common.block, 0U, true);
    if (NvSciError_Success != err) {
        printf("Consumer failed (%x) to indicate element is used\n", err);
        return 0;
    }

    /* Indicate that element import is complete */
    err = NvSciStreamBlockSetupStatusSet(consData->common.block,
                                         NvSciStreamSetup_ElementImport,
                                         true);
    if (NvSciError_Success != err) {
        printf("Consumer failed (%x) to complete element import\n", err);
        return 0;
    }

    /* Set waiter attributes for the asynchronous element. */
    err = NvSciStreamBlockElementWaiterAttrSet(consData->common.block,
                                               0U, consData->waiterAttr);
    if (NvSciError_Success != err) {
        printf("Consumer failed (%x) to send waiter atts\n", err);
        return 0;
    }

    /* Once sent, the waiting attributes are no longer needed */
    NvSciSyncAttrListFree(consData->waiterAttr);
    consData->waiterAttr = NULL;

    /* Indicate that waiter attribute export is done. */
    err = NvSciStreamBlockSetupStatusSet(consData->common.block,
                                         NvSciStreamSetup_WaiterAttrExport,
                                         true);
    if (NvSciError_Success != err) {
        printf("Consumer failed (%x) to complete waiter attr export\n", err);
        return 0;
    }

    return 1;
}

/* Handle creation of a new packet */
static int32_t handleConsumerPacketCreate(
    ConsData*         consData)
{
    NvSciError err;

    /* Retrieve handle for packet pending creation */
    NvSciStreamPacket handle;
    err = NvSciStreamBlockPacketNewHandleGet(consData->common.block,
                                             &handle);
    if (NvSciError_Success != err) {
        printf("Consumer failed (%x) to retrieve handle for the new packet\n",
               err);
        return 0;
    }

    /* Make sure there is room for more packets */
    if (MAX_PACKETS <= consData->numPacket) {
        printf("Consumer exceeded max packets\n");
        err = NvSciStreamBlockPacketStatusSet(consData->common.block,
                                              handle,
                                              NvSciStreamCookie_Invalid,
                                              NvSciError_Overflow);
        if (NvSciError_Success != err) {
            printf("Consumer failed (%x) to inform pool of packet status\n",
                   err);
        }
        return 0;
    }

    /*
     * Allocate the next entry in the array for the new packet.
     *   Use the array entry for the cookie
     */
    ConsPacket* packet = &consData->packets[consData->numPacket++];
    packet->handle = handle;

    /* Retrieve all buffers and map into application
     *   Consumers can skip querying elements that they don’t use.
     *   This use case has only 1 element.
     */
    NvSciBufObj bufObj;
    err = NvSciStreamBlockPacketBufferGet(consData->common.block,
                                          handle,
                                          0U,
                                          &bufObj);
    if (NvSciError_Success != err) {
        printf("Consumer failed (%x) to retrieve buffer (%lx/0)\n",
               err, handle);
        return 0;
    }

    /* Save buffer object */
    packet->dataObj = bufObj;

    int32_t    cudaRtErr;

    /* Map in the buffer as CUDA external memory */
    struct cudaExternalMemoryHandleDesc memHandleDesc;
    memset(&memHandleDesc, 0, sizeof(memHandleDesc));
    memHandleDesc.type = cudaExternalMemoryHandleTypeNvSciBuf;
    memHandleDesc.handle.nvSciBufObject = bufObj;
    memHandleDesc.size = consData->dataSize;
    cudaRtErr = cudaImportExternalMemory(&packet->dataExtMem, &memHandleDesc);
    if (cudaSuccess != cudaRtErr) {
        printf("Consumer failed (%x) to map buffer as external mem\n",
               cudaRtErr);
        return 0;
    }

    /* Map in the buffer as CUDA device memory */
    struct cudaExternalMemoryBufferDesc memBufferDesc;
    memset(&memBufferDesc, 0, sizeof(memBufferDesc));
    memBufferDesc.size = consData->dataSize;
    memBufferDesc.offset = 0;
    cudaRtErr = cudaExternalMemoryGetMappedBuffer(&packet->dataDevPtr,
                                                  packet->dataExtMem,
                                                  &memBufferDesc);
    if (cudaSuccess != cudaRtErr) {
        printf("Consumer failed (%x) to map buffer as device mem\n", cudaRtErr);
        return 0;
    }

    /* Allocate normal memory to use as the target for the CUDA op */
    packet->dataDstPtr = (uint8_t*)malloc(consData->dataSize);
    if (NULL == packet->dataDstPtr) {
        printf("Consumer failed to allocate target buffer\n");
        return 0;
    }

    /* Fill in with initial values */
    memset(packet->dataDstPtr, 0xD0, consData->dataSize);

    /* Inform pool of success.
     *   Note: Could inform the pool of any of the failures above.
     */
    err = NvSciStreamBlockPacketStatusSet(consData->common.block,
                                          handle,
                                          (NvSciStreamCookie)packet,
                                          NvSciError_Success);
    if (NvSciError_Success != err) {
        printf("Consumer failed (%x) to inform pool of packet status\n",
               err);
        return 0;
    }

    return 1;
}

/* Handle deletion of packet */
static void handleConsumerPacketDelete(
    ConsData*         consData)
{
    /* Get the deleted packet cookie*/
    NvSciStreamCookie cookie;
    NvSciError err =
        NvSciStreamBlockPacketOldCookieGet(consData->common.block,
                                           &cookie);
    if (NvSciError_Success != err) {
        printf("Consumer failed (%x) to retrieve the deleted packet cookie\n",
               err);
    }

    /* Get packet pointer */
    ConsPacket* packet = (ConsPacket*)cookie;

    /* Free CUDA memory mapping */
    (void)cudaFree(packet->dataDevPtr);
    (void)cudaDestroyExternalMemory(packet->dataExtMem);
    if (packet->dataDstPtr) {
        free(packet->dataDstPtr);
    }

    /* Free buffer objects */
    if (packet->dataObj) {
        NvSciBufObjFree(packet->dataObj);
    }

    /* Clear out packet information */
    memset(packet, 0, sizeof(ConsPacket));
}

/* Handle setup of supported sync attributes */
static int32_t handleConsumerSyncSupport(
    ConsData* consData)
{
    NvSciError       sciErr;
    int32_t          cudaRtErr;

    /*
     * Create sync attribute list for signaling.
     *   This will be saved until we receive the producer's attributes
     */
    sciErr = NvSciSyncAttrListCreate(sciSyncModule, &consData->signalAttr);
    if (NvSciError_Success != sciErr) {
        printf("Consumer failed (%x) to allocate signal sync attrs\n", sciErr);
        return 0;
    }

    /* Have CUDA fill the signaling attribute list */
    cudaRtErr = cudaDeviceGetNvSciSyncAttributes(consData->signalAttr,
                                                 consData->cudaDeviceId,
                                                 cudaNvSciSyncAttrSignal);
    if (cudaSuccess != cudaRtErr) {
        printf("Consumer failed (%x) to fill signal sync attrs\n", cudaRtErr);
        return 0;
    }

    /* Create sync attribute list for waiting. */
    sciErr = NvSciSyncAttrListCreate(sciSyncModule, &consData->waiterAttr);
    if (NvSciError_Success != sciErr) {
        printf("Consumer failed (%x) to allocate waiter sync attrs\n", sciErr);
        return 0;
    }

    /* Have CUDA fill the waiting attribute list */
    cudaRtErr = cudaDeviceGetNvSciSyncAttributes(consData->waiterAttr,
                                                 consData->cudaDeviceId,
                                                 cudaNvSciSyncAttrWait);
    if (cudaSuccess != cudaRtErr) {
        printf("Consumer failed (%x) to fill waiter sync attrs\n", cudaRtErr);
        return 0;
    }

    return 1;
}

/* Handle creation and export of consumer sync object */
static int32_t handleConsumerSyncExport(
    ConsData*         consData)
{
    NvSciError        sciErr;
    uint32_t          cudaRtErr;

    /* Process waiter attrs from all elements.
     * This use case has only one element.
     */
    NvSciSyncAttrList waiterAttr = NULL;
    sciErr = NvSciStreamBlockElementWaiterAttrGet(consData->common.block,
                                                  0U, &waiterAttr);
    if (NvSciError_Success != sciErr) {
        printf("Consumer failed (%x) to query waiter attr\n", sciErr);
        return 0;
    }
    if (NULL == waiterAttr) {
        printf("Consumer received NULL waiter attr for data elem\n");
        return 0;
    }

    /* Indicate that waiter attribute import is done. */
    sciErr = NvSciStreamBlockSetupStatusSet(consData->common.block,
                                            NvSciStreamSetup_WaiterAttrImport,
                                            true);
    if (NvSciError_Success != sciErr) {
        printf("Consumer failed (%x) to complete waiter attr import\n",
               sciErr);
        return 0;
    }

    /*
     * Merge and reconcile producer sync attrs with ours.
     */
    NvSciSyncAttrList unreconciled[2] = {
        consData->signalAttr,
        waiterAttr };
    NvSciSyncAttrList reconciled = NULL;
    NvSciSyncAttrList conflicts = NULL;
    sciErr = NvSciSyncAttrListReconcile(unreconciled, 2,
                                        &reconciled, &conflicts);
    if (NvSciError_Success != sciErr) {
        printf("Consumer failed (%x) to reconcile sync attributes\n", sciErr);
        return 0;
    }

    /* Allocate sync object */
    sciErr = NvSciSyncObjAlloc(reconciled, &consData->signalObj);
    if (NvSciError_Success != sciErr) {
        printf("Consumer failed (%x) to allocate sync object\n", sciErr);
        return 0;
    }

    /* Free the attribute lists */
    NvSciSyncAttrListFree(consData->signalAttr);
    consData->signalAttr = NULL;
    NvSciSyncAttrListFree(waiterAttr);
    NvSciSyncAttrListFree(reconciled);

    /* Create CUDA semaphore for sync object */
    struct cudaExternalSemaphoreHandleDesc extSemDesc;
    memset(&extSemDesc, 0, sizeof(extSemDesc));
    extSemDesc.type = cudaExternalSemaphoreHandleTypeNvSciSync;
    extSemDesc.handle.nvSciSyncObj = consData->signalObj;
    cudaRtErr = cudaImportExternalSemaphore(&consData->signalSem, &extSemDesc);
    if (cudaSuccess != cudaRtErr) {
        printf("Consumer failed (%x) to map signal object to semaphore\n",
               cudaRtErr);
        return 0;
    }

    /* Send the sync object for each element */
    sciErr = NvSciStreamBlockElementSignalObjSet(consData->common.block,
                                                 0U,
                                                 consData->signalObj);
    if (NvSciError_Success != sciErr) {
        printf("Consumer failed (%x) to send sync object\n", sciErr);
        return 0;
    }

    /* Indicate that sync object export is complete */
    sciErr = NvSciStreamBlockSetupStatusSet(consData->common.block,
                                            NvSciStreamSetup_SignalObjExport,
                                            true);
    if (NvSciError_Success != sciErr) {
        printf("Consumer failed (%x) to complete signal obj export\n",
               sciErr);
        return 0;
    }

    return 1;
}

/* Handle import of producer sync object */
static int32_t handleConsumerSyncImport(
    ConsData*         consData)
{
    uint32_t          cudaRtErr;
    NvSciError        sciErr;
    NvSciSyncObj      waiterObj = NULL;
    /* Query sync object for asynchronous elements. */
    sciErr = NvSciStreamBlockElementSignalObjGet(consData->common.block,
                                                 0U, 0U,
                                                 &waiterObj);
    if (NvSciError_Success != sciErr) {
        printf("Consumer failed (%x) to query sync object\n", sciErr);
        return 0;
    }

    /* Save object */
    consData->waiterObj = waiterObj;

    /* If the waiter sync obj is NULL,
     * it means this element is ready to use when received.
     */
    if (NULL != waiterObj) {
        /* Create CUDA semaphore for sync object */
        struct cudaExternalSemaphoreHandleDesc extSemDesc;
        memset(&extSemDesc, 0, sizeof(extSemDesc));
        extSemDesc.type = cudaExternalSemaphoreHandleTypeNvSciSync;
        extSemDesc.handle.nvSciSyncObj = waiterObj;
        cudaRtErr = cudaImportExternalSemaphore(&consData->waiterSem,
                                                &extSemDesc);
        if (cudaSuccess != cudaRtErr) {
            printf("Consumer failed (%x) to map waiter object to semaphore\n",
                   cudaRtErr);
            return 0;
        }
    }

    /* Indicate that element import is complete */
    sciErr = NvSciStreamBlockSetupStatusSet(consData->common.block,
                                            NvSciStreamSetup_SignalObjImport,
                                            true);
    if (NvSciError_Success != sciErr) {
        printf("Consumer failed (%x) to complete signal obj import\n",
            sciErr);
        return 0;
    }

    return 1;
}

/* Handle processing of payloads */
static int32_t handleConsumerPayload(
    ConsData*         consData)
{
    NvSciError        sciErr;
    int32_t           cudaRtErr;

    /* Clear space to receive fence from producer for each element.
     * This use case only has one element per packet.
     */
    NvSciSyncFence    fence = NvSciSyncFenceInitializer;

    /* Obtain packet with the new payload */
    NvSciStreamCookie cookie;
    sciErr = NvSciStreamConsumerPacketAcquire(consData->common.block,
                                              &cookie);
    if (NvSciError_Success != sciErr) {
        printf("Consumer failed (%x) to obtain packet for payload\n", sciErr);
        return 0;
    }
    ConsPacket* packet = (ConsPacket*)cookie;
    consData->counter++;

    /* If the received waiter obj if NULL,
     * the producer is done writing data into this element,
     * skip waiting on pre-fence.
     */
    if (NULL != consData->waiterObj) {

        /* Query fences for this element from producer */
        sciErr = NvSciStreamBlockPacketFenceGet(consData->common.block,
                                                packet->handle,
                                                0U, 0U,
                                                &fence);
        if (NvSciError_Success != sciErr) {
            printf("Consumer failed (%x) to query fence from producer\n",
                    sciErr);
            return 0;
        }

        /* Instruct CUDA to wait for the producer fence */
        struct cudaExternalSemaphoreWaitParams waitParams;
        memset(&waitParams, 0, sizeof(waitParams));
        waitParams.params.nvSciSync.fence = &fence;
        cudaRtErr = cudaWaitExternalSemaphoresAsync(&consData->waiterSem,
                                                    &waitParams,
                                                    1, consData->cudaStream);
        if (cudaSuccess != cudaRtErr) {
            printf("Consumer failed (%x) to wait for prefence\n", cudaRtErr);
            return 0;
        }
        NvSciSyncFenceClear(&fence);
    }

    /* Instruct CUDA to copy the packet data buffer to the target buffer */
    cudaRtErr = cudaMemcpyAsync(packet->dataDstPtr,
                                packet->dataDevPtr,
                                consData->dataSize,
                                cudaMemcpyDeviceToHost,
                                consData->cudaStream);
    if (cudaSuccess != cudaRtErr) {
        printf("Consumer failed (%x) to issue copy command\n", cudaRtErr);
        return 0;
    }

    /* Inform CUDA to signal a fence when the copy completes */
    struct cudaExternalSemaphoreSignalParams signalParams;
    memset(&signalParams, 0, sizeof(signalParams));
    signalParams.params.nvSciSync.fence = &fence;
    cudaRtErr = cudaSignalExternalSemaphoresAsync(&consData->signalSem,
                                                  &signalParams,
                                                  1,
                                                  consData->cudaStream);
    if (cudaSuccess != cudaRtErr) {
        printf("Consumer failed (%x) to signal postfence\n", cudaRtErr);
        return 0;
    }

    /* Update postfence for this element */
    sciErr = NvSciStreamBlockPacketFenceSet(consData->common.block,
                                            packet->handle,
                                            0U,
                                            &fence);
    if (NvSciError_Success != sciErr) {
        printf("Consumer failed (%x) to set fence\n", sciErr);
        return 0;
    }

    /* Release the packet back to the producer */
    sciErr = NvSciStreamConsumerPacketRelease(consData->common.block,
                                              packet->handle);
    if (NvSciError_Success != sciErr) {
        printf("Consumer failed (%x) to release packet\n", sciErr);
        return 0;
    }
    NvSciSyncFenceClear(&fence);

    return 1;
}

/* Handle events on a consumer block
 *
 * The consumer block informs the stream of the consumers buffer and
 *   synchronization requirements and capabilities, creates signaling
 *   synchronization objects and receives synchronization objects to
 *   wait for, maps buffers and synchronization objects to the consumer
 *   engine(s), and processes data.
 */
static int32_t handleConsumer(
    void*     data,
    uint32_t  wait)
{
    /* Cast to consumer data */
    ConsData* consData = (ConsData*)data;

    /* Get time to wait */
    int64_t waitTime = wait ? consData->common.waitTime : 0;

    /* Query/wait for an event on the block */
    NvSciStreamEventType event;
    NvSciError       err;
    err = NvSciStreamBlockEventQuery(consData->common.block, waitTime, &event);

    /* Handle errors */
    if (NvSciError_Success != err) {

        /* If not asked to wait, a timeout is not an error */
        if (!waitTime && (NvSciError_Timeout == err)) {
            return 0;
        }

        /* Otherwise, any error is considered fatal. A timeout probably
         *   indicates a failure to connect and complete setup in a timely
         *   fashion, so we specifically call out this case.
         */
        if (NvSciError_Timeout == err) {
            printf("Consumer timed out waiting for setup instructions\n");
        } else {
            printf("Consumer event query failed with error %x\n", err);
        }
        deleteConsumer(consData);
        return -1;
    }

    /* If we received an event, handle it based on its type
     *
     * Note that there's a lot of error checking we could choose to do for
     *   some of these events, like making sure that we only receive each
     *   event once for a given entry. But NvSciStream is expected to take
     *   care of all of that, even when the application makes a mistake.
     *   So we only check for things that don't trigger NvSciStream errors.
     */
    int32_t rv = 1;
    NvSciError status;
    switch (event) {
    /*
     * Any event we don't explicitly handle is a fatal error
     */
    default:
        printf("Consumer received unknown event %x\n", event);

        rv = -1;
        break;

    /*
     * Error events should never occur with safety-certified drivers,
     *   and are provided only in non-safety builds for debugging
     *   purposes. Even then, they should only occur when something
     *   fundamental goes wrong, like the system running out of memory,
     *   or stack/heap corruption, or a bug in NvSci which should be
     *   reported to NVIDIA.
     */
    case NvSciStreamEventType_Error:
        err = NvSciStreamBlockErrorGet(consData->common.block, &status);
        if (NvSciError_Success != err) {
            printf("%s Failed to query the error event code %x\n",
                   consData->common.name, err);
        } else{
            printf("%s received error event: %x\n",
                   consData->common.name, status);
        }

        rv = -1;
        break;

    /*
     * If told to disconnect, it means either the stream finished its
     *   business or some other block had a failure. We'll just do a
     *   clean up and return without an error.
     */
    case NvSciStreamEventType_Disconnected:
        printf("Consumer disconnected after receiving %d payloads\n",
               consData->counter);
        rv = 2;
        break;

    /*
     * On connection, the consumer should initialize the appopriate engine(s)
     *   and obtain the necessary buffer and synchronization attribute lists
     *   for the desired use case.
     */
    case NvSciStreamEventType_Connected:

        /* Initialize CUDA access */
        if (!handleConsumerInit(consData)) {
            rv = -1;
        }
        /* Determine supported buffer attributes */
        else if (!handleConsumerElemSupport(consData)) {
            rv = -1;
        }
        /* Determined supported sync attributes */
        else if (!handleConsumerSyncSupport(consData)) {
            rv = -1;
        }

        /* Now that we're fully connected, set the wait time to infinite */
        consData->common.waitTime = -1;
        break;

    /* Retrieve all element information from pool */
    case NvSciStreamEventType_Elements:
        if (!handleConsumerElemSetting(consData)) {
            rv = -1;
        }
        break;

    /* For a packet, set up an entry in the array */
    case NvSciStreamEventType_PacketCreate:
        if (!handleConsumerPacketCreate(consData)) {
            rv = -1;
        }
        break;

    /* Finish any setup related to packet resources */
    case NvSciStreamEventType_PacketsComplete:
        /* For this use case, nothing else to setup.
         *   Inform the NvSciStream that the consumer has imported all packets.
         */
        err = NvSciStreamBlockSetupStatusSet(consData->common.block,
                                             NvSciStreamSetup_PacketImport,
                                             true);
        if (NvSciError_Success != err) {
            printf("Consumer failed (%x) to complete packet import\n", err);
            rv = -1;
        }
        break;

    /* Delete a packet - usually only relevant for non-safety applications */
    case NvSciStreamEventType_PacketDelete:
        handleConsumerPacketDelete(consData);
        break;

    /* Set up signaling sync object from consumer's wait attributes */
    case NvSciStreamEventType_WaiterAttr:
        if (!handleConsumerSyncExport(consData)) {
            rv = -1;
        }
        break;

    /* Import producer sync objects for all elements */
    case NvSciStreamEventType_SignalObj:
        if (!handleConsumerSyncImport(consData)) {
            rv = -1;
        }
        break;

    /* All setup complete. Transition to runtime phase */
    case NvSciStreamEventType_SetupComplete:
        printf("Consumer setup completed\n");
        break;

    /* Processs payloads when packets arrive */
    case NvSciStreamEventType_PacketReady:
        if (!handleConsumerPayload(consData)) {
            rv = -1;
        }
        break;
    }

    /* On failure or final event, clean up the block */
    if ((rv < 0) || (1 < rv)) {
        deleteConsumer(consData);
    }

    return rv;
}

/* Create and register a new consumer block */
int32_t createConsumer_Usecase2(
    NvSciStreamBlock* consumer,
    NvSciStreamBlock  pool,
    uint32_t          index)
{
    /*
     * The index is ignored. It is provided to support use cases where
     *   there are multiple consumers that don't all do the same thing.
     */
     (void)index;

    /* Create a data structure to track the block's status */
    ConsData* consData = (ConsData*)calloc(1, sizeof(ConsData));
    if (NULL == consData) {
        printf("Failed to allocate data structure for consumer\n");
        return 0;
    }

    /* Save the name for debugging purposes */
    strcpy(consData->common.name, "Consumer");

    /* Wait time for initial connection event will be 60 seconds */
    consData->common.waitTime = 60 * 1000000;

    /* Create a pool block */
    NvSciError err =
        NvSciStreamConsumerCreate(pool, &consData->common.block);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to create consumer block\n", err);
        deleteConsumer(consData);
        return 0;
    }

    if (opts.endInfo) {
        /* Add endpoint information on consumer side.
         *  Application can specify user-defined info to help set up stream,
         *  which can be queried by other blocks after stream connection.
         */
        char info[INFO_SIZE];
        size_t infoSize =
            snprintf(info, INFO_SIZE, "%s%d", "Consumer proc: ", getpid());
        err = NvSciStreamBlockUserInfoSet(consData->common.block,
                                          ENDINFO_NAME_PROC,
                                          infoSize, info);
        if (NvSciError_Success != err) {
            printf("Failed (%x) to setup the consumer info\n", err);
            deleteConsumer(consData);
            return 0;
        }
    }

    /* Register block with event handling mechanism */
    if (!eventFuncs->reg(consData->common.block, consData, handleConsumer)) {
        deleteConsumer(consData);
        return 0;
    }

    *consumer = consData->common.block;
    return 1;
}
