/* NvSciStream Event Loop Driven Sample App - producer block for use case 1
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
 * This implements the producer for use case 1: cuda to cuda streaming
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
#include "usecase1.h"
#include "util.h"

/* Internal data structure used to track packets */
typedef struct {
    /* The packet handle use for NvSciStream functions */
    NvSciStreamPacket    handle;
    /* NvSci buffer object for the packet's data buffer */
    NvSciBufObj          dataObj;
    /* CUDA external memory handle for the data buffer */
    cudaExternalMemory_t dataExtMem;
    /* CUDA device memory pointer for the data buffer */
    void*                dataDevMem;
    /* Local system memory buffer used as the source for CUDA operations */
    uint8_t*             dataSrcMem;
    /* NvSci buffer object for the packet's CRC buffer */
    NvSciBufObj          crcObj;
    /* Virtual address for the CRC buffer */
    uint8_t*             crcPtr;
    /* Fence for the latest payload using this packet */
    NvSciSyncFence       fence;
} ProdPacket;

/* Internal data used by the producer block */
typedef struct {
    /* Common block info */
    BlockData               common;

    /* Number of consumers */
    uint32_t                numConsumers;

    /* CUDA device ID and UUID */
    int32_t                 cudaDeviceId;
    CUuuid                  cudaUuid;

    /* CUDA producer stream */
    cudaStream_t            cudaStream;

    /* NvSciSync context to do CPU waiting for fences */
    NvSciSyncCpuWaitContext cpuWaitContext;
    /* Sync attributes for CPU waiting */
    NvSciSyncAttrList       cpuWaitAttr;

    /* CUDA sync attributes required for signaling */
    NvSciSyncAttrList       signalAttr;
    /* CUDA sync attributes required for waiting */
    NvSciSyncAttrList       waiterAttr;
    /* Sync object for CUDA to signal after generating data */
    NvSciSyncObj            signalObj;
    /* CUDA semaphore mapped to sync object */
    cudaExternalSemaphore_t signalSem;
    /* Sync objects to wait for before generating data */
    NvSciSyncObj            waiterObj[MAX_CONSUMERS];
    /* CUDA semaphores mapped to sync objects */
    cudaExternalSemaphore_t waiterSem[MAX_CONSUMERS];

    /* Element index chosen by pool for the CRC buffer */
    uint32_t                crcIndex;
    /* Element index chosen by pool for the data buffer */
    uint32_t                dataIndex;
    /* Size for data buffer after reconciling all requirements */
    uint64_t                dataSize;
    /* Number of packets provided by pool */
    uint32_t                numPacket;
    /* Information about each packet */
    ProdPacket              packets[MAX_PACKETS];

    /* Number of payloads generated so far */
    uint32_t                counter;
    /* Flag indicating producer has finished generating all payloads */
    uint32_t                finished;
} ProdData;

/* Free up the packet resources */
static void deletePacket(
    ProdPacket* packet)
{
    if (packet != NULL) {
        if (packet->handle != NvSciStreamPacket_Invalid) {
            /* Free CUDA memory mapping */
            (void)cudaFree(packet->dataDevMem);
            if (packet->dataExtMem) {
                (void)cudaDestroyExternalMemory(packet->dataExtMem);
                packet->dataExtMem = 0;
            }

            if (packet->dataSrcMem) {
                free(packet->dataSrcMem);
                packet->dataSrcMem = NULL;
            }

            /* Free buffer objects */
            if (packet->dataObj) {
                NvSciBufObjFree(packet->dataObj);
                packet->dataObj = NULL;
            }

            if (packet->crcObj) {
                NvSciBufObjFree(packet->crcObj);
                packet->crcObj = NULL;
            }

            /* Clear the fences */
            NvSciSyncFenceClear(&packet->fence);
        }

        /* Clear out packet information */
        memset(packet, 0, sizeof(ProdPacket));
    }
}

/* Free up producer block resources */
static void deleteProducer(
    ProdData* prodData)
{
    /* Destroy block */
    if (prodData->common.block != 0) {
        (void)NvSciStreamBlockDelete(prodData->common.block);
        prodData->common.block = 0;
    }

    /* Free the packet resources */
    for (uint32_t i=0;i<prodData->numPacket; i++) {
        deletePacket(&prodData->packets[i]);
    }

    /* Free the sync objects */
    for (uint32_t i=0; i< prodData->numConsumers; i++) {
        if (prodData->waiterObj[i] != NULL) {
            (void)cudaDestroyExternalSemaphore(prodData->waiterSem[i]);
            prodData->waiterSem[i] = 0;
            NvSciSyncObjFree(prodData->waiterObj[i]);
            prodData->waiterObj[i] = NULL;
        }
    }

    if (prodData->signalObj != NULL) {
        (void)cudaDestroyExternalSemaphore(prodData->signalSem);
        prodData->signalSem = 0;
        NvSciSyncObjFree(prodData->signalObj);
        prodData->signalObj = NULL;
    }

    /* Free the cpu waiters */
    if (prodData->cpuWaitAttr != NULL) {
        NvSciSyncAttrListFree(prodData->cpuWaitAttr);
        prodData->cpuWaitAttr = NULL;
    }

    /* Free the CPU wait contetxt */
    if (prodData->cpuWaitContext != NULL) {
        NvSciSyncCpuWaitContextFree(prodData->cpuWaitContext);
        prodData->cpuWaitContext = NULL;
    }

    /* Destroy CUDA stream */
    (void)cudaStreamDestroy(prodData->cudaStream);

    /* Free data */
    free(prodData);
}

/* Handle query of basic stream info */
static int32_t handleStreamInit(
    ProdData* prodData)
{
    /* Query number of consumers */
    NvSciError err =
        NvSciStreamBlockConsumerCountGet(prodData->common.block,
                                         &prodData->numConsumers);

    if (NvSciError_Success != err) {
        printf("Failed (%x) to query the number of consumers\n", err);
        return 0;
    }

    if (opts.endInfo) {
    /* Query endpoint info from all consumers */
        for (uint32_t i = 0U; i < prodData->numConsumers; i++) {
            uint32_t size = INFO_SIZE;
            char info[INFO_SIZE];
            err = NvSciStreamBlockUserInfoGet(
                    prodData->common.block,
                    NvSciStreamBlockType_Consumer, i,
                    ENDINFO_NAME_PROC,
                    &size, &info);
            if (NvSciError_Success == err) {
                printf("Consumer %i info: %s\n", i, info);
            } else if (NvSciError_StreamInfoNotProvided == err) {
                printf("Info not provided by the consumer %d\n", i);
            } else {
                printf("Failed (%x) to query the consumer %d info\n", err, i);
                return 0;
            }
        }
    }

    return 1;
}

/* Handle initialization of CUDA resources for producer */
static int32_t handleProducerInit(
    ProdData* prodData)
{
    int32_t  cudaRtErr;
    CUresult cudaErr;

    /* Get stack limit */
    size_t   unused;
    cudaRtErr = cudaDeviceGetLimit(&unused, cudaLimitStackSize);
    if (cudaSuccess != cudaRtErr) {
        printf("Failed (%x) to get CUDA device limit\n", cudaRtErr);
        return 0;
    }

    /* Set CUDA device */
    prodData->cudaDeviceId = 0;
    cudaRtErr = cudaSetDevice(prodData->cudaDeviceId);
    if (cudaSuccess != cudaRtErr) {
        printf("Failed (%x) to set CUDA device\n", cudaRtErr);
        return 0;
    }

    /* Get UUID for CUDA device */
    cudaErr = cuDeviceGetUuid(&prodData->cudaUuid, prodData->cudaDeviceId);
    if (CUDA_SUCCESS != cudaErr) {
        printf("Failed (%x) to get CUDA UUID\n", cudaErr);
        return 0;
    }

    /* Get CUDA stream for asynchronous operation */
    cudaRtErr = cudaStreamCreateWithFlags(&prodData->cudaStream,
                                          cudaStreamNonBlocking);
    if (cudaSuccess != cudaRtErr) {
        printf("Failed (%x) to create CUDA stream\n", cudaRtErr);
        return 0;
    }

    return 1;
}

/* Handle setup of supported buffer attributes */
static int32_t handleProducerElemSupport(
    ProdData* prodData)
{
    /*
     * Note: To illustrate that NvSciStream producer and consumer do
     *       not need to specify the same set of element types, or use
     *       the same order for element types, the producer for this
     *       use case sends the CRC attributes first, followed by the
     *       primary data, while the consumer uses the opposite order.
     *       Our pool implementation will end up using the producer
     *       ordering, but that is not required either.
     */

    NvSciError             sciErr;
    uint32_t               bufName[2];
    NvSciBufAttrList       bufAttrs[2];

    /*
     * CRC buffer requires write access by CPU, and uses a raw 64 byte
     *   data buffer with 1 byte alignment.
     */
    NvSciBufAttrValAccessPerm crcPerm = NvSciBufAccessPerm_ReadWrite;
    uint8_t crcCpu                    = 1U;
    NvSciBufType crcBufType           = NvSciBufType_RawBuffer;
    uint64_t crcSize                  = 64U;
    uint64_t crcAlign                 = 1U;
    NvSciBufAttrKeyValuePair crcKeyVals[] = {
        { NvSciBufGeneralAttrKey_Types, &crcBufType, sizeof(crcBufType) },
        { NvSciBufRawBufferAttrKey_Size, &crcSize, sizeof(crcSize) },
        { NvSciBufRawBufferAttrKey_Align, &crcAlign, sizeof(crcAlign) },
        { NvSciBufGeneralAttrKey_RequiredPerm, &crcPerm, sizeof(crcPerm) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &crcCpu, sizeof(crcCpu) }
    };

    /* Create and fill attribute list for CRC checksum buffer */
    bufName[0] = ELEMENT_NAME_CRC;
    sciErr = NvSciBufAttrListCreate(sciBufModule, &bufAttrs[0]);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to create CRC attribute list\n", sciErr);
        return 0;
    }
    sciErr = NvSciBufAttrListSetAttrs(bufAttrs[0],
                                      crcKeyVals,
                                      sizeof(crcKeyVals) /
                                          sizeof(NvSciBufAttrKeyValuePair));
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to fill CRC attribute list\n", sciErr);
        return 0;
    }

    /*
     * Data buffer requires write access by CPU and the GPU of the cuda
     *   device, and uses a raw 128KB data buffer with 4KB alignment.
     */
    NvSciBufAttrValAccessPerm dataPerm = NvSciBufAccessPerm_ReadWrite;
    uint8_t dataCpu                    = 1U;
    NvSciRmGpuId dataGpu               = { 0 };
    NvSciBufType dataBufType           = NvSciBufType_RawBuffer;
    uint64_t dataSize                  = 128U * 1024U;
    uint64_t dataAlign                 =   4U * 1024U;
    memcpy(&dataGpu.bytes, &prodData->cudaUuid.bytes, sizeof(dataGpu.bytes));
    NvSciBufAttrKeyValuePair dataKeyVals[] = {
        { NvSciBufGeneralAttrKey_GpuId, &dataGpu, sizeof(dataGpu) },
        { NvSciBufGeneralAttrKey_Types, &dataBufType, sizeof(dataBufType) },
        { NvSciBufRawBufferAttrKey_Size, &dataSize, sizeof(dataSize) },
        { NvSciBufRawBufferAttrKey_Align, &dataAlign, sizeof(dataAlign) },
        { NvSciBufGeneralAttrKey_RequiredPerm, &dataPerm, sizeof(dataPerm) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &dataCpu, sizeof(dataCpu) }
    };

    /* Create and fill attribute list for data buffer */
    bufName[1] = ELEMENT_NAME_DATA;
    sciErr = NvSciBufAttrListCreate(sciBufModule, &bufAttrs[1]);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to create data attribute list\n", sciErr);
        return 0;
    }
    sciErr = NvSciBufAttrListSetAttrs(bufAttrs[1],
                                      dataKeyVals,
                                      sizeof(dataKeyVals) /
                                          sizeof(NvSciBufAttrKeyValuePair));
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to fill data attribute list\n", sciErr);
        return 0;
    }

    /*
     * Inform stream of the attributes
     *   Once sent, the attribute lists are no longer needed
     */
    for (uint32_t i=0; i<2U; ++i) {
        sciErr = NvSciStreamBlockElementAttrSet(prodData->common.block,
                                                bufName[i], bufAttrs[i]);
        if (NvSciError_Success != sciErr) {
            printf("Producer failed (%x) to send element %d attribute\n",
                   sciErr, i);
            return 0;
        }
        NvSciBufAttrListFree(bufAttrs[i]);
    }

    /* Indicate that all element information has been exported */
    sciErr = NvSciStreamBlockSetupStatusSet(prodData->common.block,
                                            NvSciStreamSetup_ElementExport,
                                            true);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to complete element export\n", sciErr);
        return 0;
    }

    return 1;
}

/* Handle receipt of chosen element attributes */
static int32_t handleProducerElemSetting(
    ProdData*              prodData)
{
    NvSciError err;

    /*
     * This application does not need to query the element count, because we
     *   know it is always 2. But we do so anyways to show how it is done.
     */
    uint32_t count;
    err = NvSciStreamBlockElementCountGet(prodData->common.block,
                                          NvSciStreamBlockType_Pool,
                                          &count);
    if (NvSciError_Success != err) {
        printf("Producer failed (%x) to query element count\n", err);
        return 0;
    }
    if (2U != count) {
        printf("Producer received unexpected element count (%d)\n", count);
        return 0;
    }

    /* Process all elements */
    for (uint32_t i=0U; i<2U; ++i) {

        /* Query element type and attributes */
        uint32_t type;
        NvSciBufAttrList bufAttr;
        err = NvSciStreamBlockElementAttrGet(prodData->common.block,
                                             NvSciStreamBlockType_Pool, i,
                                             &type, &bufAttr);
        if (NvSciError_Success != err) {
            printf("Producer failed (%x) to query element attr %d\n", err, i);
            return 0;
        }

        /* For data element, need to extract size and save index */
        if (ELEMENT_NAME_DATA == type) {
            prodData->dataIndex = i;
            NvSciBufAttrKeyValuePair keyVals[] = {
                { NvSciBufRawBufferAttrKey_Size, NULL, 0 }
            };
            err = NvSciBufAttrListGetAttrs(bufAttr, keyVals, 1);
            if (NvSciError_Success != err) {
                printf("Producer failed (%x) to obtain buffer size\n", err);
                return 0;
            }
            prodData->dataSize = *((const uint64_t*)(keyVals[0].value));

            /* Set waiter attributes for the asynchronous element. */
            err = NvSciStreamBlockElementWaiterAttrSet(prodData->common.block,
                                                       i,
                                                       prodData->waiterAttr);
            if (NvSciError_Success != err) {
                printf("Producer failed (%x) to send waiter attr for elem %d\n",
                       err, i);
                return 0;
            }

            /* Once sent, the waiting attributes are no longer needed */
            NvSciSyncAttrListFree(prodData->waiterAttr);
            prodData->waiterAttr = NULL;
        }

        /* For CRC element, just need to save the index */
        else if (ELEMENT_NAME_CRC == type) {
            prodData->crcIndex = i;

            /* CRC element is a synchronous element.
             * Pass NULL for the attr to indicate no sync object is needed.
             * This call could be omitted since NULL is the default. */
            err = NvSciStreamBlockElementWaiterAttrSet(prodData->common.block,
                                                       i, NULL);
            if (NvSciError_Success != err) {
                printf("Producer failed (%x) to send waiter attr for elem %d\n",
                       err, i);
                return 0;
            }
        }

        /* Report any unknown element */
        else {
            printf("Producer received unknown element type (%x)\n", type);
            return 0;
        }

        /* Don't need to keep attribute list */
        NvSciBufAttrListFree(bufAttr);
    }

    /* Indicate that element import is complete */
    err = NvSciStreamBlockSetupStatusSet(prodData->common.block,
                                         NvSciStreamSetup_ElementImport,
                                         true);
    if (NvSciError_Success != err) {
        printf("Producer failed (%x) to complete element import\n", err);
        return 0;
    }

    /* Indicate that waiter attribute export is done. */
    err = NvSciStreamBlockSetupStatusSet(prodData->common.block,
                                         NvSciStreamSetup_WaiterAttrExport,
                                         true);
    if (NvSciError_Success != err) {
        printf("Producer failed (%x) to complete waiter attr export\n", err);
        return 0;
    }

    return 1;
}

/* Handle creation of a new packet */
static int32_t handleProducerPacketCreate(
    ProdData*         prodData)
{
    NvSciError err;

    /* Retrieve handle for packet pending creation */
    NvSciStreamPacket handle;
    err = NvSciStreamBlockPacketNewHandleGet(prodData->common.block,
                                             &handle);
    if (NvSciError_Success != err) {
        printf("Producer failed (%x) to retrieve handle for the new packet\n",
               err);
        return 0;
    }

    /* Make sure there is room for more packets */
    if (MAX_PACKETS <= prodData->numPacket) {
        printf("Producer exceeded max packets\n");
        err = NvSciStreamBlockPacketStatusSet(prodData->common.block,
                                              handle,
                                              NvSciStreamCookie_Invalid,
                                              NvSciError_Overflow);
        if (NvSciError_Success != err) {
            printf("Producer failed (%x) to send packet status\n", err);
        }
        return 0;
    }

    /*
     * Allocate the next entry in the array for the new packet.
     *   Use the array entry for the cookie
     */
    ProdPacket* packet = &prodData->packets[prodData->numPacket++];
    packet->handle = handle;
    packet->fence = NvSciSyncFenceInitializer;

    /* Retrieve all buffers and map into application
     *   This use case has 2 elements.
     */
    for (uint32_t index = 0; index < 2; index++) {
        NvSciBufObj bufObj;
        err = NvSciStreamBlockPacketBufferGet(prodData->common.block,
                                              handle,
                                              index,
                                              &bufObj);
        if (NvSciError_Success != err) {
            printf("Producer failed (%x) to retrieve buffer (%lx/%d)\n",
                   err, handle, index);
            return 0;
        }

        /* Handle mapping of data buffer */
        NvSciError sciErr;
        int32_t    cudaRtErr;

        if (index == prodData->dataIndex) {

            /* Save buffer object */
            packet->dataObj = bufObj;

            /* Map in the buffer as CUDA external memory */
            struct cudaExternalMemoryHandleDesc memHandleDesc;
            memset(&memHandleDesc, 0, sizeof(memHandleDesc));
            memHandleDesc.type = cudaExternalMemoryHandleTypeNvSciBuf;
            memHandleDesc.handle.nvSciBufObject = bufObj;
            memHandleDesc.size = prodData->dataSize;
            cudaRtErr = cudaImportExternalMemory(&packet->dataExtMem,
                                                 &memHandleDesc);
            if (cudaSuccess != cudaRtErr) {
                printf("Producer failed (%x) to map buffer as external mem\n",
                       cudaRtErr);
                return 0;
            }

            /* Map in the buffer as CUDA device memory */
            struct cudaExternalMemoryBufferDesc memBufferDesc;
            memset(&memBufferDesc, 0, sizeof(memBufferDesc));
            memBufferDesc.size = prodData->dataSize;
            memBufferDesc.offset = 0;
            cudaRtErr = cudaExternalMemoryGetMappedBuffer(&packet->dataDevMem,
                                                          packet->dataExtMem,
                                                          &memBufferDesc);
            if (cudaSuccess != cudaRtErr) {
                printf("Producer failed (%x) to map buffer as device mem\n",
                       cudaRtErr);
                return 0;
            }

            /* Allocate normal memory to use as the source for the CUDA op */
            packet->dataSrcMem = (uint8_t*)malloc(prodData->dataSize);
            if (NULL == packet->dataSrcMem) {
                printf("Producer failed to allocate source buffer\n");
                return 0;
            }

            /* Fill in with initial values */
            memset(packet->dataSrcMem, 0x5A, prodData->dataSize);

        }

        /* Handle mapping of CRC buffer */
        else if (index == prodData->crcIndex) {

            /* Save buffer object */
            packet->crcObj = bufObj;

            /* Get a CPU pointer for the buffer from NvSci */
            sciErr = NvSciBufObjGetCpuPtr(bufObj, (void**)&packet->crcPtr);
            if (NvSciError_Success != sciErr) {
                printf("Producer failed (%x) to map CRC buffer\n", sciErr);
                return 0;
            }

        }

        /* Shouldn't be any other index */
        else {
            printf("Producer received buffer for unknown element (%d)\n",
                   index);
            return 0;
        }

    }

    /* Inform pool of success.
     *   Note: Could inform the pool of any of the failures above.
     */
    err = NvSciStreamBlockPacketStatusSet(prodData->common.block,
                                          handle,
                                          (NvSciStreamCookie)packet,
                                          NvSciError_Success);
    if (NvSciError_Success != err) {
        printf("Producer failed (%x) to inform pool of packet status\n", err);
        return 0;
    }

    return 1;
}

/* Handle deletion of packet */
static void handleProducerPacketDelete(
    ProdData*         prodData)
{
    /* Get the deleted packet cookie*/
    NvSciStreamCookie cookie;
    NvSciError err =
        NvSciStreamBlockPacketOldCookieGet(prodData->common.block,
                                           &cookie);
    if (NvSciError_Success != err) {
        printf("Producer failed (%x) to retrieve the deleted packet cookie\n",
               err);
    }

    /* Get packet pointer */
    ProdPacket* packet = (ProdPacket*)cookie;

    /* Free the packet resources */
    deletePacket(packet);
}

/* Handle setup of supported sync attributes */
static int32_t handleProducerSyncSupport(
    ProdData* prodData)
{
    NvSciError       sciErr;
    int32_t          cudaRtErr;

    /*
     * Create sync attribute list for signaling.
     *   This will be saved until we receive the consumer's attributes
     */
    sciErr = NvSciSyncAttrListCreate(sciSyncModule, &prodData->signalAttr);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to allocate signal sync attrs\n", sciErr);
        return 0;
    }

    /* Have CUDA fill the signaling attribute list */
    cudaRtErr = cudaDeviceGetNvSciSyncAttributes(prodData->signalAttr,
                                                 prodData->cudaDeviceId,
                                                 cudaNvSciSyncAttrSignal);
    if (cudaSuccess != cudaRtErr) {
        printf("Producer failed (%x) to fill signal sync attrs\n", cudaRtErr);
        return 0;
    }

    /* Create sync attribute list for waiting. */
    sciErr = NvSciSyncAttrListCreate(sciSyncModule, &prodData->waiterAttr);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to allocate waiter sync attrs\n", sciErr);
        return 0;
    }

    /* Have CUDA fill the waiting attribute list */
    cudaRtErr = cudaDeviceGetNvSciSyncAttributes(prodData->waiterAttr,
                                                 prodData->cudaDeviceId,
                                                 cudaNvSciSyncAttrWait);
    if (cudaSuccess != cudaRtErr) {
        printf("Producer failed (%x) to fill waiter sync attrs\n", cudaRtErr);
        return 0;
    }

    /*
     * Most producers will only need to signal their own sync objects and
     *   wait for the consumer sync object(s). But to protect a local
     *   data buffer, this producer will also need the ability to do
     *   CPU waits on the sync objects it signals.
     */

    /* Create attribute list for CPU waiting */
    sciErr = NvSciSyncAttrListCreate(sciSyncModule, &prodData->cpuWaitAttr);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to allocate cpu wait sync attrs\n",
               sciErr);
        return 0;
    }

    /* Fill attribute list for CPU waiting */
    uint8_t                   cpuSync = 1;
    NvSciSyncAccessPerm       cpuPerm = NvSciSyncAccessPerm_WaitOnly;
    NvSciSyncAttrKeyValuePair cpuKeyVals[] = {
        { NvSciSyncAttrKey_NeedCpuAccess, &cpuSync, sizeof(cpuSync) },
        { NvSciSyncAttrKey_RequiredPerm,  &cpuPerm, sizeof(cpuPerm) }
    };
    sciErr = NvSciSyncAttrListSetAttrs(prodData->cpuWaitAttr, cpuKeyVals, 2);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to fill cpu wait sync attrs\n", sciErr);
        return 0;
    }

    /* Create a context for CPU waiting */
    sciErr = NvSciSyncCpuWaitContextAlloc(sciSyncModule,
                                          &prodData->cpuWaitContext);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to allocate cpu wait context\n", sciErr);
        return 0;
    }

    return 1;
}

/* Handle creation and export of producer sync object */
static int32_t handleProducerSyncExport(
    ProdData*         prodData)
{
    NvSciError        sciErr;
    uint32_t          cudaRtErr;

    /* Process waiter attrs from all elements.
     * As CRC element is a synchronous element,
     * no need to query the sync object for it.
     */
    NvSciSyncAttrList waiterAttr = NULL;
    sciErr = NvSciStreamBlockElementWaiterAttrGet(prodData->common.block,
                                                  prodData->dataIndex,
                                                  &waiterAttr);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to query waiter attr\n",
                sciErr);
        return 0;
    }
    if (NULL == waiterAttr) {
        printf("Producer received NULL waiter attr for data elem\n");
        return 0;
    }

    /* Indicate that waiter attribute import is done. */
    sciErr = NvSciStreamBlockSetupStatusSet(prodData->common.block,
                                            NvSciStreamSetup_WaiterAttrImport,
                                            true);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to complete waiter attr import\n",
               sciErr);
        return 0;
    }

    /*
     * Merge and reconcile consumer sync attrs with ours.
     * Note: Many producers would only require their signaler attributes
     *       and the consumer waiter attributes. As noted above, we also
     *       add in attributes to allow us to CPU wait for the syncs that
     *       we signal.
     */
    NvSciSyncAttrList unreconciled[3] = {
        prodData->signalAttr,
        waiterAttr,
        prodData->cpuWaitAttr };
    NvSciSyncAttrList reconciled = NULL;
    NvSciSyncAttrList conflicts = NULL;
    sciErr = NvSciSyncAttrListReconcile(unreconciled, 3,
                                        &reconciled, &conflicts);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to reconcile sync attributes\n", sciErr);
        return 0;
    }

    /* Allocate sync object */
    sciErr = NvSciSyncObjAlloc(reconciled, &prodData->signalObj);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to allocate sync object\n", sciErr);
        return 0;
    }

    /* Free the attribute lists */
    NvSciSyncAttrListFree(prodData->signalAttr);
    prodData->signalAttr = NULL;
    NvSciSyncAttrListFree(waiterAttr);
    NvSciSyncAttrListFree(reconciled);

    /* Create CUDA semaphore for sync object */
    struct cudaExternalSemaphoreHandleDesc extSemDesc;
    memset(&extSemDesc, 0, sizeof(extSemDesc));
    extSemDesc.type = cudaExternalSemaphoreHandleTypeNvSciSync;
    extSemDesc.handle.nvSciSyncObj = prodData->signalObj;
    cudaRtErr = cudaImportExternalSemaphore(&prodData->signalSem,
                                            &extSemDesc);
    if (cudaSuccess != cudaRtErr) {
        printf("Producer failed (%x) to map signal object to semaphore\n",
               cudaRtErr);
        return 0;
    }

    /* Only send the sync object for the asynchronous element.
     * If this function is not called for an element,
     * the sync object is assumed to be NULL.
     * In this use case, CRC element doesn't use sync object.
     */
    sciErr = NvSciStreamBlockElementSignalObjSet(prodData->common.block,
                                                 prodData->dataIndex,
                                                 prodData->signalObj);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to send sync object\n", sciErr);
        return 0;
    }

    /* Indicate that sync object export is complete */
    sciErr = NvSciStreamBlockSetupStatusSet(prodData->common.block,
                                            NvSciStreamSetup_SignalObjExport,
                                            true);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to complete signal obj export\n",
               sciErr);
        return 0;
    }

    return 1;
}

/* Handle import of consumer sync object */
static int32_t handleProducerSyncImport(
    ProdData*         prodData)
{
    uint32_t          cudaRtErr;
    NvSciError        sciErr;

    /* Query sync objects for asynchronous elements
     * from all consumers.
     */
    for (uint32_t c = 0U; c < prodData->numConsumers; c++) {
        NvSciSyncObj waiterObj = NULL;
        sciErr = NvSciStreamBlockElementSignalObjGet(
                    prodData->common.block,
                    c, prodData->dataIndex,
                    &waiterObj);
        if (NvSciError_Success != sciErr) {
            printf("Producer failed (%x) to query sync obj from consumer %d\n",
                   sciErr, c);
            return 0;
        }

        /* Save object */
        prodData->waiterObj[c] = waiterObj;

        /* If the waiter sync obj is NULL,
         * it means this element is ready to use when received.
         */
        if (NULL != waiterObj) {
            /* Create CUDA semaphore for sync object */
            struct cudaExternalSemaphoreHandleDesc extSemDesc;
            memset(&extSemDesc, 0, sizeof(extSemDesc));
            extSemDesc.type = cudaExternalSemaphoreHandleTypeNvSciSync;
            extSemDesc.handle.nvSciSyncObj = waiterObj;
            cudaRtErr = cudaImportExternalSemaphore(&prodData->waiterSem[c],
                                                    &extSemDesc);
            if (cudaSuccess != cudaRtErr) {
                printf("Producer failed (%x) to map waiter obj from cons %d\n",
                       cudaRtErr, c);
                return 0;
            }
        }
    }

    /* Indicate that element import is complete */
    sciErr = NvSciStreamBlockSetupStatusSet(prodData->common.block,
                                            NvSciStreamSetup_SignalObjImport,
                                            true);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to complete signal obj import\n",
               sciErr);
        return 0;
    }

    return 1;
}

/* Handle generation of payloads */
static int32_t handleProducerPayload(
    ProdData*         prodData)
{
    NvSciError        sciErr;
    int32_t           cudaErr;

    /* Obtain packet for the new payload */
    NvSciStreamCookie cookie;
    sciErr = NvSciStreamProducerPacketGet(prodData->common.block,
                                          &cookie);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to obtain packet for payload\n", sciErr);
        return 0;
    }
    ProdPacket* packet = (ProdPacket*)cookie;

    /*
     * Before modifying the contents of the source buffer, make sure the
     *   previous copy from the buffer has completed. Once done, the
     *   fence can be cleared.
     * Note: This CPU wait on the previously generated payload for this
     *       packet is only necesary to protect the source buffer contents.
     *       If this producer were processing data coming in from an external
     *       source or generating data that didn't involve copying from a
     *       fixed source, this wait would not be necessary. For most
     *       producers, it is sufficient to have the engine wait for the
     *       consumer prefences.
     *       However, this wait does add some throttling, preventing the
     *       producer from issuing commands for many payloads in advance,
     *       which can be valuable in some use cases.
     */
    sciErr = NvSciSyncFenceWait(&packet->fence, prodData->cpuWaitContext, -1);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to CPU wait for previous fence\n",
               sciErr);
        return 0;
    }
    NvSciSyncFenceClear(&packet->fence);

    /*
     * Modify 1 byte of the source.
     * We do this mainly so we're not sending the same thing every time,
     *   but also to have an excuse to illustrate the CPU wait above,
     *   for use cases where the producer needs to wait for more than
     *   just an available buffer to continue.
     */
    packet->dataSrcMem[prodData->counter] =
                (uint8_t)(prodData->counter % 256);
    prodData->counter++;

    /* Query fences for data element from each consumer */
    for (uint32_t i = 0U; i < prodData->numConsumers; i++) {
        /* If the received waiter obj if NULL,
         * the consumer is done using this element,
         * skip waiting on pre-fence.
         */
        if (NULL == prodData->waiterObj[i]) {
            continue;
        }

        NvSciSyncFence prefence = NvSciSyncFenceInitializer;
        sciErr = NvSciStreamBlockPacketFenceGet(prodData->common.block,
                                                packet->handle,
                                                i, prodData->dataIndex,
                                                &prefence);
        if (NvSciError_Success != sciErr) {
            printf("Producer failed (%x) to query fence from consumer %d\n",
                    sciErr, i);
            return 0;
        }

        /* Instruct CUDA to wait for each of the consumer fences */
        struct cudaExternalSemaphoreWaitParams waitParams;
        memset(&waitParams, 0, sizeof(waitParams));
        waitParams.params.nvSciSync.fence = &prefence;
        waitParams.flags = 0;
        cudaErr = cudaWaitExternalSemaphoresAsync(
                    &prodData->waiterSem[i],
                    &waitParams, 1,
                    prodData->cudaStream);
        if (cudaSuccess != cudaErr) {
            printf("Producer failed (%x) to wait for prefence from cons %d\n",
                    cudaErr, i);
            return 0;
        }
        NvSciSyncFenceClear(&prefence);
    }

    /* Instruct CUDA to copy the source buffer to the packet data buffer */
    cudaErr = cudaMemcpy2DAsync(packet->dataDevMem,
                                prodData->dataSize,
                                packet->dataSrcMem,
                                prodData->dataSize,
                                prodData->dataSize,
                                1,
                                cudaMemcpyHostToDevice,
                                prodData->cudaStream);
    if (cudaSuccess != cudaErr) {
        printf("Producer failed (%x) to issue copy command\n", cudaErr);
        return 0;
    }

    /* Inform CUDA to signal a fence when the copy completes */
    struct cudaExternalSemaphoreSignalParams signalParams;
    memset(&signalParams, 0, sizeof(signalParams));
    signalParams.params.nvSciSync.fence = &packet->fence;
    signalParams.flags = 0;
    cudaErr = cudaSignalExternalSemaphoresAsync(&prodData->signalSem,
                                                &signalParams,
                                                1,
                                                prodData->cudaStream);
    if (cudaSuccess != cudaErr) {
        printf("Producer failed (%x) to signal postfence\n", cudaErr);
        return 0;
    }

    /* Generate a checkum and save to the CRC buffer of the packet */
    *((uint32_t*)(packet->crcPtr)) = generateCRC(packet->dataSrcMem,
                                                 1,
                                                 prodData->dataSize,
                                                 prodData->dataSize);


    /* Update postfence for data element */
    sciErr = NvSciStreamBlockPacketFenceSet(prodData->common.block,
                                            packet->handle,
                                            prodData->dataIndex,
                                            &packet->fence);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to set postfence\n", sciErr);
        return 0;
    }

    /* Send the new payload to the consumer(s) */
    sciErr = NvSciStreamProducerPacketPresent(prodData->common.block,
                                              packet->handle);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to present packet\n", sciErr);
        return 0;
    }

    /* If counter has reached the limit, indicate finished */
    if (prodData->counter == 32) {
        /* Make sure all operations have been completed
         * before resource cleanup.
         */
        sciErr =  NvSciSyncFenceWait(&packet->fence,
                            prodData->cpuWaitContext,
                            0xFFFFFFFF);
        if (NvSciError_Success != sciErr) {
            printf("Producer failed (%x) to wait for all operations done\n",
                   sciErr);
            return 0;
        }

        printf("Producer finished sending %d payloads\n", prodData->counter);
        prodData->finished = 1;
    }

    return 1;
}

/* Handle events on a producer block
 *
 * The producer block informs the stream of the producers buffer and
 *   synchronization requirements and capabilities, creates signaling
 *   synchronization objects and receives synchronization objects to
 *   wait for, maps buffers and synchronization objects to the producer
 *   engine(s), and generates data.
 */
static int32_t handleProducer(
    void*     data,
    uint32_t  wait)
{
    /* Cast to producer data */
    ProdData* prodData = (ProdData*)data;

    /* Get time to wait */
    int64_t waitTime = wait ? prodData->common.waitTime : 0;

    /* Query/wait for an event on the block */

    NvSciStreamEventType event;
    NvSciError       err;
    err = NvSciStreamBlockEventQuery(prodData->common.block, waitTime, &event);

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
            printf("Producer timed out waiting for setup instructions\n");
        } else {
            printf("Producer event query failed with error %x\n", err);
        }
        deleteProducer(prodData);
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
        printf("Producer received unknown event %x\n", event);

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
        err = NvSciStreamBlockErrorGet(prodData->common.block, &status);
        if (NvSciError_Success != err) {
            printf("%s Failed to query the error event code %x\n",
                   prodData->common.name, err);
        } else {
            printf("%s received error event: %x\n",
                   prodData->common.name, status);
        }

        rv = -1;
        break;

    /*
     * If told to disconnect, it means either the stream finished its
     *   business or some other block had a failure. We'll just do a
     *   clean up and return without an error.
     */
    case NvSciStreamEventType_Disconnected:
        rv = 2;
        break;

    /*
     * On connection, the producer should initialize the appopriate engine(s)
     *   and obtain the necessary buffer and synchronization attribute lists
     *   for the desired use case.
     */
    case NvSciStreamEventType_Connected:

        /* Initialize basic stream info */
        if (!handleStreamInit(prodData)) {
            rv = -1;
        }
        /* Initialize CUDA access */
        else if (!handleProducerInit(prodData)) {
            rv = -1;
        }
        /* Determine supported buffer attributes */
        else if (!handleProducerElemSupport(prodData)) {
            rv = -1;
        }
        /* Determined supported sync attributes */
        else if (!handleProducerSyncSupport(prodData)) {
            rv = -1;
        }

        /* Now that we're fully connected, set the wait time to infinite */
        prodData->common.waitTime = -1;
        break;

    /* Retrieve all element information from pool */
    case NvSciStreamEventType_Elements:
        if (!handleProducerElemSetting(prodData)) {
            rv = -1;
        }
        break;

    /* For a packet, set up an entry in the array */
    case NvSciStreamEventType_PacketCreate:
        if (!handleProducerPacketCreate(prodData)) {
            rv = -1;
        }
        break;

    /* Finish any setup related to packet resources */
    case NvSciStreamEventType_PacketsComplete:
        /* For this use case, nothing else to setup.
         *   Inform the NvSciStream that the producer has imported all packets.
         */
        err = NvSciStreamBlockSetupStatusSet(prodData->common.block,
                                             NvSciStreamSetup_PacketImport,
                                             true);
        if (NvSciError_Success != err) {
            printf("Producer failed (%x) to complete packet import\n", err);
            rv = -1;
        }
        break;

    /* Delete a packet - usually only relevant for non-safety applications */
    case NvSciStreamEventType_PacketDelete:
        handleProducerPacketDelete(prodData);
        break;

    case NvSciStreamEventType_WaiterAttr:
        if (!handleProducerSyncExport(prodData)) {
            rv = -1;
        }
        break;

    /* Import consumer sync objects for all elements */
    case NvSciStreamEventType_SignalObj:
        if (!handleProducerSyncImport(prodData)) {
            rv = -1;
        }
        break;

    /* All setup complete. Transition to runtime phase */
    case NvSciStreamEventType_SetupComplete:
        printf("Producer setup completed\n");
        break;

    /* Generate payloads when packets are available */
    case NvSciStreamEventType_PacketReady:
        if (!handleProducerPayload(prodData)) {
            rv = -1;
        } else if (prodData->finished) {
            rv = 2;
        }
        break;
    }

    /* On failure or final event, clean up the block */
    if ((rv < 0) || (1 < rv)) {
        deleteProducer(prodData);
    }

    return rv;
}

/* Create and register a new producer block */
int32_t createProducer_Usecase1(
    NvSciStreamBlock* producer,
    NvSciStreamBlock  pool)
{
    /* Create a data structure to track the block's status */
    ProdData* prodData = (ProdData*)calloc(1, sizeof(ProdData));
    if (NULL == prodData) {
        printf("Failed to allocate data structure for producer\n");
        return 0;
    }

    /* Save the name for debugging purposes */
    strcpy(prodData->common.name, "Producer");

    /* Wait time for initial connection event will be 60 seconds */
    prodData->common.waitTime = 60 * 1000000;

    /* Create a pool block */
    NvSciError err =
        NvSciStreamProducerCreate(pool, &prodData->common.block);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to create producer block\n", err);
        deleteProducer(prodData);
        return 0;
    }

    if (opts.endInfo) {
        /* Add endpoint information on producer side.
         *  Application can specify user-defined info to help set up stream,
         *  which can be queried by other blocks after stream connection.
         */
        char info[INFO_SIZE];
        size_t infoSize =
            snprintf(info, INFO_SIZE, "%s%d", "Producer proc: ", getpid());
        err = NvSciStreamBlockUserInfoSet(prodData->common.block,
                                          ENDINFO_NAME_PROC,
                                          infoSize, info);
        if (NvSciError_Success != err) {
            printf("Failed (%x) to setup the producer info\n", err);
            deleteProducer(prodData);
            return 0;
        }
    }

    /* Register block with event handling mechanism */
    if (!eventFuncs->reg(prodData->common.block, prodData, handleProducer)) {
        deleteProducer(prodData);
        return 0;
    }

    *producer = prodData->common.block;
    return 1;
}
