/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/*
 * NvSciStream Event Loop Driven Sample App - producer block for use case 3
 *
 * This implements the producer for use case 3: cuda to cuda streaming
 *  with with ASIL-D safety.
 */

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include "nvscibuf.h"
#include "nvscisync.h"
#include "nvscistream.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "block_info.h"
#include "event_loop.h"
#include "usecase3.h"
#include "util.h"

/* Internal data structure used to store all the init-time validation data */
typedef struct {
    /*
     * Init-time CRC values
     */

    /* CRC value for producer's endpoint info */
    uint32_t    prodInfo;
    /* CRC value for the buffer info
     * buf[0..consCount-1] is consumer's buf crc values
     * buf[consCount] is the producer's crc values
     */

    uint32_t*   buf;
    /* CRC value for producer's sync objects */
    uint32_t    prodSync;
    /* CRC value for consumer's sync objects */
    uint32_t    consSync[MAX_CONSUMERS];

    /*
     * Other init-time validation data
     */

    /* Stream id received from producer */
    uint32_t    streamId;
} ProdInitCrcData;

/* Internal data structure used to track buffer attributes of data element.
 *  Data buffer requires write access by CPU and the GPU of the cuda
 *  device, and uses a raw 128KB data buffer with 4KB alignment.
 */
typedef struct {
    NvSciBufAttrValAccessPerm   perm ;
    uint8_t                     cpu;
    NvSciRmGpuId                gpu;
    NvSciBufType                bufType;
    uint64_t                    size;
    uint64_t                    align;
} ProdDataBufAttrs;

/* Internal data structure used to track buffer attributes of CRC element.
 *  CRC buffer requires write access by CPU, and uses a raw 64 byte
 *   data buffer with 1 byte alignment.
 */
typedef struct {
    NvSciBufAttrValAccessPerm   perm;
    uint8_t                     cpu;
    NvSciBufType                bufType;
    uint64_t                    size;
    uint64_t                    align;
} ProdCrcBufAttrs;

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

    /* NvSci buffer object for the packet's producer CRC buffer */
    NvSciBufObj          prodCrcObj;
    /* Virtual address for the CRC buffer */
    void*                prodCrcPtr;
    /* NvSci buffer object for the packet's consumer CRC buffer */
    NvSciBufObj          consCrcObj[MAX_CONSUMERS];
    /* Virtual address for the CRC buffer */
    void*                consCrcPtr[MAX_CONSUMERS];
    /* Fence for the latest payload using this packet */
    NvSciSyncFence       fence;
    /* Indicate whether the packet is reused */
    bool                 reuse;
} ProdPacket;

/* Internal data used by the producer block */
typedef struct {
    /* Common block info */
    BlockData               common;

    /* Init-time validation data*/
    ProdInitCrcData         initCrc;

    /* Number of consumers */
    uint32_t                numConsumers;

    /* Number of elements per packet decided by pool */
    uint32_t                elemCount;

    /* CUDA device ID and UUID */
    int32_t                 cudaDeviceId;
    CUuuid                  cudaUuid;

    /* CUDA producer stream */
    cudaStream_t            cudaStream;

    /* Buffer attributes for data element */
    ProdDataBufAttrs        dataBufAttrs;
    /* Buffer attributes for CRC element */
    ProdCrcBufAttrs         crcBufAttrs;

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

    /* Element index chosen by pool for the producer's CRC element */
    uint32_t                prodCrcIndex;
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

            if (packet->prodCrcObj) {
                NvSciBufObjFree(packet->prodCrcObj);
                packet->prodCrcObj = NULL;
            }

            for (uint32_t i = 0; i < MAX_CONSUMERS; i++) {
                if (packet->consCrcObj[i]) {
                    NvSciBufObjFree(packet->consCrcObj[i]);
                    packet->consCrcObj[i] = NULL;
                }
            }

            if (packet->prodCrcObj) {
                NvSciBufObjFree(packet->prodCrcObj);
                packet->prodCrcObj = NULL;
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

    /* Destroy CUDA stream */
    (void)cudaStreamDestroy(prodData->cudaStream);

    /* Free crc resources */
    if (prodData->initCrc.buf) {
        free(prodData->initCrc.buf);
        prodData->initCrc.buf = NULL;
    }

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
    NvSciError             sciErr;
    uint32_t               elemCount;
    uint32_t               elemIndex = 0;
    uint32_t               bufName[MAX_ELEMS];
    NvSciBufAttrList       bufAttrs[MAX_ELEMS];
    ProdDataBufAttrs*      dataAttr = &prodData->dataBufAttrs;
    ProdCrcBufAttrs*       crcAttr = &prodData->crcBufAttrs;

    /*
     * In this use case, producer application requires
     *   one data element,
     *   one producer CRC element
     *   and one consumer CRC element for each consumer.
     */
    elemCount = 1 + 1 + prodData->numConsumers;
    assert(elemCount <= MAX_ELEMS);

    crcAttr->perm = NvSciBufAccessPerm_ReadWrite;
    crcAttr->cpu = 1U;
    crcAttr->bufType = NvSciBufType_RawBuffer;
    crcAttr->size = 64U;
    crcAttr->align = 1U;

    NvSciBufAttrKeyValuePair crcKeyVals[] = {
        { NvSciBufGeneralAttrKey_Types,
            &crcAttr->bufType, sizeof(crcAttr->bufType) },
        { NvSciBufRawBufferAttrKey_Size,
            &crcAttr->size, sizeof(crcAttr->size) },
        { NvSciBufRawBufferAttrKey_Align,
            &crcAttr->align, sizeof(crcAttr->align) },
        { NvSciBufGeneralAttrKey_RequiredPerm,
            &crcAttr->perm, sizeof(crcAttr->perm) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess,
            &crcAttr->cpu, sizeof(crcAttr->cpu) }
    };

    /* Create and fill attribute list for CRC buffer */
    bufName[0] = ELEMENT_NAME_CONS_CRC_BASE;
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

    /* All CRC elements share the same buffer requirement in this case. */
    for (elemIndex = 1; elemIndex < prodData->numConsumers; elemIndex++) {
        bufName[elemIndex] = ELEMENT_NAME_CONS_CRC_BASE + elemIndex;
        sciErr = NvSciBufAttrListClone(bufAttrs[0], &bufAttrs[elemIndex]);
        if (NvSciError_Success != sciErr) {
            printf("Producer failed (%x) to clone buffer attribute list\n",
                    sciErr);
            return 0;
        }
    }

    bufName[elemIndex] = ELEMENT_NAME_PROD_CRC;
    sciErr = NvSciBufAttrListClone(bufAttrs[0], &bufAttrs[elemIndex]);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to clone buffer attribute list\n",
                sciErr);
        return 0;
    }
    elemIndex++;

    dataAttr->perm = NvSciBufAccessPerm_ReadWrite;
    dataAttr->cpu = 1U;
    dataAttr->bufType = NvSciBufType_RawBuffer;
    dataAttr->size = 128U * 1024U;
    dataAttr->align = 4U * 1024;
    memcpy(&dataAttr->gpu.bytes,
           &prodData->cudaUuid.bytes,
           sizeof(dataAttr->gpu.bytes));

    NvSciBufAttrKeyValuePair dataKeyVals[] = {
        { NvSciBufGeneralAttrKey_GpuId,
            &dataAttr->gpu, sizeof(dataAttr->gpu) },
        { NvSciBufGeneralAttrKey_Types,
            &dataAttr->bufType, sizeof(dataAttr->bufType) },
        { NvSciBufRawBufferAttrKey_Size,
            &dataAttr->size, sizeof(dataAttr->size) },
        { NvSciBufRawBufferAttrKey_Align,
            &dataAttr->align, sizeof(dataAttr->align) },
        { NvSciBufGeneralAttrKey_RequiredPerm,
            &dataAttr->perm, sizeof(dataAttr->perm) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess,
            &dataAttr->cpu, sizeof(dataAttr->cpu) }
    };

    /* Create and fill attribute list for data buffer */
    bufName[elemIndex] = ELEMENT_NAME_DATA;
    sciErr = NvSciBufAttrListCreate(sciBufModule, &bufAttrs[elemIndex]);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to create data attribute list\n", sciErr);
        return 0;
    }
    sciErr = NvSciBufAttrListSetAttrs(bufAttrs[elemIndex],
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
    for (uint32_t i = 0; i < elemCount; ++i) {
        sciErr = NvSciStreamBlockElementAttrSet(
                    prodData->common.block, bufName[i], bufAttrs[i]);
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
    ProdData* prodData)
{
    NvSciError err;
    uint32_t crcBufCount;

    err = NvSciStreamBlockElementCountGet(prodData->common.block,
                                          NvSciStreamBlockType_Pool,
                                          &prodData->elemCount);
    if (NvSciError_Success != err) {
        printf("Producer failed (%x) to query element count\n", err);
        return 0;
    }
    assert(prodData->elemCount <= MAX_ELEMS);

    crcBufCount = prodData->numConsumers + 1;
    prodData->initCrc.buf = (uint32_t*)calloc(crcBufCount, sizeof(uint32_t));
    memset(prodData->initCrc.buf, 0, crcBufCount * sizeof(uint32_t));

    for (uint32_t i = 0U; i < prodData->elemCount; ++i) {
        /* Query element type and attributes and
         *   update the buf CRC value.
         */
        uint32_t type;
        NvSciBufAttrList bufAttr;

        err = NvSciStreamBlockElementAttrGetWithCrc(
                prodData->common.block,
                NvSciStreamBlockType_Pool, i,
                &type, &bufAttr,
                &prodData->initCrc.buf[0]);

        if (NvSciError_Success != err) {
            printf("Producer failed (%x) to query element attr %d\n", err, i);
            return 0;
        }

        /* For data element, need to extract size and save index */
        if (ELEMENT_NAME_DATA == type) {
            assert(i == prodData->elemCount - 1);
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

        /* For producer CRC element*/
        else if (ELEMENT_NAME_PROD_CRC == type) {
            /* CRC element is a synchronous element.
             * No need to set the waiter attr, which is NULL by default. */
            assert(i == prodData->numConsumers);
            prodData->prodCrcIndex = i;
        }

        /* For consumer CRC element*/
        else if ((type >= ELEMENT_NAME_CONS_CRC_BASE) &&
                 (type < (ELEMENT_NAME_CONS_CRC_BASE +
                          prodData->numConsumers))) {
            assert(i == type - ELEMENT_NAME_CONS_CRC_BASE);
        }

        /* Report any unknown element */
        else {
            printf("Producer received unknown element type (%x)\n", type);
            return 0;
        }

        /* Don't need to keep attribute list */
        NvSciBufAttrListFree(bufAttr);
    }

    /* Init the array with buffuer attr CRC values */
    for (uint32_t i = 1U; i < crcBufCount; i++) {
        prodData->initCrc.buf[i] = prodData->initCrc.buf[0];
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
    uint32_t crcBufCount = prodData->numConsumers + 1;
    uint32_t tmpBuf[MAX_CONSUMERS+1];

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

    /* Retrieve all buffers with the updated CRC value
     *   and map the buffer object in the client space */
    for (uint32_t index = 0; index < prodData->elemCount; index++) {
        NvSciBufObj bufObj;

        memcpy(tmpBuf,
               prodData->initCrc.buf,
               crcBufCount * sizeof(uint32_t));

        err = NvSciStreamBlockPacketBufferGetWithCrc(
                prodData->common.block,
                handle,
                index,
                &bufObj,
                crcBufCount,
                tmpBuf);
        if (NvSciError_Success != err) {
            printf("Producer failed (%x) to retrieve buffer (%lx/%d)\n",
                   err, handle, index);
            return 0;
        }

        if (index < prodData->prodCrcIndex) {
            /* For consumer CRC elements, only save the CRC value
             * for producer and the owner
             */
            prodData->initCrc.buf[index] = tmpBuf[index];
            prodData->initCrc.buf[prodData->prodCrcIndex] =
                tmpBuf[prodData->prodCrcIndex];

        } else {
            /* For producer CRC element and data element,
             *   save the CRC value for producer and all consumers */
            for (uint32_t i = 0U; i <= prodData->prodCrcIndex; i++) {
                prodData->initCrc.buf[i] = tmpBuf[i];
            }
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

        /* Handle mapping of producer's CRC buffer */
        else if (index == prodData->prodCrcIndex) {

            /* Save buffer object */
            packet->prodCrcObj = bufObj;

            /* Get a CPU pointer for the buffer from NvSci */
            sciErr = NvSciBufObjGetCpuPtr(bufObj, (void**)&packet->prodCrcPtr);
            if (NvSciError_Success != sciErr) {
                printf("Producer failed (%x) to map CRC buffer\n", sciErr);
                return 0;
            }

        }

        /* Handle mapping of consumer's CRC buffer */
        else if (index < prodData->prodCrcIndex) {
            /* Save buffer object */
            packet->consCrcObj[index] = bufObj;

            /* Get a CPU pointer for the buffer from NvSci */
            sciErr = NvSciBufObjGetCpuPtr(
                        bufObj,
                        (void**)&packet->consCrcPtr[index]);
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
    NvSciSyncAttrList unreconciled[2] = {
        prodData->signalAttr,
        waiterAttr };
    NvSciSyncAttrList reconciled = NULL;
    NvSciSyncAttrList conflicts = NULL;
    sciErr = NvSciSyncAttrListReconcile(unreconciled, 2,
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
    sciErr = NvSciStreamBlockElementSignalObjSetWithCrc(
                prodData->common.block,
                prodData->dataIndex,
                prodData->signalObj,
                &prodData->initCrc.prodSync);
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
        sciErr = NvSciStreamBlockElementSignalObjGetWithCrc(
                    prodData->common.block,
                    c, prodData->dataIndex,
                    &waiterObj,
                    &prodData->initCrc.consSync[c]);
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

/* Handle validating the init-time data */
static int32_t handleProducerInitValidation(
    ProdData* prodData)
{
    NvSciError        sciErr;
    bool              consValidated[MAX_CONSUMERS] = { 0 };
    ProdDataBufAttrs* dataAttr = &prodData->dataBufAttrs;
    ProdCrcBufAttrs*  crcAttr = &prodData->crcBufAttrs;

    /* Retrieve and validate the buffer data created by pool */
    PoolInitCrcData const* poolCrcData =
        (PoolInitCrcData const*)prodData->packets[0].prodCrcPtr;

    /* Validate the magic number to ensure
     *  the data is for init-time validatoin
     */
    if (poolCrcData->magicNum != CRC_INIT) {
        printf("Pool init-time magic number NOT match.\n");
        return 0;
    }

    /* Validate the CRC value of all buffers with that from the pool */
    if (poolCrcData->buf != prodData->initCrc.buf[prodData->prodCrcIndex]) {
        printf("CRC of pool's buffers NOT match.\n");
        return 0;
    }


    /* Retrieve and validate the data from consumer's crc elements */
    for (uint32_t i = 0; i < prodData->numConsumers; i++) {
        uint32_t consIndex;

        ConsInitCrcData* consCrcData =
            (ConsInitCrcData*)prodData->packets[0].consCrcPtr[i];

        /* Producer application knows which NvSciIpc channel will used to
         *  communicate with the ASIL-D consumer. Only need to validate
         *  the data from the ASIL-D consumer.
         */

        /* Validate the magic number to ensure
         *  the data is for init-time validatoin
         */
        if (consCrcData->magicNum != CRC_INIT) {
            printf("Consumer init-time magic number NOT match.\n");
            return 0;
        }

        /* Validate the stream id to ensure
         *  data not from another stream.
         */
        if (consCrcData->streamId != prodData->initCrc.streamId) {
            printf("Stream id NOT match.\n");
            return 0;
        }

        /* Validate the consumer index is unique and in valid range */
        consIndex = consCrcData->consIndex;
        if ((consIndex >= prodData->numConsumers) ||
            consValidated[consIndex]) {
            printf("Consumer index not in a valid range or not unique.\n");
            return 0;
        }

        /* Validate the CRC values for endpoint info, buffer and sync */
        if (consCrcData->prodInfo != prodData->initCrc.prodInfo) {
            printf("CRC of producer's user information NOT match.\n");
            return 0;
        }
        if (consCrcData->buf != prodData->initCrc.buf[consIndex]) {
            printf("CRC of buffers NOT match.\n");
            return 0;
        }
        if (consCrcData->prodSync != prodData->initCrc.prodSync) {
            printf("CRC of producer's sync object NOT match.\n");
            return 0;
        }
        if(consCrcData->consSync != prodData->initCrc.consSync[consIndex]) {
            printf("CRC of consumer's sync object NOT match.\n");
            return 0;
        }
        memset(consCrcData, 0, sizeof(ConsInitCrcData));
    }

    /* Validate the received buffers objects whether it
     * - meets the producer's buffer requirements
     * - whether grants correct permission for each consumer
     */
    NvSciBufAttrKeyValuePair crcKeyVals[] = {
        { NvSciBufGeneralAttrKey_Types,
            &crcAttr->bufType, sizeof(crcAttr->bufType) },
        { NvSciBufRawBufferAttrKey_Size,
            &crcAttr->size, sizeof(crcAttr->size) },
        { NvSciBufRawBufferAttrKey_Align,
            &crcAttr->align, sizeof(crcAttr->align) },
        { NvSciBufGeneralAttrKey_RequiredPerm,
            &crcAttr->perm, sizeof(crcAttr->perm) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess,
            &crcAttr->cpu, sizeof(crcAttr->cpu) }
    };

    NvSciBufAttrKeyValuePair dataKeyVals[] = {
        { NvSciBufGeneralAttrKey_GpuId,
            &dataAttr->gpu, sizeof(dataAttr->gpu) },
        { NvSciBufGeneralAttrKey_Types,
            &dataAttr->bufType, sizeof(dataAttr->bufType) },
        { NvSciBufRawBufferAttrKey_Size,
            &dataAttr->size, sizeof(dataAttr->size) },
        { NvSciBufRawBufferAttrKey_Align,
            &dataAttr->align, sizeof(dataAttr->align) },
        { NvSciBufGeneralAttrKey_RequiredPerm,
            &dataAttr->perm, sizeof(dataAttr->perm) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess,
            &dataAttr->cpu, sizeof(dataAttr->cpu) }
    };

    for (uint32_t p = 0; p < prodData->numPacket; p++) {
        ProdPacket* packet = &prodData->packets[p];

        /* Validate the crc buffer objects */
        for (uint32_t e = 0; e < prodData->elemCount; e++) {
            NvSciBufObj obj;
            NvSciBufAttrKeyValuePair* pairArray;
            size_t pairArrayCount;

            if (e < prodData->prodCrcIndex) {
                obj = packet->consCrcObj[e];
                pairArray = crcKeyVals;
                pairArrayCount =
                    sizeof(crcKeyVals) / sizeof(NvSciBufAttrKeyValuePair);
            } else if (e == prodData->prodCrcIndex) {
                obj = packet->prodCrcObj;
                pairArray = crcKeyVals;
                pairArrayCount =
                    sizeof(crcKeyVals) / sizeof(NvSciBufAttrKeyValuePair);
            } else {
                obj = packet->dataObj;
                pairArray = dataKeyVals;
                pairArrayCount =
                    sizeof(dataKeyVals) / sizeof(NvSciBufAttrKeyValuePair);
            }

            /* Retrieve the reconciled attribute list from the buffer object */
            NvSciBufAttrList reconciled = NULL;
            sciErr = NvSciBufObjGetAttrList(obj, &reconciled);
            if(NvSciError_Success != sciErr) {
                printf("Producer failed (%x) to get reconciled buf attrList\n",
                        sciErr);
                return 0;
            }

            /* Verify the reconciled attribute meets the producer's
             *   buffer requirement.
             */
            sciErr = NvSciBufAttrListValidateReconciledAgainstAttrs(
                        reconciled, pairArray, pairArrayCount);
            if (NvSciError_Success != sciErr) {
                printf("Reconciled list not meet producer's buf attrs(%x)\n",
                        sciErr);
                return 0;
            }

            /* Verify the buffer object satisfies the constraints of the
             *  the associated reconciled buffer attribute list.
             */
            sciErr = NvSciBufObjValidate(obj);
            if (NvSciError_Success != sciErr) {
                printf("Buffer object not satisfy the constraints(%x)\n",
                        sciErr);
                return 0;
            }

            /* For the lower ASIL consumer, check whether ready-only permission
             *  is given by calling NvSciBufObjGetMaxPerm API
             */
        }
    }

    /* Validate the received waiter sync objects whether it meets
     *   the producer's waiter requirement.
     */
    for (uint32_t i = 0; i < prodData->numConsumers; i++) {
        /* Retrieve the reconciled attribute list from the waiter object */
        NvSciSyncAttrList reconciled = NULL;
        sciErr = NvSciSyncObjGetAttrList(prodData->waiterObj[i], &reconciled);
        if (NvSciError_Success != sciErr) {
            printf("Producer failed (%x) to get reconciled sync attrList\n",
                    sciErr);
            return 0;
        }

        /* Verify the reconciled attribute meets the producer's waiter
         *   requirement.
         *
         * The application only needs to validate the waiter attributes set
         *   by application. Those set by CUDA/NvMedia should be validated by
         *   CUDA/NvMeida when registering the waiter objects.
         *
         * In this use case, as all the waiter attributes are set by CUDA,
         *   the application just provides an empty array of attributes.
         */
        sciErr = NvSciSyncAttrListValidateReconciledAgainstAttrs(
                    reconciled,
                    NULL, 0,
                    NvSciSyncAccessPerm_WaitOnly);
        if (NvSciError_Success != sciErr) {
            printf("Reconciled list not meet producer's waiter attrs(%x)\n",
                    sciErr);
            return 0;
        }

        sciErr = NvSciSyncObjValidate(prodData->waiterObj[i]);
        if (NvSciError_Success != sciErr) {
            printf("Waiter object not satisfy the constraints(%x)\n",
                    sciErr);
            return 0;
        }
    }


    /* Retrieve the reconciled attribute list from the signal object */
    NvSciSyncAttrList reconciled = NULL;
    sciErr = NvSciSyncObjGetAttrList(prodData->signalObj, &reconciled);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to get reconciled sync attrList\n",
                sciErr);
        return 0;
    }

    /* Verify the reconciled attribute meets the producer's signalling
     *   requirement.
     */
    sciErr = NvSciSyncAttrListValidateReconciledAgainstAttrs(
                reconciled,
                NULL, 0,
                NvSciSyncAccessPerm_SignalOnly);
    if (NvSciError_Success != sciErr) {
        printf("Reconciled list not meet producer's signal attrs(%x)\n",
                sciErr);
        return 0;
    }

    /* Validate the signal sync objects. */
    sciErr = NvSciSyncObjValidate(prodData->signalObj);
    if (NvSciError_Success != sciErr) {
        printf("Signal object not satisfy the constraints(%x)\n",
                sciErr);
        return 0;
    }


    /* Indicate that init-time data validation is complete */
    sciErr = NvSciStreamBlockSetupStatusSet(prodData->common.block,
                                            NvSciStreamSetup_CrcImport,
                                            true);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to complete crc import\n",
                sciErr);
        return 0;
    }

    printf("Producer validation PASS\n");
    return 1;
}

/* Handle generation of payloads */
static int32_t handleProducerPayload(
    ProdData*         prodData)
{
    NvSciError        sciErr;
    int32_t           cudaErr;
    NvSciSyncFence    prefence[MAX_CONSUMERS];
    uint32_t          consFenceCrc;

    /* Obtain packet for the new payload */
    NvSciStreamCookie cookie;
    sciErr = NvSciStreamProducerPacketGet(prodData->common.block,
                                          &cookie);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to obtain packet for payload\n", sciErr);
        return 0;
    }
    ProdPacket* packet = (ProdPacket*)cookie;
    prodData->counter++;

    /* Clear the magic number in producer's CRC element,
     *   in case that the consumer tries to access the
     *   this packet.
     */
    *(uint32_t*)packet->prodCrcPtr = 0;

    /* Query fences for data element from each consumer */
    for (uint32_t i = 0U; i < prodData->numConsumers; i++) {
        /* If the received waiter obj if NULL,
         *  the consumer is done using this element,
         *  skip waiting on pre-fence.
         */
        consFenceCrc = 0U;
        if (NULL != prodData->waiterObj[i]) {
            prefence[i] = NvSciSyncFenceInitializer;
            sciErr = NvSciStreamBlockPacketFenceGetWithCrc(
                        prodData->common.block,
                        packet->handle,
                        i, prodData->dataIndex,
                        &prefence[i],
                        &consFenceCrc);
            if (NvSciError_Success != sciErr) {
                printf("Producer failed (%x) to query fence from consumer %d\n",
                        sciErr, i);
                return 0;
            }
        }

        /*
         * Before modifying the contents of the source buffer, need to validate
         *  all the runtime data from ASIL-D consumers:
         *  - Runtime CRC magic number
         *  - Runtime CRC values
         */
        ConsPayloadCrcData* consCrcData =
            (ConsPayloadCrcData*)packet->consCrcPtr[i];

        /* Skip validaiton if packtet is first in use. */
        if (packet->reuse) {
            /* Validate the magic number to ensure the packet
             *  is not accessing by others.
             */
            if (consCrcData->magicNum != CRC_RUNTIME) {
                printf("Consumer %d runtime magic number NOT match.\n", i);
                return 0;
            }

            /* Validate the fence CRC.
             *   If the magic number matches and the fence CRC in the consumer's
             *   CRC element is 0, the packet may be returned by mailbox queue.
             *   Skip validating the fence.
             */
            if ((consCrcData->fence != 0U) &&
                (consCrcData->fence != consFenceCrc)) {
                printf("Crc of consumer %d fence NOT match.\n", i);
                return 0;
            }
        }
        /* Clear data in consumer's CRC elements except the magic number */
        memset(consCrcData, 0, sizeof(ConsPayloadCrcData));
        consCrcData->magicNum = CRC_RUNTIME;
    }

    /*
     * Validation done. Process payload
     */
    for (uint32_t i = 0U; i < prodData->numConsumers; i++) {
        if (NULL == prodData->waiterObj[i]) {
            continue;
        }

        /* Instruct CUDA to wait for each of the consumer fences */
        struct cudaExternalSemaphoreWaitParams waitParams;
        memset(&waitParams, 0, sizeof(waitParams));
        waitParams.params.nvSciSync.fence = &prefence[i];
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
        NvSciSyncFenceClear(&prefence[i]);
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

    /* Init the fence CRC to 0 */
    uint32_t prodFenceCrc = 0;
    /* Update postfence for data element and update the CRC*/
    sciErr = NvSciStreamBlockPacketFenceSetWithCrc(
                prodData->common.block,
                packet->handle,
                prodData->dataIndex,
                &packet->fence,
                &prodFenceCrc);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to set postfence\n", sciErr);
        return 0;
    }

    /* Write the valdation data into producer's CRC element
     *   before sending the packet:
     *   - Runtime CRC magic number
     *   - Runtime CRC values
     *   - frame count
     *   - present time
     */
    ProdPayloadCrcData* prodCrcData = (ProdPayloadCrcData*)packet->prodCrcPtr;
    prodCrcData->magicNum = CRC_RUNTIME;
    prodCrcData->fence = prodFenceCrc;
    prodCrcData->frameCount = prodData->counter;

    /* Send the new payload to the consumer(s) */
    sciErr = NvSciStreamProducerPacketPresent(prodData->common.block,
                                              packet->handle);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to present packet\n", sciErr);
        return 0;
    }
    packet->reuse = true;

    /* If counter has reached the limit, indicate finished */
    if (prodData->counter == 32) {
        printf("Producer finished sending %d payloads\n", prodData->counter);
        prodData->finished = 1;
    }

    NvSciSyncFenceClear(&packet->fence);

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

    /* Validate the init-time data in CRC elements */
    case NvSciStreamEventType_Validate:
        if (!handleProducerInitValidation(prodData)) {
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
int32_t createProducer_Usecase3(
    NvSciStreamBlock* producer,
    NvSciStreamBlock  pool,
    uint32_t          numFrames)
{
    /* The number of frames are ignored as this is significant only
    * for late/re-attach usecase.
    */
    (void)numFrames;
    
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

    /* Create a producer block and indicate CRC validation is required */
    NvSciError err =
        NvSciStreamProducerCreate2(pool, true, &prodData->common.block);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to create producer block\n", err);
        deleteProducer(prodData);
        return 0;
    }

    /* Producer process defines a unique stream id
     *  and pass it to all consumers.
     *  And update the CRC value with the endpoint info.
     */
    prodData->initCrc.streamId = 1U;
    err = NvSciStreamBlockUserInfoSetWithCrc(
            prodData->common.block,
            ENDINFO_NAME_STREAM_ID,
            sizeof(prodData->initCrc.streamId),
            (void const* const)&prodData->initCrc.streamId,
            &prodData->initCrc.prodInfo);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to send the producer stream id\n", err);
        deleteProducer(prodData);
        return 0;
    }

    /* Register block with event handling mechanism */
    if (!eventFuncs->reg(prodData->common.block, prodData, handleProducer)) {
        deleteProducer(prodData);
        return 0;
    }

    *producer = prodData->common.block;
    return 1;
}
