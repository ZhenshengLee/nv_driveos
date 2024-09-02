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
 * NvSciStream Event Loop Driven Sample App - consumer block for use case 3
 *
 * This implements the consumer for use case 3: cuda to cuda streaming
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

/* Internal data structure used to track buffer attributes of data element.
 *  Data buffer requires write access by CPU and the GPU of the cuda
 *   device, and uses a raw data buffer. (Size is specified by producer.)
 */
typedef struct {
    NvSciBufAttrValAccessPerm   perm;
    uint8_t                     cpu;
    NvSciRmGpuId                gpu;
    NvSciBufType                bufType;
} ConsDataBufAttrs;

/* Internal data structure used to track buffer attributes of CRC element.
 *  CRC buffer requires write access by CPU, and uses a raw 64 byte
 *    data buffer with 1 byte alignment.
 */
typedef struct {
    NvSciBufAttrValAccessPerm   perm;
    uint8_t                     cpu;
    NvSciBufType                bufType;
    uint64_t                    size;
    uint64_t                    align;
} ConsCrcBufAttrs;

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
    /* Local system memory buffer used as the target for CUDA operations */
    uint8_t*             dataDstMem;
    /* NvSci buffer object for the packet's producer CRC buffer */
    NvSciBufObj          prodCrcObj;
    /* Pointer to the producer's CRC value */
    void const*          prodCrcPtr;
    /* NvSci buffer object for the packet's consumer CRC buffer */
    NvSciBufObj          consCrcObj;
    /* Pointer to the consumer's CRC value */
    void*                consCrcPtr;
} ConsPacket;

/* Internal data used by the consumer block */
typedef struct {
    /* Common block info */
    BlockData               common;

    /* Validation data*/
    ConsInitCrcData         initCrc;

    /* Number of elements per packet decided by pool */
    uint32_t                elemCount;

    /* CUDA device ID and UUID */
    int32_t                 cudaDeviceId;
    CUuuid                  cudaUuid;

    /* CUDA consumer stream */
    cudaStream_t            cudaStream;

    /* Buffer attributes for data element */
    ConsDataBufAttrs        dataBufAttrs;
    /* Buffer attributes for CRC element */
    ConsCrcBufAttrs         crcBufAttrs;

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

    /*  Element index chosen by pool for the producer's CRC element */
    uint32_t                prodCrcIndex;
    /*  Element index chosen by pool for the consumer's CRC element */
    uint32_t                consCrcIndex;
    /* Element index chosen by pool for the data buffer */
    uint32_t                dataIndex;
    /* Size for data buffer after reconciling all requirements */
    uint64_t                dataSize;
    /* Number of packets provided by pool */
    uint32_t                numPacket;
    /* Information about each packet */
    ConsPacket              packets[MAX_PACKETS];

    /* Track different status of setup */
    bool                    bufDone;
    bool                    syncImportDone;
    bool                    syncExportDone;
    bool                    validationDone;

    /* Number of payloads processed so far */
    uint32_t                counter;
} ConsData;

/* Free up the packet resources */
static void deletePacket(
    ConsPacket* packet)
{
    if (packet != NULL) {
        if (packet->handle != NvSciStreamPacket_Invalid) {
            /* Free CUDA memory mapping */
            (void)cudaFree(packet->dataDevMem);
            if (packet->dataExtMem) {
                (void)cudaDestroyExternalMemory(packet->dataExtMem);
                packet->dataExtMem = 0;
            }

            if (packet->dataDstMem) {
                free(packet->dataDstMem);
                packet->dataDstMem = NULL;
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

            if (packet->consCrcObj) {
                NvSciBufObjFree(packet->consCrcObj);
                packet->consCrcObj = NULL;
            }
        }

        /* Clear out packet information */
        memset(packet, 0, sizeof(ConsPacket));
    }
}

/* Free up consumer block resources */
static void deleteConsumer(
    ConsData* consData)
{
    /* Destroy block */
    if (consData->common.block != 0) {
        (void)NvSciStreamBlockDelete(consData->common.block);
        consData->common.block = 0;
    }

    /* Free the packet resources */
    for (uint32_t i=0;i<consData->numPacket; i++) {
        deletePacket(&consData->packets[i]);
    }

    /* Free the sync objects */
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

    /* Destroy CUDA stream */
    (void)cudaStreamDestroy(consData->cudaStream);

    /* Free data */
    free(consData);
}

/* Handle query of basic stream info */
static int32_t handleStreamInit(
    ConsData* consData)
{
    uint32_t infoSize = sizeof(uint32_t);

    /* Query the connection index*/
    NvSciError err = NvSciStreamConsumerIndexGet(
        consData->common.block,
        &consData->initCrc.consIndex);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to query the consumer index\n", err);
        return 0;
    }

    /* Query endpoint info from producer.
     *   And update the CRC value with the received endpoint info.
     */
    err = NvSciStreamBlockUserInfoGetWithCrc(
            consData->common.block,
            NvSciStreamBlockType_Producer, 0U,
            ENDINFO_NAME_STREAM_ID,
            &infoSize,
            &consData->initCrc.streamId,
            &consData->initCrc.prodInfo);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to query the producer stream id\n", err);
        return 0;
    }

    return 1;
}

/* Handle initialization of CUDA resources for consumer */
static int32_t handleConsumerInit(
    ConsData * consData)
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
    /*
     * In this use case, consumer application requires
     *   one data element,
     *   one producer CRC element
     *   and one consumer CRC element.
     */
    NvSciError             sciErr;
    uint32_t               bufName[3];
    NvSciBufAttrList       bufAttrs[3];
    ConsDataBufAttrs*      dataAttr = &consData->dataBufAttrs;;
    ConsCrcBufAttrs*       crcAttr = &consData->crcBufAttrs;


    /* All CRC elements share the same buffer requirement in this case. */
    crcAttr->perm = NvSciBufAccessPerm_Readonly;
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

    /* Create and fill attribute list for producer CRC buffer */
    bufName[0] = ELEMENT_NAME_PROD_CRC;
    sciErr = NvSciBufAttrListCreate(sciBufModule, &bufAttrs[0]);
    if (NvSciError_Success != sciErr) {
        printf("Consumer failed (%x) to create CRC attribute list\n", sciErr);
        return 0;
    }
    sciErr = NvSciBufAttrListSetAttrs(bufAttrs[0],
                                      crcKeyVals,
                                      sizeof(crcKeyVals) /
                                        sizeof(NvSciBufAttrKeyValuePair));
    if (NvSciError_Success != sciErr) {
        printf("Consumer failed (%x) to fill CRC attribute list\n", sciErr);
        return 0;
    }

    /* Create and fill attribute list for consumer CRC buffer.
     *   It has ReadWrite permission.
     */
    crcAttr->perm = NvSciBufAccessPerm_ReadWrite;
    bufName[1] = ELEMENT_NAME_CONS_CRC_BASE + consData->initCrc.consIndex;
    sciErr = NvSciBufAttrListCreate(sciBufModule, &bufAttrs[1]);
    if (NvSciError_Success != sciErr) {
        printf("Consumer failed (%x) to create CRC attribute list\n", sciErr);
        return 0;
    }
    sciErr = NvSciBufAttrListSetAttrs(bufAttrs[1],
                                      crcKeyVals,
                                      sizeof(crcKeyVals) /
                                        sizeof(NvSciBufAttrKeyValuePair));
    if (NvSciError_Success != sciErr) {
        printf("Consumer failed (%x) to fill CRC attribute list\n", sciErr);
        return 0;
    }

    /* Fill in attributes for data element */
    dataAttr->perm = NvSciBufAccessPerm_Readonly;
    dataAttr->cpu = 1U;
    dataAttr->bufType = NvSciBufType_RawBuffer;
    memcpy(&dataAttr->gpu.bytes,
           &consData->cudaUuid.bytes,
           sizeof(dataAttr->gpu.bytes));

    dataAttr = &consData->dataBufAttrs;
    NvSciBufAttrKeyValuePair dataKeyVals[] = {
        { NvSciBufGeneralAttrKey_GpuId,
            &dataAttr->gpu, sizeof(dataAttr->gpu) },
        { NvSciBufGeneralAttrKey_Types,
            &dataAttr->bufType, sizeof(dataAttr->bufType) },
        { NvSciBufGeneralAttrKey_RequiredPerm,
            &dataAttr->perm, sizeof(dataAttr->perm) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess,
            &dataAttr->cpu, sizeof(dataAttr->cpu) }
    };

    /* Create and fill attribute list for data buffer */
    bufName[2] = ELEMENT_NAME_DATA;
    sciErr = NvSciBufAttrListCreate(sciBufModule, &bufAttrs[2]);
    if (NvSciError_Success != sciErr) {
        printf("Consumer failed (%x) to create data attribute list\n", sciErr);
        return 0;
    }
    sciErr = NvSciBufAttrListSetAttrs(bufAttrs[2],
                                      dataKeyVals,
                                      sizeof(dataKeyVals) /
                                          sizeof(NvSciBufAttrKeyValuePair));
    if (NvSciError_Success != sciErr) {
        printf("Consumer failed (%x) to fill data attribute list\n", sciErr);
        return 0;
    }


    /*
     * Inform stream of the attributes
     *   Once sent, the attribute lists are no longer needed
     */
    for (uint32_t i = 0; i < 3U; ++i) {
        sciErr = NvSciStreamBlockElementAttrSet(consData->common.block,
                                                bufName[i], bufAttrs[i]);
        if (NvSciError_Success != sciErr) {
            printf("Consumer failed (%x) to send element %d attribute\n",
                   sciErr, i);
            return 0;
        }
        NvSciBufAttrListFree(bufAttrs[i]);
    }

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

    err = NvSciStreamBlockElementCountGet(consData->common.block,
                                          NvSciStreamBlockType_Pool,
                                          &consData->elemCount);
    if (NvSciError_Success != err) {
        printf("Consumer failed (%x) to query element count\n", err);
        return 0;
    }
    assert(consData->elemCount <= MAX_ELEMS);

    /* Process all elements */
    for (uint32_t i = 0U; i < consData->elemCount; ++i) {
        /* Query element type and attributes and
         *   update the buf CRC value.
         */
        uint32_t type;
        NvSciBufAttrList bufAttr;
        err = NvSciStreamBlockElementAttrGetWithCrc(
                consData->common.block,
                NvSciStreamBlockType_Pool, i,
                &type, &bufAttr,
                &consData->initCrc.buf);
        if (NvSciError_Success != err) {
            printf("Consumer failed (%x) to query element attr %d\n", err, i);
            return 0;
        }

        /* For data element, need to extract size and save index */
        if (ELEMENT_NAME_DATA == type) {
            consData->dataIndex = i;
            NvSciBufAttrKeyValuePair keyVals[] = {
                { NvSciBufRawBufferAttrKey_Size, NULL, 0 }
            };
            err = NvSciBufAttrListGetAttrs(bufAttr, keyVals, 1);
            if (NvSciError_Success != err) {
                printf("Consumer failed (%x) to obtain buffer size\n", err);
                return 0;
            }
            consData->dataSize = *((const uint64_t*)(keyVals[0].value));

            /* Set waiter attributes for the asynchronous element. */
            err = NvSciStreamBlockElementWaiterAttrSet(consData->common.block,
                                                       i,
                                                       consData->waiterAttr);
            if (NvSciError_Success != err) {
                printf("Consumer failed (%x) to send waiter attr for elem %d\n",
                       err, i);
                return 0;
            }

            /* Once sent, the waiting attributes are no longer needed */
            NvSciSyncAttrListFree(consData->waiterAttr);
            consData->waiterAttr = NULL;
        }

        /* For CRC element, just need to save the index */
        else if (ELEMENT_NAME_PROD_CRC == type) {
            consData->prodCrcIndex = i;

        }
        else if ((ELEMENT_NAME_CONS_CRC_BASE + consData->initCrc.consIndex) ==
                 type){
            assert(i == consData->initCrc.consIndex);
            consData->consCrcIndex = i;
        }

        /* Don't need to keep attribute list */
        NvSciBufAttrListFree(bufAttr);

        /*err = NvSciStreamBlockElementUsageSet(consData->common.block, i, true);
        if (NvSciError_Success != err) {
            printf("Consumer failed (%x) to indicate element %d is used\n",
                   err, i);
            return 0;
        }*/
    }

    /* Indicate that element import is complete */
    err = NvSciStreamBlockSetupStatusSet(consData->common.block,
                                         NvSciStreamSetup_ElementImport,
                                         true);
    if (NvSciError_Success != err) {
        printf("Consumer failed (%x) to complete element import\n", err);
        return 0;
    }

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
            printf("Consumer failed (%x) to send packet status\n", err);
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
     *   Consumers can skip querying elements that they don't use.
     *   This use case has 2 elements.
     */
    for (uint32_t index = 0U; index < consData->elemCount; index++) {
        NvSciBufObj bufObj;
        err = NvSciStreamBlockPacketBufferGetWithCrc(
                consData->common.block,
                handle,
                index,
                &bufObj,
                1U,
                &consData->initCrc.buf);
        if (NvSciError_Success != err) {
            printf("Consumer failed (%x) to retrieve buffer (%lx/%d)\n",
                   err, handle, index);
            return 0;
        }

        /* CRC elements owned by other consumers not visible */
        if (bufObj == NULL) {
            continue;
        }

        /* Handle mapping of data buffer */
        NvSciError sciErr;
        int32_t    cudaRtErr;

        if (index == consData->dataIndex) {

            /* Save buffer object */
            packet->dataObj = bufObj;

            /* Map in the buffer as CUDA external memory */
            struct cudaExternalMemoryHandleDesc memHandleDesc;
            memset(&memHandleDesc, 0, sizeof(memHandleDesc));
            memHandleDesc.type = cudaExternalMemoryHandleTypeNvSciBuf;
            memHandleDesc.handle.nvSciBufObject = bufObj;
            memHandleDesc.size = consData->dataSize;
            cudaRtErr = cudaImportExternalMemory(&packet->dataExtMem,
                                                 &memHandleDesc);
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
            cudaRtErr = cudaExternalMemoryGetMappedBuffer(&packet->dataDevMem,
                                                          packet->dataExtMem,
                                                          &memBufferDesc);
            if (cudaSuccess != cudaRtErr) {
                printf("Consumer failed (%x) to map buffer as device mem\n",
                       cudaRtErr);
                return 0;
            }

            /* Allocate normal memory to use as the target for the CUDA op */
            packet->dataDstMem = (uint8_t*)malloc(consData->dataSize);
            if (NULL == packet->dataDstMem) {
                printf("Consumer failed to allocate target buffer\n");
                return 0;
            }

            /* Fill in with initial values */
            memset(packet->dataDstMem, 0xD0, consData->dataSize);

        }

        /* Handle mapping of producer's CRC buffer */
        else if (index == consData->prodCrcIndex) {

            /* Save buffer object */
            packet->prodCrcObj = bufObj;

            /* Get a CPU pointer for the buffer from NvSci */
            sciErr = NvSciBufObjGetConstCpuPtr(
                        bufObj,
                        (void const**)&packet->prodCrcPtr);
            if (NvSciError_Success != sciErr) {
                printf("Consumer failed (%x) to map CRC buffer\n", sciErr);
                return 0;
            }

        }

        /* Handle mapping of consumer's CRC buffer */
        else if (index < consData->prodCrcIndex) {
            if (index == consData->consCrcIndex) {

                /* Save buffer object */
                packet->consCrcObj = bufObj;

                /* Get a CPU pointer for the buffer from NvSci */
                sciErr = NvSciBufObjGetCpuPtr(
                            bufObj,
                            (void**)&packet->consCrcPtr);
                if (NvSciError_Success != sciErr) {
                    printf("Consumer failed (%x) to map CRC buffer\n", sciErr);
                    return 0;
                }
            } else {
                assert(bufObj == NULL);
            }
        }

        /* Shouldn't be any other index */
        else {
            printf("Consumer received buffer for unknown element (%d)\n",
                    index);
            return 0;
        }
    }

    /* Inform pool of success.
     *   Note: Could inform the pool of any of the failures above.
     */
    err = NvSciStreamBlockPacketStatusSet(consData->common.block,
                                          handle,
                                          (NvSciStreamCookie)packet,
                                          NvSciError_Success);
    if (NvSciError_Success != err) {
        printf("Consumer failed (%x) to inform pool of packet status\n", err);
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

    /* Free up the packet resources */
    deletePacket(packet);
}

/* Handle export of the init-time validation data */
static int32_t handleConsumerInitValidation(
    ConsData* consData)
{
    /* Validate the received buffers objects whether
     *  it meets its consumer's buffer requirements.
     */
    NvSciError          sciErr;
    ConsDataBufAttrs*   dataAttr = &consData->dataBufAttrs;
    ConsCrcBufAttrs*    crcAttr = &consData->crcBufAttrs;

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
        { NvSciBufGeneralAttrKey_RequiredPerm,
            &dataAttr->perm, sizeof(dataAttr->perm) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess,
            &dataAttr->cpu, sizeof(dataAttr->cpu) }
    };
    for (uint32_t p = 0; p < consData->numPacket; p++) {
        ConsPacket* packet = &consData->packets[p];

        /* Validate the crc buffer objects */
        for (uint32_t e = 0; e < consData->elemCount; e++) {
            NvSciBufObj obj;
            NvSciBufAttrKeyValuePair* pairArray;
            size_t pairArrayCount;
            if (e == consData->consCrcIndex) {
                obj = packet->consCrcObj;
                crcAttr->perm = NvSciBufAccessPerm_ReadWrite;
                pairArray = crcKeyVals;
                pairArrayCount =
                    sizeof(crcKeyVals) / sizeof(NvSciBufAttrKeyValuePair);
            } else if (e == consData->prodCrcIndex) {
                obj = packet->prodCrcObj;
                crcAttr->perm = NvSciBufAccessPerm_Readonly;
                pairArray = crcKeyVals;
                pairArrayCount =
                    sizeof(crcKeyVals) / sizeof(NvSciBufAttrKeyValuePair);
            } else if (e == consData->dataIndex){
                obj = packet->dataObj;
                pairArray = dataKeyVals;
                pairArrayCount =
                    sizeof(dataKeyVals) / sizeof(NvSciBufAttrKeyValuePair);
            } else {
                /* Skip CRC elements of other consumers */
                continue;
            }

            /* Retrieve the reconciled attribute list from the buffer object */
            NvSciBufAttrList reconciled = NULL;
            sciErr = NvSciBufObjGetAttrList(obj, &reconciled);
            if (NvSciError_Success != sciErr) {
                printf("Consumer failed (%x) to get reconciled buf attrList\n",
                    sciErr);
                return 0;
            }

            /* Verify the reconciled attribute meets the consumer's
             *   buffer requirement.
             */
            sciErr = NvSciBufAttrListValidateReconciledAgainstAttrs(
                        reconciled, pairArray, pairArrayCount);
            if (NvSciError_Success != sciErr) {
                printf("Reconciled list not meet consumer's buf attrs(%x)\n",
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
        }
    }

    /* Retrieve the reconciled attribute list from the waiter object */
    NvSciSyncAttrList syncReconciled = NULL;
    sciErr = NvSciSyncObjGetAttrList(consData->waiterObj, &syncReconciled);
    if (NvSciError_Success != sciErr) {
        printf("Consumer failed (%x) to get reconciled sync attrList\n",
                sciErr);
        return 0;
    }

    /* Verify the reconciled attribute meets the consumer's waiter
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
                syncReconciled,
                NULL, 0,
                NvSciSyncAccessPerm_WaitOnly);
    if (NvSciError_Success != sciErr) {
        printf("Reconciled list not meet consumer's waiter attrs(%x)\n",
                sciErr);
        return 0;
    }

    /* Validate the received waiter sync objects whether it meets
     *  the consumer's waiter requirement.
     */
    sciErr = NvSciSyncObjValidate(consData->waiterObj);
    if (NvSciError_Success != sciErr) {
        printf("Waiter object not satisfy the constraints(%x)\n",
                sciErr);
        return 0;
    }


    /* Retrieve the reconciled attribute list from the signal object */
    sciErr = NvSciSyncObjGetAttrList(consData->signalObj, &syncReconciled);
    if (NvSciError_Success != sciErr) {
        printf("Consumer failed (%x) to get reconciled sync attrList\n",
                sciErr);
        return 0;
    }

    /* Verify the reconciled attribute meets the consumer's signalling
     *   requirement.
     */
    sciErr = NvSciSyncAttrListValidateReconciledAgainstAttrs(
                syncReconciled,
                NULL, 0,
                NvSciSyncAccessPerm_SignalOnly);
    if (NvSciError_Success != sciErr) {
        printf("Reconciled list not meet consumer's signal attrs(%x)\n",
                sciErr);
        return 0;
    }

    /* Validate the signal sync objects. */
    sciErr = NvSciSyncObjValidate(consData->signalObj);
    if (NvSciError_Success != sciErr) {
        printf("Signal object not satisfy the constraints(%x)\n",
                sciErr);
        return 0;
    }


    /* Write all the validation data into the consumer's crc buffer
     *  Producer application will validate whether the data matched.
     */
    consData->initCrc.magicNum = CRC_INIT;
    sprintf(consData->initCrc.ipcChannel,
            "nvscistream_%d", 2 * consData->initCrc.consIndex + 1);

    memcpy(consData->packets[0].consCrcPtr,
           &consData->initCrc,
           sizeof(consData->initCrc));

    /* Indicate that crc validation data export is complete */
    sciErr = NvSciStreamBlockSetupStatusSet(consData->common.block,
                                            NvSciStreamSetup_CrcExport,
                                            true);
    if (NvSciError_Success != sciErr) {
        printf("Consumer failed (%x) to complete crc export\n",
                sciErr);
        return 0;
    }

    consData->validationDone = true;
    printf("Consumer validation PASS\n");
    return 1;
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
     * As CRC element is a synchronous element,
     * no need to query the sync object for it.
     */
    NvSciSyncAttrList waiterAttr = NULL;
    sciErr = NvSciStreamBlockElementWaiterAttrGet(consData->common.block,
                                                  consData->dataIndex,
                                                  &waiterAttr);
    if (NvSciError_Success != sciErr) {
        printf("Consumer failed (%x) to query waiter attr\n",
                sciErr);
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
    cudaRtErr = cudaImportExternalSemaphore(&consData->signalSem,
                                            &extSemDesc);
    if (cudaSuccess != cudaRtErr) {
        printf("Consumer failed (%x) to map signal object to semaphore\n",
               cudaRtErr);
        return 0;
    }

    /* Only send the sync object for the asynchronous element.
     * If this function is not called for an element,
     * the sync object is assumed to be NULL.
     * In this use case, CRC element doesn't use sync object.
     */
    sciErr = NvSciStreamBlockElementSignalObjSetWithCrc(
                consData->common.block,
                consData->dataIndex,
                consData->signalObj,
                &consData->initCrc.consSync);
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

    consData->syncExportDone = true;
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
    sciErr = NvSciStreamBlockElementSignalObjGetWithCrc(
                consData->common.block,
                0U, consData->dataIndex,
                &waiterObj,
                &consData->initCrc.prodSync);
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

    consData->syncImportDone = true;
    return 1;
}

/* Handle processing of payloads */
static int32_t handleConsumerPayload(
    ConsData*         consData)
{
    NvSciError        sciErr;
    int32_t           cudaRtErr;
    NvSciSyncFence    prefence;

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

    /* Clear the magic number in consumer's CRC element,
     *   in case that the producer tries to access the
     *   this packet.
     */
    *(uint32_t*)packet->consCrcPtr = 0;

    /*
     * Before modifying the contents of the source buffer, need to validate
     *  all the runtime data from ASIL-D consumers:
     *  - Runtime CRC magic number
     *  - Runtime CRC values
     */
    ProdPayloadCrcData* prodCrcData =
        (ProdPayloadCrcData*)packet->prodCrcPtr;

    /* Validate the magic number to ensure
     *  the data is for init-time validatoin
     */
    if (prodCrcData->magicNum != CRC_RUNTIME) {
        printf("Producer runtime magic number NOT match.\n");
        return 0;
    }

    /* For FIFO queue, validate whether the framcount is incremented by 1. */
    if (prodCrcData->frameCount < consData->counter) {
        printf("Consumer failed, receiving a old payload from producer.\n");
        return 0;
    }

    /* For mailbox queue, if it only want to process latest packet.
     *  It can check the prodCrcData->timestamp to decide whether to process.
     */

    /* If the received waiter obj if NULL,
     * the producer is done writing data into this element,
     * skip waiting on pre-fence.
     */
    if (NULL != consData->waiterObj) {


        /* Query fences from producer for data element */
        uint32_t prodFenceCrc = 0U;
        prefence = NvSciSyncFenceInitializer;
        sciErr = NvSciStreamBlockPacketFenceGetWithCrc(
                    consData->common.block,
                    packet->handle,
                    0U, consData->dataIndex,
                    &prefence,
                    &prodFenceCrc);
        if (NvSciError_Success != sciErr) {
            printf("Consumer failed (%x) to query fence from producer\n",
                    sciErr);
            return 0;
        }
        /* Validate the fence CRC.
         *   If the magic number matches and the fence CRC in the consumer's
         *   CRC element is 0, the packet may be returned by mailbox queue.
         *   Skip validating the fence.
         */
        if (prodCrcData->fence != prodFenceCrc) {
            printf("Crc of producer fence NOT match.\n");
            return 0;
        }
    }

    /*
     * Validation done. Process payload
     */
    if (NULL != consData->waiterObj) {
        /* Instruct CUDA to wait for the producer fence */
        struct cudaExternalSemaphoreWaitParams waitParams;
        memset(&waitParams, 0, sizeof(waitParams));
        waitParams.params.nvSciSync.fence = &prefence;
        waitParams.flags = 0;
        cudaRtErr = cudaWaitExternalSemaphoresAsync(&consData->waiterSem,
                                                    &waitParams, 1,
                                                    consData->cudaStream);
        if (cudaSuccess != cudaRtErr) {
            printf("Consumer failed (%x) to wait for prefence\n", cudaRtErr);
            return 0;
        }
        NvSciSyncFenceClear(&prefence);

        /*
         * Before reading the contents of the source buffer, need to validate
         *  all the runtime data from the producer:
         *  - Runtime CRC magic number
         *  - Runtime CRC values
         *  - For FIFO queue, validate the frameCount is incremented by 1.
         *  - For mailbox queue, the frameCount is incremented. Check whether
         *    the latency is accpectable.
         */
    }

    /* Instruct CUDA to copy the packet data buffer to the target buffer */
    cudaRtErr = cudaMemcpy2DAsync(packet->dataDstMem,
                                  consData->dataSize,
                                  packet->dataDevMem,
                                  consData->dataSize,
                                  consData->dataSize,
                                  1,
                                  cudaMemcpyDeviceToHost,
                                  consData->cudaStream);
    if (cudaSuccess != cudaRtErr) {
        printf("Consumer failed (%x) to issue copy command\n", cudaRtErr);
        return 0;
    }

    /* Inform CUDA to signal a fence when the copy completes */
    NvSciSyncFence postfence = NvSciSyncFenceInitializer;
    struct cudaExternalSemaphoreSignalParams signalParams;
    memset(&signalParams, 0, sizeof(signalParams));
    signalParams.params.nvSciSync.fence = &postfence;
    signalParams.flags = 0;
    cudaRtErr = cudaSignalExternalSemaphoresAsync(&consData->signalSem,
                                                  &signalParams,
                                                  1,
                                                  consData->cudaStream);
    if (cudaSuccess != cudaRtErr) {
        printf("Consumer failed (%x) to signal postfence\n", cudaRtErr);
        return 0;
    }

    /* Init the fence CRC to 0 */
    uint32_t consFenceCrc = 0;
    /* Update postfence for data element and update the CRC*/
    sciErr = NvSciStreamBlockPacketFenceSetWithCrc(
                consData->common.block,
                packet->handle,
                consData->dataIndex,
                &postfence,
                &consFenceCrc);
    if (NvSciError_Success != sciErr) {
        printf("Consumer failed (%x) to set fence\n", sciErr);
        return 0;
    }

    /* Write the valdation data into producer's CRC element
     *   before sending the packet:
     *   - Runtime CRC magic number
     *   - Runtime CRC values
     */
    ConsPayloadCrcData* consCrcData = (ConsPayloadCrcData*)packet->consCrcPtr;
    consCrcData->magicNum = CRC_RUNTIME;
    consCrcData->fence = consFenceCrc;

    /* Release the packet back to the producer */
    sciErr = NvSciStreamConsumerPacketRelease(consData->common.block,
                                              packet->handle);

    if (NvSciError_Success != sciErr) {
        printf("Consumer failed (%x) to release packet\n", sciErr);
        return 0;
    }
    NvSciSyncFenceClear(&postfence);

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
        } else {
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

        /* Initialize basic stream info */
        if (!handleStreamInit(consData)) {
            rv = -1;
        }
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
        } else {
            consData->bufDone = true;
        }
        break;

    /* Delete a packet - usually only relevant for non-safety applications */
    case NvSciStreamEventType_PacketDelete:
        handleConsumerPacketDelete(consData);
        break;

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

    /* Validate the init-time data */
    if ((rv == 1) &&
        (!consData->validationDone) &&
        consData->bufDone &&
        consData->syncExportDone &&
        consData->syncImportDone) {
        if (!handleConsumerInitValidation(consData)) {
            rv = -1;
        }
    }

    /* On failure or final event, clean up the block */
    if ((rv < 0) || (1 < rv)) {
        deleteConsumer(consData);
    }

    return rv;
}

/* Create and register a new consumer block */
int32_t createConsumer_Usecase3(
    NvSciStreamBlock* consumer,
    NvSciStreamBlock  queue,
    uint32_t          index,
    uint32_t          frames)
{
    /*
     * The index is ignored. It is provided to support use cases where
     *   there are multiple consumers that don't all do the same thing.
     */
     (void)index;

    /* The number of frames are ignored as this is significant only
    * for late/re-attach usecase.
    */
    (void)frames;

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

    /* Create a consumer block and indicate CRC validation is required */
    NvSciError err =
        NvSciStreamConsumerCreate2(queue, true, &consData->common.block);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to create consumer block\n", err);
        deleteConsumer(consData);
        return 0;
    }

    /* Register block with event handling mechanism */
    if (!eventFuncs->reg(consData->common.block, consData, handleConsumer)) {
        deleteConsumer(consData);
        return 0;
    }

    *consumer = consData->common.block;
    return 1;
}
