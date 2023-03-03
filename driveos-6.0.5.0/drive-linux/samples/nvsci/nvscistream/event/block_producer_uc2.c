/* NvSciStream Event Loop Driven Sample App - producer block for use case 2
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
 * This implements the producer for use case 2: nvmedia to cuda streaming
 */

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include "nvscibuf.h"
#include "nvscisync.h"
#include "nvscistream.h"
#include "nvmedia_core.h"
#include "nvmedia_2d.h"
#include "nvmedia_2d_sci.h"
#include "block_info.h"
#include "event_loop.h"
#include "usecase2.h"

/* Dimensions for buffers */
uint32_t const WIDTH  = 1920;
uint32_t const HEIGHT = 1080;

/* Internal data structure used to track packets */
typedef struct {
    /* The packet handle use for NvSciStream functions */
    NvSciStreamPacket    handle;

    /* NvSci buffer for source image */
    NvSciBufObj          srcBuf;

    /* NvSci buffer object for the packet's data buffer */
    NvSciBufObj          dataBuf;

} ProdPacket;

/* Internal data used by the producer block */
typedef struct {
    /* Common block info */
    BlockData               common;

    /* Number of consumers */
    uint32_t                numConsumers;

    /* NvMedia device/engine access */
    NvMedia2D*              nvm2d;

    /* Attributes to use for allocating source buffers */
    NvSciBufAttrList        sourceAttr;

    /* NvSciSync context to do CPU waiting for fences */
    NvSciSyncCpuWaitContext cpuWaitContext;

    /* Sync attributes for CPU waiting */
    NvSciSyncAttrList       cpuWaitAttr;

    /* NvMeida sync attributes required for signaling */
    NvSciSyncAttrList       signalAttr;
    /* NvMeida sync attributes required for waiting */
    NvSciSyncAttrList       waiterAttr;
    /* Sync object for NvMedia to signal after generating data */
    NvSciSyncObj            signalObj;
    /* Sync objects to wait for before generating data */
    NvSciSyncObj            waiterObj[MAX_CONSUMERS];

    /* Number of packets provided by pool */
    uint32_t                numPacket;
    /* Information about each packet */
    ProdPacket              packets[MAX_PACKETS];

    /* Number of payloads generated so far */
    uint32_t                counter;
    /* Flag indicating producer has finished generating all payloads */
    uint32_t                finished;
} ProdData;

/* Free up producer block resources */
static void deleteProducer(
    ProdData* prodData)
{
    /* Destroy block */
    if (prodData->common.block != 0) {
        (void)NvSciStreamBlockDelete(prodData->common.block);
    }

    /* Unregister the buffers */
    ProdPacket *packet;
    for (uint32_t i=0;i<prodData->numPacket;i++)
    {
        packet = &prodData->packets[i];
        if (packet->dataBuf != NULL) {
            (void)NvMedia2DUnregisterNvSciBufObj(prodData->nvm2d,
                                                packet->dataBuf);
            NvSciBufObjFree(packet->dataBuf);
            packet->dataBuf = NULL;
        }
        if (packet->srcBuf != NULL) {
            (void)NvMedia2DUnregisterNvSciBufObj(prodData->nvm2d,
                                                 packet->srcBuf);
            NvSciBufObjFree(packet->srcBuf);
            packet->srcBuf = NULL;
        }

        /* Clear out packet information */
        memset(packet, 0, sizeof(ProdPacket));
    }

    /* Free the sync objects */
    if (prodData->signalObj != NULL) {
        (void)NvMedia2DUnregisterNvSciSyncObj(prodData->nvm2d,
                                            prodData->signalObj);
        NvSciSyncObjFree(prodData->signalObj);
        prodData->signalObj = NULL;
    }

    for (uint32_t i=0; i< prodData->numConsumers; i++)
    {
        if (prodData->waiterObj[i] != NULL) {
            (void)NvMedia2DUnregisterNvSciSyncObj(prodData->nvm2d,
                                                prodData->waiterObj[i]);
            NvSciSyncObjFree(prodData->waiterObj[i]);
            prodData->waiterObj[i] = NULL;
        }
    }

    /* Free the buffer attributes */
    if (prodData->sourceAttr != NULL) {
        NvSciBufAttrListFree(prodData->sourceAttr);
        prodData->sourceAttr = NULL;
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

    /* Destory NvMedia2D */
    if (prodData->nvm2d != NULL)
    {
        (void)NvMedia2DDestroy(prodData->nvm2d);
        prodData->nvm2d = NULL;
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

/* Handle initialization of NvMedia resources for producer */
static int32_t handleProducerInit(
    ProdData* prodData)
{
    NvMediaStatus          nvmErr;

    /* Create NvMedia 2D engine access */
    nvmErr = NvMedia2DCreate(&prodData->nvm2d, NULL);
    if (NVMEDIA_STATUS_OK != nvmErr) {
        printf("Producer failed (%x) to create 2D NvMedia object\n",
            nvmErr);
        return 0;
    }

    return 1;
}

/* Handle setup of supported buffer attributes */
static int32_t handleProducerElemSupport(
    ProdData*     prodData)
{
    NvSciError             sciErr;

    uint32_t               bufName = ELEMENT_NAME_IMAGE;
    NvSciBufAttrList       bufAttr = NULL;

    /* Create unreconciled attribute list for NvMedia buffers */
    sciErr = NvSciBufAttrListCreate(sciBufModule, &bufAttr);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to create buffer attribute list\n",
               sciErr);
        return 0;
    }

    /* Add read/write permission to attribute list */
    NvSciBufAttrValAccessPerm bufPerm = NvSciBufAccessPerm_ReadWrite;
    NvSciBufAttrKeyValuePair  bufKeyVal =
        { NvSciBufGeneralAttrKey_RequiredPerm, &bufPerm, sizeof(bufPerm) };
    sciErr = NvSciBufAttrListSetAttrs(bufAttr, &bufKeyVal, 1);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to set source permission\n", sciErr);
        return 0;
    }

    /* Get Nvmedia surface type for A8R8G8B8 buffers */
    NvSciBufType bufType = NvSciBufType_Image;
    NvSciBufAttrValColorFmt colorFmt = NvSciColor_A8R8G8B8;
    NvSciBufAttrValImageLayoutType layout = NvSciBufImage_PitchLinearType;
    uint32_t planeCount = 1;
    NvSciBufAttrKeyValuePair bufFormatKeyVal[4] =
        {
            { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
            { NvSciBufImageAttrKey_PlaneColorFormat, &colorFmt, sizeof(colorFmt) },
            { NvSciBufImageAttrKey_Layout, &layout, sizeof(layout) },
            { NvSciBufImageAttrKey_PlaneCount, &planeCount, sizeof(planeCount) },
        };

    sciErr = NvSciBufAttrListSetAttrs(bufAttr, bufFormatKeyVal, 4);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to set buffer format attributes\n", sciErr);
        return 0;
    }

    /* Set NvMedia surface allocation attributes */
    bool enableCpuCache = true;
    bool needCpuAccess = true;
    uint64_t topPadding = 0;
    uint64_t bottomPadding = 0;
    NvSciBufAttrValColorStd colorStd = NvSciColorStd_REC601_ER;
    NvSciBufAttrValImageScanType scanType = NvSciBufScan_ProgressiveType;
    NvSciBufAttrKeyValuePair bufAllocKeyVal[8] =
        {
            { NvSciBufImageAttrKey_PlaneWidth, &WIDTH, sizeof(WIDTH) },
            { NvSciBufImageAttrKey_PlaneHeight, &HEIGHT, sizeof(HEIGHT) },
            { NvSciBufGeneralAttrKey_NeedCpuAccess, &needCpuAccess, sizeof(needCpuAccess) },
            { NvSciBufGeneralAttrKey_EnableCpuCache, &enableCpuCache, sizeof(enableCpuCache) },
            { NvSciBufImageAttrKey_TopPadding, &topPadding, sizeof(topPadding) },
            { NvSciBufImageAttrKey_BottomPadding, &bottomPadding, sizeof(bottomPadding) },
            { NvSciBufImageAttrKey_PlaneColorStd, &colorStd, sizeof(colorStd) },
            { NvSciBufImageAttrKey_ScanType, &scanType, sizeof(scanType) }
        };

    sciErr = NvSciBufAttrListSetAttrs(bufAttr, bufAllocKeyVal, 8);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to set buffer alloc attributes\n", sciErr);
        return 0;
    }

    /* Setting VIC specific buffer attributes */
    NvMediaStatus nvmErr = NvMedia2DFillNvSciBufAttrList(prodData->nvm2d, bufAttr);
    if (NVMEDIA_STATUS_OK != nvmErr) {
        printf("Producer failed (%x) to set 2D attributes\n", nvmErr);
        return 0;
    }

    /* Inform stream of the attributes */
    sciErr = NvSciStreamBlockElementAttrSet(prodData->common.block,
                                            bufName, bufAttr);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to send element attribute\n", sciErr);
        return 0;
    }

    /* Indicate that all element information has been exported */
    sciErr = NvSciStreamBlockSetupStatusSet(prodData->common.block,
                                            NvSciStreamSetup_ElementExport,
                                            true);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to complete element export\n", sciErr);
        return 0;
    }

    /* Also reconcile and save the attributes for source buffer creation */
    NvSciBufAttrList conflicts = NULL;
    sciErr = NvSciBufAttrListReconcile(&bufAttr, 1,
                                       &prodData->sourceAttr, &conflicts);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to reconcile source attributes\n",
               sciErr);
        return 0;
    }

    /* Clean up */
    NvSciBufAttrListFree(bufAttr);
    if (NULL != conflicts) {
        NvSciBufAttrListFree(conflicts);
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
     *   know it is always 1. But we do so anyways to show how it is done.
     */
    uint32_t count;
    err = NvSciStreamBlockElementCountGet(prodData->common.block,
                                          NvSciStreamBlockType_Pool,
                                          &count);
    if (NvSciError_Success != err) {
        printf("Producer failed (%x) to query element count\n", err);
        return 0;
    }
    if (1U != count) {
        printf("Producer received unexpected element count (%d)\n", count);
        return 0;
    }

    /*
     * Query element type and attributes.
     *   For this simple use case, there is only one type and the attribute
     *   list is not needed, so we could skip this call. We do it only to
     *   illustrate how it is done.
     */
    uint32_t type;
    NvSciBufAttrList bufAttr;
    err = NvSciStreamBlockElementAttrGet(prodData->common.block,
                                         NvSciStreamBlockType_Pool, 0U,
                                         &type, &bufAttr);
    if (NvSciError_Success != err) {
        printf("Producer failed (%x) to query element attr\n", err);
        return 0;
    }

    /* Validate type */
    if (ELEMENT_NAME_IMAGE != type) {
        printf("Producer received unknown element type (%x)\n", type);
        return 0;
    }

    /* Don't need to keep attribute list */
    NvSciBufAttrListFree(bufAttr);

    /* Indicate that element import is complete */
    err = NvSciStreamBlockSetupStatusSet(prodData->common.block,
                                         NvSciStreamSetup_ElementImport,
                                         true);
    if (NvSciError_Success != err) {
        printf("Producer failed (%x) to complete element import\n", err);
        return 0;
    }

    /* Set waiter attributes for the asynchronous element. */
    err = NvSciStreamBlockElementWaiterAttrSet(prodData->common.block,
                                               0U, prodData->waiterAttr);
    if (NvSciError_Success != err) {
        printf("Producer failed (%x) to send waiter atts\n", err);
        return 0;
    }

    /* Once sent, the waiting attributes are no longer needed */
    NvSciSyncAttrListFree(prodData->waiterAttr);
    prodData->waiterAttr = NULL;

    /* Indicate that waiter attribute export is done. */
    err = NvSciStreamBlockSetupStatusSet(prodData->common.block,
                                         NvSciStreamSetup_WaiterAttrExport,
                                         true);
    if (NvSciError_Success != err) {
        printf("Producer failed (%x) to complete waiter attr export\n",
               err);
        return 0;
    }

    return 1;
}

/* Handle creation of a new packet */
static int32_t handleProducerPacketCreate(
    ProdData*         prodData)
{
    NvSciError err;
    NvMediaStatus nvmErr;

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
            printf("Producer failed (%x) to inform pool of packet status\n",
                   err);
        }
        return 0;
    }

    /* Allocate the next entry in the array for the new packet. */
    NvSciError    sciErr;
    uint32_t p         = prodData->numPacket++;
    ProdPacket* packet = &prodData->packets[p];
    packet->handle = handle;

    /* Allocate a source buffer for the NvMedia operations */
    sciErr = NvSciBufObjAlloc(prodData->sourceAttr, &packet->srcBuf);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to allocate source buffer\n", sciErr);
    } else {
        /* Register the source buffer */
        nvmErr = NvMedia2DRegisterNvSciBufObj(prodData->nvm2d, packet->srcBuf);
        if (NVMEDIA_STATUS_OK != nvmErr) {
            printf("Producer failed (%x) to register sources buffer.\n", nvmErr);
            return 0;
        }

        /* Get CPU pointer */
        uint8_t *cpu_ptr = NULL;
        sciErr = NvSciBufObjGetCpuPtr(packet->srcBuf, (void **)&cpu_ptr);
        if (NvSciError_Success != sciErr) {
             printf("Producer failed (%x) to get cpu pointer\n", sciErr);
        } else {
            /* Get width, height and pitch attributes */
            NvSciBufAttrKeyValuePair attr[] =
            {
               { NvSciBufImageAttrKey_PlaneWidth, NULL,0},
               { NvSciBufImageAttrKey_PlaneHeight, NULL, 0},
               { NvSciBufImageAttrKey_PlanePitch, NULL, 0}
            };

            sciErr = NvSciBufAttrListGetAttrs(prodData->sourceAttr, attr,
                                     sizeof(attr)/sizeof(NvSciBufAttrKeyValuePair));
            if (NvSciError_Success != sciErr) {
                printf("Producer failed (%x) to get attributes\n", sciErr);
            }

            if (NvSciError_Success == sciErr) {
               uint32_t width  = *(uint32_t *)attr[0].value;
               uint32_t height = *(uint32_t *)attr[1].value;
               uint32_t pitch  = *(uint32_t *)attr[2].value;
               uint32_t nWidth = 4U;
               uint8_t* srcPtr = cpu_ptr;

               for (uint32_t y = 0U; y < height; y++) {
                  for (uint32_t x = 0U; x < width * nWidth; x++) {
                      srcPtr[x] = (p + ((x % 32)+ (y % 32) )) % (1 << 8);
                  }
                  srcPtr += pitch;
               }
            }
        }
    }

    /* Inform pool of failure */
    if (NvSciError_Success != sciErr) {
        err = NvSciStreamBlockPacketStatusSet(prodData->common.block,
                                              handle,
                                              NvSciStreamCookie_Invalid,
                                              sciErr);
        if (NvSciError_Success != err) {
            printf("Producer failed (%x) to inform pool of packet status\n",
                   err);
        }
        return 0;
    }


    /* Handle mapping of a packet buffer */
    sciErr = NvSciError_Success;

    /* Retrieve all buffers and map into application
     *   This use case has only 1 element.
     */
    NvSciBufObj bufObj;
    err = NvSciStreamBlockPacketBufferGet(prodData->common.block,
                                          handle,
                                          0U,
                                          &bufObj);
    if (NvSciError_Success != err) {
        printf("Producer failed (%x) to retrieve buffer (%lx/0)\n",
               err, handle);
        return 0;
    }

    /* Save buffer object */
    packet->dataBuf = bufObj;

    /* Register the data buffer */
    nvmErr = NvMedia2DRegisterNvSciBufObj(prodData->nvm2d, packet->dataBuf);
    if (NVMEDIA_STATUS_OK != nvmErr) {
        printf("Producer failed (%x) to register data buffer.\n", nvmErr);
        return 0;
    }

    /* Get datda buffer attributes */
    NvSciBufAttrList databufAttr = NULL;
    sciErr = NvSciBufObjGetAttrList(packet->dataBuf, &databufAttr);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to get attribute list\n", sciErr);
        return 0;
    }

    /* Get CPU pointer */
    uint8_t *cpu_ptr2 = NULL;
    sciErr = NvSciBufObjGetCpuPtr(packet->dataBuf, (void **)&cpu_ptr2);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to get cpu pointer\n", sciErr);
        return 0;
    }

    /* Get width, height and pitch attributes */
    NvSciBufAttrKeyValuePair attr2[] =
    {
        { NvSciBufImageAttrKey_PlaneHeight, NULL, 0},
        { NvSciBufImageAttrKey_PlanePitch, NULL, 0}
    };

    sciErr = NvSciBufAttrListGetAttrs(databufAttr, attr2,
                                      sizeof(attr2)/sizeof(NvSciBufAttrKeyValuePair));
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to get attributes\n", sciErr);
        return 0;
    } else {
        uint32_t height2 = *(uint32_t *)attr2[0].value;
        uint32_t pitch2  = *(uint32_t *)attr2[1].value;
        uint8_t* data_ptr = cpu_ptr2;

        (void)memset(data_ptr, 0, pitch2 * height2);
    }

    /* Inform pool of succes or failure */
    if (NvSciError_Success != sciErr) {
        err = NvSciStreamBlockPacketStatusSet(prodData->common.block,
                                              packet->handle,
                                              NvSciStreamCookie_Invalid,
                                              sciErr);
    } else {
        err = NvSciStreamBlockPacketStatusSet(prodData->common.block,
                                              packet->handle,
                                              (NvSciStreamCookie)packet,
                                              NvSciError_Success);
    }
    if (NvSciError_Success != err) {
        printf("Producer failed (%x) to inform pool of packet status\n",
               err);
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

    /* Free buffer objects */
    if (packet->dataBuf) {
        NvSciBufObjFree(packet->dataBuf);
    }
    if (packet->srcBuf) {
        NvSciBufObjFree(packet->srcBuf);
    }

    /* Clear out packet information */
    memset(packet, 0, sizeof(ProdPacket));
}

/* Handle setup of supported sync attributes */
static int32_t handleProducerSyncSupport(
    ProdData* prodData)
{
    NvSciError       sciErr;
    NvMediaStatus    nvmErr;

    /*
     * Create sync attribute list for signaling.
     *   This will be saved until we receive the consumer's attributes
     */
    sciErr = NvSciSyncAttrListCreate(sciSyncModule, &prodData->signalAttr);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to allocate signal sync attrs\n", sciErr);
        return 0;
    }

    /* Have NvMedia fill the signaling attribute list */
    nvmErr = NvMedia2DFillNvSciSyncAttrList(prodData->nvm2d,
                                            prodData->signalAttr,
                                            NVMEDIA_SIGNALER);
    if (NVMEDIA_STATUS_OK != nvmErr) {
        printf("Producer failed (%x) to fill signal sync attrs\n", nvmErr);
        return 0;
    }

    /* Create sync attribute list for waiting. */
    sciErr = NvSciSyncAttrListCreate(sciSyncModule, &prodData->waiterAttr);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to allocate waiter sync attrs\n", sciErr);
        return 0;
    }

    /* Have NvMedia fill the waiting attribute list */
    nvmErr = NvMedia2DFillNvSciSyncAttrList(prodData->nvm2d,
                                            prodData->waiterAttr,
                                            NVMEDIA_WAITER);
    if (NVMEDIA_STATUS_OK != nvmErr) {
        printf("Producer failed (%x) to fill waiter sync attrs\n", nvmErr);
        return 0;
    }

    /*
     * To ensure that the producers have completed all operations before
     *  unregistering buffers and cleaning up resources, the producers will
     *  need the ability to do CPU waits on the last fence generated by
     *  its sync object.
     */

    uint8_t                   cpuSync = 1;
    NvSciSyncAccessPerm       cpuPerm = NvSciSyncAccessPerm_WaitOnly;
    NvSciSyncAttrKeyValuePair cpuKeyVals[] = {
        { NvSciSyncAttrKey_NeedCpuAccess, &cpuSync, sizeof(cpuSync) },
        { NvSciSyncAttrKey_RequiredPerm,  &cpuPerm, sizeof(cpuPerm) }
    };

    /* Create attribute list for CPU waiting */
    sciErr = NvSciSyncAttrListCreate(sciSyncModule,
                                &prodData->cpuWaitAttr);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to create CPU waiter attibute list\n",
        sciErr);
        return 0;
    }

    /* Fill attribute list for CPU waiting */
    sciErr = NvSciSyncAttrListSetAttrs(prodData->cpuWaitAttr,
                                    cpuKeyVals, 2);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to fill cpu wait sync attrs\n", sciErr);
        return 0;
    }

    /* Create a context for CPU waiting */
    sciErr = NvSciSyncCpuWaitContextAlloc(sciSyncModule,
                                &prodData->cpuWaitContext);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to create CPU wait context\n", sciErr);
        return 0;
    }

    return 1;
}

/* Handle creation and export of producer sync object */
static int32_t handleProducerSyncExport(
    ProdData*         prodData)
{
    NvSciError        sciErr;
    NvMediaStatus     nvmErr;

    /* Process waiter attrs from all elements.
     * This use case has only one element.
     */
    NvSciSyncAttrList waiterAttr = NULL;
    sciErr = NvSciStreamBlockElementWaiterAttrGet(prodData->common.block,
                                                  0U, &waiterAttr);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to query waiter attr\n", sciErr);
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
     *       add in attributes to allow us to CPU wait for the last fence
     *       generated by its sync object.
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

    /* Register sync object with NvMedia */
    nvmErr = NvMedia2DRegisterNvSciSyncObj(prodData->nvm2d,
                                           NVMEDIA_EOFSYNCOBJ,
                                           prodData->signalObj);
    if (NVMEDIA_STATUS_OK != nvmErr) {
        printf("Producer failed (%x) to register signal sync object\n",
               nvmErr);
        return 0;
    }

    /* Send the sync object for each element */
    sciErr = NvSciStreamBlockElementSignalObjSet(prodData->common.block,
                                                 0U,
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
    NvMediaStatus     nvmErr;

    NvSciError        sciErr;

    /* Query sync objects for each element
     * from all consumers.
     */
    for (uint32_t c = 0U; c < prodData->numConsumers; c++) {
        NvSciSyncObj waiterObj = NULL;
        sciErr = NvSciStreamBlockElementSignalObjGet(
                    prodData->common.block,
                    c, 0U, &waiterObj);
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
            /* Register sync object with NvMedia */
            nvmErr = NvMedia2DRegisterNvSciSyncObj(prodData->nvm2d,
                                                   NVMEDIA_PRESYNCOBJ,
                                                   waiterObj);
            if (NVMEDIA_STATUS_OK != nvmErr) {
                printf("Producer failed (%x) to register waiter sync object\n",
                       nvmErr);
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
    NvMediaStatus     nvmErr;

    /* Obtain packet for the new payload */
    NvSciStreamCookie cookie;
    sciErr = NvSciStreamProducerPacketGet(prodData->common.block,
                                          &cookie);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to obtain packet for payload\n", sciErr);
        return 0;
    }
    ProdPacket* packet = (ProdPacket*)cookie;

    NvMedia2DComposeParameters params;
    nvmErr = NvMedia2DGetComposeParameters(prodData->nvm2d,
                                               &params);
    if (NVMEDIA_STATUS_OK != nvmErr) {
        printf("Producer failed (%x) to get compose parameters\n",
                nvmErr);
        return 0;
    }

    nvmErr = NvMedia2DSetNvSciSyncObjforEOF(prodData->nvm2d,
                                            params,
                                            prodData->signalObj);
    if (NVMEDIA_STATUS_OK != nvmErr) {
        printf("Producer failed (%x) to set EOF sync object\n", nvmErr);
        return 0;
    }

    /* Query fences for this element from each consumer */
    for (uint32_t i = 0U; i < prodData->numConsumers; ++i) {
        /* If the received waiter obj if NULL,
         * the consumer is done using this element,
         * skip waiting on pre-fence.
         */
        if (NULL == prodData->waiterObj[i]) {
            continue;
        }

        NvSciSyncFence prefence = NvSciSyncFenceInitializer;
        sciErr = NvSciStreamBlockPacketFenceGet(
                    prodData->common.block,
                    packet->handle,
                    i, 0U,
                    &prefence);
        if (NvSciError_Success != sciErr) {
            printf("Producer failed (%x) to query fence from cons %d\n",
                   sciErr, i);
            return 0;
        }

        /* Instruct NvMedia to wait for each of the consumer fences */
        nvmErr = NvMedia2DInsertPreNvSciSyncFence(prodData->nvm2d,
                                                  params,
                                                  &prefence);
        NvSciSyncFenceClear(&prefence);

        if (NVMEDIA_STATUS_OK != nvmErr) {
            printf("Producer failed (%x) to wait for prefence %d\n",
                   nvmErr, i);
            return 0;
        }
    }

    uint32_t index = 0;
    nvmErr = NvMedia2DSetSrcNvSciBufObj(prodData->nvm2d, params, index, packet->srcBuf);
    if (NVMEDIA_STATUS_OK != nvmErr) {
        printf("Producer failed (%x) to set source buf\n",
               nvmErr);
        return 0;
    }

    nvmErr = NvMedia2DSetDstNvSciBufObj(prodData->nvm2d, params, packet->dataBuf);
    if (NVMEDIA_STATUS_OK != nvmErr) {
        printf("Producer failed (%x) to set source buf\n",
               nvmErr);
        return 0;
    }

    NvMedia2DComposeResult result;
    nvmErr = NvMedia2DCompose(prodData->nvm2d, params, &result);
    if (NVMEDIA_STATUS_OK != nvmErr) {
        printf("Producer failed (%x) to compose\n",
               nvmErr);
        return 0;
    }

    /* Instruct NvMedia to signal the post fence */
    NvSciSyncFence postfence = NvSciSyncFenceInitializer;
    nvmErr = NvMedia2DGetEOFNvSciSyncFence(prodData->nvm2d,
                                           &result,
                                           &postfence);
    if (NVMEDIA_STATUS_OK != nvmErr) {
        printf("Producer failed (%x) to signal postfence\n", nvmErr);
        return 0;
    }

    /* Update postfence for this element */
    sciErr = NvSciStreamBlockPacketFenceSet(prodData->common.block,
                                            packet->handle,
                                            0U,
                                            &postfence);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to set fence\n", sciErr);
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
    if (++(prodData->counter) == 32) {
        /* Make sure all operations have been completed
         * before resource cleanup.
         */
        sciErr =  NvSciSyncFenceWait(&postfence,
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

    NvSciSyncFenceClear(&postfence);

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
        /* Initialize NvMedia access */
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

    /* Set up signaling sync object from consumer's wait attributes */
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
int32_t createProducer_Usecase2(
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
