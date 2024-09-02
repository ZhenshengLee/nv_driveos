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
 * NvSciStream Event Loop Driven Sample App - pool block for sue case 3
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "nvscistream.h"
#include "block_info.h"
#include "event_loop.h"
#include "usecase3.h"

/* Internal data used by the pool block */
typedef struct {
    BlockData           common;
    PoolInitCrcData     initCrc;
    uint32_t            numConsumers;
    uint32_t            numProdElem;
    uint32_t            numConsElem;
    uint32_t            elemCount;
    bool                elementsDone;
    ElemAttr            prodElem[MAX_ELEMS];
    ElemAttr            consElem[MAX_ELEMS];
    ElemAttr            elem[MAX_ELEMS];
    uint32_t            numPacket;
    uint32_t            numPacketCreated;
    bool                packetsDone;
    NvSciStreamPacket   packet[MAX_PACKETS];

    /*
     * Data used in ASIL-D process
     */
    /*  Element index chosen by pool for the producer's CRC element */
    uint32_t            crcIndex;
    /* NvSci buffer object for the packet's producer CRC buffer */
    NvSciBufObj         crcObj;
    /* Pointer to the producer's CRC value */
    void*               crcPtr;
} PoolData;

/* Free up pool block resources */
static void deletePool(
    PoolData* poolData)
{
    /* Free buffer attribute lists */
    for (uint32_t e = 0; e < poolData->elemCount; ++e) {
        if (NULL != poolData->elem[e].attrList) {
            NvSciBufAttrListFree(poolData->elem[e].attrList);
            poolData->elem[e].attrList = NULL;
        }
    }

    /* Free buffer objects */
    if (poolData->crcObj) {
        NvSciBufObjFree(poolData->crcObj);
        poolData->crcObj = NULL;
    }


    /* Destroy block */
    if (poolData->common.block != 0) {
        (void)NvSciStreamBlockDelete(poolData->common.block);
    }

    /* Free data */
    free(poolData);
}

/* Handle query of basic stream info */
static int32_t handleStreamInit(
    PoolData* poolData)
{
    /* Query number of consumers */
    NvSciError err =
        NvSciStreamBlockConsumerCountGet(poolData->common.block,
                                         &poolData->numConsumers);
    if (NvSciError_Success != err) {
        printf("Pool failed (%x) to query number of consumers", err);
        return 0;
    }

    /* Query producer and consumer(s) endpoint info if needed */

    return 1;
}

/*
 * If ready, handle setup of the pool buffers.
 *
 * Most of the work the pool application has to do resides in this function.
 */
static int32_t handlePoolBufferSetup(
    PoolData* poolData)
{
    NvSciError err;

    /* Query producer element count */
    err = NvSciStreamBlockElementCountGet(poolData->common.block,
                                          NvSciStreamBlockType_Producer,
                                          &poolData->numProdElem);
    if (NvSciError_Success != err) {
        printf("Pool failed (%x) to query producer element count\n", err);
        return 0;
    }
    assert(poolData->numProdElem <= MAX_ELEMS);

    /* Query consumer element count */
    err = NvSciStreamBlockElementCountGet(poolData->common.block,
                                          NvSciStreamBlockType_Consumer,
                                          &poolData->numConsElem);
    if (NvSciError_Success != err) {
        printf("Pool failed (%x) to query consumer element count\n", err);
        return 0;
    }
    assert(poolData->numProdElem <= poolData->numConsElem);

    /* Query all producer elements */
    for (uint32_t i=0U; i<poolData->numProdElem; ++i) {
        err = NvSciStreamBlockElementAttrGet(poolData->common.block,
                                             NvSciStreamBlockType_Producer, i,
                                             &poolData->prodElem[i].userName,
                                             &poolData->prodElem[i].attrList);
        if (NvSciError_Success != err) {
            printf("Pool failed (%x) to query producer element %d\n", err, i);
            return 0;
        }
    }

    /* Query all consumer elements */
    for (uint32_t i=0U; i<poolData->numConsElem; ++i) {
        err = NvSciStreamBlockElementAttrGet(poolData->common.block,
                                             NvSciStreamBlockType_Consumer, i,
                                             &poolData->consElem[i].userName,
                                             &poolData->consElem[i].attrList);
        if (NvSciError_Success != err) {
            printf("Pool failed (%x) to query consumer element %d\n", err, i);
            return 0;
        }
    }

    /* Indicate that all element information has been imported */
    poolData->elementsDone = true;
    err = NvSciStreamBlockSetupStatusSet(poolData->common.block,
                                         NvSciStreamSetup_ElementImport,
                                         true);
    if (NvSciError_Success != err) {
        printf("Pool failed (%x) to complete element import\n", err);
        return 0;
    }

    /*
     * Go through requested elements from producer and consumer and line
     *   them up.
     */
    uint32_t numElem = 0, elemIndex = 0, nextBufName, p, c, e, i;
    ElemAttr tmp[MAX_ELEMS];
    for (p=0; p<poolData->numProdElem; ++p) {
        ElemAttr* prodElem = &poolData->prodElem[p];
        for (c=0; c<poolData->numConsElem; ++c) {
            ElemAttr* consElem = &poolData->consElem[c];

            /* If requested element types match, combine the entries */
            if (prodElem->userName == consElem->userName) {
                ElemAttr* poolElem = &tmp[numElem++];
                poolElem->userName = prodElem->userName;
                poolElem->attrList = NULL;

                /* Combine and reconcile the attribute lists */
                NvSciBufAttrList oldAttrList[2] = { prodElem->attrList,
                                                    consElem->attrList };
                NvSciBufAttrList conflicts = NULL;
                err = NvSciBufAttrListReconcile(oldAttrList, 2,
                                                &poolElem->attrList,
                                                &conflicts);

                /* Discard any conflict list.
                 *  (Could report its contents for additional debug info)
                 */
                if (NULL != conflicts) {
                    NvSciBufAttrListFree(conflicts);
                }

                /* Abort on error */
                if (NvSciError_Success != err) {
                    printf("Failed to reconcile element %x attrs (%x)\n",
                           poolElem->userName, err);
                    return 0;
                }

                /* Found a match for this producer element so move on */
                break;
            }  /* if match */
        } /* for all requested consumer elements */
    } /* for all requested producer elements */

    /* Should be at least one element */
    if (0 == numElem) {
        printf("Pool didn't find any common elements\n");
        return 0;
    }

    /* The requested attribute lists are no longer needed, so discard them */
    for (p=0; p<poolData->numProdElem; ++p) {
        ElemAttr* prodElem = &poolData->prodElem[p];
        if (NULL != prodElem->attrList) {
            NvSciBufAttrListFree(prodElem->attrList);
            prodElem->attrList = NULL;
        }
    }
    for (c=0; c<poolData->numConsElem; ++c) {
        ElemAttr* consElem = &poolData->consElem[c];
        if (NULL != consElem->attrList) {
            NvSciBufAttrListFree(consElem->attrList);
            consElem->attrList = NULL;
        }
    }

    /* Re-arrange the elements with the followng order:
     *   - ELEMENT_NAME_CONS_CRC_BASE [0..consumerCount]
     *   - ELEMENT_NAME_PROD_CRC
     *   - ELEMENT_NAME_DATA
     */
    nextBufName = ELEMENT_NAME_CONS_CRC_BASE;
    while (elemIndex < numElem) {
        for (e = 0; e < numElem; ++e) {
            ElemAttr* srcElem = &tmp[e];
            if (srcElem->userName == nextBufName) {
                poolData->elem[elemIndex].userName = srcElem->userName;
                poolData->elem[elemIndex].attrList = srcElem->attrList;
                elemIndex++;

                if (elemIndex == numElem) {
                    break;
                }

                if (nextBufName == ELEMENT_NAME_PROD_CRC) {
                    nextBufName = ELEMENT_NAME_DATA;
                } else {
                    nextBufName++;
                    if (nextBufName == (ELEMENT_NAME_CONS_CRC_BASE +
                                        poolData->numConsumers)) {
                        nextBufName = ELEMENT_NAME_PROD_CRC;
                        poolData->crcIndex = elemIndex;
                    }
                }
            }
        }
    }

    /* Inform the stream of the chosen elements */
    for (e = 0; e < numElem; ++e) {
        ElemAttr* poolElem = &poolData->elem[e];

        /* Update the CRC value with the reconciled buf attributes */
        err = NvSciStreamBlockElementAttrSetWithCrc(
                poolData->common.block,
                poolElem->userName,
                poolElem->attrList,
                &poolData->initCrc.buf);
        if (NvSciError_Success != err) {
            printf("Pool failed (%x) to send element %d info\n",
                err, e);
            return 0;
        }
    }

    /* Mark the consmer's CRC elements only visible to its owner */
    for (e = 0; e < poolData->numConsumers; ++e) {
        uint32_t uesrName = ELEMENT_NAME_CONS_CRC_BASE + e;
        for (i = 0; i < poolData->numConsumers; i++) {
            if (i == e) {
                continue;
            }
            err = NvSciStreamBlockElementNotVisible(
                    poolData->common.block,
                    uesrName,
                    i);
            if (NvSciError_Success != err) {
                printf("Failed (%x) to mark elem %d invisible to cons %d\n",
                        err, e, i);
                return 0;
            }
        }
    }

    /* Indicate that all element information has been exported */
    err = NvSciStreamBlockSetupStatusSet(poolData->common.block,
                                         NvSciStreamSetup_ElementExport,
                                         true);
    if (NvSciError_Success != err) {
        printf("Consumer failed (%x) to complete element export\n", err);
        return 0;
    }

    poolData->elemCount = numElem;
    return 1;
}

/* Handle creating a packet */
static int32_t handlePacketCreate(
    PoolData* poolData)
{
    NvSciError err;
    uint32_t i = poolData->numPacketCreated;

    /*
     * Create a new packet
     * Our pool implementation doesn't need to save any packet-specific
     *   data, but we do need to provide unique cookies, so we just
     *   use the pointer to the location we save the handle. For other
     *   blocks, this will be a pointer to the structure where the
     *   packet information is kept.
     */
    NvSciStreamCookie cookie = (NvSciStreamCookie)&poolData->packet[i];
    err = NvSciStreamPoolPacketCreate(poolData->common.block,
                                     cookie,
                                     &poolData->packet[i]);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to create packet %d\n", err, i);
        return 0;
    }

    /* Create buffers for the packet */
    for (uint32_t e = 0; e < poolData->elemCount; ++e) {

        /* Allocate a buffer object */
        NvSciBufObj obj;
        err = NvSciBufObjAlloc(poolData->elem[e].attrList, &obj);
        if (NvSciError_Success != err) {
            printf("Failed (%x) to allocate buffer %d of packet %d\n",
                    err, e, i);
            return 0;
        }

        /* Get the access pointer to producer's crc element
            *   in the first packet.
            */
        if ((i == 0U) && (e == poolData->crcIndex)) {
            err = NvSciBufObjGetCpuPtr(obj, &poolData->crcPtr);
            if (NvSciError_Success != err) {
                printf("Failed (%x) to get the CRC buffer\n", err);
                return 0;
            }
        }

        /* Insert the buffer in the packet */
        err = NvSciStreamPoolPacketInsertBufferWithCrc(
                poolData->common.block,
                poolData->packet[i],
                e, obj,
                &poolData->initCrc.buf);

        if (NvSciError_Success != err) {
            printf("Failed (%x) to insert buffer %d of packet %d\n",
                    err, e, i);
            return 0;
        }

        /* The pool doesn't need to keep a copy of the object handle */
        if ((i == 0U) && (e == poolData->crcIndex)) {
            poolData->crcObj = obj;
        }
        else {
            NvSciBufObjFree(obj);
        }
    }

    /* Indicate packet setup is complete */
    err = NvSciStreamPoolPacketComplete(poolData->common.block,
                                        poolData->packet[i]);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to complete packet %d setup\n", err, i);
        return 0;
    }

    poolData->numPacketCreated++;

    /* All packets are created */
    if (poolData->numPacketCreated == poolData->numPacket) {
        /* Write the buffer CRC values into the producer's crc element. */
        poolData->initCrc.magicNum = CRC_INIT;
        memcpy(poolData->crcPtr,
               &poolData->initCrc,
               sizeof(poolData->initCrc));

        /* Indicate that all packets have been sent. */
        err = NvSciStreamBlockSetupStatusSet(poolData->common.block,
                                             NvSciStreamSetup_PacketExport,
                                             true);
        if (NvSciError_Success != err) {
            printf("Failed (%x) to complete packet export\n", err);
            return 0;
        }

        return 1;
    }
    return 1;
}

static int32_t handlePacketsStatus(
    PoolData* poolData)
{
    /* Check packet acceptance */
    bool packetFailure = false, accept;
    NvSciError err, status;
    uint32_t p = poolData->numPacketCreated - 1;

    err = NvSciStreamPoolPacketStatusAcceptGet(poolData->common.block,
                                               poolData->packet[p],
                                               &accept);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to retrieve packet %d's acceptance-statue\n",
                err, p);
        return 0;
    }

    /* On rejection, query and report details */
    if (!accept) {
        packetFailure = true;
    }

    /* Check packet status from producer */
    err = NvSciStreamPoolPacketStatusValueGet(
            poolData->common.block,
            poolData->packet[p],
            NvSciStreamBlockType_Producer, 0U,
            &status);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to retrieve packet %d's statue from producer\n",
                err, p);
        return 0;
    }
    if (status != NvSciError_Success) {
        printf("Producer rejected packet %d with error %x\n", p, status);
    }

    /* Check packet status from consumers */
    for (uint32_t c = 0; c < poolData->numConsumers; ++c) {
        err = NvSciStreamPoolPacketStatusValueGet(
                poolData->common.block,
                poolData->packet[p],
                NvSciStreamBlockType_Consumer, c,
                &status);
        if (NvSciError_Success != err) {
            printf("Failed (%x) to retrieve packet %d's statue from consumer %d\n",
                    err, p, c);
            return 0;
        }
        if (status != NvSciError_Success) {
            printf("Consumer %d rejected packet %d with error %x\n",
                    c, p, status);
        }
    }

    if (packetFailure) {
        return 0;
    }

    if (poolData->numPacketCreated == poolData->numPacket) {
        /* Indicate that status for all packets has been received. */
        poolData->packetsDone = true;
        err = NvSciStreamBlockSetupStatusSet(poolData->common.block,
                                             NvSciStreamSetup_PacketImport,
                                             true);
        if (NvSciError_Success != err) {
            printf("Pool failed (%x) to complete packets import\n", err);
            return 0;
        }
    }
    return 1;
}

/* Handle events on a pool block
 *
 * The pool block coordinates allocation of packets based on producer
 *   and consumer requirements during setup. After that, no further
 *   events should be received until the stream is torn down.
 */
static int32_t handlePool(
    void*     data,
    uint32_t  wait)
{
    /* Cast to pool data */
    PoolData* poolData = (PoolData*)data;

    /* Get time to wait */
    int64_t waitTime = wait ? poolData->common.waitTime : 0;

    /* Query/wait for an event on the block */
    NvSciStreamEventType event;
    NvSciError       err;
    err = NvSciStreamBlockEventQuery(poolData->common.block, waitTime, &event);

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
            printf("Pool timed out waiting for setup instructions\n");
        } else {
            printf("Pool event query failed with error %x\n", err);
        }
        deletePool(poolData);
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
        printf("Pool received unknown event %x\n", event);

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
        err = NvSciStreamBlockErrorGet(poolData->common.block, &status);
        if (NvSciError_Success != err) {
            printf("%s Failed to query the error event code %x\n",
                   poolData->common.name, err);
        } else {
            printf("%s received error event: %x\n",
                   poolData->common.name, status);
        }

        rv = -1;
        break;

    /*
     * If told to disconnect, it means either the stream finished its
     *   business or some other block had a failure. We'll just do a
     *   clean up and return without an error. But if it happened before
     *   all the pool setup operations finished, we'll report it for
     *   debugging purposes.
     */
    case NvSciStreamEventType_Disconnected:
        if (!poolData->elementsDone) {
            printf("Warning: Pool disconnect before element support\n");
        } else if (!poolData->packetsDone) {
            printf("Warning: Pool disconnect before packet setup\n");
        }
        rv = 2;
        break;

    /*
     * The pool doesn't have to do anything immediately on connection, but
     *   now that the stream is complete we can reduce the timeout to wait
     *   for the producer and consumer events to arrive.
     */
    case NvSciStreamEventType_Connected:
        /* Initialize basic stream info */
        if (!handleStreamInit(poolData)) {
            rv = -1;
        }

        poolData->common.waitTime = 10 * 1000000;
        break;

    /* Process all element support from producer and consumer(s) */
    case NvSciStreamEventType_Elements:
        /* Process elements */
        if (!handlePoolBufferSetup(poolData)) {
            rv = -1;
        }
        /* Create a new packet and allocate buffers in it*/
        else if (!handlePacketCreate(poolData)) {
            rv = -1;
        }
        break;

    /*
     * Check packet/buffer status returned from producer/consumer
     *   A more sophisticated application might have the means to recover
     *   from any failures. But in general we expect that in a production
     *   application, any failures are due to something fundamental going
     *   wrong like lack of memory/resources, which hopefully has been
     *   designed out. So these status checks are more useful during
     *   development, where we just report the issue for debugging purposes.
     *
     * Once all the status events have been received for all packets
     *   and buffers, the pool should require no further interaction
     *   until the time comes to shut down the application. We set the
     *   wait time to infinite.
     */
    case NvSciStreamEventType_PacketStatus:
        /*
         * In this use case, create packet and then check packet status
         *  one by one.
         */
        if (!handlePacketsStatus(poolData)) {
            rv = -1;
        }
        /* Create another new packet if not all packets created */
        else if (poolData->numPacketCreated < poolData->numPacket) {
            if (!handlePacketCreate(poolData)) {
                rv = -1;
            } else {
                poolData->common.waitTime = -1;
            }
        }
        break;

    /* All setup complete. Transition to runtime phase */
    case NvSciStreamEventType_SetupComplete:
        break;
    }

    /* On failure or final event, clean up the block */
    if ((rv < 0) || (1 < rv)) {
        deletePool(poolData);
    }

    return rv;
}


/* Create and register a new pool block */
int32_t createPool_Usecase3(
    NvSciStreamBlock* pool,
    uint32_t          numPacket,
    bool              isC2cPool)
{
    /* Use case 3 not supported in C2C */
    (void)isC2cPool;

    /* Create a data structure to track the block's status */
    PoolData* poolData = (PoolData*)calloc(1, sizeof(PoolData));
    if (NULL == poolData) {
        printf("Failed to allocate data structure for pool\n");
        return 0;
    }

    /* Save the name for debugging purposes */
    strcpy(poolData->common.name, "Pool");

    /* Save the packet count */
    poolData->numPacket = numPacket;

    /* Wait time for initial connection event will be 60 seconds */
    poolData->common.waitTime = 60 * 1000000;

    /* Create a pool block */
    NvSciError err =
        NvSciStreamStaticPoolCreate(poolData->numPacket,
                                    &poolData->common.block);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to create pool block\n", err);
        deletePool(poolData);
        return 0;
    }

    /* Register block with event handling mechanism */
    if (!eventFuncs->reg(poolData->common.block, poolData, handlePool)) {
        deletePool(poolData);
        return 0;
    }

    *pool = poolData->common.block;
    return 1;
}
