/* NvSciStream Event Loop Driven Sample App - pool block
 *
 * Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software, related documentation and any
 * modifications thereto. Any use, reproduction, disclosure or distribution
 * of this software and related documentation without an express license
 * agreement from NVIDIA Corporation is strictly prohibited.
 */

#include <stdlib.h>
#include <stdio.h>
#include "nvscistream.h"
#include "block_info.h"
#include "event_loop.h"

/* Internal data used by the pool block */
typedef struct {
    BlockData         common;
    bool              isC2cPool;
    uint32_t          numConsumers;
    uint32_t          numProdElem;
    uint32_t          numConsElem;
    bool              elementsDone;
    ElemAttr          prodElem[MAX_ELEMS];
    ElemAttr          consElem[MAX_ELEMS];
    uint32_t          numPacket;
    uint32_t          numPacketReady;
    bool              packetsDone;
    NvSciStreamPacket packet[MAX_PACKETS];
} PoolData;

/* Free up pool block resources */
static void deletePool(
    PoolData* poolData)
{
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

    /* Query consumer element count */
    err = NvSciStreamBlockElementCountGet(poolData->common.block,
                                          NvSciStreamBlockType_Consumer,
                                          &poolData->numConsElem);
    if (NvSciError_Success != err) {
        printf("Pool failed (%x) to query consumer element count\n", err);
        return 0;
    }

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
     *   them up. A general streaming application might not have a one to
     *   one correspondence, and the pool may have to decide what subset
     *   of elements to select based on knowledge of the data types that
     *   the application suite supports. This sample application is much
     *   simpler, but we still go through the process rather than assuming
     *   producer and consumer have requested the same things in the same
     *   order.
     */
    uint32_t numElem = 0, p, c, e, i;
    ElemAttr elem[MAX_ELEMS];
    for (p=0; p<poolData->numProdElem; ++p) {
        ElemAttr* prodElem = &poolData->prodElem[p];
        for (c=0; c<poolData->numConsElem; ++c) {
            ElemAttr* consElem = &poolData->consElem[c];

            /* If requested element types match, combine the entries */
            if (prodElem->userName == consElem->userName) {
                ElemAttr* poolElem = &elem[numElem++];
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

    /* Inform the stream of the chosen elements */
    for (e=0; e<numElem; ++e) {
        ElemAttr* poolElem = &elem[e];
        err = NvSciStreamBlockElementAttrSet(poolData->common.block,
                                             poolElem->userName,
                                             poolElem->attrList);
        if (NvSciError_Success != err) {
            printf("Pool failed (%x) to send element %d info\n", err, e);
            return 0;
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

    /*
     * Create and send all the packets and their buffers
     * Note: Packets and buffers are not guaranteed to be received by
     *       producer and consumer in the same order sent, nor are the
     *       status messages sent back guaranteed to preserve ordering.
     *       This is one reason why an event driven model is more robust.
     */
    for (i=0; i<poolData->numPacket; ++i) {

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
        for (e=0; e<numElem; ++e) {

            /* Allocate a buffer object */
            NvSciBufObj obj;
            err = NvSciBufObjAlloc(elem[e].attrList, &obj);
            if (NvSciError_Success != err) {
                printf("Failed (%x) to allocate buffer %d of packet %d\n",
                       err, e, i);
                return 0;
            }

            /* Insert the buffer in the packet */
            err = NvSciStreamPoolPacketInsertBuffer(poolData->common.block,
                                                    poolData->packet[i],
                                                    e, obj);
            if (NvSciError_Success != err) {
                printf("Failed (%x) to insert buffer %d of packet %d\n",
                       err, e, i);
                return 0;
            }

            /* The pool doesn't need to keep a copy of the object handle */
            NvSciBufObjFree(obj);
        }

        /* Indicate packet setup is complete */
        err = NvSciStreamPoolPacketComplete(poolData->common.block,
                                            poolData->packet[i]);
        if (NvSciError_Success != err) {
            printf("Failed (%x) to complete packet %d setup\n",
                   err, i);
            return 0;
        }
    }

    /*
     * Indicate that all packets have been sent.
     * Note: An application could choose to wait to send this until
     *  the status has been received, in order to try to make any
     *  corrections for rejected packets.
     */
    err = NvSciStreamBlockSetupStatusSet(poolData->common.block,
                                         NvSciStreamSetup_PacketExport,
                                         true);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to complete packet export\n",
               err);
        return 0;
    }

    /* Once all packets are set up, no longer need to keep the attributes */
    for (e=0; e<numElem; ++e) {
        ElemAttr* poolElem = &elem[e];
        if (NULL != poolElem->attrList) {
            NvSciBufAttrListFree(poolElem->attrList);
            poolElem->attrList = NULL;
        }
    }

    return 1;
}

/*
* If ready, handle setup of the C2C pool buffers.
*
* Most of the work the pool application has to do resides in this function.
*/
static int32_t handleC2cPoolBufferSetup(
    PoolData* poolData)
{
    NvSciError err;

    /* Query allocated element count from the primary pool */
    uint32_t          numElem;
    err = NvSciStreamBlockElementCountGet(poolData->common.block,
                                          NvSciStreamBlockType_Producer,
                                          &numElem);
    if (NvSciError_Success != err) {
        printf("C2C pool failed (%x) to query allocated element count\n",
               err);
        return 0;
    }

    /* Query all allocated elements from the primary pool */
    ElemAttr elem[MAX_ELEMS];
    for (uint32_t i = 0U; i<numElem; ++i) {
        err = NvSciStreamBlockElementAttrGet(poolData->common.block,
                                             NvSciStreamBlockType_Producer, i,
                                             &elem[i].userName,
                                             &elem[i].attrList);
        if (NvSciError_Success != err) {
            printf("C2C pool failed (%x) to query allocated element %d\n",
                   err, i);
            return 0;
        }
    }

    /* If necessary, query the consumer elements for validation */

    /* Indicate that all element information has been imported */
    poolData->elementsDone = true;
    err = NvSciStreamBlockSetupStatusSet(poolData->common.block,
                                         NvSciStreamSetup_ElementImport,
                                         true);
    if (NvSciError_Success != err) {
        printf("C2C pool failed (%x) to complete element import\n", err);
        return 0;
    }

    /*
    * Create and send all the packets and their buffers
    */
    for (uint32_t i = 0; i<poolData->numPacket; ++i) {

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
        for (uint32_t e = 0; e<numElem; ++e) {

            /* Allocate a buffer object */
            NvSciBufObj obj;
            err = NvSciBufObjAlloc(elem[e].attrList, &obj);
            if (NvSciError_Success != err) {
                printf("Failed (%x) to allocate buffer %d of packet %d\n",
                       err, e, i);
                return 0;
            }

            /* Insert the buffer in the packet */
            err = NvSciStreamPoolPacketInsertBuffer(poolData->common.block,
                                                    poolData->packet[i],
                                                    e, obj);
            if (NvSciError_Success != err) {
                printf("Failed (%x) to insert buffer %d of packet %d\n",
                       err, e, i);
                return 0;
            }

            /* The pool doesn't need to keep a copy of the object handle */
            NvSciBufObjFree(obj);
        }

        /* Indicate packet setup is complete */
        err = NvSciStreamPoolPacketComplete(poolData->common.block,
                                            poolData->packet[i]);
        if (NvSciError_Success != err) {
            printf("Failed (%x) to complete packet %d setup\n",
                   err, i);
            return 0;
        }
    }

    /*
     * Indicate that all packets have been sent.
     * Note: An application could choose to wait to send this until
     *  the status has been received, in order to try to make any
     *  corrections for rejected packets.
     */
    err = NvSciStreamBlockSetupStatusSet(poolData->common.block,
                                         NvSciStreamSetup_PacketExport,
                                         true);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to complete packet export\n",
               err);
        return 0;
    }

    /* Once all packets are set up, no longer need to keep the attributes */
    for (uint32_t e = 0; e<numElem; ++e) {
        ElemAttr* poolElem = &elem[e];
        if (NULL != poolElem->attrList) {
            NvSciBufAttrListFree(poolElem->attrList);
            poolElem->attrList = NULL;
        }
    }

    return 1;
}

/* Check packet status */
static int32_t handlePacketsStatus(
    PoolData*     poolData)
{
    bool packetFailure = false;
    NvSciError err;

    /* Check each packet */
    for (uint32_t p = 0; p < poolData->numPacket; ++p) {
        /* Check packet acceptance */
        bool accept;
        err = NvSciStreamPoolPacketStatusAcceptGet(poolData->common.block,
                                                   poolData->packet[p],
                                                   &accept);
        if (NvSciError_Success != err) {
            printf("Failed (%x) to retrieve packet %d's acceptance-statue\n",
                   err, p);
            return 0;
        }
        if (accept) {
            continue;
        }

        /* On rejection, query and report details */
        packetFailure = true;
        NvSciError status;

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
    }

    /* Indicate that status for all packets has been received. */
    poolData->packetsDone = true;
    err = NvSciStreamBlockSetupStatusSet(poolData->common.block,
                                         NvSciStreamSetup_PacketImport,
                                         true);
    if (NvSciError_Success != err) {
        printf("Pool failed (%x) to complete packets export\n", err);
        return 0;
    }

    return packetFailure ? 0 : 1;
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
        if (poolData->isC2cPool) {
            if (!handleC2cPoolBufferSetup(poolData)) {
                rv = -1;
            }
        } else {
            if (!handlePoolBufferSetup(poolData)) {
                rv = -1;
            }
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
        /* There are multiple ways the status handling could be organized.
        *   In particular, waiting for status could be interleaved with
        *   sending the packets. This example waits for status from all
        *   packets before checking each packet's status.
        */
        if (++poolData->numPacketReady < poolData->numPacket) {
            break;
        }

        if (!handlePacketsStatus(poolData)) {
            rv = -1;
        }

        poolData->common.waitTime = -1;
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
int32_t createPool(
    NvSciStreamBlock* pool,
    uint32_t          numPacket,
    bool              isC2cPool)
{
    /* Create a data structure to track the block's status */
    PoolData* poolData = (PoolData*)calloc(1, sizeof(PoolData));
    if (NULL == poolData) {
        printf("Failed to allocate data structure for pool\n");
        return 0;
    }

    /* Save the name for debugging purposes */
    strcpy(poolData->common.name, "Pool");

    /* Save the c2c pool flag */
    poolData->isC2cPool = isC2cPool;

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
