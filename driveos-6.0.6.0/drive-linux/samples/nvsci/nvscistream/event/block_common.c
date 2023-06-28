/* NvSciStream Event Loop Driven Sample App - common block event handling
 *
 * Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software, related documentation and any
 * modifications thereto. Any use, reproduction, disclosure or distribution
 * of this software and related documentation without an express license
 * agreement from NVIDIA Corporation is strictly prohibited.
 */

/*
 * Block types which do not require type-specific interactions make use of
 *   this common code.
 */

#include <stdlib.h>
#include <stdio.h>
#include "nvscistream.h"
#include "block_info.h"
#include "event_loop.h"

/* Delete common block */
void deleteCommon(
    void* data)
{
    BlockData* blockData = (BlockData*)data;

    /* Destroy block */
    if (blockData->block != 0) {
        (void)NvSciStreamBlockDelete(blockData->block);
    }

    /* Free data */
    free(blockData);
}

/* Handle events on a common block
 *
 * Blocks that don't require interaction after connection usually just
 *   receive notification of connection and disconnection.
 */
int32_t handleCommon(
    void*     data,
    uint32_t  wait)
{
    /* Cast to common data */
    BlockData* blockData = (BlockData*)data;

    /* Get time to wait */
    int64_t waitTime = wait ? blockData->waitTime : 0;

    /* Query/wait for an event on the block */

    NvSciStreamEventType event;
    NvSciError       err;
    err = NvSciStreamBlockEventQuery(blockData->block, waitTime, &event);

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
            printf("%s timed out waiting for setup instructions\n",
                   blockData->name);
        } else {
            printf("%s event query failed with error %x\n",
                   blockData->name, err);
        }
        blockData->deleteFunc(blockData);
        return -1;
    }

    /* If we received an event, handle it based on its type */
    int32_t rv = 1;
    NvSciError status;
    switch (event) {

    /*
     * Any event we don't explicitly handle is a fatal error
     */
    default:
        printf("%s received unknown event %x\n",
               blockData->name, event);

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
        err = NvSciStreamBlockErrorGet(blockData->block, &status);
        if (NvSciError_Success != err) {
            printf("%s Failed to query the error event code %x\n",
                   blockData->name, err);
        } else {
            printf("%s received error event: %x\n",
                   blockData->name, status);
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
     * The block doesn't have to do anything on connection, but now we may
     *   wait forever for any further events, so the timeout becomes infinite.
     */
    case NvSciStreamEventType_Connected:
        /* Query producer and consumer(s) endpoint info if needed */
        blockData->waitTime = -1;
        break;

    /* All setup complete. Transition to runtime phase */
    case NvSciStreamEventType_SetupComplete:
        break;
    }

    /* On failure or final event, clean up the block */
    if ((rv < 0) || (1 < rv)) {
        blockData->deleteFunc(blockData);
    }

    return rv;
}

/* Create and register a new common block */
BlockData* createCommon(
    char const* name,
    size_t      size)
{
    /* If no size specified, just use BlockData */
    if (0 == size) {
        size = sizeof(BlockData);
    }

    /* Create a data structure to track the block's status */
    BlockData* commonData = (BlockData*)calloc(1, size);
    if (NULL == commonData) {
        printf("Failed to allocate data structure for %s\n", name);
        return NULL;
    }

    /* Save the name for debugging purposes */
    strcpy(commonData->name, name);

    /* Wait time for initial connection event will be 60 seconds */
    commonData->waitTime = 60 * 1000000;

    /* Use the common delete function */
    commonData->deleteFunc = deleteCommon;

    return commonData;
}
