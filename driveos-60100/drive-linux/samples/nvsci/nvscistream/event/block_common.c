/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * NvSciStream Event Loop Driven Sample App - common block event handling
 *
 * Block types which do not require type-specific interactions make use of
 *   this common code.
 */

#include <stdlib.h>
#if (QNX == 1)
#include <sys/neutrino.h>
#endif
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include "nvscistream.h"
#include "block_info.h"
#include "event_loop.h"

/* Variable indicates whether the thread handling the
* late consumer connetions is started or not
*/
static bool threadStarted = false;

/* Delete common block */
void deleteCommon(
    void* data)
{
    BlockData* blockData = (BlockData*)data;

    /* Destroy block */
    if (blockData->block != 0) {
        (void)NvSciStreamBlockDelete(blockData->block);
    }

    /* Check if it is late/re-attach usecase */
    if (opts.numLateConsumer > 0U) {
        /* Close the endpoints used by the IpcSrc/C2CSrc
        * blocks for next late-/re-attach consumer connection
        */
        pthread_mutex_lock(&mutex);
        if ((!strcmp(blockData->name, "IpcSrc")) ||
            (!strcmp(blockData->name, "C2cSrc"))) {
            for (uint32_t i=0; i< MAX_CONSUMERS; i++) {
                if (ipcEP[i].ipcBlock == blockData->block) {
                    /* close the Ipc endpoint */
                    if (ipcEP[i].ipcEndpoint) {
#if (QNX == 1)
                        if (ipcEP[i].coid != 0) {
                            (void)ConnectDetach_r(ipcEP[i].coid);
                            ipcEP[i].coid = 0;
                        }
                        if (ipcEP[i].chid != 0) {
                            (void)ChannelDestroy_r(ipcEP[i].chid);
                            ipcEP[i].chid = 0;
                        }
#endif
                        if (NvSciError_Success !=
                            NvSciIpcCloseEndpointSafe(ipcEP[i].ipcEndpoint, false)) {
                            printf("Failed to close ipc endpoint\n");
                        }
                        sleep(2);
                        ipcEP[i].ipcEndpoint = 0U;
                    }
                    /* close the C2C endpoint */
                    if (ipcEP[i].c2cEndpoint) {
                        if (NvSciError_Success !=
                            NvSciIpcCloseEndpointSafe(ipcEP[i].c2cEndpoint, false)) {
                            printf("Failed to close ipc endpoint\n");
                        }
                        ipcEP[i].c2cEndpoint = 0U;
                    }

                    /* clear the informaton as this is needed
                    * for next late-/re-attach connection
                    */
                    ipcEP[i].ipcBlock = 0U;
                    ipcEP[i].ipcConnected = false;
                    ipcEP[i].c2cConnected = false;
                    ipcEP[i].ipcOpened = false;
                    ipcEP[i].c2cOpened = false;
                    break;
                }
            }
            /* Wakeup the thread to handle the next set of
            * late-/re-attach consumer connections
            */
            pthread_cond_signal(&cond);
        }
        pthread_mutex_unlock(&mutex);
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
            rv = -1;
        } else {
            if ((opts.numLateConsumer > 0U) &&
                (status == NvSciError_StreamNotConnected)) {
                printf("[WARN] %s received error event: %x\n",
                        blockData->name, status);
                rv = 2;
            } else {
                printf("%s received error event: %x\n",
                       blockData->name, status);
                rv = -1;
            }
        }
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
        if (opts.numLateConsumer > 0U) {
            /* Check if it is multicast block */
            if (!strcmp(blockData->name, "Multicast")) {
                /* Wakeup the thread to handle the next set
                * of late-/re-attach consumer connections
                */
                pthread_cond_signal(&cond);
                if (!threadStarted) {
                    threadStarted = true;
                    /* Spawn a thread to handle the late attach connections */
                    int32_t status = pthread_create(&dispatchThread,
                                                    NULL,
                                                    handleLateConsumerThreadFunc,
                                                    NULL);
                    if (status != 0) {
                        printf("Failed to spawn thread to monitor late consumer connections\n");
                        /* Abort the process as this thread is important
                        * to process the late-/re-attach consumer connections.
                        * Failed to create this thread makes the late/re-attach usecase
                        * unusable.
                        */
                        abort();
                    }
                }
            }
        }
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
