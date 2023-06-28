/* NvSciStream Event Loop Driven Sample App - service-based event handling
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
 * This file implements the option to handle events for all blocks
 *   through an event service. Each block adds an event notifier to
 *   a list. That notifier will be signaled when an event is ready
 *   on the block. A single main loop waits for one or more of the
 *   notifiers to trigger, processes events on the corresponding
 *   blocks, and goes back to waiting. When all blocks have been
 *   destroyed either due to failure or all payloads being processed,
 *   the loop exits and the function returns.
 */

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#if (QNX == 1)
#include <sys/neutrino.h>
#endif
#include "nvscievent.h"
#include "event_loop.h"

/* Event service */
static NvSciEventLoopService* service = NULL;

/* Structure to track block info */
typedef struct {
    NvSciStreamBlock    handle;
    void*               data;
    BlockFunc           func;
    NvSciEventNotifier* notifier;
} BlockEventData;

/* List of blocks */
#define MAX_BLOCKS 100
static int32_t numBlocks = 0;
static BlockEventData blocks[MAX_BLOCKS];
static uint32_t success = 1U;

/* Initialize service-based event handling */
static int32_t eventServiceInit(void)
{
    /*
     * The OS configuration should be NULL for Linux and should
     * have a valid configuration for QNX.
     * See NvSciEventLoopServiceCreateSafe API Specification for more
     * information.
     */
    void *osConfig = NULL;

#if (QNX == 1)
    struct nto_channel_config config = {0};

    /*
     * The number of pulses could be calculated based on the
     * number of notifiers bind to the event service, number of packets and
     * number of events handled by each block.
     * (num_of_pulses = num_of_notifiers * 4 + \
     *                  (num_packets + 5) * num_of_endpoints)
     * If experienced pulse pool shortage issue in normal operation, increase
     * the number of pulses.
     * If there are no available pulses in the pool, SIGKILL is delivered
     * by default. You may configure the sigevent that you want to be
     * delivered when a pulse can't be obtained from the pool.
     *
     * See NvSciEventLoopServiceCreateSafe API Specification for more
     * information.
     */

    /* The num_pulses set below is just an example number and should be
     * adjusted depending on the use case.
     */
    config.num_pulses = 100U;
    config.rearm_threshold = 0;
    osConfig = &config;
#endif

    /* Create event loop service */
    NvSciError err = NvSciEventLoopServiceCreateSafe(1U, osConfig, &service);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to create event service\n", err);
        return 0;
    }

    return 1;
}

/* Register a new block with the event management */
static int32_t eventServiceRegister(
    NvSciStreamBlock blockHandle,
    void*            blockData,
    BlockFunc        blockFunc)
{
    /* Sanity check to make sure we left room for enough blocks */
    if (numBlocks >= MAX_BLOCKS) {
        printf("Exceeded maximum number of blocks\n");
        return 0;
    }

    /* Grab the next entry in the list for the new block and fill it in */
    BlockEventData* entry = &blocks[numBlocks++];
    entry->handle = blockHandle;
    entry->data   = blockData;
    entry->func   = blockFunc;

    /* Create a notifier for events on this block */
    NvSciError err =
        NvSciStreamBlockEventServiceSetup(entry->handle,
                                          &service->EventService,
                                          &entry->notifier);
    if (NvSciError_Success != err ) {
        printf("Failed (%x) to create event notifier for block\n", err);
        return 0;
    }

    return 1;
}

/* Main service-based event loop */
static int32_t eventServiceLoop(void)
{
    int32_t i;

    /*
     * Notes on handling notificiations:
     *   If more than one signal occurs on a notifier in between calls
     *     to check for events, then NvSciEvent will squash the notifications,
     *     so only one is received. This means the application must drain
     *     all pending events on a block after its notifier signals. It won't
     *     receive new notifications for those pending events.
     *   A simple implementation might process each block's events in a loop
     *     until there are no more, and then move on to the next block. But
     *     this poses a risk of starvation. Consider the case of a stream in
     *     mailbox mode, where the mailbox already has a waiting payload.
     *     If the producer receives a PacketReady event, it will obtain
     *     the packet, fill it with data, and present it to the stream.
     *     Because the mailbox is full, the packet will immediately be
     *     returned, resulting in a new PacketReady event. The application
     *     can go into an infinite loop, generating new payloads on the
     *     producer without giving the consumer a chance to process them.
     *   We therefore use an event loop that only processes one event
     *     per block for each iteration, but keeps track of whether there
     *     was an event on a block for the previous pass, and if so
     *     retries it even if no new signal occurred. The event loop
     *     waits for events only when there was no prior event. Otherwise
     *     it only polls for new ones.
     */

    /* Pack all notifiers into an array */
    NvSciEventNotifier* notifiers[MAX_BLOCKS];
    for (i=0; i<numBlocks; ++i) {
        notifiers[i] = blocks[i].notifier;
    }

    /* Initialize loop control parameters */
    uint32_t numAlive = numBlocks;
    int64_t timeout = -1;
    bool retry[MAX_BLOCKS];
    bool event[MAX_BLOCKS];
    memset(retry, 0, sizeof(retry));

    /* Main loop - Handle events until all blocks report completion or fail */
    while (numAlive) {

        /* Wait/poll for events, depending on current timeout */
        memset(event, 0, sizeof(event));
        NvSciError err = service->WaitForMultipleEventsExt(
                                                        &service->EventService,
                                                        notifiers,
                                                        numBlocks,
                                                        timeout,
                                                        event);
        if ((NvSciError_Success != err) && (NvSciError_Timeout != err)) {
            printf("Failure (%x) while waiting/polling event service\n", err);
            return 0;
        }

        /* Timeout for next pass will be infinite unless we need to retry */
        timeout = -1;

        /*
         * Check for events on new blocks that signaled or old blocks that
         *   had an event on the previous pass. This is done in reverse
         *   of the order in which blocks were registered. This is because
         *   producers are created before consumers, and for mailbox mode
         *   we want to give the consumer a chance to use payloads before
         *   the producer replaces them.
         */
        for (i=numBlocks-1; i>=0; --i) {
            if (event[i] || retry[i]) {

                /* Get block info */
                BlockEventData* entry = &blocks[i];

                /* Reset to no retry for next pass */
                retry[i] = false;

                /* Skip if this block is no longer in use */
                if (entry->data) {

                    /* Call the block's event handler function */
                    int32_t rv = entry->func(entry->data, 0);
                    if (rv < 0) {
                        /* On failure, no longer check block and app failed */
                        success = 0U;
                        entry->data = NULL;
                        numAlive--;
                    } else if (rv == 2) {
                        /* On completion, no longer check block */
                        entry->data = NULL;
                        numAlive--;
                    } else if (rv == 1) {
                        /* If event found, retry next loop */
                        timeout = 0;
                        retry[i] = true;
                    }
                }
            }
        }
    }

    /* Delete notifiers */
    for (i=0; i<numBlocks; ++i) {
        notifiers[i]->Delete(notifiers[i]);
    }

    /* Delete service */
    service->EventService.Delete(&service->EventService);

    return success;
}

/* Table of functions for service-based event handling */
EventFuncs const eventFuncs_Service = {
    .init = eventServiceInit,
    .reg  = eventServiceRegister,
    .loop = eventServiceLoop
};
