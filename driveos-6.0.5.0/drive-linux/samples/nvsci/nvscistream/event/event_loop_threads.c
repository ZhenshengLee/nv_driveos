/* NvSciStream Event Loop Driven Sample App - thread-based event handling
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
 * This file implements the option to handle events for each block in
 *   a separate thread. Each thread just has a loop that waits for a
 *   block event to occur and handles it appropriately, until the block
 *   has performed all required actions or receives notification that
 *   the stream has disconnected.
 *
 * In practice, only a few block types (producer, consumer, and pool)
 *   receive any events that need to be handled. So a more streamlined
 *   application might choose to only monitor them, assuming that the
 *   other blocks can be left alone until the time comes to tear them
 *   down.
 *
 * Note: We use standard pthread functions here because it allows this
 *       sample to run on all operating systems. QNX has its own thread
 *       management functions which might be more efficient when using
 *       this approach.
 */

#include <stdint.h>
#include <stdio.h>
#include <pthread.h>
#include "event_loop.h"

/* Structure to track block info */
typedef struct {
    NvSciStreamBlock handle;
    void*            data;
    BlockFunc        func;
    pthread_t        thread;
} BlockEventData;

/* List of blocks */
#define MAX_BLOCKS 100U
static uint32_t numBlocks = 0U;
static BlockEventData blocks[MAX_BLOCKS];
static uint32_t success = 1U;

/* The per-thread loop function for each block */
static void* eventThreadFunc(void* arg)
{
    /* Simple loop, waiting for events on the block until the block is done */
    BlockEventData* entry = (BlockEventData*)arg;
    while (1) {
        int32_t rv = entry->func(entry->data, 1);
        if (rv < 0) {
            success = 0U;
            break;
        } else if (rv == 2) {
            break;
        }
    }
    return NULL;
}

/* Initialize per-thread event handling */
static int32_t eventThreadInit(void)
{
    /* No special initialization required for this method */
    return 1;
}

/* Register a new block with the event management */
static int32_t eventThreadRegister(
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

    /* Spawn a thread */
    int32_t rv = pthread_create(&entry->thread,
                                NULL,
                                eventThreadFunc,
                                (void*)entry);
    if (rv != 0) {
        printf("Failed to spawn thread to monitor block\n");
        return 0;
    }

    return 1;
}

/* Main per-thread event loop */
static int32_t eventThreadLoop(void)
{
    /*
     * Each block has its own thread loop. This main function just needs
     *   to wait for all of them to exit, and then return any error. This
     *   waiting can be done in any order.
     */
    for (uint32_t i=0; i<numBlocks; ++i) {
        (void)pthread_join(blocks[i].thread, NULL);
    }

    return success;
}

/* Table of functions for per-thread event handling */
EventFuncs const eventFuncs_Threads = {
    .init = eventThreadInit,
    .reg  = eventThreadRegister,
    .loop = eventThreadLoop
};
