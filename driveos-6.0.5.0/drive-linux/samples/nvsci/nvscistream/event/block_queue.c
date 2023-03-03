/* NvSciStream Event Loop Driven Sample App - queue block
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

/*
 * Queue blocks do not require any block-specific interactions so we
 *   use the set of common functions to handle its events.
 */

/* Create and register a new limiter block */
int32_t createQueue(
    NvSciStreamBlock* queue,
    uint32_t          useMailbox)
{
    /* Create a data structure to track the block's status */
    BlockData* blockData = createCommon(useMailbox ? "Mailbox" : "FIFO", 0);
    if (NULL == blockData) {
        return 0;
    }

    /* Create a queue block */
    NvSciError err = useMailbox
                   ? NvSciStreamMailboxQueueCreate(&blockData->block)
                   : NvSciStreamFifoQueueCreate(&blockData->block);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to create queue block\n", err);
        deleteCommon(blockData);
        return 0;
    }

    /* Register block with event handling mechanism */
    if (!eventFuncs->reg(blockData->block, blockData, handleCommon)) {
        deleteCommon(blockData);
        return 0;
    }

    *queue = blockData->block;
    return 1;
}
