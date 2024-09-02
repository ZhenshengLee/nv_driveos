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
 * NvSciStream Event Loop Driven Sample App - multicast block
 */

#include <stdlib.h>
#include <stdio.h>
#include "nvscistream.h"
#include "block_info.h"
#include "event_loop.h"

/*
 * Multicast blocks do not require any block-specific interactions so we
 *   use the set of common functions to handle its events.
 */

/* Create and register a new multicast block */
int32_t createMulticast(
    NvSciStreamBlock* multicast,
    uint32_t          numConsumer)
{
    /* Create a data structure to track the block's status */
    BlockData* blockData = createCommon("Multicast", 0);
    if (NULL == blockData) {
        return 0;
    }

    /* Create a multicast block */
    NvSciError err =
        NvSciStreamMulticastCreate(numConsumer, &blockData->block);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to create limiter block\n", err);
        deleteCommon(blockData);
        return 0;
    }

    /* Register block with event handling mechanism */
    if (!eventFuncs->reg(blockData->block, blockData, handleCommon)) {
        deleteCommon(blockData);
        return 0;
    }

    *multicast = blockData->block;
    return 1;
}

