/* NvSciStream Event Loop Driven Sample App - PresentSync block
 *
 * Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <stdlib.h>
#include <stdio.h>
#include "nvscistream.h"
#include "block_info.h"
#include "event_loop.h"

/*
 * PresentSync blocks do not require any block-specific interactions so we
 *   use the set of common functions to handle its events.
 */

/* Create and register a new presentSync block */
int32_t createPresentSync(
    NvSciStreamBlock* presentSync)
{
    /* Create a data structure to track the block's status */
    BlockData* blockData = createCommon("PresentSync", 0);
    if (NULL == blockData) {
        return 0;
    }

    /* Create a PresentSync block */
    NvSciError err =
        NvSciStreamPresentSyncCreate(sciSyncModule, &blockData->block);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to create PresentSync block\n", err);
        deleteCommon(blockData);
        return 0;
    }

    /* Register block with event handling mechanism */
    if (!eventFuncs->reg(blockData->block, blockData, handleCommon)) {
        deleteCommon(blockData);
        return 0;
    }

    *presentSync = blockData->block;
    return 1;
}
