/* NvSciStream Event Loop Driven Sample App - pool block
 *
 * Copyright (c) 2021-2022 NVIDIA Corporation. All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software, related documentation and any
 * modifications thereto. Any use, reproduction, disclosure or distribution
 * of this software and related documentation without an express license
 * agreement from NVIDIA Corporation is strictly prohibited.
 */

#include <stdlib.h>
#include <stdio.h>
#include "nvsciipc.h"
#include "nvscistream.h"
#include "block_info.h"
#include "event_loop.h"

/* Create and register a new C2C src block */
int32_t createC2cSrc(
    NvSciStreamBlock* c2cSrc,
    const char*       channel,
    NvSciStreamBlock  queue)
{
    NvSciError err;

    /* Create a data structure to track the block's status */
    BlockData* blockData = createCommon("C2cSrc", 0);
    if (NULL == blockData) {
        return 0;
    }

    /* Open the named channel */
    err = NvSciIpcOpenEndpoint(channel, &ipcEndpoint);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to open channel (%s) for C2C src\n",
               err, channel);
        deleteCommon(blockData);
        return 0;
    }
    err = NvSciIpcResetEndpointSafe(ipcEndpoint);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to reset IPC endpoint", err);
    }

    /* Create a C2C src block */
    err = NvSciStreamIpcSrcCreate2(ipcEndpoint,
                                  sciSyncModule,
                                  sciBufModule,
                                  queue,
                                  &blockData->block);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to create C2C src block\n", err);
        deleteCommon(blockData);
        return 0;
    }

    /* Register block with event handling mechanism */
    if (!eventFuncs->reg(blockData->block, blockData, handleCommon)) {
        deleteCommon(blockData);
        return 0;
    }

    *c2cSrc = blockData->block;
    return 1;
}

/* Create and register a new C2C dst block */
int32_t createC2cDst(
    NvSciStreamBlock* c2cDst,
    const char*       channel,
    NvSciStreamBlock  pool)
{
    NvSciError err;

    /* Create a data structure to track the block's status */
    BlockData* blockData = createCommon("C2cDst", 0);
    if (NULL == blockData) {
        return 0;
    }

    /* Open the named channel */
    err = NvSciIpcOpenEndpoint(channel, &ipcEndpoint);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to open channel (%s) for C2C dst\n",
               err, channel);
        deleteCommon(blockData);
        return 0;
    }
    err = NvSciIpcResetEndpointSafe(ipcEndpoint);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to reset IPC endpoint", err);
    }

    /* Create a C2C dst block */
    err = NvSciStreamIpcDstCreate2(ipcEndpoint,
                                  sciSyncModule,
                                  sciBufModule,
                                  pool,
                                  &blockData->block);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to create C2C dst block\n", err);
        deleteCommon(blockData);
        return 0;
    }

    /* Register block with event handling mechanism */
    if (!eventFuncs->reg(blockData->block, blockData, handleCommon)) {
        deleteCommon(blockData);
        return 0;
    }

    *c2cDst = blockData->block;
    return 1;
}
