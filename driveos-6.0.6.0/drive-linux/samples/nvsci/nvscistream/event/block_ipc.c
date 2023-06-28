/* NvSciStream Event Loop Driven Sample App - ipc blocks
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

/*
 * Ipc blocks do not require any block-specific interactions so we
 *   use the set of common functions to handle its events. However
 *   they do have an additional data field which needs to be cleaned
 *   up when the block is destroyed, so we use more than the common
 *   data structure and delete function.
 */

/* Create and register a new ipcsrc block */
int32_t createIpcSrc(
    NvSciStreamBlock* ipcsrc,
    const char*       channel)
{
    NvSciError err;

    /* Create a data structure to track the block's status */
    BlockData* blockData = createCommon("IpcSrc", 0);
    if (NULL == blockData) {
        return 0;
    }

    /* Open the named channel */
    err = NvSciIpcOpenEndpoint(channel, &ipcEndpoint);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to open channel (%s) for IpcSrc\n",
               err, channel);
        deleteCommon(blockData);
        return 0;
    }
    err = NvSciIpcResetEndpointSafe(ipcEndpoint);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to reset IPC endpoint", err);
    }

    /* Create a ipcsrc block */
    err = NvSciStreamIpcSrcCreate(ipcEndpoint,
                                  sciSyncModule,
                                  sciBufModule,
                                  &blockData->block);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to create IpcSrc block\n", err);
        deleteCommon(blockData);
        return 0;
    }

    /* Register block with event handling mechanism */
    if (!eventFuncs->reg(blockData->block, blockData, handleCommon)) {
        deleteCommon(blockData);
        return 0;
    }

    *ipcsrc = blockData->block;
    return 1;
}

/* Create and register a new ipcdst block */
int32_t createIpcDst(
    NvSciStreamBlock* ipcdst,
    const char*       channel)
{
    NvSciError err;

    /* Create a data structure to track the block's status */
    BlockData* blockData = createCommon("IpcDst", 0);
    if (NULL == blockData) {
        return 0;
    }

    /* Open the named channel */
    err = NvSciIpcOpenEndpoint(channel, &ipcEndpoint);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to open channel (%s) for IpcDst\n",
               err, channel);
        deleteCommon(blockData);
        return 0;
    }
    err = NvSciIpcResetEndpointSafe(ipcEndpoint);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to reset IPC endpoint", err);
    }

    /* Create a ipcdst block */
    err = NvSciStreamIpcDstCreate(ipcEndpoint,
                                  sciSyncModule,
                                  sciBufModule,
                                  &blockData->block);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to create IpcDst block\n", err);
        deleteCommon(blockData);
        return 0;
    }

    /* Register block with event handling mechanism */
    if (!eventFuncs->reg(blockData->block, blockData, handleCommon)) {
        deleteCommon(blockData);
        return 0;
    }

    *ipcdst = blockData->block;
    return 1;
}
