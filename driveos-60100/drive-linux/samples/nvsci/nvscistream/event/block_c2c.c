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
 * NvSciStream Event Loop Driven Sample App - C2C block
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
    uint32_t i;
    uint32_t slot = 0;

    /* Create a data structure to track the block's status */
    BlockData* blockData = createCommon("C2cSrc", 0);
    if (NULL == blockData) {
        return 0;
    }

    for (i=0; i< MAX_CONSUMERS; i++) {
        if (!strcmp(ipcEP[i].c2cChannel, channel)) {
            slot = i;
            break;
        }
    }

    /* Open the named channel */
    err = NvSciIpcOpenEndpoint(channel, &ipcEP[slot].c2cEndpoint);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to open channel (%s) for C2C src\n",
               err, channel);
        deleteCommon(blockData);
        return 0;
    }
    err = NvSciIpcResetEndpointSafe(ipcEP[slot].c2cEndpoint);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to reset IPC endpoint", err);
    }

    /* Create a C2C src block */
    err = NvSciStreamIpcSrcCreate2(ipcEP[slot].c2cEndpoint,
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
    ipcEP[slot].ipcBlock = blockData->block;
    ipcEP[slot].c2cOpened = true;
    ipcEP[slot].c2cConnected = true;

    return 1;
}


/* Create and register a new C2C src block */
int32_t createC2cSrc2(
    NvSciStreamBlock* c2cSrc,
    NvSciIpcEndpoint endpoint,
    NvSciStreamBlock  queue)
{
    NvSciError err;

    /* Create a data structure to track the block's status */
    BlockData* blockData = createCommon("C2cSrc", 0);
    if (NULL == blockData) {
        return 0;
    }

    /* Create a C2C src block */
    err = NvSciStreamIpcSrcCreate2(endpoint,
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
    err = NvSciIpcOpenEndpoint(channel, &ipcEP[0].ipcEndpoint);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to open channel (%s) for C2C dst\n",
               err, channel);
        deleteCommon(blockData);
        return 0;
    }
    err = NvSciIpcResetEndpointSafe(ipcEP[0].ipcEndpoint);
    if (NvSciError_Success != err) {
        printf("Failed (%x) to reset IPC endpoint", err);
    }

    /* Create a C2C dst block */
    err = NvSciStreamIpcDstCreate2(ipcEP[0].ipcEndpoint,
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
