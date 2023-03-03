/*
 * Copyright (c) 2013-2018 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software and related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "deinterlace_utils.h"
#include "log_utils.h"

NvMediaStatus DeinterlaceInit(
    DeinterlaceContext **ctx,
    EDeinterlaceMode mode)
{
    DeinterlaceContext *deinterlaceCtx;
    uint32_t i;

    deinterlaceCtx = malloc (sizeof(DeinterlaceContext));
    if (!deinterlaceCtx) {
        LOG_ERR("Deinterlace: Out of memory\n");
        return NVMEDIA_STATUS_OUT_OF_MEMORY;
    }

    for(i = 0; i < DEINT_FRAME_QUEUE_ELEMENTS; i++) {
        deinterlaceCtx->pDeinterlaceFrameQueue[i] = 0;
    }

    deinterlaceCtx->mode = mode;
    switch(mode) {
        case DEINTERLACE_MODE_BOB:
            deinterlaceCtx->framesQueueDepth = 1;
            break;
        case DEINTERLACE_MODE_WEAVE:
            deinterlaceCtx->framesQueueDepth = 1;
            break;
        case DEINTERLACE_MODE_ADVANCED_FRAMERATE:
            deinterlaceCtx->framesQueueDepth = 2;
            break;
        case DEINTERLACE_MODE_ADVANCED_FIELDRATE:
            deinterlaceCtx->framesQueueDepth = 3;
            break;
        default:
            LOG_ERR("DeinterlaceInit: Encountered unsupported deinterlacing mode (%d)\n", mode);
            return NVMEDIA_STATUS_ERROR;
    }
    *ctx = deinterlaceCtx;

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus DeinterlaceFini(
    DeinterlaceContext *ctx)
{
    free(ctx);

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus Deinterlace(
    DeinterlaceContext *ctx,
    FrameBuffer* pInputSurface,
    uint32_t *puOutputLoopCount,
    NvMediaPictureStructure* pPictureStucture,
    NvMediaVideoSurface* pOutputSurface[DEINT_LOOPS][DEINT_OUTPUT_SURFACES],
    FrameBuffer** ppReleaseSurface)
{
    uint32_t i;
    FrameBuffer **pDeinterlaceFrameQueue = ctx->pDeinterlaceFrameQueue;
    FrameBuffer* pReleaseSurface;

    if(!puOutputLoopCount || !pPictureStucture || !pOutputSurface) {
        LOG_ERR("Deinterlace: Bad parameter\n");
        return NVMEDIA_STATUS_ERROR;
    }

    // Reset output
    *puOutputLoopCount = 0;
    memset(pOutputSurface, 0, sizeof(FrameBuffer*) * DEINT_OUTPUT_SURFACES * DEINT_LOOPS);
    pReleaseSurface = 0;

    // Shortcut for Progressive material
    if(pInputSurface->progressiveFrameFlag) {
        ctx->mode = DEINTERLACE_MODE_WEAVE;
        ctx->framesQueueDepth = 1;
    }

    // Shift queue and add new frame at the end
    for(i = 0; i < ctx->framesQueueDepth - 1; i++) {
        pDeinterlaceFrameQueue[i] = pDeinterlaceFrameQueue[i + 1];
    }
    pDeinterlaceFrameQueue[ctx->framesQueueDepth - 1] = pInputSurface;

    switch(ctx->mode) {
        case DEINTERLACE_MODE_WEAVE:
            *puOutputLoopCount = 1;
            pPictureStucture[DEINT_LOOP_1ST] = NVMEDIA_PICTURE_STRUCTURE_FRAME;
            pOutputSurface[DEINT_LOOP_1ST][DEINT_OUTPUT_CURR] = pInputSurface->videoSurface;
            pReleaseSurface = pInputSurface;
            break;
        case DEINTERLACE_MODE_BOB:
            *puOutputLoopCount = 2;
            // Set field order
            if(pInputSurface->topFieldFirstFlag) {
                pPictureStucture[DEINT_LOOP_1ST] = NVMEDIA_PICTURE_STRUCTURE_TOP_FIELD;
                pPictureStucture[DEINT_LOOP_2ND] = NVMEDIA_PICTURE_STRUCTURE_BOTTOM_FIELD;
            } else {
                pPictureStucture[DEINT_LOOP_1ST] = NVMEDIA_PICTURE_STRUCTURE_BOTTOM_FIELD;
                pPictureStucture[DEINT_LOOP_2ND] = NVMEDIA_PICTURE_STRUCTURE_TOP_FIELD;
            }
            pOutputSurface[DEINT_LOOP_1ST][DEINT_OUTPUT_CURR] = pInputSurface->videoSurface;
            pOutputSurface[DEINT_LOOP_2ND][DEINT_OUTPUT_CURR] = pInputSurface->videoSurface;
            pReleaseSurface = pInputSurface;
            break;
        case DEINTERLACE_MODE_ADVANCED_FRAMERATE:
            // Set field order
            if(pInputSurface->topFieldFirstFlag) {
                pPictureStucture[DEINT_LOOP_1ST] = NVMEDIA_PICTURE_STRUCTURE_TOP_FIELD;
            } else {
                pPictureStucture[DEINT_LOOP_1ST] = NVMEDIA_PICTURE_STRUCTURE_BOTTOM_FIELD;
            }
            // Set frame history (1st loop)
            if(pDeinterlaceFrameQueue[DEINT_FRAME_QUEUE_PAST]) {
                pOutputSurface[DEINT_LOOP_1ST][DEINT_OUTPUT_PREV2] = pDeinterlaceFrameQueue[DEINT_FRAME_QUEUE_PAST]->videoSurface;
                pOutputSurface[DEINT_LOOP_1ST][DEINT_OUTPUT_PREV]  = pDeinterlaceFrameQueue[DEINT_FRAME_QUEUE_PAST]->videoSurface;
            }
            pOutputSurface[DEINT_LOOP_1ST][DEINT_OUTPUT_CURR]  = pDeinterlaceFrameQueue[DEINT_FRAME_QUEUE_NOW]->videoSurface;
            pOutputSurface[DEINT_LOOP_1ST][DEINT_OUTPUT_NEXT]  = pDeinterlaceFrameQueue[DEINT_FRAME_QUEUE_NOW]->videoSurface;
            // Release past frame if not the same (note this will be save for reuse if needed)
            pReleaseSurface = pDeinterlaceFrameQueue[DEINT_FRAME_QUEUE_PAST];
            // Set output loop count
            *puOutputLoopCount = 1;
            break;
        case DEINTERLACE_MODE_ADVANCED_FIELDRATE:
            // Check history is available
            if(pDeinterlaceFrameQueue[DEINT_FRAME_QUEUE_NOW]) {
                // Set field order
                if(pInputSurface->topFieldFirstFlag) {
                    pPictureStucture[DEINT_LOOP_1ST] = NVMEDIA_PICTURE_STRUCTURE_TOP_FIELD;
                    pPictureStucture[DEINT_LOOP_2ND] = NVMEDIA_PICTURE_STRUCTURE_BOTTOM_FIELD;
                } else {
                    pPictureStucture[DEINT_LOOP_1ST] = NVMEDIA_PICTURE_STRUCTURE_BOTTOM_FIELD;
                    pPictureStucture[DEINT_LOOP_2ND] = NVMEDIA_PICTURE_STRUCTURE_TOP_FIELD;
                }
                // Set frame history (1st loop)
                if(pDeinterlaceFrameQueue[DEINT_FRAME_QUEUE_PAST]) {
                    pOutputSurface[DEINT_LOOP_1ST][DEINT_OUTPUT_PREV2] = pDeinterlaceFrameQueue[DEINT_FRAME_QUEUE_PAST]->videoSurface;
                    pOutputSurface[DEINT_LOOP_1ST][DEINT_OUTPUT_PREV]  = pDeinterlaceFrameQueue[DEINT_FRAME_QUEUE_PAST]->videoSurface;
                }
                pOutputSurface[DEINT_LOOP_1ST][DEINT_OUTPUT_CURR]  = pDeinterlaceFrameQueue[DEINT_FRAME_QUEUE_NOW]->videoSurface;
                pOutputSurface[DEINT_LOOP_1ST][DEINT_OUTPUT_NEXT]  = pDeinterlaceFrameQueue[DEINT_FRAME_QUEUE_NOW]->videoSurface;
                // Set frame history (2nd loop)
                if(pDeinterlaceFrameQueue[DEINT_FRAME_QUEUE_PAST]) {
                    pOutputSurface[DEINT_LOOP_2ND][DEINT_OUTPUT_PREV2] = pDeinterlaceFrameQueue[DEINT_FRAME_QUEUE_PAST]->videoSurface;
                }
                pOutputSurface[DEINT_LOOP_2ND][DEINT_OUTPUT_PREV]  = pDeinterlaceFrameQueue[DEINT_FRAME_QUEUE_NOW]->videoSurface;
                pOutputSurface[DEINT_LOOP_2ND][DEINT_OUTPUT_CURR]  = pDeinterlaceFrameQueue[DEINT_FRAME_QUEUE_NOW]->videoSurface;
                pOutputSurface[DEINT_LOOP_2ND][DEINT_OUTPUT_NEXT]  = pDeinterlaceFrameQueue[DEINT_FRAME_QUEUE_FUTURE]->videoSurface;
                // Release past frame if not the same (note this will be save for reuse if needed)
                pReleaseSurface = pDeinterlaceFrameQueue[DEINT_FRAME_QUEUE_PAST];
                // Set output loop count
                *puOutputLoopCount = 2;
            }
            break;
        default:
            return NVMEDIA_STATUS_ERROR;
    }

    if (ppReleaseSurface) {
        *ppReleaseSurface = pReleaseSurface;
    }

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
DeinterlaceFlush(
    DeinterlaceContext *ctx,
    FrameBuffer** ppReleaseSurface)
{
    int i;

    for(i = 0; i < 2; i++) {
        ctx->pDeinterlaceFrameQueue[i] = ctx->pDeinterlaceFrameQueue[i + 1];
    }
    ctx->pDeinterlaceFrameQueue[2] = NULL;

    *ppReleaseSurface = ctx->pDeinterlaceFrameQueue[DEINT_FRAME_QUEUE_PAST];

    return NVMEDIA_STATUS_OK;
}
