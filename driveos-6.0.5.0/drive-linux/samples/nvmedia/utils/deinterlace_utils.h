/*
 * Copyright (c) 2013-2019, NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software and related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef _NVMEDIA_TEST_DEINTERLACE_UTILS_H_
#define _NVMEDIA_TEST_DEINTERLACE_UTILS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "video_utils.h"
#include "nvmedia_core.h"
#include "nvmedia_vmp.h"

typedef enum {
    DEINTERLACE_MODE_WEAVE              = 0,
    DEINTERLACE_MODE_BOB                = 1,
    DEINTERLACE_MODE_ADVANCED_FRAMERATE = 2,
    DEINTERLACE_MODE_ADVANCED_FIELDRATE = 3
} EDeinterlaceMode;

enum {
    DEINT_LOOP_1ST = 0,
    DEINT_LOOP_2ND,
    DEINT_LOOPS
};

enum {
    DEINT_OUTPUT_PREV2 = 0,
    DEINT_OUTPUT_PREV,
    DEINT_OUTPUT_CURR,
    DEINT_OUTPUT_NEXT,
    DEINT_OUTPUT_SURFACES
};

enum {
    DEINT_FRAME_QUEUE_PAST = 0,
    DEINT_FRAME_QUEUE_NOW,
    DEINT_FRAME_QUEUE_FUTURE,
    DEINT_FRAME_QUEUE_ELEMENTS
};

typedef struct {
    FrameBuffer *pDeinterlaceFrameQueue[DEINT_FRAME_QUEUE_ELEMENTS];
    EDeinterlaceMode mode;
    uint32_t framesQueueDepth;
} DeinterlaceContext;

NvMediaStatus
DeinterlaceInit(
    DeinterlaceContext **ctx,
    EDeinterlaceMode mode);

NvMediaStatus
DeinterlaceFini(
    DeinterlaceContext *ctx);

NvMediaStatus Deinterlace(
    DeinterlaceContext *ctx,
    FrameBuffer* pInputSurface,
    uint32_t *puOutputLoopCount,
    NvMediaPictureStructure* pPictureStucture,
    NvMediaVideoSurface* pOutputSurface[DEINT_LOOPS][DEINT_OUTPUT_SURFACES],
    FrameBuffer** ppReleaseSurface);

NvMediaStatus
DeinterlaceFlush(
    DeinterlaceContext *ctx,
    FrameBuffer** ppReleaseSurface);

#ifdef __cplusplus
}
#endif

#endif /* _NVMEDIA_TEST_DEINTERLACE_UTILS_H_ */
