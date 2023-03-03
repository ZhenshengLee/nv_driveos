/*
 * Copyright (c) 2019-2021 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software and related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef _NVMEDIA_TEST_VIDEO_UTILS_H_
#define _NVMEDIA_TEST_VIDEO_UTILS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include "misc_utils.h"
#include "nvmedia_core.h"
#include "nvscibuf.h"
#include "nvscisync.h"

#define PACK_RGBA(R, G, B, A)  (((uint32_t)(A) << 24) | ((uint32_t)(B) << 16) | \
                                ((uint32_t)(G) << 8) | (uint32_t)(R))
#define DEFAULT_ALPHA   0x80

typedef struct {
    int                     refCount;
    int                     width;
    int                     height;
    int                     frameNum;
    int                     index;
    NvSciBufObj             videoSurface;
    bool                    progressiveFrameFlag;
    bool                    topFieldFirstFlag;
    // used for specifying crop rectange information
    int                     lDARWidth;
    int                     lDARHeight;
    int                     displayLeftOffset;
    int                     displayTopOffset;
    int                     displayWidth;
    int                     displayHeight;
    NvSciSyncFence          preFence;
} FrameBuffer;

#ifdef __cplusplus
}
#endif

#endif /* _NVMEDIA_TEST_VIDEO_UTILS_H_ */
