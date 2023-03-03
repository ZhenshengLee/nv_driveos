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
#include "nvmedia_surface.h"
#include "nvmedia_video.h"

#define PACK_RGBA(R, G, B, A)  (((uint32_t)(A) << 24) | ((uint32_t)(B) << 16) | \
                                ((uint32_t)(G) << 8) | (uint32_t)(R))
#define DEFAULT_ALPHA   0x80

typedef struct {
    int                     refCount;
    int                     width;
    int                     height;
    int                     frameNum;
    int                     index;
    NvMediaVideoSurface    *videoSurface;
    bool                    progressiveFrameFlag;
    bool                    topFieldFirstFlag;
    // used for specifying crop rectange information
    int                     lDARWidth;
    int                     lDARHeight;
    int                     displayLeftOffset;
    int                     displayTopOffset;
    int                     displayWidth;
    int                     displayHeight;
} FrameBuffer;

//  WriteFrame
//
//    WriteFrame()  Save RGB or YUV video surface to a file
//
//  Arguments:
//
//   filename
//      (in) Output file name
//
//   videoSurface
//      (out) Pointer to a surface
//
//   bOrderUV
//      (in) Flag for UV order. If true - UV; If false - VU;
//           Used only in YUV type surface case
//
//   bAppend
//      (in) Apped to exisitng file if true otherwise create new file
//
//   srcRect
//      (in) structure containing co-ordinates of the rectangle in the source surface
//           from which the client surface is to be copied. Setting srcRect to NULL
//           implies rectangle of full surface size.

NvMediaStatus
WriteFrame(
    char *filename,
    NvMediaVideoSurface *videoSurface,
    NvMediaBool bOrderUV,
    NvMediaBool bAppend,
    NvMediaRect *srcRect);


//  ReadFrame
//
//    ReadFrame()  Read specific frame from YUV or RGBA file
//
//  Arguments:
//
//   filename
//      (in) Input file name
//
//   uFrameNum
//      (in) Frame number to read
//
//   width
//      (in) Surface width
//
//   height
//      (in) Surface height
//
//   pFrame
//      (out) Pointer to pre-allocated output surface
//
//   bOrderUV
//      (in) Flag for UV order. If true - UV; If false - VU;
//
//   pixelAlignment
//      (in) Alignment of bits in pixel.
//         0 - LSB Aligned
//         1 - MSB Aligned

NvMediaStatus
ReadFrame(
    char *fileName,
    uint32_t uFrameNum,
    uint32_t uWidth,
    uint32_t uHeight,
    NvMediaVideoSurface *pFrame,
    NvMediaBool bOrderUV,
    uint32_t pixelAlignment);

//  ReadPPMFrame
//
//    ReadPPMFrame()  Read surface from PPM file
//
//  Arguments:
//
//   filename
//      (in) Input file name
//
//   pFrame
//      (out) Pointer to pre-allocated surface

NvMediaStatus
ReadPPMFrame(
    char *fileName,
    NvMediaVideoSurface *pFrame);

NvMediaStatus
GetSurfaceCrc_New(
    NvMediaVideoSurface *surf,
    NvMediaRect *srcRect,
    NvMediaBool monochromeFlag,
    uint32_t *crcOut);

NvMediaStatus
GetSurfaceCrc(
    NvMediaVideoSurface *surf,
    uint32_t width,
    uint32_t height,
    NvMediaBool monochromeFlag,
    uint32_t *crcOut);

NvMediaStatus
CheckSurfaceCrc_New(
    NvMediaVideoSurface *surf,
    NvMediaRect *srcRect,
    NvMediaBool monochromeFlag,
    uint32_t ref,
    NvMediaBool *isMatching);

NvMediaStatus
CheckSurfaceCrc(
    NvMediaVideoSurface *surf,
    uint32_t width,
    uint32_t height,
    NvMediaBool monochromeFlag,
    uint32_t ref,
    NvMediaBool *isMatching);


#ifdef __cplusplus
}
#endif

#endif /* _NVMEDIA_TEST_VIDEO_UTILS_H_ */
