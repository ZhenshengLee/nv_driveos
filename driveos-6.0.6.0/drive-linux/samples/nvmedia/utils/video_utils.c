/*
 * Copyright (c) 2019-2021 NVIDIA Corporation.  All rights reserved.
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

#include "log_utils.h"
#include "misc_utils.h"
#include "video_utils.h"
#include "nvmedia_surface.h"

#define MAXM_NUM_SURFACES 6

typedef struct {
    float heightFactor[6];
    float widthFactor[6];
    unsigned int numSurfaces;
} VidUtilSurfParams;

VidUtilSurfParams VidSurfParamsTable_RGBA  = {
    .heightFactor = {1, 0, 0, 0, 0, 0},
    .widthFactor = {1, 0, 0, 0, 0, 0},
    .numSurfaces = 1,
};

VidUtilSurfParams VidSurfParamsTable_RAW  = {
    .heightFactor = {1, 0, 0, 0, 0, 0},
    .widthFactor = {1, 0, 0, 0, 0, 0},
    .numSurfaces = 1,
};

VidUtilSurfParams VidSurfParamsTable_YUV[][4] = {
    { /* PLANAR */
        { /* 420 */
            .heightFactor = {1, 0.5, 0.5, 0, 0, 0},
            .widthFactor = {1, 0.5, 0.5, 0, 0, 0},
            .numSurfaces = 3,
        },
        { /* 422 */
            .heightFactor = {1, 1, 1, 0, 0, 0},
            .widthFactor = {1, 0.5, 0.5, 0, 0, 0},
            .numSurfaces = 3,
        },
        { /* 444 */
            .heightFactor = {1, 1, 1, 0, 0, 0},
            .widthFactor = {1, 1, 1, 0, 0, 0},
            .numSurfaces = 3,
        },
        { /* 422R */
            .heightFactor = {1, 0.5, 0.5, 0, 0, 0},
            .widthFactor = {1, 1, 1, 0, 0, 0},
            .numSurfaces = 3,
        },
    },
    { /* SEMI_PLANAR */
        { /* 420 */
            .heightFactor = {1, 0.5, 0, 0, 0, 0},
            .widthFactor = {1, 0.5, 0, 0, 0, 0},
            .numSurfaces = 2,
        },
        { /* 422 */
            .heightFactor = {1, 1, 0, 0, 0, 0},
            .widthFactor = {1, 0.5, 0, 0, 0, 0},
            .numSurfaces = 2,
        },
        { /* 444 */
            .heightFactor = {1, 1, 0.5, 0, 0, 0},
            .widthFactor = {1, 1, 0.5, 0, 0, 0},
            .numSurfaces = 2,
        },
        { /* 422R */
            .heightFactor = {1, 0.5, 0.5, 0, 0, 0},
            .widthFactor = {1, 1, 0.5, 0, 0, 0},
            .numSurfaces = 2,
        },
    },
    { /* PACKED */
        { /* 420 */
            .heightFactor = {1, 0.5, 0.5, 0, 0, 0},
            .widthFactor = {1, 0.5, 0.5, 0, 0, 0},
            .numSurfaces = 3,
        },
        { /* 422 */
            .heightFactor = {1, 1, 1, 0, 0, 0},
            .widthFactor = {1, 0.5, 0.5, 0, 0, 0},
            .numSurfaces = 3,
        },
        { /* 444 */
            .heightFactor = {1, 1, 1, 0, 0, 0},
            .widthFactor = {1, 1, 1, 0, 0, 0},
            .numSurfaces = 3,
        },
        { /* 422R */
            .heightFactor = {1, 0.5, 0.5, 0, 0, 0},
            .widthFactor = {1, 1, 1, 0, 0, 0},
            .numSurfaces = 3,
        },
    },
};

VidUtilSurfParams VidSurfParamsTable_Packed  = {
    .heightFactor = {1, 0, 0, 0, 0, 0},
    .widthFactor = {1, 0, 0, 0, 0, 0},
    .numSurfaces = 1,
};


unsigned int VidBytesPerPixelTable_RGBA[][6] = {
    {4, 0, 0, 0, 0, 0}, /* 8 */
};

unsigned int VidBytesPerPixelTable_RGBA16[][6] = {
    {8, 0, 0, 0, 0, 0}, /* 16 */
};

unsigned int VidBytesPerPixelTable_RG16[6] =
    {4, 0, 0, 0, 0, 0};

unsigned int VidBytesPerPixelTable_Alpha[][6] = {
    {1, 0, 0, 0, 0, 0}, /* 8 */
    {2, 0, 0, 0, 0, 0}, /* 10 */
    {2, 0, 0, 0, 0, 0}, /* 12 */
    {2, 0, 0, 0, 0, 0}, /* 14 */
    {2, 0, 0, 0, 0, 0}, /* 16 */
    {4, 0, 0, 0, 0, 0}, /* 32 */
};

unsigned int VidBytesPerPixelTable_RAW[][6] = {
    {1, 0, 0, 0, 0, 0}, /* 8 */
    {2, 0, 0, 0, 0, 0}, /* 10 */
    {2, 0, 0, 0, 0, 0}, /* 12 */
    {2, 0, 0, 0, 0, 0}, /* 14 */
    {2, 0, 0, 0, 0, 0}, /* 16 */
    {4, 0, 0, 0, 0, 0}, /* 32 */
    {4, 0, 0, 0, 0, 0}, /* 16_8_8 */
    {4, 0, 0, 0, 0, 0}, /* 10_8_8 */
    {4, 0, 0, 0, 0, 0}, /* 2_10_10_10 */
    {4, 0, 0, 0, 0, 0}, /* 20 */
};

unsigned int VidBytesPerPixelTable_YUV[][9][6] = {
    { /* PLANAR */
        {1, 1, 1, 0, 0, 0}, /* 8 */
        {2, 2, 2, 0, 0, 0}, /* 10 */
        {2, 2, 2, 0, 0, 0}, /* 12 */
        {2, 2, 2, 0, 0, 0}, /* 14 */
        {2, 2, 2, 0, 0, 0}, /* 16 */
        {4, 4, 4, 0, 0, 0}, /* 32 */
        {2, 1, 1, 0, 0, 0}, /* 16_8_8 */
        {2, 1, 1, 0, 0, 0}, /* 10_8_8 */
        {4, 0, 0, 0, 0, 0}, /* 2_10_10_10 */
    },
    { /* SEMI_PLANAR */
        {1, 2, 0, 0, 0, 0}, /* 8 */
        {2, 4, 0, 0, 0, 0}, /* 10 */
        {2, 4, 0, 0, 0, 0}, /* 12 */
        {2, 4, 0, 0, 0, 0}, /* 14 */
        {2, 4, 0, 0, 0, 0}, /* 16 */
        {4, 8, 0, 0, 0, 0}, /* 32 */
        {2, 2, 0, 0, 0, 0}, /* 16_8_8 */
        {2, 2, 0, 0, 0, 0}, /* 10_8_8 */
        {4, 0, 0, 0, 0, 0}, /* 2_10_10_10 */
    }
};

static NvMediaStatus
GetBytesPerCompForPackedYUV(unsigned int surfBPCidx,
                unsigned int *bytespercomp
)
{
    switch(surfBPCidx) {
    case NVM_SURF_ATTR_BITS_PER_COMPONENT_8:
    case NVM_SURF_ATTR_BITS_PER_COMPONENT_LAYOUT_2_10_10_10:
        *bytespercomp = 1;
        break;
    case NVM_SURF_ATTR_BITS_PER_COMPONENT_10:
    case NVM_SURF_ATTR_BITS_PER_COMPONENT_12:
    case NVM_SURF_ATTR_BITS_PER_COMPONENT_14:
    case NVM_SURF_ATTR_BITS_PER_COMPONENT_16:
        *bytespercomp = 2;
        break;
    case NVM_SURF_ATTR_BITS_PER_COMPONENT_20:
        *bytespercomp = 3;
        break;
    case NVM_SURF_ATTR_BITS_PER_COMPONENT_32:
        *bytespercomp = 4;
        break;
    default:
        return NVMEDIA_STATUS_ERROR;
    }
    return NVMEDIA_STATUS_OK;

}

static NvMediaStatus
GetSurfParams(unsigned int surfaceType,
             float **xScale,
             float **yScale,
             unsigned int **bytePerPixel,
             uint32_t *numSurfacesVal)
{
    NvMediaStatus status;
    unsigned int surfType, surfMemoryType, surfSubSamplingType, surfBPC, surfCompOrder;
    float *xScalePtr = NULL, *yScalePtr = NULL;
    unsigned int *bytePerPixelPtr = NULL;
    NVM_SURF_FMT_DEFINE_ATTR(srcAttr);
    uint32_t numSurfaces = 1;
    static unsigned int yuvpackedtbl[6] = {1, 0, 0, 0, 0, 0};
    unsigned int numcomps = 1;

    status = NvMediaSurfaceFormatGetAttrs(surfaceType,
                                          srcAttr,
                                          NVM_SURF_FMT_ATTR_MAX);
    if (status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s:NvMediaSurfaceFormatGetAttrs failed\n", __func__);
        return NVMEDIA_STATUS_ERROR;
    }

    surfType = srcAttr[NVM_SURF_ATTR_SURF_TYPE].value;
    surfMemoryType = srcAttr[NVM_SURF_ATTR_MEMORY].value;
    surfSubSamplingType = srcAttr[NVM_SURF_ATTR_SUB_SAMPLING_TYPE].value;
    surfBPC = srcAttr[NVM_SURF_ATTR_BITS_PER_COMPONENT].value;
    surfCompOrder = srcAttr[NVM_SURF_ATTR_COMPONENT_ORDER].value;

    switch(surfType) {
        case NVM_SURF_ATTR_SURF_TYPE_YUV:
            if (surfSubSamplingType == NVM_SURF_ATTR_SUB_SAMPLING_TYPE_NONE &&
                surfMemoryType == NVM_SURF_ATTR_MEMORY_PACKED) {

                xScalePtr =  &VidSurfParamsTable_Packed.widthFactor[0];
                yScalePtr = &VidSurfParamsTable_Packed.heightFactor[0];
                numSurfaces = VidSurfParamsTable_Packed.numSurfaces;

                if (NVMEDIA_STATUS_OK != GetBytesPerCompForPackedYUV(surfBPC, &yuvpackedtbl[0])) {
                    LOG_ERR("Invalid Bits per component and Packed YUV combination\n");
                    return NVMEDIA_STATUS_ERROR;
                }

                switch(surfCompOrder) {
                    case NVM_SURF_ATTR_COMPONENT_ORDER_VUYX:
                    case NVM_SURF_ATTR_COMPONENT_ORDER_XYUV:
                    case NVM_SURF_ATTR_COMPONENT_ORDER_XUYV:
                        numcomps = 4;
                        break;
                    case NVM_SURF_ATTR_COMPONENT_ORDER_UYVY:
                    case NVM_SURF_ATTR_COMPONENT_ORDER_VYUY:
                    case NVM_SURF_ATTR_COMPONENT_ORDER_YVYU:
                    case NVM_SURF_ATTR_COMPONENT_ORDER_YUYV:
                        numcomps = 2;
                        break;
                    case NVM_SURF_ATTR_COMPONENT_ORDER_LUMA:
                        numcomps = 1;
                        break;
                    default:
                        LOG_ERR("Invalid component Order  and Packed YUV combination\n");
                        return NVMEDIA_STATUS_ERROR;
                }
                yuvpackedtbl[0] = yuvpackedtbl[0] * numcomps;
                bytePerPixelPtr = &yuvpackedtbl[0];

            } else {
                xScalePtr = &VidSurfParamsTable_YUV[0][surfSubSamplingType - NVM_SURF_ATTR_SUB_SAMPLING_TYPE_420].widthFactor[0];
                yScalePtr = &VidSurfParamsTable_YUV[0][surfSubSamplingType - NVM_SURF_ATTR_SUB_SAMPLING_TYPE_420].heightFactor[0];
                numSurfaces = VidSurfParamsTable_YUV[0][surfSubSamplingType - NVM_SURF_ATTR_SUB_SAMPLING_TYPE_420].numSurfaces;
                bytePerPixelPtr = &VidBytesPerPixelTable_YUV[0][surfBPC - NVM_SURF_ATTR_BITS_PER_COMPONENT_8][0];
            }

            break;
        case NVM_SURF_ATTR_SURF_TYPE_RGBA:
            if (surfCompOrder == NVM_SURF_ATTR_COMPONENT_ORDER_ALPHA) {
                bytePerPixelPtr = &VidBytesPerPixelTable_Alpha[surfBPC - NVM_SURF_ATTR_BITS_PER_COMPONENT_8][0];
            } else if (surfCompOrder == NVM_SURF_ATTR_COMPONENT_ORDER_RG) {
                if(surfBPC == NVM_SURF_ATTR_BITS_PER_COMPONENT_16) {
                    bytePerPixelPtr = &VidBytesPerPixelTable_RG16[0];
                } else {
                    LOG_ERR("Invalid RGorder & Bitspercomp combination.Only RG16 is supported\n");
                    return NVMEDIA_STATUS_ERROR;
                }
            } else { /* RGBA, ARGB, BGRA */
                if (surfBPC == NVM_SURF_ATTR_BITS_PER_COMPONENT_16) {
                    bytePerPixelPtr = &VidBytesPerPixelTable_RGBA16[0][0];
                } else if (surfBPC == NVM_SURF_ATTR_BITS_PER_COMPONENT_8) {
                    bytePerPixelPtr = &VidBytesPerPixelTable_RGBA[0][0];
                } else {
                    LOG_ERR("RGBA orders with 8 and 16bits only is supported \n");
                    return NVMEDIA_STATUS_ERROR;
                }
            }
            xScalePtr = &VidSurfParamsTable_RGBA.widthFactor[0];
            yScalePtr = &VidSurfParamsTable_RGBA.heightFactor[0];
            numSurfaces =  VidSurfParamsTable_RGBA.numSurfaces;
            break;
        case NVM_SURF_ATTR_SURF_TYPE_RAW:
            bytePerPixelPtr = &VidBytesPerPixelTable_RAW[surfBPC - NVM_SURF_ATTR_BITS_PER_COMPONENT_8][0];
            xScalePtr = &VidSurfParamsTable_RAW.widthFactor[0];
            yScalePtr = &VidSurfParamsTable_RAW.heightFactor[0];
            numSurfaces =  VidSurfParamsTable_RAW.numSurfaces;
            break;
        default:
            LOG_ERR("%s: Unsupported Pixel Format %d", __func__, surfType);
            return NVMEDIA_STATUS_ERROR;
    }

    if (xScale) {
        *xScale = xScalePtr;
    }
    if (yScale) {
        *yScale = yScalePtr;
    }
    if (bytePerPixel) {
        *bytePerPixel = bytePerPixelPtr;
    }
    if (numSurfacesVal) {
        *numSurfacesVal = numSurfaces;
    }

    return NVMEDIA_STATUS_OK;
}


static NvMediaStatus
WriteFrameNew(
    char *filename,
    NvMediaVideoSurface *videoSurface,
    NvMediaBool bOrderUV,
    NvMediaBool bAppend,
    NvMediaRect *srcRect)
{
    uint8_t **pBuff = NULL;
    uint32_t *pBuffPitches = NULL;
    uint32_t imageSize = 0;
    unsigned int size[3] ={0};
    uint8_t *buffer = NULL;
    uint32_t i, k, newk = 0;
    float *xScalePtr = NULL, *yScalePtr = NULL;
    unsigned int *bytePerPixelPtr = NULL;
    uint32_t numSurfaces = 1;
    unsigned int height, width;
    NvMediaVideoSurfaceMap surfaceMap;
    NvMediaStatus status;
    FILE *file = NULL;
    uint8_t *cropBuffer = NULL, *srcAddr, *dstAddr;
    uint32_t lineWidth, numRows, startOffset;

    if(!videoSurface || !filename) {
        LOG_ERR("%s: Bad parameter\n", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    height = videoSurface->height;
    width  = videoSurface->width;

    file = fopen(filename, bAppend ? "ab" : "wb");
    if(!file) {
        LOG_ERR("%s: File open failed: %s\n", __func__, filename);
        status = NVMEDIA_STATUS_ERROR;
        goto done;
    }

    pBuff = malloc(sizeof(uint8_t*) * MAXM_NUM_SURFACES);
    if(!pBuff) {
        LOG_ERR("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    pBuffPitches = calloc(1, sizeof(uint32_t) * MAXM_NUM_SURFACES);
    if(!pBuffPitches) {
        LOG_ERR("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    status = GetSurfParams(videoSurface->type,
                           &xScalePtr,
                           &yScalePtr,
                           &bytePerPixelPtr,
                           &numSurfaces);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: GetSurfParams() failed\n", __func__);
        goto done;
    }

    imageSize = 0;
    for(i = 0; i < numSurfaces; i++) {
        size[i] = width * xScalePtr[i] * height * yScalePtr[i] * bytePerPixelPtr[i];
        imageSize += size[i];
        pBuffPitches[i] = (uint32_t)((float)width * xScalePtr[i]) * bytePerPixelPtr[i];
    }

    buffer = calloc(1, imageSize);
    if(!buffer) {
        LOG_ERR("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    memset(buffer, 0xFF,imageSize);
    pBuff[0] = buffer;
    for(i = 1; i < numSurfaces; i++) {
        pBuff[i] = pBuff[i - 1] + (uint32_t)(height * yScalePtr[i - 1] * pBuffPitches[i - 1]);
    }

    status = NvMediaVideoSurfaceLock(videoSurface, &surfaceMap);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: NvMediaVideoSurfaceLock() failed\n", __func__);
        goto done;
    }
    LOG_DBG("WriteFrame: %s Size: %dx%d Luma pitch: %d Chroma pitch: %d Chroma type: %d\n",
            filename, surfaceMap.lumaWidth, surfaceMap.lumaHeight, surfaceMap.pitchY, surfaceMap.pitchU, videoSurface->type);

    status = NvMediaVideoSurfaceGetBits(videoSurface, NULL, (void **)pBuff, pBuffPitches);
    NvMediaVideoSurfaceUnlock(videoSurface);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: NvMediaVideoSurfaceGetBits() failed\n", __func__);
        goto done;
    }

    /* Check if src rect is within the boundaries of image */
    if (srcRect) {
        if ((srcRect->x0 > videoSurface->width) ||
            (srcRect->y0 > videoSurface->height) ||
            (srcRect->x1 > videoSurface->width) ||
            (srcRect->y1 > videoSurface->height) ||
            (srcRect->x0 >= srcRect->x1) ||
            (srcRect->y0 >= srcRect->y1)) {
           LOG_ERR("%s: Invalid srcRect parameters. Ignoring srcRect..\n", __func__);
        }
        else if (((uint32_t)(srcRect->x1 - srcRect->x0) != videoSurface->width) ||
                 ((uint32_t)(srcRect->y1 - srcRect->y0) != videoSurface->height)) {
            /* Copy only if src and dst dimensions are different */
            imageSize = 0;
            for (i = 0; i < numSurfaces; i++) {
            imageSize += ((srcRect->x1 - srcRect->x0) * xScalePtr[i] * bytePerPixelPtr[i]) *
                          ((srcRect->y1 - srcRect->y0) * yScalePtr[i]);
            }

            /* Allocate destination buffer */
            cropBuffer = calloc(1, imageSize);
            if (!cropBuffer) {
                LOG_ERR("%s: Out of memory\n", __func__);
                status = NVMEDIA_STATUS_OUT_OF_MEMORY;
                goto done;
            }

            dstAddr     = cropBuffer;
            for (k = 0; k < numSurfaces; k++) {
                startOffset = (srcRect->x0 * xScalePtr[k] * bytePerPixelPtr[k]) +
                              (srcRect->y0 * yScalePtr[k] * pBuffPitches[k]);
                srcAddr     = pBuff[k] + startOffset;
                numRows     = (srcRect->y1 - srcRect->y0) * yScalePtr[k];
                lineWidth   = (srcRect->x1 - srcRect->x0) * xScalePtr[k] * bytePerPixelPtr[k];

                pBuff[k]    = dstAddr;
                size[k]     = lineWidth * numRows;

                for (i = 0; i < numRows ; i++) {
                    memcpy (dstAddr, srcAddr, lineWidth);
                    dstAddr += lineWidth;
                    srcAddr += pBuffPitches[k];
                }
            }
        }
    }

    for(k = 0; k < numSurfaces; k++) {
       newk = (!bOrderUV && k ) ? (numSurfaces - k) : k;
       if (fwrite(pBuff[newk], size[newk], 1, file) != 1) {
           LOG_ERR("%s: File write failed\n", __func__);
           status = NVMEDIA_STATUS_ERROR;
           goto done;
       }
    }

done:
    if(file) {
        fclose(file);
    }

    if(pBuff) {
        free(pBuff);
    }

    if(buffer) {
        free(buffer);
    }

    if(pBuffPitches) {
        free(pBuffPitches);
    }

    if (cropBuffer) {
        free (cropBuffer);
    }

    return status;
}


NvMediaStatus
WriteFrame(
    char *filename,
    NvMediaVideoSurface *videoSurface,
    NvMediaBool bOrderUV,
    NvMediaBool bAppend,
    NvMediaRect *srcRect)
{
    NvMediaStatus status;
    NVM_SURF_FMT_DEFINE_ATTR(srcAttr);

    if(!videoSurface || !filename) {
        LOG_ERR("WriteFrame: Bad parameter\n");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    status = NvMediaSurfaceFormatGetAttrs(videoSurface->type,
                                          srcAttr,
                                          NVM_SURF_FMT_ATTR_MAX);
    if (status == NVMEDIA_STATUS_OK) {
        return WriteFrameNew(
                         filename,
                         videoSurface,
                         bOrderUV,
                         bAppend,
                         srcRect);
    } else {
        LOG_ERR("%s:NvMediaSurfaceFormatGetAttrs failed\n", __func__);
        return status;
    }
}

static NvMediaStatus
ReadFrameNew(
    char *fileName,
    uint32_t uFrameNum,
    uint32_t uWidth,
    uint32_t uHeight,
    NvMediaVideoSurface *pFrame,
    NvMediaBool bOrderUV,
    uint32_t pixelAlignment)
{
    uint8_t **pBuff = NULL;
    uint32_t *pBuffPitches = NULL;
    uint32_t frameSize = 0, surfaceSize = 0;
    uint8_t *buffer = NULL;
    uint8_t *pBuffer = NULL;
    uint32_t i, j, k, newk = 0;
    float *xScalePtr = NULL, *yScalePtr = NULL;
    unsigned int *bytePerPixelPtr = NULL;
    uint32_t numSurfaces = 1;
    unsigned int uHeightSurface, uWidthSurface;
    NvMediaVideoSurfaceMap surfaceMap;
    NvMediaStatus status = NVMEDIA_STATUS_ERROR;
    FILE *file = NULL;
    unsigned int count, index;
    unsigned int surfType;
    NVM_SURF_FMT_DEFINE_ATTR(srcAttr);
    uint16_t *psrc;
    uint32_t surfBPC;

    if(!fileName || !pFrame) {
        LOG_ERR("%s: Bad parameter\n", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    file = fopen(fileName, "rb");
    if(!file) {
        LOG_ERR("%s: File open failed: %s\n", __func__, fileName);
        status = NVMEDIA_STATUS_ERROR;
        goto done;
    }

    pBuff = malloc(sizeof(uint8_t*) * MAXM_NUM_SURFACES);
    if(!pBuff) {
        LOG_ERR("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    pBuffPitches = calloc(1, sizeof(uint32_t) * MAXM_NUM_SURFACES);
    if(!pBuffPitches) {
        LOG_ERR("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    status = NvMediaSurfaceFormatGetAttrs(pFrame->type,
                                          srcAttr,
                                          NVM_SURF_FMT_ATTR_MAX);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: NvMediaSurfaceFormatGetAttrs() failed\n", __func__);
        goto done;
    }

    surfType = srcAttr[NVM_SURF_ATTR_SURF_TYPE].value;
    surfBPC = srcAttr[NVM_SURF_ATTR_BITS_PER_COMPONENT].value;

    if (surfType == NVM_SURF_ATTR_SURF_TYPE_RGBA) {
        uHeightSurface = pFrame->height;
        uWidthSurface  = pFrame->width;
    } else {
        status = NvMediaVideoSurfaceLock(pFrame, &surfaceMap);
        if(status != NVMEDIA_STATUS_OK) {
            LOG_ERR("%s: NvMediaVideoSurfaceLock() failed\n", __func__);
            goto done;
        }

        NvMediaVideoSurfaceUnlock(pFrame);

        uHeightSurface = surfaceMap.lumaHeight;
        uWidthSurface  = surfaceMap.lumaWidth;
    }

    status = GetSurfParams(pFrame->type,
                           &xScalePtr,
                           &yScalePtr,
                           &bytePerPixelPtr,
                           &numSurfaces);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: GetSurfParams() failed\n", __func__);
        goto done;
    }

    surfaceSize = 0;
    frameSize = 0;
    for(i = 0; i < numSurfaces; i++) {
        surfaceSize += (uWidthSurface * xScalePtr[i] * uHeightSurface * yScalePtr[i] * bytePerPixelPtr[i]);
        frameSize += (uWidth * xScalePtr[i] * uHeight * yScalePtr[i] * bytePerPixelPtr[i]);
        pBuffPitches[i] = (uint32_t)((float)uWidthSurface * xScalePtr[i]) * bytePerPixelPtr[i];
    }

    pBuffer = calloc(1, surfaceSize);
    if(!pBuffer) {
        LOG_ERR("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    buffer = pBuffer;
    memset(buffer,0x10,surfaceSize);
    for(i = 0; i < numSurfaces; i++) {
        pBuff[i] = buffer;
        if (i) {
            memset(pBuff[i], 0x80, (uHeightSurface * yScalePtr[i] * pBuffPitches[i]));
        }
        buffer = buffer + (uint32_t)(uHeightSurface * yScalePtr[i] * pBuffPitches[i]);
    }

    if(uFrameNum > 0) {
        if(fseeko(file, uFrameNum * (off_t)frameSize, SEEK_SET)) {
            LOG_ERR("%s: Error seeking file\n", __func__);
            status = NVMEDIA_STATUS_ERROR;
            goto done;
        }
    }

    if((surfType == NVM_SURF_ATTR_SURF_TYPE_RGBA ) && strstr(fileName, ".png")) {
        LOG_ERR("%s: Does not support PNG format\n", __func__);
        status = NVMEDIA_STATUS_ERROR;
        goto done;
    }
    for(k = 0; k < numSurfaces; k++) {
        for(j = 0; j < uHeight*yScalePtr[k]; j++) {
            count = uWidth * xScalePtr[k] * bytePerPixelPtr[k];
            newk = (!bOrderUV && k ) ? (numSurfaces - k) : k;
            index = j * pBuffPitches[newk];
            if (fread(pBuff[newk] + index, count, 1, file) != 1) {
                LOG_ERR("ReadFrame: Error reading file: %s\n", fileName);
                status = NVMEDIA_STATUS_ERROR;
                goto done;
            }
            if((surfType == NVM_SURF_ATTR_SURF_TYPE_YUV) && (pixelAlignment == LSB_ALIGNED)) {
                psrc = (uint16_t*)(pBuff[newk] + index);
                switch(surfBPC) {
                    case NVM_SURF_ATTR_BITS_PER_COMPONENT_10:
                        for(i = 0; i < count/2; i++) {
                            *(psrc + i) = (*(psrc + i)) << (16 - 10);
                        }
                        break;
                    case NVM_SURF_ATTR_BITS_PER_COMPONENT_12:
                        for(i = 0; i < count/2; i++) {
                            *(psrc + i) = (*(psrc + i)) << (16 - 12);
                        }
                        break;
                    case NVM_SURF_ATTR_BITS_PER_COMPONENT_14:
                        for(i = 0; i < count/2; i++) {
                            *(psrc + i) = (*(psrc + i)) << (16 - 14);
                        }
                        break;
                    default:
                        break;
                }
            }
        }
    }

    status = NvMediaVideoSurfaceLock(pFrame, &surfaceMap);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: NvMediaVideoSurfaceLock() failed\n", __func__);
        goto done;
    }

    status = NvMediaVideoSurfacePutBits(pFrame, NULL, (void **)pBuff, pBuffPitches);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: NvMediaVideoSurfacePutBits() failed\n", __func__);
    }
    NvMediaVideoSurfaceUnlock(pFrame);

done:
    if(file) {
        fclose(file);
    }

    if (pBuffer) {
        free(pBuffer);
    }

    if (pBuffPitches) {
        free(pBuffPitches);
    }

    if(pBuff) {
        free(pBuff);
    }

    return status;
}

NvMediaStatus
ReadFrame(
    char *fileName,
    uint32_t uFrameNum,
    uint32_t uWidth,
    uint32_t uHeight,
    NvMediaVideoSurface *pFrame,
    NvMediaBool bOrderUV,
    uint32_t pixelAlignment)
{
    NvMediaStatus status;
    NVM_SURF_FMT_DEFINE_ATTR(srcAttr);

    status = NvMediaSurfaceFormatGetAttrs(pFrame->type,
                                          srcAttr,
                                          NVM_SURF_FMT_ATTR_MAX);
    if (status == NVMEDIA_STATUS_OK) {
        return ReadFrameNew(
                        fileName,
                        uFrameNum,
                        uWidth,
                        uHeight,
                        pFrame,
                        bOrderUV,
                        pixelAlignment);
    } else {
        LOG_ERR("%s:NvMediaSurfaceFormatGetAttrs failed\n", __func__);
        return status;
    }
}

NvMediaStatus
ReadPPMFrame(
    char *fileName,
    NvMediaVideoSurface *pFrame)
{
    uint32_t uSurfaceWidth;
    uint32_t uSurfaceHeight;
    uint32_t maxValue = 0;
    int num = 0;
    uint32_t x, y;
    char buf[256], *c;
    uint8_t *pBuff = NULL, *pRGBBuff;
    uint32_t uFrameSize;
    FILE *file;
    NvMediaVideoSurfaceMap surfaceMap;
    NvMediaStatus ret = NVMEDIA_STATUS_OK;

    if(!pFrame) {
        LOG_ERR("ReadPPMFrame: Failed allocating memory\n");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    file = fopen(fileName, "rb");
    if(!file) {
        LOG_ERR("ReadPPMFrame: Error opening file: %s\n", fileName);
        return NVMEDIA_STATUS_ERROR;
    }

    c = fgets(buf, 256, file);
    if(!c || strncmp(buf, "P6\n", 3)) {
        LOG_ERR("ReadPPMFrame: Invalid PPM header in file: %s\n", fileName);
        ret = NVMEDIA_STATUS_ERROR;
        goto done;
    }
    do {
        c = fgets(buf, 255, file);
        if(!c) {
            LOG_ERR("ReadPPMFrame: Invalid PPM header in file: %s\n", fileName);
            ret = NVMEDIA_STATUS_ERROR;
            goto done;
        }
    } while(!strncmp(buf, "#", 1));

    num = sscanf(buf, "%u %u %u", &uSurfaceWidth, &uSurfaceHeight, &maxValue);
    switch(num) {
    case 2:
        c = fgets(buf, 255, file);
        if(!c || strncmp(buf, "255\n", 4)) {
            LOG_ERR("ReadPPMFrame: Invalid PPM header in file: %s\n", fileName);
            ret = NVMEDIA_STATUS_ERROR;
            goto done;
        } else
            break;
    case 3:
        if(maxValue != 255) {
            LOG_ERR("ReadPPMFrame: Invalid PPM header in file: %s\n", fileName);
            ret = NVMEDIA_STATUS_ERROR;
            goto done;
        } else
            break;
    default:
        LOG_ERR("ReadPPMFrame: Error getting PPM file resolution in file: %s\n", fileName);
        ret = NVMEDIA_STATUS_ERROR;
        goto done;
    }

    if(uSurfaceWidth < 16 || uSurfaceWidth > 2048 ||
            uSurfaceHeight < 16 || uSurfaceHeight > 2048) {
        LOG_ERR("ReadPPMFrame: Invalid PPM file resolution: %ux%u for file: %s\n", uSurfaceWidth, uSurfaceHeight, fileName);
        ret = NVMEDIA_STATUS_ERROR;
        goto done;
    }

    uFrameSize = uSurfaceWidth * uSurfaceHeight * 3;
    pBuff = malloc(uFrameSize);
    if(!pBuff) {
        LOG_ERR("ReadPPMFrame: Out of memory\n");
        ret = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }
    pRGBBuff = pBuff;

    if(fread(pRGBBuff, uFrameSize, 1, file) != 1) {
        LOG_ERR("ReadPPMFrame: Error reading file: %s\n", fileName);
        ret = NVMEDIA_STATUS_ERROR;
        goto done;
    }

    ret = NvMediaVideoSurfaceLock(pFrame, &surfaceMap);
    if (ret != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: NvMediaVideoSurfaceLock() failed\n", __func__);
        goto done;
    }

    for(y = 0; y < uSurfaceHeight; y++) {
        uint8_t *pPixel = (uint8_t *)surfaceMap.pRGBA + surfaceMap.pitchRGBA * y;
        for(x = 0; x < uSurfaceWidth; x++) {
            *pPixel++ = *pRGBBuff++; // R
            *pPixel++ = *pRGBBuff++; // G
            *pPixel++ = *pRGBBuff++; // B
            *pPixel++ = 255;         // Alpha
        }
    }

done:
    NvMediaVideoSurfaceUnlock(pFrame);
    if(pBuff) free(pBuff);
    if(file) fclose(file);

    return ret;
}

static NvMediaStatus
GetSurfaceCrcNew_Crop(
    NvMediaVideoSurface *videoSurface,
    NvMediaRect *srcRect,
    NvMediaBool monochromeFlag,
    uint32_t *crcOut)
{
    NvMediaVideoSurfaceMap surfMap;
    uint32_t lines, crc = 0;
    NvMediaStatus status;
    uint8_t **pBuff = NULL;
    uint32_t *pBuffPitches = NULL;
    uint32_t *pBuffWidthInBytes = NULL;
    uint8_t *bufferSurf = NULL;
    uint8_t *bufferTmp = NULL;
    uint32_t i = 0;
    uint32_t width = srcRect->x1;
    uint32_t width_temp;
    uint32_t height = srcRect->y1;
    float *xScalePtr = NULL, *yScalePtr = NULL;
    unsigned int *bytePerPixelPtr = NULL;
    uint32_t numSurfaces = 1;
    uint32_t uHeightSurface, uWidthSurface, imageSize;

    if(!videoSurface || !crcOut) {
        LOG_ERR("%s: Bad parameter\n", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    status = NvMediaVideoSurfaceLock(videoSurface, &surfMap);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: NvMediaVideoSurfaceLock() failed\n", __func__);
        return status;
    }
    NvMediaVideoSurfaceUnlock(videoSurface);

    uHeightSurface = surfMap.lumaHeight;
    uWidthSurface  = surfMap.lumaWidth;

    if(width > uWidthSurface || height > uHeightSurface) {
        LOG_ERR("%s: Bad parameter\n", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    status = GetSurfParams(videoSurface->type,
                           &xScalePtr,
                           &yScalePtr,
                           &bytePerPixelPtr,
                           &numSurfaces);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: GetSurfParams() failed\n", __func__);
        return status;
    }

    pBuff = calloc(1,sizeof(uint8_t*)*MAXM_NUM_SURFACES);
    if(!pBuff) {
        LOG_ERR("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    pBuffPitches = calloc(1,sizeof(uint32_t) * MAXM_NUM_SURFACES);
    if(!pBuffPitches) {
        LOG_ERR("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    pBuffWidthInBytes = calloc(1,sizeof(uint32_t) * MAXM_NUM_SURFACES);
    if(!pBuffWidthInBytes) {
        LOG_ERR("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    imageSize = 0;
    for(i = 0; i < numSurfaces; i++) {
        imageSize += (uWidthSurface * xScalePtr[i] * uHeightSurface * yScalePtr[i] * bytePerPixelPtr[i]);
        pBuffWidthInBytes[i] = (uint32_t)((float)uWidthSurface * xScalePtr[i]) * bytePerPixelPtr[i];
        pBuffPitches[i] = (uint32_t)((float)uWidthSurface * xScalePtr[i]) * bytePerPixelPtr[i];
    }

    bufferSurf = calloc(1, imageSize);
    if(!bufferSurf) {
        LOG_ERR("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    memset(bufferSurf,0xFF,imageSize);
    bufferTmp = bufferSurf;
    for(i = 0; i < numSurfaces; i++) {
        pBuff[i] = bufferTmp;
        bufferTmp = bufferTmp + (uint32_t)(uHeightSurface * yScalePtr[i] * pBuffPitches[i]);
    }

    status = NvMediaVideoSurfaceLock(videoSurface, &surfMap);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: NvMediaVideoSurfaceLock() failed\n", __func__);
        goto done;
    }
    status = NvMediaVideoSurfaceGetBits(videoSurface, NULL, (void **)pBuff, pBuffPitches);
    NvMediaVideoSurfaceUnlock(videoSurface);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: NvMediaVideoSurfaceGetBits failed\n", __func__);
        goto done;
    }
    NvMediaVideoSurfaceUnlock(videoSurface);

    for(i = 0; i < numSurfaces; i++) {
        if(i > 0 && monochromeFlag == true) {
            break;
        }
        lines = height*yScalePtr[i];
        if (i > 0)
            lines = (((height + 0x1) & (~0x1)) * yScalePtr[i]);

        bufferTmp = pBuff[i] + (srcRect->x0 * bytePerPixelPtr[i]) + (srcRect->y0 * pBuffPitches[i]);
        if (i > 0)
            width_temp  = (((width + 0x1) & (~0x1)) * xScalePtr[i]);
        else
            width_temp = width;
        while(lines--) {
            crc = CalculateBufferCRC(width_temp * bytePerPixelPtr[i], crc, bufferTmp);
            bufferTmp += pBuffPitches[i];
        }
    }

    *crcOut = crc;

done:
    if(pBuff) {
        free(pBuff);
    }

    if(pBuffPitches) {
        free(pBuffPitches);
    }

    if(pBuffWidthInBytes) {
        free(pBuffWidthInBytes);
    }

    if(bufferSurf) {
        free(bufferSurf);
    }

    return status;
}

static NvMediaStatus
GetSurfaceCrcNew(
    NvMediaVideoSurface *videoSurface,
    uint32_t width,
    uint32_t height,
    NvMediaBool monochromeFlag,
    uint32_t *crcOut)
{
    NvMediaVideoSurfaceMap surfMap;
    uint32_t lines, crc = 0;
    NvMediaStatus status;
    uint8_t **pBuff = NULL;
    uint32_t *pBuffPitches = NULL;
    uint32_t *pBuffWidthInBytes = NULL;
    uint8_t *bufferSurf = NULL;
    uint8_t *bufferTmp = NULL;
    uint32_t i = 0;
    float *xScalePtr = NULL, *yScalePtr = NULL;
    unsigned int *bytePerPixelPtr = NULL;
    uint32_t numSurfaces = 1;
    uint32_t uHeightSurface, uWidthSurface, imageSize;

    if(!videoSurface || !crcOut) {
        LOG_ERR("%s: Bad parameter\n", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    status = NvMediaVideoSurfaceLock(videoSurface, &surfMap);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: NvMediaVideoSurfaceLock() failed\n", __func__);
        return status;
    }
    NvMediaVideoSurfaceUnlock(videoSurface);

    uHeightSurface = surfMap.lumaHeight;
    uWidthSurface  = surfMap.lumaWidth;

    if(width > uWidthSurface || height > uHeightSurface) {
        LOG_ERR("%s: Bad parameter\n", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    status = GetSurfParams(videoSurface->type,
                           &xScalePtr,
                           &yScalePtr,
                           &bytePerPixelPtr,
                           &numSurfaces);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: GetSurfParams() failed\n", __func__);
        return status;
    }

    pBuff = calloc(1,sizeof(uint8_t*)*MAXM_NUM_SURFACES);
    if(!pBuff) {
        LOG_ERR("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    pBuffPitches = calloc(1,sizeof(uint32_t) * MAXM_NUM_SURFACES);
    if(!pBuffPitches) {
        LOG_ERR("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    pBuffWidthInBytes = calloc(1,sizeof(uint32_t) * MAXM_NUM_SURFACES);
    if(!pBuffWidthInBytes) {
        LOG_ERR("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    imageSize = 0;
    for(i = 0; i < numSurfaces; i++) {
        imageSize += (uWidthSurface * xScalePtr[i] * uHeightSurface * yScalePtr[i] * bytePerPixelPtr[i]);
        pBuffWidthInBytes[i] = (uint32_t)((float)width * xScalePtr[i]) * bytePerPixelPtr[i];
        pBuffPitches[i] = (uint32_t)((float)uWidthSurface * xScalePtr[i]) * bytePerPixelPtr[i];
    }

    bufferSurf = calloc(1, imageSize);
    if(!bufferSurf) {
        LOG_ERR("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    memset(bufferSurf,0xFF,imageSize);
    bufferTmp = bufferSurf;
    for(i = 0; i < numSurfaces; i++) {
        pBuff[i] = bufferTmp;
        bufferTmp = bufferTmp + (uint32_t)(uHeightSurface * yScalePtr[i] * pBuffPitches[i]);
    }

    status = NvMediaVideoSurfaceLock(videoSurface, &surfMap);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: NvMediaVideoSurfaceLock() failed\n", __func__);
        goto done;
    }
    status = NvMediaVideoSurfaceGetBits(videoSurface, NULL, (void **)pBuff, pBuffPitches);
    NvMediaVideoSurfaceUnlock(videoSurface);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: NvMediaVideoSurfaceGetBits failed\n", __func__);
        goto done;
    }
    NvMediaVideoSurfaceUnlock(videoSurface);

    for(i = 0; i < numSurfaces; i++) {
        if(i > 0 && monochromeFlag == true) {
            break;
        }
        lines = height*yScalePtr[i];
        bufferTmp = pBuff[i];
        while(lines--) {
            crc = CalculateBufferCRC(pBuffWidthInBytes[i], crc, bufferTmp);
            bufferTmp += pBuffPitches[i];
        }
    }

    *crcOut = crc;

done:
    if(pBuff) {
        free(pBuff);
    }

    if(pBuffPitches) {
        free(pBuffPitches);
    }

    if(pBuffWidthInBytes) {
        free(pBuffWidthInBytes);
    }

    if(bufferSurf) {
        free(bufferSurf);
    }

    return status;
}



NvMediaStatus
GetSurfaceCrc_New(
    NvMediaVideoSurface *surf,
    NvMediaRect *srcRect,
    NvMediaBool monochromeFlag,
    uint32_t *crcOut)
{
    NvMediaStatus status;
    NVM_SURF_FMT_DEFINE_ATTR(srcAttr);

    if(!surf || !crcOut) {
        LOG_ERR("GetImageCrc: Bad parameter\n");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    status = NvMediaSurfaceFormatGetAttrs(surf->type,
                                          srcAttr,
                                          NVM_SURF_FMT_ATTR_MAX);
    if (status == NVMEDIA_STATUS_OK) {
        return GetSurfaceCrcNew_Crop(
                                surf,
                                srcRect,
                                monochromeFlag,
                                crcOut);
    } else {
        LOG_ERR("%s:NvMediaSurfaceFormatGetAttrs failed\n", __func__);
        return status;
    }
}

NvMediaStatus
GetSurfaceCrc(
    NvMediaVideoSurface *surf,
    uint32_t width,
    uint32_t height,
    NvMediaBool monochromeFlag,
    uint32_t *crcOut)
{
    NvMediaStatus status;
    NVM_SURF_FMT_DEFINE_ATTR(srcAttr);

    if(!surf || !crcOut) {
        LOG_ERR("GetImageCrc: Bad parameter\n");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    status = NvMediaSurfaceFormatGetAttrs(surf->type,
                                          srcAttr,
                                          NVM_SURF_FMT_ATTR_MAX);
    if (status == NVMEDIA_STATUS_OK) {
        return GetSurfaceCrcNew(
                                surf,
                                width,
                                height,
                                monochromeFlag,
                                crcOut);
    } else {
        LOG_ERR("%s:NvMediaSurfaceFormatGetAttrs failed\n", __func__);
        return status;
    }
}

NvMediaStatus
CheckSurfaceCrc_New(
    NvMediaVideoSurface *surf,
    NvMediaRect *srcRect,
    NvMediaBool monochromeFlag,
    uint32_t ref,
    NvMediaBool *isMatching)
{
    uint32_t crc = 0;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    status = GetSurfaceCrc_New(surf, srcRect, monochromeFlag, &crc);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("CheckSurfaceCrc_New: GetSurfaceCrc failed\n");
        return NVMEDIA_STATUS_ERROR;
    }

    if(crc != ref) {
        LOG_WARN("CheckSurfaceCrc_New: Encountered CRC mismatch.\n");
        LOG_WARN("CheckSurfaceCrc_New: Calculated CRC: %8x (%d). Expected CRC: %8x (%d).\n", crc, crc, ref, ref);
        *isMatching = NVMEDIA_FALSE;
    } else {
        *isMatching = NVMEDIA_TRUE;
    }

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
CheckSurfaceCrc(
    NvMediaVideoSurface *surf,
    uint32_t width,
    uint32_t height,
    NvMediaBool monochromeFlag,
    uint32_t ref,
    NvMediaBool *isMatching)
{
    uint32_t crc = 0;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    status = GetSurfaceCrc(surf, width, height, monochromeFlag, &crc);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("CheckSurfaceCrc: GetSurfaceCrc failed\n");
        return NVMEDIA_STATUS_ERROR;
    }

    if(crc != ref) {
        LOG_WARN("CheckSurfaceCrc: Encountered CRC mismatch.\n");
        LOG_WARN("CheckSurfaceCrc: Calculated CRC: %8x (%d). Expected CRC: %8x (%d).\n", crc, crc, ref, ref);
        *isMatching = NVMEDIA_FALSE;
    } else {
        *isMatching = NVMEDIA_TRUE;
    }

    return NVMEDIA_STATUS_OK;
}


