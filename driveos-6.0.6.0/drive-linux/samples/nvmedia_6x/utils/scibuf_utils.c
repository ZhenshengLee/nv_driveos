/*
 * Copyright (c) 2021-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software and related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "nvscibuf.h"
#include "nvmedia_nvscibuf.h"

#include "misc_utils.h"
#include "scibuf_utils.h"
#include "log_utils.h"

#define MAX_NUM_SURFACES 6

typedef struct {
    float heightFactor[6];
    float widthFactor[6];
    unsigned int numSurfaces;
} SurfDesc;

#define SURF_DESC_INDEX_PLANAR      0
#define SURF_DESC_INDEX_SEMI_PLANAR 1
#define SURF_DESC_INDEX_PACKED      2
#define SURF_DESC_INDEX_420         0
#define SURF_DESC_INDEX_422         1
#define SURF_DESC_INDEX_444         2

SurfDesc SurfDescTable_RGBA  = {
    .heightFactor = {1, 0, 0, 0, 0, 0},
    .widthFactor = {1, 0, 0, 0, 0, 0},
    .numSurfaces = 1,
};

SurfDesc SurfDescTable_YUV[][4] = {
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
    },
};

SurfDesc SurfDescTable_Packed  = {
    .heightFactor = {1, 0, 0, 0, 0, 0},
    .widthFactor = {1, 0, 0, 0, 0, 0},
    .numSurfaces = 1,
};

unsigned int SurfBytesPerPixelTable_YUV[][9][6] = {
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

unsigned int SurfBytesPerPixelTable_RGBA[][6] = {
    {4, 0, 0, 0, 0, 0}, /* 8 */
};

unsigned int SurfBytesPerPixelTable_RGBA16[][6] = {
    {8, 0, 0, 0, 0, 0}, /* 16 */
};

unsigned int SurfBytesPerPixelTable_RG16[6] =
    {4, 0, 0, 0, 0, 0};

unsigned int SurfBytesPerPixelTable_Alpha[][6] = {
    {1, 0, 0, 0, 0, 0}, /* 8 */
    {2, 0, 0, 0, 0, 0}, /* 10 */
    {2, 0, 0, 0, 0, 0}, /* 12 */
    {2, 0, 0, 0, 0, 0}, /* 14 */
    {2, 0, 0, 0, 0, 0}, /* 16 */
    {4, 0, 0, 0, 0, 0}, /* 32 */
};

NvMediaStatus
ClearSurface(
    uint32_t width,
    uint32_t height,
    NvSciBufObj bufObj,
    ChromaFormat DataFormat)
{
    NvSciError err;
    uint8_t **pBuff = NULL;
    uint32_t *pBuffPitches = NULL;
    uint8_t *buffer = NULL;
    uint8_t *pBuffer = NULL;
    uint32_t *pBuffSizes = NULL;
    uint32_t imageSize = 0, surfaceSize = 0;
    uint32_t i;
    /* Number of surfaces in the ChromaFormat of the input file */
    uint32_t numSurfaces = 1;
    uint32_t lumaWidth, lumaHeight;
    NvMediaStatus status = NVMEDIA_STATUS_ERROR;
    /* Extract surface info from NvSciBuf */
    NvSciBufAttrList attrList;
    uint32_t const *planeWidth, *planeHeight;
    float *xScalePtr = NULL, *yScalePtr = NULL;
    unsigned int *srcBytesPerPixelPtr = NULL;
    uint32_t bitsPerPixel = 0U;
    /* Passed outside the function, needs to be static */
    static unsigned int yuvpackedtbl[6] = {1, 0, 0, 0, 0, 0};
    unsigned int numcomps = 1;

    if(!bufObj) {
        LOG_ERR("%s: Bad parameter\n", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    /* Assumptions: This function assumes that
       - Input file is in planar format
       - Input format is of type YUV
       - Output surface is in semi-planar format
       - Output surface can be written to (ops on the surface have completed)
       */

    /* TODO: Add support for other formats */
    switch(DataFormat) {
        case YUV400P_8bit:
        case YUV400P_10bit:
        case YUV400P_12bit:
        case YUV400P_16bit:
            xScalePtr =  &SurfDescTable_Packed.widthFactor[0];
            yScalePtr = &SurfDescTable_Packed.heightFactor[0];
            numSurfaces = SurfDescTable_Packed.numSurfaces;
            numcomps = 1U;
            break;
        case YUV420P_8bit:
        case YUV420P_10bit:
        case YUV420P_12bit:
        case YUV420P_16bit:
            xScalePtr = &SurfDescTable_YUV[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_420].widthFactor[0];
            yScalePtr = &SurfDescTable_YUV[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_420].heightFactor[0];
            numSurfaces = SurfDescTable_YUV[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_420].numSurfaces;
            break;
        case YUV422P_8bit:
        case YUV422P_10bit:
        case YUV422P_12bit:
        case YUV422P_16bit:
            xScalePtr = &SurfDescTable_YUV[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_422].widthFactor[0];
            yScalePtr = &SurfDescTable_YUV[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_422].heightFactor[0];
            numSurfaces = SurfDescTable_YUV[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_422].numSurfaces;
            break;
        case YUV444P_8bit:
        case YUV444P_10bit:
        case YUV444P_12bit:
        case YUV444P_16bit:
            xScalePtr = &SurfDescTable_YUV[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_444].widthFactor[0];
            yScalePtr = &SurfDescTable_YUV[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_444].heightFactor[0];
            numSurfaces = SurfDescTable_YUV[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_444].numSurfaces;
            break;
        case A8:
        case A16:
        case RG16:
            xScalePtr = &SurfDescTable_RGBA.widthFactor[0];
            yScalePtr = &SurfDescTable_RGBA.heightFactor[0];
            numSurfaces = SurfDescTable_RGBA.numSurfaces;
            break;
        default:
            LOG_ERR("Unsuported input format type: %u\n", DataFormat);
            return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    switch(DataFormat) {
        case YUV400P_8bit:
            bitsPerPixel = 8U;
            yuvpackedtbl[0] = 1U;
            yuvpackedtbl[0] = yuvpackedtbl[0] * numcomps;
            srcBytesPerPixelPtr = &yuvpackedtbl[0];
            break;
        case YUV400P_10bit:
            bitsPerPixel = 10U;
            yuvpackedtbl[0] = 2U;
            yuvpackedtbl[0] = yuvpackedtbl[0] * numcomps;
            srcBytesPerPixelPtr = &yuvpackedtbl[0];
            break;
        case YUV400P_12bit:
            bitsPerPixel = 12U;
            yuvpackedtbl[0] = 2U;
            yuvpackedtbl[0] = yuvpackedtbl[0] * numcomps;
            srcBytesPerPixelPtr = &yuvpackedtbl[0];
            break;
        case YUV400P_16bit:
            bitsPerPixel = 16U;
            yuvpackedtbl[0] = 2U;
            yuvpackedtbl[0] = yuvpackedtbl[0] * numcomps;
            srcBytesPerPixelPtr = &yuvpackedtbl[0];
            break;
        case YUV420P_8bit:
        case YUV422P_8bit:
        case YUV444P_8bit:
            bitsPerPixel = 8U;
            srcBytesPerPixelPtr = &SurfBytesPerPixelTable_YUV[SURF_DESC_INDEX_PLANAR][0][0];
            break;
        case YUV420P_10bit:
        case YUV422P_10bit:
        case YUV444P_10bit:
            bitsPerPixel = 10U;
            srcBytesPerPixelPtr = &SurfBytesPerPixelTable_YUV[SURF_DESC_INDEX_PLANAR][1][0];
            break;
        case YUV420P_12bit:
        case YUV422P_12bit:
        case YUV444P_12bit:
            bitsPerPixel = 12U;
            srcBytesPerPixelPtr = &SurfBytesPerPixelTable_YUV[SURF_DESC_INDEX_PLANAR][2][0];
            break;
        case YUV420P_16bit:
        case YUV422P_16bit:
        case YUV444P_16bit:
            bitsPerPixel = 16U;
            srcBytesPerPixelPtr = &SurfBytesPerPixelTable_YUV[SURF_DESC_INDEX_PLANAR][4][0];
            break;
        case A8:
            bitsPerPixel = 8U;
            srcBytesPerPixelPtr = &SurfBytesPerPixelTable_Alpha[0][0];
            break;
        case A16:
            bitsPerPixel = 16U;
            srcBytesPerPixelPtr = &SurfBytesPerPixelTable_Alpha[4][0];
            break;
        case RG16:
            bitsPerPixel = 16U;
            srcBytesPerPixelPtr = &SurfBytesPerPixelTable_RG16[0];
            break;
        default:
            LOG_ERR("Unsuported input format type: %u\n", DataFormat);
            return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (0U == bitsPerPixel) {
        LOG_ERR("Failed to deduce bits per pixel");
        return NVMEDIA_STATUS_ERROR;
    }

    err = NvSciBufObjGetAttrList(bufObj, &attrList);
    if(err != NvSciError_Success) {
        LOG_ERR(" NvSciBufObjGetAttrList failed");
        return NVMEDIA_STATUS_ERROR;
    }

    NvSciBufAttrKeyValuePair imgattrs[] = {
        { NvSciBufImageAttrKey_PlaneWidth, NULL, 0 },       /* 0 */
        { NvSciBufImageAttrKey_PlaneHeight, NULL, 0 },      /* 1 */
    };
    err = NvSciBufAttrListGetAttrs(attrList, imgattrs,
                                    sizeof(imgattrs) / sizeof(NvSciBufAttrKeyValuePair));
    if(err != NvSciError_Success) {
        LOG_ERR(" NvSciBufAttrListGetAttrs failed");
        return NVMEDIA_STATUS_ERROR;
    }

    planeWidth = (const uint32_t*)(imgattrs[0].value);
    planeHeight = (const uint32_t*)(imgattrs[1].value);
    lumaWidth = planeWidth[0];
    lumaHeight = planeHeight[0];

    /* Check if requested read width, height are lesser than the width and
       height of the surface - checking only for Luma */
    if((width > lumaWidth) || (height > lumaHeight)) {
        LOG_ERR("%s: Bad parameter %ux%u vs %ux%u\n", __func__,
        width, height, lumaWidth, lumaHeight);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    pBuff = malloc(sizeof(uint8_t*)*MAX_NUM_SURFACES);
    if(!pBuff) {
        LOG_ERR("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    pBuffSizes = malloc(sizeof(uint32_t)*MAX_NUM_SURFACES);
    if(!pBuffSizes) {
        LOG_ERR("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    pBuffPitches = calloc(1, sizeof(uint32_t) * MAX_NUM_SURFACES);
    if(!pBuffPitches) {
        LOG_ERR("%s: out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    surfaceSize = 0;
    imageSize = 0;
    for(i = 0; i < numSurfaces; i++) {
        surfaceSize += (lumaWidth * xScalePtr[i] * lumaHeight * yScalePtr[i] * srcBytesPerPixelPtr[i]);
        imageSize += (width * xScalePtr[i] * height * yScalePtr[i] * srcBytesPerPixelPtr[i]);
        pBuffPitches[i] = (uint32_t)((float)lumaWidth * xScalePtr[i]) * srcBytesPerPixelPtr[i];
    }

    buffer = calloc(1, surfaceSize);
    if(!buffer) {
        LOG_ERR("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    pBuffer = buffer;
    for(i = 0; i < numSurfaces; i++) {
        pBuff[i] = buffer;
        pBuffSizes[i] = (uint32_t)(lumaHeight * yScalePtr[i] * pBuffPitches[i]);
        buffer = buffer + pBuffSizes[i];
    }
    err = NvSciBufObjPutPixels(bufObj, NULL, (const void **)pBuff, pBuffSizes,
            pBuffPitches);
    if (err != NvSciError_Success) {
        LOG_ERR("NvSciBufObjPutPixels failed.");
        status = NVMEDIA_STATUS_ERROR;
        goto done;
    } else {
        status = NVMEDIA_STATUS_OK;
    }

done:
    if(pBuff) {
        free(pBuff);
    }

    if (pBuffSizes) {
        free(pBuffSizes);
    }

    if (pBuffer) {
        free(pBuffer);
    }

    if (pBuffPitches) {
        free(pBuffPitches);
    }

    return status;
}

NvMediaStatus
ReadInput(
    char *fileName,
    uint32_t frameNum,
    uint32_t width,
    uint32_t height,
    NvSciBufObj bufObj,
    ChromaFormat inputFileChromaFormat,
    bool uvOrderFlag,
    uint32_t pixelAlignment)
{
    NvSciError err;
    /* Temporary buffer that stores the data read from file */
    uint8_t **pBuff = NULL;
    uint32_t *pBuffPitches = NULL;
    uint8_t *buffer = NULL;
    uint8_t *pBuffer = NULL;
    uint32_t *pBuffSizes = NULL;
    uint32_t imageSize = 0, surfaceSize = 0;
    uint32_t i, j, k, newk = 0;
    /* Number of surfaces in the ChromaFormat of the input file */
    uint32_t numSurfaces = 1;
    uint32_t lumaWidth, lumaHeight;
    NvMediaStatus status = NVMEDIA_STATUS_ERROR;
    FILE *file = NULL;
    unsigned int count, index;
    /* Extract surface info from NvSciBuf */
    NvSciBufAttrList attrList;
    uint32_t const *planeWidth, *planeHeight;
    float *xScalePtr = NULL, *yScalePtr = NULL;
    unsigned int *srcBytesPerPixelPtr = NULL;
    uint32_t bitsPerPixel = 0U;
    /* Passed outside the function, needs to be static */
    static unsigned int yuvpackedtbl[6] = {1, 0, 0, 0, 0, 0};
    unsigned int numcomps = 1;

    if(!bufObj || !fileName) {
        LOG_ERR("%s: Bad parameter\n", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    /* Assumptions: This function assumes that
       - Input file is in planar format
       - Input format is of type YUV
       - Output surface is in semi-planar format
       - Output surface can be written to (ops on the surface have completed)
       */

    /* TODO: Add support for other formats */
    switch(inputFileChromaFormat) {
        case YUV400P_8bit:
        case YUV400P_10bit:
        case YUV400P_12bit:
        case YUV400P_16bit:
            xScalePtr =  &SurfDescTable_Packed.widthFactor[0];
            yScalePtr = &SurfDescTable_Packed.heightFactor[0];
            numSurfaces = SurfDescTable_Packed.numSurfaces;
            numcomps = 1U;
            break;
        case YUV420P_8bit:
        case YUV420P_10bit:
        case YUV420P_12bit:
        case YUV420P_16bit:
            xScalePtr = &SurfDescTable_YUV[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_420].widthFactor[0];
            yScalePtr = &SurfDescTable_YUV[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_420].heightFactor[0];
            numSurfaces = SurfDescTable_YUV[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_420].numSurfaces;
            break;
        case YUV422P_8bit:
        case YUV422P_10bit:
        case YUV422P_12bit:
        case YUV422P_16bit:
            xScalePtr = &SurfDescTable_YUV[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_422].widthFactor[0];
            yScalePtr = &SurfDescTable_YUV[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_422].heightFactor[0];
            numSurfaces = SurfDescTable_YUV[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_422].numSurfaces;
            break;
        case YUV444P_8bit:
        case YUV444P_10bit:
        case YUV444P_12bit:
        case YUV444P_16bit:
            xScalePtr = &SurfDescTable_YUV[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_444].widthFactor[0];
            yScalePtr = &SurfDescTable_YUV[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_444].heightFactor[0];
            numSurfaces = SurfDescTable_YUV[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_444].numSurfaces;
            break;
        default:
            LOG_ERR("Unsuported input format type: %u\n", inputFileChromaFormat);
            return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    switch(inputFileChromaFormat) {
        case YUV400P_8bit:
            bitsPerPixel = 8U;
            yuvpackedtbl[0] = 1U;
            yuvpackedtbl[0] = yuvpackedtbl[0] * numcomps;
            srcBytesPerPixelPtr = &yuvpackedtbl[0];
            break;
        case YUV400P_10bit:
            bitsPerPixel = 10U;
            yuvpackedtbl[0] = 2U;
            yuvpackedtbl[0] = yuvpackedtbl[0] * numcomps;
            srcBytesPerPixelPtr = &yuvpackedtbl[0];
            break;
        case YUV400P_12bit:
            bitsPerPixel = 12U;
            yuvpackedtbl[0] = 2U;
            yuvpackedtbl[0] = yuvpackedtbl[0] * numcomps;
            srcBytesPerPixelPtr = &yuvpackedtbl[0];
            break;
        case YUV400P_16bit:
            bitsPerPixel = 16U;
            yuvpackedtbl[0] = 2U;
            yuvpackedtbl[0] = yuvpackedtbl[0] * numcomps;
            srcBytesPerPixelPtr = &yuvpackedtbl[0];
            break;
        case YUV420P_8bit:
        case YUV422P_8bit:
        case YUV444P_8bit:
            bitsPerPixel = 8U;
            srcBytesPerPixelPtr = &SurfBytesPerPixelTable_YUV[SURF_DESC_INDEX_PLANAR][0][0];
            break;
        case YUV420P_10bit:
        case YUV422P_10bit:
        case YUV444P_10bit:
            bitsPerPixel = 10U;
            srcBytesPerPixelPtr = &SurfBytesPerPixelTable_YUV[SURF_DESC_INDEX_PLANAR][1][0];
            break;
        case YUV420P_12bit:
        case YUV422P_12bit:
        case YUV444P_12bit:
            bitsPerPixel = 12U;
            srcBytesPerPixelPtr = &SurfBytesPerPixelTable_YUV[SURF_DESC_INDEX_PLANAR][2][0];
            break;
        case YUV420P_16bit:
        case YUV422P_16bit:
        case YUV444P_16bit:
            bitsPerPixel = 16U;
            srcBytesPerPixelPtr = &SurfBytesPerPixelTable_YUV[SURF_DESC_INDEX_PLANAR][4][0];
            break;
        default:
            LOG_ERR("Unsuported input format type: %u\n", inputFileChromaFormat);
            return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (0U == bitsPerPixel) {
        LOG_ERR("Failed to deduce bits per pixel");
        return NVMEDIA_STATUS_ERROR;
    }

    err = NvSciBufObjGetAttrList(bufObj, &attrList);
    if(err != NvSciError_Success) {
        LOG_ERR(" NvSciBufObjGetAttrList failed");
        return NVMEDIA_STATUS_ERROR;
    }

    NvSciBufAttrKeyValuePair imgattrs[] = {
        { NvSciBufImageAttrKey_PlaneWidth, NULL, 0 },       /* 0 */
        { NvSciBufImageAttrKey_PlaneHeight, NULL, 0 },      /* 1 */
    };
    err = NvSciBufAttrListGetAttrs(attrList, imgattrs,
                                    sizeof(imgattrs) / sizeof(NvSciBufAttrKeyValuePair));
    if(err != NvSciError_Success) {
        LOG_ERR(" NvSciBufAttrListGetAttrs failed");
        return NVMEDIA_STATUS_ERROR;
    }

    planeWidth = (const uint32_t*)(imgattrs[0].value);
    planeHeight = (const uint32_t*)(imgattrs[1].value);
    lumaWidth = planeWidth[0];
    lumaHeight = planeHeight[0];

    /* Check if requested read width, height are lesser than the width and
       height of the surface - checking only for Luma */
    if((width > lumaWidth) || (height > lumaHeight)) {
        LOG_ERR("%s: Bad parameter %ux%u vs %ux%u\n", __func__,
        width, height, lumaWidth, lumaHeight);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    pBuff = malloc(sizeof(uint8_t*)*MAX_NUM_SURFACES);
    if(!pBuff) {
        LOG_ERR("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    pBuffSizes = malloc(sizeof(uint32_t)*MAX_NUM_SURFACES);
    if(!pBuffSizes) {
        LOG_ERR("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    pBuffPitches = calloc(1, sizeof(uint32_t) * MAX_NUM_SURFACES);
    if(!pBuffPitches) {
        LOG_ERR("%s: out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    surfaceSize = 0;
    imageSize = 0;
    for(i = 0; i < numSurfaces; i++) {
        surfaceSize += (lumaWidth * xScalePtr[i] * lumaHeight * yScalePtr[i] * srcBytesPerPixelPtr[i]);
        imageSize += (width * xScalePtr[i] * height * yScalePtr[i] * srcBytesPerPixelPtr[i]);
        pBuffPitches[i] = (uint32_t)((float)lumaWidth * xScalePtr[i]) * srcBytesPerPixelPtr[i];
    }

    buffer = calloc(1, surfaceSize);
    if(!buffer) {
        LOG_ERR("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    pBuffer = buffer;
    memset(buffer, 0x10, surfaceSize);
    for(i = 0; i < numSurfaces; i++) {
        pBuff[i] = buffer;
        if (i) {
            memset(pBuff[i], 0x80, (lumaHeight * yScalePtr[i] * pBuffPitches[i]));
        }
        pBuffSizes[i] = (uint32_t)(lumaHeight * yScalePtr[i] * pBuffPitches[i]);
        buffer = buffer + pBuffSizes[i];
    }

    file = fopen(fileName, "rb");
    if(!file) {
        LOG_ERR("%s: Error opening file: %s\n", __func__, fileName);
        status = NVMEDIA_STATUS_ERROR;
        goto done;
    }

    if(frameNum > 0) {
        if(fseeko(file, frameNum * (off_t)imageSize, SEEK_SET)) {
            LOG_ERR("ReadInput: Error seeking file: %s\n", fileName);
            status = NVMEDIA_STATUS_ERROR;
            goto done;
        }
    }

    for(k = 0; k < numSurfaces; k++) {
        for(j = 0; j < height * yScalePtr[k]; j++) {
            newk = (!uvOrderFlag && k ) ? (numSurfaces - k) : k;
            index = j * pBuffPitches[newk];
            count = width * xScalePtr[newk] * srcBytesPerPixelPtr[newk];
            if (fread(pBuff[newk] + index, count, 1, file) != 1) {
                status = NVMEDIA_STATUS_ERROR;
                LOG_ERR("ReadInput: Error reading file: %s\n", fileName);
                goto done;
            }

            /* TODO: Assuming YUV input */
            if (pixelAlignment == LSB_ALIGNED) {
                uint16_t *psrc = (uint16_t*)(pBuff[newk] + index);
                if (bitsPerPixel > 8U) {
                    for(i = 0; i < count/2; i++) {
                        *(psrc + i) = (*(psrc + i)) << (16 - bitsPerPixel);
                    }
                }
            }
        }
    }

    err = NvSciBufObjPutPixels(bufObj, NULL, (const void **)pBuff, pBuffSizes,
            pBuffPitches);
    if (err != NvSciError_Success) {
        LOG_ERR("NvSciBufObjPutPixels failed.");
        status = NVMEDIA_STATUS_ERROR;
        goto done;
    } else {
        status = NVMEDIA_STATUS_OK;
    }

done:
    if(pBuff) {
        free(pBuff);
    }

    if (pBuffSizes) {
        free(pBuffSizes);
    }

    if (pBuffer) {
        free(pBuffer);
    }

    if (pBuffPitches) {
        free(pBuffPitches);
    }

    if(file) {
        fclose(file);
    }

    return status;
}

static NvMediaStatus
GetSurfParams(
        BufAttrValues *imgAttrValues,
        float **xScalePtr,
        float **yScalePtr,
        unsigned int **bytePerPixel,
        uint32_t *numSurfaces,
        bool forcePlanarOutput)
{
    static unsigned int yuvpackedtbl[6] = {1, 0, 0, 0, 0, 0};

    if (NULL == imgAttrValues)
    {
        LOG_ERR("%s: Invalid input\n", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    /* Non-planar output is not supported */
    if (!forcePlanarOutput)
    {
        LOG_ERR("%s: Non-planar output write is not supported\n", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    /* TODO: Supporting specific color formats for MM engines */
    if ((1U == imgAttrValues->planeCount) &&
            ((imgAttrValues->planeColorFormat[0] == NvSciColor_A8) ||
             (imgAttrValues->planeColorFormat[0] == NvSciColor_Signed_A16) ||
             (imgAttrValues->planeColorFormat[0] == NvSciColor_Signed_R16G16) ||
             (imgAttrValues->planeColorFormat[0] == NvSciColor_B8G8R8A8) ||
             (imgAttrValues->planeColorFormat[0] == NvSciColor_A8B8G8R8))
       )
    {
        *xScalePtr = &SurfDescTable_RGBA.widthFactor[0];
        *yScalePtr = &SurfDescTable_RGBA.heightFactor[0];
        *numSurfaces = SurfDescTable_RGBA.numSurfaces;

        if (imgAttrValues->planeColorFormat[0] == NvSciColor_A8) {
            *bytePerPixel = &SurfBytesPerPixelTable_Alpha[0][0];
        } else if (imgAttrValues->planeColorFormat[0] == NvSciColor_Signed_A16) {
            *bytePerPixel = &SurfBytesPerPixelTable_Alpha[4][0];
        } else if (imgAttrValues->planeColorFormat[0] == NvSciColor_Signed_R16G16) {
            *bytePerPixel = &SurfBytesPerPixelTable_RG16[0];
        } else if ((imgAttrValues->planeColorFormat[0] == NvSciColor_B8G8R8A8) ||
                   (imgAttrValues->planeColorFormat[0] == NvSciColor_A8B8G8R8)) {
            *bytePerPixel = &SurfBytesPerPixelTable_RGBA[0][0];
        } else {
            LOG_ERR("%s: Unsupported color format: %u\n",
                    __func__, imgAttrValues->planeColorFormat[0]);
            return NVMEDIA_STATUS_BAD_PARAMETER;
        }
    }
    /* Rudimentary check for YUV color formats */
    else if ((NvSciColor_Y8 == imgAttrValues->planeColorFormat[0]) ||
            (NvSciColor_Y10 == imgAttrValues->planeColorFormat[0]) ||
            (NvSciColor_Y12 == imgAttrValues->planeColorFormat[0]) ||
            (NvSciColor_Y16 == imgAttrValues->planeColorFormat[0]))
    {
        uint8_t subSampling = 0U;
        uint8_t subSamplingType = 0U;
        uint8_t bitDepthIndex = 0U;

        if (forcePlanarOutput)
        {
            subSampling = SURF_DESC_INDEX_PLANAR;
        }
        else if (3U == imgAttrValues->planeCount)
        {
            if ((1U == imgAttrValues->planeChannelCount[0]) &&
                (1U == imgAttrValues->planeChannelCount[1]) &&
                (1U == imgAttrValues->planeChannelCount[1]))
            {
                subSampling = SURF_DESC_INDEX_PLANAR;
            }
            else
            {
                LOG_ERR("%s: Unsupported channel count\n", __func__);
                return NVMEDIA_STATUS_BAD_PARAMETER;
            }
        }
        else if (2U == imgAttrValues->planeCount)
        {
            if (2U == imgAttrValues->planeChannelCount[1])
            {
                subSampling = SURF_DESC_INDEX_SEMI_PLANAR;
            }
            else
            {
                LOG_ERR("%s: Unsupported color format. Channel Count: %u\n",
                        __func__, imgAttrValues->planeChannelCount[1]);
                return NVMEDIA_STATUS_BAD_PARAMETER;
            }
        }
        else if (1U == imgAttrValues->planeCount)
        {
            if (1U == imgAttrValues->planeChannelCount[0])
            {
                subSampling = SURF_DESC_INDEX_PACKED;
            }
            else
            {
                LOG_ERR("%s: Unsupported color format. Channel Count: %u\n",
                        __func__, imgAttrValues->planeChannelCount[0]);
                return NVMEDIA_STATUS_BAD_PARAMETER;
            }
        }
        else
        {
            LOG_ERR("%s: Unsupported color format. Plane Count: %u\n",
                    __func__, imgAttrValues->planeCount);
            return NVMEDIA_STATUS_BAD_PARAMETER;
        }

        /* Handle Luma only packed formats */
        if (1U == imgAttrValues->planeCount)
        {
            *xScalePtr =  &SurfDescTable_Packed.widthFactor[0];
            *yScalePtr = &SurfDescTable_Packed.heightFactor[0];
            *numSurfaces = SurfDescTable_Packed.numSurfaces;

            switch(imgAttrValues->planeColorFormat[0]) {
                case NvSciColor_Y8:
                    yuvpackedtbl[0] = 1U;
                    break;
                case NvSciColor_Y10:
                case NvSciColor_Y12:
                case NvSciColor_Y16:
                    yuvpackedtbl[0] = 2U;
                    break;
                default:
                    LOG_ERR("%s: Unsupported color format for luma only: %u\n",
                            __func__, imgAttrValues->planeColorFormat[0]);
                    return NVMEDIA_STATUS_BAD_PARAMETER;
            }

            yuvpackedtbl[0] = yuvpackedtbl[0] * imgAttrValues->planeChannelCount[0];
            *bytePerPixel = &yuvpackedtbl[0];

            return NVMEDIA_STATUS_OK;
        }

        /* The values are channel specific and as a result should be same for
           planar and semi-planar formats */
        if ((imgAttrValues->planeHeight[0] ==
                (imgAttrValues->planeHeight[1])) &&
            (imgAttrValues->planeWidth[0] ==
                (imgAttrValues->planeWidth[1]))
           )
        {
            subSamplingType = SURF_DESC_INDEX_444;
        }
        else if ((imgAttrValues->planeHeight[0] ==
                    (imgAttrValues->planeHeight[1])) &&
                 (imgAttrValues->planeWidth[0] ==
                    (2U * imgAttrValues->planeWidth[1]))
                )
        {
            subSamplingType = SURF_DESC_INDEX_422;
        }
        else if ((imgAttrValues->planeHeight[0] ==
                    (2U * imgAttrValues->planeHeight[1])) &&
                 (imgAttrValues->planeWidth[0] ==
                    (2U * imgAttrValues->planeWidth[1]))
                )
        {
            subSamplingType = SURF_DESC_INDEX_420;
        }
        else
        {
            LOG_ERR("%s: Unsupported channel count\n", __func__);
            return NVMEDIA_STATUS_BAD_PARAMETER;
        }

        switch(imgAttrValues->planeColorFormat[0]) {
            case NvSciColor_Y8:
                bitDepthIndex = 0U;
                break;
            case NvSciColor_Y10:
                bitDepthIndex = 1U;
                break;
            case NvSciColor_Y12:
                bitDepthIndex = 2U;
                break;
            case NvSciColor_Y16:
                bitDepthIndex = 4U;
                break;
            default:
                LOG_ERR("Unsuported input format type: %u\n",
                        imgAttrValues->planeColorFormat[0]);
                return NVMEDIA_STATUS_BAD_PARAMETER;
        }

        *xScalePtr = &SurfDescTable_YUV[subSampling][subSamplingType].widthFactor[0];
        *yScalePtr = &SurfDescTable_YUV[subSampling][subSamplingType].heightFactor[0];
        *numSurfaces = SurfDescTable_YUV[subSampling][subSamplingType].numSurfaces;
        *bytePerPixel = &SurfBytesPerPixelTable_YUV[subSampling][bitDepthIndex][0];
    }
    else
    {
        LOG_ERR("%s: YUV write is not implemented\n", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    return NVMEDIA_STATUS_OK;
}

static void
FreeBuff(uint8_t **pTempBuffer)
{
    if ((pTempBuffer) && (*pTempBuffer)) {
        free(*pTempBuffer);
        *pTempBuffer = NULL;
    }
}

static NvMediaStatus
AllocCopyToCropBuff(
        NvMediaRect   *srcRect,
        uint8_t       **pCropBuffer,
        BufAttrValues *imgAttrValues,
        uint8_t       *pBuff[],
        uint32_t      pBuffPitches[],
        uint32_t      size[],
        bool          forcePlanarOutput
)
{
    uint8_t *cropBuffer = NULL;
    uint8_t *srcAddr = NULL;
    uint8_t *dstAddr = NULL;
    uint32_t imageSize = 0;
    uint32_t i = 0, k = 0;
    uint32_t numSurfaces = 1U;
    float *xScalePtr = NULL, *yScalePtr = NULL;
    unsigned int *bytePerPixelPtr = NULL;
    unsigned int uHeightSurface = 0U, uWidthSurface = 0U;
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint32_t lineWidth, numRows, startOffset;

    uWidthSurface = imgAttrValues->planeWidth[0];
    uHeightSurface = imgAttrValues->planeHeight[0];

    status = GetSurfParams(imgAttrValues, &xScalePtr, &yScalePtr,
            &bytePerPixelPtr, &numSurfaces, forcePlanarOutput);
    if (NVMEDIA_STATUS_OK != status) {
        LOG_ERR("%s: Failed to get attributes\n", __func__);
        goto error;
    }

    /* Check if src rect is within the boundaries of image */
    if ((srcRect->x0 > uWidthSurface)  ||
        (srcRect->y0 > uHeightSurface) ||
        (srcRect->x1 > uWidthSurface)  ||
        (srcRect->y1 > uHeightSurface) ||
        (srcRect->x0 >= srcRect->x1)   ||
        (srcRect->y0 >= srcRect->y1))
    {
       LOG_ERR("%s: Invalid srcRect parameters. Ignoring srcRect: (%u, %u) (%u, %u)\n",
               __func__, srcRect->x0, srcRect->y0, srcRect->x1, srcRect->y1);
       status = NVMEDIA_STATUS_OK;
       goto error;
    }

    /* Copy only if src and dst dimensions are different */
    if (((uint32_t)(srcRect->x1 - srcRect->x0) != uWidthSurface) ||
        ((uint32_t)(srcRect->y1 - srcRect->y0) != uHeightSurface))
    {
        imageSize = 0;
        for (i = 0; i < numSurfaces; i++) {
            imageSize += 
            (
             ((srcRect->x1 - srcRect->x0) * xScalePtr[i] * bytePerPixelPtr[i]) *
             ((srcRect->y1 - srcRect->y0) * yScalePtr[i])
            );
        }

        /* Allocate destination buffer */
        cropBuffer = calloc(1, imageSize);
        if (!cropBuffer) {
            LOG_ERR("%s: Out of memory\n", __func__);
            status = NVMEDIA_STATUS_OUT_OF_MEMORY;
            goto error;
        }

        dstAddr = cropBuffer;
        for (k = 0; k < numSurfaces; k++) {
            startOffset = 
                (
                 (srcRect->x0 * xScalePtr[k] * bytePerPixelPtr[k]) +
                 (srcRect->y0 * yScalePtr[k] * pBuffPitches[k])
                );
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

            /* Update crop width as pitch */
            pBuffPitches[k] = lineWidth;
        }

        *pCropBuffer = cropBuffer;
    }

    return NVMEDIA_STATUS_OK;

error:

    if (cropBuffer) {
        free(cropBuffer);
    }

    return status;
}

static NvMediaStatus
AllocTempBuff(
        BufAttrValues *imgAttrValues,
        uint8_t       *pTempBuff[],
        uint32_t      pTempBuffSizes[],
        uint32_t      pTempBuffPitches[],
        uint8_t       **pTempBuffer,
        uint32_t      size[],
        bool          forcePlanarOutput
)
{
    uint8_t *buffer = NULL;
    uint8_t *pBuffer = NULL;
    uint32_t imageSize = 0;
    uint32_t i = 0;
    uint32_t numSurfaces = 1U;
    float *xScalePtr = NULL, *yScalePtr = NULL;
    unsigned int *bytePerPixelPtr = NULL;
    unsigned int uHeightSurface = 0U, uWidthSurface = 0U;
    NvMediaStatus status = NVMEDIA_STATUS_ERROR;

    uWidthSurface = imgAttrValues->planeWidth[0];
    uHeightSurface = imgAttrValues->planeHeight[0];

    status = GetSurfParams(imgAttrValues, &xScalePtr, &yScalePtr,
            &bytePerPixelPtr, &numSurfaces, forcePlanarOutput);
    if (NVMEDIA_STATUS_OK != status) {
        LOG_ERR("%s: Failed to get attributes\n", __func__);
        goto error;
    }

    imageSize = 0;
    for(i = 0; i < numSurfaces; i++) {
        size[i] = (uWidthSurface * xScalePtr[i] * uHeightSurface * yScalePtr[i] * bytePerPixelPtr[i]);
        imageSize += size[i];
        pTempBuffPitches[i] = (uint32_t)((float)uWidthSurface * xScalePtr[i]) * bytePerPixelPtr[i];
    }

    buffer = calloc(1, imageSize);
    if(!buffer) {
        LOG_ERR("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto error;
    }

    pBuffer = buffer;
    memset(buffer, 0xFF, imageSize);
    for(i = 0; i < numSurfaces; i++) {
        pTempBuff[i] = buffer;
        pTempBuffSizes[i] = (uint32_t)(uHeightSurface * yScalePtr[i] * pTempBuffPitches[i]);
        buffer = buffer + pTempBuffSizes[i];
    }

    *pTempBuffer = pBuffer;

    return NVMEDIA_STATUS_OK;

error:

    if (pBuffer) {
        free(pBuffer);
    }

    return status;
}

NvMediaStatus
WriteOutput(
    char *filename,
    NvSciBufObj bufObj,
    bool uvOrderFlag,
    bool appendFlag,
    NvMediaRect *srcRect)
{
    uint8_t *pBuff[6] = {0};
    uint32_t pBuffSizes[6] = {0};
    uint8_t *pBuffer = NULL;
    uint8_t *pCropBuffer = NULL;
    uint32_t pBuffPitches[6] = {0};
    uint32_t size[6] ={0};
    uint32_t k = 0, newk = 0;
    uint32_t numSurfaces = 1U;
    unsigned int *bytePerPixelPtr = NULL;
    float *xScalePtr = NULL, *yScalePtr = NULL;
    FILE *file = NULL;
    NvMediaStatus status = NVMEDIA_STATUS_ERROR;
    NvSciError err = NvSciError_Success;
    NvSciBufAttrList attrList = {0};
    BufAttrValues imgAttrValues = {0};
    bool isPlanarOutput = true;

    if(!bufObj || !filename) {
        LOG_ERR("%s: Bad parameter\n", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    /* Assumptions: This function assumes that
       - Output surface can be written to (ops on the surface have completed)
       */

    file = fopen(filename, appendFlag ? "ab" : "wb");
    if(!file) {
        LOG_ERR("%s: file open failed: %s\n", __func__, filename);
        return NVMEDIA_STATUS_ERROR;
    }

    err = NvSciBufObjGetAttrList(bufObj, &attrList);
    if(err != NvSciError_Success) {
        LOG_ERR(" NvSciBufObjGetAttrList failed");
        return NVMEDIA_STATUS_ERROR;
    }

    status = GetNvSciBufAttributes(attrList, &imgAttrValues);
    if (NVMEDIA_STATUS_OK != status) {
        LOG_ERR("%s: Failed to get attributes\n", __func__);
        goto done;
    }

    status = GetSurfParams(&imgAttrValues, &xScalePtr, &yScalePtr,
            &bytePerPixelPtr, &numSurfaces, isPlanarOutput);
    if (NVMEDIA_STATUS_OK != status) {
        LOG_ERR("%s: Failed to get Surface Params\n", __func__);
        goto done;
    }

    status = AllocTempBuff(&imgAttrValues, pBuff, pBuffSizes, pBuffPitches,
                &pBuffer, size, isPlanarOutput);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: Failed to allocate temp buffer %u\n", __func__, status);
        goto done;
    }

    err = NvSciBufObjGetPixels(bufObj, NULL, (void **)&pBuff, pBuffSizes,
            pBuffPitches);
    if (err != NvSciError_Success) {
        LOG_ERR("NvSciBufObjGetPixels failed.");
        status = NVMEDIA_STATUS_ERROR;
        goto done;
    } else {
        status = NVMEDIA_STATUS_OK;
    }

    if (srcRect) {
        status = AllocCopyToCropBuff(srcRect, &pCropBuffer, &imgAttrValues, pBuff,
                pBuffPitches, size, isPlanarOutput);
        if(status != NVMEDIA_STATUS_OK) {
            LOG_ERR("%s: Failed to allocate temp buffer %u\n", __func__, status);
            goto done;
        }
    }

    for(k = 0; k < numSurfaces; k++) {
       newk = (!uvOrderFlag && k ) ? (numSurfaces - k) : k;
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

    FreeBuff(&pBuffer);
    FreeBuff(&pCropBuffer);

    return status;
}

NvMediaStatus
GetNvSciBufObjCrcNoSrcRect(
    NvSciBufObj bufObj,
    bool monochromeFlag,
    uint32_t *crcOut)
{
    uint8_t *pBuff[6] = {0};
    uint32_t pBuffSizes[6] = {0};
    uint8_t *pBuffer = NULL;
    uint8_t *bufferTmp = NULL;
    uint32_t pBuffPitches[6] = {0};
    uint32_t crc = 0;
    uint32_t size[6] ={0};
    float *xScalePtr = NULL, *yScalePtr = NULL;
    unsigned int *bytePerPixelPtr = NULL;
    uint32_t numSurfaces = 1U;
    NvSciError err = NvSciError_Success;
    NvMediaStatus status = NVMEDIA_STATUS_ERROR;
    NvSciBufAttrList attrList = {0};
    BufAttrValues imgAttrValues = {0};
    bool isPlanarOutput = true;

    if(!bufObj || !crcOut) {
        LOG_ERR("%s: Invalid Input argument passed\n", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    err = NvSciBufObjGetAttrList(bufObj, &attrList);
    if(err != NvSciError_Success) {
        LOG_ERR(" NvSciBufObjGetAttrList failed");
        return NVMEDIA_STATUS_ERROR;
    }

    status = GetNvSciBufAttributes(attrList, &imgAttrValues);
    if (NVMEDIA_STATUS_OK != status) {
        LOG_ERR("%s: Failed to get attributes\n", __func__);
        goto done;
    }

    status = GetSurfParams(&imgAttrValues, &xScalePtr, &yScalePtr,
            &bytePerPixelPtr, &numSurfaces, isPlanarOutput);
    if (NVMEDIA_STATUS_OK != status) {
        LOG_ERR("%s: Failed to get Surface Params\n", __func__);
        goto done;
    }

    status = AllocTempBuff(&imgAttrValues, pBuff, pBuffSizes, pBuffPitches,
                &pBuffer, size, isPlanarOutput);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: Failed to allocate temp buffer %u\n", __func__, status);
        goto done;
    }

    err = NvSciBufObjGetPixels(bufObj, NULL, (void **)&pBuff, pBuffSizes,
            pBuffPitches);
    if (err != NvSciError_Success) {
        LOG_ERR("NvSciBufObjGetPixels failed.");
        status = NVMEDIA_STATUS_ERROR;
        goto done;
    } else {
        status = NVMEDIA_STATUS_OK;
    }

    for(uint32_t i = 0; i < numSurfaces; i++) {
        if (monochromeFlag && i) {
            break;
        }
        bufferTmp = pBuff[i];
        uint32_t lines = size[i]/pBuffPitches[i];
        uint32_t widthInBytes = pBuffPitches[i];
        while(lines--) {
            crc = CalculateBufferCRC(widthInBytes, crc, bufferTmp);
            bufferTmp += pBuffPitches[i];
        }
    }

    *crcOut = crc;

done:

    FreeBuff(&pBuffer);
    return status;
}

NvMediaStatus
GetNvSciBufObjCrc(
    NvSciBufObj bufObj,
    NvMediaRect *srcRect,
    bool monochromeFlag,
    uint32_t *crcOut)
{
    uint8_t *pBuff[6] = {0};
    uint32_t pBuffSizes[6] = {0};
    uint8_t *pBuffer = NULL;
    uint8_t *bufferTmp = NULL;
    uint8_t *pCropBuffer = NULL;
    uint32_t pBuffPitches[6] = {0};
    uint32_t crc = 0;
    uint32_t i = 0, lines;
    uint32_t size[6] ={0};
    float *xScalePtr = NULL, *yScalePtr = NULL;
    unsigned int *bytePerPixelPtr = NULL;
    uint32_t numSurfaces = 1U;
    NvSciError err = NvSciError_Success;
    NvMediaStatus status = NVMEDIA_STATUS_ERROR;
    NvSciBufAttrList attrList = {0};
    BufAttrValues imgAttrValues = {0};
    uint32_t width = srcRect->x1;
    uint32_t width_temp;
    uint32_t height = srcRect->y1;
    bool isPlanarOutput = true;

    if(!bufObj || !crcOut) {
        LOG_ERR("%s: Invalid Input argument passed\n", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    err = NvSciBufObjGetAttrList(bufObj, &attrList);
    if(err != NvSciError_Success) {
        LOG_ERR(" NvSciBufObjGetAttrList failed");
        return NVMEDIA_STATUS_ERROR;
    }

    status = GetNvSciBufAttributes(attrList, &imgAttrValues);
    if (NVMEDIA_STATUS_OK != status) {
        LOG_ERR("%s: Failed to get attributes\n", __func__);
        goto done;
    }

    status = GetSurfParams(&imgAttrValues, &xScalePtr, &yScalePtr,
            &bytePerPixelPtr, &numSurfaces, isPlanarOutput);
    if (NVMEDIA_STATUS_OK != status) {
        LOG_ERR("%s: Failed to get Surface Params\n", __func__);
        goto done;
    }

    status = AllocTempBuff(&imgAttrValues, pBuff, pBuffSizes, pBuffPitches,
                &pBuffer, size, isPlanarOutput);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: Failed to allocate temp buffer %u\n", __func__, status);
        goto done;
    }

    err = NvSciBufObjGetPixels(bufObj, NULL, (void **)&pBuff, pBuffSizes,
            pBuffPitches);
    if (err != NvSciError_Success) {
        LOG_ERR("NvSciBufObjGetPixels failed.");
        status = NVMEDIA_STATUS_ERROR;
        goto done;
    } else {
        status = NVMEDIA_STATUS_OK;
    }

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

    FreeBuff(&pBuffer);
    FreeBuff(&pCropBuffer);

    return status;
}

NvMediaStatus
CheckSurfaceCrc(
    NvSciBufObj surf,
    NvMediaRect *srcRect,
    bool        monochromeFlag,
    uint32_t    ref,
    bool        *isMatching)
{
    uint32_t crc = 0;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    status = GetNvSciBufObjCrc(surf,
                               srcRect,
                               monochromeFlag,
                               &crc);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("CheckSurfaceCrc: GetSurfaceCrc failed\n");
        return NVMEDIA_STATUS_ERROR;
    }
    if(crc != ref) {
        LOG_WARN("CheckSurfaceCrc: Encountered CRC mismatch.\n");
        LOG_WARN("CheckSurfaceCrc: Calculated CRC: %8x (%d). Expected CRC: %8x (%d).\n", crc, crc, ref, ref);
        *isMatching = false;
    } else {
        *isMatching = true;
    }

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
PopulateNvSciBufAttrList (
    ChromaFormat                    chromaFormat,
    uint32_t                        width,
    uint32_t                        height,
    bool                            needCpuAccess,
    NvSciBufAttrValImageLayoutType  layout,
    uint32_t                        planeCount,
    NvSciBufAttrValAccessPerm       access_perm,
    uint32_t                        lumaBaseAddressAlign,
    NvSciBufAttrValColorStd         lumaColorStd,
    NvSciBufAttrValImageScanType    scanType,
    NvSciBufAttrList                bufAttributeList
)
{
    NvSciError err;
    NvSciBufAttrValColorFmt colorFormat[NV_SCI_BUF_IMAGE_MAX_PLANES] = {0};
    NvSciBufAttrValColorStd colorStd[NV_SCI_BUF_IMAGE_MAX_PLANES] = {0};
    uint32_t planeWidth[NV_SCI_BUF_IMAGE_MAX_PLANES] = {0};
    uint32_t planeHeight[NV_SCI_BUF_IMAGE_MAX_PLANES] = {0};
    uint32_t baseAddrAlign[NV_SCI_BUF_IMAGE_MAX_PLANES] = {0};
    uint64_t padding[NV_SCI_BUF_IMAGE_MAX_PLANES] = {0};
    bool vprFlag = false;
    bool isYUV = true;
    bool isPlanar = false;

    /* TODO: Add support for more planes/formats */
    /* Set color format */
    switch (chromaFormat)
    {
        /* Semi-planar formats */
        case YUV420SP_8bit:
        case YUV422SP_8bit:
        case YUV444SP_8bit:
            colorFormat[0]  = NvSciColor_Y8;
            colorFormat[1]  = NvSciColor_V8U8;
            break;
        case YUV420SP_10bit:
        case YUV422SP_10bit:
        case YUV444SP_10bit:
            colorFormat[0] = NvSciColor_Y10;
            colorFormat[1] = NvSciColor_V10U10;
            break;
        case YUV420SP_12bit:
        case YUV422SP_12bit:
        case YUV444SP_12bit:
            colorFormat[0] = NvSciColor_Y12;
            colorFormat[1] = NvSciColor_V12U12;
            break;
        case YUV420SP_16bit:
        case YUV422SP_16bit:
        case YUV444SP_16bit:
            colorFormat[0] = NvSciColor_Y16;
            colorFormat[1] = NvSciColor_V16U16;
            break;

        /* Planar formats */
        case YUV400P_8bit:
            colorFormat[0] = NvSciColor_Y8;
            isPlanar = true;
            break;
        case YUV400P_10bit:
            colorFormat[0] = NvSciColor_Y10;
            isPlanar = true;
            break;
        case YUV400P_12bit:
            colorFormat[0] = NvSciColor_Y12;
            isPlanar = true;
            break;
        case YUV400P_16bit:
            colorFormat[0] = NvSciColor_Y16;
            isPlanar = true;
            break;
        case YUV420P_8bit:
        case YUV422P_8bit:
        case YUV444P_8bit:
            colorFormat[0] = NvSciColor_Y8;
            colorFormat[1] = NvSciColor_U8;
            colorFormat[2] = NvSciColor_V8;
            isPlanar = true;
            break;
        case YUV420P_10bit:
        case YUV422P_10bit:
        case YUV444P_10bit:
            colorFormat[0] = NvSciColor_Y10;
            colorFormat[1] = NvSciColor_U10;
            colorFormat[2] = NvSciColor_V10;
            isPlanar = true;
            break;
        case YUV420P_12bit:
        case YUV422P_12bit:
        case YUV444P_12bit:
            colorFormat[0] = NvSciColor_Y12;
            colorFormat[1] = NvSciColor_U12;
            colorFormat[2] = NvSciColor_V12;
            isPlanar = true;
            break;
        case YUV420P_16bit:
        case YUV422P_16bit:
        case YUV444P_16bit:
            colorFormat[0] = NvSciColor_Y16;
            colorFormat[1] = NvSciColor_U16;
            colorFormat[2] = NvSciColor_V16;
            isPlanar = true;
            break;

        /* Packed formats */
        case RGBA_8bit:
            colorFormat[0] = NvSciColor_A8B8G8R8;
            isYUV = false;
            isPlanar = true;
            break;
        case ARGB_8bit:
            colorFormat[0] = NvSciColor_B8G8R8A8;
            isYUV = false;
            isPlanar = true;
            break;
        case RG16:
            colorFormat[0] = NvSciColor_Signed_R16G16;
            isYUV = false;
            isPlanar = true;
            break;
        case A16:
            colorFormat[0] = NvSciColor_Signed_A16;
            isYUV = false;
            isPlanar = false;
            break;
        case A8:
            colorFormat[0] = NvSciColor_A8;
            isYUV = false;
            isPlanar = false;
            break;
        default:
            LOG_ERR("%s: Unsupported color format: %u\n",
                    __func__, planeCount);
            return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    /* Set image dimensions */
    planeWidth[0]   = width;
    planeHeight[0]  = height;
    if (planeCount > 1)
    {
    switch (chromaFormat)
    {
        /* Planar formats */
        case YUV420P_8bit:
        case YUV420P_10bit:
        case YUV420P_12bit:
        case YUV420P_16bit:
            planeWidth[1]  = width >> 1;
            planeHeight[1] = height >> 1;
            planeWidth[2] = planeWidth[1];
            planeHeight[2] = planeHeight[1];
            break;
        case YUV422P_8bit:
        case YUV422P_10bit:
        case YUV422P_12bit:
        case YUV422P_16bit:
            planeWidth[1]   = width >> 1;
            planeHeight[1]  = height;
            planeWidth[2] = planeWidth[1];
            planeHeight[2] = planeHeight[1];
            break;
            break;
        case YUV444P_8bit:
        case YUV444P_10bit:
        case YUV444P_12bit:
        case YUV444P_16bit:
            planeWidth[1]   = width;
            planeHeight[1]  = height;
            planeWidth[2] = planeWidth[1];
            planeHeight[2] = planeHeight[1];
            break;

        /* Semi-planar formats */
        /* Note: Plane resolution for chroma is per channel (Cb or Cr) and not
           combined resolution */
        case YUV420SP_8bit:
        case YUV420SP_10bit:
        case YUV420SP_12bit:
        case YUV420SP_16bit:
            planeWidth[1]   = width >> 1;
            planeHeight[1]  = height >> 1;
            break;
        case YUV422SP_8bit:
        case YUV422SP_10bit:
        case YUV422SP_12bit:
        case YUV422SP_16bit:
            planeWidth[1]   = width >> 1;
            planeHeight[1]  = height;
            break;
        case YUV444SP_8bit:
        case YUV444SP_10bit:
        case YUV444SP_12bit:
        case YUV444SP_16bit:
            planeWidth[1]   = width;
            planeHeight[1]  = height;
            break;

        case YUV400P_8bit:
        case YUV400P_10bit:
        case YUV400P_12bit:
        case YUV400P_16bit:
        case RGBA_8bit:
        case ARGB_8bit:
            /* Single plane format */
            break;
        default:
            LOG_ERR("%s: Unsupported color format: %u\n",
                    __func__, chromaFormat);
            return NVMEDIA_STATUS_BAD_PARAMETER;
    }
    }

    colorStd[0] = lumaColorStd;
    colorStd[1] = lumaColorStd;
    colorStd[2] = lumaColorStd;
    baseAddrAlign[0] = lumaBaseAddressAlign;
    baseAddrAlign[1] = lumaBaseAddressAlign;
    baseAddrAlign[2] = lumaBaseAddressAlign;

    /* Set buffer type */
    NvSciBufType bufType = NvSciBufType_Image;

    LOG_DBG("%s: Color Format:%u isYUV:%u isPlanar:%u planeCount:%u\n",
            __func__, chromaFormat, isYUV, isPlanar, planeCount);

    for (uint32_t i = 0; i < planeCount; i++) {
        LOG_DBG("%d: Color:%u %u Addr:%u W:%u H:%u\n",
        i, colorFormat[i], colorStd[i], baseAddrAlign[i],
        planeWidth[i], planeHeight[i]);
    }

    /* Set all key-value pairs */
    NvSciBufAttrKeyValuePair attributes[] = {
        {NvSciBufGeneralAttrKey_RequiredPerm, &access_perm, sizeof(access_perm)},
        {NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType)},
        {NvSciBufGeneralAttrKey_NeedCpuAccess, &needCpuAccess, sizeof(needCpuAccess)},
        {NvSciBufGeneralAttrKey_EnableCpuCache, &needCpuAccess, sizeof(needCpuAccess)},
        {NvSciBufImageAttrKey_TopPadding, &padding, planeCount * sizeof(padding[0])},
        {NvSciBufImageAttrKey_BottomPadding, &padding, planeCount * sizeof(padding[0])},
        {NvSciBufImageAttrKey_LeftPadding, &padding, planeCount * sizeof(padding[0])},
        {NvSciBufImageAttrKey_RightPadding, &padding, planeCount * sizeof(padding[0])},
        {NvSciBufImageAttrKey_Layout, &layout, sizeof(layout)},
        {NvSciBufImageAttrKey_PlaneCount, &planeCount, sizeof(planeCount)},
        {NvSciBufImageAttrKey_PlaneColorFormat, &colorFormat, planeCount * sizeof(NvSciBufAttrValColorFmt)},
        {NvSciBufImageAttrKey_PlaneColorStd, &colorStd, planeCount * sizeof(NvSciBufAttrValColorStd)},
        {NvSciBufImageAttrKey_PlaneBaseAddrAlign, &baseAddrAlign, planeCount * sizeof(uint32_t)},
        {NvSciBufImageAttrKey_PlaneWidth, &planeWidth, planeCount * sizeof(uint32_t)},
        {NvSciBufImageAttrKey_PlaneHeight, &planeHeight, planeCount * sizeof(uint32_t)},
        {NvSciBufImageAttrKey_VprFlag, &vprFlag, sizeof(vprFlag)},
        {NvSciBufImageAttrKey_ScanType, &scanType, sizeof(NvSciBufAttrValImageScanType)}
    };

    err = NvSciBufAttrListSetAttrs(bufAttributeList, attributes,
            sizeof(attributes)/sizeof(NvSciBufAttrKeyValuePair));
    if (err != NvSciError_Success) {
        LOG_ERR("%s: SetAttr for input frame failed: %u\n", __func__, err);
        return NVMEDIA_STATUS_ERROR;
    }

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
GetNvSciBufAttributes (
    NvSciBufAttrList bufAttrList,
    BufAttrValues *imgAttrValues)
{
    NvSciError err = NvSciError_Success;

    NvSciBufAttrKeyValuePair imgAttrList[] = {
        { NvSciBufImageAttrKey_Layout, NULL, 0 },               /* 0 */
        { NvSciBufImageAttrKey_PlaneCount, NULL, 0 },           /* 1 */
        { NvSciBufImageAttrKey_PlaneWidth, NULL, 0 },           /* 2 */
        { NvSciBufImageAttrKey_PlaneHeight, NULL, 0 },          /* 3 */
        { NvSciBufImageAttrKey_PlaneColorFormat, NULL, 0 },     /* 4 */
        { NvSciBufImageAttrKey_PlaneBitsPerPixel, NULL, 0 },    /* 5 */
        { NvSciBufImageAttrKey_PlaneChannelCount, NULL, 0 },    /* 6 */
        { NvSciBufImageAttrKey_PlanePitch, NULL, 0 },           /* 7 */
        { NvSciBufGeneralAttrKey_NeedCpuAccess, NULL, 0},       /* 8 */
    };

    /* TODO: Check if it is safe to get array size like this */
    err = NvSciBufAttrListGetAttrs(bufAttrList, imgAttrList,
            sizeof(imgAttrList) / sizeof(NvSciBufAttrKeyValuePair));
    if (err != NvSciError_Success) {
        LOG_ERR("%s: NvSciBufAttrListGetAttrs failed: %u\n", __func__, err);
        return NVMEDIA_STATUS_ERROR;
    }

    imgAttrValues->layout =
        *((const NvSciBufAttrValImageLayoutType*)(imgAttrList[0].value));
    imgAttrValues->planeCount =
        *((const uint32_t*)(imgAttrList[1].value));
    memcpy(&imgAttrValues->planeWidth, imgAttrList[2].value,
            sizeof(uint32_t) * imgAttrValues->planeCount);
    memcpy(&imgAttrValues->planeHeight, imgAttrList[3].value,
            sizeof(uint32_t) * imgAttrValues->planeCount);
    memcpy(&imgAttrValues->planeColorFormat, imgAttrList[4].value,
            sizeof(NvSciBufAttrValColorFmt) * imgAttrValues->planeCount);
    memcpy(&imgAttrValues->planeBitsPerPixel, imgAttrList[5].value,
            sizeof(uint32_t) * imgAttrValues->planeCount);
    memcpy(&imgAttrValues->planeChannelCount, imgAttrList[6].value,
            sizeof(uint8_t) * imgAttrValues->planeCount);
    memcpy(&imgAttrValues->planePitchBytes, imgAttrList[7].value,
            sizeof(uint32_t) * imgAttrValues->planeCount);
    imgAttrValues->needCpuAccess =
        *((const bool*)(imgAttrList[8].value));

    return NVMEDIA_STATUS_OK;
}
