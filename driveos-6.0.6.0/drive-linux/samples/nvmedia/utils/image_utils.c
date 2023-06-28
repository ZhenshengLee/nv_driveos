/*
 * Copyright (c) 2019 NVIDIA Corporation.  All rights reserved.
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

#include "image_utils.h"
#include "log_utils.h"
#include "misc_utils.h"
#include "nvmedia_surface.h"

#define MAXM_NUM_SURFACES 6

typedef struct {
    float heightFactor[6];
    float widthFactor[6];
    unsigned int numSurfaces;
} ImgUtilSurfParams;

ImgUtilSurfParams ImgSurfParamsTable_RGBA  = {
    .heightFactor = {1, 0, 0, 0, 0, 0},
    .widthFactor = {1, 0, 0, 0, 0, 0},
    .numSurfaces = 1,
};

ImgUtilSurfParams ImgSurfParamsTable_RAW  = {
    .heightFactor = {1, 0, 0, 0, 0, 0},
    .widthFactor = {1, 0, 0, 0, 0, 0},
    .numSurfaces = 1,
};

ImgUtilSurfParams ImgSurfParamsTable_YUV[][4] = {
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

ImgUtilSurfParams ImgSurfParamsTable_Packed  = {
    .heightFactor = {1, 0, 0, 0, 0, 0},
    .widthFactor = {1, 0, 0, 0, 0, 0},
    .numSurfaces = 1,
};


unsigned int ImgBytesPerPixelTable_RGBA[][6] = {
    {4, 0, 0, 0, 0, 0}, /* 8 */
};

unsigned int ImgBytesPerPixelTable_RGBA16[][6] = {
    {8, 0, 0, 0, 0, 0}, /* 16 */
};

unsigned int ImgBytesPerPixelTable_RG16[6] =
    {4, 0, 0, 0, 0, 0};

unsigned int ImgBytesPerPixelTable_Alpha[][6] = {
    {1, 0, 0, 0, 0, 0}, /* 8 */
    {2, 0, 0, 0, 0, 0}, /* 10 */
    {2, 0, 0, 0, 0, 0}, /* 12 */
    {2, 0, 0, 0, 0, 0}, /* 14 */
    {2, 0, 0, 0, 0, 0}, /* 16 */
    {4, 0, 0, 0, 0, 0}, /* 32 */
};

unsigned int ImgBytesPerPixelTable_RAW[][6] = {
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

unsigned int ImgBytesPerPixelTable_YUV[][9][6] = {
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

                xScalePtr =  &ImgSurfParamsTable_Packed.widthFactor[0];
                yScalePtr = &ImgSurfParamsTable_Packed.heightFactor[0];
                numSurfaces = ImgSurfParamsTable_Packed.numSurfaces;

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
                xScalePtr = &ImgSurfParamsTable_YUV[0][surfSubSamplingType - NVM_SURF_ATTR_SUB_SAMPLING_TYPE_420].widthFactor[0];
                yScalePtr = &ImgSurfParamsTable_YUV[0][surfSubSamplingType - NVM_SURF_ATTR_SUB_SAMPLING_TYPE_420].heightFactor[0];
                numSurfaces = ImgSurfParamsTable_YUV[0][surfSubSamplingType - NVM_SURF_ATTR_SUB_SAMPLING_TYPE_420].numSurfaces;
                bytePerPixelPtr = &ImgBytesPerPixelTable_YUV[0][surfBPC - NVM_SURF_ATTR_BITS_PER_COMPONENT_8][0];
            }

            break;
        case NVM_SURF_ATTR_SURF_TYPE_RGBA:
            if (surfCompOrder == NVM_SURF_ATTR_COMPONENT_ORDER_ALPHA) {
                bytePerPixelPtr = &ImgBytesPerPixelTable_Alpha[surfBPC - NVM_SURF_ATTR_BITS_PER_COMPONENT_8][0];
            } else if (surfCompOrder == NVM_SURF_ATTR_COMPONENT_ORDER_RG) {
                if(surfBPC == NVM_SURF_ATTR_BITS_PER_COMPONENT_16) {
                    bytePerPixelPtr = &ImgBytesPerPixelTable_RG16[0];
                } else {
                    LOG_ERR("Invalid RGorder & Bitspercomp combination.Only RG16 is supported\n");
                    return NVMEDIA_STATUS_ERROR;
                }
            } else { /* RGBA, ARGB, BGRA */
                if (surfBPC == NVM_SURF_ATTR_BITS_PER_COMPONENT_16) {
                    bytePerPixelPtr = &ImgBytesPerPixelTable_RGBA16[0][0];
                } else if (surfBPC == NVM_SURF_ATTR_BITS_PER_COMPONENT_8) {
                    bytePerPixelPtr = &ImgBytesPerPixelTable_RGBA[0][0];
                } else {
                    LOG_ERR("RGBA orders with 8 and 16bits only is supported \n");
                    return NVMEDIA_STATUS_ERROR;
                }
            }
            xScalePtr = &ImgSurfParamsTable_RGBA.widthFactor[0];
            yScalePtr = &ImgSurfParamsTable_RGBA.heightFactor[0];
            numSurfaces =  ImgSurfParamsTable_RGBA.numSurfaces;
            break;
        case NVM_SURF_ATTR_SURF_TYPE_RAW:
            bytePerPixelPtr = &ImgBytesPerPixelTable_RAW[surfBPC - NVM_SURF_ATTR_BITS_PER_COMPONENT_8][0];
            xScalePtr = &ImgSurfParamsTable_RAW.widthFactor[0];
            yScalePtr = &ImgSurfParamsTable_RAW.heightFactor[0];
            numSurfaces =  ImgSurfParamsTable_RAW.numSurfaces;
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


NvMediaStatus
CreateMemRGBASurf(
    unsigned int width,
    unsigned int height,
    NvMediaBool initflag,
    unsigned int initvalue,
    MemSurf **surf_out)
{
    MemSurf *surf;
    unsigned int *p;
    unsigned int pixels;

    surf = malloc(sizeof(MemSurf));
    if(!surf) {
        LOG_ERR("CreateMemRGBASurf: Cannot allocate surface structure\n");
        return NVMEDIA_STATUS_OUT_OF_MEMORY;
    }

    memset(surf, 0, sizeof(MemSurf));
    surf->pSurf = malloc(width * height * 4);
    if(!surf->pSurf) {
        free(surf);
        LOG_ERR("CreateMemRGBASurf: Cannot allocate surface\n");
        return NVMEDIA_STATUS_OUT_OF_MEMORY;
    }
    surf->width = width;
    surf->height = height;
    surf->pitch = width * 4;
    surf->bpp = 4;

    pixels = width * height;

    if(initflag) {
        p = (unsigned int *)(void *)surf->pSurf;
        while(pixels--) {
            *p++ = initvalue;
        }
    }
    *surf_out = surf;
    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
DestroyMemSurf(
    MemSurf *surf)
{
    if(surf) {
        if(surf->pSurf)
            free(surf->pSurf);
        free(surf);
    }
    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
DrawRGBARect(
    MemSurf *surf,
    NvMediaRect *rect,
    uint8_t R,
    uint8_t G,
    uint8_t B,
    uint8_t A)
{
    uint32_t color;
    uint32_t lines;
    uint32_t pixels;
    uint8_t *pPixelBase;

    if(!surf || !rect)
        return NVMEDIA_STATUS_BAD_PARAMETER;

    color = PACK_RGBA(R, G, B, A);
    lines = rect->y1 - rect->y0;
    pixels = rect->x1 - rect->x0;
    pPixelBase = surf->pSurf + rect->y0 * surf->pitch + rect->x0 * 4;

    while(lines--) {
        uint32_t i;
        uint32_t *pPixel = (uint32_t *)(void *)pPixelBase;
        for(i = 0; i < pixels; i++) {
            *pPixel++ = color;
        }

        pPixelBase += surf->pitch;
    }
    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
PreMultiplyRGBASurf(
    MemSurf *surf)
{
    uint32_t pixels;
    uint8_t *p;
    uint8_t alpha;

    if(!surf)
        return NVMEDIA_STATUS_BAD_PARAMETER;

    pixels = surf->width * surf->height;
    p = (uint8_t *)(void *)surf->pSurf;

    while(pixels--) {
        alpha = *(p + 3);
        *p = ((uint16_t)*p * (uint16_t)alpha + 128) >> 8;
        p++;
        *p = ((uint16_t)*p * (uint16_t)alpha + 128) >> 8;
        p++;
        *p = ((uint16_t)*p * (uint16_t)alpha + 128) >> 8;
        p++;
        p++;
    }
    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
CreateMemI8Surf(
    uint32_t width,
    uint32_t height,
    uint8_t init,
    MemSurf **surf_out)
{
    MemSurf *surf;

    surf = malloc(sizeof(MemSurf));
    if(!surf) {
        LOG_ERR("CreateMemI8Surf: Cannot allocate surface structure\n");
        return NVMEDIA_STATUS_OUT_OF_MEMORY;
    }

    memset(surf, 0, sizeof(MemSurf));
    surf->pSurf = malloc(width * height);
    if(!surf->pSurf) {
        free(surf);
        LOG_ERR("CreateMemI8Surf: Cannot allocate surface\n");
        return NVMEDIA_STATUS_OUT_OF_MEMORY;
    }
    surf->width = width;
    surf->height = height;
    surf->pitch = width;
    surf->bpp = 1;

    memset(surf->pSurf, init, width * height);

    *surf_out = surf;
    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
DrawI8Rect(
    MemSurf *surf,
    NvMediaRect *rect,
    uint8_t index)
{
    uint32_t lines, pixels;
    uint8_t *pPixelBase;

    if(!surf || !rect)
        return NVMEDIA_STATUS_BAD_PARAMETER;

    lines = rect->y1 - rect->y0;
    pixels = rect->x1 - rect->x0;
    pPixelBase = surf->pSurf + rect->y0 * surf->pitch + rect->x0;

    while(lines--) {
        memset(pPixelBase, index, pixels);
        pPixelBase += surf->pitch;
    }
    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
ReadRGBA(
    char *filename,
    unsigned int width,
    unsigned int height,
    MemSurf *rgbaSurface)
{
    FILE *file;

    if(!rgbaSurface || !rgbaSurface->pSurf)
        return NVMEDIA_STATUS_BAD_PARAMETER;

    file = fopen(filename, "rb");
    if(!file) {
        LOG_ERR("ReadRGBA: Failed opening file %s\n", filename);
        return NVMEDIA_STATUS_ERROR;
    }

    if(fread(rgbaSurface->pSurf, width * height * 4, 1, file) != 1) {
        LOG_ERR("ReadRGBA: Failed reading file %s\n", filename);
        return NVMEDIA_STATUS_ERROR;
    }

    fclose(file);

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
WriteRGBA(
    char *filename,
    uint32_t outputBpp,
    uint8_t defaultAplha,
    MemSurf *rgbaSurface)
{
    uint32_t width = rgbaSurface->width;
    uint32_t height = rgbaSurface->height;
    NvMediaStatus ret = NVMEDIA_STATUS_ERROR;
    uint8_t *lineBuff = NULL;
    uint32_t i;
    FILE *f = fopen(filename, "wb");
    if(!f) {
        LOG_ERR("WriteRGBA: Cannot create file: %s", filename);
        goto WriteRGBA_end;
    }

    lineBuff = malloc(width * outputBpp);
    if(!lineBuff) {
        LOG_ERR("WriteRGBA: Error allocating line buffer");
        goto WriteRGBA_end;
    }

    for(i = 0; i < height; i++) {
        uint32_t j;
        uint8_t *srcBuff = rgbaSurface->pSurf + i * rgbaSurface->pitch;
        uint8_t *dstBuff = lineBuff;

        for(j = 0; j < width; j++) {
            dstBuff[0] = srcBuff[0]; // R
            dstBuff[1] = srcBuff[1]; // G
            dstBuff[2] = srcBuff[2]; // B
            if(outputBpp == 4) {
                dstBuff[3] = rgbaSurface->bpp == 3 ? defaultAplha : srcBuff[3];
            }
            srcBuff += rgbaSurface->bpp;
            dstBuff += outputBpp;
        }
        if(fwrite(lineBuff, width * outputBpp, 1, f) != 1) {
            LOG_ERR("WriteRGBA: Error writing file: %s", filename);
            goto WriteRGBA_end;
        }
    }

    ret = NVMEDIA_STATUS_OK;
WriteRGBA_end:
    if(lineBuff)
        free(lineBuff);
    if(f)
        fclose(f);

    return ret;
}


NvMediaStatus
GetPPMFileDimensions(
    char *fileName,
    uint16_t *uWidth,
    uint16_t *uHeight)
{
    uint32_t uFileWidth;
    uint32_t uFileHeight;
    char buf[256], *t;
    FILE *file;

    file = fopen(fileName, "rb");
    if(!file) {
        LOG_ERR("GetPPMFileDimensions: Error opening file: %s\n", fileName);
        return NVMEDIA_STATUS_ERROR;
    }

    t = fgets(buf, 256, file);
    if(!t || strncmp(buf, "P6\n", 3)) {
        LOG_ERR("GetPPMFileDimensions: Invalid PPM header in file: %s\n", fileName);
        return NVMEDIA_STATUS_ERROR;
    }
    do {
        t = fgets(buf, 255, file);
        if(!t) {
            LOG_ERR("GetPPMFileDimensions: Invalid PPM header in file: %s\n", fileName);
            return NVMEDIA_STATUS_ERROR;
        }
    } while(!strncmp(buf, "#", 1));
    if(sscanf(buf, "%u %u", &uFileWidth, &uFileHeight) != 2) {
        LOG_ERR("GetPPMFileDimensions: Error getting PPM file resolution for the file: %s\n", fileName);
        return NVMEDIA_STATUS_ERROR;
    }
    if(uFileWidth < 16 || uFileWidth > 2048 ||
            uFileHeight < 16 || uFileHeight > 2048) {
        LOG_ERR("GetPPMFileDimensions: Invalid PPM file resolution: %ux%u in file: %s\n", uFileWidth, uFileHeight, fileName);
        return NVMEDIA_STATUS_ERROR;
    }

    fclose(file);

    *uWidth  = uFileWidth;
    *uHeight = uFileHeight;

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
ReadPPM(
    char *fileName,
    uint8_t defaultAplha,
    MemSurf *rgbaSurface)
{
    uint32_t surfaceWidth;
    uint32_t surfaceHeight;
    uint32_t data = 0;
    char buf[256], *t;
    uint8_t *lineBuff = NULL;
    uint32_t i;
    NvMediaStatus ret = NVMEDIA_STATUS_ERROR;

    LOG_DBG("ReadPPM: Start - File: %s", fileName);

    FILE *f = fopen(fileName, "rb");
    if(!f) {
        LOG_ERR("ReadPPM: Error opening file: %s", fileName);
        goto ReadPPM_end;
    }

    t = fgets(buf, 256, f);
    if(!t || strncmp(buf, "P6\n", 3)) {
        LOG_ERR("ReadPPM: Invalid PPM header: %s", fileName);
        goto ReadPPM_end;
    }
    do {
        t = fgets(buf, 255, f);
        if(!t) {
            LOG_ERR("ReadPPM: Invalid PPM header: %s", fileName);
            goto ReadPPM_end;
        }
    } while(!strncmp(buf, "#", 1));
    if(sscanf(buf, "%u %u %u\n", &surfaceWidth, &surfaceHeight, &data) != 3) {
        LOG_ERR("ReadPPM: Error getting PPM file resolution - file: %s string: %s", fileName, buf);
        goto ReadPPM_end;
    }
    if(data != 255) {
        LOG_ERR("ReadPPM: Invalid PPM header (data: %u) resolution: %dx%d string: %s", data, surfaceWidth, surfaceHeight, buf);
        goto ReadPPM_end;
    }

    lineBuff = malloc(surfaceWidth * 3);
    if(!lineBuff) {
        LOG_ERR("ReadPPM: Error allocating line buffer");
        goto ReadPPM_end;
    }

    for(i = 0; i < surfaceHeight; i++) {
        uint32_t j;
        uint8_t *srcBuff = lineBuff;
        uint8_t *dstBuff = rgbaSurface->pSurf + i * rgbaSurface->pitch;

        if(fread(lineBuff, surfaceWidth * 3, 1, f) != 1) {
            LOG_ERR("ReadPPM: Error reading file: %s", fileName);
            goto ReadPPM_end;
        }
        for(j = 0; j < surfaceWidth; j++) {
            dstBuff[0] = srcBuff[0]; // R
            dstBuff[1] = srcBuff[1]; // G
            dstBuff[2] = srcBuff[2]; // B
            if(rgbaSurface->bpp == 4)
                dstBuff[3] = defaultAplha;
            srcBuff += 3;
            dstBuff += rgbaSurface->bpp;
        }
    }

    ret = NVMEDIA_STATUS_OK;
ReadPPM_end:
    if(lineBuff)
        free(lineBuff);
    if(f)
        fclose(f);

    LOG_DBG("ReadPPM: End");

    return ret;
}

NvMediaStatus
WritePPM(
    char *fileName,
    MemSurf *rgbaSurface)
{
    uint8_t *lineBuff = NULL;
    uint32_t i;
    char header[256];
    NvMediaStatus ret = NVMEDIA_STATUS_ERROR;

    LOG_DBG("WritePPM: Start - File: %s", fileName);

    FILE *f = fopen(fileName, "wb");
    if(!f) {
        LOG_ERR("WritePPM: Error opening file: %s", fileName);
        goto WritePPM_end;
    }

    sprintf(header, "P6\n# NVIDIA\n%u %u %u\n", rgbaSurface->width, rgbaSurface->height, 255);

    if(fwrite(header, strlen(header), 1, f) != 1) {
        LOG_ERR("WritePPM: Error writing PPM file header: %s", fileName);
        goto WritePPM_end;
    }

    lineBuff = malloc(rgbaSurface->width * 3);
    if(!lineBuff) {
        LOG_ERR("WritePPM: Error allocating line buffer");
        goto WritePPM_end;
    }

    for(i = 0; i < rgbaSurface->height; i++) {
        uint32_t j;
        uint8_t *srcBuff = rgbaSurface->pSurf + i * rgbaSurface->pitch;
        uint8_t *dstBuff = lineBuff;

        for(j = 0; j < rgbaSurface->width; j++) {
            dstBuff[0] = srcBuff[0]; // R
            dstBuff[1] = srcBuff[1]; // G
            dstBuff[2] = srcBuff[2]; // B
            srcBuff += rgbaSurface->bpp;
            dstBuff += 3;
        }
        if(fwrite(lineBuff, rgbaSurface->width * 3, 1, f) != 1) {
            LOG_ERR("WritePPM: Error writing file: %s", fileName);
            goto WritePPM_end;
        }
    }

    ret = NVMEDIA_STATUS_OK;
WritePPM_end:
    if(lineBuff)
        free(lineBuff);
    if(f)
        fclose(f);

    LOG_DBG("WritePPM: End");

    return ret;
}


NvMediaStatus
ReadPAL(
    char *filename,
    uint32_t *palette)
{
    NvMediaStatus ret = NVMEDIA_STATUS_ERROR;

    FILE *f = fopen(filename, "rb");
    if(!f) {
        LOG_ERR("ReadPAL: File: %s does not exist", filename);
        goto ReadPAL_end;
    }

    if(fread(palette, 256 * 4, 1, f) != 1) {
        LOG_ERR("ReadPAL: Error reading file: %s", filename);
        goto ReadPAL_end;
    }

    ret = NVMEDIA_STATUS_OK;
ReadPAL_end:
    if(f)
        fclose(f);

    return ret;
}

NvMediaStatus
ReadI8(
    char *filename,
    MemSurf *dstSurface)
{
    NvMediaStatus ret = NVMEDIA_STATUS_ERROR;
    uint32_t i;
    uint8_t *dst;

    FILE *f = fopen(filename, "rb");
    if(!f) {
        LOG_ERR("ReadI8: File: %s does not exist", filename);
        goto ReadI8_end;
    }

    dst = dstSurface->pSurf;
    for(i = 0; i < dstSurface->height; i++) {
        if(fread(dst, dstSurface->width, 1, f) != 1) {
            LOG_ERR("ReadI8: Error reading file: %s", filename);
            goto ReadI8_end;
        }
        dst += dstSurface->pitch;
    }

    ret = NVMEDIA_STATUS_OK;
ReadI8_end:
    if(f)
        fclose(f);

    return ret;
}

static NvMediaStatus
GetImageCrcNew(
    NvMediaImage *image,
    uint32_t width,
    uint32_t height,
    uint32_t *crcOut,
    uint32_t rawBytesPerPixel)
{
    NvMediaImageSurfaceMap surfaceMap;
    uint32_t lines, crc = 0;
    uint32_t uHeightSurface, uWidthSurface, imageSize;
    NvMediaStatus status;
    uint8_t **pBuff = NULL;
    uint32_t *pBuffPitches = NULL;
    uint32_t *pBuffWidthInBytes = NULL;
    uint8_t *bufferImg = NULL;
    uint8_t *bufferTmp = NULL;
    uint32_t i = 0;
    float *xScalePtr = NULL, *yScalePtr = NULL;
    unsigned int *bytePerPixelPtr = NULL;
    uint32_t numSurfaces = 1;
    NVM_SURF_FMT_DEFINE_ATTR(surfFormatAttrs);

    if(!image || !crcOut) {
        LOG_ERR("%s: Bad parameter\n", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    status = NvMediaImageLock(image, NVMEDIA_IMAGE_ACCESS_WRITE, &surfaceMap);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: NvMediaImageLock() failed\n", __func__);
        return status;
    }
    NvMediaImageUnlock(image);

    uHeightSurface = surfaceMap.height;
    uWidthSurface  = surfaceMap.width;

    status = NvMediaSurfaceFormatGetAttrs(image->type,surfFormatAttrs,NVM_SURF_FMT_ATTR_MAX);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: NvMediaSurfaceFormatGetAttrs() failed\n",__func__);
        return status;
    }
    for(i=0; i < NVM_SURF_FMT_ATTR_MAX; i++){
        if(surfFormatAttrs[i].type == NVM_SURF_ATTR_SURF_TYPE){
            if(surfFormatAttrs[i].value == NVM_SURF_ATTR_SURF_TYPE_RAW) {
                uHeightSurface = uHeightSurface +
                                 ((image->embeddedDataTopSize + image->embeddedDataBottomSize)/
                                  (width * rawBytesPerPixel));
                break;
            }
        }
    }

    if(width > uWidthSurface || height > uHeightSurface) {
        LOG_ERR("%s: Bad parameter\n", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
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

    status = GetSurfParams(image->type,
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
        imageSize += (uWidthSurface * xScalePtr[i] * uHeightSurface * yScalePtr[i] * bytePerPixelPtr[i]);
        pBuffWidthInBytes[i] = (uint32_t)((float)width * xScalePtr[i]) * bytePerPixelPtr[i];
        pBuffPitches[i] = (uint32_t)((float)uWidthSurface * xScalePtr[i]) * bytePerPixelPtr[i];
    }

    bufferImg = calloc(1, imageSize);
    if(!bufferImg) {
        LOG_ERR("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    memset(bufferImg,0xFF,imageSize);
    bufferTmp = bufferImg;
    for(i = 0; i < numSurfaces; i++) {
        pBuff[i] = bufferTmp;
        bufferTmp = bufferTmp + (uint32_t)(uHeightSurface * yScalePtr[i] * pBuffPitches[i]);
    }

    status = NvMediaImageLock(image, NVMEDIA_IMAGE_ACCESS_WRITE, &surfaceMap);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: NvMediaImageLock() failed\n", __func__);
        goto done;
    }

    status = NvMediaImageGetBits(image, NULL, (void **)pBuff, pBuffPitches);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: NvMediaImageGetBits failed\n", __func__);
        goto done;
    }
    NvMediaImageUnlock(image);

    height = height + ((image->embeddedDataTopSize + image->embeddedDataBottomSize)/(width * rawBytesPerPixel));
    for(i = 0; i < numSurfaces; i++) {
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

    if(bufferImg) {
        free(bufferImg);
    }

    return status;
}

NvMediaStatus
GetImageCrc(
    NvMediaImage *image,
    uint32_t width,
    uint32_t height,
    uint32_t *crcOut,
    uint32_t rawBytesPerPixel)
{
    NvMediaStatus status;
    NVM_SURF_FMT_DEFINE_ATTR(srcAttr);

    if(!image || !crcOut) {
        LOG_ERR("GetImageCrc: Bad parameter\n");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    status = NvMediaSurfaceFormatGetAttrs(image->type,
                                          srcAttr,
                                          NVM_SURF_FMT_ATTR_MAX);
    if (status == NVMEDIA_STATUS_OK) {
        return GetImageCrcNew(
                            image,
                            width,
                            height,
                            crcOut,
                            rawBytesPerPixel);
    } else {
        LOG_ERR("%s:NvMediaSurfaceFormatGetAttrs failed\n", __func__);
        return status;
    }
}

NvMediaStatus
CheckImageCrc(
    NvMediaImage *image,
    uint32_t width,
    uint32_t height,
    uint32_t ref,
    NvMediaBool *isMatching,
    uint32_t rawBytesPerPixel)
{
    uint32_t crc = 0;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    status = GetImageCrc(image, width, height, &crc, rawBytesPerPixel);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("CheckImageCrc: GetImageCrc failed\n");
        return NVMEDIA_STATUS_ERROR;
    }

    if(crc != ref) {
        LOG_WARN("CheckImageCrc: Encountered CRC mismatch.\n");
        LOG_WARN("CheckImageCrc: Calculated CRC: %8x (%d). Expected CRC: %8x (%d).\n", crc, crc, ref, ref);
        *isMatching = NVMEDIA_FALSE;
    } else {
        *isMatching = NVMEDIA_TRUE;
    }

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
CheckImageOutput(
    NvMediaImage *image,
    uint32_t width,
    uint32_t height,
    uint8_t *SrcBuf,
    NvMediaBool *isMatching,
    uint32_t rawBytesPerPixel)
{
    NvMediaImageSurfaceMap surfaceMap;
    uint32_t lines;
    uint32_t uHeightSurface, uWidthSurface, imageSize;
    NvMediaStatus status;
    uint8_t **pBuff = NULL;
    uint32_t *pBuffPitches = NULL;
    uint32_t *pBuffWidthInBytes = NULL;
    uint8_t *bufferImg = NULL;
    uint8_t *bufferTmp = NULL;
    uint32_t i = 0;
    float *xScalePtr = NULL, *yScalePtr = NULL;
    unsigned int *bytePerPixelPtr = NULL;
    uint32_t numSurfaces = 1;
    *isMatching = NVMEDIA_TRUE;

    if(!image || !SrcBuf) {
        LOG_ERR("%s: Bad parameter\n", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }
    NVM_SURF_FMT_DEFINE_ATTR(surfFormatAttrs);

    status = NvMediaImageLock(image, NVMEDIA_IMAGE_ACCESS_READ, &surfaceMap);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: NvMediaImageLock() failed\n", __func__);
        return status;
    }

    uHeightSurface = surfaceMap.height;
    uWidthSurface  = surfaceMap.width;

    status = NvMediaSurfaceFormatGetAttrs(image->type,surfFormatAttrs,NVM_SURF_FMT_ATTR_MAX);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: NvMediaSurfaceFormatGetAttrs() failed\n",__func__);
        return status;
    }
    for(i=0; i < NVM_SURF_FMT_ATTR_MAX; i++){
        if(surfFormatAttrs[i].type == NVM_SURF_ATTR_SURF_TYPE){
            if(surfFormatAttrs[i].value == NVM_SURF_ATTR_SURF_TYPE_RAW) {
                uHeightSurface = uHeightSurface +
                                 ((image->embeddedDataTopSize + image->embeddedDataBottomSize)/
                                  (width * rawBytesPerPixel));
                break;
            }
        }
    }

    if(width > uWidthSurface || height > uHeightSurface) {
        LOG_ERR("%s: Bad parameter\n", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
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

    status = GetSurfParams(image->type,
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
        imageSize += (uWidthSurface * xScalePtr[i] * uHeightSurface * yScalePtr[i] * bytePerPixelPtr[i]);
        pBuffWidthInBytes[i] = (uint32_t)((float)width * xScalePtr[i]) * bytePerPixelPtr[i];
        pBuffPitches[i] = (uint32_t)((float)uWidthSurface * xScalePtr[i]) * bytePerPixelPtr[i];
    }

    bufferImg = calloc(1, imageSize);
    if(!bufferImg) {
        LOG_ERR("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    memset(bufferImg,0xFF,imageSize);
    bufferTmp = bufferImg;
    for(i = 0; i < numSurfaces; i++) {
        pBuff[i] = bufferTmp;
        bufferTmp = bufferTmp + (uint32_t)(uHeightSurface * yScalePtr[i] * pBuffPitches[i]);
    }

    status = NvMediaImageGetBits(image, NULL, (void **)pBuff, pBuffPitches);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: NvMediaImageGetBits failed\n", __func__);
        goto done;
    }
    NvMediaImageUnlock(image);

    height = height + ((image->embeddedDataTopSize + image->embeddedDataBottomSize)/(width * rawBytesPerPixel));
    for(i = 0; i < numSurfaces; i++) {
        lines = height*yScalePtr[i];
        bufferTmp = pBuff[i];
        while(lines--) {
            if (memcmp(SrcBuf, bufferTmp, pBuffWidthInBytes[i]) != 0)
            {
                *isMatching = NVMEDIA_FALSE;
                goto done;
            }
            SrcBuf += pBuffPitches[i];
            bufferTmp += pBuffPitches[i];
        }
    }


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

    if(bufferImg) {
        free(bufferImg);
    }

    return status;
}


NvMediaStatus
WriteRAWImageToRGBA(
    char *filename,
    NvMediaImage *image,
    NvMediaBool appendFlag,
    uint32_t bytesPerPixel)
{
    NvMediaStatus status = NVMEDIA_STATUS_ERROR;
    NvMediaImageSurfaceMap surfaceMap;
    uint32_t width, height;
    FILE *file = NULL;
    uint32_t *rgbaBuffer;
    uint32_t i, j;
    uint8_t *rawBuffer;
    uint16_t *evenLine;
    uint16_t *oddLine;

    if(!image || !filename) {
        LOG_ERR("WriteRAWImageToRGBA: Bad parameter\n");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if(NvMediaImageLock(image, NVMEDIA_IMAGE_ACCESS_WRITE, &surfaceMap) != NVMEDIA_STATUS_OK) {
        LOG_ERR("WriteRAWImageToRGBA: NvMediaImageLock failed\n");
        return NVMEDIA_STATUS_ERROR;
    }

    height = surfaceMap.height;
    width  = surfaceMap.width;

    if(!(file = fopen(filename, appendFlag ? "ab" : "wb"))) {
        LOG_ERR("WriteRAWImageToRGBA: file open failed: %s\n", filename);
        perror(NULL);
        return NVMEDIA_STATUS_ERROR;
    }

    if(!(rgbaBuffer = calloc(1, width * height * 4))) {
        LOG_ERR("WriteRAWImageToRGBA: Out of memory\n");
        fclose(file);
        return NVMEDIA_STATUS_OUT_OF_MEMORY;
    }

    rawBuffer = (uint8_t *)surfaceMap.surface[0].mapping;

    for(j = 0; j < height - 2; j = j + 2) {
        evenLine = (uint16_t *)rawBuffer;
        oddLine = (uint16_t *)(rawBuffer + surfaceMap.surface[0].pitch);
        for(i = 0; i < width - 2; i += 2) {
            rgbaBuffer[i + j * width] = ((uint32_t)(oddLine[i + 1] >> 6)) |              // R
                                         ((uint32_t)(evenLine[i + 1] >> 6) << 8) |       // G
                                         ((uint32_t)(evenLine[i] >> 6) << 16) |          // B
                                         0xFF000000;
            rgbaBuffer[i + j * width + 1] = ((uint32_t)(oddLine[i + 1] >> 6)) |          // R
                                             ((uint32_t)(evenLine[i + 1] >> 6) << 8) |   // G
                                             ((uint32_t)(evenLine[i] >> 6) << 16) |      // B
                                             0xFF000000;
            rgbaBuffer[i + (j + 1) * width] = ((uint32_t)(oddLine[i + 1] >> 6)) |         // R
                                               ((uint32_t)(oddLine[i] >> 6) << 8) |       // G
                                               ((uint32_t)(evenLine[i] >> 6) << 16) |     // B
                                               0xFF000000;
            rgbaBuffer[i + (j + 1) * width + 1] = ((uint32_t)(oddLine[i + 1] >> 6)) |      // R
                                                   ((uint32_t)(oddLine[i] >> 6) << 8) |    // G
                                                   ((uint32_t)(evenLine[i] >> 6) << 16) |  // B
                                                   0xFF000000;
        }
        rawBuffer += surfaceMap.surface[0].pitch * 2;
    }

    if(fwrite(rgbaBuffer, width * height * 4, 1, file) != 1) {
        LOG_ERR("WriteRAWImageToRGBA: file write failed\n");
        goto done;
    }

    status = NVMEDIA_STATUS_OK;

done:
    fclose(file);

    NvMediaImageUnlock(image);

    if(rgbaBuffer)
        free(rgbaBuffer);

    return status;
}

static NvMediaStatus
WriteImageNew(
    char *filename,
    NvMediaImage *image,
    NvMediaBool uvOrderFlag,
    NvMediaBool appendFlag,
    uint32_t bytesPerPixel,
    NvMediaRect *srcRect)
{
    uint8_t **pBuff = NULL;
    uint32_t *pBuffPitches = NULL;
    uint32_t imageSize = 0;
    unsigned int size[3] ={0};
    uint8_t *buffer = NULL;
    uint8_t *pBuffer = NULL;
    uint32_t i, k, newk = 0;
    float *xScalePtr = NULL, *yScalePtr = NULL;
    unsigned int *bytePerPixelPtr = NULL;
    uint32_t numSurfaces = 1;
    unsigned int uHeightSurface, uWidthSurface;
    NvMediaImageSurfaceMap surfaceMap;
    NvMediaStatus status = NVMEDIA_STATUS_ERROR;
    FILE *file = NULL;
    uint8_t *cropBuffer = NULL, *srcAddr, *dstAddr;
    uint32_t lineWidth, numRows, startOffset;

    if(!image || !filename) {
        LOG_ERR("%s: Bad parameter\n", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    status = NvMediaImageLock(image, NVMEDIA_IMAGE_ACCESS_WRITE, &surfaceMap);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: NvMediaImageLock() failed\n", __func__);
        return status;
    }
    NvMediaImageUnlock(image);

    file = fopen(filename, appendFlag ? "ab" : "wb");
    if(!file) {
        LOG_ERR("%s: file open failed: %s\n", __func__, filename);
        return NVMEDIA_STATUS_ERROR;
    }

    pBuff = malloc(sizeof(uint8_t*)*MAXM_NUM_SURFACES);
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

    uHeightSurface = surfaceMap.height;
    uWidthSurface  = surfaceMap.width;

    status = GetSurfParams(image->type,
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
        size[i] = (uWidthSurface * xScalePtr[i] * uHeightSurface * yScalePtr[i] * bytePerPixelPtr[i]);
        imageSize += size[i];
        pBuffPitches[i] = (uint32_t)((float)uWidthSurface * xScalePtr[i]) * bytePerPixelPtr[i];
    }

    // Embedded data size needs to be included for RAW surftype
    size[0] += image->embeddedDataTopSize;
    size[0] += image->embeddedDataBottomSize;
    imageSize += image->embeddedDataTopSize;
    imageSize += image->embeddedDataBottomSize;

    buffer = calloc(1, imageSize);
    if(!buffer) {
        LOG_ERR("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    pBuffer = buffer;
    memset(buffer,0xFF,imageSize);
    for(i = 0; i < numSurfaces; i++) {
        pBuff[i] = buffer;
        buffer = buffer + (uint32_t)(uHeightSurface * yScalePtr[i] * pBuffPitches[i]);
    }

    status = NvMediaImageLock(image, NVMEDIA_IMAGE_ACCESS_WRITE, &surfaceMap);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: NvMediaImageLock() failed\n", __func__);
        goto done;
    }
    status = NvMediaImageGetBits(image, NULL, (void **)pBuff, pBuffPitches);
    NvMediaImageUnlock(image);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: NvMediaVideoSurfaceGetBits() failed %x\n", __func__);
        goto done;
    }

    /* Check if src rect is within the boundaries of image */
    if (srcRect) {
        if ((srcRect->x0 > surfaceMap.width) ||
           (srcRect->y0 > surfaceMap.height) ||
           (srcRect->x1 > surfaceMap.width) ||
           (srcRect->y1 > surfaceMap.height) ||
           (srcRect->x0 >= srcRect->x1) ||
           (srcRect->y0 >= srcRect->y1)) {
            LOG_ERR("%s: Invalid srcRect parameters. Ignoring srcRect..\n", __func__);
        }
        else if (((uint32_t)(srcRect->x1 - srcRect->x0) != surfaceMap.width) ||
                 ((uint32_t)(srcRect->y1 - srcRect->y0) != surfaceMap.height)) {
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
       newk = (!uvOrderFlag && k ) ? (numSurfaces - k) : k;
       if (fwrite(pBuff[newk],size[newk],1,file) != 1) {
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

    if (pBuffPitches) {
        free(pBuffPitches);
    }

    if (pBuffer) {
        free(pBuffer);
    }

    if (cropBuffer) {
        free (cropBuffer);
    }

    return status;
}

NvMediaStatus
WriteImage(
    char *filename,
    NvMediaImage *image,
    NvMediaBool uvOrderFlag,
    NvMediaBool appendFlag,
    uint32_t bytesPerPixel,
    NvMediaRect *srcRect)
{
    NvMediaStatus status;
    NVM_SURF_FMT_DEFINE_ATTR(srcAttr);

    if(!image || !filename) {
        LOG_ERR("WriteImage: Bad parameter\n");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    status = NvMediaSurfaceFormatGetAttrs(image->type,
                                          srcAttr,
                                          NVM_SURF_FMT_ATTR_MAX);
    if (status == NVMEDIA_STATUS_OK) {
        return WriteImageNew(filename,
                       image,
                       uvOrderFlag,
                       appendFlag,
                       bytesPerPixel,
                       srcRect);
    } else {
        LOG_ERR("%s:NvMediaSurfaceFormatGetAttrs failed\n", __func__);
        return status;
    }
}

static NvMediaStatus
ReadImageNew(
    char *fileName,
    uint32_t frameNum,
    uint32_t width,
    uint32_t height,
    NvMediaImage *image,
    NvMediaBool uvOrderFlag,
    uint32_t bytesPerPixel,
    uint32_t pixelAlignment)
{
    uint8_t **pBuff = NULL;
    uint32_t *pBuffPitches = NULL;
    uint32_t imageSize = 0,surfaceSize = 0;
    uint8_t *buffer = NULL;
    uint8_t *pBuffer = NULL;
    uint32_t i, j, k, newk = 0;
    float *xScalePtr = NULL, *yScalePtr = NULL;
    unsigned int *bytePerPixelPtr = NULL;
    uint32_t numSurfaces = 1;
    unsigned int uHeightSurface, uWidthSurface;
    NvMediaImageSurfaceMap surfaceMap;
    NvMediaStatus status = NVMEDIA_STATUS_ERROR;
    FILE *file = NULL;
    unsigned int count, index;
    NVM_SURF_FMT_DEFINE_ATTR(srcAttr);
    unsigned int surfType, surfBPC;

    if(!image || !fileName) {
        LOG_ERR("%s: Bad parameter\n", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    status = NvMediaImageLock(image, NVMEDIA_IMAGE_ACCESS_WRITE, &surfaceMap);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: NvMediaImageLock() failed\n", __func__);
        return status;
    }
    NvMediaImageUnlock(image);

    uHeightSurface = surfaceMap.height;
    uWidthSurface  = surfaceMap.width;

    if(width > uWidthSurface || height > uHeightSurface) {
        LOG_ERR("%s: Bad parameter\n", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    pBuff = malloc(sizeof(uint8_t*)*MAXM_NUM_SURFACES);
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

    status = GetSurfParams(image->type,
                           &xScalePtr,
                           &yScalePtr,
                           &bytePerPixelPtr,
                           &numSurfaces);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: GetSurfParams() failed\n", __func__);
        goto done;
    }

    status = NvMediaSurfaceFormatGetAttrs(image->type,
                                          srcAttr,
                                          NVM_SURF_FMT_ATTR_MAX);
    if (status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s:NvMediaSurfaceFormatGetAttrs failed\n", __func__);
        goto done;
    }
    surfType = srcAttr[NVM_SURF_ATTR_SURF_TYPE].value;
    surfBPC = srcAttr[NVM_SURF_ATTR_BITS_PER_COMPONENT].value;

    surfaceSize = 0;
    imageSize = 0;
    for(i = 0; i < numSurfaces; i++) {
        surfaceSize += (uWidthSurface * xScalePtr[i] * uHeightSurface * yScalePtr[i] * bytePerPixelPtr[i]);
        imageSize += (width * xScalePtr[i] * height * yScalePtr[i] * bytePerPixelPtr[i]);
        pBuffPitches[i] = (uint32_t)((float)uWidthSurface * xScalePtr[i]) * bytePerPixelPtr[i];
    }

    buffer = calloc(1, surfaceSize);
    if(!buffer) {
        LOG_ERR("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    pBuffer = buffer;
    memset(buffer,0x10,surfaceSize);
    for(i = 0; i < numSurfaces; i++) {
        pBuff[i] = buffer;
        if (i) {
            memset(pBuff[i], 0x80, (uHeightSurface * yScalePtr[i] * pBuffPitches[i]));
        }
        buffer = buffer + (uint32_t)(uHeightSurface * yScalePtr[i] * pBuffPitches[i]);
    }

    file = fopen(fileName, "rb");
    if(!file) {
        LOG_ERR("%s: Error opening file: %s\n", __func__, fileName);
        status = NVMEDIA_STATUS_ERROR;
        goto done;
    }

    if(frameNum > 0) {
        if(fseeko(file, frameNum * (off_t)imageSize, SEEK_SET)) {
            LOG_ERR("ReadImage: Error seeking file: %s\n", fileName);
            status = NVMEDIA_STATUS_ERROR;
            goto done;
        }
    }

    if((surfType == NVM_SURF_ATTR_SURF_TYPE_RGBA ) && strstr(fileName, ".png")) {
        LOG_ERR("ReadImage: Does not support png format\n");
        status = NVMEDIA_STATUS_ERROR;
        goto done;
    }
    for(k = 0; k < numSurfaces; k++) {
        for(j = 0; j < height*yScalePtr[k]; j++) {
            newk = (!uvOrderFlag && k ) ? (numSurfaces - k) : k;
            index = j * pBuffPitches[newk];
            count = width * xScalePtr[newk] * bytePerPixelPtr[newk];
            if (fread(pBuff[newk] + index, count, 1, file) != 1) {
                status = NVMEDIA_STATUS_ERROR;
                LOG_ERR("ReadImage: Error reading file: %s\n", fileName);
                goto done;
            }
            if((surfType == NVM_SURF_ATTR_SURF_TYPE_YUV) && (pixelAlignment == LSB_ALIGNED)) {
                uint16_t *psrc = (uint16_t*)(pBuff[newk] + index);
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

    status = NvMediaImageLock(image, NVMEDIA_IMAGE_ACCESS_WRITE, &surfaceMap);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: NvMediaImageLock() failed\n", __func__);
        goto done;
    }
    status = NvMediaImagePutBits(image, NULL, (void **)pBuff, pBuffPitches);
    NvMediaImageUnlock(image);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: Failed to put bits\n", __func__);
    }

done:
    if(pBuff) {
        free(pBuff);
    }

    if (pBuffPitches) {
        free(pBuffPitches);
    }

    if (pBuffer) {
        free(pBuffer);
    }

    if(file) {
        fclose(file);
    }

    return status;
}

NvMediaStatus
ReadImage(
    char *fileName,
    uint32_t frameNum,
    uint32_t width,
    uint32_t height,
    NvMediaImage *image,
    NvMediaBool uvOrderFlag,
    uint32_t bytesPerPixel,
    uint32_t pixelAlignment)
{
    NvMediaStatus status;
    NVM_SURF_FMT_DEFINE_ATTR(srcAttr);

    status = NvMediaSurfaceFormatGetAttrs(image->type,
                                          srcAttr,
                                          NVM_SURF_FMT_ATTR_MAX);
    if (status == NVMEDIA_STATUS_OK) {
        return ReadImageNew(
                        fileName,
                        frameNum,
                        width,
                        height,
                        image,
                        uvOrderFlag,
                        bytesPerPixel,
                        pixelAlignment);
    } else {
        LOG_ERR("%s:NvMediaSurfaceFormatGetAttrs failed\n", __func__);
        return status;
    }

}

NvMediaStatus
InitImage(
    NvMediaImage *image,
    uint32_t width,
    uint32_t height)
{
    uint8_t **pBuff = NULL;
    uint32_t *pBuffPitches = NULL;
    uint32_t imageSize = 0,surfaceSize = 0;
    uint8_t *buffer = NULL;
    uint8_t *pBuffer = NULL;
    float *xScalePtr = NULL, *yScalePtr = NULL;
    unsigned int *bytePerPixelPtr = NULL;
    uint32_t numSurfaces = 1;
    uint32_t i;
    unsigned int uHeightSurface, uWidthSurface;
    NvMediaImageSurfaceMap surfaceMap;
    NvMediaStatus status = NVMEDIA_STATUS_ERROR;
    NVM_SURF_FMT_DEFINE_ATTR(srcAttr);

    if(!image) {
        LOG_ERR("%s: Bad parameter\n", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    status = NvMediaImageLock(image, NVMEDIA_IMAGE_ACCESS_WRITE, &surfaceMap);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: NvMediaImageLock failed\n", __func__);
        return status;
    }
    NvMediaImageUnlock(image);


    uHeightSurface = surfaceMap.height;
    uWidthSurface  = surfaceMap.width;

    if(width > uWidthSurface || height > uHeightSurface) {
        LOG_ERR("%s: Bad parameter\n", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    pBuff = calloc(1,sizeof(uint8_t*)*MAXM_NUM_SURFACES);
    if(!pBuff) {
        LOG_ERR("%s: out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    pBuffPitches = calloc(1,sizeof(uint32_t) * MAXM_NUM_SURFACES);
    if(!pBuffPitches) {
        LOG_ERR("%s: out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    status = GetSurfParams(image->type,
                           &xScalePtr,
                           &yScalePtr,
                           &bytePerPixelPtr,
                           &numSurfaces);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: GetSurfParams failed\n", __func__);
        goto done;
    }

    status = NvMediaSurfaceFormatGetAttrs(image->type,
                                          srcAttr,
                                          NVM_SURF_FMT_ATTR_MAX);
    if (status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: NvMediaSurfaceFormatGetAttrs failed\n", __func__);
        goto done;
    }

    surfaceSize = 0;
    imageSize = 0;
    for(i = 0; i < numSurfaces; i++) {
        surfaceSize += (uWidthSurface * xScalePtr[i] * uHeightSurface * yScalePtr[i] * bytePerPixelPtr[i]);
        imageSize += (width * xScalePtr[i] * height * yScalePtr[i] * bytePerPixelPtr[i]);
        pBuffPitches[i] = (uint32_t)((float)uWidthSurface * xScalePtr[i]) * bytePerPixelPtr[i];
    }

    buffer = calloc(1, surfaceSize);
    if(!buffer) {
        LOG_ERR("%s: out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    pBuffer = buffer;
    memset(buffer,0x00,surfaceSize);
    for(i = 0; i < numSurfaces; i++) {
        pBuff[i] = buffer;
        buffer = buffer + (uint32_t)(uHeightSurface * yScalePtr[i] * pBuffPitches[i]);
    }

    status = NvMediaImageLock(image, NVMEDIA_IMAGE_ACCESS_WRITE, &surfaceMap);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: NvMediaImageLock failed\n", __func__);
        goto done;
    }
    status = NvMediaImagePutBits(image, NULL, (void **)pBuff, pBuffPitches);
    NvMediaImageUnlock(image);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: NvMediaImagePutBits failed\n", __func__);
    }

done:
    if(pBuff) {
        free(pBuff);
    }

    if (pBuffPitches) {
        free(pBuffPitches);
    }

    if (pBuffer) {
        free(pBuffer);
    }

    return status;
}

NvMediaStatus
ReadPPMImage(
    char *fileName,
    NvMediaImage *image)
{
    uint32_t uSurfaceWidth, uSurfaceHeight;
    uint32_t x, y;
    char buf[256], *c;
    uint8_t *pBuff = NULL, *pRGBBuff;
    uint32_t uFrameSize;
    FILE *file = NULL;
    NvMediaImageSurfaceMap surfaceMap;
    NvMediaStatus ret = NVMEDIA_STATUS_OK;

    if(NvMediaImageLock(image, NVMEDIA_IMAGE_ACCESS_WRITE, &surfaceMap) != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: NvMediaImageLock failed\n", __func__);
        goto done;
    }

    if(!image) {
        LOG_ERR("%s: Failed allocating memory\n", __func__);
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    file = fopen(fileName, "rb");
    if(!file) {
        LOG_ERR("%s: Error opening file: %s\n", __func__, fileName);
        return NVMEDIA_STATUS_ERROR;
    }

    c = fgets(buf, 256, file);
    if(!c || strncmp(buf, "P6\n", 3)) {
        LOG_ERR("%s: Invalid PPM header in file: %s\n", __func__, fileName);
        ret = NVMEDIA_STATUS_ERROR;
        goto done;
    }
    do {
        c = fgets(buf, 255, file);
        if(!c) {
            LOG_ERR("%s: Invalid PPM header in file: %s\n", __func__, fileName);
            ret = NVMEDIA_STATUS_ERROR;
            goto done;
        }
    } while(!strncmp(buf, "#", 1));
    if(sscanf(buf, "%u %u", &uSurfaceWidth, &uSurfaceHeight) != 2) {
        LOG_ERR("%s: Error getting PPM file resolution in file: %s\n", __func__, fileName);
        ret = NVMEDIA_STATUS_ERROR;
        goto done;
    }
    if(uSurfaceWidth < 16 || uSurfaceWidth > 2048 ||
            uSurfaceHeight < 16 || uSurfaceHeight > 2048) {
        LOG_ERR("%s: Invalid PPM file resolution: %ux%u for file: %s\n", __func__, uSurfaceWidth, uSurfaceHeight, fileName);
        ret = NVMEDIA_STATUS_ERROR;
        goto done;
    }

    uFrameSize = uSurfaceWidth * uSurfaceHeight * 3;
    pBuff = malloc(uFrameSize);
    if(!pBuff) {
        LOG_ERR("%s: Out of memory\n", __func__);
        ret = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }
    pRGBBuff = pBuff;

    if(fread(pRGBBuff, uFrameSize, 1, file) != 1) {
        LOG_ERR("%s: Error reading file: %s\n", __func__, fileName);
        ret = NVMEDIA_STATUS_ERROR;
        goto done;
    }

    for(y = 0; y < uSurfaceHeight; y++) {
        uint8_t *pPixel = (uint8_t *)surfaceMap.surface[0].mapping + surfaceMap.surface[0].pitch * y;
        for(x = 0; x < uSurfaceWidth; x++) {
            *pPixel++ = *pRGBBuff++; // R
            *pPixel++ = *pRGBBuff++; // G
            *pPixel++ = *pRGBBuff++; // B
            *pPixel++ = 255;         // Alpha
        }
    }

done:
    NvMediaImageUnlock(image);
    if(pBuff) free(pBuff);
    if(file) fclose(file);

    return ret;
}
NvMediaStatus
ReadYUVBuffer(
    FILE *file,
    uint32_t frameNum,
    uint32_t width,
    uint32_t height,
    uint8_t *pBuff,
    NvMediaBool bOrderUV)
{
    uint8_t *pYBuff, *pUBuff, *pVBuff, *pChroma;
    uint32_t frameSize = (width * height *3)/2;
    NvMediaStatus ret = NVMEDIA_STATUS_OK;
    unsigned int i;

    if(!pBuff || !file)
        return NVMEDIA_STATUS_BAD_PARAMETER;

    pYBuff = pBuff;

    //YVU order in the buffer
    pVBuff = pYBuff + width * height;
    pUBuff = pVBuff + width * height / 4;

    if(fseek(file, frameNum * frameSize, SEEK_SET)) {
        LOG_ERR("ReadYUVBuffer: Error seeking file: %p\n", file);
        ret = NVMEDIA_STATUS_ERROR;
        goto done;
    }
    //read Y U V separately
    for(i = 0; i < height; i++) {
        if(fread(pYBuff, width, 1, file) != 1) {
            LOG_ERR("ReadYUVBuffer: Error reading file: %p\n", file);
            ret = NVMEDIA_STATUS_ERROR;
            goto done;
        }
        pYBuff += width;
    }

    pChroma = bOrderUV ? pUBuff : pVBuff;
    for(i = 0; i < height / 2; i++) {
        if(fread(pChroma, width / 2, 1, file) != 1) {
            LOG_ERR("ReadYUVBuffer: Error reading file: %p\n", file);
            ret = NVMEDIA_STATUS_ERROR;
            goto done;
        }
        pChroma += width / 2;
    }

    pChroma = bOrderUV ? pVBuff : pUBuff;
    for(i = 0; i < height / 2; i++) {
        if(fread(pChroma, width / 2, 1, file) != 1) {
            LOG_ERR("ReadYUVBuffer: Error reading file: %p\n", file);
            ret = NVMEDIA_STATUS_ERROR;
            goto done;
        }
        pChroma += width / 2;
    }

done:
    return ret;
}

NvMediaStatus
ReadRGBABuffer(
    FILE *file,
    uint32_t frameNum,
    uint32_t width,
    uint32_t height,
    uint8_t *pBuff)
{
    uint32_t frameSize = width * height * 4;
    NvMediaStatus ret = NVMEDIA_STATUS_OK;

    if(!pBuff || !file)
        return NVMEDIA_STATUS_BAD_PARAMETER;

    if(fseek(file, frameNum * frameSize, SEEK_SET)) {
        LOG_ERR("ReadRGBABuffer: Error seeking file: %p\n", file);
        ret = NVMEDIA_STATUS_ERROR;
        goto done;
    }

    //read rgba data
    if(fread(pBuff, frameSize, 1, file) != 1) {
        if (feof(file))
            LOG_DBG("ReadRGBABuffer: file read to the end\n");
        else
            LOG_ERR("ReadRGBABuffer: Error reading file: %p\n", file);
        ret = NVMEDIA_STATUS_ERROR;
        goto done;
    }

done:
    return ret;
}

NvMediaStatus
GetFrameCrc(
    uint8_t   **pBuff,
    uint32_t  *widths,
    uint32_t  *heights,
    uint32_t  *pitches,
    uint32_t  numSurfaces,
    uint32_t  *crcOut)
{
    uint32_t lines, crc = 0;
    uint32_t i;
    uint8_t *bufferTmp;

    if(!pBuff || !crcOut || !widths || !heights || !pitches || (numSurfaces == 0)) {
        LOG_ERR("GetFrameCrc: Bad parameter\n");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    for(i = 0; i < numSurfaces; i++) {
        lines = heights[i];
        bufferTmp = pBuff[i];
        while(lines--) {
            crc = CalculateBufferCRC(widths[i], crc, bufferTmp);
            bufferTmp += pitches[i];
        }
    }

    *crcOut = crc;
    return NVMEDIA_STATUS_OK;
}
