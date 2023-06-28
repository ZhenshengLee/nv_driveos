/*
 * Copyright (c) 2020-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software and related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#include "image_ofa_utils.h"

#define MAX_NUM_SURFACES 6

//TODO Need to put things in a common place
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

SurfDesc SurfDescTable_RGBA_OFA  = {
    .heightFactor = {1, 0, 0, 0, 0, 0},
    .widthFactor = {1, 0, 0, 0, 0, 0},
    .numSurfaces = 1,
};

SurfDesc SurfDescTable_YUV_OFA[][4] = {
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

int32_t Kernel[5][5] =
{
    { 1,  4,  6,  4, 1 },
    { 4, 16, 24, 16, 4 },
    { 6, 24, 36, 24, 6 },
    { 4, 16, 24, 16, 4 },
    { 1,  4,  6,  4, 1 }
};

SurfDesc SurfDescTable_Packed_OFA  = {
    .heightFactor = {1, 0, 0, 0, 0, 0},
    .widthFactor = {1, 0, 0, 0, 0, 0},
    .numSurfaces = 1,
};

unsigned int SurfBytesPerPixelTable_YUV_OFA[][9][6] = {
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

static void
pyramid (
   uint8_t  *psrc,       int32_t src_width,
   int32_t  src_height,  int32_t src_pitch,
   uint8_t  *pdst,       int32_t dst_width,
   int32_t  dst_height,  int32_t dst_pitch,
   uint32_t bytesPerPixel)
{
    int32_t w, h, kh, kw;
    int32_t x, y;
    uint32_t dst;

    for (h = 0; h < dst_height; h++)
    {
        for (w = 0; w < dst_width; w++)
        {
            dst = 0;
            for (kh = 0; kh < 5; kh++)
            {
                for (kw = 0; kw < 5; kw++)
                {
                    x = 2*w+kw-2;
                    y = 2*h+kh-2;
                    if (x < 0)
                        x = -x;
                    if (x >= src_width)
                        x = 2 *(src_width -1) - x;
                    if (y < 0)
                        y = -y;
                    if (y >= src_height)
                        y = 2 *(src_height -1) - y;

                    if (bytesPerPixel == 2)
                        dst += *(uint16_t*)(psrc + y * src_pitch + 2*x) * Kernel[kh][kw];
                    else
                        dst += *(psrc + y * src_pitch + x) * Kernel[kh][kw];
                }
            }
            if (bytesPerPixel == 2)
                *(uint16_t*)(pdst + h*dst_pitch+ 2*w) = (dst+128) >> 8;
            else
                *(pdst + h*dst_pitch+w) = (dst+128) >> 8;
        }
    }
    return;
}

typedef struct {
    NvSciBufAttrValImageLayoutType layout;
    uint32_t planeCount;
    uint32_t planeWidth[NV_SCI_BUF_IMAGE_MAX_PLANES];
    uint32_t planeHeight[NV_SCI_BUF_IMAGE_MAX_PLANES];
    NvSciBufAttrValColorFmt planeColorFormat[NV_SCI_BUF_IMAGE_MAX_PLANES];
    uint32_t planeBitsPerPixel[NV_SCI_BUF_IMAGE_MAX_PLANES];
} SciBufAttributes;


NvMediaStatus
FetchSciBufAttr (
    NvSciBufAttrList bufAttrList,
    SciBufAttributes *imgAttrValues);

NvMediaStatus
FetchSciBufAttr (
    NvSciBufAttrList bufAttrList,
    SciBufAttributes *imgAttrValues)
{
    NvSciError err = NvSciError_Success;

    NvSciBufAttrKeyValuePair imgAttrList[] = {
        { NvSciBufImageAttrKey_Layout, NULL, 0 },               /* 0 */
        { NvSciBufImageAttrKey_PlaneCount, NULL, 0 },           /* 1 */
        { NvSciBufImageAttrKey_PlaneWidth, NULL, 0 },           /* 2 */
        { NvSciBufImageAttrKey_PlaneHeight, NULL, 0 },          /* 3 */
        { NvSciBufImageAttrKey_PlaneColorFormat, NULL, 0 },     /* 4 */
        { NvSciBufImageAttrKey_PlaneBitsPerPixel, NULL, 0 }     /* 5 */
    };

    /* TODO: Check if it is safe to get array size like this */
    err = NvSciBufAttrListGetAttrs(bufAttrList, imgAttrList,
            sizeof(imgAttrList) / sizeof(NvSciBufAttrKeyValuePair));
    if (err != NvSciError_Success) {
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
    return NVMEDIA_STATUS_OK;
}

static NvMediaStatus
ReadImageOFANew(
    char            *fileName,
    uint32_t        frameNum,
    uint32_t        width,
    uint32_t        height,
    NvSciBufObj     image[NVMEDIA_IOFA_MAX_PYD_LEVEL],
    bool            uvOrderFlag,
    uint32_t        pixelAlignment,
    uint8_t         pyrLevel,
    uint32_t        bitDepth,
    ChromaFormat    inputFileChromaFormat
)
{
    SciBufAttributes attr = {0};
    uint8_t *pBuff[NVMEDIA_IOFA_MAX_PYD_LEVEL][MAX_NUM_SURFACES] = {NULL};
    uint32_t pBuffSizes[NVMEDIA_IOFA_MAX_PYD_LEVEL][MAX_NUM_SURFACES] = {};
    uint32_t *pBuffPitches = NULL;
    uint32_t imageSize = 0,surfaceSize = 0;
    uint8_t *buffer[NVMEDIA_IOFA_MAX_PYD_LEVEL] = {NULL};
    uint8_t *pBuffer[NVMEDIA_IOFA_MAX_PYD_LEVEL] = {NULL};
    uint32_t i, j, k, newk = 0;
    float *xScalePtr = NULL, *yScalePtr = NULL;
    uint32_t *bytePerPixelPtr = NULL;
    uint32_t numSurfaces = 1;
    uint32_t uHeightSurface, uWidthSurface;

    NvMediaStatus status = NVMEDIA_STATUS_ERROR;
    FILE *file = NULL;
    uint32_t count, index;
    uint32_t arrWidth[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint32_t arrHeight[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint8_t  pyrIdx = 0;
    uint32_t bytesPerPixel = 1;
    unsigned int yuvpackedtbl[6] = {1, 0, 0, 0, 0, 0};
    unsigned int numcomps = 1;
    NvSciBufAttrList bufAttrList;
    NvSciError err;

    if (bitDepth > 8)
    {
        bytesPerPixel = 2;
    }

    if(!image || !fileName)
    {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    status = NvSciBufObjGetAttrList(image[0], &bufAttrList);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR(" NvSciBufAttrListGetAttrs failed");
        return false;
    }

    status = FetchSciBufAttr(bufAttrList, &attr);
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR(" FetchSciBufAttr failed");
        return false;
    }
    uWidthSurface = attr.planeWidth[0];
    uHeightSurface = attr.planeHeight[0];

    if(width > uWidthSurface || height > uHeightSurface)
    {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    pBuffPitches = calloc(1,sizeof(uint32_t) * MAX_NUM_SURFACES);
    if(!pBuffPitches)
    {
        LOG_ERR("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    switch(inputFileChromaFormat) {
        case YUV400P_8bit:
        case YUV400P_10bit:
        case YUV400P_12bit:
        case YUV400P_16bit:
            xScalePtr =  &SurfDescTable_Packed_OFA.widthFactor[0];
            yScalePtr = &SurfDescTable_Packed_OFA.heightFactor[0];
            numSurfaces = SurfDescTable_Packed_OFA.numSurfaces;
            numcomps = 1U;
            break;
        case YUV420P_8bit:
        case YUV420P_10bit:
        case YUV420P_12bit:
        case YUV420P_16bit:
                xScalePtr = &SurfDescTable_YUV_OFA[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_420].widthFactor[0];
                yScalePtr = &SurfDescTable_YUV_OFA[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_420].heightFactor[0];
                numSurfaces = SurfDescTable_YUV_OFA[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_420].numSurfaces;
                break;
        case YUV422P_8bit:
        case YUV422P_10bit:
        case YUV422P_12bit:
        case YUV422P_16bit:
            xScalePtr = &SurfDescTable_YUV_OFA[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_422].widthFactor[0];
            yScalePtr = &SurfDescTable_YUV_OFA[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_422].heightFactor[0];
            numSurfaces = SurfDescTable_YUV_OFA[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_422].numSurfaces;
            break;
        case YUV444P_8bit:
        case YUV444P_10bit:
        case YUV444P_12bit:
        case YUV444P_16bit:
                xScalePtr = &SurfDescTable_YUV_OFA[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_444].widthFactor[0];
                yScalePtr = &SurfDescTable_YUV_OFA[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_444].heightFactor[0];
                numSurfaces = SurfDescTable_YUV_OFA[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_444].numSurfaces;
                break;
        default:
            LOG_ERR("Unsuported input format type: %u\n", inputFileChromaFormat);
            return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    switch(inputFileChromaFormat) {
        case YUV400P_8bit:
            //bitsPerPixel = 8U;
            yuvpackedtbl[0] = 1U;
            yuvpackedtbl[0] = yuvpackedtbl[0] * numcomps;
            bytePerPixelPtr = &yuvpackedtbl[0];
            break;
        case YUV400P_10bit:
            //bitsPerPixel = 10U;
            yuvpackedtbl[0] = 2U;
            yuvpackedtbl[0] = yuvpackedtbl[0] * numcomps;
            bytePerPixelPtr = &yuvpackedtbl[0];
            break;
        case YUV400P_12bit:
            //bitsPerPixel = 12U;
            yuvpackedtbl[0] = 2U;
            yuvpackedtbl[0] = yuvpackedtbl[0] * numcomps;
            bytePerPixelPtr = &yuvpackedtbl[0];
            break;
        case YUV400P_16bit:
            //bitsPerPixel = 16U;
            yuvpackedtbl[0] = 2U;
            yuvpackedtbl[0] = yuvpackedtbl[0] * numcomps;
            bytePerPixelPtr = &yuvpackedtbl[0];
            break;
        case YUV420P_8bit:
        case YUV422P_8bit:
        case YUV444P_8bit:
            //bitsPerPixel = 8U;
            bytePerPixelPtr = &SurfBytesPerPixelTable_YUV_OFA[SURF_DESC_INDEX_PLANAR][0][0];
            break;
        case YUV420P_10bit:
        case YUV422P_10bit:
        case YUV444P_10bit:
            //bitsPerPixel = 10U;
            bytePerPixelPtr = &SurfBytesPerPixelTable_YUV_OFA[SURF_DESC_INDEX_PLANAR][1][0];
            break;
        case YUV420P_12bit:
        case YUV422P_12bit:
        case YUV444P_12bit:
            //bitsPerPixel = 12U;
            bytePerPixelPtr = &SurfBytesPerPixelTable_YUV_OFA[SURF_DESC_INDEX_PLANAR][2][0];
            break;
        case YUV420P_16bit:
        case YUV422P_16bit:
        case YUV444P_16bit:
            //bitsPerPixel = 16U;
            bytePerPixelPtr = &SurfBytesPerPixelTable_YUV_OFA[SURF_DESC_INDEX_PLANAR][4][0];
            break;
        default:
            LOG_ERR("Unsuported input format type: %u\n", inputFileChromaFormat);
            return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    surfaceSize = 0;
    imageSize = 0;
    for(i = 0; i < numSurfaces; i++)
    {
        surfaceSize += (uWidthSurface * xScalePtr[i] * uHeightSurface * yScalePtr[i] * bytePerPixelPtr[i]);
        imageSize += (width * xScalePtr[i] * height * yScalePtr[i] * bytePerPixelPtr[i]);
        pBuffPitches[i] = (uint32_t)((float)uWidthSurface * xScalePtr[i]) * bytePerPixelPtr[i];
    }

    buffer[0] = calloc(1, surfaceSize);
    if(!buffer[0])
    {
        LOG_ERR("%s: Out of memory\n", __func__);
        status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        goto done;
    }

    pBuffer[0] = buffer[0];
    memset(buffer[0],0x10,surfaceSize);
    for(i = 0; i < numSurfaces; i++)
    {
        pBuff[0][i] = buffer[0];
        if (i)
        {
            memset(pBuff[0][i], 0x80, (uHeightSurface * yScalePtr[i] * pBuffPitches[i]));
        }
        pBuffSizes[0][i] = (uint32_t)(uHeightSurface * yScalePtr[i] * pBuffPitches[i]);
        buffer[0] = buffer[0] + pBuffSizes[0][i];
    }

    file = fopen(fileName, "rb");
    if(!file)
    {
        LOG_ERR("%s: Error opening file: %s\n", __func__, fileName);
        status = NVMEDIA_STATUS_ERROR;
        goto done;
    }

    if(frameNum > 0)
    {
        if(fseeko(file, frameNum * (off_t)imageSize, SEEK_SET))
        {
            LOG_ERR("ReadImage: Error seeking file: %s\n", fileName);
            status = NVMEDIA_STATUS_ERROR;
            goto done;
        }
    }

    for(k = 0; k < numSurfaces; k++)
    {
        for(j = 0; j < height*yScalePtr[k]; j++)
        {
            newk = (!uvOrderFlag && k ) ? (numSurfaces - k) : k;
            index = j * pBuffPitches[newk];
            count = width * xScalePtr[newk] * bytePerPixelPtr[newk];
            if (fread(pBuff[0][newk] + index, count, 1, file) != 1)
            {
                status = NVMEDIA_STATUS_ERROR;
                LOG_ERR("ReadImage: Error reading file: %s\n", fileName);
                goto done;
            }
            //if((surfType == NVM_SURF_ATTR_SURF_TYPE_YUV) && (pixelAlignment == LSB_ALIGNED)) //ColorFormat
            if((pixelAlignment == LSB_ALIGNED)) //ColorFormat Need to fix
            {
                uint16_t *psrc = (uint16_t*)(pBuff[0][newk] + index);
                switch(bitDepth)
                {
                    case 10:
                        for(i = 0; i < count/2; i++)
                        {
                            *(psrc + i) = (*(psrc + i)) << (16 - 10);
                        }
                        break;
                    case 12:
                        for(i = 0; i < count/2; i++)
                        {
                            *(psrc + i) = (*(psrc + i)) << (16 - 12);
                        }
                        break;
                    case 14:
                        for(i = 0; i < count/2; i++)
                        {
                            *(psrc + i) = (*(psrc + i)) << (16 - 14);
                        }
                        break;
                    default:
                        break;
                }
            }
        }
    }
    err = NvSciBufObjPutPixels(image[0], NULL, (const void **)pBuff[0], pBuffSizes[0],
            pBuffPitches);
    if (err != NvSciError_Success)
    {
        status = NVMEDIA_STATUS_ERROR;
        LOG_ERR("%s: NvSciBufObjPutPixels() failed\n", __func__);
        goto done;
    } else {
        status = NVMEDIA_STATUS_OK;
    }

        {
        arrWidth[0] = width;
        arrHeight[0] = height;

        for (pyrIdx = 1; pyrIdx < pyrLevel; pyrIdx++)
        {
            arrWidth[pyrIdx] = (arrWidth[pyrIdx-1] + 1)/2;
            arrHeight[pyrIdx] = (arrHeight[pyrIdx-1] + 1)/2;

            status = NvSciBufObjGetAttrList(image[pyrIdx], &bufAttrList);
            if(status != NVMEDIA_STATUS_OK) {
                LOG_ERR(" NvSciBufAttrListGetAttrs failed");
                return false;
            }

            status = FetchSciBufAttr(bufAttrList, &attr);
            if(status != NVMEDIA_STATUS_OK) {
                LOG_ERR(" FetchSciBufAttr failed");
                return false;
            }
            uWidthSurface = attr.planeWidth[0];
            uHeightSurface = attr.planeHeight[0];

            surfaceSize = 0;
            for(i = 0; i < numSurfaces; i++)
            {
                surfaceSize += (uWidthSurface * xScalePtr[i] * uHeightSurface * yScalePtr[i] * bytePerPixelPtr[i]);
                pBuffPitches[i] = (uint32_t)((float)uWidthSurface * xScalePtr[i]) * bytePerPixelPtr[i];
            }

            buffer[pyrIdx] = calloc(1, surfaceSize);
            if (!buffer[pyrIdx])
            {
                LOG_ERR("%s: Out of memory\n", __func__);
                status = NVMEDIA_STATUS_OUT_OF_MEMORY;
                goto done;
            }

            pBuffer[pyrIdx] = buffer[pyrIdx];
            memset(buffer[pyrIdx],0x10,surfaceSize);
            for(i = 0; i < numSurfaces; i++)
            {
                pBuff[pyrIdx][i] = buffer[pyrIdx];
                if (i != 0)
                {
                    memset(pBuff[pyrIdx][i], 0x80, (uHeightSurface * yScalePtr[i] * pBuffPitches[i]));
                }
                pBuffSizes[pyrIdx][i] = (uint32_t)(uHeightSurface * yScalePtr[i] * pBuffPitches[i]);
                buffer[pyrIdx] = buffer[pyrIdx] + pBuffSizes[pyrIdx][i];
            }

            if ((bitDepth == 10) || (bitDepth == 12))
            {
                uint16_t *src = (uint16_t*)pBuff[pyrIdx-1][0];
                uint16_t *dst = (uint16_t*)pBuff[pyrIdx-1][0];
                for (i=0; i<arrWidth[pyrIdx-1]*arrHeight[pyrIdx-1]; i++)
                {
                    *(dst+i) = *(src+i) >> (16-bitDepth);
                }
            }

            pyramid(pBuff[pyrIdx-1][0], arrWidth[pyrIdx-1], arrHeight[pyrIdx-1], arrWidth[pyrIdx-1]*bytesPerPixel,
                    pBuff[pyrIdx][0], arrWidth[pyrIdx], arrHeight[pyrIdx], arrWidth[pyrIdx]*bytesPerPixel, bytesPerPixel);

            if ((bitDepth == 10) || (bitDepth == 12))
            {
                uint16_t *src = (uint16_t*)pBuff[pyrIdx][0];
                uint16_t *dst = (uint16_t*)pBuff[pyrIdx][0];
                for (i=0; i<arrWidth[pyrIdx]*arrHeight[pyrIdx]; i++)
                {
                    *(dst+i) = *(src+i) << (16-bitDepth);
                }
            }
 
            err = NvSciBufObjPutPixels(image[pyrIdx], NULL, (const void **)pBuff[pyrIdx],
                    pBuffSizes[pyrIdx], pBuffPitches);
            if (err != NvSciError_Success)
            {
                status = NVMEDIA_STATUS_ERROR;
                LOG_ERR("%s: NvSciBufObjPutPixels() failed\n", __func__);
                goto done;
            } else {
                status = NVMEDIA_STATUS_OK;
            }
        }
    }

done:
    for (pyrIdx = 0; pyrIdx < pyrLevel; pyrIdx++)
    {
        if (pBuffer[pyrIdx])
        {
            free(pBuffer[pyrIdx]);
        }
    }

    if (pBuffPitches)
    {
        free(pBuffPitches);
    }

    if(file)
    {
        fclose(file);
    }

    return status;
}

NvMediaStatus
ReadImageOFA (
    char         *fileName,
    uint32_t     frameNum,
    uint32_t     width,
    uint32_t     height,
    NvSciBufObj image[NVMEDIA_IOFA_MAX_PYD_LEVEL],
    bool  uvOrderFlag,
    uint32_t     pixelAlignment,
    uint8_t      pyrLevel,
    uint32_t     bitdepth,
    ChromaFormat inputFileChromaFormat
)
{
    return ReadImageOFANew(fileName, frameNum, width, height, image,
                            uvOrderFlag, pixelAlignment, pyrLevel, bitdepth, inputFileChromaFormat);
}

