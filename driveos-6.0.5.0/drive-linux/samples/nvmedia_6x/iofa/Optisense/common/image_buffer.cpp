/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "image_buffer.h"

#define SURF_DESC_INDEX_PLANAR      0
#define SURF_DESC_INDEX_SEMI_PLANAR 1
#define SURF_DESC_INDEX_PACKED      2
#define SURF_DESC_INDEX_420         0
#define SURF_DESC_INDEX_422         1
#define SURF_DESC_INDEX_444         2

typedef struct {
    float heightFactor[6];
    float widthFactor[6];
    unsigned int numSurfaces;
} SurfDesc;

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

ImageBuffer::ImageBuffer(uint32_t bitDepth, ChromaFormat chromaFormat, uint16_t width, uint16_t height) :
    Buffer(bitDepth, chromaFormat, width, height)
{
}

NvSciBufObj* ImageBuffer::getHandle()
{
    return &m_image;
}

//fixme
bool ImageBuffer::checkOpDone()
{
    return true;
}

NvMediaStatus ImageBuffer::PopulateNvSciBufAttrList (
    SurfaceFormat format,
    bool needCpuAccess,
    NvSciBufAttrValImageLayoutType layout,
    uint32_t planeCount,
    NvSciBufAttrValAccessPerm access_perm,
    uint32_t lumaBaseAddressAlign,
    NvSciBufAttrValColorStd lumaColorStd,
    NvSciBufAttrValImageScanType scanType,
    NvSciBufAttrList bufAttributeList
)
{
    NvSciError err;
    NvSciBufAttrValColorFmt colorFormat[NV_SCI_BUF_IMAGE_MAX_PLANES] = {};
    NvSciBufAttrValColorStd colorStd[NV_SCI_BUF_IMAGE_MAX_PLANES] = {};
    uint32_t planeWidth[NV_SCI_BUF_IMAGE_MAX_PLANES] = {0};
    uint32_t planeHeight[NV_SCI_BUF_IMAGE_MAX_PLANES] = {0};
    uint32_t baseAddrAlign[NV_SCI_BUF_IMAGE_MAX_PLANES] = {0};
    uint64_t padding[NV_SCI_BUF_IMAGE_MAX_PLANES] = {0};
    bool vprFlag = false;

    switch (format)
    {
        /* Semi-planar formats */
        case YUV420SP_8bit:
        case YUV422SP_8bit:
        case YUV444SP_8bit:
            colorFormat[0]  = NvSciColor_Y8;
            colorFormat[1]  = NvSciColor_U8V8;
            break;
        case YUV420SP_10bit:
        case YUV422SP_10bit:
        case YUV444SP_10bit:
            colorFormat[0] = NvSciColor_Y10;
            colorFormat[1] = NvSciColor_U10V10;
            break;
        case YUV420SP_12bit:
        case YUV422SP_12bit:
        case YUV444SP_12bit:
            colorFormat[0] = NvSciColor_Y12;
            colorFormat[1] = NvSciColor_U12V12;
            break;
        case YUV420SP_16bit:
        case YUV422SP_16bit:
        case YUV444SP_16bit:
            colorFormat[0] = NvSciColor_Y16;
            colorFormat[1] = NvSciColor_U16V16;
            break;

        /* Planar formats */
        case YUV400P_8bit:
            colorFormat[0] = NvSciColor_Y8;
            break;
        case YUV400P_10bit:
            colorFormat[0] = NvSciColor_Y10;
            break;
        case YUV400P_12bit:
            colorFormat[0] = NvSciColor_Y12;
            break;
        case YUV400P_16bit:
            colorFormat[0] = NvSciColor_Y16;
            break;
        case YUV420P_8bit:
        case YUV422P_8bit:
        case YUV444P_8bit:
            colorFormat[0] = NvSciColor_Y8;
            colorFormat[1] = NvSciColor_U8;
            colorFormat[2] = NvSciColor_V8;
            break;
        case YUV420P_10bit:
        case YUV422P_10bit:
        case YUV444P_10bit:
            colorFormat[0] = NvSciColor_Y10;
            colorFormat[1] = NvSciColor_U10;
            colorFormat[2] = NvSciColor_V10;
            break;
        case YUV420P_12bit:
        case YUV422P_12bit:
        case YUV444P_12bit:
            colorFormat[0] = NvSciColor_Y12;
            colorFormat[1] = NvSciColor_U12;
            colorFormat[2] = NvSciColor_V12;
            break;
        case YUV420P_16bit:
        case YUV422P_16bit:
        case YUV444P_16bit:
            colorFormat[0] = NvSciColor_Y16;
            colorFormat[1] = NvSciColor_U16;
            colorFormat[2] = NvSciColor_V16;
            break;

        /* Packed formats */
        case SF_RG16:
            colorFormat[0] = NvSciColor_Signed_R16G16;
            break;
        case SF_A16:
            colorFormat[0] = NvSciColor_Signed_A16;
            break;
        case SF_A8:
            colorFormat[0] = NvSciColor_A8;
            break;
        default:
            cerr<<"PopulateNvSciBufAttrList: Unsupported color format: planeCount\n";
            return NVMEDIA_STATUS_BAD_PARAMETER;
    }
     /* Set image dimensions */
    planeWidth[0]   = m_width;
    planeHeight[0]  = m_height;
    if (planeCount > 1)
    {
    switch (format)
    {
        /* Planar formats */
        case YUV420P_8bit:
        case YUV420P_10bit:
        case YUV420P_12bit:
        case YUV420P_16bit:
            planeWidth[1]  = m_width >> 1;
            planeHeight[1] = m_height >> 1;
            planeWidth[2] = planeWidth[1];
            planeHeight[2] = planeHeight[1];
            break;
        case YUV422P_8bit:
        case YUV422P_10bit:
        case YUV422P_12bit:
        case YUV422P_16bit:
            planeWidth[1]   = m_width >> 1;
            planeHeight[1]  = m_height;
            planeWidth[2] = planeWidth[1];
            planeHeight[2] = planeHeight[1];
            break;
            break;
        case YUV444P_8bit:
        case YUV444P_10bit:
        case YUV444P_12bit:
        case YUV444P_16bit:
            planeWidth[1]   = m_width;
            planeHeight[1]  = m_height;
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
            planeWidth[1]   = m_width >> 1;
            planeHeight[1]  = m_height >> 1;
            break;
        case YUV422SP_8bit:
        case YUV422SP_10bit:
        case YUV422SP_12bit:
        case YUV422SP_16bit:
            planeWidth[1]   = m_width >> 1;
            planeHeight[1]  = m_height;
            break;
        case YUV444SP_8bit:
        case YUV444SP_10bit:
        case YUV444SP_12bit:
        case YUV444SP_16bit:
            planeWidth[1]   = m_width;
            planeHeight[1]  = m_height;
            break;

        case YUV400P_8bit:
        case YUV400P_10bit:
        case YUV400P_12bit:
        case YUV400P_16bit:
            /* Single plane format */
            break;
        default:
            cerr<<"PopulateNvSciBufAttrList: Unsupported color format: chromaFormat 1\n";
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
    if (err != NvSciError_Success)
    {
        return NVMEDIA_STATUS_ERROR;
    }
    return NVMEDIA_STATUS_OK;
}

bool ImageBuffer::GetFormatType(
    SurfaceFormat       *chromaTarget)
{
    if(m_bitdepth == 8)
    {
        switch(m_chroma_format)
        {
            case YUV_400:
                *chromaTarget = YUV400P_8bit;
                break;
            case YUV_420:
                *chromaTarget = YUV420SP_8bit;
                break;
            case YUV_422:
                *chromaTarget = YUV422SP_8bit;
                break;
            case YUV_444:
                *chromaTarget = YUV444SP_8bit;
                break;
            case A8:
                *chromaTarget = SF_A8;
                break;
            default:
                return false;
        }

    }
    else if(m_bitdepth == 10)
    {
        switch(m_chroma_format)
        {
            case YUV_400:
                *chromaTarget = YUV400P_10bit;
                break;
            case YUV_420:
                *chromaTarget = YUV420SP_10bit;
                break;
            case YUV_422:
                *chromaTarget = YUV422SP_10bit;
                break;
            case YUV_444:
                *chromaTarget = YUV444SP_10bit;
                break;
            default:
                return false;
        }
    }
    else if(m_bitdepth == 12)
    {
        switch(m_chroma_format)
        {
            case YUV_400:
                *chromaTarget = YUV400P_12bit;
                break;
            case YUV_420:
                *chromaTarget = YUV420SP_12bit;
                break;
            case YUV_422:
                *chromaTarget = YUV422SP_12bit;
                break;
            case YUV_444:
                *chromaTarget = YUV444SP_12bit;
                break;
            default:
                return false;
        }
    }
    else if(m_bitdepth == 16)
    {
        switch(m_chroma_format)
        {
            case YUV_400:
                *chromaTarget = YUV400P_16bit;
                break;
            case YUV_420:
                *chromaTarget = YUV420SP_16bit;
                break;
            case YUV_422:
                *chromaTarget = YUV422SP_16bit;
                break;
            case YUV_444:
                *chromaTarget = YUV444SP_16bit;
                break;
            case A16:
                *chromaTarget = SF_A16;
                break;
            case RG16:
                *chromaTarget = SF_RG16;
                break;
            default:
                return false;
        }
    }
    else
    {
        return false;
    }
    return true;
}

bool ImageBuffer::GetFileFormat(
    SurfaceFormat  *chromaTarget)
{
    if(m_bitdepth == 8)
    {
        switch(m_chroma_format)
        {
            case YUV_400:
                *chromaTarget = YUV400P_8bit;
                break;
            case YUV_420:
                *chromaTarget = YUV420P_8bit;
                break;
            case YUV_422:
                *chromaTarget = YUV422P_8bit;
                break;
            case YUV_444:
                *chromaTarget = YUV444P_8bit;
                break;
            case A8:
                *chromaTarget = SF_A8;
                break;
            default:
                return false;
        }

    }
    else if(m_bitdepth == 10)
    {
        switch(m_chroma_format)
        {
            case YUV_400:
                *chromaTarget = YUV400P_10bit;
                break;
            case YUV_420:
                *chromaTarget = YUV420P_10bit;
                break;
            case YUV_422:
                *chromaTarget = YUV422P_10bit;
                break;
            case YUV_444:
                *chromaTarget = YUV444P_10bit;
                break;
            default:
                return false;
        }
    }
    else if(m_bitdepth == 12)
    {
        switch(m_chroma_format)
        {
            case YUV_400:
                *chromaTarget = YUV400P_12bit;
                break;
            case YUV_420:
                *chromaTarget = YUV420P_12bit;
                break;
            case YUV_422:
                *chromaTarget = YUV422P_12bit;
                break;
            case YUV_444:
                *chromaTarget = YUV444P_12bit;
                break;
            default:
                return false;
        }
    }
    else if(m_bitdepth == 16)
    {
        switch(m_chroma_format)
        {
            case YUV_400:
                *chromaTarget = YUV400P_16bit;
                break;
            case YUV_420:
                *chromaTarget = YUV420P_16bit;
                break;
            case YUV_422:
                *chromaTarget = YUV422P_16bit;
                break;
            case YUV_444:
                *chromaTarget = YUV444P_16bit;
                break;
            case A16:
                *chromaTarget = SF_A16;
                break;
            case RG16:
                *chromaTarget = SF_RG16;
                break;
            default:
                return false;
        }
    }
    else
    {
        return false;
    }
    return true;
}

NvMediaStatus ImageBuffer::createBuffer()
{
    NvSciError err;
    NvMediaStatus status = NVMEDIA_STATUS_ERROR;
    uint32_t i, planeCount;
    SurfaceFormat InputFormat;
    SurfaceFormat FileFormat;
    NvSciBufAttrList reconciledAttributeList=NULL;
    NvSciBufAttrList conflictList=NULL;

    if(m_chroma_format == YUV_400 || m_chroma_format == A8 || m_chroma_format == A16 || m_chroma_format == RG16)
    {
        planeCount = 1;
    }
    else
    {
        planeCount = 2;
    }
    if(!GetFormatType(&InputFormat))
    {
        cerr<<"createBuffer: Incorrect input format type\n";
        return status;
    }
    if(!GetFileFormat(&FileFormat))
    {
        cerr<<"createBuffer: Incorrect file format type\n";
        return status;
    }
    err = NvSciBufModuleOpen(&m_bufModule);
    if(err != NvSciError_Success)
    {
        return status;
    }
    err = NvSciBufAttrListCreate(m_bufModule, &m_attributeList);
    if(err != NvSciError_Success)
    {
        return status;
    }
    status = NvMediaIOFAFillNvSciBufAttrList(m_attributeList);
    if(status != NVMEDIA_STATUS_OK)
    {
        cerr<<"createBuffer: Failed to fill IOFA internal attributes for input surface \n";
        return status;
    }

    status = PopulateNvSciBufAttrList(InputFormat, true, NvSciBufImage_BlockLinearType, planeCount, NvSciBufAccessPerm_ReadWrite, 256U, NvSciColorStd_REC601_ER, NvSciBufScan_ProgressiveType, m_attributeList);
    if(NVMEDIA_STATUS_OK != status)
    {
        cerr<<"createBuffer: failed to fill IOFA external attributes for input surface\n";
        return NVMEDIA_STATUS_ERROR;
    }

    err = NvSciBufAttrListReconcile(&m_attributeList, 1U, &reconciledAttributeList, &conflictList);
    if(err!=NvSciError_Success)
    {
        cerr<<"createBuffer: Reconciliation for input frame failed for input surface\n";
        return NVMEDIA_STATUS_ERROR;
    }

    err = NvSciBufObjAlloc(reconciledAttributeList, &m_image);
    if(err != NvSciError_Success)
    {
        cerr<<"createBuffer: NvSciBuf Obj creation for input surface failed\n";
        return NVMEDIA_STATUS_ERROR;
    }

    static unsigned int yuvpackedtbl[6] = {1, 0, 0, 0, 0, 0};
    unsigned int numcomps = 1;
    uint32_t const *planeWidth, *planeHeight;
    NvSciBufAttrList attrList;

    switch(FileFormat)
    {
        case YUV400P_8bit:
        case YUV400P_10bit:
        case YUV400P_12bit:
        case YUV400P_16bit:
            m_xScalePtr =  &SurfDescTable_Packed.widthFactor[0];
            m_yScalePtr = &SurfDescTable_Packed.heightFactor[0];
            m_numSurfaces = SurfDescTable_Packed.numSurfaces;
            numcomps = 1U;
            break;
        case YUV420P_8bit:
        case YUV420P_10bit:
        case YUV420P_12bit:
        case YUV420P_16bit:
            m_xScalePtr = &SurfDescTable_YUV[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_420].widthFactor[0];
            m_yScalePtr = &SurfDescTable_YUV[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_420].heightFactor[0];
            m_numSurfaces = SurfDescTable_YUV[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_420].numSurfaces;
            break;
        case YUV422P_8bit:
        case YUV422P_10bit:
        case YUV422P_12bit:
        case YUV422P_16bit:
            m_xScalePtr = &SurfDescTable_YUV[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_422].widthFactor[0];
            m_yScalePtr = &SurfDescTable_YUV[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_422].heightFactor[0];
            m_numSurfaces = SurfDescTable_YUV[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_422].numSurfaces;
            break;
        case YUV444P_8bit:
        case YUV444P_10bit:
        case YUV444P_12bit:
        case YUV444P_16bit:
            m_xScalePtr = &SurfDescTable_YUV[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_444].widthFactor[0];
            m_yScalePtr = &SurfDescTable_YUV[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_444].heightFactor[0];
            m_numSurfaces = SurfDescTable_YUV[SURF_DESC_INDEX_PLANAR][SURF_DESC_INDEX_444].numSurfaces;
            break;
        case SF_A8:
        case SF_A16:
        case SF_RG16:
            m_xScalePtr = &SurfDescTable_RGBA.widthFactor[0];
            m_yScalePtr = &SurfDescTable_RGBA.heightFactor[0];
            m_numSurfaces = SurfDescTable_RGBA.numSurfaces;
            break;
        default:
            cerr<<"Unsupported input format type\n";
            return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    switch(FileFormat)
    {
        case YUV400P_8bit:
            m_bitdepth = 8U;
            yuvpackedtbl[0] = 1U;
            yuvpackedtbl[0] = yuvpackedtbl[0] * numcomps;
            m_bytePerPixelPtr = &yuvpackedtbl[0];
            break;
        case YUV400P_10bit:
            m_bitdepth = 10U;
            yuvpackedtbl[0] = 2U;
            yuvpackedtbl[0] = yuvpackedtbl[0] * numcomps;
            m_bytePerPixelPtr = &yuvpackedtbl[0];
            break;
        case YUV400P_12bit:
            m_bitdepth = 12U;
            yuvpackedtbl[0] = 2U;
            yuvpackedtbl[0] = yuvpackedtbl[0] * numcomps;
            m_bytePerPixelPtr = &yuvpackedtbl[0];
            break;
        case YUV400P_16bit:
            m_bitdepth = 16U;
            yuvpackedtbl[0] = 2U;
            yuvpackedtbl[0] = yuvpackedtbl[0] * numcomps;
            m_bytePerPixelPtr = &yuvpackedtbl[0];
            break;
        case YUV420P_8bit:
        case YUV422P_8bit:
        case YUV444P_8bit:
            m_bitdepth = 8U;
            m_bytePerPixelPtr = &SurfBytesPerPixelTable_YUV[SURF_DESC_INDEX_PLANAR][0][0];
            break;
        case YUV420P_10bit:
        case YUV422P_10bit:
        case YUV444P_10bit:
            m_bitdepth = 10U;
            m_bytePerPixelPtr = &SurfBytesPerPixelTable_YUV[SURF_DESC_INDEX_PLANAR][1][0];
            break;
        case YUV420P_12bit:
        case YUV422P_12bit:
        case YUV444P_12bit:
            m_bitdepth = 12U;
            m_bytePerPixelPtr = &SurfBytesPerPixelTable_YUV[SURF_DESC_INDEX_PLANAR][2][0];
            break;
        case YUV420P_16bit:
        case YUV422P_16bit:
        case YUV444P_16bit:
            m_bitdepth = 16U;
            m_bytePerPixelPtr = &SurfBytesPerPixelTable_YUV[SURF_DESC_INDEX_PLANAR][4][0];
            break;
        case SF_A8:
            m_bitdepth = 8U;
            m_bytePerPixelPtr = &SurfBytesPerPixelTable_Alpha[0][0];
            break;
        case SF_A16:
            m_bitdepth = 16U;
            m_bytePerPixelPtr = &SurfBytesPerPixelTable_Alpha[4][0];
            break;
        case SF_RG16:
            m_bitdepth = 16U;
            m_bytePerPixelPtr = &SurfBytesPerPixelTable_RG16[0];
            break;
        default:
            cerr<<"Unsuppored input format type\n";
            return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (0U == m_bitdepth)
    {
        cerr<<"Failed to deduce bits per pixel\n";
        return NVMEDIA_STATUS_ERROR;
    }

    err = NvSciBufObjGetAttrList(m_image, &attrList);
    if (err != NvSciError_Success)
    {
        cerr<<" NvSciBufObjGetAttrList failed\n";
        return NVMEDIA_STATUS_ERROR;
    }

    NvSciBufAttrKeyValuePair imgattrs[] = {
        { NvSciBufImageAttrKey_PlaneWidth, NULL, 0 },       /* 0 */
        { NvSciBufImageAttrKey_PlaneHeight, NULL, 0 },      /* 1 */
    };

    err = NvSciBufAttrListGetAttrs(attrList, imgattrs,
                                    sizeof(imgattrs) / sizeof(NvSciBufAttrKeyValuePair));
    if (err != NvSciError_Success)
    {
        cerr<<" NvSciBufAttrListGetAttrs failed\n";
        return NVMEDIA_STATUS_ERROR;
    }

    planeWidth = (const uint32_t*)(imgattrs[0].value);
    planeHeight = (const uint32_t*)(imgattrs[1].value);
    m_lumaWidth = planeWidth[0];
    m_lumaHeight = planeHeight[0];

    /* Check if requested read width, height are lesser than the width and
       height of the surface - checking only for Luma */
    if((m_width > m_lumaWidth) || (m_height > m_lumaHeight))
    {
        cerr<<"createBuffer: Bad parameter m_widthxm_height vs m_lumawidthxm_lumaheight";
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    for(uint32_t iter = 0; iter<MAXM_NUM_SURFACES; iter++)
    {
        m_pBuffPitches[iter] = 1;
    }
    m_surfaceSize = 0;
    m_imageSize = 0;
    for(i = 0; i<m_numSurfaces; i++)
    {
        m_surfaceSize += (m_lumaWidth * m_xScalePtr[i] * m_lumaHeight * m_yScalePtr[i] * m_bytePerPixelPtr[i]);
        m_imageSize += (m_width * m_xScalePtr[i] * m_height * m_yScalePtr[i] * m_bytePerPixelPtr[i]);
        m_pBuffPitches[i] = (uint32_t)((float)m_lumaWidth * m_xScalePtr[i]) * m_bytePerPixelPtr[i];
        m_pBuffSizes[i] = m_pBuffPitches[i] * m_height * m_yScalePtr[i];
    }

    m_buffer = new uint8_t[m_surfaceSize];
    if (m_buffer == NULL)
    {
        cerr<<"Temporary surface allocation failed\n";
        return NVMEDIA_STATUS_OUT_OF_MEMORY;
    }
    m_p_buffer = m_buffer;
    memset(m_buffer, 0x10, m_surfaceSize);
    for (i = 0; i < m_numSurfaces; i++)
    {
        m_pBuff[i] = m_buffer;
        if (i != 0)
        {
            memset(m_pBuff[i], 0x80, (m_lumaHeight * m_yScalePtr[i] * m_pBuffPitches[i]));
        }
        m_buffer = m_buffer + (uint32_t)(m_lumaHeight * m_yScalePtr[i] * m_pBuffPitches[i]);
    }

    m_imageSize_write = 0;
    for (i = 0; i < m_numSurfaces; i++)
    {
        m_size_write[i] = (m_lumaWidth * m_xScalePtr[i] * m_lumaHeight * m_yScalePtr[i] * m_bytePerPixelPtr[i]);
        m_imageSize_write += m_size_write[i];
    }

    m_buffer_write = new uint8_t[m_imageSize_write];
    if (m_buffer_write == NULL)
    {
        cerr << "Temporary surface allocation for write surface failed\n";
        return NVMEDIA_STATUS_OUT_OF_MEMORY;
    }

    m_pBuffer_write = m_buffer_write;
    memset(m_buffer_write, 0xFF, m_imageSize_write);
    for (i = 0; i < m_numSurfaces; i++)
    {
        m_pBuff_write[i] = m_buffer_write;
        m_buffer_write = m_buffer_write + (uint32_t)(m_lumaHeight * m_yScalePtr[i] * m_pBuffPitches[i]);
    }

    if (reconciledAttributeList)
    {
        NvSciBufAttrListFree(reconciledAttributeList);
    }
    if (conflictList)
    {
        NvSciBufAttrListFree(conflictList);
    }

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus ImageBuffer::writeBuffer(uint8_t *p_file, NvMediaBool isPNGfile)
{
    uint32_t i, j, k;
    NvSciError err;
    NvMediaStatus status = NVMEDIA_STATUS_ERROR;
    uint32_t count, index;

    memset(m_p_buffer, 0x10, m_surfaceSize);
    for(i = 0; i < m_numSurfaces; i++)
    {
        if (i!=0)
        {
            memset(m_pBuff[i], 0x80, (m_lumaHeight * m_yScalePtr[i] * m_pBuffPitches[i]));
        }
    }

    if (isPNGfile)
    {
        uint32_t bpp = m_bitdepth > 8U ? 2U : 1U;
        memcpy(m_pBuff[0], p_file, bpp*m_width*m_height);
    }
    else
    {
        for(k = 0; k < m_numSurfaces; k++) {
            for(j = 0; j < m_height * m_yScalePtr[k]; j++) {
                index = j * m_pBuffPitches[k];
                count = m_width * m_xScalePtr[k] * m_bytePerPixelPtr[k];
                copy(p_file, p_file + count, m_pBuff[k] + index);
                p_file += count;

                /* TODO: Assuming YUV input */
                uint16_t *psrc = (uint16_t*)(m_pBuff[k] + index);

                if (m_bitdepth > 8U && m_chroma_format != A16 && m_chroma_format != RG16) {
                    for(i = 0; i < count/2; i++) {
                        *(psrc + i) = (*(psrc + i)) << (16 - m_bitdepth);
                    }
                }
            }
        }
    }
    err = NvSciBufObjPutPixels(m_image, NULL, (const void **)m_pBuff, m_pBuffSizes, m_pBuffPitches);
    if (err != NvSciError_Success)
    {
        status = NVMEDIA_STATUS_ERROR;
        cerr << "NvSciBufObjPutPixels failed\n";
    }
    else
    {
        status = NVMEDIA_STATUS_OK;
    }

    return status;
}

NvMediaStatus ImageBuffer::readBuffer()
{
    NvSciError err;

    memset(m_pBuffer_write, 0xFF, m_imageSize_write);
    err = NvSciBufObjGetPixels(m_image, NULL, (void **)m_pBuff_write, m_pBuffSizes, m_pBuffPitches);
    if (err != NvSciError_Success)
    {
        cerr << "NvSciBufObjGetPixels failed\n";
        return NVMEDIA_STATUS_ERROR;
    }

    return NVMEDIA_STATUS_OK;
}

void ImageBuffer::destroyBuffer()
{
    if(m_p_buffer != NULL)
    {
        delete[] m_p_buffer;
    }
    if(m_pBuffer_write != NULL)
    {
        delete[] m_pBuffer_write;
    }
    if(m_image != NULL)
    {
        NvSciBufObjFree(m_image);
    }
    if(NULL != m_bufModule)
    {
        NvSciBufModuleClose(m_bufModule);
    }
}

