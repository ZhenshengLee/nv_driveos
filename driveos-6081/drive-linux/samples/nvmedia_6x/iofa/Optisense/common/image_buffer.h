/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef IMAGE_BUFFER_OFA_H
#define IMAGE_BUFFER_OFA_H

#include <iostream>
#include <string>
#include <cstring>
#include "common_defs.h"
#include "nvscibuf.h"
#include "nvmedia_iofa.h"
#include "nvmedia_core.h"

#define MAXM_NUM_SURFACES 6U

using namespace std;

typedef struct
{
    float heightFactor[6];
    float widthFactor[6];
    unsigned int numSurfaces;
} ImgUtilSurfParams;

typedef struct {
    NvSciBufAttrValImageLayoutType layout;
    uint32_t planeCount;
    uint32_t planeWidth[NV_SCI_BUF_IMAGE_MAX_PLANES];
    uint32_t planeHeight[NV_SCI_BUF_IMAGE_MAX_PLANES];
    NvSciBufAttrValColorFmt planeColorFormat[NV_SCI_BUF_IMAGE_MAX_PLANES];
    uint32_t planeBitsPerPixel[NV_SCI_BUF_IMAGE_MAX_PLANES];
    uint8_t planeChannelCount[NV_SCI_BUF_IMAGE_MAX_PLANES];
    uint32_t planePitchBytes[NV_SCI_BUF_IMAGE_MAX_PLANES];
    bool needCpuAccess;
} BufAttrValues;

class Buffer
{
protected:
    uint32_t m_bitdepth;
    ChromaFormat m_chroma_format;
    uint16_t m_width;
    uint16_t m_height;
public:
    Buffer(uint32_t bitDepth, ChromaFormat chromaFormat, uint16_t width, uint16_t height) :
    m_bitdepth(bitDepth),
    m_chroma_format(chromaFormat),
    m_width(width),
    m_height(height)
    {
    }
    virtual NvMediaStatus createBuffer() = 0;
    virtual void destroyBuffer() = 0;
    virtual NvSciBufObj *getHandle() = 0;
    virtual NvMediaStatus writeBuffer(uint8_t *p_file, NvMediaBool isPNGfile) = 0;
    virtual NvMediaStatus readBuffer() = 0;
    virtual ~Buffer() {}
    virtual bool checkOpDone() = 0;
    virtual uint32_t getNumOfSurfaces() = 0;
    virtual uint8_t **getWriteBufferPointer() = 0;
    virtual uint32_t *getWriteBufferSize() = 0;
    virtual bool GetFormatType(SurfaceFormat* format) = 0;
    virtual bool GetFileFormat(SurfaceFormat* format) = 0;
    virtual NvMediaStatus PopulateNvSciBufAttrList (SurfaceFormat format,
                                                    bool needCpuAccess,
                                                    NvSciBufAttrValImageLayoutType layout,
                                                    uint32_t planeCount,
                                                    NvSciBufAttrValAccessPerm access_perm,
                                                    uint32_t lumaBaseAddressAlign,
                                                    NvSciBufAttrValColorStd lumaColorStd,
                                                    NvSciBufAttrValImageScanType scanType,
                                                    NvSciBufAttrList bufAttributeList) = 0;

    uint16_t getWidth()
    {
        return m_width;
    }

    uint16_t getHeight()
    {
        return m_height;
    }
};

class ImageBuffer : public Buffer
{
private:
    NvSciBufObj m_image;
    NvSciBufAttrList m_attributeList;
    uint32_t m_lumaHeight, m_lumaWidth;
    float *m_xScalePtr = NULL, *m_yScalePtr = NULL;
    uint32_t *m_bytePerPixelPtr = NULL;
    uint32_t m_numSurfaces = 1;
    uint32_t m_surfType, m_surfBPC;
    uint8_t *m_buffer = NULL;
    uint8_t *m_p_buffer = NULL;
    uint8_t *m_pBuff[MAXM_NUM_SURFACES] = {NULL};
    uint32_t m_pBuffPitches[MAXM_NUM_SURFACES];
    uint32_t m_pBuffSizes[MAXM_NUM_SURFACES];
    uint32_t m_imageSize = 0, m_surfaceSize = 0;
    uint8_t *m_pBuff_write[MAXM_NUM_SURFACES] = {NULL};
    uint8_t *m_buffer_write = NULL;
    uint8_t *m_pBuffer_write = NULL;
    uint32_t m_imageSize_write = 0;
    uint32_t m_size_write[3] = {0};
    NvSciBufModule m_bufModule;
public:
    ImageBuffer(uint32_t bitDepth, ChromaFormat chromaFormat, uint16_t width, uint16_t height);
    ImageBuffer(uint16_t width, uint16_t height, NvSciBufAttrList attrList);
    NvMediaStatus createBuffer() override;
    void destroyBuffer() override;
    NvSciBufObj *getHandle();
    NvMediaStatus writeBuffer(uint8_t *p_file, NvMediaBool isPNGfile) override;
    NvMediaStatus readBuffer() override;
    bool checkOpDone() override;
    bool GetFormatType(SurfaceFormat* format) override;
    bool GetFileFormat(SurfaceFormat* format) override;
    uint32_t getNumOfSurfaces() override { return m_numSurfaces; }
    uint8_t **getWriteBufferPointer() override { return &m_pBuffer_write; }
    uint32_t *getWriteBufferSize() override { return m_size_write; }
    NvMediaStatus PopulateNvSciBufAttrList (SurfaceFormat format,
                                            bool needCpuAccess,
                                            NvSciBufAttrValImageLayoutType layout,
                                            uint32_t planeCount,
                                            NvSciBufAttrValAccessPerm access_perm,
                                            uint32_t lumaBaseAddressAlign,
                                            NvSciBufAttrValColorStd lumaColorStd,
                                            NvSciBufAttrValImageScanType scanType,
                                            NvSciBufAttrList bufAttributeList) override;
};

#endif
