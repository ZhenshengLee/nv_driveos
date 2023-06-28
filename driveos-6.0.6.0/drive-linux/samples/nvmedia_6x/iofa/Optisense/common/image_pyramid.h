/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef IMAGE_PYRAMID_H
#define IMAGE_PYRAMID_H

#include "common_defs.h"
#include "image_buffer.h"
#include "nvmedia_iofa.h"

using namespace std;

class ImagePyramid
{
protected:
    uint32_t m_bitdepth;
    ChromaFormat m_chroma_format;
    uint16_t *m_width;
    uint16_t *m_height;
    uint16_t m_pyd_level;
    uint32_t m_bytes_per_pixel;
    NvSciBufAttrList m_out_format;
    uint16_t m_input_type; // 0: For input pyramid, 1: for output pyramid
    ImageBuffer *m_image_pyramid[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    void pyramid(uint8_t *psrc, uint8_t *pdst, int16_t idx);

public:
    ImagePyramid(uint32_t bitDepth, ChromaFormat chromaFormat, uint16_t *width, uint16_t *height, uint16_t pydLevel);
    ~ImagePyramid();
    NvMediaStatus createPyramid();
    NvMediaStatus writePyramid(uint8_t *p_file, bool isPNGfile);
    void destroyPyramid();
    ImageBuffer *getImageBuffer(uint8_t i) { return m_image_pyramid[i]; }
};

#endif
