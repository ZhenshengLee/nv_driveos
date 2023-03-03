/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "image_pyramid.h"

int32_t Kernel[5][5] =
{
    { 1,  4,  6,  4, 1 },
    { 4, 16, 24, 16, 4 },
    { 6, 24, 36, 24, 6 },
    { 4, 16, 24, 16, 4 },
    { 1,  4,  6,  4, 1 }
};

ImagePyramid::ImagePyramid(uint32_t bitDepth, ChromaFormat chromaFormat, uint16_t *width, uint16_t *height, uint16_t pydLevel) :
    m_bitdepth(bitDepth),
    m_chroma_format(chromaFormat),
    m_width(width),
    m_height(height),
    m_pyd_level(pydLevel)
{
    m_bytes_per_pixel = (m_bitdepth > 8U) ? 2U: 1U;
    if (m_chroma_format == RG16 || m_chroma_format == A8)
    {
        m_input_type = 2U;
    }
    else
    {
        m_input_type = 0U;
    }
}

ImagePyramid::~ImagePyramid()
{
}

void ImagePyramid::pyramid(uint8_t *psrc, uint8_t *pdst, int16_t idx)
{
    int16_t w, h, kh, kw;
    int16_t x, y;
    uint32_t dst;
    int16_t src_width, src_height, src_pitch, dst_width, dst_height, dst_pitch;

    src_width = m_width[idx - 1U];
    src_height = m_height[idx - 1U];
    src_pitch = (m_width[idx - 1U] * m_bytes_per_pixel);
    dst_width = m_width[idx];
    dst_height = m_height[idx];
    dst_pitch = (m_width[idx] * m_bytes_per_pixel);

    for (h = 0; h < dst_height; h++)
    {
        for (w = 0; w < dst_width; w++)
        {
            dst = 0;
            for (kh = 0; kh < 5; kh++)
            {
                for (kw = 0; kw < 5; kw++)
                {
                    x = (2 * w) + kw - 2;
                    y = (2 * h) + kh - 2;
                    if (x < 0)
                    {
                        x = -x;
                    }
                    if (x >= src_width)
                    {
                        x = 2 *(src_width -1) - x;
                    }
                    if (y < 0)
                    {
                        y = -y;
                    }
                    if (y >= src_height)
                    {
                        y = 2 *(src_height -1) - y;
                    }

                    if (m_bytes_per_pixel == 2)
                    {
                        dst += *(uint16_t*)(psrc + (y * src_pitch) + (2 * x)) * Kernel[kh][kw];
                    }
                    else
                    {
                        dst += *(psrc + (y * src_pitch) + x) * Kernel[kh][kw];
                    }
                }
            }
            if (m_bytes_per_pixel == 2)
            {
                *(uint16_t*)(pdst + (h * dst_pitch) + (2 * w)) = (dst + 128) >> 8;
            }
            else
            {
                *(pdst + (h * dst_pitch) + w) = (dst + 128) >> 8;
            }
        }
    }
    return;
}

NvMediaStatus ImagePyramid::createPyramid()
{
    uint16_t i;
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    for (i = 0; i < m_pyd_level; i++)
    {
        if (m_input_type == 0U) // For input pyramid buffers
        {
            if (i == 0)
            {
                m_image_pyramid[i] = new ImageBuffer(m_bitdepth, m_chroma_format, m_width[i], m_height[i]);
            }
            else
            {
                m_image_pyramid[i] = new ImageBuffer(m_bitdepth, YUV_400, m_width[i], m_height[i]);
            }
        }
        else if (m_input_type == 2U)
        {
            m_image_pyramid[i] = new ImageBuffer(m_bitdepth, m_chroma_format, m_width[i], m_height[i]);
        }
        else // For output pyramid buffers
        {
            cerr << "not supported pyramid \n";
            return NVMEDIA_STATUS_ERROR;
	}
        if (m_image_pyramid[i] == NULL)
        {
            cerr << "ImageBuffer class creation failed in image pyramid class \n";
            return NVMEDIA_STATUS_ERROR;
        }
        status = m_image_pyramid[i]->createBuffer();
        if (status != NVMEDIA_STATUS_OK)
        {
            cerr << "image createBuffer failed in image pyramid class \n";
            return status;
        }
    }
    return status;
}

NvMediaStatus ImagePyramid::writePyramid(uint8_t *p_file, bool isPNGfile)
{
    uint8_t *tempBuff[NVMEDIA_IOFA_MAX_PYD_LEVEL] = {NULL};
    uint8_t i;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    status = m_image_pyramid[0]->writeBuffer(p_file, isPNGfile);
    if (status != NVMEDIA_STATUS_OK)
    {
        cerr << "writeBuffer failed in image pyramid class \n";
        goto writePyramidExit;
    }
    tempBuff[0] = new uint8_t[(m_width[0] * m_height[0] * m_bytes_per_pixel)];
    memcpy(tempBuff[0], p_file, (m_width[0] * m_height[0] * m_bytes_per_pixel));
    for (i = 1; i < m_pyd_level; i++)
    {
        tempBuff[i] = new uint8_t[(m_width[i] * m_height[i] * m_bytes_per_pixel)];
        pyramid(tempBuff[i - 1U], tempBuff[i], i);
        status = m_image_pyramid[i]->writeBuffer(tempBuff[i], isPNGfile);
        if (status != NVMEDIA_STATUS_OK)
        {
            cerr << "writeBuffer failed in image pyramid class \n";
            goto writePyramidExit;
        }
    }

writePyramidExit:
    for (i = 0; i < m_pyd_level; i++)
    {
        delete [] tempBuff[i];
    }
    return status;
}

void ImagePyramid::destroyPyramid()
{
    uint8_t i;
    for (i = 0; i < m_pyd_level; i++)
    {
        if (m_image_pyramid[i] != NULL)
        {
            m_image_pyramid[i]->destroyBuffer();
            delete m_image_pyramid[i];
        }
    }
}

