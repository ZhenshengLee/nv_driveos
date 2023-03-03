/* Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "median.h"

bool StereoMedianFilterCPU::initialize()
{
    m_outSurface = new uint16_t[m_width * m_height];
    if (m_outSurface == NULL)
    {
        cerr << "Median filter out surface allocation failed\n";
        return false;
    }
    memset(m_outSurface, 0x0000, sizeof(uint16_t) * m_width * m_height);

    return true;
}

void StereoMedianFilterCPU::release()
{
    if (m_outSurface != NULL)
    {
        uint16_t *ptr = static_cast <uint16_t*>(m_outSurface);
        delete[] ptr;
    }
}

bool StereoMedianFilterCPU::process(Buffer *inpBuffer,Buffer *outBuffer)
{
    NvMediaStatus status;
    bool ret;

    if ((inpBuffer == NULL) || (outBuffer ==NULL))
    {
        cerr << "NULL input and output image \n";
        return false;
    }

    if (inpBuffer->readBuffer() != NVMEDIA_STATUS_OK)
    {
        cerr << "ReadBuffer failed for image \n";
        return false;
    }
    if (inpBuffer->getNumOfSurfaces() != 1)
    {
        cerr << "invalid number of surface \n";
        return false;
    }

    uint8_t **pList = inpBuffer->getWriteBufferPointer();
    uint16_t *pSrc = reinterpret_cast<uint16_t *>(pList[0]);
    uint16_t *pDst = reinterpret_cast<uint16_t *>(m_outSurface);

    if (m_shape == 2)
    {
         ret = stereomedian_cross(pSrc, m_width, pDst, m_width, m_width, m_height, m_windowsize);
    }
    else
    {
         ret = stereomedian_square(pSrc, m_width, pDst, m_width, m_width, m_height, m_windowsize);
    }
    if (!ret)
    {
        cerr << "stereo median processing failed \n";
        return false;
    }

    status = outBuffer->writeBuffer(reinterpret_cast<uint8_t *>(m_outSurface), false);
    if (status != NVMEDIA_STATUS_OK)
    {
        cerr << "stereo median write buffer failed \n";
        return false;
    }

    return true;
}

//Support 3x3 and 5x5 square shape median filter
bool StereoMedianFilterCPU::stereomedian_square(uint16_t *src, int32_t srcpitch,
                                                       uint16_t *dst, int32_t dstpitch,
                                                       int32_t width, int32_t height,
                                                       int32_t ws)
{

    uint16_t neighbours[5*5];
    int32_t  wh = ws/2;

    for (int32_t j = 0; j < height; j++)
    {
        for (int32_t i = 0; i < width; i++)
        {
            int32_t w = 0;
            for (int32_t wy = -1*wh; wy <= wh; wy++)
            {
                for (int32_t wx = -1*wh; wx <= wh; wx++)
                {
                    int32_t y = j + wy;
                    if (y < 0)
                    {
                        y = 0;
                    }
                    if (y >= height)
                    {
                        y = (height -1);
                    }

                    int32_t x = i + wx;
                    if (x < 0)
                    {
                        x = 0;
                    }
                    if (x >= width)
                    {
                        x = (width -1);
                    }
                    neighbours[w++] = *(src + y*srcpitch + x);
                }
            }

            if (w != ws*ws)
            {
               return false;
            }
            for(int32_t k = 0; k <= w/2; k++)
            {
                for(int32_t l = 0; l < w-k-1; l++)
                {
                    if (neighbours[l] > neighbours[l+1])
                    {
                        uint16_t temp;
                        temp            = neighbours[l];
                        neighbours[l]   = neighbours[l+1];
                        neighbours[l+1] = temp;
                    }
                }
            }
            *(dst + j*dstpitch + i) = neighbours[w/2];
        }
    }

    return true;
}

//Support 3x3 and 5x5 cross shape median filter
bool StereoMedianFilterCPU::stereomedian_cross(uint16_t *src, int32_t srcpitch,
                                                      uint16_t *dst, int32_t dstpitch,
                                                      int32_t width, int32_t height,
                                                      int32_t ws)
{
    uint16_t neighbours[3*3];
    int32_t  wh = ws/2;

    for (int32_t j = 0; j < height; j++)
    {
        for (int32_t i = 0; i < width; i++)
        {
            int32_t w = 0;
            for (int32_t wy = -1*wh; wy <= wh; wy++)
            {
                for (int32_t wx = -1*wh; wx <= wh; wx++)
                {
                    if ((wy != 0) && (wx != 0))
                    {
                        continue;
                    }
                    int32_t y = j + wy;
                    if (y < 0)
                    {
                        y = 0;
                    }
                    if (y >= height)
                    {
                        y = (height -1);
                    }

                    int32_t x = i + wx;
                    if (x < 0)
                    {
                        x = 0;
                    }
                    if (x >= width)
                    {
                        x = (width -1);
                    }
                    neighbours[w++] = *(src + y*srcpitch + x);
                }
            }

            if (w != 2*ws-1)
            {
               return false;
            }
            for(int32_t k = 0; k <= w/2; k++)
            {
                for(int32_t l = 0; l < w-k-1; l++)
                {
                    if (neighbours[l] > neighbours[l+1])
                    {
                        uint16_t temp;
                        temp            = neighbours[l];
                        neighbours[l]   = neighbours[l+1];
                        neighbours[l+1] = temp;
                    }
                }
            }
            *(dst + j*dstpitch + i) = neighbours[w/2];
        }
    }

    return true;
}

bool OFMedianFilterCPU::initialize()
{
    m_outSurface = new int16_t[2*m_width * m_height];
    if (m_outSurface == NULL)
    {
        cerr << "Median filter out surface allocation failed\n";
        return false;
    }
    memset(m_outSurface, 0x0000, 2 * sizeof(int16_t) * m_width * m_height);

    return true;
}

void OFMedianFilterCPU::release()
{
    if (m_outSurface != NULL)
    {
        int16_t *ptr = static_cast <int16_t*>(m_outSurface);
        delete[] ptr;
    }
}

bool OFMedianFilterCPU::process(Buffer *inpBuffer,Buffer *outBuffer)
{
    NvMediaStatus status;
    bool ret;

    if ((inpBuffer == NULL) || (outBuffer ==NULL))
    {
        cerr << "NULL input and output image \n";
        return false;
    }

    if (inpBuffer->readBuffer() != NVMEDIA_STATUS_OK)
    {
        cerr << "ReadBuffer failed for image \n";
        return false;
    }
    if (inpBuffer->getNumOfSurfaces() != 1)
    {
        cerr << "invalid number of surface \n";
        return false;
    }

    uint8_t **pList = inpBuffer->getWriteBufferPointer();
    int16_t *pSrc = reinterpret_cast<int16_t *>(pList[0]);
    int16_t *pDst = reinterpret_cast<int16_t *>(m_outSurface);
    if (pSrc == NULL)
    {
        cerr << "median source pointer null \n";
        return false;
    }
    if (m_shape == 2)
    {
        ret = ofmedian_cross(pSrc, 2*m_width, pDst, 2*m_width, m_width, m_height, m_windowsize);
    }
    else
    {
        ret = ofmedian_square(pSrc, 2*m_width, pDst, 2*m_width, m_width, m_height, m_windowsize);
    }
    if (!ret)
    {
        cerr << "optical flow median processing failed \n";
        return false;
    }

    status = outBuffer->writeBuffer(reinterpret_cast<uint8_t *>(m_outSurface), false);
    if (status != NVMEDIA_STATUS_OK)
    {
        cerr << "optical flow median write buffer failed \n";
        return false;
    }

    return true;
}


//Support 3x3 and 5x5 square median filter
bool OFMedianFilterCPU::ofmedian_square(int16_t *src, int32_t srcpitch,
                                               int16_t *dst, int32_t dstpitch,
                                               int32_t width, int32_t height,
                                               int32_t ws)
{

    int16_t neighbours[5*5];
    int32_t  wh = ws/2;

    for (int32_t j = 0; j < height; j++)
    {
        for (int32_t i = 0; i < width; i++)
        {
            for (int32_t com = 0; com < 2; com++)
            {
                int32_t w = 0;
                for (int32_t wy = -1*wh; wy <= wh; wy++)
                {
                    for (int32_t wx = -1*wh; wx <= wh; wx++)
                    {
                       int32_t y = j + wy;
                       if (y < 0)
                       {
                           y = 0;
                       }
                       if (y >= height)
                       {
                           y = (height -1);
                       }

                       int32_t x = i + wx;
                       if (x < 0)
                       {
                           x = 0;
                       }
                       if (x >= width)
                       {
                           x = (width -1);
                       }
                       neighbours[w++] = *(src + y*srcpitch + 2*x + com);
                    }
                }

                if (w != ws*ws)
                {
                    return false;
                }
                for(int32_t k = 0; k <= w/2; k++)
                {
                    for(int32_t l = 0; l < w-k-1; l++)
                    {
                        if (neighbours[l] > neighbours[l+1])
                        {
                            int16_t temp;
                            temp            = neighbours[l];
                            neighbours[l]   = neighbours[l+1];
                            neighbours[l+1] = temp;
                        }
                    }
                }
                *(dst + j*dstpitch + 2*i + com) = neighbours[w/2];
            }
        }
    }

    return true;
}

//Support 3x3 and 5x5 cross median filter
bool OFMedianFilterCPU::ofmedian_cross(int16_t *src, int32_t srcpitch,
                                              int16_t *dst, int32_t dstpitch,
                                              int32_t width, int32_t height,
                                              int32_t ws)
{

    int16_t neighbours[9];
    int32_t  wh = ws/2;

    for (int32_t j = 0; j < height; j++)
    {
        for (int32_t i = 0; i < width; i++)
        {
            for (int32_t com = 0; com < 2; com++)
            {
                int32_t w = 0;
                for (int32_t wy = -1*wh; wy <= wh; wy++)
                {
                    for (int32_t wx = -1*wh; wx <= wh; wx++)
                    {
                       if ((wy != 0) && (wx != 0))
                       {
                           continue;
                       }
                       int32_t y = j + wy;
                       if (y < 0)
                       {
                           y = 0;
                       }
                       if (y >= height)
                       {
                           y = (height -1);
                       }

                       int32_t x = i + wx;
                       if (x < 0)
                       {
                           x = 0;
                       }
                       if (x >= width)
                       {
                           x = (width -1);
                       }
                       neighbours[w++] = *(src + y*srcpitch + 2*x + com);
                    }
                }

                if (w != 2*ws-1)
                {
                    return false;
                }
                for(int32_t k = 0; k <= w/2; k++)
                {
                    for(int32_t l = 0; l < w-k-1; l++)
                    {
                        if (neighbours[l] > neighbours[l+1])
                        {
                            int16_t temp;
                            temp            = neighbours[l];
                            neighbours[l]   = neighbours[l+1];
                            neighbours[l+1] = temp;
                        }
                    }
                }
                *(dst + j*dstpitch + 2*i + com) = neighbours[w/2];
            }
        }
    }

    return true;
}

