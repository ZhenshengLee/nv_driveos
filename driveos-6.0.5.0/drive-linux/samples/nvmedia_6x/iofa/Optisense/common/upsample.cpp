/* Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "upsample.h"

bool NNUpSampleStereo::initialize()
{
    m_outSurface = new uint16_t[m_outWidth * m_outHeight];
    if (m_outSurface == NULL)
    {
        cerr << "NNUpSampleStereo out surface allocation failed\n";
        return false;
    }
    memset(m_outSurface, 0x0000, sizeof(uint16_t) * m_outWidth * m_outHeight);

    return true;
}

void NNUpSampleStereo::release()
{
    if (m_outSurface != NULL)
    {
        uint16_t *ptr = static_cast <uint16_t*>(m_outSurface);
        delete[] ptr;
    }
}

bool NNUpSampleStereo::process(Buffer *inpBuffer,Buffer *outBuffer)
{
    NvMediaStatus status;
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
    if (!stereoUpSample(pSrc,pDst))
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

bool NNUpSampleStereo::stereoUpSample(uint16_t *src, uint16_t *dst)
{
    uint32_t x, y, xin, yin, offset = 0;

    if (m_gridSize > 0)
    {
        offset = (1<<m_gridSize)/2 -1;
    }
    for (y=0; y<m_outHeight; y++)
    {
        for (x=0; x<m_outWidth; x++)
        {
            yin = (y+offset)>>m_gridSize;
            xin = (x+offset)>>m_gridSize;

            if (xin >= m_inWidth)
            {
                xin = m_inWidth -1;
            }
            if (yin >= m_inHeight)
            {
                yin = m_inHeight -1;
            }
            *(dst + y*m_outWidth + x) = *(src + yin*m_inWidth + xin) * m_scale;
        }
    }

    return true;
}

bool NNUpSampleFlow::initialize()
{
    m_outSurface = new int16_t[2 * m_outWidth * m_outHeight];
    if (m_outSurface == NULL)
    {
        cerr << "NNUpSampleFlow out surface allocation failed\n";
        return false;
    }
    memset(m_outSurface, 0x0000, sizeof(int16_t) * 2 * m_outWidth * m_outHeight);

    return true;
}

void NNUpSampleFlow::release()
{
    if (m_outSurface != NULL)
    {
        int16_t *ptr = static_cast <int16_t*>(m_outSurface);
        delete[] ptr;
    }
}

bool NNUpSampleFlow::process(Buffer *inpBuffer,Buffer *outBuffer)
{
    NvMediaStatus status ;
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
    if (!flowUpSample(pSrc,pDst))
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

bool NNUpSampleFlow::flowUpSample(int16_t *src,int16_t *dst)
{
    uint32_t x, y, xin, yin, offset = 0;

    if (m_gridSize > 0)
    {
        offset = (1<<m_gridSize)/2 -1;
    }

    for (y=0; y<m_outHeight; y++)
    {
        for (x=0; x<m_outWidth; x++)
        {
            yin = (y+offset)>>m_gridSize;
            xin = (x+offset)>>m_gridSize;

            if (xin >= m_inWidth)
            {
                xin = m_inWidth -1;
            }
            if (yin >= m_inHeight)
            {
                yin = m_inHeight -1;
            }

            *(dst + 2*y*m_outWidth + 2*x) = *(src + 2*yin*m_inWidth + 2*xin);
            *(dst + 2*y*m_outWidth + 2*x + 1) = *(src + 2*yin*m_inWidth + 2*xin + 1);
        }
    }

    return true;
}

