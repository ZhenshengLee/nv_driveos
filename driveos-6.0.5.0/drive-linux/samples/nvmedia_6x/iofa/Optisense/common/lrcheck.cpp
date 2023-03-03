/* Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "lrcheck.h"

bool LRCheckStereo::initialize()
{
    m_outSurface = new uint16_t[m_inWidth * m_inHeight];
    if (m_outSurface == NULL)
    {
        cerr << "LR check surface allocation failed\n";
        return false;
    }
    memset(m_outSurface, 0x0000, sizeof(uint16_t) * m_inWidth * m_inHeight);

    return true;
}

void LRCheckStereo::release()
{
    if (m_outSurface != NULL)
    {
        uint16_t *ptr = static_cast <uint16_t*>(m_outSurface);
        delete[] ptr;
    }
}

bool LRCheckStereo::process(Buffer *leftDisp, Buffer *rightDisp, Buffer *outBuffer, uint16_t threshold)
{
    NvMediaStatus status ;
    if ((leftDisp == NULL) || (rightDisp == NULL) || (outBuffer ==NULL))
    {
        cerr << "NULL input and output image \n";
        return false;
    }

    if ((leftDisp->readBuffer() != NVMEDIA_STATUS_OK) || (rightDisp->readBuffer() != NVMEDIA_STATUS_OK))
    {
        cerr << "ReadBuffer failed for image \n";
        return false;
    }
    if ((leftDisp->getNumOfSurfaces() != 1) || (rightDisp->getNumOfSurfaces() != 1))
    {
        cerr << "invalid number of surface \n";
        return false;
    }

    uint8_t **pList1 = leftDisp->getWriteBufferPointer();
    uint16_t *pSrc1 = reinterpret_cast<uint16_t *>(pList1[0]);
    uint8_t **pList2 = rightDisp->getWriteBufferPointer();
    uint16_t *pSrc2 = reinterpret_cast<uint16_t *>(pList2[0]);
    uint16_t *pDst = reinterpret_cast<uint16_t *>(m_outSurface);

    stereoLRCheck(pSrc1, pSrc2, pDst, threshold);

    status = outBuffer->writeBuffer(reinterpret_cast<uint8_t *>(m_outSurface), false);
    if (status != NVMEDIA_STATUS_OK)
    {
        cerr << "LRCheck write buffer failed \n";
        return false;
    }

    return true;
}

//The following API support LR consistency check. RL consistency check is not supported.
void LRCheckStereo::stereoLRCheck(uint16_t *src_leftDisp, uint16_t *src_rightDisp, uint16_t *dst, uint16_t threshold)
{
    uint16_t leftDisp_value,rightDisp_value;
    for (uint32_t j = 0; j < m_inHeight; j++)
    {
        for (uint32_t i = 0; i < m_inWidth; i++)
        {
            leftDisp_value = (*(src_leftDisp + j*m_inWidth + i) + 15) >> 5;
            if ((i-leftDisp_value) < 0)
            {
                *(dst + j*m_inWidth + i) = 0;
                continue;
            }
            rightDisp_value = (*(src_rightDisp + j*m_inWidth + i - leftDisp_value) + 15) >> 5;
            if((abs(leftDisp_value-rightDisp_value) > threshold) || (rightDisp_value==0))
            {
                *(dst + j*m_inWidth + i) = 0;
                continue;
            }
            *(dst + j*m_inWidth + i) = *(src_leftDisp + j*m_inWidth + i);
        }
    }
}

bool FBCheckFlow::initialize()
{
    m_outSurface = new int16_t[2*m_inWidth * m_inHeight];
    if (m_outSurface == NULL)
    {
        cerr << "FB check out surface allocation failed\n";
        return false;
    }
    memset(m_outSurface, 0x0000, 2 * sizeof(int16_t) * m_inWidth * m_inHeight);

    return true;
}

void FBCheckFlow::release()
{
    if (m_outSurface != NULL)
    {
        int16_t *ptr = static_cast <int16_t*>(m_outSurface);
        delete[] ptr;
    }
}

bool FBCheckFlow::process(Buffer *fwdFlow, Buffer *bwdFlow, Buffer *outBuffer, uint16_t threshold)
{
    NvMediaStatus status ;
    if ((fwdFlow == NULL) || (bwdFlow == NULL) || (outBuffer ==NULL))
    {
        cerr << "NULL input and output image \n";
        return false;
    }

    if ((fwdFlow->readBuffer() != NVMEDIA_STATUS_OK) || (bwdFlow->readBuffer() != NVMEDIA_STATUS_OK))
    {
        cerr << "ReadBuffer failed for image \n";
        return false;
    }
    if ((fwdFlow->getNumOfSurfaces() != 1) || (bwdFlow->getNumOfSurfaces() != 1))
    {
        cerr << "invalid number of surface \n";
        return false;
    }

    uint8_t **pList1 = fwdFlow->getWriteBufferPointer();
    int16_t *pSrc1 = reinterpret_cast<int16_t *>(pList1[0]);
    uint8_t **pList2 = bwdFlow->getWriteBufferPointer();
    int16_t *pSrc2 = reinterpret_cast<int16_t *>(pList2[0]);
    int16_t *pDst = reinterpret_cast<int16_t *>(m_outSurface);

    flowFBCheck(pSrc1, pSrc2, pDst, threshold);

    status = outBuffer->writeBuffer(reinterpret_cast<uint8_t *>(m_outSurface), false);
    if (status != NVMEDIA_STATUS_OK)
    {
        cerr << "FBCheck write buffer failed \n";
        return false;
    }

    return true;
}

//The following API support Fwd bwd  consistency check. 
void FBCheckFlow::flowFBCheck(int16_t *src_fwdFlow, int16_t *src_bwdFlow, int16_t *dst, uint16_t threshold)
{
    int16_t fwdFlow_valuex, fwdFlow_valuey, bwdFlow_valuex, bwdFlow_valuey;
    uint32_t srcpitch = 2*m_inWidth;
    int16_t dstx, dsty, newx, newy;
    for (int16_t j = 0; j < (int16_t)m_inHeight; j++)
    {
        for (int16_t i = 0; i < (int16_t)m_inWidth; i++)
        {
            fwdFlow_valuex = (*(src_fwdFlow + j*srcpitch + 2*i) + 15) >> 5;
            fwdFlow_valuey = (*(src_fwdFlow + j*srcpitch + 2*i + 1) + 15) >> 5;
            dstx = i + fwdFlow_valuex;
            dsty = j + fwdFlow_valuey;
            if (dstx < 0 || dsty < 0 || dstx >(int16_t)(m_inWidth - 1) || dsty >(int16_t)(m_inHeight - 1))
            {
                *(dst + j*srcpitch + 2*i) = UNKNOWN_FLOW_THRESH;
                *(dst + j*srcpitch + 2*i +1) = UNKNOWN_FLOW_THRESH;
                continue;
            }
            bwdFlow_valuex = (*(src_bwdFlow + dsty*srcpitch + 2*dstx) + 15) >> 5;
            bwdFlow_valuey = (*(src_bwdFlow + dsty*srcpitch + 2*dstx +1) + 15) >> 5;
            newx = dstx + bwdFlow_valuex;
            newy = dsty + bwdFlow_valuey;
            if (newx < 0 || newy < 0 || newx >(int16_t)(m_inWidth - 1) || newy >(int16_t)(m_inHeight - 1))
            {
                *(dst + j*srcpitch + 2*i) = UNKNOWN_FLOW_THRESH;
                *(dst + j*srcpitch + 2*i +1) = UNKNOWN_FLOW_THRESH;
                continue;
            }
            if((abs(i-newx) > threshold) || (abs(j-newy) >threshold))
            {
               *(dst + j*srcpitch + 2*i) = UNKNOWN_FLOW_THRESH;
               *(dst + j*srcpitch + 2*i +1) = UNKNOWN_FLOW_THRESH;
                continue;
            }

            *(dst + j*srcpitch + 2*i) = *(src_fwdFlow + j*srcpitch + 2*i);
            *(dst + j*srcpitch + 2*i +1) = *(src_fwdFlow + j*srcpitch + 2*i + 1);
        }
    }
}


