/* Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef CONSISTENCY_CHECK_H
#define CONSISTENCY_CHECK_H

#include <iostream>
#include "image_buffer.h"
#include "common_defs.h"

#define UNKNOWN_FLOW_THRESH 32766 
class ConstCheck
{
protected:
    uint32_t m_inWidth;
    uint32_t m_inHeight;
    void *m_outSurface;
public:
    ConstCheck(uint32_t inWidth, uint32_t inHeight):
        m_inWidth(inWidth),
        m_inHeight(inHeight)
    {
        m_outSurface = NULL;
    }
    virtual ~ConstCheck() {}
    virtual bool initialize() = 0;
    virtual bool process(Buffer *leftDisp, Buffer *rightDisp, Buffer *outBuffer, uint16_t threshold) = 0;
    virtual void release() = 0;
    void *getOutputSurface() { return m_outSurface; }
};

class LRCheckStereo : public ConstCheck
{

public:
    LRCheckStereo(uint32_t inWidth, uint32_t inHeight):
        ConstCheck(inWidth, inHeight)
    {
    }
    bool initialize();
    bool process(Buffer *leftDisp, Buffer *rightDisp, Buffer *outBuffer, uint16_t threshold);
    void release();

private:
    void stereoLRCheck(uint16_t *src_leftDisp,uint16_t *src_rightDisp, uint16_t *dst, uint16_t threshold);
    uint32_t m_scale;
};

class FBCheckFlow : public ConstCheck
{

public:
    FBCheckFlow(uint32_t inWidth, uint32_t inHeight):
        ConstCheck(inWidth, inHeight)
    {
    }
    bool initialize();
    bool process(Buffer *fwdFlow, Buffer *bwdFlow, Buffer *outBuffer, uint16_t threshold);
    void release();

private:
    void flowFBCheck(int16_t *src_fwdFlow,int16_t *src_bwdFlow, int16_t *dst, uint16_t threshold);
    uint32_t m_scale;
};



#endif
