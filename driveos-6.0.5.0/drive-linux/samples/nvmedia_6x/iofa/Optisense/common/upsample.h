/* Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef UPSAMPLE_H
#define UPSAMPLE_H

#include <iostream>
#include "image_buffer.h"
#include "common_defs.h"

class UpSample
{
protected:
    uint32_t m_inWidth;
    uint32_t m_inHeight;
    uint32_t m_outWidth;
    uint32_t m_outHeight;
    uint8_t m_gridSize;
    void *m_outSurface;
public:
    UpSample(uint32_t inWidth, uint32_t inHeight, uint32_t outWidth, uint32_t outHeight, uint8_t gridSize):
        m_inWidth(inWidth),
        m_inHeight(inHeight),
        m_outWidth(outWidth),
        m_outHeight(outHeight),
        m_gridSize(gridSize)
    {
        m_outSurface = NULL;
    }
    virtual ~UpSample() {}
    virtual bool initialize() = 0;
    virtual bool process(Buffer *inpBuffer, Buffer *outBuffer) = 0;
    virtual void release() = 0;
    void *getOutputSurface() { return m_outSurface; }
};

class NNUpSampleStereo : public UpSample
{

public:
    NNUpSampleStereo(uint32_t inWidth, uint32_t inHeight, uint32_t outWidth, uint32_t outHeight, uint8_t gridSize):
         UpSample(inWidth, inHeight, outWidth, outHeight, gridSize)
    {
        m_scale = 1;
    }

    NNUpSampleStereo(uint32_t inWidth, uint32_t inHeight, uint32_t outWidth, uint32_t outHeight, uint8_t gridSize, uint32_t scale):
         UpSample(inWidth, inHeight, outWidth, outHeight, gridSize)
    {
        m_scale = scale;
    }

    bool initialize();
    bool process(Buffer *inpBuffer, Buffer *outBuffer);
    void release();

private:
    bool stereoUpSample(uint16_t *src, uint16_t *dst);
    uint32_t m_scale;
};

class NNUpSampleFlow : public UpSample
{

public:
    NNUpSampleFlow(uint32_t inWidth, uint32_t inHeight, uint32_t outWidth, uint32_t outHeight, uint8_t gridSize):
         UpSample(inWidth, inHeight, outWidth, outHeight, gridSize)
    {

    }
    bool initialize();
    bool process(Buffer *inpBuffer, Buffer *outBuffer);
    void release();

private:
    bool flowUpSample(int16_t *src, int16_t *dst);
};

#endif
