/* Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef MEDIAN_H
#define MEDIAN_H

#include <iostream>
#include "image_buffer.h"
#include "common_defs.h"

class MedianFilter
{
protected:
    uint32_t m_shape;
    uint32_t m_windowsize;
    uint32_t m_width;
    uint32_t m_height;
    void *m_outSurface;
public:
    MedianFilter(uint32_t shape, uint32_t window_size, uint32_t width, uint32_t height):
        m_shape(shape),
        m_windowsize(window_size),
        m_width(width),
        m_height(height)
    {
        m_outSurface = NULL;
    }
    virtual ~MedianFilter() {}
    virtual bool initialize() = 0;
    virtual bool process(Buffer *inpBuffer, Buffer *outBuffer) = 0;
    virtual void release() = 0;
    void *getOutputSurface() { return m_outSurface; }
};

class StereoMedianFilterCPU : public MedianFilter
{
public:
    StereoMedianFilterCPU(uint32_t shape, uint32_t window_size, uint32_t width, uint32_t height):
         MedianFilter(shape, window_size, width, height)
    {
    }
    bool initialize();
    bool process(Buffer *inpBuffer, Buffer *outBuffer);
    void release();

private:
    bool stereomedian_square(uint16_t *src, int32_t srcpitch,
                                    uint16_t *dst, int32_t dstpitch,
                                    int32_t width, int32_t height,
                                    int32_t ws);
    bool stereomedian_cross(uint16_t *src, int32_t srcpitch,
                                   uint16_t *dst, int32_t dstpitch,
                                   int32_t width, int32_t height,
                                   int32_t ws);
};

class OFMedianFilterCPU : public MedianFilter
{
public:
    OFMedianFilterCPU(uint32_t shape, uint32_t window_size, uint32_t width, uint32_t height):
         MedianFilter(shape, window_size, width, height)
    {

    }
    bool initialize();
    bool process(Buffer *inpBuffer, Buffer *outBuffer);
    void release();

private:
    bool ofmedian_square(int16_t *src, int32_t srcpitch,
                                int16_t *dst, int32_t dstpitch,
                                int32_t width, int32_t height,
                                int32_t ws);
    bool ofmedian_cross(int16_t *src, int32_t srcpitch,
                               int16_t *dst, int32_t dstpitch,
                               int32_t width, int32_t height,
                               int32_t ws);
};

#endif
