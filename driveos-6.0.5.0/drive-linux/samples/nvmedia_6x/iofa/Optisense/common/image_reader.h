/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef IMAGE_READER_H
#define IMAGE_READER_H

#include <fstream>
#include <iostream>

#include "common_defs.h"
#include <string.h>
#if PNG_SUPPORT
#include "lodepng.h"
#endif
using namespace std;

class ImageReader
{
protected:
    ChromaFormat m_chroma_format;
    string m_file_name;
    ifstream m_file;
    uint32_t m_height;
    uint32_t m_width;
    uint32_t m_bitdepth;
    uint32_t m_image_size;
    uint32_t m_num_of_frames;
    uint8_t m_frame_num;
    uint8_t *m_buffer;
    uint8_t *m_pFrameData;

public:
    virtual bool read_file() = 0;
    ImageReader(uint32_t height, uint32_t width, uint32_t bitdepth, ChromaFormat chroma_format, string file_name, uint32_t frame_num);
    virtual uint8_t *current_item() = 0;
    virtual bool initialize() = 0;
    uint32_t getHeight() {return m_height; }
    uint32_t getWidth() {return m_width; }
    uint32_t getBitdepth() {return m_bitdepth; }
    virtual ~ImageReader() {}
};

class YUVReader : public ImageReader
{
private:
    uint8_t     m_readIdx;
public:
    YUVReader(uint32_t height, uint32_t width, uint32_t bitdepth, ChromaFormat chroma_format, string file_name, uint32_t frame_num);
    void calc_image_size();
    void calc_num_of_frames();
    bool read_file();
    bool initialize() override;
    uint8_t *current_item() override
    {
        return m_pFrameData;
    }
    ~YUVReader();
};
#if PNG_SUPPORT
class PNGReader : public ImageReader
{
private:
    uint8_t     m_readIdx;
    bool getPNGInfo();
    bool pngRead();
public:
    PNGReader(uint32_t height, uint32_t width, uint32_t bitdepth, ChromaFormat chroma_format, string file_name);
    ~PNGReader();
    bool read_file() override;
    bool initialize() override;
    uint8_t *current_item() override
    {
        return m_pFrameData;
    }
};
#endif
#endif
