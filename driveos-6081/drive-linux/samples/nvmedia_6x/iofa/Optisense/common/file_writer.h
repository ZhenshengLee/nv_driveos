/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef FILE_WRITER_H
#define FILE_WRITER_H

#include <iostream>
#include <fstream>

#include "image_buffer.h"

using namespace std;

class FileWriter
{
private:
    string  m_filename;
    fstream m_pfile;
public:
    FileWriter(string filename) :
    m_filename(filename)
    {
    }
    bool initialize();
    bool writeOutputBinFile(uint32_t numSurfaces, uint8_t **pBuff_write, uint32_t *size_write);
    bool writeSurface(Buffer *image);
    ~FileWriter()
    {
        m_filename.clear();
        m_pfile.close();
    }
};

#endif
