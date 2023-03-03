/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "file_writer.h"

bool FileWriter::initialize()
{
    m_pfile.open(m_filename, ios::out | ios::binary);
    if (!m_pfile.is_open())
    {
        cerr << "Opening of out file for writing failed\n";
        return false;
    }
    return true;
}
bool FileWriter::writeSurface(Buffer *image)
{
    if (image == NULL)
    {
        cerr << "NULL Buffer passed to writeSurface \n";
        return false;
    }
    if (image->readBuffer() != NVMEDIA_STATUS_OK)
    {
        cerr << "ReadBuffer failed for image \n";
        return false;
    }
    if (!writeOutputBinFile(image->getNumOfSurfaces(), image->getWriteBufferPointer(), image->getWriteBufferSize()))
    {
        cerr << "writeOutputBinFile failed for writeSurface surface\n";
        return false;
    }

    return true;
}

bool FileWriter::writeOutputBinFile(uint32_t numSurfaces, uint8_t **pBuff_write, uint32_t *size_write)
{
    bool status = true;
    uint32_t k, newk = 0;

    for (k = 0; k < numSurfaces; k++)
    {
        newk =  k ? (numSurfaces - k) : k;
        if (!m_pfile.write(reinterpret_cast<char *> (pBuff_write[newk]), size_write[newk]))
        {
            cerr << "File write falied\n";
            status = false;
            break;
        }
    }

    return status;
}

