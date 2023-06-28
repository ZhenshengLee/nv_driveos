/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */
#ifndef CRC_CLASS_H
#define CRC_CLASS_H

#include "image_buffer.h"

class CRC
{
public:
    CRC(uint32_t CrcPolynomial)
    {
        uint32_t i, j, tempcrc;

        for (i = 0; i <= 255; i++)
        {
            tempcrc = i;
            for (j = 8; j > 0; j--)
            {
                if (tempcrc & 1)
                {
                    tempcrc = (tempcrc >> 1) ^ CrcPolynomial;
                }
                else
                {
                    tempcrc >>= 1;
                }
            }
            m_crcTable[i] = tempcrc;
        }
    };

    ~CRC(){};

    uint32_t getCRC() const
    {
        return m_crcValue;
    };

    void resetCRC()
    {
        m_crcValue = 0;
    };

    bool CalculateCRC(Buffer *image)
    {
        if (image == NULL)
        {
            cerr << "NULL Buffer passed to CalculateCRC \n";
            return false;
        }
        if (image->readBuffer() != NVMEDIA_STATUS_OK)
        {
            cerr << "ReadBuffer failed for image \n";
            return false;
        }
        if(!CalculateCRC(image->getNumOfSurfaces(), image->getWriteBufferPointer(), image->getWriteBufferSize()))
        {
            cerr << "CalculateCRC failed \n";
            return false;
        }
        return true;
    }

    bool CalculateCRC(uint32_t numSurfaces, uint8_t **pBuff_write, uint32_t *size_write)
    {
        uint32_t k, newk = 0;
        bool status;

        for (k = 0; k < numSurfaces; k++)
        {
            newk =  k ? (numSurfaces - k) : k;
            status = CalculateCRC_in(pBuff_write[newk], size_write[newk]);
            if (status != true)
            {
                return status;
            }
        }
        return status;
    }

    bool CalculateCRC_in(const uint8_t* buffer, size_t count)
    {
         uint32_t temp1, temp2;

        //return if count is 0 or  buffer pointer is null
        if ((!count) || (!buffer))
        {
            return false;
        }
        while (count--)
        {
            temp1 = (m_crcValue >> 8) & 0x00FFFFFFL;
            temp2 = m_crcTable[(m_crcValue ^ *buffer++) & 0xFF];
            m_crcValue   = temp1 ^ temp2;
        }

        return true;
    };

protected:
    uint32_t m_crcValue;
    uint32_t m_crcTable[256];
};

class CRCGen : public CRC
{
public:
    CRCGen (string fileWrite, uint32_t CrcPolynomial) :
            CRC(CrcPolynomial)
    {
        m_filename = fileWrite;
    }

    bool fileOpen()
    {
        resetCRC();
        m_file.open(m_filename, std::ios::out);
        if (!m_file.is_open())
        {
            return false;
        }

        return true;
    }

    bool fileWrite()
    {
        if (m_file.is_open())
        {
            m_file << hex << m_crcValue << "\n";
            resetCRC();
            return true;
        }
        return false;
    }

    void fileClose()
    {
        if (m_file.is_open())
        {
            m_file.close();
        }
    }

private:
    string m_filename;
    std::ofstream m_file;
};

class CRCCmp : public CRC
{
public:
    CRCCmp(string fileRead, uint32_t CrcPolynomial) :
            CRC(CrcPolynomial)
    {
        m_filename = fileRead;
    }

    bool fileOpen()
    {
        resetCRC();
        m_file.open(m_filename, std::ios::in);
        if (!m_file.is_open())
        {
            return false;
        }

        return true;
    }

    bool CmpCRC()
    {
        if (m_file.is_open())
        {
            m_file >> hex >> m_crcGolden;
            if (m_crcValue != m_crcGolden)
            {
                return false;
            }

            resetCRC();
            return true;
        }
        return false;
    }

    void fileClose()
    {
        if (m_file.is_open())
        {
            m_file.close();
        }
    }

private:
    string m_filename;
    std::ifstream m_file;
    uint32_t m_crcGolden;
};

#endif

