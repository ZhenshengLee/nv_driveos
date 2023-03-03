/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "image_reader.h"
#include "math.h"

ImageReader::ImageReader(uint32_t height, uint32_t width, uint32_t bitdepth, ChromaFormat chroma_format, string file_name, uint32_t frame_num) :
    m_chroma_format(chroma_format),
    m_file_name(file_name),
    m_height(height), // initialization list
    m_width(width),
    m_bitdepth(bitdepth),

    m_frame_num(frame_num)
{
}

YUVReader::YUVReader(uint32_t height, uint32_t width, uint32_t bitdepth, ChromaFormat chroma_format, string file_name, uint32_t frame_num) : ImageReader(height, width, bitdepth, chroma_format, file_name, frame_num)
{
    m_readIdx = 0;
    calc_image_size();
    calc_num_of_frames();
}

YUVReader::~YUVReader()
{
    delete[] m_pFrameData;
    m_file.close();
}

bool YUVReader::initialize()
{
    m_file.open(m_file_name, std::ios::in | std::ios::binary);
    if (m_file.is_open())
    {
        m_file.seekg(m_frame_num * m_image_size, ios::beg);
    }
    else
    {
        cerr << "Opening of YUV file failed\n";
        return false;
    }
    m_pFrameData = new (nothrow) uint8_t[m_image_size];
    if (m_pFrameData == NULL)
    {
        cerr << "YUV buffer allocation failed\n";
        return false;
    }
    memset(m_pFrameData, 0, m_image_size);
    return true;
}

bool YUVReader::read_file()
{
    if (m_readIdx > m_num_of_frames)
    {
        cout << m_file_name << " yuv File fully processed so \n";
        return false;
    }
    m_file.read(reinterpret_cast<char *>(m_pFrameData), m_image_size);
    if (m_file)
    {
	    m_readIdx++;
      	return true;
    }
    else
	{
       return false;
	}
}

void YUVReader::calc_image_size()
{
    switch (m_chroma_format)
    {
    case YUV_400:
        m_image_size = (m_width * m_height);
        break;
    case YUV_420:
        m_image_size = (m_width * m_height * 3 / 2);
        break;
    case YUV_422:
        m_image_size = (m_width * m_height * 2);
        break;
    case YUV_444:
        m_image_size = (m_width * m_height * 3);
        break;
    default:
        m_image_size = (m_width * m_height);
        break;
    }
    if (m_bitdepth > 8)
    {
        m_image_size *= 2;
    }
}
void YUVReader::calc_num_of_frames()
{
    FILE *input_file = NULL;
    uint32_t file_length;
    uint32_t frames_num = 0U;
    if (!m_file_name.empty())
    {
        input_file = fopen(m_file_name.c_str(), "rb");
        if (input_file == NULL)
        {
            cout << "Unable to open input YUV file \n";
            return;
        }
        fseek(input_file, 0, SEEK_END);
        file_length = ftell(input_file);
        fclose(input_file);

        if (file_length == 0U)
        {
            cout << "Zero file length for file " << m_file_name << endl;
            return;
        }
        frames_num = file_length / m_image_size;
        if (frames_num < 2)
        {
            cout << " At least 2 frames are needed for IOFA \n";
            return;
        }
        m_num_of_frames = frames_num - 1;
    }
    else
    {
        m_num_of_frames = 1;
    }
}
#if PNG_SUPPORT

PNGReader::PNGReader(uint32_t height, uint32_t width, uint32_t bitdepth, ChromaFormat chroma_format, string file_name) : ImageReader(height, width, bitdepth, chroma_format, file_name, 0)
{
    m_readIdx = 0;
}

bool PNGReader::read_file()
{
    if (m_readIdx > 0)
    {
        cout << m_file_name << " PNG File already processed \n";
        return false;
    }
    if (!pngRead())
    {
        cerr << m_file_name << " PNG File reading failed \n";
        return false;
    }
    m_readIdx++;
    return true;
}

PNGReader::~PNGReader()
{
    delete[] m_pFrameData;
}

bool PNGReader::getPNGInfo()
{
    unsigned error;
    unsigned char* image;

    // decode
    error = lodepng_decode32_file(&image, &m_width, &m_height, m_file_name.c_str());
    //if there's an error, display it
    if (error)
    {
        cerr << "png decoder error " << error << endl;
        return false;
    }

    free(image);
    cout << "PNG image width: " << m_width << " height: " << m_height << endl ;
    return true;
}

bool PNGReader::initialize()
{
    if (!getPNGInfo())
    {
        cerr << "getPNGInfo failed\n";
        return false;
    }
    m_image_size = m_width*m_height * ((m_bitdepth > 8U) ? 2U : 1U);
    m_pFrameData = new (nothrow) uint8_t[m_image_size];
    if (m_pFrameData == NULL)
    {
        cerr << "PNG buffer allocation failed\n";
        return false;
    }
    memset(m_pFrameData, 0, m_image_size);

    return true;
}

bool PNGReader::pngRead()
{
    uint32_t i, j, offset, pixel;
    uint32_t error;
    unsigned char* image;

    // decode
    error = lodepng_decode32_file(&image, &m_width, &m_height, m_file_name.c_str());
    if (error)
    {
        cerr << "png decoder error " << error << endl;
        return false;
    }

    if (m_bitdepth == 8)
    {
        //Y <- 0.299·R+0.587·G+0.114·B
        for (i=0;  i<m_height; i++)
        {
            for (j=0; j<m_width; j++)
            {
                offset = 4 * (i*m_width + j);
                pixel = round(0.299*image[offset] + 0.587*image[offset+1] + 0.114*image[offset+2]);
                if (pixel > 255)
                {
                    pixel = 255;
                }
                m_pFrameData[i*m_width + j] = static_cast<uint8_t>(pixel);
            }
        }
    }
    else
    {
        cerr << "bitdepth higher than 8 bits are not supported." << endl;
        return false;
    }

    free(image);
    return true;
}
#endif
