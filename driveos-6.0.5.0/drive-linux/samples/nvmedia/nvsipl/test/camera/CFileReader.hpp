/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


/* STL Headers */
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <vector>

#include "NvSIPLPipelineMgr.hpp" // Pipeline manager
#include "NvSIPLDeviceBlockInfo.hpp"
#include "nvscibuf.h"
#include "CUtils.hpp"

#ifndef CFILEREADER_HPP
#define CFILEREADER_HPP

#define TSC_33_MS (1043829U)
#define TSC_SOF_EOF_DIFF (987696U)

using namespace std;
using namespace nvsipl;

/*** CFileReader class. Used as source of buffer for ISP stand alone processing */
class CFileReader: public NvSIPLImageGroupWriter
{
public:

    virtual ~CFileReader() {
        Deinit();
    }

    SIPLStatus Init(std::vector<std::string> inputRawFiles,
                    const SensorInfo::VirtualChannelInfo& vcinfo,
                    std::atomic<bool> *quitflag)
    {
        for(auto rawFile : inputRawFiles) {
            FILE *handle = fopen(rawFile.c_str(), "rb");
            if (handle == NULL) {
                LOG_ERR("Failed to open file %s\n", rawFile.c_str());
                return NVSIPL_STATUS_BAD_ARGUMENT;
            }
            m_vFileHandles.push_back(handle);
        }

        m_vInputRawFiles = inputRawFiles;

        if (quitflag == nullptr) {
            LOG_ERR("Invalid quit flag\n");
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }

        if (vcinfo.inputFormat == NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW8) {
            m_bytesPerPixel = 1;
        } else if ((vcinfo.inputFormat == NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW10) or
                   (vcinfo.inputFormat == NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12) or
                   (vcinfo.inputFormat == NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW14) or
                   (vcinfo.inputFormat == NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW16)) {
            m_bytesPerPixel = 2;
        } else if (vcinfo.inputFormat == NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW20) {
            m_bytesPerPixel = 4;
        } else {
            LOG_ERR("Unsupported input format\n");
            return NVSIPL_STATUS_NOT_SUPPORTED;
        }

        auto surfwidth = vcinfo.resolution.width;
        auto surfheight = vcinfo.embeddedTopLines + vcinfo.resolution.height + vcinfo.embeddedBottomLines;
        auto surfSize = surfwidth * surfheight * m_bytesPerPixel;
        m_vBuff.resize(surfSize);
        m_quit = quitflag;

        return NVSIPL_STATUS_OK;
    }

    void Deinit(void) {
        m_vBuff.resize(0);
        for(auto &handle : m_vFileHandles) {
            if (handle != NULL) {
                fclose(handle);
            }
        }
        m_vFileHandles.resize(0);
        return;
    }

    //! Implement the callback function
    SIPLStatus FillRawBuffer(RawBuffer& oRawBuffer) final
    {
        auto bufPtr = oRawBuffer.image;
        BufferAttrs bufAttrs;
        auto status = PopulateBufAttr(bufPtr, bufAttrs);
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("PopulateBufAttr failed\n");
            *m_quit = true;
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }
        auto pitch = bufAttrs.planeWidths[0] * m_bytesPerPixel; // Assuming no padding
        auto height = bufAttrs.planeHeights[0];
        auto imageSize = pitch * height;

        // validate surface size
        if (imageSize != m_vBuff.size()) {
            LOG_ERR("imageSize(%u) does not match expected surface size(%u)\n", imageSize, m_vBuff.size());
            *m_quit = true;
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }

        while (fread(m_vBuff.data(), imageSize, 1, m_vFileHandles[m_fileIndex]) != 1) {
            if (!feof(m_vFileHandles[m_fileIndex])) {
                LOG_ERR("Error reading file: %s\n", m_vInputRawFiles[m_fileIndex].c_str());
                *m_quit = true;
                return NVSIPL_STATUS_ERROR;
            } else {
                LOG_MSG("End of file reached for: %s\n", m_vInputRawFiles[m_fileIndex].c_str());
                m_fileIndex++;
                if (m_fileIndex >= m_vFileHandles.size()) {
                    *m_quit = true;
                    return NVSIPL_STATUS_EOF;
                }
                oRawBuffer.discontinuity = true;
            }
        }

        uint32_t formatPitch[MAX_NUM_SURFACES]= {pitch, 0, 0};
        uint32_t size[MAX_NUM_SURFACES]= {imageSize, 0, 0};
        uint8_t *pFormatBuff[MAX_NUM_SURFACES] = {m_vBuff.data(), NULL, NULL};
        auto nvsci_err = NvSciBufObjPutPixels(bufPtr, NULL, (const void **) pFormatBuff, size, formatPitch);
        if (nvsci_err != NvSciError_Success) {
            LOG_ERR("NvSciBufObjPutPixels failed\n");
            *m_quit = true;
            return NVSIPL_STATUS_ERROR;
        }
        /** In case of applications where we need to be able to have a unique identifier
         *  across Engines we can populate capture timestamp in
         *  RawBuffer struct's members frameCaptureTSC and frameCaptureStartTSC
         *  and this will be reflected in the ImageMetaData
         *  Adding TSC_33_MS per frame to mimic TSC SOF timestamp
         *  Adding TSC_SOF_EOF_DIFF per frame to TSC SOF to mimic TSC EOF Timestamp
         */
        oRawBuffer.frameCaptureStartTSC = m_timestamp;
        oRawBuffer.frameCaptureTSC = m_timestamp + TSC_SOF_EOF_DIFF;
        m_timestamp += TSC_33_MS;

        if (bufAttrs.needSwCacheCoherency) {
            nvsci_err = NvSciBufObjFlushCpuCacheRange(bufPtr, 0, bufAttrs.planePitches[0] * height);
            if (nvsci_err != NvSciError_Success) {
                LOG_ERR("NvSciBufObjFlushCpuCacheRange failed\n");
                return NVSIPL_STATUS_ERROR;
            }
        }

        LOG_INFO("Fed frame: %u\n", m_frames++);

        return NVSIPL_STATUS_OK;
    }

private:
    std::vector<std::string> m_vInputRawFiles;
    std::vector<FILE *> m_vFileHandles;
    std::atomic<bool> *m_quit = nullptr;
    std::vector<uint8_t> m_vBuff;
    uint32_t m_bytesPerPixel;
    uint32_t m_frames = 0U;
    uint32_t m_fileIndex = 0U;
    uint64_t m_timestamp = 0U;
};

#endif //CFILEREADER_HPP
