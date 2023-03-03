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
#include <csignal>
#include <thread>
#include <chrono>
#include <ctime>
#include <getopt.h>

/* NvSIPL Header for INvSIPLNvMBuffer to Flip */
#include "NvSIPLClient.hpp" // Client
#include "nvscibuf.h"
#include "CUtils.hpp"

#ifndef CFILEWRITER_HPP
#define CFILEWRITER_HPP

using namespace std;
using namespace nvsipl;

/** FileWriter class */
class CFileWriter
{
 public:

    SIPLStatus Init(string sFilename, bool isRawDump)
    {
        m_pOutFile = nullptr;
        remove(sFilename.c_str());
        m_pOutFile = fopen(sFilename.c_str(), "wb");
        if (!m_pOutFile) {
            LOG_ERR("Failed to create output file\n");
            return NVSIPL_STATUS_ERROR;
        }
        m_isRawOutput = isRawDump;
        m_filename = sFilename;

        return NVSIPL_STATUS_OK;
    }

    SIPLStatus WriteBufferToFile(INvSIPLClient::INvSIPLNvMBuffer *pNvmBuf,
                                 NvSciSyncCpuWaitContext cpuWaitContext)
    {
        NvSciError sciErr;
        SIPLStatus status;

        // Wait on EOF fence if its not ICP
        if (!m_isRawOutput) {
            NvSciSyncFence fence = NvSciSyncFenceInitializer;
            status = pNvmBuf->GetEOFNvSciSyncFence(&fence);
            CHK_STATUS_AND_RETURN(status,
                    "INvSIPLClient::INvSIPLNvMBuffer::GetEOFNvSciSyncFence");

            sciErr = NvSciSyncFenceWait(&fence, cpuWaitContext, FENCE_FRAME_TIMEOUT_MS * 1000UL);
            if (sciErr != NvSciError_Success) {
                if (sciErr == NvSciError_Timeout) {
                    LOG_ERR("Frame done NvSciSyncFenceWait timed out\n");
                } else {
                    LOG_ERR("Frame done NvSciSyncFenceWait failed\n");
                }
            }
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncFenceWait Failed");
            NvSciSyncFenceClear(&fence);
        }

        // Write Buffer
        NvSciBufObj bufPtr = pNvmBuf->GetNvSciBufImage();
        BufferAttrs bufAttrs;
        status = PopulateBufAttr(bufPtr, bufAttrs);
        if(status != NVSIPL_STATUS_OK) {
            LOG_ERR("FileWriter: PopulateBufAttr failed\n");
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }

        uint32_t numSurfaces = -1U;
        float *xScalePtr = nullptr, *yScalePtr = nullptr;
        uint32_t *bytesPerPixelPtr = nullptr;
        bool isPackedYUV = false;
        status = GetBuffParams(bufAttrs,
                               &xScalePtr,
                               &yScalePtr,
                               &bytesPerPixelPtr,
                               &numSurfaces,
                               &isPackedYUV);
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("GetBuffParams failed\n");
            return status;
        }

        if (isPackedYUV && m_isRawOutput)
        {
            NvSciError sciErr;
            if ((bufAttrs.needSwCacheCoherency)) {
                uint32_t flushSize = bufAttrs.planePitches[0] * bufAttrs.planeHeights[0];
                sciErr = NvSciBufObjFlushCpuCacheRange(bufPtr, 0U, flushSize);
                CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufObjFlushCpuCacheRange Failed");
            }
            void* va_ptr = nullptr;
            sciErr = NvSciBufObjGetConstCpuPtr(bufPtr, (const void**)&va_ptr);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufObjGetConstCpuPtr Failed");

            uint8_t* basePtr = static_cast<uint8_t*>(va_ptr);
            for (uint32_t j = 0U; j < bufAttrs.planeHeights[0]; j++){
                fwrite(basePtr + j * bufAttrs.planePitches[0],
                       bufAttrs.planeWidths[0] * bytesPerPixelPtr[0],
                       1U,
                       m_pOutFile);
            }
        } else {
            uint32_t pBuffPitches[MAX_NUM_SURFACES] = { 0U };
            uint8_t *pBuff[MAX_NUM_SURFACES] = { 0U };
            uint32_t size[MAX_NUM_SURFACES] = { 0U };
            uint32_t imageSize = 0U;

            uint32_t height = bufAttrs.planeHeights[0];
            uint32_t width = bufAttrs.planeWidths[0];
            for (uint32_t i = 0U; i < numSurfaces; i++) {
                size[i] = (width * xScalePtr[i] * height * yScalePtr[i] * bytesPerPixelPtr[i]);
                imageSize += size[i];
                pBuffPitches[i] = (uint32_t)((float)width * xScalePtr[i]) * bytesPerPixelPtr[i];
            }

            if (m_pBuff == nullptr) {
                m_pBuff = new (std::nothrow) uint8_t[imageSize];
                if (m_pBuff == nullptr) {
                    LOG_ERR("Out of memory\n");
                    return NVSIPL_STATUS_OUT_OF_MEMORY;
                }
                std::fill(m_pBuff, m_pBuff + imageSize, 0x00);
            }

            uint8_t *buffIter = m_pBuff;
            for (uint32_t i = 0U; i < numSurfaces; i++) {
                pBuff[i] = buffIter;
                buffIter += (uint32_t)(height * yScalePtr[i] * pBuffPitches[i]);
            }

            if ((bufAttrs.needSwCacheCoherency) && (m_isRawOutput)) {
                sciErr = NvSciBufObjFlushCpuCacheRange(bufPtr, 0U, bufAttrs.planePitches[0] * height);
                CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufObjFlushCpuCacheRange Failed");
            }

            sciErr = NvSciBufObjGetPixels(bufPtr, nullptr, (void **)pBuff, size, pBuffPitches);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufObjGetPixels Failed");

            for (uint32_t i = 0U; i < numSurfaces; i++) {
                if (fwrite(pBuff[i], size[i], 1U, m_pOutFile) != 1U) {
                    LOG_ERR("File write failed\n");
                    return NVSIPL_STATUS_ERROR;
                }
            }
        }

        return NVSIPL_STATUS_OK;
    }

    void Deinit(void)
    {
        if (m_pOutFile != nullptr) {
            fflush(m_pOutFile);
            fclose(m_pOutFile);
            delete [] m_pBuff;
        }
    }

 private:
    uint8_t *m_pBuff {nullptr};
    FILE *m_pOutFile = nullptr;
    bool m_isRawOutput = false;
    string m_filename = "";

    typedef struct {
        float heightFactor[MAX_NUM_SURFACES];
        float widthFactor[MAX_NUM_SURFACES];
        uint32_t numSurfaces;
    } BufUtilSurfParams;

    BufUtilSurfParams BufSurfParamsTable_Default = {
        .heightFactor = {1, 0, 0},
        .widthFactor = {1, 0, 0},
        .numSurfaces = 1,
    };

    BufUtilSurfParams BufSurfParamsTable_YUV[3] = {
        /* Shift factors for SEMI_PLANAR to PLANAR conversion */
        { /* 420 */
            .heightFactor = {1, 0.5, 0.5},
            .widthFactor = {1, 0.5, 0.5},
            .numSurfaces = 3,
        },
        { /* 444 */
            .heightFactor = {1, 1, 1},
            .widthFactor = {1, 1, 1},
            .numSurfaces = 3,
        },
        { /* 422 */
            .heightFactor = {1, 1, 1},
            .widthFactor = {1, 0.5, 0.5},
            .numSurfaces = 3,
        },
    };

    SIPLStatus GetBuffParams(BufferAttrs buffAttrs,
                             float **xScale,
                             float **yScale,
                             uint32_t **bytesPerPixel,
                             uint32_t *numSurfacesVal,
                             bool *isPackedYUV)
    {
        uint32_t subSamplingType;
        uint32_t numSurfaces = 1U;
        uint32_t *bytesPerPixelPtr = nullptr;
        float *xScalePtr = nullptr, *yScalePtr = nullptr;
        static uint32_t bpp[6] = {1U, 0U, 0U}; // Initializing default array for Bytes Per Pixels

        if ((buffAttrs.planeColorFormats[0] >= NvSciColor_A8Y8U8V8) &&
            (buffAttrs.planeColorFormats[0] <= NvSciColor_A16Y16U16V16)){
            // YUV PACKED
            if(!CUtils::GetBpp(buffAttrs.planeBitsPerPixels[0], &bpp[0])) {
                return NVSIPL_STATUS_ERROR;
            }
            bytesPerPixelPtr = &bpp[0];
            xScalePtr =  &BufSurfParamsTable_Default.widthFactor[0];
            yScalePtr = &BufSurfParamsTable_Default.heightFactor[0];
            numSurfaces = BufSurfParamsTable_Default.numSurfaces;
            *isPackedYUV = true;
        } else if ((1U == buffAttrs.planeCount) &&
                   (buffAttrs.planeColorFormats[0] == NvSciColor_Y16)) {
            //LUMA16
            bpp[0] = 2U;
            xScalePtr = &BufSurfParamsTable_Default.widthFactor[0];
            yScalePtr = &BufSurfParamsTable_Default.heightFactor[0];
            numSurfaces = BufSurfParamsTable_Default.numSurfaces;
            bytesPerPixelPtr = &bpp[0];
        } else if ((1U == buffAttrs.planeCount) &&
                   ((buffAttrs.planeColorFormats[0] == NvSciColor_Float_A16B16G16R16) ||
                    ((buffAttrs.planeColorFormats[0] >= NvSciColor_B8G8R8A8) &&
                     (buffAttrs.planeColorFormats[0] <= NvSciColor_A8B8G8R8)))) {
            // RBGA
            xScalePtr = &BufSurfParamsTable_Default.widthFactor[0];
            yScalePtr = &BufSurfParamsTable_Default.heightFactor[0];
            numSurfaces = BufSurfParamsTable_Default.numSurfaces;
            if (buffAttrs.planeColorFormats[0] == NvSciColor_Float_A16B16G16R16) {
                bpp[0] = 8U;
            } else {
                bpp[0] = 4U;
            }
            bytesPerPixelPtr = &bpp[0];
        } else if ((1U == buffAttrs.planeCount) &&
                   ((buffAttrs.planeColorFormats[0] < NvSciColor_U8V8) ||
                   ((buffAttrs.planeColorFormats[0] >= NvSciColor_X6Bayer10BGGI_RGGI) &&
                   (buffAttrs.planeColorFormats[0] <= NvSciColor_Bayer16IGGR_IGGB)))) {
            // RAW
            if(!CUtils::GetBpp(buffAttrs.planeBitsPerPixels[0], &bpp[0])) {
                return NVSIPL_STATUS_ERROR;
            }
            bytesPerPixelPtr = &bpp[0];
            xScalePtr = &BufSurfParamsTable_Default.widthFactor[0];
            yScalePtr = &BufSurfParamsTable_Default.heightFactor[0];
            numSurfaces = BufSurfParamsTable_Default.numSurfaces;
        } else if ((2U == buffAttrs.planeCount) &&
                   ((NvSciColor_Y8 == buffAttrs.planeColorFormats[0]) ||
                   (NvSciColor_Y10 == buffAttrs.planeColorFormats[0]) ||
                   (NvSciColor_Y12 == buffAttrs.planeColorFormats[0]) ||
                   (NvSciColor_Y16 == buffAttrs.planeColorFormats[0]))) {
            // YUV Semi Planar
            if ((buffAttrs.planeHeights[0] == buffAttrs.planeHeights[1]) &&
                (buffAttrs.planeWidths[0] == buffAttrs.planeWidths[1])) {
                subSamplingType = 1; // 444
            } else if ((buffAttrs.planeHeights[0] == (2U * buffAttrs.planeHeights[1])) &&
                       (buffAttrs.planeWidths[0] == (2U * buffAttrs.planeWidths[1]))) {
                subSamplingType = 0; // 420
            } else if ((buffAttrs.planeHeights[0] == buffAttrs.planeHeights[1]) &&
                       (buffAttrs.planeWidths[0] == (2U * buffAttrs.planeWidths[1]))) {
                subSamplingType = 2; // 422
            } else {
                LOG_ERR("Unsupported channel count\n");
                return NVSIPL_STATUS_NOT_SUPPORTED;
            }
            xScalePtr = &BufSurfParamsTable_YUV[subSamplingType].widthFactor[0];
            yScalePtr = &BufSurfParamsTable_YUV[subSamplingType].heightFactor[0];
            numSurfaces = BufSurfParamsTable_YUV[subSamplingType].numSurfaces;
            if(!CUtils::GetBpp(buffAttrs.planeBitsPerPixels[0], &bpp[0])) {
                return NVSIPL_STATUS_ERROR;
            }
            bpp[1] = bpp[0];
            bpp[2] = bpp[0];
            bytesPerPixelPtr = &bpp[0];
        } else {
            LOG_ERR("Unsupported plane format\n");
            return NVSIPL_STATUS_NOT_SUPPORTED;
        }

        if (xScale) {
            *xScale = xScalePtr;
        }
        if (yScale) {
            *yScale = yScalePtr;
        }
        if (bytesPerPixel) {
            *bytesPerPixel = bytesPerPixelPtr;
        }
        if (numSurfacesVal) {
            *numSurfacesVal = numSurfaces;
        }

        return NVSIPL_STATUS_OK;
    }
};

#endif //CFileWriter_HPP
