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
#include <cmath>

#include "NvSIPLClient.hpp"
#include "CProfiler.hpp"
#include "CFrameFeeder.hpp"
#include "CFileWriter.hpp"

#if !NV_IS_SAFETY
#include "CComposite.hpp"
#include "CNvSIPLMasterNvSci.hpp"
#endif // !NV_IS_SAFETY

#ifndef CNVSIPLCONSUMER_HPP
#define CNVSIPLCONSUMER_HPP

#define IMAGE_QUEUE_TIMEOUT_US (1000000U)

using namespace std;
using namespace nvsipl;

/** NvSIPL consumer class.
 * NvSIPL consumer consumes the output buffers of NvSIPL.
 */
class CNvSIPLConsumer
{
 public:
    SIPLStatus Init(
#if !NV_IS_SAFETY
                    CComposite *pComposite=nullptr,
                    CNvSIPLMasterNvSci *pMasterNvSci=nullptr,
#endif // !NV_IS_SAFETY
                    uint32_t uID=-1,
                    uint32_t uSensor=-1,
                    INvSIPLClient::ConsumerDesc::OutputType outputType=INvSIPLClient::ConsumerDesc::OutputType::ICP,
                    CProfiler *pProfiler=nullptr,
                    CFrameFeeder *pFrameFeeder=nullptr,
                    std::string sFilenamePrefix="",
                    uint32_t uNumSkipFrames=0u,
                    uint64_t uNumWriteFrames=-1u)
    {
#if !NV_IS_SAFETY
        if ((pComposite != nullptr) && (pMasterNvSci != nullptr)) {
            LOG_ERR("CNvSIPLConsumer expects only one of pComposite and pMasterNvSci\n");
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }

        m_pComposite = pComposite;
        m_pMasterNvSci = pMasterNvSci;
#endif // !NV_IS_SAFETY

        m_uID = uID;
        m_uSensor = uSensor;
        m_outputType = outputType;
        m_pProfiler = pProfiler;
        m_pFrameFeeder = pFrameFeeder;
        m_sFilenamePrefix = sFilenamePrefix;
        m_uNumSkipFrames = uNumSkipFrames;
        m_uNumWriteFrames = uNumWriteFrames;

        // Create file writer if necessary
        if (m_sFilenamePrefix != "") {
            m_pFileWriter.reset(new CFileWriter);
        }

        return NVSIPL_STATUS_OK;
    }

    void Deinit(void)
    {
        if (m_pFileWriter != nullptr) {
            m_pFileWriter->Deinit();
            m_pFileWriter = nullptr;
        }

        return;
    }

    void EnableMetadataLogging(void) {
        m_bShowMetadata = true;
    }

#if !NV_IS_SAFETY
    bool IsLEDEnabled(void) {
        return m_toggleLED_ON;
    }
    bool IsPrevFrameLEDEnabled(void) {
        return m_prevFrameLEDEnabled;
    }

    void EnableLEDControl(void) {
        m_LEDControl = true;
    }

    bool IsLEDControlEnabled(void) {
        return m_LEDControl;
    }
#endif // !NV_IS_SAFETY

    uint32_t m_uSensor = -1;
    INvSIPLClient::ConsumerDesc::OutputType m_outputType;

    virtual ~CNvSIPLConsumer() {
        Deinit();
    }

    bool IsFrameWriteComplete(void) {
        return m_bFrameWriteDone;
    }

    SIPLStatus OnFrameAvailable(INvSIPLClient::INvSIPLBuffer* pBuffer,
                                NvSciSyncCpuWaitContext cpuWaitContext)
    {
        auto pNvMBuffer = (INvSIPLClient::INvSIPLNvMBuffer*)pBuffer;
        if (pNvMBuffer == nullptr) {
            LOG_ERR("Invalid INvSIPLClient::INvSIPLNvMBuffer pointer\n");
            return NVSIPL_STATUS_ERROR;
        }
        const auto& md = pNvMBuffer->GetImageData();
        auto& EmdData = pNvMBuffer->GetImageEmbeddedData();
        LOG_INFO("EmdData.embeddedBufTopSize: %u EmdData.embeddedBufBottomSize: %u\n",
                 EmdData.embeddedBufTopSize, EmdData.embeddedBufBottomSize);

        // Send to profiler
        if (m_pProfiler != nullptr) {
            SIPLStatus status = m_pProfiler->ProfileFrame(pNvMBuffer);
            if (status != NVSIPL_STATUS_OK) {
                LOG_ERR("Frame profiling failed\n");
                return status;
            }
        }

        if (m_pFrameFeeder != nullptr) {
            if (m_outputType == INvSIPLClient::ConsumerDesc::OutputType::ICP) {
                SIPLStatus status = m_pFrameFeeder->SetInputFrame(pBuffer);
                if (status != NVSIPL_STATUS_OK) {
                    LOG_ERR("Failed to set input frame for two-pass ISP feeder\n");
                    return status;
                }
            } else {
                SIPLStatus status = m_pFrameFeeder->DeliverOutputFrame(pBuffer);
                if (status != NVSIPL_STATUS_OK) {
                    LOG_ERR("Failed to send output frame to feeder for synchronization purposes\n");
                    return status;
                }
            }
        }

        // Send to file writer
        if (m_pFileWriter && (m_uFrameCounter >= m_uNumSkipFrames)) {
            if ((m_uNumWriteFrames == -1u) || (m_uFrameCounter < (m_uNumSkipFrames + m_uNumWriteFrames))) {
                // create file if it isn't created
                if (!(m_uFrameCounter - m_uNumSkipFrames)) {
                    string sFileExt;
                    NvSciBufObj bufPtr = pNvMBuffer->GetNvSciBufImage();
                    BufferAttrs bufAttrs;
                    auto status = PopulateBufAttr(bufPtr, bufAttrs);
                    if(status != NVSIPL_STATUS_OK) {
                        LOG_ERR("Consumer: PopulateBufAttr failed\n");
                        return NVSIPL_STATUS_BAD_ARGUMENT;
                    }
                    if (((bufAttrs.planeColorFormats[0] > NvSciColor_LowerBound) &&
                         (bufAttrs.planeColorFormats[0] < NvSciColor_U8V8)) ||
                        ((bufAttrs.planeColorFormats[0] > NvSciColor_Float_A16) &&
                         (bufAttrs.planeColorFormats[0] < NvSciColor_UpperBound))) {
                        sFileExt = ".raw";
                    } else if ((bufAttrs.planeColorFormats[0] == NvSciColor_Y16) && (bufAttrs.planeCount == 1U)) {
                        sFileExt = ".luma";
                    } else if (((bufAttrs.planeColorFormats[0] > NvSciColor_V16U16) &&
                                (bufAttrs.planeColorFormats[0] < NvSciColor_U8)) ||
                               ((bufAttrs.planeColorFormats[0] > NvSciColor_V16) &&
                                (bufAttrs.planeColorFormats[0] < NvSciColor_A8))) {
                        sFileExt = ".yuv";
                    } else if ((bufAttrs.planeColorFormats[0] > NvSciColor_A16Y16U16V16) &&
                               (bufAttrs.planeColorFormats[0] < NvSciColor_X6Bayer10BGGI_RGGI)) {
                        sFileExt = ".rgba";
                    }
                    string sFilename = m_sFilenamePrefix + "_cam_" + std::to_string(m_uSensor)
                                           + "_out_" + std::to_string((uint32_t)m_outputType) + sFileExt;
                    auto bRawOut = (m_outputType == INvSIPLClient::ConsumerDesc::OutputType::ICP);
                    status = m_pFileWriter->Init(sFilename, bRawOut);
                    if (status != NVSIPL_STATUS_OK) {
                        LOG_ERR("Failed to initialize file writer\n");
                        return status;
                    }
                }
                auto status = m_pFileWriter->WriteBufferToFile(pNvMBuffer, cpuWaitContext);
                if (status != NVSIPL_STATUS_OK) {
                    LOG_ERR("WriteBufferToFile failed\n");
                    return status;
                }
            } else {
                m_bFrameWriteDone = true;
            }
        }

        if (m_bShowMetadata) {
            PrintMetadata(md);
        }

#if !NV_IS_SAFETY
        if(m_LEDControl) {
            SetLEDFlag(md);
        }
#endif


#if !NV_IS_SAFETY
        if (m_pComposite != nullptr) {
            auto status = m_pComposite->Post(m_uID, pNvMBuffer);
            if (status != NVSIPL_STATUS_OK) {
                LOG_ERR("Compositor post failed\n");
                return status;
            }
        }
        if (m_pMasterNvSci != nullptr) {
            auto status = m_pMasterNvSci->Post(m_uSensor, m_outputType, pNvMBuffer);
            if (status != NVSIPL_STATUS_OK) {
                LOG_ERR("Master post failed\n");
                return status;
            }
        }
#endif // !NV_IS_SAFETY

        m_uFrameCounter++;
        return NVSIPL_STATUS_OK;
    }

private:

    void PrintMetadata(const INvSIPLClient::ImageMetaData &md)
    {
        cout << "Camera ID: " << m_uSensor << endl;
        cout << " Frame Counter: " << (md.frameSeqNumInfo.frameSeqNumValid ?
                                       md.frameSeqNumInfo.frameSequenceNumber : m_uFrameCounter)
                                   << endl;
        cout << " TSC SOF: " << md.frameCaptureStartTSC << endl;
        cout << " TSC EOF: " << md.frameCaptureTSC << endl;
        if (md.badPixelStatsValid) {
            cout << " Bad pixel stats:" << endl;
            cout << "     highInWin: " << md.badPixelStats.highInWin << endl;
            cout << "     lowInWin: " << md.badPixelStats.lowInWin << endl;
            cout << "     highMagInWin: " << md.badPixelStats.highMagInWin << endl;
            cout << "     lowMagInWin: " << md.badPixelStats.lowMagInWin << endl;
            cout << "     highOutWin: " << md.badPixelStats.highOutWin << endl;
            cout << "     lowOutWin: " << md.badPixelStats.lowOutWin << endl;
            cout << "     highMagOutWin: " << md.badPixelStats.highMagOutWin << endl;
            cout << "     lowMagOutWin: " << md.badPixelStats.lowMagOutWin << endl;

            cout << " Bad pixel settings:" << endl;
            cout << "     ROI: " << md.badPixelSettings.rectangularMask.x0 << ", "
                                 << md.badPixelSettings.rectangularMask.x1 << ", "
                                 << md.badPixelSettings.rectangularMask.y0 << ", "
                                 << md.badPixelSettings.rectangularMask.y1 << endl;
        }

        if (md.controlInfo.valid) {
            cout << "alpha: " << md.controlInfo.alpha << endl;
            if (md.controlInfo.isLuminanceCalibrated) {
                cout << "luminanceCalibrationFactor: " << md.controlInfo.luminanceCalibrationFactor << endl;
            } else {
                cout << "luminance is not calibrated" << endl;
            }
            if (md.controlInfo.wbGainTotal.valid) {
                cout << "wbGains[0]: " << md.controlInfo.wbGainTotal.gain[0] << endl;
                cout << "wbGains[1]: " << md.controlInfo.wbGainTotal.gain[1] << endl;
                cout << "wbGains[2]: " << md.controlInfo.wbGainTotal.gain[2] << endl;
                cout << "wbGains[3]: " << md.controlInfo.wbGainTotal.gain[3] << endl;
            } else {
                cout << "wbGain info not enabled" << endl;
            }
            cout << "cct: " << md.controlInfo.cct << endl;
            cout << "brightnessKey: " << md.controlInfo.brightnessKey << endl;
            cout << "rawImageMidTone: " << md.controlInfo.rawImageMidTone << endl;
            if (md.controlInfo.gtmSplineInfo.enable) {
                cout << "gtmSpline control points: " << endl;
                for (uint32_t i = 0U; i < NUM_GTM_SPLINE_POINTS; i++) {
                    cout << "    index " << i << " -- x: " << md.controlInfo.gtmSplineInfo.gtmSplineControlPoint[i].x << " y: " << \
                    md.controlInfo.gtmSplineInfo.gtmSplineControlPoint[i].y << " slope: " << md.controlInfo.gtmSplineInfo.gtmSplineControlPoint[i].slope << endl;
                }
            } else {
                cout << "gtmSpline info not enabled" << endl;
            }

        } else {
            cout << "controlInfo not valid" << endl;
        }

        for (uint32_t i = 0; i < 2; i++) {
            if (md.histogramStatsValid[i]) {
                cout << " Histogram[" << i << "] stats:" << endl;
                for (uint32_t comp = 0; comp < NVSIPL_ISP_MAX_COLOR_COMPONENT; comp++) {
                    cout << "     data[][" << comp << "] (first 8 values): ";
                    for (uint32_t j = 0; j < 8; j++) {
                        cout << md.histogramStats[i].data[j][comp] << " ";
                    }
                    cout << endl;
                }

                cout << " Histogram[" << i << "] settings:" << endl;
                cout << "     knees[]: ";
                for (uint32_t point = 0; point < NVSIPL_ISP_HIST_KNEE_POINTS; point++) {
                    cout << unsigned(md.histogramSettings[i].knees[point]) << " ";
                }
                cout << endl;

                cout << "     ranges[]: ";
                for (uint32_t point = 0; point < NVSIPL_ISP_HIST_KNEE_POINTS; point++) {
                    cout << unsigned(md.histogramSettings[i].ranges[point]) << " ";
                }
                cout << endl;
            }
        }

        for (uint32_t i = 0; i < 2; i++) {
            if (md.localAvgClipStatsValid[i]) {
                cout << " LocalAvgClip[" << i << "] stats:" << endl;
                for (uint32_t roi = 0; roi < NVSIPL_ISP_MAX_LAC_ROI; roi++) {
                    cout << "     data[" << roi << "].numWindowsH: " << md.localAvgClipStats[i].data[0].numWindowsH << endl;
                    cout << "     data[" << roi << "].numWindowsV: " << md.localAvgClipStats[i].data[0].numWindowsH << endl;
                    for (uint32_t comp = 0; comp < NVSIPL_ISP_MAX_COLOR_COMPONENT; comp++) {
                        cout << "     data[" << roi << "].average[][" << comp << "] (first 8 values): ";
                        for (uint32_t j = 0; j < 8; j++) {
                            cout << md.localAvgClipStats[i].data[roi].average[j][comp] << " ";
                        }
                        cout << endl;
                    }
                }

                cout << " LocalAvgClip[" << i << "] settings:" << endl;
                for (uint32_t roi = 0; roi < NVSIPL_ISP_MAX_LAC_ROI; roi++) {
                    cout << "     windows[" << roi << "].width: " << md.localAvgClipSettings[i].windows[roi].width << endl;
                    cout << "     windows[" << roi << "].height: " << md.localAvgClipSettings[i].windows[roi].height << endl;
                    cout << "     windows[" << roi << "].numWindowsH: " << md.localAvgClipSettings[i].windows[roi].numWindowsH << endl;
                    cout << "     windows[" << roi << "].numWindowsV: " << md.localAvgClipSettings[i].windows[roi].numWindowsV << endl;
                    cout << "     windows[" << roi << "].horizontalInterval: " << md.localAvgClipSettings[i].windows[roi].horizontalInterval << endl;
                    cout << "     windows[" << roi << "].verticalInterval: " << md.localAvgClipSettings[i].windows[roi].verticalInterval << endl;
                    cout << "     windows[" << roi << "].startOffset: (" << md.localAvgClipSettings[i].windows[roi].startOffset.x << ", "
                                                                        << md.localAvgClipSettings[i].windows[roi].startOffset.y << ")" << endl;
                }
            }
        }
#if !NV_IS_SAFETY
        if (md.frameTimestampInfo.frameTimestampValid) {
            cout << " Frame Timestamp from the sensor: " << md.frameTimestampInfo.frameTimestamp << endl;
        }
#endif //!NV_IS_SAFETY

        cout << " errorFlag: " << (int)md.errorFlag << " (meaning determined by driver)" << endl;
    }

#if !NV_IS_SAFETY
    void SetLEDFlag(const INvSIPLClient::ImageMetaData &md)
    {
        if (md.sensorExpInfo.expTimeValid) {
            constexpr double_t MAX_EXP_THRESHOLD_AR0234 = 0.0013; // 1.3ms
            constexpr double_t MIN_EXP_THRESHOLD_AR0234 = 0.0004; //0.4ms

            m_prevFrameLEDEnabled = IsLEDEnabled();
            if (md.sensorExpInfo.exposureTime[0] >= MAX_EXP_THRESHOLD_AR0234) {
                m_toggleLED_ON = true; //turn on LED when exposure hits max threshold
            }
            if (md.sensorExpInfo.exposureTime[0] <= MIN_EXP_THRESHOLD_AR0234) {
                m_toggleLED_ON = false; //turn off LED when exposure hits min threshold
            }
        }
    }
#endif //!NV_IS_SAFETY

    unique_ptr<CFileWriter> m_pFileWriter = nullptr;
    string m_sFilenamePrefix = "";
    uint32_t m_uNumSkipFrames = 0u;
    uint64_t m_uNumWriteFrames = -1u;
    bool m_bFrameWriteDone = false;
#if !NV_IS_SAFETY
    CComposite *m_pComposite = nullptr;
    CNvSIPLMasterNvSci *m_pMasterNvSci = nullptr;
    bool m_prevFrameLEDEnabled = true;
    bool m_toggleLED_ON = true;
    bool m_LEDControl = false;
#endif // !NV_IS_SAFETY
    uint32_t m_uID = -1;
    uint32_t m_uFrameCounter = 0u;
    CProfiler *m_pProfiler = nullptr;
    CFrameFeeder *m_pFrameFeeder = nullptr;
    bool m_bShowMetadata = false;
};

#endif //CNVSIPLCONSUMER_HPP
