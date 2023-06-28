/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "CSamplePlugin.hpp"

/* STL Headers */
#include <cstring>

CAutoControlPlugin::CAutoControlPlugin()
{
    m_prevExpVal = {};
    m_targetLuma = 0.10F;
    m_dampingFactor = 0.60F;
}

nvsipl::SIPLStatus CAutoControlPlugin::ProcessAE(const nvsipl::SiplControlAutoInputParam& inParam,
                                nvsipl::SiplControlAutoSensorSetting& sensorSett)
{
    uint32_t i;
    uint32_t j;
    uint32_t kneeL;
    uint32_t kneeU;
    uint32_t rangeL;
    uint32_t rangeU;
    uint32_t rangeIdx;
    uint32_t numPixels;
    uint32_t totalpixels;
    uint32_t histWidth;
    uint32_t histHeight;
    float_t beta;
    float_t linVal;
    float_t avgVal;
    float_t nextExpVal;
    float_t targetExpVal;
    float_t currentLuma;
    float_t nextExpTime[DEVBLK_CDI_MAX_EXPOSURES] {};
    float_t nextExpGain[DEVBLK_CDI_MAX_EXPOSURES] {};
    nvsipl::NvSiplRect rect;

    auto& stats = inParam.statsInfo;
    auto histStatsData = stats.histData[0];
    auto histControls = stats.histSettings[0];

    auto& parsedEmbData = inParam.embedData.embedInfo;
    auto& sensorAttr = inParam.sensorAttr;

    if (histControls->enable == NVSIPL_TRUE) {
        rect = histControls->rectangularMask;
        histWidth = (uint32_t)(rect.x1 - rect.x0);
        histHeight = (uint32_t)(rect.y1 - rect.y0);
        totalpixels = (histWidth * histHeight) / NVSIPL_ISP_MAX_COLOR_COMPONENT;

        /*
            * Calculating the average pixel value for each channel is done by:
            * average value = sum(lin[i] * H[i], i = 0, i < 256) / numPixels
            *
            * The lin value is a value between the upper and lower bounds of the current
            * range, this value is calculated by interpolating between the two using the
            * current bin index.
            * The H represents the frequency of the pixel data in each bin in that channel.
            * In order to normalize this value divide by numpixels.
            *
            * At the end average all the channels average values and divide by the total
            * range of the histogram to normalize the lin values that were used.
            *
            * This will result in a current luma in the range 0-1.
            */

        currentLuma = 0.0F;

        for (i = 0U; i < NVSIPL_ISP_MAX_COLOR_COMPONENT; i++) {
            avgVal = 0.0F;
            rangeIdx = 0U;

            kneeL = 0U;
            kneeU = histControls->knees[rangeIdx];

            rangeL = 0U;
            rangeU = rangeL + (1U << histControls->ranges[rangeIdx]);

            numPixels = totalpixels - histStatsData->excludedCount[i];

            for (j = 0U; j < NVSIPL_ISP_HIST_BINS; j++) {
                beta = ((float_t)(j - kneeL)) / ((float_t)(kneeU - kneeL));
                linVal = (float_t)(((rangeL * (1.0F - beta)) + (rangeU * beta)));
                avgVal += (linVal * ((float_t)histStatsData->data[j][i]) /
                    (float_t)numPixels);

                if ((j == kneeU) && (j != (NVSIPL_ISP_HIST_BINS - 1U))) {
                    rangeIdx++;

                    kneeL = kneeU;
                    kneeU = histControls->knees[rangeIdx];

                    rangeL = rangeU;
                    rangeU = rangeL + (1U << histControls->ranges[rangeIdx]);
                }
            }
            currentLuma += avgVal;
        }

        if (sensorAttr.numActiveExposures == 1U) {
            m_targetLuma = 0.18F;
        }
        if (sensorAttr.numActiveExposures > 4U) {
            return nvsipl::NVSIPL_STATUS_NOT_SUPPORTED;
        }

        currentLuma /= (float_t) NVSIPL_ISP_MAX_COLOR_COMPONENT;
        currentLuma /= (float_t) (1U << ISP_OUT_MAX_BITS);

        if (currentLuma > 0.0F) {
            targetExpVal =  parsedEmbData.sensorExpInfo.exposureTime[0] *
                            parsedEmbData.sensorExpInfo.sensorGain[0] *
                            (m_targetLuma / currentLuma);
        } else {
            /* Scene is dark, double the exposure values. */
            targetExpVal = parsedEmbData.sensorExpInfo.exposureTime[0] *
                            parsedEmbData.sensorExpInfo.sensorGain[0] *
                            2.0F;
        }

        nextExpVal = m_dampingFactor * targetExpVal + (1.0F - m_dampingFactor) * m_prevExpVal;

        /*
            * This is an example calculation of exposure control. Using fixed hdr ratio of 64 &
            * setting all exposure gains to same value as 1st exposure.
            * For 3-exposure, hdr ratio is 8. 4-exposure, hdr ratio is 4.
            */
        nextExpTime[0] = sensorAttr.sensorExpRange[0].max;
        nextExpGain[0] = nextExpVal / nextExpTime[0];
        nextExpGain[0] = CLIP(nextExpGain[0], sensorAttr.sensorGainRange[0].min, sensorAttr.sensorGainRange[0].max);
        nextExpTime[0] = nextExpVal / nextExpGain[0];
        nextExpTime[0] = CLIP(nextExpTime[0], sensorAttr.sensorExpRange[0].min, sensorAttr.sensorExpRange[0].max);

        for (i = 1U; i < sensorAttr.numActiveExposures; i++) {
            if (sensorAttr.numActiveExposures == 2U) {
                nextExpVal = nextExpVal / 64.0F;
            } else if (sensorAttr.numActiveExposures == 3U) {
                nextExpVal = nextExpVal / 8.0F;
            } else {
                nextExpVal = nextExpVal / 4.0F;
            }
            nextExpGain[i] = nextExpGain[0];
            nextExpGain[i] = CLIP(nextExpGain[i], sensorAttr.sensorGainRange[i].min, sensorAttr.sensorGainRange[i].max);
            nextExpTime[i] = nextExpVal / nextExpGain[i];
            nextExpTime[i] = CLIP(nextExpTime[i], sensorAttr.sensorExpRange[i].min, sensorAttr.sensorExpRange[i].max);
        }

        sensorSett.exposureControl[0].expTimeValid = true;
        sensorSett.exposureControl[0].gainValid = true;
        for (i = 0U; i < sensorAttr.numActiveExposures; i++) {
            sensorSett.exposureControl[0].exposureTime[i] = nextExpTime[i];
            sensorSett.exposureControl[0].sensorGain[i] = nextExpGain[i];
        }
        sensorSett.numSensorContexts = 1;
        m_prevExpVal = nextExpTime[0] * nextExpGain[0];
    }
    return nvsipl::NVSIPL_STATUS_OK;
}

nvsipl::SIPLStatus CAutoControlPlugin::ProcessAWB(const nvsipl::SiplControlAutoInputParam& inParam,
                                nvsipl::SiplControlAutoOutputParam& outParam)
{
    uint32_t i;
    uint32_t j;
    uint32_t GsampleCnt, RsampleCnt, BsampleCnt;
    uint32_t totalpixelsCmpnt, sampleCnt;
    float_t maxGain;
    float_t Gsample, Rsample, Bsample;
    float_t RAvgStats, GAvgStats1, GAvgStats2, BAvgStats;
    float_t invgains[NVSIPL_ISP_MAX_COLOR_COMPONENT] {};
    float_t awbValues[NVSIPL_ISP_MAX_COLOR_COMPONENT] {};
    float_t nextAwbGains[NVSIPL_ISP_MAX_COLOR_COMPONENT] {};

    auto& stats = inParam.statsInfo;
    auto& lacSetting = stats.lacSettings[0];
    auto lacStatsData = stats.lacData[0];

    auto& parsedEmbData = inParam.embedData.embedInfo;

    Rsample = 0.0F;
    Gsample = 0.0F;
    Bsample = 0.0F;

    RsampleCnt = 0U;
    GsampleCnt = 0U;
    BsampleCnt = 0U;

    if (parsedEmbData.sensorWBInfo.wbValid == NVSIPL_TRUE) {
        invgains[0] = 1.0F / parsedEmbData.sensorWBInfo.wbGain[0].value[0];
        invgains[1] = 1.0F / parsedEmbData.sensorWBInfo.wbGain[0].value[1];
        invgains[2] = 1.0F / parsedEmbData.sensorWBInfo.wbGain[0].value[2];
        invgains[3] = 1.0F / parsedEmbData.sensorWBInfo.wbGain[0].value[3];
    } else {
        invgains[0] = 1.0F;
        invgains[1] = 1.0F;
        invgains[2] = 1.0F;
        invgains[3] = 1.0F;
    }

    if ((lacSetting != NULL) && (lacSetting->enable == NVSIPL_TRUE)) {
        for (j = 0U; j < (uint32_t)NVSIPL_ISP_MAX_LAC_ROI; j++) {
            if (lacSetting->roiEnable[j] == NVSIPL_FALSE) {
                continue;
            }
            totalpixelsCmpnt = lacSetting->windows[j].width * lacSetting->windows[j].height / 4U;

            for (i = 0U; i < lacStatsData->data[j].numWindowsH * lacStatsData->data[j].numWindowsV; i++) {
                RAvgStats = (float_t)lacStatsData->data[j].average[i][0];
                GAvgStats1 = (float_t)lacStatsData->data[j].average[i][1];
                GAvgStats2 = (float_t)lacStatsData->data[j].average[i][2];
                BAvgStats = (float_t)lacStatsData->data[j].average[i][3];
                if((RAvgStats > 0.0F) && (GAvgStats1 > 0.0F) && (GAvgStats2 > 0.0F) && (BAvgStats > 0.0F)){
                    sampleCnt   = totalpixelsCmpnt -
                                    lacStatsData->data[j].maskedOffCount[i][0] -
                                    lacStatsData->data[j].clippedCount[i][0];
                    Rsample    += RAvgStats * (float_t)sampleCnt;
                    RsampleCnt += sampleCnt;

                    sampleCnt   = totalpixelsCmpnt -
                                    lacStatsData->data[j].maskedOffCount[i][1] -
                                    lacStatsData->data[j].clippedCount[i][1];
                    Gsample    += GAvgStats1 * (float_t)sampleCnt;
                    GsampleCnt += sampleCnt;
                    sampleCnt   = totalpixelsCmpnt -
                                    lacStatsData->data[j].maskedOffCount[i][2] -
                                    lacStatsData->data[j].clippedCount[i][2];
                    Gsample    += GAvgStats1 * (float_t)sampleCnt;
                    GsampleCnt += sampleCnt;

                    sampleCnt   = totalpixelsCmpnt -
                                    lacStatsData->data[j].maskedOffCount[i][3] -
                                    lacStatsData->data[j].clippedCount[i][3];
                    Bsample    += BAvgStats * (float_t)sampleCnt;
                    BsampleCnt += sampleCnt;
                }
            }
        }
    }

    if ((RsampleCnt > 0U) && (GsampleCnt > 0U) && (BsampleCnt > 0U)) {
        awbValues[0] = (Rsample / (float_t)RsampleCnt) * invgains[0];
        awbValues[1] = (Gsample / (float_t)GsampleCnt) * ((invgains[1] + invgains[2]) / 2.0F);
        awbValues[2] = awbValues[1];
        awbValues[3] = (Bsample / (float_t)BsampleCnt) * invgains[3];

        for (i = 0U; i < NVSIPL_ISP_MAX_COLOR_COMPONENT; i++) {
            if(awbValues[1] > 0.0F) {
                nextAwbGains[i] = awbValues[i] / awbValues[1];
            } else {
                nextAwbGains[i] = 1.0F;

            }
        }

        maxGain = nextAwbGains[0];
        for (i = 1U; i < NVSIPL_ISP_MAX_COLOR_COMPONENT; i++) {
            if (nextAwbGains[i] > maxGain) {
                maxGain = nextAwbGains[i];
            }
        }

        for (i = 0U; i < NVSIPL_ISP_MAX_COLOR_COMPONENT; i++) {
            if(nextAwbGains[i] > 0.0F) {
                nextAwbGains[i] = maxGain / nextAwbGains[i];
            } else {
                nextAwbGains[i] = 1.0F;
            }
            if (nextAwbGains[i] > 8.0F)
            {
                nextAwbGains[i] = 8.0F;
            }
            if (!(nextAwbGains[i] >= 1.0F)) {
                nextAwbGains[i] = 1.0F;
            }
        }

        for (auto p = 0U; p < NVSIPL_ISP_MAX_INPUT_PLANES; p++) {
            outParam.awbSetting.wbGainTotal[p].valid = NVSIPL_TRUE;
            outParam.awbSetting.wbGainTotal[p].gain[0] = nextAwbGains[0];
            outParam.awbSetting.wbGainTotal[p].gain[1] = nextAwbGains[1];
            outParam.awbSetting.wbGainTotal[p].gain[2] = nextAwbGains[2];
            outParam.awbSetting.wbGainTotal[p].gain[3] = nextAwbGains[3];
        }
    } else {
        for (auto p = 0U; p < NVSIPL_ISP_MAX_INPUT_PLANES; p++) {
            outParam.awbSetting.wbGainTotal[p].valid = NVSIPL_TRUE;
            outParam.awbSetting.wbGainTotal[p].gain[0] = 1.0F;
            outParam.awbSetting.wbGainTotal[p].gain[1] = 1.0F;
            outParam.awbSetting.wbGainTotal[p].gain[2] = 1.0F;
            outParam.awbSetting.wbGainTotal[p].gain[3] = 1.0F;
        }
    }

    outParam.awbSetting.cct = 4000.0F;
    for (auto i = 0U; i < NVSIPL_ISP_MAX_COLORMATRIX_DIM; i++) {
        outParam.awbSetting.ccmMatrix[i][i] = 1.0F;
    }
    outParam.ispDigitalGain = 1.0F;
    outParam.newStatsSetting = {};
    return nvsipl::NVSIPL_STATUS_OK;
}

nvsipl::SIPLStatus CAutoControlPlugin::Process(const nvsipl::SiplControlAutoInputParam& inParam,
                            nvsipl::SiplControlAutoOutputParam& outParam) noexcept
{
    auto status = nvsipl::NVSIPL_STATUS_OK;
    status = ProcessAE(inParam, outParam.sensorSetting);
    if (status != nvsipl::NVSIPL_STATUS_OK) {
        return status;
    }
    status = ProcessAWB(inParam, outParam);
    if (status != nvsipl::NVSIPL_STATUS_OK) {
        return status;
    }
    return nvsipl::NVSIPL_STATUS_OK;
}

nvsipl::SIPLStatus CAutoControlPlugin::Reset()
{
    m_prevExpVal = 0.0F;
    return nvsipl::NVSIPL_STATUS_OK;
}

nvsipl::ISiplControlAuto* CreatePlugin() {
    return new CAutoControlPlugin();
}
