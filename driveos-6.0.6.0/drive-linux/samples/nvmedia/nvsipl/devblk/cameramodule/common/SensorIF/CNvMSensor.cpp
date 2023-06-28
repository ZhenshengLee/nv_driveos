/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include <cstring>

#include "CNvMSensor.hpp"
#include "sipl_error.h"

namespace nvsipl
{

CNvMSensor::CNvMSensor(): CNvMDevice(), ISensorControl(), m_pipelineIndex {0U},
    m_embLinesTop {0U}, m_embLinesBot {0U}, m_bEmbDataType {false},
    m_width {0U}, m_height {0U}, m_ePixelOrder {0U},
    m_frameRate {static_cast<float_t>(0.0)}, m_bEnableExtSync {false},
    m_bIsAuthEnabled {false}
{
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    m_bEnabletpg = false;
    m_patternMode = 0U;
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
    m_eInputFormat.inputFormatType = NVSIPL_CAP_INPUT_FORMAT_TYPE_YUV422;
    m_eInputFormat.bitsPerPixel = NVSIPL_BITS_PER_PIXEL_8;
}

SIPLStatus CNvMSensor::SetConfig(SensorInfo const* const sensorInformation, DeviceParams const* const params)
{
    SIPLStatus status {NVSIPL_STATUS_OK};
    bool exitFlag {false};

    if ((sensorInformation == nullptr) || (params == nullptr)) {
        SIPL_LOG_ERR_STR("Invalid arguments passed");
        status = NVSIPL_STATUS_BAD_ARGUMENT;
        exitFlag = true;
    }

    if ((!exitFlag) && (m_eState != CREATED)) {
        SIPL_LOG_ERR_STR("CDI invalid state");
        status = NVSIPL_STATUS_INVALID_STATE;
        exitFlag = true;
    }

    if (!exitFlag) {
        m_embLinesTop = sensorInformation->vcInfo.embeddedTopLines;
        m_embLinesBot = sensorInformation->vcInfo.embeddedBottomLines;
        m_bEmbDataType = sensorInformation->vcInfo.isEmbeddedDataTypeEnabled;
        m_width = sensorInformation->vcInfo.resolution.width;
        m_height = sensorInformation->vcInfo.resolution.height;
        m_eInputFormat.inputFormatType = sensorInformation->vcInfo.inputFormat;

        if ((m_embLinesTop == 0U) and m_bEmbDataType) {
            SIPL_LOG_ERR_STR("Embedded data type must be disabled if top emb data lines is 0");
            status = NVSIPL_STATUS_NOT_SUPPORTED;
            exitFlag = true;
        }
    }

    if (!exitFlag) {
        status = SetInputFormatProperty();
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("CNvMSensor::SetConfig failed with SIPL error",
                                  static_cast<int32_t>(status));
            exitFlag = true;
        }
    }

    if (!exitFlag) {
        m_ePixelOrder = sensorInformation->vcInfo.cfa;
        m_frameRate = sensorInformation->vcInfo.fps;
        m_bEnableExtSync = sensorInformation->isTriggerModeEnabled;
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
        m_bEnabletpg = sensorInformation->isTPGEnabled;
        m_patternMode = sensorInformation->patternMode;
        m_sensorDescription = sensorInformation->description;
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
        m_oDeviceParams = *params;
        if ((!m_oDeviceParams.bEnableSimulator) and (!m_oDeviceParams.bPassive)) {
            m_nativeI2CAddr =  sensorInformation->i2cAddress;
            m_oDeviceParams.pI2CAddrMgr->RegisterNativeI2CAddr(m_nativeI2CAddr);
        }
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
#if (USE_CDAC == 0)
        m_oDeviceParams.bUseCDIv2API = NVMEDIA_FALSE;
#else
        m_oDeviceParams.bUseCDIv2API = sensorInformation->useCDIv2API;
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#else
        m_oDeviceParams.bUseCDIv2API = NVMEDIA_TRUE;
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
        m_ePixelOrder = sensorInformation->vcInfo.cfa;
        m_eState = CDI_DEVICE_CONFIG_SET;
        m_bIsAuthEnabled = sensorInformation->isAuthEnabled;
    }
    return status;
}

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
void CNvMSensor::SetPipelineIndex(uint32_t const index)
{
    m_pipelineIndex = index;
}

uint32_t CNvMSensor::GetPipelineIndex() const
{
    return m_pipelineIndex;
}
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif // !NV_IS_SAFETY

uint32_t CNvMSensor::GetWidth() const
{
    return m_width;
}

uint32_t CNvMSensor::GetHeight() const
{
    return m_height;
}

uint32_t CNvMSensor::GetEmbLinesTop() const
{
    return m_embLinesTop;
}

uint32_t CNvMSensor::GetEmbLinesBot() const
{
    return m_embLinesBot;
}

bool CNvMSensor::GetEmbDataType() const
{
    return m_bEmbDataType;
}

NvSiplCapInputFormat CNvMSensor::GetInputFormat() const
{
    return m_eInputFormat;
}

uint32_t CNvMSensor::GetPixelOrder() const
{
    return m_ePixelOrder;
}

float_t CNvMSensor::GetFrameRate() const
{
    return m_frameRate;
}

bool CNvMSensor::GetEnableExtSync() const
{
    return m_bEnableExtSync;
}

bool CNvMSensor::IsAuthenticationEnabled() const
{
    return m_bIsAuthEnabled;
}

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
bool CNvMSensor::GetEnableTPG() const
{
    return m_bEnabletpg;
}

uint32_t CNvMSensor::GetPatternMode() const
{
    return m_patternMode;
}
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

std::string CNvMSensor::GetSensorDescription() const
{
    return m_sensorDescription;
}

SIPLStatus CNvMSensor::SetInputFormatProperty()
{
    SIPLStatus status {NVSIPL_STATUS_OK};

    if (m_eInputFormat.inputFormatType == NVSIPL_CAP_INPUT_FORMAT_TYPE_YUV422) {
        m_eInputFormat.bitsPerPixel = NVSIPL_BITS_PER_PIXEL_8;
    } else if (m_eInputFormat.inputFormatType == NVSIPL_CAP_INPUT_FORMAT_TYPE_YUV422_10) {
        m_eInputFormat.bitsPerPixel = NVSIPL_BITS_PER_PIXEL_10;
    } else if (m_eInputFormat.inputFormatType == NVSIPL_CAP_INPUT_FORMAT_TYPE_RGB888) {
        m_eInputFormat.bitsPerPixel = NVSIPL_BITS_PER_PIXEL_8;
    } else if (m_eInputFormat.inputFormatType == NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW8) {
        m_eInputFormat.bitsPerPixel = NVSIPL_BITS_PER_PIXEL_8;
    } else if (m_eInputFormat.inputFormatType == NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW10) {
        m_eInputFormat.bitsPerPixel = NVSIPL_BITS_PER_PIXEL_10;
    } else if ((m_eInputFormat.inputFormatType == NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12) ||
               (m_eInputFormat.inputFormatType == NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12RJ)) {
        m_eInputFormat.bitsPerPixel = NVSIPL_BITS_PER_PIXEL_12;
    } else if (m_eInputFormat.inputFormatType == NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW16) {
        m_eInputFormat.bitsPerPixel = NVSIPL_BITS_PER_PIXEL_16;
    } else {
        SIPL_LOG_ERR_STR_INT("Unknown input format",
                              static_cast<int32_t>(m_eInputFormat.inputFormatType));
        status = NVSIPL_STATUS_BAD_ARGUMENT;
    }

    return status;
}

#if !NV_IS_SAFETY
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
CNvMSensor::EEPROMRequestHandle CNvMSensor::GetEEPROMRequestHandle() const
{
    return m_EEPROMRequestHandle;
}

void CNvMSensor::SetEEPROMRequestHandle(void *const handle, const uint32_t size)
{
    if (NULL != handle) {
        m_EEPROMRequestHandle.handle = handle;
        m_EEPROMRequestHandle.size = size;
    }
}
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif // !NV_IS_SAFETY

NvMediaStatus CNvMSensor::SIPLParseTopEmbDataInfo(
    DevBlkCDIEmbeddedDataChunk const* const embeddedTopDataChunk,
    size_t const embeddedDataChunkStructSize,
    DevBlkCDIEmbeddedDataInfo *const embeddedDataInfo,
    size_t const dataInfoStructSize)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    if (m_upCDIDevice == nullptr) {
        status = NVMEDIA_STATUS_NOT_INITIALIZED;
    } else {
        status = DevBlkCDIParseTopEmbDataInfo(m_upCDIDevice.get(),
                                             embeddedTopDataChunk,
                                             embeddedDataChunkStructSize,
                                             embeddedDataInfo,
                                             dataInfoStructSize);
    }
    return status;
}

NvMediaStatus CNvMSensor::SIPLParseBotEmbDataInfo(
    DevBlkCDIEmbeddedDataChunk const* const embeddedBotDataChunk,
    size_t const embeddedDataChunkStructSize,
    DevBlkCDIEmbeddedDataInfo *const embeddedDataInfo,
    size_t const dataInfoStructSize)
{
    NvMediaStatus status;
    if (m_upCDIDevice == nullptr) {
        status = NVMEDIA_STATUS_NOT_INITIALIZED;
    } else {
        status = DevBlkCDIParseBotEmbDataInfo(m_upCDIDevice.get(),
                                             embeddedBotDataChunk,
                                             embeddedDataChunkStructSize,
                                             embeddedDataInfo,
                                             dataInfoStructSize);
    }
    return status;
}

NvMediaStatus CNvMSensor::SIPLSetSensorControls(
                            DevBlkCDISensorControl const* const sensorControl,
                            size_t const sensrCtrlStructSize)
{
    NvMediaStatus status {NVMEDIA_STATUS_OK};
    if (m_upCDIDevice == nullptr) {
        status = NVMEDIA_STATUS_NOT_INITIALIZED;
    } else {
        status = DevBlkCDISetSensorControls(m_upCDIDevice.get(),
                                            sensorControl,
                                            sensrCtrlStructSize);
    }
    return status;
}

NvMediaStatus CNvMSensor::SIPLGetSensorAttributes(
                            DevBlkCDISensorAttributes *const sensorAttr,
                            size_t const sensorAttrStructSize)
{
    NvMediaStatus status {NVMEDIA_STATUS_OK};
    if (m_upCDIDevice == nullptr) {
        status = NVMEDIA_STATUS_NOT_INITIALIZED;
    } else {
        status = DevBlkCDIGetSensorAttributes(m_upCDIDevice.get(),
                                              sensorAttr,
                                              sensorAttrStructSize);
    }
    return status;
}

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
NvMediaStatus CNvMSensor::SIPLSetSensorCharMode(uint8_t expNo)
{
    if (m_upCDIDevice == nullptr) {
        return NVMEDIA_STATUS_NOT_INITIALIZED;
    }

    return DevBlkCDISetSensorCharMode(m_upCDIDevice.get(),
                                      expNo);
}
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif // !NV_IS_SAFETY

NvMediaStatus
CNvMSensor::SIPLAuthenticateImage(DevBlkImageDesc const * const imageDesc) const
{
    NvMediaStatus status;

    if (m_upCDIDevice == nullptr) {
        status = NVMEDIA_STATUS_NOT_INITIALIZED;
    } else {
        status = DevBlkCDIAuthenticateImage(m_upCDIDevice.get(), imageDesc);
    }

    return status;
}

} // end of nvsipl
