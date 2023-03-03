/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "CNvMMAX96712TPG.hpp"
#include "devices/MAX96712DeserializerDriver/CNvMMax96712.hpp"
#include "ModuleIF/CNvMCameraModuleExport.hpp"
#include "sipl_error.h"

#include <chrono>
#include <thread>

namespace nvsipl
{

SIPLStatus CNvMMAX96712TPG::Init() {
    NvMediaStatus nvmediaStatus = NVMEDIA_STATUS_OK;
    DevBlkCDIDevice *cdiDeserializer = m_pDeserializer->GetCDIDeviceHandle();
    DataTypeMAX96712 dataType = CDI_MAX96712_DATA_TYPE_INVALID;
    WriteParametersParamMAX96712 paramsMAX96712 = {};
    NvSiplCapInputFormatType inputFormatType = m_upCameraModuleProperty->sensorProperty.inputFormat.inputFormatType;
    LinkPipelineMapMAX96712 *pipeLineMap = NULL;

    if (m_bPassive) {
        return NVSIPL_STATUS_OK;
    }

    if (m_initLinkMask != 0) {
        LOG_INFO("Set MAX96712 PG seting\n");
        paramsMAX96712.SetPGSetting.width = m_upCameraModuleProperty->sensorProperty.width;
        paramsMAX96712.SetPGSetting.height = m_upCameraModuleProperty->sensorProperty.height;
        paramsMAX96712.SetPGSetting.frameRate = m_upCameraModuleProperty->sensorProperty.frameRate;
        paramsMAX96712.SetPGSetting.linkIndex = m_linkIndex;
        nvmediaStatus = MAX96712WriteParameters(cdiDeserializer,
                                                CDI_WRITE_PARAM_CMD_MAX96712_SET_PG,
                                                sizeof(paramsMAX96712.SetPGSetting),
                                                &paramsMAX96712);
        if (nvmediaStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96712TPG: CDI_WRITE_PARAM_CMD_MAX96712_SET_PG failed with NvMedia error", (int32_t)nvmediaStatus);
            return ConvertNvMediaStatus(nvmediaStatus);
        }

        if (inputFormatType == NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW10) {
            dataType = CDI_MAX96712_DATA_TYPE_RAW10;
        } else if (inputFormatType == NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12) {
            dataType = CDI_MAX96712_DATA_TYPE_RAW12;
        } else if (inputFormatType == NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW16) {
            dataType = CDI_MAX96712_DATA_TYPE_RAW16;
        } else if (inputFormatType == NVSIPL_CAP_INPUT_FORMAT_TYPE_RGB888) {
            dataType = CDI_MAX96712_DATA_TYPE_RGB;
        } else if (inputFormatType == NVSIPL_CAP_INPUT_FORMAT_TYPE_YUV422) {
            dataType = CDI_MAX96712_DATA_TYPE_YUV_8;
        } else if (inputFormatType == NVSIPL_CAP_INPUT_FORMAT_TYPE_YUV422_10) {
            dataType = CDI_MAX96712_DATA_TYPE_YUV_10;
        } else {
            SIPL_LOG_ERR_STR_INT("MAX96712TPG: Input format not supported", (int32_t)inputFormatType);
            return NVSIPL_STATUS_ERROR;
        }

        /* Update mapping table */
        pipeLineMap = &paramsMAX96712.PipelineMappingTPG.linkPipelineMap[m_linkIndex];
        pipeLineMap->isEmbDataType = false;
        pipeLineMap->vcID = m_linkIndex;
        pipeLineMap->dataType = dataType;
        pipeLineMap->isDTOverride = true;

        LOG_INFO("Set MAX96712 pipeline mapping\n");
        paramsMAX96712.PipelineMappingTPG.linkIndex = m_linkIndex;
        LOG_DEBUG("Set pipeline mapping\n");
        nvmediaStatus = MAX96712WriteParameters(cdiDeserializer,
                                                CDI_WRITE_PARAM_CMD_MAX96712_PIPELINE_MAPPING_TPG,
                                                sizeof(paramsMAX96712.PipelineMappingTPG),
                                                &paramsMAX96712);
        if (nvmediaStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96712TPG: CDI_WRITE_PARAM_CMD_MAX96712_PIPELINE_MAPPING_TPG failed with NvMedia error", (int32_t)nvmediaStatus);
            return ConvertNvMediaStatus(nvmediaStatus);
        }

        // Override DataType
        pipeLineMap = &paramsMAX96712.PipelineMapping.linkPipelineMap[m_linkIndex];
        pipeLineMap->isEmbDataType = false;
        pipeLineMap->vcID = m_linkIndex;
        pipeLineMap->dataType = dataType;
        pipeLineMap->isDTOverride = true;
        paramsMAX96712.PipelineMapping.link = (LinkMAX96712)(1U << m_linkIndex);
        nvmediaStatus = MAX96712WriteParameters(cdiDeserializer,
                                                CDI_WRITE_PARAM_CMD_MAX96712_OVERRIDE_DATATYPE,
                                                sizeof(paramsMAX96712.PipelineMapping),
                                                &paramsMAX96712);
        if (nvmediaStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96712TPG: CDI_WRITE_PARAM_CMD_MAX96712_OVERRIDE_DATATYPE failed with NvMedia error", (int32_t)nvmediaStatus);
            return ConvertNvMediaStatus(nvmediaStatus);
        }

        LOG_INFO("Set MAX96712 maps unused pipelines\n");
        nvmediaStatus = MAX96712SetDeviceConfig(cdiDeserializer,
                                                CDI_CONFIG_MAX96712_MAP_UNUSED_PIPE);
        if (nvmediaStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96712TPG: CDI_CONFIG_MAX96712_MAP_UNUSED_PIPE failed with NvMedia error", (int32_t)nvmediaStatus);
            return ConvertNvMediaStatus(nvmediaStatus);
        }

        LOG_INFO("Set MAX96712 enable PG\n");
        nvmediaStatus = MAX96712SetDeviceConfig(cdiDeserializer,
                                                  CDI_CONFIG_MAX96712_ENABLE_PG);
        if (nvmediaStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96712TPG: CDI_CONFIG_MAX96712_ENABLE_PG failed with NvMedia error", (int32_t)nvmediaStatus);
            return ConvertNvMediaStatus(nvmediaStatus);
        }

        // Delay to skip some frames at the beginning
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMMAX96712TPG::DoSetConfig(const CameraModuleConfig *cameraModuleConfig, const uint8_t linkIndex)
{

    SIPLStatus status;

    m_pDeserializer = cameraModuleConfig->deserializer;
    m_initLinkMask = cameraModuleConfig->initLinkMask;
    m_bPassive = cameraModuleConfig->params->bPassive;

    const CameraModuleInfo *moduleInfo = cameraModuleConfig->cameraModuleInfo;

    // Create camera module property
    m_upCameraModuleProperty = std::move(std::unique_ptr<Property>(new Property));

    m_upSensor.reset(new CNvMSensor());
    status = m_upSensor->SetConfig(&moduleInfo->sensorInfo, cameraModuleConfig->params);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712TPG: Sensor SetConfig failed with SIPL error", (int32_t)status);
        return status;
    }

    CNvMSensor* sensor = m_upSensor.get();
    CNvMCameraModule::Property::SensorProperty oSensorProperty = {
        .id = moduleInfo->sensorInfo.id,
        .virtualChannelID = linkIndex,
        .inputFormat = sensor->GetInputFormat(),
        .pixelOrder = sensor->GetPixelOrder(),
        .width = sensor->GetWidth(),
        .height = sensor->GetHeight(),
        .startX = 0,
        .startY = 0,
        .embeddedTop = sensor->GetEmbLinesTop(),
        .embeddedBot = sensor->GetEmbLinesBot(),
        .frameRate = sensor->GetFrameRate(),
        .embeddedDataType = sensor->GetEmbDataType(),
        .pSensorControlHandle = nullptr,
    };

    m_upCameraModuleProperty->sensorProperty = oSensorProperty;

    return NVSIPL_STATUS_OK;
}

CNvMCameraModule::Property* CNvMMAX96712TPG::GetCameraModuleProperty() {
    return m_upCameraModuleProperty.get();
}

uint16_t CNvMMAX96712TPG::GetPowerOnDelayMs() {
    return 0;
}

uint16_t CNvMMAX96712TPG::GetPowerOffDelayMs() {
    return 0;
}

std::string CNvMMAX96712TPG::GetSupportedDeserailizer() {
    return "MAX96712";
}

CNvMDeserializer::LinkMode CNvMMAX96712TPG::GetLinkMode() {
    return CNvMDeserializer::LinkMode::LINK_MODE_GMSL1;
}

CNvMCameraModule *CNvMCameraModule_Create() {
    return new CNvMMAX96712TPG();
}

const char** CNvMCameraModule_GetNames() {
    static const char* names[] = {
        "MAX96712TPG",
        "MAX96712TPG_YUV_8",
        NULL
    };
    return names;
}

SIPLStatus
CNvMMAX96712TPG::GetSerializerErrorSize(size_t & serializerErrorSize)
{
#if USE_MOCK_ERRORS
    serializerErrorSize = 1;
#else
    serializerErrorSize = 0;
#endif
    return NVSIPL_STATUS_OK;
}

SIPLStatus
CNvMMAX96712TPG::GetSensorErrorSize(size_t & sensorErrorSize)
{
#if USE_MOCK_ERRORS
    sensorErrorSize = 1;
#else
    sensorErrorSize = 0;
#endif
    return NVSIPL_STATUS_OK;
}

SIPLStatus
CNvMMAX96712TPG::GetSerializerErrorInfo(std::uint8_t * const buffer,
                                        const std::size_t bufferSize,
                                        std::size_t &size)
{
#if USE_MOCK_ERRORS
    size = 1U;
#else
    size = 0U;
#endif
    return NVSIPL_STATUS_OK;
}

SIPLStatus
CNvMMAX96712TPG::GetSensorErrorInfo(std::uint8_t * const buffer,
                                    const std::size_t bufferSize,
                                    std::size_t &size)
{
#if USE_MOCK_ERRORS
    size = 1U;
#else
    size = 0U;
#endif
    return NVSIPL_STATUS_OK;
}

SIPLStatus
CNvMMAX96712TPG::NotifyLinkState(const NotifyLinkStates linkState)
{
    return NVSIPL_STATUS_OK;
}

Interface* CNvMMAX96712TPG::GetInterface(const UUID &interfaceId)
{
    return nullptr;
}

SIPLStatus CNvMMAX96712TPG::DoSetPower(bool powerOn)
{
    return NVSIPL_STATUS_OK;
}


SIPLStatus CNvMMAX96712TPG::GetInterruptStatus(const uint32_t gpioIdx,
                                               IInterruptNotify &intrNotifier)
{
    return NVSIPL_STATUS_NOT_SUPPORTED;
}

} // end of namespace
