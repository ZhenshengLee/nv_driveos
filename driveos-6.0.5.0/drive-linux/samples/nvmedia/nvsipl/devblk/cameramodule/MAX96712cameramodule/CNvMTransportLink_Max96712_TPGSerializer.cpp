/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "CNvMTransportLink_Max96712_TPGSerializer.hpp"
#include "sipl_error.h"

#include <chrono>
#include <thread>

// Include necessary CDI driver headers
extern "C" {
#include "cdi_max96712.h"
}

namespace nvsipl
{

SIPLStatus CNvMTransportLink_Max96712_TPGSerializer::Init(DevBlkCDIDevice* brdcstSerCDI, uint8_t linkMask, bool groupInitProg) {
    NvMediaStatus nvmediaStatus = NVMEDIA_STATUS_OK;
    DevBlkCDIDevice *cdiDeserializer = m_oLinkParams.pDeserCDIDevice;
    DataTypeMAX96712 dataTypeMAX96712 = CDI_MAX96712_DATA_TYPE_INVALID;
    WriteParametersParamMAX96712 paramsMAX96712 = {};
    LinkPipelineMapMAX96712 *pipeLineMap = NULL;

    if (m_oLinkParams.bPassive or m_oLinkParams.bEnableSimulator) {
        return NVSIPL_STATUS_OK;
    }

    LOG_INFO("Set MAX96712 PG seting\n");
    paramsMAX96712.SetPGSetting.width = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.width;
    paramsMAX96712.SetPGSetting.height = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.height +
                                         m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.embeddedTop +
                                         m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.embeddedBot;
    paramsMAX96712.SetPGSetting.frameRate = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.fFrameRate;
    paramsMAX96712.SetPGSetting.linkIndex = m_oLinkParams.ulinkIndex;
    nvmediaStatus = MAX96712WriteParameters(cdiDeserializer,
                                            CDI_WRITE_PARAM_CMD_MAX96712_SET_PG,
                                            sizeof(paramsMAX96712.SetPGSetting),
                                            &paramsMAX96712);
    if (nvmediaStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712TPG: CDI_WRITE_PARAM_CMD_MAX96712_SET_PG failed with NvMedia error", (int32_t)nvmediaStatus);
        return ConvertNvMediaStatus(nvmediaStatus);
    }

    switch (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.inputFormat.inputFormatType)
    {
        case NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW10:
            dataTypeMAX96712 = CDI_MAX96712_DATA_TYPE_RAW10;
            break;
        case NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12:
            dataTypeMAX96712 = CDI_MAX96712_DATA_TYPE_RAW12;
            break;
        case NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW16:
            dataTypeMAX96712 = CDI_MAX96712_DATA_TYPE_RAW16;
            break;
        case NVSIPL_CAP_INPUT_FORMAT_TYPE_RGB888:
            dataTypeMAX96712 = CDI_MAX96712_DATA_TYPE_RGB;
            break;
        case NVSIPL_CAP_INPUT_FORMAT_TYPE_YUV422:
            dataTypeMAX96712 = CDI_MAX96712_DATA_TYPE_YUV_8;
            break;
        default:
            SIPL_LOG_ERR_STR_INT(
                "MAX96712: Input format not supported",
                (int32_t)m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.inputFormat.inputFormatType
            );
            return NVSIPL_STATUS_NOT_SUPPORTED;
    }

    /* Update mapping table */
    pipeLineMap = &paramsMAX96712.PipelineMappingTPG.linkPipelineMap[m_oLinkParams.ulinkIndex];
    pipeLineMap->isEmbDataType = false;
    pipeLineMap->vcID = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.uVCID;
    pipeLineMap->dataType = dataTypeMAX96712;
    pipeLineMap->isDTOverride = true;

    LOG_INFO("Set MAX96712 pipeline mapping\n");
    paramsMAX96712.PipelineMappingTPG.linkIndex = m_oLinkParams.ulinkIndex;
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
    pipeLineMap = &paramsMAX96712.PipelineMapping.linkPipelineMap[m_oLinkParams.ulinkIndex];
    pipeLineMap->isEmbDataType = false;
    pipeLineMap->vcID = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.uVCID;
    pipeLineMap->dataType = dataTypeMAX96712;
    pipeLineMap->isDTOverride = true;
    paramsMAX96712.PipelineMapping.link = GetMAX96712Link(m_oLinkParams.ulinkIndex);
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
    return NVSIPL_STATUS_OK;
}



} // end of namespace
