/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "CNvMTransportLink_Max96712_96759.hpp"
#include "sipl_error.h"

#include <chrono>
#include <string>
#include <thread>

// Include necessary CDI driver headers
extern "C" {
#include "cdi_max96712.h"
#include "cdi_max96759.h"
}

namespace nvsipl {

#if !NV_IS_SAFETY
void CNvMTransportLink_Max96712_96759::DumpLinkParams()
{
    LOG_INFO("Link parameters\n");
    LOG_INFO("Link Index: %u \n", m_oLinkParams.ulinkIndex);
    LOG_INFO("Broadcast serializer addr: 0x%x \n", m_oLinkParams.uBrdcstSerAddr);
    LOG_INFO("Serializer addr: 0x%x \n", m_oLinkParams.uSerAddr);

    auto &sensor = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty;
    LOG_INFO("\nBroadcast sensor addr: 0x%x \n", sensor.uBrdcstSensorAddrs);
    LOG_INFO("Sensor addr: 0x%x \n", sensor.uSensorAddrs);
    LOG_INFO("VCID: %u \n", sensor.uVCID);
    LOG_INFO("Embedded data type: %s \n", sensor.bEmbeddedDataType ? "true" : "false");
    LOG_INFO("Trigger mode sync: %s \n", sensor.bEnableTriggerModeSync ? "true" : "false");
    LOG_INFO("Frame rate: %.2f fps \n", sensor.fFrameRate);

    LOG_INFO("Simulator mode: %u \n", m_oLinkParams.bEnableSimulator);
    LOG_INFO("Passive mode: %u \n", m_oLinkParams.bPassive);
}
#endif

SIPLStatus
CNvMTransportLink_Max96712_96759::SetupAddressTranslations(
    DevBlkCDIDevice* brdcstSerCDI)
{
    NvMediaStatus nvmStatus = NVMEDIA_STATUS_OK;
    ReadWriteParams96759 paramsMAX96759 = {};
    const auto &sensorProperties =
        m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty;

    // Setup address translation for the serializer
    paramsMAX96759.Translator = {};
    paramsMAX96759.Translator.source = m_oLinkParams.uSerAddr;
    paramsMAX96759.Translator.destination = m_oLinkParams.uBrdcstSerAddr;
    LOG_INFO("Translate image sensor device addr %x to %x\n",
              paramsMAX96759.Translator.source, paramsMAX96759.Translator.destination);
    nvmStatus = MAX96759WriteParameters(brdcstSerCDI,
                                        CDI_WRITE_PARAM_CMD_MAX96759_SET_TRANSLATOR_A,
                                        sizeof(paramsMAX96759.Translator),
                                        &paramsMAX96759);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96759: CDI_WRITE_PARAM_CMD_MAX96759_SET_TRANSLATOR_A failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    // Setup address translation for the FTDI chip
    paramsMAX96759.Translator = {};
    paramsMAX96759.Translator.source = sensorProperties.uSensorAddrs;
    paramsMAX96759.Translator.destination = sensorProperties.uBrdcstSensorAddrs;
    LOG_INFO("Translate image sensor device addr %x to %x\n",
              paramsMAX96759.Translator.source, paramsMAX96759.Translator.destination);
    nvmStatus = MAX96759WriteParameters(brdcstSerCDI,
                                        CDI_WRITE_PARAM_CMD_MAX96759_SET_TRANSLATOR_B,
                                        sizeof(paramsMAX96759.Translator),
                                        &paramsMAX96759);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96759: CDI_WRITE_PARAM_CMD_MAX96759_SET_TRANSLATOR_B failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    return NVSIPL_STATUS_OK;
}

// This will do the init for 1 maxim deser and up to MAX_LINKS_PER_DESER maxim serializers.
SIPLStatus CNvMTransportLink_Max96712_96759::Init(DevBlkCDIDevice* brdcstSerCDI, uint8_t linkMask, bool groupInitProg)
{
    NvMediaStatus nvmStatus = NVMEDIA_STATUS_OK;
    WriteParametersParamMAX96712 paramsMAX96712 {};
    ReadWriteParams96759 paramsMAX96759 = {};
    DataTypeMAX96712 dataType = {};
    LinkMAX96712 link = GetMAX96712Link(m_oLinkParams.ulinkIndex);
    DevBlkCDIDevice *deserializerCDI = m_oLinkParams.pDeserCDIDevice;
    LinkPipelineMapMAX96712 *pipeLineMap = nullptr;
    const auto & sensorProperties =
        m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty;
    std::string identifier;

    LOG_INFO("Initializing link %u in CNvMTransportLink_Max96712_96759::Init \n",
              m_oLinkParams.ulinkIndex);

#if !NV_IS_SAFETY
    DumpLinkParams();
#endif

    if (m_oLinkParams.bPassive or m_oLinkParams.bEnableSimulator) {
        return NVSIPL_STATUS_OK;
    }

    // Enable one link of the deserializer
    LOG_INFO("Enable link\n");
    paramsMAX96712.link = link;
    nvmStatus = MAX96712WriteParameters(deserializerCDI,
                                        CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_SPECIFIC_LINKS,
                                        sizeof(paramsMAX96712.link),
                                        &paramsMAX96712);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96759: CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_SPECIFIC_LINK failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    /* Check link lock */
    LOG_INFO("%s: Check config link lock \n", __func__);
    nvmStatus = MAX96712CheckLink(deserializerCDI,
                                  GetMAX96712Link(m_oLinkParams.ulinkIndex),
                                  CDI_MAX96712_LINK_LOCK_GMSL2,
                                  true);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96759: MAX96712CheckLink(CDI_MAX96712_GMSL2_LINK_LOCK) failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    // Check serializer presence
    nvmStatus = MAX96759CheckPresence(brdcstSerCDI);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96759: MAX96759CheckPresence failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    // Reset serializer
    nvmStatus = MAX96759SetDeviceConfig(brdcstSerCDI,
                                        CDI_CONFIG_MAX96759_SET_RESET);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96759: CDI_CONFIG_MAX96759_SET_RESET failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Set serializer defaults
    nvmStatus = MAX96759SetDefaults(brdcstSerCDI);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96759: MAX96759SetDefaults failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    // Setup serializer to forward video to link B
    paramsMAX96759.LinkMode = {};
    paramsMAX96759.LinkMode.mode = LINK_MODE_MAX96759_LINK_B;
    nvmStatus = MAX96759WriteParameters(brdcstSerCDI,
                                        CDI_WRITE_PARAM_CMD_MAX96759_SET_LINK_MODE,
                                        sizeof(paramsMAX96759.LinkMode),
                                        &paramsMAX96759.LinkMode);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96759: CDI_WRITE_PARAM_CMD_MAX96759_SET_LINK_MODE failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Setup the BPP to truncate to 12 bpp
    paramsMAX96759.BitsPerPixel = 12u;
    nvmStatus = MAX96759WriteParameters(brdcstSerCDI,
                                        CDI_WRITE_PARAM_CMD_MAX96759_SET_BPP,
                                        sizeof(paramsMAX96759.BitsPerPixel),
                                        &paramsMAX96759.BitsPerPixel);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96759: CDI_WRITE_PARAM_CMD_MAX96759_SET_BPP failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    SetupAddressTranslations(brdcstSerCDI);

    switch (sensorProperties.inputFormat.inputFormatType)
    {
        case NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW10:
            dataType = CDI_MAX96712_DATA_TYPE_RAW10;
            break;
        case NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12:
        case NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12RJ:
            dataType = CDI_MAX96712_DATA_TYPE_RAW12;
            break;
        default:
            SIPL_LOG_ERR_STR_INT("MAX96759: Input format not supported", (int32_t)sensorProperties.inputFormat.inputFormatType);
            return NVSIPL_STATUS_NOT_SUPPORTED;
    }

    // Update mapping table
    pipeLineMap = &paramsMAX96712.PipelineMapping.linkPipelineMap[m_oLinkParams.ulinkIndex];
    pipeLineMap->isEmbDataType = sensorProperties.bEmbeddedDataType;
    pipeLineMap->vcID = sensorProperties.uVCID;
    pipeLineMap->dataType = dataType;
    pipeLineMap->isDTOverride = !sensorProperties.bEmbeddedDataType;
    paramsMAX96712.PipelineMapping.link = GetMAX96712Link(m_oLinkParams.ulinkIndex);
    nvmStatus = MAX96712WriteParameters(deserializerCDI,
                                        CDI_WRITE_PARAM_CMD_MAX96712_PIPELINE_MAPPING,
                                        sizeof(paramsMAX96712.PipelineMapping),
                                        &paramsMAX96712);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: CDI_WRITE_PARAM_CMD_MAX96712_PIPELINE_MAPPING failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    // Override DataType
    pipeLineMap = &paramsMAX96712.PipelineMapping.linkPipelineMap[m_oLinkParams.ulinkIndex];
    pipeLineMap->isEmbDataType = sensorProperties.bEmbeddedDataType;
    pipeLineMap->vcID = sensorProperties.uVCID;
    pipeLineMap->dataType = dataType;
    pipeLineMap->isDTOverride = !sensorProperties.bEmbeddedDataType;
    paramsMAX96712.PipelineMapping.link = GetMAX96712Link(m_oLinkParams.ulinkIndex);
    nvmStatus = MAX96712WriteParameters(deserializerCDI,
                                        CDI_WRITE_PARAM_CMD_MAX96712_OVERRIDE_DATATYPE,
                                        sizeof(paramsMAX96712.PipelineMapping),
                                        &paramsMAX96712);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: CDI_WRITE_PARAM_CMD_MAX96712_OVERRIDE_DATATYPE failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    // Enable TPG
    if (sensorProperties.bEnableTPG) {
        paramsMAX96759.EDID = {};
        paramsMAX96759.TPG.width  = sensorProperties.width;
        paramsMAX96759.TPG.height = sensorProperties.height +
                                    sensorProperties.embeddedTop +
                                    sensorProperties.embeddedBot;
        paramsMAX96759.TPG.frameRate = sensorProperties.fFrameRate;
        nvmStatus = MAX96759WriteParameters(brdcstSerCDI,
                                            CDI_WRITE_PARAM_CMD_MAX96759_SET_TPG,
                                            sizeof(paramsMAX96759.TPG),
                                            &paramsMAX96759.TPG);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96759: CDI_WRITE_PARAM_CMD_MAX96759_SET_TPG failed with NvMedia error", (int32_t)nvmStatus);
            return ConvertNvMediaStatus(nvmStatus);
        }
    }
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif // !NV_IS_SAFETY

    // Program the EDID
    paramsMAX96759.EDID = {};
    identifier = sensorProperties.sensorDescription;
    paramsMAX96759.EDID.width  = sensorProperties.width;
    paramsMAX96759.EDID.height = sensorProperties.height +
                                 sensorProperties.embeddedTop +
                                sensorProperties.embeddedBot;
    paramsMAX96759.EDID.frameRate = sensorProperties.fFrameRate;
    paramsMAX96759.EDID.identifier = identifier.c_str();
    nvmStatus = MAX96759WriteParameters(brdcstSerCDI,
                                        CDI_WRITE_PARAM_CMD_MAX96759_SET_EDID,
                                        sizeof(paramsMAX96759.EDID),
                                        &paramsMAX96759.EDID);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96759: CDI_WRITE_PARAM_CMD_MAX96759_SET_EDID failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    return NVSIPL_STATUS_OK;
}

} // end of namespace nvsipl
