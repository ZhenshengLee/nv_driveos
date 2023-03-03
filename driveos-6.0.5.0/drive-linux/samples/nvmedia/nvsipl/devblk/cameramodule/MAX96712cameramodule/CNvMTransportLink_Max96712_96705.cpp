/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "CNvMTransportLink_Max96712_96705.hpp"
#include "sipl_error.h"

#include <chrono>
#include <string>
#include <thread>

// Include necessary CDI driver headers
extern "C" {
#include "cdi_max96712.h"
#include "cdi_max96705.h"
}

namespace nvsipl {

#if !NV_IS_SAFETY
void CNvMTransportLink_Max96712_96705::DumpLinkParams()
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

    auto broadEEPROMAddr = m_oLinkParams.moduleConnectionProperty.brdcstEepromAddr;
    LOG_INFO("\nBroadcast EEPROM addr: 0x%x \n", broadEEPROMAddr);

    auto EEPROMAddr = m_oLinkParams.moduleConnectionProperty.eepromAddr;
    LOG_INFO("EEPROM addr: 0x%x \n", EEPROMAddr);

    LOG_INFO("Simulator mode: %u \n", m_oLinkParams.bEnableSimulator);
    LOG_INFO("Passive mode: %u \n", m_oLinkParams.bPassive);
}
#endif

SIPLStatus CNvMTransportLink_Max96712_96705::SetupAddressTranslationsSer(DevBlkCDIDevice* brdcstSerCDI, bool groupInitProg)
{
    NvMediaStatus nvmStatus = NVMEDIA_STATUS_OK;
    WriteReadParametersParamMAX96705 paramsMAX96705 = {};

    // Check serializer is present
    LOG_INFO("Check serializer is present\n");
    nvmStatus = MAX96705CheckPresence(brdcstSerCDI);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96705: MAX96705CheckPresence(Ser) failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    // Set address translation for the sensor
    paramsMAX96705.Translator.source = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.uSensorAddrs;
    paramsMAX96705.Translator.destination = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.uBrdcstSensorAddrs;
    LOG_INFO("Translate image sensor device addr %x to %x\n",
              paramsMAX96705.Translator.source, paramsMAX96705.Translator.destination);
    nvmStatus = MAX96705WriteParameters(brdcstSerCDI,
                                        CDI_WRITE_PARAM_CMD_MAX96705_SET_TRANSLATOR_A,
                                        sizeof(paramsMAX96705.Translator),
                                        &paramsMAX96705);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96705: CDI_WRITE_PARAM_CMD_MAX96705_SET_TRANSLATOR_A failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    // Set unique address with broadcast address
    LOG_INFO("Set unique address\n");
    paramsMAX96705.DeviceAddress.address = m_oLinkParams.uSerAddr;
    nvmStatus = MAX96705WriteParameters(brdcstSerCDI,
                                        CDI_WRITE_PARAM_CMD_MAX96705_SET_DEVICE_ADDRESS,
                                        sizeof(paramsMAX96705.DeviceAddress.address),
                                        &paramsMAX96705);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96705: CDI_WRITE_PARAM_CMD_MAX96705_SET_DEVICE_ADDRESS failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMTransportLink_Max96712_96705::SetupAddressTranslationsEEPROM() const
{
    NvMediaStatus nvmStatus = NVMEDIA_STATUS_OK;
    WriteReadParametersParamMAX96705 paramsMAX96705 = {};
    DevBlkCDIDevice *serCDI = m_oLinkParams.pSerCDIDevice;

    // Set address translation for the eeprom
    if (m_oLinkParams.moduleConnectionProperty.eepromAddr != UINT8_MAX) {
        paramsMAX96705.Translator.source = m_oLinkParams.moduleConnectionProperty.eepromAddr;
        paramsMAX96705.Translator.destination = m_oLinkParams.moduleConnectionProperty.brdcstEepromAddr;

        LOG_INFO("Translate serializer device broadcast addr %x to %x\n", paramsMAX96705.Translator.source,
                                                                          paramsMAX96705.Translator.destination);
        nvmStatus = MAX96705WriteParameters(serCDI,
                                            CDI_WRITE_PARAM_CMD_MAX96705_SET_TRANSLATOR_B,
                                            sizeof(paramsMAX96705.Translator),
                                            &paramsMAX96705);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96705: CDI_WRITE_PARAM_CMD_MAX96705_SET_TRANSLATOR_B failed with NvMedia error", (int32_t)nvmStatus);
            return ConvertNvMediaStatus(nvmStatus);
        }
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMTransportLink_Max96712_96705::SetupConfigLink(DevBlkCDIDevice* brdcstSerCDI, uint8_t linkMask, bool groupInitProg)
{
    NvMediaStatus nvmStatus = NVMEDIA_STATUS_OK;
    WriteParametersParamMAX96712 writeParamsMAX96712 {};
    WriteReadParametersParamMAX96705 paramsMAX96705 = {};
    DevBlkCDIDevice *deserializerCDI = m_oLinkParams.pDeserCDIDevice;

    LOG_INFO("Set GMSL1 HIM DESER defaults-step0\n");
    writeParamsMAX96712.GMSL1HIMEnabled.link = GetMAX96712Link(m_oLinkParams.ulinkIndex);
    writeParamsMAX96712.GMSL1HIMEnabled.step = 0;
    nvmStatus = MAX96712WriteParameters(deserializerCDI,
                                        CDI_WRITE_PARAM_CMD_MAX96712_SET_DEFAULT_GMSL1_HIM_ENABLED,
                                        sizeof(writeParamsMAX96712.GMSL1HIMEnabled),
                                        &writeParamsMAX96712);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96705: CDI_WRITE_PARAM_CMD_MAX96712_SET_DEFAULT_GMSL1_HIM_ENABLED, Step 1 failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    LOG_INFO("Turn off HIM of serializer\n");
    nvmStatus = MAX96705SetDeviceConfig(brdcstSerCDI, CDI_CONFIG_MAX96705_DISABLE_HIM_MODE);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96705: CDI_CONFIG_MAX96705_DISABLE_HIM_MODE failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(15));

    LOG_INFO("Set GMSL1 HIM DESER defaults-step1\n");
    writeParamsMAX96712.GMSL1HIMEnabled.link = GetMAX96712Link(m_oLinkParams.ulinkIndex);
    writeParamsMAX96712.GMSL1HIMEnabled.step = 1;
    nvmStatus = MAX96712WriteParameters(deserializerCDI,
                                        CDI_WRITE_PARAM_CMD_MAX96712_SET_DEFAULT_GMSL1_HIM_ENABLED,
                                        sizeof(writeParamsMAX96712.GMSL1HIMEnabled),
                                        &writeParamsMAX96712);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96705: CDI_WRITE_PARAM_CMD_MAX96712_SET_DEFAULT_GMSL1_HIM_ENABLED, Step 2 failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    LOG_INFO("Enable reverse channel for serializer\n");
    nvmStatus = MAX96705SetDeviceConfig(brdcstSerCDI, CDI_CONFIG_MAX96705_ENABLE_REVERSE_CHANNEL);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96705: CDI_CONFIG_MAX96705_ENABLE_REVERSE_CHANNEL failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    LOG_INFO("Disable all forward channel for deserializer\n");
    writeParamsMAX96712.link = GetMAX96712Link(m_oLinkParams.ulinkIndex);
    nvmStatus = MAX96712WriteParameters(deserializerCDI,
                                        CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_FORWARD_CHANNELS,
                                        sizeof(writeParamsMAX96712.link),
                                        &writeParamsMAX96712);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96705: CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_FORWARD_CHANNELS failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    LOG_INFO("Enable HIM for serializer\n");
    nvmStatus = MAX96705SetDeviceConfig(brdcstSerCDI, CDI_CONFIG_MAX96705_ENABLE_HIM_MODE);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96705: CDI_CONFIG_MAX96705_ENABLE_HIM_MODE failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    LOG_INFO("Set GMSL1 HIM DESER defaults-step2\n");
    writeParamsMAX96712.GMSL1HIMEnabled.link = GetMAX96712Link(m_oLinkParams.ulinkIndex);
    writeParamsMAX96712.GMSL1HIMEnabled.step = 2;
    nvmStatus = MAX96712WriteParameters(deserializerCDI,
                                        CDI_WRITE_PARAM_CMD_MAX96712_SET_DEFAULT_GMSL1_HIM_ENABLED,
                                        sizeof(writeParamsMAX96712.GMSL1HIMEnabled),
                                        &writeParamsMAX96712);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96705: CDI_WRITE_PARAM_CMD_MAX96712_SET_DEFAULT_GMSL1_HIM_ENABLED, Step 3 failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    LOG_INFO("Enable the forward channel for deserializer\n");
    writeParamsMAX96712.link = GetMAX96712Link(m_oLinkParams.ulinkIndex);
    nvmStatus = MAX96712WriteParameters(deserializerCDI,
                                        CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_FORWARD_CHANNELS,
                                        sizeof(writeParamsMAX96712.link),
                                        &writeParamsMAX96712);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96705: CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_FORWARD_CHANNELS failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    LOG_INFO("Disable AUTO ACK\n");
    writeParamsMAX96712.link = GetMAX96712Link(m_oLinkParams.ulinkIndex);
    nvmStatus = MAX96712WriteParameters(deserializerCDI,
                                        CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_AUTO_ACK,
                                        sizeof(writeParamsMAX96712.link),
                                        &writeParamsMAX96712);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96705: CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_AUTO_ACK failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    // Set Vsync regeneration
    paramsMAX96705.vsyncRegen.vsync_high = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.vsyncHigh;
    paramsMAX96705.vsyncRegen.vsync_low = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.vsyncLow;
    paramsMAX96705.vsyncRegen.vsync_delay = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.vsyncDelay;
    paramsMAX96705.vsyncRegen.vsync_trig = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.vsyncTrig;
    paramsMAX96705.vsyncRegen.pclk = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.pclk;

    LOG_INFO("Set Vsync regeneration\n");
    nvmStatus = MAX96705WriteParameters(brdcstSerCDI,
                                        CDI_WRITE_PARAM_CMD_MAX96705_REGEN_VSYNC,
                                        sizeof(paramsMAX96705.vsyncRegen),
                                        &paramsMAX96705);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96705: CDI_WRITE_PARAM_CMD_MAX96705_REGEN_VSYNC failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    LOG_INFO("Enable the packet based control channel in MAX96712\n");
    writeParamsMAX96712.link = GetMAX96712Link(m_oLinkParams.ulinkIndex);
    nvmStatus = MAX96712WriteParameters(deserializerCDI,
                                        CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_PACKET_BASED_CONTROL_CHANNEL,
                                        sizeof(writeParamsMAX96712.link),
                                        &writeParamsMAX96712);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96705: CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_PACKET_BASED_CONTROL_CHANNEL failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(15));

    LOG_INFO("Disable DE\n");
    writeParamsMAX96712.link = GetMAX96712Link(m_oLinkParams.ulinkIndex);
    nvmStatus = MAX96712WriteParameters(deserializerCDI,
                                        CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_DE,
                                        sizeof(writeParamsMAX96712.link),
                                        &writeParamsMAX96712);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96705: CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_DE failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    LOG_INFO("Check config link lock \n");
    nvmStatus = MAX96712CheckLink(deserializerCDI,
                                  GetMAX96712Link(m_oLinkParams.ulinkIndex),
                                  CDI_MAX96712_LINK_LOCK_GMSL1_CONFIG,
                                  true);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96705: MAX96712CheckLink(CDI_MAX96712_GMSL1_CONFIG_LINK_LOCK) failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    LOG_INFO("Set periodic AEQ\n");
    writeParamsMAX96712.link = GetMAX96712Link(m_oLinkParams.ulinkIndex);
    nvmStatus = MAX96712WriteParameters(deserializerCDI,
                                        CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_PERIODIC_AEQ,
                                        sizeof(writeParamsMAX96712.link),
                                        &writeParamsMAX96712);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96705: CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_PERIODIC_AEQ failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    // Set PreEmphasis
    paramsMAX96705.preemp = CDI_SET_PREEMP_MAX96705_PLU_3_3DB;
    LOG_INFO("Set Preemphasis setting for Serializer\n");
    nvmStatus = MAX96705WriteParameters(brdcstSerCDI,
                                        CDI_WRITE_PARAM_CMD_MAX96705_SET_PREEMP,
                                        sizeof(paramsMAX96705.preemp),
                                        &paramsMAX96705);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96705: CDI_WRITE_PARAM_CMD_MAX96705_SET_PREEMP failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMTransportLink_Max96712_96705::SetupVideoLink()
{

    NvMediaStatus nvmStatus = NVMEDIA_STATUS_OK;
    WriteParametersParamMAX96712 paramsMAX96712 = {};
    DataTypeMAX96712 dataType = {};
    LinkPipelineMapMAX96712 *pipeLineMap = &paramsMAX96712.PipelineMapping.linkPipelineMap[m_oLinkParams.ulinkIndex];
    DevBlkCDIDevice *deserializerCDI = m_oLinkParams.pDeserCDIDevice;
    const auto & sensorProperties =
        m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty;

    switch (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.inputFormat.inputFormatType)
    {
        case NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW10:
            dataType = CDI_MAX96712_DATA_TYPE_RAW10;
            break;

        case NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12:
            dataType = CDI_MAX96712_DATA_TYPE_RAW12;
            break;

        case NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW16:
            dataType = CDI_MAX96712_DATA_TYPE_RAW16;
            break;

        case NVSIPL_CAP_INPUT_FORMAT_TYPE_RGB888:
            dataType =  CDI_MAX96712_DATA_TYPE_RGB;
            break;

        default:
            SIPL_LOG_ERR_STR_INT("MAX96705: Input format not supported",
                                 (int32_t)m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.inputFormat.inputFormatType);
            return NVSIPL_STATUS_NOT_SUPPORTED;
    }

    // Update mapping table
    pipeLineMap->isEmbDataType = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bEmbeddedDataType;
    pipeLineMap->vcID = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.uVCID;
    pipeLineMap->dataType = dataType;
    pipeLineMap->isDTOverride = !m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bEmbeddedDataType;
    paramsMAX96712.PipelineMapping.link = GetMAX96712Link(m_oLinkParams.ulinkIndex);
    nvmStatus = MAX96712WriteParameters(deserializerCDI,
                                        CDI_WRITE_PARAM_CMD_MAX96712_PIPELINE_MAPPING,
                                        sizeof(paramsMAX96712.PipelineMapping),
                                        &paramsMAX96712);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96705: CDI_WRITE_PARAM_CMD_MAX96712_PIPELINE_MAPPING failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    // Override DataType
    pipeLineMap = &paramsMAX96712.PipelineMapping.linkPipelineMap[m_oLinkParams.ulinkIndex];
    pipeLineMap->isEmbDataType = sensorProperties.bEmbeddedDataType;
    pipeLineMap->dataType = dataType;
    pipeLineMap->isDTOverride = !sensorProperties.bEmbeddedDataType;
    paramsMAX96712.PipelineMapping.link = GetMAX96712Link(m_oLinkParams.ulinkIndex);
    nvmStatus = MAX96712WriteParameters(deserializerCDI,
                                        CDI_WRITE_PARAM_CMD_MAX96712_OVERRIDE_DATATYPE,
                                        sizeof(paramsMAX96712.PipelineMapping),
                                        &paramsMAX96712);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96705: CDI_WRITE_PARAM_CMD_MAX96712_OVERRIDE_DATATYPE failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMTransportLink_Max96712_96705::Init(DevBlkCDIDevice *const brdcstSerCDI, uint8_t const linkMask, bool const groupInitProg) {
    SIPLStatus status = NVSIPL_STATUS_OK;
    WriteParametersParamMAX96712 paramsMAX96712 = {};
    ReadParametersParamMAX96712 readParamsMAX96712 = {};
    DevBlkCDIDevice *deserializerCDI = m_oLinkParams.pDeserCDIDevice;
    NvMediaStatus nvmStatus = NVMEDIA_STATUS_OK;
    RevisionMAX96712 revision = CDI_MAX96712_REV_INVALID;
    FSyncModeMAX96712 internalSyncMode = CDI_MAX96712_FSYNC_INVALID;

    LOG_INFO("Initializing link %u \n", m_oLinkParams.ulinkIndex);

    LOG_INFO("Read revision\n");
    nvmStatus = MAX96712ReadParameters(deserializerCDI,
                                       CDI_READ_PARAM_CMD_MAX96712_REV_ID,
                                       sizeof(readParamsMAX96712.revision),
                                       &readParamsMAX96712.revision);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96705: CDI_READ_PARAM_CMD_MAX96712_REV_ID failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }
    revision = readParamsMAX96712.revision;

#if !NV_IS_SAFETY
    DumpLinkParams();
#endif

    if (m_oLinkParams.bPassive or m_oLinkParams.bEnableSimulator) {
        return NVSIPL_STATUS_OK;
    }

    // Setup config link
    status = SetupConfigLink(brdcstSerCDI, linkMask, groupInitProg);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96705: SetupConfigLink failed with SIPL error", (int32_t)status);
        return status;
    }

    /* Enable FRSYNC */
    if (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bEnableTriggerModeSync) {
        if ((revision == CDI_MAX96712_REV_1) or (revision == CDI_MAX96712_REV_2)) {
            internalSyncMode = CDI_MAX96712_FSYNC_MANUAL;
        } else {
            internalSyncMode = CDI_MAX96712_FSYNC_OSC_MANUAL;
        }

        paramsMAX96712.FSyncSettings.FSyncMode = (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bEnableInternalSync == true) ?
                                                 internalSyncMode : CDI_MAX96712_FSYNC_EXTERNAL;
        paramsMAX96712.FSyncSettings.pclk = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.pclk;
        paramsMAX96712.FSyncSettings.fps = (uint32_t) m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.fFrameRate;
        paramsMAX96712.FSyncSettings.link = GetMAX96712Link(m_oLinkParams.ulinkIndex);
        LOG_INFO("Set FSYNC mode\n");
        nvmStatus = MAX96712WriteParameters(deserializerCDI,
                                            CDI_WRITE_PARAM_CMD_MAX96712_SET_FSYNC,
                                            sizeof(paramsMAX96712.FSyncSettings),
                                            &paramsMAX96712);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96705: CDI_WRITE_PARAM_CMD_MAX96712_FSYNC_MODE failed with NvMedia error", (int32_t)nvmStatus);
            return ConvertNvMediaStatus(nvmStatus);
        }
    }

    LOG_INFO("Setup sensor/serializer address translations\n");
    status = SetupAddressTranslationsSer(brdcstSerCDI, groupInitProg);
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96705: SetupAddressTranslationsSer failed with SIPL error", (int32_t)status);
            return status;
        }

    // Setup video link
    status = SetupVideoLink();
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96705: SetupVideoLink failed with SIPL error", (int32_t)status);
        return status;
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMTransportLink_Max96712_96705::PostSensorInit(DevBlkCDIDevice const* const brdcstSerCDI, uint8_t const linkMask, bool const groupInitProg) const {
    LOG_INFO("Post sensor initializing link %u \n", m_oLinkParams.ulinkIndex);

    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMTransportLink_Max96712_96705::MiscInit() const {
    NvMediaStatus nvmStatus = NVMEDIA_STATUS_OK;
    SIPLStatus status = NVSIPL_STATUS_OK;
    ReadParametersParamMAX96712 readParamsMAX96712 = {};
    WriteParametersParamMAX96712 writeParamsMAX96712 = {};
    DevBlkCDIDevice *deserializerCDI = m_oLinkParams.pDeserCDIDevice;
    DevBlkCDIDevice *serCDI = m_oLinkParams.pSerCDIDevice;

    if (m_oLinkParams.bPassive || m_oLinkParams.bEnableSimulator) {
        return NVSIPL_STATUS_OK;
    }

    // Disable ERRB report
    nvmStatus = MAX96712SetDeviceConfig(deserializerCDI, CDI_CONFIG_MAX96712_DISABLE_ERRB);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: CDI_CONFIG_MAX96712_DISABLE_ERRB failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    // Enable PCLK In
    LOG_INFO("Enable PCLK In for Serializer\n");
    nvmStatus = MAX96705SetDeviceConfig(serCDI, CDI_CONFIG_MAX96705_PCLKIN);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96705: CDI_CONFIG_MAX96705_PCLKIN failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    // Set DBL
    LOG_INFO("Set DBL \n");
    writeParamsMAX96712.link = GetMAX96712Link(m_oLinkParams.ulinkIndex);
    nvmStatus = MAX96712WriteParameters(deserializerCDI,
                                        CDI_WRITE_PARAM_CMD_MAX96712_SET_DBL,
                                        sizeof(writeParamsMAX96712.link),
                                        &writeParamsMAX96712);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96705: CDI_CONFIG_MAX96712_SET_DBL failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    // Enable each serial link
    LOG_INFO("Enable serial link\n");
    nvmStatus = MAX96705SetDeviceConfig(serCDI, CDI_CONFIG_MAX96705_ENABLE_SERIAL_LINK);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96705: CDI_CONFIG_MAX96705_ENABLE_SERIAL_LINK failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    // I2C bus clock stretch is observed after enabling the serial link on MAX96705 for 33ms
    std::this_thread::sleep_for(std::chrono::milliseconds(40));

    /* Set the remote-i2c-master timeout to never in MAX96705 to prevent timeout in
     * remote-i2c-master while transferring i2c data from the actual i2c master (Bug 1802338, 200419005) */
    LOG_INFO("Set remote-i2c-master timeout to never\n");
    nvmStatus = MAX96705SetDeviceConfig(serCDI, CDI_CONFIG_MAX96705_SET_MAX_REMOTE_I2C_MASTER_TIMEOUT);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96705: CDI_CONFIG_MAX96705_SET_MAX_REMOTE_I2C_MASTER_TIMEOUT failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    LOG_INFO("Check config link lock \n");
    nvmStatus = MAX96712CheckLink(deserializerCDI,
                                  GetMAX96712Link(m_oLinkParams.ulinkIndex),
                                  CDI_MAX96712_LINK_LOCK_GMSL1_CONFIG,
                                  false);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96705: MAX96712CheckLink failed with NvMedia error", (int32_t)nvmStatus);
        nvmStatus = MAX96712OneShotReset(deserializerCDI, GetMAX96712Link(m_oLinkParams.ulinkIndex));
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96705: MAX96712OneShotReset failed with NvMedia error", (int32_t)nvmStatus);
            return ConvertNvMediaStatus(nvmStatus);
        }
    }

    /* Update I2C translation for EEPROM */
    LOG_INFO("Setup EEPROM address translations\n");
    status = SetupAddressTranslationsEEPROM();
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96705: SetupAddressTranslationsEEPROM failed with SIPL error", (int32_t)status);
        return status;
    }

    /* Clear the packet based control channel CRC error. GMSL1 only */
    readParamsMAX96712.ErrorStatus.link = GetMAX96712Link(m_oLinkParams.ulinkIndex);
    nvmStatus = MAX96712ReadParameters(deserializerCDI,
                                       CDI_READ_PARAM_CMD_MAX96712_CONTROL_CHANNEL_CRC_ERROR,
                                       sizeof(readParamsMAX96712.ErrorStatus),
                                       &readParamsMAX96712);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96705: CDI_READ_PARAM_CMD_MAX96712_CONTROL_CHANNEL_CRC_ERROR failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    /* Check & Clear if ERRB set */
    readParamsMAX96712.ErrorStatus.link = GetMAX96712Link(m_oLinkParams.ulinkIndex);
    nvmStatus = MAX96712ReadParameters(deserializerCDI,
                                       CDI_READ_PARAM_CMD_MAX96712_ERRB,
                                       sizeof(readParamsMAX96712.ErrorStatus),
                                       &readParamsMAX96712);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96705: CDI_READ_PARAM_CMD_MAX96712_ERRB failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    // Enable ERRB report
    nvmStatus = MAX96712SetDeviceConfig(deserializerCDI, CDI_CONFIG_MAX96712_ENABLE_ERRB);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: CDI_CONFIG_MAX96712_ENABLE_ERRB failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMTransportLink_Max96712_96705::Reset() const
{
    NvMediaStatus nvmStatus = NVMEDIA_STATUS_OK;
    LinkMAX96712 link = GetMAX96712Link(m_oLinkParams.ulinkIndex);
    DevBlkCDIDevice *deserializerCDI = m_oLinkParams.pDeserCDIDevice;
    WriteParametersParamMAX96712 paramsMAX96712 = {};

    LOG_INFO("Resetting link %u \n", m_oLinkParams.ulinkIndex);

    // Check the input CDI handles
    if (deserializerCDI == nullptr) {
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    if (m_oLinkParams.bPassive or m_oLinkParams.bEnableSimulator) {
        return NVSIPL_STATUS_OK;
    }

    // Disable packet based control channel
    LOG_INFO("Disable packet based control channel\n");
    paramsMAX96712.link = link;
    nvmStatus = MAX96712WriteParameters(deserializerCDI,
                                        CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_PACKET_BASED_CONTROL_CHANNEL,
                                        sizeof(paramsMAX96712.link),
                                        &paramsMAX96712);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96705: CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_PACKET_BASED_CONTROL_CHANNEL failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    // Disable DBL mode
     LOG_INFO("Unset DBL \n");
     paramsMAX96712.link = link;
     nvmStatus = MAX96712WriteParameters(deserializerCDI,
                                         CDI_WRITE_PARAM_CMD_MAX96712_UNSET_DBL,
                                         sizeof(paramsMAX96712.link),
                                         &paramsMAX96712);
     if (nvmStatus != NVMEDIA_STATUS_OK) {
         SIPL_LOG_ERR_STR_INT("MAX96705: CDI_CONFIG_MAX96712_UNSET_DBL failed with NvMedia error", (int32_t)nvmStatus);
         return ConvertNvMediaStatus(nvmStatus);
     }

    return NVSIPL_STATUS_OK;
}

} // end of namespace nvsipl
