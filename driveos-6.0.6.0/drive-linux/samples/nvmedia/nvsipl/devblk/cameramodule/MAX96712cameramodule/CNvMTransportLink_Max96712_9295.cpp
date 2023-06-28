/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "CNvMTransportLink_Max96712_9295.hpp"
#include "sipl_error.h"

#include <string>
#include <chrono>
#include <thread>

// Include necessary CDI driver headers
extern "C" {
#include "cdi_max96712.h"
#include "cdi_max9295.h"
}

namespace nvsipl {

static GPIOTypeMAX9295
getMAX9295GPIO(
    uint8_t gpioInd)
{
    GPIOTypeMAX9295 gpio = CDI_MAX9295_GPIO_TYPE_INVALID;

    switch (gpioInd) {
        case 0:
            gpio = CDI_MAX9295_GPIO_TYPE_MFP0;
            break;
        case 1:
            gpio = CDI_MAX9295_GPIO_TYPE_MFP1;
            break;
        case 2:
            gpio = CDI_MAX9295_GPIO_TYPE_MFP2;
            break;
        case 3:
            gpio = CDI_MAX9295_GPIO_TYPE_MFP3;
            break;
        case 4:
            gpio = CDI_MAX9295_GPIO_TYPE_MFP4;
            break;
        case 5:
            gpio = CDI_MAX9295_GPIO_TYPE_MFP5;
            break;
        case 6:
            gpio = CDI_MAX9295_GPIO_TYPE_MFP6;
            break;
        case 7:
            gpio = CDI_MAX9295_GPIO_TYPE_MFP7;
            break;
        case 8:
            gpio = CDI_MAX9295_GPIO_TYPE_MFP8;
            break;
        default:
            SIPL_LOG_ERR_STR_UINT("MAX96712: Invalid Max9295 GPIO pin", (uint32_t)gpioInd);
    }

    return gpio;
}

#if !NV_IS_SAFETY
void CNvMTransportLink_Max96712_9295::DumpLinkParams()
{
    LOG_INFO("Link parameters\n");
    LOG_INFO("Link Index: %u \n", m_oLinkParams.ulinkIndex);
    LOG_INFO("Broadcast serializer addr: 0x%x \n", m_oLinkParams.uBrdcstSerAddr);
    LOG_INFO("Serializer addr: 0x%x \n", m_oLinkParams.uSerAddr);

    const CNvMCameraModuleCommon::ConnectionProperty::SensorConnectionProperty &sensor = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty;
    LOG_INFO("\nBroadcast sensor addr: 0x%x \n", sensor.uBrdcstSensorAddrs);
    LOG_INFO("Sensor addr: 0x%x \n", sensor.uSensorAddrs);
    LOG_INFO("VCID: %u \n", sensor.uVCID);
    LOG_INFO("Embedded data type: %s \n", sensor.bEmbeddedDataType ? "true" : "false");
    LOG_INFO("Trigger mode sync: %s \n", sensor.bEnableTriggerModeSync ? "true" : "false");
    LOG_INFO("Frame rate: %.2f fps \n", sensor.fFrameRate);

    std::uint8_t broadEEPROMAddr = m_oLinkParams.moduleConnectionProperty.brdcstEepromAddr;
    LOG_INFO("\nBroadcast EEPROM addr: 0x%x \n", broadEEPROMAddr);

    std::uint8_t EEPROMAddr = m_oLinkParams.moduleConnectionProperty.eepromAddr;
    LOG_INFO("EEPROM addr: 0x%x \n", EEPROMAddr);

    LOG_INFO("Simulator mode: %u \n", m_oLinkParams.bEnableSimulator);
    LOG_INFO("Passive mode: %u \n", m_oLinkParams.bPassive);
}
#endif

SIPLStatus CNvMTransportLink_Max96712_9295::SetupAddressTranslations(DevBlkCDIDevice* brdcstSerCDI)
{
    NvMediaStatus nvmStatus = NVMEDIA_STATUS_OK;
    ReadWriteParamsMAX9295 paramsMAX9295 = {};

    // Check serializer is present
    LOG_INFO("Check broadcast serializer is present\n");
    nvmStatus = MAX9295CheckPresence(brdcstSerCDI);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX9295: MAX9295CheckPresence(brdcstSerCDI) failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    // Set address translation for the sensor
    paramsMAX9295.Translator.source =
        m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.uSensorAddrs;
    paramsMAX9295.Translator.destination =
        m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.uBrdcstSensorAddrs;
    LOG_INFO("Translate image sensor device addr %x to %x\n",
              paramsMAX9295.Translator.source, paramsMAX9295.Translator.destination);
    nvmStatus = MAX9295WriteParameters(brdcstSerCDI,
                                       CDI_WRITE_PARAM_CMD_MAX9295_SET_TRANSLATOR_A,
                                       sizeof(paramsMAX9295.Translator),
                                       &paramsMAX9295);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX9295: CDI_WRITE_PARAM_CMD_MAX96705_SET_TRANSLATOR_A failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    // Set address translation for the eeprom
    if (m_oLinkParams.moduleConnectionProperty.eepromAddr != UINT8_MAX) {
        paramsMAX9295.Translator.source = m_oLinkParams.moduleConnectionProperty.eepromAddr;
        paramsMAX9295.Translator.destination = m_oLinkParams.moduleConnectionProperty.brdcstEepromAddr;
        LOG_INFO("Translate eeprom device addr %x to %x\n",
                            paramsMAX9295.Translator.source, paramsMAX9295.Translator.destination);
        nvmStatus = MAX9295WriteParameters(brdcstSerCDI,
                                           CDI_WRITE_PARAM_CMD_MAX9295_SET_TRANSLATOR_B,
                                           sizeof(paramsMAX9295.Translator),
                                           &paramsMAX9295);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX9295: CDI_WRITE_PARAM_CMD_MAX96705_SET_TRANSLATOR_B failed with NvMedia error", (int32_t)nvmStatus);
            return ConvertNvMediaStatus(nvmStatus);
        }
    }

    // Set unique address with broadcast address
    LOG_INFO("Set unique address\n");
    paramsMAX9295.DeviceAddress.address = m_oLinkParams.uSerAddr;
    nvmStatus = MAX9295WriteParameters(brdcstSerCDI,
                                       CDI_WRITE_PARAM_CMD_MAX9295_SET_DEVICE_ADDRESS,
                                       sizeof(paramsMAX9295.DeviceAddress.address),
                                       &paramsMAX9295);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX9295: CDI_WRITE_PARAM_CMD_MAX96705_SET_DEVICE_ADDRESS failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    return NVSIPL_STATUS_OK;
}

// This will do the init for 1 maxim deser and up to MAX_LINKS_PER_DESER maxim serializers.
SIPLStatus CNvMTransportLink_Max96712_9295::Init(DevBlkCDIDevice* brdcstSerCDI, uint8_t linkMask, bool groupInitProg)
{
    NvMediaStatus nvmStatus = NVMEDIA_STATUS_OK;
    SIPLStatus status = NVSIPL_STATUS_OK;
    WriteParametersParamMAX96712 paramsMAX96712 = {};
    LinkMAX96712 link = GetMAX96712Link(m_oLinkParams.ulinkIndex);
    ReadWriteParamsMAX9295 paramsMAX9295 = {};
    DataTypeMAX96712 dataTypeMAX96712 = CDI_MAX96712_DATA_TYPE_INVALID;
    DataTypeMAX9295 dataTypeMAX9295 = CDI_MAX9295_DATA_TYPE_INVALID;
    LinkPipelineMapMAX96712 *pipeLineMap = NULL;
    DevBlkCDIDevice *deserializerCDI = m_oLinkParams.pDeserCDIDevice;
    DevBlkCDIDevice *serCDI = m_oLinkParams.pSerCDIDevice;
    uint32_t gpioMapSize;

    LOG_INFO("Initializing link %u\n", m_oLinkParams.ulinkIndex);

    switch (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.inputFormat.inputFormatType)
    {
        case NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW10:
            dataTypeMAX96712 = CDI_MAX96712_DATA_TYPE_RAW10;
            dataTypeMAX9295 = CDI_MAX9295_DATA_TYPE_RAW10;
            break;

        case NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12:
        case NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12RJ:
            dataTypeMAX96712 = CDI_MAX96712_DATA_TYPE_RAW12;
            dataTypeMAX9295 = CDI_MAX9295_DATA_TYPE_RAW12;
            break;

        case NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW16:
            dataTypeMAX96712 = CDI_MAX96712_DATA_TYPE_RAW16;
            dataTypeMAX9295 = CDI_MAX9295_DATA_TYPE_RAW16;
            break;

        default:
            SIPL_LOG_ERR_STR_INT(
                "MAX96712: Input format not supported",
                (int32_t)m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.inputFormat.inputFormatType
            );
            return NVSIPL_STATUS_NOT_SUPPORTED;
    }

#if !NV_IS_SAFETY
    DumpLinkParams();
#endif

    if (m_oLinkParams.bPassive or m_oLinkParams.bEnableSimulator) {
        return NVSIPL_STATUS_OK;
    }

    LOG_INFO("Check config link lock \n");
    nvmStatus = MAX96712CheckLink(deserializerCDI,
                                  GetMAX96712Link(m_oLinkParams.ulinkIndex),
                                  CDI_MAX96712_LINK_LOCK_GMSL2,
                                  true);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: DevBlkCDICheckLink(CDI_MAX96712_LINK_LOCK_GMSL2) failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    // Check presence of serializer and setup address translations
    LOG_INFO("Setup address translations\n");
    status = SetupAddressTranslations(brdcstSerCDI);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: SetupAddressTranslations failed with SIPL error", (int32_t)status);
        return status;
    }

    CNvMCameraModuleCommon::ConnectionProperty::SensorConnectionProperty::EepromWriteProtect eepromWriteProtect =
            m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.eepromWriteProtect;
    if (eepromWriteProtect.isNeeded) {
        LOG_INFO("Enable EEPROM write protect\n");
        paramsMAX9295.GPIOOutp.gpioInd = getMAX9295GPIO(eepromWriteProtect.pinNum);
        paramsMAX9295.GPIOOutp.level = eepromWriteProtect.writeProtectLevel;
        nvmStatus = MAX9295WriteParameters(serCDI,
                                           CDI_WRITE_PARAM_CMD_MAX9295_SET_GPIO_OUTPUT,
                                           sizeof(paramsMAX9295.GPIOOutp),
                                           &paramsMAX9295);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
           SIPL_LOG_ERR_STR_INT("MAX96712: CDI_WRITE_PARAM_CMD_MAX9295_SET_GPIO_OUTPUT failed with Nvmedia error", (int32_t)nvmStatus);
        }
    }

    CNvMCameraModuleCommon::ConnectionProperty::SensorConnectionProperty::RefClock refClock =
            m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.refClock;
    if (refClock.isNeeded) {
        LOG_INFO("Enable ref clock\n");
        paramsMAX9295.RefClkGPIO.gpioInd = getMAX9295GPIO(refClock.pinNum); /* set source GPIO */
        paramsMAX9295.RefClkGPIO.enableRClk = true;
        nvmStatus = MAX9295WriteParameters(serCDI,
                                           CDI_WRITE_PARAM_CMD_MAX9295_ENABLE_REF_CLOCK,
                                           sizeof(paramsMAX9295.RefClkGPIO),
                                           &paramsMAX9295);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96712: CDI_WRITE_PARAM_CMD_MAX9295_ENABLE_REF_CLOCK failed with NvMedia error", (int32_t)nvmStatus);
            return ConvertNvMediaStatus(nvmStatus);
        }
    }

    CNvMCameraModuleCommon::ConnectionProperty::SensorConnectionProperty::SensorReset sensorReset =
            m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.sensorReset;
    if (sensorReset.isNeeded) {
        std::this_thread::sleep_for(std::chrono::microseconds(sensorReset.assertResetDuration));

        LOG_INFO("Release sensor reset\n");
        paramsMAX9295.GPIOOutp.gpioInd = getMAX9295GPIO(sensorReset.pinNum);
        paramsMAX9295.GPIOOutp.level = sensorReset.releaseResetLevel;
        nvmStatus = MAX9295WriteParameters(serCDI,
                                           CDI_WRITE_PARAM_CMD_MAX9295_SET_GPIO_OUTPUT,
                                           sizeof(paramsMAX9295.GPIOOutp),
                                           &paramsMAX9295);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96712: CDI_WRITE_PARAM_CMD_MAX9295_SET_GPIO_OUTPUT failed with NvMedia error", (int32_t)nvmStatus);
            return ConvertNvMediaStatus(nvmStatus);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(sensorReset.deassertResetWait));
    }

    if (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bEnableTriggerModeSync) {
        LOG_INFO("CDI_WRITE_PARAM_CMD_MAX9295_SET_FSYNC_GPIO\n");
        paramsMAX9295.FSyncGPIO.gpioInd = getMAX9295GPIO(m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.frameSync.pinNum);
        if (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bEnableInternalSync) {
            paramsMAX9295.FSyncGPIO.rxID = CDI_MAX96712_GPIO_20;
        } else {
            paramsMAX9295.FSyncGPIO.rxID = CDI_MAX96712_GPIO_2;
        }
        nvmStatus = MAX9295WriteParameters(serCDI,
                                           CDI_WRITE_PARAM_CMD_MAX9295_SET_FSYNC_GPIO,
                                           sizeof(paramsMAX9295.FSyncGPIO),
                                           &paramsMAX9295);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96712: CDI_WRITE_PARAM_CMD_MAX9295_SET_FSYNC_GPIO failed with NvMedia error", (int32_t)nvmStatus);
            return ConvertNvMediaStatus(nvmStatus);
        }
    }

    gpioMapSize = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.vGpioMap.size();
    for (uint8_t i = 0u; i < gpioMapSize; i++) {
        LOG_INFO("CDI_WRITE_PARAM_CMD_MAX9295_GPIO_FORWARD\n");
        paramsMAX9295.GPIOForward.srcGpio = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.vGpioMap[i].sourceGpio;
        paramsMAX9295.GPIOForward.dstGpio = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.vGpioMap[i].destGpio;
        nvmStatus = MAX9295WriteParameters(serCDI,
                                           CDI_WRITE_PARAM_CMD_MAX9295_GPIO_FORWARD,
                                           sizeof(paramsMAX9295.GPIOForward),
                                           &paramsMAX9295);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96712: CDI_WRITE_PARAM_CMD_MAX9295_GPIO_FORWARD failed with NvMedia error", (int32_t)nvmStatus);
            return ConvertNvMediaStatus(nvmStatus);
        }

        LOG_INFO("CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_GPIO_RX\n");
        paramsMAX96712.gpioIndex = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.vGpioMap[i].destGpio;
        nvmStatus = MAX96712WriteParameters(deserializerCDI,
                                            CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_GPIO_RX,
                                            sizeof(paramsMAX96712.gpioIndex),
                                            &paramsMAX96712);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96712: CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_GPIO_RX failed with NvMedia error", (int32_t)nvmStatus);
            return ConvertNvMediaStatus(nvmStatus);
        }
    }

    paramsMAX9295.ConfigPhy.mapping.enableMapping = false;
    paramsMAX9295.ConfigPhy.numDataLanes = 4;
    if (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.phyLanes.isLaneSwapNeeded) {
        paramsMAX9295.ConfigPhy.mapping.phy1_d0 = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.phyLanes.lane0;
        paramsMAX9295.ConfigPhy.mapping.phy1_d1 = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.phyLanes.lane1;
        paramsMAX9295.ConfigPhy.mapping.phy2_d0 = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.phyLanes.lane2;
        paramsMAX9295.ConfigPhy.mapping.phy2_d1 = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.phyLanes.lane3;
        paramsMAX9295.ConfigPhy.mapping.enableMapping = true;
    } else if (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.phyLanes.isTwoLane) {
        paramsMAX9295.ConfigPhy.numDataLanes = 2;
    }
    paramsMAX9295.ConfigPhy.polarity.setPolarity = false;
    if (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.phyLanes.isLanePolarityConfigureNeeded) {
        paramsMAX9295.ConfigPhy.polarity.phy1_d0 = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.phyLanes.lane0pol;
        paramsMAX9295.ConfigPhy.polarity.phy1_d1 = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.phyLanes.lane1pol;
        paramsMAX9295.ConfigPhy.polarity.phy1_clk= m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.phyLanes.clk1pol;
        paramsMAX9295.ConfigPhy.polarity.phy2_d0 = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.phyLanes.lane2pol;
        paramsMAX9295.ConfigPhy.polarity.phy2_d1 = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.phyLanes.lane3pol;
        paramsMAX9295.ConfigPhy.polarity.phy2_clk= m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.phyLanes.clk2pol;
        paramsMAX9295.ConfigPhy.polarity.setPolarity = true;
    }
    LOG_INFO("CDI_WRITE_PARAM_CMD_MAX9295_CONFIG_PHY\n");
    nvmStatus = MAX9295WriteParameters(serCDI,
                                       CDI_WRITE_PARAM_CMD_MAX9295_CONFIG_PHY,
                                       sizeof(paramsMAX9295.ConfigPhy),
                                       &paramsMAX9295);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: CDI_WRITE_PARAM_CMD_MAX9295_CONFIG_PHY failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    paramsMAX96712.DoublePixelMode.link = link;
    paramsMAX96712.DoublePixelMode.dataType = dataTypeMAX96712;
    paramsMAX96712.DoublePixelMode.embDataType = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bEmbeddedDataType;
    nvmStatus = MAX96712WriteParameters(deserializerCDI,
                                        CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_DOUBLE_PIXEL_MODE,
                                        sizeof(paramsMAX96712.DoublePixelMode),
                                        &paramsMAX96712);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_DOUBLE_PIXEL_MODE failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    LOG_INFO("CDI_WRITE_PARAM_CMD_MAX9295_CONFIG_VIDEO_PIPELINES\n");
    paramsMAX9295.ConfigVideoPipeline.dataType = dataTypeMAX9295;
    paramsMAX9295.ConfigVideoPipeline.embDataType = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bEmbeddedDataType;
    nvmStatus = MAX9295WriteParameters(serCDI,
                                       CDI_WRITE_PARAM_CMD_MAX9295_CONFIG_VIDEO_PIPELINES,
                                       sizeof(paramsMAX9295.ConfigVideoPipeline),
                                       &paramsMAX9295);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: CDI_WRITE_PARAM_CMD_MAX9295_CONFIG_VIDEO_PIPELINES failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    pipeLineMap = &paramsMAX96712.PipelineMapping.linkPipelineMap[m_oLinkParams.ulinkIndex];
    pipeLineMap->isEmbDataType = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bEmbeddedDataType;
    pipeLineMap->vcID = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.uVCID;
    pipeLineMap->dataType = dataTypeMAX96712;
    pipeLineMap->isDTOverride = !m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bEmbeddedDataType;
    paramsMAX96712.PipelineMapping.link = link;
    paramsMAX96712.PipelineMapping.isSinglePipeline = false; /* Assign two pipelines to process two data types */
    nvmStatus = MAX96712WriteParameters(deserializerCDI,
                                        CDI_WRITE_PARAM_CMD_MAX96712_PIPELINE_MAPPING,
                                        sizeof(paramsMAX96712.PipelineMapping),
                                        &paramsMAX96712);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: CDI_WRITE_PARAM_CMD_MAX96712_PIPELINE_MAPPING failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    paramsMAX96712 = {};
    LOG_INFO("Initializing link %u\n", m_oLinkParams.ulinkIndex);

    /* Enable FRSYNC */
    if (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bEnableTriggerModeSync &&
        (!m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bPostSensorInitFsync)) {
        LOG_INFO("Set Fsync\n");
        paramsMAX96712.FSyncSettings.FSyncMode = (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bEnableInternalSync == true) ?
                                                  CDI_MAX96712_FSYNC_OSC_MANUAL : CDI_MAX96712_FSYNC_EXTERNAL;
        paramsMAX96712.FSyncSettings.fps = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.fFrameRate;
        paramsMAX96712.FSyncSettings.link = link;
        LOG_INFO("Set FSYNC mode\n");
        nvmStatus = MAX96712WriteParameters(deserializerCDI,
                                            CDI_WRITE_PARAM_CMD_MAX96712_SET_FSYNC,
                                            sizeof(paramsMAX96712.FSyncSettings),
                                            &paramsMAX96712);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96712: CDI_WRITE_PARAM_CMD_MAX96712_FSYNC_MODE failed with NvMedia error", (int32_t)nvmStatus);
            return ConvertNvMediaStatus(nvmStatus);
        }
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMTransportLink_Max96712_9295::PostSensorInit(uint8_t const linkMask, bool const groupInitProg) const {
    WriteParametersParamMAX96712 paramsMAX96712 = {};
    LinkMAX96712 link = GetMAX96712Link(m_oLinkParams.ulinkIndex);
    DevBlkCDIDevice *deserializerCDI = m_oLinkParams.pDeserCDIDevice;
    NvMediaStatus nvmStatus = NVMEDIA_STATUS_OK;

    if (m_oLinkParams.bPassive or m_oLinkParams.bEnableSimulator) {
        return NVSIPL_STATUS_OK;
    }

    /* Enable FRSYNC */
    if (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bEnableTriggerModeSync &&
        m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bPostSensorInitFsync) {
        LOG_INFO("Set Fsync\n");
        paramsMAX96712.FSyncSettings.FSyncMode = (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bEnableInternalSync) ?
                                                  CDI_MAX96712_FSYNC_OSC_MANUAL : CDI_MAX96712_FSYNC_EXTERNAL;
        paramsMAX96712.FSyncSettings.fps = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.fFrameRate;
        paramsMAX96712.FSyncSettings.link = link;
        LOG_INFO("Set FSYNC mode\n");
        nvmStatus = MAX96712WriteParameters(deserializerCDI,
                                            CDI_WRITE_PARAM_CMD_MAX96712_SET_FSYNC,
                                            sizeof(paramsMAX96712.FSyncSettings),
                                            &paramsMAX96712);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96712: CDI_WRITE_PARAM_CMD_MAX96712_FSYNC_MODE failed with NvMedia error", (int32_t)nvmStatus);
            return ConvertNvMediaStatus(nvmStatus);
        }
    }

    return NVSIPL_STATUS_OK;
}

} // end of namespace nvsipl
