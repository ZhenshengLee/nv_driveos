/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */
#include "CNvMTransportLink_Max96712_96717.hpp"
#include "sipl_error.h"

#include <string>
// Include necessary ISC driver headers
extern "C" {
#include "cdi_max96712.h"
#include "cdi_max96717f.h"
}
namespace nvsipl {
static GPIOTypeMAX96717F
getMAX96717FGPIO(
    uint8_t gpioInd)
{
    GPIOTypeMAX96717F gpio = CDI_MAX96717F_GPIO_TYPE_INVALID;
    switch (gpioInd) {
        case 0:
            gpio = CDI_MAX96717F_GPIO_TYPE_MFP0;
            break;
        case 1:
            gpio = CDI_MAX96717F_GPIO_TYPE_MFP1;
            break;
        case 2:
            gpio = CDI_MAX96717F_GPIO_TYPE_MFP2;
            break;
        case 3:
            gpio = CDI_MAX96717F_GPIO_TYPE_MFP3;
            break;
        case 4:
            gpio = CDI_MAX96717F_GPIO_TYPE_MFP4;
            break;
        case 5:
            gpio = CDI_MAX96717F_GPIO_TYPE_MFP5;
            break;
        case 6:
            gpio = CDI_MAX96717F_GPIO_TYPE_MFP6;
            break;
        case 7:
            gpio = CDI_MAX96717F_GPIO_TYPE_MFP7;
            break;
        case 8:
            gpio = CDI_MAX96717F_GPIO_TYPE_MFP8;
            break;
        default:
            SIPL_LOG_ERR_STR_INT("Invalid Max96717F GPIO pin", gpioInd);
    }
    return gpio;
}
void CNvMTransportLink_Max96712_96717::DumpLinkParams()
{
    LOG_INFO("Link parameters\n");
    LOG_INFO("Link Index: %u \n", m_oLinkParams.ulinkIndex);
    LOG_INFO("Broadcast serializer addr: 0x%x \n", m_oLinkParams.uBrdcstSerAddr);
    LOG_INFO("Serializer addr: 0x%x \n", m_oLinkParams.uSerAddr);

    LOG_INFO("\nBroadcast sensor addr: 0x%x \n",
        m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.uBrdcstSensorAddrs);
    LOG_INFO("Sensor addr: 0x%x \n",
        m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.uSensorAddrs);
    LOG_INFO("VCID: %u \n",
        m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.uVCID);
    LOG_INFO("Embedded data type: %s \n",
        m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bEmbeddedDataType ? "true" : "false");
    LOG_INFO("Trigger mode sync: %s \n",
        m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bEnableTriggerModeSync ? "true" : "false");
    LOG_INFO("Frame rate: %.2f fps \n",
        m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.fFrameRate);

    LOG_INFO("\nBroadcast EEPROM addr: 0x%x \n", m_oLinkParams.moduleConnectionProperty.brdcstEepromAddr);
    LOG_INFO("EEPROM addr: 0x%x \n", m_oLinkParams.moduleConnectionProperty.eepromAddr);
    LOG_INFO("Simulator mode: %u \n", m_oLinkParams.bEnableSimulator);
    LOG_INFO("Passive mode: %u \n", m_oLinkParams.bPassive);
}
SIPLStatus CNvMTransportLink_Max96712_96717::SetupAddressTranslations(DevBlkCDIDevice* brdcstserCDI)
{
    NvMediaStatus nvmStatus = NVMEDIA_STATUS_OK;
    ReadWriteParamsMAX96717F paramsMAX96717F = {};
    // Check serializer is present
    LOG_INFO("Check broadcast serializer is present\n");
    nvmStatus = MAX96717FCheckPresence(brdcstserCDI);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_HEX_UINT("MAX96717CheckPresence(brdcstserCDI) failed with nvmediaErr", nvmStatus);
        goto done;
    }
    // Set address translation for the sensor
    if (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.uSensorAddrs != UINT8_MAX) {
        paramsMAX96717F.Translator.source =
            m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.uSensorAddrs;
        paramsMAX96717F.Translator.destination =
            m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.uBrdcstSensorAddrs;
        LOG_INFO("Translate image sensor device addr %x to %x\n",
                  paramsMAX96717F.Translator.source, paramsMAX96717F.Translator.destination);
        nvmStatus = MAX96717FWriteParameters(brdcstserCDI,
                                            CDI_WRITE_PARAM_CMD_MAX96717F_SET_TRANSLATOR_A,
                                            sizeof(paramsMAX96717F.Translator),
                                            &paramsMAX96717F);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_HEX_UINT("CDI_WRITE_PARAM_CMD_MAX96717F_SET_TRANSLATOR_A failed with nvmediaErr", nvmStatus);
            goto done;
        }
    }

    // Set address translation for the eeprom
    if (m_oLinkParams.moduleConnectionProperty.eepromAddr != UINT8_MAX) {
        paramsMAX96717F.Translator.source = m_oLinkParams.moduleConnectionProperty.eepromAddr;
        paramsMAX96717F.Translator.destination = m_oLinkParams.moduleConnectionProperty.brdcstEepromAddr;
        LOG_INFO("Translate eeprom device addr %x to %x\n",
                    paramsMAX96717F.Translator.source, paramsMAX96717F.Translator.destination);
        nvmStatus = MAX96717FWriteParameters(brdcstserCDI,
                                             CDI_WRITE_PARAM_CMD_MAX96717F_SET_TRANSLATOR_B,
                                             sizeof(paramsMAX96717F.Translator),
                                             &paramsMAX96717F);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_HEX_UINT("CDI_WRITE_PARAM_CMD_MAX96717F_SET_TRANSLATOR_B failed with nvmediaErr", nvmStatus);
            goto done;
        }

        // Set unique address with broadcast address
        LOG_INFO("Set unique address\n");
        paramsMAX96717F.DeviceAddress.address = m_oLinkParams.uSerAddr;
        nvmStatus = MAX96717FWriteParameters(brdcstserCDI,
                                             CDI_WRITE_PARAM_CMD_MAX96717F_SET_DEVICE_ADDRESS,
                                             sizeof(paramsMAX96717F.DeviceAddress.address),
                                             &paramsMAX96717F);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_HEX_UINT("CDI_WRITE_PARAM_CMD_MAX96717F_SET_DEVICE_ADDRESS failed with nvmediaErr", nvmStatus);
        }
    } else {
        paramsMAX96717F.Translator.source = m_oLinkParams.uSerAddr;
        paramsMAX96717F.Translator.destination = m_oLinkParams.uBrdcstSerAddr;
        LOG_INFO("Translate serializer device addr %x to %x\n",
                    paramsMAX96717F.Translator.source, paramsMAX96717F.Translator.destination);
        nvmStatus = MAX96717FWriteParameters(brdcstserCDI,
                                             CDI_WRITE_PARAM_CMD_MAX96717F_SET_TRANSLATOR_B,
                                             sizeof(paramsMAX96717F.Translator),
                                             &paramsMAX96717F);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_HEX_UINT("CDI_WRITE_PARAM_CMD_MAX96717F_SET_TRANSLATOR_B failed with nvmediaErr", nvmStatus);
            goto done;
        }
    }

done:
    return ConvertNvMediaStatus(nvmStatus);
}

// This will do the init for 1 maxim deser and up to MAX_LINKS_PER_DESER maxim serializers.
SIPLStatus CNvMTransportLink_Max96712_96717::Init(DevBlkCDIDevice* brdcstserCDI, uint8_t linkMask, bool groupInitProg)
{
    NvMediaStatus nvmStatus = NVMEDIA_STATUS_OK;
    SIPLStatus status = NVSIPL_STATUS_OK;
    WriteParametersParamMAX96712 paramsMAX96712 = {};
    ReadParametersParamMAX96712 readParamsMAX96712 = {};
    LinkMAX96712 link = GetMAX96712Link(m_oLinkParams.ulinkIndex);
    ReadWriteParamsMAX96717F paramsMAX96717F = {};
    DataTypeMAX96712 dataTypeMAX96712 = CDI_MAX96712_DATA_TYPE_INVALID;
    DataTypeMAX96717F dataTypeMAX96717F = CDI_MAX96717F_DATA_TYPE_INVALID;
    LinkPipelineMapMAX96712 *pipeLineMap = NULL;
    DevBlkCDIDevice *deserializerCDI = m_oLinkParams.pDeserCDIDevice;
    DevBlkCDIDevice *serCDI = m_oLinkParams.pSerCDIDevice;
    RevisionMAX96712 revision = CDI_MAX96712_REV_INVALID;
    FSyncModeMAX96712 internalSyncMode = CDI_MAX96712_FSYNC_INVALID;
    uint32_t gpioMapSize;

    LOG_INFO("Initializing link %u\n", m_oLinkParams.ulinkIndex);

    LOG_INFO("Read revision\n");
    nvmStatus = MAX96712ReadParameters(deserializerCDI,
                                       CDI_READ_PARAM_CMD_MAX96712_REV_ID,
                                       sizeof(readParamsMAX96712.revision),
                                       &readParamsMAX96712.revision);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_HEX_UINT("CDI_READ_PARAM_CMD_MAX96712_REV_ID failed with nvmediaErr", nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }
    revision = readParamsMAX96712.revision;

    LOG_INFO("MAX96712 Revision %d", revision);

#if 0	// TODO: Is the check below not needed?
    // Check if 1 sensor per link
    if (m_oLinkParams.moduleConnectionProperty.vSensorConnectionProperty.size() != 1 ) {
        LOG_INFO("Number of sensors is not 1 for link %u\n", m_oLinkParams.ulinkIndex);
        return NVSIPL_STATUS_NOT_SUPPORTED;
    }
#endif

    switch (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.inputFormat.inputFormatType)
    {
        case NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW10:
            dataTypeMAX96712 = CDI_MAX96712_DATA_TYPE_RAW10;
            dataTypeMAX96717F = CDI_MAX96717F_DATA_TYPE_RAW10;
            break;

        case NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12:
        case NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12RJ:
            dataTypeMAX96712 = CDI_MAX96712_DATA_TYPE_RAW12;
            dataTypeMAX96717F = CDI_MAX96717F_DATA_TYPE_RAW12;
            break;

        default:
            SIPL_LOG_ERR_STR("Input format not supported");
            return NVSIPL_STATUS_NOT_SUPPORTED;
    }

    DumpLinkParams();

    if (m_oLinkParams.bPassive or m_oLinkParams.bEnableSimulator) {
        return NVSIPL_STATUS_OK;
    }

    LOG_INFO("Check config link lock \n");
    nvmStatus = MAX96712CheckLink(deserializerCDI,
                                  GetMAX96712Link(m_oLinkParams.ulinkIndex),
                                  CDI_MAX96712_LINK_LOCK_GMSL2,
                                  true);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_HEX_UINT("CDICheckLink(CDI_MAX96712_GMSL2_LINK_LOCK failed with nvmediaErr", nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    // Check presence of serializer and setup address translations
    LOG_INFO("Setup address translations\n");
    status = SetupAddressTranslations(brdcstserCDI);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_HEX_UINT("SetupAddressTranslations failed with SIPLStatus error", status);
        return status;
    }

    nvmStatus = MAX96717FWriteParameters(serCDI,
                                         CDI_WRITE_PARAM_CMD_MAX96717F_ENABLE_PCLK_PIO_SLEW,
                                         sizeof(paramsMAX96717F.ConfigPhy),
                                         &paramsMAX96717F);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_HEX_UINT("CDI_WRITE_PARAM_CMD_MAX96717F_ENABLE_PIO_SLEW failed with nvmediaErr", nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    uint8_t sensorClock =
            m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.refClock.sensorClock;
    if (sensorClock != 0) {
        LOG_INFO("Generate sensor clock\n");
        paramsMAX96717F.ClockRate.freq = sensorClock;
        nvmStatus = MAX96717FWriteParameters(serCDI,
                                             CDI_WRITE_PARAM_CMD_MAX96717F_GENERATE_CLOCK,
                                             sizeof(paramsMAX96717F.ClockRate),
                                             &paramsMAX96717F);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_HEX_UINT("CDI_WRITE_PARAM_CMD_MAX96717F_GNERATE_CLOCK failed with nvmediaErr", nvmStatus);
            return ConvertNvMediaStatus(nvmStatus);
        }
    }

    CNvMCameraModuleCommon::ConnectionProperty::SensorConnectionProperty::RefClock refClock =
            m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.refClock;
    if (refClock.isNeeded) {
        LOG_INFO("Enable ref clock\n");
        paramsMAX96717F.RefClkGPIO.gpioInd = getMAX96717FGPIO(refClock.pinNum); /* set source GPIO */
        paramsMAX96717F.RefClkGPIO.enableRClk = true;
        nvmStatus = MAX96717FWriteParameters(serCDI,
                                             CDI_WRITE_PARAM_CMD_MAX96717F_ENABLE_REF_CLOCK,
                                             sizeof(paramsMAX96717F.RefClkGPIO),
                                             &paramsMAX96717F);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_HEX_UINT("CDI_WRITE_PARAM_CMD_MAX96717F_ENABLE_REF_CLOCK failed with nvmediaErr", nvmStatus);
            return ConvertNvMediaStatus(nvmStatus);
        }
    }

    gpioMapSize = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.vGpioMap.size();
    for (uint8_t i = 0u; i < gpioMapSize; i++) {
        LOG_INFO("CDI_WRITE_PARAM_CMD_MAX96717F_GPIO_FORWARD\n");
        paramsMAX96717F.GPIOForward.srcGpio = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.vGpioMap[i].sourceGpio;
        paramsMAX96717F.GPIOForward.dstGpio = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.vGpioMap[i].destGpio;
        nvmStatus = MAX96717FWriteParameters(serCDI,
                                            CDI_WRITE_PARAM_CMD_MAX96717F_GPIO_FORWARD,
                                            sizeof(paramsMAX96717F.GPIOForward),
                                            &paramsMAX96717F);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_UINT("CDI_WRITE_PARAM_CMD_MAX96717F_GPIO_FORWARD failed with NvMedia error:", nvmStatus);
            return ConvertNvMediaStatus(nvmStatus);
        }

        LOG_INFO("CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_GPIO_RX\n");
        paramsMAX96712.gpioIndex = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.vGpioMap[i].destGpio;
        nvmStatus = MAX96712WriteParameters(deserializerCDI,
                                            CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_GPIO_RX,
                                            sizeof(paramsMAX96712.gpioIndex),
                                            &paramsMAX96712);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_UINT("CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_GPIO_RX failed with NvMedia error:", nvmStatus);
            return ConvertNvMediaStatus(nvmStatus);
        }
    }

    paramsMAX96717F.ConfigPhy.mapping.enableMapping = false;
    paramsMAX96717F.ConfigPhy.numDataLanes = 4;

    LOG_INFO("isLaneSwapNeeded %d\n", m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.phyLanes.isLaneSwapNeeded);
    LOG_INFO("isTwoLane %d\n", m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.phyLanes.isTwoLane);

    if (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.phyLanes.isLaneSwapNeeded) {
        paramsMAX96717F.ConfigPhy.mapping.phy1_d0 = 3;
        paramsMAX96717F.ConfigPhy.mapping.phy1_d1 = 2;
        paramsMAX96717F.ConfigPhy.mapping.phy2_d0 = 1;
        paramsMAX96717F.ConfigPhy.mapping.phy2_d1 = 0;
        paramsMAX96717F.ConfigPhy.mapping.enableMapping = true;
    } else if (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.phyLanes.isTwoLane) {
        paramsMAX96717F.ConfigPhy.numDataLanes = 2;
    }
    LOG_INFO("CDI_WRITE_PARAM_CMD_MAX96717F_CONFIG_PHY\n");
    nvmStatus = MAX96717FWriteParameters(serCDI,
                                         CDI_WRITE_PARAM_CMD_MAX96717F_CONFIG_PHY,
                                         sizeof(paramsMAX96717F.ConfigPhy),
                                         &paramsMAX96717F);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_HEX_UINT("CDI_WRITE_PARAM_CMD_MAX96717F_CONFIG_PHY failed with nvmediaErr", nvmStatus);
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
        SIPL_LOG_ERR_STR_INT("MAX96712: CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_DOUBLE_PIXEL_MODE_SINGLE_PIPELINE failed with nvmediaErr", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    LOG_INFO("CDI_WRITE_PARAM_CMD_MAX96717F_CONFIG_VIDEO_PIPELINES\n");
    paramsMAX96717F.ConfigVideoPipeline.dataType = dataTypeMAX96717F;
    paramsMAX96717F.ConfigVideoPipeline.embDataType = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bEmbeddedDataType;
    nvmStatus = MAX96717FWriteParameters(serCDI,
                                         CDI_WRITE_PARAM_CMD_MAX96717F_CONFIG_VIDEO_PIPELINES,
                                         sizeof(paramsMAX96717F.ConfigVideoPipeline),
                                         &paramsMAX96717F);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_HEX_UINT("CDI_WRITE_PARAM_CMD_MAX96717F_CONFIG_VIDEO_PIPELINES failed with nvmediaErr", nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    LOG_INFO("CDI_WRITE_PARAM_CMD_MAX96712_SINGLE_PIPELINE_MAPPING\n");
    pipeLineMap = &paramsMAX96712.PipelineMapping.linkPipelineMap[m_oLinkParams.ulinkIndex];
    pipeLineMap->isEmbDataType = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bEmbeddedDataType;
    pipeLineMap->vcID = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.uVCID;
    pipeLineMap->dataType = dataTypeMAX96712;
    pipeLineMap->isDTOverride = !m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bEmbeddedDataType;
    pipeLineMap->isSinglePipeline  = true;
    pipeLineMap->isMapToUnusedCtrl = false;
    paramsMAX96712.PipelineMapping.link = link;
    nvmStatus = MAX96712WriteParameters(deserializerCDI,
                                        CDI_WRITE_PARAM_CMD_MAX96712_PIPELINE_MAPPING,
                                        sizeof(paramsMAX96712.PipelineMapping),
                                        &paramsMAX96712);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_HEX_UINT("CDI_WRITE_PARAM_CMD_MAX96712_SINGLE_PIPELINE_MAPPING failed with nvmediaErr", nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    paramsMAX96712 = {};
    LOG_INFO("Initializing link %u\n", m_oLinkParams.ulinkIndex);

    if (m_oLinkParams.bPassive or m_oLinkParams.bEnableSimulator) {
        return NVSIPL_STATUS_OK;
    }

    if (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bEnableTriggerModeSync) {
        LOG_INFO("set FSYNC pin to the low\n");
        paramsMAX96717F.GPIOOutp.gpioInd = getMAX96717FGPIO(m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.frameSync.pinNum);
        paramsMAX96717F.GPIOOutp.level = false;
        nvmStatus = MAX96717FWriteParameters(serCDI,
                                             CDI_WRITE_PARAM_CMD_MAX96717F_SET_GPIO_OUTPUT,
                                             sizeof(paramsMAX96717F.GPIOOutp),
                                             &paramsMAX96717F);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_HEX_UINT("CDI_WRITE_PARAM_CMD_MAX96717F_SET_GPIO_OUTPUT failed with nvmediaErr (0x%x)\n", nvmStatus);
            return ConvertNvMediaStatus(nvmStatus);
        }
    }

    /* Enable FRSYNC */
    if (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bEnableTriggerModeSync &&
        !m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bPostSensorInitFsync) {
        if ((revision == CDI_MAX96712_REV_1) or (revision == CDI_MAX96712_REV_2)) {
            internalSyncMode = CDI_MAX96712_FSYNC_AUTO;
        } else {
            internalSyncMode = CDI_MAX96712_FSYNC_OSC_MANUAL;
        }

        LOG_INFO("Set Fsync\n");
        paramsMAX96712.FSyncSettings.FSyncMode = (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bEnableInternalSync == true) ?
                                                  internalSyncMode : CDI_MAX96712_FSYNC_EXTERNAL;
        paramsMAX96712.FSyncSettings.fps = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.fFrameRate;
        paramsMAX96712.FSyncSettings.link = link;
        LOG_INFO("Set FSYNC mode\n");
        nvmStatus = MAX96712WriteParameters(deserializerCDI,
                                            CDI_WRITE_PARAM_CMD_MAX96712_SET_FSYNC,
                                            sizeof(paramsMAX96712.FSyncSettings),
                                            &paramsMAX96712);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_HEX_UINT("CDI_WRITE_PARAM_CMD_MAX96712_FSYNC_MODE failed with nvmediaErr", nvmStatus);
            return ConvertNvMediaStatus(nvmStatus);
        }
    }

    CNvMCameraModuleCommon::ConnectionProperty::SensorConnectionProperty::SensorReset sensorReset =
            m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.sensorReset;
    if (sensorReset.isNeeded) {
        LOG_INFO("Hold sensor reset\n");
        paramsMAX96717F.GPIOOutp.gpioInd = getMAX96717FGPIO(sensorReset.pinNum);
        paramsMAX96717F.GPIOOutp.level = !sensorReset.releaseResetLevel;
        nvmStatus = MAX96717FWriteParameters(serCDI,
                                             CDI_WRITE_PARAM_CMD_MAX96717F_SET_GPIO_OUTPUT,
                                             sizeof(paramsMAX96717F.GPIOOutp),
                                             &paramsMAX96717F);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_HEX_UINT("CDI_WRITE_PARAM_CMD_MAX96717F_SET_GPIO_OUTPUT failed with nvmediaErr 0x", nvmStatus);
            return ConvertNvMediaStatus(nvmStatus);
        }

        LOG_INFO("Release sensor reset\n");
        paramsMAX96717F.GPIOOutp.gpioInd = getMAX96717FGPIO(sensorReset.pinNum);
        paramsMAX96717F.GPIOOutp.level = sensorReset.releaseResetLevel;
        nvmStatus = MAX96717FWriteParameters(serCDI,
                                             CDI_WRITE_PARAM_CMD_MAX96717F_SET_GPIO_OUTPUT,
                                             sizeof(paramsMAX96717F.GPIOOutp),
                                             &paramsMAX96717F);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_HEX_UINT("CDI_WRITE_PARAM_CMD_MAX96717F_SET_GPIO_OUTPUT failed with nvmediaErr 0x", nvmStatus);
            return ConvertNvMediaStatus(nvmStatus);
        }
    }

    if (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bEnableTriggerModeSync) {
        paramsMAX96717F.FSyncGPIO.gpioInd = getMAX96717FGPIO(
            m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.frameSync.pinNum);
        if (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bEnableInternalSync) {
            paramsMAX96717F.FSyncGPIO.rxID = CDI_MAX96712_GPIO_20;
        } else {
            paramsMAX96717F.FSyncGPIO.rxID = CDI_MAX96712_GPIO_2;
        }

        nvmStatus = MAX96717FWriteParameters(serCDI,
                                             CDI_WRITE_PARAM_CMD_MAX96717F_SET_FSYNC_GPIO,
                                             sizeof(paramsMAX96717F.FSyncGPIO),
                                             &paramsMAX96717F);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_HEX_UINT("CDI_WRITE_PARAM_CMD_MAX96717F_SET_FSYNC_GPIO failed with nvmediaErr", nvmStatus);
            return ConvertNvMediaStatus(nvmStatus);
        }
    }

    return ConvertNvMediaStatus(nvmStatus);
}

SIPLStatus CNvMTransportLink_Max96712_96717::PostSensorInit(uint8_t const linkMask, bool const groupInitProg) const {
    WriteParametersParamMAX96712 paramsMAX96712 = {};
    ReadParametersParamMAX96712 readParamsMAX96712 = {};
    LinkMAX96712 link = (groupInitProg == true) ? (LinkMAX96712)linkMask : GetMAX96712Link(m_oLinkParams.ulinkIndex);
    DevBlkCDIDevice *deserializerCDI = m_oLinkParams.pDeserCDIDevice;
    NvMediaStatus nvmStatus = NVMEDIA_STATUS_OK;
    RevisionMAX96712 revision = CDI_MAX96712_REV_INVALID;
    FSyncModeMAX96712 internalSyncMode = CDI_MAX96712_FSYNC_INVALID;

    if (m_oLinkParams.bPassive or m_oLinkParams.bEnableSimulator) {
        return NVSIPL_STATUS_OK;
    }

    LOG_INFO("Read revision\n");
    nvmStatus = MAX96712ReadParameters(deserializerCDI,
                                       CDI_READ_PARAM_CMD_MAX96712_REV_ID,
                                       sizeof(readParamsMAX96712.revision),
                                       &readParamsMAX96712.revision);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_HEX_UINT("CDI_READ_PARAM_CMD_MAX96712_REV_ID failed with nvmediaErr 0x", nvmStatus);
        goto done;
    }

    revision = readParamsMAX96712.revision;

    LOG_INFO("MAX96712 Revision %d", revision);

    /* Enable FRSYNC */
    if (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bEnableTriggerModeSync &&
        m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bPostSensorInitFsync) {
        if ((revision == CDI_MAX96712_REV_1) or (revision == CDI_MAX96712_REV_2)) {
            internalSyncMode = CDI_MAX96712_FSYNC_AUTO;
        } else {
            internalSyncMode = CDI_MAX96712_FSYNC_OSC_MANUAL;
        }
        LOG_INFO("Set Fsync\n");
        paramsMAX96712.FSyncSettings.FSyncMode = (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bEnableInternalSync) ?
                                                  internalSyncMode : CDI_MAX96712_FSYNC_EXTERNAL;
        paramsMAX96712.FSyncSettings.fps = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.fFrameRate;
        paramsMAX96712.FSyncSettings.link = link;
        LOG_INFO("Set FSYNC mode\n");
        nvmStatus = MAX96712WriteParameters(deserializerCDI,
                                            CDI_WRITE_PARAM_CMD_MAX96712_SET_FSYNC,
                                            sizeof(paramsMAX96712.FSyncSettings),
                                            &paramsMAX96712);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_HEX_UINT("CDI_WRITE_PARAM_CMD_MAX96712_FSYNC_MODE failed with nvmediaErr", nvmStatus);
            goto done;
        }
    }

done:
    return ConvertNvMediaStatus(nvmStatus);
}

} // end of namespace nvsipl
