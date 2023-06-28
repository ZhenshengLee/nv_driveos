/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "CNvMTransportLink_DS90UB9724_971.hpp"
#include "sipl_error.h"

#include <string>

#include <chrono>
#include <thread>

// Include necessary CDI driver headers
extern "C" {
#include "cdi_ds90ub9724.h"
#include "cdi_ds90ub971.h"
}

namespace nvsipl {
#if !NV_IS_SAFETY
void CNvMTransportLink_DS90UB9724_971::DumpLinkParams()
{
    LOG_INFO("Link parameters\n");
    LOG_INFO("Link Index: %u \n", m_oLinkParams.ulinkIndex);
    LOG_INFO("Broadcast serializer addr: 0x%x \n", m_oLinkParams.uBrdcstSerAddr);
    LOG_INFO("Serializer addr: 0x%x \n", m_oLinkParams.uSerAddr);

    const CNvMCameraModuleCommon::ConnectionProperty::SensorConnectionProperty &sensor =
                m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty;
    LOG_INFO("\nBroadcast sensor addr: 0x%x \n", sensor.uBrdcstSensorAddrs);
    LOG_INFO("Sensor addr: 0x%x \n", sensor.uSensorAddrs);
    LOG_INFO("VCID: %u \n", sensor.uVCID);
    LOG_INFO("Embedded data type: %s \n", sensor.bEmbeddedDataType ? "true" : "false");
    LOG_INFO("Trigger mode sync: %s \n", sensor.bEnableTriggerModeSync ? "true" : "false");
    LOG_INFO("Frame rate: %.2f fps \n", sensor.fFrameRate);

    const std::uint8_t & broadEEPROMAddr = m_oLinkParams.moduleConnectionProperty.brdcstEepromAddr;
    LOG_INFO("\nBroadcast EEPROM addr: 0x%x \n", broadEEPROMAddr);

    const std::uint8_t & EEPROMAddr = m_oLinkParams.moduleConnectionProperty.eepromAddr;
    LOG_INFO("EEPROM addr: 0x%x \n", EEPROMAddr);

    LOG_INFO("Simulator mode: %u \n", m_oLinkParams.bEnableSimulator);
    LOG_INFO("Passive mode: %u \n", m_oLinkParams.bPassive);
}
#endif

SIPLStatus CNvMTransportLink_DS90UB9724_971::SetupAddressTranslations(DevBlkCDIDevice* brdcstSerCDI)
{
    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMTransportLink_DS90UB9724_971::SetupConfigLink(DevBlkCDIDevice* brdcstSerCDI, uint8_t linkMask, bool groupInitProg)
{
    NvMediaStatus nvmStatus = NVMEDIA_STATUS_OK;
    WriteParametersDS90UB9724 paramsDS90UB9724 = {};
    ReadWriteParamsDS90UB971 paramsDS90UB971 = {};
    DevBlkCDIDevice *deserializerCDI = m_oLinkParams.pDeserCDIDevice;
    DevBlkCDIDevice *serCDI = m_oLinkParams.pSerCDIDevice;

    paramsDS90UB9724.link = m_oLinkParams.ulinkIndex;
    nvmStatus = DS90UB9724WriteParameters(deserializerCDI,
                                          CDI_WRITE_PARAM_CMD_DS90UB9724_SET_LINK_SPEED_FPDLINK4,
                                          sizeof(paramsDS90UB9724.link),
                                          &paramsDS90UB9724);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB9724: CDI_WRITE_PARAM_CMD_DS90UB9724_SET_LINK_SPEED_FPDLINK4 failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    paramsDS90UB9724.i2cTranslation.link = m_oLinkParams.ulinkIndex;
    paramsDS90UB9724.i2cTranslation.slaveID = m_oLinkParams.uBrdcstSerAddr;
    paramsDS90UB9724.i2cTranslation.slaveAlias = m_oLinkParams.uSerAddr;
    paramsDS90UB9724.i2cTranslation.lock = true;
    paramsDS90UB9724.i2cTranslation.i2cTransID = CDI_DS90UB9724_I2C_TRANSLATE_SER;
    nvmStatus = DS90UB9724WriteParameters(deserializerCDI,
                                          CDI_WRITE_PARAM_CMD_DS90UB9724_SET_I2C_TRANSLATION,
                                          sizeof(paramsDS90UB9724.i2cTranslation),
                                          &paramsDS90UB9724);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB9724: CDI_WRITE_PARAM_CMD_DS90UB9724_SET_I2C_TRANSLATION failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    //Check Lock status()
    paramsDS90UB9724.link = m_oLinkParams.ulinkIndex;
    nvmStatus = DS90UB9724WriteParameters(deserializerCDI,
                                          CDI_WRITE_PARAM_CMD_DS90UB9724_CHECK_LOCK,
                                          sizeof(paramsDS90UB9724.link),
                                          &paramsDS90UB9724);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB9724: CDI_WRITE_PARAM_CMD_DS90UB9724_CHECK_LOCK failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    nvmStatus = DS90UB971SetDefaults(serCDI);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB971: DS90UB971SetDefaults failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    paramsDS90UB9724.i2cTranslation.link = m_oLinkParams.ulinkIndex;
    paramsDS90UB9724.i2cTranslation.slaveID = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.uBrdcstSensorAddrs;
    paramsDS90UB9724.i2cTranslation.slaveAlias = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.uSensorAddrs;
    paramsDS90UB9724.i2cTranslation.lock = false;
    paramsDS90UB9724.i2cTranslation.i2cTransID = CDI_DS90UB9724_I2C_TRANSLATE_SLAVE0;
    nvmStatus = DS90UB9724WriteParameters(deserializerCDI,
                                          CDI_WRITE_PARAM_CMD_DS90UB9724_SET_I2C_TRANSLATION,
                                          sizeof(paramsDS90UB9724.i2cTranslation),
                                          &paramsDS90UB9724);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB9724: CDI_WRITE_PARAM_CMD_DS90UB9724_SET_I2C_TRANSLATION Sensor failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    paramsDS90UB9724.i2cTranslation.link = m_oLinkParams.ulinkIndex;
    paramsDS90UB9724.i2cTranslation.slaveID = m_oLinkParams.uBrdcstSerAddr;
    paramsDS90UB9724.i2cTranslation.slaveAlias = m_oLinkParams.uSerAddr;
    paramsDS90UB9724.i2cTranslation.lock = false;
    paramsDS90UB9724.i2cTranslation.i2cTransID = CDI_DS90UB9724_I2C_TRANSLATE_SER;
    nvmStatus = DS90UB9724WriteParameters(deserializerCDI,
                                          CDI_WRITE_PARAM_CMD_DS90UB9724_SET_I2C_TRANSLATION,
                                          sizeof(paramsDS90UB9724.i2cTranslation),
                                          &paramsDS90UB9724);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB9724: CDI_WRITE_PARAM_CMD_DS90UB9724_SET_I2C_TRANSLATION failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    paramsDS90UB9724.link = m_oLinkParams.ulinkIndex;
    nvmStatus = DS90UB9724WriteParameters(deserializerCDI,
                                          CDI_WRITE_PARAM_CMD_DS90UB9724_ENABLE_AEQ_LMS,
                                          sizeof(paramsDS90UB9724.link),
                                          &paramsDS90UB9724);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB9724: CDI_WRITE_PARAM_CMD_DS90UB9724_ENABLE_AEQ_LMS failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    nvmStatus = DS90UB971CheckPresence(serCDI);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB971: DS90UB971CheckPresence failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    paramsDS90UB971.ClkOut.N = 156;
    paramsDS90UB971.ClkOut.M = 0xE4;

    nvmStatus = DS90UB971WriteParameters(serCDI,
                            CDI_WRITE_PARAM_CMD_DS90UB971_SET_CLKOUT,
                            sizeof(paramsDS90UB971.ClkOut),
                            &paramsDS90UB971);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB971: CDI_WRITE_PARAM_CMD_DS90UB971_SET_CLKOUT failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    paramsDS90UB9724.VCMap.link = m_oLinkParams.ulinkIndex;
    paramsDS90UB9724.VCMap.inVCID = 0; // Assumption is the sensor to generate frames with VC0
    paramsDS90UB9724.VCMap.outVCID = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.uVCID;
    nvmStatus = DS90UB9724WriteParameters(deserializerCDI,
                                          CDI_WRITE_PARAM_CMD_DS90UB9724_SET_VC_MAP,
                                          sizeof(paramsDS90UB9724.VCMap),
                                          &paramsDS90UB9724);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB9724: CDI_WRITE_PARAM_CMD_DS90UB9724_SET_VC_MAP failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    paramsDS90UB971.localGpio.gpio = CDI_DS90UB971_GPIO_0;
    paramsDS90UB971.localGpio.level = 0;

    nvmStatus = DS90UB971WriteParameters(serCDI,
                            CDI_WRITE_PARAM_CMD_DS90UB971_SET_LOCAL_GPIO,
                            sizeof(paramsDS90UB971.localGpio),
                            &paramsDS90UB971);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB971: CDI_WRITE_PARAM_CMD_DS90UB971_SET_CLKOUT failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    /* Set XCLR high (sensor reset) */
    paramsDS90UB971.localGpio.gpio = CDI_DS90UB971_GPIO_3;
    paramsDS90UB971.localGpio.level = 1;

    nvmStatus = DS90UB971WriteParameters(serCDI,
                            CDI_WRITE_PARAM_CMD_DS90UB971_SET_LOCAL_GPIO,
                            sizeof(paramsDS90UB971.localGpio),
                            &paramsDS90UB971);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB971: CDI_WRITE_PARAM_CMD_DS90UB971_SET_CLKOUT failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    std::this_thread::sleep_for<>(std::chrono::milliseconds(10));

    /* Assert XCLR (sensor reset) */
    paramsDS90UB971.localGpio.gpio = CDI_DS90UB971_GPIO_3;
    paramsDS90UB971.localGpio.level = 0;

    nvmStatus = DS90UB971WriteParameters(serCDI,
                            CDI_WRITE_PARAM_CMD_DS90UB971_SET_LOCAL_GPIO,
                            sizeof(paramsDS90UB971.localGpio),
                            &paramsDS90UB971);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB971: CDI_WRITE_PARAM_CMD_DS90UB971_SET_CLKOUT failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    std::this_thread::sleep_for<>(std::chrono::milliseconds(10));

    /* deassert XCLR (sensor reset) */
    paramsDS90UB971.localGpio.gpio = CDI_DS90UB971_GPIO_3;
    paramsDS90UB971.localGpio.level = 1;

    nvmStatus = DS90UB971WriteParameters(serCDI,
                            CDI_WRITE_PARAM_CMD_DS90UB971_SET_LOCAL_GPIO,
                            sizeof(paramsDS90UB971.localGpio),
                            &paramsDS90UB971);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB971: CDI_WRITE_PARAM_CMD_DS90UB971_SET_CLKOUT failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    return NVSIPL_STATUS_OK;
}

// This will do the init for 1 maxim deser and up to MAX_LINKS_PER_DESER maxim serializers.
SIPLStatus CNvMTransportLink_DS90UB9724_971::Init(DevBlkCDIDevice* brdcstSerCDI, uint8_t linkMask, bool groupInitProg)
{
    SIPLStatus status = NVSIPL_STATUS_OK;
    NvMediaStatus nvmStatus = NVMEDIA_STATUS_OK;
    DevBlkCDIDevice *deserializerCDI = m_oLinkParams.pDeserCDIDevice;
    DevBlkCDIDevice *serCDI = m_oLinkParams.pSerCDIDevice;
    WriteParametersDS90UB9724 paramsDS90UB9724 = {};
    ReadWriteParamsDS90UB971 paramsDS90UB971 = {};

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    const auto & sensorProperties = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty;
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif // !NV_IS_SAFETY

#if !NV_IS_SAFETY
    DumpLinkParams();
#endif

    if (m_oLinkParams.bPassive or m_oLinkParams.bEnableSimulator) {
        return NVSIPL_STATUS_OK;
    }

    // Setup config link
    status = SetupConfigLink(brdcstSerCDI, linkMask, groupInitProg);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB9724: SetupConfigLink failed with SIPL error", (int32_t)status);
        return status;
    }

    paramsDS90UB9724.fsyncGpio.link = m_oLinkParams.ulinkIndex;
    paramsDS90UB9724.fsyncGpio.extFsyncGpio = CDI_DS90UB9724_GPIO_1;
    paramsDS90UB9724.fsyncGpio.bc_gpio = CDI_DS90UB971_GPIO_0;
    nvmStatus = DS90UB9724WriteParameters(deserializerCDI,
                                          CDI_WRITE_PARAM_CMD_DS90UB9724_ENABLE_FSYNC_GIPO,
                                          sizeof(paramsDS90UB9724.fsyncGpio),
                                          &paramsDS90UB9724);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB9724: CDI_WRITE_PARAM_CMD_DS90UB9702_ENABLE_FSYNC_GIPO failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    paramsDS90UB971.fsyncGpio.gpioInd = CDI_DS90UB971_GPIO_0;
    paramsDS90UB971.fsyncGpio.rxGpioID = CDI_DS90UB9724_GPIO_1;

    nvmStatus = DS90UB971WriteParameters(serCDI,
                                         CDI_WRITE_PARAM_CMD_DS90UB971_SET_FSYNC_GPIO,
                                         sizeof(paramsDS90UB971.fsyncGpio),
                                         &paramsDS90UB971);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB971: CDI_WRITE_PARAM_CMD_DS90UB971_SET_FSYNC_GPIO failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    // Enable TPG
    if (sensorProperties.bEnableTPG) {
        paramsDS90UB971.TPG.width  = sensorProperties.width;
        paramsDS90UB971.TPG.height = sensorProperties.height +
                                      sensorProperties.embeddedTop +
                                      sensorProperties.embeddedBot;
        paramsDS90UB971.TPG.frameRate = sensorProperties.fFrameRate;
        nvmStatus = DS90UB971WriteParameters(serCDI,
                                             CDI_WRITE_PARAM_CMD_DS90UB971_SET_TPG,
                                             sizeof(paramsDS90UB971.TPG),
                                             &paramsDS90UB971.TPG);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("DS90UB9724: CDI_WRITE_PARAM_CMD_DS90UB971_SET_TPG failed with NvMedia error", (int32_t)nvmStatus);
            return ConvertNvMediaStatus(nvmStatus);
        }
    }
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif // !NV_IS_SAFETY

    return NVSIPL_STATUS_OK;
}

} // end of namespace nvsipl
