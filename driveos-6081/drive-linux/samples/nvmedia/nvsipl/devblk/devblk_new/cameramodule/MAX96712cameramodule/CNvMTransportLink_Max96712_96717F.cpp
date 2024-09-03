/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include "CNvMTransportLink_Max96712_96717F.hpp"
#include "CNvMCameraModuleCommon.hpp"

#include <string>
#include <vector>

// Include necessary ISC driver headers
#include "sipl_error.h"
#include "sipl_util.h"
#include "os_common.h"

#if FUSA_CDD_NV
#include "cdi_max96712_nv.h"
#else
#include "cdi_max96712.h"
#endif

#include "cdi_max96717f.h"
#include <cstdint>

namespace nvsipl {

static uint32_t const VPRBS_SLEEP_US {3000U};
static uint32_t const ERRG_DIAGNOSTICTEST_SLEEP_US {100U};

static GPIOTypeMAX96717F
getMAX96717FGPIO(
    uint8_t const gpioInd)
{
    GPIOTypeMAX96717F gpio {CDI_MAX96717F_GPIO_TYPE_INVALID};
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
            SIPL_LOG_ERR_STR_UINT("Invalid Max96717F GPIO pin",
                                   static_cast<uint32_t>(gpioInd));
            break;
    }
    return gpio;
}

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
void CNvMTransportLink_Max96712_96717F::DumpLinkParams() const
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
        m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bEmbeddedDataType
            ? "true" : "false");
    LOG_INFO("Trigger mode sync: %s \n",
        m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bEnableTriggerModeSync
            ? "true" : "false");
    LOG_INFO("Frame rate: %.2f fps \n",
        m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.fFrameRate);

    LOG_INFO("\nBroadcast EEPROM addr: 0x%x \n",
        m_oLinkParams.moduleConnectionProperty.brdcstEepromAddr);
    LOG_INFO("EEPROM addr: 0x%x \n", m_oLinkParams.moduleConnectionProperty.eepromAddr);
    LOG_INFO("Simulator mode: %u \n", m_oLinkParams.bEnableSimulator);
    LOG_INFO("Passive mode: %u \n", m_oLinkParams.bPassive);
}
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

SIPLStatus CNvMTransportLink_Max96712_96717F::SetupAddressTranslations(
    DevBlkCDIDevice const* const brdcstSerCDI) const
{
    NvMediaStatus nvmStatus {NVMEDIA_STATUS_OK};

    // Set address translation for the sensor
    ReadWriteParamsMAX96717F paramsMAX96717F {};
    paramsMAX96717F.Translator.source =
        m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.uSensorAddrs;
    paramsMAX96717F.Translator.destination =
        m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.uBrdcstSensorAddrs;
    LOG_INFO("Translate image sensor device addr %x to %x\n",
                paramsMAX96717F.Translator.source, paramsMAX96717F.Translator.destination);
    nvmStatus = MAX96717FWriteParameters(brdcstSerCDI,
                                        CDI_WRITE_PARAM_CMD_MAX96717F_SET_TRANSLATOR_A,
                                        sizeof(paramsMAX96717F.Translator),
                                        &paramsMAX96717F);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_HEX_UINT(
            "CDI_WRITE_PARAM_CMD_MAX96717F_SET_TRANSLATOR_A failed with nvmediaErr",
            static_cast<uint32_t>(nvmStatus));
    }

    if (nvmStatus == NVMEDIA_STATUS_OK) {
        // Set address translation for the eeprom
        if (m_oLinkParams.moduleConnectionProperty.eepromAddr != static_cast<uint8_t> UINT8_MAX) {
            ReadWriteParamsMAX96717F paramsMAX96717F {};
            paramsMAX96717F.Translator.source = m_oLinkParams.moduleConnectionProperty.eepromAddr;
            paramsMAX96717F.Translator.destination =
                m_oLinkParams.moduleConnectionProperty.brdcstEepromAddr;
            LOG_INFO("Translate eeprom device addr %x to %x\n",
                        paramsMAX96717F.Translator.source, paramsMAX96717F.Translator.destination);
            nvmStatus = MAX96717FWriteParameters(brdcstSerCDI,
                                                CDI_WRITE_PARAM_CMD_MAX96717F_SET_TRANSLATOR_B,
                                                sizeof(paramsMAX96717F.Translator),
                                                &paramsMAX96717F);
            if (nvmStatus != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_HEX_UINT(
                    "CDI_WRITE_PARAM_CMD_MAX96717F_SET_TRANSLATOR_B failed with nvmediaErr",
                    static_cast<uint32_t>(nvmStatus));
            }
        }
    }

    if (nvmStatus == NVMEDIA_STATUS_OK) {
        // Set unique address with broadcast address
        ReadWriteParamsMAX96717F paramsMAX96717F {};
        LOG_INFO("Set unique address\n");
        paramsMAX96717F.DeviceAddress.address = m_oLinkParams.uSerAddr;
        nvmStatus = MAX96717FWriteParameters(brdcstSerCDI,
                                           CDI_WRITE_PARAM_CMD_MAX96717F_SET_DEVICE_ADDRESS,
                                           sizeof(paramsMAX96717F.DeviceAddress.address),
                                           &paramsMAX96717F);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_HEX_UINT(
                "CDI_WRITE_PARAM_CMD_MAX96717F_SET_DEVICE_ADDRESS failed with nvmediaErr",
                static_cast<uint32_t>(nvmStatus));
        }
    }


    return ConvertNvMediaStatus(nvmStatus);
}

SIPLStatus CNvMTransportLink_Max96712_96717F::BasicCheckModule(
    DevBlkCDIDevice const* const brdcstSerCDI) const
{
    NvMediaStatus nvmStatus {NVMEDIA_STATUS_OK};
    ReadParametersParamMAX96712 readParamsMAX96712 {};
    DevBlkCDIDevice const* const deserializerCDI {m_oLinkParams.pDeserCDIDevice};
    SIPLStatus status {NVSIPL_STATUS_OK};

    LOG_INFO("Read revision\n");
    nvmStatus = MAX96712ReadParameters(deserializerCDI,
                                       CDI_READ_PARAM_CMD_MAX96712_REV_ID,
                                       sizeof(RevisionMAX96712),
                                       &readParamsMAX96712.revision);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_HEX_UINT("CDI_READ_PARAM_CMD_MAX96712_REV_ID failed with nvmediaErr",
                                   static_cast<uint32_t>(nvmStatus));
        status = ConvertNvMediaStatus(nvmStatus);
    } else {
        if (((((m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.
                inputFormat.inputFormatType !=
                NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW10) &&
            (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.
                inputFormat.inputFormatType !=
                NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12)) &&
            (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.
                inputFormat.inputFormatType !=
                NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12RJ)) &&
            (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.
                inputFormat.inputFormatType !=
                NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW8))) {
            SIPL_LOG_ERR_STR("Input format not supported");
            nvmStatus = NVMEDIA_STATUS_BAD_PARAMETER;
            status =ConvertNvMediaStatus(nvmStatus);
        }
    }

    if (NVMEDIA_STATUS_OK == nvmStatus) {
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
        DumpLinkParams();
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
        if ((m_oLinkParams.bPassive or m_oLinkParams.bEnableSimulator) or
            m_oLinkParams.bCBAEnabled) {
            status = NVSIPL_STATUS_OK;
        } else {
            LOG_INFO("Check config link lock \n");
            LinkMAX96712 const uLinkVal {GetMAX96712Link(m_oLinkParams.ulinkIndex)};

            nvmStatus = MAX96712CheckLink(deserializerCDI,
                                            uLinkVal,
                                            CDI_MAX96712_LINK_LOCK_GMSL2,
                                            true);
            if (nvmStatus != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_HEX_UINT(
                    "CDICheckLink(CDI_MAX96712_GMSL2_LINK_LOCK failed with "
                    "nvmediaErr", static_cast<uint32_t>(nvmStatus));
                status = ConvertNvMediaStatus(nvmStatus);
            } else {
                // Check presence of serializer
                LOG_INFO("Check broadcast serializer is present\n");
                nvmStatus = MAX96717FCheckPresence(brdcstSerCDI);
                if (nvmStatus != NVMEDIA_STATUS_OK) {
                    SIPL_LOG_ERR_STR_HEX_UINT(
                        "MAX96717CheckPresence(brdcstSerCDI) failed with "
                        "nvmediaErr", static_cast<uint32_t>(nvmStatus));
                    status = ConvertNvMediaStatus(nvmStatus);
                }

#ifdef NVMEDIA_QNX
                // Perform Serializer Init-time SM23(GPO Toggle & Readback)
                if (nvmStatus == NVMEDIA_STATUS_OK) {
                    std::vector<uint8_t> const vSerSM23GPIOs {m_oLinkParams.
                        moduleConnectionProperty.sensorConnectionProperty.
                        vSerGPIOsToggleTest};
                    // Skip performing SM23 diagnostic tests if
                    // vSerSM23GPIOs is empty.
                    if (!vSerSM23GPIOs.empty() && (initDone == false)) {
                        nvmStatus = MAX96717FVerifyGPIOReadBackStatus(
                            brdcstSerCDI,
                            vSerSM23GPIOs.data(),
                            static_cast<uint8_t>(vSerSM23GPIOs.size() & 0xFFU));
                        if (nvmStatus != NVMEDIA_STATUS_OK) {
                            SIPL_LOG_ERR_STR_HEX_UINT(
                                "MAX96717FVerifyGPIOReadBackStatus failed with "
                                "nvmediaErr", static_cast<uint32_t>(nvmStatus));
                            status = ConvertNvMediaStatus(nvmStatus);
                        }
                    }
                }

                // Perform Serializer Init-time SM40(GPIO open detection)
                // Only the GPIO pins that cannot run SM40 after the sensor is
                // initialized are tested here
                if (nvmStatus == NVMEDIA_STATUS_OK) {
                    std::vector<uint8_t> const vSerSM40GPIOs {m_oLinkParams.
                        moduleConnectionProperty.sensorConnectionProperty.
                        vSerGPIOsOpenDetection};
                    // Skip performing SM40 diagnostic tests if
                    // vSerSM40GPIOs is empty.
                    if (!vSerSM40GPIOs.empty()  && (initDone == false)) {
                        nvmStatus = MAX96717FVerifyGPIOOpenDetection(
                            brdcstSerCDI,
                            vSerSM40GPIOs.data(),
                            static_cast<uint8_t>(vSerSM40GPIOs.size() & 0xFFU));
                        if (nvmStatus != NVMEDIA_STATUS_OK) {
                            SIPL_LOG_ERR_STR_HEX_UINT(
                                "MAX96717FVerifyGPIOOpenDetection failed with "
                                "nvmediaErr", static_cast<uint32_t>(nvmStatus));
                            status = ConvertNvMediaStatus(nvmStatus);
                        }
                    }
                }
#endif

                // Setup address translations
                if (nvmStatus == NVMEDIA_STATUS_OK) {
                    LOG_INFO("Setup address translations\n");
                    status = SetupAddressTranslations(brdcstSerCDI);
                    if (status != NVSIPL_STATUS_OK) {
                        SIPL_LOG_ERR_STR_HEX_UINT(
                            "SetupAddressTranslations failed with SIPLStatus "
                            "error", static_cast<uint32_t>(status));
                    }
                }
            }
        }
    }

    return status;
}

// Enable/Disable EEPROM write protect
SIPLStatus CNvMTransportLink_Max96712_96717F::SetEEPROMWriteProtect(void) const
{
    NvMediaStatus nvmStatus {NVMEDIA_STATUS_OK};
    CNvMCameraModuleCommon::ConnectionProperty::SensorConnectionProperty::EepromWriteProtect
        const eepromWriteProtect {
            m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.eepromWriteProtect};

    if (eepromWriteProtect.isNeeded) {
        LOG_INFO("%s EEPROM write protection\n",
            eepromWriteProtect.writeProtectLevel ? "Enable" : "Disable");
        ReadWriteParamsMAX96717F paramsMAX96717F {};
        paramsMAX96717F.GPIOOutp.gpioInd = getMAX96717FGPIO(eepromWriteProtect.pinNum);
        paramsMAX96717F.GPIOOutp.level = eepromWriteProtect.writeProtectLevel;

        DevBlkCDIDevice const * const serCDI {m_oLinkParams.pSerCDIDevice};
        nvmStatus = MAX96717FWriteParameters(serCDI,
                                             CDI_WRITE_PARAM_CMD_MAX96717F_SET_GPIO_OUTPUT,
                                             sizeof(paramsMAX96717F.GPIOOutp),
                                             &paramsMAX96717F);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("Failed to set write protection for EEPROM");
        }
    }

    return ConvertNvMediaStatus(nvmStatus);
}

SIPLStatus CNvMTransportLink_Max96712_96717F::SetSERClk(void) const
{
    NvMediaStatus nvmStatus {NVMEDIA_STATUS_OK};
    SIPLStatus status {NVSIPL_STATUS_OK};

    /* Call MAX96717F SetDefault here */
    DevBlkCDIDevice const* const serCDI {m_oLinkParams.pSerCDIDevice};
    nvmStatus = MAX96717FSetDefaults(serCDI);
    if (NVMEDIA_STATUS_OK != nvmStatus) {
        SIPL_LOG_ERR_STR_HEX_UINT("MAX96717FSetDefaults failed with nvmediaErr",
                                    static_cast<uint32_t>(nvmStatus));
        status = ConvertNvMediaStatus(nvmStatus);
    } else {
        ReadWriteParamsMAX96717F const paramsMAX96717F {};
        nvmStatus = MAX96717FWriteParameters(serCDI,
                                             CDI_WRITE_PARAM_CMD_MAX96717F_ENABLE_PCLK_PIO_SLEW,
                                             sizeof(paramsMAX96717F.ConfigPhy),
                                             &paramsMAX96717F);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_HEX_UINT(
                "CDI_WRITE_PARAM_CMD_MAX96717F_ENABLE_PIO_SLEW failed with nvmediaErr",
                static_cast<uint32_t>(nvmStatus));
            status = ConvertNvMediaStatus(nvmStatus);
        }
    }

    if (NVMEDIA_STATUS_OK == nvmStatus) {
        CNvMCameraModuleCommon::ConnectionProperty::
        SensorConnectionProperty::RefClock const refClock
                {m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.refClock};
        if (refClock.isNeeded) {
            uint8_t const sensorClock{m_oLinkParams.moduleConnectionProperty.
                sensorConnectionProperty.refClock.sensorClock};
            if (sensorClock != 0U) {
                LOG_INFO("Generate sensor clock\n");
                if (sensorClock == (uint8_t)UINT8_MAX) {
                    nvmStatus = NVMEDIA_STATUS_ERROR;
                    status = ConvertNvMediaStatus(nvmStatus);
                    SIPL_LOG_ERR_STR(
                        "CDI_WRITE_PARAM_CMD_MAX96717F_GENERATE_CLOCK failed. sensorClock value "
                        "overflow");
                } else {
                    ReadWriteParamsMAX96717F paramsMAX96717F {};
                    paramsMAX96717F.ClockRate.freq = sensorClock;
                    nvmStatus = MAX96717FWriteParameters(
                        serCDI,
                        CDI_WRITE_PARAM_CMD_MAX96717F_GENERATE_CLOCK,
                        sizeof(paramsMAX96717F.ClockRate),
                        &paramsMAX96717F);
                    if (nvmStatus != NVMEDIA_STATUS_OK) {
                        SIPL_LOG_ERR_STR_HEX_UINT(
                            "CDI_WRITE_PARAM_CMD_MAX96717F_GNERATE_CLOCK failed with nvmediaErr",
                            static_cast<uint32_t>(nvmStatus));
                        status = ConvertNvMediaStatus(nvmStatus);
                    }
                }
            }
        }
    }

    if (NVMEDIA_STATUS_OK == nvmStatus) {
        CNvMCameraModuleCommon::ConnectionProperty::SensorConnectionProperty::RefClock const
            refClock {m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.refClock};
        if (refClock.isNeeded) {
            ReadWriteParamsMAX96717F paramsMAX96717F {};
            LOG_INFO("Enable ref clock\n");
            /* set source GPIO */
            paramsMAX96717F.RefClkGPIO.gpioInd = getMAX96717FGPIO(refClock.pinNum);
            paramsMAX96717F.RefClkGPIO.enableRClk = true;
            nvmStatus = MAX96717FWriteParameters(serCDI,
                                                 CDI_WRITE_PARAM_CMD_MAX96717F_ENABLE_REF_CLOCK,
                                                 sizeof(paramsMAX96717F.RefClkGPIO),
                                                 &paramsMAX96717F);
            if (nvmStatus != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_HEX_UINT(
                    "CDI_WRITE_PARAM_CMD_MAX96717F_ENABLE_REF_CLOCK failed with nvmediaErr",
                    static_cast<uint32_t>(nvmStatus));
                status = ConvertNvMediaStatus(nvmStatus);
            }
        }
    }

    return status;
}

SIPLStatus CNvMTransportLink_Max96712_96717F::SetSERFsyncLevel(void) const
{
    SIPLStatus status {NVSIPL_STATUS_OK};

    if (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bEnableTriggerModeSync) {
        NvMediaStatus nvmStatus {NVMEDIA_STATUS_OK};
        ReadWriteParamsMAX96717F ParamsMAX96717F {};
        LOG_INFO("set FSYNC pin to the low\n");
        ParamsMAX96717F.GPIOOutp.gpioInd = getMAX96717FGPIO(m_oLinkParams.moduleConnectionProperty.
            sensorConnectionProperty.frameSync.pinNum);
        ParamsMAX96717F.GPIOOutp.level = false;
        DevBlkCDIDevice const* const serCDI {m_oLinkParams.pSerCDIDevice};
        nvmStatus = MAX96717FWriteParameters(serCDI,
                                             CDI_WRITE_PARAM_CMD_MAX96717F_SET_GPIO_OUTPUT,
                                             sizeof(ParamsMAX96717F.GPIOOutp),
                                             &ParamsMAX96717F);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_HEX_UINT(
                "CDI_WRITE_PARAM_CMD_MAX96717F_SET_GPIO_OUTPUT failed with nvmediaErr (0x%x)\n",
                static_cast<uint32_t>(nvmStatus));
            status = ConvertNvMediaStatus(nvmStatus);
        }
    }

    return status;
}

SIPLStatus CNvMTransportLink_Max96712_96717F::SetSERFsync(void) const
{
    SIPLStatus status {NVSIPL_STATUS_OK};

    if (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bEnableTriggerModeSync) {
        NvMediaStatus nvmStatus {NVMEDIA_STATUS_OK};
        ReadWriteParamsMAX96717F syncParamsMAX96717F {};
        syncParamsMAX96717F.FSyncGPIO.gpioInd = getMAX96717FGPIO(
            m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.frameSync.pinNum);
        if (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bEnableInternalSync) {
            syncParamsMAX96717F.FSyncGPIO.rxID = CDI_MAX96712_GPIO_20;
        } else {
            syncParamsMAX96717F.FSyncGPIO.rxID = CDI_MAX96712_GPIO_2;
        }

        DevBlkCDIDevice const* const serCDIFsync {m_oLinkParams.pSerCDIDevice};
        nvmStatus = MAX96717FWriteParameters(serCDIFsync,
                                             CDI_WRITE_PARAM_CMD_MAX96717F_SET_FSYNC_GPIO,
                                             sizeof(syncParamsMAX96717F.FSyncGPIO),
                                             &syncParamsMAX96717F);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_HEX_UINT(
                "CDI_WRITE_PARAM_CMD_MAX96717F_SET_FSYNC_GPIO failed with nvmediaErr",
                static_cast<uint32_t>(nvmStatus));
            status = ConvertNvMediaStatus(nvmStatus);
        } else {
            /* Enable FRSYNC */
            if (!m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.
                bPostSensorInitFsync) {
                DevBlkCDIDevice const* const deserializerCDI {m_oLinkParams.pDeserCDIDevice};
                ReadParametersParamMAX96712 readParamsMAX96712 {};
                LOG_INFO("Read revision\n");
                /* enum is essential type so can't use for sizeof() operand.
                 * enum size is 4, WAR for changing from enum to int32_t */
                nvmStatus = MAX96712ReadParameters(deserializerCDI,
                                CDI_READ_PARAM_CMD_MAX96712_REV_ID,
                                sizeof(RevisionMAX96712),
                                &readParamsMAX96712.revision);
                if (nvmStatus != NVMEDIA_STATUS_OK) {
                    SIPL_LOG_ERR_STR_HEX_UINT(
                        "CDI_WRITE_PARAM_CMD_MAX96717F_SET_FSYNC_GPIO failed with nvmediaErr",
                        static_cast<uint32_t>(nvmStatus));
                    status = ConvertNvMediaStatus(nvmStatus);
                } else {
                    WriteParametersParamMAX96712 paramsMAX96712 {};
                    RevisionMAX96712 const revision {readParamsMAX96712.revision};
                    FSyncModeMAX96712 internalSyncMode {CDI_MAX96712_FSYNC_INVALID};

                    LOG_INFO("MAX96712 Revision %d", revision);
                    if (revision <= CDI_MAX96712_REV_2) {
                        internalSyncMode = CDI_MAX96712_FSYNC_AUTO;
                    } else {
                        internalSyncMode = CDI_MAX96712_FSYNC_OSC_MANUAL;
                    }

                    LOG_INFO("Set Fsync\n");
                    paramsMAX96712.FSyncSettings.FSyncMode = (m_oLinkParams.
                        moduleConnectionProperty.sensorConnectionProperty.bEnableInternalSync) ?
                            internalSyncMode : CDI_MAX96712_FSYNC_EXTERNAL;
                    paramsMAX96712.FSyncSettings.fps = toUint32FromFloat(
                        m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.fFrameRate);
                    LinkMAX96712 const link {GetMAX96712Link(m_oLinkParams.ulinkIndex)};
                    paramsMAX96712.FSyncSettings.link = link;
                    LOG_INFO("Set FSYNC mode\n");
                    nvmStatus = MAX96712WriteParameters(deserializerCDI,
                                                        CDI_WRITE_PARAM_CMD_MAX96712_SET_FSYNC,
                                                        sizeof(paramsMAX96712.FSyncSettings),
                                                        &paramsMAX96712);
                    if (nvmStatus != NVMEDIA_STATUS_OK) {
                        SIPL_LOG_ERR_STR_HEX_UINT(
                            "CDI_WRITE_PARAM_CMD_MAX96712_FSYNC_MODE failed with nvmediaErr",
                            static_cast<uint32_t>(nvmStatus));
                        status = ConvertNvMediaStatus(nvmStatus);
                    }
                }
            }
        }
    }

    return status;
}

SIPLStatus CNvMTransportLink_Max96712_96717F::SetSERPhy(void) const
{
    NvMediaStatus nvmStatus {NVMEDIA_STATUS_OK};
    SIPLStatus status {NVSIPL_STATUS_OK};

    ReadWriteParamsMAX96717F paramsMAX96717F {};
    paramsMAX96717F.ConfigPhy.mapping.enableMapping = false;
    paramsMAX96717F.ConfigPhy.numDataLanes = 4U;

    LOG_INFO("isLaneSwapNeeded %d\n",
        m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.phyLanes.isLaneSwapNeeded);
    LOG_INFO("isTwoLane %d\n",
        m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.phyLanes.isTwoLane);

    if (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.phyLanes.isLaneSwapNeeded) {
        paramsMAX96717F.ConfigPhy.mapping.phy1_d0 = m_oLinkParams.moduleConnectionProperty.
            sensorConnectionProperty.phyLanes.lane2;
        paramsMAX96717F.ConfigPhy.mapping.phy1_d1 = m_oLinkParams.moduleConnectionProperty.
            sensorConnectionProperty.phyLanes.lane3;
        paramsMAX96717F.ConfigPhy.mapping.phy2_d0 = m_oLinkParams.moduleConnectionProperty.
            sensorConnectionProperty.phyLanes.lane0;
        paramsMAX96717F.ConfigPhy.mapping.phy2_d1 = m_oLinkParams.moduleConnectionProperty.
            sensorConnectionProperty.phyLanes.lane1;
        paramsMAX96717F.ConfigPhy.mapping.enableMapping = true;
    } else if (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.phyLanes.isTwoLane) {
        paramsMAX96717F.ConfigPhy.numDataLanes = 2U;
    } else {
        /* code */
    }
    paramsMAX96717F.ConfigPhy.polarity.setPolarity = false;
    if (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.phyLanes.
        isLanePolarityConfigureNeeded) {
        paramsMAX96717F.ConfigPhy.polarity.phy1_d0 = m_oLinkParams.moduleConnectionProperty.
            sensorConnectionProperty.phyLanes.lane2pol;
        paramsMAX96717F.ConfigPhy.polarity.phy1_d1 = m_oLinkParams.moduleConnectionProperty.
            sensorConnectionProperty.phyLanes.lane3pol;
        paramsMAX96717F.ConfigPhy.polarity.phy1_clk= m_oLinkParams.moduleConnectionProperty.
            sensorConnectionProperty.phyLanes.clk1pol;
        paramsMAX96717F.ConfigPhy.polarity.phy2_d0 = m_oLinkParams.moduleConnectionProperty.
            sensorConnectionProperty.phyLanes.lane0pol;
        paramsMAX96717F.ConfigPhy.polarity.phy2_d1 = m_oLinkParams.moduleConnectionProperty.
            sensorConnectionProperty.phyLanes.lane1pol;
        paramsMAX96717F.ConfigPhy.polarity.phy2_clk= m_oLinkParams.moduleConnectionProperty.
            sensorConnectionProperty.phyLanes.clk2pol;
        paramsMAX96717F.ConfigPhy.polarity.setPolarity = true;
    }
    LOG_INFO("CDI_WRITE_PARAM_CMD_MAX96717F_CONFIG_PHY\n");
    DevBlkCDIDevice const* const serCDI {m_oLinkParams.pSerCDIDevice};
    nvmStatus = MAX96717FWriteParameters(serCDI,
                                         CDI_WRITE_PARAM_CMD_MAX96717F_CONFIG_PHY,
                                         sizeof(paramsMAX96717F.ConfigPhy),
                                         &paramsMAX96717F);
    if (NVMEDIA_STATUS_OK != nvmStatus) {
        SIPL_LOG_ERR_STR_HEX_UINT("CDI_WRITE_PARAM_CMD_MAX96717F_CONFIG_PHY failed with nvmediaErr",
                                  static_cast<uint32_t>(nvmStatus));
        status = ConvertNvMediaStatus(nvmStatus);
    } else {
        LinkMAX96712 const link {GetMAX96712Link(m_oLinkParams.ulinkIndex)};
        WriteParametersParamMAX96712 paramsMAX96712 {};

        paramsMAX96712.DoublePixelMode.link = link;
        paramsMAX96712.DoublePixelMode.isSharedPipeline = true;
        paramsMAX96712.DoublePixelMode.embDataType = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bEmbeddedDataType;

        switch (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.inputFormat.
            inputFormatType) {
        case NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW10:
            paramsMAX96712.DoublePixelMode.dataType = CDI_MAX96712_DATA_TYPE_RAW10;
            break;
        case NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12:
        case NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12RJ:
            paramsMAX96712.DoublePixelMode.dataType = CDI_MAX96712_DATA_TYPE_RAW12;
            break;
        default:
            paramsMAX96712.DoublePixelMode.dataType = CDI_MAX96712_DATA_TYPE_RAW8;
            break;
        }

        DevBlkCDIDevice const* const deserializerCDI {
            CNvMTransportLink::m_oLinkParams.pDeserCDIDevice};
        nvmStatus = MAX96712WriteParameters(deserializerCDI,
                                            CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_DOUBLE_PIXEL_MODE,
                                            sizeof(paramsMAX96712.DoublePixelMode),
                                            &paramsMAX96712);
        if (NVMEDIA_STATUS_OK != nvmStatus) {
            SIPL_LOG_ERR_STR_HEX_UINT(
                "CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_DOUBLE_PIXEL_MODE_SINGLE_PIPELINE failed with "
                "nvmediaErr", static_cast<uint32_t>(nvmStatus));
            status = ConvertNvMediaStatus(nvmStatus);
        }
    }

    return status;
}

SIPLStatus CNvMTransportLink_Max96712_96717F::SetSERVideo(void) const
{
    NvMediaStatus nvmStatus {NVMEDIA_STATUS_OK};
    SIPLStatus status {NVSIPL_STATUS_OK};

    ReadWriteParamsMAX96717F paramsMAX96717F {};
    LOG_INFO("CDI_WRITE_PARAM_CMD_MAX96717F_CONFIG_VIDEO_PIPELINES\n");
    paramsMAX96717F.ConfigVideoPipeline.embDataType = m_oLinkParams.moduleConnectionProperty.
        sensorConnectionProperty.bEmbeddedDataType;
    DevBlkCDIDevice const* const serCDI {m_oLinkParams.pSerCDIDevice};

    switch (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.inputFormat.
        inputFormatType) {
    case NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW10:
        paramsMAX96717F.ConfigVideoPipeline.dataType = CDI_MAX96717F_DATA_TYPE_RAW10;
        break;
    case NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12:
    case NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12RJ:
        paramsMAX96717F.ConfigVideoPipeline.dataType = CDI_MAX96717F_DATA_TYPE_RAW12;
        break;
    default:
        paramsMAX96717F.ConfigVideoPipeline.dataType = CDI_MAX96717F_DATA_TYPE_RAW8;
        break;
    }

    nvmStatus = MAX96717FWriteParameters(serCDI,
                                         CDI_WRITE_PARAM_CMD_MAX96717F_CONFIG_VIDEO_PIPELINES,
                                         sizeof(paramsMAX96717F.ConfigVideoPipeline),
                                         &paramsMAX96717F);
    if (NVMEDIA_STATUS_OK != nvmStatus) {
        SIPL_LOG_ERR_STR_HEX_UINT(
            "CDI_WRITE_PARAM_CMD_MAX96717F_CONFIG_VIDEO_PIPELINES failed with nvmediaErr",
            static_cast<uint32_t>(nvmStatus));
        status = ConvertNvMediaStatus(nvmStatus);
    } else {
        LOG_INFO("CDI_WRITE_PARAM_CMD_MAX96712_SINGLE_PIPELINE_MAPPING\n");

        if (MAX96712_MAX_NUM_LINK <= m_oLinkParams.ulinkIndex) {
            status = NVSIPL_STATUS_ERROR;
            SIPL_LOG_ERR_STR(
                "CDI_WRITE_PARAM_CMD_MAX96717F_CONFIG_VIDEO_PIPELINES ulinkIndex out of bounds");
        } else {
            LinkPipelineMapMAX96712 *pipeLineMap {NULL};
            WriteParametersParamMAX96712 paramsMAX96712 {};
            pipeLineMap = &paramsMAX96712.PipelineMapping.linkPipelineMap[m_oLinkParams.ulinkIndex];
            pipeLineMap->isEmbDataType = m_oLinkParams.moduleConnectionProperty.
                sensorConnectionProperty.bEmbeddedDataType;
            pipeLineMap->vcID = m_oLinkParams.moduleConnectionProperty.
                sensorConnectionProperty.uVCID;
            pipeLineMap->isDTOverride = !m_oLinkParams.moduleConnectionProperty.
                sensorConnectionProperty.bEmbeddedDataType;
            pipeLineMap->isSinglePipeline  = true;
            pipeLineMap->isMapToUnusedCtrl = false;
            LinkMAX96712 const link {GetMAX96712Link(m_oLinkParams.ulinkIndex)};
            paramsMAX96712.PipelineMapping.link = link;

            switch (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.inputFormat.
                inputFormatType) {
            case NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW10:
                pipeLineMap->dataType = CDI_MAX96712_DATA_TYPE_RAW10;
                break;
            case NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12:
            case NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12RJ:
                pipeLineMap->dataType = CDI_MAX96712_DATA_TYPE_RAW12;
                break;
            default:
                pipeLineMap->dataType = CDI_MAX96712_DATA_TYPE_RAW8;
                break;
            }

            DevBlkCDIDevice const* const deserializerCDI {m_oLinkParams.pDeserCDIDevice};
            nvmStatus = MAX96712WriteParameters(deserializerCDI,
                                                CDI_WRITE_PARAM_CMD_MAX96712_PIPELINE_MAPPING,
                                                sizeof(paramsMAX96712.PipelineMapping),
                                                &paramsMAX96712);
            if (NVMEDIA_STATUS_OK != nvmStatus) {
                SIPL_LOG_ERR_STR_HEX_UINT(
                    "CDI_WRITE_PARAM_CMD_MAX96712_SINGLE_PIPELINE_MAPPING failed with nvmediaErr",
                    static_cast<uint32_t>(nvmStatus));
                status = ConvertNvMediaStatus(nvmStatus);
            }
        }
    }

    return status;
}

SIPLStatus CNvMTransportLink_Max96712_96717F::SetSERDESGpioForward(void) const
{
    WriteParametersParamMAX96712 paramsMAX96712 {};
    SIPLStatus status {NVSIPL_STATUS_OK};

    uint64_t gpioMapSize {static_cast<uint64_t>(
        m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.vGpioMap.size())};
    for (uint8_t i {0U}; i < gpioMapSize; i++) {
        ReadWriteParamsMAX96717F paramsMAX96717F {};
        LOG_INFO("CDI_WRITE_PARAM_CMD_MAX96717F_GPIO_FORWARD\n");
        DevBlkCDIDevice *const serCDI {m_oLinkParams.pSerCDIDevice};
        paramsMAX96717F.GPIOForward.srcGpio = m_oLinkParams.moduleConnectionProperty.
            sensorConnectionProperty.vGpioMap[i].sourceGpio;
        paramsMAX96717F.GPIOForward.dstGpio = m_oLinkParams.moduleConnectionProperty.
            sensorConnectionProperty.vGpioMap[i].destGpio;
        NvMediaStatus nvmStatus {MAX96717FWriteParameters(serCDI,
                                 CDI_WRITE_PARAM_CMD_MAX96717F_GPIO_FORWARD,
                                 sizeof(paramsMAX96717F.GPIOForward),
                                 &paramsMAX96717F)};
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT(
                "CDI_WRITE_PARAM_CMD_MAX96717F_GPIO_FORWARD failed with NvMedia error",
                static_cast<int32_t>(nvmStatus));
            status = ConvertNvMediaStatus(nvmStatus);
        } else {
            LOG_INFO("CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_GPIO_RX\n");
            LinkMAX96712 const link {GetMAX96712Link(m_oLinkParams.ulinkIndex)};
            paramsMAX96712.link = link;
            paramsMAX96712.gpioIndex = m_oLinkParams.moduleConnectionProperty.
                sensorConnectionProperty.vGpioMap[i].destGpio;
            DevBlkCDIDevice const* const deserializerCDI {m_oLinkParams.pDeserCDIDevice};
            nvmStatus = MAX96712WriteParameters(deserializerCDI,
                                                CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_GPIO_RX,
                                                sizeof(paramsMAX96712.gpioIndex),
                                                &paramsMAX96712);
            if (NVMEDIA_STATUS_OK != nvmStatus) {
                SIPL_LOG_ERR_STR_INT(
                    "CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_GPIO_RX failed with NvMedia error",
                    static_cast<int32_t>(nvmStatus));
                status = ConvertNvMediaStatus(nvmStatus);
            }
        }
        if (NVMEDIA_STATUS_OK != nvmStatus) {
            break;
        }
    }

    return status;
}

SIPLStatus CNvMTransportLink_Max96712_96717F::ReleaseResetAndErrReporting(void) const
{
    NvMediaStatus nvmStatus {NVMEDIA_STATUS_OK};
    SIPLStatus status {NVSIPL_STATUS_OK};

    CNvMCameraModuleCommon::ConnectionProperty::SensorConnectionProperty::SensorReset const
        sensorReset = m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.sensorReset;
    if (sensorReset.isNeeded) {
        ReadWriteParamsMAX96717F paramsMAX96717F {};
        LOG_INFO("Hold sensor reset\n");
        paramsMAX96717F.GPIOOutp.gpioInd = getMAX96717FGPIO(sensorReset.pinNum);
        paramsMAX96717F.GPIOOutp.level = !sensorReset.releaseResetLevel;
        DevBlkCDIDevice const* const serCDI {m_oLinkParams.pSerCDIDevice};
        nvmStatus = MAX96717FWriteParameters(serCDI,
                                             CDI_WRITE_PARAM_CMD_MAX96717F_SET_GPIO_OUTPUT,
                                             sizeof(paramsMAX96717F.GPIOOutp),
                                             &paramsMAX96717F);
        if (nvmStatus == NVMEDIA_STATUS_OK) {
            LOG_INFO("Release sensor reset\n");
            paramsMAX96717F.GPIOOutp.gpioInd = getMAX96717FGPIO(sensorReset.pinNum);
            paramsMAX96717F.GPIOOutp.level = sensorReset.releaseResetLevel;
            nvmStatus = MAX96717FWriteParameters(serCDI,
                                                 CDI_WRITE_PARAM_CMD_MAX96717F_SET_GPIO_OUTPUT,
                                                 sizeof(paramsMAX96717F.GPIOOutp),
                                                 &paramsMAX96717F);
            if (nvmStatus != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_HEX_UINT(
                    "CDI_WRITE_PARAM_CMD_MAX96717F_SET_GPIO_OUTPUT failed with nvmediaErr",
                    static_cast<uint32_t>(nvmStatus));
                status = ConvertNvMediaStatus(nvmStatus);
            }
        } else {
            SIPL_LOG_ERR_STR_HEX_UINT(
                "CDI_WRITE_PARAM_CMD_MAX96717F_SET_GPIO_OUTPUT failed with nvmediaErr",
                static_cast<uint32_t>(nvmStatus));
            status = ConvertNvMediaStatus(nvmStatus);
        }
    }

    if (NVMEDIA_STATUS_OK == nvmStatus) {
        ReadWriteParamsMAX96717F const paramsMAX96717F {};
        DevBlkCDIDevice *const serCDI {m_oLinkParams.pSerCDIDevice};
        WriteParametersCmdMAX96717F param {CDI_WRITE_PARAM_CMD_MAX96717F_ENABLE_SM_REPORTS};
        CNvMCameraModuleCommon::ConnectionProperty::SensorConnectionProperty::RefClock const
            refClock {m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.refClock};
        if (!refClock.isNeeded) {
            // Skip reference generator DLL lock check for sensors that don't use reference clock
            param = CDI_WRITE_PARAM_CMD_MAX96717F_ENABLE_SM_REPORTS_SKIP_REFGEN_CHECK;
        }
        nvmStatus = MAX96717FWriteParameters(serCDI,
                                             param,
                                             sizeof(paramsMAX96717F),
                                             &paramsMAX96717F);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_HEX_UINT("Failed to enabled SM reporting with nvmediaErr",
                                      static_cast<uint32_t>(nvmStatus));
            status = ConvertNvMediaStatus(nvmStatus);
        }
    }

    return status;
}

SIPLStatus CNvMTransportLink_Max96712_96717F::Init_Errb(SIPLStatus const instatus) const
{
    SIPLStatus status {instatus};

    if (NVSIPL_STATUS_OK == status) {
        NvMediaStatus nvmStatus {NVMEDIA_STATUS_OK};
        DevBlkCDIDevice const * const serCDI {m_oLinkParams.pSerCDIDevice};
        ReadWriteParamsMAX96717F paramsMAX96717F {};
        LOG_INFO("Configure serializer to unset ERRB_TX_EN\n");
        paramsMAX96717F.GPIOErrb.dstGpio = 0U;

        nvmStatus = MAX96717FWriteParameters(serCDI,
                                            CDI_WRITE_PARAM_CMD_MAX96717F_UNSET_ERRB_TX,
                                            sizeof(paramsMAX96717F.GPIOErrb),
                                            &paramsMAX96717F);
        status = ConvertNvMediaStatus(nvmStatus);
        if (status == NVSIPL_STATUS_OK) {
            LOG_INFO("Configure deserializer to unset ERRB_RX_EN\n");
            WriteParametersParamMAX96712 paramsMAX96712 {};
            LinkMAX96712 const link {GetMAX96712Link(m_oLinkParams.ulinkIndex)};
            DevBlkCDIDevice const * const deserializerCDI {m_oLinkParams.pDeserCDIDevice};
            paramsMAX96712.GpioErrbSetting.link = link;
            nvmStatus = MAX96712WriteParameters(deserializerCDI,
                                                CDI_WRITE_PARAM_CMD_MAX96712_UNSET_ERRB_RX,
                                                sizeof(paramsMAX96712.GpioErrbSetting),
                                                &paramsMAX96712);

            status = ConvertNvMediaStatus(nvmStatus);
        }

        /* SER ERRB Assert */
        if (NVSIPL_STATUS_OK == status) {
            paramsMAX96717F = {};
            paramsMAX96717F.assertERRB = true;

            nvmStatus = MAX96717FWriteParameters(serCDI,
                                                CDI_WRITE_PARAM_CMD_MAX96717F_ASSERT_ERRB,
                                                sizeof(bool),
                                                &paramsMAX96717F);
            if (NVMEDIA_STATUS_OK != nvmStatus) {
                SIPL_LOG_ERR_STR_INT("CDI_WRITE_PARAM_CMD_MAX96717_ASSERT_ERRB failed with:",
                                    static_cast<int32_t>(nvmStatus));
            }
            status = ConvertNvMediaStatus(nvmStatus);
        }

        /* SER ERRB De-Assert */
        if (NVSIPL_STATUS_OK == status) {
            paramsMAX96717F = {};
            paramsMAX96717F.assertERRB = false;

            nvmStatus = MAX96717FWriteParameters(serCDI,
                                                CDI_WRITE_PARAM_CMD_MAX96717F_ASSERT_ERRB,
                                                sizeof(bool),
                                                &paramsMAX96717F);
            if (NVMEDIA_STATUS_OK != nvmStatus) {
                SIPL_LOG_ERR_STR_INT("CDI_WRITE_PARAM_CMD_MAX96717_ASSERT_ERRB failed with:",
                                    static_cast<int32_t>(nvmStatus));
            }
            status = ConvertNvMediaStatus(nvmStatus);
        }
    }

    return status;
}

#if FUSA_CDD_NV
/* Execute verification steps as per MAX96712 SM Implementation Guide for SM10 TC_1 */
SIPLStatus CNvMTransportLink_Max96712_96717F::ERRGDiagnosticTest(void) const
{
    SIPLStatus status {NVSIPL_STATUS_OK};
    NvMediaStatus nvmStatus {NVMEDIA_STATUS_OK};
    DevBlkCDIDevice const * const serCDI {m_oLinkParams.pSerCDIDevice};
    DevBlkCDIDevice const * const deserCDI {m_oLinkParams.pDeserCDIDevice};

    /* 1. Disable DEC_ERR to ERRB.
     * Not explicitly disabling IDLE_ERR reporting at ERRB since it's disabled by default */
    nvmStatus = MAX96712EnableDECErrToERRB(deserCDI, false);
    if (NVMEDIA_STATUS_OK != nvmStatus) {
        SIPL_LOG_ERR_STR_INT("Failed to disable DEC_ERR to ERRB at MAX96712: ",
                                    static_cast<int32_t>(nvmStatus));
        goto end;
    }

    /* 2. Enable ERRG at SER MAX96717/MAX96717F */
    nvmStatus = MAX96717FERRG(serCDI, true);
    if (NVMEDIA_STATUS_OK != nvmStatus) {
        SIPL_LOG_ERR_STR_INT("Failed to enable ERRG at MAX96717/MAX96717F: ",
                                    static_cast<int32_t>(nvmStatus));
        goto end;
    }

    nvsleep(ERRG_DIAGNOSTICTEST_SLEEP_US);

    /* 3. Disable ERRG at SER MAX96717/MAX96717F */
    nvmStatus = MAX96717FERRG(serCDI, false);
    if (NVMEDIA_STATUS_OK != nvmStatus) {
        SIPL_LOG_ERR_STR_INT("Failed to disable ERRG at MAX96717/MAX96717F: ",
                                    static_cast<int32_t>(nvmStatus));
        goto end;
    }

    /* 4. Verify DEC_ERR and IDLE_ERR and clear them in DES MAX96712 */
    nvmStatus = MAX96712VerifyERRGDiagnosticErrors(deserCDI);
    if (NVMEDIA_STATUS_OK != nvmStatus) {
        SIPL_LOG_ERR_STR_INT("Failed to verify ERRG diagnostic error at MAX96712: ",
                                    static_cast<int32_t>(nvmStatus));
        goto end;
    }

    /* 5. Enable the DEC_ERR to ERRB in MAX96712 */
    nvmStatus = MAX96712EnableDECErrToERRB(deserCDI, true);
    if (NVMEDIA_STATUS_OK != nvmStatus) {
        SIPL_LOG_ERR_STR_INT("Failed to disable DEC_ERR to ERRB at MAX96712: ",
                                    static_cast<int32_t>(nvmStatus));
    }

    if (NVMEDIA_STATUS_OK == nvmStatus) {
        LOG_INFO("MAX96712: SM10: ERRGDiagnosticTest is successfully executed\n");
    }

end:
    status = ConvertNvMediaStatus(nvmStatus);
    return status;
}
#endif

// This will do the init for 1 maxim deser and up to MAX_LINKS_PER_DESER maxim serializers.
SIPLStatus CNvMTransportLink_Max96712_96717F::Init(DevBlkCDIDevice *const brdcstSerCDI,
                                                   uint8_t const linkMask,
                                                   bool const groupInitProg)
{
    SIPLStatus status {NVSIPL_STATUS_OK};
    LOG_INFO("Initializing link %u\n", m_oLinkParams.ulinkIndex);

    /* not used parameter */
    (void)linkMask;
    (void)groupInitProg;

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    DumpLinkParams();
#endif

    if ((m_oLinkParams.bPassive or m_oLinkParams.bEnableSimulator) or m_oLinkParams.bCBAEnabled) {
        return status;
    }

    status = BasicCheckModule(brdcstSerCDI);
    if (NVSIPL_STATUS_OK == status) {
        status = SetEEPROMWriteProtect();
        if (NVSIPL_STATUS_OK == status) {
            status = SetSERClk();
            if (NVSIPL_STATUS_OK == status) {
                status = SetSERDESGpioForward();
                if (NVSIPL_STATUS_OK == status) {
                    status = SetSERPhy();
                    if (NVSIPL_STATUS_OK == status) {
                        status = SetSERVideo();

#if defined(NVMEDIA_QNX) && FUSA_CDD_NV
                        if (initDone == false) {
                            if (NVSIPL_STATUS_OK == status) {
                                status = VPRBSDiagnosticTest();
                            }
                            if (NVSIPL_STATUS_OK == status) {
                                status = ERRGDiagnosticTest();
                            }
                        }
#endif
                        if (NVSIPL_STATUS_OK == status) {
                            status = SetSERFsyncLevel();
                            if (NVSIPL_STATUS_OK == status) {
                                status = Init_Errb(status);
                                if (NVSIPL_STATUS_OK == status) {
                                    status = ReleaseResetAndErrReporting();
                                    if (NVSIPL_STATUS_OK == status) {
                                        status = SetSERFsync();
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (NVSIPL_STATUS_OK != status) {
         SIPL_LOG_ERR_STR("Initialize failed");
    }

    initDone = true;

    return status;
}

#if FUSA_CDD_NV
/* Execute verification steps as per GMSL2 User Guide for SM1_TC1 of MAX96712 Deserializer */
SIPLStatus CNvMTransportLink_Max96712_96717F::VPRBSDiagnosticTest(void) const {

    SIPLStatus status {NVSIPL_STATUS_OK};
    NvMediaStatus nvmStatus {NVMEDIA_STATUS_OK};
    DevBlkCDIDevice const * const desCDI {m_oLinkParams.pDeserCDIDevice};
    DevBlkCDIDevice const * const serCDI {m_oLinkParams.pSerCDIDevice};

    /* 1. Disable auto bpp on Serializer MAX96717/MAX96717F,
     * 2. Enable Serializer internal clock PCLK generation and
     * 3. Enable Serializer pattern generator.
     * */
    nvmStatus = MAX96717FConfigPatternGenerator(serCDI, true);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("vPRBSDiagnosticTest: Failed to config and enable for Pattern Generator");
        goto end;
    }

    nvsleep(VPRBS_SLEEP_US);

    /* 4. Enable Pattern Checker in DES MAX96712 */
    nvmStatus = MAX96712PRBSChecker(desCDI, true);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("vPRBSDiagnosticTest: Failed to enable checker");
        goto end;
    }

    nvsleep(VPRBS_SLEEP_US);

    /* 5. Enable Serializer VPRBS pattern generator */
    nvmStatus = MAX96717FVPRBSGenerator(serCDI, true);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("vPRBSDiagnosticTest: Failed to enable VPRBS pattern generator");
        goto end;
    }

    /* 6. Verify Serializer has PCLKDET = 1. */
    nvmStatus = MAX96717FVerifyPCLKDET(serCDI);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("vPRBSDiagnosticTest: Failed to verify PCLKDET");
        goto end;
    }

    /* 7. Verify Deserializer has VIDEO_LOCK and VPRBS_ERR */
    nvmStatus = MAX96712VerifyVPRBSErrs(desCDI);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("vPRBSDiagnosticTest: Failed to verify VPRBS_ERR and VIDEO_LOCK");
        goto end;
    }

    /* 8. Disable Pattern checker in DES MAX96712 */
    nvmStatus = MAX96712PRBSChecker(desCDI, false);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("vPRBSDiagnosticTest: Failed to disable checker");
        goto end;
    }

    /* 9. Disable VPRBS pattern generator */
    nvmStatus = MAX96717FVPRBSGenerator(serCDI, false);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("vPRBSDiagnosticTest: Failed to disable VPRBS pattern generator");
        goto end;
    }

    /* 10. Enable Autobpp and Disable the internal PCLK */
    nvmStatus = MAX96717FConfigPatternGenerator(serCDI, false);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("vPRBSDiagnosticTest: Failed Enable Autobpp and disable internal PCLK");
        goto end;
    }

    if (NVMEDIA_STATUS_OK == nvmStatus) {
        LOG_INFO("MAX96712: SM1: vPRBSDiagnosticTest is successfully executed\n");
    }

end:
    status = ConvertNvMediaStatus(nvmStatus);
    return status;
}
#endif

SIPLStatus CNvMTransportLink_Max96712_96717F::PostSensorInit(uint8_t const linkMask,
                                                             bool const groupInitProg) const
{
    ReadParametersParamMAX96712 readParamsMAX96712 {};
    DevBlkCDIDevice *const deserializerCDI {m_oLinkParams.pDeserCDIDevice};
    NvMediaStatus nvmStatus {NVMEDIA_STATUS_OK};
    LOG_INFO("Read revision\n");
    nvmStatus = MAX96712ReadParameters(deserializerCDI,
                                       CDI_READ_PARAM_CMD_MAX96712_REV_ID,
                                       sizeof(RevisionMAX96712),
                                       &readParamsMAX96712.revision);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_HEX_UINT("CDI_READ_PARAM_CMD_MAX96712_REV_ID failed with nvmediaErr",
                                   static_cast<uint32_t>(nvmStatus));
    } else if ((m_oLinkParams.bPassive || m_oLinkParams.bEnableSimulator)
        || m_oLinkParams.bCBAEnabled) {
        nvmStatus = NVMEDIA_STATUS_OK;
    } else {

        /* Enable FRSYNC */
        if (m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bEnableTriggerModeSync
            && m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.bPostSensorInitFsync)
        {
            WriteParametersParamMAX96712 paramsMAX96712 {};
            LinkMAX96712 const link {GetMAX96712Link(m_oLinkParams.ulinkIndex)};
            RevisionMAX96712 revision {CDI_MAX96712_REV_INVALID};
            revision = readParamsMAX96712.revision;
            LOG_INFO("MAX96712 Revision %d", revision);
            LOG_INFO("Set Fsync\n");

            if ((revision == CDI_MAX96712_REV_1) or (revision == CDI_MAX96712_REV_2)) {
                paramsMAX96712.FSyncSettings.FSyncMode = (m_oLinkParams.moduleConnectionProperty.
                    sensorConnectionProperty.bEnableInternalSync) ?
                        CDI_MAX96712_FSYNC_AUTO : CDI_MAX96712_FSYNC_EXTERNAL;
            } else {
                paramsMAX96712.FSyncSettings.FSyncMode = (m_oLinkParams.moduleConnectionProperty.
                    sensorConnectionProperty.bEnableInternalSync) ?
                        CDI_MAX96712_FSYNC_OSC_MANUAL : CDI_MAX96712_FSYNC_EXTERNAL;
            }
            paramsMAX96712.FSyncSettings.fps = toUint32FromFloat(
                m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.fFrameRate);
            paramsMAX96712.FSyncSettings.link = (groupInitProg) ? LinkMAX96712(linkMask) : link;
            LOG_INFO("Set FSYNC mode\n");
            nvmStatus = MAX96712WriteParameters(deserializerCDI,
                                                CDI_WRITE_PARAM_CMD_MAX96712_SET_FSYNC,
                                                sizeof(paramsMAX96712.FSyncSettings),
                                                &paramsMAX96712);
            if (nvmStatus != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_HEX_UINT(
                    "CDI_WRITE_PARAM_CMD_MAX96712_FSYNC_MODE failed with nvmediaErr",
                    static_cast<uint32_t>(nvmStatus));
                /* return is expected in this failure case */
            }
        }
    }

    return ConvertNvMediaStatus(nvmStatus);
}

SIPLStatus CNvMTransportLink_Max96712_96717F::MiscInit() const
{
    SIPLStatus status {NVSIPL_STATUS_OK};
    using SerializerErrbErrorReportInfo = CNvMCameraModuleCommon::ConnectionProperty::
        SensorConnectionProperty::SerializerErrbErrorReport;

    SerializerErrbErrorReportInfo const serErrbErrorReport
        {m_oLinkParams.moduleConnectionProperty.sensorConnectionProperty.serializerErrbErrorReport};

    if ((m_oLinkParams.bPassive || m_oLinkParams.bEnableSimulator) || m_oLinkParams.bCBAEnabled) {
        status = NVSIPL_STATUS_OK;
    } else {
        if (serErrbErrorReport.isNeeded == true) {
            LOG_INFO("Configure serializer to set ERRB_TX_EN\n");
            ReadWriteParamsMAX96717F paramsMAX96717F {};

            paramsMAX96717F.GPIOErrb.dstGpio = {0U};
            NvMediaStatus nvmStatus {NVMEDIA_STATUS_OK};
            DevBlkCDIDevice const * const serCDI {m_oLinkParams.pSerCDIDevice};
            nvmStatus = MAX96717FWriteParameters(serCDI,
                                                CDI_WRITE_PARAM_CMD_MAX96717F_SET_ERRB_TX,
                                                sizeof(paramsMAX96717F.GPIOErrb),
                                                &paramsMAX96717F);
            status = ConvertNvMediaStatus(nvmStatus);
            if (status == NVSIPL_STATUS_OK) {
                LOG_INFO("Configure deserializer to set ERRB_RX_EN\n");
                WriteParametersParamMAX96712 paramsMAX96712 {};
                LinkMAX96712 const link {GetMAX96712Link(m_oLinkParams.ulinkIndex)};
                DevBlkCDIDevice const * const deserializerCDI {m_oLinkParams.pDeserCDIDevice};
                paramsMAX96712.GpioErrbSetting.link = link;
                nvmStatus = MAX96712WriteParameters(deserializerCDI,
                                                CDI_WRITE_PARAM_CMD_MAX96712_SET_ERRB_RX,
                                                sizeof(paramsMAX96712.GpioErrbSetting),
                                                &paramsMAX96712);

                status = ConvertNvMediaStatus(nvmStatus);
            }
        }

        std::vector<uint16_t> serCRCSkipRegList {m_oLinkParams.
                        moduleConnectionProperty.sensorConnectionProperty.
                        serializerCRCSkipRegisterList};

        /* Enable configuration CRC for first MiscInit.
         * Note: Do not enable the Register CRC SM for MAX96717(F) if the
         * skip register list is empty */
        if ((status == NVSIPL_STATUS_OK) && (!serCRCSkipRegList.empty() &&
                (m_bSerRegCrcEnabled == false))) {
            LOG_INFO("MAX96717F: SM27: Enable Configuration Register CRC\n");
            NvMediaStatus nvmStatus {NVMEDIA_STATUS_OK};

            /* Add registers to skip list to avoid CRC calculation of
             * configuration registers */
            DevBlkCDIDevice const * const serCDI {m_oLinkParams.pSerCDIDevice};

            /* Refer to Table 2 of GMSL2/3 Register CRC App Note
             * for the list of Skip registers.
             * Note: Maximum we can add 8 skip registers
             */
            for (uint16_t i {0U}; i < 8U; i++) {
                /* Byte conversion from 16-bit register adress to LSB and MSB */
                std::vector<uint8_t> data {
                    static_cast<uint8_t>(serCRCSkipRegList[i] & 0xFFU),
                    static_cast<uint8_t>((serCRCSkipRegList[i] >> 8U) & 0xFFU)
                };
                nvmStatus = MAX96717FWriteRegistersVerify(
                    serCDI, REG_REGCRC8 + (i * 2U), 2U, &data[0]);
                if (nvmStatus != NVMEDIA_STATUS_OK) {
                    SIPL_LOG_ERR_STR_UINT("MAX96717F: Failed to write SKIP "
                        "register list: index: ", static_cast<uint32_t>(i));
                    break;
                }
            }

            if (nvmStatus == NVMEDIA_STATUS_OK ) {
                nvmStatus = MAX96717FSetRegCRC(serCDI);
                if (nvmStatus != NVMEDIA_STATUS_OK) {
                    SIPL_LOG_ERR_STR("MAX96717F: Failed to set config for register CRC");
                } else {
                    m_bSerRegCrcEnabled = true;
                }
            }

            status = ConvertNvMediaStatus(nvmStatus);
        }
    }

    return status;
}

} // end of namespace nvsipl
