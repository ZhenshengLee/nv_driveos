/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) 2021-2023, OmniVision Technologies.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "CNvMMAX96712_96717F_OV2311.hpp"
#include "cameramodule/MAX96712cameramodule/CNvMTransportLink_Max96712_96717F.hpp"
#include "ModuleIF/CNvMCameraModuleExport.hpp"

#include <chrono>
#include <thread>

extern "C" {
#include "cdi_ov2311.h"
#include "cdi_m24c32.h"
#include "cdi_max96717f.h"
#include "cdi_max25614.h"
}

#define OSC_MHZ   24
#define IBOOST    0xFU
#define VBOOST    0x0U
#define TSLEW     0x0U
#define ILED      0x5FU
#define TON_MAX   0xDU
#define VLED_MAX  0x60U

namespace nvsipl
{

SIPLStatus CNvMMAX96712_96717F_OV2311::SetSensorConnectionProperty(
    CNvMCameraModuleCommon::ConnectionProperty::SensorConnectionProperty
        * const sensorConnectionProperty) const noexcept
{
    SIPLStatus status {NVSIPL_STATUS_OK};

    if(sensorConnectionProperty == nullptr) {
        SIPL_LOG_ERR_STR("sensorConnectionProperty is null");
        status = NVSIPL_STATUS_BAD_ARGUMENT;
    } else {
        // These setting depends on HW configuration.
        sensorConnectionProperty->sensorReset.isNeeded = true;
        sensorConnectionProperty->sensorReset.pinNum = 0U;
        sensorConnectionProperty->sensorReset.releaseResetLevel = true;

        const auto moduleName = GetModuleName();
        sensorConnectionProperty->frameSync.pinNum = 8U;
        if (moduleName == "I0OO23111CML1050NB10" ||
            moduleName == "LI-OV2311-VCSEL-GMSL2-55H") {
            sensorConnectionProperty->bEnableInternalSync = true;
        } else {
            sensorConnectionProperty->bEnableInternalSync = false;
        }

        sensorConnectionProperty->refClock.isNeeded = false;
        sensorConnectionProperty->refClock.pinNum = 0U;
        sensorConnectionProperty->refClock.sensorClock = 0U;

        sensorConnectionProperty->phyLanes.isLaneSwapNeeded = false;
        sensorConnectionProperty->phyLanes.isTwoLane = true;

        sensorConnectionProperty->pmicProperty.isSupported = false;
        sensorConnectionProperty->pmicProperty.i2cAddress = 0U;

        sensorConnectionProperty->vcselProperty.isSupported = true;
        sensorConnectionProperty->vcselProperty.i2cAddress = 0x4AU;
    }

    return status;
}

SIPLStatus CNvMMAX96712_96717F_OV2311::SetConfigModule(SensorInfo const* const sensorInfo, CNvMDevice::DeviceParams const* const params) {
    LOG_INFO("CNvMSensor::SetConfigModule\n");

    const CNvMSensor * const sensor = GetUpSensor();

    std::string resolution;
    NvMediaStatus nvmediaStatus = NVMEDIA_STATUS_OK;

    resolution = std::to_string(sensor->GetWidth()) + "x" +  std::to_string(sensor->GetHeight());
    nvmediaStatus = GetOV2311ConfigSet((char *)resolution.c_str(),
                                       sensorInfo->vcInfo.inputFormat,
                                       &m_ConfigIndex,
                                       (uint32_t)sensor->GetFrameRate());
    if (nvmediaStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("OV2311: GetOV2311ConfigSet failed with NvMedia error", (int32_t)nvmediaStatus);
        return ConvertNvMediaStatus(nvmediaStatus);
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMMAX96712_96717F_OV2311::SetupAddressTranslationsB(DevBlkCDIDevice const* const serCDI,
                                                                 uint8_t const src,
                                                                 uint8_t const dst) const
{
    NvMediaStatus mediaStatus = NVMEDIA_STATUS_OK;

    if (src != static_cast<uint8_t> UINT8_MAX) {
        ReadWriteParamsMAX96717F paramsMAX96717F = {};
        paramsMAX96717F.Translator.source = src;
        paramsMAX96717F.Translator.destination = dst;
        LOG_INFO("Translate device addr 0x%x to 0x%x\n",
                paramsMAX96717F.Translator.source, paramsMAX96717F.Translator.destination);
        mediaStatus = MAX96717FWriteParameters(serCDI,
                                               CDI_WRITE_PARAM_CMD_MAX96717F_SET_TRANSLATOR_B,
                                               sizeof(paramsMAX96717F.Translator),
                                               &paramsMAX96717F);
        if (mediaStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("CDI_WRITE_PARAM_CMD_MAX96717F_SET_TRANSLATOR_B failed with NvMedia error",
                                 static_cast<int32_t>(mediaStatus));
        }
    } else {
        mediaStatus = NVMEDIA_STATUS_BAD_PARAMETER;
    }

    return ConvertNvMediaStatus(mediaStatus);
}

SIPLStatus CNvMMAX96712_96717F_OV2311::SetTrans2VCSEL() {
    SIPLStatus    statusSipl  = NVSIPL_STATUS_OK;
    const CNvMSerializer *serializer = GetUpSerializer();
    const CNvMVCSEL * const vcsel      = GetUpVcsel();
    DevBlkCDIDevice const *const serCDI = serializer->GetCDIDeviceHandle();
    uint8_t const src = vcsel->GetI2CAddr();
    uint8_t const dst = vcsel->GetNativeI2CAddr();

    /*! Set serializer translator-B to VCSEL */
    statusSipl = SetupAddressTranslationsB(serCDI,
                                           src,
                                           dst);
    if (statusSipl != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR("Setting up address translation in serializer failed");
    }

    return statusSipl;
}

SIPLStatus CNvMMAX96712_96717F_OV2311::DetectModule() {

    MAX25614Params vcselParams = { .iBoost = IBOOST,
                                   .vBoost = VBOOST,
                                   .iLed = ILED,
                                   .tOnMax = TON_MAX,
                                   .tSlew = TSLEW,
                                   .vLedMax = VLED_MAX};

    SIPLStatus status = NVSIPL_STATUS_OK;
    NvMediaStatus nvmStatus = NVMEDIA_STATUS_OK;
    const CNvMSensor * const sensor = GetUpSensor();
    const CNvMVCSEL * const vcsel = GetUpVcsel();

    LOG_INFO("CNvMSensor::DetectModule\n");
    if (GetDeviceParams().bEnableSimulator or GetDeviceParams().bPassive) {
        status = InitSimulatorAndPassive();
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("CNvMMAX96712_96717F_OV2311::InitSimulatorAndPassive failed with SIPL error",
                    (int32_t)status);
        }
        return status;
    }

    /*! Check SENSOR is present */
    LOG_INFO("Check SENSOR is present\n");
    nvmStatus = OV2311CheckPresence(sensor->GetCDIDeviceHandle());
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("OV2311CheckPresence failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    /*! Use VCSEL for B2 module */
    if (GetModuleName() == "LI-OV2311-VCSEL-GMSL2-55H" ||
        GetModuleName() == "LI-OV2311-VCSEL-GMSL2-55H_L" ||
        GetModuleName() == "LI-OV2311-VCSEL-GMSL2-55H_E") {
        /*! Translate the I2C address for VCSEL */
        status = SetTrans2VCSEL();
        if (status != NVSIPL_STATUS_OK) {
            return status;
        }

        /*! Check VCSEL is present */
        LOG_INFO("Check VCSEL is present\n");
        nvmStatus = MAX25614CheckPresence(vcsel->GetCDIDeviceHandle());
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX25614CheckPresence failed with NvMedia error", (int32_t)nvmStatus);
            return ConvertNvMediaStatus(nvmStatus);
        }

        /*! Initialize VCSEL */
        LOG_INFO("Initialize VCSEL\n");
        nvmStatus = MAX25614Init(vcsel->GetCDIDeviceHandle(), vcselParams);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX25614Init failed with NvMedia error", (int32_t)nvmStatus);
            return ConvertNvMediaStatus(nvmStatus);
        }
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMMAX96712_96717F_OV2311::InitModule() const {
    SIPLStatus status = NVSIPL_STATUS_OK;
    NvMediaStatus nvmediaStatus = NVMEDIA_STATUS_OK;
    const CNvMSensor * const sensor = GetUpSensor();

    WriteReadParametersParamOV2311 wrPara;

    LOG_INFO("CNvMSensor::InitModule\n");
    if (GetDeviceParams().bEnableSimulator or GetDeviceParams().bPassive) {
        status = InitSimulatorAndPassive();
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("CNvMMAX96712_96717F_OV2311::InitSimulatorAndPassive failed with SIPL error",
                    (int32_t)status);
        }
        return status;
    }

    /* Software Reset */
    LOG_INFO("Sensor software reset\n");
    nvmediaStatus = OV2311SetDeviceConfig(sensor->GetCDIDeviceHandle(),
                           CDI_CONFIG_OV2311_SOFTWARE_RESET);
    if (nvmediaStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("CDI_CONFIG_OV2311_ENABLE_STREAMING failed with NvMedia error", (int32_t)nvmediaStatus);
        return ConvertNvMediaStatus(nvmediaStatus);
    }

    /*! Set defaults */
    LOG_INFO("Set defaults in %s\n", "CNvMSensor::Init");
    nvmediaStatus = OV2311SetDefaults(sensor->GetCDIDeviceHandle());
    if (nvmediaStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("OV2311SetDefaults failed with NvMedia error", (int32_t)nvmediaStatus);
        return ConvertNvMediaStatus(nvmediaStatus);
    }

    /*! Additional setting per frame rate */
    if (sensor->GetFrameRate() == 60U) {
        LOG_INFO("Set 60FPS setting in %s\n", "CNvMSensor::Init");
        nvmediaStatus = OV2311SetDeviceConfig(sensor->GetCDIDeviceHandle(), CDI_CONFIG_OV2311_STREAM_1600x1300_RAW10_60FPS);
        if (nvmediaStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("OV2311: CDI_CONFIG_OV2311_STREAM_1600x1300_RAW10_60FPS failed with NvMedia error", (int32_t)nvmediaStatus);
            return ConvertNvMediaStatus(nvmediaStatus);
        }
    } else if (sensor->GetFrameRate() == 30U) {
        LOG_INFO("Set 30FPS setting in %s\n", "CNvMSensor::Init");
        nvmediaStatus = OV2311SetDeviceConfig(sensor->GetCDIDeviceHandle(), CDI_CONFIG_OV2311_STREAM_1600x1300_RAW10_30FPS);
        if (nvmediaStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("OV2311: CDI_CONFIG_OV2311_STREAM_1600x1300_RAW10_30FPS failed with NvMedia error", (int32_t)nvmediaStatus);
            return ConvertNvMediaStatus(nvmediaStatus);
        }
    } else {
        SIPL_LOG_ERR_STR_INT("OV2311: Unsupported frame rate : %d", (int32_t)sensor->GetFrameRate());
        return NVSIPL_STATUS_NOT_SUPPORTED;
    }

    if (sensor->GetEnableTPG()) {
        /*! Enable sensor tpg */
        nvmediaStatus = OV2311SetDeviceConfig(sensor->GetCDIDeviceHandle(), CDI_CONFIG_OV2311_ENABLE_PG);
        if (nvmediaStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("CDI_CONFIG_OV2311_ENABLE_PG failed with NvMedia error", (int32_t)nvmediaStatus);
            return ConvertNvMediaStatus(nvmediaStatus);
        }
    }

    /* Enable top emb stats */
    LOG_INFO("Config top emb\n");
    nvmediaStatus = OV2311SetDeviceConfig(sensor->GetCDIDeviceHandle(), CDI_CONFIG_OV2311_EMBLINE_TOP);
    if (nvmediaStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("CDI_CONFIG_OV2311_ENABLE_STREAMING failed with NvMedia error", (int32_t)nvmediaStatus);
        return ConvertNvMediaStatus(nvmediaStatus);
    }

    /* Enable bottom emb stats */
    LOG_INFO("Config bottom emb\n");
    nvmediaStatus = OV2311SetDeviceConfig(sensor->GetCDIDeviceHandle(), CDI_CONFIG_OV2311_EMBLINE_BOTTOM);
    if (nvmediaStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("CDI_CONFIG_OV2311_ENABLE_STREAMING failed with NvMedia error", (int32_t)nvmediaStatus);
        return ConvertNvMediaStatus(nvmediaStatus);
    }

    /* Frame Rate */
    nvmediaStatus = OV2311SetDeviceConfig(sensor->GetCDIDeviceHandle(), m_ConfigIndex);
    if (nvmediaStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("CDI_CONFIG_OV2311_STREAM_(framerate) failed with NvMedia error", (int32_t)nvmediaStatus);
        return ConvertNvMediaStatus(nvmediaStatus);
    }

    /* Get Timing info before streaming */
    nvmediaStatus = OV2311SetDeviceConfig(sensor->GetCDIDeviceHandle(), CDI_CONFIG_OV2311_SETTINGINFO);
    if (nvmediaStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("CDI_CONFIG_OX03C10_SETTINGINFO failed with NvMedia error", (int32_t)nvmediaStatus);
        return ConvertNvMediaStatus(nvmediaStatus);
    }

    //Default Strobe Width
    wrPara.strobe_ms = 0.0005f;
    nvmediaStatus = OV2311WriteParameters(sensor->GetCDIDeviceHandle(), CDI_WRITE_PARAM_CMD_STROBE, sizeof(wrPara), &wrPara);
    if (nvmediaStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("CDI_WRITE_PARAM_CMD_EXPOSURE failed with NvMedia error", (int32_t)nvmediaStatus);
        return ConvertNvMediaStatus(nvmediaStatus);
    }

    //Default Exposure
    wrPara.expogain.expo = 0.0005f;
    wrPara.expogain.gain = 1.0f;
    nvmediaStatus = OV2311WriteParameters(sensor->GetCDIDeviceHandle(), CDI_WRITE_PARAM_CMD_EXPOSURE, sizeof(wrPara), &wrPara);
    if (nvmediaStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("CDI_WRITE_PARAM_CMD_EXPOSURE failed with NvMedia error", (int32_t)nvmediaStatus);
        return ConvertNvMediaStatus(nvmediaStatus);
    }

    /*! Enable sensor streaming */
    nvmediaStatus = OV2311SetDeviceConfig(sensor->GetCDIDeviceHandle(), CDI_CONFIG_OV2311_ENABLE_STREAMING);
    if (nvmediaStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("CDI_CONFIG_OV2311_ENABLE_STREAMING failed with NvMedia error", (int32_t)nvmediaStatus);
        return ConvertNvMediaStatus(nvmediaStatus);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    /* Get FuseID */
    nvmediaStatus = OV2311SetDeviceConfig(sensor->GetCDIDeviceHandle(), CDI_CONFIG_OV2311_FUSEID);
    if (nvmediaStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("CDI_CONFIG_OX03C10_FUSEID failed with NvMedia error", (int32_t)nvmediaStatus);
        return ConvertNvMediaStatus(nvmediaStatus);
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMMAX96712_96717F_OV2311::StartModule() {
    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMMAX96712_96717F_OV2311::PostInitModule() {
    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMMAX96712_96717F_OV2311::StopModule() {
    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMMAX96712_96717F_OV2311::DeinitModule() {
    return NVSIPL_STATUS_OK;
}

DevBlkCDIDeviceDriver *CNvMMAX96712_96717F_OV2311::GetCDIDeviceDriver() const {
    return GetOV2311Driver();
}

DevBlkCDIDeviceDriver *CNvMMAX96712_96717F_OV2311::GetCDIDeviceDriver(ModuleComponent const component) const {
    DevBlkCDIDeviceDriver *status = NULL;
    switch (component) {
        case ModuleComponent::MODULE_COMPONENT_SENSOR:
            status = GetOV2311Driver();
            break;
        case ModuleComponent::MODULE_COMPONENT_EEPROM:
            status = GetM24C32Driver();
            break;
        case ModuleComponent::MODULE_COMPONENT_VCSEL:
            status = GetMAX25614Driver();
            break;
        default:
            status = NULL;
            break;
    }
    return status;
}

std::unique_ptr<CNvMDevice::DriverContext> CNvMMAX96712_96717F_OV2311::GetCDIDeviceContext() const {
    auto driverContext = new CNvMDevice::DriverContextImpl<ContextOV2311>();
    if (driverContext == nullptr) {
        return nullptr;
    }

    return std::unique_ptr<CNvMDevice::DriverContext>(driverContext);
}

SIPLStatus CNvMMAX96712_96717F_OV2311::InitSimulatorAndPassive() const
{
    return NVSIPL_STATUS_OK;
}

Interface* CNvMMAX96712_96717F_OV2311::GetInterface(
    const UUID &interfaceId)
{
    if (interfaceId == (CNvMMAX96712_96717F_OV2311::getClassInterfaceID()))
        return static_cast<OV2311_MAX96717F_CustomInterface*>(this);

    return nullptr;
}

SIPLStatus CNvMMAX96712_96717F_OV2311::SetCustomValue(
    std::uint32_t const valueToSet)
{
    SIPLStatus status = NVSIPL_STATUS_OK;
    NvMediaStatus mediaStatus = NVMEDIA_STATUS_OK;

    /*! Invoke driver function to set value */
    mediaStatus = OV2311SetDeviceValue(
        GetUpSensor()->GetCDIDeviceHandle(), valueToSet);
    if (mediaStatus != NVMEDIA_STATUS_OK) {
        LOG_ERR("Failed to set OV2311 value\n");
        status = ConvertNvMediaStatus(mediaStatus);
    }
    return status;
}

SIPLStatus CNvMMAX96712_96717F_OV2311::GetCustomValue(
    std::uint32_t * const valueToGet)
{
    SIPLStatus status = NVSIPL_STATUS_OK;
    NvMediaStatus mediaStatus = NVMEDIA_STATUS_OK;

    /*! Invoke driver function to get value*/
    mediaStatus = OV2311GetDeviceValue(
        GetUpSensor()->GetCDIDeviceHandle(), valueToGet);
    if (mediaStatus != NVMEDIA_STATUS_OK) {
        LOG_ERR("Failed to get OV2311 value\n");
        status = ConvertNvMediaStatus(mediaStatus);
    }
    return status;
}

uint16_t CNvMMAX96712_96717F_OV2311::GetPowerOnDelayMs()
{
    return POWER_ON_DELAY_MS;
}

uint16_t CNvMMAX96712_96717F_OV2311::GetPowerOffDelayMs(void) noexcept
{
    return POWER_OFF_DELAY_MS;
}

CNvMCameraModule *CNvMCameraModule_Create() {
    return new CNvMMAX96712_96717F_OV2311();
}

const char** CNvMCameraModule_GetNames() {
    static const char* names[] = {
        "I0OO23111CML1050NB10",
        "I0OO23111CML1050NB10_30FPS",
        "LI-OV2311-VCSEL-GMSL2-55H",
        "LI-OV2311-VCSEL-GMSL2-55H_L",
        "I0OO23111CML1050NB10_E",
        "LI-OV2311-VCSEL-GMSL2-55H_E",
        NULL
    };
    return names;
}

} // end of namespace
