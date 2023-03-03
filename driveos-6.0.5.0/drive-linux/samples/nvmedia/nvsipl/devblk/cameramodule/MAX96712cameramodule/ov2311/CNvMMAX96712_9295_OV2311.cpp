/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
 * Copyright (c) 2021-2022, OmniVision Technologies.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "CNvMMAX96712_9295_OV2311.hpp"
#include "cameramodule/MAX96712cameramodule/CNvMTransportLink_Max96712_9295.hpp"
#include "ModuleIF/CNvMCameraModuleExport.hpp"

#include <chrono>
#include <thread>

extern "C" {
#include "cdi_ov2311.h"
#include "cdi_max96712.h"
}

#define OSC_MHZ 24

namespace nvsipl
{

void CNvMMAX96712_9295_OV2311::SetSensorConnectionProperty(CNvMCameraModuleCommon::ConnectionProperty::SensorConnectionProperty *sensorConnectionProperty) {
    auto& sensor = m_upSensor;

    // These setting depends on HW configuration.
    sensorConnectionProperty->sensorReset.isNeeded = true;
    sensorConnectionProperty->sensorReset.pinNum = 0;
    sensorConnectionProperty->sensorReset.releaseResetLevel = true;

    sensorConnectionProperty->frameSync.pinNum = 8;
    if (sensor->GetFrameRate() == 30) {
        sensorConnectionProperty->bEnableInternalSync = false;
    } else if (sensor->GetFrameRate() == 60) {
        sensorConnectionProperty->bEnableInternalSync = true;
    }

    sensorConnectionProperty->refClock.isNeeded = false;
    sensorConnectionProperty->refClock.pinNum = 0;

    sensorConnectionProperty->phyLanes.isLaneSwapNeeded = false;
    sensorConnectionProperty->phyLanes.isTwoLane = true;

    sensorConnectionProperty->pclk = 100000000;
}

SIPLStatus CNvMMAX96712_9295_OV2311::SetConfigModule(const SensorInfo *sensorInfo, CNvMDevice::DeviceParams *params) {
    LOG_INFO("CNvMSensor::SetConfigModule\n");

    auto& sensor = m_upSensor;

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

SIPLStatus CNvMMAX96712_9295_OV2311::DetectModule() {
    SIPLStatus status = NVSIPL_STATUS_OK;
    NvMediaStatus nvmStatus = NVMEDIA_STATUS_OK;
    auto& sensor = m_upSensor;

    LOG_INFO("CNvMSensor::DetectModule\n");
    if (m_oDeviceParams.bEnableSimulator or m_oDeviceParams.bPassive) {
        status = InitSimulatorAndPassive();
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("CNvMMAX96712_9295_OV2311::InitSimulatorAndPassive failed with SIPL error",
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

    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMMAX96712_9295_OV2311::InitModule() {
    SIPLStatus status = NVSIPL_STATUS_OK;
    NvMediaStatus nvmediaStatus = NVMEDIA_STATUS_OK;
    auto& sensor = m_upSensor;

    WriteReadParametersParamOV2311 wrPara;

    LOG_INFO("CNvMSensor::InitModule\n");
    if (m_oDeviceParams.bEnableSimulator or m_oDeviceParams.bPassive) {
        status = InitSimulatorAndPassive();
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("CNvMMAX96712_9295_OV2311::InitSimulatorAndPassive failed with SIPL error",
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
    wrPara.expogain.expo = 0.015f;
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

SIPLStatus CNvMMAX96712_9295_OV2311::StartModule() {
    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMMAX96712_9295_OV2311::StopModule() {
    return NVSIPL_STATUS_OK;
}

uint16_t CNvMMAX96712_9295_OV2311::GetPowerOffDelayMs(void) {
    return 50u; /* TODO: find a right delay */
}

SIPLStatus CNvMMAX96712_9295_OV2311::DeinitModule() {
    return NVSIPL_STATUS_OK;
}

DevBlkCDIDeviceDriver *CNvMMAX96712_9295_OV2311::GetCDIDeviceDriver() {
    return GetOV2311Driver();
}

std::unique_ptr<CNvMDevice::DriverContext> CNvMMAX96712_9295_OV2311::GetCDIDeviceContext() {
    auto driverContext = new CNvMDevice::DriverContextImpl<ContextOV2311>();
    if (driverContext == nullptr) {
        return nullptr;
    }

    return std::unique_ptr<CNvMDevice::DriverContext>(driverContext);
}

SIPLStatus CNvMMAX96712_9295_OV2311::InitSimulatorAndPassive()
{
    return NVSIPL_STATUS_OK;
}

Interface* CNvMMAX96712_9295_OV2311::GetInterface(
    const UUID &interfaceId)
{
    if (interfaceId == (CNvMMAX96712_9295_OV2311::getClassInterfaceID()))
        return static_cast<OV2311NonFuSaCustomInterface*>(this);

    return nullptr;
}

SIPLStatus CNvMMAX96712_9295_OV2311::SetCustomValue(
    std::uint32_t const valueToSet)
{
    SIPLStatus status = NVSIPL_STATUS_OK;
    NvMediaStatus mediaStatus = NVMEDIA_STATUS_OK;

    /*! Invoke driver function to set value */
    mediaStatus = OV2311SetDeviceValue(
        m_upSensor->GetCDIDeviceHandle(), valueToSet);
    if (mediaStatus != NVMEDIA_STATUS_OK) {
        LOG_ERR("Failed to set OV2311 value\n");
        status = ConvertNvMediaStatus(mediaStatus);
    }
    return status;
}

SIPLStatus CNvMMAX96712_9295_OV2311::GetCustomValue(
    std::uint32_t * const valueToGet)
{
    SIPLStatus status = NVSIPL_STATUS_OK;
    NvMediaStatus mediaStatus = NVMEDIA_STATUS_OK;

    /*! Invoke driver function to get value*/
    mediaStatus = OV2311GetDeviceValue(
        m_upSensor->GetCDIDeviceHandle(), valueToGet);
    if (mediaStatus != NVMEDIA_STATUS_OK) {
        LOG_ERR("Failed to get OV2311 value\n");
        status = ConvertNvMediaStatus(mediaStatus);
    }
    return status;
}

SIPLStatus CNvMMAX96712_9295_OV2311::CheckModuleStatus()
{
    SIPLStatus status = NVSIPL_STATUS_OK;
    NvMediaStatus mediaStatus = NVMEDIA_STATUS_OK;
    WriteParametersParamMAX96712 paramsMAX96712 {};

    status = DoSetPower(true);
    if (status != NVSIPL_STATUS_OK) {
        LOG_ERR("SetPower failed with SIPL error %d", status);
        return status;
    }

    LOG_INFO("Enable link\n");
    paramsMAX96712.link = (LinkMAX96712)(1 << m_linkIndex);
    mediaStatus = MAX96712WriteParameters(m_pDeserializer->GetCDIDeviceHandle(),
                                          CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_SPECIFIC_LINKS,
                                          sizeof(paramsMAX96712.link),
                                          &paramsMAX96712);
    if (mediaStatus != NVMEDIA_STATUS_OK) {
        LOG_ERR("CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_SPECIFIC_LINKS failed with NvMedia error(%d)", (int32_t)mediaStatus);
        return ConvertNvMediaStatus(mediaStatus);
    }

    mediaStatus = MAX96712CheckLink(m_pDeserializer->GetCDIDeviceHandle(),
                                    (LinkMAX96712)(1 << m_linkIndex),
                                    CDI_MAX96712_LINK_LOCK_GMSL2,
                                    false);
    if (mediaStatus != NVMEDIA_STATUS_OK) {
        LOG_ERR("Failed to lock the link\n");
        status = ConvertNvMediaStatus(mediaStatus);

        /* Turn the camera module power off */
        SIPLStatus statusPwr = DoSetPower(false);
        if (statusPwr != NVSIPL_STATUS_OK) {
            LOG_ERR("SetPower failed with SIPL error %d", statusPwr);
            return statusPwr;
        }
    }

    return status;
}

CNvMCameraModule *CNvMCameraModule_Create() {
    return new CNvMMAX96712_9295_OV2311();
}

const char** CNvMCameraModule_GetNames() {
    static const char* names[] = {
       "LI-OV2311-VCSEL-GMSL2-60H",
       "LI-OV2311-VCSEL-GMSL2-60H_L",
        NULL
    };
    return names;
}

} // end of namespace
