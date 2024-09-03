/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "MAX96712cameramodule/CNvMMAX96712CameraModule.hpp"
#include "devices/MAX96712DeserializerDriver/CNvMMax96712.hpp"
#include "devices/MAX20087Driver/CNvMMax20087.hpp"
#include "devices/MAX20087Driver/CNvMMax20087Factory.hpp"
#include "devices/MAX20087Driver/CNvMMax20087SyncAdapter.hpp"
#include "ModuleIF/CNvMCameraModuleExport.hpp"
#include "sipl_error.h"
#include "sipl_util.h"

#include <chrono>
#include <thread>

extern "C" {
    #include "cdi_debug_log.h"
    #include "pwr_utils.h"
}

namespace nvsipl
{

SIPLStatus CNvMMAX96712CameraModule::EnableLinkAndDetect()
{
    SIPLStatus status;
    std::unique_ptr<CNvMSerializer> broadcastSerializer = std::move(CreateBroadcastSerializer(m_linkIndex));

    // Use the broadcast serializer to initialize all transport links
    status = m_upTransport->Init(broadcastSerializer->GetCDIDeviceHandle(), m_initLinkMask, m_groupInitProg);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: TransportLink Init failed with SIPL error", (int32_t)status);
        return status;
    }

    // Detect the sensor
    status = DetectModule();
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: Sensor Detection failed with SIPL error", (int32_t)status);
        return status;
    }

    return NVSIPL_STATUS_OK;
}

CNvMDevice *CNvMMAX96712CameraModule::CreateBroadcastSensor()
{
    m_broadcastSensor = std::move(DoCreateBroadcastSensor(m_linkIndex));

    return m_broadcastSensor.get();
}

SIPLStatus
CNvMMAX96712CameraModule::Init()
{
    SIPLStatus status;

    // Initialize the sensors
    status = InitModule();
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: Sensor Init failed with SIPL error", (int32_t)status);
        return status;
    }

    status = m_upTransport->PostSensorInit(m_initLinkMask, m_groupInitProg);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: TransportLink PostInit failed with SIPL error", (int32_t)status);
        return status;
    }

    return status;
}

SIPLStatus
CNvMMAX96712CameraModule::PostInit()
{
    SIPLStatus status;

    status = m_upTransport->MiscInit();
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: TransportLink misc initialization failed with SIPL error", (int32_t)status);
        return status;
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus
CNvMMAX96712CameraModule::Start()
{
    SIPLStatus status;

    // Start the individual transport links
    status = m_upTransport->Start();
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: TransportLink Start failed with SIPL error", (int32_t)status);
        return status;
    }

    status = StartModule();
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: Sensor Start failed with SIPL error", (int32_t)status);
        return status;
    }

    m_groupInitProg = false;
    m_broadcastSensor.reset();

    return NVSIPL_STATUS_OK;
}

SIPLStatus
CNvMMAX96712CameraModule::Stop()
{
    SIPLStatus status;

    // Stop the sensors
    status = StopModule();
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: Sensor Stop failed with SIPL error", (int32_t)status);
    }

    // Stop the transport links
    status = m_upTransport->Stop();
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: TransportLink Stop failed with SIPL error", (int32_t)status);
        return status;
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus
CNvMMAX96712CameraModule::Reconfigure()
{
    SIPLStatus status, statusLinkControl;

    std::unique_ptr<CNvMSerializer> broadcastSerializer = std::move(CreateBroadcastSerializer(m_linkIndex));

    std::vector<CNvMDeserializer::LinkAction> linkAction;
    CNvMDeserializer::LinkAction item;
    NvMediaStatus nvmStatus = NVMEDIA_STATUS_OK;

    status = m_upSensor->Reset();
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: Sensor Reset failed with SIPL error", (int32_t)status);
        return status;
    }

    if (m_upEeprom) {
        status = m_upEeprom->Reset();
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96712: EEPROM Reset failed with SIPL error", (int32_t)status);
            return status;
        }
    }

    status = m_upSerializer->Reset();
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: Serializer Reset failed with SIPL error", (int32_t)status);
        return status;
    }

    status = m_upTransport->Reset();
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: TransportLink Reset failed with SIPL error", (int32_t)status);
        return status;
    }

    // Oneshot reset
    nvmStatus = MAX96712OneShotReset(m_pDeserializer->GetCDIDeviceHandle(), (LinkMAX96712)(1 << m_linkIndex));
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: MAX96712OneShotReset failed with NvMedia error:", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    // Add additional delays to get module stable as link rebuild can hit over 100ms
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Use the broadcast serializer to initialize all transport links
    status = m_upTransport->Init(broadcastSerializer->GetCDIDeviceHandle(), (1 << m_linkIndex), m_groupInitProg);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: TransportLink Init failed with SIPL error", (int32_t)status);

        // Disable the link for the camera module
        item.linkIdx = m_linkIndex;
        item.eAction = CNvMDeserializer::LinkAction::Action::LINK_DISABLE;
        linkAction.clear();
        linkAction.push_back(item);
        statusLinkControl = m_pDeserializer->ControlLinks(linkAction);
        if (statusLinkControl != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96712: ControlLinks failed with SIPL error", (int32_t)statusLinkControl);
            return statusLinkControl;
        }

        return status;
    }

    // Detect the sensor
    status = DetectModule();
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: Sensor Detection failed with SIPL error", (int32_t)status);

        // Disable the link for the camera module
        item.linkIdx = m_linkIndex;
        item.eAction = CNvMDeserializer::LinkAction::Action::LINK_DISABLE;
        linkAction.clear();
        linkAction.push_back(item);
        statusLinkControl = m_pDeserializer->ControlLinks(linkAction);
        if (statusLinkControl != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96712: ControlLinks failed with SIPL error", (int32_t)statusLinkControl);
            return statusLinkControl;
        }

        return status;
    }

    // Initialize the sensors
    status = InitModule();
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: Sensor Init failed with SIPL error", (int32_t)status);

        // Disable the link for the camera module
        item.linkIdx = m_linkIndex;
        item.eAction = CNvMDeserializer::LinkAction::Action::LINK_DISABLE;
        linkAction.clear();
        linkAction.push_back(item);
        statusLinkControl = m_pDeserializer->ControlLinks(linkAction);
        if (statusLinkControl != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96712: ControlLinks failed with SIPL error", (int32_t)statusLinkControl);
            return statusLinkControl;
        }

        return status;
    }

    status = m_upTransport->PostSensorInit(1U << m_linkIndex, m_groupInitProg);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: TransportLink PostInit failed with SIPL error", (int32_t)status);

        // Disable the link for the camera module
        item.linkIdx = m_linkIndex;
        item.eAction = CNvMDeserializer::LinkAction::Action::LINK_DISABLE;
        linkAction.clear();
        linkAction.push_back(item);
        statusLinkControl = m_pDeserializer->ControlLinks(linkAction);
        if (statusLinkControl != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96712: ControlLinks failed with SIPL error", (int32_t)statusLinkControl);
            return statusLinkControl;
        }

        return status;
    }

    status = m_upTransport->MiscInit();
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: TransportLink misc initialization failed with SIPL error", (int32_t)status);

        // Disable the link for the camera module
        item.linkIdx = m_linkIndex;
        item.eAction = CNvMDeserializer::LinkAction::Action::LINK_DISABLE;
        linkAction.clear();
        linkAction.push_back(item);
        statusLinkControl = m_pDeserializer->ControlLinks(linkAction);
        if (statusLinkControl != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96712: ControlLinks failed with SIPL error", (int32_t)statusLinkControl);
            return statusLinkControl;
        }

        return status;
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus
CNvMMAX96712CameraModule::ReadEEPROMData(const std::uint16_t address,
                                         const std::uint32_t length,
                                         std::uint8_t * const buffer)
{
    if (!m_upEeprom) {
        return NVSIPL_STATUS_NOT_SUPPORTED;
    }
    return m_upEeprom->ReadData(address, length, buffer);
}

#if !NV_IS_SAFETY
SIPLStatus
CNvMMAX96712CameraModule::WriteEEPROMData(const std::uint16_t address,
                                          const std::uint32_t length,
                                          std::uint8_t * const buffer)
{
    if (!m_upEeprom) {
        return NVSIPL_STATUS_NOT_SUPPORTED;
    }
    return m_upEeprom->WriteData(address, length, buffer);
}

SIPLStatus
CNvMMAX96712CameraModule::ToggleLED(bool enable)
{
    return NVSIPL_STATUS_NOT_SUPPORTED;
}
#endif // !NV_IS_SAFETY

SIPLStatus
CNvMMAX96712CameraModule::Deinit()
{
    SIPLStatus status;

    // Deinit the sensors
    status = DeinitModule();
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: Sensor Deinit failed with SIPL error", (int32_t)status);
        return status;
    }

    // Deinit hte transport links
    status = m_upTransport->Deinit();
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: TransportLink Deinit failed with SIPL error", (int32_t)status);
        return status;
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus
CNvMMAX96712CameraModule::DoSetConfig(
    const CameraModuleConfig *cameraModuleConfig,
    const uint8_t linkIndex)
{
    SIPLStatus status;

    // Save device params and serializer info
    m_oDeviceParams = *cameraModuleConfig->params;
    m_oSerInfo = cameraModuleConfig->cameraModuleInfos->serInfo;
    m_oSensorInfo = cameraModuleConfig->cameraModuleInfos->sensorInfo;
    m_pDeserializer = cameraModuleConfig->deserializer;
    m_sModuleName = cameraModuleConfig->cameraModuleInfos->name;
    m_interfaceType = cameraModuleConfig->eInterface;

    const CameraModuleInfo *moduleInfo = cameraModuleConfig->cameraModuleInfos;

    // Config serializer
    m_upSerializer = std::move(CreateNewSerializer());

    m_oDeviceParams.bUseNativeI2C = false;
    status = m_upSerializer->SetConfig(&moduleInfo->serInfo, &m_oDeviceParams);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: Serializer SetConfig failed with SIPL error", (int32_t)status);
        return status;
    }

    // Config Sensor
    m_upSensor.reset(new CNvMSensor());
    status = m_upSensor->SetConfig(&moduleInfo->sensorInfo, cameraModuleConfig->params);
    SetConfigModule(&moduleInfo->sensorInfo, cameraModuleConfig->params);
    m_upSensor->SetDriverHandle(GetCDIDeviceDriver());
    m_upSensor->SetDriverContext(GetCDIDeviceContext());
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: Sensor SetConfig failed with SIPL error", (int32_t)status);
        return status;
    }

    // Config EEPROM
    if (moduleInfo->isEEPROMSupported) {
        m_upEeprom.reset(new CNvMEEPROM());
        status = m_upEeprom->SetConfig(&moduleInfo->eepromInfo, cameraModuleConfig->params);
        m_upEeprom->SetDriverHandle(GetCDIDeviceDriver(MODULE_COMPONENT_EEPROM));
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96712: EEPROM SetConfig failed with SIPL error", (int32_t)status);
            return status;
        }
    }

    // Set Link Index
    m_linkIndex = moduleInfo->linkIndex;
    m_initLinkMask = cameraModuleConfig->initLinkMask;
    m_groupInitProg = cameraModuleConfig->groupInitProg;

    SetupConnectionProperty(cameraModuleConfig, linkIndex);

    return NVSIPL_STATUS_OK;
}

CNvMCameraModule::Property*
CNvMMAX96712CameraModule::GetCameraModuleProperty()
{
    return m_upCameraModuleProperty.get();
}

void CNvMMAX96712CameraModule::SetSensorConnectionProperty(
        CNvMCameraModuleCommon::ConnectionProperty::SensorConnectionProperty *sensorConnectionProperty)
{
    // Default properties
    sensorConnectionProperty->sensorReset.isNeeded = false;
    sensorConnectionProperty->sensorReset.pinNum = 0;
    sensorConnectionProperty->sensorReset.releaseResetLevel = false;
    sensorConnectionProperty->sensorReset.assertResetDuration = 0;
    sensorConnectionProperty->sensorReset.deassertResetWait = 0;

    sensorConnectionProperty->frameSync.pinNum = 8;

    sensorConnectionProperty->refClock.isNeeded = false;
    sensorConnectionProperty->refClock.pinNum = 0;

    sensorConnectionProperty->phyLanes.isLaneSwapNeeded = true;
    sensorConnectionProperty->phyLanes.lane0 = SENSOR_CSI_LANE_3;
    sensorConnectionProperty->phyLanes.lane1 = SENSOR_CSI_LANE_2;
    sensorConnectionProperty->phyLanes.lane2 = SENSOR_CSI_LANE_1;
    sensorConnectionProperty->phyLanes.lane3 = SENSOR_CSI_LANE_0;
    sensorConnectionProperty->phyLanes.isLanePolarityConfigureNeeded = false;
    sensorConnectionProperty->phyLanes.isTwoLane = false;

    sensorConnectionProperty->bPostSensorInitFsync = false;

    sensorConnectionProperty->pclk = 0;

    sensorConnectionProperty->vsyncHigh  = 0;
    sensorConnectionProperty->vsyncLow   = 0;
    sensorConnectionProperty->vsyncDelay = 0;
    sensorConnectionProperty->vsyncTrig  = 0;

    sensorConnectionProperty->bEnableInternalSync = false;

    sensorConnectionProperty->eepromWriteProtect.isNeeded = false;
    sensorConnectionProperty->eepromWriteProtect.pinNum = 0;
    sensorConnectionProperty->eepromWriteProtect.writeProtectLevel = false;
}

void
CNvMMAX96712CameraModule::SetupConnectionProperty(
    const CameraModuleConfig *cameraModuleConfig,
    const uint8_t linkIndex)
{

    const CameraModuleInfo *cameraModuleInfo = cameraModuleConfig->cameraModuleInfos;

    // Create camera module property and connection property
    m_upCameraModuleProperty = std::move(std::unique_ptr<Property>(new Property));
    m_upCameraModuleConnectionProperty =  std::move(std::unique_ptr<CNvMCameraModuleCommon::ConnectionProperty>(new CNvMCameraModuleCommon::ConnectionProperty));

    CNvMSensor* sensor = m_upSensor.get();
    CNvMCameraModule::Property::SensorProperty oSensorProperty = {
        .id = cameraModuleInfo->sensorInfo.id,
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
        .isAuthEnabled = sensor->IsAuthenticationEnabled(),
        .imgAuthThreadID = sensor->GetImgAuthThreadID(),
        .imgAuthAffinity = 0U,
        .pSensorControlHandle = sensor,
    };

    m_upCameraModuleProperty->sensorProperties = oSensorProperty;

    CNvMCameraModuleCommon::ConnectionProperty::SensorConnectionProperty sensorConnectionProperty = {};

    SetSensorConnectionProperty(&sensorConnectionProperty);

    sensorConnectionProperty.uBrdcstSensorAddrs  = sensor->GetNativeI2CAddr();

    sensorConnectionProperty.uVCID = oSensorProperty.virtualChannelID;
    sensorConnectionProperty.inputFormat =  sensor->GetInputFormat();
    sensorConnectionProperty.bEmbeddedDataType =  sensor->GetEmbDataType();
    sensorConnectionProperty.bEnableTriggerModeSync =  sensor->GetEnableExtSync();
    sensorConnectionProperty.fFrameRate =  sensor->GetFrameRate();
    sensorConnectionProperty.height = sensor->GetHeight();
    sensorConnectionProperty.width = sensor->GetWidth();
    sensorConnectionProperty.embeddedTop = sensor->GetEmbLinesTop();
    sensorConnectionProperty.embeddedBot = sensor->GetEmbLinesBot();
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    sensorConnectionProperty.bEnableTPG = sensor->GetEnableTPG();
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif // !NV_IS_SAFETY
    sensorConnectionProperty.sensorDescription = sensor->GetSensorDescription();
    sensorConnectionProperty.vGpioMap.resize(cameraModuleInfo->serInfo.serdesGPIOPinMappings.size());
    for (uint8_t i = 0U; i < cameraModuleInfo->serInfo.serdesGPIOPinMappings.size(); i++){
        sensorConnectionProperty.vGpioMap[i].sourceGpio = cameraModuleInfo->serInfo.serdesGPIOPinMappings[i].sourceGpio;
        sensorConnectionProperty.vGpioMap[i].destGpio = cameraModuleInfo->serInfo.serdesGPIOPinMappings[i].destGpio;
    }

    m_upCameraModuleConnectionProperty->sensorConnectionProperty = sensorConnectionProperty;

    CNvMCameraModule::Property::EEPROMProperty oEepromProperty = {
        .isEEPROMSupported = cameraModuleInfo->isEEPROMSupported,
    };

    m_upCameraModuleProperty->eepromProperties = oEepromProperty;
}

std::unique_ptr<CNvMSerializer>
CNvMMAX96712CameraModule::CreateBroadcastSerializer(const uint8_t linkIndex)
{
    CNvMDevice::DeviceParams params = m_oDeviceParams;
    params.bUseNativeI2C = true;
    std::unique_ptr<CNvMSerializer> up = std::move(CreateNewSerializer());
    up->SetConfig(&m_oSerInfo, &params);
    up->CreateCDIDevice(m_pCDIRoot, linkIndex);
    return up;
}

std::unique_ptr<CNvMSensor>
CNvMMAX96712CameraModule::DoCreateBroadcastSensor(const uint8_t linkIndex)
{
    CNvMDevice::DeviceParams params = m_oDeviceParams;
    params.bUseNativeI2C = true;
    std::unique_ptr<CNvMSensor> up(new CNvMSensor());
    up->SetConfig(&m_oSensorInfo, &params);
    up->SetDriverHandle(GetCDIDeviceDriver());
    up->SetDriverContext(GetCDIDeviceContext());
    up->CreateCDIDevice(m_pCDIRoot, linkIndex);
    return up;
}

SIPLStatus
CNvMMAX96712CameraModule::DoCreateCDIDevice(
    DevBlkCDIRootDevice* cdiRoot, const uint8_t linkIndex)
{
    SIPLStatus status;
    NvMediaStatus nvmStatus = NVMEDIA_STATUS_OK;

    m_pCDIRoot = cdiRoot;

    if (m_interfaceType >= NVSIPL_CAP_CSI_INTERFACE_TYPE_MAX) {
        SIPL_LOG_ERR_STR_UINT("CNvMMAX96712CameraModule:: CreateCameraPowerLoadSwitch "
            "Incorrect m_interfaceType passed: ",
            static_cast<uint32_t>(m_interfaceType));
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }
    // Create the broadcast serializer to reserve the physical I2C address
    std::unique_ptr<CNvMSerializer> broadcastSerializer = std::move(CreateBroadcastSerializer(m_linkIndex));

    // Create the broadcast sensor to reserve the physical I2C address
    std::unique_ptr<CNvMSensor> broadcastSensor = std::move(DoCreateBroadcastSensor(m_linkIndex));

    // Create serializer
    status = m_upSerializer->CreateCDIDevice(cdiRoot, linkIndex);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: Serializer CreateCDIDevice failed with SIPL error", (int32_t)status);
        return status;
    }

    // Create Sensor
    status = m_upSensor->CreateCDIDevice(cdiRoot, linkIndex);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: Sensor CreateCDIDevice failed with SIPL error", (int32_t)status);
        return status;
    }

    m_upCameraModuleConnectionProperty->sensorConnectionProperty.uSensorAddrs  = m_upSensor->GetI2CAddr();

    // Create EEPROM
    if (m_upCameraModuleProperty->eepromProperties.isEEPROMSupported) {
        status = m_upEeprom->CreateCDIDevice(cdiRoot, linkIndex);
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96712: EEPROM CreateCDIDevice failed with SIPL error", (int32_t)status);
            return status;
        }

        m_upCameraModuleConnectionProperty->eepromAddr = m_upEeprom->GetI2CAddr();
        m_upCameraModuleConnectionProperty->brdcstEepromAddr = m_upEeprom->GetNativeI2CAddr();
    } else {
        m_upCameraModuleConnectionProperty->eepromAddr = UINT8_MAX;
    }

    // Get power control information
    if (!m_oDeviceParams.bPassive && !m_oDeviceParams.bEnableSimulator) {
        nvmStatus = DevBlkCDIGetCamPowerControlInfo(m_upSensor->GetCDIDeviceHandle(), &m_camPwrControlInfo);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96712: Get power control info failed with NVMEDIA error", (int32_t)nvmStatus);
            return ConvertNvMediaStatus(nvmStatus);
        }
        if ((m_camPwrControlInfo.method) && (m_camPwrControlInfo.method != UINT8_MAX)) {
            CNvMDevice::DeviceParams params = m_oDeviceParams;
            params.bUseNativeI2C = true;
            m_upCampwr = CNvMMax20087DriverFactory::RequestPowerDriver(m_pCDIRoot, m_camPwrControlInfo.links[linkIndex]);
            if (m_upCampwr == nullptr) {
                SIPL_LOG_ERR_STR("Camera power load switch failed to request driver");
                return NVSIPL_STATUS_ERROR;
            }
            status = m_upCampwr->SetConfig(m_camPwrControlInfo.i2cAddr, &params);
            if (status != NVSIPL_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("Camera power load switch SetConfig failed with SIPL error",
                    static_cast<int32_t>(status));
                return status;
            }
            status = m_upCampwr->CreatePowerDevice(m_pCDIRoot, m_camPwrControlInfo.links[linkIndex]);
            if (status != NVSIPL_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("Camera power load switch CreateCDIDevice failed with SIPL error",
                    static_cast<int32_t>(status));
                return status;
            }
            status = m_upCampwr->InitPowerDevice(m_pCDIRoot,
                                                 m_upCampwr->GetDeviceHandle(),
                                                 m_camPwrControlInfo.links[linkIndex],
                                                 CsiPortForType(m_interfaceType));
            if (status != NVSIPL_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("Camera power load switch InitDevice failed with SIPL error",
                    static_cast<int32_t>(status));
                return status;
            }
        }
    }

    // Config transport link
    m_upTransport = std::move(CreateNewTransportLink());
    CNvMTransportLink::LinkParams params{};
    params.pSerCDIDevice = m_upSerializer->GetCDIDeviceHandle();
    params.pDeserCDIDevice = m_pDeserializer->GetCDIDeviceHandle();
    params.ulinkIndex = m_linkIndex;
    params.uBrdcstSerAddr = m_upSerializer->GetNativeI2CAddr();
    params.uSerAddr = m_upSerializer->GetI2CAddr();
    params.moduleConnectionProperty = *m_upCameraModuleConnectionProperty;
    params.bEnableSimulator = m_oDeviceParams.bEnableSimulator;
    params.bPassive = m_oDeviceParams.bPassive;
    params.m_groupInitProg = m_groupInitProg;

    status = m_upTransport->SetConfig(params);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: TransportLink SetConfig failed with SIPL error", (int32_t)status);
        return status;
    }

    return NVSIPL_STATUS_OK;
}

uint16_t
CNvMMAX96712CameraModule::GetPowerOnDelayMs()
{
    if ((CNvMDeserializer::LinkMode::LINK_MODE_GMSL2_6GBPS == GetLinkMode()) ||
        (CNvMDeserializer::LinkMode::LINK_MODE_GMSL2_3GBPS == GetLinkMode())) {
        return 100u; /* GMSL2 Link lock time maximum is 100ms*/
    } else {
        return 20u; /* GMSL1 Link lock time 20ms, I2C wake time is typical 1.1ms after releasing PWDN */
    }
}

uint16_t
CNvMMAX96712CameraModule::GetPowerOffDelayMs()
{
    return 0u;
}

std::string
CNvMMAX96712CameraModule::GetSupportedDeserailizer()
{
    return "MAX96712";
}

DevBlkCDIDeviceDriver*
CNvMMAX96712CameraModule::GetCDIDeviceDriver(
    ModuleComponent component)
{
    return GetCDIDeviceDriver();
}

const CNvMCameraModule::Version*
CNvMCameraModule_GetVersion() noexcept
{
    static const CNvMCameraModule::Version version;
    return &version;
}

SIPLStatus
CNvMCameraModule_SetDebugLevel(
    INvSIPLDeviceBlockTrace::TraceLevel level) noexcept
{
#if !NV_IS_SAFETY
    // Set the trace level used by the camera module files
    INvSIPLDeviceBlockTrace * instance = INvSIPLDeviceBlockTrace::GetInstance();
    if (instance != nullptr) {
        instance->SetLevel(level);
    }
#endif
    return NVSIPL_STATUS_OK;
}

SIPLStatus
CNvMMAX96712CameraModule::GetSerializerErrorSize(size_t & serializerErrorSize)
{
    SIPLStatus status = NVSIPL_STATUS_OK;
#if USE_MOCK_ERRORS
    serializerErrorSize = 1;
#else
    status = m_upSerializer->GetErrorSize(serializerErrorSize);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: Serializer GetErrorSize failed with SIPL error", (int32_t)status);
    }
#endif /* not USE_MOCK_ERRORS */
    return status;
}

SIPLStatus
CNvMMAX96712CameraModule::GetErrorSize(size_t & sensorErrorSize)
{
    sensorErrorSize = 0U;
    LOG_WARN("GetErrorSize not implemented for sensor\n");
    return NVSIPL_STATUS_OK;
}

SIPLStatus
CNvMMAX96712CameraModule::GetSensorErrorSize(size_t & sensorErrorSize)
{
    SIPLStatus status = NVSIPL_STATUS_OK;
#if USE_MOCK_ERRORS
    sensorErrorSize = 1;
#else
    status = GetErrorSize(sensorErrorSize);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: Sensor GetErrorSize failed with SIPL error", (int32_t)status);
    }
#endif /* not USE_MOCK_ERRORS */
    return status;
}

SIPLStatus
CNvMMAX96712CameraModule::GetSerializerErrorInfo(std::uint8_t * const buffer,
                                                 const std::size_t bufferSize,
                                                 std::size_t &size)
{
    SIPLStatus status = NVSIPL_STATUS_OK;

    status = m_upSerializer->GetErrorInfo(buffer, bufferSize, size);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: Serializer GetErrorInfo failed with SIPL error", (int32_t)status);
        return status;
    }

    return status;
}

SIPLStatus
CNvMMAX96712CameraModule::GetErrorInfo(std::uint8_t * const buffer,
                            const std::size_t bufferSize,
                            std::size_t &size)
{
    size = 0U;
    LOG_WARN("GetErrorInfo not implemented for sensor");
    return NVSIPL_STATUS_OK;
}

SIPLStatus
CNvMMAX96712CameraModule::GetSensorErrorInfo(std::uint8_t * const buffer,
                                  const std::size_t bufferSize,
                                  std::size_t &size)
{
    SIPLStatus status = NVSIPL_STATUS_OK;
#if USE_MOCK_ERRORS
    size = 1U;
#else
    status = GetErrorInfo(buffer, bufferSize, size);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: Sensor GetErrorInfo failed with SIPL error", (int32_t)status);
    }
#endif /* not USE_MOCK_ERRORS */
    return status;
}

SIPLStatus
CNvMMAX96712CameraModule::NotifyLinkState(const NotifyLinkStates linkState)
{
    SIPLStatus status = NVSIPL_STATUS_OK;

    switch (linkState) {
        case NotifyLinkStates::ENABLED:
            status = Start();
            if (status != NVSIPL_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("MAX96712: Failed to start on link state enabled", (int32_t)status);
                return status;
            }
            break;
        case NotifyLinkStates::PENDING_DISABLE:
            status = Stop();
            if (status != NVSIPL_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("MAX96712: Failed to stop on link state pending disabled", (int32_t)status);
                return status;
            }
            break;
        default:
            status = NVSIPL_STATUS_BAD_ARGUMENT;
            SIPL_LOG_ERR_STR_INT("MAX96712: Invalid link state notification", (int32_t)linkState);
            return status;
    }

    return status;
}

Interface* CNvMMAX96712CameraModule::GetInterface(const UUID &interfaceId) {
    return nullptr;
}

SIPLStatus CNvMMAX96712CameraModule::DoSetPower(bool powerOn)
{
    SIPLStatus status = NVSIPL_STATUS_ERROR;

    if (!m_camPwrControlInfo.method) {
        // Default is NvCCP, other power backends can be used here based on platform/usecase.
        status = PowerControlSetUnitPower(m_pwrPort, m_linkIndex, powerOn);
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96712: CNvMMAX96712CameraModule::DoSetPower failed with SIPL error", (int32_t)status);
        }
    } else if (m_camPwrControlInfo.method == UINT8_MAX) {
        status = NVSIPL_STATUS_OK;
    } else {
        if (m_upCampwr->isSupported() == NVSIPL_STATUS_OK) {
            status = m_upCampwr->PowerControlSetUnitPower(m_upCampwr->GetDeviceHandle(), m_camPwrControlInfo.links[m_linkIndex], powerOn);
            if (status != NVSIPL_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("MAX96712: CNvMMAX96712CameraModule::DoSetPower failed with SIPL error", (int32_t)status);
            }
        }
    }

    return status;
}

SIPLStatus CNvMMAX96712CameraModule::GetInterruptStatus(const uint32_t gpioIdx,
                                                    IInterruptNotify &intrNotifier)
{
    return NVSIPL_STATUS_NOT_SUPPORTED;
}

} // end of namespace
