/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "DS90UB9724cameramodule/CNvMDS90UB9724CameraModule.hpp"
#include "devices/DS90UB9724DeserializerDriver/CNvMDS90UB9724.hpp"
#include "devices/MAX20087Driver/CNvMMax20087.hpp"
#include "devices/MAX20087Driver/CNvMMax20087Factory.hpp"
#include "devices/MAX20087Driver/CNvMMax20087SyncAdapter.hpp"
#include "ModuleIF/CNvMCameraModuleExport.hpp"
#include "sipl_error.h"

#include <thread>

extern "C" {
    #include "cdi_debug_log.h"
    #include "pwr_utils.h"
}

namespace nvsipl
{

SIPLStatus CNvMDS90UB9724CameraModule::EnableLinkAndDetect() {
    SIPLStatus status;

    std::unique_ptr<CNvMSerializer> broadcastSerializer = std::move(CreateBroadcastSerializer(m_linkIndex));

    // Use the broadcast serializer to initialize all transport links
    status = m_upTransport->Init(broadcastSerializer->GetCDIDeviceHandle(), m_initLinkMask, m_groupInitProg);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB9724: TransportLink Init failed with SIPL error", (int32_t)status);
        return status;
    }

    // Detect the sensor
    status = DetectModule();
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB9724: Sensor Detection failed with SIPL error", (int32_t)status);
        return status;
    }

    return NVSIPL_STATUS_OK;
}

CNvMDevice *
CNvMDS90UB9724CameraModule::CreateBroadcastSensor()
{
    m_broadcastSensor = std::move(DoCreateBroadcastSensor(m_linkIndex));

    return m_broadcastSensor.get();
}

SIPLStatus
CNvMDS90UB9724CameraModule::Init()
{
    SIPLStatus status {NVSIPL_STATUS_OK};

    // Initialize the sensors
    status = InitModule();
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB9724: Sensor Init failed with SIPL error", (int32_t)status);
        return status;
    }

    status = m_upTransport->PostSensorInit(m_initLinkMask, m_groupInitProg);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB9724: TransportLink PostInit failed with SIPL error", (int32_t)status);
        return status;
    }

    return status;
}

SIPLStatus
CNvMDS90UB9724CameraModule::PostInit()
{
    m_groupInitProg = false;
    m_broadcastSensor.reset();

    return NVSIPL_STATUS_OK;
}

SIPLStatus
CNvMDS90UB9724CameraModule::Start()
{
    SIPLStatus status;

    // Start the individual transport links
    status = m_upTransport->Start();
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB9724: TransportLink Start failed with SIPL error", (int32_t)status);
        return status;
    }

    status = StartModule();
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB9724: Sensor Start failed with SIPL error", (int32_t)status);
        return status;
    }

    m_upTransport->PostSensorInit( m_initLinkMask, m_groupInitProg);
    return NVSIPL_STATUS_OK;
}

SIPLStatus
CNvMDS90UB9724CameraModule::Stop()
{
    SIPLStatus status;

    // Stop the sensors
    status = StopModule();
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB9724: Sensor Stop failed with SIPL error", (int32_t)status);
        return status;
    }

    // Stop the transport links
    status = m_upTransport->Stop();
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB9724: TransportLink Stop failed with SIPL error", (int32_t)status);
        return status;
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus
CNvMDS90UB9724CameraModule::Reconfigure()
{
    return NVSIPL_STATUS_OK;
}

SIPLStatus
CNvMDS90UB9724CameraModule::ReadEEPROMData(const std::uint16_t address,
                                         const std::uint32_t length,
                                         std::uint8_t * const buffer)
{
    return NVSIPL_STATUS_OK;
}

#if !NV_IS_SAFETY
SIPLStatus
CNvMDS90UB9724CameraModule::WriteEEPROMData(const std::uint16_t address,
                                          const std::uint32_t length,
                                          std::uint8_t * const buffer)
{
    return NVSIPL_STATUS_OK;
}
#endif // !NV_IS_SAFETY

SIPLStatus
CNvMDS90UB9724CameraModule::Deinit()
{
    SIPLStatus status;

    // Deinit the sensors
    status = DeinitModule();
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB9724: Sensor Deinit failed with SIPL error", (int32_t)status);
        return status;
    }

    // Deinit the transport links
    status = m_upTransport->Deinit();
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB9724: TransportLink Deinit failed with SIPL error", (int32_t)status);
        return status;
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus
CNvMDS90UB9724CameraModule::DoSetConfig(
    const CameraModuleConfig *cameraModuleConfig,
    const uint8_t linkIndex)
{
    SIPLStatus status;

    // Save device params and serializer info
    m_oDeviceParams = *cameraModuleConfig->params;
    m_oSerInfo = cameraModuleConfig->cameraModuleInfo->serInfo;
    m_oSensorInfo = cameraModuleConfig->cameraModuleInfo->sensorInfo;
    m_pDeserializer = cameraModuleConfig->deserializer;
    m_sModuleName = cameraModuleConfig->cameraModuleInfo->name;

    const CameraModuleInfo *moduleInfo = cameraModuleConfig->cameraModuleInfo;

    // Config serializer
    m_upSerializer = std::move(CreateNewSerializer());

    m_oDeviceParams.bUseNativeI2C = false;
    status = m_upSerializer->SetConfig(&moduleInfo->serInfo, &m_oDeviceParams);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB9724: Serializer SetConfig failed with SIPL error", (int32_t)status);
        return status;
    }

    // Config Sensor
    m_upSensor.reset(new CNvMSensor());
    status = m_upSensor->SetConfig(&moduleInfo->sensorInfo, cameraModuleConfig->params);
    SetConfigModule(&moduleInfo->sensorInfo, cameraModuleConfig->params);
    m_upSensor->SetDriverHandle(GetCDIDeviceDriver());

    m_upSensor->SetDriverContext(GetCDIDeviceContext());
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB9724: Sensor SetConfig failed with SIPL error", (int32_t)status);
        return status;
    }

    // Config EEPROM
    if (moduleInfo->isEEPROMSupported) {
        m_upEeprom.reset(new CNvMEEPROM());
        status = m_upEeprom->SetConfig(&moduleInfo->eepromInfo, cameraModuleConfig->params);
        m_upEeprom->SetDriverHandle(GetCDIDeviceDriver(MODULE_COMPONENT_EEPROM));
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("DS90UB9724: EEPROM SetConfig failed with SIPL error", (int32_t)status);
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
CNvMDS90UB9724CameraModule::GetCameraModuleProperty()
{
    return m_upCameraModuleProperty.get();
}

void
CNvMDS90UB9724CameraModule::SetSensorConnectionProperty(
    CNvMCameraModuleCommon::ConnectionProperty::SensorConnectionProperty *sensorConnectionProperty)
{
    // Default properties
    sensorConnectionProperty->sensorReset.isNeeded = false;
    sensorConnectionProperty->sensorReset.pinNum = 0;
    sensorConnectionProperty->sensorReset.releaseResetLevel = false;

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
}

void
CNvMDS90UB9724CameraModule::SetupConnectionProperty(
    const CameraModuleConfig *cameraModuleConfig,
    const uint8_t linkIndex)
{
    const CameraModuleInfo *cameraModuleInfo = cameraModuleConfig->cameraModuleInfo;

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
        .pSensorControlHandle = sensor,
    };

    m_upCameraModuleProperty->sensorProperty = oSensorProperty;

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

    m_upCameraModuleConnectionProperty->sensorConnectionProperty = sensorConnectionProperty;

    CNvMCameraModule::Property::EEPROMProperty oEepromProperty = {
        .isEEPROMSupported = cameraModuleInfo->isEEPROMSupported,
    };

    m_upCameraModuleProperty->eepromProperty = oEepromProperty;
}

std::unique_ptr<CNvMSerializer>
CNvMDS90UB9724CameraModule::CreateBroadcastSerializer(const uint8_t linkIndex)
{
    CNvMDevice::DeviceParams params = m_oDeviceParams;
    params.bUseNativeI2C = true;
    std::unique_ptr<CNvMSerializer> up = std::move(CreateNewSerializer());
    up->SetConfig(&m_oSerInfo, &params);
    up->CreateCDIDevice(m_pCDIRoot, linkIndex);
    return up;
}

std::unique_ptr<CNvMSensor>
CNvMDS90UB9724CameraModule::DoCreateBroadcastSensor(const uint8_t linkIndex)
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
CNvMDS90UB9724CameraModule::DoCreateCDIDevice(
    DevBlkCDIRootDevice* cdiRoot,
    const uint8_t linkIndex)
{
    SIPLStatus status;
    NvMediaStatus nvmStatus {NVMEDIA_STATUS_OK};

    m_pCDIRoot = cdiRoot;

    // Create serializer
    status = m_upSerializer->CreateCDIDevice(cdiRoot, linkIndex);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB9724: Serializer CreateCDIDevice failed with SIPL error", (int32_t)status);
        return status;
    }

    // Create Sensor
    status = m_upSensor->CreateCDIDevice(cdiRoot, linkIndex);
    if (status != NVSIPL_STATUS_OK) {

        SIPL_LOG_ERR_STR_INT("DS90UB9724: Sensor CreateCDIDevice failed with SIPL error", (int32_t)status);
        return status;
    }

    m_upCameraModuleConnectionProperty->sensorConnectionProperty.uSensorAddrs  = m_upSensor->GetI2CAddr();


    // Create EEPROM
    if (m_upCameraModuleProperty->eepromProperty.isEEPROMSupported) {
        status = m_upEeprom->CreateCDIDevice(cdiRoot, linkIndex);
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("DS90UB9724: EEPROM CreateCDIDevice failed with SIPL error", (int32_t)status);
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
            SIPL_LOG_ERR_STR_INT("DS90UN9724: Get power control info failed with NVMEDIA error", (int32_t)nvmStatus);
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
                                                    m_camPwrControlInfo.links[linkIndex]);
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
        SIPL_LOG_ERR_STR_INT("DS90UB9724: TransportLink SetConfig failed with SIPL error", (int32_t)status);
        return status;
    }

    return NVSIPL_STATUS_OK;
}

uint16_t
CNvMDS90UB9724CameraModule::GetPowerOnDelayMs()
{
    return 20; /* Link lock time 20ms, I2C wake time is typical 1.1ms after releasing PWDN */
}

uint16_t
CNvMDS90UB9724CameraModule::GetPowerOffDelayMs()
{
    return 0;
}

std::string
CNvMDS90UB9724CameraModule::GetSupportedDeserailizer()
{
    return "DS90UB9724";
}

DevBlkCDIDeviceDriver*
CNvMDS90UB9724CameraModule::GetCDIDeviceDriver(
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
    INvSIPLDeviceBlockTrace *instance = INvSIPLDeviceBlockTrace::GetInstance();
    if (instance != nullptr) {
        instance->SetLevel(level);
    }
#endif
    return NVSIPL_STATUS_OK;
}

SIPLStatus
CNvMDS90UB9724CameraModule::GetSerializerErrorSize(size_t & serializerErrorSize)
{
    serializerErrorSize = 0;
    return NVSIPL_STATUS_OK;
}

SIPLStatus
CNvMDS90UB9724CameraModule::GetSensorErrorSize(size_t & sensorErrorSize)
{
    sensorErrorSize = 0;
    return NVSIPL_STATUS_OK;
}

SIPLStatus
CNvMDS90UB9724CameraModule::GetSerializerErrorInfo(std::uint8_t * const buffer,
                                                   const std::size_t bufferSize,
                                                   std::size_t &size)
{
    size = 0U;
    return NVSIPL_STATUS_OK;
}

SIPLStatus
CNvMDS90UB9724CameraModule::GetSensorErrorInfo(std::uint8_t * const buffer,
                                               const std::size_t bufferSize,
                                               std::size_t &size)
{
    size = 0U;
    return NVSIPL_STATUS_OK;
}

SIPLStatus
CNvMDS90UB9724CameraModule::NotifyLinkState(const NotifyLinkStates linkState)
{
    return NVSIPL_STATUS_OK;
}

Interface* CNvMDS90UB9724CameraModule::GetInterface(const UUID &interfaceId)
{
    return nullptr;
}

SIPLStatus CNvMDS90UB9724CameraModule::DoSetPower(bool powerOn)
{
    SIPLStatus status = NVSIPL_STATUS_ERROR;

    if (!m_camPwrControlInfo.method) {
        // Default is NvCCP, other power backends can be used here based on platform/usecase.
        status = PowerControlSetUnitPower(m_pwrPort, m_linkIndex, powerOn);
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("DS90UN9724: CNvMDS90UB9724CameraModule::DoSetPower failed with SIPL error", (int32_t)status);
        }
    } else if (m_camPwrControlInfo.method == UINT8_MAX) {
        status = NVSIPL_STATUS_OK;
    } else {
        if (m_upCampwr->isSupported() == NVSIPL_STATUS_OK) {
            status = m_upCampwr->PowerControlSetUnitPower(m_upCampwr->GetDeviceHandle(), m_camPwrControlInfo.links[m_linkIndex], powerOn);
            if (status != NVSIPL_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("DS90UN9724: CNvMDS90UB9724CameraModule::DoSetPower failed with SIPL error", (int32_t)status);
            }
        }
    }

    return status;
}

SIPLStatus CNvMDS90UB9724CameraModule::GetInterruptStatus(const uint32_t gpioIdx,
                                                          IInterruptNotify &intrNotifier)
{
    return NVSIPL_STATUS_NOT_SUPPORTED;
}

} // end of namespace
