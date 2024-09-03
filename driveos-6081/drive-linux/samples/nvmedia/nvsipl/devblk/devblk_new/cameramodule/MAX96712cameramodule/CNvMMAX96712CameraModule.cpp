/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "MAX96712cameramodule/CNvMMAX96712CameraModule.hpp"

#if FUSA_CDD_NV
#include "devices/MAX96712DeserializerDriver_nv/CNvMMax96712_Fusa_nv.hpp"
// Including cdi_max96712.h to fix AUTOSAR A-16-2-3
#include "devices/MAX96712DeserializerDriver_nv/cdi_max96712_nv.h"
#else
#include "devices/MAX96712DeserializerDriver/CNvMMax96712_Fusa.hpp"
#include "devices/MAX96712DeserializerDriver/cdi_max96712.h"
#endif

#include "devices/MAX20087Driver/CNvMMax20087SyncAdapter.hpp"
#include "devices/MAX20087Driver/CNvMMax20087Factory.hpp"
#include "TransportLinkIF/CNvMTransportLink.hpp"
#include "ModuleIF/CNvMCameraModuleExport.hpp"
#include "DeserializerIF/CNvMDeserializer.hpp"
#include "INvSIPLDeviceInterfaceProvider.hpp"
#include "SerializerIF/CNvMSerializer.hpp"
#include "ModuleIF/CNvMCameraModule.hpp"
#include "CNvMCameraModuleCommon.hpp"
#include "NvSIPLDeviceBlockTrace.hpp"
#include "NvSIPLDeviceBlockInfo.hpp"
#include "EEPROMIF/CNvMEEPROM.hpp"
#include "SensorIF/CNvMSensor.hpp"
#include "CampwrIF/CNvMCampwr.hpp"
#include "SiplErrorConverter.hpp"
#include "VCSELIF/CNvMVCSEL.hpp"
#include "IInterruptNotify.hpp"
#include "PMICIF/CNvMPMIC.hpp"
#include "NvSIPLCommon.hpp"
#include "CNvMDevice.hpp"
#include "CDD_Defines.hpp"

#include "NvSIPLCapStructs.h"
#include "nvos_s3_tegra_log.h"
#include "nvmedia_core.h"
#include "pwr_utils.h"
#include "devblk_cdi.h"
#include "sipl_error.h"
#include "sipl_util.h"
#include "type_traits"

#include <iosfwd>
#include <new>
#include <vector>
#include <memory>
#include <string>
#include <chrono>
#include <thread>
#include <cstdint>
#include <cstddef>

namespace nvsipl
{

/* coverity[autosar_cpp14_a8_4_4_violation] : intentional */
SIPLStatus CNvMMAX96712CameraModule::InitCreateBroadcastSerializer(
    SIPLStatus const instatus,
    std::unique_ptr<CNvMSerializer>& broadcastSerializer)
{
    SIPLStatus status = instatus;

    if(status == NVSIPL_STATUS_OK) {
        status = CreateBroadcastSerializer(m_linkIndex, broadcastSerializer);
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("CreateBroadcastSerializer failed with SIPL error",
                                  static_cast<int32_t>(status));
        }
    }

    return status;
}

// Use the broadcast serializer to initialize all transport links
SIPLStatus CNvMMAX96712CameraModule::InitTransportLinks(SIPLStatus const instatus,
                                                        CNvMSerializer const &broadcastSerializer )
{
    SIPLStatus status = instatus;

    if(status == NVSIPL_STATUS_OK) {
        status = m_upTransport->Init(broadcastSerializer.GetCDIDeviceHandle(),
                                     m_initLinkMask,
                                     m_groupInitProg);
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("TransportLink Init failed with SIPL error",
                                  static_cast<int32_t>(status));
        }
    }

    return status;
}

// Detect the sensor
SIPLStatus CNvMMAX96712CameraModule::InitDetectModule(SIPLStatus const instatus)
{
    SIPLStatus status = instatus;

    if(status == NVSIPL_STATUS_OK) {
        status = DetectModule();
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("Sensor Detection failed with SIPL error",
                              static_cast<int32_t>(status));
        }
    }

    return status;
}

/* coverity[autosar_cpp14_m0_1_10_violation] : intentional TID-2013 */
SIPLStatus CNvMMAX96712CameraModule::EnableLinkAndDetect()
{
    SIPLStatus status = NVSIPL_STATUS_OK;
    std::unique_ptr<CNvMSerializer> broadcastSerializer;

    status = InitCreateBroadcastSerializer(status, broadcastSerializer);
    status = InitTransportLinks(status, *broadcastSerializer);
    status = InitDetectModule(status);

    /* Restore power switch interrupt mask */
    if (status == NVSIPL_STATUS_OK) {
        status = MaskRestorePowerSwitchInterrupt(false);
    }

    return status;
}

/* coverity[autosar_cpp14_m0_1_10_violation] : intentional TID-2013 */
CNvMDevice *CNvMMAX96712CameraModule::CreateBroadcastSensor()
{
    SIPLStatus status;
    CNvMDevice *brcSensorDev{nullptr};

    status = DoCreateBroadcastSensor(m_linkIndex, m_broadcastSensor);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("CreateBroadcastSensor failed with SIPL error",
                             static_cast<int32_t>(status));
    } else {
        brcSensorDev = GetBroadcastSensor();
    }

    return brcSensorDev;
}

/* coverity[misra_cpp_2008_rule_0_1_10_violation] : intentional TID-1966 */
/* coverity[autosar_cpp14_m0_1_10_violation] : intentional TID-2013 */
SIPLStatus
CNvMMAX96712CameraModule::Init()
{
    SIPLStatus status;
    status = InitModule();
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_HEX_AND_UINT("Sensor Init failed",
                                       static_cast<uint32_t>(m_initLinkMask),
                                       static_cast<uint32_t>(status));
        return status;
    }

    status = m_upTransport->PostSensorInit(m_initLinkMask, m_groupInitProg);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_HEX_AND_UINT("TransportLink PostInit failed",
                                       static_cast<uint32_t>(m_initLinkMask),
                                       static_cast<uint32_t>(status));
        return status;
    }

    return NVSIPL_STATUS_OK;
}

/* coverity[autosar_cpp14_m0_1_10_violation] : intentional TID-2013 */
SIPLStatus
CNvMMAX96712CameraModule::PostInit()
{
    SIPLStatus status;

    status = m_upTransport->MiscInit();
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("TransportLink misc initialization failed with SIPL error",
                              static_cast<int32_t>(status));
    } else {
        status = PostInitModule();
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("Sensor PostInit failed with SIPL error",
                                  static_cast<int32_t>(status));
        }
    }
    return status;
}

SIPLStatus
CNvMMAX96712CameraModule::Start()
{
    SIPLStatus status;

    // Start the individual transport links
    status = m_upTransport->Start();
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("TransportLink Start failed with SIPL error",
                              static_cast<int32_t>(status));
    } else {
        status = StartModule();
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("Sensor Start failed with SIPL error",
                                  static_cast<int32_t>(status));
        }
    }

    m_groupInitProg = false;
    m_broadcastSensor.reset();

    return status;
}

SIPLStatus
CNvMMAX96712CameraModule::Stop()
{
    SIPLStatus status;

    // Stop the sensors
    status = StopModule();
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("Sensor Stop failed with SIPL error",
                              static_cast<int32_t>(status));
    }

    // Stop the transport links
    status = m_upTransport->Stop();
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("TransportLink Stop failed with SIPL error",
                              static_cast<int32_t>(status));
    }
    return status;
}

SIPLStatus CNvMMAX96712CameraModule::Reset(SIPLStatus const instatus)
{
    SIPLStatus status = instatus;

    if (status == NVSIPL_STATUS_OK) {
        status = m_upSensor->Reset();
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("Sensor Reset failed with SIPL error",
                                static_cast<int32_t>(status));
        }
    }

    if (status == NVSIPL_STATUS_OK) {
        if (m_upEeprom != nullptr) {
            status = m_upEeprom->Reset();
            if (status != NVSIPL_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("EEPROM Reset failed with SIPL error",
                                      static_cast<int32_t>(status));
            }
        }
    }

    if (status == NVSIPL_STATUS_OK) {
        status = m_upSerializer->Reset();
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("Serializer Reset failed with SIPL error",
                                  static_cast<int32_t>(status));
        }
    }

    if (status == NVSIPL_STATUS_OK) {
        status = m_upTransport->Reset();
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("TransportLink Reset failed with SIPL error",
                                  static_cast<int32_t>(status));
        }
    }

    return status;
}

void CNvMMAX96712CameraModule::LinkDisable(CNvMDeserializer::LinkAction& item)
{
    SIPLStatus statusLinkControl;
    std::vector<CNvMDeserializer::LinkAction> linkAction;

    // Disable the link for the camera module
    item.linkIdx = m_linkIndex;
    item.eAction = CNvMDeserializer::LinkAction::Action::LINK_DISABLE;
    linkAction.push_back(item);
    statusLinkControl = m_pDeserializer->ControlLinks(linkAction);
    if (statusLinkControl != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("ControlLinks failed with SIPL error",
                    static_cast<int32_t>(statusLinkControl));
    }
}

/* coverity[autosar_cpp14_a8_4_4_violation] : intentional */
SIPLStatus CNvMMAX96712CameraModule::InitializeTransportLinks(
    SIPLStatus const instatus,
    CNvMSerializer const &broadcastSerializer)
{
    SIPLStatus status {instatus};

    if (status == NVSIPL_STATUS_OK) {
        // Add additional delays to get module stable as link rebuild can hit over 100ms
        std::this_thread::sleep_for<>(std::chrono::milliseconds(100));

        // Use the broadcast serializer to initialize all transport links
        uint32_t const tmpLinkIndex {bit32(m_linkIndex)};
        if (static_cast<uint32_t>(UINT8_MAX) > tmpLinkIndex) {
            status = m_upTransport->Init(broadcastSerializer.GetCDIDeviceHandle(),
                                         static_cast<uint8_t>(tmpLinkIndex),
                                         m_groupInitProg);
            if (status != NVSIPL_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("TransportLink Init failed with SIPL error",
                                    static_cast<int32_t>(status));
            }
        } else {
            SIPL_LOG_ERR_STR_HEX_UINT("linkIndex is out of range", tmpLinkIndex);
            status = NVSIPL_STATUS_BAD_ARGUMENT;
        }
    }

    return status;
}

// Detect the sensor
/* coverity[autosar_cpp14_a8_4_4_violation] : intentional */
SIPLStatus CNvMMAX96712CameraModule::DetectSensor(
    SIPLStatus const instatus)
{
    SIPLStatus status {instatus};

    if (status == NVSIPL_STATUS_OK) {
        status = DetectModule();
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("Sensor Detection failed with SIPL error",
                                  static_cast<int32_t>(status));
        }
    }

    return status;
}

// Authenticate the sensor
/* coverity[autosar_cpp14_a8_4_4_violation] : intentional */
SIPLStatus CNvMMAX96712CameraModule::AuthenticateSensor(
    SIPLStatus const instatus)
{
    SIPLStatus status {instatus};

    if (status == NVSIPL_STATUS_OK) {
        status = Authenticate();
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("Sensor Authentication failed with SIPL error",
                                  static_cast<int32_t>(status));
        }
    }

    return status;
}

// Initialize the sensors
/* coverity[autosar_cpp14_a8_4_4_violation] : intentional */
SIPLStatus CNvMMAX96712CameraModule::InitSensors(
    SIPLStatus const instatus)
{
    SIPLStatus status {instatus};

    if (status == NVSIPL_STATUS_OK) {
        status = InitModule();
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("Sensor Init failed with SIPL error",
                                  static_cast<int32_t>(status));
        }
    }

    return status;
}

/* coverity[autosar_cpp14_a8_4_4_violation] : intentional */
SIPLStatus CNvMMAX96712CameraModule::TransportLinkPostInit(
    SIPLStatus const instatus)
{
    SIPLStatus status {instatus};

    if (status == NVSIPL_STATUS_OK) {
        if (m_linkIndex > MAX_LINK_INDEX) {
            status = NVSIPL_STATUS_INVALID_STATE;
        } else {
            uint32_t const tmpLinkIndex {bit32(m_linkIndex)};
            if (static_cast<uint32_t>(UINT8_MAX) > tmpLinkIndex) {
                status = m_upTransport->PostSensorInit(
                                static_cast<uint8_t>(tmpLinkIndex),
                                m_groupInitProg);
                if (status != NVSIPL_STATUS_OK) {
                    SIPL_LOG_ERR_STR_INT("TransportLink PostInit failed with SIPL error",
                                        static_cast<int32_t>(status));
                }
            } else {
                SIPL_LOG_ERR_STR_HEX_UINT("linkIndex is out of range",
                                          tmpLinkIndex);
                status = NVSIPL_STATUS_BAD_ARGUMENT;
            }
        }
    }

    return status;
}

/* coverity[autosar_cpp14_a8_4_4_violation] : intentional */
SIPLStatus CNvMMAX96712CameraModule::InitTransportLinkMisc(
    SIPLStatus const instatus)
{
    SIPLStatus status {instatus};

    if (status == NVSIPL_STATUS_OK) {
        status = m_upTransport->MiscInit();
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("TransportLink misc initialization failed with SIPL error",
                                  static_cast<int32_t>(status));
        }
    }

    return status;
}

/* coverity[autosar_cpp14_m0_1_10_violation] : intentional TID-2013 */
SIPLStatus
CNvMMAX96712CameraModule::Reconfigure()
{
    SIPLStatus status;
    std::unique_ptr<CNvMSerializer> broadcastSerializer;

    status = CreateBroadcastSerializer(m_linkIndex, broadcastSerializer);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("CreateBroadcastSerializer failed with SIPL error",
                              static_cast<int32_t>(status));
    }

    status = Reset(status);
    status = InitializeTransportLinks(status, *broadcastSerializer);
    status = DetectSensor(status);
    status = AuthenticateSensor(status);
    status = InitSensors(status);
    status = TransportLinkPostInit(status);
    status = InitTransportLinkMisc(status);

    if (status != NVSIPL_STATUS_OK) {
        CNvMDeserializer::LinkAction item;

        LinkDisable(item);
    }

    return status;
}

/* coverity[autosar_cpp14_m0_1_10_violation] : intentional TID-2013 */
SIPLStatus
CNvMMAX96712CameraModule::ReadEEPROMData(std::uint16_t const address,
                                         std::uint32_t const length,
                                         std::uint8_t * const buffer) noexcept
{
    SIPLStatus status;
    if (m_upEeprom == nullptr) {
        status =  NVSIPL_STATUS_NOT_SUPPORTED;
    } else {
        status = m_upEeprom->ReadData(address, length, buffer);
    }

    return status;
}

/* coverity[autosar_cpp14_a16_0_1_violation] : intentional TID-2039 */
#if !(NV_IS_SAFETY)
SIPLStatus
CNvMMAX96712CameraModule::WriteEEPROMData(std::uint16_t const address,
                                          std::uint32_t const length,
                                          std::uint8_t * const buffer)
{
    if (m_upEeprom == nullptr) {
        return NVSIPL_STATUS_NOT_SUPPORTED;
    }
    return m_upEeprom->WriteData(address, length, buffer);
}
/* coverity[autosar_cpp14_a16_0_1_violation] : intentional TID-2039 */

/* coverity[autosar_cpp14_m0_1_10_violation] : intentional TID-2013 */
SIPLStatus
CNvMMAX96712CameraModule::ToggleLED(bool const enable) noexcept
{
    static_cast<void>(enable);
    return NVSIPL_STATUS_NOT_SUPPORTED;
}
#endif // !(NV_IS_SAFETY)
/* coverity[autosar_cpp14_m0_1_10_violation] : intentional TID-2013 */
SIPLStatus
CNvMMAX96712CameraModule::Deinit()
{
    SIPLStatus status;

    // Deinit the sensors
    status = DeinitModule();
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("Sensor Deinit failed with SIPL error",
                              static_cast<int32_t>(status));
    } else {
       // Deinit hte transport links
       status = m_upTransport->Deinit();
       if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("TransportLink Deinit failed with SIPL error",
                              static_cast<int32_t>(status));
       }
    }
    return status;
}

// Function to map SIPL log verbosity level to C log verbosity level.
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
C_LogLevel CNvMMAX96712CameraModule::ConvertLogLevel(
    INvSIPLDeviceBlockTrace::TraceLevel const level
)
{
    return static_cast<C_LogLevel>(level);
}
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

/* coverity[misra_cpp_2008_rule_0_1_10_violation] : intentional TID-1966 */
/* coverity[autosar_cpp14_m0_1_10_violation] : intentional TID-2013 */
SIPLStatus
CNvMMAX96712CameraModule::DoSetConfig(
    CameraModuleConfig const* const cameraModuleCfg,
    uint8_t const linkIndex)
{
    SIPLStatus status;

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    INvSIPLDeviceBlockTrace *instance = INvSIPLDeviceBlockTrace::GetInstance();
    if (instance != nullptr) {
        INvSIPLDeviceBlockTrace::TraceLevel level = instance-> GetLevel();
        C_LogLevel c_level = ConvertLogLevel(level);
        SetCLogLevel(c_level);
    }
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

    if(cameraModuleCfg != nullptr) {
    // Save device params and serializer info
    m_oDeviceParams = *cameraModuleCfg->params;
    m_oSerInfo = cameraModuleCfg->cameraModuleInfos->serInfo;
    m_oSensorInfo = cameraModuleCfg->cameraModuleInfos->sensorInfo;
    m_pDeserializer = cameraModuleCfg->deserializer;
    m_sModuleName = cameraModuleCfg->cameraModuleInfos->name;
    m_interfaceType = cameraModuleCfg->eInterface;

    CameraModuleInfo const* const moduleInfo = cameraModuleCfg->cameraModuleInfos;

    // Config serializer
    m_upSerializer = std::move(CreateNewSerializer());

    m_oDeviceParams.bUseNativeI2C = false;
    status = m_upSerializer->SetConfig(&moduleInfo->serInfo, &m_oDeviceParams);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("Serializer SetConfig failed with SIPL error",
                              static_cast<int32_t>(status));
    } else {
        // Config Sensor

        /* coverity[misra_cpp_2008_rule_18_4_1_violation] : intentional TID-1968 */
        m_upSensor.reset(new(std::nothrow) CNvMSensor());
        if (m_upSensor == nullptr) {
            SIPL_LOG_ERR_STR("CNvMSensor allocation failed!");
            status = NVSIPL_STATUS_RESOURCE_ERROR;
        } else{
            status = m_upSensor->SetConfig(&moduleInfo->sensorInfo,
                                           cameraModuleCfg->params);
            if (status != NVSIPL_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("Sensor SetConfig failed with SIPL error",
                                      static_cast<int32_t>(status));
            } else {
                status = SetConfigModule(&moduleInfo->sensorInfo, cameraModuleCfg->params);
                if (status != NVSIPL_STATUS_OK) {
                    SIPL_LOG_ERR_STR_INT("SetConfigModule failed with SIPL error",
                                        static_cast<int32_t>(status));
                } else {
                    status = m_upSensor->SetDriverHandle(GetCDIDeviceDriver());
                    if (status != NVSIPL_STATUS_OK) {
                        SIPL_LOG_ERR_STR_INT("SetDriverHandle failed with SIPL error",
                                            static_cast<int32_t>(status));
                    } else {
                        status = m_upSensor->SetDriverContext(GetCDIDeviceContext());
                        if (status != NVSIPL_STATUS_OK) {
                            SIPL_LOG_ERR_STR_INT("SetDriverContext failed with SIPL error",
                                                static_cast<int32_t>(status));
                        } else {
                            if (moduleInfo->linkIndex > MAX_LINK_INDEX) {
                                status = NVSIPL_STATUS_BAD_ARGUMENT;
                                SIPL_LOG_ERR_STR("CNvMMAX96712CameraModule::DoSetConfig "
                                                 "Bad Argument");
                            } else {
                                // Set Link Index
                                m_linkIndex = static_cast<uint8_t>(moduleInfo->linkIndex);
                                m_initLinkMask = cameraModuleCfg->initLinkMask;
                                m_groupInitProg = cameraModuleCfg->groupInitProg;

                                // Config EEPROM
                                if (moduleInfo->isEEPROMSupported) {

                                    /* coverity[misra_cpp_2008_rule_18_4_1_violation] : intentional TID-1968 */
                                    m_upEeprom.reset(new(std::nothrow) CNvMEEPROM());
                                    if (m_upEeprom == nullptr) {
                                        SIPL_LOG_ERR_STR("CNvMEEPROM allocation failed!");
                                        status = NVSIPL_STATUS_RESOURCE_ERROR;
                                    } else {
                                        status = m_upEeprom->SetConfig(&moduleInfo->eepromInfo,
                                                                       cameraModuleCfg->params);
                                        if (status != NVSIPL_STATUS_OK) {
                                            SIPL_LOG_ERR_STR_INT("EEPROM SetConfig failed "
                                                                 "with SIPL error",
                                                                  static_cast<int32_t>(status));
                                        } else {
                                            status = m_upEeprom->SetDriverHandle(GetCDIDeviceDriver
                                                     (ModuleComponent::MODULE_COMPONENT_EEPROM));
                                            if (status != NVSIPL_STATUS_OK) {
                                                SIPL_LOG_ERR_STR_INT("EEPROM SetDriverHandle "
                                                                     "failed with SIPL error",
                                                                      static_cast<int32_t>(status));
                                            }
                                        }
                                    }
                                }

                                if (status == NVSIPL_STATUS_OK) {
                                    status = SetupConnectionProperty(cameraModuleCfg, linkIndex);
                                    if (status != NVSIPL_STATUS_OK) {
                                        SIPL_LOG_ERR_STR("SetupConnectionProperty failed");
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    }
    else {
        status = NVSIPL_STATUS_ERROR;
    }
    return status;
}

/* coverity[autosar_cpp14_m0_1_10_violation] : intentional TID-2013 */
CNvMCameraModule::Property*
CNvMMAX96712CameraModule::GetCameraModuleProperty() noexcept
{
    return m_upCameraModuleProperty.get();
}

SIPLStatus
CNvMMAX96712CameraModule::SetupConnectionProperty(
    CameraModuleConfig const* const cameraModuleConfig,
    uint8_t const linkIndex)
{
    SIPLStatus status {NVSIPL_STATUS_OK};

    if(cameraModuleConfig != nullptr) {
        CameraModuleInfo const * const moduleInfo {cameraModuleConfig->cameraModuleInfos};

        // Create camera module property and connection property
        /* coverity[misra_cpp_2008_rule_18_4_1_violation] : intentional TID-1968 */
        m_upCameraModuleProperty = std::move(std::make_unique<Property>());
        if (m_upCameraModuleProperty == nullptr) {
            SIPL_LOG_ERR_STR("CameraModuleProperty allocation failed!");
            status = NVSIPL_STATUS_RESOURCE_ERROR;
        } else {
            /* coverity[misra_cpp_2008_rule_18_4_1_violation] : intentional TID-1968 */
            m_upCameraModuleConnectionProperty =
                std::move(std::make_unique<CNvMCameraModuleCommon::ConnectionProperty>());
            if (m_upCameraModuleConnectionProperty == nullptr) {
                SIPL_LOG_ERR_STR("CameraModuleConnectionProperty allocation failed!");
                status = NVSIPL_STATUS_RESOURCE_ERROR;
            } else {
                CNvMSensor *const sensor = m_upSensor.get();
                CNvMCameraModule::Property::SensorProperty const oSensorProperty {
                    .id = moduleInfo->sensorInfo.id,
                    .virtualChannelID = linkIndex,
                    .inputFormat = sensor->GetInputFormat(),
                    .pixelOrder = sensor->GetPixelOrder(),
                    .width = sensor->GetWidth(),
                    .height = sensor->GetHeight(),
                    .startX = 0U,
                    .startY = 0U,
                    .embeddedTop = sensor->GetEmbLinesTop(),
                    .embeddedBot = sensor->GetEmbLinesBot(),
                    .frameRate = sensor->GetFrameRate(),
                    .embeddedDataType = sensor->GetEmbDataType(),
                    .isAuthEnabled = sensor->IsAuthenticationEnabled(),
                    .imgAuthThreadID = sensor->GetImgAuthThreadID(),
                    .imgAuthAffinity = 0U,
                    .pSensorControlHandle = sensor,
                    .pSensorInterfaceProvider = nullptr,
                };

                m_upCameraModuleProperty->sensorProperties = oSensorProperty;

                CNvMCameraModuleCommon::ConnectionProperty::SensorConnectionProperty
                    sensorConnectionProperty;

                sensorConnectionProperty.eepromWriteProtect.isNeeded = false;
                status = SetSensorConnectionProperty(&sensorConnectionProperty);
                if (status != NVSIPL_STATUS_OK) {
                    SIPL_LOG_ERR_STR("Call to SetSensorConnectionProperty failed.")
                } else {
                    sensorConnectionProperty.uBrdcstSensorAddrs  = sensor->GetNativeI2CAddr();

                    sensorConnectionProperty.uVCID =
                        static_cast<uint8_t>(oSensorProperty.virtualChannelID);
                    sensorConnectionProperty.inputFormat =  sensor->GetInputFormat();
                    sensorConnectionProperty.bEmbeddedDataType =  sensor->GetEmbDataType();
                    sensorConnectionProperty.bEnableTriggerModeSync =  sensor->GetEnableExtSync();
                    sensorConnectionProperty.fFrameRate =  sensor->GetFrameRate();
                    sensorConnectionProperty.height = sensor->GetHeight();
                    sensorConnectionProperty.width = sensor->GetWidth();
                    sensorConnectionProperty.embeddedTop = sensor->GetEmbLinesTop();
                    sensorConnectionProperty.embeddedBot = sensor->GetEmbLinesBot();
/* coverity[autosar_cpp14_a16_0_1_violation] : intentional TID-2039 */
#if !(NV_IS_SAFETY)
                    sensorConnectionProperty.bEnableTPG = sensor->GetEnableTPG();
/* coverity[autosar_cpp14_a16_0_1_violation] : intentional TID-2039 */
#endif // !(NV_IS_SAFETY)
                    sensorConnectionProperty.sensorDescription = sensor->GetSensorDescription();
                    sensorConnectionProperty.vGpioMap.resize(
                        moduleInfo->serInfo.serdesGPIOPinMappings.size());
                    for (uint8_t i = 0U; i < moduleInfo->serInfo.serdesGPIOPinMappings.size(); i++){
                        sensorConnectionProperty.vGpioMap[i].sourceGpio =
                            moduleInfo->serInfo.serdesGPIOPinMappings[i].sourceGpio;

                        sensorConnectionProperty.vGpioMap[i].destGpio =
                            moduleInfo->serInfo.serdesGPIOPinMappings[i].destGpio;
                    }

                    m_upCameraModuleConnectionProperty->sensorConnectionProperty =
                        sensorConnectionProperty;

                    CNvMCameraModule::Property::EEPROMProperty const oEepromProperty {
                        .isEEPROMSupported = moduleInfo->isEEPROMSupported,
                    };
                    m_upCameraModuleProperty->eepromProperties = oEepromProperty;
                }
            }
        }
    }
    else {
        status = NVSIPL_STATUS_ERROR;
    }

    return status;
}

// Create broadcast serializer handle
/* coverity[autosar_cpp14_a8_4_4_violation] : intentional */
SIPLStatus CNvMMAX96712CameraModule::CreateBroadcastSerializer(
    uint8_t const linkIndex,
    std::unique_ptr<CNvMSerializer>& serializer)
{
    SIPLStatus status = NVSIPL_STATUS_OK;
    serializer.reset(nullptr);

    CNvMDevice::DeviceParams params = m_oDeviceParams;
    params.bUseNativeI2C = true;
    std::unique_ptr<CNvMSerializer> up_var = std::move(CreateNewSerializer());
    status = up_var->SetConfig(&m_oSerInfo, &params);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("Serializer SetConfig failed with SIPL error",
                            static_cast<int32_t>(status));
    } else {
        status = up_var->CreateCDIDevice(m_pCDIRoot, linkIndex);
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("Serializer SetConfig failed with SIPL error",
                                static_cast<int32_t>(status));
        } else {
            serializer.reset(up_var.release());
        }
    }
    return status;
}

/* coverity[autosar_cpp14_a8_4_4_violation] : intentional */
SIPLStatus CNvMMAX96712CameraModule::DoCreateBroadcastSensor(
    uint8_t const linkIndex,
    std::unique_ptr<CNvMSensor>& sensor)
{
    SIPLStatus status = NVSIPL_STATUS_OK;
    sensor.reset(nullptr);

    CNvMDevice::DeviceParams params = m_oDeviceParams;
    params.bUseNativeI2C = true;

    /* coverity[misra_cpp_2008_rule_18_4_1_violation] : intentional TID-1968 */
    std::unique_ptr<CNvMSensor> up_var(std::make_unique<CNvMSensor>());
    if (up_var == nullptr) {
        SIPL_LOG_ERR_STR("CNvMSensor allocation failed!");
        status = NVSIPL_STATUS_RESOURCE_ERROR;
    } else {
        status = up_var->SetConfig(&m_oSensorInfo, &params);
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("Sensor SetConfig failed with SIPL error",
                                static_cast<int32_t>(status));
        } else {
            status = up_var->SetDriverHandle(GetCDIDeviceDriver());
            if (status != NVSIPL_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("Sensor SetDriverHandle failed with SIPL error",
                                    static_cast<int32_t>(status));
            } else {
                status = up_var->SetDriverContext(GetCDIDeviceContext());
                if (status != NVSIPL_STATUS_OK) {
                    SIPL_LOG_ERR_STR_INT("Sensor SetDriverContext failed with SIPL error",
                                        static_cast<int32_t>(status));
                } else {
                    status = up_var->CreateCDIDevice(m_pCDIRoot, linkIndex);
                    if (status != NVSIPL_STATUS_OK) {
                        SIPL_LOG_ERR_STR_INT("Sensor CreateCDIDevice failed with SIPL error",
                                            static_cast<int32_t>(status));
                    } else {
                        sensor.reset(up_var.release());
                    }
                }
            }
        }
    }

    return status;
}

// Create serializer
SIPLStatus CNvMMAX96712CameraModule::CreateSerializer(
    uint8_t const linkIndex)
{
    SIPLStatus status {NVSIPL_STATUS_OK};

    status = m_upSerializer->CreateCDIDevice(m_pCDIRoot, linkIndex);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("Serializer CreateCDIDevice failed with SIPL error",
                              static_cast<int32_t>(status));
    }

    return status;
}

// Create Sensor
SIPLStatus CNvMMAX96712CameraModule::CreateSensor(
    SIPLStatus const instatus,
    uint8_t const linkIndex)
{
    SIPLStatus status = instatus;

    if(status == NVSIPL_STATUS_OK) {
        status = m_upSensor->CreateCDIDevice(m_pCDIRoot, linkIndex);
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("Sensor CreateCDIDevice failed with SIPL error",
                                  static_cast<int32_t>(status));
        }
    }

    return status;
}


// Create EEPROM
SIPLStatus CNvMMAX96712CameraModule::CreateEEPROM(
    SIPLStatus const instatus,
    uint8_t const linkIndex)
{
    SIPLStatus status = instatus;

    if(status == NVSIPL_STATUS_OK) {
        m_upCameraModuleConnectionProperty->sensorConnectionProperty.uSensorAddrs =
        m_upSensor->GetI2CAddr();
        if (m_upCameraModuleProperty->eepromProperties.isEEPROMSupported) {
            status = m_upEeprom->CreateCDIDevice(m_pCDIRoot, linkIndex);
            if (status != NVSIPL_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("EEPROM CreateCDIDevice failed with SIPL error",
                                      static_cast<int32_t>(status));
            } else {
                m_upCameraModuleConnectionProperty->eepromAddr = m_upEeprom->GetI2CAddr();
                m_upCameraModuleConnectionProperty->brdcstEepromAddr =
                m_upEeprom->GetNativeI2CAddr();
            }
        } else {
                m_upCameraModuleConnectionProperty->eepromAddr = static_cast<uint8_t>UINT8_MAX;
        }
    }

    return status;
}

// Create PMIC
SIPLStatus CNvMMAX96712CameraModule::CreatePMIC(
    SIPLStatus const instatus,
    uint8_t const linkIndex)
{
    SIPLStatus status = instatus;

    if ((status == NVSIPL_STATUS_OK) &&
        m_upCameraModuleConnectionProperty->sensorConnectionProperty.pmicProperty.isSupported) {
        if (m_upCameraModuleConnectionProperty->sensorConnectionProperty.pmicProperty.i2cAddress !=
            I2C_ADDRESS_0x0) {

            /* coverity[misra_cpp_2008_rule_18_4_1_violation] : intentional TID-1968 */
            m_upPmic.reset(new(std::nothrow) CNvMPMIC());
            if (m_upPmic == nullptr) {
                SIPL_LOG_ERR_STR("CNvMPMIC allocation failed!");
                status = NVSIPL_STATUS_RESOURCE_ERROR;
            } else {
                status = m_upPmic->SetConfig(
                    m_upCameraModuleConnectionProperty->
                    sensorConnectionProperty.pmicProperty.i2cAddress,
                    &m_oDeviceParams);
                if (status != NVSIPL_STATUS_OK) {
                    SIPL_LOG_ERR_STR_INT("PMIC SetConfig failed with SIPL error",
                                        static_cast<int32_t>(status));
                } else {
                    status = m_upPmic->SetDriverHandle(GetCDIDeviceDriver
                                                       (ModuleComponent::MODULE_COMPONENT_PMIC));
                    if (status != NVSIPL_STATUS_OK) {
                        SIPL_LOG_ERR_STR_INT("PMIC SetDriverHandle failed with SIPL error",
                                            static_cast<int32_t>(status));
                    } else {
                        status = m_upPmic->CreateCDIDevice(m_pCDIRoot, linkIndex);
                        if (status != NVSIPL_STATUS_OK) {
                            SIPL_LOG_ERR_STR_INT("PMIC CreateCDIDevice failed with SIPL error",
                                                 static_cast<int32_t>(status));
                        }
                    }
                }
            }
        } else {
            SIPL_LOG_ERR_STR("Invalid PMIC I2C address (0)");
            status = NVSIPL_STATUS_BAD_ARGUMENT;
        }
    }

    return status;
}

// Create VCSEL
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
SIPLStatus CNvMMAX96712CameraModule::CreateVCSEL(
    SIPLStatus const instatus,
    uint8_t const linkIndex)
{
    SIPLStatus status = instatus;

    if ((status == NVSIPL_STATUS_OK) &&
        m_upCameraModuleConnectionProperty->sensorConnectionProperty.vcselProperty.isSupported) {
        if (m_upCameraModuleConnectionProperty->sensorConnectionProperty.vcselProperty.i2cAddress !=
            I2C_ADDRESS_0x0) {

            /* coverity[misra_cpp_2008_rule_18_4_1_violation] : intentional TID-1968 */
            m_upVcsel.reset(new(std::nothrow) CNvMVCSEL());
            if (m_upVcsel == nullptr) {
                SIPL_LOG_ERR_STR("CNvMVCSEL allocation failed!");
                status = NVSIPL_STATUS_RESOURCE_ERROR;
            } else {
                status = m_upVcsel->SetConfig(
                    m_upCameraModuleConnectionProperty->
                    sensorConnectionProperty.vcselProperty.i2cAddress,
                    &m_oDeviceParams);
                if (status != NVSIPL_STATUS_OK) {
                    SIPL_LOG_ERR_STR_INT("VCSEL SetConfig failed with SIPL error",
                                        static_cast<int32_t>(status));
                } else {
                    status = m_upVcsel->SetDriverHandle(GetCDIDeviceDriver(
                                                        ModuleComponent::MODULE_COMPONENT_VCSEL));
                    if (status != NVSIPL_STATUS_OK) {
                        SIPL_LOG_ERR_STR_INT("VCSEL SetDriverHandle failed with SIPL error",
                                            static_cast<int32_t>(status));
                    } else {
                        status = m_upVcsel->CreateCDIDevice(m_pCDIRoot, linkIndex);
                        if (status != NVSIPL_STATUS_OK) {
                            SIPL_LOG_ERR_STR_INT("VCSEL CreateCDIDevice failed with SIPL error",
                                static_cast<int32_t>(status));
                        }
                    }
                }
            }
        } else {
            SIPL_LOG_ERR_STR("Invalid VCSEL I2C address (0)");
            status = NVSIPL_STATUS_BAD_ARGUMENT;
        }
    }

    return status;
}

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

 /** @brief Modify state of Deserializer Error notifying pins ERRB and LOCK
     * This is required for Power switch SM.
     *
     * @param[in] handleDeser : Deserializer Devblk handle
     * @param[in] state : true to enable and false to disable Deser error pins
     * @return NVSIPL_STATUS_OK : on completion
     * @return NVSIPL_STATUS_ERROR : on failure
     * @return NVSIPL_STATUS_BAD_ARGUMENT : on incorrect arguments passed
     */
static SIPLStatus SetDeserErrorConfig(
    DevBlkCDIDevice const* const handleDeser,
    bool const state)
{
    NvMediaStatus nvmStatus {NVMEDIA_STATUS_ERROR};
    // Using braced initialization for AUTOSAR A8-5-2
    ConfigSetsMAX96712 const lock_state {(state ?
                                            CDI_CONFIG_MAX96712_ENABLE_LOCK:
                                            CDI_CONFIG_MAX96712_DISABLE_LOCK)};

    // Modifying state of ERRB and LOCK pin for powerswitch Init
    nvmStatus = MAX96712SetDeviceConfig(handleDeser, lock_state);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPLStatus const status {ConvertNvMediaStatus(nvmStatus)};
        SIPL_LOG_ERR_STR_2INT("SetDeserErrorConfig LOCK failed with error and "
            "state", static_cast<int32_t>(status), static_cast<int32_t>(state));
        return status;
    }

    ConfigSetsMAX96712 const errb_state {(state ?
                                            CDI_CONFIG_MAX96712_ENABLE_ERRB:
                                            CDI_CONFIG_MAX96712_DISABLE_ERRB)};
    nvmStatus = MAX96712SetDeviceConfig(handleDeser, errb_state);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPLStatus const status {ConvertNvMediaStatus(nvmStatus)};
        SIPL_LOG_ERR_STR_2INT("SetDeserErrorConfig ERRB failed with error and "
            "state", static_cast<int32_t>(status), static_cast<int32_t>(state));

        /* If DisableERRB fails, then reenable LOCK so as to return to original
           configuration and finally return */
        if (lock_state == CDI_CONFIG_MAX96712_DISABLE_LOCK) {
            nvmStatus = MAX96712SetDeviceConfig(handleDeser,
                                            CDI_CONFIG_MAX96712_ENABLE_LOCK);
            if (nvmStatus != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("Enabling LOCK failed with NvMedia status",
                                    static_cast<int32_t>(nvmStatus));
            }
        }
        return status;
    }
    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMMAX96712CameraModule::CreateCameraPowerLoadSwitch(
    SIPLStatus const instatus,
    uint8_t const linkIndex)
{
    SIPLStatus status = instatus;

    if (status == NVSIPL_STATUS_OK) {
        if ((linkIndex >= MAX_POWER_LINKS_PER_BLOCK) ||
            (m_interfaceType >= NVSIPL_CAP_CSI_INTERFACE_TYPE_MAX)) {
            SIPL_LOG_ERR_STR_2UINT("CNvMMAX96712CameraModule:: CreateCameraPowerLoadSwitch "
                "Incorrect linkIndex or m_interfaceType passed: ",
                static_cast<uint32_t>(linkIndex), static_cast<uint32_t>(m_interfaceType));
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }
        if ((!m_oDeviceParams.bPassive) && (!m_oDeviceParams.bEnableSimulator)) {
            // Get power control information
            NvMediaStatus const nvmStatus =
            DevBlkCDIGetCamPowerControlInfo(m_upSensor->GetCDIDeviceHandle(), &m_camPwrControlInfo);
            if (nvmStatus != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("Get power control info failed with NVMEDIA error",
                                      static_cast<int32_t>(nvmStatus));
                status = ConvertNvMediaStatus(nvmStatus);
            } else {
                if ((m_camPwrControlInfo.method != CAM_PWR_METHOD_0) &&
                    (m_camPwrControlInfo.method != static_cast<uint8_t>(UINT8_MAX))) {
                    CNvMDevice::DeviceParams params = m_oDeviceParams;
                    params.bUseNativeI2C = true;
                    m_upCampwr = CNvMMax20087DriverFactory::RequestPowerDriver(m_pCDIRoot,
                                                              m_camPwrControlInfo.links[linkIndex]);
                    if (m_upCampwr == nullptr) {
                        SIPL_LOG_ERR_STR("Camera power load switch failed to request driver");
                        status = NVSIPL_STATUS_ERROR;
                        return status;
                    }
                    status = m_upCampwr->SetConfig(m_camPwrControlInfo.i2cAddr, &params);
                    if (status != NVSIPL_STATUS_OK) {
                        SIPL_LOG_ERR_STR_INT("Camera power load switch SetConfig "
                                             "failed with SIPL error",
                            static_cast<int32_t>(status));
                        return status;
                    }
                    status = m_upCampwr->CreatePowerDevice(m_pCDIRoot,
                                                           m_camPwrControlInfo.links[linkIndex]);
                    if (status != NVSIPL_STATUS_OK) {
                        SIPL_LOG_ERR_STR_INT("Camera power load switch CreateDevice "
                                             "failed with SIPL error",
                            static_cast<int32_t>(status));
                        return status;
                    }
                    if (!m_oDeviceParams.bCBAEnabled) {
                        status = m_upCampwr->CheckPresence(m_pCDIRoot,
                                                           m_upCampwr->GetDeviceHandle());
                        if (status != NVSIPL_STATUS_OK) {
                            SIPL_LOG_ERR_STR_INT("Camera power load switch CheckPresence "
                                                 "failed with SIPL error",
                                static_cast<int32_t>(status));
                            return status;
                        }

                        status = m_pDeserializer->SetPower(true);
                        if (status != NVSIPL_STATUS_OK) {
                            SIPL_LOG_ERR_STR_INT("Deserializer SetPower failed with SIPL error",
                                                static_cast<int32_t>(status));
                            return status;
                        }
                        std::this_thread::sleep_for<>(std::chrono::milliseconds(2));
                        /**
                         * Deserializer pins which notify ERRB and LOCK errors is mapped to Orin
                         * gpio which is shared with powerswitch. We need to disable deserializer
                         * error config prior to checking the functionality of INT pin of
                         * powerswitch, and restore the configuration on completion.
                         *
                         * If caller pass 'true', enable the deserializer error config,
                         * else disable them
                         **/
                        DevBlkCDIDevice const* const handleDeser {
                                        m_pDeserializer->GetCDIDeviceHandle()};
                        status = SetDeserErrorConfig(handleDeser, false);
                        if (status != NVSIPL_STATUS_OK) {
                            return status;
                        }
                        SIPLStatus const status2 {m_upCampwr->InitPowerDevice(
                                                    m_pCDIRoot,
                                                    m_upCampwr->GetDeviceHandle(),
                                                    m_camPwrControlInfo.links[linkIndex],
                                                    CsiPortForType(m_interfaceType))};
                        /** Ensure Deser error config is set regardless of
                         *  m_upCampwr init's success. */
                        status = SetDeserErrorConfig(handleDeser, true);
                        if (status != NVSIPL_STATUS_OK) {
                            return status;
                        }
                        if (status2 != NVSIPL_STATUS_OK) {
                            SIPL_LOG_ERR_STR_INT("Camera power load switch"
                                " InitDevice failed with SIPL error",
                                static_cast<int32_t>(status2));
                            return status2;
                        }
                    }
                }
            }
        }
    }
    return status;
}

// Config transport link
SIPLStatus CNvMMAX96712CameraModule::CreateTransport(
    SIPLStatus const instatus)
{
    SIPLStatus status = instatus;

    if(status == NVSIPL_STATUS_OK) {
        m_upTransport = std::move(CreateNewTransportLink());
        CNvMTransportLink::LinkParams params;

        params.pSerCDIDevice = m_upSerializer->GetCDIDeviceHandle();
        params.pDeserCDIDevice = m_pDeserializer->GetCDIDeviceHandle();
        params.ulinkIndex = m_linkIndex;
        params.uBrdcstSerAddr = m_upSerializer->GetNativeI2CAddr();
        params.uSerAddr = m_upSerializer->GetI2CAddr();
        params.moduleConnectionProperty = *m_upCameraModuleConnectionProperty;
        params.bEnableSimulator = m_oDeviceParams.bEnableSimulator;
        params.bPassive = m_oDeviceParams.bPassive;
        params.bCBAEnabled = m_oDeviceParams.bCBAEnabled;
        params.m_groupInitProg = m_groupInitProg;

        status = m_upTransport->SetConfig(params);
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("TransportLink SetConfig failed with SIPL error",
                                  static_cast<int32_t>(status));
        }
    }

    return status;
}

/* coverity[autosar_cpp14_m0_1_10_violation] : intentional TID-2013 */
SIPLStatus
CNvMMAX96712CameraModule::DoCreateCDIDevice(
    DevBlkCDIRootDevice *const cdiRoot, uint8_t const linkIndex)
{
    SIPLStatus status = NVSIPL_STATUS_OK;

    m_pCDIRoot = cdiRoot;

    status = CreateSerializer(linkIndex);
    status = CreateSensor( status, linkIndex );
    status = CreateEEPROM( status, linkIndex );
    status = CreatePMIC( status, linkIndex);
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    status = CreateVCSEL( status, linkIndex);
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
    status = CreateCameraPowerLoadSwitch( status, linkIndex);
    status = CreateTransport( status );

    return status;
}

/* coverity[autosar_cpp14_m0_1_10_violation] : intentional TID-2013 */
uint16_t
CNvMMAX96712CameraModule::GetPowerOnDelayMs()
{
    uint16_t statusPwrOnDelay;

    if ((CNvMDeserializer::LinkMode::LINK_MODE_GMSL2_6GBPS == GetLinkMode()) ||
        (CNvMDeserializer::LinkMode::LINK_MODE_GMSL2_3GBPS == GetLinkMode())) {
        /* GMSL2 Link lock time maximum is 100ms*/
        statusPwrOnDelay = PWR_ON_DELAY_100_MSEC;
    } else {
        /* GMSL1 Link lock time 20ms, I2C wake time is typical 1.1ms after releasing PWDN */
        statusPwrOnDelay = PWR_ON_DELAY_20_MSEC;
    }
    return statusPwrOnDelay;
}

/* coverity[autosar_cpp14_m0_1_10_violation] : intentional TID-2013 */
uint16_t
CNvMMAX96712CameraModule::GetPowerOffDelayMs() noexcept
{
    return PWR_OFF_DELAY_0_MSEC;
}

/* coverity[misra_cpp_2008_rule_0_1_10_violation] : intentional TID-1498 */
/* coverity[autosar_cpp14_m0_1_10_violation] : intentional TID-2013 */
std::string
CNvMMAX96712CameraModule::GetSupportedDeserailizer() noexcept
{
    return "MAX96712";
}

/* coverity[misra_cpp_2008_rule_0_1_10_violation] : intentional TID-1498 */
/* coverity[autosar_cpp14_m0_1_10_violation] : intentional TID-2013 */
const CNvMCameraModule::Version*
CNvMCameraModule_GetVersion() noexcept
{
    static constexpr CNvMCameraModule::Version version_var{};
    return &version_var;
}

/* coverity[misra_cpp_2008_rule_0_1_10_violation] : intentional TID-1498 */
/* coverity[autosar_cpp14_m0_1_10_violation] : intentional TID-2013 */
SIPLStatus
CNvMCameraModule_SetDebugLevel(
    INvSIPLDeviceBlockTrace::TraceLevel const level) noexcept
{
/* coverity[autosar_cpp14_a16_0_1_violation] : intentional TID-2039 */
#if !(NV_IS_SAFETY)
    // Set the trace level used by the camera module files
    INvSIPLDeviceBlockTrace *instance = INvSIPLDeviceBlockTrace::GetInstance();
    if (instance != nullptr) {
        instance->SetLevel(level);
    }
/* coverity[autosar_cpp14_a16_0_1_violation] : intentional TID-2039 */
#endif //!(NV_IS_SAFETY)
    static_cast<void> (level);
    return NVSIPL_STATUS_OK;
}

/* coverity[autosar_cpp14_m0_1_10_violation] : intentional TID-2013 */
SIPLStatus
CNvMMAX96712CameraModule::GetSerializerErrorSize(size_t & serializerErrorSize)
{
    SIPLStatus status = NVSIPL_STATUS_OK;
    status = m_upSerializer->GetErrorSize(serializerErrorSize);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: Serializer GetErrorSize failed with SIPL error",
                              static_cast<int32_t>(status));
    }
    return status;
}

/* coverity[autosar_cpp14_m0_1_10_violation] : intentional TID-2013 */
SIPLStatus
CNvMMAX96712CameraModule::GetSensorErrorSize(size_t & sensorErrorSize)
{
    SIPLStatus status = NVSIPL_STATUS_OK;
    status = GetErrorSize(sensorErrorSize);
    return status;
}

/* coverity[autosar_cpp14_m0_1_10_violation] : intentional TID-2013 */
SIPLStatus
CNvMMAX96712CameraModule::GetSerializerErrorInfo(std::uint8_t * const buffer,
                                                 std::size_t const bufferSize,
                                                 std::size_t &size)
{
    SIPLStatus status = NVSIPL_STATUS_OK;

    status = m_upSerializer->GetErrorInfo(buffer, bufferSize, size);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: Serializer GetErrorInfo failed with SIPL error",
                              static_cast<int32_t>(status));
    }
    return status;
}

/* coverity[autosar_cpp14_m0_1_10_violation] : intentional TID-2013 */
SIPLStatus
CNvMMAX96712CameraModule::GetSensorErrorInfo(std::uint8_t * const buffer,
                                             std::size_t const bufferSize,
                                             std::size_t &size)
{
    SIPLStatus status = NVSIPL_STATUS_OK;
    status = GetErrorInfo(buffer, bufferSize, size);
    return status;
}

/* coverity[autosar_cpp14_m0_1_10_violation] : intentional TID-2013 */
SIPLStatus
CNvMMAX96712CameraModule::NotifyLinkState(NotifyLinkStates const linkState)
{
    SIPLStatus status = NVSIPL_STATUS_OK;

    switch (linkState) {
        case NotifyLinkStates::ENABLED:
            status = Start();
            if (status != NVSIPL_STATUS_OK) {
                SIPL_LOG_ERR_STR("MAX96712: Failed to start on link state enabled");
            } else {
                status = PostStart();
                if (status != NVSIPL_STATUS_OK) {
                    SIPL_LOG_ERR_STR("MAX96712: finalize start failed");
                }
            }
            break;
        case NotifyLinkStates::PENDING_DISABLE:
            status = Stop();
            if (status != NVSIPL_STATUS_OK) {
                SIPL_LOG_ERR_STR("MAX96712: Failed to stop on link state pending disabled");
            }
            break;
        default:
            status = NVSIPL_STATUS_BAD_ARGUMENT;
            SIPL_LOG_ERR_STR("MAX96712: Invalid link state notification");
            break;
    }

    return status;
}

/* coverity[autosar_cpp14_m0_1_10_violation] : intentional TID-2013 */
Interface* CNvMMAX96712CameraModule::GetInterface(UUID const &interfaceId) {
    static_cast<void>(interfaceId);
    SIPL_LOG_ERR_STR("CNvMMAX96712CameraModule::GetInterface not implemented for sensor");
    return nullptr;
}

/* coverity[autosar_cpp14_m0_1_10_violation] : intentional TID-2013 */
SIPLStatus CNvMMAX96712CameraModule::DoSetPower(bool const powerOn)
{
    SIPLStatus status = NVSIPL_STATUS_ERROR;

    if (m_camPwrControlInfo.method == CAM_PWR_METHOD_0) {
        // Default is NvCCP, other power backends can be used here based on platform/usecase.
        status = PowerControlSetUnitPower(m_pwrPort, m_linkIndex, powerOn);
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("CNvMMAX96712CameraModule::DoSetPower failed with SIPL error",
                                  static_cast<int32_t>(status));
        }
    } else if (static_cast<int32_t>(m_camPwrControlInfo.method) == UINT8_MAX) {
        status = NVSIPL_STATUS_OK;
    } else {
        if (m_upCampwr != nullptr) {
            if (m_upCampwr->isSupported() == NVSIPL_STATUS_OK) {
                if (m_linkIndex >= MAX_POWER_LINKS_PER_BLOCK) {
                    status = NVSIPL_STATUS_BAD_ARGUMENT;
                    SIPL_LOG_ERR_STR("CNvMMAX96712CameraModule:: DoSetPower "
                        "m_linkIndex exceeds MAX Allowed links per block");
                } else {
                    status = m_upCampwr->PowerControlSetUnitPower(m_upCampwr->GetDeviceHandle(),
                                m_camPwrControlInfo.links[m_linkIndex], powerOn);
                    if (status != NVSIPL_STATUS_OK) {
                        SIPL_LOG_ERR_STR_INT("CNvMMAX96712CameraModule::DoSetPower "
                                "failed with SIPL error", static_cast<int32_t>(status));
                    } else {
                        /* Mask all power switch interrupts.
                         * It will be restored in EnableLinkAndDetect API
                         * DES SM-26 needs to mask interrupt before running it and restore
                         * original mask after it, so below sequence is followed:
                         *  1. Masking Interrupts in DoSetPower()
                         *  2. Executing SM26 in Deserializer DoInit()
                         *  3. Restoring interrupts in EnableLinkAndDetect()
                         */
                        status = MaskRestorePowerSwitchInterrupt(true);
                    }
                }
            }
        } else {
            status = NVSIPL_STATUS_BAD_ARGUMENT;
            SIPL_LOG_ERR_STR("CNvMMAX96712CameraModule:: DoSetPower "
                "m_upCampwr is NULL");
        }
    }

    return status;
}

/* coverity[autosar_cpp14_m0_1_10_violation] : intentional TID-2013 */
SIPLStatus CNvMMAX96712CameraModule::GetInterruptStatus(uint32_t const gpioIdx,
                                                        IInterruptNotify &intrNotifier) noexcept
{
    static_cast<void>(gpioIdx);
    static_cast<void>(intrNotifier);
    return NVSIPL_STATUS_NOT_SUPPORTED;
}

/** Mask interrupt or restore power switch interrupts */
SIPLStatus
CNvMMAX96712CameraModule::MaskRestorePowerSwitchInterrupt(const bool enable)
{
    SIPLStatus status {NVSIPL_STATUS_OK};

/* Restore interrupt mask only on QNX since DES SM26 needs to run on QNX. */
#if defined(NVMEDIA_QNX)
    if (((!m_oDeviceParams.bPassive && !m_oDeviceParams.bEnableSimulator) &&
        (m_camPwrControlInfo.method != CAM_PWR_METHOD_0)) &&
        (static_cast<int32_t>(m_camPwrControlInfo.method) != UINT8_MAX)) {

        if (m_upCampwr != nullptr) {
            if (m_upCampwr->isSupported() == NVSIPL_STATUS_OK) {
                status = m_upCampwr->MaskRestoreInterrupt(enable);
                if (status != NVSIPL_STATUS_OK) {
                    SIPL_LOG_ERR_STR_INT("MaskRestorePowerSwitchInterrupt::"
                        "failed to mask or restore interrupt", static_cast<int32_t>(status));
                }
            }
        } else {
            status = NVSIPL_STATUS_BAD_ARGUMENT;
            SIPL_LOG_ERR_STR("MaskRestorePowerSwitchInterrupt: null m_upCampwr");
        }
    }
#endif

    return status;
}

} // end of namespace
