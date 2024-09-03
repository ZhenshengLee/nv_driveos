/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef _CNVMMAX96712CAMERAMODULE_HPP_
#define _CNVMMAX96712CAMERAMODULE_HPP_

#include "ModuleIF/CNvMCameraModule.hpp"
#include "SensorIF/CNvMSensor.hpp"
#include "EEPROMIF/CNvMEEPROM.hpp"
#include "SerializerIF/CNvMSerializer.hpp"
#include "TransportLinkIF/CNvMTransportLink.hpp"
#include "NvSIPLPlatformCfg.hpp"
#include "CNvMCameraModuleCommon.hpp"
#include "CampwrIF/CNvMCampwr.hpp"

namespace nvsipl
{

class CNvMMAX96712CameraModule : public CNvMCameraModule
{
public:
    SIPLStatus DoSetConfig(const CameraModuleConfig *cameraModuleConfig, const uint8_t linkIndex) override;

    SIPLStatus DoSetPower(bool powerOn) override;

    SIPLStatus EnableLinkAndDetect() override final;

    virtual CNvMDevice *CreateBroadcastSensor() final;

    virtual SIPLStatus Init();

    virtual SIPLStatus PostInit();

    virtual SIPLStatus Start();

    virtual SIPLStatus Stop();

    virtual SIPLStatus Reconfigure() override final;

    virtual SIPLStatus ReadEEPROMData(const std::uint16_t address,
                              const std::uint32_t length,
                              std::uint8_t * const buffer) override;
#if !NV_IS_SAFETY
    virtual SIPLStatus WriteEEPROMData(const std::uint16_t address,
                               const std::uint32_t length,
                               std::uint8_t * const buffer) override;

    SIPLStatus ToggleLED(bool enable) override;
#endif // !NV_IS_SAFETY

    SIPLStatus Deinit() override final;

    Property* GetCameraModuleProperty() override final;

    uint16_t GetPowerOnDelayMs() override;

    uint16_t GetPowerOffDelayMs() override;

    virtual std::string GetSupportedDeserailizer() override;

    SIPLStatus GetSerializerErrorSize(size_t & serializerErrorSize) override final;

    SIPLStatus GetSensorErrorSize(size_t & sensorErrorSize) override final;

    SIPLStatus GetSerializerErrorInfo(std::uint8_t * const buffer,
                                      const std::size_t bufferSize,
                                      std::size_t &size) override final;

    SIPLStatus GetSensorErrorInfo(std::uint8_t * const buffer,
                                  const std::size_t bufferSize,
                                  std::size_t &size) override final;

    SIPLStatus NotifyLinkState(const NotifyLinkStates linkState) override final;

    Interface* GetInterface(const UUID &interfaceId) override;

    SIPLStatus GetInterruptStatus(
        const uint32_t gpioIdx,
        IInterruptNotify &intrNotifier) override;

protected:
    enum ModuleComponent {
        MODULE_COMPONENT_SENSOR = 0,
        MODULE_COMPONENT_EEPROM
    };

    std::unique_ptr<CNvMSensor> m_upSensor;

    std::unique_ptr<CNvMEEPROM> m_upEeprom = NULL;

    std::unique_ptr<CNvMCampwr> m_upCampwr = NULL;

    std::unique_ptr<CNvMSerializer> m_upSerializer;

    std::unique_ptr<CNvMTransportLink> m_upTransport;

    std::unique_ptr<Property> m_upCameraModuleProperty;

    std::unique_ptr<CNvMCameraModuleCommon::ConnectionProperty> m_upCameraModuleConnectionProperty;

    std::unique_ptr<CNvMSensor> m_broadcastSensor;

    uint8_t m_linkIndex;

    uint8_t m_initLinkMask;

    DevBlkCDIPowerControlInfo m_camPwrControlInfo = {UINT8_MAX, 0};

    bool m_groupInitProg; // Indicate if the homogeneous camera support enabled or not

    //! Device params needs to be cached to support reinitialization
    CNvMDevice::DeviceParams m_oDeviceParams;

    SerInfo m_oSerInfo;

    SensorInfo m_oSensorInfo;

    std::string m_sModuleName;

    DevBlkCDIRootDevice* m_pCDIRoot;

    CNvMDeserializer *m_pDeserializer;

    /** stores interfaceType for CSI Port Information */
    NvSiplCapInterfaceType m_interfaceType;

    virtual void SetSensorConnectionProperty(
        CNvMCameraModuleCommon::ConnectionProperty::SensorConnectionProperty *sensorConnectionProperty);

    virtual SIPLStatus SetConfigModule(const SensorInfo *sensorInfo, CNvMDevice::DeviceParams *params) = 0;

    virtual SIPLStatus DetectModule() = 0;

    virtual SIPLStatus InitModule() = 0;

    virtual SIPLStatus StartModule() = 0;

    virtual SIPLStatus StopModule() = 0;

    virtual SIPLStatus DeinitModule() = 0;

    virtual DevBlkCDIDeviceDriver *GetCDIDeviceDriver() = 0;

    virtual DevBlkCDIDeviceDriver *GetCDIDeviceDriver(ModuleComponent component);

    virtual std::unique_ptr<CNvMDevice::DriverContext> GetCDIDeviceContext() = 0;

    virtual std::unique_ptr<CNvMSerializer> CreateNewSerializer() = 0;

    virtual std::unique_ptr<CNvMTransportLink> CreateNewTransportLink() = 0;

    std::unique_ptr<CNvMSerializer> CreateBroadcastSerializer(const uint8_t linkIndex);

    virtual SIPLStatus GetErrorSize(size_t & sensorErrorSize);

    virtual SIPLStatus GetErrorInfo(std::uint8_t * const buffer,
                                    const std::size_t bufferSize,
                                    std::size_t &size);

private:
    virtual SIPLStatus DoCreateCDIDevice(DevBlkCDIRootDevice* cdiRoot, const uint8_t linkIndex) override final;

    void SetupConnectionProperty(const CameraModuleConfig *cameraModuleConfig, const uint8_t linkIndex);

    std::unique_ptr<CNvMSensor> DoCreateBroadcastSensor(const uint8_t linkIndex);

};


} // end of namespace

#endif /* _CNVMMAX96712CAMERAMODULE_HPP_ */
