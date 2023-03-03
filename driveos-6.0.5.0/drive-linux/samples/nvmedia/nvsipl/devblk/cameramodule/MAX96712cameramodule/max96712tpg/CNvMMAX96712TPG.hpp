/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef _CNVMMAX96712TPG_HPP_
#define _CNVMMAX96712TPG_HPP_

#include "ModuleIF/CNvMCameraModule.hpp"
#include "SensorIF/CNvMSensor.hpp"

namespace nvsipl
{

class CNvMMAX96712TPG : public CNvMCameraModule
{
public:
    SIPLStatus DoSetConfig(const CameraModuleConfig *cameraModuleConfig, const uint8_t linkIndex) override final;

    SIPLStatus DoSetPower(bool powerOn) override final;

    virtual SIPLStatus Init() override final;

    virtual SIPLStatus PostInit() override final {
        return NVSIPL_STATUS_OK;
    };

    virtual SIPLStatus Reconfigure() {
        return NVSIPL_STATUS_NOT_SUPPORTED;
    };

    virtual SIPLStatus ReadEEPROMData(const std::uint16_t address,
                                               const std::uint32_t length,
                                               std::uint8_t * const buffer) override final
    {
        return NVSIPL_STATUS_NOT_SUPPORTED;
    };


#if !NV_IS_SAFETY
    SIPLStatus WriteEEPROMData(const std::uint16_t address,
                               const std::uint32_t length,
                               std::uint8_t * const buffer) override final
    {
        return NVSIPL_STATUS_NOT_SUPPORTED;
    }
#endif // !NV_IS_SAFETY

    virtual SIPLStatus ToggleLED(bool enable) override
    {
        return NVSIPL_STATUS_NOT_SUPPORTED;
    }

    virtual SIPLStatus Start() override
    {
        return NVSIPL_STATUS_OK;
    };

    virtual SIPLStatus Stop() override
    {
        return NVSIPL_STATUS_OK;
    };

    virtual SIPLStatus Deinit() override
    {
        return NVSIPL_STATUS_OK;
    };

    Property* GetCameraModuleProperty() override final;

    uint16_t GetPowerOnDelayMs() override final;

    uint16_t GetPowerOffDelayMs() override final;

    std::string GetSupportedDeserailizer() override final;

    CNvMDeserializer::LinkMode GetLinkMode() override final;

    SIPLStatus GetSerializerErrorSize(size_t & serializerErrorSize) override final;

    SIPLStatus GetSensorErrorSize(size_t & sensorErrorSize) override final;

    SIPLStatus GetSerializerErrorInfo(
        std::uint8_t * const buffer,
        const std::size_t bufferSize,
        std::size_t &size) override final;

    SIPLStatus GetSensorErrorInfo(
        std::uint8_t * const buffer,
        const std::size_t bufferSize,
        std::size_t &size) override final;

    SIPLStatus NotifyLinkState(const NotifyLinkStates linkState) override final;
    Interface* GetInterface(const UUID &interfaceId) override final;

    SIPLStatus GetInterruptStatus(
        const uint32_t gpioIdx,
        IInterruptNotify &intrNotifier) override;

private:
    uint8_t m_initLinkMask;

    bool m_bPassive;

    CNvMDeserializer *m_pDeserializer;

    std::unique_ptr<CNvMSensor> m_upSensor;

    std::unique_ptr<Property> m_upCameraModuleProperty;

    std::string m_sModuleName;
};

} // end of namespace

#endif /* _CNVMMAX96712TPG_HPP_ */
