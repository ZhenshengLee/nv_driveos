/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
 * Copyright (c) 2021-2022, OmniVision Technologies.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef _CNVMMAX96712_9295_OV2311_HPP_
#define _CNVMMAX96712_9295_OV2311_HPP_

#include "cameramodule/MAX96712cameramodule/CNvMMAX96712_9295CameraModule.hpp"
#include "OV2311NonFuSaCustomInterface.hpp"

namespace nvsipl
{

class CNvMMAX96712_9295_OV2311 : public CNvMMAX96712_9295CameraModule,
public OV2311NonFuSaCustomInterface
{
protected:
    SIPLStatus SetConfigModule(const SensorInfo *sensorInfo, CNvMDevice::DeviceParams *params) override;

    SIPLStatus DetectModule() override;

    SIPLStatus InitModule() override;

    SIPLStatus StartModule() override;

    SIPLStatus StopModule() override;

    SIPLStatus DeinitModule() override;

    DevBlkCDIDeviceDriver *GetCDIDeviceDriver() override;

    void SetSensorConnectionProperty(CNvMCameraModuleCommon::ConnectionProperty::SensorConnectionProperty *sensorConnectionProperty) override;

    std::unique_ptr<CNvMDevice::DriverContext> GetCDIDeviceContext() override;

    //! Get PowerOff Delay ms value
    virtual uint16_t GetPowerOffDelayMs(void) final override;

    Interface* GetInterface(const UUID &interfaceId) override;

    SIPLStatus SetCustomValue(std::uint32_t const valueToSet) override;

    SIPLStatus GetCustomValue(std::uint32_t * const valueToGet) override;

    SIPLStatus CheckModuleStatus() override;

    int m_ConfigIndex;

private:
    SIPLStatus InitSimulatorAndPassive();
};

} // end of namespace

#endif /* _CNVMMAX96712_9295_OV2311_HPP_ */
