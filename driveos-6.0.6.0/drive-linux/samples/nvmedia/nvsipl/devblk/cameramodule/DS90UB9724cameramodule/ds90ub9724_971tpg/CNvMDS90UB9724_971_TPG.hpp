/* Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef _CNvMDS90UB9724_971_TPG_HPP_
#define _CNvMDS90UB9724_971_TPG_HPP_

#include "cameramodule/DS90UB9724cameramodule/CNvMDS90UB9724_971CameraModule.hpp"

namespace nvsipl
{

class CNvMDS90UB9724_971_TPG : public CNvMDS90UB9724_971CameraModule
{
protected:
    SIPLStatus SetConfigModule(const SensorInfo *sensorInfo, CNvMDevice::DeviceParams *params) override;

    SIPLStatus ToggleLED(bool enable) override;

    SIPLStatus DetectModule() override;

    SIPLStatus InitModule() override;

    SIPLStatus StartModule() override;

    SIPLStatus StopModule() override;

    SIPLStatus DeinitModule() override;

    SIPLStatus DoSetPower(bool powerOn) override;

    DevBlkCDIDeviceDriver *GetCDIDeviceDriver() override;

    std::unique_ptr<CNvMDevice::DriverContext> GetCDIDeviceContext() override;

    SIPLStatus GetInterruptStatus(
        const uint32_t gpioIdx,
        IInterruptNotify &intrNotifier) override;

    int m_ConfigIndex;

private:
    SIPLStatus InitSimulatorAndPassive();
};

} // end of namespace

#endif /* _CNvMDS90UB9724_971_TPG_HPP_ */
