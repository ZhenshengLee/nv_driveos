/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "CNvMDS90UB9724_971_TPG.hpp"
#include "cameramodule/DS90UB9724cameramodule/CNvMTransportLink_DS90UB9724_971.hpp"
#include "ModuleIF/CNvMCameraModuleExport.hpp"
#include "sipl_error.h"

extern "C" {
#include "cdi_ds90ub971_tpg.h"
#include "pwr_utils.h"
}

namespace nvsipl
{
SIPLStatus CNvMDS90UB9724_971_TPG::SetConfigModule(const SensorInfo *sensorInfo, CNvMDevice::DeviceParams *params) {
    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMDS90UB9724_971_TPG::ToggleLED(bool enable) {
	return NVSIPL_STATUS_NOT_SUPPORTED;
}

SIPLStatus CNvMDS90UB9724_971_TPG::DetectModule() {
    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMDS90UB9724_971_TPG::InitModule() {
    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMDS90UB9724_971_TPG::StartModule() {
    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMDS90UB9724_971_TPG::StopModule() {
    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMDS90UB9724_971_TPG::DeinitModule() {
    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMDS90UB9724_971_TPG::DoSetPower(bool powerOn)
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

DevBlkCDIDeviceDriver *CNvMDS90UB9724_971_TPG::GetCDIDeviceDriver() {
    return GetDS90UB971TPGDriver();
}

std::unique_ptr<CNvMDevice::DriverContext> CNvMDS90UB9724_971_TPG::GetCDIDeviceContext() {
    auto driverContext = new CNvMDevice::DriverContextImpl<ContextDS90UB971TPG>();
    if (driverContext == nullptr) {
        return nullptr;
    }

    return std::unique_ptr<CNvMDevice::DriverContext>(driverContext);
}

SIPLStatus CNvMDS90UB9724_971_TPG::InitSimulatorAndPassive() {
    return NVSIPL_STATUS_OK;
}
SIPLStatus CNvMDS90UB9724_971_TPG::GetInterruptStatus(const uint32_t gpioIdx,
                                                          IInterruptNotify &intrNotifier)
{
    return NVSIPL_STATUS_NOT_SUPPORTED;
}

CNvMCameraModule *CNvMCameraModule_Create() {
    return new CNvMDS90UB9724_971_TPG();
}

const char** CNvMCameraModule_GetNames() {
    static const char* names[] = {
        "DS90UB971TPG",
        NULL
    };
    return names;
}

} // end of namespace
