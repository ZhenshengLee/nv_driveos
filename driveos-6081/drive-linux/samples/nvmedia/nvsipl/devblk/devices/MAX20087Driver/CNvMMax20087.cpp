/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "CNvMMax20087.hpp"
#include "cdi_max20087.h"
#include "sipl_error.h"

namespace nvsipl
{

SIPLStatus CNvMMax20087::SetConfig(uint8_t i2cAddress, const DeviceParams *const params)
{
    SIPLStatus status = NVSIPL_STATUS_OK;

    status = CNvMCampwr::SetConfig(i2cAddress, params);
    if (status == NVSIPL_STATUS_OK) {
        /*! Get CDI Driver */
        m_pCDIDriver = GetMAX20087Driver();
        if (m_pCDIDriver == nullptr) {
            SIPL_LOG_ERR_STR("GetMAX20087Driver() failed!");
            status = NVSIPL_STATUS_ERROR;
        }
    } else {
        SIPL_LOG_ERR_STR_INT("CNvMCampwr::SetConfig failed with SIPL error",
                             static_cast<int32_t>(status));
    }

    return status;
}

SIPLStatus CNvMMax20087::GetErrorSize(size_t & errorSize)
{
    errorSize = 0;
    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMMax20087::GetErrorInfo(std::uint8_t * const buffer,
                                     const std::size_t bufferSize,
                                     std::size_t &size)
{
    size = 0U;
    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMMax20087::isSupported()
{
    if (m_eState == DeviceState::CDI_DEVICE_CREATED) {
        return NVSIPL_STATUS_OK;
    } else {
        return NVSIPL_STATUS_NOT_SUPPORTED;
    }
}

SIPLStatus CNvMMax20087::PowerControlSetUnitPower(DevBlkCDIDevice* cdiDev, uint8_t const linkIndex, bool const enable)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    status = MAX20087SetLinkPower(cdiDev, linkIndex, enable);
    if (status == NVMEDIA_STATUS_OK) {
        return NVSIPL_STATUS_OK;
    } else {
        return NVSIPL_STATUS_ERROR;
    }
}

SIPLStatus CNvMMax20087::CreatePowerDevice(DevBlkCDIRootDevice* const cdiRootDev, const uint8_t linkIndex)
{
    return CNvMDevice::CreateCDIDevice(cdiRootDev, linkIndex);
}

DevBlkCDIDevice* CNvMMax20087::GetDeviceHandle()
{
    return CNvMDevice::GetCDIDeviceHandle();
}

SIPLStatus CNvMMax20087::InitPowerDevice(DevBlkCDIRootDevice* const cdiRootDev,
                            DevBlkCDIDevice* const cdiDev, uint8_t const linkIndex,
                            int32_t const csiPort)
{
    static_cast<void>(csiPort);

    NvMediaStatus status = NVMEDIA_STATUS_OK;

    status = MAX20087Init(cdiRootDev, cdiDev);
    if (status != NVMEDIA_STATUS_OK) {
        return NVSIPL_STATUS_ERROR;
    }

    status = MAX20087CheckPresence(cdiDev);
    if (status != NVMEDIA_STATUS_OK) {
        return NVSIPL_STATUS_ERROR;
    } else {
        return NVSIPL_STATUS_OK;
    }
}

SIPLStatus
CNvMMax20087::ReadRegister(DevBlkCDIDevice const * const handle,
              uint8_t const linkIndex, uint32_t const registerNum,
              uint32_t const dataLength, uint8_t * const dataBuff)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if ((handle == nullptr) || (dataBuff == nullptr)) {
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        constexpr uint32_t const deviceIndex {0U};
        static_cast<void>(linkIndex);
        status = MAX20087ReadRegister(handle, deviceIndex, registerNum,
                 dataLength, dataBuff);
    }

    return ConvertNvMediaStatus(status);
}

SIPLStatus
CNvMMax20087::WriteRegister(DevBlkCDIDevice const * const handle,
              uint8_t const linkIndex, uint32_t const registerNum,
              uint32_t const dataLength, uint8_t const * const dataBuff)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if ((handle == nullptr) || (dataBuff == nullptr)) {
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        constexpr uint32_t const deviceIndex {0U};
        static_cast<void>(linkIndex);
        status = MAX20087WriteRegister(handle, deviceIndex, registerNum,
                 dataLength, dataBuff);
    }

    return ConvertNvMediaStatus(status);
}

/* Mask or restore mask of interrupts */
SIPLStatus
CNvMMax20087::MaskRestoreInterrupt(const bool enableGlobalMask)
{
    return NVSIPL_STATUS_NOT_SUPPORTED;
}

} // end of namespace nvsipl
