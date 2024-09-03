/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "CNvMCampwr.hpp"

namespace nvsipl
{

// Sets the configuration for camera power device object
SIPLStatus CNvMCampwr::SetConfig(uint8_t i2cAddress,
                                 DeviceParams const *const params)
{
    m_oDeviceParams = *params;

    if ((!m_oDeviceParams.bEnableSimulator) and (!m_oDeviceParams.bPassive)) {
        m_nativeI2CAddr = i2cAddress;
        m_oDeviceParams.pI2CAddrMgr->RegisterNativeI2CAddr(m_nativeI2CAddr);
    }

    /*
    * Always use CDI version 2 API (CDAC) for QNX and not for Linux.
    */
#ifdef NVMEDIA_QNX
    m_oDeviceParams.bUseCDIv2API = NVMEDIA_TRUE;
#else
    m_oDeviceParams.bUseCDIv2API = NVMEDIA_FALSE;
#endif

    m_eState = DeviceState::CDI_DEVICE_CONFIG_SET;

    return NVSIPL_STATUS_OK;
}

// Gets camera power load switch error size
SIPLStatus CNvMCampwr::GetErrorSize(size_t & errorSize)
{
    LOG_WARN("Detailed error information is not supported for this serializer "
        "in CNvMCampwr::GetErrorInfo\n");
    errorSize = 0U;
    return NVSIPL_STATUS_NOT_SUPPORTED;
}

// Gets generic camera power load switch error information
SIPLStatus CNvMCampwr::GetErrorInfo(std::uint8_t * const buffer,
                                        std::size_t const bufferSize,
                                        std::size_t &size)
{
    LOG_WARN("Detailed error information is not supported for this serializer "
        "in CNvMCampwr::GetErrorInfo\n");
    static_cast<void>(buffer);
    static_cast<void>(bufferSize);
    static_cast<void>(size);
    return NVSIPL_STATUS_NOT_SUPPORTED;
}

// Check to see if a power control device is supported
SIPLStatus CNvMCampwr::isSupported()
{
    return NVSIPL_STATUS_NOT_SUPPORTED;
}

// Use power control device to power on/off the camera modules.
SIPLStatus CNvMCampwr::PowerControlSetUnitPower(DevBlkCDIDevice* cdiDev,
                                                uint8_t const linkIndex,
                                                bool const enable)
{
    static_cast<void>(cdiDev);
    static_cast<void>(linkIndex);
    static_cast<void>(enable);
    return NVSIPL_STATUS_NOT_SUPPORTED;
}

// Creates a new power control CDI device.
SIPLStatus CNvMCampwr::CreatePowerDevice(DevBlkCDIRootDevice* const cdiRootDev,
                                         uint8_t const linkIndex)
{
    static_cast<void>(cdiRootDev);
    static_cast<void>(linkIndex);
    return NVSIPL_STATUS_NOT_SUPPORTED;
}

// Retrieves the CDI device handle for the power device
DevBlkCDIDevice* CNvMCampwr::GetDeviceHandle()
{
    return nullptr;
}

// Check the device presence  for the power device
SIPLStatus CNvMCampwr::CheckPresence(DevBlkCDIRootDevice* const cdiRootDev,
                                     DevBlkCDIDevice* const cdiDev)
{
    static_cast<void>(cdiRootDev);
    static_cast<void>(cdiDev);
    return NVSIPL_STATUS_NOT_SUPPORTED;
}

// Initializes the power device object
SIPLStatus CNvMCampwr::InitPowerDevice(DevBlkCDIRootDevice* const cdiRootDev,
                        DevBlkCDIDevice* const cdiDev, uint8_t const linkIndex,
                        int32_t const csiPort)
{
    static_cast<void>(cdiRootDev);
    static_cast<void>(cdiDev);
    static_cast<void>(linkIndex);
    static_cast<void>(csiPort);
    return NVSIPL_STATUS_NOT_SUPPORTED;
}

// Read Register
SIPLStatus
CNvMCampwr::ReadRegister(DevBlkCDIDevice const * const handle,
                uint8_t const linkIndex, uint32_t const registerNum,
                uint32_t const dataLength, uint8_t * const dataBuff)
{
    static_cast<void>(handle);
    static_cast<void>(linkIndex);
    static_cast<void>(registerNum);
    static_cast<void>(dataLength);
    static_cast<void>(dataBuff);
    return NVSIPL_STATUS_NOT_SUPPORTED;
}

// Write Register
SIPLStatus
CNvMCampwr::WriteRegister(DevBlkCDIDevice const * const handle,
                uint8_t const linkIndex, uint32_t const registerNum,
                uint32_t const dataLength, uint8_t const * const dataBuff)
{
    static_cast<void>(handle);
    static_cast<void>(linkIndex);
    static_cast<void>(registerNum);
    static_cast<void>(dataLength);
    static_cast<void>(dataBuff);
    return NVSIPL_STATUS_NOT_SUPPORTED;
}

} // end of nvsipl
